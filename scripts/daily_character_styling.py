"""Generate the day's styling output for every evaluation character: 3 outfit picks from
different prompts, plus their weather-aware outfit-of-the-day.

Runs the same code path a real request would — StylingOrchestrator.run() and
get_or_generate_ootd() directly against the DB, exactly like scripts/seed_persona_wardrobes.py —
rather than going over HTTP. That avoids re-authenticating as a stylist for every character and
sidesteps Cloudflare entirely, and it is how every other admin script in this repo already talks
to the app when running inside the container.

**Why 3 separate calls rather than one call with top_k=3.** top_k=3 returns three outfits for the
*same* prompt and context — three variations on one theme. The panel needs three genuinely
different scenarios (an office day, a dinner out, a festive occasion) so a stylist reviewing a
character's history sees range, not three near-duplicates. Each call is real spend — request
normalisation, retrieval, compatibility scoring, image generation and visual gates — so three
prompts means three full pipeline runs, by design.

**Prompt rotation.** A bank of ~14 occasion templates, personalised with the character's own
occupation/dress-code/city, is shuffled deterministically per (UTC date, character) so the same
three prompts never repeat on the same day, the three within a day are never duplicates of each
other, and the sequence still varies day to day without needing to remember what ran yesterday.

**OOTD.** Each character is subscribed once (idempotent — enabled=True, location=ootd_location)
so the existing daily scheduler (app/worker/ootd_scheduler.py) picks them up forever without this
script's involvement. This script also calls get_or_generate_ootd() directly so today's pick
exists immediately rather than waiting for the next scheduled run.

    python -m scripts.daily_character_styling --dry-run
    python -m scripts.daily_character_styling --only shalini_rao
    python -m scripts.daily_character_styling --limit 3         # smoke test: first 3 characters
    python -m scripts.daily_character_styling                   # all 22, styling + OOTD
    python -m scripts.daily_character_styling --styling-only
    python -m scripts.daily_character_styling --ootd-only

Designed to run daily via the daily-character-styling GitHub Actions workflow (SSH + docker
compose exec, same shape as seed-wardrobes.yml). A failure on one character's one outfit does not
abort the run — see the per-call try/except below — because a bad wardrobe state or a transient
provider error on character #14 should not cost characters #15-22 their day's picks.
"""

import argparse
import asyncio
import hashlib
import random
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.ootd import OOTDSubscription
from app.models.persona import Persona
from app.schemas.styling import StylingRecommendationRequest
from app.storage import get_storage_client
from app.styling.ootd import get_or_generate_ootd
from app.styling.orchestrator import StylingOrchestrator

# {occasion label -> template}. `{occupation}`/`{dress_code}`/`{city}` are filled from the
# character; any that are blank on a given character just drop out of the sentence.
PROMPT_BANK: List[str] = [
    "A regular workday at the office. Something {dress_code} that suits {occupation_phrase}.",
    "Weekend brunch with friends — relaxed but put-together, not office wear.",
    "An evening dinner out, dressed a little more than a normal day.",
    "Travelling today — comfortable for a long day and easy to move in, still presentable.",
    "A festive family occasion at home — want to look celebratory without overdoing it.",
    "Running errands around {city} — casual, practical, nothing precious.",
    "A first date — want to feel confident and like myself, not overdone.",
    "An important client meeting. Professional and polished, {dress_code} at minimum.",
    "Casual Friday at work — relaxed but still put together for {occupation_phrase}.",
    "An outdoor day out — walking around a market or park, on my feet most of the day.",
    "A wedding or big family celebration — festive and elevated.",
    "A cold morning start — need real warmth without looking bulky.",
    "Gym this morning, then straight into the day — something that transitions well.",
    "A quiet day working from home, but might step out later — smart but comfortable.",
]

STYLING_TOP_K = 1  # one outfit per call, so 3 calls really are 3 different scenarios, not 1x3.


def build_prompts(persona: Persona, n: int, today: str) -> List[str]:
    occupation = (persona.occupation or "").strip()
    dress_code = (persona.work_dress_code or "smart casual").replace("_", " ")
    city = (persona.city or "").strip()
    occupation_phrase = f"a {occupation}" if occupation else "my day"

    # Deterministic per (date, character): stable within a run, varies day to day, never repeats
    # within the same day's pick because sample() draws without replacement.
    seed = int(hashlib.sha256(f"{today}:{persona.slug}".encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    chosen = rng.sample(PROMPT_BANK, k=min(n, len(PROMPT_BANK)))
    return [
        t.format(occupation_phrase=occupation_phrase, dress_code=dress_code, city=city or "town")
        for t in chosen
    ]


async def style_one(session, storage, persona: Persona, prompts: List[str]) -> Dict[str, Any]:
    result = {"slug": persona.slug, "outfits": [], "errors": []}
    for prompt in prompts:
        try:
            orchestrator = StylingOrchestrator(session, storage)
            resp = await orchestrator.run(
                StylingRecommendationRequest(request_text=prompt, top_k=STYLING_TOP_K),
                tenant_id=persona.user_id, member_id=persona.user_id,
            )
            for outfit in resp.outfits:
                result["outfits"].append({
                    "prompt": prompt, "garment_ids": outfit.garment_ids,
                    "image": bool(outfit.generated_image_url),
                })
            if not resp.outfits:
                result["errors"].append(f"no outfit produced for: {prompt[:60]}")
        except Exception as e:
            result["errors"].append(f"{type(e).__name__}: {e} (prompt: {prompt[:60]})")
        # Each call commits its own state via the orchestrator's session usage; rolling back here
        # would undo a genuinely-succeeded outfit because of a later prompt's unrelated failure.
        await session.commit()
    return result


async def ensure_ootd_subscription(session, persona: Persona) -> None:
    stmt = select(OOTDSubscription).where(
        OOTDSubscription.tenant_id == persona.user_id,
        OOTDSubscription.member_id == persona.user_id,
    )
    sub = (await session.execute(stmt)).scalars().first()
    location = persona.ootd_location or persona.city
    if sub:
        sub.location = location
        sub.enabled = True
    else:
        session.add(OOTDSubscription(
            tenant_id=persona.user_id, member_id=persona.user_id,
            location=location, enabled=True,
        ))
    await session.commit()


async def ootd_one(session, storage, persona: Persona) -> Dict[str, Any]:
    location = persona.ootd_location or persona.city
    try:
        resp = await get_or_generate_ootd(
            session, storage, persona.user_id, persona.user_id,
            location=location, force=False, generation_source="scheduled",
        )
        await session.commit()
        return {"slug": persona.slug, "ok": True, "location": location,
                "outfits": len(resp.outfits) if hasattr(resp, "outfits") else None}
    except Exception as e:
        await session.rollback()
        return {"slug": persona.slug, "ok": False, "error": f"{type(e).__name__}: {e}"}


async def run(only: Optional[str], limit: Optional[int], dry_run: bool,
              do_styling: bool, do_ootd: bool, prompts_per_char: int) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    storage = get_storage_client()

    async with AsyncSessionLocal() as session:
        personas = (await session.execute(select(Persona).order_by(Persona.slug))).scalars().all()
    if only:
        personas = [p for p in personas if p.slug == only]
    if limit:
        personas = personas[:limit]
    if not personas:
        raise SystemExit("no characters found — run scripts.seed_personas first")

    print(f"date (UTC): {today}")
    print(f"characters: {len(personas)}  styling={do_styling} ({prompts_per_char} prompts each)"
          f"  ootd={do_ootd}\n")

    if dry_run:
        for p in personas:
            prompts = build_prompts(p, prompts_per_char, today)
            print(f"  {p.slug:26s} " + " | ".join(pr[:36] for pr in prompts))
        print("\ndry run — nothing generated")
        return

    styling_ok = styling_fail = ootd_ok = ootd_fail = 0
    for p in personas:
        async with AsyncSessionLocal() as session:
            if do_ootd:
                await ensure_ootd_subscription(session, p)

            if do_styling:
                prompts = build_prompts(p, prompts_per_char, today)
                r = await style_one(session, storage, p, prompts)
                n_ok = len(r["outfits"])
                styling_ok += n_ok
                styling_fail += len(r["errors"])
                print(f"  {p.slug:26s} styling: {n_ok}/{len(prompts)} outfits"
                      + (f"  [{'; '.join(r['errors'])[:140]}]" if r["errors"] else ""))

            if do_ootd:
                r = await ootd_one(session, storage, p)
                if r["ok"]:
                    ootd_ok += 1
                    print(f"  {p.slug:26s} ootd: ok ({r['location']})")
                else:
                    ootd_fail += 1
                    print(f"  {p.slug:26s} ootd: FAILED — {r['error'][:140]}")
        sys.stdout.flush()

    print(f"\nstyling: {styling_ok} outfit(s) generated, {styling_fail} failure(s)")
    print(f"ootd:    {ootd_ok} succeeded, {ootd_fail} failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="a single persona slug")
    ap.add_argument("--limit", type=int, help="only the first N characters (alphabetical)")
    ap.add_argument("--dry-run", action="store_true", help="print the chosen prompts, generate nothing")
    ap.add_argument("--styling-only", action="store_true")
    ap.add_argument("--ootd-only", action="store_true")
    ap.add_argument("--prompts-per-char", type=int, default=3)
    args = ap.parse_args()
    do_styling = not args.ootd_only
    do_ootd = not args.styling_only
    asyncio.run(run(args.only, args.limit, args.dry_run, do_styling, do_ootd, args.prompts_per_char))


if __name__ == "__main__":
    main()
