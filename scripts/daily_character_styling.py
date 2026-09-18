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
    python -m scripts.daily_character_styling --only harsh_vardhan_sinha,ira_sengupta   # a list
    python -m scripts.daily_character_styling --limit 3         # smoke test: first 3 characters
    python -m scripts.daily_character_styling                   # all 22, styling + OOTD
    python -m scripts.daily_character_styling --styling-only
    python -m scripts.daily_character_styling --ootd-only
    python -m scripts.daily_character_styling --concurrency 4   # default; see note below

Designed to run daily via the daily-character-styling GitHub Actions workflow (SSH + docker
compose exec, same shape as seed-wardrobes.yml). A failure on one character's one outfit does not
abort the run — see the per-call try/except below — because a bad wardrobe state or a transient
provider error on character #14 should not cost characters #15-22 their day's picks.

**Characters run concurrently, bounded by --concurrency.** The first version of this script ran
one character after another, and the very first full run measured ~13 minutes per character for
3 styling calls + 1 OOTD (each a full pipeline: request normalisation, retrieval, compatibility
scoring, image generation, visual gates with retries) — 22 characters serially is close to 5
hours, and the run was killed by the CI step's own timeout after only 7. Each character already
gets its own AsyncSessionLocal, so nothing about correctness changes by running several at once;
what changes is that a ~5-hour job becomes roughly (5 hours / concurrency).

**The default is 4, not higher — this was tuned empirically, not guessed.** Concurrency 6 across
all 22 characters (several of which carry very large wardrobes: Shalini 2,500 garments, three
others 900) was killed by the Linux OOM killer nine minutes in (`exit status 137`, nothing logged
past the startup line) — that many concurrent retrieval/compositing/base64-encoding passes over
large wardrobes exhausted the droplet's memory before a single character finished. Concurrency 4
had already processed that exact hardest subset (Shalini plus the three 900s, among 14
characters) successfully in an earlier run, so it is the proven-safe ceiling, not an optimistic
guess. Raise it only after checking the droplet actually has memory to spare — a rate-limit
concern was the original reason for a cap at all, but memory turned out to be the tighter one.
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
                    "prompt": prompt,
                    "garment_ids": [g.garment_id for g in outfit.garments],
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
                "cached": resp.cached, "outfits": len(resp.styling.outfits)}
    except Exception as e:
        await session.rollback()
        return {"slug": persona.slug, "ok": False, "error": f"{type(e).__name__}: {e}"}


async def process_one(p: Persona, storage, today: str, do_styling: bool, do_ootd: bool,
                      prompts_per_char: int) -> Dict[str, Any]:
    """One character's full day's work, in its own session. Called concurrently — see `run()` —
    so nothing here may share a session or transaction with another character."""
    out = {"slug": p.slug, "styling_ok": 0, "styling_fail": 0, "ootd_ok": False, "ootd_err": None}
    async with AsyncSessionLocal() as session:
        if do_ootd:
            await ensure_ootd_subscription(session, p)

        if do_styling:
            prompts = build_prompts(p, prompts_per_char, today)
            r = await style_one(session, storage, p, prompts)
            out["styling_ok"] = len(r["outfits"])
            out["styling_fail"] = len(r["errors"])
            out["styling_errors"] = r["errors"]
            out["prompts_n"] = len(prompts)
            print(f"  {p.slug:26s} styling: {out['styling_ok']}/{len(prompts)} outfits"
                  + (f"  [{'; '.join(r['errors'])[:140]}]" if r["errors"] else ""))
            sys.stdout.flush()

        if do_ootd:
            r = await ootd_one(session, storage, p)
            out["ootd_ok"] = r["ok"]
            if r["ok"]:
                out["ootd_location"] = r["location"]
                print(f"  {p.slug:26s} ootd: ok ({r['location']})")
            else:
                out["ootd_err"] = r["error"]
                print(f"  {p.slug:26s} ootd: FAILED — {r['error'][:140]}")
            sys.stdout.flush()
    return out


async def run(only: Optional[List[str]], limit: Optional[int], dry_run: bool,
              do_styling: bool, do_ootd: bool, prompts_per_char: int, concurrency: int) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    storage = get_storage_client()

    async with AsyncSessionLocal() as session:
        personas = (await session.execute(select(Persona).order_by(Persona.slug))).scalars().all()
    if only:
        wanted = set(only)
        personas = [p for p in personas if p.slug in wanted]
        missing = wanted - {p.slug for p in personas}
        if missing:
            print(f"! unknown slug(s), skipped: {', '.join(sorted(missing))}")
    if limit:
        personas = personas[:limit]
    if not personas:
        raise SystemExit("no characters found — run scripts.seed_personas first, or check --only")

    print(f"date (UTC): {today}")
    print(f"characters: {len(personas)}  styling={do_styling} ({prompts_per_char} prompts each)"
          f"  ootd={do_ootd}  concurrency={concurrency}\n")

    if dry_run:
        for p in personas:
            prompts = build_prompts(p, prompts_per_char, today)
            print(f"  {p.slug:26s} " + " | ".join(pr[:36] for pr in prompts))
        print("\ndry run — nothing generated")
        return

    sem = asyncio.Semaphore(concurrency)

    async def bounded(p: Persona) -> Dict[str, Any]:
        async with sem:
            try:
                return await process_one(p, storage, today, do_styling, do_ootd, prompts_per_char)
            except Exception as e:
                print(f"  {p.slug:26s} FAILED ENTIRELY — {type(e).__name__}: {e}")
                sys.stdout.flush()
                return {"slug": p.slug, "styling_ok": 0, "styling_fail": prompts_per_char,
                        "ootd_ok": False, "ootd_err": str(e)}

    results = await asyncio.gather(*(bounded(p) for p in personas))

    styling_ok = sum(r["styling_ok"] for r in results)
    styling_fail = sum(r["styling_fail"] for r in results)
    ootd_ok = sum(1 for r in results if r["ootd_ok"])
    ootd_fail = sum(1 for r in results if do_ootd and not r["ootd_ok"])

    print(f"\nstyling: {styling_ok} outfit(s) generated, {styling_fail} failure(s)")
    print(f"ootd:    {ootd_ok} succeeded, {ootd_fail} failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="a persona slug, or a comma-separated list of slugs")
    ap.add_argument("--limit", type=int, help="only the first N characters (alphabetical)")
    ap.add_argument("--dry-run", action="store_true", help="print the chosen prompts, generate nothing")
    ap.add_argument("--styling-only", action="store_true")
    ap.add_argument("--ootd-only", action="store_true")
    ap.add_argument("--prompts-per-char", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=4,
                    help="characters processed at once (default 6 — see module docstring)")
    args = ap.parse_args()
    do_styling = not args.ootd_only
    do_ootd = not args.styling_only
    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    asyncio.run(run(only, args.limit, args.dry_run, do_styling, do_ootd,
                    args.prompts_per_char, args.concurrency))


if __name__ == "__main__":
    main()
