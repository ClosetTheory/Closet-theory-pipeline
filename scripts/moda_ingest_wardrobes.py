"""Push every evaluation character's wardrobe into Hopit's MODA, keyed by the character's own id.

Why this exists: the MODA key we hold grants `ingest` only. Search, composed search, outfit
ranking and raw vectors all return 403 `scope_forbidden`, so nothing about retrieval quality can
be measured yet. Asking Hopit to widen the scope is easier with their index already holding a
realistic, per-owner corpus rather than a 200-record sample, and the moment the wider key arrives
the comparison can run immediately instead of starting with a multi-hour ingest.

How it works: each character's garments are read through this deployment's own API, acting as
that character, and handed to MODA as `owner_ref = <the character's backing user id>` -- the same
id that is their `tenant_id` and `member_id` here, so a garment can be traced back to exactly one
wardrobe on both sides. Image URLs point at this deployment's public media route, which MODA
fetches directly; nothing is copied, re-uploaded or made public that was not already.

Two things about the corpus, stated plainly because they affect how the results should be read:

- **It is heavily duplicated by construction.** The 12,350 garments were cloned from 1,177
  distinct ones, and MODA does not deduplicate (verified: the same image under two ids yields the
  same `image_sha256` and two separate records). So their index will hold roughly ten copies of
  each distinct garment. That is faithful to production, where wardrobes are separate per user,
  but it means index size is not a measure of catalogue variety.
- **Retrieval will be scored per owner, not globally.** Every MODA retrieval call takes an
  explicit `candidates` list, so a wardrobe is whatever set we pass -- which is exactly how our
  own styling already scopes a wardrobe.

    python scripts/moda_ingest_wardrobes.py --dry-run
    python scripts/moda_ingest_wardrobes.py --only shalini_rao
    python scripts/moda_ingest_wardrobes.py               # all 22
    python scripts/moda_ingest_wardrobes.py --purge       # erase everything we sent

Resumable: every submitted garment id is written to a ledger, and a re-run skips what is already
there rather than paying to embed it twice.
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "unknown_material_samples", "moda_wardrobe_ledger.json")

APP_BASE = os.environ.get("CT_APP_BASE", "https://visualiser-dsfja238un9.closettheory.co")
STYLISTS = [f"stylist{i}@closettheory.co" for i in (1, 2, 3)]
STYLIST_PASSWORD = os.environ.get("CT_STYLIST_PASSWORD", "stylist1234")

BATCH_CAP = 200          # MODA's documented limit; 201 is a 413 batch_too_large
POLL_SECONDS = 3
ATTR_KEYS = ("category", "subcategory", "colour", "pattern", "fit", "season", "occasion",
             "material", "warmth", "formality", "layering_role")


def load_env() -> Dict[str, str]:
    out: Dict[str, str] = {}
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        with io.open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
    out.update({k: v for k, v in os.environ.items() if k.startswith("MODA_")})
    return out


ENV = load_env()
MODA_API = ENV.get("MODA_API", "").rstrip("/")
MODA_KEY = ENV.get("MODA_KEY", "")


def _request(url: str, method: str = "GET", body: Any = None,
             headers: Optional[Dict[str, str]] = None, timeout: int = 300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "content-type": "application/json",
        # Cloudflare fronts the app domain and answers urllib's default User-Agent with
        # "error code: 1010" — a 403 with no JSON body, which reads as a failed login rather
        # than a blocked client. Any ordinary UA gets through.
        "user-agent": "closet-theory-scripts/1.0",
        **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:400].decode(errors="replace")}
    except Exception as e:
        return "ERR", {"raw": f"{type(e).__name__}: {e}"}


def moda(method: str, path: str, body: Any = None, timeout: int = 300):
    return _request(MODA_API + path, method, body,
                    {"Authorization": "Bearer " + MODA_KEY}, timeout)


# --- reading the wardrobes -------------------------------------------------------------------


def login(email: str) -> Optional[str]:
    s, d = _request(f"{APP_BASE}/api/v1/auth/login", "POST",
                    {"email": email, "password": STYLIST_PASSWORD}, timeout=60)
    return d.get("token") if s == 200 else None


def collect_characters() -> List[Dict[str, Any]]:
    """Every character, with the token that can act as it.

    The three stylists between them cover the whole roster, and each only sees their own
    assignments -- so the roster is assembled by logging in as all three rather than by needing an
    admin account.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for email in STYLISTS:
        token = login(email)
        if not token:
            print(f"  ! could not log in as {email}")
            continue
        s, rows = _request(f"{APP_BASE}/api/v1/personas", headers={"Authorization": "Bearer " + token},
                           timeout=90)
        if s != 200:
            print(f"  ! {email}: personas -> {s}")
            continue
        for p in (rows if isinstance(rows, list) else rows.get("personas", [])):
            if p["slug"] not in seen:
                seen[p["slug"]] = {**p, "_token": token}
    return [seen[k] for k in sorted(seen)]


def wardrobe(character: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    offset = 0
    while True:
        s, rows = _request(
            f"{APP_BASE}/api/v1/wardrobe/garments?limit=3000&offset={offset}",
            headers={"Authorization": "Bearer " + character["_token"],
                     "X-Acting-As-Persona": character["persona_id"]},
            timeout=300)
        if s != 200 or not rows:
            break
        out.extend(rows)
        if len(rows) < 3000:
            break
        offset += len(rows)
    return out


def to_record(g: Dict[str, Any], owner_ref: str) -> Optional[Dict[str, Any]]:
    """MODA rejects unknown top-level fields outright (422 `extra_forbidden`), so only the four
    documented keys go on the wire. `attributes` itself is free-form."""
    url = g.get("canonical_image_url")
    if not url:
        return None
    attrs = {k: v for k, v in (g.get("attributes") or {}).items() if k in ATTR_KEYS and v is not None}
    rec = {
        "garment_id": g["garment_id"],
        "owner_ref": owner_ref,
        "image_url": url if url.startswith("http") else f"{APP_BASE}{url}",
    }
    if attrs:
        rec["attributes"] = attrs
    return rec


# --- ledger ----------------------------------------------------------------------------------


def read_ledger() -> Dict[str, Any]:
    if os.path.exists(LEDGER):
        return json.load(io.open(LEDGER, encoding="utf-8"))
    return {"sent": {}, "jobs": [], "failures": []}


def write_ledger(led: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    json.dump(led, io.open(LEDGER, "w", encoding="utf-8"), indent=1)


# --- stages ----------------------------------------------------------------------------------


def send_batches(records: List[Dict[str, Any]], led: Dict[str, Any], label: str) -> Dict[str, int]:
    counts = {"indexed": 0, "failed": 0}
    for i in range(0, len(records), BATCH_CAP):
        chunk = records[i:i + BATCH_CAP]
        s, d = moda("POST", "/v1/garments:batch", {"garments": chunk})
        if s != 202:
            print(f"    batch {i // BATCH_CAP + 1}: HTTP {s} {json.dumps(d)[:200]}")
            led["failures"].append({"label": label, "status": s, "body": d})
            write_ledger(led)
            continue
        job = d["job_id"]
        t0 = time.time()
        while True:
            time.sleep(POLL_SECONDS)
            _, j = moda("GET", f"/v1/jobs/{job}")
            if j.get("state") in ("completed", "failed", "partial"):
                break
            if time.time() - t0 > 1800:
                print("    timed out waiting for job")
                break
        for r in (j.get("records") or []):
            if r.get("state") == "indexed":
                counts["indexed"] += 1
                led["sent"][r["garment_id"]] = label
            else:
                counts["failed"] += 1
                led["failures"].append({"label": label, **r})
        led["jobs"].append({"job": job, "label": label, "n": len(chunk),
                            "counts": j.get("counts")})
        write_ledger(led)
        print(f"    batch {i // BATCH_CAP + 1}/{(len(records) + BATCH_CAP - 1) // BATCH_CAP}"
              f"  {j.get('counts')}  ({time.time() - t0:.0f}s)")
    return counts


def run(only: Optional[str], dry_run: bool, limit: Optional[int]) -> None:
    if not MODA_API or not MODA_KEY:
        raise SystemExit("MODA_API / MODA_KEY missing (.env or environment)")
    led = read_ledger()
    already = set(led["sent"])
    print(f"app: {APP_BASE}\nmoda: {MODA_API}\nledger: {len(already)} garments already sent\n")

    characters = collect_characters()
    if only:
        characters = [c for c in characters if c["slug"] == only]
    if not characters:
        raise SystemExit("no characters visible — check stylist credentials")

    grand = {"indexed": 0, "failed": 0, "skipped": 0, "records": 0}
    for c in characters:
        owner = c.get("user_id")
        if not owner:
            print(f"  ! {c['slug']}: no user_id in the API response — redeploy needed")
            continue
        garments = wardrobe(c)
        records = [r for r in (to_record(g, owner) for g in garments) if r]
        fresh = [r for r in records if r["garment_id"] not in already]
        grand["skipped"] += len(records) - len(fresh)
        if limit:
            fresh = fresh[:limit]
        grand["records"] += len(fresh)
        print(f"  {c['slug']:26s} owner={owner}  wardrobe={len(garments):5d}  to send={len(fresh):5d}")
        if dry_run or not fresh:
            continue
        counts = send_batches(fresh, led, c["slug"])
        grand["indexed"] += counts["indexed"]
        grand["failed"] += counts["failed"]
        sys.stdout.flush()

    print(f"\n{'dry run — nothing sent' if dry_run else 'sent'}: "
          f"{grand['records']} records, {grand['indexed']} indexed, {grand['failed']} failed, "
          f"{grand['skipped']} already in the ledger")


def purge() -> None:
    """Erase everything we put in MODA, by owner_ref — their documented erasure path."""
    led = read_ledger()
    owners = sorted(set(led["sent"].values()))
    characters = {c["slug"]: c for c in collect_characters()}
    for slug in owners:
        owner = (characters.get(slug) or {}).get("user_id")
        if not owner:
            continue
        s, d = moda("DELETE", f"/v1/owners/{owner}/garments")
        print(f"DELETE {slug:26s} ({owner}) -> {s} {json.dumps(d)[:120]}")
    write_ledger({"sent": {}, "jobs": led.get("jobs", []), "failures": []})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="a single persona slug")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="cap records per character (for a smoke test)")
    ap.add_argument("--purge", action="store_true", help="erase everything sent, then stop")
    args = ap.parse_args()
    if args.purge:
        purge()
        return
    run(args.only, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
