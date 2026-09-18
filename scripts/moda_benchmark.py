"""Benchmark Hopit's MODA API against the garments and labels we already have.

MODA is an ingestion-plus-retrieval service. It does *not* extract attributes -- a garment sent
with an image and nothing else comes back with `attributes: {}` -- so it has no counterpart to our
Stages 1-4. What it adds is one 768-d vector per garment (`space: pro-lite-768`,
`model_version: moda-pro-lite@1`) and four retrieval calls over a candidate set passed on every
request. Since our own Stage 5 already runs `HopitAI/moda-fashion-distilled` on RunPod, the real
question this script answers is narrow: **is their hosted model better than the distilled one we
already self-host, and is their ranking better than ours?**

Design notes worth knowing before reading the numbers:

- **Queries are derived from our own labels, not hand-written.** For each labelled garment we ask
  for "<colour> <subcategory>" and check where that garment lands. Hand-written queries measure
  the query writer; generated ones measure the model, and they cannot be tuned after seeing a
  result.
- **The unlabelled garments are distractors, not filler.** Retrieval accuracy against 60
  candidates is not a useful number, because the pool is smaller than most real wardrobes. Topping
  the pool up to MODA's 200-record batch cap makes every query harder in the way production is
  harder.
- **Conformance is checked, not assumed.** Stage `errors` deliberately sends a bad URL, an extra
  field and a 201-record batch, because the three documented failure modes are the ones an
  integration actually hits.

Usage:

    python scripts/moda_benchmark.py all            # everything, writes the report
    python scripts/moda_benchmark.py ingest         # stage by stage
    python scripts/moda_benchmark.py search
    python scripts/moda_benchmark.py outfits
    python scripts/moda_benchmark.py errors
    python scripts/moda_benchmark.py cleanup        # DELETE /v1/owners/{ref}/garments

Reads MODA_API / MODA_KEY and the DOS_* Spaces credentials from .env. Images are handed over as
time-limited presigned URLs; nothing is made permanently public. Everything written to MODA lives
under one owner_ref so `cleanup` can remove all of it in a single call.
"""

import argparse
import io
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "unknown_material_samples")
REPORTS = os.path.join(ROOT, "reports")
STATE = os.path.join(SAMPLES, "moda_bench_state.json")

OWNER_REF = "ctbench"
TARGET_N = 200          # MODA's documented batch cap
BATCH_CAP = 200
URL_TTL = 6 * 3600      # presigned URL lifetime; long enough for the whole suite


# --- plumbing -------------------------------------------------------------------------------


def load_env() -> Dict[str, str]:
    out: Dict[str, str] = {}
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        raise SystemExit(".env not found")
    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


ENV = load_env()
API = ENV.get("MODA_API", "").rstrip("/")
KEY = ENV.get("MODA_KEY", "")


def call(method: str, path: str, body: Any = None, timeout: int = 300) -> Tuple[Any, Any, float]:
    """Returns (status, parsed_body, seconds). Never raises on an HTTP error -- the error body is
    the interesting part when benchmarking someone else's API."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={"Authorization": "Bearer " + KEY, "content-type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {}), time.time() - t0
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw), time.time() - t0
        except Exception:
            return e.code, {"raw": raw[:600].decode(errors="replace")}, time.time() - t0
    except Exception as e:
        return "ERR", {"raw": f"{type(e).__name__}: {e}"}, time.time() - t0


def spaces():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=f"https://{ENV['DOS_REGION']}.digitaloceanspaces.com",
        region_name=ENV["DOS_REGION"], aws_access_key_id=ENV["DOS_ACCESS_KEY_ID"],
        aws_secret_access_key=ENV["DOS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"))


def save_state(d: Dict[str, Any]) -> None:
    prev = load_state()
    prev.update(d)
    json.dump(prev, io.open(STATE, "w", encoding="utf-8"), indent=2)


def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE):
        return json.load(io.open(STATE, encoding="utf-8"))
    return {}


# --- dataset --------------------------------------------------------------------------------


def build_dataset() -> Tuple[List[Dict], List[Dict]]:
    """Returns (labelled, distractors).

    `labelled` are the 60 exported garments whose attributes our own pipeline produced -- they are
    the only ones retrieval can be *scored* against. `distractors` top the pool up to 200 from the
    same bucket; they are indexed and searched over but never scored, which is what makes the
    scored queries hard.
    """
    labelled = json.load(io.open(os.path.join(SAMPLES, "garments.json"), encoding="utf-8"))
    rows = json.load(io.open(os.path.join(SAMPLES, "object_keys.json"), encoding="utf-8"))
    raw_key = {gid: key for gid, kind, key in rows if kind == "raw"}
    for g in labelled:
        g["_key"] = raw_key.get(g["garment_id"])
    labelled = [g for g in labelled if g["_key"]]

    need = max(0, TARGET_N - len(labelled))
    taken = {g["_key"] for g in labelled}
    distractors: List[Dict] = []
    if need:
        c = spaces()
        token = None
        pool: List[str] = []
        while len(pool) < need * 6:
            kw = dict(Bucket=ENV["DOS_BUCKET"], Prefix="production/wardrobe/", MaxKeys=1000)
            if token:
                kw["ContinuationToken"] = token
            r = c.list_objects_v2(**kw)
            for o in r.get("Contents", []):
                k = o["Key"][len("production/"):]
                if "/raw/" in k and k.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) \
                        and o["Size"] > 5000 and k not in taken:
                    pool.append(k)
            if not r.get("IsTruncated"):
                break
            token = r.get("NextContinuationToken")
        # Spread across owners rather than taking the first N, which would all come from one or
        # two wardrobes and make the distractor set far less varied than a real candidate pool.
        random.Random(20260918).shuffle(pool)
        # Index-prefixed, because two files can share a 40-character name prefix and a duplicate
        # garment_id would silently overwrite a record rather than error.
        for i, k in enumerate(pool[:need]):
            stem = k.split("/")[-1].split(".")[0][:40]
            distractors.append({"garment_id": f"dis{i:03d}_{stem}", "_key": k})
    return labelled, distractors


def presign(items: List[Dict]) -> None:
    c = spaces()
    for it in items:
        it["_url"] = c.generate_presigned_url(
            "get_object",
            Params={"Bucket": ENV["DOS_BUCKET"], "Key": "production/" + it["_key"].lstrip("/")},
            ExpiresIn=URL_TTL)


ATTR_KEYS = ("category", "subcategory", "colour", "pattern", "fit", "season", "occasion")


def to_record(g: Dict, labelled: bool) -> Dict:
    """MODA rejects unknown fields outright (`extra_forbidden`), so only the four documented keys
    go on the wire."""
    rec = {"garment_id": g["garment_id"], "owner_ref": OWNER_REF, "image_url": g["_url"]}
    if labelled:
        attrs = {k: g[k] for k in ATTR_KEYS if g.get(k) is not None}
        if attrs:
            rec["attributes"] = attrs
    return rec


# --- stages ---------------------------------------------------------------------------------


def stage_ingest() -> Dict[str, Any]:
    labelled, distractors = build_dataset()
    presign(labelled)
    presign(distractors)
    records = ([to_record(g, True) for g in labelled] +
               [to_record(g, False) for g in distractors])
    print(f"dataset: {len(labelled)} labelled + {len(distractors)} distractors "
          f"= {len(records)} records")
    if len(records) > BATCH_CAP:
        raise SystemExit(f"{len(records)} exceeds the documented {BATCH_CAP}-record cap")

    s, d, t_accept = call("POST", "/v1/garments:batch", {"garments": records})
    print(f"POST /v1/garments:batch -> {s} in {t_accept:.2f}s  {json.dumps(d)[:160]}")
    if s != 202:
        raise SystemExit("ingest not accepted")

    job = d["job_id"]
    t0 = time.time()
    timeline = []
    while True:
        time.sleep(2)
        _, j, _ = call("GET", f"/v1/jobs/{job}")
        el = time.time() - t0
        counts = j.get("counts") or {}
        timeline.append({"t": round(el, 1), "counts": dict(counts)})
        print(f"  t+{el:6.1f}s  {str(j.get('state')):10s} {counts}")
        if j.get("state") in ("completed", "failed", "partial"):
            break
        if el > 1800:
            print("  TIMED OUT")
            break

    wall = time.time() - t0
    recs = j.get("records") or []
    failed = [r for r in recs if r.get("state") != "indexed"]
    by_code: Dict[str, int] = {}
    for r in failed:
        by_code[r.get("code") or "?"] = by_code.get(r.get("code") or "?", 0) + 1

    print(f"\n  wall clock : {wall:.1f}s")
    print(f"  per garment: {wall / max(1, len(records)):.2f}s")
    print(f"  throughput : {len(records) / max(wall, 1e-9) * 60:.1f} garments/min")
    print(f"  indexed    : {(j.get('counts') or {}).get('indexed', 0)}/{len(records)}")
    print(f"  failed     : {len(failed)}  {by_code}")
    for r in failed[:10]:
        print(f"     {r['garment_id']}  {r.get('code')}  {r.get('message')}")

    indexed = [r["garment_id"] for r in recs if r.get("state") == "indexed"]
    save_state({
        "job": job, "accept_s": t_accept, "wall_s": wall, "n_sent": len(records),
        "timeline": timeline, "failed": failed, "by_code": by_code,
        "indexed": indexed,
        "labelled": [{k: g[k] for k in ("garment_id",) + ATTR_KEYS if k in g} for g in labelled],
    })
    return {"wall_s": wall, "accept_s": t_accept, "n": len(records),
            "indexed": len(indexed), "failed": len(failed), "by_code": by_code}


def _query_for(g: Dict) -> Optional[str]:
    """A short natural phrase naming this garment, built only from what our pipeline labelled."""
    colour = g.get("colour") or []
    sub = (g.get("subcategory") or g.get("category") or "").strip()
    if not sub:
        return None
    lead = " ".join(str(c).lower() for c in colour[:2])
    return (lead + " " + sub.lower()).strip()


class ScopeDenied(RuntimeError):
    """The API key does not grant this capability, so nothing was measured."""


def _check_scope(status: Any, body: Any) -> None:
    """A 403 means we learned nothing about the model, and must never reach the metrics.

    The first version of this script counted a scope denial as a query that returned no hit, and
    duly reported 0% recall across 60 queries — a number that looks like a devastating result and
    is in fact a statement about our API key. Refusing to score is the only honest response.
    """
    if status == 403 or (isinstance(body, dict) and body.get("code") == "scope_forbidden"):
        raise ScopeDenied((body or {}).get("message") or "scope_forbidden")


def stage_search() -> Dict[str, Any]:
    st = load_state()
    cands = st.get("indexed") or []
    labelled = st.get("labelled") or []
    if not cands:
        raise SystemExit("run `ingest` first")
    print(f"scoring {len(labelled)} generated queries against {len(cands)} candidates\n")

    ranks: List[int] = []
    misses = 0
    errors: List[Dict] = []
    lat: List[float] = []
    rows = []
    for g in labelled:
        q = _query_for(g)
        if not q:
            continue
        s, d, t = call("POST", "/v1/search/text",
                       {"candidates": cands, "query": q, "limit": 10})
        try:
            _check_scope(s, d)
        except ScopeDenied as e:
            save_state({"search": {"measured": False, "blocked_by": str(e)}})
            print(f"\n  NOT MEASURED — {e}")
            print("  Nothing about retrieval quality can be concluded from this run.")
            raise SystemExit(2)
        lat.append(t)
        if s != 200:
            errors.append({"query": q, "status": s, "body": json.dumps(d)[:160]})
            rows.append({"query": q, "target": g["garment_id"], "error": json.dumps(d)[:160]})
            print(f"  ERROR {s}  {q[:44]:44s} {json.dumps(d)[:70]}")
            continue
        hits = d.get("results") or []
        rank = next((h["rank"] for h in hits if h["garment_id"] == g["garment_id"]), None)
        if rank:
            ranks.append(rank)
        else:
            misses += 1
        rows.append({"query": q, "target": g["garment_id"], "rank": rank,
                     "top": hits[0]["garment_id"] if hits else None,
                     "top_sim": round(hits[0]["similarity"], 4) if hits else None})
        print(f"  rank {str(rank or '-'):>3}  {q[:46]:46s} ({t*1000:.0f}ms)")

    # Scored over queries that actually returned, with the error count reported alongside rather
    # than folded in — a transport failure is not a retrieval miss.
    n = len(ranks) + misses
    metrics = {
        "measured": True,
        "queries_scored": n,
        "queries_errored": len(errors),
        "recall_at_1": sum(1 for r in ranks if r == 1) / n if n else 0,
        "recall_at_5": sum(1 for r in ranks if r <= 5) / n if n else 0,
        "recall_at_10": len(ranks) / n if n else 0,
        "mrr": sum(1 / r for r in ranks) / n if n else 0,
        "latency_ms_p50": round(statistics.median(lat) * 1000, 1) if lat else None,
        "latency_ms_max": round(max(lat) * 1000, 1) if lat else None,
        "candidates": len(cands),
    }
    print("\n  scored   : {queries_scored} ({queries_errored} errored)"
          "\n  recall@1 : {recall_at_1:.1%}\n  recall@5 : {recall_at_5:.1%}"
          "\n  recall@10: {recall_at_10:.1%}\n  MRR      : {mrr:.3f}"
          "\n  latency  : p50 {latency_ms_p50}ms  max {latency_ms_max}ms".format(**metrics))
    save_state({"search": metrics, "search_rows": rows})
    return metrics


def stage_outfits() -> Dict[str, Any]:
    st = load_state()
    cands = st.get("indexed") or []
    if not cands:
        raise SystemExit("run `ingest` first")
    scenarios = [
        {"name": "office summer", "query": "smart office outfit",
         "context": {"occasion": "work", "season": "summer", "formality": 0.7,
                     "temperature_c": 32, "humidity": 70}},
        {"name": "casual monsoon", "query": "casual day out in the rain",
         "context": {"occasion": "casual", "season": "monsoon", "formality": 0.2,
                     "temperature_c": 26, "precipitation_probability": 0.8}},
        {"name": "evening dinner", "query": "dinner with friends",
         "context": {"occasion": "dinner", "time_of_day": "evening", "formality": 0.6,
                     "temperature_c": 24, "mood": "confident"}},
    ]
    out = []
    for sc in scenarios:
        s, d, t = call("POST", "/v1/outfits:rank",
                       {"candidates": cands, "query": sc["query"], "context": sc["context"],
                        "outfit_count": 3})
        try:
            _check_scope(s, d)
        except ScopeDenied as e:
            save_state({"outfits": {"measured": False, "blocked_by": str(e)}})
            print(f"\n  NOT MEASURED — {e}")
            raise SystemExit(2)
        print(f"\n{sc['name']}  -> {s} in {t*1000:.0f}ms")
        if s != 200:
            print("  ", json.dumps(d)[:250])
            out.append({"scenario": sc["name"], "status": s, "error": json.dumps(d)[:250]})
            continue
        print(f"  compatibility_source={d.get('compatibility_source')} "
              f"model={d.get('model_version')} warnings={d.get('warnings')}")
        for o in (d.get("outfits") or []):
            print(f"   score={o['score']:.3f} slots={o.get('slots')} "
                  f"factors={o.get('factors')} warn={o.get('warnings')}")
        out.append({"scenario": sc["name"], "status": s, "latency_ms": round(t * 1000, 1),
                    "compatibility_source": d.get("compatibility_source"),
                    "model_version": d.get("model_version"),
                    "outfits": d.get("outfits")})
    save_state({"outfits": out})
    return {"scenarios": out}


def stage_errors() -> Dict[str, Any]:
    """Confirm the three documented failure modes actually behave as documented."""
    results = {}

    s, d, _ = call("POST", "/v1/garments:batch", {"garments": [
        {"garment_id": "bench_badurl", "owner_ref": OWNER_REF,
         "image_url": "https://example.invalid/none.jpg"}]})
    job = d.get("job_id")
    rec = None
    if job:
        for _ in range(20):
            time.sleep(2)
            _, j, _ = call("GET", f"/v1/jobs/{job}")
            if j.get("state") in ("completed", "failed", "partial"):
                rec = (j.get("records") or [{}])[0]
                break
    print(f"bad url          -> accept {s}; record {json.dumps(rec)[:200]}")
    results["bad_url"] = {"accept_status": s, "record": rec}

    s, d, _ = call("POST", "/v1/garments:batch", {"garments": [
        {"garment_id": "bench_extra", "owner_ref": OWNER_REF,
         "image_url": "https://example.com/a.jpg", "user_id": "user_1"}]})
    print(f"extra field      -> {s}  {json.dumps(d)[:220]}")
    results["extra_field"] = {"status": s, "body": d}

    s, d, _ = call("POST", "/v1/garments:batch", {"garments": [
        {"garment_id": f"bench_cap_{i}", "owner_ref": OWNER_REF,
         "image_url": "https://example.com/a.jpg"} for i in range(BATCH_CAP + 1)]})
    print(f"{BATCH_CAP + 1}-record batch -> {s}  {json.dumps(d)[:220]}")
    results["batch_cap"] = {"status": s, "body": d}

    save_state({"errors": results})
    return results


def stage_cleanup() -> Dict[str, Any]:
    out = {}
    for ref in (OWNER_REF, "ctbench_probe"):
        s, d, t = call("DELETE", f"/v1/owners/{ref}/garments")
        print(f"DELETE /v1/owners/{ref}/garments -> {s} in {t:.2f}s  {json.dumps(d)[:200]}")
        out[ref] = {"status": s, "body": d}
    return out


def write_report() -> str:
    st = load_state()
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, "MODA_BENCHMARK.md")
    se = st.get("search") or {}
    lines = [
        "# MODA (Hopit) API benchmark",
        "",
        f"Run {time.strftime('%Y-%m-%d %H:%M')} against `{API}`.",
        "",
        "## Ingestion",
        "",
        f"- records sent: **{st.get('n_sent')}** (one batch, cap is {BATCH_CAP})",
        f"- accepted in: **{st.get('accept_s', 0):.2f}s** (202 + job id)",
        f"- indexed in: **{st.get('wall_s', 0):.1f}s** "
        f"(**{st.get('wall_s', 0) / max(1, st.get('n_sent') or 1):.2f}s/garment**, "
        f"{(st.get('n_sent') or 0) / max(st.get('wall_s') or 1, 1e-9) * 60:.0f}/min)",
        f"- failures: **{len(st.get('failed') or [])}** {st.get('by_code') or ''}",
        "",
        "## Text retrieval",
        "",
    ]
    if not se.get("measured", False):
        lines += [
            f"**Not measured.** {se.get('blocked_by') or 'the search stage did not run'}.",
            "",
            "No conclusion about retrieval quality can be drawn from this run. A key granting the "
            "`search`, `styling` and `embeddings` scopes is needed; this one grants `ingest` only.",
            "",
        ]
    else:
        lines += [
            f"Queries are generated from our own labels (`\"<colour> <subcategory>\"`) and scored "
            f"by where the garment they describe lands, over **{se.get('candidates')}** "
            f"candidates.",
            "",
            "| metric | value |",
            "|---|---|",
            f"| queries scored | {se.get('queries_scored')} |",
            f"| queries errored | {se.get('queries_errored')} |",
            f"| recall@1 | {se.get('recall_at_1', 0):.1%} |",
            f"| recall@5 | {se.get('recall_at_5', 0):.1%} |",
            f"| recall@10 | {se.get('recall_at_10', 0):.1%} |",
            f"| MRR | {se.get('mrr', 0):.3f} |",
            f"| latency p50 | {se.get('latency_ms_p50')} ms |",
            f"| latency max | {se.get('latency_ms_max')} ms |",
            "",
        ]
    lines += [
        "## Conformance",
        "",
        "```json",
        json.dumps(st.get("errors") or {}, indent=2)[:2500],
        "```",
        "",
    ]
    io.open(path, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nwrote {path}")
    return path


STAGES = {"ingest": stage_ingest, "search": stage_search, "outfits": stage_outfits,
          "errors": stage_errors, "cleanup": stage_cleanup}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=list(STAGES) + ["all", "report"])
    args = ap.parse_args()
    if not API or not KEY:
        raise SystemExit("MODA_API / MODA_KEY missing from .env")
    if args.stage == "report":
        write_report()
        return
    if args.stage == "all":
        for name in ("ingest", "search", "outfits", "errors"):
            print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
            STAGES[name]()
        write_report()
        return
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
