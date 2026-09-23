"""Does Hopit MODA decide a garment's outfit slot from its image, or from the text we send?

Motivating case: a shirt deliberately labelled "bag" (category accessory, subcategory bag) came
back from the Moda styling pane in the accessory slot. That can only happen if the slot is taken
from the attribute text rather than the photograph, so this benchmark ingests the SAME shirt
photo under many different labels and asks /v1/outfits:rank where each copy lands.

What is measured (all against the live MODA API, under a throw-away owner_ref that is deleted at
the end):

  A. the user's case and its neighbours -- a shirt image labelled as a bag, as an accessory
     only, with no attributes, with a nonsense category, uppercase, plural, subcategory only,
     casual_name only, and labelled as another body slot (bottom, footwear);
  B. the mirror image -- a real bag photo labelled as a shirt, and with no attributes;
  C. the user's case inside a realistic pool (a genuine top and a genuine bag competing);
  D. required_slots -- can the ranker be forced to treat the mislabelled shirt as a top;
  E. free-text query steering -- does asking for "a light blue shirt" rescue it;
  F. factor stability -- same image, different text: which score factors move;
  V. vocabulary -- which category words MODA maps to which slot (and which fall to "other");
  P. every distinct (attributes.category, attributes.subcategory) pair our own Stage 3 emits
     across the evaluation roster, so we know which of OUR garments MODA can and cannot slot;
  Q. precedence between category, subcategory and garment_class when they disagree.

Reads garments through this deployment's API acting as the evaluation characters (the same way
scripts/moda_ingest_wardrobes.py does) and never touches our database.

    python -m benchmarks.moda_slot_assignment.run_benchmark --dry-run     # list what would be sent
    python -m benchmarks.moda_slot_assignment.run_benchmark               # full run, ~8 minutes
    python -m benchmarks.moda_slot_assignment.run_benchmark --skip-roster # A-F, V, Q only
    python -m benchmarks.moda_slot_assignment.run_benchmark --keep        # leave the scratch owner
    python -m benchmarks.moda_slot_assignment.run_benchmark --cleanup     # just delete it

Needs MODA_API / MODA_KEY (a key with the `ingest` and `styling` scopes) in .env or the
environment, and reachable stylist logins on CT_APP_BASE (see scripts/moda_ingest_wardrobes.py).
"""

import argparse
import io
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import moda_ingest_wardrobes as m  # noqa: E402  (login, wardrobe, moda(), ATTR_KEYS, APP_BASE)

OWNER = "ctbench_slot_assignment"
RESULTS = os.path.join(HERE, "results.json")
POLL_SECONDS = 3
RANK_CONCURRENCY = 8

# Source photos: one confidently-labelled garment per role from one character's wardrobe.
SOURCE_PERSONA = "shalini_rao"
SOURCES = {
    "shirt": ("TOP", "button_down_shirt"),
    "bag": ("ACCESSORY", "bag"),
    "jeans": ("BOTTOM", "jeans"),
    "boots": ("FOOTWEAR", "boots"),
    "dress": ("ONE_PIECE", "dress"),
    "blazer": ("OUTERWEAR", "blazer"),
}

# V. category words to map. Covers our own vocabularies (garment-level TOP/BOTTOM/..., Stage 3's
# lowercase values, subcategory names) plus casing/plural variants and Indian ethnic wear.
VOCAB = [
    "top", "shirt", "blouse", "tshirt", "t-shirt", "sweater", "cardigan", "tank_top", "vest",
    "bottom", "bottomwear", "pants", "trousers", "jeans", "skirt", "shorts", "leggings", "hosiery",
    "dress", "one_piece", "one-piece", "jumpsuit", "romper", "two_piece_set",
    "outerwear", "outer", "jacket", "coat", "blazer", "suit", "suit_jacket",
    "footwear", "shoes", "boots", "sneakers", "heels", "sandals",
    "accessory", "accessories", "bag", "handbag", "headwear", "hat", "cap", "scarf", "belt",
    "sunglasses", "jewellery", "watch", "gloves",
    "TOP", "Top", "BOTTOM", "ACCESSORY", "BAG", "garment", "other",
    "traditional", "kurta", "saree", "dupatta", "lehenga", "swimwear", "underwear",
]


# --- helpers ---------------------------------------------------------------------------------


def clean(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Exactly what scripts/moda_ingest_wardrobes.py sends: the ATTR_KEYS subset, nulls dropped."""
    return {k: v for k, v in (attrs or {}).items() if k in m.ATTR_KEYS and v is not None}


def pick(wardrobe: List[Dict[str, Any]], category: str, subcategory: str) -> Dict[str, Any]:
    for g in wardrobe:
        a = g.get("attributes") or {}
        if (g.get("category") == category and g.get("subcategory") == subcategory
                and g.get("status") == "COMPLETED" and g.get("canonical_image_url")
                and a.get("confidence", 0) >= 0.9):
            url = g["canonical_image_url"]
            return {"source_garment_id": g["garment_id"],
                    "image_url": url if url.startswith("http") else m.APP_BASE + url,
                    "attributes": a}
    raise SystemExit(f"no confident {category}/{subcategory} in {SOURCE_PERSONA}'s wardrobe")


def roster_pairs(characters: List[Dict[str, Any]]) -> List[Tuple[str, Optional[str], int]]:
    """Every distinct (attributes.category, attributes.subcategory) our Stage 3 emitted, with counts."""
    pairs: Counter = Counter()
    for c in characters:
        for g in m.wardrobe(c):
            a = g.get("attributes") or {}
            if a.get("category"):
                pairs[(a["category"], a.get("subcategory"))] += 1
    return [(k[0], k[1], n) for k, n in pairs.most_common()]


def ingest(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Submit in 200-record batches, poll every job to completion, return per-record states."""
    states: Dict[str, Dict[str, Any]] = {}
    jobs = []
    for i in range(0, len(records), m.BATCH_CAP):
        s, d = m.moda("POST", "/v1/garments:batch", {"garments": records[i:i + m.BATCH_CAP]})
        if s != 202:
            raise SystemExit(f"ingest rejected: HTTP {s} {json.dumps(d)[:400]}")
        jobs.append(d["job_id"])
    t0 = time.time()
    pending = set(jobs)
    while pending and time.time() - t0 < 1200:
        time.sleep(POLL_SECONDS)
        for job in list(pending):
            j = m.moda("GET", f"/v1/jobs/{job}")[1]
            if j.get("state") in ("completed", "failed", "partial"):
                for r in j.get("records") or []:
                    states[r["garment_id"]] = r
                pending.discard(job)
    return {"jobs": jobs, "seconds": round(time.time() - t0, 1), "records": states}


class Rank:
    def __init__(self, gid: Dict[str, str]):
        self.gid = gid
        self.inv = {v: k for k, v in gid.items()}
        self.calls = 0

    def __call__(self, pool: List[str], **kw: Any) -> Dict[str, Any]:
        self.calls += 1
        s, d = m.moda("POST", "/v1/outfits:rank",
                      {"candidates": [self.gid[n] for n in pool], "outfit_count": 3, **kw})
        if s != 200:
            return {"http": s, "body": d}
        # translate garment ids back to variant names so the saved results read on their own
        for o in d.get("outfits", []):
            o["slots"] = {role: self.inv.get(g, g) for role, g in (o.get("slots") or {}).items()}
            o["garment_ids"] = [self.inv.get(g, g) for g in o.get("garment_ids", [])]
        return d

    def slot_of(self, name: str) -> Optional[str]:
        """Which slot MODA gives `name`, probed in four pools so every slot family is observable:
        with jeans+boots (reveals top/dress), with a real top+jeans+boots (reveals extra slots:
        bag/outer/hat/...), with top+boots (reveals bottom), with top+jeans (reveals footwear)."""
        for pool in ([name, "J_jeans", "F_boots"], [name, "S0_correct", "J_jeans", "F_boots"],
                     [name, "S0_correct", "F_boots"], [name, "S0_correct", "J_jeans"]):
            d = self(pool)
            for o in d.get("outfits", []):
                for role, who in o["slots"].items():
                    if who == name:
                        return role
        return None


def describe(d: Dict[str, Any]) -> str:
    if "http" in d:
        return f"HTTP {d['http']} {json.dumps(d['body'])[:160]}"
    if not d.get("outfits"):
        return f"no outfits  warnings={d.get('warnings')}"
    o = d["outfits"][0]
    f = o.get("factors") or {}
    return (f"slots={o['slots']}  score={o.get('score')}  compatibility={f.get('compatibility')}  "
            f"colour_harmony={f.get('colour_harmony')}" + (f"  text_fit={f['text_fit']}" if "text_fit" in f else ""))


# --- main ------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-roster", action="store_true", help="skip step P (all roster category pairs)")
    ap.add_argument("--keep", action="store_true", help="leave the scratch owner in MODA afterwards")
    ap.add_argument("--cleanup", action="store_true", help="delete the scratch owner and exit")
    args = ap.parse_args()

    if not m.MODA_API or not m.MODA_KEY:
        raise SystemExit("MODA_API / MODA_KEY missing (.env or environment)")
    if args.cleanup:
        s, d = m.moda("DELETE", f"/v1/owners/{OWNER}/garments")
        print(f"DELETE /v1/owners/{OWNER}/garments -> {s} {json.dumps(d)}")
        return

    characters = m.collect_characters()
    source_char = next((c for c in characters if c["slug"] == SOURCE_PERSONA), None)
    if not source_char:
        raise SystemExit(f"{SOURCE_PERSONA} not visible -- check stylist credentials / CT_APP_BASE")
    wardrobe = m.wardrobe(source_char)
    img = {name: pick(wardrobe, cat, sub) for name, (cat, sub) in SOURCES.items()}
    shirt, bag = img["shirt"], img["bag"]
    for k, v in img.items():
        print(f"  {k:7s} {v['image_url']}")

    # ---- variants: (name, image, attributes-or-None) ---------------------------------------
    variants: List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]] = [
        ("S0_correct", shirt, clean(shirt["attributes"])),
        ("S1_named_bag", shirt, {**clean(shirt["attributes"]), "category": "accessory", "subcategory": "bag"}),
        ("S2_cat_accessory_only", shirt, {"category": "accessory"}),
        ("S3_sub_bag_only", shirt, {"subcategory": "bag"}),
        ("S4_no_attrs", shirt, None),
        ("S5_cat_top_sub_bag", shirt, {"category": "top", "subcategory": "bag"}),
        ("S6_cat_bag", shirt, {"category": "bag"}),
        ("S7_CAT_UPPER", shirt, {"category": "ACCESSORY", "subcategory": "BAG"}),
        ("S8_cat_nonsense", shirt, {"category": "xyz_nonsense"}),
        ("S9_casual_name_bag_only", shirt, {"casual_name": "bag"}),
        ("S10_cat_bottom", shirt, {"category": "bottom"}),
        ("S11_cat_footwear", shirt, {"category": "footwear"}),
        ("S12_cat_shirt_word", shirt, {"category": "shirt"}),
        ("S13_cat_accessories_plural", shirt, {"category": "accessories"}),
        ("S14_cat_empty_string", shirt, {"category": ""}),
        ("S15_layering_role_accessory_only", shirt, {"layering_role": "accessory"}),
        ("B0_correct", bag, clean(bag["attributes"])),
        ("B1_named_shirt", bag, {**clean(bag["attributes"]), "category": "top", "subcategory": "button_down_shirt"}),
        ("B2_no_attrs", bag, None),
        ("J_jeans", img["jeans"], clean(img["jeans"]["attributes"])),
        ("F_boots", img["boots"], clean(img["boots"]["attributes"])),
        ("D_dress", img["dress"], clean(img["dress"]["attributes"])),
        ("O_blazer", img["blazer"], clean(img["blazer"]["attributes"])),
        # F. same image, text scrambled in every non-category field
        ("T_scrambled_text", shirt, {**clean(shirt["attributes"]), "colour": ["neon green", "red"],
                                     "material": "rubber", "pattern": "floral", "occasion": ["wedding"],
                                     "season": ["winter"], "fit": "oversized", "warmth": 0.95, "formality": 0.99}),
        # Q. precedence when the three role-bearing fields disagree
        ("Q_cat_accessory_sub_bag_gc_SHIRT", shirt, {"category": "accessory", "subcategory": "bag", "garment_class": "SHIRT"}),
        ("Q_sub_bag_gc_SHIRT", shirt, {"subcategory": "bag", "garment_class": "SHIRT"}),
        ("Q_cat_top_gc_BAG", shirt, {"category": "top", "garment_class": "BAG"}),
        ("Q_cat_top_sub_bag_gc_BAG", shirt, {"category": "top", "subcategory": "bag", "garment_class": "BAG"}),
        ("Q_gc_BAG_only", shirt, {"garment_class": "BAG"}),
        ("Q_cat_ACCESSORY_sub_bag_gc_BAG", shirt, {"category": "ACCESSORY", "subcategory": "bag", "garment_class": "BAG"}),
        ("Q_cat_list", shirt, {"category": ["accessory", "bag"]}),
        ("Q_cat_int", shirt, {"category": 3}),
    ]
    for i, word in enumerate(VOCAB):
        variants.append((f"V_{word}", shirt, {"category": word}))
    pairs: List[Tuple[str, Optional[str], int]] = []
    if not args.skip_roster:
        print("  collecting every (category, subcategory) pair the roster's Stage 3 emitted ...")
        pairs = roster_pairs(characters)
        print(f"  {len(pairs)} distinct pairs")
        for cat, sub, _ in pairs:
            attrs = {"category": cat}
            if sub:
                attrs["subcategory"] = sub
            variants.append((f"P_{cat}|{sub}", shirt, attrs))

    gid = {name: f"ctbs_{i:03d}" for i, (name, _, _) in enumerate(variants)}  # unique, case-safe
    records = []
    for name, image, attrs in variants:
        r = {"garment_id": gid[name], "owner_ref": OWNER, "image_url": image["image_url"]}
        if attrs is not None:
            r["attributes"] = attrs
        records.append(r)

    if args.dry_run:
        print(f"\ndry run -- {len(records)} records would be ingested under owner_ref={OWNER}:")
        for name, _, attrs in variants:
            print(f"  {name:40s} {json.dumps(attrs)[:100]}")
        return

    out: Dict[str, Any] = {"owner_ref": OWNER, "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "moda_api": m.MODA_API, "sources": img, "variants": {n: a for n, _, a in variants}}
    try:
        print(f"\n== ingesting {len(records)} records under {OWNER}")
        ing = ingest(records)
        out["ingest"] = {"jobs": ing["jobs"], "seconds": ing["seconds"],
                         "states": Counter(r.get("state") for r in ing["records"].values()),
                         "failed": {gid_: r for gid_, r in ing["records"].items() if r.get("state") not in ("indexed", "unchanged")}}
        print(f"   {dict(out['ingest']['states'])} in {ing['seconds']}s")
        for gid_, r in out["ingest"]["failed"].items():
            print(f"   rejected {[n for n, g in gid.items() if g == gid_][0]}: {r.get('code')} -- {r.get('message')}")

        print("\n== what MODA stored (GET /v1/garments/{id}) for the user's case and the bare image")
        out["stored"] = {}
        for name in ("S0_correct", "S1_named_bag", "S4_no_attrs"):
            s, d = m.moda("GET", f"/v1/garments/{gid[name]}")
            out["stored"][name] = d
            print(f"   {name:16s} sha={str(d.get('image_sha256'))[:12]} vectors={[(v.get('space'), v.get('dim'), v.get('model_version')) for v in d.get('vectors', [])]} attributes={json.dumps(d.get('attributes'))[:90]}")

        rank = Rank(gid)
        model_version = None

        print("\n== A/B. slot given to each single variant (None = never placed in any outfit)")
        names_ab = [n for n, _, _ in variants if n[0] in "SB"]
        with ThreadPoolExecutor(RANK_CONCURRENCY) as ex:
            slots_ab = dict(zip(names_ab, ex.map(rank.slot_of, names_ab)))
        out["A_B_slots"] = slots_ab
        for n in names_ab:
            print(f"   {n:36s} -> {slots_ab[n]}")

        print("\n== C. the user's case inside a realistic pool")
        out["C"] = {}
        for label, pool in [("C1 shirt-as-bag + real bag + jeans + boots (no real top)", ["S1_named_bag", "B0_correct", "J_jeans", "F_boots"]),
                            ("C2 shirt-as-bag + real top + real bag + jeans + boots", ["S1_named_bag", "S0_correct", "B0_correct", "J_jeans", "F_boots"]),
                            ("C3 shirt-as-bag + jeans + boots + blazer + dress", ["S1_named_bag", "J_jeans", "F_boots", "O_blazer", "D_dress"])]:
            d = rank(pool)
            model_version = model_version or d.get("model_version")
            out["C"][label] = d
            print(f"   {label}\n      {describe(d)}")
            for o in d.get("outfits", [])[1:]:
                print(f"      also: slots={o['slots']} score={o.get('score')}")

        print("\n== D. required_slots")
        out["D"] = {}
        for label, pool, req in [("D1 shirt-as-bag + jeans, require top+bottom", ["S1_named_bag", "J_jeans"], ["top", "bottom"]),
                                 ("D2 bare shirt image + jeans, require top+bottom", ["S4_no_attrs", "J_jeans"], ["top", "bottom"]),
                                 ("D3 shirt-as-bag + jeans, require accessory+bottom", ["S1_named_bag", "J_jeans"], ["accessory", "bottom"]),
                                 ("D4 shirt-as-bag + jeans, require bag+bottom", ["S1_named_bag", "J_jeans"], ["bag", "bottom"])]:
            d = rank(pool, required_slots=req)
            out["D"][label] = d
            print(f"   {label}\n      {describe(d)}")

        print("\n== E. free-text query steering")
        out["E"] = {}
        q = "a casual outfit with a light blue shirt"
        for label, pool in [("E1 shirt-as-bag + real bag + jeans", ["S1_named_bag", "B0_correct", "J_jeans"]),
                            ("E2 shirt-as-bag + bare shirt image + jeans", ["S1_named_bag", "S4_no_attrs", "J_jeans"])]:
            d = rank(pool, query=q)
            out["E"][label] = d
            print(f"   {label}, query={q!r}\n      {describe(d)}  interpretation={json.dumps(d.get('query_interpretation'))[:120]}")

        print("\n== F. which factors move when only the text changes (same shirt photo + jeans + boots)")
        out["F"] = {}
        for n in ("S0_correct", "S12_cat_shirt_word", "S5_cat_top_sub_bag", "T_scrambled_text"):
            d = rank([n, "J_jeans", "F_boots"])
            out["F"][n] = d
            print(f"   {n:20s} {describe(d)}")
        for label, pool in [("F5 shirt-as-bottom + real top + boots", ["S10_cat_bottom", "S0_correct", "F_boots"]),
                            ("F6 real jeans + real top + boots", ["J_jeans", "S0_correct", "F_boots"]),
                            ("F7 bag-as-shirt + jeans + boots", ["B1_named_shirt", "J_jeans", "F_boots"])]:
            d = rank(pool)
            out["F"][label] = d
            print(f"   {label}\n      {describe(d)}")
        out["F_query"] = {}
        for qq in ("a formal wedding outfit in neon green", "casual light blue denim shirt look"):
            d = rank(["S0_correct", "T_scrambled_text", "J_jeans", "F_boots"], query=qq)
            out["F_query"][qq] = d
            print(f"   query={qq!r}")
            for o in d.get("outfits", []):
                print(f"      top={o['slots'].get('top'):18s} score={o.get('score')} text_fit={(o.get('factors') or {}).get('text_fit')} colour_harmony={(o.get('factors') or {}).get('colour_harmony')}")

        print("\n== V. category word -> slot")
        names_v = [n for n, _, _ in variants if n.startswith("V_")]
        with ThreadPoolExecutor(RANK_CONCURRENCY) as ex:
            slots_v = dict(zip(names_v, ex.map(rank.slot_of, names_v)))
        out["V_vocab"] = {n[2:]: slots_v[n] for n in names_v}
        by_slot: Dict[str, List[str]] = {}
        for n in names_v:
            by_slot.setdefault(str(slots_v[n]), []).append(n[2:])
        for slot, words in sorted(by_slot.items()):
            print(f"   {slot:10s} <- {', '.join(words)}")

        print("\n== Q. precedence between category / subcategory / garment_class")
        names_q = [n for n, _, _ in variants if n.startswith("Q_")]
        with ThreadPoolExecutor(RANK_CONCURRENCY) as ex:
            slots_q = dict(zip(names_q, ex.map(rank.slot_of, names_q)))
        out["Q_precedence"] = slots_q
        for n in names_q:
            print(f"   {n:36s} -> {slots_q[n]}")

        if pairs:
            print("\n== P. every (attributes.category, attributes.subcategory) pair our Stage 3 emits -> slot")
            names_p = [f"P_{cat}|{sub}" for cat, sub, _ in pairs]
            with ThreadPoolExecutor(RANK_CONCURRENCY) as ex:
                slots_p = dict(zip(names_p, ex.map(rank.slot_of, names_p)))
            out["P_roster_pairs"] = [{"category": cat, "subcategory": sub, "garments": n, "slot": slots_p[f"P_{cat}|{sub}"]}
                                     for cat, sub, n in pairs]
            unslotted = [r for r in out["P_roster_pairs"] if r["slot"] in (None, "other")]
            total = sum(n for _, _, n in pairs)
            lost = sum(r["garments"] for r in unslotted)
            print(f"   {len(pairs)} pairs covering {total} garments; {len(unslotted)} pairs / {lost} garments "
                  f"({100.0 * lost / total:.1f}%) land in 'other' or nowhere:")
            for r in unslotted:
                print(f"     {r['category']:16s} | {str(r['subcategory']):16s} x{r['garments']:<4d} -> {r['slot']}")
            out["P_summary"] = {"pairs": len(pairs), "garments": total, "unslotted_pairs": len(unslotted), "unslotted_garments": lost}

        out["model_version"] = model_version
        out["rank_calls"] = rank.calls
        with io.open(RESULTS, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=str)
        print(f"\nsaved {RESULTS}  ({rank.calls} rank calls, model_version={model_version})")
    finally:
        if args.keep:
            print(f"\n--keep: scratch owner {OWNER} left in MODA (run with --cleanup to delete)")
        else:
            s, d = m.moda("DELETE", f"/v1/owners/{OWNER}/garments")
            print(f"\ncleanup: DELETE /v1/owners/{OWNER}/garments -> {s} {json.dumps(d)}")


if __name__ == "__main__":
    main()
