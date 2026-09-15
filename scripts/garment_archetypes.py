"""Derive garment-level archetypes from a wardrobe, then describe members by their mix.

The unit of analysis is the GARMENT, not the member. Two independent passes run over the same
wardrobe and are then crossed against each other:

  Layer A - rule-based archetypes. Interpretable predicates over the normalised attributes
            (role, formality, colour family, material class, pattern class, warmth). Every
            garment lands in exactly one archetype, so the buckets are mutually exclusive and
            a member's mix sums to 100%.

  Layer B - embedding clusters. Spherical k-means over the stored 768-d SigLIP vectors. This
            sees only pixels, so it is completely immune to the free-text taxonomy problem
            (production carries 76 spellings of `category` and 33 of `layering_role`). Where a
            Layer B cluster lines up with a Layer A archetype, the archetype is real and not an
            artefact of the normalisation map.

  Layer C - per-member distribution over Layer A archetypes.

READ ONLY. Every connection sets `default_transaction_read_only`, and this script issues
nothing but SELECT. It never writes to the database it reads.

    python -m scripts.garment_archetypes --dsn "$PROD_DSN" --clusters 14 --out reports/
"""

import argparse
import asyncio
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import numpy as np

# --------------------------------------------------------------------------------------
# Normalisation. Production stores these as free text, so every lookup is lowercased and
# matched on substrings rather than equality - `Base Layer`, `base layer` and `base` are one
# value, and `T-Shirt`, `Top` and `top` are one role.
# --------------------------------------------------------------------------------------

ONE_PIECE = ("dress", "jumpsuit", "romper", "bodysuit", "gown", "anarkali", "saree", "sari",
             "lehenga", "overall", "tunic", "kaftan", "set", "suit")
OUTER = ("outerwear", "jacket", "coat", "blazer", "cardigan", "hoodie", "sweatshirt", "vest",
         "parka", "trench", "shrug", "poncho", "windbreaker")
BOTTOM = ("bottom", "pant", "trouser", "jean", "short", "skirt", "legging", "chino", "culotte",
          "palazzo", "churidar", "salwar", "pyjama", "pajama", "capri", "joggers")
FOOTWEAR = ("footwear", "shoe", "sneaker", "boot", "heel", "sandal", "flat", "loafer", "slipper",
            "mule", "wedge", "pump", "espadrille")
ACCESSORY = ("accessory", "bag", "scarf", "hat", "headwear", "belt", "jewel", "watch", "sunglass",
             "glove", "tie", "clutch", "purse", "backpack", "dupatta", "stole", "shawl", "cap")
TOP = ("top", "shirt", "tee", "t-shirt", "blouse", "kurta", "kurti", "camisole", "tank", "crop",
       "polo", "sweater", "knit", "jersey", "bralette", "bustier")
NON_GARMENT = ("storage", "interior", "blanket", "towel", "hair extension", "none", "null",
               "uncategorized", "other", "garment", "clothing", "outfit", "unknown")

ETHNIC = ("kurta", "kurti", "saree", "sari", "lehenga", "anarkali", "dupatta", "sherwani",
          "salwar", "churidar", "ethnic", "traditional", "bandhani", "kanjivaram", "palazzo set",
          "indo", "chudidar", "ghagra", "choli", "angrakha", "nehru")

NEUTRAL = ("black", "white", "grey", "gray", "cream", "beige", "off-white", "ivory", "tan",
           "charcoal", "stone")
EARTH = ("brown", "khaki", "olive", "rust", "camel", "mustard", "bronze", "taupe", "sand",
         "terracotta", "maroon", "burgundy")
COOL = ("blue", "navy", "teal", "green", "mint", "aqua", "turquoise", "denim blue", "indigo")
WARM = ("red", "orange", "pink", "coral", "yellow", "peach", "magenta", "fuchsia", "purple",
        "lavender", "violet")
PASTEL = ("light pink", "light blue", "light yellow", "light green", "light purple", "pastel",
          "blush", "powder", "mint green", "lilac")
METALLIC = ("gold", "silver", "metallic", "shiny", "bronze", "rose gold")
MULTI = ("multicolor", "multicolour", "multicolored", "multicoloured", "multi", "rainbow")

LUXE_MATERIAL = ("satin", "silk", "velvet", "lace", "chiffon", "sequin", "brocade", "organza",
                 "tulle", "suede", "faux fur", "fur", "georgette", "crepe")
KNIT_MATERIAL = ("knit", "wool", "fleece", "cashmere", "jersey", "rib")
DENIM_MATERIAL = ("denim",)
LEATHER_MATERIAL = ("leather", "faux leather", "pu")
NATURAL_MATERIAL = ("cotton", "linen", "canvas", "viscose", "rayon", "bamboo", "hemp")
SYNTH_MATERIAL = ("synthetic", "polyester", "nylon", "spandex", "lycra", "acrylic", "mesh")

SOLID_PATTERN = ("solid", "plain", "none", "no pattern")
EMBELLISHED_PATTERN = ("embroider", "embellish", "sequin", "bead", "applique", "lace", "monogram",
                       "quilted", "studded")
TEXTURE_PATTERN = ("textured", "woven", "ribbed", "knitted", "cable", "waffle", "pleated")

# Occasion -> formality, 0 (loungewear) to 4 (ceremonial). A garment's formality is the MEAN of
# every tag that matches, not the max. Occasion is multi-valued and 91% of the wardrobe carries
# `casual`, so `["casual", "formal"]` - the single most common pair, on 951 garments - means "works
# either way", i.e. smart casual. Taking the max read that as formal and over-dressed 39% of the
# wardrobe into the evening bucket.
FORMALITY = {
    0.0: ("loungewear", "lounge", "sleep", "pyjama", "pajama", "home"),
    0.5: ("athletic", "sport", "active", "gym", "workout", "beach", "pool", "swim", "vacation",
          "resort"),
    1.0: ("casual", "everyday", "daywear", "day out", "weekend", "streetwear", "outdoor",
          "travel", "summer", "winter"),
    2.0: ("smart casual", "business casual", "brunch", "date", "semi-formal", "semi formal",
          "work", "office"),
    3.0: ("formal", "business", "evening", "night out", "dinner"),
    4.0: ("party", "festive", "wedding", "celebration", "ceremonial", "cocktail", "gala",
          "reception"),
}


def _txt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _has(haystack: str, needles: Tuple[str, ...]) -> bool:
    return any(n in haystack for n in needles)


def normalise_role(category: str, subcategory: str, layering: str) -> str:
    """Role is resolved most-specific-first: a `dress` is ONE_PIECE even though its category
    string may also contain `top`."""
    blob = f"{subcategory} {category}".lower()
    if _has(blob, NON_GARMENT) and not _has(blob, TOP + BOTTOM + ONE_PIECE + OUTER):
        return "NON_GARMENT"
    for needles, role in (
        (FOOTWEAR, "FOOTWEAR"),
        (ACCESSORY, "ACCESSORY"),
        (ONE_PIECE, "ONE_PIECE"),
        (OUTER, "OUTER"),
        (BOTTOM, "BOTTOM"),
        (TOP, "TOP"),
    ):
        if _has(blob, needles):
            return role
    lay = layering.lower()
    if "outer" in lay:
        return "OUTER"
    if "bottom" in lay:
        return "BOTTOM"
    if _has(lay, ("base", "top", "mid")):
        return "TOP"
    return "UNKNOWN"


def colour_family(colours: Any) -> str:
    blob = _txt(colours).lower()
    if not blob:
        return "UNKNOWN"
    if _has(blob, MULTI):
        return "MULTI"
    if _has(blob, PASTEL):
        return "PASTEL"
    if _has(blob, METALLIC):
        return "METALLIC"
    # Neutral only counts when nothing more saturated is present, so a "black and red" piece
    # reads as WARM rather than being flattened into the neutral bucket.
    saturated = [(WARM, "WARM"), (COOL, "COOL"), (EARTH, "EARTH")]
    for needles, name in saturated:
        if _has(blob, needles):
            return name
    if _has(blob, NEUTRAL):
        return "NEUTRAL"
    return "OTHER"


def material_class(material: str) -> str:
    m = (material or "").lower()
    for needles, name in (
        (LUXE_MATERIAL, "LUXE"),
        (DENIM_MATERIAL, "DENIM"),
        (LEATHER_MATERIAL, "LEATHER"),
        (KNIT_MATERIAL, "KNIT"),
        (NATURAL_MATERIAL, "NATURAL"),
        (SYNTH_MATERIAL, "SYNTHETIC"),
    ):
        if _has(m, needles):
            return name
    return "UNKNOWN"


def pattern_class(pattern: str) -> str:
    p = (pattern or "").lower()
    if not p or _has(p, SOLID_PATTERN):
        return "SOLID"
    if _has(p, EMBELLISHED_PATTERN):
        return "EMBELLISHED"
    if _has(p, TEXTURE_PATTERN):
        return "TEXTURE"
    return "PRINT"


def formality_of(occasions: Any) -> float:
    """Mean of every formality band the garment's occasion tags touch - see the note above."""
    tags = occasions if isinstance(occasions, list) else [occasions] if occasions else []
    scores = []
    for tag in tags:
        t = str(tag).lower().strip()
        if not t:
            continue
        matched = [score for score, needles in FORMALITY.items() if _has(t, needles)]
        if matched:
            scores.append(max(matched))   # one tag maps to one band
    return float(np.mean(scores)) if scores else 1.0


def is_ethnic(category: str, subcategory: str, description: str) -> bool:
    return _has(f"{subcategory} {category} {description}".lower(), ETHNIC)


def _f(value: Any, default: float = 0.5) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if 0.0 <= f <= 1.0 else default


# --------------------------------------------------------------------------------------
# Layer A - the archetypes themselves. Order matters: the first predicate that matches wins,
# so the most specific archetypes are tested before the general ones.
# --------------------------------------------------------------------------------------

def assign_archetype(g: Dict[str, Any]) -> str:
    role, fam, mat, pat = g["role"], g["colour_family"], g["material"], g["pattern"]
    formality, warmth, versatility = g["formality"], g["warmth"], g["versatility"]

    if role == "NON_GARMENT":
        return "Not a garment"
    if g["ethnic"]:
        return "Ethnic & occasion wear"
    if role == "FOOTWEAR":
        return "Footwear"
    if role == "ACCESSORY":
        return "Accessories & bags"
    if formality >= 3.0 or (formality >= 2.5 and (mat == "LUXE" or pat == "EMBELLISHED")):
        return "Evening & celebration"
    if formality <= 0.5:
        return "Lounge, active & resort"
    if role == "OUTER":
        return "Outer layers & shells"
    if mat == "DENIM":
        return "Denim staples"
    if mat in ("KNIT", "LEATHER") or warmth >= 0.65:
        return "Knitwear & warm layers"
    if mat == "LUXE" or pat == "EMBELLISHED":
        return "Luxe fabrics & embellishment"
    if pat in ("PRINT", "TEXTURE") or fam in ("WARM", "MULTI", "METALLIC"):
        return "Statement prints & colour"
    if fam in ("NEUTRAL", "EARTH") and pat == "SOLID" and versatility >= 0.6:
        return "Neutral workhorses"
    return "Everyday separates"


ARCHETYPE_ORDER = [
    "Neutral workhorses",
    "Everyday separates",
    "Statement prints & colour",
    "Denim staples",
    "Knitwear & warm layers",
    "Outer layers & shells",
    "Luxe fabrics & embellishment",
    "Evening & celebration",
    "Ethnic & occasion wear",
    "Lounge, active & resort",
    "Footwear",
    "Accessories & bags",
    "Not a garment",
]


# --------------------------------------------------------------------------------------
# Layer B - spherical k-means. The stored vectors are unit-norm, so cosine similarity is a
# plain dot product and the centroid update is a mean followed by a re-normalisation.
# --------------------------------------------------------------------------------------

def spherical_kmeans(x: np.ndarray, k: int, iters: int = 60, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]

    # k-means++ seeding on cosine distance, so the initial centroids are spread out rather
    # than clumped - a random start on fashion embeddings reliably collapses several clusters.
    centroids = [x[rng.integers(n)]]
    for _ in range(k - 1):
        sims = np.max(x @ np.array(centroids).T, axis=1)
        dist = np.clip(1.0 - sims, 1e-12, None) ** 2
        centroids.append(x[rng.choice(n, p=dist / dist.sum())])
    c = np.array(centroids)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        new_labels = np.argmax(x @ c.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = x[labels == j]
            if len(members) == 0:
                c[j] = x[rng.integers(n)]          # re-seed an emptied cluster
                continue
            v = members.mean(axis=0)
            norm = np.linalg.norm(v)
            c[j] = v / norm if norm else x[rng.integers(n)]
    return labels, c


def top_counts(values: List[str], limit: int = 5) -> List[str]:
    return [f"{v} {n}" for v, n in Counter(v for v in values if v).most_common(limit)]


# --------------------------------------------------------------------------------------

async def load(dsn: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    conn = await asyncpg.connect(dsn, ssl="require")
    try:
        await conn.execute("SET default_transaction_read_only = on")
        rows = await conn.fetch("""
            SELECT g.id::text, g.user_id::text, g.category, g.subcategory, g.title,
                   g.attributes::text AS attrs, e.embedding::text AS emb
            FROM garments g
            LEFT JOIN garment_embeddings e ON e.garment_id = g.id
            WHERE g.archived_at IS NULL AND g.status = 'active'
        """)
        users = await conn.fetch("SELECT id::text, name, email FROM users")
    finally:
        await conn.close()

    names = {u["id"]: f'{u["name"] or "?"} <{u["email"] or "?"}>' for u in users}

    garments = []
    for r in rows:
        try:
            a = json.loads(r["attrs"]) if r["attrs"] else {}
        except json.JSONDecodeError:
            a = {}
        category = _txt(a.get("category") or r["category"])
        subcategory = _txt(a.get("subcategory") or r["subcategory"])
        g = {
            "id": r["id"],
            "user_id": r["user_id"],
            "title": r["title"] or a.get("display_name") or subcategory,
            "category": category,
            "subcategory": subcategory,
            "role": normalise_role(category, subcategory, _txt(a.get("layering_role"))),
            "colour_family": colour_family(a.get("colour")),
            "colour_raw": _txt(a.get("colour")).lower(),
            "material": material_class(_txt(a.get("material"))),
            "material_raw": _txt(a.get("material")).lower(),
            "pattern": pattern_class(_txt(a.get("pattern"))),
            "pattern_raw": _txt(a.get("pattern")).lower(),
            "formality": formality_of(a.get("occasion")),
            "occasion_raw": _txt(a.get("occasion")).lower(),
            "season_raw": _txt(a.get("season")).lower(),
            "warmth": _f(a.get("warmth")),
            "versatility": _f(a.get("versatility")),
            "ethnic": is_ethnic(category, subcategory, _txt(a.get("display_name"))),
            "embedding": r["emb"],
        }
        g["archetype"] = assign_archetype(g)
        garments.append(g)
    return garments, names


def parse_vec(raw: Optional[str]) -> Optional[np.ndarray]:
    if not raw:
        return None
    try:
        v = np.array(json.loads(raw), dtype=np.float32)
    except (json.JSONDecodeError, ValueError):
        return None
    n = np.linalg.norm(v)
    return v / n if n else None


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=os.getenv("PROD_DSN"), help="postgres DSN (read-only user)")
    ap.add_argument("--clusters", type=int, default=14)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    if not args.dsn:
        raise SystemExit("--dsn or PROD_DSN is required")

    garments, names = await load(args.dsn)
    print(f"loaded {len(garments)} active garments across {len({g['user_id'] for g in garments})} members")

    # ---- Layer A ----
    by_arch: Dict[str, List[Dict]] = defaultdict(list)
    for g in garments:
        by_arch[g["archetype"]].append(g)

    layer_a = []
    for name in ARCHETYPE_ORDER:
        items = by_arch.get(name, [])
        if not items:
            continue
        layer_a.append({
            "archetype": name,
            "count": len(items),
            "share": round(100.0 * len(items) / len(garments), 1),
            "roles": top_counts([g["role"] for g in items]),
            "colours": top_counts([g["colour_family"] for g in items]),
            "materials": top_counts([g["material"] for g in items]),
            "patterns": top_counts([g["pattern"] for g in items]),
            "subcategories": top_counts([g["subcategory"] for g in items], 6),
            "mean_formality": round(float(np.mean([g["formality"] for g in items])), 2),
            "mean_warmth": round(float(np.mean([g["warmth"] for g in items])), 2),
            "mean_versatility": round(float(np.mean([g["versatility"] for g in items])), 2),
            "members_holding": len({g["user_id"] for g in items}),
        })

    # ---- Layer B ----
    vectors, indexed = [], []
    for i, g in enumerate(garments):
        v = parse_vec(g["embedding"])
        if v is not None and v.shape[0] == 768:
            vectors.append(v)
            indexed.append(i)
    layer_b = []
    if len(vectors) >= args.clusters:
        x = np.vstack(vectors)
        labels, centroids = spherical_kmeans(x, args.clusters)
        print(f"clustered {len(vectors)} embeddings into {args.clusters} groups")
        for j in range(args.clusters):
            member_idx = [indexed[p] for p in np.where(labels == j)[0]]
            items = [garments[i] for i in member_idx]
            if not items:
                continue
            rows = x[labels == j]
            cohesion = float(np.mean(rows @ centroids[j]))
            # The garments nearest the centroid are the cluster's own description of itself.
            sims = rows @ centroids[j]
            exemplars = [items[p]["title"] for p in np.argsort(-sims)[:5]]
            layer_b.append({
                "cluster": j,
                "count": len(items),
                "share": round(100.0 * len(items) / len(vectors), 1),
                "cohesion": round(cohesion, 3),
                "archetypes": top_counts([g["archetype"] for g in items], 4),
                "roles": top_counts([g["role"] for g in items], 3),
                "colours": top_counts([g["colour_family"] for g in items], 3),
                "subcategories": top_counts([g["subcategory"] for g in items], 5),
                "exemplars": exemplars,
                "members_holding": len({g["user_id"] for g in items}),
            })
        layer_b.sort(key=lambda c: -c["count"])

    # ---- Layer C ----
    per_member: Dict[str, Counter] = defaultdict(Counter)
    for g in garments:
        per_member[g["user_id"]][g["archetype"]] += 1
    layer_c = []
    for user_id, counter in per_member.items():
        total = sum(counter.values())
        if total < 5:
            continue
        items = [g for g in garments if g["user_id"] == user_id]
        layer_c.append({
            "member": names.get(user_id, user_id[:8]),
            "garments": total,
            "mix": {a: round(100.0 * counter[a] / total, 1) for a in ARCHETYPE_ORDER if counter[a]},
            "dominant": counter.most_common(1)[0][0],
            "mean_formality": round(float(np.mean([g["formality"] for g in items])), 2),
            "mean_versatility": round(float(np.mean([g["versatility"] for g in items])), 2),
            "ethnic_share": round(100.0 * sum(1 for g in items if g["ethnic"]) / total, 1),
            "top_colours": top_counts([g["colour_family"] for g in items], 3),
        })
    layer_c.sort(key=lambda m: -m["garments"])

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "garment_archetypes.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"layer_a": layer_a, "layer_b": layer_b, "layer_c": layer_c,
                   "totals": {"garments": len(garments), "embedded": len(vectors)}}, fh, indent=1)
    print(f"wrote {path}")

    print("\n=== LAYER A - garment archetypes ===")
    for a in layer_a:
        print(f"  {a['archetype']:<28} {a['count']:>5}  {a['share']:>5}%  "
              f"form={a['mean_formality']:.2f} vers={a['mean_versatility']:.2f}  "
              f"held by {a['members_holding']} members")
    print("\n=== LAYER B - embedding clusters ===")
    for c in layer_b:
        print(f"  #{c['cluster']:<3} {c['count']:>5} ({c['share']:>5}%) coh={c['cohesion']:.3f}  "
              f"{', '.join(c['archetypes'][:2])}  |  {', '.join(c['subcategories'][:3])}")


if __name__ == "__main__":
    asyncio.run(main())
