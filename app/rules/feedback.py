"""Turning *why* someone disliked (or liked) an outfit into evidence the behaviour scorer can use.

A bare 👎 on a four-piece outfit is ambiguous: the scorer (app.rules.wardrobe_behavior) has to
spread the blame evenly and recover the truth statistically over many votes. A reason resolves
it in one vote — "the shoes kill it" blames one garment, "shirt and trousers clash" blames one
pairing, "no florals" is a lesson about an attribute, not about any garment at all.

Two sources of reasons, both handled here:

- **Reason chips** (`REASON_TAGS`): deterministic, free. Each says how much of the vote is about
  the garments themselves versus their combination versus the request — "wrong occasion" is
  barely a taste signal about the pieces, "wouldn't wear these" is entirely one.
- **Free text**: structured by an extractor provider (app.providers.feedback) into a
  `FeedbackExtraction`. The LLM implementation reads the comment as *data*; the heuristic one in
  this module doubles as the mock provider and the fallback when the model call fails.

`derive_vote_weights` combines both into per-garment and per-pair weights for one vote. If a
comment names nothing recognisable, the result is exactly today's even split — text can only
sharpen a vote, never make it worse.
"""

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ----------------------------------------------------------------------------- reason chips

# tag -> (label, polarity it applies to, garment_factor, pair_factor)
# garment_factor scales how much the vote counts against/for the individual pieces;
# pair_factor scales how much it counts against/for their combination.
REASON_TAGS: Dict[str, Dict[str, Any]] = {
    "colour_clash":    {"label": "colour clash",             "polarity": "down", "garment_factor": 0.5, "pair_factor": 1.5},
    "proportions":     {"label": "proportions off",          "polarity": "down", "garment_factor": 0.7, "pair_factor": 1.2},
    "wrong_occasion":  {"label": "wrong for the occasion",   "polarity": "down", "garment_factor": 0.3, "pair_factor": 0.3},
    "too_safe":        {"label": "too safe / boring",        "polarity": "down", "garment_factor": 0.5, "pair_factor": 0.5},
    "too_bold":        {"label": "too bold / loud",          "polarity": "down", "garment_factor": 0.6, "pair_factor": 0.6},
    "wouldnt_wear":    {"label": "wouldn't wear these pieces", "polarity": "down", "garment_factor": 1.2, "pair_factor": 0.6},
    "great_pairing":   {"label": "great pairing",            "polarity": "up",   "garment_factor": 0.6, "pair_factor": 1.5},
    "love_the_pieces": {"label": "love these pieces",        "polarity": "up",   "garment_factor": 1.2, "pair_factor": 0.6},
    "perfect_fit":     {"label": "perfect for them",         "polarity": "up",   "garment_factor": 1.0, "pair_factor": 1.0},
}

# Weight the *un-named* garments keep when a comment blames specific ones: the outfit was still
# rejected, so the rest are not innocent, just not the point.
RESIDUAL_WEIGHT = 0.25
NAMED_PAIR_WEIGHT = 2.0
FACTOR_MIN, FACTOR_MAX = 0.1, 2.0

# Attribute lessons may only touch what the style profile already tracks per value.
LESSON_ATTRIBUTES = ("colour", "pattern", "material", "fit", "silhouette", "sleeve_length", "garment_class")


@dataclass
class FeedbackGarment:
    """What the extractor is allowed to know about one garment in the voted outfit."""
    garment_id: str
    role: str = ""
    category: str = ""
    subcategory: str = ""
    casual_name: str = ""
    colours: Tuple[str, ...] = ()


@dataclass
class FeedbackExtraction:
    blamed_garment_ids: List[str] = field(default_factory=list)
    praised_garment_ids: List[str] = field(default_factory=list)
    pairings: List[Tuple[str, str]] = field(default_factory=list)      # explicitly criticised/praised together
    lessons: List[Dict[str, str]] = field(default_factory=list)         # {"attribute", "value", "polarity"}
    summary: str = ""
    model: str = "none"

    @property
    def is_empty(self) -> bool:
        return not (self.blamed_garment_ids or self.praised_garment_ids or self.pairings or self.lessons)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blamed_garment_ids": list(self.blamed_garment_ids),
            "praised_garment_ids": list(self.praised_garment_ids),
            "pairings": [list(p) for p in self.pairings],
            "lessons": list(self.lessons),
            "summary": self.summary,
            "model": self.model,
        }


def sanitize_extraction(raw: Dict[str, Any], garments: Sequence[FeedbackGarment], model: str) -> FeedbackExtraction:
    """Keeps only ids that are actually in the outfit and lessons on tracked attributes — a model
    can hallucinate either, and the ledger must never carry a reference to a garment that was
    not in the voted outfit."""
    valid = {g.garment_id for g in garments}
    blamed = [g for g in dict.fromkeys(raw.get("blamed_garment_ids") or []) if g in valid]
    praised = [g for g in dict.fromkeys(raw.get("praised_garment_ids") or []) if g in valid and g not in blamed]
    pairings: List[Tuple[str, str]] = []
    for pair in raw.get("pairings") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] in valid and pair[1] in valid and pair[0] != pair[1]:
            key = tuple(sorted((str(pair[0]), str(pair[1]))))
            if key not in pairings:
                pairings.append(key)  # type: ignore[arg-type]
    lessons: List[Dict[str, str]] = []
    for lesson in raw.get("lessons") or []:
        if not isinstance(lesson, dict):
            continue
        attribute = str(lesson.get("attribute", "")).lower().strip()
        value = str(lesson.get("value", "")).lower().strip()
        polarity = str(lesson.get("polarity", "down")).lower().strip()
        if attribute in LESSON_ATTRIBUTES and value and polarity in ("up", "down"):
            lessons.append({"attribute": attribute, "value": value, "polarity": polarity})
    return FeedbackExtraction(
        blamed_garment_ids=blamed,
        praised_garment_ids=praised,
        pairings=pairings,
        lessons=lessons[:8],
        summary=str(raw.get("summary") or "")[:300],
        model=model,
    )


# ----------------------------------------------------------------------------- heuristic extractor

_NEG = {"hate", "hates", "dislike", "wrong", "clash", "clashes", "clashing", "ugly", "kill", "kills", "off",
        "bad", "terrible", "awful", "mismatch", "mismatched", "cheap", "boring", "drab", "weird", "odd",
        "ruins", "ruin", "no", "never", "worst", "meh", "dated", "frumpy", "loud", "garish", "not"}
_POS = {"love", "loves", "great", "nice", "works", "work", "perfect", "good", "beautiful", "lovely", "gorgeous",
        "fab", "fabulous", "keep", "chic", "elegant", "stunning", "best", "like", "likes", "amazing"}
_NEGATORS = {"not", "don't", "dont", "doesn't", "doesnt", "isn't", "isnt", "never", "no"}
_PAIR_CUES = {"with", "and", "together", "clash", "clashes", "clashing", "combo", "combination", "pair", "pairing", "+"}

_LESSON_VOCAB: Dict[str, Tuple[str, ...]] = {
    "pattern": ("floral", "florals", "flowery", "stripes", "striped", "stripe", "plaid", "check", "checks", "checked",
                "polka", "animal", "leopard", "paisley", "geometric", "print", "prints", "printed", "solid", "plain"),
    "colour": ("black", "white", "navy", "blue", "grey", "gray", "beige", "brown", "olive", "green", "red", "pink",
               "yellow", "orange", "purple", "cream", "maroon", "burgundy", "teal", "mustard", "neon"),
    "material": ("linen", "denim", "leather", "silk", "wool", "cotton", "satin", "velvet", "polyester", "chiffon",
                 "knit", "lace", "suede", "corduroy"),
    "fit": ("oversized", "baggy", "tight", "slim", "cropped", "loose", "fitted", "boxy", "skinny", "wide-leg", "flared"),
}
_LESSON_VALUE_NORMALISE = {"florals": "floral", "flowery": "floral", "striped": "stripes", "stripe": "stripes",
                           "checks": "checked", "check": "checked", "prints": "print", "printed": "print", "gray": "grey"}
_CATEGORY_WORDS = {
    "FOOTWEAR": ("shoes", "shoe", "footwear", "sneakers", "heels", "loafers", "boots", "sandals", "flats", "trainers"),
    "TOP": ("top", "shirt", "blouse", "tee", "t-shirt", "tshirt", "kurta", "sweater", "jumper", "hoodie"),
    "BOTTOM": ("bottom", "bottoms", "trousers", "pants", "jeans", "skirt", "shorts", "chinos", "palazzo"),
    "ONE_PIECE": ("dress", "jumpsuit", "gown", "saree", "sari", "lehenga", "romper"),
    "OUTERWEAR": ("jacket", "blazer", "coat", "cardigan", "outerwear", "shrug", "overcoat"),
    "ACCESSORY": ("bag", "belt", "scarf", "hat", "watch", "jewellery", "jewelry", "accessory", "accessories", "sunglasses"),
}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\-\+']+", text.lower())


def _garment_terms(g: FeedbackGarment) -> Set[str]:
    terms: Set[str] = set()
    for source in (g.subcategory, g.casual_name):
        terms.update(t for t in re.split(r"[\s_\-]+", (source or "").lower()) if len(t) > 2)
    terms.update(c.lower() for c in g.colours if c)
    terms.update(_CATEGORY_WORDS.get((g.category or "").upper(), ()))
    return terms


def _sentence_polarity(tokens: List[str], default: str) -> str:
    neg = sum(1 for t in tokens if t in _NEG)
    pos = sum(1 for t in tokens if t in _POS)
    # "don't like", "not great": a negator flips the positive words in the sentence.
    if any(t in _NEGATORS for t in tokens) and pos:
        neg += pos
        pos = 0
    if neg == 0 and pos == 0:
        return default
    return "down" if neg >= pos else "up"


def heuristic_extract(comment: str, vote: str, garments: Sequence[FeedbackGarment]) -> FeedbackExtraction:
    """Keyword attribution. Deliberately conservative: it only names a garment when the comment
    uses one of that garment's own words (subcategory, casual name, colour, or its category's
    everyday synonyms), and only records a lesson for vocabulary the style profile tracks."""
    comment = (comment or "").strip()
    if not comment:
        return FeedbackExtraction(model="heuristic-v1")
    terms_by_id = {g.garment_id: _garment_terms(g) for g in garments}
    blamed: List[str] = []
    praised: List[str] = []
    pairings: List[Tuple[str, str]] = []
    lessons: List[Dict[str, str]] = []

    # Clauses, not just sentences: "love the shirt, the rest is drab" carries two opinions.
    for sentence in re.split(r"[.!?;,\n]+|\s+but\s+", comment, flags=re.IGNORECASE):
        tokens = _tokens(sentence)
        if not tokens:
            continue
        token_set = set(tokens)
        polarity = _sentence_polarity(tokens, default=vote)
        mentioned = [gid for gid, terms in terms_by_id.items() if terms & token_set]
        # A colour word alone is too ambiguous to pin on a garment when several share it.
        target = blamed if polarity == "down" else praised
        for gid in mentioned:
            if gid not in target:
                target.append(gid)
        if polarity == "down" and len(mentioned) >= 2 and token_set & _PAIR_CUES:
            for a, b in combinations(mentioned[:3], 2):
                pairings.append(tuple(sorted((a, b))))  # type: ignore[arg-type]
        for attribute, vocab in _LESSON_VOCAB.items():
            for word in vocab:
                if word in token_set:
                    value = _LESSON_VALUE_NORMALISE.get(word, word)
                    if not any(l["attribute"] == attribute and l["value"] == value for l in lessons):
                        lessons.append({"attribute": attribute, "value": value, "polarity": polarity})

    praised = [g for g in praised if g not in blamed]
    parts = []
    if blamed:
        parts.append(f"blamed {len(blamed)} garment(s)")
    if praised:
        parts.append(f"praised {len(praised)}")
    if pairings:
        parts.append(f"{len(pairings)} pairing(s) named")
    if lessons:
        parts.append("lessons: " + ", ".join(f"{l['polarity']} {l['value']}" for l in lessons[:3]))
    return sanitize_extraction(
        {"blamed_garment_ids": blamed, "praised_garment_ids": praised, "pairings": pairings, "lessons": lessons,
         "summary": "; ".join(parts)},
        garments, model="heuristic-v1",
    )


# ----------------------------------------------------------------------------- weights for one vote

@dataclass
class VoteWeights:
    garment_weights: Dict[str, float]          # garment_id -> multiplier on the vote's own polarity
    pair_weights: Dict[str, float]             # "a|b" (sorted) -> multiplier
    counter_garment_ids: List[str]             # garments that get the *opposite* polarity (praise inside a 👎)

    def to_dict(self) -> Dict[str, Any]:
        return {"garment_weights": self.garment_weights, "pair_weights": self.pair_weights, "counter_garment_ids": self.counter_garment_ids}


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def _clamp_factor(x: float) -> float:
    return max(FACTOR_MIN, min(FACTOR_MAX, x))


def derive_vote_weights(
    vote: str,
    garment_ids: Iterable[str],
    tags: Iterable[str] = (),
    extraction: Optional[FeedbackExtraction] = None,
) -> VoteWeights:
    ids = list(dict.fromkeys(garment_ids))
    garment_factor = 1.0
    pair_factor = 1.0
    for tag in tags:
        spec = REASON_TAGS.get(str(tag))
        if not spec or spec["polarity"] not in (vote, "any"):
            continue
        garment_factor *= spec["garment_factor"]
        pair_factor *= spec["pair_factor"]
    garment_factor, pair_factor = _clamp_factor(garment_factor), _clamp_factor(pair_factor)

    weights = {g: garment_factor for g in ids}
    counter: List[str] = []
    named_pairs: Set[str] = set()
    if extraction is not None:
        same_polarity = extraction.blamed_garment_ids if vote == "down" else extraction.praised_garment_ids
        opposite = extraction.praised_garment_ids if vote == "down" else extraction.blamed_garment_ids
        same_polarity = [g for g in same_polarity if g in weights]
        opposite = [g for g in opposite if g in weights and g not in same_polarity]
        if same_polarity:
            for g in ids:
                weights[g] = garment_factor * (1.0 if g in same_polarity else RESIDUAL_WEIGHT)
        for g in opposite:
            weights[g] = 0.0
            counter.append(g)
        named_pairs = {pair_key(a, b) for a, b in extraction.pairings if a in weights and b in weights}

    pair_weights: Dict[str, float] = {}
    for a, b in combinations(ids, 2):
        key = pair_key(a, b)
        if key in named_pairs:
            pair_weights[key] = _clamp_factor(NAMED_PAIR_WEIGHT * pair_factor)
        elif weights[a] == 0.0 or weights[b] == 0.0:
            pair_weights[key] = 0.0  # a praised piece is not part of the disliked combination
        else:
            pair_weights[key] = round(pair_factor * (weights[a] + weights[b]) / 2.0, 4)
    return VoteWeights(garment_weights={g: round(w, 4) for g, w in weights.items()}, pair_weights=pair_weights, counter_garment_ids=counter)


def describe_feedback(tags: Iterable[str], extraction: Optional[FeedbackExtraction], labels: Dict[str, str]) -> str:
    """One line for the UI: what the reason was understood to mean."""
    parts = [REASON_TAGS[t]["label"] for t in tags if t in REASON_TAGS]
    if extraction is not None:
        if extraction.blamed_garment_ids:
            parts.append("blamed " + ", ".join(labels.get(g, g) for g in extraction.blamed_garment_ids))
        if extraction.praised_garment_ids:
            parts.append("praised " + ", ".join(labels.get(g, g) for g in extraction.praised_garment_ids))
        if extraction.pairings:
            parts.append("pairing: " + "; ".join(" + ".join(labels.get(g, g) for g in p) for p in extraction.pairings))
        if extraction.lessons:
            parts.append(", ".join(f"{'no' if l['polarity'] == 'down' else 'more'} {l['value']}" for l in extraction.lessons[:3]))
    return " · ".join(parts)
