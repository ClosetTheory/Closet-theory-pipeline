"""Wardrobe Behaviour Scoring (Styling Pipeline Stage 3).

A two-level Bayesian reputation model over the member's vote ledger (app.models.outfit_vote).
Every 👍/👎 on an outfit is split into evidence for each garment in it and for each unordered
garment *pair* in it, so the pipeline can tell two very different failures apart:

- a **bad garment**: disliked again and again next to *different* partners, and
- a **bad pairing**: two garments that keep getting disliked *together* although each one is
  fine elsewhere.

Scoring an entity (garment or pair). With L and D the time-decayed like/dislike weights:

    s = (L - D) / (L + D + prior)            in (-1, 1)

The pseudo-count prior keeps one unlucky vote from burying anything — the same shrinkage
trick app.rules.style_profile already uses for attribute affinities. A 60-day half-life lets
taste change.

Garment penalty with context diversity (credit assignment). A dislike of a four-piece outfit is
not proof that all four pieces are bad. So a garment's like/dislike evidence is scaled by how
many *distinct partners* it was liked/disliked alongside: five dislikes next to the same
trousers barely move the garment (that is the trousers-pairing's fault), five dislikes next
to five different partners do.

Pair penalty as a residual. The pair's own score minus the mean of its two garments' scores is
what "these two together" contributes beyond each one's reputation — that residual is what
Stage 6 (compatibility) consumes, and a pair with three or more confident dislikes is skipped
outright before any VLM call is spent on it.

Cold start is a neutral 0.5 for every garment, exactly like the old stub, and a small
exploration floor stops any garment from being buried forever.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from app.models.garment import Garment

NEUTRAL_BEHAVIOR_SCORE = 0.5
EXPLORATION_FLOOR = 0.15  # no garment scores below this — it can always come back

HALF_LIFE_DAYS = 60.0
GARMENT_PRIOR = 3.0   # pseudo-votes shrinking a garment's score toward neutral
PAIR_PRIOR = 2.0      # pseudo-votes shrinking a pair's score toward neutral
PARTNER_PRIOR = 3.0   # distinct partners needed before a garment's evidence is ~trusted

# A pair is skipped outright once it has this many raw dislikes AND its shrunk score is this low.
# Three pure dislikes and no likes give exactly -3 / (3 + PAIR_PRIOR) = -0.6.
HARD_REJECT_MIN_DISLIKES = 3
HARD_REJECT_SCORE = -0.6


@dataclass(frozen=True)
class BehaviorVote:
    """One ledger row, detached from SQLAlchemy so the scorer is pure and unit-testable."""

    garment_ids: Tuple[str, ...]
    vote: str            # "up" | "down"
    weight: float
    created_at: datetime


def _decay(created_at: datetime, now: datetime) -> float:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def _diversity(partner_count: int) -> float:
    """How much to trust evidence spread over this many distinct partners. A single-garment
    outfit has no partners but its vote should still count, hence the +1."""
    n = partner_count + 1
    return n / (n + PARTNER_PRIOR)


def _shrunk(likes: float, dislikes: float, prior: float) -> float:
    total = likes + dislikes
    if total <= 0:
        return 0.0
    return (likes - dislikes) / (total + prior)


class WardrobeBehaviorModel:
    """Per-member learned scores. Query methods never raise on unknown garments — they return
    the neutral value, so the pipeline behaves exactly as before for a member with no votes."""

    def __init__(
        self,
        garment_signed: Dict[str, float],
        pair_residuals: Dict[FrozenSet[str], float],
        hard_reject_pairs: Set[FrozenSet[str]],
        garment_detail: Dict[str, Dict[str, Any]],
        pair_detail: Dict[FrozenSet[str], Dict[str, Any]],
        votes_considered: int,
    ):
        self._garment_signed = garment_signed
        self._pair_residuals = pair_residuals
        self._hard_reject_pairs = hard_reject_pairs
        self.garment_detail = garment_detail
        self.pair_detail = pair_detail
        self.votes_considered = votes_considered

    # --- queries ------------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return self.votes_considered == 0

    def garment_signed_score(self, garment_id: str) -> float:
        return self._garment_signed.get(garment_id, 0.0)

    def garment_score(self, garment_id: str) -> float:
        """[EXPLORATION_FLOOR, 1.0]; 0.5 is neutral."""
        score = 0.5 + 0.5 * self.garment_signed_score(garment_id)
        return max(EXPLORATION_FLOOR, min(1.0, score))

    def pair_residual(self, a: str, b: str) -> float:
        return self._pair_residuals.get(frozenset((a, b)), 0.0)

    def outfit_behavior_score(self, garment_ids: Iterable[str]) -> float:
        ids = list(garment_ids)
        if not ids:
            return NEUTRAL_BEHAVIOR_SCORE
        return sum(self.garment_score(g) for g in ids) / len(ids)

    def outfit_pair_adjustment(self, garment_ids: Iterable[str]) -> float:
        """Mean pair residual across the outfit, in [-1, 1]; 0 when nothing is known."""
        ids = list(garment_ids)
        if len(ids) < 2:
            return 0.0
        residuals = [self.pair_residual(a, b) for a, b in combinations(ids, 2)]
        return sum(residuals) / len(residuals)

    def blocked_pair(self, garment_ids: Iterable[str], exempt: Optional[Set[str]] = None) -> Optional[Tuple[str, str]]:
        """The first hard-rejected pair in this outfit, unless *both* garments are exempt (the
        member anchored them deliberately — their choice beats their history)."""
        exempt = exempt or set()
        for a, b in combinations(list(garment_ids), 2):
            if frozenset((a, b)) in self._hard_reject_pairs and not (a in exempt and b in exempt):
                return (a, b)
        return None

    # --- explainability -----------------------------------------------------------------

    def summary(self, top_n: int = 5) -> Dict[str, Any]:
        """What Stage 3 records in the trace: which garments are penalised/boosted and why."""
        ranked = sorted(self.garment_detail.items(), key=lambda kv: kv[1]["score"])
        penalised = [{"garment_id": g, **d} for g, d in ranked if d["score"] < NEUTRAL_BEHAVIOR_SCORE - 0.02][:top_n]
        boosted = [{"garment_id": g, **d} for g, d in reversed(ranked) if d["score"] > NEUTRAL_BEHAVIOR_SCORE + 0.02][:top_n]
        worst_pairs = sorted(self.pair_detail.items(), key=lambda kv: kv[1]["residual"])
        disliked_pairs = [
            {"garment_ids": sorted(pair), **d} for pair, d in worst_pairs if d["residual"] < -0.05
        ][:top_n]
        return {
            "votes_considered": self.votes_considered,
            "garments_scored": len(self.garment_detail),
            "pairs_scored": len(self.pair_detail),
            "penalised_garments": penalised,
            "boosted_garments": boosted,
            "disliked_pairs": disliked_pairs,
            "blocked_pairs": [sorted(p) for p in self._hard_reject_pairs],
            "half_life_days": HALF_LIFE_DAYS,
        }


EMPTY_BEHAVIOR_MODEL = WardrobeBehaviorModel({}, {}, set(), {}, {}, 0)


def build_behavior_model(votes: Iterable[BehaviorVote], now: Optional[datetime] = None) -> WardrobeBehaviorModel:
    now = now or datetime.now(timezone.utc)

    likes: Dict[str, float] = {}
    dislikes: Dict[str, float] = {}
    liked_partners: Dict[str, Set[str]] = {}
    disliked_partners: Dict[str, Set[str]] = {}
    pair_likes: Dict[FrozenSet[str], float] = {}
    pair_dislikes: Dict[FrozenSet[str], float] = {}
    pair_dislike_count: Dict[FrozenSet[str], int] = {}
    pair_like_count: Dict[FrozenSet[str], int] = {}
    garment_like_count: Dict[str, int] = {}
    garment_dislike_count: Dict[str, int] = {}

    considered = 0
    for vote in votes:
        ids = sorted(set(vote.garment_ids))
        if not ids or vote.vote not in ("up", "down") or vote.weight <= 0:
            continue
        considered += 1
        w = vote.weight * _decay(vote.created_at, now)
        up = vote.vote == "up"
        for g in ids:
            others = {o for o in ids if o != g}
            if up:
                likes[g] = likes.get(g, 0.0) + w
                liked_partners.setdefault(g, set()).update(others)
                garment_like_count[g] = garment_like_count.get(g, 0) + 1
            else:
                dislikes[g] = dislikes.get(g, 0.0) + w
                disliked_partners.setdefault(g, set()).update(others)
                garment_dislike_count[g] = garment_dislike_count.get(g, 0) + 1
        for a, b in combinations(ids, 2):
            key = frozenset((a, b))
            if up:
                pair_likes[key] = pair_likes.get(key, 0.0) + w
                pair_like_count[key] = pair_like_count.get(key, 0) + 1
            else:
                pair_dislikes[key] = pair_dislikes.get(key, 0.0) + w
                pair_dislike_count[key] = pair_dislike_count.get(key, 0) + 1

    garment_signed: Dict[str, float] = {}
    garment_detail: Dict[str, Dict[str, Any]] = {}
    for g in set(likes) | set(dislikes):
        l_raw, d_raw = likes.get(g, 0.0), dislikes.get(g, 0.0)
        l_eff = l_raw * _diversity(len(liked_partners.get(g, ())))
        d_eff = d_raw * _diversity(len(disliked_partners.get(g, ())))
        signed = (l_eff - d_eff) / (l_raw + d_raw + GARMENT_PRIOR) if (l_raw + d_raw) > 0 else 0.0
        signed = max(-1.0, min(1.0, signed))
        garment_signed[g] = signed
        garment_detail[g] = {
            "score": round(max(EXPLORATION_FLOOR, min(1.0, 0.5 + 0.5 * signed)), 3),
            "likes": garment_like_count.get(g, 0),
            "dislikes": garment_dislike_count.get(g, 0),
            "distinct_disliked_partners": len(disliked_partners.get(g, ())),
            "distinct_liked_partners": len(liked_partners.get(g, ())),
        }

    pair_residuals: Dict[FrozenSet[str], float] = {}
    pair_detail: Dict[FrozenSet[str], Dict[str, Any]] = {}
    hard_reject: Set[FrozenSet[str]] = set()
    for key in set(pair_likes) | set(pair_dislikes):
        a, b = tuple(key)
        pair_score = _shrunk(pair_likes.get(key, 0.0), pair_dislikes.get(key, 0.0), PAIR_PRIOR)
        residual = pair_score - 0.5 * (garment_signed.get(a, 0.0) + garment_signed.get(b, 0.0))
        residual = max(-1.0, min(1.0, residual))
        pair_residuals[key] = residual
        blocked = pair_dislike_count.get(key, 0) >= HARD_REJECT_MIN_DISLIKES and pair_score <= HARD_REJECT_SCORE
        if blocked:
            hard_reject.add(key)
        pair_detail[key] = {
            "pair_score": round(pair_score, 3),
            "residual": round(residual, 3),
            "likes": pair_like_count.get(key, 0),
            "dislikes": pair_dislike_count.get(key, 0),
            "blocked": blocked,
        }

    return WardrobeBehaviorModel(garment_signed, pair_residuals, hard_reject, garment_detail, pair_detail, considered)


def score_wardrobe_behavior(garment: Garment, model: Optional[WardrobeBehaviorModel] = None) -> float:
    """The garment's learned behaviour score in [0, 1]; neutral 0.5 with no model or no votes."""
    if model is None:
        return NEUTRAL_BEHAVIOR_SCORE
    return model.garment_score(garment.id)
