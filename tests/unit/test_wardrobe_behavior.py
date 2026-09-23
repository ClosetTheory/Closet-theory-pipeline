"""Unit tests for Styling Stage 3's behaviour scorer: the credit-assignment cases that make it
more than a vote counter."""

from datetime import datetime, timedelta, timezone

from app.rules.wardrobe_behavior import (
    EXPLORATION_FLOOR,
    NEUTRAL_BEHAVIOR_SCORE,
    BehaviorVote,
    build_behavior_model,
)

NOW = datetime(2026, 9, 23, tzinfo=timezone.utc)


def v(ids, vote, weight=1.0, days_ago=0):
    return BehaviorVote(garment_ids=tuple(ids), vote=vote, weight=weight, created_at=NOW - timedelta(days=days_ago))


def test_no_votes_is_neutral_everywhere():
    m = build_behavior_model([], now=NOW)
    assert m.is_empty
    assert m.garment_score("anything") == NEUTRAL_BEHAVIOR_SCORE
    assert m.pair_residual("a", "b") == 0.0
    assert m.outfit_behavior_score(["a", "b"]) == NEUTRAL_BEHAVIOR_SCORE
    assert m.blocked_pair(["a", "b"]) is None


def test_one_dislike_barely_moves_anything():
    """Shrinkage: a single unlucky vote must not bury a garment."""
    m = build_behavior_model([v(["shirt", "jeans", "shoes"], "down")], now=NOW)
    for g in ("shirt", "jeans", "shoes"):
        assert 0.40 <= m.garment_score(g) < NEUTRAL_BEHAVIOR_SCORE


def test_bad_garment_vs_bad_pairing_are_told_apart():
    """The whole point of the two-level model. Same number of dislikes on `shirt` in both
    wardrobes; in one it is always next to the same trousers, in the other next to five
    different partners."""
    same_partner = build_behavior_model(
        [v(["shirt", "trousers", f"shoes{i}"], "down") for i in range(5)], now=NOW
    )
    many_partners = build_behavior_model(
        [v(["shirt", f"bottom{i}", f"shoes{i}"], "down") for i in range(5)], now=NOW
    )

    # The garment itself is penalised harder when disliked across many different partners.
    assert many_partners.garment_score("shirt") < same_partner.garment_score("shirt")

    # And the shirt+trousers *pairing* carries a strong negative residual only in the
    # same-partner wardrobe — there, the dislikes are the pairing's fault, not the shirt's.
    assert same_partner.pair_residual("shirt", "trousers") < -0.25
    assert abs(many_partners.pair_residual("shirt", "bottom0")) < 0.1


def test_pair_is_hard_blocked_after_three_confident_dislikes_but_anchors_are_exempt():
    m = build_behavior_model([v(["shirt", "trousers"], "down") for _ in range(3)], now=NOW)
    assert m.blocked_pair(["shirt", "trousers", "shoes"]) == ("shirt", "trousers")
    # Two dislikes are not enough.
    m2 = build_behavior_model([v(["shirt", "trousers"], "down") for _ in range(2)], now=NOW)
    assert m2.blocked_pair(["shirt", "trousers"]) is None
    # A like in the mix lifts it above the threshold.
    m3 = build_behavior_model([v(["shirt", "trousers"], "down") for _ in range(3)] + [v(["shirt", "trousers"], "up")], now=NOW)
    assert m3.blocked_pair(["shirt", "trousers"]) is None
    # The member anchored both on purpose — their choice beats their history.
    assert m.blocked_pair(["shirt", "trousers"], exempt={"shirt", "trousers"}) is None
    assert m.blocked_pair(["shirt", "trousers"], exempt={"shirt"}) == ("shirt", "trousers")


def test_likes_boost_symmetrically_and_exploration_floor_holds():
    liked = build_behavior_model([v(["shirt", f"b{i}"], "up") for i in range(6)], now=NOW)
    assert liked.garment_score("shirt") > 0.7
    buried = build_behavior_model([v(["coat", f"b{i}"], "down", weight=3.0) for i in range(30)], now=NOW)
    assert buried.garment_score("coat") == EXPLORATION_FLOOR


def test_old_votes_decay():
    fresh = build_behavior_model([v(["shirt", "jeans"], "down") for _ in range(4)], now=NOW)
    stale = build_behavior_model([v(["shirt", "jeans"], "down", days_ago=240) for _ in range(4)], now=NOW)
    assert stale.garment_score("shirt") > fresh.garment_score("shirt")
    assert stale.garment_score("shirt") < NEUTRAL_BEHAVIOR_SCORE  # decayed, not forgotten


def test_star_weights_count_less_than_thumbs():
    strong = build_behavior_model([v(["shirt", "jeans"], "down", weight=1.0)], now=NOW)
    mild = build_behavior_model([v(["shirt", "jeans"], "down", weight=0.5)], now=NOW)
    assert mild.garment_score("shirt") > strong.garment_score("shirt")


def test_summary_explains_penalties_and_blocks():
    m = build_behavior_model(
        [v(["shirt", "trousers"], "down") for _ in range(3)] + [v(["dress", "heels"], "up") for _ in range(3)], now=NOW
    )
    s = m.summary()
    assert s["votes_considered"] == 6
    assert {p["garment_id"] for p in s["penalised_garments"]} == {"shirt", "trousers"}
    assert {b["garment_id"] for b in s["boosted_garments"]} == {"dress", "heels"}
    assert s["blocked_pairs"] == [["shirt", "trousers"]]
    assert s["penalised_garments"][0]["dislikes"] == 3
