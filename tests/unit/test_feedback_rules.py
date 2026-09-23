"""Unit tests for reason attribution (app.rules.feedback) and how the behaviour scorer consumes it."""

from datetime import datetime, timezone

from app.rules.feedback import (
    RESIDUAL_WEIGHT,
    FeedbackExtraction,
    FeedbackGarment,
    derive_vote_weights,
    heuristic_extract,
    pair_key,
    sanitize_extraction,
)
from app.rules.style_profile import apply_attribute_lessons
from app.rules.wardrobe_behavior import BehaviorVote, build_behavior_model

NOW = datetime(2026, 9, 23, tzinfo=timezone.utc)
SHIRT = FeedbackGarment("g_shirt", role="TOP", category="TOP", subcategory="LINEN_SHIRT", casual_name="linen shirt", colours=("white",))
TROUSERS = FeedbackGarment("g_trousers", role="BOTTOM", category="BOTTOM", subcategory="TROUSERS", casual_name="grey trousers", colours=("grey",))
SNEAKERS = FeedbackGarment("g_sneakers", role="FOOTWEAR", category="FOOTWEAR", subcategory="SNEAKERS", casual_name="white sneakers", colours=("white",))
OUTFIT = [SHIRT, TROUSERS, SNEAKERS]
IDS = [g.garment_id for g in OUTFIT]


# --- heuristic extraction -------------------------------------------------------------------

def test_heuristic_blames_the_named_garment_only():
    ex = heuristic_extract("the sneakers kill it", "down", OUTFIT)
    assert ex.blamed_garment_ids == ["g_sneakers"]
    assert ex.praised_garment_ids == [] and ex.pairings == []


def test_heuristic_recognises_a_pairing_complaint():
    ex = heuristic_extract("The shirt with those trousers clashes badly.", "down", OUTFIT)
    assert set(ex.blamed_garment_ids) == {"g_shirt", "g_trousers"}
    assert ex.pairings == [("g_shirt", "g_trousers")]


def test_heuristic_praise_inside_a_dislike_and_lessons():
    ex = heuristic_extract("Love the shirt. Hate the floral trousers, no florals for her.", "down", OUTFIT)
    assert ex.praised_garment_ids == ["g_shirt"]
    assert ex.blamed_garment_ids == ["g_trousers"]
    assert {"attribute": "pattern", "value": "floral", "polarity": "down"} in ex.lessons


def test_heuristic_names_nothing_when_nothing_is_recognisable():
    ex = heuristic_extract("meh, not for a Tuesday", "down", OUTFIT)
    assert ex.is_empty


def test_sanitize_drops_ids_not_in_the_outfit_and_unknown_attributes():
    ex = sanitize_extraction(
        {"blamed_garment_ids": ["g_sneakers", "ghost"], "pairings": [["g_shirt", "ghost"], ["g_shirt", "g_trousers"]],
         "lessons": [{"attribute": "vibe", "value": "sad", "polarity": "down"}, {"attribute": "colour", "value": "Neon", "polarity": "down"}]},
        OUTFIT, model="test",
    )
    assert ex.blamed_garment_ids == ["g_sneakers"]
    assert ex.pairings == [("g_shirt", "g_trousers")]
    assert ex.lessons == [{"attribute": "colour", "value": "neon", "polarity": "down"}]


# --- weights for one vote --------------------------------------------------------------------

def test_no_reason_is_an_even_split():
    w = derive_vote_weights("down", IDS)
    assert all(v == 1.0 for v in w.garment_weights.values())
    assert all(v == 1.0 for v in w.pair_weights.values())
    assert w.counter_garment_ids == []


def test_blamed_garment_takes_the_weight_and_named_pair_is_doubled():
    ex = FeedbackExtraction(blamed_garment_ids=["g_sneakers"], pairings=[("g_shirt", "g_trousers")])
    w = derive_vote_weights("down", IDS, extraction=ex)
    assert w.garment_weights["g_sneakers"] == 1.0
    assert w.garment_weights["g_shirt"] == RESIDUAL_WEIGHT == w.garment_weights["g_trousers"]
    assert w.pair_weights[pair_key("g_shirt", "g_trousers")] == 2.0
    assert w.pair_weights[pair_key("g_shirt", "g_sneakers")] == round((1.0 + RESIDUAL_WEIGHT) / 2, 4)


def test_praised_garment_inside_a_dislike_flips_polarity_and_leaves_its_pairs_alone():
    ex = FeedbackExtraction(praised_garment_ids=["g_shirt"])
    w = derive_vote_weights("down", IDS, extraction=ex)
    assert w.counter_garment_ids == ["g_shirt"]
    assert w.garment_weights["g_shirt"] == 0.0
    assert w.pair_weights[pair_key("g_shirt", "g_trousers")] == 0.0
    assert w.pair_weights[pair_key("g_trousers", "g_sneakers")] == 1.0


def test_reason_chips_shift_blame_between_pieces_and_combination():
    clash = derive_vote_weights("down", IDS, tags=["colour_clash"])
    assert clash.garment_weights["g_shirt"] == 0.5, "a clash is about the combination, not the pieces"
    assert clash.pair_weights[pair_key("g_shirt", "g_trousers")] == 0.75, "pair factor 1.5 x mean garment weight 0.5"
    occasion = derive_vote_weights("down", IDS, tags=["wrong_occasion"])
    assert all(v == 0.3 for v in occasion.garment_weights.values())
    # An up-vote chip is ignored on a down-vote.
    assert derive_vote_weights("down", IDS, tags=["great_pairing"]).garment_weights["g_shirt"] == 1.0


# --- the scorer consuming the weights -------------------------------------------------------

def _vote(vote, weights=None, days=0):
    return BehaviorVote(garment_ids=tuple(IDS), vote=vote, weight=1.0, created_at=NOW,
                        garment_weights=weights.garment_weights if weights else None,
                        pair_weights=weights.pair_weights if weights else None,
                        counter_garment_ids=tuple(weights.counter_garment_ids) if weights else ())


def test_scorer_penalises_the_blamed_garment_more_than_the_rest():
    weighted = derive_vote_weights("down", IDS, extraction=FeedbackExtraction(blamed_garment_ids=["g_sneakers"]))
    m = build_behavior_model([_vote("down", weighted) for _ in range(3)], now=NOW)
    assert m.garment_score("g_sneakers") < m.garment_score("g_shirt")
    assert m.garment_score("g_shirt") < 0.5, "the rest still carry a residual — the outfit was rejected"


def test_scorer_counts_praise_inside_a_dislike_as_a_like():
    weighted = derive_vote_weights("down", IDS, extraction=FeedbackExtraction(praised_garment_ids=["g_shirt"]))
    m = build_behavior_model([_vote("down", weighted) for _ in range(3)], now=NOW)
    assert m.garment_score("g_shirt") > 0.5
    assert m.garment_score("g_trousers") < 0.5
    assert m.pair_residual("g_shirt", "g_trousers") == 0.0, "a praised piece is not part of the disliked combination"


def test_named_pairing_reaches_the_block_threshold_faster():
    named = derive_vote_weights("down", IDS, extraction=FeedbackExtraction(pairings=[("g_shirt", "g_trousers")]))
    m = build_behavior_model([_vote("down", named) for _ in range(3)], now=NOW)
    assert m.blocked_pair(IDS) == ("g_shirt", "g_trousers")
    assert m.pair_residual("g_shirt", "g_trousers") < m.pair_residual("g_shirt", "g_sneakers")


# --- style-profile lessons ------------------------------------------------------------------

def test_lessons_nudge_exactly_the_named_attribute_value():
    affinities = apply_attribute_lessons({}, [{"attribute": "pattern", "value": "floral", "polarity": "down"},
                                             {"attribute": "vibe", "value": "sad", "polarity": "down"}])
    assert affinities == {"pattern": {"floral": {"score": -0.5, "count": 2}}}
    again = apply_attribute_lessons(affinities, [{"attribute": "pattern", "value": "floral", "polarity": "down"}])
    assert again["pattern"]["floral"] == {"score": -0.75, "count": 4}
