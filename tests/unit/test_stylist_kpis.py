"""app/metrics/stylist_kpis.py — the numbers the admin page shows, pinned.

Scenario: two stylists, three characters. S1 is assigned A and B, S2 is assigned B only.
A has 4 outfits, B has 3, C has 2 and nobody. S1 reviewed 2 of A's outfits (one twice — a
revision, must not double count), and 1 of B's. S2 reviewed 2 of B's outfits, one of them
8 days ago (outside the 7-day window). S1 uploaded 3 garments for A, 2 completed.
"""

from datetime import datetime, timedelta, timezone

from app.metrics.stylist_kpis import compute_stylist_kpis

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def _h(hours):
    return NOW - timedelta(hours=hours)


def _scenario():
    personas = [
        {"id": "pA", "user_id": "tA", "display_name": "Ananya", "slug": "ananya", "city": "mumbai"},
        {"id": "pB", "user_id": "tB", "display_name": "Bhavna", "slug": "bhavna", "city": "pune"},
        {"id": "pC", "user_id": "tC", "display_name": "Chaiti", "slug": "chaiti", "city": "delhi"},
    ]
    assignments = [
        {"persona_id": "pA", "stylist_user_id": "s1", "status": "active", "created_at": _h(400)},
        {"persona_id": "pB", "stylist_user_id": "s1", "status": "active", "created_at": _h(400)},
        {"persona_id": "pB", "stylist_user_id": "s2", "status": "active", "created_at": _h(400)},
        {"persona_id": "pC", "stylist_user_id": "s2", "status": "revoked", "created_at": _h(400)},
    ]
    users = [
        {"id": "s1", "display_name": "Stylist1", "email": "s1@x"},
        {"id": "s2", "display_name": "Stylist2", "email": "s2@x"},
    ]
    outfits = (
        [{"id": f"oA{i}", "tenant_id": "tA", "created_at": _h(100 - i), "final_score": 0.6 + i / 10} for i in range(4)]
        + [{"id": f"oB{i}", "tenant_id": "tB", "created_at": _h(300), "final_score": 0.5} for i in range(3)]
        + [{"id": f"oC{i}", "tenant_id": "tC", "created_at": _h(10), "final_score": 0.9} for i in range(2)]
    )
    reviews = [
        # S1 on A: two outfits, oA0 revised once (same reviewer, same outfit -> one row upserted)
        {"outfit_id": "oA0", "persona_id": "pA", "reviewer_user_id": "s1", "rating": 4,
         "dimension_ratings": {"colour_harmony": 5, "fit_and_silhouette": 4, "occasion_fit": 4, "persona_fit": 3},
         "comment": "good", "would_wear": True, "tags": ["great_pairing"], "vote": "like", "created_at": _h(90), "updated_at": _h(2)},
        {"outfit_id": "oA1", "persona_id": "pA", "reviewer_user_id": "s1", "rating": 2,
         "dimension_ratings": {"colour_harmony": 3, "fit_and_silhouette": 2, "occasion_fit": 2, "persona_fit": 1},
         "comment": "", "would_wear": False, "tags": [], "vote": "dislike", "created_at": _h(50), "updated_at": _h(50)},
        # S1 on B
        {"outfit_id": "oB0", "persona_id": "pB", "reviewer_user_id": "s1", "rating": 3,
         "dimension_ratings": {}, "comment": None, "would_wear": None, "tags": [], "vote": None, "created_at": _h(24), "updated_at": _h(24)},
        # S2 on B: one inside the window, one 8 days old
        {"outfit_id": "oB0", "persona_id": "pB", "reviewer_user_id": "s2", "rating": 5,
         "dimension_ratings": {"colour_harmony": 5}, "comment": "lovely", "would_wear": True, "tags": ["love_the_pieces", "perfect_fit"],
         "vote": "like", "created_at": _h(12), "updated_at": _h(12)},
        {"outfit_id": "oB1", "persona_id": "pB", "reviewer_user_id": "s2", "rating": 4,
         "dimension_ratings": {}, "comment": "ok", "would_wear": True, "tags": [], "vote": "like", "created_at": _h(8 * 24), "updated_at": _h(8 * 24)},
    ]
    uploads = [
        {"persona_id": "pA", "uploaded_by_user_id": "s1", "garment_status": "COMPLETED"},
        {"persona_id": "pA", "uploaded_by_user_id": "s1", "garment_status": "COMPLETED"},
        {"persona_id": "pA", "uploaded_by_user_id": "s1", "garment_status": "FAILED"},
    ]
    runs = [
        {"persona_id": "pA", "initiated_by_user_id": "s1", "status": "COMPLETED", "created_at": _h(90)},
        {"persona_id": None, "initiated_by_user_id": "s1", "status": "COMPLETED", "created_at": _h(90)},
    ]
    return dict(personas=personas, assignments=assignments, users=users, outfits=outfits, reviews=reviews, uploads=uploads, runs=runs)


def _cell(m, sid, pid):
    return next(c for c in m["cells"] if c["stylist_user_id"] == sid and c["persona_id"] == pid)


def _stylist(m, sid):
    return next(s for s in m["stylists"] if s["stylist_user_id"] == sid)


def _character(m, pid):
    return next(c for c in m["characters"] if c["persona_id"] == pid)


def test_cells_cover_only_assigned_or_active_pairs():
    m = compute_stylist_kpis(now=NOW, **_scenario())
    pairs = {(c["stylist_user_id"], c["persona_id"]) for c in m["cells"]}
    # s2 x pC is a revoked assignment with no activity: no cell. s2 x pA: nothing at all.
    assert pairs == {("s1", "pA"), ("s1", "pB"), ("s2", "pB")}


def test_cell_coverage_backlog_and_weekly_progress():
    m = compute_stylist_kpis(now=NOW, **_scenario())
    c = _cell(m, "s1", "pA")
    assert c["assigned"] is True
    assert c["outfits_total"] == 4 and c["outfits_reviewed"] == 2 and c["backlog"] == 2
    assert c["coverage_pct"] == 50.0
    assert c["reviews_total"] == 2 and c["reviews_window"] == 2
    assert c["weekly_target"] == 20 and c["weekly_progress_pct"] == 10.0
    assert c["avg_rating"] == 3.0 and c["rating_histogram"] == {"2": 1, "4": 1}
    assert c["dimension_averages"] == {"colour_harmony": 4.0, "fit_and_silhouette": 3.0, "occasion_fit": 3.0, "persona_fit": 2.0}
    assert c["with_comment_pct"] == 50.0 and c["with_chips_pct"] == 50.0 and c["would_wear_pct"] == 50.0
    assert c["likes"] == 1 and c["dislikes"] == 1
    assert c["garments_uploaded"] == 3 and c["garments_completed"] == 2 and c["runs_started"] == 1

    s2b = _cell(m, "s2", "pB")
    assert s2b["reviews_total"] == 2 and s2b["reviews_window"] == 1, "the 8-day-old review is outside the window"
    assert s2b["weekly_progress_pct"] == 5.0
    assert s2b["with_chips_pct"] == 50.0 and s2b["would_wear_pct"] == 100.0


def test_stylist_rollup_uses_assigned_characters_for_coverage_and_target():
    m = compute_stylist_kpis(now=NOW, **_scenario())
    s1 = _stylist(m, "s1")
    assert s1["characters_assigned"] == 2 and s1["characters_reviewed"] == 2
    assert s1["outfits_available"] == 7  # 4 on A + 3 on B
    assert s1["outfits_reviewed"] == 3 and s1["backlog"] == 4
    assert s1["coverage_pct"] == round(100 * 3 / 7, 1)
    assert s1["weekly_target"] == 40 and s1["reviews_window"] == 3 and s1["weekly_progress_pct"] == 7.5
    assert s1["garments_uploaded"] == 3 and s1["runs_started"] == 1
    assert s1["avg_rating"] == 3.0

    s2 = _stylist(m, "s2")
    assert s2["characters_assigned"] == 1, "revoked assignments do not count"
    assert s2["weekly_target"] == 20 and s2["reviews_window"] == 1 and s2["weekly_progress_pct"] == 5.0


def test_character_rollup_counts_any_reviewer_and_measures_lag():
    m = compute_stylist_kpis(now=NOW, **_scenario())
    b = _character(m, "pB")
    assert b["assigned_to"] == ["Stylist1", "Stylist2"]
    assert b["outfits_total"] == 3 and b["outfits_reviewed"] == 2, "oB0 reviewed by two stylists is one outfit"
    assert b["backlog"] == 1 and b["coverage_pct"] == round(100 * 2 / 3, 1)
    assert b["reviews_total"] == 3 and b["reviews_window"] == 2 and b["weekly_progress_pct"] == 10.0
    assert b["avg_rating"] == 4.0
    # oB0 created 300h ago, first reviewed 24h ago (S1) -> 276h; oB1 created 300h ago, reviewed 192h ago -> 108h. Median 192.
    assert b["median_hours_to_first_review"] == 192.0
    assert b["engine_avg_score"] == 0.5

    c = _character(m, "pC")
    assert c["assigned_to"] == [] and c["reviews_total"] == 0 and c["backlog"] == 2
    assert c["avg_rating"] is None and c["coverage_pct"] == 0.0 and c["weekly_progress_pct"] == 0.0
    assert c["outfits_window"] == 2 and c["last_review_at"] is None


def test_totals_and_window_are_parameterised():
    m = compute_stylist_kpis(now=NOW, weekly_target=5, window_days=30, **_scenario())
    assert m["weekly_target_per_character"] == 5 and m["window_days"] == 30
    assert m["totals"]["characters"] == 3 and m["totals"]["characters_assigned"] == 2 and m["totals"]["stylists"] == 2
    assert m["totals"]["outfits_total"] == 9 and m["totals"]["outfits_reviewed"] == 4 and m["totals"]["backlog"] == 5
    assert m["totals"]["reviews_total"] == 5 and m["totals"]["reviews_window"] == 5, "30-day window catches the old one"
    assert m["totals"]["weekly_target"] == 15
    assert _cell(m, "s2", "pB")["weekly_progress_pct"] == 40.0


def test_naive_datetimes_are_treated_as_utc():
    data = _scenario()
    for r in data["reviews"]:
        r["updated_at"] = r["updated_at"].replace(tzinfo=None)
        r["created_at"] = r["created_at"].replace(tzinfo=None)
    m = compute_stylist_kpis(now=NOW.replace(tzinfo=None), **data)
    assert _cell(m, "s2", "pB")["reviews_window"] == 1


def test_empty_panel_is_all_zeros_not_an_error():
    m = compute_stylist_kpis(now=NOW, personas=[], assignments=[], users=[], outfits=[], reviews=[], uploads=[], runs=[])
    assert m["stylists"] == [] and m["characters"] == [] and m["cells"] == []
    assert m["totals"]["outfits_total"] == 0 and m["totals"]["coverage_pct"] is None


def test_daily_series_covers_every_day_oldest_first_with_zeros():
    m = compute_stylist_kpis(now=NOW, activity_days=10, **_scenario())
    s2 = _stylist(m, "s2")
    series = s2["daily_reviews"]
    assert len(series) == 10 and m["activity_days"] == 10
    assert series[0]["date"] == "2026-09-15" and series[-1]["date"] == "2026-09-24"
    by_date = {d["date"]: d["count"] for d in series}
    assert by_date["2026-09-24"] == 1     # 12h ago
    assert by_date["2026-09-16"] == 1     # 8 days ago, still inside the 10-day series
    assert sum(by_date.values()) == 2
    assert m["totals"]["daily_reviews"][-1]["count"] == 2  # s1's 2h-ago revision + s2's 12h-ago review
