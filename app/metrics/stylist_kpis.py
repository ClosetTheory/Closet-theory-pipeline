"""Per-stylist, per-character KPIs for the admin page.

Everything here is computed from rows the panel already writes: assignments, outfits generated
for a character, the reviews stylists leave on them, the garments stylists upload into a
character's wardrobe, and the styling runs they start. No new tables, no new columns.

Definitions (also in docs/STYLIST_KPIS.md — keep the two in step):

- A character's *outfits* are every outfit generated in that character's account.
- A stylist has *reviewed* an outfit if they have a persona_outfit_reviews row for it. One row
  per (outfit, reviewer), so revisions do not double count.
- *Coverage* = distinct outfits reviewed / outfits generated, for the pair, the character (any
  reviewer) or the stylist (over their assigned characters).
- *Backlog* = outfits generated that the stylist (cell) or anyone (character) has not reviewed.
- *This week* = reviews whose updated_at falls inside the trailing window (default 7 days).
  The target is per character per week (default 20); a stylist's target is 20 × characters
  assigned to them.
- *Depth*: share of reviews with a comment, with at least one reason chip, marked would-wear.
- Dimension averages use the four rubric keys in app.models.persona_review.REVIEW_DIMENSIONS.
- *Wardrobe built*: garments the stylist uploaded for the character, and how many finished
  ingestion (status COMPLETED).
- *Hours to first review* (character): median hours between an outfit's creation and its
  earliest review, over outfits that have one.

The function is pure: it takes plain dicts and a fixed `now`, returns plain dicts, and is
tested directly. The endpoint in app/api/v1/admin.py only loads rows and calls it.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

from app.models.persona_review import REVIEW_DIMENSIONS

DEFAULT_WEEKLY_TARGET = 20
DEFAULT_WINDOW_DAYS = 7
DEFAULT_ACTIVITY_DAYS = 28  # length of the reviews-per-day series the admin page charts


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return _utc(dt).isoformat() if dt else None


def _pct(num: float, den: float) -> Optional[float]:
    return round(100.0 * num / den, 1) if den else None


def _avg(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _review_stats(reviews: List[Dict[str, Any]], window_start: datetime) -> Dict[str, Any]:
    """Depth and quality numbers for a bag of reviews (any grouping)."""
    n = len(reviews)
    if n == 0:
        return {
            "reviews_total": 0, "reviews_window": 0, "avg_rating": None, "rating_histogram": {},
            "dimension_averages": {k: None for k in REVIEW_DIMENSIONS},
            "with_comment_pct": None, "with_chips_pct": None, "would_wear_pct": None,
            "likes": 0, "dislikes": 0, "last_review_at": None,
        }
    histogram: Dict[str, int] = defaultdict(int)
    for r in reviews:
        histogram[str(r.get("rating"))] += 1
    dims = {
        key: _avg((r.get("dimension_ratings") or {}).get(key) for r in reviews)
        for key in REVIEW_DIMENSIONS
    }
    would_wear_answered = [r for r in reviews if r.get("would_wear") is not None]
    return {
        "reviews_total": n,
        "reviews_window": sum(1 for r in reviews if _utc(r.get("updated_at") or r.get("created_at")) >= window_start),
        "avg_rating": _avg(r.get("rating") for r in reviews),
        "rating_histogram": dict(sorted(histogram.items())),
        "dimension_averages": dims,
        "with_comment_pct": _pct(sum(1 for r in reviews if (r.get("comment") or "").strip()), n),
        "with_chips_pct": _pct(sum(1 for r in reviews if r.get("tags")), n),
        "would_wear_pct": _pct(sum(1 for r in would_wear_answered if r.get("would_wear")), len(would_wear_answered)),
        "likes": sum(1 for r in reviews if r.get("vote") == "like"),
        "dislikes": sum(1 for r in reviews if r.get("vote") == "dislike"),
        "last_review_at": _iso(max(_utc(r.get("updated_at") or r.get("created_at")) for r in reviews)),
    }


def compute_stylist_kpis(
    *,
    now: datetime,
    personas: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
    users: List[Dict[str, Any]],
    outfits: List[Dict[str, Any]],
    reviews: List[Dict[str, Any]],
    uploads: List[Dict[str, Any]],
    runs: List[Dict[str, Any]],
    weekly_target: int = DEFAULT_WEEKLY_TARGET,
    window_days: int = DEFAULT_WINDOW_DAYS,
    activity_days: int = DEFAULT_ACTIVITY_DAYS,
) -> Dict[str, Any]:
    now = _utc(now)
    window_start = now - timedelta(days=window_days)

    def daily_series(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reviews per UTC calendar day over the trailing `activity_days`, oldest first,
        every day present (zeros included) so a chart never has to infer gaps."""
        counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            ts = _utc(r.get("updated_at") or r.get("created_at"))
            if ts:
                counts[ts.date().isoformat()] += 1
        today = now.date()
        return [
            {"date": (today - timedelta(days=offset)).isoformat(), "count": counts.get((today - timedelta(days=offset)).isoformat(), 0)}
            for offset in range(activity_days - 1, -1, -1)
        ]

    persona_by_id = {p["id"]: p for p in personas}
    persona_by_tenant = {p["user_id"]: p["id"] for p in personas}
    user_name = {u["id"]: (u.get("display_name") or u.get("email") or u["id"]) for u in users}
    user_email = {u["id"]: u.get("email") for u in users}

    # --- index the raw rows ---------------------------------------------------------------
    outfits_by_persona: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    outfit_persona: Dict[str, str] = {}
    for o in outfits:
        pid = persona_by_tenant.get(o["tenant_id"])
        if pid is None:
            continue
        outfits_by_persona[pid].append(o)
        outfit_persona[o["id"]] = pid

    active_assignments = [a for a in assignments if (a.get("status") or "active") == "active"]
    assigned_pairs = {(a["stylist_user_id"], a["persona_id"]) for a in active_assignments}
    assigned_personas_of: Dict[str, set] = defaultdict(set)
    assigned_stylists_of: Dict[str, set] = defaultdict(set)
    for s, p in assigned_pairs:
        assigned_personas_of[s].add(p)
        assigned_stylists_of[p].add(s)

    reviews_by_pair: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    reviews_by_persona: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    reviews_by_stylist: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in reviews:
        pid = r.get("persona_id") or outfit_persona.get(r["outfit_id"])
        if pid is None:
            continue
        r = dict(r, persona_id=pid)
        reviews_by_pair[(r["reviewer_user_id"], pid)].append(r)
        reviews_by_persona[pid].append(r)
        reviews_by_stylist[r["reviewer_user_id"]].append(r)

    uploads_by_pair: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    uploads_by_persona: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for u in uploads:
        uploads_by_pair[(u["uploaded_by_user_id"], u["persona_id"])].append(u)
        uploads_by_persona[u["persona_id"]].append(u)

    runs_by_pair: Dict[tuple, int] = defaultdict(int)
    for run in runs:
        if run.get("persona_id"):
            runs_by_pair[(run["initiated_by_user_id"], run["persona_id"])] += 1

    stylist_ids = (
        {s for s, _ in assigned_pairs}
        | set(reviews_by_stylist)
        | {s for s, _ in uploads_by_pair}
        | {s for s, _ in runs_by_pair}
    )

    # --- cells: stylist × character ---------------------------------------------------------
    cells: List[Dict[str, Any]] = []
    for sid in sorted(stylist_ids, key=lambda s: user_name.get(s, s).lower()):
        for pid in sorted(persona_by_id, key=lambda p: persona_by_id[p]["display_name"].lower()):
            pair = (sid, pid)
            pair_reviews = reviews_by_pair.get(pair, [])
            pair_uploads = uploads_by_pair.get(pair, [])
            assigned = pair in assigned_pairs
            if not (assigned or pair_reviews or pair_uploads or runs_by_pair.get(pair)):
                continue
            persona_outfits = outfits_by_persona.get(pid, [])
            reviewed_ids = {r["outfit_id"] for r in pair_reviews}
            stats = _review_stats(pair_reviews, window_start)
            cells.append({
                "stylist_user_id": sid,
                "stylist_name": user_name.get(sid, sid),
                "persona_id": pid,
                "character_name": persona_by_id[pid]["display_name"],
                "assigned": assigned,
                "outfits_total": len(persona_outfits),
                "outfits_reviewed": len(reviewed_ids),
                "coverage_pct": _pct(len(reviewed_ids), len(persona_outfits)),
                "backlog": len(persona_outfits) - len(reviewed_ids),
                "weekly_target": weekly_target,
                "weekly_progress_pct": _pct(min(stats["reviews_window"], weekly_target), weekly_target),
                "garments_uploaded": len(pair_uploads),
                "garments_completed": sum(1 for u in pair_uploads if u.get("garment_status") == "COMPLETED"),
                "runs_started": runs_by_pair.get(pair, 0),
                **stats,
            })

    # --- characters -------------------------------------------------------------------------
    characters: List[Dict[str, Any]] = []
    for pid, p in sorted(persona_by_id.items(), key=lambda kv: kv[1]["display_name"].lower()):
        persona_outfits = outfits_by_persona.get(pid, [])
        p_reviews = reviews_by_persona.get(pid, [])
        reviewed_ids = {r["outfit_id"] for r in p_reviews}
        stats = _review_stats(p_reviews, window_start)
        first_review_at: Dict[str, datetime] = {}
        for r in p_reviews:
            ts = _utc(r.get("created_at") or r.get("updated_at"))
            if ts and (r["outfit_id"] not in first_review_at or ts < first_review_at[r["outfit_id"]]):
                first_review_at[r["outfit_id"]] = ts
        lags = [
            (first_review_at[o["id"]] - _utc(o["created_at"])).total_seconds() / 3600.0
            for o in persona_outfits if o["id"] in first_review_at and o.get("created_at")
        ]
        p_uploads = uploads_by_persona.get(pid, [])
        characters.append({
            "persona_id": pid,
            "character_name": p["display_name"],
            "slug": p.get("slug"),
            "city": p.get("city"),
            "assigned_to": sorted(user_name.get(s, s) for s in assigned_stylists_of.get(pid, ())),
            "assigned_stylist_ids": sorted(assigned_stylists_of.get(pid, ())),
            "outfits_total": len(persona_outfits),
            "outfits_window": sum(1 for o in persona_outfits if _utc(o.get("created_at")) and _utc(o["created_at"]) >= window_start),
            "outfits_reviewed": len(reviewed_ids),
            "coverage_pct": _pct(len(reviewed_ids), len(persona_outfits)),
            "backlog": len(persona_outfits) - len(reviewed_ids),
            "weekly_target": weekly_target,
            "weekly_progress_pct": _pct(min(stats["reviews_window"], weekly_target), weekly_target),
            "engine_avg_score": _avg(o.get("final_score") for o in persona_outfits),
            "median_hours_to_first_review": round(median(lags), 1) if lags else None,
            "garments_uploaded": len(p_uploads),
            "garments_completed": sum(1 for u in p_uploads if u.get("garment_status") == "COMPLETED"),
            **stats,
        })

    # --- stylists ---------------------------------------------------------------------------
    stylists: List[Dict[str, Any]] = []
    for sid in sorted(stylist_ids, key=lambda s: user_name.get(s, s).lower()):
        my_cells = [c for c in cells if c["stylist_user_id"] == sid]
        assigned_ids = assigned_personas_of.get(sid, set())
        assigned_outfits = sum(len(outfits_by_persona.get(pid, [])) for pid in assigned_ids)
        assigned_reviewed = sum(
            len({r["outfit_id"] for r in reviews_by_pair.get((sid, pid), [])}) for pid in assigned_ids
        )
        s_reviews = reviews_by_stylist.get(sid, [])
        stats = _review_stats(s_reviews, window_start)
        target_total = weekly_target * len(assigned_ids)
        stylists.append({
            "stylist_user_id": sid,
            "stylist_name": user_name.get(sid, sid),
            "email": user_email.get(sid),
            "characters_assigned": len(assigned_ids),
            "characters_reviewed": len({r["persona_id"] for r in s_reviews}),
            "characters_with_reviews_window": len({
                r["persona_id"] for r in s_reviews
                if _utc(r.get("updated_at") or r.get("created_at")) >= window_start
            }),
            "outfits_available": assigned_outfits,
            "outfits_reviewed": sum(len({r["outfit_id"] for r in reviews_by_pair.get((sid, pid), [])}) for pid in {c["persona_id"] for c in my_cells}),
            "coverage_pct": _pct(assigned_reviewed, assigned_outfits),
            "backlog": assigned_outfits - assigned_reviewed,
            "weekly_target": target_total,
            "weekly_progress_pct": _pct(min(stats["reviews_window"], target_total), target_total) if target_total else None,
            "garments_uploaded": sum(c["garments_uploaded"] for c in my_cells),
            "garments_completed": sum(c["garments_completed"] for c in my_cells),
            "runs_started": sum(c["runs_started"] for c in my_cells),
            "daily_reviews": daily_series(s_reviews),
            **stats,
        })

    all_reviews = [r for rs in reviews_by_persona.values() for r in rs]
    total_outfits = sum(len(v) for v in outfits_by_persona.values())
    reviewed_any = {r["outfit_id"] for r in all_reviews}
    totals = {
        "characters": len(persona_by_id),
        "characters_assigned": len(assigned_stylists_of),
        "stylists": len(stylist_ids),
        "outfits_total": total_outfits,
        "outfits_reviewed": len(reviewed_any),
        "coverage_pct": _pct(len(reviewed_any), total_outfits),
        "backlog": total_outfits - len(reviewed_any),
        "weekly_target": weekly_target * len(persona_by_id),
        "daily_reviews": daily_series(all_reviews),
        **_review_stats(all_reviews, window_start),
    }

    return {
        "generated_at": now.isoformat(),
        "window_days": window_days,
        "window_start": window_start.isoformat(),
        "weekly_target_per_character": weekly_target,
        "activity_days": activity_days,
        "dimension_keys": list(REVIEW_DIMENSIONS),
        "totals": totals,
        "stylists": stylists,
        "characters": characters,
        "cells": cells,
    }
