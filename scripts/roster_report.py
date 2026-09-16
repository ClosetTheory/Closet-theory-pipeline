"""Render the evaluation-character roster as readable markdown.

Generated from app/personas/roster.yaml, never hand-edited — the YAML is the source of truth and
a hand-maintained copy would drift from it within a week. Re-run after editing the roster.

    python -m scripts.roster_report                    # to stdout
    python -m scripts.roster_report --out docs/ROSTER.md
"""

import argparse
from collections import Counter
from typing import List

from app.personas import load_roster
from app.schemas.persona import PersonaSeed


def _coverage(roster: List[PersonaSeed]) -> str:
    monk = Counter(p.skin_tone_monk for p in roster)
    lines = [
        "## Coverage",
        "",
        "| Axis | Spread |",
        "|---|---|",
        f"| Characters | {len(roster)} |",
        f"| Gender | " + " · ".join(f"{v} {k}" for k, v in Counter(p.gender for p in roster).most_common()) + " |",
        f"| Age | {min(p.age for p in roster)}–{max(p.age for p in roster)} |",
        f"| Cities | {len({p.city for p in roster})} across {len({p.climate.zone for p in roster})} climate zones |",
        f"| Skin tone (Monk) | " + " · ".join(f"{k}:{monk[k]}" for k in sorted(monk)) + " |",
        f"| Colour season | " + " · ".join(f"{k} {v}" for k, v in Counter(p.color_analysis.season for p in roster).most_common()) + " |",
        f"| Undertone | " + " · ".join(f"{k} {v}" for k, v in Counter(p.color_analysis.undertone for p in roster).most_common()) + " |",
        f"| Body shape | " + " · ".join(f"{k} {v}" for k, v in Counter(p.body_shape for p in roster).most_common()) + " |",
        f"| Height | " + " · ".join(f"{k} {v}" for k, v in Counter(p.height_band for p in roster).most_common()) + " |",
        f"| Build | " + " · ".join(f"{k} {v}" for k, v in Counter(p.build for p in roster).most_common()) + " |",
        f"| Budget | " + " · ".join(f"{k} {v}" for k, v in Counter(p.budget_tier for p in roster).most_common()) + " |",
        f"| Ethnic wear (share of occasions) | "
        f"{min(p.ethnic_wear.share_of_occasions for p in roster):.2f}–"
        f"{max(p.ethnic_wear.share_of_occasions for p in roster):.2f}, "
        f"daily for {sum(1 for p in roster if p.ethnic_wear.daily_ethnic)} |",
        f"| Carrying a hard constraint | {sum(1 for p in roster if p.hard_constraints)} of {len(roster)} |",
        "",
    ]
    return "\n".join(lines)


def _character(p: PersonaSeed) -> str:
    ca = p.color_analysis
    ew = p.ethnic_wear
    prefs = p.preferences or {}
    mix = prefs.get("occasion_mix") or {}
    top_occasions = ", ".join(
        f"{k.replace('_', ' ')} {int(v * 100)}%"
        for k, v in sorted(mix.items(), key=lambda kv: -kv[1])[:4]
    )
    constraints = ", ".join(c.replace("_", " ") for c in p.hard_constraints) or "none"
    pains = "; ".join(p.fit_pain_points) or "none noted"
    liked = ", ".join((prefs.get("styling_context") or {}).get("brandsLiked") or []) or "—"
    loves = ", ".join((prefs.get("colour_preferences") or {}).get("love") or []) or "—"
    avoids = ", ".join((prefs.get("colour_preferences") or {}).get("avoid") or []) or "—"

    return f"""### {p.name} · {p.age} · {p.gender}

**{p.occupation or "—"} — {p.city.title()}, {p.state_region or ''}** · {p.climate.zone.replace('_', ' ')}
(summers {p.climate.summer_severity}, winters {p.climate.winter_severity},
{p.climate.humidity}, monsoon {p.climate.monsoon_intensity})

| | |
|---|---|
| Body | {p.body_shape.replace('_', ' ')} · {p.height_cm or '?'}cm · {p.height_band} · {p.build} |
| Skin | Monk {p.skin_tone_monk} |
| Colour | {ca.season}{f" ({ca.season_sub})" if ca.season_sub else ""} · {ca.undertone} undertone · {ca.depth} depth · {ca.contrast} contrast |
| Palette | {" ".join(ca.palette)} |
| Hair | {p.hair.length.replace('_', ' ')}, {p.hair.texture}, {p.hair.colour.replace('_', ' ')}{f", {p.hair.head_covering.replace('_', ' ')}" if p.hair.head_covering != "none" else ""} |
| Occasions | {top_occasions or "—"} |
| Ethnic wear | owns {int(ew.share_of_wardrobe * 100)}%, needs {int(ew.share_of_occasions * 100)}%{" · daily" if ew.daily_ethnic else ""} · drape {ew.drape_competence} |
| Budget | {p.budget_tier} · likes {liked} |
| Colours | loves {loves} · avoids {avoids} |
| Hard constraints | **{constraints}** |
| Fit problems | {pains} |

{p.bio}

> **Why this character is on the panel.** {p.rationale}

*Colour analysis:* {ca.summary}

*Assigned to:* {p.assigned_stylist or "unassigned"}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write to this path instead of stdout")
    args = ap.parse_args()

    roster = load_roster()
    by_stylist = {}
    for p in roster:
        by_stylist.setdefault(p.assigned_stylist or "unassigned", []).append(p)

    parts = [
        "# Evaluation Character Roster",
        "",
        "Generated from `app/personas/roster.yaml` by `python -m scripts.roster_report`.",
        "Edit the YAML, not this file.",
        "",
        "These are synthetic people. They exist because production cannot measure styling "
        "quality: 163 of its 164 members are in one city, body shape is recorded on 18 "
        "accounts, the body-scan table is empty, and no member has ever accepted or rejected "
        "an outfit. Each character is specified to carry an axis the others do not.",
        "",
        _coverage(roster),
        "## Assignment",
        "",
    ]
    for stylist, people in sorted(by_stylist.items()):
        parts.append(f"**{stylist}** ({len(people)}): " + ", ".join(p.name for p in people))
        parts.append("")

    parts.append("---")
    parts.append("")
    for stylist, people in sorted(by_stylist.items()):
        parts.append(f"# {stylist}")
        parts.append("")
        for p in people:
            parts.append(_character(p))
            parts.append("---")
            parts.append("")

    text = "\n".join(parts)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out} ({len(roster)} characters)")
    else:
        print(text)


if __name__ == "__main__":
    main()
