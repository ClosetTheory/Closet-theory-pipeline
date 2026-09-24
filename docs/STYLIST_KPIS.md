# Stylist KPIs (admin page)

Per stylist, per character. Shown only to admins; stylists do not see their own numbers.
Computed live from the panel's own tables by `app/metrics/stylist_kpis.py` and served by
`GET /api/v1/admin/metrics/stylists`. No extra tables or columns.

**Target:** 20 reviews per character per week. A stylist's weekly target is 20 × the characters
currently assigned to them. The window is the trailing 7 days, both adjustable per request
(`?weekly_target=` and `?window_days=`).

| KPI | Definition | Where it shows |
|---|---|---|
| Outfits | Every outfit generated in the character's account. | cell, character, stylist ("available" = over assigned characters) |
| Reviewed | Distinct outfits with a review row. One row per (outfit, reviewer): a revised score is still one review. | cell, character (any reviewer), stylist |
| Coverage | Reviewed ÷ outfits. | all |
| Backlog | Outfits the stylist (cell) or anyone (character) has not reviewed. | all |
| This week | Reviews whose last update falls inside the window. Progress = min(this week, target) ÷ target. | all; heatmap shading |
| Avg / 5 and histogram | Mean overall rating and count per star. | all |
| Colour · Fit · Occasion · Person | Mean of each rubric dimension, when the reviewer scored it. | cards, character table |
| Depth | Share of reviews with a comment; with at least one reason chip; marked would-wear (of those answered). | cards, character table |
| Like / dislike | Count of reviews whose vote was like or dislike. | cards |
| Wardrobe built | Garments the stylist uploaded for the character, and how many finished ingestion (COMPLETED). | cell, cards, character table |
| Styling runs started | Runs the stylist launched for that character. | cell, cards |
| Hours to first review | Median hours from an outfit being generated to its earliest review, over outfits that have one. | character table |
| Engine avg | Mean of the engine's own final score across the character's outfits, next to the stylists' average, so disagreement is visible. | character table |

Layout: the page is stylist-first. Pick a stylist from the row of cards (each card shows this-week progress, so the row doubles as the comparison). Below it: six KPI tiles; reviews this week by character against a target rule; coverage by character (reviewed stacked with the backlog); the star histogram; rubric averages with a tick at the panel mean; review depth; reviews per day for the last 28 days; and a table of every character the stylist is assigned to or has touched. Per-character views are scrolling bar lists, so they scale as the roster grows.

Revoked assignments do not count toward a stylist's characters or target, but the reviews
they left while assigned still count as reviews.

Not yet measured (phase two): agreement between two stylists on the same outfit, a stylist's
leniency against the panel mean, and whether a character's engine score moved after reviews
landed.
