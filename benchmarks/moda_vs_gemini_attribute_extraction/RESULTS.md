# MODA vs Gemini — garment attribute extraction benchmark

20 images · 60 model calls · MODA_NER crop + fullbody tracks (structural only) · Gemini 2.5
Flash via OpenRouter · September 2026

## Methodology

Per the original spec: real wardrobe photos, deliberately hard cases, hand-labeled ground
truth, MODA cropped + MODA full body + Gemini, measure per-attribute accuracy, latency, cost,
and failure modes described rather than counted.

The test set is **20 images** — 10 informal Western wardrobe photos drawn from real member
uploads (hangers, home lighting, cluttered backgrounds, rotated EXIF, a garment cropped almost
out of frame), and 10 Indian ethnic-wear images drawn from production character wardrobes
tagged saree/lehenga/kurta/salwar/dhoti.

Ground truth was hand-labeled by direct visual inspection of each source image, not by a
professional garment archivist — treat the accuracy numbers below as directionally reliable,
not laboratory-grade.

**"MODA" here means `ModaNerAttributeExtractorProvider._extract_structural()` called
directly** — the raw RunPod MODA_NER call, bypassing the OpenRouter VLM top-up that
production always layers on top. That split matters: production MODA is already a MODA+GPT
hybrid, and benchmarking that hybrid against Gemini would really be benchmarking GPT against
Gemini with extra steps. This is MODA on its own.

### A methodology caveat on the ethnic-wear set

The 10 Indian ethnic-wear images turned out to be professional catalog/editorial photography
(studio backdrops, posed models, one an apparent runway shot) rather than informal member
snapshots — production wardrobes have almost no real member-uploaded ethnic wear to draw
from. That makes this half of the set systematically *easier* than the Western half, which is
built from genuinely hard informal photos. Where Indian-wear scores look strong below, read
that partly as "catalog photos are easier," not "these models are specifically good at Indian
clothing." Two of the ten ethnic images (the saree and the lehenga) also carry open provenance
questions — flagged during test-set construction and used in this benchmark run on explicit
instruction, but excluded from the image files committed to the repo as a precaution (see
README.md).

## Per-attribute accuracy

Fraction of the 20 images each model got right, or close enough to be usably right (synonyms
like `wide_leg_trousers`/`palazzo_pants` count). MODA's two columns are two tracks of the same
model, not two models.

| Attribute | MODA · crop | MODA · fullbody | Gemini |
|---|---|---|---|
| Category (broad) | 17/20 (85%) | 0/20 (0%, field not produced) | 19/20 (95%) |
| Subcategory | 0/20\* (0%) | 0/20 (0%, field not produced) | 17/20 (85%) |
| Colour | 0/20 (0%, field not produced) | 0/20 (0%, field not produced) | 18/20 (90%) |
| Pattern | 9/20\* (45%) | 8/20 (40%) | 18/20 (90%) |
| Material | 0/20 (0%, field not produced) | 2/20 (10%) | 11/20 (55%) |

\* MODA crop's subcategory field was present in only 5 of 20 outputs at all (25% coverage) and
wrong in all 5 — scored as 0/20 rather than 0/5 because a missing field is a failure to
deliver the attribute. Pattern is marked the same way: crop produced a pattern value for only
10 of 20 images.

## Latency & cost

| Call | Median | Mean | Range | Cost |
|---|---|---|---|---|
| MODA · crop | 2.2s | 2.2s (warm) | 1.2s–2.9s, one 28.9s cold start | RunPod GPU-second billing on a rented endpoint — not observable from this codebase; cost accrues to keep the worker warm regardless of call volume |
| MODA · fullbody | 2.4s | 2.5s | 1.2s–7.8s | same endpoint/worker as crop |
| Gemini 2.5 Flash | 4.8s | 4.8s | 3.7s–6.1s | Per-token via OpenRouter — purely marginal cost, no idle infra to keep warm; not logged with per-call cost in this environment, but priced as OpenRouter's budget vision tier (order of magnitude: fractions of a cent per image at this prompt length) |

MODA is roughly 2× faster per call once warm. That speed doesn't currently buy usable
attributes on its own — see below — but it's real, and matters for the recommendation.

## Failure modes

Described rather than counted, per the spec. Each is backed by concrete evidence in
`results.json`.

**[both tracks] MODA structurally cannot see colour.** Neither the crop track (Fashionpedia
vocabulary) nor the fullbody track (DeepFashion-MM vocabulary) maps a colour field at all —
confirmed in `app/providers/attributes/moda_ner_mapping.py`: `map_crop_track` and
`map_fullbody_track` never read or emit `colour`. Only the untested `catalog` track produces
it. In production this is invisibly patched by the OpenRouter top-up on every single call —
meaning colour, for crop/fullbody garments, is *always* coming from the VLM already, not from
MODA.

**[fullbody] Material defaults to "cotton" almost regardless of the garment.** Fullbody's
material guess was literally `cotton` on 13 of the 20 images and `denim` on 5 more —
including a satin evening gown, a wool sweater, a satin crop top called `denim`, and a bridal
lehenga in silk and velvet called `cotton`. It landed correctly only on a plain cotton dhoti
and a chiffon floral dress.
- `8c8cc402` (light-blue satin crop top) → fullbody material: `denim`
- `2fbaafbb` (Zegna wool-knit sweater) → fullbody material: `cotton`
- `ethnic_08` (silk/velvet bridal lehenga) → fullbody material: `cotton`

**[fullbody] Sleeve length is hallucinated on garments that don't have sleeves.** Fullbody's
`sleeve_length` field has no "not applicable" option in its output vocabulary, so it invents a
value even for trousers. On all three pairs of pants in the set, fullbody reported a sleeve
length; Gemini correctly returned `not_applicable` on all three.
- `7e088fa0` (palazzo pants) → fullbody: `long` · Gemini: `not_applicable`
- `812a6592` (wide-leg trousers) → fullbody: `sleeveless` · Gemini: `not_applicable`
- `6096f91f` (cargo pants) → fullbody: `long` · Gemini: `not_applicable`

**[crop] Subcategory is nearly absent, and wrong when it does appear.** Crop's subcategory
field only surfaced on 5 of 20 images. All five were wrong against ground truth: a bridal-set
kurta called "sheath," an Anarkali gown called "gown" (not in the known taxonomy), a lehenga
called "gown," a tank top called "shift," and a wool sweater called "classic" (not a real
subcategory). The other 15 images returned no subcategory from crop at all.

**[crop] One outright category miss: a kurta read as a bag/wallet.** `ethnic_03`, a plain
cream-and-blue cotton kurta on a model, came back from crop as `category: bag_wallet` — not a
near-miss, a completely different object class. This is the single worst individual failure in
the set from either MODA track.

**[gemini] Material goes blank on the hardest informal photos.** Gemini's material accuracy on
the catalog-quality ethnic set was strong (9/10 correct or close), but on two of the hardest
Western images — both satin garments shot in cluttered, low-light home setups — it returned
`null` for material rather than a wrong guess. It also missed "satin" twice elsewhere,
defaulting to cotton.
- `812a6592` (satin damask palazzo pants, rotated, no waistband visible) → material: `null`
- `382c210d` (satin-ish abstract-print top, rotated, cluttered background) → material: `null`

**[gemini] A dhoti read as a dress.** `ethnic_10`, dark quilted dhoti pants draped and
photographed against a plain backdrop, was classified by Gemini as
`category: dress, subcategory: dress, pattern: checkered`. The draped silhouette and quilted
stitching pattern both plausibly read as a dress hem and a checked print at a glance — this is
the one clear ethnic-wear miss Gemini made in this run.

## Indian ethnic wear: image-by-image

The specific ask: where and how much each model fails on Indian dresses. All 10 ethnic-wear
images, ground truth against both MODA tracks and Gemini's subcategory call.

| Image | Ground truth | MODA (crop / fullbody) | Gemini | Verdict |
|---|---|---|---|---|
| ethnic_01 saree | saree, black, solid, silk | category: dress; no subcategory, no colour; material "chiffon" (wrong); pattern "graphic" (wrong) | saree, black, solid, silk | Gemini correct on every core attribute; MODA misses subcategory, colour, material |
| ethnic_02 lehenga | lehenga, silver/pink, geometric/embellished, silk | subcategory "gown" (wrong taxonomy); no colour; material "cotton" (wrong) | lehenga, silver/pink, geometric, silk | Gemini correct on every core attribute |
| ethnic_03 kurta | kurta, cream/blue, geometric, cotton | category "bag_wallet" (badly wrong); material "denim" (wrong) | kurta, off-white/blue/yellow, geometric, cotton | Gemini correct (minor extra colour); MODA's worst single miss in the set |
| ethnic_04 kurta set | kurta + palazzo (2-piece), white/pink/gold, floral, cotton | subcategory "sheath" (wrong); material "denim" (wrong); pattern "solid" (wrong) | kurta, white/pink/purple/orange, floral, cotton | Gemini correct on the primary garment; both models miss the coordinated pants as a set |
| ethnic_05 kurta set | kurta + palazzo + dupatta (3-piece), mustard, floral, silk/georgette | no subcategory, no colour; material "cotton" (wrong); pattern "graphic" (wrong) | kurta, yellow, floral, silk | Gemini correctly identifies the primary garment; MODA misses almost everything |
| ethnic_06 salwar | salwar (severely cropped, low-res image), teal | category "pants" only (broadly right); nothing else usable | leggings (wrong garment type), teal (right colour) | Both wrong on subcategory — a genuinely hard, near-unreadable crop |
| ethnic_07 Anarkali gown | Anarkali gown, teal/gold, embroidered floral, silk | subcategory "gown" (not in taxonomy); material "denim" (wrong) | anarkali, teal/gold, floral, silk | Gemini correct on every core attribute |
| ethnic_08 bridal lehenga | lehenga, magenta/gold, heavy embroidery, silk/velvet | category "dress" only; material "cotton" (wrong) | lehenga, pink/gold/maroon, floral, silk | Gemini correct; MODA material badly wrong for a velvet/silk bridal piece |
| ethnic_09 lehenga (runway) | lehenga, red/gold, embroidered, silk/net | category "dress" only; material "chiffon" (close-ish) | lehenga, red/gold, floral, silk | Gemini correct on every core attribute |
| ethnic_10 dhoti | dhoti pants, dark brown/charcoal, solid/quilted, cotton | category "pants" (correct), pattern "solid" (correct), material "cotton" (correct) | dress (wrong category), checkered (wrong pattern), cotton (correct) | MODA's one clean win in the whole ethnic set |

### A related finding from production data, not this benchmark

Four of these images (05, 07, 08, 09) are currently mis-tagged in the live wardrobe database
as bare `subcategory: dupatta` — a full kurta set, a full Anarkali gown, and two full
lehengas, each reduced to just their scarf. That's a production-pipeline finding, from
whatever attribute call actually ran at ingestion time, not one of the 60 fresh calls made for
this benchmark. Notably, the fresh Gemini calls run here got the subcategory right on all four
of the same images — so today's Gemini, asked directly, does not reproduce that mistake. Worth
a separate look at what the ingestion pipeline actually called and why, but out of scope for
this comparison.

## Recommendation

**Gemini wins, and not narrowly.** On every attribute that was scoreable at all, Gemini
matched or beat MODA. On colour, MODA doesn't compete — it structurally can't produce the
field in either tested track. On subcategory, MODA's coverage is 25% and its accuracy within
that coverage is 0%, against Gemini's ~85%. On material, Gemini is roughly 5× more accurate
than MODA fullbody, which defaults to "cotton" often enough to look like a fallback value
rather than a prediction. MODA is only genuinely competitive on broad category (85% vs 95%)
and on latency (~2.2s vs ~4.8s, warm).

This is not a narrow win being reported as a big one: MODA loses 0% to 90%+ on two of five
attributes tested, and by roughly 2× on a third. It does not justify taking on MODA as a
standalone dependency for attribute extraction.

**On crop vs fullbody:** don't switch between them by input type — drop fullbody from the
extraction path entirely, or gate it to supply only `material` and `pattern` as weak hints,
never `sleeve_length` on bottoms (it hallucinates a value with no "not applicable" case in its
vocabulary). Crop's only defensible standalone use is a fast, cheap broad-category pre-filter
(top/bottom/dress) ahead of a VLM call — which is close to what the current hybrid pipeline
already does with its top-up.

The honest conclusion is that the current architecture — MODA supplying whatever structural
fields it can, GPT/Gemini topping up the rest on every single call — is already close to
correct, because MODA structurally cannot cover colour, and barely covers subcategory and
material. The interesting open question this benchmark doesn't answer is whether MODA is
worth keeping *at all* given how much the VLM is already doing on every call, versus running
Gemini alone and dropping the RunPod dependency and its idle-infra cost.

---

Ground truth is Claude-labeled by visual inspection, not professionally annotated — treat
percentages as directional. Raw per-image, per-model output is in `results.json`; full ground
truth is in `ground_truth.json`.
