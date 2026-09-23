# MODA slot assignment — image or text? Results

Run 2026-09-23 against `https://api.closettheory.hopit.ai` (`model_version: moda-pro-lite@1`,
`compatibility_source: embedding_proxy`). Key scopes: `ingest,styling`. The `embeddings` and
`search` scopes were refused (`403 scope_forbidden`), so vectors could not be read directly;
everything below is inferred from what `/v1/outfits:rank` does with the same photo under
different labels.

## The answer

**MODA's outfit slot comes entirely from the text we send. The image is never consulted for it.**

- A shirt photo labelled `category: accessory, subcategory: bag` is placed in the **`bag`**
  slot. The same photo labelled `category: bottom` becomes a bottom, `category: footwear`
  becomes footwear, and with no attributes at all it is dropped into a catch-all **`other`**
  slot. A genuine handbag photo labelled `category: top` is happily used as the outfit's top.
- The image drives only the pairwise **compatibility** factor. Every text field can be
  scrambled (colour, material, pattern, occasion, season, fit, warmth, formality) and
  compatibility stays at exactly `0.8218` for the same shirt + jeans + boots. So the stored
  `pro-lite-768` vector is an image-only embedding — consistent with it being the same
  `HopitAI/moda-fashion-distilled` SigLIP model our own Stage 5 runs, which also embeds the
  image alone (`runpod/moda_embed.py` calls `encode_image` only).
- Two factors *do* read the text: `colour_harmony` (drops from `1.0` to `0.8` when the colour
  list is changed to neon green + red) and `text_fit` (the free-text query is matched against
  the attribute text: the copy labelled `occasion: wedding` wins "a formal wedding outfit",
  the copy labelled casual/light blue wins "casual light blue denim shirt look").

MODA's own docs say attribute extraction is out of scope and `/v1/garments/{id}` confirms it:
the stored `attributes` are a byte-for-byte echo of what we sent, and a garment sent without
attributes has `attributes: {}` forever. There is no image-derived category anywhere in the
service, so nothing can catch a wrong label.

## Why the "bag" shirt came under accessories

`scripts/moda_ingest_wardrobes.py` sends `attributes.category` and `attributes.subcategory`
from our Stage 3 output. For the mis-named item that was `accessory` / `bag`. MODA does not
recognise `accessory` as a slot word (see the vocabulary table — it falls to `other`), so it
falls back to the subcategory, `bag` → slot `bag`. Our Hopit adapter then copies that slot
name straight into the outfit's `roles`, and the styling page shows `role || category`, so the
garment is rendered as an accessory on both sides.

The same shirt, under our own pipeline, would never have been an accessory: Stage 3 derives
`category` from the image, and there is no API that lets a name override it. The mislabel only
exists because the text was edited before it reached MODA.

## A. The user's case and its neighbours (same shirt photo, different labels)

Probed with jeans + boots (reveals a top), then with a real top + jeans + boots (reveals any
extra slot), then with top + boots (bottom) and top + jeans (footwear).

| Label sent with the shirt photo | Slot MODA gives it |
|---|---|
| correct (`top` / `button_down_shirt`) | `top` |
| `category: accessory`, `subcategory: bag` (**the reported case**) | `bag` |
| `category: accessory` only | `other` |
| `subcategory: bag` only | `bag` |
| no attributes at all | `other` |
| `category: top`, `subcategory: bag` | `top` (category wins) |
| `category: bag` | `bag` |
| `category: ACCESSORY`, `subcategory: BAG` (uppercase) | `bag` (case-insensitive) |
| `category: xyz_nonsense` | `other` |
| `casual_name: bag` only | `other` (name is ignored) |
| `layering_role: accessory` only | `other` (ignored) |
| `garment_class: BAG` only | `bag` (read, even though our ingest never sends it) |
| `category: bottom` | `bottom` |
| `category: footwear` | `footwear` |
| `category: shirt` | `top` |
| `category: accessories` (plural) | `other` |
| `category: ""` | `other` |
| `category: ["accessory","bag"]` | **rejected at ingest**: `attribute_invalid: category must be a string, got list` |
| `category: 3` | **rejected at ingest**: `attribute_invalid: category must be a string, got int` |

Mirror image, real handbag photo:

| Label sent with the bag photo | Slot |
|---|---|
| correct (`accessory` / `bag`) | `bag` |
| `category: top`, `subcategory: button_down_shirt` | `top` — a handbag is used as the top of the outfit |
| no attributes | `other` |

## B. Precedence when the role-bearing fields disagree

| category | subcategory | garment_class | Slot |
|---|---|---|---|
| `accessory` | `bag` | `SHIRT` | `top` |
| — | `bag` | `SHIRT` | `top` |
| `top` | — | `BAG` | `top` |
| `ACCESSORY` | `bag` | `BAG` | `bag` |
| `top` | `bag` | — | `top` |

Order of precedence is **category → garment_class → subcategory**, where `accessory` /
`accessories` do not count as a recognised category and drop through. Our ingest script only
sends `category` and `subcategory`, so in practice: category if MODA knows the word, otherwise
subcategory.

## C. The reported case inside a realistic pool

| Candidates | Result |
|---|---|
| shirt-as-bag, real bag, jeans, boots | **no outfit**: "no wearable combination found in the candidate set" — nothing is a top |
| shirt-as-bag, real top, real bag, jeans, boots | outfit = top + jeans + boots + **shirt-as-bag in the `bag` slot** (the real bag loses); score 0.6376 |
| shirt-as-bag, jeans, boots, blazer, dress | dress + blazer + boots; the shirt is left out entirely |

## D. `required_slots` cannot rescue it

| Request | Result |
|---|---|
| shirt-as-bag + jeans, `required_slots: [top, bottom]` | "no candidates for required slot(s): top" |
| bare shirt image + jeans, `required_slots: [top, bottom]` | "no candidates for required slot(s): top" |
| shirt-as-bag + jeans, `required_slots: [accessory, bottom]` | "no candidates for required slot(s): accessory" — the slot is called `bag`, not `accessory` |

## E. A free-text query cannot rescue it

"a casual outfit with a light blue shirt" over {shirt-as-bag, real bag, jeans} and over
{shirt-as-bag, bare shirt image, jeans}: no outfit either time. The query is interpreted
(`query_interpretation.engaged: true`) but only re-scores outfits that already exist.

## F. Which score factors move when only the text changes

Same shirt photo + jeans + boots:

| Variant | compatibility | colour_harmony | score |
|---|---|---|---|
| correct labels | 0.8218 | 1.0 | 0.6333 |
| `category: shirt` | 0.8218 | 1.0 | 0.6333 |
| `category: top`, `subcategory: bag` | 0.8218 | 1.0 | 0.6333 |
| every non-category field scrambled (neon green/red, rubber, floral, wedding, winter…) | **0.8218** | **0.8** | 0.6033 |

A worse consequence of image-only compatibility: the shirt photo labelled `bottom`, paired
with the same shirt as `top` and boots, scores compatibility **0.8591** — higher than the real
jeans do with that shirt (0.8218). Two copies of the same garment look maximally "compatible" to
an embedding proxy, so a mislabel is rewarded rather than caught.

## V. Which category words MODA recognises

Measured by ingesting the shirt photo with `attributes: {"category": <word>}` alone.

| Slot | Words that map to it |
|---|---|
| `top` | top, TOP, Top, shirt, blouse, tshirt, t-shirt, sweater, tank_top, kurta, vest², two_piece_set² |
| `bottom` | bottom, BOTTOM, pants, trousers, jeans, skirt, shorts, leggings, bottomwear², hosiery², garment² |
| `dress` | dress, jumpsuit, one-piece² |
| `outer` | outerwear, outer, jacket, coat, blazer, cardigan, suit², suit_jacket² |
| `footwear` | footwear, shoes, boots, sneakers, heels, sandals |
| `bag` | bag, handbag, BAG |
| `hat` | hat, cap, headwear² |
| `accessory` | scarf, gloves² |
| `belt` | belt |
| `eyewear` | sunglasses |
| `jewellery` | jewellery |
| `watch` | watch |
| **`other`** (not a wearable slot) | **accessory, accessories, ACCESSORY**, one_piece, romper, traditional, saree, dupatta, swimwear, underwear, anything unknown, empty, or missing |

² Only observed together with a recognised subcategory (e.g. `bottomwear` + `jeans`,
`headwear` + `hat`, `garment` + `jeans`), so the subcategory may be what resolved it. Every
other word was measured on its own.

Note that `cardigan` is an outer layer to MODA but a `TOP` in our taxonomy, and `one_piece` —
the word our own Stage 3 uses for the whole dress family — is **not** recognised (only the
subcategory rescues `jumpsuit`; `romper` and `overalls` do not).

## P. Coverage of the labels our Stage 3 actually emits

All 127 distinct `(attributes.category, attributes.subcategory)` pairs found across the
evaluation roster's garments were sent with the shirt photo. 123 pairs resolve to a wearable
slot; these 4 (14 garments in the scan) fall into `other`, where MODA appends them to outfits
as a free-floating extra instead of styling them as what they are:

| category | subcategory | garments | slot |
|---|---|---|---|
| one_piece | romper | 7 | other |
| one_piece | overalls | 3 | other |
| accessory | dupatta | 3 | other |
| one_piece | anarkali | 1 | other |

Everything under `accessory` / `accessories` / `headwear` only works because the subcategory
(bag, hat, scarf, belt, sunglasses, …) is recognised. An accessory whose subcategory MODA does
not know (`dupatta`, or any new one) silently becomes `other`.

## What this means for the Moda styling pane

1. **Garbage in, garbage out with no safety net.** MODA trusts our labels absolutely, so the
   Moda pane is exactly as accurate as `attributes.category` / `attributes.subcategory` at
   ingest time. Any edit to those fields (or any Stage 3 error) changes the outfit slot without
   anything on either side noticing.
2. **Send the role we already resolved.** `scripts/moda_ingest_wardrobes.py` passes Stage 3's
   free-form words (`shirt`, `pants`, `accessory`, `one_piece`, `headwear`…). It would be safer
   to also send `garment_class` (which MODA reads and which our `bundle_garment_class` already
   normalises) or to map our canonical `TOP/BOTTOM/ONE_PIECE/OUTERWEAR/FOOTWEAR/ACCESSORY`
   onto MODA's words (`top/bottom/dress/outer/footwear` + the specific accessory word), so
   `one_piece` and unknown accessories stop falling into `other`.
3. **MODA's slot names are not our role names.** The adapter copies `top`, `bottom`, `dress`,
   `outer`, `bag`, `hat`, `eyewear`, `other`… straight into `OutfitCandidate.roles`, where the
   rest of the pipeline and the UI expect `TOP`, `BOTTOM`, `ONE_PIECE`, `OUTERWEAR`,
   `ACCESSORY`. Anything keyed on role (layering rules, the swap modal, aesthetic summaries)
   sees a different vocabulary on the Hopit path than on ours.
4. **Re-ingest after relabelling.** The ingest script skips every garment id already in its
   ledger, so correcting a label locally does nothing until the id is removed from the ledger
   (or the record is re-sent; MODA reports `unchanged` only when content is identical).

## Method notes

Every variant is the same two source photos from one character's wardrobe (a light-blue
button-down shirt and a rattan handbag), served from the deployment's public media route, plus
one genuine jeans / boots / dress / blazer each as outfit partners. All records went under a
scratch `owner_ref`, and were deleted with `DELETE /v1/owners/{owner_ref}/garments` afterwards
(`deleted: 120` for the exploratory rounds, `deleted: 228` for the interrupted consolidated run).

`results.json` is assembled from the three exploratory rounds that produced the numbers above.
`run_benchmark.py` re-implements those rounds as one reproducible pass; its own end-to-end run
on the day was stopped by the local session harness (host memory pressure) after ingest and
before the probes finished, so re-run it when convenient to regenerate `results.json` in one go.
