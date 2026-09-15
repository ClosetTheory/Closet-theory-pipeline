# Do We Need the Hopit API?

Read against the Hopit integration scope doc dated 11 Sep 2026. All figures are live counts
queried from the dev database on 15 Sep 2026.

Published version (same content, formatted): https://claude.ai/artifact/XccJ3nWzTfjogPfLaaRQXP

---

## Recommendation

**Take the keys, and scope the trial to the two search endpoints.**

`/v1/search/text` fills a capability we have nothing for — there is no text-to-garment retrieval
anywhere in our API. `/v1/search/composed` generalises something we do only in one narrow place.
Those two are worth a real benchmark.

Ingestion and styling we already do more of than they offer, and their embeddings are literally
our embeddings. Their ranker — the one piece that could beat ours — is marked **in development**
and cannot be evaluated yet.

---

## The finding that reorders everything: we already run their models

Hopit publish these weights on Hugging Face, and our RunPod serverless endpoints pull them
directly. This is not inferred from behaviour; the repo IDs are in our own worker source.

| Repo | Role | State |
|---|---|---|
| `HopitAI/moda-fashion-distilled` | ViT-B-16-SigLIP, 768-d normalised vectors. Our Stage 5 embedder (`EMBEDDING_PROVIDER=siglip`). All 1,176 stored vectors carry `model = MODA SigLIP Distilled`, `v1`. See `runpod/moda_embed.py:42`. | **In production** |
| `HopitAI/moda-ner-v-crop` / `-catalog` / `-fullbody` | Three-track attribute extraction. Endpoint deployed, provider written (`app/providers/attributes/moda_ner.py`), but Stage 3 currently runs on GPT-4o (`ATTRIBUTE_PROVIDER=openrouter`). See `runpod/moda_attributes.py:49`. | Built, not selected |

Practical consequence: on `/v1/embeddings` we should expect `cosine(theirs, ours)` near **1.0**.
If it comes back low, their hosted `space` is a *different* model from the one they publish —
itself worth knowing. `scripts/benchmark_hopit_moda.py` already measures this.

---

## The five endpoints, ordered by what we'd gain

Not their doc's order — strongest case for integrating first, weakest last. The status is their
own label from the scope doc.

### 1. `POST /v1/search/text` — Available → **REAL GAP**

- **Hopit provides:** free-text query over a candidate set of garment IDs, scored in their joint
  text-image space. Returns ranked IDs with scores.
- **We run today:** nothing. There is no text search route in the API — our 27 endpoints cover
  auth, wardrobe, compatibility, outfit-of-the-day and swaps, and none take a query string.
- **Verdict:** the single strongest reason to get keys. "Show me something black and oversized
  for dinner" is a thing we cannot answer at any price today.

### 2. `POST /v1/search/composed` — Available → **PARTIAL GAP**

- **Hopit provides:** a reference garment plus a natural-language edit ("but sleeveless, in
  navy") resolved against the candidate pool. General-purpose retrieval, any garment, any
  instruction.
- **We run today:** one narrow slice — `POST /outfits/{id}/swap/chat` (`app/styling/swap.py`).
  It only swaps a garment already inside a generated outfit, and does it by asking an LLM, not
  by retrieval in an embedding space.
- **Verdict:** they generalise a feature we built for exactly one screen. Worth benchmarking
  against our swap-by-chat on the same wardrobe.

### 3. `POST /v1/embeddings` — Available → **CONVENIENCE, NOT CAPABILITY**

- **Hopit provides:** hosted vectors with an explicit `space` parameter (`moda-768`) and a
  returned `model_version`. Managed, versioned, no cold starts.
- **We run today:** the same model, self-hosted. `vector(768)` in pgvector, 1,176 rows, one
  `model_version` across the whole table. Cold starts are real — the worker pool scales to zero.
- **Verdict:** identical weights means identical quality. What we'd buy is operations — their
  versioning discipline against our RunPod cold starts — traded for per-call cost and losing
  control of the vector store.

### 4. `POST /v1/garments:batch` — Available → **DUPLICATE, AND THINNER**

- **Hopit provides:** bulk registration, up to 200 garments per batch, images fetched by URL.
  Attribute extraction is explicitly *not in scope* — we send attributes, they don't derive them.
- **We run today:** a nine-stage ingestion pipeline (classify, crop, attributes, digitise, embed,
  category, layering, structure, visual). 1,170 garments completed of 1,384 ingested, with
  per-stage provenance and a review queue.
- **Verdict:** their ingestion is a registration call. Ours crops, digitises, extracts a 21-field
  schema and gates on quality. Adopting theirs would mean giving up all of that.

### 5. `POST /v1/outfits:rank` — **In development** → **CANNOT BE EVALUATED**

- **Hopit provides:** a learned compatibility ranker taking occasion, formality, season,
  temperature, preferences and wear history. On paper the most interesting endpoint they have.
- **We run today:** a ten-stage styling pipeline — normalisation, context, wardrobe behaviour,
  filtering, retrieval, compatibility, ranking, semantic validation, image generation, gates —
  with LLM aesthetic scoring and real colour theory in the ranker.
- **Verdict:** marked in development in their own doc, so there is nothing to benchmark. Ours
  ships today. Ask for a timeline; don't plan around it.

---

## Their four open questions, answered from the database

**Q. Which styling-relevant attributes do you populate, and how completely?**

Every field their ranker depends on is at full coverage. Of 1,170 completed garments:

| Field | Populated |
|---|---|
| `layering_role` | 1170 |
| `warmth` | 1170 |
| `season` | 1170 |
| `occasion` | 1170 |
| `casual_name` | 1170 |
| `gender` | 1167 |
| `brand_label` | 1139 |

**Q. Do you track an `available` flag per garment?**

No. It does not exist in our schema in any form — no availability, laundry or out-of-rotation
state on the garment record. If their ranker treats it as required we would need to add the
column *and* a way for members to set it. That is product work, not a field mapping.

**Q. What is a typical wardrobe size?**

One large demo wardrobe rather than many small ones — worth flagging to them, because it sets
the candidate-list size on every request.

- 1,168 garments on `member_1`
- 1,170 completed overall
- 1,384 ingested overall

**Q. What is the mix of image types?**

Their doc calls image type the single largest driver of retrieval quality, so this is the number
to hold them to. Ours is a user-photo wardrobe, not a catalogue one — **73.5% of images were
taken by a person**, not supplied by a retailer. Any accuracy figure they quote on catalogue
imagery should be discounted accordingly.

| Type | Share | Count |
|---|---|---|
| `FULL_BODY` | 48.4% | 553 |
| `CATALOG` | 26.4% | 302 |
| `CROP` | 25.1% | 287 |

---

## Two things to settle before we point them at anything

**Their ingestion fetches our images by URL.** That works today because
`/api/v1/wardrobe/images/media/...` is served **unauthenticated** — verified returning HTTP 200
with image bytes to an anonymous request. It is how the integration would function, and equally
how anyone with a URL could pull member photos. Decide deliberately: signed time-limited URLs
for the vendor, or accept the exposure while the wardrobe is demo data only.

**Both search endpoints take a full candidate list.** With 1,168 garments on one member, that is
roughly 1,200 IDs on every search request. Confirm they will accept a payload that size, or that
garments stay resident server-side after `/v1/garments:batch` so we can pass a filter instead of
an enumeration.

---

## What to ask for alongside the keys

The scope doc specifies request and response shapes but contains no base URL and no auth scheme,
so neither can be inferred.

1. **Base URL and auth scheme.** Header name and token format. Our client assumes
   `Authorization: Bearer` and falls back to a raw header, both overridable.
2. **Whether `moda-768` is `moda-fashion-distilled`.** If it is the same space we already store,
   we can reuse our vectors instead of re-embedding 1,176 garments through their API.
3. **The `candidates` cap**, and whether a wardrobe persists server-side between calls.
4. **A date for `/v1/outfits:rank`**, plus what it was trained on — catalogue imagery or user
   photos.
5. **Rate limits and pricing per endpoint**, since the embeddings case is entirely a
   cost-versus-operations trade.

---

## Ready to run

`scripts/benchmark_hopit_moda.py` is committed and dry-run verified. It exercises all four
available endpoints against real garments, times each one, and diffs their vectors against our
stored SigLIP embeddings.

```bash
export HOPIT_BASE_URL=https://...
export HOPIT_API_KEY=...
python -m scripts.benchmark_hopit_moda --limit 50
```

Numbers the same hour the keys land.
