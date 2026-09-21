# MODA vs Gemini — garment attribute extraction benchmark

20 real garment images (10 informal Western wardrobe photos, 10 Indian ethnic-wear images),
hand-labeled ground truth, run through MODA_NER (crop + fullbody tracks, structural output
only — `ModaNerAttributeExtractorProvider._extract_structural()`, bypassing the OpenRouter
VLM top-up production always layers on) and Gemini 2.5 Flash (via OpenRouter). Full
methodology, per-attribute accuracy tables, latency/cost, failure modes, and an
image-by-image breakdown of all 10 Indian ethnic-wear items are in `RESULTS.md`.

## Files here

- `ground_truth.json` — hand-labeled ground truth for all 20 images (Claude-labeled by visual
  inspection, not a professional annotator — treat as directional).
- `results.json` — raw output of all 60 model calls (MODA-crop, MODA-fullbody, Gemini ×
  20 images), including latency per call.
- `run_benchmark.py` — the runner script. Re-running it requires a local copy of the 20 source
  images (see "Source images" below) and working `RUNPOD_API_KEY` /
  `RUNPOD_ATTRIBUTE_ENDPOINT_ID` / `OPENROUTER_API_KEY` in `.env`.
- `RESULTS.md` — the full write-up.

## Source images: not committed here

The 20 source images are **not** in this folder or committed to the repo. Two of the ten
Indian ethnic-wear images carry open provenance questions — a "for reference purpose only"
watermark on one, and an apparent public figure in a candid-style shot on the other — flagged
during test-set construction. They were used in this benchmark run on explicit instruction,
but committing raw image files with unresolved rights/provenance questions into permanent git
history is a separate, harder-to-reverse decision than running a one-off benchmark call, so
they were kept out of the repo. `ground_truth.json` and `results.json` fully describe what was
in each image and what each model returned, so the comparison is auditable without the images
themselves.
