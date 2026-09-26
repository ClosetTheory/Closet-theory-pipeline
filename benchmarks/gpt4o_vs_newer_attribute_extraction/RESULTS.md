# GPT-4o vs newer vision models — Stage 3 attribute extraction

23 real member-uploaded garment photos · 8 models · the exact production prompt and payload from
`OpenRouterGPTProvider.extract_attributes` · 26 September 2026.

## Why

`OPENROUTER_MODEL` has been `openai/gpt-4o` (a 2024 model) since the pipeline was built. The
question was whether it is still good at the job and whether a newer, cheaper model beats it.

## Method

- Images: the first 24 raw photos in `unknown_material_samples/images/` (gitignored, real
  wardrobe uploads on a white sheet: hangers, rotated EXIF, hands in frame). Image 21, a street
  photo of a person wearing an outfit, was excluded as unlabellable.
- Ground truth: hand-labelled by visual inspection (broad category, accepted subcategories,
  accepted colours, accepted pattern words). Directional, not archival.
- Calls: `run_benchmark.py` captures the live production prompt, then sends the production
  payload (`max_tokens` 1800, `temperature` 0.1, `response_format` json_object) to each model
  through OpenRouter with `usage.include` so the cost is the real billed cost.
- "valid" = passed `validate_extracted_attributes`, i.e. would have gone into the pipeline
  without a cross-model retry.

## Results

| Model | Schema-valid | Category | Subcategory | Colour | Pattern | p50 latency | Cost / image |
|---|---|---|---|---|---|---|---|
| openai/gpt-4o (current) | 23/23 | 20/23 | 20/23 | 23/23 | 22/23 | 6.7s | $0.0103 |
| openai/gpt-6-sol | 23/23 | 21/23 | 20/23 | 23/23 | 23/23 | 15.2s | $0.0350 |
| openai/gpt-6-luna | 22/23 | 19/23 | 19/23 | 23/23 | 20/23 | 13.5s | $0.0019 |
| google/gemini-3.5-flash-lite | 15/23 | 21/23 | 21/23 | 22/23 | 21/23 | 4.1s | $0.0025 |
| google/gemini-3.8-flash | 8/23 | 9/23 | 9/23 | 9/23 | 9/23 | 16.0s | $0.0086 |
| anthropic/claude-sonnet-5 | 9/23 | 9/23 | 9/23 | 9/23 | 9/23 | 14.7s | $0.0264 |
| qwen/qwen3.8-flash | 8/23 | 8/23 | 7/23 | 8/23 | 8/23 | 21.8s | $0.0011 |
| z-ai/glm-5.3-flash | 1/23 | 1/23 | 1/23 | 1/23 | 1/23 | 23.4s | $0.0018 |

Accuracy columns count only responses that came back; a schema-invalid response still scores if
its fields were right.

## Reading it

- **GPT-4o is still good.** Every response parsed and validated, colour was perfect, and its
  three category misses were the genuinely ambiguous photos: a rust camisole read as a sundress,
  a printed oversized shirt read as a dress, and a white sleeveless belted top read as a blazer.
  It was also the only model to identify the black jacket photographed sideways (#16) as a
  jacket; both GPT-6 models called it trousers.
- **GPT-6 Sol is the quality ceiling**, one category and one pattern better than GPT-4o, but
  3.4x the cost and twice the latency.
- **GPT-6 Luna is the cost story**: 5x cheaper than GPT-4o with one more category miss and two
  more pattern misses. It is twice as slow per call, which matters because Stage 3 runs the
  extraction inline.
- **Gemini 3.5 Flash Lite sees well but does not follow the schema.** Best raw category and
  subcategory scores and the fastest, but 8 of 23 responses failed validation, seven of them on
  enum fields (`sleeve_length` on trousers instead of null, invented `fit` and `occasion`
  values). Each of those would trigger a cross-model retry in production. It is also now the
  Stage 2/3 verifier, so it cannot be the extractor without breaking the different-vendor check.
- **Gemini 3.8 Flash, Sonnet 5, Qwen and GLM were not fairly tested by the production payload.**
  Gemini 3.8 Flash's responses were truncated JSON (its reasoning tokens eat the 1800-token
  budget). Sonnet 5 returned HTTP 400 on 14 images (request shape rejected by the Anthropic
  provider). Qwen and GLM returned non-JSON or empty content and Qwen rate-limited. Any of them
  would need prompt and payload changes before a real comparison.

## Recommendation

Keep `openai/gpt-4o` unless cost is the problem. If it is, `openai/gpt-6-luna` is the drop-in
at a fifth of the price with a small accuracy cost, and `openai/gpt-6-sol` is the upgrade if
accuracy matters more than the bill. Both keep the extractor on OpenAI, which the cross-vendor
verifier design requires.

## Files

- `run_benchmark.py` — the runner. Needs `OPENROUTER_API_KEY` in `.env` and the sample images
  in `unknown_material_samples/images/`.
- `results.json` — every response (184 calls) with latency and billed cost.
