# Image Ingestion Pipeline (V1.0)

Production-grade, first-principles backend pipeline for converting raw wardrobe/catalog images into a canonical, searchable, validated digital representation of a garment (`CanonicalGarment`).

## Architecture & Flow

```
RAW IMAGE
   │
   ▼
1. Image Classifier (MobileNetV3) ──► CATALOG / CROP / FULL_BODY
   │
   ▼
2. Facial/Person Localization & Garment Crop (RetinaFace + SAM)
   │
   ▼
3. Image -> Garment Attributes (MODA_NER / Gemini / Claude Provider + 7-Step Validation)
   │
   ▼
4. Image Digitisation (FLUX.2 with Validation & Retry Loop)
   │
   ▼
5. Image Embedding (MODA SigLIP Distilled 768d + pgvector)
   │
   ▼
6. Category Bundling (Deterministic Versioned Lookup Table)
   │
   ├───────────────────────────────┐
   ▼                               ▼
7. Layering Compatibility      8. Structural Compatibility
   (Explicit Decision Tree)       (Slot Decision Tree)
   │                               │
   └───────────────┬───────────────┘
                   ▼
9. Visual Compatibility (Deterministic Harmony Rules + VLM Fallback)
                   │
                   ▼
           CANONICAL GARMENT
```

## Core Principles

1. **Separate Bytes from Meaning**:
   - **Object Storage** (AWS S3 or MinIO) stores binary image bytes immutably under structured SHA256-addressed keys.
   - **PostgreSQL** stores garment identity, validated attributes, pipeline state runs, pgvector embeddings, compatibility evaluations, and provenance.
2. **Deterministic First**:
   - Lookup tables for category bundling (no LLMs).
   - Decision trees for layering, anatomical slots, and color/pattern harmony rules.
   - Conventional ML for perception (MobileNetV3, RetinaFace/SAM, SigLIP).
   - LLM/VLM (Gemini/Claude) only where semantic judgment or fallback is required.
3. **Provider Pattern**:
   - All AI/ML models are isolated behind interfaces (`BaseClassifierProvider`, `BaseDetectionProvider`, `BaseAttributeExtractorProvider`, `BaseDigitisationProvider`, `BaseEmbeddingProvider`, `BaseVLMProvider`).
   - Pluggable mock/heuristic providers allow instantaneous local development and hermetic CI testing without external dependencies or GPU hardware.
4. **Strict Attribute Validation Pipeline**:
   - 7-step sequence: JSON parse ➔ Schema validation ➔ Enum validation ➔ Range checks ➔ Taxonomy checks ➔ Required fields ➔ Confidence checks.
5. **Idempotency & Resiliency**:
   - Composite idempotency key: `garment_id + stage + input_hash + algorithm/model_version`.
   - Automatic skipping and caching for repeated computations.
   - Distinct classification of retryable errors, permanent failures, and human review routing.

---

## Canonical Output Contract

```json
{
  "garment_id": "garm_4e1a0df9",
  "source_image_refs": ["object://wardrobe-assets/raw/tenant_1/9f2b84...jpg"],
  "image_type": "CATALOG",
  "garment_crop_refs": ["object://wardrobe-assets/crops/tenant_1/garm_4e1a0df9_upper_body_0.jpg"],
  "attributes": {
    "category": "shirt",
    "subcategory": "oxford_shirt",
    "colour": ["white"],
    "pattern": "solid",
    "material": "cotton",
    "fit": "regular",
    "silhouette": "straight",
    "sleeve_length": "long",
    "occasion": ["smart_casual", "work"],
    "season": ["summer", "spring"],
    "layering_role": "base",
    "warmth": 0.25,
    "versatility": 0.85,
    "confidence": 0.96
  },
  "canonical_image_ref": "object://wardrobe-assets/canonical/tenant_1/garm_4e1a0df9_canonical.jpg",
  "image_embedding": [0.035, -0.012, 0.084, ...],
  "category": "TOP",
  "compatibility_features": {
    "layering": {"role": "base", "warmth": 0.25},
    "structure": {"slot": "TOP", "fit": "regular", "silhouette": "straight"},
    "visual": {"colors": ["white"], "pattern": "solid", "versatility": 0.85}
  },
  "quality_status": "APPROVED",
  "provenance": {
    "model_versions": {"classifier": "v1", "detection": "v1", "digitisation": "v1", "embedding": "v1"}
  },
  "pipeline_version": "1.0.0"
}
```

---

## REST API Specification

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/wardrobe/images` | Upload image asset (MIME validation, size limit, SHA256 deduplication) |
| `POST` | `/api/v1/wardrobe/garments` | Initiate asynchronous ingestion pipeline for an image asset |
| `GET`  | `/api/v1/wardrobe/garments/{id}` | Retrieve persistent `CanonicalGarment` record |
| `GET`  | `/api/v1/wardrobe/garments/{id}/pipeline` | Inspect stage audit history, duration, and model versions |
| `POST` | `/api/v1/wardrobe/garments/{id}/retry` | Retry or resume a failed/reviewable stage |
| `POST` | `/api/v1/wardrobe/garments/{id}/review` | Operator decision (`APPROVE`, `REJECT`, `OVERRIDE`) |
| `POST` | `/api/v1/wardrobe/compatibility` | Pairwise compatibility evaluation across layering, structure, and visual |
| `GET`  | `/health` | Service health status |
| `GET`  | `/metrics` | Pipeline telemetry metrics (latencies, counts, failure rates) |

---

## Getting Started

### Local Quickstart

```bash
# 1. Install dependencies with uv
uv sync --all-groups

# 2. Run the full test suite
uv run pytest -v

# 3. Start the FastAPI application
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### Docker Compose Deployment

```bash
docker compose up -d --build
```

This starts:
- **API service**: `http://localhost:8000`
- **PostgreSQL 16 + pgvector**: `localhost:5432`
- **MinIO S3 object storage**: `http://localhost:9000` (Console: `http://localhost:9001`)
- **Redis 7 queue**: `localhost:6379`
- **Asynchronous worker**: running background pipeline orchestrator jobs.

---

## Testing

The test suite covers unit tests, model provider contracts, rule decision trees, idempotency, pipeline state transitions, and API endpoints:

```bash
uv run pytest -v
```
