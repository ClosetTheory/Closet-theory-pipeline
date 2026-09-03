# ClosetTheory — Garment Intelligence & Styling Engine Specification

**Status:** Canonical Implementation Specification
**Authority:** This document overrides implementation assumptions where they conflict with the specification.
**Scope:** Garment ingestion, garment representation, taxonomy, compatibility intelligence, retrieval, and styling decision engine.

---

## 1. Purpose

This document defines the canonical technical specification for ClosetTheory's **Garment Intelligence** and **Styling Engine**.

It establishes:

* the garment ingestion pipeline;
* image representation classes;
* canonical garment taxonomy;
* category bundling;
* garment attributes;
* pairing and layering relationships;
* structural and visual compatibility;
* embeddings and retrieval;
* garment persistence;
* styling inputs and decision flow;
* behavioural and contextual signals;
* outfit scoring and construction;
* visual and semantic validation gates;
* external MODA/Hopit dependencies.

The implementation should be evaluated against this specification.

### Implementation principle

> **The specification is authoritative. Existing implementation should conform to the specification rather than redefining it.**

An implementation agent may determine that an existing implementation is already compliant, partially compliant, or requires changes. It should not alter the canonical design simply to match existing code.

---

# 2. Architectural Principles

| Principle                   | Requirement                                                                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Ingestion ≠ Upload          | Upload/source acquisition and garment ingestion are separate concerns. Ingestion begins once an image is available for processing. |
| Staged processing           | Ingestion consists of explicit processing stages.                                                                                  |
| DAG execution               | Stages are connected through dependencies.                                                                                         |
| Parallel execution          | Independent operations within a stage may execute concurrently.                                                                    |
| Deterministic logic         | Taxonomy, bundling, hard compatibility and filtering should be deterministic wherever practical.                                   |
| ML for perception           | ML/VLM models are used where visual or semantic interpretation is required.                                                        |
| Retrieval ≠ decision        | Vector retrieval generates candidates; it does not make the final styling decision.                                                |
| ClosetTheory owns decisions | External models provide perception/retrieval capabilities. ClosetTheory owns compatibility, scoring and final outfit construction. |
| Explicit failures           | Each stage must have defined failure behaviour.                                                                                    |
| Canonical taxonomy          | Model labels must resolve into a controlled ClosetTheory taxonomy.                                                                 |
| Preserve detail             | Bundling must not destroy fine-grained garment information.                                                                        |

---

# 3. Garment Ingestion

## 3.1 Canonical Pipeline

The production ingestion pipeline is:

|  # | Stage                             | Technology / Owner                 |
| -: | --------------------------------- | ---------------------------------- |
|  1 | Image Classifier                  | **In-house — MobileNetV3**         |
|  2 | Facial Recognition & Crop         | **In-house — RetinaFace + SAM**    |
|  3 | Image → Attributes                | **MODA_NER**                       |
|  4 | Image Digitisation                | **FLUX.2**                         |
|  5 | Category Bundling                 | **In-house — Lookup Table**        |
|  6 | Layering Compatibility Analysis   | **In-house — Decision Tree**       |
|  7 | Structural Compatibility Analysis | **In-house — Decision Tree**       |
|  8 | Visual Compatibility Analysis     | **In-house — Decision Tree + VLM** |
|  9 | Image Embedding                   | **MODA SigLIP Distilled**          |
| 10 | Garment Variety Generation        | **TBU**                            |

---

## 3.2 Stage Execution Model

The black boxes in the architecture represent **execution stages**.

Operations inside the same stage that do not depend upon each other may execute in parallel.

For example:

```text
                    ┌── Operation A ──┐
                    │                 │
Input ──────────────┼── Operation B ──┼──► Stage Barrier
                    │                 │
                    └── Operation C ──┘
```

A downstream stage must not consume incomplete outputs from the preceding stage.

The pipeline therefore behaves as a dependency DAG with explicit stage barriers.

---

# 4. Image Representation Classification

The image classifier determines **what kind of source representation is being processed**.

It does **not** determine the final garment taxonomy.

## 4.1 Canonical representation classes

```text
CATALOG
CROP
FULL_BODY
UNKNOWN
```

### `CATALOG`

A clean product/catalog representation of a garment.

Typical characteristics:

* garment is the primary subject;
* minimal environmental context;
* usually isolated or controlled background;
* suitable for direct garment analysis.

### `CROP`

A cropped garment representation where the garment is the primary subject, but the source does not necessarily constitute a clean catalog image.

### `FULL_BODY`

An image containing a person wearing one or more garments.

The system may need to identify individual garments within the image.

### `UNKNOWN`

Used when the classifier cannot confidently assign a valid representation.

Low-confidence input must not be silently forced into another class.

---

# 5. Facial Detection and Cropping

For images containing people, the pipeline must isolate or protect the relevant garment information.

Current implementation technology:

**RetinaFace + SAM**

The exact internal implementation may distinguish between:

* face detection;
* person segmentation;
* garment segmentation;
* cropping/redaction.

The important output requirement is that irrelevant facial/person information does not interfere with downstream garment analysis or catalog representation.

This stage must preserve garment pixels required by downstream attribute extraction.

---

# 6. Image → Attribute Extraction

The system extracts structured garment attributes from the image.

### Option A

**MODA_NER**

The exact MODA NER variant should correspond to the image representation:

| Input     | Appropriate MODA track |
| --------- | ---------------------- |
| Catalog   | Catalog NER            |
| Full body | Full-body NER          |
| Crop      | Crop NER               |

The output must be normalized into ClosetTheory's canonical attribute schema rather than allowing raw model output to become the permanent schema.

---

# 7. Image Digitisation

## 7.1 Definition

**Image digitisation means producing a clean, catalogable garment image.**

It is **not embedding generation**.

It is also distinct from attribute extraction.

Current technology:

**FLUX.2**

### Input

Source garment representation.

### Output

Standardized garment representation suitable for:

* catalog display;
* downstream visual analysis;
* embedding generation;
* garment comparison;
* styling visualization.

The digitisation process must preserve relevant garment characteristics and must not introduce material changes to:

* garment type;
* colour;
* pattern;
* silhouette;
* major construction details.

---

# 8. Canonical Garment Taxonomy

The garment taxonomy is controlled by ClosetTheory.

External model classifications must be mapped into this taxonomy.

The taxonomy has two levels:

```text
Fine-grained garment class
          ↓
Canonical category bundle
```

For example:

```text
T_SHIRT
   ↓
TOP
```

and:

```text
CARGO_PANTS
   ↓
BOTTOM
```

The system must retain both levels.

---

# 9. Fine-Grained Garment Classes

## 9.1 Tops

```text
T_SHIRT
SHIRT
BLOUSE
POLO
TANK_TOP
CROP_TOP
TUBE_TOP
SWEATER
SWEATSHIRT
HOODIE
CARDIGAN
VEST
TOP_OTHER
```

## 9.2 Bottoms

```text
JEANS
TROUSERS
CHINOS
CARGO_PANTS
SHORTS
SKIRT
LEGGINGS
JOGGERS
BOTTOM_OTHER
```

## 9.3 One-Piece Garments

```text
DRESS
JUMPSUIT
ROMPER
OVERALLS
ONE_PIECE_OTHER
```

## 9.4 Outerwear

```text
BLAZER
SUIT_JACKET
JACKET
COAT
TRENCH_COAT
BOMBER
DENIM_JACKET
OVERSHIRT
OUTERWEAR_OTHER
```

## 9.5 Footwear

```text
SNEAKERS
FORMAL_SHOES
LOAFERS
BOOTS
SANDALS
HEELS
FLIP_FLOPS
FOOTWEAR_OTHER
```

## 9.6 Accessories

```text
BELT
HAT
CAP
SCARF
TIE
BAG
WATCH
JEWELLERY
SUNGLASSES
ACCESSORY_OTHER
```

## 9.7 Traditional Garments

```text
SAREE
DHOTI
KURTA
LEHENGA
SHERWANI
SALWAR
DUPATTA
TRADITIONAL_OTHER
```

## 9.8 Other

```text
INNERWEAR
ACTIVEWEAR
OTHER
```

The taxonomy is extensible, but new classes should be deliberately introduced and mapped rather than being added automatically from model outputs.

---

# 10. Category Bundling

Bundling converts fine-grained garment classes into broader functional categories.

Examples:

| Fine-grained class | Bundle      |
| ------------------ | ----------- |
| `T_SHIRT`          | `TOP`       |
| `SHIRT`            | `TOP`       |
| `BLOUSE`           | `TOP`       |
| `POLO`             | `TOP`       |
| `JEANS`            | `BOTTOM`    |
| `CARGO_PANTS`      | `BOTTOM`    |
| `TROUSERS`         | `BOTTOM`    |
| `SKIRT`            | `BOTTOM`    |
| `SHORTS`           | `BOTTOM`    |
| `DRESS`            | `ONE_PIECE` |
| `JUMPSUIT`         | `ONE_PIECE` |
| `BLAZER`           | `OUTERWEAR` |
| `JACKET`           | `OUTERWEAR` |
| `SNEAKERS`         | `FOOTWEAR`  |
| `LOAFERS`          | `FOOTWEAR`  |

Traditional garments must retain their traditional classification where required.

### Required data model

A garment should conceptually contain:

```text
garment_class = T_SHIRT
category       = TOP
```

rather than:

```text
category = TOP
```

Fine-grained class information is required because two garments within the same bundle can have different:

* layering rules;
* structural compatibility;
* visual compatibility;
* styling roles;
* contextual suitability.

---

# 11. Garment Attributes

The normalized garment record should support attributes covering at minimum:

### Identity

```text
garment_id
user_id
source_image
```

### Representation

```text
representation_type
```

### Classification

```text
garment_class
category
subcategory
```

### Visual

```text
primary_colour
secondary_colours
pattern
texture
material
finish
```

### Construction

```text
silhouette
fit
length
sleeve_length
neckline
waist_style
rise
```

### Context

```text
season
weather_suitability
setting_suitability
formality
```

### Styling

```text
layer_role
pairing_compatibility
layering_compatibility
structural_compatibility
visual_compatibility
```

The exact attribute set may expand, but model-specific schemas should not become the application's canonical schema.

---

# 12. Garment Relationship Model

ClosetTheory must distinguish between:

1. **Pairing**
2. **Layering**
3. **Structural compatibility**
4. **Visual dissociation**

These are not interchangeable concepts.

---

# 13. Pairing

Pairing asks:

> **Can these garments meaningfully be worn together as part of an outfit?**

Examples:

```text
SHIRT + JEANS
T_SHIRT + CARGO_PANTS
BLAZER + TROUSERS
```

Pairing can incorporate both:

* generic fashion rules;
* user-specific historical pairing behaviour.

Historical pairing is a ranking signal, not an absolute physical rule.

---

# 14. Layering

Layering is directional.

For example:

```text
T_SHIRT
   ↓
SHIRT
   ↓
JACKET
```

represents:

```text
T_SHIRT = inner layer
SHIRT   = middle layer
JACKET  = outer layer
```

Layering rules must understand:

* whether a garment can be worn under another;
* whether it can be worn over another;
* whether multiple instances are meaningful;
* whether the resulting layer order is valid.

Examples of generally valid relationships:

```text
T_SHIRT → SHIRT
T_SHIRT → HOODIE
SHIRT → JACKET
```

Examples of incompatible layering relationships:

```text
T_SHIRT → T_SHIRT
SHIRT → SHIRT
```

The precise rule set must be explicitly represented rather than inferred ad hoc by the styling engine.

---

# 15. Structural Compatibility

Structural compatibility represents **hard constraints**.

It answers:

> **Can these garments structurally coexist in the same outfit configuration?**

A structural incompatibility produces a hard rejection.

Example:

```text
SAREE + SKIRT
```

→ **REJECT**

Structural compatibility must not be conflated with aesthetics.

---

# 16. Visual Dissociation

Visual dissociation represents **soft incompatibility**.

It answers:

> **Even though these garments can technically be worn together, do they produce a visually incoherent combination?**

Example:

```text
FULL_LENGTH_SHIRT + SHORTS
```

may be structurally valid while producing an undesirable silhouette or visual relationship.

Therefore:

```text
STRUCTURAL INCOMPATIBILITY
        ↓
      REJECT

VISUAL DISSOCIATION
        ↓
      PENALTY
```

Visual dissociation should normally affect ranking rather than cause immediate rejection.

---

# 17. Compatibility Hierarchy

Candidate outfits should pass through progressively softer constraints:

```text
Candidate garments
       │
       ▼
Structural compatibility
       │
   ┌───┴───┐
   │       │
 FAIL     PASS
   │       │
REJECT     ▼
      Layering compatibility
             │
        ┌────┴────┐
        │         │
      FAIL       PASS
        │         │
      REJECT      ▼
            Visual compatibility
                    │
                    ▼
               Score / penalty
```

Pairing history, wear behaviour and user preferences then affect ranking.

---

# 18. Image Embedding

Embeddings provide the representation required for similarity and retrieval.

### Option A

**MODA SigLIP Distilled**

The embedding layer should support:

* image → garment retrieval;
* visual similarity;
* semantic retrieval;
* candidate discovery;
* recommendation candidate generation.

Embedding generation is independent from image digitisation.

---

# 19. Vector Retrieval

The vector store is a **candidate retrieval system**.

It is not the final styling engine.

### Vector retrieval may answer:

> “Which garments in this wardrobe are visually/semantically similar to this concept?”

It should not independently answer:

> “Which outfit should the user wear?”

That decision requires structured constraints, compatibility, context, behaviour and scoring.

---

# 20. Garment Persistence

A canonical garment record should conceptually contain:

```text
Garment
│
├── Identity
│   ├── garment_id
│   └── user_id
│
├── Source
│   ├── source_image
│   └── representation_type
│
├── Classification
│   ├── garment_class
│   ├── category
│   └── subcategory
│
├── Attributes
│   ├── visual
│   ├── construction
│   └── contextual
│
├── Digitised Image
│
├── Embedding
│
├── Compatibility
│   ├── pairing
│   ├── layering
│   ├── structural
│   └── visual
│
└── Behaviour
    ├── wear_count
    ├── last_worn
    └── mostly_worn_with
```

The relational database remains the source of truth for structured garment metadata.

The vector store contains the retrieval representation.

---

# 21. Styling Engine

The styling engine is shared across all ClosetTheory styling touchpoints.

### Supported touchpoints

1. Outfit of the Day
2. Recommendations
3. Chat recommendations
4. Bulk outfit creation
5. Playground garment swapping

These should use the same underlying styling decision system rather than separate implementations.

---

# 22. Styling Decision Ownership

The styling engine is deliberately hybrid.

## Relational / structured layer

Responsible for:

* hard filtering;
* garment category;
* garment attributes;
* weather suitability;
* setting suitability;
* explicit user constraints;
* structural compatibility;
* deterministic exclusions.

## Vector / MODA retrieval

Responsible for:

* candidate discovery;
* visual similarity;
* semantic similarity;
* image → garment retrieval;
* text → garment retrieval where supported.

## Compatibility engine

Responsible for:

* pairing;
* layering;
* structural compatibility;
* visual dissociation.

## Behaviour engine

Responsible for:

* wear frequency;
* recency;
* pairing history;
* exploration;
* familiarity.

## Scoring engine

Responsible for the **actual styling decision**.

It ranks valid candidate combinations according to all available signals.

## VLM / LLM

Responsible for:

* ambiguity resolution;
* natural-language request interpretation;
* visual validation;
* semantic validation;
* structured feedback for retries.

---

# 23. Styling Pipeline

| Stage                                 | Owner / Technology                        |
| ------------------------------------- | ----------------------------------------- |
| Styling request normalisation         | In-house                                  |
| Contextual analysis                   | In-house                                  |
| Wardrobe behaviour analysis           | In-house                                  |
| Attribute candidate filtering         | Relational DB                             |
| Semantic / visual candidate retrieval | MODA Retrieval / Vector Store             |
| Candidate compatibility analysis      | In-house                                  |
| Outfit scoring & construction         | In-house — Weighted Scorer                |
| Outfit image generation               | Existing pipeline                         |
| Visual Gate                           | VLM — 0–10 score                          |
| Semantic Gate                         | VLM / LLM — Decision Validator + Feedback |

---

# 24. Styling Request Normalisation

The engine first converts the user's request into structured constraints.

Example:

> “Give me something casual for a date tonight.”

becomes conceptually:

```text
setting = DATE
formality = CASUAL
time = EVENING
```

Other possible constraints:

```text
weather
colour preference
garment preference
garment exclusion
occasion
formality
novelty
specific garment
```

Natural language should not directly drive database queries or scoring logic without normalization.

---

# 25. Contextual Analysis

The styling engine considers contextual information such as:

```text
weather
temperature
setting
occasion
time
formality
user constraints
```

Context should be converted into structured scoring/filtering signals.

---

# 26. Wardrobe Behaviour Analysis

The engine must account for wardrobe usage behaviour.

Required signals include:

```text
wear_count
last_worn
mostly_worn_with
```

Additional contextual history may be maintained.

### Wear count

Can provide:

* repetition signals;
* novelty penalties;
* frequently used garment identification.

### Last worn

Can provide a recency penalty.

### Mostly worn with

Forms a garment-pairing history graph.

Example:

```text
SHIRT A
   │
   ├── JEANS B   × 12
   ├── TROUSERS C × 4
   └── SHORTS D   × 1
```

The system can therefore identify familiar combinations.

---

# 27. Familiarity vs Exploration

The styling engine should not permanently optimize for historical behaviour.

It should support two modes:

### Familiarity

Prefer combinations the user frequently wears.

### Exploration

Use historical behaviour as context while intentionally introducing less-used combinations.

This can be implemented as a configurable scoring weight rather than separate styling systems.

---

# 28. Candidate Filtering

Candidate generation begins with deterministic constraints.

Examples:

```text
category
weather
setting
formality
availability
explicit exclusions
structural compatibility
```

The purpose of this stage is to eliminate obviously invalid candidates before expensive semantic/visual reasoning.

---

# 29. Semantic / Visual Candidate Retrieval

After structured filtering, the system retrieves candidate garments from the vector store / MODA retrieval system.

Retrieval is used to expand the candidate pool.

For example:

```text
User request
     ↓
Structured filters
     ↓
Candidate wardrobe
     ↓
Vector retrieval
     ↓
Visually / semantically relevant candidates
```

Retrieval should not override hard constraints.

---

# 30. Candidate Compatibility Analysis

Each candidate combination is evaluated against:

```text
Pairing
Layering
Structural compatibility
Visual compatibility
```

The result should be structured.

Conceptually:

```text
{
    pairing: ...,
    layering: ...,
    structural: ...,
    visual: ...
}
```

Structural failures cause rejection.

Visual incompatibilities contribute penalties.

---

# 31. Outfit Scoring and Construction

The actual styling decision happens here.

The engine should compute a weighted score incorporating signals such as:

```text
context_score
attribute_score
pairing_score
layering_score
visual_score
wear_novelty_score
recency_score
preference_score
retrieval_score
```

Conceptually:

```text
Final Score =
    Context
  + Compatibility
  + Preference
  + Retrieval Relevance
  + Behaviour
  - Penalties
```

The precise numerical weights are implementation parameters, but the architecture must keep these signals independently observable and adjustable.

---

# 32. Outfit Structure Generation

There should **not** be a mandatory standalone “Outfit Structure Generation” stage.

Outfit structure is part of:

* candidate compatibility;
* layer ordering;
* outfit construction;
* scoring.

A separate stage should only be introduced if implementation evidence demonstrates that structure generation needs an independent lifecycle or model.

---

# 33. Outfit Image Generation

Once the styling engine has selected and constructed an outfit, the existing image-generation pipeline renders the proposed outfit.

This stage is downstream of styling decisions.

The generated image is subsequently evaluated by the validation gates.

---

# 34. Visual Gate

The visual gate evaluates the **actual generated outfit image**.

### Output

```text
score: 0–10
feedback: structured
```

### Evaluation areas

At minimum:

* proportions;
* silhouette;
* colour harmony;
* layering;
* garment interaction;
* visual coherence;
* overall aesthetic quality.

The visual gate is a **quality score**, not the primary styling engine.

---

# 35. Semantic Gate

The semantic gate validates whether the generated result satisfies the original request.

### Inputs

```text
original request
context
selected garments
styling decision
generated image
```

### Outputs

```text
pass / fail
violations
feedback
```

The semantic gate acts as a **decision validator + feedback generator**.

It should answer questions such as:

* Did the outfit satisfy the requested occasion?
* Did it respect explicit constraints?
* Did the selected garments match the intended style?
* Did generation accidentally alter the intended garment configuration?

---

# 36. Final Validation Loop

The final pipeline is:

```text
Candidate retrieval
       ↓
Compatibility
       ↓
Scoring / construction
       ↓
Image generation
       ↓
┌───────────────────────┐
│                       │
│     Visual Gate       │
│     Semantic Gate     │
│                       │
└───────────┬───────────┘
            ↓
        Aggregation
            │
       ┌────┴─────┐
       │          │
      PASS     FEEDBACK
       │          │
       ▼          ▼
    Return     Re-score /
    outfit     regenerate
```

The visual and semantic gates may execute in parallel because they evaluate different dimensions of the generated result.

---

# 37. Failure Handling

Every ingestion stage must have explicit failure behaviour.

| Failure type                   | Expected behaviour                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------- |
| Invalid image                  | Reject ingestion with explicit reason.                                                            |
| Classifier failure             | Retry or mark `UNKNOWN`; do not force classification.                                             |
| Face/crop failure              | Retry where possible; otherwise flag image for review.                                            |
| Attribute extraction failure   | Retry; garment must not be marked fully ingested without required attributes.                     |
| Digitisation failure           | Retry; retain source image for diagnostics.                                                       |
| Taxonomy mapping failure       | Map to appropriate `*_OTHER` or `OTHER`; never silently discard.                                  |
| Compatibility analysis failure | Do not assume compatibility.                                                                      |
| Embedding failure              | Garment may remain structured but should be marked unavailable for vector retrieval.              |
| Variety generation failure     | Do not block canonical garment ingestion unless varieties are mandatory for a downstream feature. |

Failures should be observable and attributable to the affected garment/job.

---

# 38. Option A — MODA/Hopit

The MODA-based implementation uses:

| Capability               | Technology                   |
| ------------------------ | ---------------------------- |
| Image classification     | In-house MobileNetV3         |
| Face/crop                | In-house RetinaFace + SAM    |
| Attribute extraction     | MODA_NER                     |
| Digitisation             | FLUX.2                       |
| Category bundling        | In-house lookup table        |
| Layering                 | In-house decision tree       |
| Structural compatibility | In-house decision tree       |
| Visual compatibility     | In-house decision tree + VLM |
| Embedding                | MODA SigLIP Distilled        |
| Retrieval                | MODA / vector infrastructure |
| Styling decision         | ClosetTheory                 |
| Validation               | VLM / LLM                    |

### Important boundary

MODA is a perception/retrieval dependency.

It does not own:

* ClosetTheory's taxonomy;
* compatibility rules;
* behaviour;
* context;
* scoring;
* outfit construction;
* final validation loop.

---

# 39. MODA Capabilities Not Assumed as Dependencies

The architecture must **not depend on capabilities that are not stable production dependencies**.

In particular:

**Shop-the-look is not a production dependency.**

Similarly, compatibility/scoring capabilities must not be assumed to exist in MODA unless explicitly confirmed and integrated.

ClosetTheory's own compatibility and scoring engine remains authoritative.

---

# 40. Option B — Non-MODA Architecture

The non-MODA implementation preserves the same architecture while replacing MODA-specific components.

| Stage                    | Option B                                 |
| ------------------------ | ---------------------------------------- |
| Image Classifier         | MobileNetV3                              |
| Facial detection/crop    | RetinaFace + SAM                         |
| Image → Attributes       | VLM / fashion attribute model            |
| Image Digitisation       | FLUX.2                                   |
| Category Bundling        | Lookup Table                             |
| Layering Compatibility   | Decision Tree                            |
| Structural Compatibility | Decision Tree                            |
| Visual Compatibility     | Decision Tree + VLM                      |
| Image Embedding          | High-capacity multimodal embedding model |
| Retrieval                | Vector Store                             |
| Styling Decision         | ClosetTheory                             |
| Validation               | VLM / LLM                                |

A strong initial candidate for image-to-attribute and visual reasoning is **Qwen2.5-VL-7B-class**, but it must be treated as a baseline candidate rather than a claim of objectively superior fashion performance.

Model selection should ultimately be benchmark-driven against ClosetTheory's own garment dataset.

---

# 41. Embedding Model Requirements

The embedding system must provide:

* stable image embeddings;
* sufficient semantic/visual retrieval quality;
* efficient inference;
* predictable dimensionality;
* support for vector indexing;
* a licence compatible with the intended deployment.

If changing embedding dimensionality, the migration must account for:

* re-embedding existing garments;
* vector index rebuild;
* storage increase;
* retrieval benchmark comparison;
* deployment/inference cost.

Dimensionality reduction such as PCA must not be introduced merely to avoid migration unless retrieval quality has been benchmarked.

---

# 42. Decision Ownership Matrix

| Decision                 | Owner                              |
| ------------------------ | ---------------------------------- |
| Image representation     | In-house                           |
| Garment taxonomy         | **ClosetTheory**                   |
| Category bundling        | **ClosetTheory**                   |
| Attribute extraction     | MODA / selected model              |
| Image digitisation       | Selected image-generation pipeline |
| Pairing rules            | **ClosetTheory**                   |
| Layering rules           | **ClosetTheory**                   |
| Structural compatibility | **ClosetTheory**                   |
| Visual compatibility     | **ClosetTheory + VLM**             |
| Embedding representation | Selected embedding model           |
| Candidate retrieval      | Vector / MODA                      |
| Context interpretation   | **ClosetTheory**                   |
| Behaviour analysis       | **ClosetTheory**                   |
| Outfit scoring           | **ClosetTheory**                   |
| Outfit construction      | **ClosetTheory**                   |
| Image generation         | Existing pipeline                  |
| Visual validation        | VLM                                |
| Semantic validation      | VLM / LLM                          |
| Retry logic              | **ClosetTheory**                   |

---

# 43. Source of Truth

The system must maintain clear ownership of information.

| Information             | Source of truth                 |
| ----------------------- | ------------------------------- |
| Garment identity        | Relational DB                   |
| User ownership          | Relational DB                   |
| Fine-grained class      | Relational DB                   |
| Category bundle         | Relational DB                   |
| Structured attributes   | Relational DB                   |
| Wear history            | Relational DB                   |
| Pairing history         | Relational DB                   |
| Compatibility rules     | Application logic / rule system |
| Embedding               | Vector store                    |
| Retrieval similarity    | Vector system                   |
| Final styling decision  | Styling engine                  |
| Generated image quality | Visual gate                     |
| Request satisfaction    | Semantic gate                   |

---

# 44. Non-Negotiable Architectural Distinctions

The following concepts must not be collapsed into one another:

### Upload vs ingestion

```text
Upload = acquire source
Ingestion = process source
```

### Classification vs taxonomy

```text
Image representation = CATALOG / CROP / FULL_BODY
Garment classification = T_SHIRT / SHIRT / JEANS / etc.
```

### Fine-grained class vs category bundle

```text
T_SHIRT → TOP
JEANS → BOTTOM
```

Both must be retained.

### Pairing vs layering

```text
Pairing = can these garments form an outfit?
Layering = can these garments occupy these layer positions?
```

### Structural compatibility vs visual compatibility

```text
Structural = physically/logically valid
Visual = aesthetically coherent
```

### Retrieval vs styling

```text
Retrieval = find candidates
Styling = decide what to use
```

### Visual gate vs styling decision

```text
Styling engine = chooses
Visual gate = evaluates the generated result
```

### Semantic gate vs styling decision

```text
Styling engine = constructs the solution
Semantic gate = validates that the solution satisfies intent
```

---

# 45. Implementation Requirements

An implementation is considered aligned with this specification when:

* source images are correctly classified;
* garment representations are distinguished from garment taxonomy;
* fine-grained garment classes are preserved;
* classes resolve to canonical category bundles;
* attributes are normalized into the ClosetTheory schema;
* digitisation is separate from embedding generation;
* pairing and layering are separately represented;
* structural incompatibilities can produce hard rejection;
* visual dissociation produces a soft penalty;
* retrieval is separated from final styling decisions;
* behavioural/contextual signals participate in scoring;
* outfit construction is owned by ClosetTheory;
* generated outfits pass through visual and semantic validation;
* validation feedback can trigger a retry;
* external model capabilities are not treated as application-owned logic;
* failures are explicit and observable.

---

# 46. Implementation Review Protocol

The implementation agent should inspect the current implementation against this document.

For **every relevant decision**, classify the implementation as:

```text
COMPLIANT
PARTIALLY COMPLIANT
MISSING
CONFLICTING
AMBIGUOUS
```

### Important

The agent should **not modify this specification to match the implementation**.

If the implementation currently contains:

```text
different garment classes
different category bundles
different compatibility rules
different embedding dimensions
different scoring behaviour
```

the agent must flag the discrepancy and modify the implementation where required.

---

# 47. Canonical Decision Summary

| Area                        | Canonical Decision                         |
| --------------------------- | ------------------------------------------ |
| Image classes               | `CATALOG`, `CROP`, `FULL_BODY`, `UNKNOWN`  |
| Image classifier            | MobileNetV3                                |
| Face/crop                   | RetinaFace + SAM                           |
| Attribute extraction        | MODA_NER in Option A                       |
| Digitisation                | FLUX.2                                     |
| Fine-grained taxonomy       | Controlled ClosetTheory taxonomy           |
| Category bundling           | Lookup table                               |
| Fine-grained class retained | **Yes**                                    |
| Pairing model               | Explicit                                   |
| Layering model              | Directional                                |
| Structural compatibility    | Hard constraint                            |
| Visual dissociation         | Soft penalty                               |
| Embedding                   | MODA SigLIP Distilled in Option A          |
| Vector store                | Candidate retrieval                        |
| Relational DB               | Structured source of truth                 |
| Behaviour signals           | Wear count, last worn, mostly worn with    |
| Styling decision            | ClosetTheory weighted scorer               |
| Outfit structure            | Part of construction, not standalone stage |
| Image generation            | Existing pipeline                          |
| Visual gate                 | VLM, 0–10                                  |
| Semantic gate               | VLM/LLM validator + feedback               |
| Retry loop                  | Required                                   |
| Shop-the-look               | **Not a production dependency**            |
| Final styling ownership     | **ClosetTheory**                           |

---

# 48. Final Architectural Model

```text
                         GARMENT INGESTION
                                │
                                ▼
                     Image Representation
                                │
                                ▼
                     Garment Perception
                                │
                                ▼
                     Attribute Extraction
                                │
                                ▼
                       Image Digitisation
                                │
                                ▼
                     Canonical Taxonomy
                                │
                                ▼
                     Category Bundling
                                │
                    ┌───────────┴───────────┐
                    │                       │
             Compatibility             Embedding
                Analysis                  │
                    │                     ▼
                    │               Vector Store
                    │
                    └───────────┬─────────┘
                                │
                                ▼
                         GARMENT RECORD
                                │
                                │
                         ───────┴───────
                                │
                                ▼
                        STYLING REQUEST
                                │
                                ▼
                    Request Normalisation
                                │
                                ▼
                    Context + Behaviour
                                │
                                ▼
                     Structured Filtering
                                │
                                ▼
                    Semantic/Visual Retrieval
                                │
                                ▼
                     Candidate Compatibility
                                │
                                ▼
                     Scoring + Construction
                                │
                                ▼
                       Image Generation
                                │
                         ┌──────┴──────┐
                         │             │
                         ▼             ▼
                    Visual Gate   Semantic Gate
                         │             │
                         └──────┬──────┘
                                ▼
                           Aggregation
                                │
                    ┌───────────┴───────────┐
                    │                       │
                   PASS                  FEEDBACK
                    │                       │
                    ▼                       ▼
                  OUTPUT             RESCORE / RETRY
```

## Governing principle

> **Models perceive. Retrieval discovers. Rules constrain. Scoring decides. Generation renders. Gates validate.**

This is the canonical architecture the implementation should conform to.
