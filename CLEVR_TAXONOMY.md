# CLEVR Experiment Taxonomy

## 1. CLEVR Answer Space (28 classes)

| Category | Values                                                 | Count |
|----------|--------------------------------------------------------|-------|
| boolean  | yes, no                                                | 2     |
| counting | 0–10                                                   | 11    |
| color    | gray, red, blue, green, brown, purple, cyan, yellow    | 8     |
| shape    | cube, sphere, cylinder                                 | 3     |
| material | metal, rubber                                          | 2     |
| size     | large, small                                           | 2     |

## 2. CLEVR Question Types (program-based)

| Question Type     | Program final function                                 | Answer category |
|-------------------|--------------------------------------------------------------|-----------|
| query_color       | `query_color`                                                | color     |
| query_shape       | `query_shape`                                                | shape     |
| query_material    | `query_material`                                             | material  |
| query_size        | `query_size`                                                 | size      |
|-------------------|--------------------------------------------------------------|-----------|
| exist             | `exist`                                                      | boolean   |
|-------------------|--------------------------------------------------------------|-----------|
| counting          | `count`                                                      | counting  |
|-------------------|--------------------------------------------------------------|-----------|
| compare_integer   | `equal_integer`, `less_than`, `greater_than`                 | boolean   |
| compare_attribute | `equal_color`, `equal_shape`, `equal_material`, `equal_size` | boolean   |

## 3. Classification Schemes Used in Experiments

### A. Answer Type (what the answer is)

Used when we decode/predict from representation and check what TYPE the output is.

Categories:
- boolean
- counting
- color
- shape
- material
- size

| Scripts                      | Purpose                                       |
|------------------------------|-----------------------------------------------|
| `run_layer_decode_by_type.py`| Layer-wise decoding accuracy per answer type  |
| `run_layer_decode_dist.py`   | Prediction type distribution at each layer    |

### B. Text Corruption Type (what text element is swapped)

Used when we do activation patching with corrupted questions.

Categories:
- attribute (color/material/size swap)
- shape (shape swap)
- spatial (relation swap)
- query attribute (What color/What material/What size/What shape swap)
- quantifier (structure swap)

| Scripts                             | Purpose                                     |
|-------------------------------------|---------------------------------------------|
| `clevr_corruptions.py`              | Corruption generation                       |
| `run_headwise_by_type.py`           | Per-head patching per corruption type       |
| `run_component_patching_by_type.py` | SA/GCA decomposition per corruption type    |
| `run_all_patching.py`               | 4-way patching per corruption type          |

Note: "attribute" corruption lumps color + material + size into one category.

### C. Question Type (what the question ASKS)

Used when we classify by the question's intent, not the answer's category.

Categories: 
- query_attr
    - query_color
    - query_material
    - query_size
    - query_shape
- exist
- compare
    - compare_integer
    - compare_attribute
- counting


| Scripts                | Purpose                          |
|------------------------|----------------------------------|
| `run_retrieval_viz.py` | Steered retrieval visualisation  |

### D. Object Attribute (scene graph properties)

Used when we compare object-level representations.

**Categories: color, shape, material, size**

| Scripts                  | Purpose                      |
|--------------------------|------------------------------|
| `run_rsa.py`             | RSA with ground-truth RDMs   |
| `representation_probing.py` | Object-level probing      |

## 4. Retrieval Condition Hierarchy

Used in steered retrieval and UMAP experiments. For a given query (image, question, answer), each DB image is labeled by the highest condition it satisfies. Later conditions imply earlier ones (hierarchy). Color assignment: later overrides earlier.

Source: `clevr_condition_checker.py`

### Condition definitions

- **C1 — Feature extraction** (green): Each described attribute value exists **independently** somewhere in the DB scene (⋁ attr)
- **C2 — Feature binding** (blue): All described attributes **co-occur on a single object** in the DB scene (⋀ attr)
- **C3 — Object indexing** (orange): An object with **all 4 attributes** (color, shape, material, size) matching the query object exists in the DB scene
- **C4 — Position indexing** (purple): A DB object matches all 4 attributes **and** overlaps the pixel position of the query object (C3 ∧ dist < rad_db + rad_q)
- **C5 — Answer match** (red): Executing the CLEVR program on the DB scene produces the **same answer** as ground truth

For `same` and `spatial` queries, C2/C3/C4 split into anchor and target variants:
- Target: the object being queried
- Anchor: the object used to refer to the target (via `same_*` or `relate`)
- **C2-1 / C2-2** — Anchor / Target feature binding
- **C3-1 / C3-2** — Anchor / Target object indexing
- **C4-1 / C4-2** — Anchor / Target position indexing

### Active conditions per question type

Direct (single object):
- `attr_query_direct`: C1, C2, C3, C4, C5
- `exist_direct`: C1, C2, C5
- `count_direct`: C1, C2, C5
- `comparison_direct`: C1, C2, C3, C4, C5
- `int_comparison_direct`: C1, C2, C5

Same-attribute (anchor + target):
- `attr_query_same`: C1, C2-1, C2-2, C3-1, C3-2, C4-1, C4-2, C5
- `exist_same`: C1, C2-1, C2-2, C3-1, C3-2, C4-1, C4-2, C5
- `count_same`: C1, C2-1, C2-2, C3-1, C3-2, C4-1, C4-2, C5

Spatial (anchor + target via `relate`):
- `attr_query_spatial`: C1, C2-1, C2-2, C3-1, C3-2, C4-1, C4-2, C5
- `exist_spatial`: C1, C2-1, C2-2, C3-1, C3-2, C4-1, C4-2, C5
- `count_spatial`: C1, C2-1, C2-2, C3-1, C3-2, C4-1, C4-2, C5
- `comparison_spatial`: C1, C2-1, C2-2, C3-1, C3-2, C4-1, C4-2, C5
- `int_comparison_spatial`: C1, C2-1, C2-2, C5

## 5. Resolved Issues

- [x] Answer type missing "size" → added to `classify_answer()`, `build_answer_type_lookup()`, `ANSWER_TYPE_COLORS`
- [x] Shape corruption fixed cycle → now random selection from alternatives
- [x] "counting" vs "count" → standardised to "counting" everywhere
- [ ] RSA missing size RDM (rerun needed)
- [ ] Boolean questions not classified as a question type in retrieval
