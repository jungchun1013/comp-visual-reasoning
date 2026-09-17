# Setup manifest — scene-level organization and object-level attribute alignment

Written by Claude, 2026-09-16, in answer to the Codex plan "給 Claude:統一場景幾何與物件屬性分析"
(§2 setup 對帳, §3 條件命名, §4 補算). Every row was checked against the cache files, the
`tee_stdout` log headers and the script source on this date; line numbers refer to the
worktree copies unless marked MAIN (main checkout, script not on this branch).

## 1. Condition names

Object IDs are fixed per image: **A** = the `target` slot of `labels.json`, **B** = the
`distractors[0]` slot. Referent / non-referent are roles that change with the question. The
programme condition codes stay in the result files; this table is the mapping.

| manuscript name | programme code | question string (colour run) | how language is disabled |
|---|---|---|---|
| No question | `c0` / `noca` | none | `questions=None` → no text encoder call; the gated cross-attention receives no keys (`patch_language_condition.py:321`, `tsne_single_object.py:166`) |
| Generic attribute question | `c3` / `ca_color_object` | `What color is the object?` | — |
| Question about A | `c1` / `ca_color_refer` | `minimal_referring_question(A)`: `What color is the {A.shape}?` or `What color is the {A.size/material} object?` | — |
| Question about B | `c2` (object-level only) | same rule with the roles swapped | — |

Referring word priority: shape > size > material > colour, excluding the queried attribute,
first attribute whose value differs between A and B (`tsne_single_object.py:140-153`). The
scene-level caches have no "Question about B" condition.

## 2. Artifacts

| analysis | data (images, objects, filter) | ids / pairing | model, readout, seed | questions | feature location | files / script | comparable with the object-level analysis? |
|---|---|---|---|---|---|---|---|
| **Object-level attribute alignment (X21, colour)** | `data/clevr_object_count/{n1,n2}`, **324 of 480 pairs**: `seg_ok` = A not gray, B not gray and B.colour ≠ A.colour (`patch_language_condition.py:85-87`); no segmentation losses (`log.txt:3,27`) | `pair_index` 60–479 = index into the 480 paired renders, identical list in n1/n2 | `clevr_dinov2_decoder1l_scratch_s42/best.pt`, local-patch decoder, seed 42, grid 24 @ 336 | c0–c3 as above (`build_questions` :104-121) | every block output → trunk LayerNorm → mean over each object's own patch tokens (`obj_mean`), 64 fixed background patches (`bg_mean`); raw pre-norm copies kept; projections on unit-normalised means (`_normstd`) or raw | `outputs/analysis/patch_language_condition/n{1,2}/feats_c*.npz`, `attr_directions_v2/`, **new** `unified_role_contrasts/` | reference |
| same, other backbones | same 324 pairs | same | `clevr_siglip_decoder1l_scratch_s42` (grid 16 @ 256), `clevr_sup_decoder1l_scratch_s42` (grid 24 @ 384), `clevr_mae_decoder1l_scratch_s42` | identical strings | same | `.../{siglip,sup,mae}/` | yes (same pairs, same questions) |
| same, shape / material queried | same 324 pairs | same | DINOv2 | `What shape is the {word} object?` — the referring word changes in 223 / 324 pairs (colour run uses shape words; shape run uses size, material, colour words); material run changes it in 42 | same | `.../shape/`, `.../material/` | only on the description-fixed subset: colour↔shape 101, colour↔material 282, shape↔material 59 pairs |
| **Scene-level t-SNE / trained pooled probe** (`object_count_v2`) | same paired set, **all 480**, no filter (`object_count_v2/n1/log.txt:1,5`) | file order = `pair_index` (attrs.json byte-equal to `metadata.json`) | same DINOv2 checkpoint, seed 42; probe = StratifiedKFold 5, fold-local PCA(50), logistic (`linear_probe_single.py:99-114`) | `noca`, `ca_color_object`, `ca_shape_object`, n2 also `ca_color_refer`, `ca_shape_refer` (same generator as c1) | every block output → trunk LayerNorm → mean over all 576 patch tokens, no unit-normalisation (`tsne_single_object.py:62-64,92-97`) | `outputs/analysis/tsne/object_count_v2/n{1,2}/feats_*.npz` (`"0"…"11"`, (480,768)), `linear_probe_results.json` | yes for No question / Generic / Question about A on the 324-pair subset (rows 60–479 filtered by `seg_ok`); no Question-about-B |
| **Variance partitioning** | same caches (no forward pass), 480 images | as above | same; seed 42 | same five + four raw backbones from `raw_backbone_probe/pooled_n1n2_v2` | dependent variable = the pooled block feature matrix; unique share = R²(full) − R²(without the factor), drop-one one-hot OLS with A and B attributes as factors (MAIN `variance_partitioning.py:74-100,126-145`) | `outputs/analysis/variance_partitioning/results.json`, `results_own_axis.json` | same as the row above; descriptive within-sample share, not a decomposition |
| `object_count_v3` t-SNE | replot of `object_count_v2` features (style only) | — | — | — | — | `outputs/analysis/tsne/object_count_v3/` | not a separate experiment |
| **Referent-role probe (X17, 219 scenes)** | `data/clevr_two_object_v2` (480; target always large), eligible when shapes differ, colours differ, no gray → 221, 2 empty masks → **219** (MAIN `reference_probe.py:48-57`, log) | scene-grouped split | same DINOv2 checkpoint, seed 42; logistic, one GroupShuffleSplit 80/20 | `What color is the {shape}?` both directions; `noca`; description; irrelevant | block → LayerNorm → mean over each object's patches (unpooled cache (12,219,2,2,768)) | `outputs/analysis/reference_probe/two_object_paired/` | **no**: different stimulus set (superseded), different filter; keep as a separate design, do not mix the 219 with the 324 |
| **Workshop steered t-SNE / conditional RSA** | CLEVR v1.0 val; reference pool = first 500 unique val images with 3–5 objects in dataset order, query image excluded (MAIN `tsne_viz.py:331-350`, `conditional_rsa.py:200-218`) | — | same DINOv2 checkpoint; t-SNE cosine, perplexity 30, seed 42; RSA 72 queries / category | curated `attr_direct_queries.json` (q0 `What shape is the large cyan object?`), RSA: sampled from the 3 attr_query families | block → LayerNorm → mean over all patches | `outputs/analysis/tsne/clevr_dinov2_decoder1l_scratch/attr_direct/cache_q*.npz`, `outputs/analysis/conditional_rsa/` | **no**: synthetic CLEVR scenes with 3–5 objects, no object masks, no paired conditions; complementary evidence |
| old `outputs/analysis/tsne/object_count/` | **v1 sets**: n1 `data/clevr_single_object` (500, 320×240), n2 `data/clevr_two_object` (480, 480×320) — verified by byte-comparing `attrs.json` with the dataset files | not paired | same DINOv2 | `noca, ca_object, ca_cube, ca_shape_object, ca_shape_large` | as `object_count_v2` | also `outputs/analysis/linear_probe_v2/object_count/` (fold-local PCA redo of these caches) | **no** (render mismatch between its n1 and n2); historical only |

Correction to the registry (X18 note, "Old object_count runs … built on the defective v3/v2
stimuli"): the July `object_count/` caches were built on the v1 sets, not v3/v2. The v3/v2
sets were used by the X18 raw-backbone probe `pooled_n1n2/` (superseded by `pooled_n1n2_v2`)
and by the X17 referent probe. The old run stays historical either way.

## 3. Independence check (plan §2 特別核查)

The attribute directions are estimated on the **1-object renders** (`n1`, no question) and
evaluated on the **2-object renders** (`n2`) of the **same 324 pairs**: n1 image *i* is the
2-object scene *i* with B removed, target placement identical. So the direction set and the
evaluation set share the object instances (A of every pair), not the images. Every A
contributes to its own value's direction. **Correction (2026-09-16, after Codex review):** the
first version of this section claimed that the shared-instance term "cancels in every
contrast". That has no mathematical guarantee: a contrast is `(unit(x_question) −
unit(x_baseline)) · v`, and a direction `v` that carries information about this image can
be correlated with the difference vector. The size and sign of the overlap effect were not
tested in the first version. The sensitivity check is the pair-grouped cross-fit
(`--crossfit-folds 5`, seed 42, directions from the other folds' 1-object no-question means,
held-out pairs projected on them; `split.json` records the fold of every pair and the
per-fold counts per value). Its results are in the registry (X26, v2); the intervals remain
conditional on the fixed directions and the trained model.

## 4. What was computed from the existing caches (no new extraction)

`patch_language_condition.py --role-contrasts` (new flag, functions `role_contrasts`,
`attribute_switch`): for each object (A, B), each attribute (queried + colour + shape) and
each block, the per-image projection on the object's own attribute direction under the four
conditions, and the five contrasts

| name | contrast |
|---|---|
| `about_it` | question about it − no question |
| `about_other` | question about the other object − no question |
| `generic` | generic question − no question |
| `about_other_vs_generic` | question about the other object − generic question |
| `about_it_vs_generic` | question about it − generic question |

Bootstrap unit = image (2000 resamples, seed 42, percentile 95 % interval); `both` = the
per-image mean of A and B, so the A/B pairing is kept. Raw and unit-normalised
(`_normstd`) variants, as in `attr_directions_v2/`.

Queried-attribute switch (`attribute_switch*.json`): between two runs that ask about
different attributes, only images whose c1 **and** c2 referring words are identical in both
runs are used, so image, referent and description are fixed and only the queried attribute
changes; the projection on the own direction of attribute *d* is compared between the run
that asks about *d* and the run that asks about the other attribute, for the referent, the
non-referent and the generic condition. Because a description that avoids both queried
attributes exists only when the pair shares the remaining attributes, these subsets are
same-shape pairs (colour↔shape, shape↔material) or partly same-material pairs
(colour↔material); a `_differing` stratum keeps only pairs whose value of *d* differs between
A and B, and the shape direction has no such pairs (not estimable).

Outputs: `outputs/analysis/patch_language_condition/{,siglip/,sup/,mae/,shape/}unified_role_contrasts/`
(`role_contrasts{,_normstd}.{json,png}`; DINOv2 colour run also `attribute_switch{,_normstd}.{json,png}`),
logs `log_role_contrasts.txt` in each run directory. Results are recorded in
`docs/experiment_registry.md` (X26).

## 5. Not done / open

- Done 2026-09-16 (v2): scene-level variance share and t-SNE on the 324-pair subset
  (`outputs/analysis/variance_partitioning_324/`, `outputs/analysis/tsne/object_count_v3/n2_324/`);
  pair-grouped cross-fit directions (`unified_role_contrasts_v2/`); see registry X26
  corrections. Variance-share figures were not regenerated for the subset (plot code expects
  the raw-backbone conditions).
- Pooled scene vector and object means from one forward pass: the X21 cache keeps object
  patches + 64 background patches, not all 576, so the exact pooled mean cannot be rebuilt
  from it; the scene-level cache comes from a separate pass with the same checkpoint, the
  same LayerNorm placement and the same question strings for c0 / c1 / c3.
- Equivalence margin for "generic question does not reduce alignment": none was
  pre-specified; estimates and intervals are reported, no "no decrease" claim is made.
