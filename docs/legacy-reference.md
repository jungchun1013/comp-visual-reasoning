# Legacy reference — everything you need WITHOUT opening SteerViT-legacy/

`SteerViT-legacy/` is a huge, read-only tree. This doc distills the four things main/
still inherits from it: naming conventions, the headwise-patching methodology, the plot
style, and the legacy checkpoint format. Open the legacy tree only for Blender tooling
(`SteerViT-legacy/tools/clevr-dataset-gen/`, wired via the two render scripts).

Sources verified 2026-07-05 against the legacy files cited inline.

## 1. Naming

### 1.1 Paper v2 terminology (authoritative for all NEW artifacts)

| old (workshop paper, legacy figures) | v2 (use everywhere now) |
|---|---|
| 3-stage: binding → object grounding → answer matching | **2-stage: Binding → Retrieval** |
| "object grounding" (middle stage) | deprecated — **Grounding** now names the WHOLE language-conditioning mechanism (CA routing + refocus) |
| "answer matching" | **Retrieval** (old name was post-hoc, coined after the clustering was seen) |

Existing output files/dirs are NEVER renamed; only new code, docs, figures, and prose
use v2 names.

### 1.2 Main-repo run-name grammar

`clevr_<backbone>_<variant>_scratch_s<seed>` with backbone ∈ {dinov2, siglip, sup, mae}
and variant ∈ {decoder1l (GCA-decoder `VQADecoder`), concat_decoder1l
(`ConcatSelfAttnDecoder` — the paper's performance-table model), cls, nogate, nogca,
film, mean, gca (scratch-ViT), learned_text_decoder1l}. See
`docs/paper_artifacts.md` §1 for the paper-term ↔ config map.

### 1.3 Legacy run-name grammar (`exp_vqa/outputs/phase1_clevr/*`)

`{gca_layers}_{init}_{head}_{variant}`, optional dataset prefix (`cogent_`, `gqa_`):

- gca_layers: `odd` = GCA at blocks [1,3,5,7,9,11] · `all` = every block · `last6` = [6..11]
- init: `scratch` = GCA/connector randomly initialized · `finetune`/`pretrained` = warm-started from a prior ckpt's `steervit_trainable_state` (`train_clevr_decoder.py:217-224`)
- head: `decoder` (CoCa-style autoregressive `VQADecoder`) · `cls` (classifier `VQAHead`)
- variant: decoder `1l`/`2l` = num decoder layers; cls `cls`/`mean` = CLS-token vs mean pooling; `large` = ViT-L/14 DINOv2
- Example: `odd_scratch_decoder_1l` = the legacy ancestor of main's `clevr_dinov2_decoder1l_scratch`.

### 1.4 Retrieval categories (all analysis experiments)

3 fixed attr_query categories, family IDs (main: `src/data/clevr_sampling.py`
`RETRIEVAL_CATEGORIES`; legacy: `exp_vqa/analysis/utils/retrieval_sampling.py`):

```
attr_query_direct:  [86, 87, 88, 89]
attr_query_same:    [53, 59, 55, 57, 61, 60]
attr_query_spatial: [76, 74, 75, 77, 80, 81]
```

Sampling: n_total split evenly across families, remainder to the first families.

## 2. Corruption taxonomy (perturbations A/B and their fine types)

Canonical definition: legacy `exp_vqa/analysis/utils/clevr_corruptions.py:1-8,124-148`.
Every corruption dict has `type`, `fine_type`, `original_word`, `corrupted_word`,
`corrupted_question`.

- **Coarse `type`**: `attribute` (A), `spatial`, `attribute_query` (B), `quantifier`
- **Fine `fine_type`**: `color`, `material`, `size`, `shape` (under attribute);
  `what_color`, `what_material`, `what_size`, `what_shape` (under attribute_query);
  `spatial`, `quantifier` (their own coarse+fine)
- Word pools: colors red/green/blue/gray/brown/purple/cyan/yellow; materials
  rubber/metal/shiny/matte; sizes large/small/big/tiny; shapes cube/sphere/cylinder/block/ball;
  spatial swaps left↔right, behind↔in front of; quantifier "how many"/"what number of" ↔
  "is there a"/"are there any".
- **Targeted-corruption mode** (`run_component_patching_by_category.py:91-150`):
  program+scene-aware — locates the anchor/target object via the CLEVR program
  (`find_anchor`/`find_target`) and corrupts only that object's attribute word
  (types `anchor_attribute`/`target_attribute`). Used for the retrieval-aligned
  groups `direct`/`same`/`spatial` (family IDs as §1.4).
- Perturbation **C** (image-side) is main-repo: Blender re-renders with the queried
  attribute swapped (`scripts/analysis/render_visual_corruptions.py`, pairs.json).

## 3. Headwise patching methodology (the reference implementation)

Reference: `SteerViT-legacy/exp_vqa/analysis/dinov2_gca/patching/run_headwise_by_type.py`
(main-repo port: `scripts/analysis/activation_patching.py`, backbone-generic).

- **Per-head patching**: SA — forward-pre-hook on `blocks[layer].attn.proj`, replacing
  that head's slice of the pre-projection input; GCA — same on
  `gated_cross_attn.cross_attn.to_out` (`patching_utils.py:100-188`).
- **Directions** (same op, roles swapped; metric = patched_logit − corrupt_logit of the
  correct answer):
  - `denoising`: patch CLEAN-question activations into the CORRUPTED-question run —
    positive Δ = restoring this head recovers the answer.
  - `noising`: patch CORRUPTED activations into the CLEAN run — negative Δ = injecting
    the wrong activation degrades the answer.
- **Groups**: default 12 categories = coarse {attribute, spatial, attribute_query,
  quantifier} + fine_attribute {color,material,size,shape} + fine_attribute_query
  {what_*}; `--retrieval-categories` switches to the targeted direct/same/spatial groups.
- **CLI** (defaults): `--num-samples 50`, `--groups all`, `--directions denoising,noising`,
  `--replot` (plots only, from saved stats), `--compute-only`.
- **`headwise_by_type_stats.json` schema**: top-level `gca_layers` [1,3,5,7,9,11],
  `sa_num_heads` 12, `gca_num_heads` 16, `num_samples_per_category`; then one key per
  `{group}_{direction}` mapping category → record with `sa_mean` (12×12), `gca_mean`
  (6×16), `sa_std`, `gca_std`, `n`. PNGs `headwise_{group}_{direction}.png` draw GCA
  columns in red dashed outlines.
- Known binding heads (dinov2 GCA-decoder): CA L5H0 (color), L7H9 (material),
  L7H11 (size), L7H3 (shape); late SA block 11 = Retrieval-side integration.

## 4. Plot style

Single source of truth: **`src/analysis/plot_style.py`** (PLOT_STYLE rcParams dict,
tab10/tab20c palettes, CORRUPTION_COLORS). Import it; never copy style blocks inline.
(L2 consolidation removes the historical inline copies.)

## 5. Legacy checkpoint format

Written by `exp_vqa/train_clevr_decoder.py:356-392` (same in train_cogent_decoder /
train_gqa*):

```
epoch                     int
decoder_state_dict        decoder weights   (cls variant: vqa_head_state_dict instead)
optimizer_state_dict / scheduler_state_dict
best_acc                  float
steervit_trainable_state  only gated_cross_attn* + connector* keys (present iff unfrozen)
val_acc                   float (added post-eval)
```

**No `config` key** — loaders must fall back to defaults
(`vit_base_patch14_dinov2.lvd142m`, d_model 512, nhead 8, max_len 8, num_layers
inferred from state-dict keys), as legacy `analysis/utils/load_model.py:45-101` does.
Main-repo detection idiom: `"config" not in ckpt` ⇒ legacy format. (L3 consolidates the
five duplicated loaders into `src/model/checkpoint_io.py`.)

Main-repo checkpoints additionally store `config` and `val_acc`; the stored `val_acc`
is authoritative when logs conflict (see `docs/paper_artifacts.md` §8.1).

## 6. The only remaining legacy runtime dependency

Blender scene tooling: `scripts/analysis/render_visual_corruptions.py` and
`render_single_objects.py` sys.path into `SteerViT-legacy/tools/clevr-dataset-gen/`
(override with `BLENDER_TOOLS_ROOT` once L4 lands). Everything else in this doc means
you should not need to open the legacy tree.
