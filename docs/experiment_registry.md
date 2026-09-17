# Experiment registry & design-consistency audit (2026-07-05)

Per experiment: motivation → hypothesis → design → status → artifacts → known
inconsistencies. Then §D: cross-experiment design-consistency findings (D1–D11),
ordered by severity. Status legend: ✅ done · 🔄 running tonight · ⏳ queued ·
❌ blocked/decision needed.

## Part 1 — Experiment registry

### X1. Performance matrix (paper Tables 1/4, R1+E1)
- **Motivation**: does language conditioning elicit compositional VQA from frozen VFMs, and does it depend on pretraining type?
- **Hypothesis**: semantic/discriminative pretraining (DINOv2/SigLIP/sup) provides a usable substrate; pixel reconstruction (MAE) does not (A3.1).
- **Design**: 4 backbones × {concat decoder, cls} (+GCA-decoder as the mechanistic model), 16 epochs, s42, final-epoch full-val acc.
- **Status**: overall accs ✅ (paper_artifacts §9, all Table 1/4 cells matched); per-qtype cells 🔄 (E1b, 13 ckpts).
- **Inconsistency**: D1 (variant split), D2 (learned_text best/last), D8 (seed claim).

### X2. Ablations (−CA / scratch-ViT / learned-text / classifier)
- **Motivation**: all three pieces (pretrained ViT, pretrained text encoder, CA) necessary.
- **Hypothesis**: removing any collapses accuracy toward priors.
- **Status**: ✅ 49.4 / 52.8 / 24.6 / 90.1 all provenance-matched.
- **Current qualification (Codex edit, 2026-09-12):** learned-text has checkpoint-reconstruction and optimizer-initialization defects addressed in `7f2d8e4`; corrected retraining was started as `clevr_dinov2_learned_text_decoder1l_v2_s42`. Historical 24.6/46.7 labels and independent evaluations require provenance reconciliation, not merely a best-versus-last choice. Final corrected validation is pending in the dated record; see JOURNAL's learned-text entry. Historical entries below are retained as history, not validation of the old ablation.

### X3. Gate framing (R2)
- **Motivation**: user-decided demotion — gate = design choice enabling an analyzable mechanism handle, not a performance claim.
- **Hypothesis**: gate provides graded, monotonic control of language influence.
- **Status**: ✅ (RESULTS.md §2: α-interpolation 47/50→37/50 monotone; interventions vs random control; nogate cost 2–5 pts reported honestly). No further nogate work (byproduct policy).

### X4. CoGenT (R3)
- **Motivation**: compositional generalization beyond the training attribute pairing.
- **Hypothesis**: grounding generalizes zero-shot with a small gap that closes with few B samples, without forgetting A.
- **Status**: ✅ RESOLVED (user 2026-07-05): the paper's 92.4/88.0 was a transcription error — camera-ready adopts the reproducible main-repo numbers: zero-shot ValA 94.5 / ValB 89.5, ft(50k,8ep) → ValB 92.7 (ValA 92.4 retained). Artifacts: `cogent_sample_efficiency/sample_efficiency.json` + `50k_8ep.log`.

### X5. Transfer (Humans / Math / CLOSURE, Table 6)
- **Motivation**: does grounding transfer beyond CLEVR's synthetic language?
- **Status**: ✅ all zs/ft numbers exact from `*_ft_all/results.json`.
- **Note**: partial-freeze variants exist for Humans/Math; `clevrmath_ft_connector` died pre-eval (rerun only if the paper needs that cell).

### X6. Headwise patching, text perturbations A/B (mechanistic core)
- **Motivation**: localize where described-attr (A) vs queried-attr (B) information is causally used.
- **Hypothesis**: A concentrates in mid-layer CA (Binding); B touches both stages.
- **Design**: 50 samples/category, denoising, per-head SA(12×12)+CA(6×16), dinov2 GCA-decoder.
- **Status**: ✅ (`activation_patching/clevr_dinov2_decoder1l_scratch/headwise_by_type_stats.json`); SigLIP replication ⏳ (E3, queued).
- **Inconsistency**: D4 (sample population differs from C).

### X7. Visual perturbation C (queried attr swapped in image)
- **Motivation**: the image-side counterpart of B — is repair routed through SA instead of CA?
- **Status**: ✅ stats exist (`visual_*_stats.json`).
- **Inconsistency**: **D3 (render-domain confound: clean = original CLEVR render, corrupt = our Blender re-render)** — headline caveat for E9.

### X8. E9 A/B/C × {CA,SA} contrast
- **Motivation**: v2 §A4 rigor claim.
- **Hypothesis (revised by data)**: gradient, not absolute — CA-share A > B > C.
- **Status**: ✅ first pass (`abc_localization/`; RESULTS.md §6). Figure pending.
- **Inconsistency**: inherits D3 + D4; wording already constrained in paper_v2_outline A4.1.

### X9. Path patching / ACDC / binding interchange / back patch
- **Motivation**: circuit-level account of Binding→Retrieval.
- **Status**: ✅ artifacts exist (dinov2 GCA-decoder only). v2 uses them as A4.2 evidence; no rerun planned.

### X10. Conditional RSA + linear probes (trained models)
- **Motivation**: representational geometry of the 2 stages across the 3 attr_query categories.
- **Design**: 72 queries/category, 500 db, families direct [86,87,88,89] / same [53,59,55,57,61,60] / spatial [76,74,75,77,80,81], seed 42.
- **Status**: ✅ concat main model (`concat_decoder_1l/`), siglip GCA-decoder, dinov2 GCA-decoder. ❌ all `*_nogate_*` probe dirs are INVALID (2026-08-26, X20): `linear_probe.py`'s own loader dropped `use_gate`, so ungated checkpoints were rebuilt with zero-init gates and their GCA output nulled (decode 0.171 = unconditioned DINOv2). Kept on disk, excluded by `probe_table.py`; the variant itself is deprecated (user ruling 2026-08-26) — no rerun.

### X11. E7 add-object hallucination (v2 A1.3 core)
- **Motivation**: show the substrate bottleneck is fixation, not encoding.
- **Hypothesis**: adding a mostly-matching distractor with a bait value on the queried attribute pulls answers toward the bait iff binding is weak.
- **Design**: 100 pairs × 4 attrs; distractor flips exactly one described attr; answer invariance verified by program execution; base re-render controls render domain. Families landed exactly on attr_query_direct [86,87,88,89] ✅ (consistent with X10's "direct" category).
- **Status**: renders ✅; model eval ⏳ (queued, concat main model).

### X12. E8 raw-backbone per-object patch-token probe (v2 A1.2)
- **Motivation**: substrate is compositional per-object BEFORE language conditioning.
- **Design**: fresh zero-gated GCA = pure ViT; 3×3 patch pooling at pixel_coords; 300 scenes; per-block 5-fold logistic; 4 backbones.
- **Status**: ⏳ (chained after main queue).
- **Inconsistency**: D5 (grids differ across backbones — 24×24 vs 14×14; neighborhoods cover different image areas; compare within-backbone block curves, not absolute cross-backbone values).

### X13. E5 failure modes + autonomous diagnosis (v2 A5)
- **Motivation**: why are yes/no worst; counting/CmpInt weak.
- **Design**: per-question dump (stride 4), per-family acc, yes/no confusion, signed count errors; then pre-registered H1–H3 (paper_v2_outline §A5).
- **Status**: ⏳ (queued, concat main + GCA-decoder).

### X14. Baselines → mechanism-transfer baselines (redesigned 2026-07-05, survey-backed)
- MoT ✅ 0.7483 (keep). LLaVA-style: empty, drop. From-scratch flamingo/transfusion
  attempts: superseded by the pretrained plan below.
- **Transfusion: NO public weights anywhere** (Meta paper-only; lucidrains repo
  code-only; no HF checkpoints; no replication releases as of 2026-07) → dropped.
- **I2T (reviewer-requested, priority)**: OpenFlamingo — `openflamingo/OpenFlamingo-9B-vitl-mpt7b`
  (MIT/Apache stack, ~18–20GB bf16; CA+resampler ≈1.3B pretrained on LAION-2B+MMC4;
  gated cross-attn every 4th layer — same lineage as our GCA). Smaller:
  `-4B-vitl-rpj3b`, `-3B-vitl-mpt1b`. Plan: zero-shot mechanism analysis first
  (patching/RSA on its GCA — do binding-head structures exist without CLEVR
  training?), zero/4-shot accuracy second, readout-finetune optional.
  Replication option: `HuggingFaceM4/idefics-9b` (Llama-gated license; IDEFICS2
  does NOT fit — dropped cross-attn for early fusion).
- **T2I**: PixArt-Σ — `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (OpenRAIL++; DiT 0.6B
  + frozen T5-XXL-encoder ≈4.3B). DIFT-style small-t features; (a) per-block probing
  (3 attr_query categories), (b) frozen 1-layer decoder readout (Table-1-protocol
  comparable), (c) cross-attn map localization (zero-shot Binding evidence).
  Not fitting: SD3/FLUX (MM-DiT joint attn, no CA module), Show-o/Janus/Emu/
  Chameleon/LlamaGen (early fusion).
- Full survey with sources in session transcript 2026-07-05; details above suffice
  to implement. See paper_v2_outline.md §A6 for the claims each baseline serves.

### X15. E10 2-stage-name replots
- **Status**: 🔄 (agent replotting GPU-free figures into `*_v2names/` dirs).

### X16. Patch-level t-SNE (unpooled tokens, mechanistic model)
- **Motivation**: every existing t-SNE/probe mean-pools patch tokens (X10, object_count
  runs); test whether attribute organization exists at the individual-patch level,
  whether it survives two objects (or the objects' patches mix), and whether Grounding
  (CA with a shape-referring question) reorganizes the referent's patches.
- **Hypothesis**: object patches cluster by object/attribute per layer under no-CA; under
  `ca_refshape` the referent's patches separate or sharpen at GCA layers.
- **Design**: `clevr_dinov2_decoder1l_scratch_s42` (24×24 grid); 10 single-object (v3, no
  gray) + 10 two-object (v2, shape AND color differ, no gray); pixel segmentation →
  patch-owner masks (saturation gate + nearest-hue assignment — chromaticity is unusable,
  dim renders drift toward gray; morphology; squash-consistent 336² NEAREST; coverage
  ≥0.2 with best-patch fallback for tiny objects); t-SNE per GCA layer [1,3,5,7,9,11];
  conditions noca + `ca_refshape` ("What color is the {target.shape}?"); background
  subsampled to 100/img at plot time, subsample shared across conditions. Per-panel
  independent t-SNE fits — no cross-panel geometry claims.
- **Status**: ✅ run 08-13.
- **Results (qualitative)**: (1) noca, single 2-object image: both objects' patches form
  tight per-object clusters separate from background at every layer. (2) noca 10×1-object:
  color-major clustering emerges L7→L11 (same color, different shape merges by L11).
  (3) noca 10×2-object: object patches still separate from background but color clusters
  visibly mix relative to (2). (4) `ca_refshape`: by L9 referent patches aggregate into
  large referent-dominated clusters; at L11 object patches regroup into small per-image
  islands. Caveat: 2 distractors have only 1–2 patches (small distant objects) — flagged
  in log, do not over-read their cluster membership.
- **Artifacts**: `outputs/analysis/tsne/patch_level/{single_object_10,two_object_10}/`
  (script `scripts/analysis/tsne_patch_level.py`; feats npz cached, `--replot` and
  `--masks-only` supported; masks_debug.png per subset is the segmentation gate).
- **Consistency**: single backbone (D5 n/a); captions state n and model variant (D1/D6).

### X17. Reference probe on the GCA ViT (Song et al. §4.2 analog)
- **Motivation**: same-protocol comparison with the recode-repro Qwen §4.2 probe —
  does the in-stream (GCA) model carry a linearly decodable referent/non-referent
  signal in its patch tokens, and where does it emerge?
- **Design**: `clevr_dinov2_decoder1l_scratch_s42`; clevr_two_object_v2 filtered to
  shape≠ & color≠ & no-gray, 219/221 kept after hue segmentation (2 degenerate
  skipped). **Paired referring design**: each scene probed under both directions
  ("What color is the {target.shape}?" / "{distractor.shape}?"), so each object
  carries both labels across prompts — kills the target-is-always-large confound
  that made the first (unpaired) run trivially 1.0 in ALL conditions incl. noca
  (`reference_probe/two_object/` kept as the confound record; numbers there are
  artifacts, do not cite). Features = per-object mean over its own patches (counts
  vary; no fixed-16-token concat analog). Grouped-by-scene 80/20 logistic per block.
  Conditions: referring / noca / description / irrelevant.
- **Results**: referring 0.50 (block 0) → 0.625 (b1) → 0.761 (b4) → 0.977 (b5) →
  1.000 (b6-11); noca / description / irrelevant exactly 0.5 at every block
  (structurally: control features are direction-independent). Reference signal
  accumulates across successive GCA injections (layers 1,3,5), saturating mid-net.
- **Artifacts**: `outputs/analysis/reference_probe/two_object_paired/`
  (script `scripts/analysis/reference_probe.py`, `--replot`; feats npz cached).
  Companion Qwen runs: `../../recode-repro/outputs/probe_n200{,_prefix}/`.
- **Scene-level extension** (`scripts/analysis/reference_scene_tsne.py`,
  `outputs/analysis/reference_probe/scene_tsne/`): mean of ALL 576 patches per
  (scene, direction). noca = one undifferentiated cloud at every block; referring
  = scenes cluster by the QUERIED shape from L1 (described attribute; markers
  segregate) and reorganize into clean referent-COLOR islands by L11 (the answer
  value) — the Binding→Retrieval sequence visible at scene level, surviving mean
  pooling. Contrast: Qwen scene-level t-SNE (recode-repro `outputs/tsne_scene/`)
  shows NO referent structure under referring — its reference recoding is a small
  subspace shift that pooling (16/256 tokens) drowns, while GCA's in-stream
  reorganization dominates the pooled vector.
- **Status**: ✅ done 08-14 (probe + scene-level, both sides; Qwen full results in
  recode-repro JOURNAL).

### X18. Multi-object hallucination — pooled n1/n2 probe on raw backbones
- **Motivation** (user-directed 08-18, replaces the 3×3 per-object readout of E8
  as the site's presentation): hold the readout fixed at scene-level mean pooling
  (all patch tokens averaged) and vary ONLY object count (1 vs 2) — same method
  both sides, target attributes probed. Hypothesis: multi-object confusion arises
  at aggregation, not encoding. The experiment's public name is
  "multi-object hallucination" (hypothesis-named); measurement figures are
  "ViT backbone probing / t-SNE".
- **Design**: 4 ViT-B backbones (zero-gated GCA fwd = native), datasets
  `data/clevr_single_object_v3` (n1=500) + `data/clevr_two_object_v2` (n2=480,
  target always large → size single-class in n2, skipped). Per block: mean over
  all patch tokens → PCA(50) → logistic (5-fold), target attrs. t-SNE: DINOv2
  block 11 pooled, 288 pts/panel (per-combo 3 for n1; random 288 for n2),
  4-channel encoding (tab20 hue=color, shade=material, glyph=shape, size=size).
- **Results** (block 11, n1 → n2 target color): DINOv2 0.912 → 0.517,
  Sup-ViT 0.984 → 0.812, SigLIP 1.000 → 0.850, MAE 0.932 → 0.912; n1 all-attr
  0.91–1.00 for all 4 backbones. Shape/material stay ≥0.97 on n2 (target is the
  large object and dominates the mean). t-SNE: n1 = shape×material islands with
  color substructure; n2 = diffuse, no target-attribute organization. Note: raw
  DINOv2 n2 color 0.517 vs trained-model noca 0.356 (linear_probe_single.py) —
  same qualitative direction, protocols not point-comparable (different probe
  implementation details); recorded on the site.
- **Artifacts**: `outputs/analysis/raw_backbone_probe/pooled_n1n2/`
  (feats npz per backbone×dataset, probe_results.json, pooled_probe.png,
  pooled_tsne.png; `raw_backbone_probe.py --pooled [--only LABEL | --replot-pooled]`).
- **Status**: ✅ done 08-19 (CPU-only; GPU was occupied by the s44 cls run).
- **Dated correction (2026-09-16, X26 setup manifest)**: the July t-SNE / probe run
  `outputs/analysis/tsne/object_count/` (no `_v2`) was built on the **v1** sets
  `data/clevr_single_object` (500, 320×240) and `data/clevr_two_object` (480), verified by
  byte-comparing its `attrs.json` with the dataset files — not on the v3/v2 stimuli as the
  X18 notes elsewhere state. The v3/v2 sets were used by this entry's `pooled_n1n2/` and by
  X17. Historical status of all three unchanged; see `docs/unified_analysis_manifest.md` §2.

### X19. Patch-token PCA + KMeans on the paired renders — additive object vector
- **Motivation** (user hypothesis): a patch containing an object carries the
  local background representation plus an additive, object-specific vector.
  X16's patch-level t-SNE showed per-object clusters but is distance-based,
  nonlinear, and per-panel fit — cannot test additivity/linearity. PCA gives a
  global linear frame (n1/n2 panels share one fit); KMeans is the quantitative
  leg (user-specified: 5 random pairs, k=2 on 1-object / k=3 on 2-object,
  foreground red / distractor blue at alpha 0.3).
- **Design**: NEW paired dataset `data/clevr_object_count/{n1,n2}` (480 pairs =
  96 combos × 5 positions, target placement identical across n1/n2, ≥2-attr
  distractor, sizes free 240/240 — replaces invalidated single_object_v3 /
  two_object_v2). Model = `clevr_dinov2_decoder1l_scratch_s42` noca (frozen
  backbone ⇒ ViT backbone representation), X16 extraction+segmentation imported
  (`tsne_patch_level.py`). PCA set: 6 combos × 5 positions (position-invariance
  control); cluster set: 5 uniform-random pairs (both exclude gray / same-color
  distractors — hue segmentation limit). Offsets in full 768-d:
  offset = mean(object patches) − mean(bg patches). Clustering variants: raw
  tokens (as specified) and bgsub (per-position background template — the mean
  token at each position over images where it is background — subtracted; this
  is the additive hypothesis' own prediction).
- **Results** (script prints per layer; offset_stats.json):
  (1) Additivity holds and is object-specific: same-pair target offset is
  essentially unchanged by adding a distractor (n1-vs-n2 cos 0.998→0.962 L1→L11);
  within-combo-across-position cos > between-combo at every layer (L11
  0.912 vs 0.624), and after removing the shared "objectness" direction
  (top-1 SVD 0.60–0.79 of offset energy) residuals are combo-specific
  (L11: 0.729 within vs −0.152 between).
  (2) Raw-token KMeans k=2/3 does NOT recover objects (IoU 0.01–0.05): 550/576
  background tokens vary smoothly with position and dominate inertia — clusters
  are large spatial background regions. Consistent with (1): the object vector
  rides on a position-dependent background manifold.
  (3) bgsub KMeans recovers the foreground (n1 target IoU 0.60 at L1, 0.23 at
  L11; n2 foreground-union IoU 0.57→0.26; ARI 0.7→0.3): objects separate from
  background, but the two foreground clusters split core-vs-halo (shadow/edge)
  rather than object-vs-object — two objects' vectors are closer to each other
  than an object's core is to its own periphery, at k=3 L2 geometry. IoU is
  depressed by halo/shadow patches outside the strict pixel-based owner masks.
  (4) PCA panels: PC1+2 hold only ~25–39% variance (background positional
  manifold); object patches collapse into one tight clump by L11 in the global
  frame, color separation not visible in 2 PCs.
- **SigLIP leg** (user-ordered same day; `clevr_siglip_decoder1l_scratch_s42`
  noca, ViT-B/16 @256 → 16×16 grid, same 35 pairs/seed): additivity replicates
  (n1↔n2 target cos 0.999→0.922; within > between everywhere, L11 resid 0.616
  vs −0.116; top-1 SVD share higher than DINOv2, 0.59–0.84). Depth trend
  REVERSES vs DINOv2: bgsub KMeans at L1 separates OBJECT-vs-OBJECT (target
  IoU 0.568, distractor 0.629 ≈ foreground 0.664 — not the core-vs-halo split
  DINOv2 shows), then foreground clusters fragment into background scatter
  with depth (L11 IoU 0.07–0.11 vs DINOv2's 0.20–0.27). Raw-token KMeans fails
  at all layers like DINOv2 (IoU ≤0.08). Caveat: 16×16 grid → small objects
  hold only 2–3 patches (9 warnings, see log).
- **Artifacts**: `outputs/analysis/patch_pca_cluster/` (feats npz + labels +
  masks_debug per subset; pca_n{1,2}.png, offset_stats.json,
  cluster_overlay_n{1,2}{,_bgsub}.png, cluster_metrics{,_bgsub}.png,
  cluster_metrics.json, log.txt); SigLIP leg in
  `outputs/analysis/patch_pca_cluster/siglip/` (same layout). Script
  `scripts/analysis/patch_pca_cluster.py` (--masks-only / cached extraction /
  --replot, X16 three-phase pattern; CPU-only runs; SigLIP via --checkpoint
  ... --grid 16 --resolution 256 --out-dir .../siglip).
- **MAE and Sup-ViT (2026-08-27, user-ordered after X20 showed MAE's probe
  deficit; same 35 pairs / seed, CPU)**. Sup-ViT
  (`clevr_sup_decoder1l_scratch_s42`, `vit_base_patch16_384`, 24×24 grid, 1
  warning; `.../sup/`): behaves like DINOv2 — additivity holds (n1↔n2 target
  0.998→0.921; L11 within 0.889 vs between 0.607; resid 0.690 vs −0.143; top-1
  0.65–0.80), raw KMeans fails (≤0.12), bgsub foreground IoU 0.64 at L1 →
  0.16 at L11 (L1 separates target 0.60 / distractor 0.50). MAE
  (`clevr_mae_decoder1l_scratch_s42`, `vit_base_patch16_224.mae`, 14×14 = 196
  patches, 23 small-object warnings; `.../mae/`): DIFFERENT regime — object
  patches sit far from the background cloud at every layer in the
  single-image PCA (PC1+2 hold 45–62% vs 28–41% for DINOv2), offsets are the
  most position-invariant and type-specific of the four (L11 within 0.869 vs
  between 0.418; resid 0.757 vs −0.160; n1↔n2 0.986), and bgsub KMeans does
  NOT decay with depth (n1 target IoU 0.79–0.83 at every layer; n2 target
  0.62–0.81, distractor 0.41–0.61, foreground 0.62–0.73 at L11 — the only
  backbone where k=3 keeps splitting object-vs-object through L11). Raw
  KMeans still fails (n2 foreground ≤0.11; n1 ≤0.06). Reading: MAE's patch
  tokens stay appearance-local through the whole trunk (pixel-reconstruction
  objective), which is exactly the regime where per-patch object identity is
  strongest and pooled/answer readout is weakest (X20 decode 0.817) — the
  patch-level and probe-level pictures of MAE agree. Caveat: 14×14 grid
  inflates per-patch IoU (objects are 1–8 patches) — compare trends, not
  absolute IoU, across grids. `pca_single_*.png` now carries the scene with
  owner overlay in a left column (all four backbones replotted).
- **Caveats**: cluster-set pair 478 distractor has few patches at high layers
  (owner counts in labels.json); owner masks exclude shadows so halo patches
  count against IoU; PCA-set combos skew metal/large (5/6) under seed 42 —
  color is the diverse axis.
- **Status**: ✅ done 2026-08-19 (CPU-only; GPU left to the running s44 job).

### X20. Comprehensive linear probe — story-vs-evidence consistency check
- **Motivation** (user, 2026-08-26): several mechanism claims on the results
  site rest on accuracy alone; a linear probe is the representational view that
  can show whether the claimed difference exists. Audit found probe coverage of
  4/12 paper cells (DINOv2 × 3 readouts, SigLIP local patches), zero probes on
  the −CA ablation (the site's "causal baseline"), and an uncited
  near-chance probe on the ungated-CA model (decode 0.171 vs val acc 0.91–0.97).
- **Design**: same protocol as X10 (`linear_probe.py`, 72 queries × 500 db,
  answer_decode / answer_match, seed 42), unchanged except a `--categories`
  CLI (prefix of the default order reproduces the same queries). Queue on one
  GPU (`outputs/analysis/linear_probe/x20_probe_queue_2026-08-26.log`):
  Tier 1 `clevr_dinov2_concat_decoder1l_nogca_scratch` (−CA; GCA layers present
  with attn_gate frozen at 0) all 3 categories; Tier 2 direct only:
  mae/sup × {decoder1l, concat_decoder1l}; Tier 3 direct only: siglip concat,
  siglip/sup/mae cls. Aggregation `scripts/analysis/probe_table.py` →
  `outputs/analysis/linear_probe/probe_table.{md,json,png}` (readout ×
  backbone; L11 decode/match, peak, half-rise) + section in
  `docs/results_tables.md`.
- **Pre-registered readings** (written before results):
  (a) pretraining-objective claim — MAE decode under local patches ≥0.05 below
  the other three backbones → supported; equal → difference lives in readout
  training, rewrite. (b) Sup-ViT readout-interaction claim — Sup's two readouts
  give the same probe curve (Δ ≤0.02) while acc differs by 0.07 → supported;
  probe also drops → representation changed, rewrite. (c) −CA causal baseline —
  −CA answer_match near chance at every layer and decode well below CA models →
  language conditioning has a readable contribution in the ViT stream,
  supported; decode still high with match low → rewrite as "encoding present,
  selection absent". (d) mechanism-not-readout — DINOv2's three readouts share
  the direct half-rise layer → supported (existing data). (e) ungated-CA 0.171 —
  first rule out a loader artifact (strict=False key mismatch); if real, cite
  it at the site's gate design note.
- **Reading (e) resolved before results**: the ungated-CA 0.171 IS a loader
  artifact — `linear_probe.load_model` rebuilt the backbone with the default
  `use_gate=True`, `strict=False` left the missing `attn_gate` at 0, tanh(0)
  nulled every GCA block (the same bug `checkpoint_io.py` documents for the
  old eval loader). Fixed by switching `linear_probe.py` to
  `load_any_checkpoint` (+ `getattr(model, "decoder", None)` for CLS
  classifiers); smoke test with the fixed loader gives decode 0.80 / match
  0.91 at L11 on 8 queries × 60 db. The Tier-1 −CA run started under the old
  loader — equivalent for that checkpoint (its attn_gate keys exist, frozen
  at 0); all later runs use the canonical loader. **User ruling 2026-08-26:
  the ungated-CA variant is deprecated — not rerun, not compared, no row in
  the probe table; reading (e) is closed.**
- **Results** (queue done 2026-08-27 07:54; direct category, L11
  answer_decode acc / answer_match F1; `outputs/analysis/linear_probe/probe_table_direct.{md,json,png}`):

  | readout | DINOv2 | SigLIP | Sup-ViT | MAE |
  |---|---|---|---|---|
  | CLS token | 0.922 / 0.774 | 0.918 / 0.757 | 0.935 / 0.755 | 0.916 / 0.680 |
  | local patches | 0.922 / 0.773 | 0.921 / 0.798 | 0.937 / 0.709 | 0.817 / 0.533 |
  | local patches + question | 0.933 / 0.819 | 0.921 / 0.777 | 0.930 / 0.745 | 0.879 / 0.587 |
  | −CA (local patches + question) | 0.171 / 0.225 | — | — | — |

  Decode half-rise is L1 in every CA cell; −CA is flat at chance in all 12
  layers on all three categories (same: 0.225/0.068, spatial: 0.163/0.065 at
  L11).
- **Readings against the pre-registered criteria**:
  (a) MAE — supported under the local patches readout (0.817 vs 0.92–0.94,
  Δ ≥ 0.10) and in answer_match under every readout (0.53–0.68 vs
  0.71–0.82). BUT under the CLS-token readout MAE's ViT stream decodes the
  answer at 0.916, on par with the others, while its accuracy stays at 0.77:
  MAE's accuracy deficit is not purely representational — with a CLS readout
  the information is in the stream and the loss sits in the answer
  classification. Site wording must say "pretraining objective × readout",
  not "pretraining objective, not architecture". (b) Sup-ViT — supported:
  local patches vs +question decode 0.937 vs 0.930 (Δ 0.007 ≤ 0.02) while
  accuracy drops 0.07 → representation unchanged, the drop is on the readout
  side. (c) −CA — supported in the strong form (chance at every layer), with
  the label caveat: the label is the question's answer, so this shows the
  question never enters the ViT stream; object attributes remain readable
  (no-question pooled probe, X18) → wording "attributes present, selection
  needs cross-attention", not "causal baseline". (d) mechanism-not-readout —
  supported: within each backbone the three readouts share the decode curve
  (half-rise L1, peaks within 0.02) except MAE local patches. (e) closed
  (variant deprecated).
- **Status**: ✅ done 2026-08-27. Not yet: same/spatial for the eight new
  cells; site edits (await user).

### X21. Language condition on the patch object vector — measurement + causal additivity
- **Motivation** (user, 2026-08-27): X19 measured the additive object vector
  only without a question (a design gap the user identified); X20 shows the
  question enters the ViT stream through cross-attention from L1. Missing:
  what a referring question does to the object vector, at which layer, and
  whether the vector is causally additive. User asked for a literature survey
  first (three sweeps, 2026-08-27).
- **References that fix the design**: Song, Lepori & Pavlick 2025
  (arXiv 2608.00035) — concept-vector projections, Δ_ref / Δ_nonref, late-layer
  amplification of the queried attribute, steering/freezing; Feng & Steinhardt
  2024 (ICLR) and Saravanan, Tapaswi & Gandhi 2025 (CVPRW, on image patches) —
  difference-in-means binding vectors, additive swap, norm-matched random
  control; Assouel, Campbell, Bengio & Webb 2025 (arXiv 2506.15871) —
  additive binding IDs in VLMs are position pointers, identity-vs-position RSM
  dissociation; Lepori et al. 2024 (NeurIPS) — disentangled shape/color
  subspaces in the object's own tokens, cross-position injection; Darcet et
  al. 2024 — high-norm background tokens; Dai et al. 2024 — PC1 as index axis;
  Campbell et al. 2024 (NeurIPS) — set-size / conjunctive-search capacity
  conditions (not run here: only 1–2 objects). Opposing framing to position
  against: Haputhanthri, …, Webb 2026 (arXiv 2605.25427) — superposed object
  codes cause binding failure, serial attention fixes it; our claim: the
  additive vector is the substrate, gated cross-attention is the selection.
- **Design**: model `clevr_dinov2_decoder1l_scratch_s42` (local patches;
  loader `load_any_checkpoint`). n2 conditions c0 none / c1 refer target /
  c2 refer distractor / c3 "What color is the object?" (questions from
  `minimal_referring_question`, referent by shape>size>material); n1 c0, c1.
  All eligible pairs (X19 segmentation filter; ~324). Sparse cache per
  condition: object patches + 64 fixed background patches, object/background
  means (normed and pre-norm), token norms, GCA writes, per-patch attention
  onto the referent word. Part A: projections onto V = normalized c0 offset,
  Δ_ref/Δ_nonref with bootstrap CIs, per-patch change norm/cosine grouped by background / target / distractor,
  GCA write norm/cosine, offset stats per condition, identity-vs-position RSA,
  Darcet norm control (+ `_normstd` variant). Part B: Δ_ℓ(A→B) colour vectors
  by difference-in-means on n1 raw target means; residual-edit hook on block ℓ
  adds α·Δ to target / random norm-matched / background subset / background
  all / distractor patches under c1, and to distractor (Δ for its colour)
  under c2, target under c2; readout = decoder first-token argmax (checked
  against `generate`); flip rate on baseline-correct trials, logit(B)−logit(A);
  α ∈ {0.5,1,2}, ℓ 0–11. Part C: single-patch probes (bg/object, four
  attributes, referent vs non-referent under c1 ∪ c2 with c0 control) under
  random-by-image, slot-LOO and spatial-LOO (3×3 cell of the owner's
  centroid) splits, GCA layers. Script
  `scripts/analysis/patch_language_condition.py` (masks → extract →
  --intervene → --replot); `offset_statistics_from_offsets` factored out of
  `patch_pca_cluster.py` for reuse. Output
  `outputs/analysis/patch_language_condition/`.
- **Pre-registered expectations**: Δ_ref > 0 and Δ_nonref < 0 emerging by
  L5–L7 (X17 referent probe reaches 0.98 at block 5), growing with depth;
  Δ under c3 ≈ 0 relative to Δ_ref; referent-patch change aligned with V,
  background change ≈ 0; c0 offset stats reproduce X19 on its 30 pairs
  (±0.02); interventions: target+Δ flips at early/mid layers, random-vector and
  background controls at baseline error rate, distractor+Δ inert under c1 and
  effective under c2; probes: bg/object ≥ 0.95, referent probe 0.5 on c0,
  spatial-LOO ≈ random split if the object code is position-invariant.
- **Results (2026-08-27; 324 pairs; user framing: a replication of the cited
  methods on the GCA ViT)**:
  (A) Δ_ref(ℓ) = 0 through block 4, +2.8 (5), +11.2 (7), +26.5 (9), +21.3
  (10), +6.2 (11); Δ_nonref mirrors it (−26.4 at 9); relative to the mean
  offset norm: ±0.40 at blocks 9–10. Against no question: refer-this-object
  +3 at block 9, refer-other-object −23 — suppression of the non-referent
  carries the selection. Non-referring c3 moves both objects like the
  referring ones do (common component), and block 11 lowers the object
  projection by ≈15 under every question. Per-patch relative change grows to
  1.4× the token norm at block 11 for all owners alike; background-token
  change aligns +0.2 with V from block 7, object-token change −0.3 at 9–11.
  GCA write norm peaks at layer 9 (≈8–10) with no owner difference;
  cos(write, V) ≤ 0.11; patch→referent-word attention: objects > background
  at layer 5 (0.28 vs 0.22) but target = distractor. RSA on the target's
  patch mean: position RDM 0.6–0.78 through block 8, identity ≤ 0.16;
  questions reduce the position correlation earlier (blocks 7–9). c0
  reproduces X19 on the 30 pairs exactly. Norm-standardised variant gives the
  same picture (Δ_ref/‖offset‖ 0.36 at 9–10).
  (B) baseline c1/c2 accuracy 0.994; α=1 flip rate: target+Δ 0.80 (block 0),
  0.93 (2), 0.97 (5), 0.99 (8–10), 0.27 (11); random 0.00 everywhere;
  background subset 0.00; background all ≤0.04 through block 10 then 0.79 at
  11; distractor+Δ under c1 0.00 (0.16 at 11); distractor+Δ_D under c2 0.97;
  target+Δ under c2 0.00. α=0.5 reaches only 0.25–0.3; α=2 saturates.
  Verdict: the colour component of the patch token is causally additive and
  object-specific at every block but the last; the last block's readout
  draws on background tokens.
  (C) single-patch probes (≤6 tokens/object, 12 background/image, GCA
  layers; random-by-image / slot-LOO / spatial-LOO): background vs object
  0.99 at every layer; colour 0.99 (L1) → 0.91 (L11); shape 0.72 (L1) → 1.00
  (L5+); material 0.92 → 0.99; size 0.94 → 0.99; referent vs non-referent
  (c1 ∪ c2) 0.58 (L1), 0.94 (L5), ≥0.997 (L7–L11), no-question control 0.50.
  Spatial-LOO within 0.05 of the random split for every task → the
  per-patch object code generalises to unseen positions (supervised
  position-invariance). Colour's decline with depth mirrors X19's
  fragmentation and the intervention's block-11 exception.
  (A, RSA redone 2026-08-28, `--rsa-template`, `partA_rsa_template.json`,
  `rsa_template.png`): subtracting the image-wide background mean leaves the
  target's offset position-dominated (position RDM 0.6–0.8 through block 8);
  subtracting X19's per-position background template (built from the sparse
  cache: 19–60 background tokens per position, mean 36) removes it — position
  correlation 0.1–0.3 without a question (peak 0.59 at block 3, 0.11 at 11)
  and ≈0 from block 5 under any question. What remains is colour: the
  colour RDM correlates 0.43 at block 0 in every condition; without a
  question it decays to 0.01 by block 11 (X19's fragmentation); when the
  target is the referent (refer target / non-referring) it rises to
  0.59 / 0.53 at block 11; when the distractor is the referent the target's
  colour correlation falls to 0.06 at block 11 — the non-referent's colour
  is removed from its own patches, matching Δ_nonref. The 84-way identity
  RDM (all four attributes) is a weak model RDM (nearly every pair differs)
  and stays at 0.13–0.17; colour is the informative one.
  (D, readout check 2026-08-29, `--readout`, `readout_attention.json`,
  `readout_swap.json`, `readout_swap_trials.jsonl`, `readout.png`): the
  claim "the last block's readout draws on background tokens" (from B)
  tested directly. (i) Decoder cross-attention (1-layer VQADecoder, bos →
  576 patches, 8 heads, head-mean): 81% of the mass sits on background
  tokens because they are 97% of the tokens; per patch the referent object
  receives 13.4×1e-3 vs 1.5×1e-3 for a background patch (≈9×) and 2.0×1e-3
  for the non-referent; the top-attended patch is the referent in 76–77%
  of images, the non-referent in 1%, background in 21–23%; without a
  question both objects get ≈7–8×1e-3. (ii) Activation patching between
  conditions at every block output (receiver: refer target; masked patch
  tokens replaced by the same tokens from another forward pass at that block;
  identity control (replaced from the same forward pass) reproduces the baseline 1.00 at all 12 blocks; n = 320
  images correct under both questions). Tokens from the forward pass with the question about the distractor:
  swapping the two objects' tokens makes the answer the distractor's colour
  0.88 (block 7), 0.97 (9–10), 0.24 (11); swapping the background tokens
  0.00–0.02 through block 10, 0.71 at block 11; swapping only the
  distractor's tokens 0.36–0.50 at 7–10, only the target's ≤0.03 (0.10 at
  11). Tokens from the forward pass without a question: object tokens swapped drop P(target colour) to
  0.32–0.57 at 7–10 (answers become "other", not the distractor) and 1.00
  at 11; background swapped 0.99 through 10 and 0.87 at 11. Reading: the
  referent selection is carried by the object tokens from block 7 to 10
  (dominantly by the non-referent's tokens — suppression, matching Δ_nonref
  and the template RSA), and at block 11 it is copied into the background
  tokens, from which the decoder reads it; the block-11 exceptions in B
  (target+Δ 0.27, background+Δ 0.79) are the same effect.
  (E, attribute-specific directions 2026-08-29, `--attr-directions`,
  `partA_attr_directions.json`, `attr_directions.png`; Song, Lepori &
  Pavlick 2025 concept vectors): V[attr][value] = unit(mean patch-mean of
  1-object targets with that value − mean over all 1-object targets), an
  independent image set; projections of each 2-object object's patch mean
  per condition. Asked attribute (colour), own value, refer target − refer
  distractor on the target: 0 through block 4, +1.0/+1.3 (5–6), +0.8 (8),
  +5.3 (9), +6.0 (10), +11.4 (11); other colour values −1.3 to −2.1 at 9–11;
  the distractor mirrors it (−4.6, −5.2, −11.1). Against no question: any
  colour question (refer target, refer distractor, non-referring) raises
  the target's own-colour projection alike through block 8 (+12 to +15 at
  block 8); from block 9 the referent keeps it (+9 to +10) while the
  non-referent's falls (+4.1, +4.5, −3.4 at 9–11). The unasked attribute
  (shape) falls under any question on both objects alike (−14 to −18 at
  9–11, referent = non-referent). A separate shape-direction dip of the
  referent at blocks 5–8 (−5 to −8) occurs only when the referring word is
  a shape word (n=223: −7.7/−8.1/−12.9 at 5–7; size/material referring
  words: ≈0 to +4) — it belongs to matching the referring word, not to the
  unasked attribute. **Correction 2026-09-13 (X24 A3; the stratification was
  not implemented in code when the sentence was written — recomputed from
  `labels.referent_words`, `attr_directions_v2/partA_attr_directions.json`
  `by_ref_word`):** the referent's raw shape dip at blocks 5–7 is present for
  every referring-word class (shape n=223: −7.8/−6.6/−12.1; material n=42:
  −5.1/−3.9/−6.0; size n=59: −6.3/−6.1/−8.6). What is specific to shape
  referring words is the NON-referent: its shape projection stays ≈0 at 5–7
  (−0.0/+1.5/+0.8) whereas under material/size words it dips like the referent
  (−6.0/−5.9/−8.9; −6.3/−6.9/−9.2). The "≈0 to +4" row was the shape-word
  non-referent, not the size/material referent. After unit-normalisation the
  three strata are indistinguishable at blocks 9–11 (−0.42 / −0.45 / −0.46).
  **Limit added 2026-09-14 (X25 clarification):** the colour directions along which
  DINOv2's blocks 9–11 removal is measured classify held-out 1-object colours at only
  0.43 / 0.40 / 0.41 (2-fold by image, unit-normalised object means, chance 0.125),
  against 0.94 / 0.93 / 0.94 for SigLIP and 0.96 / 0.95 / 0.96 for MAE; DINOv2's late
  object means carry colour weakly (its decoder reads background tokens), so its
  late-block attribute numbers are less interpretable than SigLIP's.
  Reading: the model does both things — the asked
  attribute is amplified on every object by any question (Song et al.'s
  amplification, not selective), and from block 9 the selection is
  expressed as removing the asked attribute from the non-referent; the
  whole-vector Δ_nonref of −26 at block 9 is mostly this colour component.
  (F, SigLIP replication 2026-08-29, `clevr_siglip_decoder1l_scratch_s42`,
  grid 16 @ 256, same 324 pairs, `outputs/analysis/patch_language_condition/siglip/`,
  all phases A–E; masks inspected; c0 reproduces X19-SigLIP on its 30 pairs:
  L11 within 0.915 / between 0.743 / n1↔n2 0.922 vs 0.915 / 0.742). Same
  mechanism, earlier and without the last-block copy into the background tokens: Δ_ref 0 through
  block 4, +15.3 (5), +18.9 (6), +28.7 (7), then +21 to +25 through block
  11 (Δ_nonref the mirror); baseline accuracy 1.000 / 1.000. Interventions
  α=1: target+Δ 0.76 (block 0), 0.90–0.95 (1–5), 0.83–0.89 (6–11) — no
  block-11 exception; random / background-subset / distractor-under-c1 /
  target-under-c2 0.00 everywhere; background-all ≤0.18; α=0.5 0.03–0.33;
  distractor+Δ_D under c2 0.77–0.95. Readout: decoder attention per patch
  113×1e-3 on the referent vs 1.8×1e-3 background and 0.1×1e-3 on the
  non-referent (57% of the mass on the referent's few patches; top patch is
  the referent in 76–78%); swaps (tokens from the forward pass with the question about the distractor): object tokens
  → distractor's colour 0.85 (5), 0.80–0.88 (6–11); background tokens
  0.05–0.21; distractor's tokens alone 0.26–0.44, target's alone ≤0.03.
  Tokens from the forward pass without a question: object tokens swapped keep the target's colour
  0.90–1.00 at every block, but background tokens swapped at blocks 9–11
  make the answer "no" in 93% of images — in SigLIP the background tokens
  at 9–11 carry the question type (that a colour is asked), the object
  tokens carry which object and its colour. Template RSA: position 0.05–0.09
  at block 11 under all conditions (image-mean offset 0.12–0.17); colour
  0.60 when the target is the referent (refer target / non-referring),
  0.23 without a question, 0.10 when the distractor is the referent.
  Attribute directions: own-colour refer target − refer distractor +1.8
  (6), +7.8 (7), +13.0 (8), +17.7 (11), other colours −1 to −3; vs no
  question the referent's own colour rises (+6 to +10 at 8–11) and the
  non-referent's falls (−7 at 7–11) — here the split starts at block 7,
  not 9, and the non-referent goes below no-question already at block 7;
  shape falls under any question on both objects (−10 to −12 at 7–11);
  referent shape dip at 5–6 (−3). Single-patch probes (random / spatial-
  LOO): background vs object 0.99–1.00; colour 1.00 (L1) → 0.94/0.93 (L11);
  shape 0.86 → 1.00 (L3+); material 0.94 → 0.99; size 0.99 → 1.00;
  referent 0.57 (L1), 0.77 (L3), 1.00 (L5–L11), no-question control 0.50;
  spatial-LOO within 0.05 of random. Caveat: 16×16 grid — objects are
  1–20 patches, so per-patch probes and attention masses rest on fewer
  tokens than on DINOv2.
  (G, pre-registered 2026-08-29, queried attribute = shape): every
  analysis so far asked about colour, so "the queried attribute is
  amplified on both objects, the non-referent's queried-attribute component
  is removed from block 9" rests on one attribute. Rerun the whole suite
  (`--queried shape`, out-dir `patch_language_condition/shape/`, DINOv2,
  same 324 pairs) with questions "What shape is the {referring word} object?"
  where the referring word follows the dataset's fixed rule (first differing
  attribute in the order shape → size → material → colour, excluding the
  queried one): size for 186 pairs, material for 92, colour for 46; 101 pairs
  have the same shape for both objects and are excluded from the token swaps,
  the non-referring "What shape is the object?", and shape difference-of-
  means vectors for the interventions. Expectations written before the run:
  (i) the target's projection on its own shape direction rises on both
  objects through block 8 under any shape question, and its colour
  projection falls on both; (ii) from block 9 the non-referent's shape
  component is removed (refer target − refer distractor on own shape > 0,
  mirrored on the distractor); (iii) the block 5–8 dip previously seen on the
  shape direction when the referring word was a shape word now appears on the
  direction of the referring attribute (size for most pairs; split by
  referring-word type as before) — this tests the "matching the referring
  word" reading; (iv) token swaps: object tokens
  carry the selection at 7–10 and background at 11 as for colour; (v) shape
  difference vectors flip the answer at blocks 0–10 with the same controls at
  0. If (i)–(ii) fail for shape, the claim is narrowed to colour.
  Result (2026-08-29, `patch_language_condition/shape/`, all phases; the
  no-question condition reproduces the colour run's numbers exactly, so
  every difference is due to the questions): (ii) holds — target's own-shape
  projection, refer target − refer distractor: 0 through block 4, +0.9 (5),
  +2.0 (6), +2.3 (7), +5.0 (8), +5.8 (9), +8.9 (10), +14.7 (11), mirrored on
  the distractor (−15.0 at 11); the colour direction shows no selection
  (|Δ| ≤ 1, +2.5 at 11). Template RSA, shape RDM: without a question the
  target's offset organises by shape more and more with depth (0.02 → 0.77
  at block 11, the opposite of colour which faded to 0.01); refer target
  0.81 and non-referring 0.79 at block 11; refer distractor 0.05 (already
  0.14 at block 5 vs 0.31 without a question). Colour RDM under shape
  questions ≈0.01 at block 11 in every condition, i.e. colour fades as it
  does without a question. (i) FAILS: relative to no question, the target's
  own-shape projection does not rise under any shape question (refer target
  −0.6 … −5.5, non-referring similar), the non-referent's falls far more
  (−5.6 at 5, −11.3 at 9, −17.5 at 11); colour projections fall on both
  (−2 to −7). Offset norms behave as in the colour run (block 11: 46 no
  question, 31 referent, 23 non-referent), so this is not a norm artefact
  peculiar to shape. (iii) the block 5–8 dip on the referring attribute's
  direction was not tested (size/material directions have two values
  each; not analysed). (iv) holds: token swaps (219 images with distinct
  shapes) — both objects' tokens → distractor's shape 0.18 (3), 0.25 (5),
  0.74 (7), 0.92 (9–10), 0.13 (11); background 0.00–0.05 through 10, 0.87
  at 11; distractor only 0.32–0.37 at 7–10, target only ≤0.06; from the
  no-question forward: objects 0.83–0.93 keep the target's shape, background
  0.82/0.81 at 9–10 and 0.32 at 11. Decoder attention per patch (refer
  target): target 11.6, distractor 3.3, background 1.5 ×1e-3. (v) holds
  from block 5: shape difference vectors flip the answer 0.10 (5), 0.34 (6),
  0.65 (7), 0.92–0.96 (8–10), 0.02 (11), all controls 0.00; background-all
  0.90 at 11 — unlike colour vectors, which flipped from block 0, matching
  the single-patch shape probe (0.72 at L1, 1.00 from L5). Probes: colour
  0.99 → 0.91, shape 0.72 → 1.00 (L5+), referent 0.62 (L1) → 0.96 (L5) →
  1.00, spatial-LOO within 0.06. Reading: what generalises across the two
  queried attributes is (a) the non-referent loses the queried attribute's
  component from its own tokens (RSA colour 0.59 vs 0.06; shape 0.81 vs
  0.05), (b) the selection is carried by object tokens at blocks 7–10 and
  copied into background tokens at block 11, (c) the queried attribute's
  component is causally additive with all controls at 0, (d) the unqueried
  attribute is not maintained. What does not generalise is the rise of the
  queried attribute on both objects relative to no question: it occurs for
  colour, whose representation otherwise decays with depth, and not for
  shape, which the backbone keeps by itself. Inference (not measured
  directly): the question keeps the queried attribute on the referent
  against the backbone's default fate and removes it from the non-referent;
  when the default already preserves the attribute, only the removal is
  visible. The site claim is narrowed accordingly.
  (H, pre-registered 2026-08-29, head ablation scan, `--head-scan`,
  `head_scan.json`, `head_scan.png`): links the patch-level selection effect
  to the head-level causal localisation. Zero-ablate every self-attention
  head (12 × 12) and every GCA head (6 × 16) one at a time, plus all heads of
  one layer at a time, with `analysis.patching_utils.HeadAblator`; under
  each ablation run refer-target and refer-distractor on the 324 pairs and
  measure per block the target's projection on its own colour direction,
  refer target − refer distractor (baseline +5.3 / +6.0 / +11.4 at blocks
  9–11), plus accuracy. Report the change at block 11 per head and the
  Spearman correlation across heads between this change and the headwise
  activation-patching recovery on the same checkpoint
  (`headwise_by_type_stats.json`, colour-described and query groups).
  Expectations: (i) the largest drops come from GCA heads at layers 5–9,
  in particular the query-routing heads patching found at L7/L9; (ii) SA
  heads at blocks 9–10 reduce the effect if the removal is executed by
  self-attention after the GCA write; (iii) SA heads at block 11 change
  accuracy but not the effect measured at block 10; (iv) no SA head at
  blocks 0–4 changes the effect; (v) the per-head drop correlates
  positively with the patching recovery of the query group.
  Result (2026-08-29, 258 ablations, zero mode, 324 pairs; on-the-fly
  baseline reproduces the cached curve exactly: +5.3 / +6.0 / +11.4 at
  blocks 9–11, accuracy 0.994): no single head is necessary — over the 240
  single-head ablations the change of the block-11 effect has median 0.0,
  5th percentile −0.7, and no single-head ablation lowers accuracy below
  0.97; the one exception is SA block 11 head 7, whose removal halves the
  block-11 effect (11.4 → 5.1) while blocks 9–10 stay at baseline and
  accuracy stays 0.99 — a head of the final block's copy step, not of the
  selection. The selection is written by whole GCA layers: zeroing all 16
  heads of GCA layer 7 or layer 9 brings the effect at blocks 9–10 to
  +0.9/+0.8 and −0.8/−0.4 (baseline +5.3/+6.0) and the block-11 effect to
  +4.2, with accuracy 0.73/0.69 and 0.76/0.75; layer 5 halves it
  (+1.9/+2.3); layers 1, 3 and 11 do nothing (block-11 effect 11.1 / 10.6 /
  10.8, accuracy ≥0.98). Zeroing a whole SA block reduces the block 9–10
  effect by 2–3 for blocks 0, 3, 5–8 (distributed, each with accuracy
  0.83–0.97) and zeroing SA block 11 leaves blocks 9–10 at baseline but
  drops the block-11 effect to +2.0 and accuracy to 0.20. Correlation
  across heads between the block-11 drop and the headwise patching
  recovery is weak (Spearman −0.02 to +0.28; largest for the what_size and
  what_color query groups on GCA heads, +0.28 / +0.21). Reading against the
  expectations: (i) confirmed at the layer level (GCA 7 and 9 are the
  writing sites; GCA 5 partial), not at the single-head level — the
  selection is distributed over many heads within those layers; (ii) SA
  heads at 9–10 do not carry it individually; SA blocks 5–8 contribute
  collectively; (iii) confirmed — SA block 11 (and its head 7) affect the
  block-11 value and accuracy, not blocks 9–10; (iv) confirmed for single
  heads, but SA block 0 and 3 as wholes reduce the effect by ~2–3 (with
  accuracy loss, i.e. generic damage); (v) not confirmed — head-level
  patching recovery and head-level selection drop are only weakly
  related. Link to the causal localisation therefore holds at the layer
  level (patching's query-routing heads sit in GCA 7/9) and not head by
  head.
  (I, pre-registered 2026-08-31, head combinations, `--head-combos`,
  `head_combos.json`): sufficiency test on GCA layers 7 and 9. Conditions
  per layer: keep only the top-4 heads of the single-head scan (L7:
  H11/H2/H9/H3, L9: H2/H10/H9/H7) and zero the other 12; zero only the
  top-4; keep 4 random heads (3 seeds); zero 8 random heads (3 seeds);
  zero all 16 (reference); plus keep-top-4 on both layers at once.
  Expectations: if the writing is distributed, keeping the top-4 recovers
  less than half of the selection effect at blocks 9–10, zeroing the top-4
  costs no more than the sum of the single-head drops (≈2.8 at L7, ≈1.3 at
  L9, against a baseline of 11.4 at block 11), random-4 ≈ top-4, and
  zeroing 8 random heads removes roughly half — i.e. the effect scales
  with the number of heads kept, not with which ones.
  Result (2026-08-31, `head_combos.json`; baseline S9/S10/S11 = +5.3/+6.0/
  +11.4, accuracy 0.994): the writing is graded, not uniform. Keeping only
  the top-4 heads of layer 7 preserves S9/S10 at +3.6/+4.2 (≈70% of the
  layer's contribution) with accuracy 0.98; keeping 4 random heads
  preserves +1.3 to +2.0 (3 seeds) with accuracy 0.75–0.93 — so the top-4
  are far better than random, against the fully-distributed expectation.
  Zeroing only the top-4 of layer 7 still leaves +2.4/+2.7 (about half),
  matching the near-additive sum of single-head drops, so the top-4 are
  not necessary either. Layer 9 same pattern, weaker concentration
  (keep-top-4 +2.8/+2.8, random +0.2 to +1.0, zero-top-4 +3.0/+3.2).
  Zeroing 8 random heads ≈ zeroing the top-4 (S9 +2.4 to +3.6). Keeping the
  top-4 of both layers at once gives +2.2/+2.4 with accuracy 0.73–0.81.
  Reading: within GCA layers 7 and 9 the selection writing is concentrated
  in about a quarter of the heads, which carry roughly half to two thirds
  of it, and the rest is spread over the remaining heads; no subset of 4 is
  sufficient for the full effect and none is strictly necessary. The site
  sentence "分散在許多 head 上" is refined accordingly.
  (J, pre-registered 2026-08-31, queried = material and queried = size,
  out-dirs `patch_language_condition/{material,size}/`, no intervention
  phase — with two values per attribute a flip target B ∉ {A, A_distractor}
  rarely exists): direct test of the inference from G. Measured on the
  existing no-question cache before launching, the template-RSA correlation
  of the target's offset with the material RDM stays at 0.09–0.23 across
  depth and with the size RDM at 0.09–0.19 — both attributes sit between
  colour (0.43 → 0.01, discarded) and shape (0.02 → 0.77, kept).
  Expectations: (i) the removal generalises — at block 11 the queried-
  attribute RDM correlation is high for the referent and near the
  no-question value or lower for the non-referent, and the projection
  contrast (refer target − refer distractor on the own-value direction) is
  positive from the middle blocks; (ii) the both-object rise relative to no
  question is small (clearly below colour's +12 to +15 at block 8) because
  the backbone loses little of either attribute — if material or size shows
  a colour-sized rise, the inference "the rise exists only where the
  backbone would discard the attribute" is wrong; (iii) token swaps behave
  as before (object tokens carry the selection mid-depth, background at
  block 11), on the subset of pairs whose two objects differ in the queried
  attribute.
  Result (2026-08-31, both runs complete; template-RSA correlations of the
  target's offset with the queried attribute's RDM, template subtraction):
  material — no question stays at 0.16–0.23 from block 5 on; refer target
  0.86 at block 11, non-referring 0.80, refer distractor (target is the
  non-referent) 0.04. Size — no question 0.10–0.19; refer target 0.73–0.75
  from block 9, non-referring 0.73–0.81, refer distractor 0.09 at block 11.
  So (i) holds: the non-referent loses the queried attribute in both.
  Against (ii), the referent's organisation by the queried attribute RISES
  far above the no-question level (0.19 → 0.86 material, 0.10 → 0.73 size),
  i.e. material and size behave like colour (an attribute the backbone
  keeps only weakly is boosted when queried), not like shape; the
  projection rise stays small (refer target − no question on the own-value
  direction ≤ +2.8 for both, vs colour's +12 to +15), so the boost shows in
  the rank-order organisation more than in the raw projection — the
  difference between the two measures is recorded, not explained. The
  whole-vector selection contrast peaks at block 9 (+25.7 material, +28.4
  size) and the own-value contrast at block 11 (+10.3, +4.9). (iii) holds:
  token swaps (183 / 174 images with distinct answers) — objects' tokens
  move the answer from block 7 (0.99 material, 0.86–0.99 size at 7–10),
  background tokens only at block 11 (0.96–0.97 material, 0.67 size),
  target keeps its value at block 11 when objects are swapped (0.96, 0.75)
  — the DINOv2 final-block copy appears for all four queried attributes.
  Summary across the four: the removal from the non-referent and the
  final-block copy are attribute-general; the boost of the queried
  attribute tracks how weakly the backbone keeps it by default (colour,
  material, size boosted; shape, already kept at 0.77, not), which
  supports the inference of (G) in rank-order terms.
  (K, launched 2026-08-31): MAE replication of the whole suite
  (`clevr_mae_decoder1l_scratch_s42`, grid 14 @ 224, out-dir
  `patch_language_condition/mae/`, all phases). Purpose: third backbone for
  the backbone-specific final-block step — DINOv2 copies the selection into
  background tokens at block 11, SigLIP does not; MAE keeps object-vs-object
  separation to the last block (X19), so if it also shows no final-block
  copy, the copy is DINOv2-specific rather than shared. Caveat recorded in
  advance: MAE's VQA accuracy is 0.742 and the 14×14 grid gives few patches
  per object; baseline accuracy on the referring questions may be low, which
  shrinks the usable image set for swaps and interventions.
  Result (2026-08-31, all phases, 324 pairs, 14×14 grid; c0 reproduces
  X19-MAE on its 30 pairs: within 0.869 / between 0.417 vs 0.869 / 0.418;
  baseline accuracy on the referring questions 0.997 / 0.997, so the swap
  set is 323 images): MAE selects WITHOUT removal. (a) The token-level
  selection effect is an order of magnitude smaller than in DINOv2 — Δ_ref
  on the whole-object direction ≤ +1.3 (block 11; DINOv2 +26 at 9), the
  own-colour contrast ≤ +1.6, and the colour RDM correlation of the target's
  offset stays at 0.48–0.55 in every block and every condition (refer
  distractor 0.51 at block 11 vs refer target 0.54): the non-referent keeps
  its colour. Question − no question projections are ≤ 3 in magnitude and
  identical for referent and non-referent. (b) Referent identity is
  nevertheless linearly readable per patch from mid-depth (probe 0.53 L1,
  0.79 L7, 0.97 L9, 0.99 L11; no-question control 0.50) and GCA write norms
  are largest at layers 9 and 11 (17–25), so the question writes a
  referent marker onto the object tokens without altering their attribute
  content. (c) The decoder's attention does the selecting: per patch
  127.8×1e-3 on the referent, 1.3 on the non-referent, 2.9 on background
  (top patch on the referent in 90% of images; no question: 36.6 / 33.4,
  no preference). Token swaps: both objects' tokens from the other question
  change the answer 0.13–0.15 at blocks 7–10 and 0.90 at block 11;
  background tokens 0.00 through 10 and 0.12 at 11; distractor only 0.49 at
  11, target only 0.03; from the no-question forward, background tokens at
  block 11 drop P(target colour) to 0.53 (a question-type dependence of
  background tokens at the last block, as in SigLIP) and object tokens to
  0.67. (d) Colour difference vectors flip the answer at every block (0.54
  at 0, 0.81 at 2, ≥0.92 from 5, 0.95 at 11), random / distractor / target-
  under-c2 controls 0.00, background-all ≤ 0.23 — no final-block exception.
  Reading for the backbone-specific question: the DINOv2 copy into
  background tokens at block 11 is not shared by SigLIP or MAE, so it is
  DINOv2-specific. Larger point: the "removal from the non-referent" seen
  in DINOv2 and SigLIP is not a property of gated cross-attention per se —
  on MAE, whose tokens keep every attribute to the last block (X19), the
  same architecture leaves the attributes intact and marks the referent
  for the decoder's attention, with the answer-determining step at block
  11. Whether this reflects the backbone's representational default (MAE
  keeps both objects separate to the end) is an inference, not measured.
  Caveat: 14×14 grid, 1–8 patches per object.
- **Status**: ✅ A–K done 2026-08-31. Site section updated the same day.

### X22. Relational selection in the visual stream — pre-registered mechanism tests
- **Naming** (user, 2026-09-06): *relational* covers the same-as and spatial
  question types. A relational question is a selection conditioned on a
  property of another selected object: select the anchor, make one of its
  properties (an attribute value, or its location) available, select the
  answer object by that property, read the queried attribute of the answer
  object. Attribute-match relations compare in an attribute subspace;
  geometric relations compare positions and, under absolute position codes,
  need the anchor's position broadcast before a relative comparison exists.
  The story is stated in these general terms; CLEVR facts (camera-frame 3D
  relations, templated questions) are controls or limitations, never claims.
- **Motivation**: the 2026-09-02 three-object batch (transplant, projection,
  write-position) was exploratory. Five literature sweeps (2026-09-06) turn it
  into hypotheses with predictions and disconfirmation criteria written down
  before any run. Registered here before the first GPU job.
- **References that fix the design**: depth as sequential hops — Sanford,
  Hsu & Telgarsky 2024 (arXiv 2402.09268), Peng, Narayanan & Papadimitriou
  2024 (arXiv 2402.08164), Biran et al. 2024 "Hopping too late" (EMNLP), Yang
  et al. 2024 (ACL), Liu et al. 2023 shortcuts (arXiv 2210.10749), Wang et
  al. 2024 grokked composition (NeurIPS); relations as inner products —
  Webb et al. 2024 relational bottleneck (TICS), Kerg et al. 2022 CoRelNet,
  Altabaa et al. 2024 Abstractors (ICLR), Altabaa & Lafferty 2024/2025
  (arXiv 2402.08856, 2405.16727), Lepori et al. 2024 (NeurIPS); position
  infrastructure — Ke, He & Liu 2021 TUPE (QK decomposition into word/pos
  terms), Kazemnejad et al. 2023, Heo et al. 2024 RoPE-ViT, Yamamoto et al.
  2026 left/right heads from pos-embed × QK (arXiv 2601.12809), Meng et al.
  2021 Conditional DETR (content vs spatial query), Liu et al. 2022 DAB-DETR,
  Cui, Prakash, Bau et al. 2026 spatial binding rides on background tokens
  (arXiv 2603.22278); suppression and scratch — McDougall et al. 2023 copy
  suppression, Lad, Gurnee & Tegmark 2024 residual sharpening, Darcet et al.
  2024 registers (high-norm tokens lose position), Sun et al. 2024 massive
  activations (registers as biases), Pfau et al. 2024 filler tokens;
  binding substrates — Smolensky 1990, Plate 1995, Kanerva 2009 (unbinding =
  inner product, crosstalk grows with shared components), Hersche et al.
  2023 NVSA, Ramsauer et al. 2020 Hopfield attention (metastable mixtures
  for similar patterns); VLM binding — Assouel et al. 2025, Campbell et al.
  2024, Haputhanthri et al. 2026, Song, Lepori & Pavlick 2026, Feng &
  Steinhardt 2024; spatial failure modes — Chen et al. 2025 (ICML), Kamath
  et al. 2023 What's Up, Qi et al. 2025, Ma et al. 2026.
- **Model / data**: `clevr_dinov2_decoder1l_scratch_s42` (GCA at blocks
  1,3,5,7,9,11; 1-layer decoder on patches); `data/clevr_three_object_v2`
  (672 scenes, pairwise-distinct non-gray colours); roles A anchor (named by
  colour), T answer, D third; conditions c0 no question, c1 clean, c2
  corrupted (same-as: T named; spatial: opposite word), new c3 (spatial:
  same word, D named — the 09-02 transplant could not test anchor necessity
  for spatial because c2 keeps the anchor). New: queried attribute q ∈
  {shape, material, size} with shared attribute s ≠ q (anchor still named by
  colour); dissociation render `data/clevr_three_object_edge` (anchor pushed
  to the image edge so the answer object lies on the opposite image half from
  the asked direction); generality on `clevr_siglip_decoder1l_scratch_s42`,
  `clevr_dinov2_decoder1l_scratch_s43` (MAE optional).
- **Hypotheses, predictions, disconfirmation** (fixed before running):
  - H1 ordering: the anchor's conditioning property is linearly decodable
    before candidate tokens become causally necessary. Pass: probe onset ≤
    transplant onset (same-as ≤ 3, spatial ≤ 9). Fail: candidates causal first.
  - H2 transport, exactly one of: (a) property broadcast into candidate
    patches, (b) held in background patches, (c) stays on the anchor and is
    read by self-attention at the comparison block. Measured by per-block
    probes from A/T/D/background (low- and high-norm) tokens and by SA
    candidate→anchor mass c1−c0 per block and head. Fail: none holds, or the
    c1−c0 change appears only at GCA layers with no SA change (GCA cannot
    compare two patches: its keys are text).
  - H3 attribute-specific comparison: projecting the anchor tokens onto the
    complement of the shared-attribute subspace inside the anchor's causal
    window breaks the answer; removing a non-shared attribute subspace or a
    random subspace of the same rank does not. Fail: everything breaks, or
    nothing does.
  - H4 match marker: the selected candidate's c1−c0 change projects
    positively and increasingly on the single-hop referent-marker direction
    (X21); A and D do not. Fail: no T/D difference or orthogonal direction.
  - H5 suppression after use: (i) block-11 GCA write on A has negative cosine
    with A's own answer-attribute direction; (ii) masking only that write
    shifts answers toward A's attribute; (iii) the effect is weaker when A
    and T share the queried attribute. Fail: orthogonal write, or masking
    has no effect.
  - H6 interference: logit margin falls monotonically with the number of
    non-queried attributes D shares with A (0/1/2: n = 204/339/109). Fail:
    flat.
  - H7 geometry, absolute first then relative: (i) the shallow-layer field is
    anchor- and content-independent and follows the positional embedding
    (cross-scene field correlation; flipping the pos-embed grid flips the
    field); (ii) the relative field appears ≥ 1 GCA layer after the anchor's
    location is decodable from background tokens; (iii) on the dissociation
    set the model answers the relative relation (accuracy near overall), not
    the image half. Fail: field follows content; relative before anchor;
    dissociation accuracy near 0.5.
  - H8 anchor-position broadcast: SA heads at blocks 8–9 carrying anchor →
    background mass (selection rule fixed: |c1−c0| ≥ 10× median, as in
    `select_ablation_heads.py`; prior candidates from the 07-15 anchor-swap
    patching: GCA L5H4, L11H8, L9H8, SA 8:0). Ablation leaves the absolute
    field, lowers relative R² and dissociation accuracy; disjoint random
    heads do not. Fail: no head meets the rule, or both fields collapse.
    **2026-09-12 rerun; edited by Codex from Claude's record.** The earlier
    runs used the maximum of background-query→anchor-key and
    candidate-query→anchor-key attention-change ratios. The rerun uses only
    the registered background statistic (`--h8-measure background`); the
    default remains `both` for reproducing older runs. New artifacts are
    `outputs/analysis/patch_language_condition/relational_*/h8_head_ablation_bgrule/results.json`.

    Selection normalizes |c1−c0| by its median over all 144 block/head
    combinations, then considers the 72 heads in blocks 5–10, ranks those
    reaching 10× the median, and ablates at most eight. “Cell” previously
    meant either a head or an experimental setting; use those explicit terms.

    | Task / backbone | Qualifying / ablated heads | Baseline accuracy | Selected ablation | Random seed 0 / 1 |
    |---|---:|---:|---:|---:|
    | same-as DINOv2 | 10 / 8 | 0.972 | 0.732 | 0.954 / 0.914 |
    | same-as SigLIP | 14 / 8 | 0.977 | 0.628 | 0.975 / 0.966 |
    | spatial DINOv2 | 12 / 8 | 1.000 | 0.998 | 0.998 / 0.964 |
    | spatial DINOv2, edge-render data | 12 / 8 | 1.000 | 1.000 | 0.994 / 0.982 |
    | spatial SigLIP | 0 / 0 | 1.000 | Not applicable | Not applicable |

    Same-as remains sensitive to the selected ablation on both backbones
    (earlier selected accuracies: 0.744 / 0.611). Random sets have equal head
    counts but are sampled across all blocks, not matched by layer; only two
    random sets were tested. Spatial SigLIP's saved selected/random rows use
    empty sets and therefore are not head-ablation evidence. The relation
    contrast is backbone-specific: DINOv2 has qualifying heads for both tasks,
    whereas SigLIP has them only for same-as under this rule.

    **Measurement correction:** `write_position_r2` regresses background-patch
    GCA update-norm differences (c1−c2) on patch coordinates, not anchor
    coordinates on activations. Its block-11 absolute-coordinate R² is
    0.360 / 0.072 / 0.502 / 0.464 (none / selected / random 0 / random 1)
    for spatial DINOv2, and 0.344 / 0.064 / 0.495 / 0.465 for edge-render
    data. These are in-sample spatial fits, not held-out position-decoding
    scores. Random ablation increases the fit for unexplained reasons.
    Report the observations; they do not establish anchor-position broadcast
    or the full original H8 hypothesis. Definitions, sample counts, answer-role
    denominators and interpretation limits are in JOURNAL's 2026-09-12 rerun entry.
  - H9 generality: the ordering results (H1, H2, H5(i), transplant windows)
    replicate on SigLIP and seed 43; numbers are not required to match.
- **Controls fixed**: c0 no-question; transplant self-control (1.00);
  random subspace of equal rank; random norm-matched vectors; disjoint random
  heads; random background rows for pos-embed edits; probes GroupKFold(5) by
  scene; bootstrap CIs over scenes; all cells reported whatever the sign.
- **Known limitations declared up front**: spatial accuracy is at ceiling
  (margins replace error rates); CLEVR relations are camera-frame 3D while
  the analysis uses 2D centroids (disagreeing scenes reported, not claimed);
  pos-embed edits perturb frozen features (interpret against the random-row
  control); probes are correlational, transplant / projection / masking /
  ablation are causal — the write-up keeps the two levels apart.
- **Code**: `scripts/analysis/patch_language_condition.py` (`--relational-probes`
  CPU mode on the existing caches; `--relational-v2` GPU passes with
  `--sa-attn`, `--project-attr`, `--gca-mask`, `--pos-flip`, `--head-ablate`,
  `--same-queried`, `--spatial-c3`); hooks `SubspaceProjector`,
  `GCAWriteMasker`, `PosEmbedEditor`, `SAAttnCapture` in
  `src/analysis/patching_utils.py`; render option `--anchor-edge` in
  `scripts/analysis/render_single_objects.py`. Outputs in new directories
  `patch_language_condition/relational_{same,spatial}_{probes,v2}`,
  `relational_same_q{shape,material,size}`, `relational_spatial_edge`.
- **Status**: registered 2026-09-07 before the first GPU job. ✅ All stages
  run 2026-09-07/08 (DINOv2 s42, SigLIP s42, DINOv2 s43 epoch 14, edge
  render, CLEVR-val dissociation). Tally in JOURNAL 2026-09-08: H1 pass;
  H2 (b)+(c) with (c) causal for same-as; H3 specific but small; H4 partial;
  H5 disconfirmed; H6 disconfirmed; H7 deep-layer stage pass, shallow-layer field
  content-driven; H8 disconfirmed for spatial, supported for same-as; H9
  order replicates, anchor-relative field s42-only. Addendum 2026-09-08:
  SigLIP H5/H8 cells (`relational_same_v2_siglip/{h5_gca_mask,
  h8_head_ablation}`, `relational_spatial_v2_siglip/h8_head_ablation`) —
  same-as head ablation causal (0.977 → 0.611, heads at blocks 5–6),
  spatial null, GCA-write mask at blocks 9/11 no effect (read-out finished
  by block 7 on SigLIP); see JOURNAL 2026-09-08 addendum.

### X23. CLEVR mechanism observations on real images (GQA) — pre-registered
- **Design document**: `docs/gqa_relational_experiment_design.md` (v5, 2026-09-11,
  reviewed by the user in ten points; all adopted). Four primary hypotheses with
  frozen statistical criteria (H1 referent selection generalizes; H2 the anchor's
  property is decodable outside the anchor before the target is causally
  necessary, ordering score ≥ 0.9; H3 head-ablation interaction same-as vs
  spatial, CI lower bound > 0, plus cumulative ablation curve; H4 the target
  acquires the single-hop referent marker, target − third object CI > 0).
  Everything else secondary / exploratory. Analysis populations: S_eligible
  (scene graph / program / geometry only) for observational analyses, S_correct
  for interventions, coverage reported everywhere. Bootstrap by image, 95 % CI.
- **Models**: GQA-trained `gqa_siglip_decoder1l_scratch_s42` (primary; its
  `best.pt` was chosen on balanced val in 2026-06 with aggregate accuracy only —
  X23 items are held-out but not checkpoint-independent, stated in the paper);
  CLEVR-trained `clevr_siglip_decoder1l_scratch_s42` as a transfer test on
  questions whose answers are CLEVR colours (direct only, see step 0).
- **Data**: GQA val balanced (only train/val carry scene graphs); counterfactual
  questions are natural GQA questions of the same image from
  `val_all_questions.json` (spatial c2 = same anchor + opposite relation; c3 =
  same target + other anchor; direct c2 = other referent, same query; same-as
  c2 = roles swapped). Geometry presets: relaxed (patch counts as an object's at
  ≥ 0.3 cover, ≥ 2 patches per role, ≤ 50 % image area, role IoU ≤ 0.05) for
  the primary analysis; strict (0.5 / 4 / 30 % / 0) as a robustness subset;
  margin along the relation axis 2 patches (1 / 3 as robustness). Background
  (b) = patches no box covers at the preset threshold; (a) = non-role object
  patches. Left/right only; front/behind excluded.
- **Step 0 (2026-09-11, CPU, no model output seen)**: code
  `src/analysis/gqa_roles.py` + `--gqa-filter {direct,spatial,same}` in
  `patch_language_condition.py`; outputs
  `outputs/analysis/patch_language_condition/x23_step0b_{spatial,same,direct}/`
  (records, owner maps, funnel per rule and per variant, blind audit page
  `audit/index.html`). S_eligible (relaxed, margin 2): spatial 263 questions /
  157 images (c2 80, c3 188, both 5; dissociated 30; all category answers —
  no attribute-answer spatial question has a natural counterfactual); direct
  1,016 / 759 images (233 answerable by the CLEVR model); same-as 25 (only 2
  with a counterfactual → same-as on GQA is observational and behavioural only;
  H3's same-as side stays on CLEVR). Strict subsets 48 / 500 / 10. Earlier
  draft directories `gqa_{spatial,same,direct}`, `_v2`, `_v3`,
  `gqa_step0_*`, `x23_step0_*` are superseded drafts of the same step (first
  rules; constructed counterfactuals; background definition; referent-word
  casing) and are not used.
- **Rule revisions made in step 0** (before any model output): natural instead
  of constructed counterfactuals; relaxed geometry preset as primary; background
  (b) redefined. Reasons and the first-version funnel (76 / 8 / 125) are in the
  design document §5a.
- **Step 0 audit (2026-09-11, model annotators)**: eight Opus subagents
  audited the blind samples (rule: tick only when confident). Four core checks
  (roles, boxes, relation, unique) pass for 6 / 60 spatial, 10 / 60 direct,
  2 / 25 same-as; `unique` is the main failure (scene-graph uniqueness is not
  visual uniqueness), then 2–3-patch objects on cluttered regions and boxes
  dominated by a neighbour. Answers: `x23_step0b_<mode>/audit/audit_answers_model.json`.
  Same-as on GQA is dropped (2 usable items).
- **Step 0c — visual selection (rules frozen 2026-09-11 before running)**:
  applied to the whole S_eligible of step 0b (spatial 263, direct 1,016);
  output `x23_step0c_{spatial,direct}/` (records, owner maps, funnel, all
  annotator answers). Rules, in order: (V1) every role object covers ≥ 4
  patches at the relaxed threshold (spatial 159, direct 640 survive; from
  the records, no annotator); (V2) an Opus annotator sees the box-overlaid
  image and the questions and marks roles / boxes / relation / unique / c2,
  ticking only when confident; the item is kept iff roles, boxes, relation
  and unique are all true. `unique` in this pass: a second instance of the
  anchor's (direct: referent's) category counts as a competitor only if it is
  clearly visible — roughly ≥ 4 grid patches, not cut off at the border, not
  heavily occluded; tiny background instances are ignored. (V3) spatial:
  a c2 / c3 whose check fails is removed from the item (has_c2 / has_c3 set
  false) but the item stays; direct: the c2 check must pass (the pair defines
  the third object), else the item is dropped. Annotators see no model
  output. The populations S_eligible of every hypothesis are read from
  step 0c from here on; 0b stays as the pre-audit funnel. Expected sizes at
  the sampled pass rates are small (spatial tens, direct low hundreds) and
  are reported as-is.
- **Step 0c result (2026-09-11)**: 33 Opus annotators, 799 items, every
  item answered. Funnel (first failing check in the order roles, boxes,
  relation, unique): spatial 263 → V1 159 → roles 113 → boxes 98 → unique
  **35** questions / 27 images (c2 10, c3 18; relation held in 157 / 159);
  direct 1,016 → V1 640 → roles 515 → boxes 406 → relation 352 → unique 203
  → c2 **184** questions / 148 images (45 answerable by the CLEVR model).
  Outputs `x23_step0c_{spatial,direct}/` (records with `visual_audit` per
  item, owner maps, funnel, `audit_answers_model_full.json`, all box-overlaid
  images under `audit/`). Consequences: the CLEVR-trained model's transfer
  interventions are below the n_correct ≥ 100 gate (45 eligible) and are
  reported as behavioural only; spatial interventions use n ≤ 18 per
  counterfactual and are reported with their CIs, no further relaxation
  without a new directory.
- **H2 probe method and its lineage (recorded 2026-09-12 at the user's
  request)**: the anchor-position probe is a linear decoding probe in the
  sense of Alain & Bengio 2016 / Belinkov 2022 — a linear read-out from
  background (b) tokens to the anchor centroid, so that held-out R² can be
  attributed to the representation rather than to the read-out's own
  capacity. Ridge regression for a continuous target with the regularisation
  strength chosen by inner cross-validation and held-out R² reported follows
  Gurnee & Tegmark 2024 (space/time probes on the residual stream) and the
  neuroscience encoding/decoding practice (Naselaris et al. 2011, Huth et al.
  2016); cross-validation grouped by image prevents leakage from correlated
  tokens of the same picture; the shuffled-label run is the control task of
  Hewitt & Liang 2019. The CLEVR version (X22 `_ridge_r2`: Ridge alpha = 1,
  GroupKFold(5) by scene, mean R²) is the simplified form — alpha was not
  literature-derived; with ~500 scenes per condition it mattered little. On
  GQA (27 images) a fixed alpha (10) gave negative R² at every block
  (`x23_gqa_spatial/`, kept), so the GQA version (`gqa_h2_position_probe`,
  `x23_gqa_spatial_v2/`) selects alpha by RidgeCV on the training fold,
  adds the shuffled-label control and an image bootstrap (200, refit) for
  the ΔR² CI; the k_condition rule (ΔR² CI lower > 0.1) is unchanged. The
  probe is correlational evidence only; causal claims rest on transplant and
  head ablation.
- **Results (2026-09-12, JOURNAL same day)**: H1 pass on query attr
  (+2.21 [+1.83, +2.61], n = 184); H4 pass on spatial (+1.74 [+0.90,
  +2.56], n = 35); **H2 not decidable on this population** — the corrected
  probe (`x23_gqa_spatial_h2/`, image-grouped alpha, leakage-free bootstrap)
  gives ΔR² −0.019 to −0.003 with every CI containing zero and k_condition
  None, but a power control recovering the anchor centroid from the anchor's
  own patches reaches only R² +0.234 at block 0 and is negative by block 9,
  so the registered threshold of 0.1 is unreachable at 27 images; report as
  untestable, never as evidence against transport (JOURNAL 2026-09-12 later); transplant k_target None
  (n = 3 / 12 per counterfactual); H3 rule selects 0 heads, cumulative
  curve no difference from random up to m = 8; CLEVR-trained model: H1
  +0.26 [−0.09, +0.64] not supported, referent / background attention 4.0.
  Reverse direction (added 2026-09-12 at the user's request, not
  pre-registered, behavioural + observational): GQA-trained SigLIP on the
  CLEVR query-attr pairs (`x23_gqamodel_clevr/`, 215 pairs after excluding
  cyan, `--exclude-values`): accuracy c1 0.474 / c2 0.433, decoder attention
  target 31 vs other object 29 per token (CLEVR-trained 113 vs 0.1),
  Δ_ref ≈ 0 through block 10 — no transfer in either direction.
- **Status**: thresholds frozen with this entry; steps 1–4 run 2026-09-12
  (`x23_gqa_direct/`, `x23_gqa_spatial/` superseded for c2/c3 by
  `x23_gqa_spatial_v2/` — its H2 still running, `x23_gqa_spatial_v2_causal/`,
  `x23_clevrmodel_direct/`); populations = step 0c.

## Part 2 — Design-consistency findings (D1–D11)

**D1 [major, disclosure required] Performance model ≠ mechanistic model.** Tables use
the concat readout (92.4); ALL causal analyses (patching, path patching, ACDC,
interventions) use the GCA-decoder (91.0). Probe/RSA exist for both; patching cannot
trivially run on concat (patcher assumes the VQADecoder logit API). v2 must state per
figure which model it uses; the two share the identical frozen ViT + GCA trunk, so
the claim "the mechanism lives in the trunk" is defensible — but say it explicitly.

**D2 [major, actionable] Checkpoint policy: paper = final epoch; analyses use best.pt.**
Verified: best==last (ep15) for every tabled run EXCEPT `learned_text` (best ep2
0.4667 vs final 0.2456). E1b will therefore produce a learned_text breakdown at 46.7
that contradicts the ablation table's 24.6. Fix (queued as TODO): copy last.pt into a
new dir name (`..._lastep/`) and rerun eval_generalization there — filename is derived
from the parent dir, so this avoids overwriting the best-based JSON. Everything else
is unaffected.

**D3 [major, caveat or rerun] Perturbation C has a render-domain confound.** C pairs
= original CLEVR render (clean) vs our Blender re-render (corrupt); Δ therefore
includes renderer differences, not only the attribute swap. E7 already fixed this
pattern (base re-render); the same fix for C = re-render the clean scenes with the
same pipeline (script change is trivial; ~800 CPU renders). Until then, E9 must carry
the caveat; per-head *relative* maps remain informative.

**D4 [moderate] E9 compares different question populations.** A/B samples come from
unrestricted corruption sampling (any question containing the attribute word); C
samples only direct query_X questions with a unique target. The CA-share gradient may
partly reflect population, not pathway. Mitigation: rerun A/B in targeted/retrieval
mode (direct families) to match C — legacy methodology supports it (legacy-reference
§3); check whether main `activation_patching.py` exposes it; if not, port the flag.

**D5 [minor, note in captions] Backbone geometry differs**: DINOv2 336px/14 (24×24
tokens) vs others 224px/16 (14×14). Cross-backbone comparisons are qualitative;
within-backbone trends are the claim.

**D6 [minor] Sample sizes differ per analysis** (patching 50/cat, RSA/probe 72×500,
α-sweep 50, cogent-zs 30/cat, E7 100/attr, E5 stride-4 ≈37k). Fine as estimators —
every figure caption must state its n (aggregate_results records them).

**D7 [RESOLVED 2026-07-05] Two CLEVR data roots in code**: verified identical copies
(val-questions md5 match). P2 standardized all code on `CLEVR_ROOT` (release/public,
commit 4e41c8a).

**D8 [decision pending] Seed claim**: paper says 3 seeds; artifacts support s42 only
(s43 incomplete, s44 absent). Either E1c reruns or v2 states single-seed.

**D9 [consistency ✅] E7 render params match X7's C renders** (480×320@128, same
Blender + base scene); E7 families == attr_query_direct == X10's direct category.

**D10 [non-issue] Eval batch sizes differ** (train-loop val 512 vs eval_generalization
64) — no metric effect; final numbers cross-checked via checkpoint val_acc.

**D11 [minor, unexplained ~0.6pt] CoGenT protocol gap**: training-loop final ValA
0.9408 vs sample-efficiency "before" 0.94466. Likely best-vs-last or subset protocol;
one bounded check when CoGenT numbers are finalized for camera-ready.

## Action items extracted (also in JOURNAL TODO)
1. learned_text last-epoch per-qtype rerun via copied-dir trick (D2).
2. C-perturbation clean re-renders OR caveat sentence in A4 (D3) — recommend rerun (cheap, CPU).
3. Targeted-mode A/B patching to match C's population (D4) — check main script's support first.
4. Verify the two CLEVR roots are identical copies (D7).
5. Figure-caption rule: every figure states model variant (D1) + n (D6).


### X24. Selection → removal dependency, GQA marker injection, GQA attribute decomposition — pre-registered

Design document: `docs/mechanism_followup_design.md` (2026-09-13). Registered before the
first GPU job. Everything below is a prediction; results are reported for every cell
with n and image-bootstrap CI, whichever way they come out.

**Why.** Three gaps in the selection → removal account: (a) the attribute-direction
projections (`partA_attr_directions.json`, six dirs) have no gain control and their
own-vs-other statistic is an algebraic identity (−1.000 for binary attributes); (b) the
block-5 selection onset and the block-9 (DINOv2) / block-7 (SigLIP) removal onset give
an *order* but no *dependency*; (c) GQA has no causal evidence (transplant n=3, zero
qualifying heads) and no attribute decomposition.

**A. Repairs (CPU, no new data).**
- A1 `attr_direction_analysis(..., norm_std)`: unit-normalised object means; `gain` series
  saved. Output `attr_directions_v2/partA_attr_directions{,_normstd}.json` next to each
  existing run; the un-normalised variant must reproduce the 2026-08 JSON value-for-value.
  Prediction: after normalisation refer-it / refer-other / refer-neither stay
  indistinguishable through block 8 and the non-referent stays below the no-question
  baseline from block 9 (DINOv2) / 7 (SigLIP). Falsifier: normalised removal ≈ 0.
- A2 `*_other` keys marked deprecated; new statistics `own_vs_c0` (existing `*vs0_*`) and
  `qvu_*` = Δ(queried-attribute own-value) − Δ(unqueried-attribute own-value) of the same
  object. Prediction: `qvu_nonrefvs0_target` < 0 from the removal onset; `qvu_refvs0_target`
  ≈ 0 through block 8.
- A3 referring-word strata (`labels.json: referent_words.c1` → attribute family) with n per
  stratum, replacing the unimplemented n=223/59/42 sentence at X21.
- A4 fold-local PCA for the trained pooled probe; registered statistics written into the GQA
  result files; marker cross-fit by image for the two overlapping spatial images.

**B1. Dependency (GPU, CLEVR n2, 324 images × c0–c3).**
H-dep: the block ≥ 9 attribute removal from the non-referent depends on the selection
established by block 5.
- Intervention 1, `GCAWriteMasker`: mask GCA layers {1}, {1,3}, {1,3,5}, {1,3,5,7} on all
  patches (dose); {1,3,5} on target patches only / distractor only / background only
  (role split by object identity: the measured object is always the target, which is the
  referent under c1 and the non-referent under c2); {9,11} as late control.
- Intervention 2, `SubspaceProjector`: project out the marker direction (n2 c1−c2 target
  `raw_obj_mean`, 2-fold cross-fit by image) at block 7 or 8; controls: same-norm random
  unit vector × 5 seeds; same basis at block 10 (timing control).
- Statistics: A1/A2 normalised series, selection contrast (`delta.ref_imgdir`), accuracy,
  P(answer = non-referent value), margin.
- Pass: `nonrefvs0_*_own` (normalised) at blocks 9–11 shrinks ≥ 50 % toward 0 with CI
  excluding the unblocked value while the block 5–8 selection contrast vanishes and the
  random controls do not move. Falsifier (parallel paths): selection contrast gone,
  removal within the unblocked CI. Partial: report dose-response, no binary call.
- DINOv2 primary; SigLIP same grid; MAE baseline + {1,3,5} only (negative control).
- Caches: `n2_int_<spec>/` (aggregates only if disk is a concern).

**B2. GQA marker injection (GPU, small).**
H-inj: the GQA selection marker is causally effective, not merely correlated.
- 184 query-attr records, filtered to D's queried value ∈ answer vocabulary and ≠ T's value
  (n reported). Marker as `gqa_h4_marker`, 2-fold cross-fit by image. `ResidualAdder` at
  block 7 or 9, alpha ∈ {1, 2}; +marker on D's patches, −marker on T's patches.
- Controls: same-norm random direction × 5 seeds; marker on background patches; alpha = 0.
- Pass: ΔP(D's value) > 0 and Δaccuracy < 0 with CI excluding 0 and exceeding the random
  control. Falsifier: indistinguishable from random.
- Output `x23_gqa_direct_inject/`. Only causal claim permitted on GQA.

**B3. GQA attribute decomposition.**
- Object pool: GQA val scene-graph objects with a colour attribute, ≥ 4 patches, from images
  outside the 184 records; ≥ 30 objects per value, ≥ 8 values; one c0 extraction
  (`gqa_attr_pool/`, aggregates only). Directions as `attribute_directions`.
- Projection of the 184 records' T and D (c0/c1/c2) onto own colour value, normalised;
  statistics as A2, blocks 7–11. Prediction: CLEVR pattern (non-selective amplification,
  non-referent below baseline from block 9); "selection without removal" is reportable.
- Secondary: 36 records whose T has a second plain select→query question about another
  attribute (19 colour↔material); paired projection difference, CI only, no headline claim.

**Not run.** Spatial population expansion (35 q / 27 images stays "not decidable");
compositional factorial (future work); CLEVR SigLIP cumulative ablation curve.
**User decision.** GQA dev-split retrain (~36 h GPU; needs a `data.dev_fraction` option,
split is hard-coded today).

**Rules.** Controls fixed above; marker always cross-fit by image; claims graded
correlational (A, B3) vs causal (B1, B2); no hypothesis edited after results; design
changes get a new directory, old directories stay.


#### X24 Results (2026-09-13; all cells reported; JOURNAL 2026-09-13 has the full tables)

**A (CPU).** Un-normalised v2 reproduces the 2026-08 JSON value-for-value in all six runs.
Gain is real (DINOv2 target norm c1 vs c0 +16 % at b8–11; SigLIP c2 +38 % at b11).
Normalised: DINOv2 colour b8 +0.13 / +0.13 / +0.15 (non-selective), b11 referent +0.15 vs
non-referent −0.08 [−0.098, −0.071] — PASS; SigLIP non-referent −0.12 at b7 → −0.22 at
b11 — PASS; MAE +0.10 / +0.05, no removal (raw b7–9 dip was gain) — no signature;
shape / material / size queried: non-referent −0.42 / −0.20 / −0.12 at b11, referent
−0.11 / +0.02 / −0.02 (removal for all four attributes, referent enhancement only for
colour). `qvu_*` is dominated by a general shape drop (≈ −0.42 on both objects under any
question), so attribute specificity holds for the referential contrast, not for a single
object's projection. X21 referring-word sentence corrected (see note there).

**B2 (GPU, 1.5 min) `x23_gqa_direct_inject/`.** 125 clean items / 107 images, marker
2-fold cross-fit. +marker on the third object: ΔP(its value) b7 α1 +0.032 [+0.008,
+0.063], b7 α2 +0.040, b9 α1 +0.032, b9 α2 +0.064 [+0.024, +0.105]; random same-norm
directions ≈ 0; paired marker − random CI excludes 0 in all four cells (+0.029 / +0.035 /
+0.027 / +0.056) — registered criterion PASS. −marker on the referent: acc 1.0 → 0.688
(b9 α2); both: 0.536. **Background control not separated**: +marker on background b9
α2 +0.040 [+0.008, +0.076], acc 0.736. Claim allowed: the direction is causally
effective on natural images; not allowed: object-specific role marker.

**B1 (GPU) `n2_int_<spec>/`, figure `n2_int_summary/dependency_test.png`.** Non-referent
normalised queried-attribute projection at b11 (no intervention DINOv2 −0.084, SigLIP
−0.218). Marker direction projected out before the removal onset: DINOv2 @b8 −0.004
(gone), @b7 −0.028; SigLIP @b7 −0.043, @b8 −0.099; five random directions unchanged in
both — H-dep PASS by the registered criterion (≥ 50 % shrink, CI excludes the unblocked
value, random controls flat). GCA masking: {1,3,5} all patches DINOv2 −0.047 (44 %,
partial), SigLIP +0.084 (gone, reversed); {1,3,5,7} gone in both (acc 0.07 / 0.25);
{9,11} late control DINOv2 +0.058 (reversed, acc 0.50) vs SigLIP −0.169 (intact, acc
0.99). Role split {1,3,5}: DINOv2 redundant (target / distractor / bg alone all intact);
SigLIP target-only kills it (+0.003) with acc 0.997. Timing control @b10: DINOv2 −0.023,
acc 0.52; SigLIP −0.128, acc 0.997. MAE: no removal (+0.054), masking {1,3,5} leaves the
projection (+0.043) but drops acc 0.997 → 0.892.
Reading: same dependency on the marker direction, different implementation (DINOv2:
executed by GCA 9/11 reading the marker, early writes redundant; SigLIP: done by b7 via
target-patch writes, late blocks unnecessary). Two dissociations to report: SigLIP
target-only mask and @b10 projection remove the signature without harming accuracy.

**Not run / pending user.** B3 (GQA attribute pool), A4, C1, the attribute-restoration
intervention (Codex contribution 2), Sup-ViT as a fourth backbone.


### X25. Attribute restoration on the non-referent (CLEVR) and GQA attribute decomposition — pre-registered

Registered 2026-09-13 before any GPU run (user go: "編輯修正和 intervention 還有 GQA").
Follows X24: under a question the non-referent's alignment with its own queried-attribute
direction falls below the no-question baseline late in the trunk (DINOv2 blocks 9–11, SigLIP
7–11, MAE none). The paper outline (`PAPER_FRAMING_AND_ABSTRACT_CODEX_2026-09-13.md`,
contribution 2) asks for a functional test with a **predicted alternative answer**: put the
reduced component back and see whether the answer moves to the non-referent's value. R2 is
X24 B3 (GQA attribute pool), unchanged in design.

**R1 Restoration (CLEVR 2-object set, 324 images, question c1 about the target; the
distractor is the non-referent).** Code: `run_restoration` in
`scripts/analysis/patch_language_condition.py`, flag `--restore`, output
`<out-dir>/n2_restore/restoration.{json,png}` per backbone (DINOv2 root, `siglip/`, `mae/`).

- Direction: raw-space own-value direction of the distractor's colour,
  `V_raw[colour][Ad] = unit(mean raw_obj_mean of the 1-object targets with colour Ad − grand
  mean)` per block, from the independent 1-object set (`attribute_directions(space="raw")`).
- Dose: at block l the vector `alpha · |s_l| · V_raw[Ad]` is added (`ResidualAdder`) to every
  patch owned by the distractor, where `s_l` = mean over images of the raw own-value
  projection change of the target under c2 minus c0 (the non-referent's measured removal in
  raw units, negative where removal happens). alpha ∈ {1, 2, 4}; alpha = 1 restores the mean
  removal exactly. `s_l` is reported per backbone.
- Blocks: single-block 7, 8, 9, 10, 11 and joint {9, 10, 11} (each block with its own s_l).
- Variants: `own` (above); `other_value` (another colour B ∉ {A, Ad}, round-robin, same dose,
  distractor patches); `random` (same-norm random unit direction, 5 seeds, distractor
  patches); `own_on_referent` (V_raw[Ad] on the target's patches — dose-efficacy reference:
  the referent's colour is changed); `own_on_bg` (V_raw[Ad] on the background patches);
  `own_c2` (V_raw[Ad] on the distractor under c2, where it is the referent — sanity).
- Items: both colours in the vocabulary and different, clean c1 answer correct.
- Readouts on the clean items: P(answer = Ad), P(answer = A), accuracy, flip rate to Ad
  (argmax = Ad), and for `other_value` P(B) / flip rate to B; per-image bootstrap CIs;
  paired `own − random` per image.
- Predictions. **H-rest pass**: at alpha ≤ 2 on the blocks where removal was measured,
  `own` raises P(Ad) with CI > 0 and paired `own − random` CI > 0, and the flip rate to Ad
  is > 0 with CI. **Falsifier**: `own` ≈ `random` at every alpha ≤ 2 while `own_on_referent`
  flips at the same dose → the non-referent's late queried-attribute content does not govern
  the answer (removal is a signature, not a behavioural cause). Secondary reading:
  `other_value` moving P(B) as much as `own` moves P(Ad) → the decoder reads whatever
  colour content the non-referent's patches carry (generic), otherwise the effect is
  specific to the removed component. `own_on_bg` ≈ `own` would repeat the X24 B2 caveat
  (direction effective, not object-specific). MAE: no removal measured, `s_l` near zero →
  descriptive only.
- Cost: ≈ 324 × 10 direction sets × 6 block specs × 3 alphas forwards ≈ 5–8 min GPU per
  backbone. Nothing is re-extracted.

**R2 GQA attribute decomposition (X24 B3; GQA-trained SigLIP, `gqa_siglip_decoder1l_scratch_s42`).**
Code: `run_gqa_attr`, flag `--gqa-attr` with `--cache-dir` = `x23_gqa_direct`; output
`x24_gqa_attr/`.

- Pool: GQA val scene-graph objects with a colour attribute (`COLOR_WORDS`) or a material
  attribute (`MATERIAL_WORDS`), box passing the relaxed geometry (cover 0.3, ≥ 4 patches at
  grid 16, ≤ 50 % of the image), image not among the 148 direct or 27 spatial images, at most
  2 objects per image, per-value cap 60 (seeded). Directions kept for colour values with ≥ 30
  objects and material values with ≥ 15. Extraction: c0 only (no question), one object per
  record, aggregates only. Directions in `trunk.norm` space and raw space:
  `V[attr][value] = unit(mean − pool grand mean)`.
- Projection on the 184 direct records (c0 / c1 / c2 caches of `x23_gqa_direct`, unit-
  normalised object means): own-value projection of each object on the record's queried
  attribute. Referent series = T under c1 − c0 and D under c2 − c0; non-referent series = T
  under c2 − c0 and D under c1 − c0; unqueried series where the object has a second attribute
  in V (colour-queried records with a material value, and vice versa); qvu = queried change −
  unqueried change of the same object. Image-bootstrap CIs (`_group_boot`).
- Prediction (main): non-referent below zero with CI excluding 0 at blocks 9–11 (GQA onset is
  b7); referent ≥ 0. Falsifier: non-referent CI includes 0 at every block 7–11 → "selection
  reproduces on real images, removal does not" (reportable). Both are correlational.
- Secondary (same object, different question): records whose T has another natural
  `select|query` question on a different attribute (colour ↔ material) in the question pool;
  extract T under both questions (c1 and c4); paired difference of the own-colour projection
  (colour question − material question) and of the own-material projection (material − colour).
  n ≈ 20–30; CI reported, no conclusion sentence.

Registration rules as in X24: all cells reported; no design change after seeing results
(changes → new directory).


#### X25 Results (2026-09-13 23:28–23:55; all cells reported; JOURNAL 2026-09-13 has the tables)

**R1 Restoration — falsifier outcome in all three backbones.** Dose `s_l` (raw units, target under
c2 − c0 on its own colour direction), b7–11: DINOv2 +0.19 / +0.55 / −0.11 / −0.92 / −5.13 (removal
concentrated at b11; raw object norm ≈ 53); SigLIP −1.09 / −1.82 / −2.53 / −3.81 / −7.87 (norm ≈ 47);
MAE +0.98 / +0.72 / +0.65 / +0.39 / −0.16 (norm ≈ 134, no removal). Clean items 322 / 324 / 323.
Adding `alpha·|s_l|·V_raw[Ad]` to the non-referent's patches at any single block 7–11 or jointly at
9–11, alpha 1 / 2 / 4: ΔP(non-referent's colour) = 0.000 with CI width < 0.001, flip rate 0.000,
accuracy 1.000, in every cell of every backbone; another colour on the same patches: identical
zeros; five random directions: zeros; paired own − random: zeros. **The same vector on the
referent's patches is efficacious**: SigLIP b9 α4 ΔP(that colour) +0.54 [+0.50, +0.58], flip 0.55;
b10 α4 +0.58, flip 0.61; joint 9–11 α2 +0.69 [+0.64, +0.73], flip 0.71; DINOv2 only b8 α4 +0.15
[+0.12, +0.19], flip 0.14 (its decoder reads background tokens, X24); MAE ≈ 0 everywhere (dose ≈ 0).
Background: SigLIP joint 9–11 α4 +0.50 [+0.46, +0.55], α2 +0.10; DINOv2 ≤ +0.003. c2 sanity
accuracy 1.000 everywhere. **Registered reading**: H-rest fails and the falsifier holds for
SigLIP (dose efficacious on the referent, zero on the non-referent) — the non-referent's late
queried-attribute content at the magnitude actually removed does not govern the answer; the
removal is a signature of the mechanism, not the behavioural cause of ignoring the non-referent.
For DINOv2 the test is underpowered (the dose is barely efficacious anywhere); for MAE there is no
removal to restore. Open discrepancy to reconcile (not resolved here): the 2026-08 colour-swap
intervention (`intervention_results.json`, direction = means[B] − means[A], mean norm 9.7 at b11
for DINOv2) flipped 54 % of DINOv2 answers to B when added to the distractor at b11 with alpha 2,
a norm comparable to alpha 4 here (20.5); the direction type differs (difference of two colour
means vs one value's unit direction), and SigLIP showed 0 flips in both tests.

**R2 GQA attribute decomposition (`x24_gqa_attr/`).** Pool 1664 objects / 1180 images (173 images
excluded), 17 colour values (≥ 30 each) and 16 material values (≥ 15 each); extraction 1664 c0
passes, 0.13 GB. Projections on the 184 records (349 object-role rows / 145 images; unit-normalised
object means): non-referent queried-attribute change vs c0, b7–11: +0.006 / +0.010 / +0.003 /
−0.001 / **−0.044 [−0.055, −0.033]**; referent: +0.001 / +0.016 / +0.005 / +0.010 / **+0.028
[+0.019, +0.038]**. Colour-only (319 rows): non-referent b11 −0.046 [−0.058, −0.034], referent
+0.035 [+0.026, +0.045]. Material-only (30 rows / 11 images): both negative at b11 (−0.047 /
−0.029), too few images. Unqueried attribute (36 rows / 28 images): referent b11 +0.041 [+0.008,
+0.078], non-referent +0.005 [−0.036, +0.047]; queried − unqueried within the same object (32
rows): referent +0.007 [−0.025, +0.042], non-referent −0.002 [−0.043, +0.032] at b11 (wide).
**Registered criterion**: non-referent CI excludes 0 at b11 only, not at b9–10 → the removal
reproduces on real images but only at the last block (selection onset b7, X23 H1); referent
enhancement present from b8. Secondary same-object test (16 items: 10 colour→material, 6
material→colour; c1 re-extraction identical to the x23 cache): colour direction, colour asked −
material asked, b11 −0.061 [−0.095, −0.026], b10 +0.019 [+0.005, +0.036]; material direction,
material asked − colour asked, b10 +0.033 [+0.011, +0.053], b11 +0.004 [−0.052, +0.059]. CI
reported only, as registered; the b11 colour sign is opposite to "amplify the asked attribute".

**Not run / pending user.** A4 (fold-local PCA, registered statistics in GQA result files, spatial
marker cross-fit), C1 (GQA dev-split retrain), Sup-ViT, reconciling R1 with the 2026-08 colour
swap, MAE CPU checks.


**Clarification 2026-09-14 (scope of the R1 null; the paragraph above is kept as written).**
"Zero" means P(non-referent's colour) stayed at the 1e−5 level (DINOv2 b11 α4 own: 8.4e−6 →
1.5e−5; SigLIP 8.9e−6 → 9.3e−6), P(referent's colour) ≥ 0.9996, 0 / 322 flips. The registered
falsifier requires the efficacy control ("own_on_referent flips at the same dose"); only SigLIP
meets it (b9–11 α2 flip 0.71), so the sentence "removal is a signature, not the behavioural
cause" is licensed for SigLIP only. DINOv2: the add-only vector is ineffective on the referent
too (b11 α4 flip 0; b8 α4 0.14) → no power. MAE: no removal to restore. What is refuted is the
prediction for this direction, magnitude and location, not the grounding function.
Comparison with the 2026-08 colour swap (`intervention_results.json`), from the JSONs: the
old vector means[B] − means[A] has cos 0.75 with +V[B] and 0.75 with −V[A] (it removes A as
much as it adds B); in DINOv2 at b11 α2 it flips 54 % on the distractor but also 52 % on a
background subset of the target's size and 99 % on all background (random 0 %), i.e. an
any-token effect of the decoder's background readout, not evidence about the non-referent;
SigLIP: 0 flips in both tests. No contradiction; no extra GPU cell is needed. Direction
usability (1-object set, 2-fold held-out colour classification by argmax projection of
unit-normalised object means, chance 0.125), b7–11: DINOv2 0.51 / 0.51 / 0.43 / 0.40 / 0.41;
SigLIP 0.96 / 0.94 / 0.94 / 0.93 / 0.94; MAE 0.98 / 0.96 / 0.96 / 0.95 / 0.96 → MAE's
"no removal" is not a measurement floor; DINOv2's late colour directions are weak, a limit to
carry with its b9–11 removal. GQA checkpoint: best.pt = last.pt = epoch 17 (val 0.6358), so
val-based selection changed nothing; a dev-split retrain (C1) cannot undo the exploration of
val and is not recommended.


#### X24 A4 + Sup-ViT results (2026-09-14; user go "照你的建議做" on the ordered plan)

**A4-3 H4 marker cross-fit (validity fix, CPU).** `--gqa-h4-crossfit`: the two spatial records
whose images are among the 184 direct pairs (2400661: 1 pair, 2407031: 2 pairs) are projected
on a marker estimated without those pairs. target − third object, blocks 9–11: +1.74 [+0.91,
+2.56] (n = 35 questions, 27 images), previously +1.74 [+0.90, +2.56]. File
`x23_gqa_spatial_h2/h4_marker_crossfit.{json,png}`. Conclusion unchanged.

**A4-2 registered endpoints.** `x23_registered_endpoints.json` (`--gqa-summary`) collects
H1 (ref +1.174 [+0.872, +1.490]; contrast +2.21), H2 (ΔR² per block, k_condition = none),
H3 (selected − random drop 0.00 [0.00, 0.00] on 20 clean-correct spatial items; cumulative
curve), H4 (+1.74) and its cross-fit, B2, R2, each with its source path. Nothing recomputed.

**A4-1 fold-local PCA (X18 pooled probe, `linear_probe_single.py`).** PCA(50) is now fitted
inside each of the 5 folds. DINOv2, 1-object set (blocks 1/3/5/7/9/11), no-question colour:
0.972 / 0.984 / 0.980 / 0.984 / 0.956 / 0.914 (was 0.964 / 0.918 at 9 / 11); largest change
in any matched cell 0.036 (shape, block 1). No reported comparison changes. Outputs in
`outputs/analysis/linear_probe_v2/<model>/single_object/`; the two-object set has no
`scenes.json` and was never run through this script (the earlier "noca 0.356" note came from
t-SNE features), so it is not rerun. Other backbones (no-question colour, blocks 1/3/5/7/9/11, fold-local PCA): SigLIP 0.992 / 1.000 / 1.000 / 1.000 / 1.000 / 1.000; MAE 0.998 / 0.986 / 0.986 / 0.980 / 0.978 / 0.952; Sup-ViT 0.998 / 1.000 / 1.000 / 1.000 / 1.000 / 0.998 — all within 0.01 of the earlier values.

**Sup-ViT as fourth backbone (`clevr_sup_decoder1l_scratch_s42`, `vit_base_patch16_384`,
grid 24, `outputs/analysis/patch_language_condition/sup/`).** Hypothesis registered in the
2026-09-14 Q&A: if MAE's missing removal reflects its pretraining objective, the supervised
backbone (discriminative, like DINOv2 / SigLIP) should show removal and late referent
enhancement. Result (unit-normalised own-colour projection vs no question, 324 images):
non-referent b5–11 −0.133 / −0.072 / −0.129 / −0.094 / −0.116 / −0.108 / **−0.124 [−0.136,
−0.111]**; referent −0.116 / −0.061 / −0.061 / +0.052 / +0.038 / +0.061 / **+0.105 [+0.095,
+0.115]** — both objects fall at blocks 5–7, the referent recovers from block 8, the
non-referent stays below baseline. Selection contrast (ref_imgdir) +10.5 at b5 rising to
+36.3 at b11 (largest of the four backbones); gain c1/c0 at b8–11 1.03 / 1.07 / 1.00 / 0.97
(no gain); 2-object accuracy c1 / c2 1.000 / 1.000. Direction usability (held-out colour
classification, b7–11): 1.00 / 0.99 / 0.95 / 0.89 / 0.87. Summary across backbones at b11
(referent / non-referent): DINOv2 +0.15 / −0.08, SigLIP +0.20 / −0.22, Sup-ViT +0.11 / −0.12,
MAE +0.10 / +0.05. The three discriminatively pretrained backbones show the pattern, the
reconstruction-pretrained one does not — consistent with the pretraining-objective reading;
one model per objective, so it is a consistency, not a test of the objective. Sup-ViT differs
from the others in timing: the removal is present from block 5 (SigLIP 7, DINOv2 11 in raw
units) and the referent is initially reduced too.

**A4-1 correction (2026-09-14, later the same day).** The statement above that the two-object
set "was never run through this script" is wrong: the RESULTS §16 probe table (trained DINOv2
model, `outputs/analysis/tsne/object_count/n{1,2}/linear_probe_results.json`) was produced by
`linear_probe_single.py --features-dir` from cached features with the global PCA, so it carried
the same leakage. Rerun from the same cached features with fold-local PCA (CPU), new outputs
`outputs/analysis/linear_probe_v2/object_count/n{1,2}/`. Two-object set, block 11, target
colour, old → new: no question 0.356 → 0.365; "What color is the object?" 0.458 → 0.496; "What
color is the cube?" 0.331 → 0.323; shape questions 0.317 / 0.294 → 0.310 / 0.285. Target shape:
no question 0.667 → 0.656, "What color is the object?" 0.619 → 0.627. Largest change in any of
the 60 cells per set: 0.037 (n2), 0.014 (n1). The reported comparison (aligned colour question
raises two-object colour readout above the no-question value; shape questions do not) is
unchanged; the gap widens from +0.10 to +0.13. The X18 raw-backbone pooled probe
(`raw_backbone_probe.py`, 0.517 / 0.812 / 0.850 / 0.912) uses no PCA and is unaffected.

#### Corrections after the Codex response of 2026-09-14 (framing document, "Codex response to Claude's six pending items")

**H3 reporting status.** The sentence in "A4-2 registered endpoints" above, "H3 (selected −
random drop 0.00 [0.00, 0.00] on 20 clean-correct spatial items)", misstates the outcome. The
X23 record (step 1–4 results) is correct: the registered selection rule selected **0 heads**
(`x23_gqa_spatial_v2_causal/head_ablation.json`, `selection.n_meeting_rule = 0`; median |Δ| of
the candidate statistics 0.0009 / 0.0011), so the registered test is **not estimable: no heads
qualified**. The 0.00 [0.00, 0.00] is the accuracy drop of an empty head set and carries no
information. The cumulative ablation (`cum_1 … cum_16`: 0.00 / 0.05 / 0.05 / 0.15 / 0.15 drop,
CIs include 0) is a separate exploratory curve, not the registered test. The endpoint JSON is an
output file and is left as written; this note is the correction.

**Rounding of the two-object probe difference.** The reported maximum cell change 0.037 is the
full-precision value 0.0375 (0.45833 → 0.49583, n = 480, one item = 1/480 = 0.0021); the rounded
endpoints 0.458 → 0.496 differ by 0.038. Both are the same 18-item change.

**GQA analyses: cohort and correctness filter (one row per analysis).**

| analysis | cohort | correctness filter | c1 accuracy on the cohort | source |
|---|---|---|---|---|
| H1 direct (referent own-V projection) | 184 questions / 148 images, step 0c S_eligible | none (observational) | 0.679 (c2 0.663) | `x23_gqa_direct/x23_results.json` |
| H2 spatial (anchor ridge probe ΔR²) | 35 q / 27 images, S_eligible | none | — | `x23_gqa_spatial_h2/x23_results.json` |
| H4 spatial marker, and its cross-fit | 35 q / 27 images; marker from the 184 direct pairs | none | — | same; `h4_marker_crossfit.json` |
| H3 head ablation | 20 of the 35 spatial questions | clean-correct (S_correct) | 1.0 by construction | `x23_gqa_spatial_v2_causal/head_ablation.json` |
| B2 marker injection | 125 items / 107 images from the 184 direct records | clean-correct, plus both values in the answer vocabulary and different | 1.0 by construction | `x23_gqa_direct_inject/marker_injection.json` |
| R2 attribute projection (queried attribute) | 349 objects / 145 images from the 184 records (colour 136 images, material 11) | none (projection only) | — | `x24_gqa_attr/attr_decomposition.json` |
| R2 same-object secondary test | 16 items | none | — | `x24_gqa_attr/same_object/` |

So "GQA analysis uses only clean-correct items" holds for the two interventions (H3, B2) only;
the observational analyses (H1, H2, H4, R2) run on the eligible population, whose c1 accuracy
is 0.679 on the direct records. Registry line for X23 "Analysis populations: S_eligible for
observational analyses, S_correct for interventions" already states this rule.

### X26. Unified role contrasts — object-level attribute alignment by question condition (from the X21 caches)

- **Origin**: Codex plan "給 Claude:統一場景幾何與物件屬性分析" (2026-09-16), §2–§4; user
  instruction to execute it reusing the existing code. Setup manifest, condition-name mapping
  and comparability table: `docs/unified_analysis_manifest.md`.
- **Status of the hypotheses**: the four conditions (No question / Generic attribute question /
  Question about A / Question about B = `c0` / `c3` / `c1` / `c2`) were already extracted in
  X21 for the same 324 pairs, so H1–H3 below are **re-analyses of existing observations on the
  registered pipeline**, not pre-registered predictions; no new forward pass was run.
- **Design**: script `scripts/analysis/patch_language_condition.py --role-contrasts`
  (functions `role_contrasts`, `attribute_switch`; CPU, from `n{1,2}/feats_c*.npz`). Object
  IDs fixed (A = `target` slot, B = `distractors[0]`); attribute directions from the 1-object
  renders (`attribute_directions`, unchanged); per object, attribute and block the projection
  on the object's own value direction under each condition, unit-normalised object means
  (primary, `_normstd`) and raw. Five contrasts: about it − no question; about the other
  object − no question; generic − no question; about the other object − generic; about it −
  generic. Bootstrap unit = image, 2000 resamples, seed 42, 95 % percentile interval;
  `both` = per-image mean of A and B. Description-fixed queried-attribute switch between the
  colour / shape / material runs on the pairs whose c1 and c2 referring words coincide
  (colour↔shape 101, colour↔material 282, shape↔material 59), with a differing-value stratum.
  Outputs `outputs/analysis/patch_language_condition/{,siglip/,sup/,mae/,shape/}unified_role_contrasts/`.
- **Results (2026-09-16; unit-normalised own-direction projection, mean of A and B, block
  11, n = 324 images; intervals in the JSON files)**

  Queried attribute (colour runs; DINOv2 shape run on its own shape direction):

  | run | about it − none | about other − none | generic − none | about other − generic | about it − generic |
  |---|---|---|---|---|---|
  | DINOv2 colour | +0.156 | −0.077 | +0.151 | **−0.227** [−0.236, −0.219] | +0.005 [+0.001, +0.009] |
  | SigLIP colour | +0.194 | −0.217 | +0.216 | **−0.433** [−0.440, −0.426] | −0.022 [−0.026, −0.018] |
  | Sup-ViT colour | +0.106 | −0.121 | +0.047 | **−0.168** [−0.174, −0.162] | +0.059 [+0.057, +0.062] |
  | MAE colour | +0.093 | +0.050 | +0.099 | −0.049 [−0.051, −0.046] | −0.006 [−0.008, −0.004] |
  | DINOv2 shape | −0.106 | −0.416 | −0.172 | **−0.244** [−0.262, −0.226] | +0.066 [+0.061, +0.072] |

  Unqueried attribute (shape direction in the colour runs), block 11: all three question
  conditions move the shape projection together (DINOv2 −0.42 / −0.44 / −0.42, SigLIP
  −0.30 / −0.31 / −0.30, Sup-ViT −0.31 / −0.32 / −0.32, MAE −0.03 / −0.01 / −0.04); about
  other − generic = −0.015 / −0.011 / −0.004 / +0.031, about it − generic = −0.001 / +0.001 /
  +0.008 / +0.009. In the shape run the unqueried colour direction behaves the same way (about
  other − generic −0.027, about it − generic +0.018).

  Reading, per Codex's hypothesis list: (H1) the generic question changes both objects' queried
  attribute alignment by about as much as a referring question changes the referent's (DINOv2
  +0.151 vs +0.156; SigLIP +0.216 vs +0.194) — the question adds attribute information without
  a role distinction; (H2) relative to the generic question the referent gains little or
  nothing (+0.005 / −0.022 / +0.059) and the non-referent loses a lot (−0.227 / −0.433 / −0.168),
  so the role effect is "mainly a reduction of the non-referent" in the three discriminatively
  pretrained backbones; MAE shows the same sign at a fifth of the size (−0.049). In the shape run
  every question lowers the shape projection below the no-question level, yet the same
  contrasts hold (about other − generic −0.244, about it − generic +0.066); (H3) on the
  unqueried attribute the role contrasts are an order of magnitude smaller (|Δ| ≤ 0.03), so the
  role effect is specific to the queried attribute. Per-object curves (A and B separately) agree
  with the pooled ones in every run. Timing: the about-other − generic gap opens at block 9 in
  DINOv2 (−0.10 → −0.13 → −0.23), at block 7–9 in SigLIP and Sup-ViT.

  Queried-attribute switch, image + referent + description fixed (DINOv2, object A, block 11,
  ask attribute *d* − ask the other attribute, projection on A's own *d* direction):

  | pair (n same description) | direction | referent | generic | non-referent | non-referent, differing values (n) |
  |---|---|---|---|---|---|
  | colour vs shape (101) | colour | +0.227 | +0.223 | +0.034 | same (colours always differ) |
  | colour vs material (282) | colour | +0.195 | +0.202 | +0.018 | same |
  | colour vs material (282) | material | +0.166 | +0.184 | +0.006 | **−0.127** [−0.140, −0.114] (141) |
  | shape vs material (59) | material | +0.151 | +0.172 | +0.016 | −0.108 [−0.144, −0.070] (31) |
  | colour vs shape (101) / shape vs material (59) | shape | +0.315 / +0.308 | +0.269 / +0.250 | +0.259 / +0.268 | **not estimable: 0 differing-shape pairs** |

  With everything but the queried word held fixed, asking about *d* raises the referent's and
  the generic-condition alignment on *d* by 0.15–0.23 at block 11 and leaves the non-referent
  near zero (colour) or below zero (material, differing values). The shape direction cannot be
  assessed: a description that avoids shape words exists only when A and B share the shape, so
  on that subset the non-referent's "own" shape direction is also the referent's.
- **Cautions**: (i) the description-fixed subsets are, by construction, pairs whose other
  attributes coincide (see the shape row); every switch contrast on a shared value is
  confounded and is reported only in the JSON; (ii) attribute directions are estimated on the
  1-object partner of the same pairs (see manifest §3) — the shared-instance term cancels in
  every contrast but not in the levels; pair-wise cross-fitting not run; (iii) no equivalence
  margin was pre-specified, so "the generic question does not reduce alignment" is not claimed;
  the estimates are: generic − none is positive in all colour runs and negative in the shape run;
  (iv) "generic question" here is the one string `What {attribute} is the object?`, a
  non-referring question on a two-object scene — a competing reading (the model picks a default
  object) is not excluded by these numbers; (v) H1–H3 are replications on a pipeline whose
  results were known; the intervals are within-sample.
- **Not run (plan items left open)**: scene-level variance share / t-SNE restricted to the 324
  pairs; pair-wise cross-fit directions; a pooled scene vector from the same forward pass (the
  X21 cache holds object + 64 background patches only).

#### X26 corrections and v2 results (2026-09-16, after the Codex review `writing/X26_REVIEW_AND_FOLLOWUP_CODEX_2026-09-16.md`)

Codex re-derived the five "about the other object − generic" contrasts and intervals from the
caches and confirmed them; the following interpretations of the entry above are withdrawn or
narrowed. The old text stays as written.

1. **"Shared-instance term cancels in every contrast" — withdrawn.** There is no such
   guarantee (a contrast is `(unit(x_q) − unit(x_base)) · v`; an in-sample `v` can correlate
   with the difference). The effect was untested. Sensitivity check now run: pair-grouped
   5-fold cross-fit directions (seed 42; `unified_role_contrasts_v2/split.json`; every
   training fold keeps ≥ 30 one-object examples per colour value, ≥ 82 per shape value).
   Held-out results, block 11, mean of A and B, unit-normalised, old in-sample → cross-fit:
   DINOv2 about other − generic −0.227 → **−0.216** [−0.224, −0.208], about other − none
   −0.077 → −0.055, about it − generic +0.005 → +0.005; SigLIP −0.433 → −0.429; Sup-ViT
   −0.168 → −0.163; MAE −0.049 → −0.049; DINOv2 shape run −0.244 → −0.243. No contrast
   moves by more than 0.022 and no sign changes. Intervals are conditional on the fixed
   directions and the single trained model (not on direction or training uncertainty).
2. **"Generic question ≈ question about it" — narrowed.** About it − generic is +0.005
   [+0.001, +0.009] (DINOv2), −0.022 [−0.026, −0.018] (SigLIP), **+0.059** [+0.055, +0.060]
   (Sup-ViT): the magnitudes are close in DINOv2 / SigLIP, and Sup-ViT has an additional
   referent increase; no equivalence test was run. The similar mean response of A and B under
   the generic question does not exclude that the model selects a different default object per
   image; nothing here shows both objects being bound at once or no selection.
3. **MAE.** MAE has a smaller role contrast (about other − generic −0.049; self − other
   +0.043), not none; what it lacks is the fall below the no-question baseline (about other −
   none +0.051). Its unqueried-shape about other − generic is +0.030 [+0.026, +0.034], so the
   colour/shape ratio is not an order of magnitude there.
4. **H3 is preferential, not exclusive.** Unqueried-attribute role contrasts have intervals
   excluding zero (DINOv2 shape about other − generic −0.014 [−0.017, −0.011]; self − other on
   shape is −0.065 / −0.095 at blocks 5 / 7 before returning to +0.013 at 11). Supported
   statement: at block 11 the queried-attribute role contrast is larger (|Δ| 0.16–0.43 vs
   ≤ 0.03 for the three discriminative backbones); the block-11 statement does not cover the
   middle blocks. "The question adds attribute information" is replaced by "the question
   increases alignment with the measured attribute-value direction".
5. **Shape switch.** The full same-description subsets are computable and reported (colour
   vs shape, shape direction, n = 101: referent +0.314, generic +0.267, non-referent +0.259);
   only the **differing-shape stratum** is empty (n = 0), so the shape direction cannot answer
   how two different shape values compete. Shared-value and differing-value strata are now
   reported side by side, not deleted.
6. Manifest: the workshop t-SNE / RSA pool is synthetic CLEVR (3–5 objects), not natural
   scenes; the switch figure now shows the mean of A and B with the differing-value stratum on
   its own row; the first figure showed object A only.

**v2 additions (`--role-contrasts --role-dir unified_role_contrasts_v2 --crossfit-folds 5`,
in-sample and cross-fit files side by side; DINOv2 also `--attribute-switch`).**
Direct self − other (about it − about the other object, per image, mean of A and B,
cross-fit, queried attribute, block 11): DINOv2 +0.221 [+0.214, +0.228], SigLIP +0.407,
Sup-ViT +0.221, MAE +0.043, DINOv2 shape +0.309; it opens at block 9 (DINOv2), 7 (SigLIP,
Sup-ViT). Switch difference-in-differences `dd` = (self − other when asking d) − (self −
other when asking the other attribute), cross-fit, block 11: colour (vs shape, n 101) +0.184
[+0.171, +0.196]; colour (vs material, n 282) +0.167; material, differing stratum (n 141)
+0.295 [+0.279, +0.309] with non-referent −0.120; material (vs shape, differing n 31) +0.239;
shape (shared-value pairs only) +0.055 / +0.041. Cross-run checks recorded in the JSON
(`checks`): same pair list, same object attributes and positions, identical owner masks,
same checkpoint for all three run pairs. Every estimator's valid pair indices are stored
(`valid_pair_index`, `pair_index_same_description`, `pair_index_differing`). These are
exploratory quantifications, not pre-registered hypotheses.

**Scene-level analyses on the same 324 pairs (P3).** Reconciliation
(`check_scene_questions`): `object_count_v2/n2/attrs.json` equals the dataset metadata; for
all 324 pairs the scene-cache rows carry the same objects and filenames, and the scene
`ca_color_refer` question string equals the object-level c1 (324/324); same checkpoint. No
Question-about-B scene cache exists (not fabricated). `variance_partitioning.py
--subset-labels --skip-raw` → `outputs/analysis/variance_partitioning_324/results.json`
(figures not regenerated: the plot functions expect the raw-backbone keys). Unique variance
share of the scene vector, block 11, 480 → 324: no question A colour 0.032 → 0.030, B
colour 0.035 → 0.030; generic colour question 0.273 → 0.300 / 0.287 → 0.319; question about
A 0.562 → 0.546 / 0.005 → 0.006; shape question about A 0.554 → 0.556 / 0.005 → 0.005.
Descriptive within-sample shares; the subset changes nothing by more than 0.03. t-SNE on
the 324 rows (perplexity 30, seed 42, each panel fitted separately, no cross-panel
coordinate comparison): `outputs/analysis/tsne/object_count_v3/n2_324/` (no question /
generic / question about A grids) and `composite_block11_conditions.png` (1 object, 2 objects
no question, generic, question about A); the full-480 composite with the same four panels is
`outputs/analysis/tsne/object_count_v3/composite_block11_conditions.png`. One-object scenes
are the baseline in which the generic question is itself uniquely referring.

**Manuscript figure candidate** `unified_role_contrasts_v2/role_contrasts_paper.png`
(`--role-paper`, cross-fit JSONs): top row per model, the three question conditions − no
question; bottom, block-11 about it − generic and about other − generic with intervals.
Sentence adopted from the review: in the two-object colour analysis, referring questions
produce a larger reduction in the non-referent's colour alignment than any additional
increase in the referent's alignment, relative to a generic colour question; strongest in
DINOv2, SigLIP and the supervised ViT; MAE shows a smaller role-dependent contrast without a
decrease below the no-question baseline. This is a representational dependence on the
referring expression; it is not evidence of information removal, of behavioural necessity,
of completed retrieval, or of two serial stages.

#### X26 v3 (2026-09-17, after the Codex v2 review `writing/X26_V2_REVIEW_CODEX_2026-09-17.md`; main result accepted, scope and P3 fixes)

Codex independently re-derived the five cross-fit "about the other object − generic" contrasts,
the self − other contrasts, the switch difference-in-differences and the block-11 n2 unique
shares; all match. The following statements above are narrowed; numbers below were re-verified
here (`check_scope.py` / `check_v3.py` in the job directory, from the JSON files).

1. **Scope of "no contrast moves by more than 0.022, no sign change".** True for the **six
   block-11, queried-attribute, A/B-mean contrasts of the five runs** (max change 0.02161). Over
   all objects / attributes / blocks the largest change is 0.0457 (DINOv2, object A, own colour,
   about the other object − no question, block 11: −0.084 → −0.039) and 19 near-zero values change
   sign (e.g. SigLIP generic − none at block 5: −0.00007 → +0.0023). The DINOv2 non-referent −
   no-question value is −0.055 after cross-fitting (28 % smaller in magnitude than −0.077); the
   cross-fit numbers are the ones to quote.
2. **Sup-ViT about it − generic (cross-fit) is +0.058** [+0.055, +0.060]; the +0.059 in the v2
   section mixed in the in-sample mean.
3. **Scene-level shares on the 324 pairs: "changes nothing by more than 0.03" withdrawn.**
   Block-11 shares used in the text, 480 → 324: no question A 0.032 → 0.030, B 0.035 → 0.030;
   generic colour question A 0.273 → 0.300, B 0.287 → **0.319** (difference 0.0316); question
   about A: A 0.562 → 0.546, B 0.005 → 0.006; shape question about A 0.554 → 0.556 / 0.005 →
   0.005. Largest difference over all conditions, layers and factors: 0.0578 (1-object shape
   question, block 1, colour).
4. **Interaction field on the unbalanced subset (P3 code fix).** `variance_partitioning.py`
   computed `r2_cells − Σ marginal` as "interaction" for n1; on the 324 subset the factors are
   not orthogonal, so that number (0.155 at block 11, no question) is not an interaction
   variance. New field `cells_beyond_additive = r2_cells − r2_full` (0.169; in-sample incremental
   fit of the full cell model beyond additive main effects, not evidence of hierarchy) and the
   `interaction` field is written only for the balanced 480 set. New output
   `outputs/analysis/variance_partitioning_324_v2/` (unique shares identical to the 324 v1 file
   to 1e-16; the v1 `interaction` field is not to be cited). `compute_own_axis` now takes the same
   subset. Plotting: the three raw-backbone figures are skipped with a message under
   `--skip-raw`; `queried_share_by_layer.png` is produced. Status for the 324 v1 directory:
   numbers computed, plotting step failed (`KeyError: raw_DINOv2_n1`).
5. **Provenance / assertions.** `pair_folds` builds the fold map from sorted unique pair ids
   and asserts uniqueness and order (reproduces `split.json` exactly); `attribute_switch`
   requires a recorded checkpoint in both runs and asserts, per retained image, that the c1/c2/c3
   strings differ only in the queried word. `--role-provenance` writes
   `unified_role_contrasts_v2/provenance.json` per run: command, code commit, checkpoint path and
   **sha256** (DINOv2 22f4e90f…, SigLIP eee4673c…, Sup-ViT 6a01ed64…, MAE 57b008ea…), cache
   paths / mtimes / shapes, the 324 pair ids, split file, bootstrap spec, and the verbatim
   question check (colour vs shape 101, colour vs material 282: pass).
   `variance_partitioning_324_v2/provenance.json` records command, commit, caches, checkpoint
   from the cache logs, pair ids, layers and the definition of every statistic.
6. **Figures.** `role_contrasts_paper.{png,pdf}` (cross-fit JSONs; caption must state 5-fold
   pair-grouped cross-fit, image bootstrap 2000 / seed 42, conditional intervals, and that the
   top-row y-axes are scaled per model). Composite v2
   `outputs/analysis/tsne/object_count_v3/{n2_324/,}composite_block11_conditions_v2.png`: panels
   "No question / Generic colour question / Question about A" (object count on its own line, no
   question strings), attribute legend, n per panel; real question examples belong in the
   caption ("What color is the cube?", "What color is the large object?"); each panel is an
   independent t-SNE. Figures 1(b), 3 and A1 in `unified_role_contrasts_v2/figures/`.
7. X18 stimulus note: dated correction inserted in the X18 entry above.

Manuscript sentence adopted from the review: language conditioning changes object
representations according to both the referring expression and the requested attribute; in the
two-object colour analysis the contrast with a generic question is dominated by lower colour
alignment in the non-referent, and this remains after pair-grouped cross-fitting of the attribute
directions. Model differences and the MAE limitation are reported alongside; no claim of
information deletion, behavioural necessity, completed retrieval, or fixed serial stages.
