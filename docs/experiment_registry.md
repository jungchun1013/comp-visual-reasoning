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
- **Inconsistency**: D2 — learned_text degrades after ep2 (best 46.7 → final 24.6); paper reports final. E1b's best.pt-based breakdown for this run will NOT match the table (fix queued, see D2).

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
  unasked attribute. Reading: the model does both things — the asked
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
- **Status**: ✅ A, B, C done 2026-08-27; RSA position control closed
  2026-08-28; readout check (D), attribute directions (E) and SigLIP
  replication (F) done 2026-08-29; G and H launched 2026-08-29. Site
  edits await the user.

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
