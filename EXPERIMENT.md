# EXPERIMENT
> [!NOTE] Living document. Always reflects current state. All changes tracked in JOURNAL.md.
> [!NOTE] Initialized 2026-07-05 from the user's v2 paper outline (2-stage reframe of the accepted ICML 2026 CompLearning workshop paper). Pending user confirmation.

## Research Question
Can language conditioning elicit compositional visual reasoning from frozen pretrained
vision foundation models (VFMs) — and through what mechanism?

## Hypothesis
Pretrained VFMs already encode a structured, compositional representation space (the
substrate); their bottleneck is *fixation on the described object* when multiple objects
are present. Injecting the question through gated cross-attention achieves **grounding**
(the whole language-conditioning mechanism) via routing and refocus, unfolding as a
2-stage process: **Binding** (bind the described attribute to the right object) →
**Retrieval** (retrieve the queried attribute). Perturbation localization: text-side
perturbations (A) should affect only cross-attention; image-side perturbations (C)
should affect only self-attention.

## Objective
A release-quality repo where every paper claim maps to a traceable artifact
(`docs/paper_artifacts.md`), plus the v2 experiment additions: substrate probing on raw
backbones (E8), add-object hallucination (E7), A/B/C × {CA, SA} localization contrast
(E9), second-backbone mechanistic replication (E3), and an agent-driven failure-mode
analysis explaining why yes/no questions are worst (E5).

## Methods

### Model
- **CrossAttnViT (main model)** (src/model/, forked from legacy SteerViT)
  - Architecture: frozen pretrained ViT-B/14-or-16 + gated cross-attention (GCA) inserted at layers [1,3,5,7,9,11]; frozen RoBERTa-large text encoder + trainable connector; readout head.
  - Readouts: `concat_self_attention` 1-layer decoder (**paper performance tables**) vs `VQADecoder` cross-attention 1-layer decoder (**mechanistic-analysis model**) vs classifier.
  - Input resolution: 336 (DINOv2) / backbone-native for others.
  - Pretraining strategies compared: DINOv2 (self-distillation), SigLIP (image-text), supervised (augreg_in21k), MAE (pixel reconstruction).
  - Gate: tanh-gated CA, a *design choice* (stability + analyzable language-influence handle); ungated `nogate` variant exists but is not a paper claim.

### Data
- Dataset: CLEVR v1.0 (`/home/jungchun/data/clevr/CLEVR_v1.0`), 700K train / 150K val, 28 answers.
- CLEVR CoGenT (condition A train, valA/valB) for compositional generalization.
- Transfer: CLEVR-Humans, CLEVR-Math, CLOSURE.
- Perturbations: A = swap described attribute in text; B = swap queried attribute in text; C = swap queried attribute in image (Blender re-renders, `pairs.json`).

### Tasks
1. CLEVR VQA (overall + per-question-type: QryAttr, EqAttr, Exist, Count, CmpInt).
2. Mechanistic: activation/path patching (headwise, denoising), binding interchange, conditional RSA on 3 attr_query categories (direct [86,87,88,89], same [53,59,55,57,61,60], spatial [76,74,75,77,80,81]), linear probes per block.
3. Substrate (v2): raw-backbone probing (E8), add-object hallucination (E7).

## Experiment Setup
- Hardware: 2× RTX A6000 (GPU 0 flaky — prefer GPU 1); NFS storage.
- Framework: PyTorch + Hydra; interpreter `SteerViT-legacy/.venv-aspen/bin/python` (no venv in main/).
- Random seed: 42 (paper claims 42/43/44 — s43 incomplete, s44 missing; open decision E1c).
- Frozen: ViT + RoBERTa; trainable: GCA + connector + readout.

### Inference Settings
Deterministic greedy decoding; eval at final epoch only (project policy).

### Evaluation Protocol
Full CLEVR val (150K); accuracy = exact answer match. Training logs report per-epoch
`Val acc`; paper numbers use final-epoch accuracy. Per-type breakdown via
`scripts/eval_generalization.py`. Never overwrite existing outputs — new runs get new names.

## Dependent Variables (Metrics)
- Primary: val accuracy (overall, per question type).
- Mechanistic: patching effect per head (denoising recovery), RSA correlation (Spearman, conditional chains), probe accuracy per block, intervention deltas.
- Independent variables: backbone pretraining strategy; readout; perturbation type (A/B/C); attention type (CA vs SA); layer/head.
- Comparison: 4 backbones × {concat decoder, GCA-decoder, cls, nogate} + ablations (−CA, scratch ViT, learned text) + baselines (MoT).

## Ablation Study
- −CA (`concat_decoder1l_nogca`): 49.4 — language injection necessary.
- Scratch ViT (`gca_scratch`): 52.8 — pretrained substrate necessary.
- Learned text (`learned_text_decoder1l`): 24.6 — pretrained text encoder necessary.
- Classifier readout (`cls`): 90.1 — decoder helps but is not the story.
- Gate (`nogate`): design-choice ablation, documented but unreported as claim.

## Current Status
R0/R1 complete (provenance + accuracy matrix in `docs/paper_artifacts.md`); E1 per-qtype
eval batch running; DINOv2 92.4 and CoGenT 92.4/88.0 provenance still unresolved
(legacy hunt in progress); E3–E10 pending.
