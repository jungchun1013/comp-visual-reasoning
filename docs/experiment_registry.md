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
- **Status**: ✅ concat main model (`concat_decoder_1l/`), siglip GCA-decoder, nogate (byproduct, keep unused); dinov2 GCA-decoder ⏳ (E4, queued — completes the mechanistic-model pair).

### X11. E7 add-object hallucination (v2 A1.3 core)
- **Motivation**: show the substrate bottleneck is fixation, not encoding.
- **Hypothesis**: adding a mostly-matching distractor with a bait value on the queried attribute pulls answers toward the bait iff binding is weak.
- **Design**: 100 pairs × 4 attrs; distractor flips exactly one described attr; answer invariance verified by program execution; base re-render controls render domain. Families landed exactly on attr_query_direct [86,87,88,89] ✅ (consistent with X10's "direct" category).
- **Status**: renders ✅; model eval ⏳ (queued, concat main model).

### X12. E8 raw-backbone per-object probe (v2 A1.2)
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
