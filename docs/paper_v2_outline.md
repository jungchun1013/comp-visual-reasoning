# v2 paper outline — claims, evidence, wording (Fable pre-registration, 2026-07-05)

Reframe of the accepted workshop paper into the v2 architecture (2-stage naming).
Each claim states: the exact wording to defend, the artifact(s) backing it, and its
status — ✅ supported now / ⏳ artifact incoming (in tonight's GPU queue) / ⚠ wording
constraint discovered during R0/E9. Numbers cite `docs/paper_artifacts.md` (P) and
`RESULTS.md` (R).

Terminology (fixed): **Grounding** = the whole language-conditioning mechanism,
achieved via routing and refocus. Stage-wise: **Binding → Retrieval**. Never use
"object grounding" as a stage name; "answer matching" → Retrieval.

---

## A1 — Substrate: VFMs already encode structured, compositional representations

**Claim A1.1** (✅): Frozen VFM features linearly encode object attributes without any
language conditioning — single-object probes/t-SNE on raw DINOv2 separate
color/shape/material/size. Evidence: `outputs/analysis/single_objects` +
`linear_probe/single_object` + `dino_attribute_tsne` outputs.

**Claim A1.2** (⏳ E8): The encoding persists in multi-object scenes at the
*per-object* level (the substrate is compositional, not scene-global). Evidence
incoming: raw-backbone multi-object probe (E8, `--raw-backbone` mode). Existing
trained-model multi-object probe (`linear_probe/multi_object`) serves as the upper
reference.

**Claim A1.3 — the bottleneck** (⏳ E7 eval): What pretrained VFMs *lack* is not
attribute encoding but **fixation on the described object when multiple objects
compete**. Operationalization: add-object hallucination — adding a distractor that
matches most described attributes but differs on the queried one (bait) leaves the
question's answer invariant; a fixating model is unaffected, a bag-of-features
binder retrieves the bait. Metrics: acc_base vs acc_added, hallucination_rate,
bait_share_of_errors (4 attributes × 100 pairs, rendered ✅; eval ⏳).
Wording note: report the *bait share of errors* as the headline (isolates
hallucination from generic distribution shift; acc_base controls render-domain shift).

## A2 — Grounding via language conditioning (routing + refocus)

**Claim A2.1** (✅): Injecting the query through gated CA turns the frozen substrate
into a compositional reasoner: 92.4 on CLEVR vs 49.4 without CA, 52.8 with a scratch
ViT, 24.6 with learned text (P§2–3 — all single-provenance, concat s42).

**Claim A2.2** (✅ relabel only, E10): The conditioning unfolds in 2 stages —
Binding (described attributes bound mid-network, CA heads L5H0/L7H9/L7H11/L7H3)
→ Retrieval (queried attribute read out late, SA block 11). Evidence: existing
RSA/t-SNE/patching figures relabeled with 2-stage names into NEW dirs (E10);
conditional RSA on the main model exists (`conditional_rsa/concat_decoder_1l`).

**Claim A2.3 — gate as design choice** (✅, demoted per user decision): the tanh gate
is the analyzability handle, not a performance claim. Graded control: activation
interpolation degrades monotonically (47/50 → 37/50 as α 0→1); gate-mediated
interventions move Binding→Retrieval geometry predictably, random control does not
(R§2). Ungated CA scores higher IID but fuses language unanalyzably — report cost
honestly (~2–5 pts), no OOD claim.

## A3 — Performance & architecture

**Claim A3.1** (✅): Works across semantic/discriminative backbones: DINOv2 92.4 /
SigLIP 92.6 / supervised 86.6; pixel-reconstruction pretraining (MAE 74.8) is the
consistent laggard across every readout (74.2–77.0) → the substrate quality claim.

**Claim A3.2** (⏳ E1b): Sup-ViT is a *readout interaction*, not a weak substrate —
it reaches 93.8 with the GCA-decoder vs 86.6 concat. E1b per-qtype tables will
localize where concat loses it (prediction: Count/CmpInt, not QryAttr — sup already
shows 94.0 QryAttr in the old draft). If confirmed, the paper's "sup-ViT mid-tier"
framing softens to "readout-sensitive".
⚠ Constraint from R0: Table 1 per-category cells must be regenerated from the concat
checkpoints (E1b) — the draft's DINOv2 row cells came from a legacy GCA-decoder run.

**Claim A3.3 — CoGenT** (✅ with number fix): zero-shot compositional gap is small
(94.5 → 89.5) and closes to ~2 pts with 50k B-finetuning without forgetting A
(92.4/92.7 retained). ⚠ Replace the draft's unreproducible 92.4/88.0 (legacy ep-11
checkpoint, ValB never persisted) with the reproducible main-repo numbers — they are
*better*. (User decision pending, recommended.)

## A4 — Mechanistic rigor: perturbation triad with localization

**Claim A4.1** (✅ E9, ⚠ wording): Recovery from text-side perturbations routes
through cross-attention; image-side through late self-attention. The correct wording
is a **gradient, not an absolute**: CA-share A 0.53–0.55 > B 0.43–0.49 > C 0.20–0.43;
C's strongest head is late SA (L11H0/H11) for every attribute; A's strongest heads
are mid-layer CA (R§6). DO NOT write "A only affects CA, C only affects SA" — the
data does not support "only" (e.g. C-size has one large CA effect, L9H14 Δ+3.36).
Suggested sentence: "Text-side perturbations are repaired predominantly by
cross-attention heads (Binding), image-side perturbations by late self-attention
(Retrieval-side integration), with the queried-attribute text perturbation (B)
intermediate — consistent with B touching both stages."

**Claim A4.2** (✅): Interventions on binding heads causally chain into Retrieval
(existing back-patch / binding-interchange / grounding-manipulation artifacts).

**Claim A4.3** (⏳ E3): The circuit replicates across backbones — SigLIP GCA-decoder
patching incoming. Acceptance: analogous mid-layer CA binding heads + late-SA
retrieval concentration. If only partial, claim "qualitative replication" and show
both heatmaps.

## A5 — Failure modes (+ agent-autonomous diagnosis)

**Claim A5.1** (⏳ E5): Failures concentrate in counting and integer comparison;
yes/no questions are the worst-calibrated. Evidence incoming: failure_modes.py on
concat main + GCA-decoder (per-family tables, confusion, signed count errors).

**Pre-registered diagnosis hypotheses (E5b — run after E5 tables land):**
- **H1 (yes/no = comparison-chain break)**: yes/no errors concentrate in
  equal_*/same_* families (two-object comparison chains), not exist families —
  i.e. Binding handles one referent fine; comparing TWO bound objects breaks.
  Test: per-family acc split exist vs equal_/same_; if exist ≫ equal_, H1 holds.
- **H2 (answer-prior collapse)**: within wrong yes/no answers, pred distribution
  collapses to the majority answer (pred_no_rate ≫ gt_no_rate). Test: E5's
  yesno.confusion + pred_no_rate. If confirmed, connects to Retrieval reading a
  prior rather than the bound comparison outcome.
- **H3 (counting = off-by-one at Binding)**: count errors are dominated by ±1
  (missed/double-counted object), not uniform — i.e. near-correct enumeration,
  failure at set-boundary binding. Test: signed_error_hist mass at ±1 vs ≥2.
  Follow-up if holds: correlate error with n_scene_objects (E7 metadata has it) and
  with object proximity (scenes JSON).
- Each confirmed H gets one paragraph: hypothesis → test → result → mechanism link
  (Binding vs Retrieval stage attribution). Budget: ≤2 GPU-hours total, existing
  dumps only where possible.

---

## A6 (pre-registered 2026-07-05) — Compositionality taxonomy & the understanding-vs-query decomposition

User framing (verbatim intent): the architecture achieves **substitutivity**;
**systematicity** and **productivity** lean on language understanding; and since our
"visual reasoning" = *querying language to adjust visual representations*, CLOSURE
failures must be attributed to either **(i) understanding** (frozen encoder cannot
compose the novel template) or **(ii) query limitation** (representation fine; the
learned GCA query/routing is template-specialized and cannot express the novel
composition).

Taxonomy mapping:
- **Substitutivity ✅**: CoGenT zs 89.5 (novel attribute pairings), perturbation
  triad (value swaps repaired head-locally), E7 (referent stability under scene edits).
- **Systematicity**: CLOSURE (template recombination) — currently zs 0.579 / ft-all
  0.703; per-type worst = compare_* (0.62–0.67).
- **Productivity**: accuracy-vs-program-depth axis (now recorded per-question in E5's
  `per_depth`). Note: CLEVR val depth is within the training distribution — this
  measures within-distribution depth robustness; true beyond-depth productivity needs
  generated deeper questions (future work, one line in limitations).

### Discriminating experiments for (i) understanding vs (ii) query limitation

| # | Test | Verdict rule | Cost |
|---|---|---|---|
| T1 | **Text-side probe transfer** (encoder only, no vision): train probes on CLEVR-template sentence reprs to decode program structure (queried attr, anchor attrs, relation, comparison type); test on CLOSURE sentences | transfer holds → understanding intact → (ii); breaks → (i) | tiny (CPU-class) |
| T2 | **Encoder capacity axis**: learned-text → RoBERTa-L → t5-large → t5-xl (param-matched at the low end), eval IID + Humans-zs + CLOSURE-zs per type | CLOSURE flat across capacity → (ii); rises → (i) partial | 2–3 training runs |
| T3 | **Freeze-mode CLOSURE recovery**: closure ft with connector-only vs gca+connector vs all (ft_all=0.703 exists; other two are cheap) | connector-only recovers most → interface remap suffices, understanding fine → (ii) at the encoder→query interface; needs GCA → routing itself template-specialized | 2 short ft runs |
| T4 | **Binding-stage inspection on CLOSURE items**: do binding heads land on the correct referents for novel templates (attention maps / patching vs matched CLEVR questions)? | mislands → (ii) at Binding; lands correctly but answer wrong → break downstream, links to H1 compare-chain | analysis-only |

Pre-registered predictions: T1 transfers (understanding intact); T2 CLOSURE mostly
flat except embed_* (+3–5) with compare_* immobile (±1); T3 connector-only recovers
a large fraction of ft_all's gain; T4 binding lands correctly on embed_*, breaks on
compare_* → overall verdict "query/routing limitation, concentrated at the
two-referent comparison chain" (consistent with H1). If T1 breaks instead, A6 flips
to an understanding-bottleneck story and T2's capacity axis becomes the headline.

### T5-vs-RoBERTa pre-registration (capacity axis, controlled)
Fixed grounding architecture; text encoder ∈ {learned (24.6 ✅), roberta-large
(92.4 ✅), t5-large-encoder (~335M, param-matched), t5-xl-encoder, (opt) flan-t5-xl}.
Predictions: IID saturates at roberta (≤1pt differences); Humans-zs rises with
capacity; convergence speed + binding-head sharpness favor T5 (its encoder was
consumed by cross-attention during pretraining — "CA-ready" K/V); CLOSURE per §A6.
Mechanistic add-on: compare binding-head concentration across encoders.

### Baseline reframe (X14 → mechanism-transfer baselines; Transfusion has NO public weights anywhere — dropped)
- **I2T zero-shot mechanism analysis (priority)**: OpenFlamingo-9B/3B (open licenses;
  gated CA same lineage as ours) — port patching/RSA onto its GCA; question: do
  binding-head-like structures exist zero-shot in scale-pretrained CA? If yes, the
  mechanism is a general property of language conditioning, not a CLEVR-training
  artifact (kills the overfit objection).
- **T2I analysis**: PixArt-Σ (DiT 0.6B + frozen T5) — DIFT-style small-t feature
  extraction; (a) per-block probing on the 3 attr_query categories, (b) frozen
  1-layer decoder readout (Table-1-protocol comparable), (c) cross-attn map
  localization on the referent (zero-shot Binding evidence). Caveat: questions ≠
  captions (domain mismatch, state in text).
- Scale disclosure: OpenFlamingo CA ≈1.3B vs ours ≈0.1B — comparison axis is
  *origin of cross-attention*, not parameter-matched performance.

## Number-consistency rules (from R0 — apply to every draft revision)

1. Every number in the paper must match a row in `docs/paper_artifacts.md`; new
   numbers enter via `aggregate_results.py` output, never hand-typed.
2. Accuracy convention: final-epoch val acc; checkpoint-stored `val_acc` is
   authoritative over text logs (dir-contamination lesson, P§8.1).
3. Single-provenance per table: never mix variants/runs within one table row-set.
4. Seed status: single seed (s42) unless E1c runs; do not claim "3 seeds" in v2
   without artifacts (currently none for s44).
