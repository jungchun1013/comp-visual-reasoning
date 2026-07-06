# RESULTS

Curated findings with artifact pointers. Machine-generated tables live in
`docs/results_tables.md` (via `scripts/analysis/aggregate_results.py`, W2); paper-number
provenance lives in `docs/paper_artifacts.md`. Never hand-edit numbers here without an
artifact path beside them.

Naming (v2): **Grounding** = the whole language-conditioning mechanism; stages =
**Binding → Retrieval**. Old 3-stage paper terms map via `docs/legacy-reference.md`.

## 1. Accuracy matrix (R1)

Authoritative table: `docs/paper_artifacts.md` §9. Headline (final-epoch val acc, s42):
concat readout — DINOv2 0.9237, SigLIP 0.9256, Sup 0.8655, MAE 0.7476 (all four =
paper Table 1 exactly, from checkpoint-stored `val_acc`); GCA-decoder —
0.9095 / 0.9297 / 0.9376 / 0.7420; cls — 0.9014 / 0.8476 / 0.8663 / 0.7701.

**Backbone claim (v2 framing)**: semantic/discriminative pretraining (DINOv2, SigLIP,
supervised) yields a usable substrate; pixel-reconstruction pretraining (MAE) is the
consistent laggard across ALL readouts (0.742–0.770). Sup-ViT is not a weak substrate —
it reaches 0.9376 with the GCA-decoder; its low concat number (0.8655) is a
readout interaction, not a substrate failure. (Per-qtype E1b breakdowns will resolve
where the concat readout loses it.)

## 2. Gate = design choice (R2)

Framing (user-decided): the tanh gate is not a performance claim — it is the design
choice that makes the mechanism *analyzable*. The scalar gate is the single
language-influence handle every mechanistic analysis operates through (patching,
interventions, alpha interpolation). Ungated CA trains fine and even scores higher IID,
but fuses language into the stream with no separable handle, risking collapse into
unanalyzable fusion.

Honest numbers (unreported in paper as claims):

| Variant | DINOv2 | SigLIP | Sup | MAE | artifact |
|---|---|---|---|---|---|
| gated (concat readout) | 0.9237 | 0.9256 | 0.8655 | 0.7476 | §9 of paper_artifacts.md |
| nogate (ungated CA) | 0.9683 | 0.9457 | 0.9439 | 0.9136 | `clevr_<bb>_nogate_scratch_s42` logs |

Cost of the gate: ~2–5 pts IID (and MAE benefits most from ungated fusion — consistent
with a weak substrate leaning on the text stream).

Evidence that the gate is a *graded, monotonic* control handle:
- Activation interpolation sweep (`outputs/analysis/cogent_zeroshot/zeroshot_alpha_sweep.json`,
  50 samples): accuracy degrades monotonically as GCA activations are interpolated away
  from the true question toward a composed substitute — α=0: 47/50, α=0.25–0.75: 40/50,
  α=1.0: 37/50. Graded control, no cliff.
- Gate-mediated interventions: `outputs/analysis/grounding_manipulation/clevr_dinov2_decoder1l_scratch/`
  (manipulation_{grounding,answer,random}.json + retrieval/tsne figures) — steering the
  gated pathway moves Binding→Retrieval geometry predictably; random control does not.
- Per-layer gate magnitudes: `outputs/analysis/gate_values.png`.

## 3. CoGenT (R3)

Main-repo numbers (all reproducible):

| Quantity | Value | Artifact |
|---|---|---|
| trainA final ValA | 0.9408 | `outputs/model/cogent_dinov2_decoder1l_scratch_s42/stdout.log` |
| ValA (eval protocol) | 0.94466 | `outputs/analysis/cogent_sample_efficiency/sample_efficiency.json` "before" |
| ValB zero-shot | 0.89479 | same |
| ValB after 50k/8ep ft on B | 0.927 (ValA 0.924 retained) | `cogent_sample_efficiency/50k_8ep.log` |
| Sample efficiency (1k–50k, 4ep) | valB 0.904 → 0.926, valA stable 0.92± | `sample_efficiency.json` "runs" |

DECIDED (user 2026-07-05): the draft's 92.4/88.0 was a transcription error;
camera-ready uses the reproducible numbers above. Zero-shot compositional gap is only
~5 pts (94.5→89.5) and closes to ~2 pts with 50k B-samples.

## 4. Baseline triage (R4)

| Baseline | State | Disposition |
|---|---|---|
| MoT (`clevr_mot_scratch_s42`) | complete, ep15 val 0.7483 | **keep** in baseline table |
| Flamingo-style (`clevr_flamingo_dinov2_early_s42`) | last.pt only, no eval, training log has no val acc | evaluate via E1-style run if needed; otherwise drop |
| LLaVA-style (`clevr_llava_dinov2_lora_s42`) | dir empty | **drop** — attempted, no recoverable checkpoint |
| Transfusion (`clevr_transfusion_scratch_s42`) | config only, never trained | **drop** (or retrain — user decision, JOURNAL TODO) |

## 5. Per-question-type breakdown (E1b) — 9/13 landed 2026-07-06

Artifacts: `outputs/analysis/generalization/<run>.json` (single-provenance, concat/cls
s42 best.pt = final epoch except learned_text, see registry D2).

| run (concat) | overall | QryAttr | EqAttr | Exist | Count | CmpInt |
|---|---|---|---|---|---|---|
| dinov2 | 0.924 | 0.991 | 0.925 | 0.960 | 0.853 | 0.785 |
| siglip | 0.926 | 0.990 | 0.921 | 0.964 | 0.863 | 0.786 |
| sup | 0.866 | 0.940 | 0.839 | 0.914 | 0.792 | 0.742 |
| mae | 0.748 | 0.921 | **0.586** | 0.777 | 0.603 | 0.718 |

cls rows: dinov2 0.901 / siglip 0.847 / sup 0.866 / mae 0.770 (full breakdowns in the
JSONs).

Ablation rows (landed 2026-07-06, same artifact dir):

| run | overall | QryAttr | EqAttr | Exist | Count | CmpInt | paper cell |
|---|---|---|---|---|---|---|---|
| −CA (`nogca`) | 0.459 | 0.516 | 0.522 | 0.570 | **0.246** | 0.501 | 49.4 = ckpt best_acc 0.4945 ✓ |
| scratch-ViT (`gca_scratch`) | 0.528 | 0.490 | 0.517 | 0.664 | 0.459 | 0.672 | 52.8 = eval 0.5277 EXACT ✓ |
| learned-text (best.pt **ep2**) | 0.207 | **0.000** | 0.512 | 0.505 | 0.0004 | 0.511 | ⚠ paper 24.6 = last.pt ep15 (D2) |

**Findings (Fable interpretation) — the three ablations fail in three different ways,
which is the A3 "all three components necessary" claim with mechanism-level signatures:**
4. **Remove CA → counting dies first** (0.246, far below the 0.50-ish answer-prior
   plateau the other types sit at). Text at the readout without CA supports
   prior-matching but nothing that requires iterating over visual tokens.
5. **Remove visual pretraining (scratch-ViT, CA intact) → binding survives, retrieval
   starves**: binary types recover above prior (Exist 0.664, CmpInt 0.672 — the CA
   mechanism still routes) but QryAttr stays at 0.490 — there is no structured
   substrate to retrieve from. Mirror image of MAE's failure (substrate fine for
   retrieval, weak for multi-referent binding).
6. **Remove language pretraining → generation collapses to closed-set types**:
   QryAttr exactly 0.0 and Count 0.0004 — the decoder emits degenerate strings for
   open-vocabulary answers; only binary types land at chance (~0.51). Provenance
   caveat: this row is best.pt(ep2); full-val eval of that ckpt gives 0.207 while its
   stored windowed acc says 0.4667 — one more reason the paper cell must come from the
   last-epoch rerun (registry D2 action item, still open).

D2 lastep rerun (landed 03:51,
`generalization/clevr_dinov2_learned_text_decoder1l_lastep_s42.json`): last.pt(ep15)
independent eval = **0.1974** — same qualitative collapse (QryAttr 0.000, Count 0.003,
binary types at chance ~0.48). The paper cell 24.6 therefore matches only the
training-loop log (0.2456), not the independent protocol (0.197). Camera-ready needs
either a footnote (protocol difference) or renumbering to 19.7 — user decision (TODO).

**Findings (Fable interpretation):**
1. **A3.2 prediction partially wrong — sup-ViT's concat deficit is UNIFORM, not
   Count/CmpInt-concentrated** (loses ~5pts on every type incl. QryAttr 0.940 vs
   0.991). Since sup reaches 0.938 with the GCA-decoder readout, the readout
   interaction is real but the mechanism is a broad conditioning-quality drop with
   the concat self-attn readout, which the GCA-decoder's extra cross-attn stage
   compensates. v2 wording: "readout-sensitive", not "weak at counting".
2. **MAE's failure is two-referent, not retrieval**: QryAttr 0.921 (single-referent
   retrieval nearly fine) but EqAttr collapses to 0.586 and Count 0.603 —
   reconstruction pretraining supports single-referent attribute retrieval yet fails
   multi-referent binding/comparison. This is the substrate-quality claim (A3.1) AND
   an H1-shaped failure at the substrate level — strong bridge between A3 and A5.
3. **SigLIP needs the decoder**: cls reversal (dinov2 0.901 > siglip 0.847 despite
   concat 0.926 > 0.924) — SigLIP's pooled features are weak for classification
   readout; its strength is token-level, consumed by decoders.

## 6. A/B/C × {CA, SA} localization (E9) — first pass done

Aggregated from existing patching stats (denoising, dinov2 GCA-decoder, n=50/category):
`outputs/analysis/abc_localization/clevr_dinov2_decoder1l_scratch/abc_contrast.{json,md}`
(generated by `scripts/analysis/abc_localization.py`).

**Finding: the localization claim holds as a clean gradient, not an absolute.**
CA share of per-head effect mass: **A (described-attr, text) 0.53–0.55 > B
(queried-attr, text) 0.43–0.49 > C (queried-attr, image) 0.20–0.43**. CA heads in the
10 strongest: A 4–5/10, B 2–5/10, C 0–3/10. C's strongest head is consistently *late
self-attention* (L11H0 / L11H11) for every attribute; A's strongest heads are mid-layer
*cross-attention* (L3–L9, incl. the known binding heads L7H9, L7H3). Notables:
B peaks hard on CA L7H3 (Δ +1.5–2.0 for what-color/what-material); C-size has one huge
CA outlier (L9H14, Δ +3.36) worth a follow-up; C-shape is the purest SA case (CA share
0.199, 0 CA heads in top-10).

v2 wording suggestion: "text-side perturbations route their recovery through
cross-attention (Binding), image-side perturbations through late self-attention
(Retrieval-side integration)" — supported; the absolute phrasing "A only affects CA,
C only affects SA" is not.

## 7. Failure modes (E5) — concat main model landed; H1–H3 adjudicated

Artifacts: `outputs/analysis/failure_modes/clevr_dinov2_concat_decoder1l_scratch_s42/`
(records.jsonl n=37,498 stride=4; overall 0.9240 ≈ full-val 0.9237 — subsample faithful).
Pre-registered hypotheses (docs/paper_v2_outline.md):

- **H2 (answer-prior collapse on yes/no): REFUTED.** pred-no rate 0.504 vs gt-no 0.503;
  confusion symmetric (yes→no 710 / no→yes 690); yes/no acc 0.9075. No majority-class
  bias whatsoever — the draft's "worst = yes/no because prior" story is dead.
- **H3 (counting off-by-one): CONFIRMED.** Of 1,315 counting errors, 86.9% are ±1
  (−1: 586 / +1: 557 — symmetric). Approximate-numerosity behavior, not random guessing.
  Drift: '0' underpredicted (−63) — the model dislikes answering zero.
- **H1 (two-referent chains): CONFIRMED, refined.** The 8 worst families are ALL
  two-set cardinality questions: count-over-union ("what number of things are either X
  or Y" — fam 67/71/70, acc 0.52–0.64), compare-counts ("are there an equal number of
  X and Y" — fam 6/7/3, acc 0.61–0.74), count-over-intersection (fam 31/25). The
  failure is not the yes/no format and not counting per se — it is **enumerating and
  combining MULTIPLE referent sets**.

**Headline finding — the difficulty axis is referent multiplicity, NOT program depth**
(per-depth × qtype table in records.jsonl):

- query_attribute (one referent chain) is **flat 0.97–1.00 from depth 4 to depth 20** —
  productivity along single-referent composition is fully achieved.
- count 0.99→0.69 and compare_integer 0.91→0.74 as depth grows; equal_attribute
  1.00→0.88; exist decays only past depth ~11 (where deep exist = union constructs).
- The aggregate depth curve is non-monotonic (dip 0.85 at depth 9–11, recovery to
  0.92+ at 13+) — a pure composition artifact: deep buckets are dominated by
  single-chain query_attribute, mid buckets by multi-referent types.

**Autonomous-diagnosis follow-up — relations are free for localization, costly for
enumeration** (same records.jsonl, question-text stratification):

- Deep query_attribute chains (depth ≥18, 3–4 spatial hops, e.g. "the tiny gray object
  left of the tiny green ball in front of the small gray thing that is left of…"):
  **acc 0.992 (n=498)**. Relational-chain *localization of one object* is at ceiling.
- Single-set count stratified by number of spatial relations: 0 rel 0.982 → 1 rel
  0.789 → 2 rel 0.767 → 3 rel 0.656. The SAME relation vocabulary that costs nothing
  in query_attribute collapses counting — because there the relation defines a
  *region* (a half-plane of variable cardinality) whose members must all be bound,
  not a stepping stone to one object.
- count substructure: single-set chain 0.929 > union (either/or) 0.763 > intersection
  0.662. equal_attribute by relations: 0 rel 0.976 → 2 rel 0.819.

Refined v2 wording: "Difficulty is set by the cardinality of what a query step must
bind: one object (ceiling accuracy at any program depth), a variable-size set to
enumerate (each relational region-constraint compounds the cost), or several sets to
be enumerated and combined (worst). Program depth and relational vocabulary per se
cost nothing — the bottleneck is the query mechanism's one-referent-at-a-time
binding, not language understanding of long or relational programs." Bridges to MAE's
EqAttr collapse (§5) and A6 substitutivity-vs-systematicity framing.

**Cross-readout replication (GCA-decoder, landed 03:09)** —
`outputs/analysis/failure_modes/clevr_dinov2_decoder1l_scratch_s42/` (overall 0.9074
≈ full-val 0.9095): the failure structure is **mechanism-level, not readout-level**.
Per-family accuracy Spearman ρ = **0.927** across 89 families vs the concat model;
same top-4 worst families {67, 70, 71, 6}; deep qryattr chains 0.994; single-set
count by #relations 0.977→0.736→0.710→0.594 (same monotone collapse, slightly
steeper); off-by-one = 87.9% of counting errors, symmetric (−1: 645 / +1: 648).
Readout-level differences (the only ones): (a) the generative decoder shows a mild
"no" bias absent in the classification readout (pred-no 0.512 vs gt 0.503, drift
no +142 / yes −142; yes/no acc 0.879 vs 0.908) — autoregressive calibration is
slightly worse, still nowhere near H2 collapse; (b) zero non-numeric outputs on
counting — the decoder's answer vocabulary is well-behaved.

## 8. Add-object hallucination (E7) — landed 03:51

Artifacts: `outputs/analysis/add_object/<attr>/add_object_eval_clevr_dinov2_concat_decoder1l_scratch_s42.json`
(n=100 pairs/attr; distractor = one described-attr flipped + bait value on the queried
attr; answer invariance verified by program execution; base re-render controls the
render-domain shift).

| queried attr | acc_base | acc_added | hallucination_rate | bait_share_of_errors | flip_rate |
|---|---|---|---|---|---|
| color | 0.98 | 0.97 | 0.02 | 0.67 | 0.02 |
| material | 1.00 | 1.00 | 0.00 | — (0 errors) | 0.00 |
| shape | 1.00 | 0.98 | 0.01 | 0.50 | 0.02 |
| size | 0.90 | 0.94 | 0.06 | 1.00 | 0.04 |

**Finding (reframes A1's bottleneck story): trained binding fixation is ROBUST.**
An adversarial lure that matches the description except one attribute and carries a
bait value on the queried attribute captures the query in ≤6% of cases; accuracy
moves ≤2 pts (size +4 is n=100 noise). The few errors that DO occur are bait-shaped
(bait_share 0.5–1.0), so the failure mode exists — it is just rare. Size is the
weakest attribute throughout (lowest acc_base 0.90, highest hallucination 0.06 —
consistent with size being the least separable probe attribute).

v2 wording: the multi-object "fixation" problem the paper motivated A1 with is
*solved by the trained grounding mechanism* for direct queries; the remaining
multi-object bottleneck is set enumeration/combination (E5, §7). Whether the RAW
substrate (before language conditioning) has the fixation problem is exactly what E8
adjudicates — if raw probes confuse objects that trained binding separates, the
grounding mechanism is what fixes fixation, completing the A1→A2 arc.

### 8b. The −CA ablation fails BY fixation (E7+E5 on nogca, landed 09:06 07-06)

Same E7 pairs run on `clevr_dinov2_concat_decoder1l_nogca_scratch_s42` (−CA: no
language conditioning in the trunk; question enters only through the concat readout):

| queried attr | acc_base | acc_added | hallucination_rate | bait_share_of_errors |
|---|---|---|---|---|
| color | 0.34 | 0.28 | 0.24 | 0.33 |
| material | 0.55 | 0.45 | 0.55 | 1.00 |
| shape | 0.46 | 0.39 | 0.46 | 0.75 |
| size | 0.46 | 0.41 | 0.59 | 1.00 |

Causal contrast on identical stimuli: the bait captures the −CA model's answer
24–59% of the time vs 0–6% for the GCA model — a ~10× hallucination gap
attributable to the single added object.

E5-on-nogca supplies the observational counterpart (records at
`outputs/analysis/failure_modes/clevr_dinov2_concat_decoder1l_nogca_scratch_s42/`).
Classifying all 6,516 query_attribute errors against scene ground truth: **98.6% are
another scene object's attribute value**; out-of-scene hallucination is 1/6516.
Restricting to color (8 values, so in-scene membership is non-trivial): **100.0% of
2,276 wrong colors are present in the scene** vs a 52.3% chance baseline for a
random wrong color. The −CA model reads out a *real* object — just not the described
one: attribute encoding is intact, selection is broken. (The trained model's rare
errors have the same in-scene structure, 131/133 — the failure mode is shared; GCA
changes its *rate* by ~50×: qryattr error rate 48.6% → 1.0%.)

**Fixation triangle (A1→A2, closes when E8 lands):** (i) E5/E7-on-nogca — without
GCA, failure is object mis-selection, not encoding loss; (ii) E7-on-trained — with
GCA, fixation on the described object is robust (≤6% capture); (iii) E8 — does the
raw substrate encode attributes per-object (info present, selection absent)? If yes,
grounding's causal contribution is precisely the selection/fixation step.

## 9–10. E3, E4, E8, E10 — pending

Placeholders: E3 SigLIP patching (running) · E4 GCA-decoder probe/RSA (probe done,
RSA running) · E8 raw-backbone substrate · E10 grounding_manipulation replots.
