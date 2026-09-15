# Terminology and reporting contract

**Author: Codex — September 14, 2026. Current editorial authority, requested by the user.**

## Message to Claude and other agents

Do not invent a processing stage, mechanism, or acronym to summarize a plot.
Do not promote a correlation, a label assigned by the analyst, or a successful
intervention into a broader claim without checking what the experiment distinguishes.
This is a required correction to recurring reporting errors, not a stylistic suggestion.

The historical words “Retrieval,” “removal,” “marker,” and “binding” have referred
to different measurements. Never infer a measurement from its display name alone.
Read the result schema, semantic condition ID, producing code and model provenance.
If they disagree, report the conflict rather than guessing which narrative is intended.

## Stable language

| Concept | Preferred wording | Do not substitute |
|---|---|---|
| Task scope | Grounding in attribute-query tasks | General symbol processing established |
| Functional requirements | Referent identification; attribute-value retrieval | Two empirically proven serial stages |
| Direct RSA condition 1 | Description satisfaction | Proven feature-binding mechanism |
| Direct RSA condition 2 | Object-profile match | Attribute-value retrieval; instance identity |
| Direct RSA condition 4 | Answer agreement | An identified retrieval algorithm |
| Relational conditions 2 / 3 / 6 | Anchor profile match / answer-object profile match / answer agreement | Reuse direct indices without the task family |
| Normalized attribute projection | Cosine alignment of the normalized object-region mean with its own attribute-value direction | Accuracy; information content |
| Below-baseline projection | Reduced attribute alignment in the non-referent representation | Removal / erasure of information; selection by removal |
| Data-estimated reference contrast | Reference-related direction (state estimator) | A proven object-specific pointer |
| RDM subset structure | Conditional semantic organization | Causal dependency; linear subspace carved out by binding |
| Empirical answer choice | Fraction of items answered with the other object's value (`pick_D`) | Mean softmax probability (`p_D`) |
| No qualifying heads | Registered intervention not estimable | Ablation had no effect |
| Weak position probe | Inconclusive position decoding | Absence of position information |
| Different backbone results | Consistency/differences across tested models | Causal effect of pretraining objective |

“Removal” remains appropriate for an actual specified intervention, such as removing
a projection or disabling a module. Do not globally replace code identifiers or
historical registration text. Correct current summaries and annotate historical text.

## Required structure of an experiment record

1. Question and hypothesis; mark exploratory versus prospectively specified.
2. Model/run/checkpoint/readout and code version.
3. Dataset, question family, image and question counts, exclusions and correctness filter.
4. Representation unit, layer/hook location, normalization and comparison baseline.
5. Operation and controls; distinguish equal direction norm from equal perturbation norm.
6. Statistic with units, denominator, interval type and resampling unit.
7. Result artifact and field/index, not only a figure or a prior prose summary.
8. What the result supports; remaining alternatives; next necessary check.

Avoid compressed strings of internal experiment IDs. Introduce the experiment in
ordinary language, then put its stable ID and source in parentheses. Distinguish
object identity from question-dependent role: code `target` may be a non-referent
under the alternate question. No acronym for a long sentence is needed.

## Specific corrections that must propagate

- The old direct RSA 0.572687 is condition 4 (answer agreement); condition 2 is
  0.247484 (object-profile match). The current plot labels are semantic, not stage labels.
- Existing RSA shading in the inspected script is mean ± SEM, not SD or 95% CI.
- GQA injection 0.064 is 8/125 answers, not a mean probability increase.
- X25 denominators are DINOv2 322, SigLIP 324, MAE 323; no flips is not exact zero
  probability change. SigLIP has the strongest relevant efficacy control.
- Projection-out random directions are unit vectors, not automatically equal-energy
  perturbations. Independent random directions do not inherently require cross-fitting.
- Do not claim every task has an identical two-layer lag without checking every curve.
- Do not infer an algorithm from an answer-aligned RDM or call full profile matching
  requested-value retrieval. Three semantic comparisons can study two task requirements.

## Website and historical records

Only verified current summaries should be presented as current conclusions. Preserve
older interpretations under an explicit historical/superseded label. Archived figures
may retain old labels until regenerated from the same artifacts in a NEW output path.
Never silently change an immutable result to match prose. A corrected code label does
not certify an old PNG or mean the experiment has been rerun.

## Publication-repository plan (recorded; no release authorized by this note)

- One source of truth for semantic IDs, task-family-specific condition mapping and labels.
- A manifest for each reported cell/figure: checkpoint hash, code/config, cohort, feature
  extraction, statistic, artifact field, and output build version.
- Separate reproducible current analyses, exploratory history and deprecated scripts.
- Generate tables/site/manuscript values from the same manifests; no manual transcription.
- Resolve paired model comparisons, fold grouping, direction fitting, intervention-energy
  controls, and uncertainty before labeling an analysis publication-ready.
- Provide environment, data acquisition/license notes, run commands and small semantic
  label checks; audit paths and private/log data before release.
- Retain provenance and correction history without presenting obsolete interpretations
  as accepted conclusions. Public release/hosting remains a separate action.

**Signed: Codex — September 14, 2026**
