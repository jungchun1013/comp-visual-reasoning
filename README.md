# Language Elicits Compositional Reasoning in Pretrained Vision Foundation Models

Code for the ICML 2026 CompLearning workshop paper. A frozen pretrained ViT +
frozen RoBERTa-large, connected by gated cross-attention (GCA) at alternating
blocks, learns compositional VQA on CLEVR — and the mechanism is analyzable:
language conditioning achieves **grounding** via routing and refocus, unfolding
in two stages, **Binding** (attach the described attributes to the right object)
→ **Retrieval** (read out the queried attribute).

> Terminology note: the workshop paper uses an older 3-stage naming
> (binding → object grounding → answer matching). All current code and docs use
> the 2-stage naming; the old↔new map is in `docs/legacy-reference.md` §1.1.

## Setup

```bash
pip install -e .            # or: uv sync (NOT on the lab cluster — CUDA mismatch)
export CLEVR_ROOT=/path/to/CLEVR_v1.0
```

Data: [CLEVR v1.0](https://cs.stanford.edu/people/jcjohns/clevr/) (700K train /
150K val). Optional: CLEVR-CoGenT (same page), CLEVR-Humans, CLEVR-Math, CLOSURE
for the transfer experiments.

## Training

Single entry point (run from the repo root; every experiment yaml sets `wandb.name`,
which names the checkpoint dir under `outputs/model/`):

```bash
PYTHONPATH=src python scripts/train.py +experiment=clevr_dinov2_concat_decoder_scratch
```

The paper's model matrix:

| Table 1/4 cell | experiment config |
|---|---|
| Main model (concat self-attn decoder), backbone ∈ {dinov2, siglip, sup, mae} | `clevr_<bb>_concat_decoder_scratch` |
| Classifier readout | `clevr_<bb>_cls_scratch` |
| −CA ablation | `clevr_dinov2_concat_decoder_nogca_scratch` |
| Scratch-ViT ablation | `clevr_dinov2_gca_scratch` |
| Learned-text ablation | `clevr_dinov2_learned_text_decoder1l` |
| Mechanistic-analysis model (GCA decoder) | `clevr_dinov2_decoder1l_scratch` |
| CoGenT | `cogent_dinov2_decoder1l_scratch` |

## Evaluation

```bash
# overall accuracy
PYTHONPATH=src python scripts/evaluate.py +experiment=<config> +checkpoint=outputs/model/<run>/best.pt
# per-question-type breakdown (+ optional CoGenT / CLOSURE / Humans)
PYTHONPATH=src python scripts/eval_generalization.py --checkpoint outputs/model/<run>/best.pt \
    --data-root $(dirname $CLEVR_ROOT) --skip-cogent --skip-closure --skip-humans
```

All checkpoint loading (main-format and legacy) goes through
`src/model/checkpoint_io.py:load_any_checkpoint`.

## Mechanistic analyses

| Analysis | script |
|---|---|
| Headwise activation patching (perturbations A/B text, C image) | `scripts/analysis/activation_patching.py` |
| A/B/C × {CA, SA} localization contrast | `scripts/analysis/abc_localization.py` |
| Conditional RSA (3 attr_query categories) | `scripts/analysis/conditional_rsa.py` |
| Linear probes per block | `scripts/analysis/linear_probe.py` |
| Gate-mediated interventions | `scripts/analysis/grounding_manipulation.py` |
| Failure modes (per-family, yes/no confusion, counting errors) | `scripts/analysis/failure_modes.py` |
| Add-object hallucination (substrate/binding stress test) | `scripts/analysis/render_add_object.py` (Blender) + `add_object_eval.py` |
| Results aggregation → `docs/results_tables.md` | `scripts/analysis/aggregate_results.py` |

Methodology details (corruption taxonomy, patching directions, stats schema):
`docs/legacy-reference.md`. Blender renders need the CLEVR dataset-generation
tooling; point `BLENDER_TOOLS_ROOT` at a checkout of
[clevr-dataset-gen](https://github.com/facebookresearch/clevr-dataset-gen).

## Reproducing the paper's numbers

`docs/paper_artifacts.md` maps **every table cell and figure** of the paper to the
run directory, analysis artifact, and exact regenerating command — including the
cells that cannot be reproduced from a repo artifact (flagged there, never guessed).
Curated findings live in `RESULTS.md`; generated tables in `docs/results_tables.md`.

## Repo layout

```
configs/            Hydra configs (model / task / data / experiment)
scripts/            entry points (train, evaluate, eval_*) — see scripts/README.md
scripts/analysis/   mechanistic + failure analyses
src/model/          CrossAttnViT (frozen ViT + GCA), checkpoint_io
src/tasks/          decoder / classification / MoT heads
src/data/           CLEVR datasets, sampling categories
src/analysis/       patching utils, plot style (single source of truth)
docs/               paper_artifacts, legacy-reference, results_tables
```

## License

MIT (see `LICENSE`).
