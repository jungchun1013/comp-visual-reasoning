# scripts/ — map of entry points

Run everything from `main/` with `PYTHONPATH=src` and the venv interpreter (see `CLAUDE.md`).
All checkpoint loading goes through `src/model/checkpoint_io.py:load_any_checkpoint` — never write a new loader.

**Training:** `train.py` (unified entry, `+experiment=<name>`); variants `train_refseg.py`, `train_llava.py`, `train_flamingo_clevr.py`; `gcog/` (gCOG task launchers).

**Evaluation:** `evaluate.py` (standalone); `eval_generalization.py` (CLEVR / CLOSURE / CoGenT), `eval_closure.py`, `eval_clevr_math.py`, `eval_legacy_humans.py`; `finetune_cogent_b.py` (CoGenT sample efficiency).

**CoGenT patching (top level):** `run_cogent_patching.py`, `plot_cogent_patching_diff.py`, `back_patch.py` — note the activation-patching reference implementation is the legacy `run_headwise_by_type.py`, not `run_cogent_patching.py`.

**`analysis/`** — figure style comes from `src/analysis/plot_style.py` (import it, don't copy):
- Patching / circuits: `activation_patching.py`, `path_patching.py`, `acdc.py`, `patching_tsne.py`, `binding_interchange.py`, `cogent_zeroshot.py`, `abc_localization.py`.
- Probes + RSA: `linear_probe.py` (answer match/decode per layer), `linear_probe_single.py` / `linear_probe_multi.py` (attribute decodability on single-/multi-object images), `conditional_rsa.py`, `grounding_manipulation.py`.
- t-SNE / retrieval: `tsne_viz.py` (qtype / steered / cross-model), `tsne_single_object.py` (no-CA vs CA on `data/clevr_single_object*`), `dino_attribute_tsne.py` (pretrained DINOv2), `retrieval_viz.py`.
- Renders: `render_single_objects.py` (Blender, all 96 attribute combos → `outputs/analysis/single_objects/`), `render_single_object.py` (pyrender approximation → `data/clevr_single_object*`, consumed by the `*_single_object` scripts above), `render_visual_corruptions.py`, `render_add_object.py` (+ companion `add_object_eval.py`).
- Aggregation / reporting: `aggregate_results.py` (→ `docs/results_tables.md`), `failure_modes.py`.
