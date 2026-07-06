# CLAUDE.md — SteerViT Experiments (main repo)

Vault-root `../CLAUDE.md` has the full environment facts, pre-flight checklists,
and playbook routing — read it first. This file only holds main/-specific notes.

Before opening `../SteerViT-legacy/`, read `docs/legacy-reference.md` — naming,
patching methodology, plot style, checkpoint format are all distilled there; the
legacy tree is needed only for Blender tooling (`BLENDER_TOOLS_ROOT`).
Paper-number provenance + regenerating commands: `docs/paper_artifacts.md`.
Results narrative: `RESULTS.md`; generated tables: `docs/results_tables.md`
(`scripts/analysis/aggregate_results.py`). Journal: `JOURNAL.md` (research-journal
skill). Checkpoint loading: always `src/model/checkpoint_io.py:load_any_checkpoint`
— never write a new loader.

## Environment
- Python interpreter (absolute — there is NO `.venv` under `main/`):
  `/nfs/turbo/coe-chaijy/jungchun/vault/a-concept/comp-visual-reasoning/SteerViT-legacy/.venv-aspen/bin/python`
- NEVER use `uv run` — CUDA driver mismatch.
- GPU cluster with NFS storage — avoid concurrent heavy I/O.

## Running experiments
- Single entry point: `PYTHONPATH=src <interpreter> scripts/train.py +experiment=<name>`
- Override any config from CLI: `... +experiment=clevr_cls training.lr=5e-5`
- Every experiment yaml must set `wandb.name` (checkpoint dir naming depends on it).
- Hydra outputs go to `outputs/` (auto-managed). Never overwrite existing results.
- Long runs: `nohup ... > outputs/model/<wandb.name>/train.log 2>&1 &` — logs
  live with the run, never in `/tmp`.
