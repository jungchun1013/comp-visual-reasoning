# CLAUDE.md — SteerViT Experiments

## Environment
- Python env: `.venv-aspen/bin/python` (NEVER use `uv run` — CUDA driver mismatch)
- GPU cluster with NFS storage — avoid concurrent heavy I/O

## Running experiments
- Single entry point: `python scripts/train.py +experiment=<name>`
- Override any config from CLI: `python scripts/train.py +experiment=clevr_cls training.lr=5e-5`
- Hydra outputs go to `outputs/` (auto-managed)
