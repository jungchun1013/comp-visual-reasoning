#!/bin/bash
# Launch all model × split combinations for gCOG Phase 1 experiments.
#
# Usage:
#   bash scripts/gcog/launch_all.sh          # run all
#   bash scripts/gcog/launch_all.sh gca      # run only gca model across all splits

cd /nfs/turbo/coe-chaijy/jungchun/vault/a-concept/comp-visual-reasoning/main

PYTHON="../SteerViT-legacy/.venv-aspen/bin/python"
SCRIPT="scripts/gcog/train_gcog.py"

MODELS=("transformer" "crossattn" "gca" "perceiver" "multimodal")
SPLITS=("distractor" "opsys" "comptree" "productive")

# Filter models if argument provided
if [ -n "$1" ]; then
    MODELS=("$1")
fi

SEED=42
N_EPOCHS=50
NTRIALS=5000

for model in "${MODELS[@]}"; do
    for split in "${SPLITS[@]}"; do
        echo "=========================================="
        echo "Model: ${model} | Split: ${split}"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
            --model "$model" \
            --split "$split" \
            --n-epochs $N_EPOCHS \
            --ntrials $NTRIALS \
            --seed $SEED \
            2>&1 | tee -a "../outputs/gcog/${model}_${split}_seed${SEED}/stdout.log"
    done
done
