#!/bin/bash
# 3 core models × 4 splits = 12 runs
cd /nfs/turbo/coe-chaijy/jungchun/vault/a-concept/comp-visual-reasoning/main

PYTHON="../SteerViT-legacy/.venv-aspen/bin/python"
SCRIPT="scripts/gcog/train_gcog.py"

MODELS=("transformer" "crossattn" "gca")
SPLITS=("distractor" "opsys" "comptree" "productive")

SEED=42
N_EPOCHS=200
NTRIALS=5000

for model in "${MODELS[@]}"; do
    for split in "${SPLITS[@]}"; do
        OUTDIR="../outputs/gcog/${model}_${split}_ep${N_EPOCHS}_seed${SEED}"
        mkdir -p "$OUTDIR"
        echo "=========================================="
        echo "[$(date)] Model: ${model} | Split: ${split}"
        echo "=========================================="
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
            --model "$model" \
            --split "$split" \
            --n-epochs $N_EPOCHS \
            --ntrials $NTRIALS \
            --seed $SEED \
            --output-dir "$OUTDIR" \
            2>&1 | tee "$OUTDIR/stdout.log"
    done
done

echo ""
echo "=========================================="
echo "[$(date)] ALL 12 RUNS COMPLETE"
echo "=========================================="
