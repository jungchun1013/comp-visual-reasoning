#!/bin/bash
# ViT models (vit_crossattn, vit_gca) × 4 splits = 8 runs
# With cached ViT features, ntrials=2000, ep50
cd /nfs/turbo/coe-chaijy/jungchun/vault/a-concept/comp-visual-reasoning/main

PYTHON="../SteerViT-legacy/.venv-aspen/bin/python"
SCRIPT="scripts/gcog/train_gcog.py"

MODELS=("vit_crossattn" "vit_gca")
SPLITS=("distractor" "opsys" "comptree" "productive")

SEED=42
N_EPOCHS=50
NTRIALS=2000

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
echo "[$(date)] ALL 8 VIT RUNS COMPLETE"
echo "=========================================="
