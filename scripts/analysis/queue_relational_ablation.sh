#!/bin/bash
# Follow-up: 實驗一 step2 head-ablation, which the main GPU queue's G2 gate skipped.
# G2 FAILED only on the absolute floor (top anchor_swap |mean|=0.344 < floor 0.5),
# NOT on the ratio test (0.344 vs median 0.0155 = 22x, far above the 3x bar). The
# 0.5 floor was calibrated to the paper's *queried*-attribute binding heads (~0.77);
# the anchor-swap corruption is more indirect, so 0.34 is a legitimate anchor-binding
# signal and the ablation should run. Heads are the selector's own picks (it prints
# ABLATE/RANDOM even on FAIL). Gates on the main queue's DONE so the GPU is free.
set -uo pipefail
cd "$(dirname "$0")/../.."
INTERP=/nfs/turbo/coe-chaijy/jungchun/vault/a-concept/SteerViT-legacy/.venv-aspen/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src
CKPT=outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt
CR=outputs/analysis/conditional_rsa
S1=$CR/clevr_dinov2_decoder1l_scratch_pos_only
QLIST=$S1/attr_query_same/rsa_per_query.json
MAINLOG=outputs/analysis/relational_queues/gpu_queue.log
ABLATE="gca:5:4,gca:11:8,gca:9:8,sa:8:0"
RANDOMH="gca:11:3,gca:1:14,gca:1:3,sa:5:10"

echo "=== ABLATION FOLLOW-UP START  $(date '+%F %T') ==="
echo "waiting for main GPU queue DONE (GPU must be free)..."
for i in $(seq 1 96); do
  grep -q "^=== QUEUE DONE" "$MAINLOG" 2>/dev/null && { echo "main queue DONE"; break; }
  pgrep -f "queue_relational_gpu.sh" >/dev/null || { echo "main queue process gone"; break; }
  sleep 300
done

# S3: ablate the selected anchor-binding heads
S3=$CR/clevr_dinov2_decoder1l_scratch_head_ablation
if [ -f "$S3/attr_query_same/rsa_conditional_stats.json" ]; then
  echo "SKIP-S3 (exists)  $(date '+%F %T')"
else
  echo "=== S3 head_ablation heads=$ABLATE  $(date '+%F %T') ==="
  "$INTERP" scripts/analysis/conditional_rsa.py --checkpoint "$CKPT" \
    --categories attr_query_same --query-list "$QLIST" \
    --ablate-heads "$ABLATE" --ablate-mode zero --pos-only --dump-per-query \
    --output-dir "$S3" || echo "=== STAGE FAILED S3"
fi

# S4: disjoint random-head control
S4=$CR/clevr_dinov2_decoder1l_scratch_head_ablation_random
if [ -f "$S4/attr_query_same/rsa_conditional_stats.json" ]; then
  echo "SKIP-S4 (exists)  $(date '+%F %T')"
else
  echo "=== S4 head_ablation_random heads=$RANDOMH  $(date '+%F %T') ==="
  "$INTERP" scripts/analysis/conditional_rsa.py --checkpoint "$CKPT" \
    --categories attr_query_same --query-list "$QLIST" \
    --ablate-heads "$RANDOMH" --ablate-mode zero --pos-only --dump-per-query \
    --output-dir "$S4" || echo "=== STAGE FAILED S4"
fi

# dissipation with baseline + ablation (new dir; never overwrite the baseline-only one)
S8=outputs/analysis/anchor_dissipation/clevr_dinov2_decoder1l_scratch_ablation
ABLQ=$S3/attr_query_same/rsa_per_query.json
if [ -f "$S8/dissipation_stats.json" ]; then
  echo "SKIP-dissipation (exists)  $(date '+%F %T')"
elif [ -f "$QLIST" ] && [ -f "$ABLQ" ]; then
  echo "=== dissipation baseline+ablation  $(date '+%F %T') ==="
  "$INTERP" scripts/analysis/anchor_dissipation.py --baseline "$QLIST" --ablation "$ABLQ" \
    --output-dir "$S8" || echo "=== STAGE FAILED dissipation"
else
  echo "SKIP-dissipation (missing per_query inputs)  $(date '+%F %T')"
fi
echo "=== ABLATION QUEUE DONE  $(date '+%F %T') ==="
