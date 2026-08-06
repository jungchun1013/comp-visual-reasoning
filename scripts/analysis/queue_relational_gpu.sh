#!/bin/bash
# GPU queue for the relational-mechanism batch (實驗三 -> 一 -> 二 + 四-eval).
# This node: only physical GPU 0 can init a CUDA context (CVD=1 -> "No CUDA GPUs
# available"; CVD=0 -> alloc OK), verified 2026-07-15 — so CUDA_VISIBLE_DEVICES=0.
# (nvidia-smi shows GPU 0 util as [N/A], a cosmetic driver quirk; compute works.)
# Resume-safe (each stage skipped if its terminal artifact exists). Per-stage
# stdout tees into each output dir; this queue log carries stage banners + gates.
# Launch:
#   nohup bash scripts/analysis/queue_relational_gpu.sh \
#     > outputs/analysis/relational_queues/gpu_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
INTERP=/nfs/turbo/coe-chaijy/jungchun/vault/a-concept/SteerViT-legacy/.venv-aspen/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src
CKPT=outputs/model/clevr_dinov2_decoder1l_scratch_s42/best.pt
CR=outputs/analysis/conditional_rsa
AP=outputs/analysis/activation_patching
CPULOG=outputs/analysis/relational_queues/cpu_queue.log

echo "=== GPU QUEUE START  $(date '+%F %T') ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/GPU /'

# ---- S1: pos_only conditional RSA (+per_query dump) — 實驗三 + 實驗二a ----
S1=$CR/clevr_dinov2_decoder1l_scratch_pos_only
if [ -f "$S1/attr_query_spatial/rsa_conditional_stats.json" ]; then
  echo "SKIP-S1 pos_only RSA (exists)  $(date '+%F %T')"
else
  echo "=== S1 pos_only RSA  $(date '+%F %T') ==="
  "$INTERP" scripts/analysis/conditional_rsa.py --checkpoint "$CKPT" \
    --categories attr_query_direct,attr_query_same,attr_query_spatial \
    --pos-only --dump-per-query --output-dir "$S1" \
    || { echo "=== STAGE FAILED S1  $(date '+%F %T')"; exit 1; }
fi

# ---- G1: baseline-repro DIAGNOSTIC (never aborts; logs per-curve L6/9/11 diffs) ----
echo "=== G1 baseline-repro diagnostic  $(date '+%F %T') ==="
"$INTERP" - "$S1" "$CR/clevr_dinov2_decoder1l_scratch" "$CR/clevr_dinov2_decoder1l_scratch_v2" <<'PY' || echo "G1 diagnostic errored (non-fatal)"
import json, sys
from pathlib import Path
new, *bases = sys.argv[1:]
cat = "attr_query_spatial"
def load(d):
    p = Path(d)/cat/"rsa_conditional_stats.json"
    if not p.exists(): return None
    cr = json.load(open(p))["conditional_rsa"]
    return {e["condition_index"]: e for e in cr}
n = load(new)
if n is None:
    print("G1: new spatial stats missing"); sys.exit(0)
for base in bases:
    b = load(base)
    if b is None:
        print(f"G1: baseline {base} has no spatial stats, skip"); continue
    print(f"G1 vs {Path(base).name}:")
    for idx, e in sorted(n.items()):
        if idx not in b: continue
        row=[]
        gross=False
        for L in ("6","9","11"):
            mn=e["per_layer"].get(L,{}).get("mean"); mb=b[idx]["per_layer"].get(L,{}).get("mean")
            if mn is None or mb is None: continue
            d=mn-mb; row.append(f"L{L} Δ{d:+.4f}")
            if abs(d)>0.1: gross=True
        tag=" <<< G1-GROSS(>0.1)" if gross else ""
        print(f"  idx{idx} {e['name'][:28]:28s} " + " ".join(row) + tag)
PY

# ---- S2: anchor_swap headwise patching — 實驗一 step1 ----
S2=$AP/clevr_dinov2_decoder1l_scratch_anchor_swap
if [ -f "$S2/headwise_by_type_stats.json" ]; then
  echo "SKIP-S2 anchor_swap patching (exists)  $(date '+%F %T')"
else
  echo "=== S2 anchor_swap patching  $(date '+%F %T') ==="
  "$INTERP" scripts/analysis/activation_patching.py --checkpoint "$CKPT" \
    --groups anchor_swap --directions noising,denoising --num-samples 50 \
    --output-dir "$S2" \
    || { echo "=== STAGE FAILED S2  $(date '+%F %T')"; exit 1; }
fi

# ---- G2: automatic head selection (or skip ablation) ----
echo "=== G2 head selection  $(date '+%F %T') ==="
SEL=$("$INTERP" scripts/analysis/select_ablation_heads.py \
        --stats "$S2/headwise_by_type_stats.json" \
        --group anchor_swap_noising --category anchor_swap); G2RC=$?
echo "$SEL"
ABLATE=$(printf '%s\n' "$SEL" | awk '/^ABLATE/{print $2}')
RANDOMH=$(printf '%s\n' "$SEL" | awk '/^RANDOM/{print $2}')
QLIST=$S1/attr_query_same/rsa_per_query.json

if [ "$G2RC" -ne 0 ] || [ -z "$ABLATE" ]; then
  echo "=== G2 FAILED (rc=$G2RC) — SKIP ablation stages S3/S4; leaving head-ablation for user review  $(date '+%F %T')"
else
  # ---- S3: ablate top anchor-binding heads — 實驗一 step2 ----
  S3=$CR/clevr_dinov2_decoder1l_scratch_head_ablation
  if [ -f "$S3/attr_query_same/rsa_conditional_stats.json" ]; then
    echo "SKIP-S3 head_ablation (exists)  $(date '+%F %T')"
  else
    echo "=== S3 head_ablation heads=$ABLATE  $(date '+%F %T') ==="
    "$INTERP" scripts/analysis/conditional_rsa.py --checkpoint "$CKPT" \
      --categories attr_query_same --query-list "$QLIST" \
      --ablate-heads "$ABLATE" --ablate-mode zero --pos-only --dump-per-query \
      --output-dir "$S3" || echo "=== STAGE FAILED S3 (non-fatal, continue)"
  fi
  # ---- S4: random-head control ----
  S4=$CR/clevr_dinov2_decoder1l_scratch_head_ablation_random
  if [ -f "$S4/attr_query_same/rsa_conditional_stats.json" ]; then
    echo "SKIP-S4 head_ablation_random (exists)  $(date '+%F %T')"
  else
    echo "=== S4 head_ablation_random heads=$RANDOMH  $(date '+%F %T') ==="
    "$INTERP" scripts/analysis/conditional_rsa.py --checkpoint "$CKPT" \
      --categories attr_query_same --query-list "$QLIST" \
      --ablate-heads "$RANDOMH" --ablate-mode zero --pos-only --dump-per-query \
      --output-dir "$S4" || echo "=== STAGE FAILED S4 (non-fatal, continue)"
  fi
fi

# ---- S5: anchor_probe — 實驗二b ----
S5=outputs/analysis/anchor_probe/clevr_dinov2_decoder1l_scratch
if [ -f "$S5/probe_results.json" ]; then
  echo "SKIP-S5 anchor_probe (exists)  $(date '+%F %T')"
else
  echo "=== S5 anchor_probe  $(date '+%F %T') ==="
  "$INTERP" scripts/analysis/anchor_probe.py --checkpoint "$CKPT" \
    --output-dir "$S5" || echo "=== STAGE FAILED S5 (non-fatal, continue)"
fi

# ---- S6: wait for CPU render queue DONE (poll, cap ~24h) — 實驗四 gate ----
echo "=== S6 waiting for CPU render queue DONE  $(date '+%F %T') ==="
for i in $(seq 1 48); do
  grep -q "^=== QUEUE DONE" "$CPULOG" 2>/dev/null && { echo "CPU queue DONE seen"; break; }
  # if the CPU queue process is gone without DONE, stop waiting and eval what exists
  pgrep -f "queue_relational_cpu.sh" >/dev/null || { echo "CPU queue process gone (no DONE marker) — evaluating available sets"; break; }
  sleep 1800
done

# ---- S7: add_object_eval on each rendered set that has >=100 pairs (G4) ----
SR=outputs/analysis/shortcut_renders
for set in shared_anchor_attr_query_same shared_anchor_attr_query_spatial \
           translate_attr_query_same translate_attr_query_spatial; do
  PAIRS=$SR/$set/pairs.json
  N=$([ -f "$PAIRS" ] && "$INTERP" -c "import json;print(len(json.load(open('$PAIRS'))))" 2>/dev/null || echo 0)
  if [ "${N:-0}" -lt 100 ]; then
    echo "SKIP-EVAL $set (only ${N:-0} pairs, <100 — G4)  $(date '+%F %T')"; continue
  fi
  # skip if an eval json already sits next to pairs.json
  if ls "$SR/$set"/add_object_eval_*.json >/dev/null 2>&1; then
    echo "SKIP-EVAL $set (eval json exists)  $(date '+%F %T')"; continue
  fi
  echo "=== S7 add_object_eval $set (n=$N)  $(date '+%F %T') ==="
  "$INTERP" scripts/analysis/add_object_eval.py --checkpoint "$CKPT" --pairs "$PAIRS" \
    || echo "=== EVAL-FAILED $set (non-fatal)"
done

# ---- S8: anchor_dissipation (CPU) — 實驗二c ----
S8=outputs/analysis/anchor_dissipation/clevr_dinov2_decoder1l_scratch
BASEQ=$S1/attr_query_same/rsa_per_query.json
ABLQ=$CR/clevr_dinov2_decoder1l_scratch_head_ablation/attr_query_same/rsa_per_query.json
if [ -f "$S8/dissipation_stats.json" ]; then
  echo "SKIP-S8 anchor_dissipation (exists)  $(date '+%F %T')"
elif [ -f "$BASEQ" ]; then
  echo "=== S8 anchor_dissipation  $(date '+%F %T') ==="
  if [ -f "$ABLQ" ]; then
    "$INTERP" scripts/analysis/anchor_dissipation.py --baseline "$BASEQ" --ablation "$ABLQ" \
      --output-dir "$S8" || echo "=== STAGE FAILED S8 (non-fatal)"
  else
    "$INTERP" scripts/analysis/anchor_dissipation.py --baseline "$BASEQ" \
      --output-dir "$S8" || echo "=== STAGE FAILED S8 (non-fatal)"
  fi
else
  echo "SKIP-S8 (no per_query dump from S1)  $(date '+%F %T')"
fi

echo "=== QUEUE DONE  $(date '+%F %T') ==="
