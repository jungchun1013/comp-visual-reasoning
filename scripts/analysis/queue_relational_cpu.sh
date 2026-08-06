#!/bin/bash
# CPU render queue for 實驗四 (shortcut_renders shortcut-exclusion sets).
# Two parallel Blender lanes (attr_query_same / attr_query_spatial); each lane
# renders shared_anchor then translate. CPU Cycles (no GPU). Resume-safe: a set
# whose pairs.json already holds >=100 pairs is skipped. Launch:
#   nohup bash scripts/analysis/queue_relational_cpu.sh \
#     > outputs/analysis/relational_queues/cpu_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
INTERP=/nfs/turbo/coe-chaijy/jungchun/vault/a-concept/SteerViT-legacy/.venv-aspen/bin/python
BLENDER=/nfs/turbo/coe-chaijy/jungchun/vault/a-concept/SteerViT-legacy/tools/blender/blender
ROOT=outputs/analysis/shortcut_renders
SCRIPT=scripts/analysis/render_add_object.py

render_set() {
  local mode="$1" category="$2"
  local dir="$ROOT/${mode}_${category}"
  if [ -f "$dir/pairs.json" ] && \
     [ "$("$INTERP" -c "import json;print(len(json.load(open('$dir/pairs.json'))))" 2>/dev/null || echo 0)" -ge 100 ]; then
    echo "SKIP-RENDER ${mode}_${category} (>=100 pairs)  $(date '+%F %T')"
    return 0
  fi
  mkdir -p "$dir"
  echo "=== RENDER ${mode}_${category}  $(date '+%F %T') ==="
  "$BLENDER" --background --python "$SCRIPT" -- \
    --mode "$mode" --category "$category" \
    --num-pairs 100 --seed 42 --output-dir "$ROOT" \
    > "$dir/render.log" 2>&1 || echo "RENDER-FAILED ${mode}_${category}  $(date '+%F %T')"
}

lane() {  # one category: both modes, sequential
  local category="$1"
  render_set shared_anchor "$category"
  render_set translate     "$category"
  echo "=== LANE DONE $category  $(date '+%F %T') ==="
}

echo "=== CPU RENDER QUEUE START  $(date '+%F %T') ==="
lane attr_query_same &
L1=$!
lane attr_query_spatial &
L2=$!
wait $L1 $L2
echo "=== QUEUE DONE  $(date '+%F %T') ==="
