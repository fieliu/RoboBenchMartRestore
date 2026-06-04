#!/bin/bash
# Build per-SKILL LeRobot datasets from generated_data/ trajectories.
#
# STREAMING design (low disk footprint): process ONE skill end-to-end before the
# next -- replay its items to rgbd, convert to a LeRobot dataset, then DELETE that
# skill's rgbd intermediates. Peak disk = one skill's rgbd, not all skills' at once.
#
#   Per skill:
#     1. replay each item's first MAX_PER_ITEM trajectories with --obs-mode rgbd
#     2. convert the skill (merging its products) into one LeRobot dataset
#     3. delete the skill's *.rgbd.*.h5 + *.replay.log  (ONLY if convert succeeded)
#
# Idempotent: a skill whose dataset already exists is skipped; rgbd is kept when
# convert fails so you can retry. Run AFTER run_mp_vla_training.sh.

set -e

P=${PYTHON:-python}
RB_ROOT=${RB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUTPUT_ROOT=${OUTPUT_ROOT:-generated_data}
DATASET_BASE=${DATASET_BASE:-datasets/warehouse_fetch}
MAX_PER_ITEM=${MAX_PER_ITEM:-30}
FPS=${FPS:-15}
cd "$RB_ROOT"

# env_dir | task instruction  (grouped per skill below)
PICK_TO_BASKET=(
  "PickToBasketContDuffEnv|move to shelf and pick Duff Beer Can to basket"
  "PickToBasketContFantaEnv|move to shelf and pick Fanta to basket"
  "PickToBasketContStarsEnv|move to shelf and pick Nestle Honey Stars to basket"
)
RESTOCK=(
  "RestockBasketToShelfContDuffEnv|pick Duff Beer Can from basket and place on shelf"
  "RestockBasketToShelfContFantaEnv|pick Fanta from basket and place on shelf"
  "RestockBasketToShelfContStarsEnv|pick Nestle Honey Stars from basket and place on shelf"
)
PICK_FROM_FLOOR=(
  "PickFromFloorBeansContEnv|pick beans from floor and place in basket"
  "PickFromFloorSlamContEnv|pick slam from floor and place in basket"
)

# ---- concurrency: each replay proc uses ~4.5GB RAM; default = free_GB/5 ----
if [ -z "${REPLAY_JOBS:-}" ]; then
  _free_gb=$(free -g | awk '/Mem/{print $7}')
  REPLAY_JOBS=$(( _free_gb / 5 )); [ "$REPLAY_JOBS" -lt 1 ] && REPLAY_JOBS=1
  [ "$REPLAY_JOBS" -gt 8 ] && REPLAY_JOBS=8
  echo "  auto REPLAY_JOBS=$REPLAY_JOBS (free=${_free_gb}GB, ~4.5GB each)"
fi

# replay_one <h5> <count>: render first <count> trajectories of this shard to rgbd.
replay_one() {
  local h5="$1" count="$2" base="${1%.h5}"
  if ls "${base}".rgbd.*.h5 >/dev/null 2>&1; then
    echo "  already replayed: $h5"; return 0
  fi
  echo "  replaying (count=$count): $h5"
  $P scripts/replay_trajectory.py --traj-path "$h5" -b cpu -o rgbd --save-traj \
     --allow-failure --count "$count" > "${base}.replay.log" 2>&1 \
     && echo "  done: $h5" || echo "  FAILED: $h5 (see ${base}.replay.log)"
}
export -f replay_one
export P

# shard_eps <h5>: episode count from the shard's sibling .json
shard_eps() { $P -c \
  "import json,sys;print(len(json.load(open(sys.argv[1]))['episodes']))" "${1%.h5}.json" 2>/dev/null || echo 0; }

# replay_item <ENV>: replay up to MAX_PER_ITEM trajectories for one product, in parallel.
replay_item() {
  local ENV="$1"
  [ -d "$OUTPUT_ROOT/$ENV" ] || { echo "  skip (no dir): $ENV"; return 0; }
  local remaining=$MAX_PER_ITEM
  while IFS= read -r h5; do
    [ -z "$h5" ] && continue
    [ "$remaining" -le 0 ] && { echo "  cap reached for $ENV, skipping: $h5"; continue; }
    local eps take
    eps=$(shard_eps "$h5")
    take=$(( remaining < eps ? remaining : eps ))
    [ "$take" -le 0 ] && continue
    replay_one "$h5" "$take" &
    remaining=$(( remaining - take ))
    while [ "$(jobs -rp | wc -l)" -ge "$REPLAY_JOBS" ]; do wait -n; done
  done < <(find "$OUTPUT_ROOT/$ENV" -name "*.h5" ! -name "*.rgbd.*" 2>/dev/null | sort)
}

# convert_skill <skill_name> <item...>: merge a skill's products into one dataset.
# Returns nonzero if the converter fails (so the caller keeps rgbd for retry).
convert_skill() {
  local skill_name="$1"; shift
  local -a items=("$@") args=()
  local it ENV TASK
  for it in "${items[@]}"; do
    IFS='|' read -r ENV TASK <<< "$it"
    args+=(--item "$OUTPUT_ROOT/$ENV|$TASK")
  done
  echo "  converting -> $DATASET_BASE/$skill_name"
  $P scripts/convert_skill_to_lerobot.py \
      --output-dir "$DATASET_BASE/$skill_name" \
      "${args[@]}" \
      --max-per-item "$MAX_PER_ITEM" --fps "$FPS"
}

# cleanup_skill <item...>: delete the skill's rgbd intermediates + replay logs.
cleanup_skill() {
  local it ENV
  for it in "$@"; do
    IFS='|' read -r ENV _ <<< "$it"
    find "$OUTPUT_ROOT/$ENV" \( -name "*.rgbd.*.h5" -o -name "*.replay.log" \) \
      -delete 2>/dev/null
  done
}

# process_skill <skill_name> <item...>: STREAM one skill end-to-end.
process_skill() {
  local skill_name="$1"; shift
  local -a items=("$@")
  local it ENV
  echo "===== Skill: $skill_name ====="
  if [ -f "$DATASET_BASE/$skill_name/meta/info.json" ]; then
    echo "  already built, skipping: $DATASET_BASE/$skill_name"; return 0
  fi
  echo "  -- replay (parallel x$REPLAY_JOBS, <=$MAX_PER_ITEM traj/item) --"
  for it in "${items[@]}"; do
    IFS='|' read -r ENV _ <<< "$it"
    replay_item "$ENV"
  done
  wait
  echo "  -- convert --"
  if convert_skill "$skill_name" "${items[@]}"; then
    echo "  -- cleanup rgbd intermediates for $skill_name --"
    cleanup_skill "${items[@]}"
  else
    echo "  CONVERT FAILED for $skill_name -- keeping rgbd for retry"
    return 1
  fi
}

process_skill "pick_to_basket"          "${PICK_TO_BASKET[@]}"
process_skill "restock_basket_to_shelf" "${RESTOCK[@]}"
process_skill "pick_from_floor"         "${PICK_FROM_FLOOR[@]}"

echo "===== Done. Datasets at: $DATASET_BASE ====="
find "$DATASET_BASE" -maxdepth 2 -name info.json -exec dirname {} \; 2>/dev/null
