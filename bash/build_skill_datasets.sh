#!/bin/bash
# Build per-SKILL LeRobot datasets from generated_data/ trajectories.
#
#   Step 1: replay each generated h5 with --obs-mode rgbd (renders camera images,
#           which motion planning does NOT store). Handles multi-proc subdirs
#           (traj.0/, traj.1/, ...). Skips files already replayed.
#   Step 2: convert each SKILL (merging its products) into one LeRobot dataset,
#           capping episodes per product (MAX_PER_ITEM).
#
# Run AFTER run_mp_vla_training.sh. Idempotent: re-running skips done replays.

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

ALL_ITEMS=("${PICK_TO_BASKET[@]}" "${RESTOCK[@]}" "${PICK_FROM_FLOOR[@]}")

# ---- Step 1: replay each item's first MAX_PER_ITEM trajectories with rgbd ----
# replay_trajectory.py has no --num-procs, so we parallelize at the bash level.
# IMPORTANT: each replay process uses ~4-4.5GB RAM (measured), so concurrency is
# memory-bound. Default REPLAY_JOBS = free_GB / 5 (clamped [1,4]); override via env.
# We cap replays PER ITEM at MAX_PER_ITEM *before* rendering (using --count across
# shards), matching what the converter later selects -- no wasted rendering.
if [ -z "${REPLAY_JOBS:-}" ]; then
  _free_gb=$(free -g | awk '/Mem/{print $7}')
  REPLAY_JOBS=$(( _free_gb / 5 )); [ "$REPLAY_JOBS" -lt 1 ] && REPLAY_JOBS=1
  [ "$REPLAY_JOBS" -gt 4 ] && REPLAY_JOBS=4
  echo "  auto REPLAY_JOBS=$REPLAY_JOBS (free=${_free_gb}GB, ~4.5GB each)"
fi
echo "===== Step 1: replay (parallel x$REPLAY_JOBS, <=$MAX_PER_ITEM traj/item) ====="

# replay_one <h5> <count>: render first <count> trajectories of this shard.
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

# For each item (env), walk its shards in sorted order and dispatch replays until
# MAX_PER_ITEM trajectories are covered, then stop -- later shards are skipped.
# Shard episode counts come from each shard's .json (episodes list length).
shard_eps() { $P -c \
  "import json,sys;print(len(json.load(open(sys.argv[1]))['episodes']))" "${1%.h5}.json" 2>/dev/null || echo 0; }

for item in "${ALL_ITEMS[@]}"; do
  IFS='|' read -r ENV _ <<< "$item"
  [ -d "$OUTPUT_ROOT/$ENV" ] || { echo "  skip (no dir): $ENV"; continue; }
  remaining=$MAX_PER_ITEM
  while IFS= read -r h5; do
    [ -z "$h5" ] && continue
    if [ "$remaining" -le 0 ]; then
      echo "  cap reached for $ENV, skipping: $h5"; continue
    fi
    eps=$(shard_eps "$h5")
    take=$(( remaining < eps ? remaining : eps ))
    [ "$take" -le 0 ] && continue
    replay_one "$h5" "$take" &
    remaining=$(( remaining - take ))
    while [ "$(jobs -rp | wc -l)" -ge "$REPLAY_JOBS" ]; do wait -n; done
  done < <(find "$OUTPUT_ROOT/$ENV" -name "*.h5" ! -name "*.rgbd.*" 2>/dev/null | sort)
done
wait
echo "  Step 1 complete."

# ---- Step 2: convert each skill (merge its products) into one dataset ----
echo "===== Step 2: convert to per-skill LeRobot datasets ====="

convert_skill() {
  local skill_name="$1"; shift
  local -a items=("$@")
  local -a args=()
  for it in "${items[@]}"; do
    IFS='|' read -r ENV TASK <<< "$it"
    args+=(--item "$OUTPUT_ROOT/$ENV|$TASK")
  done
  echo "----- $skill_name -----"
  $P scripts/convert_skill_to_lerobot.py \
      --output-dir "$DATASET_BASE/$skill_name" \
      "${args[@]}" \
      --max-per-item "$MAX_PER_ITEM" --fps "$FPS"
}

convert_skill "pick_to_basket"            "${PICK_TO_BASKET[@]}"
convert_skill "restock_basket_to_shelf"   "${RESTOCK[@]}"
convert_skill "pick_from_floor"           "${PICK_FROM_FLOOR[@]}"

echo "===== Done. Datasets at: $DATASET_BASE ====="
find "$DATASET_BASE" -maxdepth 2 -name info.json -exec dirname {} \; 2>/dev/null

