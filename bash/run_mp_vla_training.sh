#!/bin/bash
# Generate training trajectories for VLA fine-tuning.
# 3 skills, 8 envs, 30 successful trajectories each (~240 total).
# CPU backend, parallel processes for speed.
# Output goes to a CLEAN dir (generated_data/), NOT into demo_envs/.
# A per-env log records how many episodes ran to collect 30 successes
# (reflects MP success rate -- deliverable #5).
# Usage: bash bash/run_mp_vla_training.sh

set -e

P=${PYTHON:-/home/lh/software/miniconda3/envs/robort_mart/bin/python}
NUM_TRAJ=${NUM_TRAJ:-30}     # TOTAL trajectories wanted per env (across all procs)
NUM_PROCS=${NUM_PROCS:-4}
# run_mp.py's -n is PER-PROCESS, so divide to hit NUM_TRAJ total (ceil).
PER_PROC=$(( (NUM_TRAJ + NUM_PROCS - 1) / NUM_PROCS ))
OUTPUT_ROOT=${OUTPUT_ROOT:-generated_data}
LOG_DIR="$OUTPUT_ROOT/gen_logs"
mkdir -p "$LOG_DIR"
echo "Target: $NUM_TRAJ traj/env = $PER_PROC per-proc x $NUM_PROCS procs"

SCENE_PTB=demo_envs/pick_to_basket
SCENE_PFF=demo_envs/pick_from_floor

# env | scene-dir | traj-name
# pick_to_basket Duff/Fanta already generated (60 each) under generated_data/.
# Only Stars pick_to_basket is (re)generated here -- it was interrupted before.
# Plus restock x3 and floor x2. Nivea excluded everywhere: shelf board_idxs=2
# (high), top-down grasp IK blocked by the shelf above.
JOBS=(
  "PickToBasketContStarsEnv|$SCENE_PTB|pick_to_basket_stars"
  "RestockBasketToShelfContDuffEnv|$SCENE_PTB|restock_duff"
  "RestockBasketToShelfContFantaEnv|$SCENE_PTB|restock_fanta"
  "RestockBasketToShelfContStarsEnv|$SCENE_PTB|restock_stars"
  "PickFromFloorBeansContEnv|$SCENE_PFF|pick_from_floor_beans"
  "PickFromFloorSlamContEnv|$SCENE_PFF|pick_from_floor_slam"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r ENV SCENE TRAJ <<< "$job"
  echo "===== Generating $TRAJ ($ENV) -> $OUTPUT_ROOT ====="
  # No --only-count-success: solver returning non-(-1) is accepted and saved
  # (same lenient criterion as the verified single `-n 1 -b cpu` run). The
  # strict info['success'] gate made MP almost never collect a trajectory
  # because of gripper-vs-decorative-shelf collisions on the post-grasp move.
  $P scripts/run_mp.py -e "$ENV" \
      --scene-dir "$SCENE" -r ds_fetch_basket \
      -b cpu --num-procs "$NUM_PROCS" -n "$PER_PROC" \
      --traj-name "$TRAJ" \
      --output-root "$OUTPUT_ROOT" 2>&1 | tee "$LOG_DIR/${TRAJ}.log"
done

echo "===== All trajectory generation complete ====="
echo "Trajectories: $OUTPUT_ROOT/<EnvName>/<traj_name>/"
echo "Per-env logs: $LOG_DIR/"
