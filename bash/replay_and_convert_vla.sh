#!/bin/bash
# Step 1: replay each generated trajectory with rgbd obs mode (to get camera images)
# Step 2: convert to LeRobot format for VLA training.
# Run AFTER run_mp_vla_training.sh. Reads from the clean OUTPUT_ROOT, not demo_envs.

set -e

P=${PYTHON:-/home/lh/software/miniconda3/envs/robort_mart/bin/python}
P_VLA=${PYTHON_VLA:-$P}
RB_ROOT=${RB_ROOT:-/home/lh/VLA/RoboBenchMart-main}
VLA_ROOT=${VLA_ROOT:-/home/lh/VLA/GalaxeaVLA-main}
OUTPUT_ROOT=${OUTPUT_ROOT:-generated_data}          # where run_mp wrote h5 (relative to RB_ROOT)
DATASET_BASE=${DATASET_BASE:-$VLA_ROOT/datasets/warehouse_fetch}
FPS=15

# env | traj-name | task instruction
JOBS=(
  "PickToBasketContDuffEnv|pick_to_basket_duff|pick duff from shelf and place in basket"
  "PickToBasketContFantaEnv|pick_to_basket_fanta|pick fanta from shelf and place in basket"
  "PickToBasketContStarsEnv|pick_to_basket_stars|pick stars from shelf and place in basket"
  "RestockBasketToShelfContDuffEnv|restock_duff|pick duff from basket and place on shelf"
  "RestockBasketToShelfContFantaEnv|restock_fanta|pick fanta from basket and place on shelf"
  "RestockBasketToShelfContStarsEnv|restock_stars|pick stars from basket and place on shelf"
  "PickFromFloorBeansContEnv|pick_from_floor_beans|pick beans from floor and place in basket"
  "PickFromFloorSlamContEnv|pick_from_floor_slam|pick slam from floor and place in basket"
)

echo "===== Step 1: replay trajectories with rgbd ====="
cd "$RB_ROOT"
for job in "${JOBS[@]}"; do
  IFS='|' read -r ENV TRAJ TASK <<< "$job"
  traj_dir="$OUTPUT_ROOT/$ENV/$TRAJ"
  h5_file=$(ls "$traj_dir"/*.h5 2>/dev/null | head -1)
  if [ -z "$h5_file" ]; then
    echo "WARNING: no h5 in $traj_dir, skipping"; continue
  fi
  echo "Replaying: $h5_file"
  $P scripts/replay_trajectory.py --traj-path "$h5_file" -b cpu -o rgbd --save-traj
done

echo "===== Step 2: convert to LeRobot ====="
for job in "${JOBS[@]}"; do
  IFS='|' read -r ENV TRAJ TASK <<< "$job"
  h5_dir="$RB_ROOT/$OUTPUT_ROOT/$ENV/$TRAJ"
  if [ -d "$h5_dir" ]; then
    echo "Converting: $h5_dir  ->  $DATASET_BASE/$TRAJ"
    cd "$VLA_ROOT"
    $P_VLA scripts/convert_robobenchmart_to_lerobot.py \
        --h5-dir "$h5_dir" \
        --output-dir "$DATASET_BASE/$TRAJ" \
        --task "$TASK" \
        --fps "$FPS"
    cd "$RB_ROOT"
  fi
done

echo "===== All conversions complete ====="
echo "Datasets: $DATASET_BASE"
