#!/bin/bash
# Step 1: Replay trajectories with rgbd observation mode (for LeRobot conversion)
# Step 2: Convert to LeRobot format
# Run AFTER run_mp_vla_training.sh completes

set -e

P=/home/lh/software/miniconda3/envs/robort_mart/bin/python
P_VLA=/home/lh/software/miniconda3/envs/robort_mart/bin/python
SCENE_DIR_PICK_TO_BASKET=demo_envs/pick_to_basket
SCENE_DIR_PICK_FROM_FLOOR=demo_envs/pick_from_floor
OUTPUT_BASE=/home/lh/VLA/GalaxeaVLA-main/datasets/warehouse_fetch

# ============================================================
# Step 1: Replay with rgbd obs mode to get camera images
# ============================================================
echo "===== Step 1: Replaying trajectories with rgbd ====="

TRAJ_DIRS=(
    "$SCENE_DIR_PICK_TO_BASKET/demos/PickToBasketContNiveaEnv/pick_to_basket_nivea"
    "$SCENE_DIR_PICK_TO_BASKET/demos/PickToBasketContFantaEnv/pick_to_basket_fanta"
    "$SCENE_DIR_PICK_TO_BASKET/demos/PickToBasketContStarsEnv/pick_to_basket_stars"
    "$SCENE_DIR_PICK_TO_BASKET/demos/RestockBasketToShelfContNiveaEnv/restock_nivea"
    "$SCENE_DIR_PICK_TO_BASKET/demos/RestockBasketToShelfContFantaEnv/restock_fanta"
    "$SCENE_DIR_PICK_TO_BASKET/demos/RestockBasketToShelfContStarsEnv/restock_stars"
    "$SCENE_DIR_PICK_FROM_FLOOR/demos/PickFromFloorBeansContEnv/pick_from_floor_beans"
    "$SCENE_DIR_PICK_FROM_FLOOR/demos/PickFromFloorSlamContEnv/pick_from_floor_slam"
)

for traj_dir in "${TRAJ_DIRS[@]}"; do
    # Find the merged h5 file
    h5_file=$(ls $traj_dir/*.h5 2>/dev/null | head -1)
    if [ -z "$h5_file" ]; then
        echo "WARNING: No h5 file found in $traj_dir, skipping"
        continue
    fi
    echo "Replaying: $h5_file"
    $P scripts/replay_trajectory.py --traj_path "$h5_file" -b cpu -o rgbd --save-traj
done

# ============================================================
# Step 2: Convert to LeRobot format
# ============================================================
echo "===== Step 2: Converting to LeRobot format ====="

# Skill: pick_to_basket
for ITEM in nivea fanta stars; do
    ENV_NAME="PickToBasketCont${ITEM^}Env"
    TRAJ_NAME="pick_to_basket_${ITEM}"
    H5_DIR="$SCENE_DIR_PICK_TO_BASKET/demos/${ENV_NAME}/${TRAJ_NAME}"
    if [ -d "$H5_DIR" ]; then
        echo "Converting: $H5_DIR"
        cd /home/lh/VLA/GalaxeaVLA-main
        $P_VLA scripts/convert_robobenchmart_to_lerobot.py \
            --h5-dir "/home/lh/VLA/RoboBenchMart-main/$H5_DIR" \
            --output-dir "$OUTPUT_BASE/pick_to_basket_${ITEM}" \
            --task "pick ${ITEM} from shelf and place in basket" \
            --fps 15
        cd /home/lh/VLA/RoboBenchMart-main
    fi
done

# Skill: restock_basket_to_shelf
for ITEM in nivea fanta stars; do
    ENV_NAME="RestockBasketToShelfCont${ITEM^}Env"
    TRAJ_NAME="restock_${ITEM}"
    H5_DIR="$SCENE_DIR_PICK_TO_BASKET/demos/${ENV_NAME}/${TRAJ_NAME}"
    if [ -d "$H5_DIR" ]; then
        echo "Converting: $H5_DIR"
        cd /home/lh/VLA/GalaxeaVLA-main
        $P_VLA scripts/convert_robobenchmart_to_lerobot.py \
            --h5-dir "/home/lh/VLA/RoboBenchMart-main/$H5_DIR" \
            --output-dir "$OUTPUT_BASE/restock_${ITEM}" \
            --task "pick ${ITEM} from basket and place on shelf" \
            --fps 15
        cd /home/lh/VLA/RoboBenchMart-main
    fi
done

# Skill: pick_from_floor
for ITEM in beans slam; do
    ENV_NAME="PickFromFloor${ITEM^}ContEnv"
    TRAJ_NAME="pick_from_floor_${ITEM}"
    H5_DIR="$SCENE_DIR_PICK_FROM_FLOOR/demos/${ENV_NAME}/${TRAJ_NAME}"
    if [ -d "$H5_DIR" ]; then
        echo "Converting: $H5_DIR"
        cd /home/lh/VLA/GalaxeaVLA-main
        $P_VLA scripts/convert_robobenchmart_to_lerobot.py \
            --h5-dir "/home/lh/VLA/RoboBenchMart-main/$H5_DIR" \
            --output-dir "$OUTPUT_BASE/pick_from_floor_${ITEM}" \
            --task "pick ${ITEM} from floor and place in basket" \
            --fps 15
        cd /home/lh/VLA/RoboBenchMart-main
    fi
done

echo "===== All conversions complete ====="
echo "Datasets saved to: $OUTPUT_BASE"
