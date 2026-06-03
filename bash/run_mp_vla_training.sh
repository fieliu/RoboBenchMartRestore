#!/bin/bash
# Generate training trajectories for VLA fine-tuning
# 3 skills, 8 envs, 30 successful trajectories each (~240 total)
# Uses CPU backend with 2 parallel processes for speed
# Usage: bash bash/run_mp_vla_training.sh

set -e

P=/home/lh/software/miniconda3/envs/robort_mart/bin/python
NUM_TRAJ=30
NUM_PROCS=4
SCENE_DIR_PICK_TO_BASKET=demo_envs/pick_to_basket
SCENE_DIR_PICK_FROM_FLOOR=demo_envs/pick_from_floor

echo "===== Skill 1: pick_to_basket (Nivea, Fanta, Stars) ====="

$P scripts/run_mp.py -e PickToBasketContNiveaEnv \
    --scene-dir $SCENE_DIR_PICK_TO_BASKET \
    -b cpu --only-count-success --num-procs $NUM_PROCS -n $NUM_TRAJ \
    --traj-name pick_to_basket_nivea

$P scripts/run_mp.py -e PickToBasketContFantaEnv \
    --scene-dir $SCENE_DIR_PICK_TO_BASKET \
    -b cpu --only-count-success --num-procs $NUM_PROCS -n $NUM_TRAJ \
    --traj-name pick_to_basket_fanta

$P scripts/run_mp.py -e PickToBasketContStarsEnv \
    --scene-dir $SCENE_DIR_PICK_TO_BASKET \
    -b cpu --only-count-success --num-procs $NUM_PROCS -n $NUM_TRAJ \
    --traj-name pick_to_basket_stars

echo "===== Skill 2: restock_basket_to_shelf (Nivea, Fanta, Stars) ====="

$P scripts/run_mp.py -e RestockBasketToShelfContNiveaEnv \
    --scene-dir $SCENE_DIR_PICK_TO_BASKET \
    -b cpu --only-count-success --num-procs $NUM_PROCS -n $NUM_TRAJ \
    --traj-name restock_nivea

$P scripts/run_mp.py -e RestockBasketToShelfContFantaEnv \
    --scene-dir $SCENE_DIR_PICK_TO_BASKET \
    -b cpu --only-count-success --num-procs $NUM_PROCS -n $NUM_TRAJ \
    --traj-name restock_fanta

$P scripts/run_mp.py -e RestockBasketToShelfContStarsEnv \
    --scene-dir $SCENE_DIR_PICK_TO_BASKET \
    -b cpu --only-count-success --num-procs $NUM_PROCS -n $NUM_TRAJ \
    --traj-name restock_stars

echo "===== Skill 3: pick_from_floor (Beans, Slam) ====="

$P scripts/run_mp.py -e PickFromFloorBeansContEnv \
    --scene-dir $SCENE_DIR_PICK_FROM_FLOOR \
    -b cpu --only-count-success --num-procs $NUM_PROCS -n $NUM_TRAJ \
    --traj-name pick_from_floor_beans

$P scripts/run_mp.py -e PickFromFloorSlamContEnv \
    --scene-dir $SCENE_DIR_PICK_FROM_FLOOR \
    -b cpu --only-count-success --num-procs $NUM_PROCS -n $NUM_TRAJ \
    --traj-name pick_from_floor_slam

echo "===== All trajectory generation complete ====="
