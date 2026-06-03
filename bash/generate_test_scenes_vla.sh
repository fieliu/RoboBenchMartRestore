#!/bin/bash
# Generate test scenes for closed-loop evaluation
# These use different seeds and layouts from training scenes
# Usage: bash bash/generate_test_scenes_vla.sh

set -e

P=/home/lh/software/miniconda3/envs/robort_mart/bin/python

echo "===== Generating test scenes for pick_to_basket + restock ====="
$P scripts/generate_scene_continuous.py ds_continuous=test_unseen_scenes_pick_to_basket_1
$P scripts/generate_scene_continuous.py ds_continuous=test_unseen_scenes_pick_to_basket_2
$P scripts/generate_scene_continuous.py ds_continuous=test_unseen_items_pick_to_basket_1
$P scripts/generate_scene_continuous.py ds_continuous=test_unseen_items_pick_to_basket_2

echo "===== Generating test scenes for pick_from_floor ====="
$P scripts/generate_scene_continuous.py ds_continuous=test_unseen_scenes_pick_from_floor_1
$P scripts/generate_scene_continuous.py ds_continuous=test_unseen_scenes_pick_from_floor_2
$P scripts/generate_scene_continuous.py ds_continuous=test_unseen_items_pick_from_floor_1
$P scripts/generate_scene_continuous.py ds_continuous=test_unseen_items_pick_from_floor_2

echo "===== Test scene generation complete ====="
echo "Test scenes saved to:"
echo "  demo_envs/test_unseen_scenes_pick_to_basket/"
echo "  demo_envs/test_unseen_items_pick_to_basket/"
echo "  demo_envs/test_unseen_scenes_pick_from_floor/"
echo "  demo_envs/test_unseen_items_pick_from_floor/"
