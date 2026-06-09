
抓取到篮子(场景 demo_envs/pick_to_basket)
P=/home/lh/software/miniconda3/envs/robort_mart/bin/python
$P scripts/run_mp.py -e PickToBasketContNiveaEnv  --scene-dir demo_envs/pick_to_basket -n 1 -b cpu
$P scripts/run_mp.py -e PickToBasketContStarsEnv  --scene-dir demo_envs/pick_to_basket -n 1 -b cpu
$P scripts/run_mp.py -e PickToBasketContFantaEnv  --scene-dir demo_envs/pick_to_basket -n 1 -b cpu
$P scripts/run_mp.py -e PickToBasketContDuffEnv   --scene-dir demo_envs/pick_to_basket -n 1 -b cpu


篮子放回货架(场景 generated_envs/restock_scene)
P=/home/lh/software/miniconda3/envs/robort_mart/bin/python
$P scripts/run_mp.py -e RestockBasketToShelfContNiveaEnv --scene-dir demo_envs/pick_to_basket -n 1 -b cpu
$P scripts/run_mp.py -e RestockBasketToShelfContFantaEnv --scene-dir demo_envs/pick_to_basket -n 1 -b cpu
$P scripts/run_mp.py -e RestockBasketToShelfContStarsEnv --scene-dir demo_envs/pick_to_basket -n 1 -b cpu
$P scripts/run_mp.py -e RestockBasketToShelfContDuffEnv  --scene-dir demo_envs/pick_to_basket -n 1 -b cpu

P=/home/lh/software/miniconda3/envs/robort_mart/bin/python

# 篮子放 1 个目标 + 3 个干扰品(共4件)
$P scripts/run_mp.py -e RestockBasketToShelfContEnv      --scene-dir demo_envs/pick_to_basket -n 1 -b cpu --env-kwargs '{"num_basket_distractors": 3}'
$P scripts/run_mp.py -e RestockBasketToShelfContNiveaEnv --scene-dir demo_envs/pick_to_basket -n 1 -b cpu --env-kwargs '{"num_basket_distractors": 3}'
$P scripts/run_mp.py -e RestockBasketToShelfContFantaEnv --scene-dir demo_envs/pick_to_basket -n 1 -b cpu --env-kwargs '{"num_basket_distractors": 3}'
$P scripts/run_mp.py -e RestockBasketToShelfContStarsEnv --scene-dir demo_envs/pick_to_basket -n 1 -b cpu --env-kwargs '{"num_basket_distractors": 3}'
$P scripts/run_mp.py -e RestockBasketToShelfContDuffEnv  --scene-dir demo_envs/pick_to_basket -n 1 -b cpu --env-kwargs '{"num_basket_distractors": 3}'

 --no-retry --save-video

地面拾取(场景 demo_envs/pick_from_floor)
$P scripts/run_mp.py -e PickFromFloorSlamContEnv  --scene-dir demo_envs/pick_from_floor -n 1 -b cpu
$P scripts/run_mp.py -e PickFromFloorBeansContEnv --scene-dir demo_envs/pick_from_floor -n 1 -b cpu
$P scripts/run_mp.py -e PickFromFloorFantaContEnv --scene-dir demo_envs/pick_from_floor -n 1 -b cpu
$P scripts/run_mp.py -e PickFromFloorDuffContEnv  --scene-dir demo_envs/pick_from_floor -n 1 -b cpu


MAX_PER_ITEM=30 REPLAY_JOBS=8 nohup bash bash/build_skill_datasets.sh > generated_data/build.log 2>&1

watch -n 5 bash bash/replay_progress.sh

nohup python scripts/build_dataset_mp.py --split-root split_data --out datasets/warehouse_fetch --jobs 6 --per-scene 30 --fps 15 > split_data/build.log 2>&1 &
watch -n 5 bash bash/build_progress.sh
