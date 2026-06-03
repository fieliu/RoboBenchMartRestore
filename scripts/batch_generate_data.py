#!/usr/bin/env python3
"""Batch motion planning data generation for VLA fine-tuning.

Generates successful trajectories for 3 manipulation skills:
  - pick_to_basket: PickToBasketCont{Nivea,Fanta,Stars}Env
  - restock_basket_to_shelf: RestockBasketToShelfCont{Nivea,Fanta,Stars}Env
  - pick_from_floor: PickFromFloorCont{Slam,Beans}Env

Each env runs until N successful trajectories are collected.
Errors are caught and skipped — the script never aborts mid-batch.
"""

import os
import sys
import time
import json
import traceback
import subprocess
import datetime

PYTHON = "/home/lh/software/miniconda3/envs/robort_mart/bin/python"
PROJECT_ROOT = "/home/lh/VLA/RoboBenchMart-main"
SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_mp.py")
LOG_DIR = os.path.join(PROJECT_ROOT, "generated_envs", "data_gen_logs")

TASKS = [
    {
        "skill": "pick_to_basket",
        "envs": [
            {"name": "PickToBasketContNiveaEnv", "scene": "demo_envs/pick_to_basket"},
            {"name": "PickToBasketContFantaEnv", "scene": "demo_envs/pick_to_basket"},
            {"name": "PickToBasketContStarsEnv", "scene": "demo_envs/pick_to_basket"},
        ],
        "num_success": 30,
    },
    {
        "skill": "restock_basket_to_shelf",
        "envs": [
            {"name": "RestockBasketToShelfContNiveaEnv", "scene": "demo_envs/pick_to_basket"},
            {"name": "RestockBasketToShelfContFantaEnv", "scene": "demo_envs/pick_to_basket"},
            {"name": "RestockBasketToShelfContStarsEnv", "scene": "demo_envs/pick_to_basket"},
        ],
        "num_success": 30,
    },
    {
        "skill": "pick_from_floor",
        "envs": [
            {"name": "PickFromFloorContSlamEnv", "scene": "demo_envs/pick_from_floor"},
            {"name": "PickFromFloorContBeansEnv", "scene": "demo_envs/pick_from_floor"},
        ],
        "num_success": 30,
    },
]


def run_one(env_name, scene_dir, num_success, traj_name):
    cmd = [
        PYTHON, "-u", SCRIPT,
        "-e", env_name,
        "--scene-dir", scene_dir,
        "-n", str(num_success),
        "-b", "cpu",
        "--only-count-success",
        "--traj-name", traj_name,
    ]
    print(f"\n{'='*60}")
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=False,
            timeout=7200,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 7200s for {env_name}")
        return -1
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return -1


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"batch_gen_{date_str}.log")

    summary = []
    total_start = time.time()

    for task in TASKS:
        skill = task["skill"]
        print(f"\n{'#'*60}")
        print(f"# Skill: {skill}")
        print(f"{'#'*60}")

        for env_cfg in task["envs"]:
            env_name = env_cfg["name"]
            scene_dir = env_cfg["scene"]
            num_success = task["num_success"]
            traj_name = f"{skill}_{env_name}_{date_str}"

            env_start = time.time()
            retcode = run_one(env_name, scene_dir, num_success, traj_name)
            elapsed = time.time() - env_start

            status = "OK" if retcode == 0 else f"FAIL(rc={retcode})"
            summary.append({
                "skill": skill,
                "env": env_name,
                "scene": scene_dir,
                "traj_name": traj_name,
                "num_success_target": num_success,
                "status": status,
                "elapsed_sec": round(elapsed, 1),
            })
            print(f"  [{status}] {env_name} took {elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE in {total_elapsed:.1f}s")
    print(f"{'='*60}")
    for s in summary:
        print(f"  [{s['status']}] {s['env']} ({s['elapsed_sec']}s)")

    with open(log_path, "w") as f:
        json.dump({
            "date": date_str,
            "total_elapsed_sec": round(total_elapsed, 1),
            "tasks": summary,
        }, f, indent=2)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
