#!/usr/bin/env python3
"""Multiprocess builder: split_data per-traj h5 -> per-skill LeRobot datasets.

Per trajectory (each worker, ONE file at a time -> tiny disk footprint):
    replay (rgbd) -> convert_single (append) -> delete rgbd intermediate

The full work list (all trajectories, capped --per-scene per scene) is built up
front and assigned a global episode_index (per skill) + task_index, then split
across N processes. Each process owns a FIXED slice. After all workers finish,
each skill dataset is finalized (fragments -> LeRobot meta).

Usage:
    python scripts/build_dataset_mp.py --split-root split_data \
        --out datasets/warehouse_fetch --jobs 6 --per-scene 30 --fps 15
"""
import argparse
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

TASK_OF = {
    "PickToBasketContDuffEnv": "move to shelf and pick Duff Beer Can to basket",
    "PickToBasketContFantaEnv": "move to shelf and pick Fanta to basket",
    "PickToBasketContStarsEnv": "move to shelf and pick Nestle Honey Stars to basket",
    "RestockBasketToShelfContDuffEnv": "pick Duff Beer Can from basket and place on shelf",
    "RestockBasketToShelfContFantaEnv": "pick Fanta from basket and place on shelf",
    "RestockBasketToShelfContStarsEnv": "pick Nestle Honey Stars from basket and place on shelf",
    "PickFromFloorBeansContEnv": "pick beans from floor and place in basket",
    "PickFromFloorSlamContEnv": "pick slam from floor and place in basket",
}

def collect_tasks(split_root: Path, out_root: Path, per_scene: int, fps: int):
    """Build the full work list up front.

    Walks split_data/<skill>/<Env>/traj_*/traj_*.h5, caps each scene (Env) at
    per_scene trajectories, assigns a per-skill global episode_index and a
    per-skill task_index. Returns (jobs, skills) where each job is a dict the
    worker can run standalone.
    """
    jobs = []
    skills = {}  # skill -> {"out": dir, "ep": next_index, "tasks": {task: idx}}
    for skill_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        skill = skill_dir.name
        sk = skills.setdefault(skill, {"out": out_root / skill, "ep": 0, "tasks": {}})
        for env_dir in sorted(p for p in skill_dir.iterdir() if p.is_dir()):
            env = env_dir.name
            task = TASK_OF.get(env, env)
            if task not in sk["tasks"]:
                sk["tasks"][task] = len(sk["tasks"])
            task_idx = sk["tasks"][task]
            traj_h5s = sorted(env_dir.rglob("traj_*.h5"))[:per_scene]
            for h5 in traj_h5s:
                jobs.append({
                    "h5": str(h5), "skill": skill, "out": str(sk["out"]),
                    "ep": sk["ep"], "task": task, "task_idx": task_idx, "fps": fps,
                })
                sk["ep"] += 1
    return jobs, skills


def process_one(job):
    """ONE trajectory: replay -> convert_single (append) -> delete rgbd. Worker-safe.

    Replay's tqdm frame progress is streamed to <base>.replay.log so a watcher
    (bash/build_progress.sh) can show per-trajectory frame extraction live. The
    log is deleted on success, kept on failure for debugging.
    """
    h5 = Path(job["h5"])
    log_path = h5.parent / f"{h5.stem}.replay.log"
    # 1. replay to rgbd; stream progress to the per-traj log (visible to watcher)
    with open(log_path, "w") as lf:
        r = subprocess.run(
            [PY, str(ROOT / "scripts/replay_trajectory.py"), "--traj-path", str(h5),
             "-b", "cpu", "-o", "rgbd", "--save-traj", "--allow-failure"],
            stdout=lf, stderr=subprocess.STDOUT, text=True)
    rgbd = sorted(h5.parent.glob(f"{h5.stem}.rgbd.*.h5"))
    if r.returncode != 0 or not rgbd:
        return ("FAIL_REPLAY", job["ep"], job["h5"])
    rgbd_path = rgbd[0]
    # 2. convert_single (append one episode with pre-assigned indices)
    with open(log_path, "a") as lf:
        c = subprocess.run(
            [PY, str(ROOT / "scripts/convert_skill_to_lerobot.py"),
             "--output-dir", job["out"], "--single-h5", str(rgbd_path),
             "--episode-index", str(job["ep"]), "--task", job["task"],
             "--task-index", str(job["task_idx"]), "--fps", str(job["fps"])],
            stdout=lf, stderr=subprocess.STDOUT, text=True)
    # 3. delete rgbd intermediate (always; source split h5 is kept)
    rgbd_path.unlink(missing_ok=True)
    if c.returncode != 0:
        return ("FAIL_CONVERT", job["ep"], job["h5"])
    log_path.unlink(missing_ok=True)  # success: drop log, watcher shows in-flight only
    return ("OK", job["ep"], job["h5"])


def main():
    ap = argparse.ArgumentParser(description="MP builder: split_data -> LeRobot datasets")
    ap.add_argument("--split-root", default="split_data")
    ap.add_argument("--out", default="datasets/warehouse_fetch")
    ap.add_argument("--jobs", type=int, default=4, help="parallel processes")
    ap.add_argument("--per-scene", type=int, default=30, help="trajectories per scene")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    split_root, out_root = Path(args.split_root), Path(args.out)
    jobs, skills = collect_tasks(split_root, out_root, args.per_scene, args.fps)
    if not jobs:
        print(f"No trajectories under {split_root}/ -- run split_episodes.py first")
        return
    print(f"Collected {len(jobs)} trajectories across {len(skills)} skills, "
          f"jobs={args.jobs}, per_scene={args.per_scene}")

    ok = fail_r = fail_c = 0
    with mp.Pool(processes=args.jobs) as pool:
        for status, ep, h5 in pool.imap_unordered(process_one, jobs):
            if status == "OK":
                ok += 1
            elif status == "FAIL_REPLAY":
                fail_r += 1; print(f"  FAIL_REPLAY ep={ep}: {h5}")
            else:
                fail_c += 1; print(f"  FAIL_CONVERT ep={ep}: {h5}")
            if (ok + fail_r + fail_c) % 10 == 0:
                print(f"  progress: {ok} ok / {fail_r} replay-fail / {fail_c} convert-fail "
                      f"of {len(jobs)}")
    print(f"Replay+convert done: {ok} ok, {fail_r} replay-fail, {fail_c} convert-fail")

    print("===== finalize =====")
    for skill, sk in skills.items():
        subprocess.run(
            [PY, str(ROOT / "scripts/convert_skill_to_lerobot.py"),
             "--output-dir", str(sk["out"]), "--finalize", "--fps", str(args.fps)],
            check=False)
    print(f"===== Done. Datasets at: {out_root} =====")


if __name__ == "__main__":
    main()

