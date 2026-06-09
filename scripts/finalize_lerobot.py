#!/usr/bin/env python3
"""Repair / finalize a LeRobot v2.1 dataset from per-episode fragments.

build_dataset_mp.py writes meta/_fragments/<idx>.json + episode_*.parquet + mp4s
per episode, but the meta assembly may be missing -- especially
meta/episodes_stats.jsonl (normalization stats), which the loader hard-requires.

This script (idempotent) scans fragments, reads each parquet for state/action
stats, samples each mp4 for per-channel image stats, fixes the global frame
`index`, and writes the full meta set:
    meta/info.json  meta/tasks.jsonl  meta/episodes.jsonl  meta/episodes_stats.jsonl

Usage:
    python scripts/finalize_lerobot.py --output-dir datasets/warehouse_fetch/pick_from_floor
    python scripts/finalize_lerobot.py --output-dir datasets/warehouse_fetch   # all skills
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import imageio.v2 as imageio

STATE_DIM, ACTION_DIM = 15, 13
HEAD_HW, WRIST_HW = (360, 640), (128, 128)
CAM_KEYS = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
STATE_NAMES = ["base_x", "base_y", "base_yaw", "torso_lift", "shoulder_pan",
               "shoulder_lift", "upperarm_roll", "elbow_flex", "forearm_roll",
               "wrist_flex", "wrist_roll", "head_pan", "head_tilt",
               "r_gripper", "l_gripper"]

def _stats_1d(arr):
    """min/max/mean/std/count for a (T, D) float array -> per-dim lists."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "count": [int(a.shape[0])],
    }


def _img_stats(mp4_path: Path, max_frames: int = 8):
    """Per-channel image stats normalized to [0,1], shape (3,1,1) as LeRobot wants.

    Samples up to max_frames evenly to keep finalize fast.
    """
    rd = imageio.get_reader(str(mp4_path))
    frames = []
    try:
        n = rd.count_frames()
        idxs = np.linspace(0, max(n - 1, 0), min(max_frames, max(n, 1)), dtype=int)
        for i in idxs:
            frames.append(np.asarray(rd.get_data(int(i)), dtype=np.float64) / 255.0)
    finally:
        rd.close()
    f = np.stack(frames)                       # (S, H, W, 3)
    def per_ch(fn): return fn(f, axis=(0, 1, 2)).reshape(3, 1, 1).tolist()
    return {
        "min": per_ch(np.min), "max": per_ch(np.max),
        "mean": per_ch(np.mean), "std": per_ch(np.std),
        "count": [int(f.shape[0])],
    }


def _col(tbl, key):
    """Read a parquet column as (T, D) ndarray (handles list-of-list cells)."""
    v = tbl[key]
    return np.array([row for row in v])


def finalize_one(output_dir: Path, fps_default: int = 15):
    """Assemble all meta files for one dataset from its _fragments + parquet + mp4."""
    meta_dir = output_dir / "meta"
    frag_dir = meta_dir / "_fragments"
    data_dir = output_dir / "data" / "chunk-000"
    video_dir = output_dir / "videos" / "chunk-000"
    frags = sorted(frag_dir.glob("*.json"), key=lambda p: int(p.stem))
    if not frags:
        print(f"  skip (no fragments): {output_dir}")
        return 0
    eps = [json.loads(p.read_text()) for p in frags]

    # contiguous task indices in first-seen order
    task_to_index = {}
    for e in eps:
        t = e["tasks"][0]
        task_to_index.setdefault(t, len(task_to_index))

    total_frames = 0
    dims = {"head": list(HEAD_HW), "wrist": list(WRIST_HW)}
    fps = eps[0].get("fps", fps_default)
    ep_meta, ep_stats = [], []
    for e in eps:
        idx, T = e["episode_index"], e["length"]
        dims = e.get("dims", dims)
        task_idx = task_to_index[e["tasks"][0]]
        pq_path = data_dir / f"episode_{idx:06d}.parquet"
        tbl = pq.read_table(pq_path).to_pydict()
        # fix global frame index + task_index, persist
        tbl["index"] = list(range(total_frames, total_frames + T))
        tbl["task_index"] = [task_idx] * T
        pq.write_table(__import__("pyarrow").table(tbl), pq_path)
        # per-episode stats (normalization-critical)
        st = {
            "observation.state": _stats_1d(_col(tbl, "observation.state")),
            "action": _stats_1d(_col(tbl, "action")),
            "timestamp": _stats_1d(np.array(tbl["timestamp"])),
            "frame_index": _stats_1d(np.array(tbl["frame_index"])),
            "episode_index": _stats_1d(np.array(tbl["episode_index"])),
            "index": _stats_1d(np.array(tbl["index"])),
            "task_index": _stats_1d(np.array(tbl["task_index"])),
        }
        for ck in CAM_KEYS:
            mp4 = video_dir / f"observation.images.{ck}" / f"episode_{idx:06d}.mp4"
            if mp4.exists():
                st[f"observation.images.{ck}"] = _img_stats(mp4)
        ep_meta.append({"episode_index": idx, "tasks": e["tasks"], "length": T})
        ep_stats.append({"episode_index": idx, "stats": st})
        total_frames += T

    _write_meta(meta_dir, fps, total_frames, len(eps), task_to_index, dims)
    _write_jsonl(meta_dir / "episodes.jsonl", ep_meta)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", ep_stats)
    print(f"  finalized {output_dir.name}: {len(eps)} eps, {total_frames} frames, "
          f"{len(task_to_index)} tasks, stats OK")
    return len(eps)


def _write_jsonl(path: Path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_meta(meta_dir: Path, fps, total_frames, total_episodes, task_to_index, dims):
    meta_dir.mkdir(parents=True, exist_ok=True)
    head_hw = dims.get("head", list(HEAD_HW))
    wrist_hw = dims.get("wrist", list(WRIST_HW))
    info = {
        "codebase_version": "v2.1", "robot_type": "fetch", "fps": fps,
        "total_episodes": total_episodes, "total_frames": total_frames,
        "total_tasks": len(task_to_index), "total_videos": total_episodes * len(CAM_KEYS),
        "total_chunks": 1, "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float64", "shape": [ACTION_DIM], "names": None},
            "observation.state": {"dtype": "float64", "shape": [STATE_DIM], "names": STATE_NAMES},
            **{f"observation.images.{ck}": {
                "dtype": "video",
                "shape": [*(head_hw if ck == "head_rgb" else wrist_hw), 3],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": float(fps), "video.codec": "h264",
                         "video.pix_fmt": "yuv420p", "has_audio": False},
            } for ck in CAM_KEYS},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "splits": {"train": f"0:{total_episodes}"},
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    _write_jsonl(meta_dir / "tasks.jsonl",
                 [{"task_index": i, "task": t}
                  for t, i in sorted(task_to_index.items(), key=lambda kv: kv[1])])


def main():
    ap = argparse.ArgumentParser(description="Repair/finalize LeRobot dataset meta")
    ap.add_argument("--output-dir", required=True,
                    help="a skill dataset dir, or a parent containing several")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    root = Path(args.output_dir)
    # a dataset has meta/_fragments; otherwise treat as parent of datasets
    targets = ([root] if (root / "meta" / "_fragments").is_dir()
               else sorted(p for p in root.iterdir()
                           if (p / "meta" / "_fragments").is_dir()))
    if not targets:
        print(f"No datasets with meta/_fragments under {root}")
        return
    total = 0
    for d in targets:
        total += finalize_one(d, args.fps)
    print(f"Done. Finalized {len(targets)} dataset(s), {total} episodes.")


if __name__ == "__main__":
    main()


