"""
Convert RoboBenchMart replayed rgbd h5 trajectories to per-SKILL LeRobot v2.1
datasets. One dataset per skill, merging multiple products, each episode keeping
its own task instruction. Caps episodes per product (default 30).

Usage:
    python scripts/convert_skill_to_lerobot.py \
        --output-dir datasets/warehouse_fetch/pick_to_basket \
        --item "generated_data/PickToBasketContDuffEnv|move to shelf and pick Duff Beer Can to basket" \
        --item "generated_data/PickToBasketContFantaEnv|move to shelf and pick Fanta to basket" \
        --max-per-item 30 --fps 15

Reads replayed h5 (replay_trajectory.py --obs-mode rgbd --save-traj), real layout:
  traj_{i}/
    obs/agent/qpos                       (T+1, 15)  -> observation.state[:T]
    obs/sensor_data/head_camera/rgb      (T+1, 360, 640, 3)
    obs/sensor_data/fetch_hand/rgb       (T+1, 128, 128, 3)
    actions                              (T, 13)
"""
import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import imageio.v2 as imageio  # uses bundled imageio-ffmpeg, no system ffmpeg needed

HEAD_HW = (360, 640)
WRIST_HW = (128, 128)
STATE_DIM = 15
ACTION_DIM = 13


def read_instruction(traj) -> str | None:
    """Decode per-episode language instruction from obs/extra, if present."""
    try:
        ex = traj["obs"]["extra"]
        b = np.array(ex["language_instruction_bytes"][0])
        return bytes(b[b > 0]).decode("utf-8", "ignore").strip() or None
    except Exception:
        return None


def extract_episode(traj):
    """Return (state[T,15], action[T,13], head[T,...], wrist[T,...]) aligned to T."""
    actions = np.array(traj["actions"])                    # (T, 13)
    T = len(actions)
    qpos = np.array(traj["obs"]["agent"]["qpos"])[:T]      # (T, 15), drop trailing
    head = np.array(traj["obs"]["sensor_data"]["head_camera"]["rgb"])[:T]
    wrist = np.array(traj["obs"]["sensor_data"]["fetch_hand"]["rgb"])[:T]
    return qpos.astype(np.float64), actions.astype(np.float64), head, wrist


def encode_frames_to_mp4(frames: np.ndarray, output_path: Path, fps: int = 15):
    """Encode (T, H, W, 3) uint8 frames to mp4 via imageio-ffmpeg (bundled ffmpeg).

    macro_block_size=1 keeps exact H,W (no auto-resize to multiples of 16).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, C = frames.shape
    assert C == 3, f"expected 3 channels, got {C}"
    writer = imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=8,
        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    for t in range(T):
        writer.append_data(np.ascontiguousarray(frames[t]))
    writer.close()


def write_parquet(parquet_data, path):
    cols = {}
    for k, v in parquet_data.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            cols[k] = pa.array([row.tolist() for row in v])
        elif isinstance(v, np.ndarray):
            cols[k] = pa.array(v.tolist())
        else:
            cols[k] = pa.array(v)
    pq.write_table(pa.table(cols), path)


def find_h5_files(item_dir: Path):
    """All replayed rgbd h5 under an env dir (incl. .0/.1/... proc subdirs)."""
    files = sorted(item_dir.rglob("*.rgbd.*.h5"))
    return files


def convert_skill(items, output_dir: Path, max_per_item: int, fps: int):
    """items: list of (env_dir: Path, fallback_task: str)."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    data_dir = output_dir / "data" / "chunk-000"
    video_dir = output_dir / "videos" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    cam_keys = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
    for ck in cam_keys:
        (video_dir / f"observation.images.{ck}").mkdir(parents=True, exist_ok=True)

    task_to_index = {}
    episode_meta = []
    episode_index = 0
    total_frames = 0
    dims = {"head": list(HEAD_HW), "wrist": list(WRIST_HW)}  # fallback; overwritten by real frames

    for env_dir, fallback_task in items:
        env_dir = Path(env_dir)
        h5_files = find_h5_files(env_dir)
        if not h5_files:
            print(f"  WARNING: no rgbd h5 under {env_dir}, skipping")
            continue
        taken = 0
        for h5_path in h5_files:
            if taken >= max_per_item:
                break
            with h5py.File(h5_path, "r") as f:
                traj_ids = sorted([k for k in f.keys() if k.startswith("traj_")],
                                  key=lambda s: int(s.split("_")[1]))
                for tid in traj_ids:
                    if taken >= max_per_item:
                        break
                    episode_index, total_frames, taken, dims = _write_episode(
                        f[tid], output_dir, data_dir, video_dir, fps, fallback_task,
                        task_to_index, episode_meta, episode_index, total_frames, taken, dims)

    _write_meta(output_dir, fps, total_frames, episode_index, task_to_index, dims)
    with open(output_dir / "meta" / "episodes.jsonl", "w") as f:
        for ep in episode_meta:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print(f"\n{output_dir.name}: {episode_index} episodes, {total_frames} frames, "
          f"{len(task_to_index)} tasks")
    return episode_index


def _write_episode(traj, output_dir, data_dir, video_dir, fps, fallback_task,
                   task_to_index, episode_meta, episode_index, total_frames, taken, dims):
    try:
        state, action, head, wrist = extract_episode(traj)
    except Exception as e:
        print(f"  skip traj: {e}")
        return episode_index, total_frames, taken, dims
    T = len(action)
    if T < 2:
        return episode_index, total_frames, taken, dims

    # capture actual frame dims (H, W) so info.json matches reality
    dims = {"head": [head.shape[1], head.shape[2]],
            "wrist": [wrist.shape[1], wrist.shape[2]]}

    task = read_instruction(traj) or fallback_task
    if task not in task_to_index:
        task_to_index[task] = len(task_to_index)
    task_idx = task_to_index[task]

    ep = f"episode_{episode_index:06d}"
    encode_frames_to_mp4(head, video_dir / "observation.images.head_rgb" / f"{ep}.mp4", fps)
    lw = video_dir / "observation.images.left_wrist_rgb" / f"{ep}.mp4"
    encode_frames_to_mp4(wrist, lw, fps)
    shutil.copy2(lw, video_dir / "observation.images.right_wrist_rgb" / f"{ep}.mp4")

    write_parquet({
        "observation.state": state,
        "action": action,
        "episode_index": np.full(T, episode_index, dtype=np.int64),
        "frame_index": np.arange(T, dtype=np.int64),
        "index": np.arange(total_frames, total_frames + T, dtype=np.int64),
        "timestamp": (np.arange(T, dtype=np.float32) / fps),
        "task_index": np.full(T, task_idx, dtype=np.int64),
    }, data_dir / f"{ep}.parquet")

    episode_meta.append({"episode_index": episode_index, "tasks": [task], "length": int(T)})
    return episode_index + 1, total_frames + T, taken + 1, dims


def _write_meta(output_dir: Path, fps, total_frames, total_episodes, task_to_index, dims):
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    head_hw = dims.get("head", list(HEAD_HW))
    wrist_hw = dims.get("wrist", list(WRIST_HW))
    state_names = ["base_x", "base_y", "base_yaw", "torso_lift", "shoulder_pan",
                   "shoulder_lift", "upperarm_roll", "elbow_flex", "forearm_roll",
                   "wrist_flex", "wrist_roll", "head_pan", "head_tilt",
                   "r_gripper", "l_gripper"]
    info = {
        "codebase_version": "v2.1",
        "robot_type": "fetch",
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_chunks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float64", "shape": [ACTION_DIM], "names": None},
            "observation.state": {"dtype": "float64", "shape": [STATE_DIM], "names": state_names},
            "observation.images.head_rgb": {"dtype": "video", "shape": [*head_hw, 3],
                                            "names": ["height", "width", "channels"]},
            "observation.images.left_wrist_rgb": {"dtype": "video", "shape": [*wrist_hw, 3],
                                                  "names": ["height", "width", "channels"]},
            "observation.images.right_wrist_rgb": {"dtype": "video", "shape": [*wrist_hw, 3],
                                                   "names": ["height", "width", "channels"]},
        },
        "splits": {"train": f"0:{total_episodes}"},
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task, idx in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")
    return meta_dir


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert RoboBenchMart rgbd h5 to a per-skill LeRobot dataset")
    p.add_argument("--output-dir", required=True, help="Skill dataset output dir")
    p.add_argument("--item", action="append", required=True, dest="items",
                   help="'<env_dir>|<fallback_task>' (repeatable, one per product)")
    p.add_argument("--max-per-item", type=int, default=30)
    p.add_argument("--fps", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    items = []
    for spec in args.items:
        env_dir, _, task = spec.partition("|")
        items.append((env_dir.strip(), task.strip()))
    output_dir = Path(args.output_dir)
    convert_skill(items, output_dir, args.max_per_item, args.fps)
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()




