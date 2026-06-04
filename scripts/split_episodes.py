#!/usr/bin/env python3
"""Split multi-episode RoboBenchMart h5 trajectories into per-trajectory files.

Input  (motion-planning output, MANY episodes per file):
    generated_data/<Env>/<env>.<shard>/<name>.h5   + sibling .json
Output (ONE trajectory per dir, grouped by skill/scene):
    split_data/<skill>/<Env>/traj_<NNNN>/traj_<NNNN>.h5    (single group "traj_0")
                                          traj_<NNNN>.json  (episodes=[one episode])

Each output is self-contained and replayable: episode_id is reset to 0 and the h5
group is renamed traj_0, matching replay_trajectory.py's traj_{episode_id} lookup.
"""
import argparse
import json
from pathlib import Path

import h5py

SKILL_OF = {
    "PickToBasketContDuffEnv": "pick_to_basket",
    "PickToBasketContFantaEnv": "pick_to_basket",
    "PickToBasketContStarsEnv": "pick_to_basket",
    "RestockBasketToShelfContDuffEnv": "restock_basket_to_shelf",
    "RestockBasketToShelfContFantaEnv": "restock_basket_to_shelf",
    "RestockBasketToShelfContStarsEnv": "restock_basket_to_shelf",
    "PickFromFloorBeansContEnv": "pick_from_floor",
    "PickFromFloorSlamContEnv": "pick_from_floor",
}


def find_source_h5(env_dir: Path):
    """Original (non-rgbd) h5 shards under an env dir, sorted for stable indexing."""
    return sorted(p for p in env_dir.rglob("*.h5") if ".rgbd." not in p.name)

def _copy_group(src_grp, dst_grp):
    """Recursively deep-copy an h5 group's datasets/attrs into dst_grp."""
    for k, v in src_grp.attrs.items():
        dst_grp.attrs[k] = v
    for key in src_grp:
        item = src_grp[key]
        if isinstance(item, h5py.Group):
            _copy_group(item, dst_grp.create_group(key))
        else:
            dst_grp.create_dataset(key, data=item[()], compression=item.compression)


def split_one_h5(h5_path: Path, out_root: Path, skill: str, env: str, start_idx: int):
    """Split every traj_* group in one source h5 into its own traj_<NNNN> dir.

    Returns the next global index for this env (so shards accumulate cleanly).
    """
    json_path = h5_path.with_suffix(".json")
    meta = json.loads(json_path.read_text())
    eps_by_id = {ep["episode_id"]: ep for ep in meta["episodes"]}
    written = []
    with h5py.File(h5_path, "r") as f:
        traj_ids = sorted((k for k in f if k.startswith("traj_")),
                          key=lambda s: int(s.split("_")[1]))
        for tid in traj_ids:
            ep_id = int(tid.split("_")[1])
            if ep_id not in eps_by_id:
                continue
            idx = start_idx + len(written)
            traj_dir = out_root / skill / env / f"traj_{idx:04d}"
            traj_dir.mkdir(parents=True, exist_ok=True)
            base = traj_dir / f"traj_{idx:04d}"
            # h5: single group renamed traj_0 (matches replay's traj_{episode_id})
            with h5py.File(f"{base}.h5", "w") as out:
                _copy_group(f[tid], out.create_group("traj_0"))
            # json: one episode, episode_id reset to 0
            ep = dict(eps_by_id[ep_id]); ep["episode_id"] = 0
            single = {k: meta[k] for k in meta if k != "episodes"}
            single["episodes"] = [ep]
            Path(f"{base}.json").write_text(json.dumps(single))
            written.append(str(base) + ".h5")
    return start_idx + len(written), written


def main():
    ap = argparse.ArgumentParser(description="Split multi-episode h5 into per-trajectory files")
    ap.add_argument("--src", default="generated_data", help="source root with <Env>/ dirs")
    ap.add_argument("--out", default="split_data", help="output root")
    args = ap.parse_args()
    src_root, out_root = Path(args.src), Path(args.out)
    grand = 0
    for env, skill in SKILL_OF.items():
        env_dir = src_root / env
        if not env_dir.is_dir():
            print(f"  skip (no dir): {env}")
            continue
        idx = 0
        for h5_path in find_source_h5(env_dir):
            idx, written = split_one_h5(h5_path, out_root, skill, env, idx)
            print(f"  {env}: +{len(written)} from {h5_path.name} (total {idx})")
        grand += idx
    print(f"Done. {grand} trajectories under {out_root}/")


if __name__ == "__main__":
    main()
