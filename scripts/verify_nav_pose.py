"""PD-verified nav pose check + corridor gap measurement.

FK-set poses are fantasy if the PD controller can't hold them. So here we
ACTUALLY drive each candidate arm pose with pd_joint_pos (what nav uses),
let it settle, then measure the real footprint radius, ground clearance,
joint tracking error, and head-camera occlusion.

Also measures the free lateral gap between the obstacle box and the nearest
shelves on each side of the travel corridor.
"""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsynth.envs import *
from dsynth.robots import *
import gymnasium as gym

HALF_FOV = 1.442 / 2.0
# (name, shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll)
CANDIDATES = [
    ("current",      -1.57, 1.518, 0.0,  0.0,  0.0, 1.57, 0.0),
    ("tuck_up",      -1.00, 1.520, 0.0,  2.0,  0.0, 1.50, 0.0),
    ("elbow_fold",   -1.57, 1.518, 0.0,  1.8,  0.0, 1.40, 0.0),
    ("side_bent",    -1.30, 1.518, 0.0,  1.5,  0.0, 0.8,  0.0),
    ("compact_side", -1.57, 1.400, 0.0,  2.0,  0.0, 1.50, 0.0),
]

def arm_qpos(env):  return env.unwrapped.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
def body_qpos(env): return env.unwrapped.agent.controller.controllers["body"].qpos[0].cpu().numpy()

def make_action(arm7, body3):
    return np.hstack([arm7, 0.015, body3, [0.0, 0.0]]).astype(np.float32)


def measure_settled(env, arm7, body3, steps=150):
    """PD-drive to (arm7, body3), settle, then measure real footprint."""
    obs = None
    for _ in range(steps):
        obs, _, _, _, _ = env.step(make_action(arm7, body3))
    robot = env.unwrapped.agent.robot
    links = robot.get_links()
    base = next(l for l in links if l.get_name() == "base_link")
    bp = base.pose.p[0].cpu().numpy()
    brot = base.pose.to_transformation_matrix()[0].cpu().numpy()[:3, :3]

    actual_arm = arm_qpos(env)
    track_err = float(np.abs(actual_arm - arm7).max())

    max_r, min_z, occl = 0.0, 1e9, False
    for lk in links:
        nm = lk.get_name()
        if not any(k in nm for k in ("shoulder", "upperarm", "elbow", "forearm", "wrist", "gripper")):
            continue
        p = lk.pose.p[0].cpu().numpy()
        rel = brot.T @ (p - bp)
        r = float(np.hypot(rel[0], rel[1]))
        max_r = max(max_r, r); min_z = min(min_z, float(p[2]))
        if rel[0] > 0.1 and p[2] > 0.5 and abs(np.arctan2(rel[1], rel[0])) < HALF_FOV:
            occl = True
    return max_r, min_z, track_err, occl


def measure_corridor(env):
    sb = env.unwrapped.scene_builder
    shlv = env.unwrapped.actors["fixtures"]["shelves"]           # dict
    act = env.unwrapped.active_shelves[0]                        # list of key(s)
    sp = shlv[act[0]].pose.sp
    facing = sp.to_transformation_matrix()[:3, 1]
    perp = sp.to_transformation_matrix()[:3, 0]
    start = sp.p - 1.5 * facing
    fwd = perp.copy()
    if fwd[0] < 0: fwd = -fwd
    obstacle = start + 2.5 * fwd
    lateral = np.array([-fwd[1], fwd[0], 0.0]); lateral /= np.linalg.norm(lateral)

    left_gap, right_gap = 1e9, 1e9
    for key, actor in shlv.items():
        c = actor.pose.sp.p
        if abs(np.dot(c - obstacle, fwd)) > 1.5:
            continue
        lat = float(np.dot(c - obstacle, lateral))
        if lat < -0.1:  left_gap = min(left_gap, abs(lat))
        elif lat > 0.1: right_gap = min(right_gap, abs(lat))
    return obstacle, fwd, left_gap, right_gap


def main():
    env = gym.make("DarkstoreContinuousBaseEnv", robot_uids="ds_fetch_basket",
                   config_dir_path="demo_envs/pick_to_basket", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   obs_mode="state", sim_backend="cpu", render_backend="cpu")
    env.reset(options={"reconfigure": True})

    print(f"\n{'pose':<14}{'radius':>8}{'min_z':>8}{'trackErr':>10}{'cam':>12}")
    print("-" * 52)
    for name, *vals in CANDIDATES:
        r, z, te, occl = measure_settled(env, np.array(vals, dtype=np.float32),
                                         np.array([0.0, 0.0, 0.0], dtype=np.float32))
        cam = "BLOCKED" if occl else "clear"
        drag = " DRAG" if z < 0.08 else ""
        print(f"{name:<14}{r:>8.3f}{z:>8.3f}{te:>10.3f}{cam:>12}{drag}")

    obstacle, fwd, lg, rg = measure_corridor(env)
    print(f"\n=== CORRIDOR ===  obstacle=({obstacle[0]:.2f},{obstacle[1]:.2f}) fwd=({fwd[0]:.2f},{fwd[1]:.2f})")
    print(f"obstacle->left shelf  = {lg:.2f} m")
    print(f"obstacle->right shelf = {rg:.2f} m")
    print(f"obstacle box half-width = 0.25 m")
    print(f"free gap LEFT  = {lg-0.25:.2f} m ;  free gap RIGHT = {rg-0.25:.2f} m")
    env.close()


if __name__ == "__main__":
    main()
