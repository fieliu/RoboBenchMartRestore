"""Find + PD-verify a 'human relaxed' hang-down arm pose for Fetch.

Target: upper arm + forearm vertical (hanging down), hand/palm forward, low
enough to clear the head camera, minimal horizontal footprint. A hang-down
pose is gravity-ASSISTED (unlike folded poses that fight gravity), so PD
should hold it. We report PER-JOINT tracking error to see what sags.
"""
import sys, os, itertools
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsynth.envs import *
from dsynth.robots import *
import gymnasium as gym

HALF_FOV = 1.442 / 2.0
ARM_KEYS = ("shoulder", "upperarm", "elbow", "forearm", "wrist", "gripper")
JNAMES = ["pan", "lift", "uroll", "elbow", "froll", "wflex", "wroll"]


def fk_eval(robot, q0, links, arm7):
    q = q0.copy()
    q[3] = q[4] = q[6] = 0.0
    q[5], q[7], q[8], q[9], q[10], q[11], q[12] = arm7
    robot.set_qpos(torch.tensor(q[None], dtype=torch.float32))
    base = next(l for l in links if l.get_name() == "base_link")
    bp = base.pose.p[0].cpu().numpy()
    brot = base.pose.to_transformation_matrix()[0].cpu().numpy()[:3, :3]
    max_r, min_z, occl = 0.0, 1e9, False
    for lk in links:
        if not any(k in lk.get_name() for k in ARM_KEYS):
            continue
        p = lk.pose.p[0].cpu().numpy()
        rel = brot.T @ (p - bp)
        max_r = max(max_r, float(np.hypot(rel[0], rel[1])))
        min_z = min(min_z, float(p[2]))
        if rel[0] > 0.1 and p[2] > 0.5 and abs(np.arctan2(rel[1], rel[0])) < HALF_FOV:
            occl = True
    return max_r, min_z, occl


def arm_qpos(env):
    return env.unwrapped.agent.controller.controllers["arm"].qpos[0].cpu().numpy()


def pd_verify(env, arm7, torso, steps=160):
    """Drive to (arm7, torso) with PD, settle, report per-joint error + footprint."""
    body = np.array([0.0, 0.0, torso], dtype=np.float32)  # head_pan, head_tilt, torso
    for _ in range(steps):
        act = np.hstack([arm7, 0.015, body, [0.0, 0.0]]).astype(np.float32)
        env.step(act)
    actual = arm_qpos(env)
    err = np.abs(actual - np.array(arm7))
    robot = env.unwrapped.agent.robot
    links = robot.get_links()
    base = next(l for l in links if l.get_name() == "base_link")
    bp = base.pose.p[0].cpu().numpy()
    brot = base.pose.to_transformation_matrix()[0].cpu().numpy()[:3, :3]
    max_r, min_z, occl = 0.0, 1e9, False
    for lk in links:
        if not any(k in lk.get_name() for k in ARM_KEYS):
            continue
        p = lk.pose.p[0].cpu().numpy()
        rel = brot.T @ (p - bp)
        max_r = max(max_r, float(np.hypot(rel[0], rel[1])))
        min_z = min(min_z, float(p[2]))
        if rel[0] > 0.1 and p[2] > 0.5 and abs(np.arctan2(rel[1], rel[0])) < HALF_FOV:
            occl = True
    return actual, err, max_r, min_z, occl


def main():
    env = gym.make("DarkstoreContinuousBaseEnv", robot_uids="ds_fetch_basket",
                   config_dir_path="demo_envs/pick_to_basket", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   obs_mode="state", sim_backend="cpu", render_backend="cpu")
    env.reset(options={"reconfigure": True})
    robot = env.unwrapped.agent.robot
    q0 = robot.get_qpos()[0].cpu().numpy().copy()
    links = robot.get_links()

    # hang-down candidates: arm FORWARD (pan~0), upper-arm down (lift=1.518),
    # forearm continues down (elbow~0), torso raised to avoid floor drag.
    # [pan, lift, uroll, elbow, froll, wflex, wroll], torso
    cands = [
        ("fwd_hang_t0",   [0.0, 1.518, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
        ("fwd_hang_t15",  [0.0, 1.518, 0.0, 0.0, 0.0, 0.0, 0.0], 0.15),
        ("fwd_hang_t30",  [0.0, 1.518, 0.0, 0.0, 0.0, 0.0, 0.0], 0.30),
        ("fwd_elbow_t15", [0.0, 1.518, 0.0, 0.4, 0.0, 0.4, 0.0], 0.15),
        ("fwd_palmfwd",   [0.0, 1.518, 0.0, 0.0, 0.0, 1.3, 0.0], 0.20),
    ]
    print("=== FK (ideal) ===")
    for name, arm7, _t in cands:
        r, z, occl = fk_eval(robot, q0, links, arm7)
        print(f"  {name:<14} r={r:.3f} min_z={z:.3f} cam={'BLOCK' if occl else 'clear'}")

    print("\n=== PD (real) ===  per-joint err shown for worst joints")
    env.reset(options={"reconfigure": True})
    for name, arm7, torso in cands:
        actual, err, r, z, occl = pd_verify(env, arm7, torso)
        worst = np.argsort(err)[::-1][:2]
        we = ", ".join(f"{JNAMES[i]}:{err[i]:.2f}" for i in worst)
        print(f"  {name:<14} r={r:.3f} min_z={z:.3f} cam={'BLOCK' if occl else 'clear'} "
              f"maxErr={err.max():.2f} ({we})")
        env.reset(options={"reconfigure": True})
    env.close()


if __name__ == "__main__":
    main()
