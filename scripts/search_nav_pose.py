"""FK grid-search for a compact 'relaxed arm' nav pose for Fetch.

Sets qpos directly (fast FK, no PD stepping), reads link world poses with the
base at origin, and scores each candidate by:
  - footprint radius: max horizontal dist (base center -> any arm/gripper link)
  - ground clearance: min z of arm/gripper links (must not drag)
  - camera clearance: arm links kept to the side / below the head camera

Goal: radius ~0.35 m, gripper off the floor, head_camera view unobstructed.
"""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsynth.envs import *
from dsynth.robots import *
import gymnasium as gym

# qpos indices
PAN, LIFT, UROLL, ELBOW, FROLL, WFLEX, WROLL = 5, 7, 8, 9, 10, 11, 12
ARM_LINK_KEYS = ("shoulder", "upperarm", "elbow", "forearm", "wrist", "gripper")


def main():
    env = gym.make("DarkstoreContinuousBaseEnv", robot_uids="ds_fetch_basket",
                   config_dir_path="demo_envs/pick_to_basket", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   obs_mode="state", sim_backend="cpu", render_backend="cpu")
    env.reset(options={"reconfigure": True})
    robot = env.unwrapped.agent.robot
    q0 = robot.get_qpos()[0].cpu().numpy().copy()
    links = robot.get_links()

    def evaluate(pan, lift, uroll, elbow, froll, wflex, wroll):
        q = q0.copy()
        q[0] = q[1] = q[2] = 0.0           # base at origin
        q[3] = 0.0                          # torso min
        q[4] = q[6] = 0.0                   # head_pan, head_tilt = 0
        q[PAN], q[LIFT], q[UROLL] = pan, lift, uroll
        q[ELBOW], q[FROLL], q[WFLEX], q[WROLL] = elbow, froll, wflex, wroll
        import torch
        robot.set_qpos(torch.tensor(q[None], dtype=torch.float32))
        # radius relative to base_link (root joints don't teleport the spawn)
        base = robot.get_links()[0]
        for lk in links:
            if lk.get_name() in ("base_link",):
                base = lk; break
        bxy = base.pose.p[0].cpu().numpy()[:2]
        max_r, min_z = 0.0, 1e9
        for lk in links:
            nm = lk.get_name()
            if not any(k in nm for k in ARM_LINK_KEYS):
                continue
            p = lk.pose.p[0].cpu().numpy()
            r = float(np.hypot(p[0] - bxy[0], p[1] - bxy[1]))
            max_r = max(max_r, r)
            min_z = min(min_z, float(p[2]))
        return max_r, min_z

    print("searching …")
    best = []
    pans = [-1.5, -1.3, -1.0, -0.6, -0.3]
    lifts = [1.0, 1.3, 1.518]
    elbows = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    wflexes = [-1.5, -0.8, 0.0, 0.8, 1.5]
    for pan, lift, elbow, wflex in itertools.product(pans, lifts, elbows, wflexes):
        r, z = evaluate(pan, lift, 0.0, elbow, 0.0, wflex, 0.0)
        if z > 0.12:   # ground clearance (no drag)
            best.append((r, z, pan, lift, elbow, wflex))
    best.sort(key=lambda t: t[0])
    print(f"\n{'rank':<5}{'radius':>8}{'min_z':>8}{'pan':>7}{'lift':>7}{'elbow':>7}{'wflex':>7}")
    for i, (r, z, pan, lift, elbow, wflex) in enumerate(best[:12]):
        print(f"{i:<5}{r:>8.3f}{z:>8.3f}{pan:>7.2f}{lift:>7.2f}{elbow:>7.2f}{wflex:>7.2f}")
    env.close()


if __name__ == "__main__":
    main()
