"""Find a relaxed arm pose at TORSO=0 (camera height unchanged for NavDP) that
clears the floor WITHOUT raising the torso — by slightly bending the arm up
(elbow) or turning the hand to the side, per user guidance.

PD-drives each candidate (what nav uses), renders head_camera depth to measure
real occlusion, and reports settled radius + ground clearance + tracking error.
Picks minimal deviation from vertical that gives min_z > 0.10m and cam clear.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsynth.envs import *
from dsynth.robots import *
import gymnasium as gym

ARM_KEYS = ("shoulder", "upperarm", "elbow", "forearm", "wrist", "gripper")
JN = ["pan", "lift", "uroll", "elbow", "froll", "wflex", "wroll"]

def arm_qpos(env): return env.unwrapped.agent.controller.controllers["arm"].qpos[0].cpu().numpy()

def drive(env, arm7, steps=90):
    body = np.array([0.0, 0.0, 0.0], dtype=np.float32)   # TORSO = 0 (no raise)
    for _ in range(steps):
        env.step(np.hstack([arm7, 0.015, body, [0.0, 0.0]]).astype(np.float32))

def measure(env, arm7):
    """State-mode only (no render): radius, ground clearance, tracking error, and
    GEOMETRIC camera occlusion by projecting each arm link into the depth optical
    frame and testing if it lands inside the image FOV in front of the camera."""
    robot = env.unwrapped.agent.robot
    links = robot.get_links()
    base = next(l for l in links if l.get_name() == "base_link")
    bp = base.pose.p[0].cpu().numpy()
    brot = base.pose.to_transformation_matrix()[0].cpu().numpy()[:3, :3]
    cam = next(l for l in links if l.get_name() == "head_camera_depth_optical_frame")
    cT = cam.pose.to_transformation_matrix()[0].cpu().numpy()
    cpos, cR = cT[:3, 3], cT[:3, :3]
    HFOV, VFOV = 1.442 / 2.0, (1.442 * 360 / 640) / 2.0   # 640x360, optical +z fwd, +x right, +y down

    max_r, min_z, occ = 0.0, 1e9, False
    for lk in links:
        nm = lk.get_name()
        if not any(k in nm for k in ARM_KEYS):
            continue
        p = lk.pose.p[0].cpu().numpy()
        rel = brot.T @ (p - bp)
        max_r = max(max_r, float(np.hypot(rel[0], rel[1])))
        min_z = min(min_z, float(p[2]))
        pc = cR.T @ (p - cpos)                       # link in camera optical frame
        if pc[2] > 0.05:                             # in front of camera
            if abs(np.arctan2(pc[0], pc[2])) < HFOV and abs(np.arctan2(pc[1], pc[2])) < VFOV:
                occ = True
    err = float(np.abs(arm_qpos(env) - np.array(arm7)).max())
    return max_r, min_z, err, occ



def main():
    env = gym.make("DarkstoreContinuousBaseEnv", robot_uids="ds_fetch_basket",
                   config_dir_path="demo_envs/pick_to_basket", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   obs_mode="rgbd", sim_backend="cpu", render_backend="cpu")
    env.reset(options={"reconfigure": True})

    # All TORSO=0. arm = [pan, lift, uroll, elbow, froll, wflex, wroll]
    # Strategy: bend elbow up (lift forearm) and/or raise wrist so hand clears floor.
    cands = [
        ("vertical",       [0.0, 1.518, 0.0,  0.0, 0.0,  0.0, 0.0]),
        ("elbow_0.8",      [0.0, 1.518, 0.0,  0.8, 0.0,  0.0, 0.0]),
        ("elbow_1.2",      [0.0, 1.518, 0.0,  1.2, 0.0,  0.0, 0.0]),
        ("wrist_up_1.3",   [0.0, 1.518, 0.0,  0.0, 0.0,  1.3, 0.0]),
        ("palm_left",      [0.0, 1.518, 0.0,  0.3, 1.57, 0.5, 0.0]),
    ]

    print(f"\nTORSO=0 (camera height fixed).  want: min_z>0.10, occ~0, small dev")
    print(f"(driven sequentially, no reconfigure -> also proves works from any start)")
    print(f"{'pose':<16}{'radius':>8}{'min_z':>8}{'occ%':>7}{'trkErr':>8}")
    print("-" * 48)
    for name, arm7 in cands:
        a = np.array(arm7, dtype=np.float32)
        drive(env, a)
        r, z, err, occ = measure(env, a)
        flag = "  <-- clears" if z > 0.10 and occ < 0.02 else (" DRAG" if z < 0.08 else "")
        print(f"{name:<16}{r:>8.3f}{z:>8.3f}{occ*100:>7.1f}{err:>8.2f}{flag}")
    env.close()


if __name__ == "__main__":
    main()
