"""Decisive test: at TORSO=0, with FULL settling (220 steps), can a front-arm
pose clear the floor by shortening vertical reach (elbow bent OR wrist folded up)?
State mode, 2 poses only, to stay within time/memory.
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

def drive(env, arm7, steps=220):
    body = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(steps):
        env.step(np.hstack([arm7, 0.015, body, [0.0, 0.0]]).astype(np.float32))

def measure(env, arm7):
    robot = env.unwrapped.agent.robot
    links = robot.get_links()
    base = next(l for l in links if l.get_name() == "base_link")
    bp = base.pose.p[0].cpu().numpy()
    brot = base.pose.to_transformation_matrix()[0].cpu().numpy()[:3, :3]
    cam = next(l for l in links if l.get_name() == "head_camera_depth_optical_frame")
    cT = cam.pose.to_transformation_matrix()[0].cpu().numpy()
    cpos, cR = cT[:3, 3], cT[:3, :3]
    HFOV, VFOV = 1.442/2.0, (1.442*360/640)/2.0
    max_r, lat_r, min_z, occ = 0.0, 0.0, 1e9, False
    for lk in links:
        if not any(k in lk.get_name() for k in ARM_KEYS): continue
        p = lk.pose.p[0].cpu().numpy()
        rel = brot.T @ (p - bp)
        max_r = max(max_r, float(np.hypot(rel[0], rel[1])))
        lat_r = max(lat_r, abs(float(rel[1])))      # LATERAL extent (corridor width)
        min_z = min(min_z, float(p[2]))
        pc = cR.T @ (p - cpos)
        if pc[2] > 0.05 and abs(np.arctan2(pc[0],pc[2]))<HFOV and abs(np.arctan2(pc[1],pc[2]))<VFOV:
            occ = True
    err = np.abs(arm_qpos(env) - np.array(arm7))
    return max_r, lat_r, min_z, occ, err

def main():
    env = gym.make("DarkstoreContinuousBaseEnv", robot_uids="ds_fetch_basket",
                   config_dir_path="demo_envs/pick_to_basket", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   obs_mode="state", sim_backend="cpu", render_backend="cpu")
    env.reset(options={"reconfigure": True})
    cands = [
        ("elbow90_fwd", [0.0, 1.518, 0.0, 1.57, 0.0, 0.0, 0.0]),
        ("wrist_fold",  [0.0, 1.518, 0.0, 0.0,  0.0, 1.57, 0.0]),
    ]
    print(f"\nTORSO=0, 220 settle steps.  fwd_r=forward, LAT_r=lateral(corridor), min_z, cam")
    print(f"{'pose':<14}{'fwd_r':>7}{'LAT_r':>7}{'min_z':>7}{'cam':>7}{'maxErr':>8}{'worst':>14}")
    print("-"*64)
    for name, arm7 in cands:
        a = np.array(arm7, dtype=np.float32)
        drive(env, a)
        r, lat, z, occ, err = measure(env, a)
        wi = int(np.argmax(err))
        flag = " CLEARS" if z > 0.10 else " drag"
        print(f"{name:<14}{r:>7.3f}{lat:>7.3f}{z:>7.3f}{('BLK' if occ else 'ok'):>7}"
              f"{err.max():>8.2f}{(JN[wi]+':'+format(err[wi],'.2f')):>14}{flag}")
    env.close()

if __name__ == "__main__":
    main()
