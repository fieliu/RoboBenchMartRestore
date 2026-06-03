"""Render head_camera with arm in relaxed hang-down pose at several shoulder_pan
values, and measure how much of the view the arm occludes (near-depth pixels).

Picks the smallest |pan| (arm most 'in front') that keeps the camera view
mostly clear, per the user's spec: arm may hang in front, must not block view.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsynth.envs import *
from dsynth.robots import *
import gymnasium as gym


def arm_qpos(env):  return env.unwrapped.agent.controller.controllers["arm"].qpos[0].cpu().numpy()

def drive(env, arm7, torso, steps=160):
    body = np.array([0.0, 0.0, torso], dtype=np.float32)
    obs = None
    for _ in range(steps):
        obs, _, _, _, _ = env.step(np.hstack([arm7, 0.015, body, [0.0, 0.0]]).astype(np.float32))
    return obs

def occlusion(obs):
    """fraction of head_camera pixels with depth < 0.5m (arm right in front)."""
    d = obs["sensor_data"]["head_camera"]["depth"][0].cpu().numpy().squeeze().astype(np.float32)
    d = d / 1000.0 if d.max() > 100 else d   # mm->m
    H, W = d.shape
    near = (d > 0.01) & (d < 0.5)
    # also report lower-center band (where forward obstacles appear for NavDP)
    band = near[H//2:, W//3:2*W//3]
    return float(near.mean()), float(band.mean())


def main():
    env = gym.make("DarkstoreContinuousBaseEnv", robot_uids="ds_fetch_basket",
                   config_dir_path="demo_envs/pick_to_basket", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   obs_mode="rgbd", sim_backend="cpu", render_backend="cpu")
    env.reset(options={"reconfigure": True})

    print(f"\n{'pan':>7}{'radius_proxy':>14}{'occl_all%':>11}{'occl_band%':>12}")
    print("-" * 44)
    for pan in [0.0, -0.5, -1.0, -1.57]:
        arm7 = np.array([pan, 1.518, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs = drive(env, arm7, torso=0.30)
        oa, ob = occlusion(obs)
        print(f"{pan:>7.2f}{'':>14}{oa*100:>10.1f}{ob*100:>11.1f}")
        env.reset(options={"reconfigure": True})
    print("\nlower 'occl_band%' = arm blocking NavDP's forward view. want ~0.")
    env.close()


if __name__ == "__main__":
    main()
