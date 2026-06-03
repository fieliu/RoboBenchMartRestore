"""Measure Fetch footprint radius in the navigation pose (_NAV_ARM).

Loads the env, drives the robot to the nav posture, then computes the max
horizontal distance from base_link origin to every robot link's collision
geometry -> the effective occupancy radius NavDP must respect.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dsynth.envs import *
from dsynth.robots import *
import gymnasium as gym

_NAV_ARM = np.array([-1.57, 1.518, 0.0, 0.0, 0.0, 1.57, 0.0], dtype=np.float32)
_NAV_BODY = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def main():
    env = gym.make("DarkstoreContinuousBaseEnv", robot_uids="ds_fetch_basket",
                   config_dir_path="demo_envs/pick_to_basket", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   obs_mode="state", sim_backend="cpu", render_backend="cpu")
    env.reset(options={"reconfigure": True})

    for _ in range(150):
        act = np.hstack([_NAV_ARM, 0.015, _NAV_BODY, [0.0, 0.0]]).astype(np.float32)
        env.step(act)

    agent = env.unwrapped.agent
    base_pose = agent.base_link.pose.to_transformation_matrix()[0].cpu().numpy()
    base_xy = base_pose[:2, 3]

    print(f"\nbase_link world xy = ({base_xy[0]:.3f}, {base_xy[1]:.3f})")
    print(f"{'link':<28}{'dx':>7}{'dy':>7}{'horiz_dist':>12}")
    print("-" * 54)

    max_r = 0.0; max_link = ""
    for link in agent.robot.get_links():
        p = link.pose.p[0].cpu().numpy()
        dx, dy = p[0] - base_xy[0], p[1] - base_xy[1]
        r = float(np.hypot(dx, dy))
        name = link.get_name()
        if r > 0.05:
            print(f"{name:<28}{dx:>7.3f}{dy:>7.3f}{r:>12.3f}")
        if r > max_r:
            max_r, max_link = r, name

    print("-" * 54)
    print(f"\nMAX link-origin radius = {max_r:.3f} m  (link: {max_link})")
    print("NOTE: this is link ORIGIN distance; add collision mesh extent (~0.05-0.15m)")
    print(f"=> conservative footprint radius ~ {max_r + 0.10:.2f} m, diameter ~ {2*(max_r+0.10):.2f} m")
    env.close()


if __name__ == "__main__":
    main()
