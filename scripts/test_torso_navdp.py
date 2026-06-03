"""Option B test: does raising the torso (to lift the hand off the floor) degrade
NavDP? Measures, per torso height:
  - camera height delta (how much the head_camera actually moves)
  - hand ground clearance (min_z of arm/gripper links)
  - NavDP output: feed the SAME observation, compare planned trajectory + critic
    across heights. If traj/critic barely change, NavDP is height-insensitive.
"""
import sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dsynth", "navigation", "navdp_models"))
from dsynth.envs import *
from dsynth.robots import *
import gymnasium as gym

CKPT = "dsynth/navigation/navdp_models/navdp-cross-modal.ckpt"
ARM_KEYS = ("shoulder", "upperarm", "elbow", "forearm", "wrist", "gripper")
RELAX_ARM = np.array([0.0, 1.518, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def drive(env, torso, steps=160):
    body = np.array([0.0, 0.0, float(torso)], dtype=np.float32)
    obs = None
    for _ in range(steps):
        obs, _, _, _, _ = env.step(np.hstack([RELAX_ARM, 0.015, body, [0.0, 0.0]]).astype(np.float32))
    return obs


def cam_height_and_clearance(env):
    robot = env.unwrapped.agent.robot
    links = robot.get_links()
    cam = next(l for l in links if l.get_name() == "head_camera_link")
    cz = float(cam.pose.p[0].cpu().numpy()[2])
    min_z = 1e9
    for lk in links:
        if any(k in lk.get_name() for k in ARM_KEYS):
            min_z = min(min_z, float(lk.pose.p[0].cpu().numpy()[2]))
    return cz, min_z


def main():
    env = gym.make("DarkstoreContinuousBaseEnv", robot_uids="ds_fetch_basket",
                   config_dir_path="demo_envs/pick_to_basket", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   obs_mode="rgbd", sim_backend="cpu", render_backend="cpu")
    env.reset(options={"reconfigure": True})

    from policy_agent import NavDP_Agent
    K = np.array([[200.0, 0, 320.0], [0, 200.0, 180.0], [0, 0, 1.0]], dtype=np.float32)
    agent = NavDP_Agent(image_intrinsic=K, navi_model=CKPT, device="cuda:0")
    goal = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)   # 5m straight ahead

    print(f"\n{'torso':>7}{'cam_z':>8}{'hand_z':>8}{'critic_max':>12}{'traj_end_xy':>20}")
    print("-" * 56)
    base_cam = None
    for torso in [0.0, 0.05, 0.10, 0.15]:
        obs = drive(env, torso)
        cz, hz = cam_height_and_clearance(env)
        if base_cam is None: base_cam = cz
        rgb = obs["sensor_data"]["head_camera"]["rgb"][0].cpu().numpy()
        depth = obs["sensor_data"]["head_camera"]["depth"][0].cpu().numpy()
        dm = (depth.astype(np.float32) / 1000.0) if np.issubdtype(depth.dtype, np.integer) else depth.astype(np.float32)
        if dm.ndim == 2: dm = dm[:, :, None]
        agent.reset(batch_size=1, threshold=-3.0)
        traj, _, values, _ = agent.step_pointgoal(goal, rgb[None].astype(np.float32), dm[None])
        end = traj[0][-1][:2]
        print(f"{torso:>7.2f}{cz:>8.3f}{hz:>8.3f}{float(values.max()):>12.2f}"
              f"   ({end[0]:.2f},{end[1]:.2f})")
    print(f"\ncam moved {base_cam:.3f} -> see above. If critic/traj stable across")
    print("heights, NavDP is height-insensitive and raising torso is safe.")
    env.close()


if __name__ == "__main__":
    main()
