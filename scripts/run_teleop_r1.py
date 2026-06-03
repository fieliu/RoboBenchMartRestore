"""Simple teleop + record script for R1 robot.
Use keyboard to control the robot, press 'c' to save trajectory and start new episode.
"""
import argparse
import sys
import time
import numpy as np
import gymnasium as gym

sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *

from mani_skill.utils.wrappers.record import RecordEpisode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", help="Path to scene directory")
    parser.add_argument("-e", "--env-id", type=str, default="DarkstoreContinuousBaseEnv")
    parser.add_argument("-r", "--robot-uids", type=str, default="ds_r1")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--sim-backend", type=str, default="cpu")
    parser.add_argument("--render-backend", type=str, default="cpu")
    parser.add_argument("--no-shadow", action="store_true", help="Disable shadow")
    parser.add_argument("--sim-freq", type=int, default=30)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--num-episodes", type=int, default=2, help="Number of episodes to record")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir for trajectories")
    return parser.parse_args()


R1_ARM_JOINT_NAMES = [
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
]
R1_BODY_JOINT_NAMES = [
    "root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint",
    "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
]
R1_GRIPPER_JOINT_NAMES = [
    "left_gripper_finger_joint1", "left_gripper_finger_joint2",
]


def find_qpos_indices(robot, joint_names):
    indices = []
    for jn in joint_names:
        for i, j in enumerate(robot.active_joints):
            if j.name == jn:
                indices.append(i)
                break
    return np.array(indices, dtype=int)


def print_help():
    print("""
=== R1 Keyboard Teleop + Record ===
  W/S:    Move base forward/backward
  A/D:    Rotate base left/right
  Q/E:    Torso down/up
  I/K:    Arm joint1 (shoulder rot) up/down
  J/L:    Arm joint2 (shoulder lift) up/down
  U/O:    Arm joint4 (elbow) up/down
  Y/H:    Arm joint5 (forearm roll) up/down
  G:      Toggle gripper (open/close)
  R:      Reset arm to rest pose
  C:      Save trajectory & start new episode
  ESC:    Quit
""")


def main():
    args = parse_args()
    print_help()

    output_dir = args.output_dir or f"{args.scene_dir}/demos/teleop"
    traj_name = time.strftime("%Y%m%d_%H%M%S")

    sim_config = {}
    if args.sim_freq is not None:
        sim_config['sim_freq'] = args.sim_freq
    if args.control_freq is not None:
        sim_config['control_freq'] = args.control_freq

    camera_config = dict(width=args.camera_width, height=args.camera_height)

    env = gym.make(
        args.env_id,
        robot_uids=args.robot_uids,
        config_dir_path=args.scene_dir,
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="human",
        reward_mode="none",
        enable_shadow=not args.no_shadow,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        sim_config=sim_config,
        human_render_camera_configs=dict(render_camera=camera_config),
    )

    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name=traj_name,
        save_video=False,
        source_type="teleoperation",
        source_desc="keyboard teleoperation",
        record_reward=False,
        save_on_reset=False,
    )

    agent = env.unwrapped.agent
    robot = agent.robot

    arm_joint_idx = find_qpos_indices(robot, R1_ARM_JOINT_NAMES)
    body_joint_idx = find_qpos_indices(robot, R1_BODY_JOINT_NAMES)
    gripper_joint_idx = find_qpos_indices(robot, R1_GRIPPER_JOINT_NAMES)

    print(f"Arm joint indices: {arm_joint_idx}")
    print(f"Body joint indices: {body_joint_idx}")
    print(f"Gripper joint indices: {gripper_joint_idx}")
    print(f"Output directory: {output_dir}")
    print(f"Saving trajectory data (no video)")

    step_count = 0
    gripper_open = True
    episode_count = 0
    arm_step = 0.05
    base_step = 0.02
    torso_step = 0.02

    obs, info = env.reset(seed=[args.seed], options={'reconfigure': True})
    viewer = env.render()

    print(f"\nStarting episode {episode_count + 1}/{args.num_episodes}")
    print("Control the robot with keyboard...")

    while True:
        viewer.window.wait_key_down()

        action_dict = env.unwrapped.agent.controller.action_dict
        arm_action = action_dict.get("arm", None)
        body_action = action_dict.get("body", None)
        gripper_action = action_dict.get("gripper", None)

        if arm_action is None:
            arm_action = np.zeros(len(arm_joint_idx), dtype=np.float32)
        if body_action is None:
            body_action = np.zeros(len(body_joint_idx), dtype=np.float32)
        if gripper_action is None:
            gripper_action = np.array([0.04, 0.04], dtype=np.float32)

        # Get keyboard state
        w = viewer.window.key_press('w')
        a = viewer.window.key_press('a')
        s = viewer.window.key_press('s')
        d = viewer.window.key_press('d')
        q = viewer.window.key_press('q')
        e = viewer.window.key_press('e')
        i = viewer.window.key_press('i')
        k = viewer.window.key_press('k')
        j = viewer.window.key_press('j')
        l = viewer.window.key_press('l')
        u = viewer.window.key_press('u')
        o = viewer.window.key_press('o')
        g = viewer.window.key_press('g')
        c_key = viewer.window.key_press('c')
        r_key = viewer.window.key_press('r')
        esc = viewer.window.key_press('escape')

        if esc:
            break

        # Base movement
        body_action[0] += (w - s) * base_step
        body_action[2] += (a - d) * base_step * 0.5
        body_action[6] += (e - q) * torso_step

        # Arm movement
        arm_action[0] += (i - k) * arm_step
        arm_action[1] += (j - l) * arm_step * 0.5
        arm_action[3] += (u - o) * arm_step

        # Gripper toggle
        if g:
            gripper_open = not gripper_open
            if gripper_open:
                gripper_action[:] = 0.04
                print("Gripper OPEN")
            else:
                gripper_action[:] = -0.01
                print("Gripper CLOSE")

        # Arm reset
        if r_key:
            arm_action[:] = 0
            body_action[3:7] = 0
            print("Arm reset")

        # Build action
        action = np.zeros(15, dtype=np.float32)
        action[body_joint_idx] = body_action
        action[arm_joint_idx] = arm_action
        action[gripper_joint_idx] = gripper_action

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if c_key:
            episode_count += 1
            print(f"\nEpisode {episode_count} saved. Steps: {step_count}")
            env.flush_trajectory()
            step_count = 0

            if episode_count >= args.num_episodes:
                print(f"\nAll {args.num_episodes} episodes recorded!")
                break

            obs, info = env.reset(seed=[args.seed + episode_count], options={'reconfigure': True})
            print(f"\nStarting episode {episode_count + 1}/{args.num_episodes}")

    env.close()
    print(f"\nDone! Trajectories saved to: {output_dir}")


if __name__ == "__main__":
    main()
