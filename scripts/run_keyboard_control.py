import argparse
import sys
import numpy as np
import gymnasium as gym

sys.path.append('.')
from dsynth.envs import *
from dsynth.robots import *


def parse_args():
    parser = argparse.ArgumentParser(description="Keyboard teleoperation")
    parser.add_argument("scene_dir", help="Path to scene directory")
    parser.add_argument("-e", "--env-id", type=str, default="DarkstoreContinuousBaseEnv")
    parser.add_argument("-r", "--robot-uids", type=str, default="ds_fetch_basket")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--sim-backend", type=str, default="cpu")
    parser.add_argument("--render-backend", type=str, default="cpu")
    parser.add_argument("--no-shadow", action="store_true", help="Disable shadow rendering for better CPU performance")
    parser.add_argument("--sim-freq", type=int, default=None, help="Simulation frequency (default: 100)")
    parser.add_argument("--control-freq", type=int, default=None, help="Control frequency (default: 20)")
    parser.add_argument("--camera-width", type=int, default=256, help="Camera render width (default: 256)")
    parser.add_argument("--camera-height", type=int, default=256, help="Camera render height (default: 256)")
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

FETCH_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
    "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint",
]
FETCH_BODY_JOINT_NAMES = ["head_pan_joint", "head_tilt_joint", "torso_lift_joint"]


def find_qpos_indices(robot, joint_names):
    indices = []
    for jn in joint_names:
        for i, j in enumerate(robot.active_joints):
            if j.name == jn:
                indices.append(i)
                break
    return indices


def run_fetch(env, args, viewer, agent, robot):
    home_qpos = np.array(agent.keyframes["rest"].qpos)
    action_low = env.action_space.low
    action_high = env.action_space.high

    arm_qpos_idx = find_qpos_indices(robot, FETCH_ARM_JOINT_NAMES)
    body_qpos_idx = find_qpos_indices(robot, FETCH_BODY_JOINT_NAMES)

    current_qpos = robot.get_qpos()[0].cpu().numpy()
    arm_target = np.array([current_qpos[i] for i in arm_qpos_idx], dtype=np.float32)
    body_target = np.array([current_qpos[i] for i in body_qpos_idx], dtype=np.float32)
    gripper_action = -1.0

    arm_step = 0.05
    body_step = 0.05
    base_vel = 0.5
    rot_vel = 0.5

    print("""
╔══════════════════════════════════════════════════════════════════╗
║            Fetch Robot Keyboard Teleoperation                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Action space: 13-dim (pd_joint_pos)                            ║
║  [0-6]   arm: shoulder_pan, shoulder_lift, upperarm_roll,       ║
║           elbow_flex, forearm_roll, wrist_flex, wrist_roll      ║
║  [7]     gripper: mimic (1 value controls both fingers)         ║
║  [8-10]  body: head_pan, head_tilt, torso_lift                  ║
║  [11-12] base: forward_vel, rotation_vel (VELOCITY control)     ║
╠══════════════════════════════════════════════════════════════════╣
║  Up/Down    : Base forward/backward (velocity)                  ║
║  Left/Right : Base rotate left/right (velocity)                 ║
║  w/s        : Shoulder lift up/down                              ║
║  a/d        : Shoulder pan left/right                            ║
║  q/e        : Elbow flex extend/retract                          ║
║  r/f        : Wrist flex up/down                                 ║
║  t/y        : Upperarm roll                                      ║
║  z/x        : Forearm roll                                       ║
║  c/v        : Wrist roll                                         ║
║  u/j        : Torso lift up/down                                 ║
║  i/k        : Head pan left/right                                ║
║  o/l        : Head tilt up/down                                  ║
║  g          : Toggle gripper open/close                          ║
║  0          : Reset to home position                             ║
║  n          : Next episode (new seed)                            ║
║  ESC/close  : Quit                                               ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    def build_action(base_forward=0.0, base_rotation=0.0):
        a = np.zeros(13, dtype=np.float32)
        a[0:7] = arm_target
        a[7] = gripper_action
        a[8:11] = body_target
        a[11] = base_forward
        a[12] = base_rotation
        return np.clip(a, action_low, action_high)

    while True:
        env.render()
        if viewer.closed:
            break

        base_forward = 0.0
        base_rotation = 0.0
        arm_changed = False
        body_changed = False
        do_step = False

        if viewer.window.key_down("up"):
            base_forward = base_vel
            do_step = True
        elif viewer.window.key_down("down"):
            base_forward = -base_vel
            do_step = True
        if viewer.window.key_down("left"):
            base_rotation = rot_vel
            do_step = True
        elif viewer.window.key_down("right"):
            base_rotation = -rot_vel
            do_step = True

        if viewer.window.key_down("a"):
            arm_target[0] += arm_step; arm_changed = True
        elif viewer.window.key_down("d"):
            arm_target[0] -= arm_step; arm_changed = True
        if viewer.window.key_down("w"):
            arm_target[1] += arm_step; arm_changed = True
        elif viewer.window.key_down("s"):
            arm_target[1] -= arm_step; arm_changed = True
        if viewer.window.key_down("t"):
            arm_target[2] += arm_step; arm_changed = True
        elif viewer.window.key_down("y"):
            arm_target[2] -= arm_step; arm_changed = True
        if viewer.window.key_down("q"):
            arm_target[3] -= arm_step; arm_changed = True
        elif viewer.window.key_down("e"):
            arm_target[3] += arm_step; arm_changed = True
        if viewer.window.key_down("z"):
            arm_target[4] += arm_step; arm_changed = True
        elif viewer.window.key_down("x"):
            arm_target[4] -= arm_step; arm_changed = True
        if viewer.window.key_down("r"):
            arm_target[5] -= arm_step; arm_changed = True
        elif viewer.window.key_down("f"):
            arm_target[5] += arm_step; arm_changed = True
        if viewer.window.key_down("c"):
            arm_target[6] += arm_step; arm_changed = True
        elif viewer.window.key_down("v"):
            arm_target[6] -= arm_step; arm_changed = True

        if viewer.window.key_down("i"):
            body_target[0] += body_step; body_changed = True
        elif viewer.window.key_down("k"):
            body_target[0] -= body_step; body_changed = True
        if viewer.window.key_down("o"):
            body_target[1] += body_step; body_changed = True
        elif viewer.window.key_down("l"):
            body_target[1] -= body_step; body_changed = True
        if viewer.window.key_down("u"):
            body_target[2] += body_step; body_changed = True
        elif viewer.window.key_down("j"):
            body_target[2] -= body_step; body_changed = True

        if arm_changed:
            arm_target = np.clip(arm_target, action_low[0:7], action_high[0:7])
            do_step = True
        if body_changed:
            body_target = np.clip(body_target, action_low[8:11], action_high[8:11])
            do_step = True

        if viewer.window.key_press("g"):
            gripper_action = 1.0 if gripper_action < 0 else -1.0
            do_step = True
            print(f"Gripper {'closed' if gripper_action > 0 else 'opened'}")

        if viewer.window.key_press("0"):
            arm_target = np.array([home_qpos[i] for i in arm_qpos_idx], dtype=np.float32)
            body_target = np.array([home_qpos[i] for i in body_qpos_idx], dtype=np.float32)
            gripper_action = -1.0
            do_step = True
            print("Reset to home position")

        if viewer.window.key_press("n"):
            args.seed += 1
            obs, info = env.reset(seed=[args.seed], options={'reconfigure': True})
            current_qpos = robot.get_qpos()[0].cpu().numpy()
            arm_target = np.array([current_qpos[i] for i in arm_qpos_idx], dtype=np.float32)
            body_target = np.array([current_qpos[i] for i in body_qpos_idx], dtype=np.float32)
            gripper_action = -1.0
            viewer = env.render()
            print(f"New episode seed={args.seed}")
            continue

        if do_step:
            action = build_action(base_forward, base_rotation)
            obs, reward, terminated, truncated, info = env.step(action)

    env.close()


def run_r1(env, args, viewer, agent, robot):
    home_qpos = np.array(agent.keyframes["rest"].qpos)
    action_low = env.action_space.low
    action_high = env.action_space.high
    action_dim = env.action_space.shape[0]

    arm_qpos_idx = find_qpos_indices(robot, R1_ARM_JOINT_NAMES)
    body_qpos_idx = find_qpos_indices(robot, R1_BODY_JOINT_NAMES)
    gripper_qpos_idx = find_qpos_indices(robot, R1_GRIPPER_JOINT_NAMES)

    current_qpos = robot.get_qpos()[0].cpu().numpy()
    arm_target = np.array([current_qpos[i] for i in arm_qpos_idx], dtype=np.float32)
    body_target = np.array([current_qpos[i] for i in body_qpos_idx], dtype=np.float32)
    gripper_target = np.array([current_qpos[i] for i in gripper_qpos_idx], dtype=np.float32)

    arm_step = 0.05
    body_step = 0.05
    gripper_step = 0.01

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║            R1 Robot Keyboard Teleoperation                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Action space: {action_dim}-dim (pd_joint_pos)                     ║
║  [0-2]   base: x, y, rotation (POSITION control)                ║
║  [3-6]   torso: joint1~4                                        ║
║  [7-12]  left arm: joint1~6                                     ║
║  [13-14] left gripper: finger1, finger2                         ║
║  Right arm: FIXED (not controllable)                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Up/Down    : Base y forward/backward                            ║
║  Left/Right : Base rotation left/right                           ║
║  w/s        : Arm joint1 (shoulder pan)                          ║
║  a/d        : Arm joint2 (shoulder lift)                         ║
║  q/e        : Arm joint3 (upperarm roll)                         ║
║  r/f        : Arm joint4 (elbow flex)                            ║
║  t/y        : Arm joint5 (forearm roll)                          ║
║  z/x        : Arm joint6 (wrist)                                 ║
║  u/j        : Torso joint1 (lift up/down)                        ║
║  i/k        : Torso joint2                                       ║
║  o/l        : Torso joint3                                       ║
║  b/n        : Torso joint4                                       ║
║  g          : Toggle gripper open/close                          ║
║  0          : Reset to home position                             ║
║  n          : Next episode (new seed)                            ║
║  ESC/close  : Quit                                               ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    def build_action():
        a = np.zeros(action_dim, dtype=np.float32)
        for i, idx in enumerate(body_qpos_idx):
            a[idx] = body_target[i]
        for i, idx in enumerate(arm_qpos_idx):
            a[idx] = arm_target[i]
        for i, idx in enumerate(gripper_qpos_idx):
            a[idx] = gripper_target[i]
        return np.clip(a, action_low, action_high)

    while True:
        env.render()
        if viewer.closed:
            break

        arm_changed = False
        body_changed = False
        gripper_changed = False
        do_step = False

        if viewer.window.key_down("up"):
            body_target[1] += body_step; body_changed = True
        elif viewer.window.key_down("down"):
            body_target[1] -= body_step; body_changed = True
        if viewer.window.key_down("left"):
            body_target[2] += arm_step; body_changed = True
        elif viewer.window.key_down("right"):
            body_target[2] -= arm_step; body_changed = True

        if viewer.window.key_down("w"):
            arm_target[0] += arm_step; arm_changed = True
        elif viewer.window.key_down("s"):
            arm_target[0] -= arm_step; arm_changed = True
        if viewer.window.key_down("a"):
            arm_target[1] += arm_step; arm_changed = True
        elif viewer.window.key_down("d"):
            arm_target[1] -= arm_step; arm_changed = True
        if viewer.window.key_down("q"):
            arm_target[2] += arm_step; arm_changed = True
        elif viewer.window.key_down("e"):
            arm_target[2] -= arm_step; arm_changed = True
        if viewer.window.key_down("r"):
            arm_target[3] += arm_step; arm_changed = True
        elif viewer.window.key_down("f"):
            arm_target[3] -= arm_step; arm_changed = True
        if viewer.window.key_down("t"):
            arm_target[4] += arm_step; arm_changed = True
        elif viewer.window.key_down("y"):
            arm_target[4] -= arm_step; arm_changed = True
        if viewer.window.key_down("z"):
            arm_target[5] += arm_step; arm_changed = True
        elif viewer.window.key_down("x"):
            arm_target[5] -= arm_step; arm_changed = True

        if viewer.window.key_down("u"):
            body_target[3] += body_step; body_changed = True
        elif viewer.window.key_down("j"):
            body_target[3] -= body_step; body_changed = True
        if viewer.window.key_down("i"):
            body_target[4] += body_step; body_changed = True
        elif viewer.window.key_down("k"):
            body_target[4] -= body_step; body_changed = True
        if viewer.window.key_down("o"):
            body_target[5] += body_step; body_changed = True
        elif viewer.window.key_down("l"):
            body_target[5] -= body_step; body_changed = True
        if viewer.window.key_down("b"):
            body_target[6] += body_step; body_changed = True
        elif viewer.window.key_down("n"):
            body_target[6] -= body_step; body_changed = True

        if arm_changed:
            for i, idx in enumerate(arm_qpos_idx):
                arm_target[i] = np.clip(arm_target[i], action_low[idx], action_high[idx])
            do_step = True
        if body_changed:
            for i, idx in enumerate(body_qpos_idx):
                body_target[i] = np.clip(body_target[i], action_low[idx], action_high[idx])
            do_step = True

        if viewer.window.key_press("g"):
            if gripper_target[0] > 0.01:
                gripper_target = np.array([0.0, 0.0], dtype=np.float32)
            else:
                gripper_target = np.array([0.04, 0.04], dtype=np.float32)
            gripper_changed = True
            do_step = True
            print(f"Gripper {'closed' if gripper_target[0] > 0.01 else 'opened'}")

        if viewer.window.key_press("0"):
            arm_target = np.array([home_qpos[i] for i in arm_qpos_idx], dtype=np.float32)
            body_target = np.array([home_qpos[i] for i in body_qpos_idx], dtype=np.float32)
            gripper_target = np.array([home_qpos[i] for i in gripper_qpos_idx], dtype=np.float32)
            do_step = True
            print("Reset to home position")

        if viewer.window.key_press("p"):
            args.seed += 1
            obs, info = env.reset(seed=[args.seed], options={'reconfigure': True})
            current_qpos = robot.get_qpos()[0].cpu().numpy()
            arm_target = np.array([current_qpos[i] for i in arm_qpos_idx], dtype=np.float32)
            body_target = np.array([current_qpos[i] for i in body_qpos_idx], dtype=np.float32)
            gripper_target = np.array([current_qpos[i] for i in gripper_qpos_idx], dtype=np.float32)
            viewer = env.render()
            print(f"New episode seed={args.seed}")
            continue

        if do_step:
            action = build_action()
            obs, reward, terminated, truncated, info = env.step(action)

    env.close()


def main():
    args = parse_args()

    sim_config = {}
    if args.sim_freq is not None:
        sim_config['sim_freq'] = args.sim_freq
    if args.control_freq is not None:
        sim_config['control_freq'] = args.control_freq

    camera_config = dict(
        width=args.camera_width,
        height=args.camera_height,
    )
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

    obs, info = env.reset(seed=[args.seed], options={'reconfigure': True})
    viewer = env.render()

    agent = env.unwrapped.agent
    robot = agent.robot

    if args.robot_uids == "ds_r1":
        run_r1(env, args, viewer, agent, robot)
    else:
        run_fetch(env, args, viewer, agent, robot)


if __name__ == "__main__":
    main()
