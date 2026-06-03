import numpy as np
import gymnasium as gym
import imageio
from dsynth.envs import *
from dsynth.robots import *
from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver
from dsynth.planning.fetch_skills import reset_arm, reset_head, ARM_HOME_QPOS

env = gym.make(
    'PickToBasketContDuffEnv',
    robot_uids='ds_fetch_basket',
    config_dir_path='generated_envs/ds_small_scene',
    num_envs=1,
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    obs_mode='none',
    sim_backend='cpu',
    render_backend='cpu',
)

obs, info = env.reset(options={'reconfigure': True})
env_uw = env.unwrapped
planner = FetchMotionPlanningSapienSolver(env_uw, debug=False, vis=False, print_env_info=False)

frames = []

def capture_frame():
    img = env.render()
    if hasattr(img, 'cpu'):
        img = img.cpu().numpy()
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if img.ndim == 4:
        img = img[0]
    frames.append(img.astype(np.uint8))

arm_qpos_before = env_uw.agent.controller.controllers['arm'].qpos[0].cpu().numpy()
print("Arm qpos BEFORE reset:", np.round(arm_qpos_before, 4))
print("Target ARM_HOME_QPOS:  ", np.round(ARM_HOME_QPOS, 4))

for _ in range(20):
    planner.idle_steps(t=1)
    capture_frame()

print("Resetting arm...")
target_straight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
arm_qpos = env_uw.agent.controller.controllers['arm'].qpos[0].cpu().numpy()
if np.max(np.abs(arm_qpos - target_straight)) > 0.3:
    print("  Step 1: Straightening arm first...")
    planner.move_arm_to_qpos(target_straight)
    for _ in range(10):
        planner.idle_steps(t=1)
        capture_frame()

print("  Step 2: Moving to fold pose...")
planner.move_arm_to_qpos(ARM_HOME_QPOS)
for _ in range(10):
    planner.idle_steps(t=1)
    capture_frame()

print("Resetting head...")
res = reset_head(env_uw, planner)
for _ in range(20):
    planner.idle_steps(t=1)
    capture_frame()

arm_qpos_after = env_uw.agent.controller.controllers['arm'].qpos[0].cpu().numpy()
print()
print("Arm qpos AFTER reset:", np.round(arm_qpos_after, 4))
print("Target ARM_HOME_QPOS:  ", np.round(ARM_HOME_QPOS, 4))

joint_names = ['shoulder_pan', 'shoulder_lift', 'upperarm_roll', 'elbow_flex', 'forearm_roll', 'wrist_flex', 'wrist_roll']
for i, name in enumerate(joint_names):
    actual = arm_qpos_after[i]
    target = ARM_HOME_QPOS[i]
    status = "OK" if abs(actual - target) < 0.1 else "LIMITED"
    print(f"  {name}: actual={actual:.4f}, target={target:.4f}, [{status}]")

output_path = 'generated_envs/ds_small_scene/demos/motionplanning/test_arm_reset/arm_reset_front_view_20260602.mp4'
imageio.mimwrite(output_path, frames, fps=30, quality=8)
print(f"\nVideo saved to: {output_path}")
print(f"Total frames: {len(frames)}")
print(f"Frame shape: {frames[0].shape}")

env.close()
