import numpy as np
import gymnasium as gym
from dsynth.envs import *
from dsynth.robots import *
from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver

MANISKILL_FETCH_REST_ARM = np.array([-0.370, -1.032, 0.695, 0.955, -0.1, 2.077, 0.0])

DSFETCHBASKET_REST_ARM = np.array([0.0, 1.518, 0.0, np.pi/2, 0.0, np.pi/2, 0.0])

STRAIGHT_ARM = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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

joint_names = ['shoulder_pan', 'shoulder_lift', 'upperarm_roll', 'elbow_flex', 'forearm_roll', 'wrist_flex', 'wrist_roll']

def check_collision_at_qpos(qpos, label):
    planner.move_arm_to_qpos(qpos)
    planner.planner.update_from_simulation()
    collisions = planner.planner.planning_world.check_collision()
    print(f"\n[{label}] qpos = {np.round(qpos, 4)}")
    print(f"  Collisions: {len(collisions)}")
    if len(collisions) > 0:
        for c in collisions:
            print(f"    {c.link_name1} <-> {c.link_name2}")
    return len(collisions)

def try_plan_to(target_qpos, label):
    planner.planner.update_from_simulation()
    current_qpos = env_uw.agent.controller.controllers['arm'].qpos[0].cpu().numpy()
    print(f"\n[{label}] Planning from current to target...")
    print(f"  Current: {np.round(current_qpos, 3)}")
    print(f"  Target:  {np.round(target_qpos, 3)}")
    result = planner.plan_reset_arm()
    if result == -1:
        print(f"  RRT plan: FAILED")
    else:
        n_steps = len(result.get('position', []))
        print(f"  RRT plan: SUCCESS ({n_steps} waypoints)")
    return result

print("=" * 60)
print("Test 1: Check ManiSkill Fetch rest keyframe self-collision")
print("=" * 60)
planner.move_arm_to_qpos(STRAIGHT_ARM)
planner.idle_steps(t=5)
n_col = check_collision_at_qpos(MANISKILL_FETCH_REST_ARM, "ManiSkill Fetch Rest")

print("\n" + "=" * 60)
print("Test 2: Check DSFetchBasket rest keyframe self-collision")
print("=" * 60)
planner.move_arm_to_qpos(STRAIGHT_ARM)
planner.idle_steps(t=5)
n_col2 = check_collision_at_qpos(DSFETCHBASKET_REST_ARM, "DSFetchBasket Rest")

print("\n" + "=" * 60)
print("Test 3: RRT plan from straight to ManiSkill Fetch rest")
print("=" * 60)
planner.move_arm_to_qpos(STRAIGHT_ARM)
planner.idle_steps(t=5)

env_uw.agent.keyframes['rest'].qpos[5] = MANISKILL_FETCH_REST_ARM[0]
env_uw.agent.keyframes['rest'].qpos[7] = MANISKILL_FETCH_REST_ARM[1]
env_uw.agent.keyframes['rest'].qpos[8] = MANISKILL_FETCH_REST_ARM[2]
env_uw.agent.keyframes['rest'].qpos[9] = MANISKILL_FETCH_REST_ARM[3]
env_uw.agent.keyframes['rest'].qpos[10] = MANISKILL_FETCH_REST_ARM[4]
env_uw.agent.keyframes['rest'].qpos[11] = MANISKILL_FETCH_REST_ARM[5]
env_uw.agent.keyframes['rest'].qpos[12] = MANISKILL_FETCH_REST_ARM[6]
result1 = try_plan_to(MANISKILL_FETCH_REST_ARM, "Straight -> ManiSkill Rest")

print("\n" + "=" * 60)
print("Test 4: RRT plan from straight to DSFetchBasket rest")
print("=" * 60)
env_uw.agent.keyframes['rest'].qpos[5] = DSFETCHBASKET_REST_ARM[0]
env_uw.agent.keyframes['rest'].qpos[7] = DSFETCHBASKET_REST_ARM[1]
env_uw.agent.keyframes['rest'].qpos[8] = DSFETCHBASKET_REST_ARM[2]
env_uw.agent.keyframes['rest'].qpos[9] = DSFETCHBASKET_REST_ARM[3]
env_uw.agent.keyframes['rest'].qpos[10] = DSFETCHBASKET_REST_ARM[4]
env_uw.agent.keyframes['rest'].qpos[11] = DSFETCHBASKET_REST_ARM[5]
env_uw.agent.keyframes['rest'].qpos[12] = DSFETCHBASKET_REST_ARM[6]
planner.move_arm_to_qpos(STRAIGHT_ARM)
planner.idle_steps(t=5)
result2 = try_plan_to(DSFETCHBASKET_REST_ARM, "Straight -> DSFetchBasket Rest")

print("\n" + "=" * 60)
print("Test 5: RRT plan from random poses to ManiSkill Fetch rest")
print("=" * 60)
env_uw.agent.keyframes['rest'].qpos[5] = MANISKILL_FETCH_REST_ARM[0]
env_uw.agent.keyframes['rest'].qpos[7] = MANISKILL_FETCH_REST_ARM[1]
env_uw.agent.keyframes['rest'].qpos[8] = MANISKILL_FETCH_REST_ARM[2]
env_uw.agent.keyframes['rest'].qpos[9] = MANISKILL_FETCH_REST_ARM[3]
env_uw.agent.keyframes['rest'].qpos[10] = MANISKILL_FETCH_REST_ARM[4]
env_uw.agent.keyframes['rest'].qpos[11] = MANISKILL_FETCH_REST_ARM[5]
env_uw.agent.keyframes['rest'].qpos[12] = MANISKILL_FETCH_REST_ARM[6]

random_poses = [
    np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    np.array([-0.5, 0.8, -0.3, 1.0, 0.5, 1.5, 0.3]),
    np.array([1.0, -0.5, 1.0, -0.5, 1.0, -0.5, 1.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
]

success_count = 0
for i, pose in enumerate(random_poses):
    planner.move_arm_to_qpos(STRAIGHT_ARM)
    planner.idle_steps(t=3)
    planner.move_arm_to_qpos(pose)
    planner.idle_steps(t=3)
    result = try_plan_to(MANISKILL_FETCH_REST_ARM, f"Random #{i+1} -> ManiSkill Rest")
    if result != -1:
        success_count += 1

print(f"\n  RRT planning success: {success_count}/{len(random_poses)}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"ManiSkill Fetch rest self-collision: {'YES - PROBLEM' if n_col > 0 else 'NO - OK'}")
print(f"DSFetchBasket rest self-collision:   {'YES - PROBLEM' if n_col2 > 0 else 'NO - OK'}")
print(f"RRT to ManiSkill rest (from straight): {'SUCCESS' if result1 != -1 else 'FAILED'}")
print(f"RRT to DSFetchBasket rest (from straight): {'SUCCESS' if result2 != -1 else 'FAILED'}")
print(f"RRT to ManiSkill rest (from random): {success_count}/{len(random_poses)}")

env.close()
