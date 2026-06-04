"""Restock: pick item from robot basket -> place back on shelf empty slot.

Mirrors pick_from_floor exactly, with two intentional differences:
  - No "drive to the object" phase: the target is already in the robot basket
    and the robot already faces the shelf, so we start from looking at it.
  - Head control: look at the basket target BEFORE grasping, and only reset the
    head to forward AFTER the grasp is confirmed.
Everything else (grasp computation, attach, lift, drive to shelf, place) is the
same logic as solve_fetch_pick_from_floor_cont.
"""
import numpy as np
import sapien
from dsynth.envs import *
from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver
from dsynth.planning.utils import (
    get_fcl_object_name,
    BAD_ENV_ERROR_CODE,
)
from dsynth.planning.fetch_skills import (
    align_to_target_pose,
    place_object_to_pos,
    look_at_target,
    reset_head,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

MAX_GRASP_RETRIES = 3


def _get_obb_center(obb):
    return np.array(obb.primitive.transform)[:3, 3].copy()


def _get_item_z(target_actor):
    return np.array(get_actor_obb(target_actor).primitive.transform)[2, 3]


def solve_fetch_restock_basket_to_shelf(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        disable_actors_collision=False,
        verbose=debug,
    )

    def get_base_pose():
        return env.agent.base_link.pose

    def get_tcp_pose():
        return env.agent.tcp.pose

    def get_tcp_matrix():
        return get_tcp_pose().to_transformation_matrix()[0].cpu().numpy()

    if len(planner.planner.planning_world.check_collision()) > 0:
        return BAD_ENV_ERROR_CODE

    FINGER_LENGTH = 0.04
    env = env.unwrapped

    # Let the scene settle (~0.5s) before doing anything: items dropped into the
    # basket on reset are still moving, so grasp poses computed too early are stale.
    settle_steps = max(1, int(0.5 * planner.base_env.control_freq))
    planner.idle_steps(t=settle_steps)
    planner.planner.update_from_simulation()

    # ------------------------------------------------------------------ #
    # Target product (the item sitting in the robot's basket)
    # ------------------------------------------------------------------ #
    target_product_name = ''
    max_dist = np.inf
    for actor_name in env.target_products_df['actor_name']:
        prod_pos = env.actors['products'][actor_name].pose.sp.p
        d = np.linalg.norm(prod_pos - get_base_pose().sp.p)
        if d < max_dist:
            max_dist = d
            target_product_name = actor_name
    target_product_actor = env.actors['products'][target_product_name]
    target_fcl = get_fcl_object_name(target_product_actor)

    # Whitelist every item in the basket so reaching for the target does not
    # fail IK on neighbouring distractor items.
    basket_names = getattr(env, 'basket_item_names', {}).get(0, [target_product_name])
    acm = planner.planner.planning_world.get_allowed_collision_matrix()
    for item_name in basket_names:
        acm.set_default_entry(get_fcl_object_name(env.actors['products'][item_name]), True)
    planner.planner.update_from_simulation()

    # ------------------------------------------------------------------ #
    # Phase 1: Grasp from basket (head looks at target during the grasp)
    # ------------------------------------------------------------------ #
    grasp_success = False
    for attempt in range(MAX_GRASP_RETRIES):
        # settle, then re-read the target's CURRENT position every attempt and
        # plan the grasp from that fresh pose (a failed attempt may have nudged it).
        planner.idle_steps(t=max(1, int(0.2 * planner.base_env.control_freq)))
        planner.planner.update_from_simulation()
        obb = get_actor_obb(target_product_actor)
        target_product_center = _get_obb_center(obb)

        # look at the basket target BEFORE grasping
        res = look_at_target(env, planner, target_product_center)
        if res == -1:
            continue

        z_before = _get_item_z(target_product_actor)

        grasp_info = compute_grasp_info_by_obb(
            obb, approaching=[0, 0, -1],
            target_closing=get_tcp_matrix()[:3, 1], depth=FINGER_LENGTH)
        grasp_pose = env.agent.build_grasp_pose(
            grasp_info["approaching"], grasp_info["closing"], grasp_info["center"])
        pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.2])

        # 1. pre-grasp 0.2m above target (lift joint locked, arm only)
        res = planner.static_manipulation(pre_grasp_pose, n_init_qpos=100, disable_lift_joint=True)
        if res == -1:
            print(f"[Grasp] pre-grasp failed attempt {attempt + 1}")
            continue
        # 2. descend straight down onto the target
        res = planner.static_manipulation(grasp_pose, n_init_qpos=100, disable_lift_joint=False)
        if res == -1:
            print(f"[Grasp] descend failed attempt {attempt + 1}")
            continue

        planner.close_gripper()
        planner.planner.update_from_simulation()

        # lift clear of the basket to verify the object is actually held
        planner.static_manipulation(pre_grasp_pose, n_init_qpos=100, disable_lift_joint=False)
        planner.planner.update_from_simulation()

        if _get_item_z(target_product_actor) - z_before > 0.02:
            grasp_success = True
            print(f"[Grasp] SUCCESS on attempt {attempt + 1}")
            break

        print(f"[Grasp] FAILED attempt {attempt + 1}, object not held — retrying")
        planner.open_gripper()
        planner.planner.update_from_simulation()

    if not grasp_success:
        print(f"[Grasp] All {MAX_GRASP_RETRIES} attempts failed")
        return -1

    # attach the held object so planning treats it as part of the arm
    planner.planner.planning_world.attach_object(
        name=target_fcl,
        art_name='scene-0_ds_fetch_basket_1',
        link_id=planner.planner.move_group_link_id)
    planner.planner.update_from_simulation()

    # reset head to forward ONLY after grasp confirmed
    res = reset_head(env, planner)
    if res == -1:
        return res

    # ------------------------------------------------------------------ #
    # Phase 2: Place on shelf — identical to move_from_board_to_board:
    #   align_to_target_pose (drive + rotate the base to face the shelf slot)
    #   then place_object_to_pos (gradual-approach placement: try IK on a fan
    #   of candidate poses, and if the arm can't reach, step the base forward
    #   and retry until it can). Low rotate speed so the carried item is not
    #   flung out on the turn.
    # ------------------------------------------------------------------ #
    shelf_target_p = env.products_initial_poses[target_product_name][0, :3].cpu().numpy().copy()
    target_center = shelf_target_p.copy()
    target_center[2] += 0.15

    direction = env.directions_to_shelf[0]
    closing = np.cross(direction, [0., 0., 1.])
    final_pose = env.agent.build_grasp_pose(direction, closing, target_center)

    res = align_to_target_pose(env, planner, final_pose, rotate_max_vel=0.3)
    if res == -1:
        return res

    res = place_object_to_pos(env, planner, target_center, direction, n_grasps=10)
    if res == -1:
        return res

    res = planner.idle_steps(t=1)
    planner.render_wait()
    return res

