import sapien
import numpy as np

from mani_skill.utils import common
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

from dsynth.planning.utils import (
    get_tcp_pose,
    get_tcp_matrix,
    get_base_pose,
    get_fcl_object_name,
    compute_box_grasp_thin_side_info,
    compute_cylinder_grasp_info,
    is_mesh_cylindrical,
    generate_sphere_grasp_info,
    get_distance_tcp_to_shelf
)
from dsynth.planning.r1_motionplanner import R1MotionPlanningSapienSolver


def r1_align_ee_to_target_pos(env, planner: R1MotionPlanningSapienSolver, target_pos: np.ndarray):
    delta_h = target_pos[2] - get_tcp_pose(env).p[2]
    res = r1_lift_body(planner, delta_h)
    if res == -1:
        return res

    base_pos = get_base_pose(env).p
    base_to_tcp_distance = np.linalg.norm(get_tcp_pose(env).p - base_pos)
    base_to_item_direction = (target_pos - base_pos)
    base_to_item_direction[2] = 0.
    base_to_item_direction = common.np_normalize_vector(base_to_item_direction)
    target_tcp_pos = base_pos + base_to_item_direction * base_to_tcp_distance
    target_tcp_pos[2] = target_pos[2]

    closing = np.cross(base_to_item_direction, [0., 0., 1.])
    closing = common.np_normalize_vector(closing)
    approaching = np.array([0., 0., -1.])
    target_tcp_pose = env.agent.build_grasp_pose(approaching, closing, target_tcp_pos)

    res = planner.static_manipulation(target_tcp_pose)
    return res


def r1_align_ee_to_target_product(env, planner: R1MotionPlanningSapienSolver, target_product_actor):
    obb = get_actor_obb(target_product_actor)
    item_center = np.array(obb.primitive.transform)[:3, 3]
    res = r1_align_ee_to_target_pos(env, planner, item_center)
    return res


def r1_align_to_target_pose(env, planner: R1MotionPlanningSapienSolver, pose: sapien.Pose, offset_from_pose=1.0):
    reset_arm_actions = planner.plan_reset_arm()
    if reset_arm_actions == -1:
        reset_arm_actions = None

    direction = pose.to_transformation_matrix()[:3, 2]
    base_pos_near_target = pose.p - offset_from_pose * direction
    base_pos_near_target[2] = 0

    res = planner.drive_base(base_pos_near_target, direction, arm_actions=reset_arm_actions)
    if res == -1:
        return res

    delta_h = pose.p[2] - get_tcp_pose(env).p[2]
    res = r1_lift_body(planner, delta_h)
    return res


def r1_align_to_target_product(env, planner: R1MotionPlanningSapienSolver, target_product_actor):
    obb = get_actor_obb(target_product_actor)
    item_center = np.array(obb.primitive.transform)[:3, 3]
    direction = env.directions_to_shelf[0]
    closing = np.cross(direction, [0., 0., 1.])

    target_pose = env.agent.build_grasp_pose(direction, closing, item_center)
    return r1_align_to_target_pose(env, planner, target_pose)


def r1_lift_body(planner: R1MotionPlanningSapienSolver, delta_h: float,
                 k_p=1.0, k_d=0.2, tol=1e-2, max_steps=200):
    TORSO_LIFT_IDX = 6

    current_torso_qpos = planner.robot.get_qpos().cpu().numpy()[0, TORSO_LIFT_IDX]
    qlimits = planner.robot.qlimits[0, TORSO_LIFT_IDX].cpu().numpy()

    target_torso_qpos = current_torso_qpos + delta_h
    target_torso_qpos = np.clip(target_torso_qpos, qlimits[0], qlimits[1])
    true_delta_h = target_torso_qpos - current_torso_qpos

    if abs(true_delta_h) < tol:
        return planner.idle_steps(t=1)

    arm_action = planner.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = planner.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
    gripper_state = planner.gripper_state

    last_error = 0.0
    dt = 1 / planner.base_env.control_freq
    n_steps = 0

    while True:
        current_torso_qpos = planner.robot.get_qpos().cpu().numpy()[0, TORSO_LIFT_IDX]
        current_error = target_torso_qpos - current_torso_qpos
        error_diff = (current_error - last_error) / dt
        last_error = current_error

        if np.abs(current_error) < tol:
            planner.update_base_pose()
            planner.planner.update_from_simulation()
            return planner.idle_steps(t=1)

        control_delta = k_p * current_error + k_d * error_diff
        control_delta = np.clip(control_delta, -0.2, 0.2)

        body_action[TORSO_LIFT_IDX - 3] = current_torso_qpos + control_delta
        action = planner._build_action(arm_action, gripper_state, body_action)

        obs, reward, terminated, truncated, info = planner.env.step(action)
        n_steps += 1
        planner.update_base_pose()
        planner.elapsed_steps += 1

        if planner.print_env_info:
            print(f"[{planner.elapsed_steps:3}] Env Output: reward={reward} info={info}")
        if planner.vis:
            planner.base_env.render_human()

        if n_steps > max_steps:
            print("Reached max steps in r1_lift_body.")
            return -1


def r1_approach_and_manipulate_to_pose_in_shelf(
    env,
    planner: R1MotionPlanningSapienSolver,
    target_center_pos: np.ndarray,
    target_poses_list: list,
    num_tries=5,
    switch_approach_target_from_shelf_to_target_distance=0.05,
):
    success = False

    res = r1_align_ee_to_target_pos(env, planner, target_center_pos)
    if res == -1:
        return res

    for try_num in range(num_tries):
        ik_solvable_grasps = []

        for grasp in target_poses_list:
            planner._update_grasp_visual(grasp)
            planner.update_base_pose()
            planner.render_wait()
            if planner.check_IK(grasp):
                res = planner.static_manipulation(grasp)
                if res != -1:
                    success = True
                    break
                ik_solvable_grasps.append(grasp)

        if success:
            break

        approach_dist = get_distance_tcp_to_shelf(env)
        if approach_dist <= switch_approach_target_from_shelf_to_target_distance:
            approach_dist = np.linalg.norm(get_tcp_pose(env).p - target_center_pos)

        res, _ = planner.move_base_forward_delta(approach_dist / 2)
        if res == -1:
            return res

        res = r1_align_ee_to_target_pos(env, planner, target_center_pos)
        if res == -1:
            return res

    if not success:
        return -1

    return res


def r1_fetch_object_from_shelf(
    env,
    planner: R1MotionPlanningSapienSolver,
    target_product_actor,
    n_grasps=10,
    num_tries=5,
):
    FINGER_LENGTH = 0.03
    obb = get_actor_obb(target_product_actor)
    target_center_pos = np.array(obb.primitive.transform)[:3, 3]

    dir_to_shelf = env.directions_to_shelf[0]
    perpendicular_to_shelf = np.cross(dir_to_shelf, [0., 0., 1.])

    if is_mesh_cylindrical(target_product_actor):
        grasp_infos = compute_cylinder_grasp_info(
            target_product_actor,
            target_closing=perpendicular_to_shelf,
            ee_direction=dir_to_shelf,
            depth=FINGER_LENGTH,
            n_grasps_central=n_grasps,
            n_grasps_lateral=n_grasps,
            central_angle_range=[-np.pi / 4, np.pi / 4],
            lateral_angle_range=[-np.pi / 4, np.pi / 4],
        )
    else:
        grasp_infos = compute_box_grasp_thin_side_info(
            obb,
            target_closing=get_tcp_matrix(env)[:3, 1],
            ee_direction=get_tcp_matrix(env)[:3, 2],
            depth=FINGER_LENGTH,
            n_grasps=n_grasps,
        )

    grasps = []
    for grasp_info in grasp_infos:
        grasp_closing = grasp_info["closing"]
        grasp_center = grasp_info["center"]
        grasp_approaching = grasp_info["approaching"]
        grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
        grasps.append(grasp_pose)

    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        get_fcl_object_name(target_product_actor), True
    )
    planner.planner.update_from_simulation()

    start_base_pos = get_base_pose(env).p

    res = r1_approach_and_manipulate_to_pose_in_shelf(
        env,
        planner,
        target_center_pos,
        grasps,
        num_tries=num_tries,
    )
    if res == -1:
        return res

    res = planner.close_gripper()
    if res == -1:
        return res

    res = r1_lift_body(planner, 0.05)
    if res == -1:
        return res

    delta_approach = start_base_pos - get_base_pose(env).p
    delta_approach = np.linalg.norm(delta_approach) + 5e-2

    res = planner.move_base_forward_delta(-delta_approach)
    if res == -1:
        return res

    return res


def r1_place_object_to_pos(
    env,
    planner: R1MotionPlanningSapienSolver,
    target_center_pos: np.ndarray,
    target_ee_direction: np.ndarray,
    n_grasps=10,
    num_tries=5,
):
    FINGER_LENGTH = 0.03

    grasp_infos = generate_sphere_grasp_info(
        center=target_center_pos,
        ee_direction=target_ee_direction,
        n_grasps_central=n_grasps,
        n_grasps_lateral=n_grasps,
        central_angle_range=[-np.pi / 4, np.pi / 4],
        lateral_angle_range=[-np.pi / 4, np.pi / 4],
    )

    grasps = []
    for grasp_info in grasp_infos:
        grasp_closing = grasp_info["closing"]
        grasp_center = grasp_info["center"]
        grasp_approaching = grasp_info["approaching"]
        grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
        grasps.append(grasp_pose)

    start_base_pos = get_base_pose(env).p

    res = r1_approach_and_manipulate_to_pose_in_shelf(
        env,
        planner,
        target_center_pos,
        grasps,
        num_tries=num_tries,
    )
    if res == -1:
        return res

    res = planner.open_gripper()
    if res == -1:
        return res

    delta_approach = start_base_pos - get_base_pose(env).p
    delta_approach = np.linalg.norm(delta_approach) + 5e-2

    res = planner.move_base_forward_delta(-delta_approach)
    if res == -1:
        return res

    return res


def r1_drop_to_basket(env, planner: R1MotionPlanningSapienSolver):
    goal_center = env.calc_target_pose().sp.p
    goal_center = goal_center + np.array([0.05, 0., 0.4])

    goal_approaching = np.array([0, 0., -1.])
    goal_closing = -get_base_pose(env).to_transformation_matrix()[:3, 1]

    goal_pose = env.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)

    res = r1_lift_body(planner, 0.3)
    if res == -1:
        return res

    res = planner.static_manipulation(goal_pose)
    if res == -1:
        return res

    res = planner.open_gripper()
    if res == -1:
        return res

    res = planner.idle_steps(t=10)
    if res == -1:
        return res
    return res
