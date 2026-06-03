import sapien
import numpy as np

from mani_skill.utils import common
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

from dsynth.planning.utils import (
    get_tcp_pose,
    get_tcp_matrix,
    get_base_pose,
    get_shoulder_pan_pose,
    get_fcl_object_name, 
    compute_box_grasp_thin_side_info,
    compute_cylinder_grasp_info,
    is_mesh_cylindrical,
    get_base_shift_tcp_to_target,
    generate_sphere_grasp_info,
    get_head_pose,
    get_distance_tcp_to_shelf
)
from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver

def align_ee_to_target_pos(env, planner: FetchMotionPlanningSapienSolver, target_pos: np.ndarray):
    delta_h = target_pos[2] - get_tcp_pose(env).p[2]
    res = planner.lift_body(delta_h)
    if res == -1:
        return res
    
    head_pos = get_head_pose(env).p
    head_to_tcp_distance = np.linalg.norm(head_pos - get_tcp_pose(env).p)
    head_to_item_direction = (target_pos - head_pos)
    head_to_item_direction = common.np_normalize_vector(head_to_item_direction)
    target_tcp_pos = head_pos + head_to_item_direction * head_to_tcp_distance

    closing = np.cross(head_to_item_direction, [0., 0., 1.])
    closing = common.np_normalize_vector(closing)
    target_tcp_pose = env.agent.build_grasp_pose(head_to_item_direction, closing, target_tcp_pos)

    res = planner.static_manipulation(target_tcp_pose)
    return res

def align_ee_to_target_product(env, planner: FetchMotionPlanningSapienSolver, target_product_actor):
    obb = get_actor_obb(target_product_actor)
    item_center = np.array(obb.primitive.transform)[:3, 3]
    res = align_ee_to_target_pos(env, planner, item_center)
    return res

def align_to_target_pose(env, planner: FetchMotionPlanningSapienSolver, pose: sapien.Pose, offset_from_pose=1.35, rotate_max_vel=0.8):

    reset_arm_actions = planner.plan_reset_arm()
    if reset_arm_actions == -1:
        reset_arm_actions = None
        reset_arm(env, planner)

    direction = pose.to_transformation_matrix()[:3, 2]
    base_pos_near_target = pose.p - offset_from_pose * direction

    base_pos_near_target[2] = 0
    res = planner.drive_base(base_pos_near_target, direction, arm_actions=reset_arm_actions, rotate_max_vel=rotate_max_vel)
    if res == -1:
        return res

    delta_h = pose.p[2] - get_tcp_pose(env).p[2]
    res = planner.lift_body(delta_h)
    return res


def align_to_target_product(env, planner: FetchMotionPlanningSapienSolver, target_product_actor):
    obb = get_actor_obb(target_product_actor)
    item_center = np.array(obb.primitive.transform)[:3, 3]
    direction = env.directions_to_shelf[0]
    closing = np.cross(direction, [0., 0., 1.])

    target_pose = env.agent.build_grasp_pose(direction, closing, item_center)
    return align_to_target_pose(env, planner, target_pose)

def approach_and_manipulate_to_pose_in_shelf(
    env, 
    planner: FetchMotionPlanningSapienSolver,
    target_center_pos: np.ndarray,
    target_poses_list: list[sapien.Pose],
    num_tries = 5,
    switch_approach_target_from_shelf_to_target_distance = 0.05,
):
    success = False

    res = align_ee_to_target_pos(env, planner, target_center_pos)
    if res == -1:
        return res

    for try_num in range(num_tries):
        ik_solvable_graps = []

        for grasp in target_poses_list:
            planner._update_grasp_visual(grasp)
            planner.update_torso_pose()
            planner.render_wait()
            if planner.check_IK(grasp):
                res = planner.static_manipulation(grasp)
                if res != -1:
                    success = True
                    break
                ik_solvable_graps.append(grasp)

        if success:
            break

        approach_dist = get_distance_tcp_to_shelf(env)
        if approach_dist <= switch_approach_target_from_shelf_to_target_distance:
            approach_dist = np.linalg.norm(get_tcp_pose(env).p - target_center_pos)
         
        res, _ = planner.move_base_forward_delta(approach_dist / 2)
        if res == -1:
            return res

        res = align_ee_to_target_pos(env, planner, target_center_pos)
        if res == -1:
            return res
    if not success:
        return -1

    return res

def place_object_to_pos(
    env, 
    planner: FetchMotionPlanningSapienSolver, 
    target_center_pos: np.ndarray,
    target_ee_direction: np.ndarray,
    n_grasps=10, 
    num_tries = 5,
):
    FINGER_LENGTH = 0.03

    grasp_infos = generate_sphere_grasp_info(
        center=target_center_pos,
        ee_direction=target_ee_direction,
        n_grasps_central=n_grasps,
        n_grasps_lateral=n_grasps,
        central_angle_range=[-np.pi/4, np.pi/4],
        lateral_angle_range=[-np.pi/4, np.pi/4],
    )

    grasps = []
    for grasp_info in grasp_infos:
        grasp_closing, grasp_center, grasp_approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
        grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
        grasps.append(grasp_pose)

    start_tcp_pos = get_tcp_pose(env).p
    start_base_pos = get_base_pose(env).p
    
    res = approach_and_manipulate_to_pose_in_shelf(
        env, 
        planner, 
        target_center_pos, 
        grasps, 
        num_tries=num_tries, 
        # approach_to_shelf=False
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

def fetch_object_from_shelf(
    env, 
    planner: FetchMotionPlanningSapienSolver, 
    target_product_actor, 
    n_grasps=10, 
    num_tries = 5, 
):
    FINGER_LENGTH = 0.03
    obb = get_actor_obb(target_product_actor)
    target_center_pos = np.array(obb.primitive.transform)[:3, 3]

    dir_to_shelf = env.directions_to_shelf[0]
    perpendicular_to_shelf = np.cross(dir_to_shelf, [0., 0., 1.])

    if is_mesh_cylindrical(target_product_actor):
        grasp_infos = compute_cylinder_grasp_info(
            target_product_actor,
            # target_closing=get_tcp_matrix(env)[:3, 1],
            target_closing=perpendicular_to_shelf,
            # ee_direction=get_tcp_matrix(env)[:3, 2],
            ee_direction=dir_to_shelf,
            depth=FINGER_LENGTH,
            n_grasps_central=n_grasps,
            n_grasps_lateral=n_grasps,
            central_angle_range=[-np.pi/4, np.pi/4],
            lateral_angle_range=[-np.pi/4, np.pi/4],
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
        grasp_closing, grasp_center, grasp_approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
        grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
        grasps.append(grasp_pose)

    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        get_fcl_object_name(target_product_actor), True
    )
    planner.planner.update_from_simulation()

    start_tcp_pos = get_tcp_pose(env).p
    start_base_pos = get_base_pose(env).p

    res = approach_and_manipulate_to_pose_in_shelf(
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
    
    res = planner.lift_body(0.05)
    if res == -1:
        return res

    delta_approach = start_base_pos - get_base_pose(env).p
    delta_approach = np.linalg.norm(delta_approach) + 5e-2

    res = planner.move_base_forward_delta(-delta_approach)
    if res == -1:
        return res

    return res

def drop_to_basket(env, planner: FetchMotionPlanningSapienSolver):
    goal_center = env.calc_target_pose().sp.p
    goal_center = goal_center + np.array([0.05, 0., 0.4]) # add shift from base to basket

    goal_approaching = np.array([0, 0., -1.])
    goal_closing = - get_base_pose(env).to_transformation_matrix()[:3, 1]

    goal_pose = env.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)

    res = planner.lift_body(0.3)
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


def look_at_basket(env, planner: FetchMotionPlanningSapienSolver):
    base_T = get_base_pose(env).to_transformation_matrix()
    basket_center_local = np.array([0.312, 0.245, 0.182, 1.0])
    basket_center_world = (base_T @ basket_center_local)[:3]
    return look_at_target(env, planner, basket_center_world)


def look_at_target(env, planner: FetchMotionPlanningSapienSolver, target_pos: np.ndarray):
    base_T = get_base_pose(env).to_transformation_matrix()
    base_T_inv = np.eye(4)
    base_T_inv[:3, :3] = base_T[:3, :3].T
    base_T_inv[:3, 3] = -base_T[:3, :3].T @ base_T[:3, 3]

    cam_pos = None
    for link in env.agent.robot.get_links():
        if link.name == 'head_camera_link':
            cam_pos = link.pose.sp.p
            break

    if cam_pos is None:
        return -1

    cam_local = base_T_inv @ np.append(cam_pos, 1)
    target_local = base_T_inv @ np.append(target_pos, 1)
    diff = target_local[:3] - cam_local[:3]
    dx, dy, dz = diff[0], diff[1], diff[2]

    target_pan = np.arctan2(dy, dx)
    target_tilt = np.arctan2(-dz, np.sqrt(dx**2 + dy**2))

    return planner.move_head(target_pan=target_pan, target_tilt=target_tilt)


def reset_head(env, planner: FetchMotionPlanningSapienSolver):
    return planner.move_head(target_pan=0.0, target_tilt=0.0)


ARM_HOME_QPOS = np.array([0.0, 1.518, 0.0, np.pi/2, 0.0, np.pi/2, 0.0])

_SAFE_INTERMEDIATE_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_SAFE_INTERMEDIATE_2 = np.array([0.0, 1.518, 0.0, 0.0, 0.0, 0.0, 0.0])


def reset_arm(env, planner: FetchMotionPlanningSapienSolver):
    arm_qpos = env.agent.controller.controllers['arm'].qpos[0].cpu().numpy()

    if np.max(np.abs(arm_qpos - ARM_HOME_QPOS)) < 0.1:
        return 0

    reset_arm_actions = planner.plan_reset_arm()
    if reset_arm_actions != -1:
        return planner.follow_path(reset_arm_actions)

    planner.move_arm_to_qpos(_SAFE_INTERMEDIATE_1)
    planner.move_arm_to_qpos(_SAFE_INTERMEDIATE_2)
    return planner.move_arm_to_qpos(ARM_HOME_QPOS)