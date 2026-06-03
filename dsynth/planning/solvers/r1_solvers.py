import numpy as np
np.set_printoptions(suppress=True)

from dsynth.envs import *
from dsynth.planning.r1_motionplanner import R1MotionPlanningSapienSolver
from dsynth.planning.utils import (
    get_base_pose,
    BAD_ENV_ERROR_CODE,
)
from dsynth.planning.r1_skills import (
    r1_align_to_target_product,
    r1_fetch_object_from_shelf,
    r1_drop_to_basket,
    r1_place_object_to_pos,
    r1_align_to_target_pose,
)


def solve_r1_pick_to_basket_cont_one_prod_w_skills(env, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = R1MotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        verbose=debug,
    )

    collisions = planner.planner.planning_world.check_collision()
    external_collisions = [c for c in collisions if c.object_name1 != c.object_name2]
    if len(external_collisions) > 0:
        return BAD_ENV_ERROR_CODE

    env = env.unwrapped

    max_dist = np.inf
    target_product_name = ''
    for target_actor_name in env.target_products_df['actor_name']:
        prod_pos = env.actors['products'][target_actor_name].pose.sp.p
        if np.linalg.norm(prod_pos - get_base_pose(env).p) < max_dist:
            max_dist = np.linalg.norm(prod_pos - get_base_pose(env).p)
            target_product_name = target_actor_name

    target_product_actor = env.actors['products'][target_product_name]

    res = r1_align_to_target_product(env, planner, target_product_actor)
    if res == -1:
        return res

    res = r1_fetch_object_from_shelf(env, planner, target_product_actor, n_grasps=10, num_tries=5)
    if res == -1:
        return res

    return r1_drop_to_basket(env, planner)


def solve_r1_move_to_board_cont_one_prod_w_skills(env, seed=None, debug=False, vis=False):
    from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb

    env.reset(seed=seed, options={'reconfigure': True})
    planner = R1MotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        verbose=debug,
    )

    collisions = planner.planner.planning_world.check_collision()
    external_collisions = [c for c in collisions if c.object_name1 != c.object_name2]
    if len(external_collisions) > 0:
        return BAD_ENV_ERROR_CODE

    env = env.unwrapped

    max_dist = np.inf
    target_product_name = ''
    for target_actor_name in env.target_products_df['actor_name']:
        prod_pos = env.actors['products'][target_actor_name].pose.sp.p
        if np.linalg.norm(prod_pos - get_base_pose(env).p) < max_dist:
            max_dist = np.linalg.norm(prod_pos - get_base_pose(env).p)
            target_product_name = target_actor_name

    target_product_actor = env.actors['products'][target_product_name]

    obb = get_actor_obb(target_product_actor)
    target_center = np.array(obb.primitive.transform)[:3, 3]
    target_center[2] += env.get_interboard_height() + 0.05
    direction = env.directions_to_shelf[0]
    closing = np.cross(direction, [0., 0., 1.])
    final_pose = env.agent.build_grasp_pose(direction, closing, target_center)

    res = r1_align_to_target_product(env, planner, target_product_actor)
    if res == -1:
        return res

    res = r1_fetch_object_from_shelf(env, planner, target_product_actor, n_grasps=10, num_tries=5)
    if res == -1:
        return res

    res = r1_align_to_target_pose(env, planner, final_pose)
    if res == -1:
        planner.render_wait()
        return res

    res = r1_place_object_to_pos(env, planner, target_center, direction, n_grasps=10)
    if res == -1:
        return res

    res = planner.idle_steps(t=1)
    return res


def solve_r1_pick_from_floor_cont(env, seed=None, debug=False, vis=False):
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb, get_actor_obb,
    )

    env.reset(seed=seed, options={'reconfigure': True})
    planner = R1MotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        verbose=debug,
    )

    collisions = planner.planner.planning_world.check_collision()
    external_collisions = [c for c in collisions if c.object_name1 != c.object_name2]
    if len(external_collisions) > 0:
        return BAD_ENV_ERROR_CODE

    FINGER_LENGTH = 0.04
    env = env.unwrapped

    target_pose = env.target_pose[0]
    target_center = target_pose.sp.p
    target_center[2] += 0.15

    target_product_actor = env.actors['products'][env.fallen_items[0]]
    obb = get_actor_obb(target_product_actor)
    target_product_center = np.array(obb.primitive.transform)[:3, 3]

    drive_vec = target_product_center - get_base_pose(env).p
    drive_vec[2] = 0

    if np.linalg.norm(drive_vec) > 0.5:
        drive_pos = get_base_pose(env).p + drive_vec * (1 - 0.5 / np.linalg.norm(drive_vec))
        res = planner.drive_base(drive_pos)
    else:
        res = planner.rotate_base_z(drive_vec)

    planner.planner.update_from_simulation()

    obb = get_actor_obb(target_product_actor)
    target_product_center = np.array(obb.primitive.transform)[:3, 3]

    from dsynth.planning.utils import get_tcp_matrix
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=[0, 0, -1],
        target_closing=get_tcp_matrix(env)[:3, 1],
        depth=FINGER_LENGTH,
    )

    grasp_closing = grasp_info["closing"]
    grasp_center = grasp_info["center"]
    grasp_approaching = grasp_info["approaching"]
    grasp_pose = env.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)
    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.3])

    res = planner.static_manipulation(pre_grasp_pose)
    if res == -1:
        return res

    from dsynth.planning.utils import get_fcl_object_name
    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        get_fcl_object_name(target_product_actor), True
    )

    res = planner.static_manipulation(grasp_pose)
    if res == -1:
        return res

    planner.planner.update_from_simulation()
    res = planner.close_gripper()

    kwargs = {
        "name": get_fcl_object_name(target_product_actor),
        "art_name": planner.planner.planning_world.get_articulation_names()[0],
        "link_id": planner.planner.move_group_link_id,
    }
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()

    res = planner.static_manipulation(pre_grasp_pose)
    if res == -1:
        return res

    planner.planner.update_from_simulation()

    actor_shelf_name = env.active_shelves[0][0]
    shelf_pose = env.actors["fixtures"]["shelves"][actor_shelf_name].pose.sp
    origin = shelf_pose.p - 1.4 * env.directions_to_shelf[0]

    res = planner.drive_base(origin)

    view_to_target = target_center - get_base_pose(env).p
    view_to_target[2] = 0.
    res = planner.rotate_base_z(view_to_target)

    from dsynth.planning.r1_skills import r1_lift_body
    res = r1_lift_body(planner, target_center[2] - get_tcp_pose(env).p[2])
    if res == -1:
        return res

    from dsynth.planning.utils import get_tcp_pose
    tcp_pose = get_tcp_pose(env).sp
    drive_vec = target_center - tcp_pose.to_transformation_matrix()[:3, 3]
    drive_vec[2] = 0
    drive_delta = drive_vec * (1 - 0.2 / np.linalg.norm(drive_vec))
    drive_pos = get_base_pose(env).p + drive_delta

    res = planner.drive_base(drive_pos)

    target_approaching = env.directions_to_shelf[0]
    target_closing = np.cross(target_approaching, [0, 0, 1])
    target_pose = env.agent.build_grasp_pose(target_approaching, target_closing, target_center)

    res = planner.static_manipulation(target_pose)
    if res == -1:
        return res

    planner.planner.update_from_simulation()
    res = planner.open_gripper()
    res = planner.idle_steps(t=10)

    planner.render_wait()
    return res
