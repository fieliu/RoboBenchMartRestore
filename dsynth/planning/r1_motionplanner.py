import mplib
import numpy as np
from collections import deque
import sapien
from transforms3d.euler import euler2quat
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import build_two_finger_gripper_grasp_pose_visual

from dsynth.planning.utils import SapienPlanningWorldV2
from mplib.sapien_utils import SapienPlanner

OPEN = np.array([1.0, 1.0])
CLOSED = np.array([-1.0, -1.0])

R1_ARM_LINK_NAMES = [
    'torso_link1', 'torso_link2', 'torso_link3', 'torso_link4',
    'left_arm_base_link', 'left_arm_link1', 'left_arm_link2',
    'left_arm_link3', 'left_arm_link4', 'left_arm_link5', 'left_arm_link6',
    'left_gripper_link', 'left_gripper_finger_link1', 'left_gripper_finger_link2',
]

R1_ARM_JOINT_NAMES = [
    'torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4',
    'left_arm_joint1', 'left_arm_joint2', 'left_arm_joint3',
    'left_arm_joint4', 'left_arm_joint5', 'left_arm_joint6',
    'left_gripper_finger_joint1', 'left_gripper_finger_joint2',
]


class R1MotionPlanningSapienSolver:
    OPEN = np.array([1.0, 1.0])
    CLOSED = np.array([-1.0, -1.0])
    MOVE_GROUP = "left_gripper_link"

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        visualize_target_grasp_pose: bool = True,
        verbose: bool = False,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self._sim_scene: sapien.Scene = self.base_env.scene.sub_scenes[0]

        self.planner = self.setup_planner()
        self.update_base_pose()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.verbose = verbose
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

        self.gripper_state = self.OPEN
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_two_finger_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp_pose)

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self, objects=[]):
        link_names = R1_ARM_LINK_NAMES
        joint_names = R1_ARM_JOINT_NAMES

        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(
            self._sim_scene,
            user_link_names=link_names,
            user_joint_names=joint_names,
            planned_articulations=[planned_articulation],
            planned_urdf_paths=[self.env_agent.urdf_arm_ik_path],
            disable_actors_collision=False,
            new_package_keyword="",
            use_convex=False,
            verbose=False,
            qpos_offset=3,
            base_link_name="torso_link1",
        )
        planner = SapienPlanner(
            planning_world,
            "left_gripper_link",
            joint_vel_limits=np.ones(10) * self.joint_vel_limits,
            joint_acc_limits=np.ones(10) * self.joint_acc_limits,
        )
        return planner

    def update_base_pose(self):
        base_pose = self.base_env.agent.torso_link1.pose.sp
        self.planner.set_base_pose(mplib.Pose(base_pose.p, base_pose.q))

    def _update_grasp_visual(self, target: sapien.Pose) -> None:
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target)

    def _transform_pose_for_planning(self, target: sapien.Pose) -> sapien.Pose:
        return target

    def _get_arm_qpos_indices(self):
        return slice(7, 13)

    def _get_body_qpos_indices(self):
        return slice(0, 7)

    def _build_action(self, arm_qpos, gripper_state, body_qpos):
        return np.hstack([body_qpos, arm_qpos, gripper_state])

    def base_x_direction(self):
        base_link_pose = self.base_env.agent.base_link.pose.sp
        return base_link_pose.to_transformation_matrix()[:3, 0]

    def rotate_base_z(
        self,
        new_direction,
        arm_actions={"position": [], "velocity": []},
        tol=1e-2,
        k_p=1.7,
        k_d=0.01,
        max_vel=0.8,
        max_stuck_steps: int = 20,
        stuck_tol: float = 1e-3,
        abort_when_collision: bool = True,
        steps_to_abortion: int = 300,
    ):
        def _calc_current_error(target_direction):
            cur_x_direction = self.base_x_direction()
            current_error = np.arccos(
                np.clip(
                    np.dot(target_direction, cur_x_direction)
                    / np.linalg.norm(cur_x_direction)
                    / np.linalg.norm(target_direction),
                    -1,
                    1,
                )
            )
            if np.cross(cur_x_direction, target_direction)[2] < 0:
                current_error = -current_error
            return current_error

        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        action = self._build_action(arm_action, self.gripper_state, body_action)

        last_error = 0
        dt = 1 / self.base_env.control_freq

        if self.grasp_pose_visual is not None:
            current_error = _calc_current_error(new_direction)
            base_link_pose = self.base_env.agent.base_link.pose.sp
            tcp_pose = self.env.unwrapped.agent.tcp.pose.sp
            rotation_wrt_base_link = sapien.Pose(q=euler2quat(0, 0, current_error))
            target_tcp_pose = (
                base_link_pose
                * rotation_wrt_base_link
                * base_link_pose.inv()
                * tcp_pose
            )
            self.grasp_pose_visual.set_pose(target_tcp_pose)

        self.render_wait()

        n_steps = 0
        rotations = deque(maxlen=max_stuck_steps)
        while True:
            cur_x_direction = self.base_x_direction()
            current_error = np.arccos(
                np.clip(
                    np.dot(new_direction, cur_x_direction)
                    / np.linalg.norm(cur_x_direction)
                    / np.linalg.norm(new_direction),
                    -1,
                    1,
                )
            )

            rotations.append(current_error)

            if (
                len(rotations) >= max_stuck_steps
                and np.std(rotations) < stuck_tol
            ):
                print("Robot is stuck")
                self.planner.update_from_simulation()
                return self.idle_steps(t=1), arm_actions

            if np.cross(cur_x_direction, new_direction)[2] < 0:
                current_error = -current_error

            error_diff = (current_error - last_error) / dt
            last_error = current_error

            if (
                np.abs(current_error) < tol
                and np.abs(error_diff) < tol**2
            ):
                self.planner.update_from_simulation()
                return self.idle_steps(t=1), arm_actions

            control_omega = k_p * current_error + k_d * error_diff
            control_omega = np.clip(control_omega, -max_vel, max_vel)

            if len(arm_actions["position"]) > 0:
                arm_action = arm_actions["position"][0]
                arm_actions["position"] = np.delete(
                    arm_actions["position"], 0, axis=0
                )
                arm_actions["velocity"] = np.delete(
                    arm_actions["velocity"], 0, axis=0
                )

            action[7:13] = arm_action
            action[2] = control_omega

            if self.verbose:
                print(
                    f"n_steps: {n_steps} base Action:",
                    np.round(action[:3], 4),
                )
                print(
                    "Full: ",
                    np.round(self.robot.get_qpos().cpu().numpy()[0], 4),
                )
            obs, reward, terminated, truncated, info = self.env.step(action)
            n_steps += 1
            self.update_base_pose()

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

            if abort_when_collision:
                if len(
                    collisions := self.planner.planning_world.check_collision()
                ) > 0:
                    print("Collision detected while rotating base")
                    for collision in collisions:
                        print(
                            f"Collision between {collision.link_name1} of entity "
                            f"{collision.object_name1} with {collision.link_name2} "
                            f"of entity {collision.object_name2}"
                        )
                    return -1, None
            if n_steps > steps_to_abortion:
                print("Reached max steps. Something went wrong.")
                return -1, None

    def drive_base(self, target_pos=None, target_view_vec=None, arm_actions=None):
        if arm_actions is None:
            arm_actions = {"position": [], "velocity": []}

        if target_pos is not None:
            moving_direction = target_pos - self.base_env.agent.base_link.pose.sp.p
            moving_direction[2] = 0.0

            if np.linalg.norm(moving_direction) < 1e-2:
                res = self.idle_steps(t=1)
                if res == -1:
                    return res
                self.planner.update_from_simulation()
            else:
                res, arm_actions = self.rotate_base_z(
                    moving_direction, arm_actions=arm_actions
                )
                if res == -1:
                    return res
                self.planner.update_from_simulation()

                delta = np.linalg.norm(moving_direction)
                res, arm_actions = self.move_base_forward_delta(
                    delta, arm_actions=arm_actions
                )
                if res == -1:
                    return res
                self.planner.update_from_simulation()

        if target_view_vec is not None:
            res, arm_actions = self.rotate_base_z(
                target_view_vec, arm_actions=arm_actions
            )
            if res == -1:
                return res

        if len(arm_actions["position"]) > 0:
            res = self.follow_path(arm_actions)
            if res == -1:
                return res
        self.planner.update_from_simulation()

        return res

    def plan_reset_arm(self):
        self.planner.update_from_simulation()
        self.update_base_pose()
        current_qpos = self.robot.get_qpos().cpu().numpy()[0, 3:13]
        goal_qpos = [self.env_agent.keyframes["rest"].qpos[3:13]]
        if np.all(np.isclose(goal_qpos, current_qpos)):
            return {"position": [], "velocity": []}
        result = self.planner.plan_qpos(
            goal_qpos,
            current_qpos,
            time_step=0.1,
            rrt_range=0.1,
            planning_time=1,
            fix_joint_limits=True,
            simplify=True,
            constraint_function=None,
            constraint_jacobian=None,
            constraint_tolerance=1e-3,
            verbose=self.verbose,
        )
        if result["status"] != "Success":
            return -1
        return result

    def base_x_pos(self):
        base_link_pose = self.base_env.agent.base_link.pose.sp
        return base_link_pose.to_transformation_matrix()[:3, 3]

    def move_base_forward_delta(
        self,
        delta,
        arm_actions={"position": [], "velocity": []},
        tol=1e-2,
        k_p=1.7,
        k_d=0.02,
        max_vel=0.8,
        max_stuck_steps: int = 20,
        stuck_tol: float = 1e-3,
        abort_when_collision: bool = True,
        steps_to_abortion: int = 300,
    ):
        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        action = self._build_action(arm_action, self.gripper_state, body_action)

        last_errors = deque(maxlen=max_stuck_steps)
        base_direction = self.base_x_direction()

        dst_pos = self.base_x_pos() + delta * base_direction

        if self.grasp_pose_visual is not None:
            tcp_pose = self.base_env.agent.tcp.pose.sp
            tcp_pos_dst = tcp_pose.p + delta * base_direction
            self.grasp_pose_visual.set_pose(
                sapien.Pose(p=tcp_pos_dst, q=tcp_pose.q)
            )

        last_error = 0
        dt = 0.01

        self.render_wait()

        n_steps = 0
        while True:
            cur_pos = self.base_x_pos()
            current_error = np.dot(base_direction, dst_pos - cur_pos)

            last_errors.append(current_error)
            error_diff = (current_error - last_error) / dt
            last_error = current_error

            if (
                len(last_errors) >= max_stuck_steps
                and np.std(last_errors) < stuck_tol
            ):
                print("Robot is stuck")
                self.planner.update_from_simulation()
                return self.idle_steps(t=1), arm_actions

            if np.abs(current_error) < tol:
                self.planner.update_from_simulation()
                return self.idle_steps(t=1), arm_actions

            control_vel = k_p * current_error + k_d * error_diff
            control_vel = np.clip(control_vel, -max_vel, max_vel)

            if len(arm_actions["position"]) > 0:
                arm_action = arm_actions["position"][0]
                arm_actions["position"] = np.delete(
                    arm_actions["position"], 0, axis=0
                )
                arm_actions["velocity"] = np.delete(
                    arm_actions["velocity"], 0, axis=0
                )
            action[7:13] = arm_action
            action[0] = control_vel

            if self.verbose:
                print(
                    f"n_steps: {n_steps} base Action:",
                    np.round(action[:3], 4),
                )
                print(
                    "Full: ",
                    np.round(self.robot.get_qpos().cpu().numpy()[0], 4),
                )
            obs, reward, terminated, truncated, info = self.env.step(action)
            n_steps += 1
            self.update_base_pose()

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

            if abort_when_collision:
                if len(
                    collisions := self.planner.planning_world.check_collision()
                ) > 0:
                    print("Collision detected while moving base")
                    for collision in collisions:
                        print(
                            f"Collision between {collision.link_name1} of entity "
                            f"{collision.object_name1} with {collision.link_name2} "
                            f"of entity {collision.object_name2}"
                        )
                    return -1, None
            if n_steps > steps_to_abortion:
                print("Reached max steps. Something went wrong.")
                return -1, None

        return obs, reward, terminated, truncated, info

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        body_action = self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                raise NotImplementedError
            else:
                torso_qpos = qpos[:4]
                arm_qpos = qpos[4:]
                body_action[:4] = torso_qpos
                action = self._build_action(arm_qpos, self.gripper_state, body_action)
                if self.verbose:
                    print("arm action:", np.round(qpos, 4))
                    print(
                        "Full: ",
                        np.round(self.robot.get_qpos().cpu().numpy()[0], 4),
                    )
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        self.planner.update_from_simulation()
        return obs, reward, terminated, truncated, info

    def static_manipulation(
        self, target_tcp_pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        res = self.move_to_pose_with_screw(
            target_tcp_pose, dry_run=dry_run, refine_steps=refine_steps
        )
        if res == -1:
            res = self.move_to_pose_with_RRTConnect(
                target_tcp_pose, dry_run=dry_run, refine_steps=refine_steps
            )
            if res == -1:
                return res
        self.planner.update_from_simulation()
        return res

    def check_IK(self, pose: sapien.Pose):
        self.update_base_pose()
        pose = to_sapien_pose(pose)
        pose = self._transform_pose_for_planning(pose)
        current_qpos = self.robot.get_qpos().cpu().numpy()[0, 3:13]
        pose = mplib.Pose(p=pose.p, q=pose.q)
        current_qpos = np.clip(
            current_qpos,
            self.planner.joint_limits[:, 0],
            self.planner.joint_limits[:, 1],
        )
        current_qpos = self.planner.pad_move_group_qpos(current_qpos)
        pose = self.planner._transform_goal_to_wrt_base(pose)
        ik_status, goal_qpos = self.planner.IK(pose, current_qpos, [])
        return ik_status == "Success"

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        self.update_base_pose()
        pose = to_sapien_pose(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        current_qpos = self.robot.get_qpos().cpu().numpy()[0, 3:13]
        result = self.planner.plan_pose(
            mplib.Pose(p=pose.p, q=pose.q),
            current_qpos,
            time_step=self.base_env.control_timestep,
            wrt_world=True,
            verbose=True,
            planning_time=2,
            rrt_range=0.1,
            simplify=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        self.update_base_pose()
        pose = to_sapien_pose(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        current_qpos = self.robot.get_qpos().cpu().numpy()[0, 3:13]
        result = self.planner.plan_screw(
            mplib.Pose(p=pose.p, q=pose.q),
            current_qpos,
            time_step=self.base_env.control_timestep,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                mplib.Pose(p=pose.p, q=pose.q),
                current_qpos,
                time_step=self.base_env.control_timestep,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def open_gripper(self, t=6, gripper_state=None):
        return self.change_gripper_state(t=t, gripper_state=self.OPEN)

    def close_gripper(self, t=6, gripper_state=None):
        return self.change_gripper_state(t=t, gripper_state=self.CLOSED)

    def change_gripper_state(self, t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.CLOSED
        self.gripper_state = gripper_state
        qpos = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = self._build_action(qpos, self.gripper_state, body_action)
            else:
                raise NotImplementedError
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def idle_steps(self, t=20):
        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        action = self._build_action(arm_action, self.gripper_state, body_action)
        for i in range(t):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info
