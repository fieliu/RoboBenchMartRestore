import torch
from copy import deepcopy
from transforms3d import euler
import numpy as np
import sapien
from mani_skill.utils import sapien_utils
from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Pose
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.utils.structs.link import Link
from dsynth import PACKAGE_DIR


@register_agent()
class DSR1(BaseAgent):
    uid = "ds_r1"
    urdf_path = f"{PACKAGE_DIR}/assets/urdf/r1/r1.urdf"
    urdf_arm_ik_path = f"{PACKAGE_DIR}/assets/urdf/r1/r1_arm_ik.urdf"

    arm_joint_names = [
        "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
        "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
    ]
    gripper_joint_names = [
        "left_gripper_finger_joint1", "left_gripper_finger_joint2",
    ]
    body_joint_names = [
        "root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint",
        "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
    ]
    torso_joint_names = [
        "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
    ]

    arm_stiffness = [1e3, 1e3, 1e3, 5e2, 5e2, 5e2]
    arm_damping = [1e2, 1e2, 1e2, 1e2, 1e2, 1e2]
    arm_force_limit = [40, 40, 27, 7, 7, 7]

    body_stiffness = [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]
    body_damping = [1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2]
    body_force_limit = [100, 100, 100, 100, 100, 100, 100]

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            qpos=np.array([
                0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0.04, 0.04,
            ]),
        )
    )

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="r1_hand",
                pose=Pose.create_from_pq([0.1, 0, 0], euler.euler2quat(np.pi, -np.pi / 2, 0)),
                width=128,
                height=128,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="left_gripper_link",
            ),
            CameraConfig(
                uid="left_base_camera_link",
                pose=Pose.create_from_pq([-0.3, 0.3, 0], euler.euler2quat(0, 0.3, -0.2)),
                width=256,
                height=256,
                fov=1.5,
                near=0.01,
                far=100,
                entity_uid="zed_link",
            ),
            CameraConfig(
                uid="right_base_camera_link",
                pose=Pose.create_from_pq([-0.3, -0.3, 0], euler.euler2quat(0, 0.3, 0.2)),
                width=256,
                height=256,
                fov=1.5,
                near=0.01,
                far=100,
                entity_uid="zed_link",
            ),
        ]

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=False,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            -0.04,
            0.04,
            [1e2] * 2,
            [1e1] * 2,
            [100] * 2,
            use_delta=False,
            normalize_action=True,
        )

        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            None,
            None,
            self.body_stiffness,
            self.body_damping,
            self.body_force_limit,
            use_delta=False,
            normalize_action=False,
        )
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            -0.1,
            0.1,
            self.body_stiffness,
            self.body_damping,
            self.body_force_limit,
            use_delta=True,
        )
        stiff_body_pd_joint_target_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            None,
            None,
            1e5,
            1e5,
            1e5,
            normalize_action=False,
            use_target=True,
        )

        controller_configs = deepcopy_dict(
            dict(
                pd_joint_pos=dict(
                    arm=arm_pd_joint_pos,
                    gripper=gripper_pd_joint_pos,
                    body=body_pd_joint_pos,
                ),
                pd_joint_delta_pos=dict(
                    arm=arm_pd_joint_delta_pos,
                    gripper=gripper_pd_joint_pos,
                    body=body_pd_joint_delta_pos,
                ),
                pd_ee_delta_pos=dict(
                    arm=PDJointPosControllerConfig(
                        self.arm_joint_names,
                        -0.1,
                        0.1,
                        self.arm_stiffness,
                        self.arm_damping,
                        self.arm_force_limit,
                        use_delta=True,
                    ),
                    gripper=gripper_pd_joint_pos,
                    body=stiff_body_pd_joint_target_pos,
                    ee_delta=PDEEPoseControllerConfig(
                        self.arm_joint_names,
                        -0.1, 0.1,  # pos_lower, pos_upper (delta)
                        stiffness=1e2,
                        damping=1e2,
                        force_limit=100,
                    ),
                ),
            )
        )
        return controller_configs

    def _after_init(self):
        super()._after_init()
        self.torso_link1: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "torso_link1"
        )
        self.torso_link4: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "torso_link4"
        )
        self.left_gripper_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_gripper_link"
        )

    @property
    def tcp(self):
        return self.left_gripper_link

    @property
    def base_link(self):
        return self.torso_link1

    @property
    def tcp_pose(self) -> Pose:
        return self.left_gripper_link.pose

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    def is_static(self, threshold: float = 0.2, base_threshold: float = 0.05):
        qvel = self.robot.get_qvel()
        body_qvel = qvel[..., 3:13]
        base_qvel = qvel[..., :3]
        return torch.all(torch.abs(body_qvel) <= threshold, dim=1) & torch.all(
            torch.abs(base_qvel) <= base_threshold, dim=1
        )
