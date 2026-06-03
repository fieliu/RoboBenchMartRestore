"""NavDP point-goal navigation controller.

Wraps InternRobotics' NavDP diffusion policy behind an interface that mirrors
IPlannerController, so it drops into the same navigation loop.

Key differences vs iPlanner:
  - NavDP needs RGB *and* depth (both -> DepthAnythingV2 feature encoders).
  - NavDP is stateful: it keeps an 8-frame RGB memory queue internally, so you
    must call reset() per episode and step it frame-by-frame in order.
  - NavDP outputs 24 absolute waypoints (x, y, yaw) in the *robot* frame.

This controller caches the planned trajectory in the *world* frame so it can be
re-projected into the current robot frame every control step. That lets the
re-plan interval N be > 1 (the robot moves between plans, world-frame anchoring
keeps tracking correct).
"""
import sys
import math
import numpy as np
from pathlib import Path

_MODELS_DIR = str(Path(__file__).parent / "navdp_models")


def _register_navdp_modules():
    """Inject navdp_models/ into sys.path so NavDP's bare imports resolve.

    Mirrors IPlannerController._register_legacy_modules(): NavDP source uses
    `from policy_network import ...` / `from depth_anything... import ...`,
    which assume the package dir is on sys.path.
    """
    if _MODELS_DIR not in sys.path:
        sys.path.insert(0, _MODELS_DIR)


class NavDPController:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        image_intrinsic: np.ndarray = None,
        stop_threshold: float = -3.0,
        sample_num: int = 16,
    ):
        _register_navdp_modules()
        from policy_agent import NavDP_Agent

        if image_intrinsic is None:
            # only used by NavDP's internal trajectory visualisation, not inference
            image_intrinsic = np.array([[200.0, 0, 112.0],
                                        [0, 200.0, 112.0],
                                        [0, 0, 1.0]], dtype=np.float32)

        self.device = device
        self.stop_threshold = stop_threshold
        self.sample_num = sample_num
        self.agent = NavDP_Agent(
            image_intrinsic=image_intrinsic,
            navi_model=model_path,
            device=device,
        )
        self.traj_world = None     # cached (24, 2) world-frame XY waypoints
        self.last_values = None    # critic values from last plan
        self.last_traj_robot = None
        self.last_target_robot = None   # selected look-ahead wp (robot frame)
        self.last_target_world = None   # selected look-ahead wp (world frame)
        self.last_bearing = 0.0

    def reset(self, threshold: float = None):
        """Clear NavDP's 8-frame RGB memory. Call once per episode before the loop."""
        if threshold is not None:
            self.stop_threshold = threshold
        self.agent.reset(batch_size=1, threshold=self.stop_threshold)
        self.traj_world = None
        self.last_values = None
        self.last_traj_robot = None
        self.last_target_robot = None
        self.last_target_world = None
        self.last_bearing = 0.0

    # ── SE(2) frame transforms (planar, yaw only) ──────────────────
    @staticmethod
    def _yaw_t(pose: np.ndarray):
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        return yaw, pose[:2, 3].astype(np.float64)

    def _robot_to_world(self, xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """(N,2) robot-frame XY -> world-frame XY using pose at plan time."""
        yaw, t = self._yaw_t(pose)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        return (R @ xy.T).T + t

    def _world_to_robot(self, xy_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """(N,2) world-frame XY -> current robot-frame XY using current pose."""
        yaw, t = self._yaw_t(pose)
        c, s = math.cos(yaw), math.sin(yaw)
        Rt = np.array([[c, s], [-s, c]])   # R^T
        return (Rt @ (xy_world - t).T).T

    def compute_goal_in_robot_frame(self, goal_world: np.ndarray,
                                    robot_pose_matrix: np.ndarray) -> np.ndarray:
        """Same as IPlannerController: world goal -> robot frame (x forward)."""
        robot_pos = robot_pose_matrix[:3, 3]
        robot_rot = robot_pose_matrix[:3, :3]
        return robot_rot.T @ (goal_world - robot_pos)

    def plan(self, rgb: np.ndarray, depth: np.ndarray,
             goal_robot: np.ndarray, robot_pose: np.ndarray):
        """Run one NavDP inference. Call only on re-plan steps.

        Args:
            rgb:   (H, W, 3) uint8/float, 0-255   (single current frame)
            depth: (H, W, 1) or (H, W) depth. mm (int) or m (float) auto-handled.
            goal_robot: (3,) point goal in robot frame, x forward, metres.
            robot_pose: (4, 4) base_link pose in world (at this instant).

        Returns:
            traj_robot: (24, 3) waypoints (x, y, yaw) in robot frame, metres.
            values:     (sample_num,) critic scores for the sampled trajectories.
        """
        # mm -> m if integer depth (ManiSkill returns uint16 mm); NavDP wants metres
        d = depth
        if np.issubdtype(d.dtype, np.integer):
            d = d.astype(np.float32) / 1000.0
        else:
            d = d.astype(np.float32)
        if d.ndim == 2:
            d = d[:, :, None]                      # -> (H, W, 1)

        rgb_in = rgb[None].astype(np.float32)      # (1, H, W, 3), 0-255
        depth_in = d[None]                         # (1, H, W, 1), metres
        goal_in = np.asarray(goal_robot, dtype=np.float32)[None]   # (1, 3)

        traj, _all_traj, values, _vis = self.agent.step_pointgoal(
            goal_in, rgb_in, depth_in)
        traj_robot = traj[0]                       # (24, 3)

        self.last_traj_robot = traj_robot
        self.last_values = values[0] if values.ndim > 1 else values
        # anchor XY to world frame so it can be re-projected next control steps
        self.traj_world = self._robot_to_world(traj_robot[:, :2], robot_pose)
        return traj_robot, self.last_values

    def compute_base_velocity(self, robot_pose: np.ndarray,
                              look_ahead_dist: float = 0.7,
                              conv_dist: float = 0.25,
                              max_linear_vel: float = 0.3,
                              max_angular_vel: float = 0.4):
        """Pure-pursuit over the cached world-frame trajectory.

        Re-projects the world-frame waypoints into the *current* robot frame
        each call, so tracking stays correct between re-plans (N > 1).
        Mirrors IPlannerController.compute_base_velocity.
        """
        if self.traj_world is None or len(self.traj_world) == 0:
            return 0.0, 0.0

        traj_robot = self._world_to_robot(self.traj_world, robot_pose)   # (24, 2)

        # first waypoint beyond conv_dist; fall back to look_ahead, then last
        target = None
        for wp in traj_robot:
            if np.linalg.norm(wp[:2]) >= conv_dist:
                target = wp
                break
        if target is None:
            target = traj_robot[-1]

        K_A = 1.0
        bearing = math.atan2(float(target[1]), float(target[0]))
        angular_vel = float(np.clip(K_A * bearing, -max_angular_vel, max_angular_vel))

        # cache selected look-ahead target (robot + world frame) for logging
        self.last_target_robot = np.asarray(target[:2], dtype=np.float64)
        self.last_target_world = self._robot_to_world(
            self.last_target_robot[None, :], robot_pose)[0]
        self.last_bearing = float(bearing)

        turn_fraction = abs(angular_vel) / max_angular_vel
        linear_vel = max_linear_vel * (1.0 - 0.8 * turn_fraction)
        linear_vel = max(0.05, linear_vel)
        return linear_vel, angular_vel

    @property
    def is_stopped(self) -> bool:
        """True when NavDP's critic found no safe trajectory (stop signal)."""
        if self.last_values is None:
            return False
        return float(np.max(self.last_values)) < self.stop_threshold
