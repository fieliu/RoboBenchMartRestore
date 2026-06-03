import math
import sys
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms

from dsynth.navigation.iplanner_models.planner_net import PlannerNet
from dsynth.navigation.iplanner_models.traj_opt import TrajOpt


def _register_legacy_modules():
    models_dir = str(Path(__file__).parent / "iplanner_models")
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    if "percept_net" not in sys.modules:
        from dsynth.navigation.iplanner_models import percept_net as _pn
        sys.modules["percept_net"] = _pn
    if "planner_net" not in sys.modules:
        from dsynth.navigation.iplanner_models import planner_net as _pln
        sys.modules["planner_net"] = _pln


class IPlannerController:
    def __init__(
        self,
        model_path: str = None,
        config_path: str = None,
        device: str = "cpu",
    ):
        if model_path is None:
            model_path = str(Path(__file__).parent / "iplanner_models" / "plannernet.pt")
        if config_path is None:
            config_path = str(Path(__file__).parent / "iplanner_models" / "iplanner.yaml")

        self.device = device
        self.traj_generate = TrajOpt()
        self.load_model(model_path, config_path)

    def load_model(self, model_path: str, config_path: str):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.img_input_size = tuple(self.cfg["img_input_size"])
        self.sensor_offset_x = self.cfg["sensor_offset_x"]
        self.sensor_offset_y = self.cfg["sensor_offset_y"]
        self.max_depth = self.cfg["max_depth"]
        self.max_goal_distance = self.cfg["max_goal_distance"]
        self.is_traj_shift = False
        if math.hypot(self.sensor_offset_x, self.sensor_offset_y) > 1e-1:
            self.is_traj_shift = True

        self.net = PlannerNet(encoder_channel=16)
        _register_legacy_modules()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, tuple):
            model_obj = checkpoint[0]
            if hasattr(model_obj, 'state_dict'):
                model_state_dict = model_obj.state_dict()
            elif isinstance(model_obj, dict):
                model_state_dict = model_obj
            else:
                model_state_dict = model_obj
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model_state_dict = checkpoint["state_dict"]
        elif hasattr(checkpoint, 'state_dict'):
            model_state_dict = checkpoint.state_dict()
        else:
            model_state_dict = checkpoint
        self.net.load_state_dict(model_state_dict, strict=True)
        self.net.eval()
        if torch.cuda.is_available() and self.device != "cpu":
            self.net = self.net.cuda()

    def process_depth(self, depth: np.ndarray) -> torch.Tensor:
        if isinstance(depth, np.ndarray):
            depth = depth.copy()
            if np.issubdtype(depth.dtype, np.integer):
                depth = depth.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth)] = 0.0
            depth[depth > self.max_depth] = 0.0
            while depth.ndim > 2:
                if depth.shape[0] == 1:
                    depth = depth.squeeze(0)
                elif depth.shape[-1] == 1:
                    depth = depth.squeeze(-1)
                else:
                    depth = depth.squeeze(0)
            if depth.ndim == 2:
                depth = depth.astype(np.float32)

        pil_img = Image.fromarray(depth, mode='F')
        depth_transform = transforms.Compose([
            transforms.Resize(tuple(self.img_input_size)),
            transforms.ToTensor()])
        tensor = depth_transform(pil_img)
        tensor = tensor.expand(1, 3, -1, -1).clone()
        return tensor

    def _inflate_depth(self, depth: torch.Tensor, robot_radius: float = 0.5) -> torch.Tensor:
        """Downsample + iterative dilation for efficient obstacle inflation.

        Strategy: downsample 4x, run 8-connected iterative dilation on the
        small image, then upsample back.  On the full-resolution image each
        pass on the small image corresponds to ~4 px of inflation, so 15
        passes ≈ 60 px total.  At 2 m distance that is ~0.38 m per side,
        enough to block narrow gaps that the robot (width ~1 m) cannot fit
        through.
        """
        _, _, H, W = depth.shape
        obstacle_mask = (depth > 0) & (depth < self.max_depth)
        if not obstacle_mask.any():
            return depth

        scale = 4
        small_h, small_w = H // scale, W // scale

        small_mask = torch.nn.functional.max_pool2d(
            obstacle_mask.float(), scale, stride=scale) > 0.5

        neg_depth = depth.clone()
        neg_depth[~obstacle_mask] = 0.0
        small_neg = torch.nn.functional.max_pool2d(
            -neg_depth, scale, stride=scale)
        small_depth = -small_neg
        small_depth[~small_mask] = 0.0

        inflated_small = small_depth.clone()
        obs_depth = small_depth.clone()
        obs_depth[~small_mask] = float('inf')
        cur_mask = small_mask.clone()

        n_passes = 15
        for _ in range(n_passes):
            padded_obs = torch.nn.functional.pad(
                obs_depth, [1, 1, 1, 1], mode='constant', value=float('inf'))
            padded_mask = torch.nn.functional.pad(
                cur_mask.float(), [1, 1, 1, 1], mode='constant', value=0.0)

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    n_mask = padded_mask[:, :,
                                        1 + dy:small_h + 1 + dy,
                                        1 + dx:small_w + 1 + dx]
                    n_obs = padded_obs[:, :,
                                      1 + dy:small_h + 1 + dy,
                                      1 + dx:small_w + 1 + dx]

                    spread = (~cur_mask) & (n_mask > 0.5)
                    inflated_small = torch.where(
                        spread, torch.min(inflated_small, n_obs), inflated_small)
                    cur_mask = cur_mask | spread
                    obs_depth = torch.where(
                        spread, torch.min(obs_depth, n_obs), obs_depth)

        inflated = torch.nn.functional.interpolate(
            inflated_small, size=(H, W), mode='nearest')

        original_valid = (depth > 0) & (depth < self.max_depth)
        inflated = torch.where(original_valid, depth, inflated)

        inflated[inflated > self.max_depth] = 0.0
        inflated[inflated < 0] = 0.0
        return inflated

    def check_forward_clearance(self, depth_raw: np.ndarray,
                                safe_dist: float = 0.8,
                                lateral_margin: float = 0.4,
                                forward_range: float = 1.5) -> dict:
        """Check if the path ahead is clear using the raw (un-inflated) depth image.

        Scans a trapezoidal region in front of the robot:
          - forward: 0.3 m to forward_range m
          - lateral: ±lateral_margin m at each depth

        Returns dict with:
          'clear'      : bool  – True if no obstacle within safe_dist
          'min_dist'   : float – closest obstacle distance in the region
          'obstacle_lateral': float – lateral offset of closest obstacle
                                      (negative=left, positive=right)
          'left_clear' : bool  – left side has more room
          'right_clear': bool  – right side has more room
        """
        if isinstance(depth_raw, np.ndarray):
            d = depth_raw.copy()
            if np.issubdtype(d.dtype, np.integer):
                d = d.astype(np.float32) / 1000.0
        else:
            d = depth_raw.cpu().numpy().squeeze()

        if d.ndim == 3 and d.shape[-1] == 1:
            d = d.squeeze(-1)
        H, W = d.shape[:2]

        fx = 205.47

        step = 4
        min_dist = float('inf')
        obs_lateral = 0.0
        left_min = float('inf')
        right_min = float('inf')

        for v in range(0, H, step):
            for u in range(0, W, step):
                z = d[v, u]
                if z <= 0.3 or z > forward_range:
                    continue
                x_3d = (u - W / 2) * z / fx
                if abs(x_3d) > lateral_margin:
                    continue
                if z < min_dist:
                    min_dist = z
                    obs_lateral = x_3d
                if x_3d < 0:
                    left_min = min(left_min, z)
                else:
                    right_min = min(right_min, z)

        clear = min_dist > safe_dist
        return {
            'clear': clear,
            'min_dist': min_dist,
            'obstacle_lateral': obs_lateral,
            'left_clear': left_min > safe_dist,
            'right_clear': right_min > safe_dist,
            'left_min': left_min,
            'right_min': right_min,
        }

    def plan(self, depth_image: np.ndarray, goal_robot_frame: np.ndarray):
        goal_camera = goal_robot_frame.copy()
        if self.is_traj_shift:
            goal_camera[0] -= self.sensor_offset_x
            goal_camera[1] -= self.sensor_offset_y

        tensor_depth = self.process_depth(depth_image)
        tensor_goal = torch.as_tensor(goal_camera[:3], device=self.device, dtype=torch.float32).unsqueeze(0)

        if torch.cuda.is_available() and self.device != "cpu":
            tensor_depth = tensor_depth.cuda()
            tensor_goal = tensor_goal.cuda()

        with torch.no_grad():
            keypoints, fear = self.net(tensor_depth, tensor_goal)

        if self.is_traj_shift:
            batch_size, _, dims = keypoints.shape
            keypoints = torch.cat(
                (torch.zeros(batch_size, 1, dims, device=keypoints.device, requires_grad=False), keypoints), dim=1
            )
            keypoints[..., 0] += self.sensor_offset_x
            keypoints[..., 1] += self.sensor_offset_y

        keypoints_np = keypoints.squeeze(0).cpu().numpy()

        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)
        traj_np = traj.squeeze(0).cpu().numpy()
        fear_val = fear.squeeze().cpu().item()

        return keypoints_np, traj_np, fear_val

    def compute_base_velocity(self, traj: np.ndarray, current_base_pose_matrix: np.ndarray,
                              look_ahead_dist: float = 0.5, conv_dist: float = 0.25,
                              max_linear_vel: float = 0.3,
                              max_angular_vel: float = 0.3):
        """Pure-pursuit controller with adaptive look-ahead.

        Uses the first waypoint beyond conv_dist as the tracking target.
        This ensures the robot follows the planned path closely, especially
        during obstacle avoidance where distant waypoints may already be
        past the obstacle and point straight ahead.
        """
        target_point = None
        for wp in traj:
            dist = np.linalg.norm(wp[:2])
            if dist < conv_dist:
                continue
            target_point = wp
            break
        if target_point is None:
            if len(traj) > 0:
                target_point = traj[-1]
            else:
                return 0.0, 0.0

        K_A = 1.0

        bearing = np.arctan2(target_point[1], target_point[0])
        angular_vel = K_A * bearing
        angular_vel = np.clip(angular_vel, -max_angular_vel, max_angular_vel)

        turn_fraction = abs(angular_vel) / max_angular_vel
        linear_vel = max_linear_vel * (1.0 - 0.8 * turn_fraction)
        linear_vel = max(0.05, linear_vel)

        return linear_vel, angular_vel

    def compute_goal_in_robot_frame(self, goal_world: np.ndarray, robot_pose_matrix: np.ndarray) -> np.ndarray:
        robot_pos = robot_pose_matrix[:3, 3]
        robot_rot = robot_pose_matrix[:3, :3]
        goal_relative_world = goal_world - robot_pos
        goal_robot = robot_rot.T @ goal_relative_world
        return goal_robot
