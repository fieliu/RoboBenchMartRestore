"""
Store Map Utilities for VLM-based Navigation Planning
=====================================================
从 RoboBenchMart 仿真环境自动提取地图信息，供 VLM 规划器使用。

提供三类信息:
  1. 标注了货架名称的俯视图 (numpy RGB image)
  2. 货架物品清单 (text)
  3. 坐标查找表 (text, 含路点+货架接近点+走廊)

初始化时获取一次静态地图，运行时只更新机器人位置。

Usage:
    from dsynth.navigation.map_utils import StoreMapProvider

    provider = StoreMapProvider(env)
    provider.initialize()          # 获取静态地图（只需一次）

    # 每次规划前调用，获取最新地图+机器人位置
    map_info = provider.get_map_info()
    # map_info.topdown_image   -> np.ndarray (H, W, 3) 带标注的俯视图
    # map_info.inventory_text  -> str 物品清单
    # map_info.coordinates_text-> str 坐标表
    # map_info.robot_state     -> dict 机器人位姿
    # map_info.full_prompt_block-> str  拼好的文本块，可直接喂给VLM
"""

import math
import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("StoreMapProvider")


# ============================================================
# Data classes
# ============================================================
@dataclass
class ShelfInfo:
    """单个货架的完整信息"""
    name: str              # 语义名, 如 "zone_0_shelf_0"
    actor_name: str        # 系统名, 如 "[ENV#0]_SHELF_0_0.0"
    zone_id: str
    shelf_id: str
    position: np.ndarray   # (3,) 世界坐标
    facing: np.ndarray     # (3,) 朝向向量
    approach_pos: np.ndarray  # (3,) 机器人接近位置
    approach_yaw_deg: float   # 接近时机器人朝向(度)
    products: List[Dict] = field(default_factory=list)  # 该货架上的物品


@dataclass
class RobotState:
    """机器人当前状态"""
    position: np.ndarray   # (3,) 世界坐标
    yaw: float             # 弧度
    yaw_deg: float         # 度
    pose_matrix: np.ndarray  # (4, 4) 齐次变换矩阵


@dataclass
class MapInfo:
    """VLM规划器需要的全部地图信息"""
    topdown_image: np.ndarray      # (H, W, 3) 带标注的俯视图
    inventory_text: str            # 物品清单文本
    coordinates_text: str          # 坐标表文本
    robot_state: RobotState        # 机器人当前状态
    shelves: Dict[str, ShelfInfo]  # 货架信息字典
    waypoints: Dict[str, List[float]]  # 走廊路点
    scene_size: Tuple[float, float]    # (x_size, y_size)

    @property
    def full_prompt_block(self) -> str:
        """拼好的文本块，可直接喂给VLM"""
        lines = [
            "=== STORE MAP (top-down image with labeled shelves) ===",
            f"Scene size: {self.scene_size[0]:.1f}m x {self.scene_size[1]:.1f}m",
            "",
            self.inventory_text,
            "",
            self.coordinates_text,
            "",
            f"=== ROBOT NOW: at ({self.robot_state.position[0]:.2f}, "
            f"{self.robot_state.position[1]:.2f}) "
            f"facing {self.robot_state.yaw_deg:.0f}deg ===",
        ]
        return "\n".join(lines)


# ============================================================
# StoreMapProvider
# ============================================================
class StoreMapProvider:
    """从仿真环境提取地图信息，供VLM规划器使用。

    初始化时获取静态地图（俯视图+货架+物品），运行时只更新机器人位置。
    """

    def __init__(self, env, camera_height: float = 8.0,
                 image_size: int = 1024):
        """
        Args:
            env: RoboBenchMart 的 gym 环境 (ManiSkill)
            camera_height: 俯视相机高度 (米)
            image_size: 俯视图分辨率
        """
        self.env = env.unwrapped if hasattr(env, 'unwrapped') else env
        self.camera_height = camera_height
        self.image_size = image_size

        # 静态数据（initialize后填充）
        self._shelves: Dict[str, ShelfInfo] = {}
        self._waypoints: Dict[str, List[float]] = {}
        self._scene_size: Tuple[float, float] = (0.0, 0.0)
        self._base_topdown: Optional[np.ndarray] = None  # 不含机器人标注的底图
        self._inventory_text: str = ""
        self._coordinates_text: str = ""
        self._initialized = False

    # ----------------------------------------------------------
    # 初始化：获取静态地图（只需调用一次）
    # ----------------------------------------------------------
    def initialize(self):
        """获取静态地图信息。在 env.reset() 之后调用一次。"""
        self._extract_scene_size()
        self._extract_shelves()
        self._extract_waypoints()
        self._render_base_topdown()
        self._build_inventory_text()
        self._build_coordinates_text()
        self._initialized = True
        logger.info(
            f"StoreMapProvider initialized: "
            f"{len(self._shelves)} shelves, "
            f"{len(self._waypoints)} waypoints, "
            f"scene {self._scene_size[0]:.1f}x{self._scene_size[1]:.1f}m"
        )

    # ----------------------------------------------------------
    # 实时获取：更新机器人位置，返回完整MapInfo
    # ----------------------------------------------------------
    def get_map_info(self) -> MapInfo:
        """获取最新地图信息（含机器人位置）。每次规划前调用。"""
        if not self._initialized:
            raise RuntimeError("Call initialize() first after env.reset()")

        robot = self._get_robot_state()

        # 在底图上标注机器人位置（不修改底图）
        topdown = self._annotate_robot(self._base_topdown.copy(), robot)

        return MapInfo(
            topdown_image=topdown,
            inventory_text=self._inventory_text,
            coordinates_text=self._coordinates_text,
            robot_state=robot,
            shelves=self._shelves,
            waypoints=self._waypoints,
            scene_size=self._scene_size,
        )

    # ----------------------------------------------------------
    # 便捷方法：直接获取VLM prompt
    # ----------------------------------------------------------
    def get_vlm_prompt_block(self) -> str:
        """获取拼好的文本块，可直接喂给VLM"""
        return self.get_map_info().full_prompt_block

    def get_vlm_image(self) -> np.ndarray:
        """获取带机器人标注的俯视图，可直接编码给VLM"""
        return self.get_map_info().topdown_image

    # ----------------------------------------------------------
    # 根据物品名定位货架
    # ----------------------------------------------------------
    def find_shelf_by_product(self, product_name: str) -> Optional[ShelfInfo]:
        """根据物品名找到对应货架。

        Args:
            product_name: 物品名（支持模糊匹配），如 "Fanta", "Nivea"

        Returns:
            ShelfInfo 或 None（未找到）
        """
        product_name_lower = product_name.lower()
        for shelf_name, shelf_info in self._shelves.items():
            for p in shelf_info.products:
                if product_name_lower in p['name'].lower():
                    return shelf_info
        return None

    def find_approach_by_product(self, product_name: str) -> Optional[Dict]:
        """根据物品名找到对应货架的接近位置。

        Returns:
            {"shelf_name": str, "approach_pos": [x, y], "approach_yaw_deg": float}
            或 None
        """
        shelf = self.find_shelf_by_product(product_name)
        if shelf is None:
            return None
        return {
            "shelf_name": shelf.name,
            "approach_pos": [float(shelf.approach_pos[0]), float(shelf.approach_pos[1])],
            "approach_yaw_deg": shelf.approach_yaw_deg,
        }

    def find_product_layer(self, product_name: str) -> Optional[str]:
        """根据物品名找到所在货架的层号。

        Returns:
            层号字符串如 "0", "1", "2" 或 None
        """
        product_name_lower = product_name.lower()
        for shelf_name, shelf_info in self._shelves.items():
            for p in shelf_info.products:
                if product_name_lower in p['name'].lower():
                    return str(p.get('board', '?'))
        return None

    # ===========================================================
    # 内部方法
    # ===========================================================

    def _extract_scene_size(self):
        """获取场景尺寸"""
        sb = self.env.scene_builder
        self._scene_size = (float(sb.x_size[0]), float(sb.y_size[0]))

    def _extract_shelves(self):
        """从环境actor提取所有货架信息"""
        self._shelves = {}
        shelves_actors = self.env.actors.get('fixtures', {}).get('shelves', {})

        for actor_name, actor in shelves_actors.items():
            # actor_name: "[ENV#0]_SHELF_0_0.0" -> zone_id=0, shelf_id=0
            try:
                shelf_suffix = re.sub(r"\[ENV#\d+\]_SHELF_\d+_", "", actor_name)
                zone_id, shelf_id = shelf_suffix.split('.')
            except ValueError:
                logger.warning(f"Cannot parse shelf name: {actor_name}, skipping")
                continue

            semantic_name = f"zone_{zone_id}_shelf_{shelf_id}"

            # 货架位姿
            mat = actor.pose.sp.to_transformation_matrix()
            pos = actor.pose.sp.p.cpu().numpy()
            facing = mat[:3, 1]  # y轴方向为货架朝向

            # 机器人接近位置：货架前方1.4m
            approach_pos = pos - 1.4 * facing
            approach_yaw = math.atan2(facing[1], facing[0])

            self._shelves[semantic_name] = ShelfInfo(
                name=semantic_name,
                actor_name=actor_name,
                zone_id=zone_id,
                shelf_id=shelf_id,
                position=pos,
                facing=facing,
                approach_pos=approach_pos,
                approach_yaw_deg=math.degrees(approach_yaw),
            )

        # 填充物品信息
        self._fill_shelf_products()

    def _fill_shelf_products(self):
        """将物品信息关联到对应货架"""
        products_df = getattr(self.env, 'products_df', None)
        if products_df is None:
            logger.warning("No products_df found, inventory will be empty")
            return

        for shelf_name, shelf_info in self._shelves.items():
            # 匹配: actor_name 包含该货架的标识
            pattern = shelf_info.actor_name.split(']_')[1] if ']_' in shelf_info.actor_name else ''
            if not pattern:
                continue
            mask = products_df['actor_name'].str.contains(re.escape(pattern), regex=True)
            shelf_products = products_df[mask]

            for _, row in shelf_products.iterrows():
                shelf_info.products.append({
                    'name': row.get('product_name', 'unknown'),
                    'asset_name': row.get('asset_name', ''),
                    'board': row.get('board_idxs', '?'),
                    'col': row.get('col_idxs', '?'),
                    'row': row.get('row_idxs', '?'),
                })

    def _extract_waypoints(self):
        """生成走廊路点。

        策略: 在货架之间自动生成走廊交叉点。
        每两个相邻货架的中间位置生成一个路点。
        """
        self._waypoints = {}
        x_size, y_size = self._scene_size

        if not self._shelves:
            return

        # 收集所有货架位置，按zone分组
        zone_positions: Dict[str, List[np.ndarray]] = {}
        for name, info in self._shelves.items():
            zid = info.zone_id
            zone_positions.setdefault(zid, []).append(info.position)

        # 在货架之间生成走廊路点
        wp_idx = 0
        sorted_zones = sorted(zone_positions.keys())

        for i in range(len(sorted_zones)):
            positions = zone_positions[sorted_zones[i]]
            # 每个zone的中间位置作为路点
            mean_pos = np.mean(positions, axis=0)
            self._waypoints[f"wp_{wp_idx}"] = [
                round(float(mean_pos[0]), 2),
                round(float(mean_pos[1]), 2),
            ]
            wp_idx += 1

        # 在zone之间生成连接路点（走廊中间点）
        for i in range(len(sorted_zones) - 1):
            pos_a = np.mean(zone_positions[sorted_zones[i]], axis=0)
            pos_b = np.mean(zone_positions[sorted_zones[i + 1]], axis=0)
            mid = (pos_a + pos_b) / 2
            self._waypoints[f"wp_{wp_idx}"] = [
                round(float(mid[0]), 2),
                round(float(mid[1]), 2),
            ]
            wp_idx += 1

        # 场景入口路点
        self._waypoints["entrance"] = [round(x_size * 0.1, 2), round(y_size / 2, 2)]

    def _render_base_topdown(self):
        """渲染俯视图底图（不含机器人标注）"""
        try:
            import cv2
            from mani_skill.utils import sapien_utils
            from mani_skill.sensors.camera import CameraConfig

            x_size, y_size = self._scene_size
            cx, cy = x_size / 2, y_size / 2
            h = self.camera_height

            # 设置俯视相机
            cam_pose = sapien_utils.look_at(
                [cx, cy, h], [cx, cy, 0.0]
            )
            self.env._custom_human_render_camera_configs["render_camera"] = {
                "uid": "render_camera",
                "pose": list(cam_pose.raw_pose[0].cpu().numpy()),
                "width": self.image_size,
                "height": self.image_size,
                "fov": 1.0,
                "near": 0.01,
                "far": 100,
            }

            # 渲染
            images = self.env.render()
            if images is not None and len(images) > 0:
                base_img = images[0].copy()
                # 确保是uint8
                if base_img.dtype != np.uint8:
                    base_img = np.clip(base_img, 0, 255).astype(np.uint8)
            else:
                # 渲染失败，生成空白图
                base_img = np.ones((self.image_size, self.image_size, 3),
                                   dtype=np.uint8) * 240

            # 在底图上标注货架名称
            self._base_topdown = self._annotate_shelves(base_img)

        except Exception as e:
            logger.warning(f"Failed to render topdown: {e}, using blank image")
            self._base_topdown = np.ones((self.image_size, self.image_size, 3),
                                          dtype=np.uint8) * 240
            self._annotate_shelves_fallback()

    def _world_to_pixel(self, world_xy: np.ndarray) -> Tuple[int, int]:
        """世界坐标 -> 图像像素坐标"""
        x_size, y_size = self._scene_size
        H, W = self._base_topdown.shape[:2]

        # 简单线性映射 (假设俯视图覆盖整个场景)
        px = int(world_xy[0] / x_size * W)
        py = int((y_size - world_xy[1]) / y_size * H)  # y轴翻转
        px = np.clip(px, 0, W - 1)
        py = np.clip(py, 0, H - 1)
        return px, py

    def _annotate_shelves(self, img: np.ndarray) -> np.ndarray:
        """在俯视图上标注货架名称"""
        try:
            import cv2
        except ImportError:
            return img

        for name, info in self._shelves.items():
            px, py = self._world_to_pixel(info.position[:2])

            # 画货架位置标记
            cv2.rectangle(img, (px - 15, py - 10), (px + 15, py + 10),
                          (0, 128, 255), 1)
            cv2.putText(img, name, (px - 20, py - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 255), 1)

            # 画接近位置标记
            apx, apy = self._world_to_pixel(info.approach_pos[:2])
            cv2.drawMarker(img, (apx, apy), (0, 255, 0),
                           cv2.MARKER_DIAMOND, 8, 1)

        # 画路点
        for wp_name, wp_xy in self._waypoints.items():
            wpx, wpy = self._world_to_pixel(np.array(wp_xy))
            cv2.drawMarker(img, (wpx, wpy), (255, 200, 0),
                           cv2.MARKER_SQUARE, 6, 1)
            cv2.putText(img, wp_name, (wpx + 5, wpy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 200, 0), 1)

        return img

    def _annotate_shelves_fallback(self):
        """渲染失败时在空白图上标注"""
        try:
            import cv2
            self._base_topdown = self._annotate_shelves(self._base_topdown)
        except ImportError:
            pass

    def _annotate_robot(self, img: np.ndarray, robot: RobotState) -> np.ndarray:
        """在图上标注机器人位置和朝向"""
        try:
            import cv2
        except ImportError:
            return img

        px, py = self._world_to_pixel(robot.position[:2])

        # 画机器人位置（红色圆点）
        cv2.circle(img, (px, py), 8, (0, 0, 255), -1)

        # 画朝向箭头
        arrow_len = 20
        dx = int(arrow_len * math.cos(robot.yaw))
        dy = int(-arrow_len * math.sin(robot.yaw))  # y轴翻转
        cv2.arrowedLine(img, (px, py), (px + dx, py + dy),
                        (0, 255, 0), 2, tipLength=0.3)

        # 标注文字
        cv2.putText(img, "ROBOT", (px - 20, py - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(img,
                    f"({robot.position[0]:.1f},{robot.position[1]:.1f}) {robot.yaw_deg:.0f}deg",
                    (px - 30, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        return img

    def _get_robot_state(self) -> RobotState:
        """获取机器人当前位姿"""
        mat = self.env.agent.base_link.pose.to_transformation_matrix()[0].cpu().numpy()
        pos = self.env.agent.base_link.pose.p[0].cpu().numpy()
        yaw = math.atan2(mat[1, 0], mat[0, 0])

        return RobotState(
            position=pos,
            yaw=yaw,
            yaw_deg=math.degrees(yaw),
            pose_matrix=mat,
        )

    def _build_inventory_text(self) -> str:
        """生成物品清单文本"""
        lines = ["=== SHELF INVENTORY ==="]

        for name, info in sorted(self._shelves.items()):
            lines.append(f"\n{name} (zone_{info.zone_id}, shelf_{info.shelf_id}):")
            if info.products:
                # 按层分组
                boards: Dict[str, List[str]] = {}
                for p in info.products:
                    b = str(p.get('board', '?'))
                    boards.setdefault(b, []).append(p['name'])

                for board_id in sorted(boards.keys()):
                    items = boards[board_id]
                    # 统计每种物品数量
                    from collections import Counter
                    counts = Counter(items)
                    items_str = ", ".join(
                        f"{n} x{c}" for n, c in counts.items()
                    )
                    lines.append(f"  layer_{board_id}: {items_str}")
            else:
                lines.append("  (empty)")

        self._inventory_text = "\n".join(lines)

    def _build_coordinates_text(self) -> str:
        """生成坐标表文本"""
        lines = ["=== SHELF APPROACH COORDINATES ==="]

        for name, info in sorted(self._shelves.items()):
            lines.append(
                f"  {name}_approach: "
                f"({info.approach_pos[0]:.2f}, {info.approach_pos[1]:.2f}), "
                f"face={info.approach_yaw_deg:.0f}deg"
            )

        lines.append("")
        lines.append("=== CORRIDOR WAYPOINTS ===")
        for wp_name, wp_xy in sorted(self._waypoints.items()):
            lines.append(f"  {wp_name}: ({wp_xy[0]:.2f}, {wp_xy[1]:.2f})")

        self._coordinates_text = "\n".join(lines)
