"""RestockFlow: 补货流程专用仿真环境(给 VLA 模型测试用, 不含求解器/运动规划)。

与 RestockBasketToShelfContEnv 的区别:
  - 激活两个货架: 仓库源货架(存放目标物品) + 商业目标货架(待补货上架);
  - 机器人初始放在场景角落的"休息区", 机械臂处于 rest 姿态;
  - 不依赖"机器人站在唯一货架正前方"那套初始位姿计算(故不触发单货架 assert)。

动作由外部 VLA 策略给出, 本环境只负责场景构建/reset/step/render。
"""
import torch
import numpy as np
import sapien
from transforms3d.euler import euler2quat
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from dsynth.envs.pick_to_basket import PickToBasketContEnv


@register_env('RestockFlowContEnv', max_episode_steps=200000)
class RestockFlowContEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Fanta Sabor Naranja 2L'
    ROBOT_INIT_POSE_RANDOM_ENABLED = False

    # 休息区: 场景右上角空地(无货架, 避免机器人生成在货架内导致穿模/碰撞失效)。
    # restock_scene 为 16x10, 货架占 x:0~15 y:0~6.4; 右上角 (x>14,y>8) 为空。
    REST_AREA_XY = [14.5, 8.5]
    REST_AREA_YAW_DEG = 180.0   # 面朝西(store 内部)
    # 休息区相对场景的内缩(米), 仅在 REST_AREA_XY 为 None 时用
    REST_AREA_MARGIN = 1.2

    # 只给这些货架上货并激活(大幅减少商品 actor 数 -> 提速)。匹配 fixture 名子串。
    # None = 全部激活(旧行为)。默认: Nivea 所在仓库 C 排某架 + 商业货架。
    STOCKED_SHELF_PATTERNS = ['commercial', 'row_C_daily_col0']

    def _rest_area_xy(self):
        if self.REST_AREA_XY is not None:
            return np.array(self.REST_AREA_XY, dtype=np.float32)
        sb = self.scene_builder
        m = self.REST_AREA_MARGIN
        return np.array([m, m], dtype=np.float32)

    def _compute_robot_init_pose(self, env_idx=None):
        """机器人固定在角落休息区, 朝向 REST_AREA_YAW_DEG。

        覆盖父类(它要求站在唯一目标货架正前方)。连续环境链按 3 元组消费:
        (origins, angles, directions) —— 见 darkstore_cont_base._initialize_episode。
        """
        n = len(env_idx)
        xy = self._rest_area_xy()
        origin = np.array([xy[0], xy[1], 0.0], dtype=np.float32)
        angle = float(np.radians(self.REST_AREA_YAW_DEG))
        direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)

        origins = np.tile(origin, (n, 1))
        angles = np.full((n,), angle, dtype=np.float32)
        directions = np.tile(direction, (n, 1))
        return origins, angles, directions

    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            name = self.target_product_names.get(scene_idx, self.TARGET_PRODUCT_NAME)
            self.language_instructions.append(
                f'restock {name}: take it from the warehouse shelf and '
                f'place it on the commercial shelf')


@register_env('RestockFlowContNiveaEnv', max_episode_steps=200000)
class RestockFlowContNiveaEnv(RestockFlowContEnv):
    TARGET_PRODUCT_NAME = 'Nivea Body Milk'


@register_env('RestockFlowContDuffEnv', max_episode_steps=200000)
class RestockFlowContDuffEnv(RestockFlowContEnv):
    TARGET_PRODUCT_NAME = 'Duff Beer Can'

