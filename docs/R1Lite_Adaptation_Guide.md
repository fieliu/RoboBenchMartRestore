# R1Lite 机器人适配说明

本文档描述为 RoboBenchMart 项目添加 R1Lite（`ds_r1`）机器人适配的所有新增和修改内容，包括文件清单、架构设计、使用方法和关键参数说明。

---

## 目录

1. [概述](#概述)
2. [新增文件](#新增文件)
3. [修改文件](#修改文件)
4. [R1 vs Fetch 对比](#r1-vs-fetch-对比)
5. [使用方法](#使用方法)
6. [场景配置参数](#场景配置参数)
7. [CPU 仿真性能优化](#cpu-仿真性能优化)
8. [架构设计](#架构设计)

---

## 概述

R1Lite 是一款单臂移动操作机器人，与原有的 Fetch 双臂机器人存在以下关键差异：

| 特性 | Fetch | R1Lite |
|------|-------|--------|
| 可动手臂 | 1（右臂） | 1（左臂） |
| 右臂状态 | 可动 | **固定（fixed joint）** |
| 底盘控制 | 速度型（vx, vy, ω） | **位置型（x, y, θ）** |
| 躯干关节 | 1个（torso_lift_joint） | **4个（torso_joint1-4）** |
| 动作空间维度 | 13维 | **15维** |
| 夹爪 | 1个（2指） | 1个（2指，左臂末端） |

由于这些差异，需要从机器人模型、运动规划器、技能函数、任务求解器到场景配置进行全链路适配。

---

## 新增文件

### 1. `dsynth/robots/ds_r1.py` — R1 机器人模型定义

注册为 ManiSkill Agent（uid=`ds_r1`），定义了：

- **URDF 路径**：`dsynth/assets/urdf/r1/r1.urdf`（完整模型）、`r1_arm_ik.urdf`（IK 求解用）
- **关节分组**：
  - `arm_joint_names`：左臂6关节（`left_arm_joint1-6`）
  - `gripper_joint_names`：左夹爪2关节
  - `body_joint_names`：底盘3关节（x, y, z旋转）+ 躯干4关节
- **控制器配置**：
  - `pd_joint_pos`：关节位置控制
  - `pd_joint_delta_pos`：关节增量位置控制
  - `pd_ee_delta_pos`：末端执行器位姿控制（使用 `PDEEPoseControllerConfig`，需提供 `urdf_path`）
- **传感器**：3个相机（手部、左基座、右基座）
- **关键链接**：`torso_link1`（规划基准）、`torso_link4`、`left_gripper_link`（TCP）

### 2. `dsynth/planning/r1_motionplanner.py` — R1 运动规划器

`R1MotionPlanningSapienSolver` 类，核心功能：

- **规划器初始化**：基于 mplib 的 `SapienPlanner`，规划组为 `left_gripper_link`
- **底盘驱动**：
  - `drive_base(target_pos, target_view_vec)`：驱动底盘到目标位置
  - `move_base_forward_delta(delta)`：沿底盘朝向前进/后退
  - `rotate_base_z(new_direction)`：旋转底盘朝向
  - 底盘控制采用 PID 控制器（位置型），区别于 Fetch 的速度型
- **手臂规划**：
  - `static_manipulation(target_tcp_pose)`：先尝试 Screw 规划，失败后回退到 RRTConnect
  - `move_to_pose_with_screw()` / `move_to_pose_with_RRTConnect()`
- **轨迹跟随**：`follow_path(result)` — 执行规划结果，将躯干和手臂关节分开映射到动作空间
- **动作构建**：`_build_action(arm_qpos, gripper_state, body_qpos)` → 15维动作 `[body(7), arm(6), gripper(2)]`
- **规划世界**：使用 `SapienPlanningWorldV2`，`qpos_offset=3`（跳过底盘3个位置关节），`base_link_name="torso_link1"`

### 3. `dsynth/planning/r1_skills.py` — R1 高层技能函数

针对 R1 的运动学特性实现的高层操作技能：

| 技能函数 | 功能 | 关键实现 |
|----------|------|----------|
| `r1_lift_body(planner, delta_h)` | 升降躯干 | PID 控制 `torso_joint4`（索引6），k_p=1.0, k_d=0.2 |
| `r1_align_ee_to_target_pos(env, planner, target_pos)` | 对齐末端到目标位置 | 先升降躯干，再静态规划手臂 |
| `r1_align_to_target_product(env, planner, target_product_actor)` | 对齐到目标商品 | 基于 OBB 中心计算目标位姿 |
| `r1_align_to_target_pose(env, planner, pose)` | 对齐到目标位姿 | 驱动底盘 + 升降躯干 |
| `r1_fetch_object_from_shelf(env, planner, target_product_actor)` | 从货架抓取商品 | 对齐→抓取姿态生成→接近→闭合夹爪→提升→后退 |
| `r1_place_object_to_pos(env, planner, target_center_pos, target_ee_direction)` | 放置商品到指定位置 | 对齐→放置姿态生成→接近→打开夹爪→后退 |
| `r1_drop_to_basket(env, planner)` | 将商品放入篮子 | 提升躯干→移动到篮子上方→打开夹爪 |

**关键设计决策**：
- `r1_lift_body` 使用 PID 控制而非直接设置关节位置，因为 R1 的4个躯干关节存在耦合，直接设置可能导致不稳定
- 抓取姿态生成区分圆柱体（`compute_cylinder_grasp_info`）和方体（`compute_box_grasp_thin_side_info`）

### 4. `dsynth/planning/solvers/r1_solvers.py` — R1 任务求解器

为每个任务类型实现 R1 版本的求解函数：

| 求解函数 | 对应任务 | 流程 |
|----------|----------|------|
| `solve_r1_pick_to_basket_cont_one_prod_w_skills` | PickToBasket | 对齐→抓取→放入篮子 |
| `solve_r1_move_to_board_cont_one_prod_w_skills` | MoveFromBoardToBoard | 对齐→抓取→对齐目标板→放置 |
| `solve_r1_pick_from_floor_cont` | PickFromFloor | 驱动到商品→预抓取→抓取→提升→驱动到货架→放置 |

这些求解器在 `dsynth/planning/__init__.py` 中注册到 `R1_MP_SOLUTIONS` 字典。

---

## 修改文件

### 1. `dsynth/planning/__init__.py`

新增 `R1_MP_SOLUTIONS` 字典，注册 R1 的3个任务求解器（PickToBasket、MoveFromBoardToBoard、PickFromFloor）。

### 2. `scripts/run_mp.py`

- 新增 `--robot-uids ds_r1` 命令行参数选项
- 根据 `robot_uids` 选择 `MP_SOLUTIONS` 或 `R1_MP_SOLUTIONS`

### 3. `scripts/run_keyboard_control.py`

- 新增 `--no-shadow` 参数：关闭阴影渲染，大幅提升 CPU 性能
- 新增 `--sim-freq` 参数：自定义仿真频率
- 新增 `--control-freq` 参数：自定义控制频率
- 传递 `sim_config` 到环境创建

### 4. `scripts/generate_restock_scene.py`

- 新增 `--robot` 参数（`fetch` / `r1`），控制场景生成配置
- 新增 R1 专用参数：
  - `R1_WAREHOUSE_ROWS = 2`、`R1_WAREHOUSE_COLS = 3`、`R1_WAREHOUSE_BOARDS = 2`
  - `R1_ROW_PRODUCTS`：R1 专用商品配置（2行×2层，每层2种×4件）
- 修改 `generate_commercial_layout()`：R1 模式下跳过50%的 inactive 货架，移除 wall 货架
- 修改 `generate_commercial_arrangement()`：R1 模式下商品密度降低（3板×2种×3件 vs 5板×3种×4件）
- 修改 `build_warehouse_shelves()`：R1 模式下2行×3列，跳过底层板
- 修改 `generate_warehouse_arrangement()`：R1 模式下填充板1-2（跳过底层0），每层4件

---

## R1 vs Fetch 对比

### 动作空间布局

```
Fetch (13维): [vx, vy, ω, torso_lift, arm_j1~j7, gripper]
R1   (15维): [base_x, base_y, base_θ, torso_j1~j4, arm_j1~j6, gripper_j1~j2]
              └── body (7) ──┘└── arm (6) ──┘└─ grip (2) ─┘
```

### 场景配置对比

| 配置项 | Fetch | R1 |
|--------|-------|----|
| 仓储区行数 | 3 | 2 |
| 仓储区列数 | 4 | 3 |
| 仓储区货架总数 | 12 | 6 |
| 仓储区填充板层 | 0,1,2（全部3层） | 1,2（跳过底层） |
| 每层商品种类×数量 | 2种×6-8件 | 2种×4件 |
| 商业区货架板数 | 5 | 3 |
| 商业区商品密度 | 3板×3种×4-5件 | 2板×2种×3件 |
| 商业区 inactive 跳过率 | 10% | 50% |
| 仓储区总商品数 | ~716 | ~264 |

### 为什么跳过底层货架？

R1 手臂基座高度约 1.403m，而货架底层板高度仅 0.206m。操作底层商品需要 R1 大幅弯腰，单臂操作空间极其局促，容易导致规划失败或碰撞。中层（0.653m）和顶层（1.105m）在 R1 的舒适操作范围内。

---

## 使用方法

### 1. 生成 R1 专用场景

```bash
# 生成 R1 场景（小货架、少商品、跳过底层）
python scripts/generate_restock_scene.py -r r1

# 生成 Fetch 场景（默认）
python scripts/generate_restock_scene.py -r fetch

# 指定输出目录
python scripts/generate_restock_scene.py -r r1 -o my_r1_scene
```

输出目录默认为 `generated_envs/restock_scene_r1/`。

### 2. 键盘遥操作

```bash
# R1 键盘控制（CPU，关闭阴影）
python scripts/run_keyboard_control.py generated_envs/restock_scene_r1/ \
    --sim-backend cpu --render-backend cpu -r ds_r1 --no-shadow

# R1 键盘控制（降低仿真频率）
python scripts/run_keyboard_control.py generated_envs/restock_scene_r1/ \
    --sim-backend cpu --render-backend cpu -r ds_r1 \
    --no-shadow --sim-freq 50 --control-freq 10

# Fetch 键盘控制（默认）
python scripts/run_keyboard_control.py generated_envs/restock_scene/ \
    --sim-backend cpu --render-backend cpu
```

### 3. 运动规划轨迹生成

```bash
# R1 轨迹生成
python scripts/run_mp.py -e PickToBasketContNiveaEnv \
    --scene-dir generated_envs/restock_scene_r1/ \
    -r ds_r1 -n 10 --vis

# Fetch 轨迹生成（默认）
python scripts/run_mp.py -e PickToBasketContNiveaEnv \
    --scene-dir generated_envs/restock_scene/ \
    -n 10 --vis
```

### 4. 可视化场景

```bash
python scripts/show_env_in_sim.py generated_envs/restock_scene_r1/ \
    --sim-backend cpu --render-backend cpu --gui
```

---

## 场景配置参数

### R1 仓储区参数（`generate_restock_scene.py`）

```python
R1_WAREHOUSE_ROWS = 2          # 2行（vs Fetch 3行）
R1_WAREHOUSE_COLS = 3          # 3列（vs Fetch 4列）
R1_WAREHOUSE_BOARDS = 2        # 填充2层板（vs Fetch 3层）
R1_WAREHOUSE_START_Y = 2.5     # Y方向起始偏移（给机器人更多空间）

R1_ROW_PRODUCTS = {
    'row_A_drinks': {
        1: {'food.DRINKS_SODA.FantaSaborNaranja2L': 4, 'food.DRINKS_SODA.Coca-ColaOriginal0.33L': 4},
        2: {'food.BEER.DuffBeerCan': 4, 'food.JUICE.TropicaOrangeJuice': 4},
    },
    'row_B_daily': {
        1: {'food.HYGIENE.NiveaBodyMilk': 4, 'food.HYGIENE.NiveaBodyLotion': 4},
        2: {'food.HOUSEHOLD.VanishStainRemover': 4, 'food.HOUSEHOLD.AjaxDishSoap': 4},
    },
}
```

板层编号说明：`1` = 中层板（0.653m），`2` = 顶层板（1.105m）。底层板（0，0.206m）被跳过。

### R1 商业区参数

```python
# R1 商业区货架
num_products_per_block = 4      # vs Fetch 7
num_products_per_board = 6      # vs Fetch 10
num_boards = 3                  # vs Fetch 5
board_product_numcol = {
    1: {'food.HYGIENE.NiveaBodyMilk': 3, 'food.drinks.coffeePackaging': 3},
    2: {'food.BEER.DuffBeerCan': 3, 'food.DRINKS_SODA.FantaSaborNaranja2L': 3},
}
inactive_shelvings_skip_prob = 0.5  # vs Fetch 0.1
```

---

## CPU 仿真性能优化

CPU 仿真卡顿的主要原因和解决方案：

| 瓶颈 | 原因 | 解决方案 | 预期提升 |
|------|------|----------|----------|
| 阴影渲染 | `shadow=True, shadow_map_size=2048` | `--no-shadow` | 2-3倍 |
| 物理子步过多 | sim_freq=100, control_freq=20 → 每帧5次子步 | `--sim-freq 50` | 2倍 |
| 物体数量 | 400+ 物理体 | R1 场景减少到 ~300 | 1.5倍 |
| nonconvex碰撞 | active货架使用非凸碰撞 | 场景已优化 | - |

**推荐配置**：

```bash
# 平衡性能和精度
python scripts/run_keyboard_control.py <scene_dir>/ \
    -r ds_r1 --no-shadow --sim-freq 50 --control-freq 10 \
    --sim-backend cpu --render-backend cpu

# 最大性能（牺牲精度）
python scripts/run_keyboard_control.py <scene_dir>/ \
    -r ds_r1 --no-shadow --sim-freq 40 --control-freq 20 \
    --sim-backend cpu --render-backend cpu
```

---

## 架构设计

### 模块依赖关系

```
scripts/run_mp.py
  ├── dsynth/planning/__init__.py
  │     ├── MP_SOLUTIONS (Fetch)
  │     └── R1_MP_SOLUTIONS (R1)
  │           ├── solve_r1_pick_to_basket_cont_one_prod_w_skills
  │           ├── solve_r1_move_to_board_cont_one_prod_w_skills
  │           └── solve_r1_pick_from_floor_cont
  │                 └── dsynth/planning/solvers/r1_solvers.py
  │                       └── dsynth/planning/r1_skills.py
  │                             ├── r1_lift_body
  │                             ├── r1_align_to_target_product
  │                             ├── r1_fetch_object_from_shelf
  │                             ├── r1_place_object_to_pos
  │                             └── r1_drop_to_basket
  │                                   └── dsynth/planning/r1_motionplanner.py
  │                                         └── R1MotionPlanningSapienSolver
  │                                               ├── drive_base (位置型PID)
  │                                               ├── move_base_forward_delta
  │                                               ├── rotate_base_z
  │                                               ├── static_manipulation
  │                                               │     ├── Screw规划
  │                                               │     └── RRTConnect规划
  │                                               └── follow_path
  └── dsynth/robots/ds_r1.py
        └── DSR1 (uid="ds_r1")

scripts/generate_restock_scene.py
  ├── -r fetch → 默认配置（3行×4列×3层）
  └── -r r1    → R1配置（2行×3列×2层，跳过底层）

scripts/run_keyboard_control.py
  ├── -r ds_r1 → R1 15维动作空间
  ├── --no-shadow → 关闭阴影
  ├── --sim-freq → 仿真频率
  └── --control-freq → 控制频率
```

### R1 动作空间映射

```python
# _build_action(arm_qpos, gripper_state, body_qpos)
action = np.hstack([body_qpos, arm_qpos, gripper_state])
# body_qpos: [base_x, base_y, base_θ, torso_j1, torso_j2, torso_j3, torso_j4]  (7维)
# arm_qpos:  [left_arm_j1, j2, j3, j4, j5, j6]                                 (6维)
# gripper:   [left_gripper_finger_j1, j2]                                        (2维)
# 总计: 15维
```

### R1 规划器 qpos 偏移

mplib 规划器使用的关节向量不包含底盘的3个位置关节（base_x, base_y, base_θ），因此：

```python
# 规划器 qpos = [torso_j1~j4, arm_j1~j6, gripper_j1~j2]  (12维)
# 机器人 qpos = [base_x, base_y, base_θ, torso_j1~j4, arm_j1~j6, gripper_j1~j2]  (15维)
# qpos_offset = 3  # 跳过前3个底盘关节
```

规划器中的 `base_link_name="torso_link1"` 表示底盘位姿以 `torso_link1` 为参考，`set_base_pose()` 在每次规划前更新底盘位姿。
