# RoboBenchMart 任务场景文档

## 1. 场景概述

RoboBenchMart 是一个基于 ManiSkill3 仿真框架的**暗仓/超市机器人操作基准测试平台**。场景模拟了一个真实的暗仓（Darkstore）环境，包含多排货架、多种商品、冰箱/展示柜等设施，机器人需要在此环境中完成导航与操作任务。

### 1.1 环境参数

| 参数 | 值 |
|------|-----|
| 场景类型 | 暗仓（Darkstore） |
| 单元格尺寸 | 1.55m × 1.55m |
| 房间高度 | 2.7m |
| 货架类型 | 普通货架、冰箱、玻璃顶展示柜、单面小货架、双面小货架 |
| 货架层数 | 默认5层 |
| 每层商品数 | 默认10个 |
| 商品总数 | 100~150个/场景 |

### 1.2 机器人平台

| 机器人 | UID | 说明 |
|--------|-----|------|
| Fetch（带篮子） | `ds_fetch_basket` | 移动操作机器人，底盘+7自由度机械臂+夹爪+前置篮子 |
| Fetch（无篮子） | `ds_fetch` | 同上，无篮子 |
| Fetch（静态+篮子） | `ds_fetch_basket_static` | 底盘锁定，仅手臂操作 |
| Fetch（静态） | `ds_fetch_static` | 底盘锁定，无篮子 |
| R1 | `ds_r1` | 双臂移动操作机器人 |
| Panda | `panda_wristcam` | 固定基座单臂机器人（腕部相机） |

### 1.3 传感器配置（DSFetchBasket）

| 传感器 | 分辨率 | FOV | 位置 |
|--------|--------|-----|------|
| 头部深度相机 | 640×360 | 82.6°（水平114.7°） | 头部相机链接 |
| 手部相机 | 128×128 | 114.6° | 夹爪链接 |

---

## 2. 导航任务

### 2.1 Point-to-Point 导航（iPlanner）

**环境注册名**：自定义脚本 `scripts/run_iplanner_nav.py`

**任务描述**：机器人从起点出发，使用 iPlanner 深度学习导航模型，在暗仓走廊中自主导航至目标点，途中需避开货架、障碍物等。

**详细流程**：

```
1. 初始化阶段
   ├── 设置导航姿态（手臂收至右侧，低头0°）
   ├── 原地旋转对准目标方向（P控制，Kp=0.8，容差5°）
   └── 设置头部相机角度（0°平视，匹配iPlanner训练条件）

2. 导航循环（15Hz重规划）
   ├── 获取深度图（640×360，uint16毫米级）
   ├── 深度图预处理
   │   ├── 毫米→米转换（/1000）
   │   ├── 超过10m的深度值置零（在resize前）
   │   ├── 双线性插值resize至360×640
   │   └── 复制3通道（单通道→三通道）
   ├── 计算目标在机器人坐标系的坐标
   ├── iPlanner推理：深度图+目标→关键点+恐惧值
   ├── 轨迹插值：3个关键点→样条插值生成密集路径
   ├── Pure-Pursuit路径跟踪
   │   ├── 跳过距离<0.25m的路点（conv_dist）
   │   ├── 选择距离>0.5m的前瞻路点（look_ahead）
   │   └── 计算线速度和角速度
   └── 恐惧值处理
       ├── isForwardTracking判断（路点方向与前方夹角<60°）
       ├── fear>0.5且前进中→fear_buffer+1
       ├── fear≤0.5→fear_buffer-1
       └── fear_buffer>3→停车（lv=0, av=0）

3. 结束条件
   ├── 到达目标（距离<0.25m）
   ├── 超过最大步数
   └── 恐惧停车
```

**关键参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| 规划频率 | 15Hz（每7个仿真步） | 匹配官方iPlanner频率 |
| 最大深度 | 10.0m | 超过此距离置零 |
| 最大目标距离 | 10.0m | 目标裁剪范围 |
| look_ahead | 0.5m | Pure-Pursuit前瞻距离 |
| conv_dist | 0.25m | 路点收敛距离 |
| max_linear_vel | 0.3 m/s | 最大线速度 |
| max_angular_vel | 0.3 rad/s | 最大角速度 |
| fear阈值 | 0.5 | 触发恐惧反应的fear值 |
| fear_buffer | 3 | 连续fear触发次数阈值 |
| 传感器偏移 | (0.164, 0.045)m | 相机相对于机器人中心的偏移 |

**成功条件**：机器人到达目标点（距离<0.25m）

### 2.2 区域导航（NavMoveToZoneEnv）

**环境注册名**：`NavMoveToZoneEnv`

**任务描述**：机器人从随机起点导航至指定区域（zone），到达后需面向货架方向。

**详细流程**：

```
1. 初始化
   ├── 随机选择空闲单元格作为起点
   ├── 随机选择朝向角度
   └── 随机选择目标区域（或指定TARGET_ZONE_NAME）

2. 目标区域定义
   ├── 货架前方的单元格（面对货架方向）
   └── 对面单元格（如果存在且空闲）

3. 导航执行
   └── 机器人自主导航至目标区域
```

**成功条件**：
- 机器人位于目标单元格内
- 机器人朝向与货架方向一致（误差<0.11）
- 机器人静止（关节速度<0.2）

---

## 3. 操作任务

### 3.1 从货架拿到篮子（PickToBasket）

**环境注册名**：`PickToBasketContEnv` 及其变体

**任务描述**：机器人接近货架，用机械臂从货架上抓取指定商品，放入自身前置篮子中。

**详细流程**：

```
1. 初始化
   ├── 随机选择目标商品（或指定TARGET_PRODUCT_NAME）
   │   ├── 随机选择区域（zone）
   │   ├── 随机选择货架（shelf）
   │   └── 随机选择该货架上的商品
   ├── 机器人放置在货架前方1.4m处
   │   ├── 朝向货架方向
   │   └── 可选随机偏移（ROBOT_INIT_POSE_RANDOM_ENABLED）
   └── 手臂初始化为操作姿态

2. 操作执行
   ├── 机器人导航至货架前方
   ├── 识别并定位目标商品
   ├── 控制机械臂抓取商品
   └── 将商品放入篮子（机器人底盘前方0.3m、右侧0.25m、高0.14m）

3. 目标位置
   └── 篮子位置 = 机器人底盘位姿 × [0.3, 0.25, 0.14]
```

**成功条件**：
- 目标商品位于篮子容差范围内（0.2m）
- 其他商品未被移动（位移<0.1m）
- 机器人静止（关节速度<0.2）

**语言指令**：`"move to the shelf and pick {product_name} and put to the basket"`

**商品变体**：

| 环境名 | 目标商品 | 类型 |
|--------|---------|------|
| PickToBasketContNiveaEnv | Nivea Body Milk | 训练集 |
| PickToBasketContStarsEnv | Nestle Honey Stars | 训练集 |
| PickToBasketContFantaEnv | Fanta Sabor Naranja 2L | 训练集 |
| PickToBasketContNestleEnv | Nestle Fitness Chocolate Cereals | 测试集（未见） |
| PickToBasketContSlamEnv | SLAM luncheon meat | 测试集（未见） |
| PickToBasketContDuffEnv | Duff Beer Can | 测试集（未见） |

### 3.2 从篮子拿到货架（RestockBasketToShelf）

**环境注册名**：`RestockBasketToShelfContEnv`

**任务描述**：机器人从自身篮子中拿起商品，放回货架上的原始位置。这是 PickToBasket 的逆任务。

**详细流程**：

```
1. 初始化
   ├── 目标商品默认为 Duff Beer Can
   ├── 商品初始放置在篮子中
   │   ├── 位置：篮子中心偏右0.05m、偏上0.08m
   │   └── 姿态：随机旋转（±27°倾斜，0~360°偏航）
   └── 机器人放置在货架前方（无随机偏移）

2. 操作执行
   ├── 从篮子中抓取商品
   ├── 导航至货架前方
   └── 将商品放回原始货架位置
```

**成功条件**：
- 商品回到初始位置（位移<0.1m）
- 机器人静止（关节速度<0.2）

**语言指令**：`"pick {product_name} from basket and place on shelf"`

### 3.3 从地面捡起放回货架（PickFromFloor）

**环境注册名**：`PickFromFloorContEnv` 及其变体

**任务描述**：机器人发现掉落在地面的商品，将其捡起并放回货架上的原始位置。

**详细流程**：

```
1. 初始化
   ├── 选择货架边缘的商品（第0行，最外侧）
   ├── 记录商品原始位置作为目标
   ├── 将商品放置在地面上
   │   ├── 位置：货架前方1.4m处 + 随机偏移
   │   │   ├── 沿货架方向偏移：0.3~0.7m
   │   │   └── 垂直方向偏移：±0.25m
   │   ├── 高度：商品最大截面半径/2 + 0.01m
   │   └── 姿态：随机偏航角（±45°）
   └── 机器人放置在货架前方

2. 操作执行
   ├── 识别地面上的商品
   ├── 导航至商品附近
   ├── 弯腰/伸展手臂抓取地面商品
   └── 将商品放回货架原始位置
```

**成功条件**：
- 商品回到原始货架位置（容差0.2m）
- 其他商品未被移动（位移<0.1m）
- 机器人静止（关节速度<0.2）

**语言指令**：`"pick {product_name} from floor and place it on shelf"`

**商品变体**：

| 环境名 | 目标商品 | 类型 |
|--------|---------|------|
| PickFromFloorBeansContEnv | Heinz Beans in a rich tomato sauce | 训练集 |
| PickFromFloorSlamContEnv | SLAM luncheon meat | 训练集 |
| PickFromFloorFantaContEnv | Fanta Sabor Naranja 2L | 测试集（未见） |
| PickFromFloorDuffContEnv | Duff Beer Can | 测试集（未见） |

---

## 4. 其他操作任务

### 4.1 层间移动（MoveFromBoardToBoard）

**环境注册名**：`MoveFromBoardToBoardContEnv`

**任务描述**：机器人从货架上抓取指定商品，将其移动到同一货架的上一层板。

**关键参数**：层间距 = 0.397m

**成功条件**：
- 商品位于上一层板的目标位置（容差0.15m）
- 其他商品未被移动
- 机器人静止

### 4.2 开/关冰箱门（OpenDoorShowcase / CloseDoorShowcase）

**环境注册名**：`OpenDoorShowcaseContEnv` / `CloseDoorShowcaseContEnv`

**任务描述**：机器人接近展示柜/冰箱，打开/关闭指定门。

**门编号**：first、second、third、fourth（4扇门）

**成功条件**：
- 开门：门角度与展示柜成90°（容差0.2rad）
- 关门：门角度与展示柜成0°（容差0.2rad）
- 机器人静止

### 4.3 开/关冰箱（OpenDoorFridge / CloseDoorFridge）

**环境注册名**：`OpenDoorFridgeContEnv` / `CloseDoorFridgeContEnv`

**任务描述**：机器人接近冰箱，打开/关闭冰箱门。

**成功条件**：
- 开门：右盖关节角度=0.624rad（容差0.1rad）
- 关门：右盖关节角度=0rad（容差0.1rad）
- 机器人静止

---

## 5. 组合任务

### 5.1 组合任务框架

组合任务通过 `get_composite_task()` 工厂函数创建，将多个子任务串联执行。

**执行逻辑**：
```
1. 执行第1个子任务
2. 子任务成功 → 切换到下一个子任务
3. 更新目标商品、语言指令
4. 所有子任务完成 → 总体成功
```

**评估指标**：
- `task_0`, `task_1`, ... : 各子任务是否完成
- `success_length` : 已完成子任务比例
- `success` : 所有子任务是否全部完成

### 5.2 预定义组合任务

| 环境名 | 子任务序列 |
|--------|-----------|
| PickNiveaFantaEnv | PickNivea → PickFanta |
| PickNiveaFantaStarsEnv | PickNivea → PickFanta → PickStars |
| OpenPickDuffCloseEnv | 开门 → 拿Duff → 关门 |

---

## 6. 随机化

所有任务均支持以下随机化：

| 随机化项 | 说明 |
|---------|------|
| 场景布局 | 货架位置、朝向、间距 |
| 商品排列 | 商品在货架上的摆放顺序、位置 |
| 墙壁/地板纹理 | 视觉外观变化 |
| 机器人初始位置 | 货架前方的随机偏移（平行0~0.4×CELL_SIZE，垂直±0.4×CELL_SIZE） |
| 机器人初始朝向 | ±22.5°随机偏转 |
| 目标商品 | 从场景中随机选择（如未指定） |
| 目标门/区域 | 随机选择（如未指定） |

---

## 7. 通用评估指标

| 指标 | 说明 |
|------|------|
| `success` | 任务是否完全成功 |
| `is_obj_placed` | 物体是否放置到目标位置 |
| `is_robot_static` | 机器人是否静止（关节速度<0.2） |
| `is_non_target_products_displaced` | 非目标商品是否被意外移动 |
| `is_door_opened` / `is_door_closed` | 门是否打开/关闭 |
| `is_robot_placed` | 机器人是否到达目标位置 |
| `is_target_in_view` | 机器人是否面向目标方向 |
