# 按技能构建 LeRobot 数据集

## 目标
从 `generated_data/` 的 state-only h5 → 3 个**按技能合并**的 LeRobot v2.1 数据集,
每技能合并多个物品,每条 episode 带自己的指令,每个物品最多取 30 条。

```
datasets/warehouse_fetch/
├── pick_to_basket/            (Duff + Fanta + Stars 合并)
├── restock_basket_to_shelf/   (Duff + Fanta + Stars 合并)
└── pick_from_floor/           (Beans + Slam 合并)
```

## 关键事实(已实测确认)
1. 生成的 h5 **无图像**(`obs:[]`),必须先 `replay_trajectory.py --obs-mode rgbd --save-traj`。
2. 多进程输出分目录:`{env}/{traj}.0/`、`.1/`、`.2/`,各含一个 h5(各 ~10 条)。
3. replay 输出命名:`{traj}.rgbd.pd_joint_pos.physx_cpu[.{proc}].h5`,与源同目录。
4. replayed h5 真实结构(每个 `traj_N`):
   - `obs/agent/qpos (T+1,15)` → observation.state
   - `obs/sensor_data/head_camera/rgb (T+1,360,640,3)`
   - `obs/sensor_data/fetch_hand/rgb (T+1,128,128,3)`
   - `actions (T,13)`,`obs/extra/language_instruction_bytes`
   - **off-by-one**:actions=T,obs=T+1 → 取 state/图像的前 T 帧对齐 action。
5. 现有 `convert_robobenchmart_to_lerobot.py` 已过时(期望旧 key、15维action、单task、256²),需重写。

## 实现步骤

### 1. 新转换脚本 `scripts/convert_skill_to_lerobot.py`(RoboBenchMart 内,用 robort_mart env)
- 入参:`--skill-dir`(多个 env 目录)、`--output-dir`、`--task-map`(env→指令)、`--max-per-item 30`、`--fps 15`。
- 递归找每个 env 下所有 `*.rgbd.*.h5`(含 proc 子目录),每个物品**最多取 30 条 traj**。
- 按真实结构读:`obs/agent/qpos[:T]`(state 15维)、`actions`(13维)、两路 rgb 取前 T 帧。
- 每条 episode 分配 `task_index`(按指令文本去重建表),写 per-episode 指令到 episodes.jsonl。
- 编码 mp4:head_rgb(360×640)、left_wrist_rgb(128×128)、right_wrist_rgb(复制 left)。
- 写 parquet:observation.state(15)、action(13)、index/episode_index/frame_index/timestamp/task_index。
- 写 meta:info.json(features 真实 shape)、tasks.jsonl(多条)、episodes.jsonl。

### 2. 新编排脚本 `bash/build_skill_datasets.sh`
- Step 1 replay:对每个 env 的**每个 proc 子目录**的 h5 跑 replay --obs-mode rgbd(跳过已存在的)。
- Step 2 convert:每个技能调用一次新脚本,传该技能所有 env 目录 + task-map。
- 三技能映射:
  - pick_to_basket: Duff/Fanta/Stars Env(各自指令)
  - restock_basket_to_shelf: Restock Duff/Fanta/Stars
  - pick_from_floor: Beans/Slam

### 3. 验证
- 转换后用 lerobot/galaxea loader 读一个数据集确认 shape;或脚本内打印 episode 数 + 抽检 parquet/mp4。

## 待确认
- action 13维 vs GalaxeaVLA 期望:info.json 按**真实 13维**写(忠实数据),训练侧维度适配由 VLA config 处理,不在此脚本伪造。
- state 用 `obs/agent/qpos`(15维)。

## 不做
- 不改 run_mp.py / 已生成数据 / 已有的 Duff·Fanta 60条数据(转换时各取前30)。
- 不调用 Workflow(无需多代理)。
