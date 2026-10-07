# AM-Bench 功能复现与验证记录

本手册对应 [AM-Bench 公开文档](https://ambench.github.io/docs/) 的任务、机型、控制器、扰动、脚本专家、数据集和 ACT / Diffusion Policy / OpenPI 工作流。Docker 构建、远端策略容器和 SSH 隧道先按 [note.md](note.md) 完成。宿主机命令从仓库根目录执行；标有“容器内”的仿真命令先执行 `docker exec -it ambench-sim-research bash`，在默认的可写 `/workspace/ambench-run` 下运行。这个目录链接到只读源码和可写数据目录，并为 MPC 保留可写的 acados 生成路径；命令中的 `scripts/...` 相对路径保持不变。远端策略容器仍以 `/workspace/ambench` 为工作目录。

![AM-Bench 公开总览](assets/ambench-overview.png)

图片是仓库自带的项目总览。下文的验证表只记录实际运行结果；静态注册检查、仿真 reset/step、脚本任务成功和模型评估分别记录，不能互相代替。

## 阅读路线

| 目标 | 章节 | 可得到的结果 |
| --- | --- | --- |
| 首次运行 Docker 仿真 | `note.md`、第 0–1 节 | 容器启动、106 个注册 ID 与第一个有界场景 |
| 检查五种机型及全部任务 | 第 1–3 节 | 控制组合、12 族场景、相机与扰动的逐项记录 |
| 采集示范并核验数据 | 第 4 节 | 脚本专家、遥操作入口与 canonical LeRobot 校验 |
| 训练及运行高层模型 | 第 5–6 节 | ACT、DP、π₀、π₀.₅ 的远端训练、服务和闭环评估 |
| 判断本分支已实际验证什么 | 第 7 节 | 测试产物、通过项和仍需完成的环节 |

## 0. 复现范围与证据

| 层级 | 通过条件 | 结果位置 |
| --- | --- | --- |
| 注册 | Isaac Sim 启动后 Gym registry 恰好含 106 个 AM-Bench ID，12 个任务族且各有脚本专家入口 | `list_envs.py` 输出、矩阵 JSON |
| 场景与低层 | 每个 ID 能构造、reset、执行至少 8 步；进程按上限退出 | `outputs/research/rebuild-20261007/matrix_verified.json` 与逐 ID 日志 |
| 脚本专家 | 对应任务生成至少 1 个完成成功判定的 episode | `<session>/lerobot/meta/episodes/` 与图像帧；使用 `--video` 时另存 MP4 |
| 数据 | canonical LeRobot validator 成功；DP/OpenPI 派生数据各自验证 | `<session>/lerobot/meta/info.json`、转换结果 |
| 模型 | 匹配 checkpoint 经过远端服务完成闭环 rollout | `eval_summary.json` 中 `status: "completed"` 和足量 rollout |

`zero_agent` 的成功只证明场景与控制流水线能推进。它不预示脚本专家必定成功，也不代表论文任务成功率。完整模型复现需要任务数据和 checkpoint；本仓库不提交这些生成物。[验证快照](usage_assets/validation_snapshot.json)保存最终 106 个 ID、12 个成功专家和四种高层模型 20 秒闭环的摘要，并记录本机原始报告的 SHA-256；大日志和模型权重仍留在忽略目录。

## 1. 环境注册与命名

```text
<Task>-Am-<Robot>-<Action>-<Controller>-Direct-v0
<Task>-Am-FAHexa-BaseJoint-Abs-<Controller>-Direct-v0
```

所有 12 族都注册下面 8 种基础配置，额外配置见表后。`EE` 是悬浮末端执行器的任务逻辑 oracle；物理飞行器是 UAQuad、UAHexa、FAHexa 和 OmniHexa。

| 机型 | 控制与动作 | 单族基础 ID 后缀 |
| --- | --- | --- |
| EE | 6 DoF PID、L1；EE 绝对位姿 | `EE-Abs-PID`、`EE-Abs-L1` |
| UAQuad | 4 DoF PID + Pyroki IK；EE 绝对位姿 | `UAQuad-Abs-PID` |
| UAHexa | 4 DoF PID + Pyroki IK；EE 绝对位姿 | `UAHexa-Abs-PID` |
| FAHexa | 6 DoF PID、L1 + Pyroki IK；另有 whole-body MPC | `FAHexa-Abs-PID`、`FAHexa-Abs-L1`、`FAHexa-Abs-MPC` |
| OmniHexa | 6 DoF PID + Pyroki IK；EE 绝对位姿 | `OmniHexa-Abs-PID` |

`PressButton`、`PushSlider`、`RotateValve` 各有 FAHexa BaseJoint PID/L1 两个额外 ID；这三个任务和 `LemonHarvesting` 各有一个 `EE-Abs-PID-Direct-Fast-v0`。合计 `12×8 + 3×2 + 4 = 106`。真实注册表永远由运行时 Gym registry 判定，不要靠字符串拼出未注册组合。

容器内执行：

```bash
python scripts/environments/list_envs.py
python scripts/research/verify_environment.py --list-json outputs/research/registry.json --headless
```

### 1.1 五种机型与三类控制器

下面每条命令各验证一个不同的实体或控制路径。单卡 16 GB 上每次只开一个 Isaac Sim 进程、`--num_envs 1`；若某个物理机型要首次生成 MPC 代码，应保留该项完整日志。

```bash
python scripts/research/verify_matrix.py --task-id PressButton-Am-EE-Abs-PID-Direct-v0 --steps 8 --timeout-s 180 --output-dir outputs/research/ee
python scripts/research/verify_matrix.py --task-id PressButton-Am-UAQuad-Abs-PID-Direct-v0 --steps 8 --timeout-s 180 --output-dir outputs/research/uaquad
python scripts/research/verify_matrix.py --task-id PressButton-Am-UAHexa-Abs-PID-Direct-v0 --steps 8 --timeout-s 180 --output-dir outputs/research/uahexa
python scripts/research/verify_matrix.py --task-id PressButton-Am-FAHexa-Abs-PID-Direct-v0 --steps 8 --timeout-s 180 --output-dir outputs/research/fahexa
python scripts/research/verify_matrix.py --task-id PressButton-Am-OmniHexa-Abs-PID-Direct-v0 --steps 8 --timeout-s 180 --output-dir outputs/research/omnihexa
python scripts/research/verify_matrix.py --task-id PressButton-Am-FAHexa-Abs-L1-Direct-v0 --steps 8 --timeout-s 180 --output-dir outputs/research/l1
python scripts/research/verify_matrix.py --task-id PressButton-Am-FAHexa-Abs-MPC-Direct-v0 --steps 8 --timeout-s 240 --output-dir outputs/research/mpc
python scripts/research/verify_matrix.py --task-id PressButton-Am-FAHexa-BaseJoint-Abs-PID-Direct-v0 --steps 8 --timeout-s 180 --output-dir outputs/research/basejoint
```

控制器公共契约在 `source/ambench/ambench/controllers/`，机型与动作组合在 `source/ambench/ambench/tasks/base/robot_profiles.py`。Pyroki 求逆运动学，PID/L1/MPC 负责低层控制。BaseJoint 动作绕过 IK，直接给基座位姿和关节目标。

## 2. 十二个任务与全部 106 种场景配置

| 任务前缀 | 时限 | 成功信号 | 额外配置 |
| --- | ---: | --- | --- |
| `CabinetPickPlace` | 60 s | 罐体进抽屉指定高度且速度低于阈值 | 无 |
| `FrameAssembly` | 30 s | 框架中心落在 peg 容差内 | 无 |
| `LemonHarvesting` | 30 s | 柠檬进入容器且夹爪打开 | EE Fast |
| `NDT` | 20 s | 末端在检测点保持指定步数 | 无 |
| `OpenDoor` | 15 s | 门关节超过开启角 | 无 |
| `PegInHole` | 20 s | peg 尖端穿过孔坐标系的深度与横向边界 | 无 |
| `PressButton` | 20 s | 按钮关节达到阈值 | BaseJoint PID/L1、EE Fast |
| `PullLever` | 20 s | 杠杆关节超过最小角度 | 无 |
| `PushSlider` | 26 s | 滑块关节达到目标 | BaseJoint PID/L1、EE Fast |
| `RotateValve` | 20 s | 阀门关节达到目标角 | BaseJoint PID/L1、EE Fast |
| `TossBall` | 10 s | 球在容器内释放、机体保持在要求区域 | 无 |
| `WipeWindow` | 20 s | 每个污渍满足接触清除条件 | 无 |

WipeWindow 的公开评估时限是 20 秒。修正夹持并缩短后续局部擦拭轨迹后，EE PID 脚本专家已在该默认时限内完成四处污点清除：成功示范为 1747 帧、120 Hz（约 14.6 秒），canonical LeRobot validator 通过。研究分支在 WipeWindow 场景中用固定关节保持海绵工具的夹持，因为原实现接触窗面时会脱手；这会改变工具动力学，跨分支比较成功率时应使用相同实现。此前 40 秒测试保留在合并报告的尝试历史中。

固定工具关节还通过双环境克隆检查：`outputs/research/wipe-fixed-two-envs-20260930T2044/results.json` 完成 8 步、动作形状 `(2, 8)`，两个环境均含接触传感器与 EE 相机。

先按每族 EE PID 做独立 smoke；随后运行注册表驱动的全量矩阵，含物理机型、L1、MPC、BaseJoint 与 Fast。脚本为每个 ID 保存退出状态和日志，中途一个失败不会掩盖其他条目。当前矩阵入口默认 `--seed 42`，每个 child 和汇总报告都保存 seed；此前 2026-09-30 的无固定 seed 报告作为历史记录保留。

```bash
python scripts/research/verify_matrix.py \
  --all --steps 8 --seed 42 --timeout-s 600 \
  --video-family-representatives --video-steps 60 \
  --output-dir outputs/research/rebuild-20261007/matrix
```

已有矩阵输出时，按 `results.json` 中的失败 ID 用 `--task-id` 重跑；修复后重新执行全量命令并换一个空的 `--output-dir` 生成新记录。工具会拒绝非空输出目录，防止旧结果混入。`--family PressButton` 与 `--robot FAHexa` 可用于缩小范围。NDT 首次加载远端仓库资产时，本次单项超过 240 秒，因此冷启动的全量命令使用 600 秒上限。机器上只有一张仿真 GPU，不要同时跑多个矩阵任务。

2026-10-07 删除旧镜像、重新构建后，固定 seed 42 的整份矩阵一次通过 106/106；12 个 EE PID 代表各录制 60 步，其他 ID 各推进 8 步。下面的合并器读取这份新报告和每个 child JSON，逐项核对步数、退出状态、日志、seed 与来源 SHA-256；在仿真容器内运行。以下输出目录是本次验收路径，重复运行应换成新的空目录。2026-09-30 的首轮失败及修复记录作为历史保留在 `outputs/research/combined-final/`，不混入本次复测：

```bash
python scripts/research/merge_matrix_results.py \
  outputs/research/rebuild-20261007/matrix/results.json \
  --output outputs/research/rebuild-20261007/matrix_verified.json \
  --expect-count 106 --expect-families 12
```

矩阵中每族 EE PID 的 60 步录像可提取成可追溯的中间帧；脚本会同时写出源 MP4 和 PNG 的 SHA-256。以下命令在仿真容器内执行，`output-dir` 必须是空目录：

```bash
python scripts/research/extract_scene_frames.py \
  --results-json outputs/research/rebuild-20261007/matrix/results.json \
  --output-dir outputs/research/rebuild-20261007/scene_frames
```

### 2.1 十二类场景的实测相机帧

下图均来自 2026-10-07 删除旧镜像并重新构建后，固定 seed 42 的 Isaac Sim 5.1 Docker EE PID 场景录像。来源是 `outputs/research/rebuild-20261007/matrix/results.json`：每类运行 60 步，从 MP4 提取中间帧；[帧清单](usage_assets/scenes/manifest.json)记录源录像路径、帧位置和文件 SHA-256。它们证明场景和相机可运行，不表示任务成功。

| CabinetPickPlace | FrameAssembly |
| --- | --- |
| ![CabinetPickPlace 相机帧](usage_assets/scenes/cabinetpickplace_ee_pid.png) | ![FrameAssembly 相机帧](usage_assets/scenes/frameassembly_ee_pid.png) |
| LemonHarvesting | NDT |
| ![LemonHarvesting 相机帧](usage_assets/scenes/lemonharvesting_ee_pid.png) | ![NDT 相机帧](usage_assets/scenes/ndt_ee_pid.png) |
| OpenDoor | PegInHole |
| ![OpenDoor 相机帧](usage_assets/scenes/opendoor_ee_pid.png) | ![PegInHole 相机帧](usage_assets/scenes/peginhole_ee_pid.png) |
| PressButton | PullLever |
| ![PressButton 相机帧](usage_assets/scenes/pressbutton_ee_pid.png) | ![PullLever 相机帧](usage_assets/scenes/pulllever_ee_pid.png) |
| PushSlider | RotateValve |
| ![PushSlider 相机帧](usage_assets/scenes/pushslider_ee_pid.png) | ![RotateValve 相机帧](usage_assets/scenes/rotatevalve_ee_pid.png) |
| TossBall | WipeWindow |
| ![TossBall 相机帧](usage_assets/scenes/tossball_ee_pid.png) | ![WipeWindow 相机帧](usage_assets/scenes/wipewindow_ee_pid.png) |

### 2.2 观察、动作、奖励与成功率

所有任务返回 Gymnasium 五元组 `obs, reward, terminated, truncated, info`。`obs["policy"]` 含末端位姿/速度、基座位姿/速度、夹爪宽度，物理机型还含机械臂关节状态；任务再添加目标与物体状态。四元数采用 WXYZ；位置相对各环境 origin。EE 绝对动作为位置 3 + 四元数 4 + 夹爪 1，共 8 维；BaseJoint 为基座位置 3 + 四元数 4 + 机械臂关节 4 + 夹爪 1，共 12 维。

当前任务 reward 为零。最终成功由 `terminated` 表示，时间上限由 `truncated` 表示。有命名子任务的场景在 `info["success_criteria"]` 中返回布尔状态；评估器记录每项是否曾达到。要比较模型，使用相同任务 ID、配置、seed、动作语义、策略频率、相机、扰动和 episode 上限。

## 3. 物理、扰动与相机

默认物理步长为 `1/120 s`。`BaseEnvCfg` 提供动作噪声、观察噪声、转子饱和、气动效应、风力五个独立开关；默认全关。气动效应包括地面、近墙与阻力，风向量由具体 robot spec 给出；只打开风力开关而保持默认零向量不会产生风。EE oracle 没有转子，因此不应用于飞行动力学结论。

一个不改注册表的扰动评估入口是模型 evaluator 的 `--disturbance`，它共同开启饱和、气动和风力开关。研究单个效应时，应在派生环境配置中只改变目标字段并注册新的 ID，记录完整 `env_cfg.yaml`。相机由机型 profile 配置，常用 `ee_camera`、`base_camera`；部分场景还可用 `scene_camera`。

要检查相机实际画面，容器内运行一个有界录像：

```bash
timeout --signal=INT 45s python scripts/environments/zero_agent.py \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --num_envs 1 --video --camera_names ee_camera --headless --device cuda:0
```

这会在忽略目录 `videos/` 产生 MP4。`zero_agent.py` 是连续运行器，超时应发生在 scene/reset/step 已开始之后；检查子进程确实退出。该录像展示相机和场景，不代表专家动作。

下图保留 2026-09-30 的历史 Docker 实测：`PressButton-Am-EE-Abs-PID-Direct-v0`、`verify_matrix.py --task-id ... --video-task-id ... --video-steps 8`、单环境、EE 相机；8/8 步通过。帧取自 `outputs/research/press-ee-video-20260930T1744/` 的 MP4 中间帧，画面显示灰墙、红色按钮和白色夹爪。它是场景检查，不是按下按钮后的成功画面。

![PressButton EE PID 在 Isaac Sim 5.1 Docker 中的 EE 相机实测帧](usage_assets/press_button_ee_pid.png)

物理机型也已跑通独立相机：下面保留 2026-09-30 的历史画面，UAQuad PID 在 PressButton 场景用 `base_camera` 录制 30 步，取第 14 帧。容器内复现命令为：

```bash
python scripts/research/verify_matrix.py \
  --task-id PressButton-Am-UAQuad-Abs-PID-Direct-v0 \
  --video-task-id PressButton-Am-UAQuad-Abs-PID-Direct-v0 \
  --video-camera-name base_camera --video-steps 30 \
  --timeout-s 240 --output-dir outputs/research/uaquad_base_camera
```

![PressButton UAQuad PID 的 base_camera 实测帧](usage_assets/press_button_uaquad_base_camera.png)

FAHexa PID 的扰动组合也已执行 8 步，包括转子饱和、气动、1 N 的 X 向风、动作噪声和观察噪声；这些设置与实际测试一致：

```bash
python scripts/research/verify_matrix.py \
  --task-id PressButton-Am-FAHexa-Abs-PID-Direct-v0 \
  --disturbance --wind-force 1 0 0 --action-noise --observation-noise \
  --steps 8 --timeout-s 240 --output-dir outputs/research/fahexa_disturbance
```

MPC 的 acados 会在当前目录生成 C 代码和模型 JSON。仿真容器默认位于可写 `/workspace/ambench-run`，因此普通 `zero_agent.py` 和 ACT/DP 评估入口可直接使用 MPC 任务 ID，源码仍只读。GUI 遥操作的 X11 启动命令和无头恢复命令见 [note.md](note.md)。本分支的两项额外实测如下：

| 功能 | 有界实测结果 |
| --- | --- |
| FAHexa MPC 普通入口 | `python scripts/environments/zero_agent.py --task PressButton-Am-FAHexa-Abs-MPC-Direct-v0 --num_envs 1 --headless` 完成 reset 和 2,386 次步进；`outputs/mpc_zero_agent_normal_cwd.log` 无权限错误或 traceback，生成文件仅在 Docker 工作卷。 |
| 本地 X11 GUI 与键盘设备 | `PressButton-Am-EE-Abs-PID-Direct-v0` 出现 1440×900 Isaac Sim 5.1.0 窗口，场景构造及 `Se3Keyboard` 初始化完成；`outputs/gui_teleop_x11.log` 出现 `Teleoperation started`。连续运行器在 180 秒上限结束，exit 124 为预期；没有手动键盘动作示范。 |

## 4. 十二个脚本专家与 canonical 数据

每个注册 ID 都携带 `scripted_policy_entry_point`。脚本专家使用任务状态机，经当前机器人控制流水线执行；BaseJoint 录制器通过当前 profile 的 IK 配置把专家 EE 目标转换为基座与关节绝对命令。先在 EE PID 下为每族取一条成功示范，随后检查各机型上的适用性。

全部 12 族的有界采集和 canonical 验证可逐项运行，失败后继续并保存 JSON/CSV 与日志：

```bash
python scripts/research/verify_scripted.py \
  --timeout-s 600 --output-dir outputs/research/scripted_all
```

先单独查一个任务可加 `--family PressButton` 并改用空的输出目录；给该单族测试加 `--video` 会保存额外 MP4，已在 PressButton 上实测。该脚本以每族 EE PID 作为任务逻辑基线；FAHexa BaseJoint PressButton 另有一条成功示范。四种物理飞行器的 PressButton PID 专家也均完成一条成功示范和 canonical validator，记录如下；这些单次结果不能推断其他任务的飞行专家成功率。

| PressButton 物理机型 | 成功示范数 | canonical 帧数（120 Hz） | recorder + validator |
| --- | ---: | ---: | --- |
| UAQuad PID | 1 | 1016 | 通过 |
| UAHexa PID | 1 | 986 | 通过 |
| FAHexa PID | 1 | 1002 | 通过 |
| OmniHexa PID | 1 | 997 | 通过 |

结果来自 `outputs/research/scripted-press-physical-20261007T1300/results.json`。2026-10-07 这轮补测沿用旧采集器的随机 seed（各 session 的 `env_cfg.yaml` 如实保存 `seed: null`）；从当前版本开始，验证脚本默认显式使用 `--seed 42`，报告记录 seed 与 episode 上限。使用公共采集器时也可加 `--seed 42` 固定任务随机化与专家轨迹：

```bash
python scripts/research/verify_scripted.py \
  --task-id PressButton-Am-UAQuad-Abs-PID-Direct-v0 \
  --task-id PressButton-Am-UAHexa-Abs-PID-Direct-v0 \
  --task-id PressButton-Am-FAHexa-Abs-PID-Direct-v0 \
  --task-id PressButton-Am-OmniHexa-Abs-PID-Direct-v0 \
  --seed 42 --timeout-s 180 --output-dir outputs/research/scripted_press_physical
```

容器内运行单任务示例：

```bash
python scripts/data/record_demos_scripted.py \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --dataset_root datasets/press_button_smoke \
  --repo_id am_bench/pressbutton_ee_absolute \
  --state_keys ee_pos ee_quat gripper_width \
  --task_prompt "press the button" \
  --step_hz 120 --num_envs 1 --num_demos 1 --env_length_s 20 --seed 42 \
  --camera_names ee_camera --video --headless --device cuda:0
```

脚本默认只保存成功 episode，`--num_demos 1` 的含义是一个成功 episode；失败的尝试不计数。调试失败可加 `--save_failed_episodes`，但这些 episode 不能算成功示范。录制成功后，在同一个容器终端定位最新 session 并用验证器检查：

```bash
SESSION_INFO=$(find datasets/press_button_smoke -type f -path '*/lerobot/meta/info.json' | sort | tail -n 1)
SESSION_ROOT=${SESSION_INFO%/lerobot/meta/info.json}
test -f "$SESSION_ROOT/lerobot/meta/info.json"
python scripts/data/validate_lerobotdataset.py \
  --dataset_root "$SESSION_ROOT" \
  --repo_id am_bench/pressbutton_ee_absolute --target_hz 20
```

下面保留 2026-09-30 的历史成功 PressButton 脚本示范画面，来自 EE 相机 MP4 第 237 帧，画面显示夹爪已靠近按钮。该 session 的一条 episode 和 validator 都通过；成功由环境终止条件判定，不靠图片推断。[附加图像清单](usage_assets/evidence_manifest.json)记录这个视频和上方 UAQuad 视频的 SHA-256、帧号及源报告。

![PressButton 脚本专家成功示范接近结束时的 EE 相机帧](usage_assets/press_button_scripted_success_near_final.png)

BaseJoint 数据应使用匹配 ID 和有序 state keys：

```bash
python scripts/data/record_demos_scripted.py \
  --task PressButton-Am-FAHexa-BaseJoint-Abs-PID-Direct-v0 \
  --dataset_root datasets/press_button_base_joint \
  --repo_id am_bench/press_button_base_joint_absolute \
  --state_keys base_pos base_quat arm_joint_pos gripper_width \
  --task_prompt "press the button" \
  --step_hz 120 --num_envs 1 --num_demos 1 --env_length_s 20 --seed 42 \
  --camera_names ee_camera \
  --headless --device cuda:0
```

这个 BaseJoint 配置还通过 `verify_scripted.py --task-id PressButton-Am-FAHexa-BaseJoint-Abs-PID-Direct-v0 --output-dir outputs/research/scripted-press-basejoint-20260930T2005` 做了完整采集和校验：1 条成功 episode、1019 帧、12 维 `observation.state` 与 `action`，`base_joint_absolute` 动作语义。独立结果不会计入上面的 12 族 EE PID 汇总。

保留首次失败及各次复测，核对每条成功示范的 recorder、validator、LeRobot metadata 和视频，在仿真容器内合并本次分批采集：

```bash
python scripts/research/merge_scripted_results.py \
  outputs/research/scripted-pressbutton-20260930T1850/results.json \
  outputs/research/scripted-other-families-20260930T1852/results.json \
  outputs/research/scripted-wipewindow-40s-20260930T1903/results.json \
  outputs/research/scripted-pressbutton-video-20260930T1930/results.json \
  outputs/research/scripted-wipe-fixed-success-20260930T1950/results.json \
  outputs/research/scripted-wipe-default20-20260930T2045/results.json \
  --output outputs/research/combined-final/scripted_12_default20.json \
  --expect-families 12
```

人工遥操作使用 `python scripts/data/record_demos_teleop.py --task PressButton-Am-EE-Abs-PID-Direct-v0 --help` 查看设备选项；它需要可见的 Isaac Sim 窗口和受支持输入设备。与脚本录制相同，遥操作输出也是 canonical LeRobot 数据。

### 4.1 数据格式与派生格式

```text
<session-root>/
  env_cfg.yaml
  lerobot/meta/info.json
  lerobot/meta/episodes/
  lerobot/data/
  lerobot/images/ 或 lerobot/videos/
  videos/  # 可选
```

源数据 LeRobot 0.4.4 的帧含 `observation.state`、`action`、`task` 与 `observation.images.<camera>`。`meta/info.json` 中 `ambench.action_semantics` 必须为 `ee_absolute` 或 `base_joint_absolute`，`state_keys` 记录 state 拼接顺序。模型训练使用相对目标时，由策略适配器或导出脚本从绝对动作源计算，源数据仍保留绝对语义。

把 canonical session 按 [note.md](note.md) 同步到远端 `/data/datasets/press_button_ee` 后，在**远端策略容器的 DP 环境**转换与验证。以下命令在远端容器内执行：

```bash
/opt/venvs/dp/bin/python scripts/data/dp/lerobot_to_zarr.py \
  --input_path /data/datasets/press_button_ee \
  --output_path /data/datasets/press_button_ee.zarr.zip --omit_base_image
/opt/venvs/dp/bin/python scripts/data/dp/validate_zarr.py \
  /data/datasets/press_button_ee.zarr.zip --image_size 224
```

OpenPI 固定 reader 的 v2.1 派生导出在远端策略容器的 ACT 环境执行，直接读取同一份已同步的 canonical session；实际验证生成了 169 个 20 Hz 逻辑帧：

```bash
/opt/venvs/act/bin/python scripts/data/export_lerobot_to_openpi.py \
  --dataset_roots /data/datasets/press_button_ee \
  --repo_id am_bench/multitask_openpi_original_20hz_ee_local_relative \
  --output_root /data/datasets/openpi/am_bench/multitask_openpi_original_20hz_ee_local_relative \
  --target_hz 20 \
  --omit_base_image --task_prompt "press the button"
```

多任务导出使用 `--task_prompt_map scripts/data/am_bench_language_instructions.json --require_task_prompt_map`，把各任务的 canonical session 一起传入。若已在本地仿真容器导出，也可用 `tools/research/sync_openpi_export.sh` 将 v2.1 repo 上传到远端相同布局；同步脚本拒绝覆盖已有目录。数据、配置和评估命令见下一节。

## 5. ACT、Diffusion Policy 与 OpenPI 的双容器闭环

远端策略 Docker 持有 checkpoint 和高层模型；本地 Isaac Docker 采集相机与状态、运行低层控制并保存评估。先按 [note.md](note.md) 部署两个镜像并建立 SSH 隧道。2026-10-07 重建使用新采集的 seed 42 PressButton canonical 数据；本节的训练各为 **1 步接口验证**，EE 评估 20 秒、BaseJoint 评估 0.5 秒。此前 2026-09-30 的 OpenPI 两步训练和旧目录属于历史记录。

以下路径使用空的新目录。再次复现时统一更换 `rebuild-20261007` 这个运行名，保留已有结果。当前的新模型结果待重建后门禁完成，更新到第 5.5 节。

### 5.0 同步本次 canonical 数据

在本地仿真容器分别采集并验证 EE 与 BaseJoint fixture，两个命令各执行一次：

```bash
python scripts/research/verify_scripted.py \
  --family PressButton --seed 42 --video --timeout-s 600 \
  --output-dir outputs/research/rebuild-20261007/scripted_press_ee
python scripts/research/verify_scripted.py \
  --task-id PressButton-Am-FAHexa-BaseJoint-Abs-PID-Direct-v0 \
  --seed 42 --video --timeout-s 600 \
  --output-dir outputs/research/rebuild-20261007/scripted_press_base_joint
```

本次 EE 得到 1 个成功 episode / 994 个 120 Hz 帧，state/action 为 8 维 `ee_absolute`；BaseJoint 为 1 个成功 episode / 999 帧、12 维 `base_joint_absolute`。两者均在默认 20 秒任务内成功，canonical validator 通过，`env_cfg.yaml` 记录 seed 42。从本地宿主机定位对应 session 并上传：

```bash
EE_INFO=$(find outputs/research/rebuild-20261007/scripted_press_ee -type f -path '*/lerobot/meta/info.json' | sort | tail -n 1)
BJ_INFO=$(find outputs/research/rebuild-20261007/scripted_press_base_joint -type f -path '*/lerobot/meta/info.json' | sort | tail -n 1)
EE_SESSION=${EE_INFO%/lerobot/meta/info.json}
BJ_SESSION=${BJ_INFO%/lerobot/meta/info.json}
test -f "$EE_SESSION/lerobot/meta/info.json"
test -f "$BJ_SESSION/lerobot/meta/info.json"
bash tools/research/sync_dataset.sh "$EE_SESSION" press_button_ee_rebuild_20261007
bash tools/research/sync_dataset.sh "$BJ_SESSION" press_button_base_joint_rebuild_20261007
```

上传脚本拒绝覆盖已有远端目录。后续训练与导出命令均在**远端策略容器**执行：

```bash
ssh -t tencent-86 'docker exec -it -w /workspace/ambench ambench-policy-research bash'
```

### 5.1 ACT

ACT 直接读取 canonical LeRobot，在训练时从绝对源计算相对动作。EE 使用以下固定的 20 Hz、一训练步配置：

```bash
/opt/venvs/act/bin/python -m ambench_learn.policies.act.train \
  --dataset.repo_id=am_bench/pressbutton_ee_absolute \
  --dataset.root=/data/datasets/press_button_ee_rebuild_20261007/lerobot \
  --dataset.use_imagenet_stats=true \
  --policy.type=act --policy.chunk_size=16 --policy.n_action_steps=8 \
  --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=/data/checkpoints/rebuild-20261007/act_press_button_smoke \
  --job_name=press_button_ee_act_rebuild --batch_size=8 --steps=1 --seed=42 \
  --num_workers=0 --save_freq=1 --wandb.enable=false --policy_target_hz=20 \
  --policy_action_representation=ee_local_relative
```

BaseJoint 的命令替换四项：

| 参数 | BaseJoint 值 |
| --- | --- |
| `--dataset.repo_id` | `am_bench/press_button_base_joint_absolute` |
| `--dataset.root` | `/data/datasets/press_button_base_joint_rebuild_20261007/lerobot` |
| `--output_dir` | `/data/checkpoints/rebuild-20261007/act_press_button_base_joint_smoke` |
| `--policy_action_representation` | `base_joint_relative` |

保持其余参数不变并用独立 job name。`benchmark_dataset_report.json` 记录实际逻辑帧数：EE 为 `994 // 6 = 165`，BaseJoint 为 `999 // 6 = 166`；不足一个完整采样间隔的尾部帧不进入训练。ACT 相对动作不能跨观测锚点做 temporal ensembling。

### 5.2 Diffusion Policy

在远端容器从相同 canonical session 导出 UMI zarr 并验证：

```bash
/opt/venvs/dp/bin/python scripts/data/dp/lerobot_to_zarr.py \
  --input_path /data/datasets/press_button_ee_rebuild_20261007 \
  --output_path /data/datasets/press_button_ee_rebuild_20261007.zarr.zip --omit_base_image
/opt/venvs/dp/bin/python scripts/data/dp/validate_zarr.py \
  /data/datasets/press_button_ee_rebuild_20261007.zarr.zip --image_size 224
```

BaseJoint 导出使用同样命令，将两个 `press_button_ee_rebuild_20261007` 改为 `press_button_base_joint_rebuild_20261007`。下面一步训练使用随机初始化的 ResNet18，不下载默认视觉编码器权重；checkpoint 保存完整 Hydra config：

```bash
cd source/ambench_learn/ambench_learn/policies/dp/universal_manipulation_interface
WANDB_MODE=disabled /opt/venvs/dp/bin/python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task=umi_drone_ee_pos \
  task.dataset_path=/data/datasets/press_button_ee_rebuild_20261007.zarr.zip \
  task.obs_down_sample_steps=6 task.action_horizon=16 \
  task.pose_repr.obs_pose_repr=relative task.pose_repr.action_pose_repr=relative \
  policy.obs_encoder.model_name=resnet18 policy.obs_encoder.pretrained=false \
  policy.obs_encoder.feature_aggregation=avg policy.obs_encoder.transforms=null \
  'policy.down_dims=[64,128]' policy.diffusion_step_embed_dim=64 \
  policy.num_inference_steps=4 policy.noise_scheduler.num_train_timesteps=8 \
  training.num_epochs=1 training.max_train_steps=1 training.max_val_steps=1 \
  training.device=cuda:0 training.seed=42 \
  dataloader.batch_size=1 dataloader.num_workers=0 dataloader.persistent_workers=false \
  val_dataloader.batch_size=1 val_dataloader.num_workers=0 val_dataloader.persistent_workers=false \
  exp_name=press_button_dp_ee_rebuild logging.name=press_button_dp_ee_rebuild \
  logging.mode=disabled hydra.run.dir=/data/outputs/rebuild-20261007/press_button_dp_smoke
```

BaseJoint 使用 `task=umi_drone_base_joint`、对应 BaseJoint zarr、独立的 exp/logging name 和 `hydra.run.dir=/data/outputs/rebuild-20261007/press_button_dp_base_joint_smoke`。只有一个示范 episode 时，验证集为空；一步训练检查数据、优化和 checkpoint 保存。DP evaluator 当前每次运行一个环境。

### 5.3 OpenPI π₀ / π₀.₅

OpenPI 使用 `ext/openpi` 的固定版本和独立 Python 环境。两个模型共享 canonical 源；训练前分别准备配置、norm stats 和官方 base 权重。以下两个配置对应 EE：

| 模型 | 配置名 |
| --- | --- |
| π₀ | `pi0_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative` |
| π₀.₅ | `pi05_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative` |

先从**本地宿主机**执行校验和同步脚本；下载与文件校验在无 GPU 策略 Docker 内执行，已有有效缓存可复用：

```bash
bash tools/research/fetch_openpi_base.sh pi0
bash tools/research/fetch_openpi_base.sh pi05
```

远端策略容器的 ACT 环境将同一份 canonical 数据导出成固定 reader 的 LeRobot v2.1。此次使用独立的 dataset home 保存新导出，同时保留配置要求的 repo ID：

```bash
cd /workspace/ambench
/opt/venvs/act/bin/python scripts/data/export_lerobot_to_openpi.py \
  --dataset_roots /data/datasets/press_button_ee_rebuild_20261007 \
  --repo_id am_bench/multitask_openpi_original_20hz_ee_local_relative \
  --output_root /data/datasets/openpi-rebuild-20261007/am_bench/multitask_openpi_original_20hz_ee_local_relative \
  --target_hz 20 --omit_base_image --task_prompt 'press the button'
```

BaseJoint 替换源为 `/data/datasets/press_button_base_joint_rebuild_20261007`，repo ID 和 output root 末尾替换为 `am_bench/multitask_base_joint_openpi_original_20hz_base_joint_relative`。多任务导出把各任务 canonical session 传给 `--dataset_roots`，并用 `--task_prompt_map scripts/data/am_bench_language_instructions.json --require_task_prompt_map` 保持任务语言映射。

以下以 π₀.₅ EE 为例，转换在 CPU 上执行，norm stats 和训练读取新的 dataset home。运行 π₀ 时将开头的 `pi05` 改为 `pi0`：

```bash
cd /opt/openpi
export HF_LEROBOT_HOME=/data/datasets/openpi-rebuild-20261007
export JAX_PLATFORMS=cpu
PI_CONFIG=pi05_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative
BASE_NAME=pi05_base
/opt/openpi/.venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir "/data/cache/openpi/openpi-assets/checkpoints/$BASE_NAME" \
  --config-name "$PI_CONFIG" \
  --output-path "/data/checkpoints/openpi-rebuild-20261007/${BASE_NAME}_pytorch"
/opt/openpi/.venv/bin/python -m scripts.compute_norm_stats \
  --config-name "$PI_CONFIG" --assets-base-dir /data/assets/rebuild-20261007 \
  --repo-id am_bench/multitask_openpi_original_20hz_ee_local_relative --num-workers 2
/opt/openpi/.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m scripts.train_pytorch "$PI_CONFIG" --exp-name rebuild-20261007 \
  --pytorch-weight-path "/data/checkpoints/openpi-rebuild-20261007/${BASE_NAME}_pytorch" \
  --assets-base-dir /data/assets/rebuild-20261007 \
  --checkpoint-base-dir /data/checkpoints/openpi-rebuild-20261007 \
  --data.repo-id am_bench/multitask_openpi_original_20hz_ee_local_relative \
  --batch-size 1 --num-train-steps 1 --num-workers 0 --seed 42 \
  --save-interval 1 --no-resume --no-overwrite --no-wandb-enabled
```

`--pytorch-weight-path` 指向包含 `model.safetensors` 的目录。BaseJoint 使用 `pi0_am_bench_multitask_base_joint_openpi_original_20hz_h50_base_joint_relative` 或对应 `pi05_...` 配置、BaseJoint repo ID，并单独计算 norm stats 和训练；两个模型可复用各自已经转换的 base PyTorch 目录。数据 repo ID、配置、stats 和 checkpoint 必须匹配。

### 5.4 启动真实模型服务并运行 Isaac 闭环

从本地宿主机启动服务，等待健康检查完成。`start_policy.sh` 顶部显式配置 `INFERENCE_SEED=42`，在远端 Python 中设置 random、NumPy 和 Torch 的随机种子，日志输出 `REMOTE_INFERENCE_SEED=42`。每次启动会停止前一个高层服务；重启恢复远端随机推理的起点。ACT/DP 用 HTTP 8001，OpenPI 用 WebSocket 8000；SSH 隧道只向本地 loopback 转发。

先以 EE ACT 为例检查真实权重调用，再重新启动同一服务开始固定 seed 的评估：

```bash
bash tools/research/start_policy.sh act \
  /data/checkpoints/rebuild-20261007/act_press_button_smoke/checkpoints/last/pretrained_model
bash tools/research/wait_policy.sh act
docker exec ambench-sim-research python \
  source/ambench_learn/tests/policies/remote/smoke_remote_transport.py \
  --policy act --remote-url http://127.0.0.1:8001

bash tools/research/start_policy.sh act \
  /data/checkpoints/rebuild-20261007/act_press_button_smoke/checkpoints/last/pretrained_model
bash tools/research/wait_policy.sh act
docker exec ambench-sim-research python -m ambench_learn.policies.act.eval \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --remote-url http://127.0.0.1:8001 --policy-id act_ee_rebuild_step1 \
  --num-rollouts 1 --num-envs 1 --seed 42 --n-action-steps 8 --policy-target-hz 20 \
  --episode-length-s 20 --output-dir outputs/research/rebuild-20261007/policy_rpc/act_isaac_full20s \
  --save-video --video-camera-names ee_camera --progress-every 240 --headless --device cuda:0
```

DP 对应的真实 checkpoint 与评估如下；可在评估前用 `smoke_remote_transport.py --policy dp` 做 HTTP 前向，再重启同一服务：

```bash
bash tools/research/start_policy.sh dp \
  /data/outputs/rebuild-20261007/press_button_dp_smoke/checkpoints/latest.ckpt
bash tools/research/wait_policy.sh dp
docker exec ambench-sim-research python -m ambench_learn.policies.dp.eval \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --remote-url http://127.0.0.1:8001 --policy-id dp_ee_rebuild_step1 \
  --num-rollouts 1 --num-envs 1 --seed 42 --episode-length-s 20 \
  --output-dir outputs/research/rebuild-20261007/policy_rpc/dp_isaac_full20s \
  --save-video --video-camera-names ee_camera --progress-every 240 --headless --device cuda:0
```

OpenPI 从训练生成的一步目录加载；把下面 `pi05` 改为 `pi0`，即可运行另一模型。WebSocket 前向检查用 `smoke_openpi_transport.py --host 127.0.0.1 --port 8000`，随后重启服务再评估：

```bash
PI_CONFIG=pi05_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative
bash tools/research/start_policy.sh pi "$PI_CONFIG" \
  "/data/checkpoints/openpi-rebuild-20261007/$PI_CONFIG/rebuild-20261007/1"
bash tools/research/wait_policy.sh pi
docker exec ambench-sim-research python -m ambench_learn.policies.pi.eval \
  --host 127.0.0.1 --port 8000 --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --prompt 'press the button' --policy-id pi05_ee_rebuild_step1 \
  --num-rollouts 1 --num-envs 1 --seed 42 --n-action-steps 8 --policy-target-hz 20 \
  --episode-length-s 20 --output-dir outputs/research/rebuild-20261007/policy_rpc/pi05_isaac_full20s \
  --save-video --video-camera-names ee_camera --progress-every 240 --headless --device cuda:0
```

BaseJoint 分别启动下表的真实一步权重，再用相应 `eval` 命令。共同替换环境为 `PressButton-Am-FAHexa-BaseJoint-Abs-PID-Direct-v0`，设置 `--seed 42 --episode-length-s 0.5 --progress-every 10`，保持相机录像和单环境。输出目录使用 `outputs/research/rebuild-20261007/policy_rpc/<model>_base_joint_isaac_step1`，policy ID 使用 `<model>_base_joint_rebuild_step1`：

| 模型 | BaseJoint checkpoint |
| --- | --- |
| ACT | `/data/checkpoints/rebuild-20261007/act_press_button_base_joint_smoke/checkpoints/last/pretrained_model` |
| DP | `/data/outputs/rebuild-20261007/press_button_dp_base_joint_smoke/checkpoints/latest.ckpt` |
| π₀ | `/data/checkpoints/openpi-rebuild-20261007/pi0_am_bench_multitask_base_joint_openpi_original_20hz_h50_base_joint_relative/rebuild-20261007/1` |
| π₀.₅ | `/data/checkpoints/openpi-rebuild-20261007/pi05_am_bench_multitask_base_joint_openpi_original_20hz_h50_base_joint_relative/rebuild-20261007/1` |

OpenPI 的 BaseJoint 服务配置也替换成表内 checkpoint 的配置名。模型名在路径中分别为 `act`、`dp`、`pi0`、`pi05`。HTTP/WS 前向检查的 BaseJoint 参数为 `--action-semantics base_joint_absolute`。

### 5.5 本次重建的策略结果

待新镜像的数据、训练和八项闭环门禁完成后填写。成功完成一个 rollout 证明跨容器模型接口可用；一步训练与单条示范不能作为论文性能复现。

生产镜像没有开发测试工具。以下命令在远端策略 Docker 内安装固定版本 pytest，再核对动作变换、数据导出、评估记录与协议；测试依赖只写入容器：

```bash
uv pip install --python /opt/venvs/act/bin/python pytest==8.4.2
uv pip install --python /opt/venvs/dp/bin/python pytest==8.4.2
uv pip install --python /opt/openpi/.venv/bin/python pytest==8.4.2 pynvml==13.0.1 nvidia-ml-py==13.590.48
cd /workspace/ambench
/opt/venvs/act/bin/python -m pytest -q \
  source/ambench_learn/tests/data/test_action_contract.py \
  source/ambench_learn/tests/data/test_openpi_export.py \
  source/ambench_learn/tests/policies/act/test_act_eval_utils.py \
  source/ambench_learn/tests/policies/act/test_se_relative_processor.py \
  source/ambench_learn/tests/policies/pi/test_pi_eval_utils.py \
  source/ambench_learn/tests/eval/test_eval_common.py \
  source/ambench_learn/tests/eval/test_eval_run.py \
  source/ambench_learn/tests/policies/remote/test_protocol.py
/opt/venvs/dp/bin/python -m pytest -q \
  source/ambench_learn/tests/data/test_dp_zarr_export.py \
  source/ambench_learn/tests/policies/dp/test_dp_eval_utils.py \
  source/ambench_learn/tests/policies/remote/test_protocol.py
cd /opt/openpi
JAX_PLATFORMS=cpu /opt/openpi/.venv/bin/python -m pytest -q src/openpi/policies/am_bench_policy_test.py
```

## 6. 评估记录与可视化

三个公共 evaluator 均保存解析后的完整 `env_cfg.yaml`、`results.txt`、`eval_summary.json`、tracking JSONL 和 `tracking/analysis.json`。ACT/DP 的远端 checkpoint 路径记录在 summary 的 `metadata.Server metadata`；OpenPI 的来源按服务配置、实际启动参数和 checkpoint 哈希核对。所有本次公开命令显式指定 seed 42。

验收一个评估目录时，同时检查 `status: "completed"`、rollout 数、步骤、环境 ID、seed、动作语义、服务来源和 checkpoint。保存录像时用 `--save-video --video-camera-names ee_camera`，还要实际解码 MP4 并确认帧数大于零。进程 exit 0 而缺少 summary 不算通过；本次曾遇到长期运行的旧容器失去 GPU 访问，恢复方法见 [note.md](note.md)。

EE 20 秒评估与 BaseJoint 0.5 秒评估分别记录。任务成功以环境的终止条件判定，视频仅提供画面证据；`completed` 本身不能推断任务成功。原始录像、checkpoint、示范和日志保持在忽略目录；经核验的代表帧及小型结果摘要放到 `usage_assets/`，并记录出处与哈希。

## 7. 本分支验证记录

下表在每次复现后按实际输出填写。`未执行` 是明确状态，不代表失败或通过。在**本地主机的仓库根目录**用下面的纯标准库脚本从忽略目录中的原始报告重新生成[验证快照](usage_assets/validation_snapshot.json)；它要求 106 个环境、12 族专家和四个真实 20 秒策略 episode 均齐全：

```bash
python3 scripts/research/snapshot_validation.py \
  --matrix outputs/research/combined-final/verified_106_after_wipe_fix.json \
  --scripted outputs/research/combined-final/scripted_12_default20.json \
  --policy-root outputs/policy_rpc \
  --require-base-joint \
  --output usage_assets/validation_snapshot.json
```

| 项目 | 状态 | 证据或原因 |
| --- | --- | --- |
| 仿真镜像构建与依赖导入 | 通过 | 最终镜像 `sha256:f6b7d491…` 已运行；Python 3.11.13、NumPy 1.26.0、Torch 2.7.0+cu128、CUDA 可用、LeRobot 0.4.4 与 accelerate 1.12 可导入，acados `t_renderer` 可执行 |
| 策略镜像构建与远端 GPU | 通过 | 远端容器 `ambench-policy-research` 仅看见 GPU 0 RTX PRO 5000 72 GB；8000/8001 仅绑定服务器 `127.0.0.1` |
| Gym registry | 通过 | 2026-09-30，`outputs/research/registry-smoke-20260930T1740/registry.json`：106 ID、12 族 |
| PressButton EE PID 场景与相机 | 通过 | `outputs/research/press-ee-video-20260930T1744/results.json`：8/8 步，384×384 MP4 与上图 |
| UAQuad base_camera 与 FAHexa 扰动 | 通过 | UAQuad `outputs/research/press-uaquad-base-camera-20260930T1933/results.json`：30/30 步、base_camera MP4；FAHexa `press-disturbance-20260930T1932/results.json`：1 N 风、饱和、气动、动作/观察噪声下 8/8 步 |
| ACT/DP 随机权重跨机模型推理 | 通过（接口烟测） | `outputs/policy_rpc/`：远端 GPU 加载两种真实模型；本地仿真 Docker 经 SSH tunnel 取得有限 8D 动作及 `(8, 8)` 计划；随机权重无任务能力 |
| OpenPI 配置与 WebSocket | 通过（π₀、π₀.₅ 真实模型接口） | 远端 adapter 17/17；两个官方 base 分别为 33 文件/12,014,440,489 B 与 29 文件/12,441,749,581 B，经本地及远端 GCS 校验、转换与各自两步训练；本地仿真 Docker 经隧道分别收到真实 checkpoint 的 `(50, 8)` 动作序列，见 `outputs/policy_rpc/pi0_trained_remote_ws.log`、`pi05_trained_remote_ws.log` |
| 106/106 注册与 reset/step | 通过（首轮及针对性复测） | `outputs/research/combined-final/verified_106_after_wipe_fix.json`：逐 ID 核对 child JSON、步数及来源，106/106、12 族；首轮 103/106，MPC/NDT 三项复测 3/3，WipeWindow 修复后 8/8 重新通过；12 类 EE 相机录像及上方图集已生成 |
| 12/12 脚本专家成功示范 | 通过（每族 EE PID 一条） | PressButton `outputs/research/scripted-pressbutton-20260930T1850/results.json`：1 episode/1018 帧；`scripted-other-families-20260930T1852` 的另外 10 类通过；WipeWindow 默认 20 s 修正后见 `scripted-wipe-default20-20260930T2045/results.json`：1 episode/1747 帧。全部 12 个 session 均通过 LeRobot validator；合并报告为 `scripted_12_default20.json` |
| 四种物理机型 PressButton 专家 | 通过（每机型一条） | `outputs/research/scripted-press-physical-20261007T1300/results.json`：UAQuad1016、UAHexa986、FAHexa1002、OmniHexa997 帧；四条成功 episode 均通过 canonical validator。旧采集器 seed未固定，不能推断多任务成功率 |
| FAHexa BaseJoint 专家示范 | 通过（单任务） | `outputs/research/scripted-press-basejoint-20260930T2005/results.json`：PressButton 1 条成功 episode/1019 帧，12 维 state/action，`base_joint_absolute`，LeRobot validator exit0 |
| canonical LeRobot + DP/OpenPI 派生格式 | 通过（单任务数据链） | PressButton canonical LeRobot 0.4.4 已通过 validator 并同步远端，`info.json` 两端 SHA-256 一致；DP zarr 导出、validator 1 episode/1018 steps 通过；OpenPI v2.1 导出 169 个 20 Hz 逻辑帧，π₀/π₀.₅ 两个 config 的 norm stats 均生成。见 `outputs/policy_rpc/dp_convert.log`、`dp_validate.log`、`openpi_export.log`、`openpi_norm_pi0.log`、`openpi_norm_pi05.log` |
| ACT 一步训练与跨机调用 | 通过（模型接口） | 远端从 PressButton canonical 的 169 个 20 Hz 帧训练 1 步，checkpoint `/data/checkpoints/act_press_button_smoke/checkpoints/last/pretrained_model`；训练及本地 Docker → 远端真实 checkpoint HTTP 见 `outputs/policy_rpc/act_train.log`、`act_trained_remote_http.log` |
| DP 一步训练与跨机调用 | 通过（模型接口） | 远端从 PressButton zarr 训练 1 步，checkpoint `/data/outputs/press_button_dp_smoke/checkpoints/latest.ckpt`；训练及本地 Docker → 远端真实 checkpoint HTTP 见 `outputs/policy_rpc/dp_train.log`、`dp_trained_remote_http.log` |
| ACT / DP Isaac 闭环 | 通过（任务成功均 0/1） | 0.5 秒录像闭环见 `outputs/policy_rpc/act_isaac_step1/eval_summary.json`、`dp_isaac_step1/eval_summary.json`：均 `completed`、59 步、0/1。默认 20 秒 episode 见 `outputs/policy_rpc/act_isaac_full20s/eval_summary.json`、`dp_isaac_full20s/eval_summary.json`：均 `completed`、2399 步、0/1；远端服务日志各有 400 次 `/infer` 请求，ACT 耗时 103.5 s、DP 63 s。一步训练 checkpoint 验证完整跨机链路，不代表有效任务成绩 |
| OpenPI π₀ / π₀.₅ 远端闭环 | 通过（20 秒闭环，任务成功均 0/1） | 两个官方 base 均经本地及远端 GCS 校验、转成 PyTorch、从同一 canonical 派生数据各训练 2 步；本地 Docker 经 WebSocket 分别收到真实 checkpoint 的 `(50, 8)` 动作块。两个 0.5 秒 Isaac 闭环均 `completed`、59 步、0/1 且有 MP4；`outputs/policy_rpc/pi0_isaac_full20s/eval_summary.json` 与 `pi05_isaac_full20s/eval_summary.json` 均 `completed`、2399 步、0/1 |
| FAHexa BaseJoint 四种高层模型 | 通过（0.5 秒闭环，任务成功均 0/1） | ACT、DP、π₀、π₀.₅ 从同一 12 维成功示范各训练 1 步，HTTP/WebSocket 返回有限 12 维动作；`outputs/policy_rpc/{act,dp,pi0,pi05}_base_joint_isaac_step1/eval_summary.json` 均 `completed`、59 步、0/1 且有 MP4。π₀.₅ 闭环于 2026-10-07 补齐 |

公开文档提供的是代码与运行方法，仓库没有打包训练数据和任务 checkpoint。任何未完成的阶段都会保留真实状态及直接日志，不推断模型精度或任务成功率。
