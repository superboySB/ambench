# AM-Bench 双容器研究环境

这份手册在 `research` 分支使用 Isaac Lab 2.3.2 / Isaac Sim 5.1.0，把仿真、机器人控制与脚本专家放在本地仿真容器，把 ACT、Diffusion Policy 和 OpenPI 的模型推理放在独立策略容器。默认部署策略容器到 `tencent-86`，本地和远端均只通过 Docker 安装项目依赖。[usage.md](usage.md) 给出功能清单、逐项命令和验证记录；[在线实验视频总览](https://zipengdai.com/ambench/)支持直接播放、暂停、拖动、调速，并按任务、飞机和模型查看 seed 与视角对照。成功、超时和场景烟测逐段标明。网络不可用时，下载或克隆仓库后用浏览器打开[离线播放页](usage_assets/playback.html)，同时保留其 `animations/` 目录。重新录制、导出和发布步骤见 usage.md 第 6.1–6.3 节。

命令中的 `rebuild-20261007` 是 **2026-10-07 历史验收**的运行名，对应一步训练与接口闭环。当前充分训练结果另见 [usage.md 的当前结果](usage.md#当前正式训练结果2026-10-15)：UAQuad PressButton 的 ACT 完成 40,000 次更新，独立最终测试 0/30；DP 完成 40 epochs / 46,480 次更新，选中 epoch 16 / 18,592 次更新的 EMA 权重，独立最终测试 30/30。[新实验播放页](https://zipengdai.com/ambench/trained/)和[证据快照](usage_assets/trained/validation.json)与旧录像分别保存；仓库内有[本地页面副本](usage_assets/trained/index.html)，新视频使用外链，播放时仍需网络。

当前机器已有历史示范和 checkpoint，可以核验并复用；需要重新采集或训练时，统一替换所有本地输出、远端数据、stats、checkpoint 和评估路径中的运行名，使用空目录。验证采集入口拒绝非空输出目录，上传入口拒绝同名远端目录；训练也应使用新的目录。

已完成训练的 UAQuad PressButton ACT/DP 选中权重现可[公开下载并重放](trained_models.md)，包含各自的配置、归一化、来源与 SHA；无需先重新训练这两个模型。模型 ZIP 通过 Release 提供，已匿名完整下载验真，不能用于恢复优化器训练。

2026-10-15 按当前挂载源码完成 **106/106** 注册/reset/8 步 smoke、**126 项 CPU 回归**和 **12/12** 历史 canonical 重验；另以已选 DP epoch 16 权重在新 seed 34002 首次闭环成功（1055 步、约 8.792 秒），单独记录，不合入原最终测试 30/30。此前 34001 的原生成功与含 reset 尾帧的视频保留；录像修复后 34002 三路各 264 帧完整解码、末帧接触画面检查通过。上述复测批次复用已有仿真镜像 `sha256:09bb607e…`，其烘焙源码仍为 `02d1499`；源码来自挂载，**上述复测批次没有重建镜像或重新训练模型**。在线新实验的 686 附件匿名访问、选定视频全解码与实际浏览器播放已验收；CDN 返回 `application/octet-stream`，不宣称媒体 MIME 匹配。逐项来源、哈希与验证边界见 [usage.md 本轮复测](usage.md#本轮逐步复测2026-10-15)和[快照](usage_assets/reproduction_20261015.json)。

[![本轮DP seed34002按钮成功过程，录像末帧保持接触](https://github.com/superboySB/ambench/releases/download/research-media-20261015-uaquad-replay/uaquad_dp_replay_seed34002_ee_camera.gif)](https://github.com/superboySB/ambench/releases/download/research-media-20261015-uaquad-replay/uaquad_dp_replay_seed34002_ee_camera.mp4)

上图是本轮新试次的采样预览，点击播放 MP4；[重放来源清单](usage_assets/trained/replay.json)保留两个 seed 的全部原片与录制版本，两个试次不合并为新成功率。

**2026-10-15 追加长训练阶段：** 用户已授权全部 **18 次正式训练**，范围、完整预算、80/20 划分和选模/最终测试种子见[公开预声明](usage_assets/extensions_20261015_plan.json)。UAQuad WipeWindow ACT/DP 已在远端 GPU 4/6 启动；OpenPI 的旧 exit 139 环境问题已通过独立探针定位；新的单卡 full-batch 显式 OOM 单独保留。双卡 FSDP2 已通过完整模型真实更新，UAQuad 多任务正式长训已启动。当前进展以真实训练状态与验收记录为准，尚不报告这批训练完成或新成功率；状态详见 [usage.md](usage.md#当前正式训练结果2026-10-15)。

Omni Wipe 的原 52001 只清除 2/4 污点；仅加长安装点的 gen02 在 52011 仍失败，另发现工具位置计算遗漏 authored USD scale。采集前预声明的 gen03 保留专属 authored EE 局部坐标中的 `(0, 0, 0.15)` 安装偏移，并用实际 USD scale 修正通用坐标计算，明确绑定两个源码文件 SHA；新数据及评估统一使用 gen03，52021/52022 仅作新预检。原生阈值、UAQuad 物理版本与旧失败证据保留，详见预声明的 `fixture_amendments`。

### 先选择复现版本

`research` 会继续变化，Compose 挂载和远端同步会使用当前 checkout；旧 checkpoint 路径和旧镜像标签不能单独固定运行源码。重放历史实验时，从下表指定提交建立独立 worktree，并在该 worktree 中使用相应手册、子模块、镜像依赖和保存的配置。不要切换正在被训练或评估容器挂载的 checkout。

| 记录 | 固定源码 | 范围 |
| --- | --- | --- |
| 2026-10-07 通用接口基线 | `02d14993b717509d0dad940d4695689aad70f9a3` | 106-ID 矩阵、EE/BaseJoint 一步模型 |
| 2026-10-09 UAQuad 专题 | `042e2f2ec079505ab9a386634c419d9ce564a482` | 历史专家、迁移与一步模型 |
| 2026-10-09 OmniHexa 专题 | `aea900e4000cb85d3e8f774108552ea2bb1b6282` | 历史主动倾转专题；来源文件哈希另见 training provenance |
| 2026-10-13–15 充分训练 | [逐模型快照](usage_assets/trained/validation.json) | right-v2 动作坐标、reset 相机重绘、独立选模和最终测试 |

例如，以下命令只准备历史源码，不启动或替换当前容器：

```bash
git worktree add --detach ../ambench-replay-20261007 02d14993b717509d0dad940d4695689aad70f9a3
cd ../ambench-replay-20261007
git submodule update --init --recursive ext/pyroki ext/acados ext/openpi
```

随后按这个 worktree 内的 `note.md` 构建和启动；历史容器与当前容器使用不同名称、输出目录和源码挂载。重编码已有录像只需原始文件与导出配置；重新运行 Isaac 还需要匹配源码、权重、归一化、seed 和 `env_cfg.yaml`，不承诺视频字节逐一相同。

> 本分支的容器流程是研究验证路径。主分支 README 所述的原生安装仍是项目目前的维护路径。

## 1. 机器与目录

| 角色 | 设备 | 容器 | 持久目录 |
| --- | --- | --- | --- |
| 仿真、低层控制、数据采集 | 本地 RTX 4070 Ti SUPER 16 GiB | `ambench-sim-research` | 仓库的 `datasets/`、`outputs/`、`videos/`（均不提交） |
| 高层策略训练与推理 | `tencent-86` 上空闲编号最小的 GPU；无空闲 GPU 时可在策略 Docker 内用 CPU 推理 | `ambench-policy-research` | `/diff/dzp_is_sb/ambench-research` |

```mermaid
flowchart LR
  subgraph Local[本地仿真 Docker]
    Isaac[Isaac Sim / Isaac Lab]
    Low[PID / L1 / MPC 与 IK]
    Expert[十二类脚本专家]
    Isaac --> Low
    Expert --> Low
  end
  Data[Canonical LeRobot 数据]
  subgraph Remote[远端策略 Docker：空闲 GPU 或 CPU]
    Model[ACT / DP / π₀ / π₀.₅]
    Train[数据转换与训练]
    Train --> Model
  end
  Isaac --> Data
  Data -->|rsync| Train
  Isaac -->|观察：SSH 隧道| Model
  Model -->|动作：HTTP / WebSocket| Low
  Low --> Isaac
```

本机已经把现有 `~/.ssh/id_ed25519.pub` 安装到 `tencent-86`，可用下面的只读检查确认免密登录。换一台本地机器时，先准备自己的公钥并执行 `ssh-copy-id -i ~/.ssh/id_ed25519.pub -p 22 tencent-86`，输入一次该服务器的登录凭据，随后再运行检查命令。不要把密码、私钥、Hugging Face token 或模型文件写进 Git。

```bash
ssh -o BatchMode=yes -p 22 tencent-86 'id -un; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
docker compose version
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L
```

本次连续启动服务时遇到过短 SSH 连接偶发中断，已为本机现有 `tencent-86` 配置连接复用，并通过禁用密码的 BatchMode 检查。其他机器可在已有的 `Host tencent-86` 配置下加入以下三项；保留原来的 HostName、User、IdentityFile 等设置：

```sshconfig
Host tencent-86
  ControlMaster auto
  ControlPersist 30m
  ControlPath ~/.ssh/ambench-%C
```

远端 GPU 编号会随他人的任务变化。每次启动容器前重新执行 `nvidia-smi`；部署脚本默认选取没有计算进程的最小编号，可给脚本传入明确的 GPU ID。不要在已有任务的卡上叠加模型进程。

## 2. 获取源码和构建镜像

首次获取本研究分支时执行：

```bash
git clone --branch research --recurse-submodules https://github.com/superboySB/ambench.git ambench-research
cd ambench-research
```

已有 checkout 时跳过克隆，从仓库根目录执行下面的命令。构建上下文包含固定 Git 子模块版本；`ext/` 的生成物仍留在容器层和忽略目录中。

```bash
git switch research
git submodule update --init --recursive ext/pyroki ext/acados ext/openpi
bash tools/research/build_images.sh sim
bash tools/research/build_images.sh policy
```

镜像分别是 `ambench:research-sim` 和 `ambench:research-policy`。仿真镜像基于 NVIDIA 官方 `nvcr.io/nvidia/isaac-lab:2.3.2`，其配套 Isaac Sim 为 5.1.0，Python 为 3.11，Torch 为 2.7.0+cu128。第一次拉取镜像较大，应留出磁盘空间。构建只在容器内安装 Pyroki、acados、AM-Bench 和策略依赖；宿主机不需要安装 Conda、Isaac Lab Python 包或模型包。

2026-10-07 重建的最终两张镜像约为 18.9 GB 和 32.4 GB；策略镜像传输包约 15.1 GB。构建机还需要 Docker 层、下载缓存和该传输包的临时空间；远端需要镜像与传输包的空间。传输包放在本机 `~/.cache/ambench-research/` 和远端研究目录的 `images/`，均不属于 Git 仓库。

2026-10-08 为多视角录制与 CPU 推理改动再次构建本地两张镜像，源码为 `85d5957`、构建时工作区干净，大小约 18.9 GB 和 32.9 GB。脱离仓库挂载、禁用 GPU 与网络的 12 项导入/CLI/内置源码检查全部通过，仿真镜像也包含默认的 21 条录制配置。详见[多视角验收快照](usage_assets/multiview_validation_snapshot.json)的 `local_docker_refresh`。这一轮未重新上传远端镜像；远端 CPU 实测使用既有策略容器及更新后的源码挂载。该检查不代替前述完整 GPU 仿真与闭环实验记录。

OpenPI 的 π₀ 和 π₀.₅ 官方 base checkpoint 各约 11–12 GiB。`fetch_openpi_base.sh` 只在本地无 GPU Docker 中下载到忽略目录 `outputs/openpi-cache/`，校验后用 rsync 传到远端持久缓存并再次校验；同时使用两个模型时，本地和远端分别预留至少 25 GiB 权重空间。

高层镜像内部使用三个互不影响的 Python 环境：`/opt/venvs/act/bin/python`、`/opt/venvs/dp/bin/python`、`/opt/openpi/.venv/bin/python`。ACT 的 LeRobot 0.4.4 依赖与本仓 DP 固定的 diffusers 版本不兼容；OpenPI 也有独立的锁定依赖。三者仍属于**一个**远端 Docker 镜像。

`docker/constraints.sim.txt`、`docker/constraints.act.txt` 和 `docker/constraints.dp.txt` 固定已验证的 Python 运行依赖；OpenPI 使用其子模块的 `uv.lock`。Pyroki 的 JAXLS Git 依赖也固定到成功环境的 commit，未修改第三方子模块。仿真继续采用官方安装脚本要求的 JAX/JAXLIB 0.4.28，即使 JAXLS 元数据声明了更新的版本范围；IK 的实际运行以全量矩阵验证为准。

## 3. 启动仿真容器

```bash
mkdir -p datasets outputs videos
docker compose -f docker/compose.sim.yml up -d
docker exec ambench-sim-research true
```

进入容器：

```bash
docker exec -it ambench-sim-research bash
```

下文中 `SIM='docker exec ambench-sim-research'` 可简化宿主机命令。仿真容器默认工作目录是可写 Docker 卷 `/workspace/ambench-run`：其中 `scripts/`、`source/`、`datasets/`、`outputs/` 等链接到 `/workspace/ambench` 的对应目录。仓库源码仍只读，acados 的相对路径生成物留在工作卷；直接执行 `docker exec ... python scripts/...` 或 `python -m ...` 即可，运行 MPC 时不要先切到只读的 `/workspace/ambench`。Isaac Sim 需要 GPU、NVIDIA 容器运行时及 `ACCEPT_EULA=Y`；compose 文件负责这些设置和持久缓存。先做最轻量的注册与场景检查：

```bash
docker exec ambench-sim-research python scripts/environments/list_envs.py
docker exec ambench-sim-research python scripts/research/verify_environment.py \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 --steps 8 --headless --device cuda:0
```

应看到 106 个注册 ID，随后得到环境构造、reset 和 step 成功记录。若第一步失败，查看容器日志，并确认 `nvidia-smi`、Isaac Sim 5.1 资产网络和镜像内的 acados 路径；不要把单次失败记作通过。

### 3.1 可选的本地 GUI 与遥操作

本机有 X11 桌面时，可用同一仿真镜像的 GUI 覆盖配置。先在宿主机终端确认 `DISPLAY` 与 `XAUTHORITY` 指向现有会话；切换模式会重建仿真容器，应先结束正在运行的评估或矩阵任务。

```bash
test -n "$DISPLAY" && test -f "$XAUTHORITY"
docker compose -f docker/compose.sim.yml -f docker/compose.gui.yml \
  up -d --no-deps --force-recreate sim
docker exec -it ambench-sim-research python scripts/environments/teleop_se3_agent.py \
  --task PressButton-Am-EE-Abs-PID-Direct-v0
```

`teleop_se3_agent.py` 持续运行，结束时按 Ctrl+C。在可见窗口按 R 应触发环境重置。2026-10-07 最终镜像的有界 X11 检查完成 PressButton 场景和 `Se3Keyboard` 初始化；软件事件只发送给本次容器的窗口，日志出现 `Reset triggered` 和 `Environment reset complete`。180 秒上限触发的 exit 124 是连续运行器的预期结束方式，日志为 `outputs/research/rebuild-20261007/gui/teleop.log`，结束后没有残留 Isaac 子进程。公共连续入口没有 seed 参数，此项为 `seed: null`。这项检查验证 GUI 与键盘回调，真人示范、SpaceMouse 和 gamepad 硬件仍未执行。完成后恢复无头服务：

```bash
docker compose -f docker/compose.sim.yml up -d --no-deps --force-recreate sim
```

正式录制人工示范时在容器内执行 `python scripts/data/record_demos_teleop.py --task PressButton-Am-EE-Abs-PID-Direct-v0 --help`，选择 keyboard、SpaceMouse 或 gamepad，再按 `usage.md` 的 canonical 数据验证步骤检查产物。Isaac Lab 2.3.2 会在显示帮助前预解析必填参数，因此这里也传入 task。设备透传取决于设备类型；键盘使用 X11 会话。无桌面的服务器继续使用上一节的无头 compose 配置。

## 4. 上传并启动远端策略容器

远端只在 `/diff/dzp_is_sb/ambench-research` 保存源码、镜像归档、缓存、数据与结果。上传脚本只传策略镜像，仿真镜像留在本地。

```bash
bash tools/research/deploy_policy.sh
ssh -o BatchMode=yes tencent-86 \
  'docker exec ambench-policy-research nvidia-smi -L'
```

需要指定空闲卡时传入编号，例如 `bash tools/research/deploy_policy.sh 0`。容器内只看见所选卡，因此服务命令统一使用 `cuda:0`。不要把服务器八卡全映射给策略容器。

### 4.1 启动服务与 SSH 隧道

远端容器的 ACT/DP HTTP 端口为 8001，OpenPI WebSocket 端口为 8000；只发布到远端环回地址。另开终端建立 SSH 本地转发，仿真容器使用 host 网络访问本地环回地址：

```bash
bash tools/research/tunnel_policy.sh
```

在另一个本地终端选择一个策略服务：

```bash
# ACT：先按 usage.md 第 5 节完成一步训练
bash tools/research/start_policy.sh act \
  /data/checkpoints/rebuild-20261007/act_press_button_smoke/checkpoints/last/pretrained_model

# Diffusion Policy：同一 canonical 数据的一步训练 checkpoint
bash tools/research/start_policy.sh dp \
  /data/outputs/rebuild-20261007/press_button_dp_smoke/checkpoints/latest.ckpt

# OpenPI π₀.₅：第 5.3 节的一步训练 checkpoint
bash tools/research/start_policy.sh pi \
  pi05_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative \
  /data/checkpoints/openpi-rebuild-20261007/pi05_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative/rebuild-20261007/1
```

三个示例都要求先按 `usage.md` 第 5 节生成训练 checkpoint 和对应数据统计；再次复现时统一改用新的运行名和空输出目录。运行 π₀ 时将 OpenPI 命令中的两处 `pi05` 替换为 `pi0`。

这些 `rebuild-20261007` 服务示例保留历史一步模型身份。当前 ACT/DP 的已选权重与完整训练预算分别记录在 [usage.md 当前结果](usage.md#当前正式训练结果2026-10-15)，不要把本节旧路径当作新训练模型。

启动后在本地宿主机等待服务加载完 checkpoint；把 `act` 替换为实际选择的 `dp` 或 `pi`：

```bash
bash tools/research/wait_policy.sh act 300
```

这三个命令是互斥服务选择，先停止旧服务再启动另一种策略。脚本顶部 `INFERENCE_SEED=42` 固定远端 Python、NumPy 与 Torch 的随机种子，日志包含 `REMOTE_INFERENCE_SEED=42`；以上基线评估也使用 `--seed 42`。前向 smoke 会消耗模型随机状态，开始可复现的评估前重新启动对应服务。ACT/DP 的仿真侧评估命令使用 `--remote-url http://127.0.0.1:8001`；OpenPI 使用 `--host 127.0.0.1 --port 8000`。跨机器流量经过 SSH 隧道，不需要打开公网推理端口。

### 4.2 无空闲 GPU 时的远端 CPU 推理与多视角记录

2026-10-08 补录时远端 GPU 均被其他任务占用，新增模型评估使用同一策略 Docker 内的 CPU 推理，本地 Isaac 仿真继续使用本地 GPU。服务入口加 `--cpu` 后会隐藏 CUDA 设备并记录 `REMOTE_INFERENCE_DEVICE=cpu`；推理 seed 仍为 42。以下命令从本地宿主机执行，只启动服务，不重新训练：

```bash
bash tools/research/start_policy.sh --cpu act \
  /data/checkpoints/rebuild-20261007/act_press_button_smoke/checkpoints/last/pretrained_model
bash tools/research/wait_policy.sh act 600
```

DP 和 OpenPI 使用各自原有 checkpoint，按 usage.md 第 6.2 节的 CPU 服务与双视角评估命令依次运行。CPU 推理可能明显慢于 GPU，应等待服务就绪后再启动单个 rollout；不要为了补录覆盖正在运行的他人 GPU 任务。

在线总览将 seed 42 基线和 seed 43 补录分组显示。新模型录像的环境 seed 为 43、推理 seed 为 42、训练 seed 为 42 且仅训练一步；与旧 GPU 记录同时改变了环境 seed 和推理设备，无法把结果差异单独归因于布局。新模型同一次 rollout 的 EE 和场景录像共享 `trial_id`，只计一次试验。十二任务的新视角按每次调用一条 episode 记录，基础设施异常的重跑独立标明；NDT 固定几何，仅补充观察视角。物理飞机的基座与外部场景视角分别运行，使用不同 `trial_id`。精确动作语义、相机、初始观测、步数、结果和来源哈希见每段的技术详情。

## 5. Canonical LeRobot 数据流

唯一源数据格式是记录器写出的 LeRobot 0.4.4 数据集，位于 `<session>/lerobot`，包含 `observation.state`、`action`、`task`、相机图像，以及 `meta/info.json` 的 `ambench.action_semantics` 和有序 `state_keys`。EE 绝对动作 8 维；当前四关节 BaseJoint 动作 12 维；四元数顺序 WXYZ。

当前 right-v2 契约区分 **命令姿态**与**实测 link 姿态**。令机器人配置偏移为 `O=command_to_link_quat_wxyz`，则 `q_link=q_command⊗O`，逆变换为 `q_command=q_link⊗O⁻¹`；canonical `ee_quat` 仍是实际 authored EE-link 的世界姿态。工具尖端位置也不同于 raw link 原点。历史实现使用 `O⊗q_command`；非单位姿态动作不可直接混用。若显式迁移旧动作，保持原物理目标需要 `q_command_new=O⊗q_command_old⊗O⁻¹`，同时记录版本；仅说明 WXYZ 或 8D 不足以确定坐标语义。

```text
本地 Isaac 容器：脚本专家 / teleop → canonical LeRobot
远端策略容器：同一份 canonical ├─ ACT：直接读取
                              ├─ DP：导出 UMI zarr
                              └─ OpenPI：导出固定 reader 的 LeRobot v2.1 派生集
远端策略容器：训练 / checkpoint → 推理服务
本地 Isaac 容器：观察经 SSH RPC → 动作 → 低层控制 → 评估记录
```

源数据不要改写为相对动作；相对轨迹由各策略适配器在训练时构造。DP zarr 与 OpenPI v2.1 是可重新生成的派生格式，不能取代 canonical 源。每次采集后用容器中的 `scripts/data/validate_lerobotdataset.py` 验证，再通过部署目录的同步步骤上传远端。原始数据、checkpoint、缓存和完整录像保持 Git 未跟踪；文档使用的短 GIF/MP4 与来源清单保存在 `usage_assets/animations/`。

完成 `usage.md` 第 5 节开头的 EE 和 BaseJoint 两条固定 seed 示范采集后，在本地宿主机运行。下面的目录和远端名称与后续训练、推理命令一致：

```bash
SESSION_INFO=$(find outputs/research/rebuild-20261007/scripted_press_ee -type f -path '*/lerobot/meta/info.json' | sort | tail -n 1)
SESSION_ROOT=${SESSION_INFO%/lerobot/meta/info.json}
test -f "$SESSION_ROOT/lerobot/meta/info.json"
bash tools/research/sync_dataset.sh "$SESSION_ROOT" press_button_ee_rebuild_20261007
SESSION_INFO=$(find outputs/research/rebuild-20261007/scripted_press_base_joint -type f -path '*/lerobot/meta/info.json' | sort | tail -n 1)
SESSION_ROOT=${SESSION_INFO%/lerobot/meta/info.json}
test -f "$SESSION_ROOT/lerobot/meta/info.json"
bash tools/research/sync_dataset.sh "$SESSION_ROOT" press_button_base_joint_rebuild_20261007
ssh tencent-86 'ls -l /diff/dzp_is_sb/ambench-research/datasets/press_button_{ee,base_joint}_rebuild_20261007/lerobot/meta/info.json'
```

DP zarr 转换在远端的 DP 环境执行；ACT 直接读取上传的 canonical `lerobot/`；OpenPI 的 v2.1 派生数据在远端 ACT 环境从同一 canonical session 导出到 `/data/datasets/openpi-rebuild-20261007` 下的对应 repo ID。完整命令见 `usage.md` 第 5.1–5.3 节；OpenPI 的 norm stats、训练和推理都使用这个独立 dataset home。

同步脚本拒绝覆盖远端同名目录。重新采集时使用新的本地输出目录与远端名称，并一致替换后续命令中的路径。所有训练输出与 checkpoint 放在远端 `/data/checkpoints` 或 `/data/outputs`。

## 6. 完整验证与故障定位

全部 106 个注册 ID 的有界 reset/step 验证：

```bash
docker exec ambench-sim-research python scripts/research/verify_matrix.py \
  --all --steps 8 --timeout-s 600 \
  --video-family-representatives --video-steps 60 \
  --output-dir outputs/research/verify-01
```

`usage.md` 列出任务、机型、控制器、脚本专家、数据、三种策略与扰动的逐项命令。NDT 首次会从 Isaac 资产源加载大型仓库场景，本次首次启动超过了 240 秒，后续在缓存就绪后通过；完整矩阵因此给每项 600 秒上限。镜像导入检查、一次 reset/step 和一个短 rollout 是不同验证级别；只有实际执行且产生相应产物的项目才计为通过。验证中出现超时或失败，先查该 ID 的日志与 GPU 使用，再修复并单独重跑；每次重跑使用新的空输出目录，不要把零动作 smoke 误写成任务成功率。

| 工作目录相关实测 | 结果与证据 |
| --- | --- |
| MPC 验证器的相对输出路径 | 最终镜像上 `verify_environment.py` 完成 seed 42 的 8 步，结果写入预期的 `outputs/research/rebuild-20261007/direct_mpc_relative/records/probe.json`；同级 `videos/` 含 9 帧可解码 MP4，新求解器编译完成。 |
| 普通 MPC 入口 | 最终镜像上 `zero_agent.py --task PressButton-Am-FAHexa-Abs-MPC-Direct-v0 --num_envs 1 --headless` 在可写工作卷完成构造、reset 和 3,151 条步进输出，记录为 `outputs/research/rebuild-20261007/mpc_normal_cwd.json`；75 秒有界运行的 exit 124 为预期，结束后没有残留子进程。公共连续运行器没有 seed 参数，此项为 `seed: null`。 |
| X11 GUI 遥操作 | 最终镜像完成 PressButton EE PID 场景、键盘初始化及定向 R 键重置回调；日志 `outputs/research/rebuild-20261007/gui/teleop.log`，180 秒有界退出后恢复无头 Compose，未将软件事件计为真人示范。 |

常用定位命令：

```bash
docker logs --tail 100 ambench-sim-research
docker exec ambench-sim-research nvidia-smi
ssh tencent-86 'docker logs --tail 100 ambench-policy-research'
ssh tencent-86 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

公开训练数据和任务专用 checkpoint 不在本仓库中。没有匹配 checkpoint 的策略评估应标为“未执行”，不能以服务健康检查冒充有效任务成绩。完整评估以 `eval_summary.json` 的 `status: "completed"` 和要求的 rollout 数量为准。

## 7. 删除研究镜像、同步官方源码并完整重建

先结束当前采集、训练和评估，提交本分支的源码改动，再执行下列命令。清理脚本只处理命名的 AM-Bench 研究容器、研究镜像标签及其策略镜像传输包；示范、checkpoint、模型下载缓存和 Docker 持久卷保留。共享基础镜像由 `--pull` 检查，应用层通过 `--no-cache` 重新构建，pip/uv 下载缓存可复用。

```bash
bash tools/research/reset_images.sh all
git fetch --no-tags https://github.com/ambench/ambench.git \
  main:refs/remotes/official/main
git log --oneline research..official/main
git merge --no-edit official/main
git submodule update --init --recursive ext/pyroki ext/acados ext/openpi
bash tools/research/build_images.sh all clean
docker compose -f docker/compose.sim.yml up -d
bash tools/research/deploy_policy.sh
```

2026-10-07 与 2026-10-08 的官方 main 检查均得到 `60bf5b73041df4eab571f7d9f0a297aeecbf2e0d`，该提交已包含在 research 历史中。存在新提交时先审查差异和解决合并冲突，再运行后续构建；子模块使用合并后仓库记录的版本。

构建脚本把源码 revision、是否有未提交改动和递归子模块版本写入镜像标签，并将 image inspect 与构建模式保存到忽略目录 `outputs/docker-builds/<image-id>/`。两种镜像内的 `/opt/ambench-build/` 保留各 Python 环境的依赖版本清单；应用安装和导入检查在构建中执行。重建后重新运行本手册第 6 节全量矩阵、`usage.md` 第 4 节示范采集及其策略数据、训练、推理和闭环检查，使用新的空输出目录。

[2026-10-07 重建快照](usage_assets/rebuild_snapshot.json)记录删除旧镜像、无缓存构建、后续源码修复重建、当时两端镜像和传输包哈希，以及 35 个原始证据的 SHA-256。该次运行镜像嵌入源码 `02d1499`。2026-10-08 的多视角录制、CPU 服务改动及本地镜像刷新另见第 2 节与[多视角验收快照](usage_assets/multiview_validation_snapshot.json)，两次记录分别保留源码和检查范围。

如果宿主机 `nvidia-smi` 正常，但旧容器内提示 `Failed to initialize NVML: Unknown Error`，先结束该容器内的任务，再重新创建容器，随后检查驱动访问：

```bash
docker compose -f docker/compose.sim.yml up -d --no-deps --force-recreate sim
docker exec ambench-sim-research nvidia-smi
```

本次 π₀.₅ BaseJoint 补测遇到过这一状态，重新创建容器后已完成真实闭环。若宿主机本身的驱动检查也失败，需要先恢复宿主机驱动，再运行研究流程。
