# 长训练扩展：协议、当前结果与复现

本页记录 `extensions-20261015` 研究扩展：UAQuad、OmniHexa 在 PressButton、WipeWindow 上的 **18 次正式训练、22 个模型任务评估单元**。每个单元锁定模型后运行 30 个封存 seeds，计划共 **660 次最终原生试验**，对应至少 1,980 段三视角原片。多任务模型对两个任务分别选模，因此一次训练可以对应两个评估单元。

**当前完成了 1/18 条完整训练链和 1/22 个完整评估单元：UAQuad WipeWindow ACT，实际训练 40,000 次更新，最终 0/30。其余项目仍在训练或准备，不能写成全部完成。** 下表是 2026-10-15 已有收据的快照，运行计数不是实时面板。

| 训练组 | 训练次数／评估单元 | 必须完成的预算 | 当前实际状态 |
| --- | ---: | --- | --- |
| UAQuad Wipe ACT | 1／1 | 40,000 次 AdamW 更新 | 完整预算、8 个候选的全量留出验证、80 次选模、锁定及 30 次最终测试均已验收；**0/30** |
| UAQuad Wipe DP | 1／1 | 40 完整 epochs；88,360 次更新 | 正在原训练链运行；11:17 UTC 收据记录实际 **28,700** 次更新、299 个 AdamW 状态一致；最终库存与原生评估未完成 |
| OmniHexa 两任务 ACT | 2／2 | 每任务 40,000 次更新 | fresh200 数据通过；旧 CPU 准备失败证据保留，新 v3 的真实 CPU 准备验证中，尚无正式训练完成收据 |
| OmniHexa 两任务 DP | 2／2 | 每任务完整 40 epochs；更新数由自身实际 sampler 计算 | 旧 CPU 准备失败证据保留，新 v3 的转换与接口验证中；不能套用 UA 的 88,360，也尚未完成正式训练 |
| UAQuad π₀／π₀.₅ | 6／8 | 两个 MT 各 40,000，四个 ST 各 20,000；合计 160,000 | π₀.₅ MT 正在训练；首次名义 `5000` checkpoint 已实读验证为 **5,001** 次更新，尚未完成 MT 全预算；ST 等待真正 MT final |
| OmniHexa π₀／π₀.₅ | 6／8 | 同上，合计 160,000 | 同一份 fresh200 已被 CPU sourcefix-v3 消费；MT160 导出已完成，ST 导出与独立统计流程继续，尚无正式 GPU 训练 admission |
| **合计** | **18／22** | ACT 共 120,000 次；OpenPI 共 320,000 次；三个 DP 各完整 40 epochs | **12 次 OpenPI 的完整预算验收均未完成；只有上述 ACT 单元具备完整最终结果** |

[公开预声明](usage_assets/extensions_20261015_plan.json)固定了全部训练清单、候选规则、数据划分和 seeds。已有 UAQuad PressButton 的 [ACT 0/30、DP 30/30](trained_models.md) 属于此前研究，单独保存，**不计入本次 18／22／660**。本页也不把脚本专家预检、CPU tests、8 步 smoke 或独立 ACT absolute-action 消融算作正式模型成绩。

## 1. 先固定源码、数据和动作接口

双容器部署、Isaac Sim 5.1／Isaac Lab 2.3.2 环境与 SSH 隧道见 [note.md](note.md)；基础逐项验证见 [usage.md](usage.md)。本扩展的学习源码固定为 `9ab28b02ab332e47e4a15dbeccca31f83452aed8`，291 文件 source manifest 的 SHA-256 为：

```text
3f546f2f29c86925b6062ff3d27afc49b623986ccebf2b3168f49f60ec2e4410
```

包含 Omni 夹具修订的公开预声明 SHA-256 为：

```text
90e039f683e484e4199b0181a73cad40e81ea74926ca77177811bc5271a1cdde
```

已经激活的 UA ACT 原生队列保留其原预声明 `efdda7cc…`，不反向替换成新字节；其精确来源见 [ACT 清单](usage_assets/extensions/act_manifest.json)。源码冻结、数据就绪、训练预算、候选库存、选模锁和最终结果各有独立收据，后一步不能代替前一步。

### 数据与归一化

每个机型、每个任务使用 100 条原生成功 canonical 示范：源 episode **0–79 训练、80–99 留出**。采集按每个 app 最先获得的 20 条真实成功保留明确映射，失败尝试仍保存。20 条留出示范不进入训练，也不参与归一化拟合。

ACT、DP 使用各自机型任务的 train80 统计。OpenPI 每个机型有一份两任务各 80 条组成的 **MT160**，以及 Press、Wipe 各自的 **ST80** 导出和统计；不能把 MT 统计复制后改名成 ST 统计。本研究不声称 OpenPI 的离线验证 loss，也不用训练 loss 选模。

Omni 的正式 fresh200 已完整通过原始记录、官方合并与 validator、全部合并数值行的独立逐行比对及每任务 300 个 RGB 样本检查。正式 data-ready SHA 为：

```text
74fad093f24d1e7fe020b3436d0d6d45d0b49f8eea0280b3dd1c1961342bd3d3
```

其中 Press 为 100 条成功、99,907 个原始数值帧，来自 101 次原始尝试；Wipe 为 100 条成功、194,806 帧，来自 100 次原始尝试。这些是 canonical100 的原始 120 Hz 数值帧数，不是 train80 的 20 Hz 样本数，也不是视频帧数。已有 UA Wipe 的 train80 为 148,555 个原始帧、24,726 个 20 Hz 导出帧；DP 的 141,355 是有效序列样本位置，不能当成原始帧数。

Omni ACT/DP 新流程要求独立 CPU profile、train-only normalizer 参考，以及**首次实际 AdamW 更新之前**的精确 tensor 比对收据。当前准备验证不能写成该更新前 gate 已通过。UA DP 已有训练中独立 normalizer 比对，属于事后核验，不倒写成首次更新前检查。

### Omni Wipe 的物理版本

Omni Wipe 使用批准的 gen03：base `9ab28b0` 完整 394 文件仿真树，仅叠加 Wipe 环境和 Omni 配置两个 runtime 文件。variant manifest SHA 为 `08cd3b75da798670ad631d08a6830f252536b21b7f84656644964175f050e395`。这不等于整个后来 HEAD；学习源码仍是原 291 文件。

该版本修复 authored EE 工具坐标与 USD 非均匀 scale 的变换，并将 Omni 海绵挂载 offset 定为 authored EE 局部 `(0, 0, 0.15)` 米，定义在 USD scale 之前。原生条件仍为 20 秒、清除全部 4 个污点、力严格大于 1.25 N、接触半径 80 mm。原 base／gen02 失败与成功专家预检均保留，且排除训练、选模和封存成绩；[夹具专家预检播放器](https://zipengdai.com/ambench/extensions/)与[来源清单](usage_assets/extensions/fixture_manifest.json)单独标明身份。

### 时间与动作语义

| 接口 | 已固定的执行方式 |
| --- | --- |
| 原生仿真与 canonical | 120 Hz；源动作是 8D 绝对末端位姿与夹爪命令，动作 frame 为 `task_command_world_right_v2` |
| ACT | 20 Hz 逻辑轨迹；chunk 16，每次执行 8 个；内部为 `ee_local_relative`，由配套 processor 转回绝对命令 |
| DP | 两个观察时刻，horizon 16，原始帧 stride 6；每 48 个原生步重新查询，预测并插值到 96 个原生步；DDIM 推理 16 步、训练扩散 50 步 |
| OpenPI | 20 Hz，horizon 50，每次执行 8 个；使用对应模型自己的统计与已保存 serving config |
| 图像与录像 | canonical EE RGB 为 384×384；DP 编码器按自身配置处理图像。模型只用 EE 输入，场景与基座相机用于记录；原片 30 FPS |

DP 的 16 个训练目标位于原生索引 `0,6,…,90`，原有插值输出覆盖 `0…95`。本研究保留并披露这个接口边界，不静默改变模型目标或执行长度。远端推理的墙钟耗时不等于原生仿真时间。

## 2. 完整训练后，才建立候选库存

ACT 是完整 ImageNet ResNet18，batch 8、AdamW LR `1e-5`、KL 权重 10，每任务实际 40,000 次更新。5k 到 40k 的 8 个候选都要检查真实 153 个 AdamW 状态、保存的 RNG、配置与 processor，并对完整 20 条留出示范计算归一化 MAE。MAE 是无量纲指标，不是厘米，也不是成功率。

DP 使用完整预训练 CLIP ViT-B/16 与 U-Net `[256,512,1024]`、EMA；microbatch 8、梯度累积 8，保留 epoch 尾窗口，训练完整 40 epochs。预算由实际训练样本、DataLoader microbatches 和更新时点计算；验收检查 299 个 AdamW 状态、EMA、scheduler、RNG、episode 划分，以及全部 epochs 的完整留出 EMA 验证。

**DP 候选是训练结束时实际保留的 top4 加 latest final40，按整个 checkpoint 文件的 SHA-256 去重，最多 5 个。** epoch 1／6／11／16／21／26／31／36／40 是保存机会，不是必然保留的候选。epoch 1 只有最终仍在 top4 才进入库存。retained40 与 latest40 即使都完成 epoch 40，只要 whole-file SHA 不同且各自验收通过，就分别保留；不能仅按 epoch 或模型 tensor 去重。

每个机型的 OpenPI 六跑包括 π₀、π₀.₅ 各一条 MT160 和两条任务 ST80：MT 各完整 40,000 次、global batch 32，ST 各完整 20,000 次、global batch 8，EMA decay 0.99，均为完整微调。实际容量验证后采用双卡 FSDP2 串行，不降低模型、batch 或预算，也不宣称与单卡逐位等价。此前环境导致的 exit139 与后来的真实单卡 OOM 分别保留，不能混为同一个失败原因。

OpenPI MT 有 8 个、ST 有 4 个名义 5k 间隔候选，含实际 final；实际计数从 Orbax `train_state.step` 与 Optax 状态读取。已验证的首个 UA checkpoint 目录 `5000` 存储的是 **5,001** 次更新；MT final 目录 `39999` 应对应实际 **40,000**。目录出现本身不证明训练或完整预算已完成。

**每个 ST 的母本必须是同机型、同 π₀／π₀.₅ 变体真正完成的 MT final actual40000。** MT 对 Press、Wipe 分别选中的较早 checkpoint 不改变这个父模型，封存结果也不能用于选择 ST 母本。

Omni 的 CPU 工具准备需要完整 base394 中的公开 converter／exporter，同时逐 SHA 核对原 291 学习文件；GPU 训练继续挂载原学习源码。旧 Torch CPU v2 因 NumPy 统计复制假设和缺少 converter 而失败，未开始正式 GPU 训练，证据保留；独立 v3 修复正在验证。OpenPI 的 sourcefix-v3 同样只修 CPU 工具挂载与来源绑定，原六份请求及训练数学保持冻结。

## 3. 选模、封存与成功判定

| 机型／任务 | 每候选的 10 个选模 seeds | 锁定后的 30 个封存 seeds |
| --- | --- | --- |
| UAQuad WipeWindow | 43001–43010 | 53001–53030 |
| UAQuad PressButton | 83001–83010 | 93001–93030 |
| OmniHexa PressButton | 63001–63010 | 73001–73030 |
| OmniHexa WipeWindow | 64001–64010 | 74001–74030 |

四组区间互不相交。相同机型任务的不同模型共享这些固定 seeds，候选、排名和锁各自独立；旧 Press 最终 seeds、34001／34002 重放与专家预检均不进入这些选模集合。每个 seed 保留第一次有效原生尝试，真实任务失败不能选择性重跑。基础设施异常另存证据，完成固定有效 seed 集合后才排名，不能把缺失遥测补成零分。

ACT／DP 按以下顺序比较：**原生成功数降序 → 自身完整留出集 loss 升序 → 实际更新数降序 → model SHA 字典序**。ACT 用全量归一化 MAE；DP 用自身 train-only 归一化、固定验证随机性的完整留出 EMA diffusion loss。

OpenPI 按：**原生成功数降序 → 终态 named progress 均值降序 → 成功试次平均原生步数升序 → 实际更新数升序 → model SHA 字典序**。Press progress 为按钮位移米数，Wipe 为已清污点数，均从自动 reset **之前**的终态物理数据取得，并对 10 个有效尝试取均值。没有成功时平均成功步数记为 `null` 加明确无成功标志，比较时排在有成功者之后，不输出非标准 JSON `Infinity`。MT 对每个任务单独排名和封存。

第一次最终试验前，锁定候选全集、每 seed 结果、精确排序，以及模型、归一化、数据、源码、配置、预算、inventory 与协议 SHA。之后不得根据 30 个封存结果重选或调参。最终分母固定为 30；一条 episode 的三路相机不增加分母。

新研究每 trial 使用独立应用与独立远端 CUDA 服务，推理 seed 42、无前向预热，环境为单实例、20 秒、扰动关闭、reset 重绘 1 次。新 Orbax/JAX 服务还必须证明实际采样 key `[0,42]`、首握手 forward 数为零及后续 key 链；仅设置 Python／NumPy／Torch seed 不足以证明 JAX 种子。GPU 使用前核容量并持共同 UUID 锁，完成后只清理己方确证容器，HOST compute 与 graphics 进程均空、显存达清理门槛后才释放。

## 4. 已完成单元：UAQuad Wipe ACT

完整训练 40,000 次更新后，8 个候选各运行 10 次选模，均为 0/10。按预声明的完整留出 MAE 锁定 step 40,000，MAE 为 **0.0578722822**。随后 53001–53030 全部运行完毕：**最终 0/30，均在原生第 2399 步超时，终态清污 0/4**；95% Wilson 区间约 0–11.35%。选模中有一次清除了 1 个污点，仍未达到全部 4 个污点的成功条件。

选中模型 SHA 为 `93509c8f5a563c7a3c8bf88d17181f55972506789fc039631587758858cb29c6`；四个 processor 文件摘要组成的 canonical JSON normalizer SHA 为 `23d0c3c30a5cd464c97ead3a50a9cbc08f95567b5070389f6ade0a2ab36eeb48`；选模锁 SHA 为 `97f3416a291ac2d2d65e646f4b562f55e52b62ccd92c9e957e797d541d395d0c`。

下面固定展示最终首、中、末 seeds 的 EE 预览，三条均为 **TIMEOUT**。本单元没有成功视频；不能用另一任务或脚本专家的成功替代。

**seed 53001 · TIMEOUT · 0/4**

[![UAQuad Wipe ACT seed53001最终超时EE预览](https://github.com/superboySB/ambench/releases/download/research-media-20261015-uaquad-wipe-act/uaquad_wipe_act_sealed_seed53001_ee_camera.gif)](https://github.com/superboySB/ambench/releases/download/research-media-20261015-uaquad-wipe-act/uaquad_wipe_act_sealed_seed53001_ee_camera.mp4)

**seed 53015 · TIMEOUT · 0/4**

[![UAQuad Wipe ACT seed53015最终超时EE预览](https://github.com/superboySB/ambench/releases/download/research-media-20261015-uaquad-wipe-act/uaquad_wipe_act_sealed_seed53015_ee_camera.gif)](https://github.com/superboySB/ambench/releases/download/research-media-20261015-uaquad-wipe-act/uaquad_wipe_act_sealed_seed53015_ee_camera.mp4)

**seed 53030 · TIMEOUT · 0/4**

[![UAQuad Wipe ACT seed53030最终超时EE预览](https://github.com/superboySB/ambench/releases/download/research-media-20261015-uaquad-wipe-act/uaquad_wipe_act_sealed_seed53030_ee_camera.gif)](https://github.com/superboySB/ambench/releases/download/research-media-20261015-uaquad-wipe-act/uaquad_wipe_act_sealed_seed53030_ee_camera.mp4)

[全部 110 次试验与三视角原片播放器](https://zipengdai.com/ambench/extensions/act.html)提供 80 次选模及 30 次最终试验，共 **330 段原片**。来源、逐片 SHA、完整候选排序与权重包见 [act_manifest.json](usage_assets/extensions/act_manifest.json)，交付验证见 [act_validation.json](usage_assets/extensions/act_validation.json)；[Release](https://github.com/superboySB/ambench/releases/tag/research-media-20261015-uaquad-wipe-act)另有原始证据压缩包及文件索引。

生产者已对 330 原片完整 CPU 解码；根审查独立重复解码并查看固定三 seeds 的 9 段代表原片，两个验证范围分别记录。原片每 4 个原生步采样一次，末图在第 **2396** 步，终态成绩来自第 **2399** 步的 reset 前物理 journal；末图不等于终态物理时刻，也不追加下一 episode 的 reset 画面。

想查看已经成功的学习策略，可另看 [此前 UAQuad Press DP 的 30/30 成功与视频](trained_models.md)；[Omni 专家夹具成功预检](https://zipengdai.com/ambench/extensions/)属于脚本专家，二者都不改变这里的 ACT 0/30。

## 5. 下载选中权重并单独重放

[选中 ACT 推理 ZIP](https://github.com/superboySB/ambench/releases/download/research-media-20261015-uaquad-wipe-act/uaquad_wipe_act_selected_step040000_inference.zip)为 191,426,512 bytes，SHA-256 是 `e7640d2b7ef37536e14c7ee2d7be44ae4a3c1489028f2503e8064bc1c0fb21cb`。解压得到原始 `pretrained_model/` 的 **7 个文件、206,396,168 bytes**：权重、配置、训练配置和四个 processor 文件。**这是推理包，不包含恢复训练所需的 optimizer／scheduler／RNG 状态。** 权重与自己的 normalizer 必须一起使用。

先在仓库根目录从清单读取 URL、大小及 SHA，验证 ZIP 和全部成员后解压到新的输出目录。以下是读者使用的下载模板；本轮文档编写没有重复下载或运行实验。

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

manifest = json.loads(Path("usage_assets/extensions/act_manifest.json").read_text())
package = manifest["selected_inference_package"]
output = Path("outputs/extensions-replay-seed53001")
output.mkdir(parents=True, exist_ok=False)
archive = output / "inference.zip"
urllib.request.urlretrieve(package["url"], archive)

def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()

assert archive.stat().st_size == package["bytes"]
with archive.open("rb") as stream:
    assert digest(stream) == package["sha256"]
expected = {"pretrained_model/" + name: ref for name, ref in package["files"].items()}
assert len(expected) == 7
with zipfile.ZipFile(archive) as bundle:
    assert len(bundle.infolist()) == len(expected)
    assert set(bundle.namelist()) == set(expected)
    for name, ref in expected.items():
        assert bundle.getinfo(name).file_size == ref["bytes"]
        with bundle.open(name) as stream:
            assert digest(stream) == ref["sha256"]
    bundle.extractall(output)
print("Verified ZIP and seven original inference files:", output / "pretrained_model")
PY
```

为 UA Wipe 重放固定原学习与仿真版本，使用独立 worktree，不切换已有训练挂载的 checkout：

```bash
git worktree add --detach ../ambench-extensions-replay 9ab28b02ab332e47e4a15dbeccca31f83452aed8
```

按 [trained_models.md 的双容器传输与启动步骤](trained_models.md#3-下载并放进远端策略容器)，把已校验目录同步进远端策略容器的 `/data/outputs/extensions-replay-seed53001/`，并配置该 worktree 的源码挂载、独立容器与可用 GPU。以下沿用 `ambench-policy-research`／`ambench-sim-research` 名称和现有隧道脚本；同名服务存在时应先使用完整独立部署，不触碰训练作业。

以下是**尚未作为本轮新实验执行的重放模板**，复用公开入口与既有参数。实际正式 0/30 来自冻结的审计 worker；便捷重放不替代其身份、锁、journal 和 30 次完整验收，也不承诺视频字节或随机数状态逐位一致。

```bash
# 远端策略服务：默认 CUDA、推理 seed42；一次 trial 从新服务开始。
bash tools/research/start_policy.sh act \
  /data/outputs/extensions-replay-seed53001/pretrained_model
```

另一个终端保持 `bash tools/research/tunnel_policy.sh`，随后等待 health 就绪，避免用模型前向预热：

```bash
bash tools/research/wait_policy.sh act 600
docker exec ambench-sim-research python -m ambench_learn.policies.act.eval \
  --task WipeWindow-Am-UAQuad-Abs-PID-Direct-v0 \
  --num-envs 1 --num-rollouts 1 --seed 53001 --episode-length-s 20 \
  --remote-url http://127.0.0.1:8001 \
  --n-action-steps 8 --policy-target-hz 20 --save-video \
  --video-camera-names scene_camera base_camera ee_camera \
  --output-dir outputs/extensions-replay-seed53001/rollout \
  --headless --device cuda:0
```

`53001` 用于检查同一个已发布 seed 的重放，不新增到原最终分母。读取新 `eval_summary.json` 中任务、seed、模型来源、真实步数、终止原因和原生 `success`；`completed` 或进程 exit0 只表示执行完成。正式测试若要重建，应重新落实上文所有库存、选模锁、seed 与服务身份门禁，而非循环运行这条便捷模板后改写原成绩。

## 6. 如何阅读和验收后续交付

本地查看交互页面时，从仓库根目录运行 `python3 -m http.server 8765`，再打开 `http://127.0.0.1:8765/usage_assets/extensions/act.html`。页面通过 `fetch` 读取 JSON，直接双击 HTML 可能被浏览器限制；视频与 GIF 是外链，本地页面仍需要网络。源码、HTML／JSON／Markdown 留在 Git，视频、GIF、数据、checkpoint 与权重包通过外部 Release 提供，不依赖 Git LFS。

后续每个单元要在**完整训练预算 → 实际候选库存 → 全部选模 → 不可变选模锁 → 完整 30 次最终试验 → 原片及来源审查**通过后，才补最终成功率、选中实际更新、模型／normalizer SHA、Wilson 区间和代表视频。有成功才展示成功；没有成功就明确展示超时。发布附件还需核匿名下载 SHA、实际浏览器播放／暂停／拖动及三视角同步；CDN 的 `application/octet-stream` 响应与已验证的播放功能分别报告。

现有 106/106 reset／8 步 smoke、126 项 CPU 回归和 12/12 历史 canonical 重验属于 [usage.md](usage.md) 的另一轮验证，不能推出本次 18 条训练已完成，也不能推出所有十二任务的学习能力。这里没有新实机、硬件外设或 real2sim2real 完成结论；当前可公开的本次完整学习结果仍只有 UAQuad Wipe ACT **0/30**。
