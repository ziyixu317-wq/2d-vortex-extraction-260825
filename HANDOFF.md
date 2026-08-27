# HANDOFF — 迹线 Transformer 涡提取项目交接文档

> 本文档是项目的**唯一权威上下文（单一事实来源）**。任何新会话接手本项目时：先完整阅读本文档，再按 §9 工作流推进。
> 维护协议见 §11：事实变化时直接改正文对应小节，并在 §11 追加变更日志条目；不要把过程叙述堆进正文。
> 历史文件 `工作计划_迹线Transformer涡提取.md` 已被本文档取代，仅作考古用，以本文档为准。

---

## 0. 项目一句话

用 VortexTransformer 论文（CGF 2025）的**迹线 Transformer**（`PathlineTransformerV0`）在 2D 非定常仿真流场（当前数据集：ETH "2D Unsteady Cylinder Flow Around Corners" 的 `pipedcylinder2d.nc`）上做**涡提取**：从迹线（pathline）直接输出每条迹线的涡概率，再投影回网格。

- 论文：Zotero `zotero://user/0/item/T76G9Z3A`，DOI 10.1111/cgf.70042，开源仓库 `PyflowVis-main`（Apache 2.0）
- 原始参考仓库（只读参考，不直接依赖）：`C:\Users\徐子屹\Desktop\AI CFD\PyflowVis-main`
- 数据集文件：`C:\Users\徐子屹\Desktop\AI CFD\CFD数据集\pipedcylinder2d.nc`
- 本项目目录（自包含，见 §4）：`C:\Users\徐子屹\Desktop\AI CFD\cylinder_vortex_pipeline\`

## 1. 用户已拍板的决策（默认不再争论，除非用户主动改口）

1. **训练标签**：直接用 IVD/Q-criterion 阈值给仿真数据打**弱标注**（标签 = 迹线种子点处 IVD ≥ τ）。**不走** Vatistas 参数拟合/合成数据管线。
2. **模型**：论文主体的迹线 Transformer，**从头训练**；不使用仓库预训练权重与 demo 验证集（用户已在 Kaggle 验证过模型可用）。
3. **训练**：Kaggle T4×2，**200 epoch**；本地（核显、torch 2.10.0+cpu、无 CUDA）只做 CPU 冒烟。
4. **评估**：定性对比（IVD/Q-criterion 为参考）+ 弱定量表；不涉及 Vatistas 验证集。
5. **不做 ivd 遮除消融**：IVD 是公认高精度涡判据，模型学习其近似是可接受且预期的。
6. **迹线口径 = 256 条**：64 组 × 4 卫星点（不含中心），`KpathlinePerGroup=4`，对齐论文与发布数据。
7. **代码组织**：独立自包含工程，迁移（复制）所需代码进本项目，不依赖原始 `PyflowVis-main`；保留 Apache 2.0 署名。
8. **geometry 掩膜是逐数据集预处理**，不进入模型输入，不影响多数据集泛化（后续会加更多仿真数据集）。

## 2. 已核实事实（带来源，直接用，勿重新考证）

### 数据集 `pipedcylinder2d.nc`（实测）

| 项 | 值 |
|---|---|
| 格式 | NetCDF-4/HDF5；变量 `tdim/xdim/ydim/u/v`（另有 Re=160、nu、radius=0.0625、const） |
| 网格/时间 | 450×150，1501 步（dt=0.01，t∈[0,15]） |
| 域 | x∈[-0.5,5.5]，y∈[-0.5,1.5]；u∈[-3.0,4.6]，v∈[-2.3,2.4] |
| 质量 | 全量无 NaN；**41.8% 细胞近零速（每帧固定 28213 个）**= 静态固体几何（台阶管道壁面+圆柱+死区） |
| 固体几何（票 02 实测） | 全帧取与（\|v\|<1e-5）后 **4 个连通块**：两块矩形壁面（x∈[2.5,5.5],y∈[-0.5,0.5] 与 x∈[-0.5,1.5],y∈[0.5,1.5]，接触域边界）+ **两个孤立圆柱**：入口 (≈0,0) 与拐角后管道 (≈3,1)，零速盘 43/45 格（圆角方块，等效半径≈0.047）；radius=0.0625 与 Re=160=U·D/ν（U=1, D=2r, ν=0.00078125）自洽；零速区为圆柱**内切区**（表面约 1 格厚格被插值为非零速），物理半径≈面积等效+1 格≈0.063 |
| 坑 | netCDF4 的 C 库打不开中文路径（实测）；**h5py 可直接读中文路径** → 一律用 h5py。Kaggle 路径为 ASCII 无此问题 |
| 可用 patch（票 05 实测） | 216 个 patch 位置中 **128 个可用**（88 个不可用：种子-中心线段全固体——壁面区/圆柱包围区；票 03 "全固体 patch 应避开"边界）；池组合 31,360（train 片：正 13,652=43.5%、负 17,708） |

### 模型事实（`DeepUtils/models/segmentation/pathline_transformer.py`）

- 输入迹线簇 `[B, L=16, K=256, C=7]`；C = **[px, py, t, ivd, distance(距种子点), u, v]**
- 十字采样：8×8=64 组 × 4 卫星点 `(x±Δ,y),(x,y±Δ)`（**不含中心**），Δ = patch 边长×0.05；**组主序编组**（组 0 的 4 条在前）
- PSL：组置换 + 空间下采样（64→32 组）+ 时间下采样（L 16→8）→ N = 32×4×8 = **1024 点**
- 3 层 KNN Point-Transformer（k=16，相对位置编码 MLP）；全局池化 mean+max；特征传播回全部 256 条；sigmoid 输出每迹线涡概率；损失 BCE
- 训练超参（论文附录 C）：AdamW(wd 1e-6)、lr 1e-4、batch 100、200 epochs、warmup 后降 lr（仓库 config 为 TwoStep：warmup 60、二段 5e-6）
- **KNN 在 (x,y,t) 混合坐标上暴力计算 O(N²)** → 时空归一化尺度影响邻居选择；原训练数据 t/空间比≈0.2 → 用 `t_scale=0.25` 复刻（可调）
- **推理非确定**：PSL 采样在 forward 恒 `random=True` → 评估用 TTA（同一样本随机采样 5 次取平均），或临时改 `random=False`

### 代码事实（依赖与迁移）

- `DeepUtils/models`、`DeepUtils/loss`、`DeepUtils/utils/{registry,ckpt_util,random}` 导入链**全纯 torch**；`get_missing_parameters_message` 等位于 `utils/ckpt_util.py`
- 原仓库 `utils/__init__.py → config.py` 需要 `multimethod` → 迁移时**剔除 config.py**，重写 utils 的 `__init__.py` 只导出所需符号
- **不要**导入原仓库的 `DeepUtils/dataset`、`FLowUtils`、`train.py`、`test.py`、`MiscFunctions.py`（拖入 numba/pybind/GUI/wandb）
- `loss/build.py` 已注册 `BCELoss`；`registry.build` 接受普通 dict 配置
- 最终依赖清单：torch、numpy、h5py、PyYAML（pip 包名；import 名 yaml）、matplotlib、tqdm（本地 Python 3.12 已装除 torch 外全部；Kaggle 自带 torch）
- C++ 生成器只支持解析场（`PathlineIntegrationInfoCollect2D` 对离散场 assert）→ 迹线提取必须自写（`extractor.py`）

## 3. 目标与验收标准

**总目标**：在 `pipedcylinder2d.nc` 上从头训练迹线 Transformer，输出可信的涡提取结果。

验收标准（全部满足才算完成）：
1. 预测涡区域与 IVD/Q-criterion 参考结构一致（涡街、拐角回流区），无明显碎裂噪声
2. 200 epoch 训练完成，checkpoint 归档，训练可跨 Kaggle 会话断点续训
3. 交付：多个代表性时间步的对比图 + mp4 动画 + 弱定量表（对 IVD 阈值的 F1/IoU、涡面积占比、帧间连续性）+ 复现 README
4. 多阈值敏感性报告（τ 的稳健性说明）

## 4. 目录结构与代码清单（阶段 0 迁移产物）

```
cylinder_vortex_pipeline/
├── vendor/DeepUtils/            # 从 PyflowVis-main 复制的最小纯 torch 子集
│   ├── models/                  #   pathline_transformer.py, base_seg.py, build.py, samplingLayers.py, layers/, segmentation/, classification/, reconstruction/
│   ├── loss/                    #   build.py（BCELoss 注册）, cross_entropy.py, distill_loss.py
│   └── utils/                   #   registry.py, ckpt_util.py, random.py + 重写的 __init__.py（剔除 config.py）
├── LICENSE  NOTICE              # 随迁移代码保留 Apache 2.0 署名
├── geometry.py                  # 固体掩膜（逐数据集预处理；进不了模型输入）
├── extractor.py                 # 迹线生成（256 条/样本，7 通道，RK4+三线性插值，掩膜处理）
├── weak_labels.py               # IVD（5×5 局部邻域均值）+ Q-criterion + 阈值标签
├── dataset.py                   # WeakLabelPathlineDataset（h5py+memmap，on-the-fly，多数据集扩展）
├── train_kaggle.py              # 自写训练脚本（TwoStep、断点续训、可选 DataParallel/AMP；票 07 增 --report-f1）
├── evaluate.py                  # TTA 推理、网格投影、对比图/动画、弱定量表
├── kaggle/                      # 票 07：Kaggle Notebook/打包/分块/自检/操作手册
├── config/pathline_transformer_cylinder.yaml
├── CLAUDE.md                   # agent 入口说明（唯一权威仍是本文件）
├── docs/agents/                # ask-matt 配置：issue-tracker.md / triage-labels.md / domain.md
├── HANDOFF.md                   # 本文件
└── NEW_SESSION_PROMPT.md        # 新会话启动 prompt
```

各模块职责（实现时逐条落实）：
- **geometry.py**：|v|<ε 逐帧取与 → 连通域标记 → 输出 `mask.npy`（种子排除、迹线截断、IVD 置零共用）+ `geometry_meta.json`（块统计与圆柱定位）；圆柱 = 不与壁相连的孤立连通块（无尺寸判据；pipedcylinder2d 实测 2 个：入口 (≈0,0) 与拐角后 (≈3,1)）；无障碍物数据集输出空掩膜，代码路径不变。**每个新数据集各自跑一遍**，掩膜随数据集的 (T,Y,X) 存储。
- **extractor.py**：全局场积分（允许迹线离开 patch）；种子落固体 → 重播种（仿 C++ `JittorReSeeding`）；迹线入固体 → 截断并重复末点（不引入 -1000 毒值）；位置按 patch 归一化到 [-1,1]（可超界）。
- **weak_labels.py**：中心差分 ω=∂v/∂x−∂u/∂y，IVD=|ω−5×5 邻域均值|；标签 = IVD(种子,t0)≥τ + 5×5 最小面积连通域过滤；固体区 IVD=0。
- **dataset.py**：时间划分 train [0,10]s / val (10,12.5] / test (12.5,15]（帧 0-1000 / 1000-1250 / 1250-1500，无时间泄漏）；patch 32×32 stride 16，窗口 T_win=24 帧、起点步长 4 帧；每 epoch 40000 样本、50% 正样本过采样；u,v 与预计算 IVD 存 memmap（IVD 一次算好 ≈405MB）；返回 `((dummy_field, pathlines), labels)` 匹配模型输入；正样本 = patch 内存在 ≥1 条涡迹线。
- **train_kaggle.py**：不 import 原仓库 train.py；TwoStep（warmup 60 epoch @1e-4 → 5e-6）；梯度裁剪 1.0；每 epoch 存 checkpoint（含 optimizer 状态）；batch 100。
- **evaluate.py**：滑窗推理 → TTA 5 次平均 → 投影（累积+计数平均消 patch 重叠）；展示帧用加密种子（每 2×2 输出像素一组十字）；底图用速度模场（不依赖 LIC 渲染器）。

## 5. 阶段计划（每阶段有完成判据）

- **阶段 0 代码迁移**：按 §4 复制 vendor 子集、重写 utils `__init__.py`、复制 LICENSE/NOTICE。
  判据：本地 `python -c "from vendor.DeepUtils.models import build_model_from_cfg"` 通过；此后全项目不引用原 PyflowVis-main。
- **阶段 1 本地 CPU 冒烟**：geometry 掩膜可视化+圆柱定位；extractor 在 3~5 个 patch×24 帧窗口生成 256 条迹线并目检；weak_labels 出 τ 对比图定 τ；CPU torch 跑通一次前向。
  判据：掩膜/迹线/标签图目检通过；τ 定值写入参数表；前向输出形状 `(B, 256)` 概率正确。
- **阶段 2 弱标签校验**：IVD/Q 图 + 正样本占比统计。
  判据：标签在涡街/拐角涡处成连通块、非涡区干净。
- **阶段 3 数据集类**：dataset.py + memmap 预计算 + 归一化（px,py→[-1,1] patch 内；t→[0,1]×t_scale；ivd 标准化；distance 用归一化坐标；u,v÷全局最大速度）。
  判据：单样本生成 <5ms（**2026-08-25 用户放宽：时间性能不纠结、能跑即可**——on-the-fly 实测 ~35ms 已记录，Kaggle 8-worker 下不构成训练瓶颈）；train/val/test 划分无泄漏。
- **阶段 4 Kaggle 上传**：Dataset A = nc 文件；Dataset B = 整个 pipeline 目录。
  判据：Notebook 里 `pip install h5py PyYAML matplotlib tqdm` 后能 import vendor 并加载数据。
- **阶段 5 训练**：先 1 epoch 冒烟实测每步耗时 → 校准 epoch 样本数（T4 单卡预计 0.8~1.3s/步 → 80k 步约 18~29h，超预算则用 DataParallel/AMP/降样本数）；**Kaggle 会话硬上限 12h** → 分块 ≤8h，每 epoch checkpoint，块尾打包为 Kaggle Dataset 新版本，下次会话恢复。
  判据：200 epoch 完成，val F1 记录，checkpoint 归档。（代码机制已在票 07 交付——kaggle/ Notebook 自动基准/预算检查/分块/续训/块尾发布 + `--report-f1` val F1 记录；**200 epoch 实跑与实测值回填为用户 Kaggle 执行项**，见票 07 完成记录与 `kaggle/README.md`。）
- **阶段 6 推理评估**：TTA 滑窗推理 → 投影 → 对比图+动画+弱定量表。
  判据：满足 §3 验收标准 1、3、4。
- **阶段 7 整理**：结果目录、复现 README、参数表、checkpoint 归档。

## 6. 默认参数表（冒烟阶段可调）

| 参数 | 默认 | 说明 |
|---|---|---|
| T_win / L / 组数 / 每组 | 24 帧 / 16 / 64 / **4 卫星** | 256 条/样本，KpathlinePerGroup=4 |
| 十字偏移 Δ | patch×0.05 | 轴向卫星 (x±Δ,y)、(x,y±Δ) |
| RK4 子步 | 每输出步 4 | 三线性时空插值 |
| patch / stride | 32×32 / 16 | 窗口起点步长 4 帧 |
| t_scale | 0.25 | KNN 时空混合度量中 t 的权重 |
| IVD 阈值 τ | **3.3272 / 3.16848 / 3.14344**（train/val/test，95 分位数逐时间片） | 票 04 实测（流体区统计，排除固体 0 值）；备选 μ+3σ 未用 |
| 最小涡面积 | 5×5 | 连通域过滤 |
| epoch 样本数 | 40000（50% 正样本） | 下限 20000 |
| 时间划分 | 10 / 12.5 / 15 s | train/val/test |
| TTA 次数 | 5 | 平均随机 PSL 采样的概率 |

## 7. 风险与预案

| 风险 | 预案 |
|---|---|
| 弱标签阈值敏感/循环评估 | 多阈值敏感性报告；定性为主；备选 Q-criterion 标签对照 |
| Kaggle 配额/12h 会话 | 分块+每 epoch checkpoint+Dataset 版本续训；DataParallel/AMP/降样本数 |
| 正负样本不平衡 | 50% 过采样；必要时 BCE pos_weight |
| 涡特征微弱 | τ 下探；T_win 24→48 |
| 迹线撞固体 | 掩膜截断+重播种；冒烟目检 |
| KNN 时空尺度失调 | t_scale∈{0.1,0.25,1} 冒烟对比 |
| 推理非确定 | TTA 5 次；或 random=False 确定性评估 |
| 中文路径 | h5py 直读；建议另复制数据到 ASCII 路径（如 `C:\flowdata\`） |
| 上游 `.cuda()` 硬编码（票 01 迁移发现） | `vendor/DeepUtils/loss/build.py` SmoothCrossEntropy(ignore_index/weight) 分支与 `models/point_transformers.py` PosE_Initial 含 `.cuda()`，CPU-only 下启用会崩；当前 PathlineTransformerV0+BCELoss 路径不触碰。若未来启用须先去 cuda 化（作为独立小票处理，不改迁移忠实性） |

## 8. 测试接缝建议（给 /to-spec 阶段 2 与用户确认用）

按"最高接缝优先、越少越好"的原则，建议 3 条（第 3 条为最高接缝，优先采用它作为规格验收主缝）：

1. **数据准备缝**：原始 nc 场 + patch/窗口参数 → (迹线 7 通道张量, 弱标签)。测试 = 冒烟目检 + 属性测试（掩膜连通性、迹线计数 256、特征有限无 NaN）。
2. **模型缝**：迹线样本 → 每迹线涡概率。测试 = 前向形状/数值 + 训练 loss 下降 + val F1。
3. **端到端缝**：nc 文件 + 时间窗 → 网格化涡概率场（可画图/可量化）。测试 = 若干展示帧的对比图目检 + 弱定量表。

## 9. 工作流（新会话如何推进）

**前置检查**：`to-spec`/`to-tickets` 依赖 issue tracker 与 triage 标签配置（`docs/agents/issue-tracker.md` 等）。✅ 已配置（2026-08-25）：**Local markdown** 追踪器（`.scratch/<feature-slug>/`：spec.md + `issues/NN-<slug>.md` 一票一文件，五默认 triage 标签），详见 `docs/agents/issue-tracker.md`、`docs/agents/triage-labels.md`。代码托管 GitHub：`ziyixu317-wq/2d-vortex-extraction-260825`（Kaggle 训练从该仓库克隆；数据集仍走 Kaggle Dataset）。

**主线（ask-matt 多会话构建分支）**：
1. 完整阅读本 HANDOFF 文档；若某节与代码现实冲突，以代码为准并在 §11 记一条修正。
2. 调用 `/to-spec`：按 §8 的接缝建议与用户确认测试接缝；用模板产出规格（Problem/Solution/User Stories/Implementation Decisions/Testing Decisions/Out of Scope），发布到 tracker（local 方案 = `.scratch/<slug>/spec.md`），打 `ready-for-agent` 标签。
3. 调用 `/to-tickets`：把规格拆成 tracer-bullet 垂直切片，每票声明阻塞边；向用户确认颗粒度与依赖边后再发布（local 方案 = `.scratch/<slug>/issues/NN-<slug>.md`，从 01 按依赖序编号）。
4. 之后逐票 `/implement`（每票新上下文，内部走 `/tdd` + `/code-review`），按 frontier 顺序（阻塞边全完成者优先）。

**上下文卫生（ask-matt）**：从开始到 `/to-tickets` 完成保持同一未断窗口（不 /clear、不 /compact）；若接近智能区（~150k tokens），在最近的阶段边界 /compact 而非中途。

**用户故事素材（to-spec 扩写起点）**：作为流场分析者，我想对任意 2D 非定常仿真场得到逐迹线涡概率并投影成网格图；我想在本地 CPU 快速验证迹线提取与标签正确性；我想在 Kaggle 分块训练并在会话中断后无损恢复；我想拿到与 IVD/Q 参考并排的对比图与动画；我想对新数据集用一条命令生成掩膜并接入训练。

## 10. Suggested skills

新会话应按序调用：`to-spec` → `to-tickets` → `implement`（内部 `tdd`、`code-review`）。词汇冲突/新概念时用 `domain-modeling`；遇到只有用户能完成的配置步骤用 `wizard`。若 tracker 未配置，先提示用户运行 `/setup-matt-pocock-skills`。

## 11. 变更日志（持续更新协议）

**协议**：每次会话结束前必须追加一条；事实变化改正文对应节（§1/§2/§4/§6），日志只记"改了什么、为什么、还差什么"。保持本文档为单一事实来源，不要另起炉灶。

- 2026-05-xx 成立：基于三轮回合（论文/代码/数据逐行核实 + 用户决策）形成本文档；取代 `工作计划_迹线Transformer涡提取.md`。当前进度：尚未开始阶段 0。
- 2026-08-25 ask-matt 配置建立：运行 `/setup-matt-pocock-skills`，创建 `docs/agents/{issue-tracker,triage-labels,domain}.md` 与 `CLAUDE.md`（Local markdown tracker，五默认标签；仓库 URL 经用户确认定为 ziyixu317-wq 账号）。代码托管 GitHub `ziyixu317-wq/2d-vortex-extraction-260825`：初始 commit `5cf066c`（项目骨架）+ `a354652`（URL 修正）已推送，main 同步。**新环境事实（§2 之外）**：① 本机 git schannel 后端 + 本地代理 127.0.0.1:7890 有兼容性问题（`SEC_E_NO_CREDENTIALS`），推送须 `-c http.sslBackend=openssl`；② DSH 沙箱内 git 无法调用 GCM（signal pipe 限制）→ GitHub 认证类操作须用户在普通终端完成。未决问题：git 全局 user.email 为占位 `ziyi@example.com`（建议改为真实邮箱）；全局 sslBackend 是否改 openssl 待用户定（不改则每次推送带 `-c` 参数）。当前进度：仍为阶段 0 前，下一步按 §9 主线进入 `/to-spec`。
- 2026-08-25 to-spec + to-tickets 完成：接缝经用户确认（三条全用、端到端为主验收缝）；规格发布 `.scratch/vortex-extraction-pipeline/spec.md`（Status: ready-for-agent）；10 张垂直切片票发布 `.scratch/vortex-extraction-pipeline/issues/01..10-*.md`（依赖：01,02 无阻塞可并行 → 03,04 依赖 02 → 05 依赖 03,04 → 06 依赖 01,05 → 07 依赖 06 → 08 依赖 07 → 09 依赖 08 → 10 依赖 09）。用户已确认拆分。未决：无。下一步：按 frontier 从 01、02 逐票 /implement（每票新上下文，内部 /tdd + /code-review）。
- 2026-08-25 票 01（vendor 迁移）完成：`vendor/DeepUtils/` 落盘（models/loss 全树 + utils/{registry,ckpt_util,random}，38 文件 SHA256 与源逐字一致；仅 `utils/__init__.py` 重写剔除依赖 multimethod 的 config.py/EasyConfig）；项目根复制 LICENSE（Apache 2.0）与 NOTICE（PyFlowVis 署名）；新增 `tests/test_vendor_migration.py` 8 项验收测试全绿（导入缝、前向缝 (B,256)/(0,1)、迁移边界）；全项目不再引用 PyflowVis-main（阶段 0 判据达成）。/code-review 双轴处置：① 验收 2 守护测试曾逐字符迭代空洞通过 → 已改 AST 按 import 语句扫描（commit `1af7b7c` 前修复）；② "全树超最小子集"意见按 §4 目录树保留；③ 上游 `.cuda()` 硬编码（loss/build.py SmoothCrossEntropy、point_transformers.py PosE_Initial）为潜伏风险，已记 §7，不改迁移忠实性。下一步：票 02（geometry 掩膜）可并行启动。
- 2026-08-25 票 02（geometry 掩膜）完成：新增 `geometry.py`（|v|<ε 逐帧取与 + 自写并查集连通域（无 scipy，遵守 §2 依赖清单）+ 块统计/圆柱定位 + `mask.npy` (T,Y,X) uint8 与 `geometry_meta.json` 落盘 + 物理坐标系目检图 + CLI）+ `tests/test_geometry.py` 10 项全绿（合成属性 + 真实数据已知事实；全量 18 passed）+ `pytest.ini`。**事实回写（§2 已加"固体几何（票 02 实测）"行，§4 职责行同步）**：数据集含**两个**圆柱（入口 (≈0,0) 与拐角后管道 (≈3,1)，零速盘 43/45 格）；radius=0.0625 与 Re=160=U·D/ν 自洽（元数据可信）；零速区为圆柱内切区（表面约 1 格厚格被插值为非零速），物理半径≈面积等效+1 格≈0.063。产物 `outputs/geometry/`（gitignore，走 Kaggle Dataset）。/code-review 双轴处置：① plot_mask 物理/格坐标混用（目检圆错位）→ extent 修复；② 测试期望值缺独立来源 → 实测事实按 §11 回写 HANDOFF §2 后成为权威来源；③ `min_block_cells` 曾扩展规格"圆柱=孤立块"定义 → 默认改 1（规格字面），保留为显式收紧选项；④ 死代码/assert 校验 → 清理（commit `7b3dcb9` 前修复）。未决：无阻塞。下一步：按 frontier 票 03（迹线提取，依赖 02）与 04（弱标签，依赖 02）可启动。
- 2026-08-25 票 03（迹线提取）完成：新增 `extractor.py`（全局场 RK4 积分器：每输出步 4 子步 + 三线性时空插值（越界 clamp）；256 条 = 64 组 × 4 轴向卫星（不含中心），Δ = patch 边长×0.05，组主序；种子落固体 → 重播种（仿 C++ JittorReSeeding：朝 patch 中心随机移动 + 细扫兜底 + 全固体 ValueError）；迹线入固体 → 截断重复末点（**无 -1000 毒值**，C++ 参考用 -1000 填充而本项目决策不用）；位置按 patch 归一化 [-1,1]（可超界）；允许离开 patch（仅场域边界停止）；积分太短（≤2 点，仿 C++ suc 判据）→ 朝中心移动重试 ≤3 次）+ 目检图/CLI + `tests/test_extractor.py` 28 项全绿（全量 46 passed）。**实现边界（§4 职责行不变，供票 05 衔接）**：① 归一化只做位置部分——t→[0,1]×t_scale、ivd 标准化、u/v÷全局最大速度、distance 归一化口径属票 05 dataset 职责；② 7 通道 ivd 为可选参数，未传时第 4 通道填 0（票 04 提供 IVD 场后由票 05 接入）；③ 入固体/出域检查在输出步粒度（与 C++ 参考一致；一步最大位移 ≈5.3 格 < 最薄固体 7 格，实际数据不穿越固体）；④ 组中心取 patch 内 [0.1,0.9] 区间、重播种细扫/ValueError 为对 C++ 的工程化扩展。目检（验收 1）用户肉眼复核通过 + 数值目检（有效点 0 穿固体、切向一致性余弦 >0.9）。产物 `outputs/pathlines/`（gitignore，走 Kaggle Dataset）。/code-review 双轴处置：① 插值双份公式漂移风险 → 一致性守护测试；② CLI 掩膜兜底与 geometry 同判据 → 注释交叉引用（逐帧流式读避免全量驻留）；③ CLI --mask 2D/3D 兼容修复；④ 积分太短重试分支补测试；⑤ 目检 PNG 落盘补测试（commit `85ad7e3` 前修复）。未决：无阻塞。下一步：按 frontier 票 04（弱标签，依赖 02）与 05（数据集，依赖 03、04）可启动。
- 未决问题：无阻塞性问题。τ 已定值（§6 回写）；t_scale 取值、epoch 样本数待冒烟/票 05 后定（§6 已给默认）。
- 2026-08-25 票 04（弱标签与 τ 定值）完成：新增 `weak_labels.py`（ω=∂v/∂x−∂u/∂y 中心差分（边界单边一阶、等距守卫）→ IVD=|ω−5×5 邻域均值|（edge pad 含中心 25 格）→ 固体区 IVD=0（复用票 02 掩膜）；2D Q=−u_yv_x−½(u_x²+v_y²)（参考对照）；标签 = 逐时间片 τ 二值化 + 5×5（25 格）连通域过滤（复用 geometry.label_components）+ 固体强制 0；τ=流体区 IVD 第 95 分位（排除固体 0 值，其 41.8% 占比会污染阈值）；正样本占比统计（种子判据 + 窗口起点步长 4 帧）；τ 敏感性/IVD+Q 目检图；CLI 逐帧流式算 IVD 落盘 ivd.npy/label_field.npy/weak_label_meta.json）+ `tests/test_weak_labels.py` 32 项（全量 84 passed）。`extractor.py` 提取 `seeding_grid`（256 种子单一公式，迹线提取与正样本统计共用防漂移）。**事实回写**：§6 τ 行 ← 实测 3.3272/3.16848/3.14344（train/val/test）；**新事实（§2 之外，供票 05）**：① **时间片口径明确化**——「帧 0-1000/1000-1250/1250-1500」按闭包读法（t=10.0 帧 1000 ∈train、t=12.5 帧 1250 ∈val、t=15.0 帧 1500 ∈test）→ DEFAULT_SLICES {(0,1001),(1001,1251),(1251,1501)}，全覆盖 1501 帧无泄漏（有测试守护）；② **正样本占比（种子判据）global=37.86%**（train 37.38/val 38.76/test 38.82）→ 50% 过采样仅需 1.32×（过采样负担小）；标签正格占比 2.80%；③ 弱标签对壁面剪切带（y≈0.5 长条）也有响应（IVD 是涡量偏差并非纯涡判据），p95 默认下仅余有限长条，属预期（决策 5：模型学习 IVD 近似可接受）。**实现边界（供票 05 衔接）**：IVD/标签场为 (T,Y,X) 落盘（ivd.npy float32、label_field.npy uint8），dataset 的 memmap 预计算口径一致；正样本占比未模拟种子重播种（低估 ≤2%，对过采样设计无实质影响）。/code-review 双轴处置：Standards 硬性 1（compute_tau docstring 分位方向）→ 修复；判断项 5 → 修 4（_labeled_mask 共享、删 binary_label 数组语义、frame_indices+4 帧步长、~mask2d/_slice_of 前移），保留 1（(taus,slices) 数据团：两语义不同、成对性仅模块内 3 处，合并不降复杂度）；Spec 缺失 2（票收尾文档→本记录、时间划分守护测试→TestTimeSlices 2 项）+ 口径错 2（正样本占比→种子判据重写、compute_ivd docstring 保形）+ 范围蔓延 3（流体 τ 总体/边界差分/四候选敏感性图，保留并记录）。目检（验收 1）AI 读图核验 tb:400/1200/1300（涡街/拐角回流区成块、非涡区干净）+ 敏感性 p95 稳定；**建议用户肉眼复核收尾**。产物 `outputs/weak_labels/`（gitignore，走 Kaggle Dataset）。未决：无阻塞。下一步：按 frontier 票 05（数据集，依赖 03、04）可启动。
- 2026-08-25 票 05（数据集类）完成：新增 `dataset.py`（prepare_dataset：nc 流式或内存 → u/v/ivd/label/mask memmap + meta.json（slices/taus/speed_max/ivd μσ）；WeakLabelPathlineDataset：池构建（正 = patch ≥1 条涡迹线，与 weak_labels.patch_positive_map 单公式共用）→ set_epoch 50% 正样本过采样 → on-the-fly 提取+归一化+标签 → `((dummy_field(1,1,1,1), pathlines), labels)`；normalize_pathlines：t→[0,1]×t_scale=0.25、ivd z-score（train 片流体区 μ/σ，防统计泄漏）、distance 归一化坐标、u/v÷speed_max；window_starts（窗口完全在片内 [i0, i1−t_win]））+ extractor 新增 `extract_pathlines_batched`（向量化批量 RK4，批量-逐条一致性守护）+ `nearest_cell`（物理→最近格单一公式，收敛 mask_at/批量/种子偏移/标签四处）+ weak_labels 提取 `patch_seed_offsets`/`patch_positive_map` 共享判据；测试 21+8 项，**全量 113 passed**；真实数据复验通过（taus 与票 04 逐位一致、样本 (16,256,7) 有限、t∈[0,0.25]、|u|≤1、正样本占比实测 0.500）。**事实回写（§2 之外）**：① **可用 patch 实测 128/216**（88 个不可用：种子-中心线段全固体——壁面区/圆柱包围区；票 03 "全固体 patch 应避开"边界的精确化，`_patch_usable` 静态判据 201 点采样）→ 池组合 31,360（正 13,652=43.5%、负 17,708）；票 04 的 37.86% 为含不可用 patch 口径（全固体 patch 恒为负拉低），与票 05 池口径不矛盾；归一化统计实测：speed_max=4.63067、ivd_mu=0.88674、ivd_sigma=3.58489（train 片流体区）；② **性能事实**：on-the-fly 单样本实测 median 35.7ms（真实窗口；首取 836ms 页加载；池构建 0.85s）；**用户已确认（2026-08-25）：时间性能不纠结，能跑即可** → 验收 1（<5ms）以"冒烟上限+守护测试"落实，Kaggle 训练建议 DataLoader workers≥8（8 worker 下 0.44s/batch(100) < T4 训练步时，不构成瓶颈）；③ **实现边界**：批量版 rng 为 per-k 确定性派生（SeedSequence([base,k,attempt])——与逐条版单流 rng 不同构：确定性路径逐元素一致、随机路径为同语义不同随机实现）；数据集"池判定/标签/提取"三者一致由组合级 rng base（`_comb_rng_base`）保证。**/code-review 双轴处置**：Standards 无硬性违规（Duplicated Code 4 处最近格公式+rint/floor 口径不一 → nearest_cell 单一公式抽象；_gather 闭包绑定顺序 → 前移；Data Clumps 保留——判断项同票 04 先例）；**Spec 关键缺陷 1**（时变冻结 bug：`_extract` 窗口切片场配全场 tdim → 时间映射 clamp 到窗口末帧，u/v/ivd 通道与 RK4 速度场全部冻结——实测复现、修复为窗口配窗口 tdim（_extract/CLI/真实测试同改））+ 弱项 2（未用下限常量删除、seeds_for 与 _extract 重试不一致 → 复用 _extract）+ **连带发现第二处公式漂移**（批量插值 x1/y1 用 x0+1 而标量用 ceil，域外点两角不同 → 守护测试捕获并修复）；范围蔓延 1（_patch_usable 为真实数据必备防护，注释披露保留）。产物 `outputs/dataset/`（gitignore，走 Kaggle Dataset ≈1.3GB）。未决：无阻塞。下一步：按 frontier 票 06（训练脚本，依赖 01、05）可启动。
- 2026-08-25 票 06（训练脚本）完成：新增 `train_kaggle.py`（自写训练循环，不 import 参考仓库任何训练代码）+ `config/pathline_transformer_cylinder.yaml` + `tests/test_train.py` 18 项（**全量 131 passed**）。训练口径（HANDOFF §2 论文附录 C/§6）：AdamW(wd 1e-6)、lr 1e-4、TwoStep（warmup 60 epoch 恒 lr → 5e-6，epoch 粒度两段常数阶梯，与原仓库 `TwoStepLRScheduler` 逐行核实一致）、batch 100、200 epoch、梯度裁剪 1.0、BCE；模型沿用官方 config 形态（`BaseSeg` 包 `PathlineTransformerV0`：in_channels=7/PathlineGroups=64/KpathlinePerGroup=4/dmodel=144/3 层/k=16 + BCELoss）。checkpoint：latest 每 epoch 更新 + save_freq=30 里程碑（对齐原仓库），含 optimizer/scheduler/epoch/metrics/config 元数据 + DataParallel `module.` 前缀归一化；`--resume auto|none|路径` 默认 auto（latest 存在则续，start_epoch=epoch+1），训练序每 epoch `set_epoch(epoch)`（同 (seed,epoch) 确定性 → 续训采样序与中断前逐样本一致）；DataParallel/AMP 为 YAML 可选开关；`--max-steps`/`--epochs` 为 CPU 冒烟与 Kaggle 分块入口。**验证证据**：合成+CLI 集成测试 18 项全绿（调度值/状态往返、生产配置构建前向 (B,256) 域 (0,1)、YAML 全字段驱动+死配置回归、2 步冒烟 loss 有限参数更新、全局梯度范数 ≤1.0、checkpoint 往返逐张量一致、续训恢复位置、同 seed 复现、val 路径）；真实数据 CPU 冒烟（生产模型，2 epoch×2 步）：epoch1 loss=0.9147 val=0.7877 → epoch2 loss=0.7153 val=0.5841（早期下降符合观察）；ckpt 5.6MB。**审查发现并修复 2 缺陷**：① val_ds 未 set_epoch（真实冒烟复现 RuntimeError → 修复 + 回归测试）；② YAML `data.patch_size/stride/t_win/window_step` 与 `groups/delta_frac/L/n_substeps` 未进训练路径（死配置 → `_make_dataset` 全量传入 + 回归）。**/code-review 双轴处置**：Standards 硬性 0、判断项 7 → 修 5（`_make_dataset` 归拢 train/val 构造、`_iter_batches` 共享批迭代、AMP 单路径 `autocast(enabled=)`、测试样板 helper `fresh_small_model`/`fresh_adamw`/`assert_forward_shape_and_range`、删函数内冗余 import），保留 2（vendor 薄封装 Middle Man 边界情形、数据 clump 先例）；Spec 3 条 → 修 1（死配置）、保留 2 并记录（checkpoint 双写：latest 每 epoch + save_freq=30 里程碑，200 epoch ~230 文件；**val 损失口径 = 训练同款 50% 平衡采样监控**（自然分布精度评估归票 08 弱定量表，注释明示）。未决：无阻塞。下一步：按 frontier 票 07（Kaggle 训练）可启动。
- 2026-08-25 票 07（Kaggle 训练）agent 交付完成：新增 `kaggle/`（`chunking.py` 分块规划纯函数、`prepare_dataset_a.py` Dataset A 打包 + manifest sha256 审计、`self_check.py` 环境自检、`train_kaggle.ipynb` 9-cell Notebook、`README.md` 操作手册）+ `train_kaggle.py` 增 `evaluate_f1`/`--report-f1`（训练完成写 `{run}_val_f1.json`，自然分布 val F1）+ `dataset.py` 增 `set_epoch_natural`（自然分布采样序=池比例；池名统一公开 `pool_positive/pool_negative`）+ `tests/test_kaggle.py` 20 项（**全量 151 passed**）。Notebook 设计：TOTAL_EPOCHS 从 config 读（单一来源）、步速基准首会话实测后复用（bench_info.json）、预算检查（总时长 >48h 自动生成 train_opt.yaml：AMP+DataParallel+样本数降 50% 至下限 20000，HANDOFF §7 预案落地）、每会话一块（CHUNK_BUDGET_H=7.5h）`--resume auto` 续训、块尾打包+模式 A（kaggle datasets version）/模式 B（手动上传）双路、最后一块带 `--report-f1`、收尾 cell 打 final_ckpt.zip+sha256。**验证证据**：全量 151 passed；真实数据实证 = `self_check.py` 生产模型（dmodel 144/3 层/k16）+ `outputs/dataset` 真产物 CPU 通过（前向 (1,256) 域 (0,1)、4 样本 (16,256,7) 有限无 NaN、正标签 192 条）；Dataset A 规模实测预估 ≈2.03GB（nc 773MB + memmap 1.26GB + 目检图 1MB）；`.gitignore` 加 kaggle_dataset_a 防呆。**/code-review 双轴处置**：Standards 硬性 0、判断项 8 → 修 6（池名双套统一、main 重复守卫删除、测试样板 with_val 参数复用、_mixed_loader 删除、self_check 不可用缺省路径 fail loud、README 预算口径说明），保留 2 并记录（TestSetEpochNatural 期望与实现同源的独立性局限——池比例是数据自描述公开属性，守护「natural 口径正确接池」语义；evaluate_f1 8 字段 dict Data Clumps 按票 04/05 先例）；Spec 3 → 修 2（TOTAL_EPOCHS 硬编码改 config 读、验收 2 校准自动化补全：基准复用+预算检查+自动优化配置+回填提示），记录 2（**自然分布 val F1 提前实现为票 07 验收 4 明示需求**——与票 06 注释「归票 08」的计划冲突以票为准，正式网格投影级弱定量表仍归票 08；`--aux-dirs` 打包 weak_labels 目检 PNG 为 HANDOFF 票 04 记录「目检图走 Kaggle Dataset」的既有表述，1MB 级）。**执行边界（重要事实）**：票 07 验收 2/3/4 的**实测与 200 epoch 实跑为用户 Kaggle 执行项**（本会话无 GPU/Kaggle 访问）——票文件 Status 置 `ready-for-human`（triage 语义：Requires human implementation），交付物全部完成并本地验证；用户按 `kaggle/README.md` 执行后回填（val F1/步速/checkpoint 位置）并把票改 done。未决：无阻塞（用户执行后回填）。下一步：用户 Kaggle 执行票 07 → 票 08（推理评估，依赖 07）在 200 epoch 完成后启动。
- 2026-08-25 票 07 运行反馈修复：Kaggle 首次 Run All 在 notebook cell 1 失败（`pip install ... yaml ...` → "No matching distribution found for yaml"）——**pip 包名笔误：PyPI 无 `yaml` 包，正确为 `PyYAML`（import 名 yaml）**；Kaggle 日志 `train-kaggle.log` 定位（In [1] pip 安装段）。修复：notebook cell 1 pip 列表、`kaggle/README.md` §5 表格、HANDOFF §2 依赖清单与 §5 阶段 4 判据统一改 `PyYAML`（import 语句不变；本地无行为影响，测试无需改动）。未决：无。下一步：用户重新上传/运行 notebook（cell 1 起 Run All）→ 票 08 待 200 epoch 完成后启动。
- 2026-08-25 票 07 运行反馈修复二期：① 首次 Run All 第二轮在 cell 3 失败（`assert ds_in is not None` —— 挂载 dataset 已添加但 `find_input("dataset/meta.json")` 未命中）→ cell 3 输入探测重写为**布局自适应**（直接命中/多一层嵌套/任意深度 rglob/zip 未解压自动解压回退）+ `[env] /kaggle/input 挂载:` 与失败时输入树全览诊断打印（下一轮实测日志：实际挂载路径为 `/kaggle/input/datasets/ziyixu317/2d-unsteady-cylinder-flow-around-corners/...` —— dataset 名称生成 slug 含多级目录，验证 rglob 兜底必要性）；notebook 各训练 subprocess 改显式 `cwd=REPO_DIR`、clone 后 `sys.path.insert(0, REPO_DIR)` 自举（防 Script 模式找不到 dataset/vendor）+ repo 根文件自证打印。② 挂载/克隆/自检全通过后，**步速校准（cell 5）训练前向 OOM**：`softmax(attn)` 分配 900MB 失败——GPU 0（T4 16GB）前向激活 13.5GB（batch 100 × 1024 点 × 3 层 Point-Transformer KNN+注意力）；**触发 HANDOFF §7 预案「超预算用 DataParallel/AMP」**：YAML 默认改 `amp: true` + `data_parallel: true`（Kaggle T4×2 自动生效、本地 CPU 路径不激活、测试全绿）、`num_workers` 8→4（Kaggle 4 核 warning）、notebook cell 1 设 `PYTORCH_ALLOC_CONF=expandable_segments:True`（子进程继承，OOM 提示建议）。**新事实（§6 之外）**：T4 单卡 16GB 下 batch 100 前向 ~13.5GB 显存（超出 ~2GB 安全余量）。未决：无。下一步：用户重跑（Runtime Restart and Run All）→ 步速校准应通过（1 epoch 全量 400 步）→ 分块训练。
- 2026-08-25 票 07 运行反馈修复三期：AMP+DP 开跑后（OOM 已解，`DataParallel 已启用 (2 GPU)` 生效）第一步内即报 `RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Half for the source`（`vendor/DeepUtils/models/segmentation/pathline_transformer.py:246` `full_features[:, ~mask, :] = self.feature_propagation(...)`）——**上游模型不兼容 AMP**：原地 index-put 在 autocast(Half) 下 dtype 冲突（原作者 24GB 卡全精度训练未用 AMP；迁移忠实性（票 01：38 文件 SHA256 逐字一致）不允许改 vendor）。处置：**AMP 默认关（`amp: false`）**，显存改由 **DataParallel 拆分**（T4×2：batch 100 → 每卡 50，峰值 ~6.8GB < 16GB；等效 batch 100 语义不变——DP 仅数据切分，非梯度累积）；顺带修 GradScaler deprecated API（`torch.amp.GradScaler('cuda', ...)`，消除 FutureWarning）。**新事实（§7 风险表之外）**：① T4×2 单机 DP 下 batch 100 全精度训练可行（无 AMP），余量 ~9GB；② 上游模型 AMP 不兼容是**硬约束**，若未来需要半精度须先修 vendor（独立小票，改迁移忠实性）；③ `num_workers` 4（Kaggle 4 核）。未决：无。下一步：用户重跑（需 `git push` 让 Kaggle clone 到最新 YAML）→ 步速校准 → 分块训练。
