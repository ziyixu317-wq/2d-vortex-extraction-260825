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
- 最终依赖清单：torch、numpy、h5py、yaml、matplotlib、tqdm（本地 Python 3.12 已装除 torch 外全部；Kaggle 自带 torch）
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
├── train_kaggle.py              # 自写训练脚本（TwoStep、断点续训、可选 DataParallel/AMP）
├── evaluate.py                  # TTA 推理、网格投影、对比图/动画、弱定量表
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
  判据：单样本生成 <5ms；train/val/test 划分无泄漏。
- **阶段 4 Kaggle 上传**：Dataset A = nc 文件；Dataset B = 整个 pipeline 目录。
  判据：Notebook 里 `pip install h5py yaml matplotlib tqdm` 后能 import vendor 并加载数据。
- **阶段 5 训练**：先 1 epoch 冒烟实测每步耗时 → 校准 epoch 样本数（T4 单卡预计 0.8~1.3s/步 → 80k 步约 18~29h，超预算则用 DataParallel/AMP/降样本数）；**Kaggle 会话硬上限 12h** → 分块 ≤8h，每 epoch checkpoint，块尾打包为 Kaggle Dataset 新版本，下次会话恢复。
  判据：200 epoch 完成，val F1 记录，checkpoint 归档。
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
| IVD 阈值 τ | 95 分位数（逐时间片） | 阶段 1 校准；备选 μ+3σ |
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
- 未决问题：无阻塞性问题。τ 具体值、t_scale 取值、epoch 样本数待冒烟后定（§6 已给默认）。
