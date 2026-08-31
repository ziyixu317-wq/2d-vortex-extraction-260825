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
3. **训练**：Kaggle T4×2，**200 epoch**；本地（核显、torch 2.10.0+cpu、无 CUDA）只做 CPU 冒烟。（2026-08-29 多数据集线**结算口径改为 130 epoch**：43-86 loss 已趋稳（0.0873→0.0811），用户拍板 130 结算、原 200 中止；§5/§6/§11 已回写。单数据集线（暂缓）仍 200。）
4. **评估**：定性对比（IVD/Q-criterion 为参考）+ 弱定量表；不涉及 Vatistas 验证集。
5. **不做 ivd 遮除消融**：IVD 是公认高精度涡判据，模型学习其近似是可接受且预期的。
6. **迹线口径 = 256 条**：64 组 × 4 卫星点（不含中心），`KpathlinePerGroup=4`，对齐论文与发布数据。
7. **代码组织**：独立自包含工程，迁移（复制）所需代码进本项目，不依赖原始 `PyflowVis-main`；保留 Apache 2.0 署名。
8. **geometry 掩膜是逐数据集预处理**，不进入模型输入，不影响多数据集泛化（后续会加更多仿真数据集）。（2026-08-25 票 07 延伸落实：7 数据集联合训练——`prepare_multi.py` 逐数据集预处理 → `MultiDatasetPathlineDataset` 联合采样池，frac 60/40 逐数据集时间划分，配置 `config/pathline_transformer_multi.yaml`。）

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

### 多数据集盘点（票 07 延伸实测 2026-08-25，`CFD数据集/` 下 6 个新数据集 + 既有 pipedcylinder2d）

> 注（2026-08-29）：**forceddampedduffing2d 已移出训练池**（用户判定该数据集有问题；
> `config/pathline_transformer_multi.yaml` roots 7→6，本行保留作盘点记录）。
> **问题实证（2026-08-29，三项）**：① ParaView 打开显示 1×1×1 退化网格（# of Cells: 1、
> Bounds 全 0、Data Arrays 仅余 alpha/beta）+ `vtkPVImageSliceMapper: Incorrect
> dimensionality`——**reader 路径问题**（被当 Image 而非 rectilinear grid；文件本身
> netCDF4/h5py 读写正常、与参考数据集逐字段同构，无结构缺陷）；② **时间维冻结**
> （u[0]=u[50]=…=u[511]，全帧扫描 max|du|=0.00e+00——512 帧完全相同，tdim/dt 元数据
> 齐全但时间演化丢失）；③ **IVD/标签退化**（ω∈[-11.8,-0.002] 全域单一负涡带、无 5×5
> 尺度突变 → IVD 处处≈0.006 常数（p75=p95=0.006）、max 仅 0.148 且只在域边界；
> 标签 p85 正格 8.3% **全部落在 x=±2 边界条带**（正格最小边界距离 0 格）= 边界差分
> 伪影，未捕获任何涡结构）——**与官网示意（LIC/FTLE 两个动态椭圆涡）不符**。
> **用户"数据有问题"判定成立且实质**：对"非定常迹线 + IVD 弱标注"管线该文件无有效
> 信息（静态场伪时间维 + 边界伪影正样本 + 标签无涡语义），入池属负价值。
> 用户保持移出决策（2026-08-29 已执行 roots 7→6）。另：此前 7 数据集盘点只覆盖
> 统计量、未逐数据集验证"时间演化 + 涡结构符合官网"——duffing 暴露此盲点
> （§7 已建议对 6 个既有数据集补做体检）。Kaggle Dataset A 无需重传——config root 即
> 训练池唯一来源，cell 3 多链接的目录不参与池构建/训练（本地打包目录可留可删）。

| 数据集 | (T,Y,X) | 域 / t 范围 | speed_max | 固体（全帧取与 \|v\|<1e-5） | τ（p85，train/test frac 片） | 标签正格（流体区） |
|---|---|---|---|---|---|---|
| boussinesq | 2001×450×150 | x[-0.5,0.5] y[-0.5,2.5] / [0,20] | 1.254 | 242 格 1 块（≤1 圆柱，有 obstacle_pos/radius 元数据） | 0.0555 / 0.5955 | 14.3% |
| cylinder2d | 1501×80×640 | x[-0.5,7.5] y[-0.5,0.5] / [0,15] | 1.836 | 60 格 1 块（1 圆柱，Re/nu/radius 元数据） | 0.0545 / 0.1022 | 14.3% |
| doublegyre2d | 512×128×256 | x[0,2] y[0,1] / [0,10] | 0.471 | 4 单格噪声点（无实质障碍） | 0.0022 / 0.0024 | 15.0% |
| forceddampedduffing2d | 512×128×128 | [-2,2]² / [0,4] | 7.184 | 无 | 0.0060 / 0.0060 | 8.3% |
| fourcenters2d | 512×128×128 | [-2,2]² / [0,6.28] | 2.848 | 无 | 0.0240 / 0.0240 | 15.0% |
| jungtelziemniak2d | 500×200×450 | x[-3,7] y[-2,2] / [1.107,5.535] | 65.71 | 12 单格噪声点（无实质障碍） | 0.2189 / 0.2179 | 15.0% |
| pipedcylinder2d | 1501×150×450 | x[-0.5,5.5] y[-0.5,1.5] / [0,15] | 4.631 | 28213 格 4 块 2 圆柱（§2 上文） | 0.6408 / 0.6526 | 14.8% |

- 全部 h5py 直读（中文路径无碍）、等距网格、采样帧无 NaN；**IVD 量纲跨数据集差 ~300×**（doublegyre τ=0.0022 vs jung 0.2189 vs pipedcylinder 0.64）→ **τ 必须逐数据集同分位**（绝对固定值不可移植，票 07 延伸定案）；
- **时间片必须按帧比例划分**（frac）：jung 的 t 从 1.107 起、各数据集帧数 512~2001 且 dt 各异（0.0078~0.0196）——绝对秒数（10/12.5/15s）只对 pipedcylinder2d 有效；
- 多数据集预处理产物：`outputs/datasets/<名>/{geometry,dataset,previews}` + `outputs/datasets/multi_meta.json`（逐数据集 shape/slices/taus/统计汇总；≈3.8GB，gitignore 走 Kaggle Dataset）；
- 多数据集池（train，7 root 实测：正 78,351 / 负 82,757 组合，池构建 ~2s；联合池量远超每 epoch 2 万样本需求（50% 过采样仅取正池 1/8）。**2026-08-29 起 6 root（duffing 移除），池量相应减小，口径不变**）。

### 模型事实（`DeepUtils/models/segmentation/pathline_transformer.py`）

- 输入迹线簇 `[B, L=16, K=256, C=7]`；C = **[px, py, t, ivd, distance(距种子点), u, v]**
- 十字采样：8×8=64 组 × 4 卫星点 `(x±Δ,y),(x,y±Δ)`（**不含中心**），Δ = patch 边长×0.05；**组主序编组**（组 0 的 4 条在前）
- PSL：组置换 + 空间下采样（64→32 组）+ 时间下采样（L 16→8）→ N = 32×4×8 = **1024 点**
- 3 层 KNN Point-Transformer（k=16，相对位置编码 MLP）；全局池化 mean+max；特征传播回全部 256 条；sigmoid 输出每迹线涡概率；损失 BCE
- 训练超参（论文附录 C 原文核实 2026-08-25，Zotero T76G9Z3A）：AdamW(wd 1e-6)、**cosine lr 调度（warmup 5 epoch + 195 epoch，共 200）**、batch 100、ReLU（末层 sigmoid）；**硬件 = Intel Xeon Gold 6230R ×2 + NVIDIA A100（单卡），200 epoch ≈ 14 小时**；训练集 = 3000 合成稳态场 × 20 个 Killing 变换 = **60,000 非定常场**（9:1 train/val，Vatistas 参数拟合生成——非仿真数据）。**实现口径（与论文的已知偏差，按仓库 config 复刻）**：lr 调度为 TwoStep（warmup 60、二段 5e-6——原仓库 `TwoStepLRScheduler`，票 06 已核实；cosine 为论文原文；如需对齐论文须改调度器，属用户决策）
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
1. 预测涡区域与 IVD/Q-criterion 参考结构一致（涡街、拐角回流区），无明显碎裂噪声（**票 08 交付**：对比图/弱定量表已实现并经真实数据 smoke（130-epoch 权重，pipdycylinder2d test 帧 1300，涡街/拐角涡对应良好）；最终目检待用户复核）
2. 200 epoch 训练完成，checkpoint 归档，训练可跨 Kaggle 会话断点续训（**2026-08-29 多数据集线结算口径改 130 epoch**——用户决策，§11；单数据集线仍 200）
3. 交付：多个代表性时间步的对比图 + mp4 动画 + 弱定量表（对 IVD 阈值的 F1/IoU、涡面积占比、帧间连续性）+ 复现 README（**票 08 交付**：evaluate.py 产对比图/mp4（ffmpeg 缺失回退 gif）/弱定量表；复现 README 在 kaggle/README.md，票 07 已交付）
4. 多阈值敏感性报告（τ 的稳健性说明）（**已交付**：multi_tau_report，票 07 延伸）

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
├── weak_labels.py               # IVD（5×5 局部邻域均值）+ Q-criterion + 阈值标签 + 多阈值敏感性报告
├── dataset.py                   # WeakLabelPathlineDataset（h5py+memmap，on-the-fly）+ _DatasetStore/MultiDatasetPathlineDataset（多数据集联合池，票 07 延伸）
├── prepare_multi.py             # 多数据集逐数据集预处理驱动（geometry→IVD/label/τ→memmap+meta，票 07 延伸）
├── train_kaggle.py              # 自写训练脚本（TwoStep、断点续训、可选 DataParallel/AMP；票 07 增 --report-f1；票 07 延伸增 --f1-split/F1+IoU）
├── evaluate.py                  # TTA 推理、网格投影、对比图/动画、弱定量表
├── kaggle/                      # 票 07：Kaggle Notebook/打包/分块/自检/操作手册
├── config/pathline_transformer_cylinder.yaml  # 单数据集训练配置
├── config/pathline_transformer_multi.yaml     # 多数据集联合训练配置（票 07 延伸）
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
- **evaluate.py**：滑窗推理（stride 16 全场覆盖 + 贴边补全）→ TTA 5 次平均 → 网格投影（累积+计数平均消 patch 重叠）；展示帧模型概率面板用滑窗 prob_sw（patch 归一化与训练一致——加密种子 dense 全场归一化会分布偏移致输出退化，保留为独立工具+单测）；对比图（模型/IVD/Q/速度模+弱标签等值线）→ mp4 动画（ffmpeg 缺失回退 gif）+ 弱定量表（F1/IoU、标签参考与模型预测涡面积占比、帧间连续性）；底图用速度模场（不依赖 LIC 渲染器）。

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
  判据：200 epoch 完成，val F1 记录，checkpoint 归档。（代码机制已在票 07 交付——kaggle/ Notebook 自动基准/预算检查/分块/续训/块尾发布 + `--report-f1` val F1 记录；**200 epoch 实跑与实测值回填为用户 Kaggle 执行项**，见票 07 完成记录与 `kaggle/README.md`。**2026-08-29 多数据集线结算口径改 130 epoch**（86 续跑 44；loss 趋稳观察）——阶段 5 判据在多数据集线上以 130 + `{run}_test_f1.json` 为准。）
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
| IVD 阈值 τ | **0.6539 / 0.6512 / 0.6113**（train/val/test，**85 分位**逐时间片；单数据集 abs 划分） | 票 07 延伸：p95（3.3272/3.16848/3.14344）弱标签相比论文 Fig.6 列 1 捕获稀疏（正格 4.8%）→ τ 下探至 **p85**（正格 14.8%、5×5 过滤后 9.8 连通块/帧；多阈值敏感性表 `outputs/weak_labels/multi_tau/multi_tau_stats.json`：p90=9.9%/p80=19.7%）。**多数据集 τ 逐数据集各自**（§2 盘点表：boussinesq 0.0555、cylinder2d 0.0545、doublegyre 0.0022、duffing 0.006、fourcenters 0.024、jung 0.2189、pipedcylinder2d frac 片 0.6408/0.6526） |
| 最小涡面积 | 5×5 | 连通域过滤（票 07 延伸实测：过滤删的是 <9 格小碎片而非大涡结构，p85 下 ma9 与 ma25 覆盖差仅 0.15%，保留） |
| epoch 样本数 | **20000**（50% 正样本） | 2026-08-25 票 07 步速校准回写：T4×2 实测 ~5s/步（DP+全精度）→ 40000×200 epoch≈110h 超预算 4×；降半 + TF32 → 预计 35~55h；下限 20000 不变 |
| 时间划分 | abs：10 / 12.5 / 15 s（**仅 1501 帧×dt=0.01 适用**）；**多数据集 frac：各数据集帧前 60% 训 / 后 40% 测（无 val，逐数据集各自）** | train/val/test 无时间泄漏（守护测试）；frac 口径（fraction_slices）对帧数/时长各异的数据集通用（§2 盘点——jung t 从 1.107 起，绝对秒数不通用） |
| 多数据集池 | roots = **6 个** `outputs/datasets/<名>/dataset`（config/pathline_transformer_multi.yaml；**2026-08-29 duffing 移出**，原 7） | τ 与归一化逐数据集各自（各 meta ivd μ/σ、speed_max）；50% 过采样、t_scale 0.25 等与单数据集同参；留出评估 `--report-f1 --f1-split test`；**结算口径 130 epoch**（2026-08-29 用户决策，原 200） |
| TTA 次数 | 5 | 平均随机 PSL 采样的概率 |

## 7. 风险与预案

| 风险 | 预案 |
|---|---|
| 弱标签阈值敏感/循环评估 | 多阈值敏感性报告（**已交付**：multi_tau_report + 统计表/目检图，票 07 延伸）；定性为主；备选 Q-criterion 标签对照 |
| **T4×2 显存与速度约束（票 07 实测）** | batch 100 单卡前向 13.5GB > 16GB → **DP 默认开**（每卡 50 ≈6.8GB，等效 batch 100）；**AMP 默认关**（上游 `propagate_features` 原地 index-put Half/Float 冲突——迁移忠实性不改 vendor；如需半精度先修 vendor 独立小票）；**TF32 开**（训练界标准，1.5~2×）；实测 ~5s/步 → samples_per_epoch 20000（§6 已回写）+ 20000 时每 epoch 200 步 |
| Kaggle 配额/12h 会话 | 分块+每 epoch checkpoint+Dataset 版本续训；DataParallel/AMP/降样本数 |
| 正负样本不平衡 | 50% 过采样；必要时 BCE pos_weight |
| 涡特征微弱 | τ 下探（**已执行**：p95→p85，§6）；T_win 24→48 |
| 迹线撞固体 | 掩膜截断+重播种；冒烟目检 |
| KNN 时空尺度失调 | t_scale∈{0.1,0.25,1} 冒烟对比 |
| 推理非确定 | TTA 5 次；或 random=False 确定性评估 |
| 中文路径 | h5py 直读；建议另复制数据到 ASCII 路径（如 `C:\flowdata\`） |
| **逐数据集结构验证盲点（2026-08-29 duffing 教训）** | 7 数据集盘点只覆盖统计量（shape/τ/正格），未逐数据集验证「时间演化 + 涡结构符合数据集官网」——duffing 实锤：512 帧时间冻结（max\|du\|=0）、IVD 全域常数（p75=p95=0.006）、标签正格全为域边界伪影，与官网 LIC/FTLE 双涡不符。**预案：新数据集接入前补体检**（帧差扫描 max\|du\| + ω/IVD 结构图 + 与官网示意对照；对既有 6 数据集同样补做——2026-08-29 已排程，见 §11） |
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
- 2026-08-25 票 07 运行反馈修复四期（步速优化与中途评估入口）：用户确认「TF32 + samples_per_epoch 20000」组合（回答：论文训练硬件为 H100/3090 级（README 46 行 benchmark 目标；迹线积分 GPU CUDA 内核、C++ 离线预生成数据集），T4 fp32 8.1 TFLOPS 差 4-8×，故 1GB 数据 200 epoch 也需百小时级——计算密集负载而非体积问题）。实施：① `train_kaggle.enable_tf32()`（matmul/cudnn allow_tf32=True，数值仍 fp32 语义）+ main 调用；② YAML `samples_per_epoch` 40000→20000（§6 回写；每 epoch 200 步）；③ 中途评估入口验证：`--epochs == 续训进度` 时训练循环为空 → 仅 `--report-f1` 评估（**无需改代码的既有行为**，加守护测试）；④ 新增 `kaggle/preview_eval.py`（单帧滑窗 stride16+TTA n 次+投影（累积/计数平均）+ 模型/IVD/弱标签三联图——票 08 评估的简版子集，标注正式评估属票 08）+ `dataset.sample_at(y0,x0,frame)` 公开入口（与 __getitem__ 同路径，预览/票 08 滑窗共用）；⑤ notebook 追加 cell 9（可选中途预览）。**验证证据**：全量 155 passed（+4：TF32 flags、中途评估、投影字面量、预览端到端）；真实数据冒烟（随机初始化生产模型）：帧 1300 场 (150,450)、概率域 [0,0.754]、正格 2047、三联图落盘（中文标题在无 CJK 字体环境出豆腐块 → 标题改英文）。**新事实（§6 之外）**：中途几十 epoch（lr=1e-4 段）模型高概率区应落涡街/拐角回流区但边界毛糙、背景有噪声——仅作管线正确性检查，非交付结果。未决：无。下一步：用户 git push + 重跑 → 步速校准（TF32+20000）→ 分块训练 ~4-6 会话 → 票 08。

- 2026-08-25 票 07 论文训练细节核实（Zotero T76G9Z3A 全文，用户授权查证）：论文 §3.1 「3000 steady × 20 Killing = 60000 unsteady fields（9:1 train/val，Vatistas 参数拟合合成，非仿真数据）」；附录 C Training Details 原文「batch 100、cosine lr scheduler、warm-up 5 epochs + 195 epochs、共 200、AdamW(wd 1e-6)、n=3 PT blocks h=144、64 组×4/组、L=16、ReLU（末层 sigmoid）、BCE」；性能节「Intel Xeon Gold 6230R×2 + NVIDIA A100 单卡，200 epochs ≈ 14 小时」。
**事实修正（§2 模型事实行已改）**：论文调度器实为 cosine+warmup5（此前 HANDOFF 记录口吻为
「论文附录 C：warmup 后降 lr（仓库 config 为 TwoStep：warmup60/5e-6）」未核实全文）；
**实现口径维持仓库 config TwoStep**（票 06 已按此实现并测试；改 cosine 属用户决策，
若对齐论文须改 train_kaggle 调度器 + 测试）。
**速度差距结论（回答用户「为何 1501 帧(1 个仿真场)反而比 60,000 场慢」）**：训练时间 = 
每 epoch 样本数 × epoch 数 × 每步耗时——**与数据集场数无关**（我们每 epoch 采样数固定配置）；
每步 5s vs 论文 ~0.6s（按 14h/200epoch=4.2min/epoch 反推）来自 ① A100 vs T4×2 的 5-8× 硬件差
（论文可开 TF32/AMP；A100 fp32 19.5 TFLOPS vs T4 8.1）、② 论文离线预生成样本张量（训练纯 GPU
读缓存）vs 我们 on-the-fly CPU 提取（~0.9s/批）、③ DP 同步开销。Kaggle 免费层无 A100；已做最优
（TF32 + samples 20000 → 预计 35-55h）；进一步预案：样本预缓存（消数据管线，~15-20%）/修 vendor 
AMP（独立小票，改迁移忠实性）——目标 <20h 需 A100 级硬件。未决：调度器是否对齐论文 cosine 待用户决策。

- 2026-08-25 票 07 运行反馈修复六期（首块实测与预算检查修复）：Kaggle 首次完整块运行——cell 5 步速校准实测 1 epoch = 1058s（20000 样本 × 200 步 ≈ 5.3 s/步；**TF32 实测收益 <10%**：KNN 距离计算/softmax 为逐元素与归约操作，TF32 仅加速 matmul——此前 1.5-2× 预估不成立）；**发现并修复**：notebook cell 5 预算检查在 total_h > 48h 时生成 train_opt.yaml 并 `amp=True` ——与三期结论（上游模型 Half/Float 原地赋值冲突必崩）直接冲突，总时长 58.7h > 48h 必然触发 → cell 6 第一步即 RuntimeError；修复为**不启用 AMP**（仅样本数降级 + 提示块数），基准文案改按 `bench['samples_per_epoch']` 打印（原写死 40000）。**新事实（§6 之外）**：T4×2 DP+TF32+20000 实测基准 = 5.3 s/步、17.6 min/epoch、200 epoch ≈ 59h ≈ 8 个 7.5h 块会话（README §5 已回写）；TF32 保留（无副作用、小收益）而非回滚。未决：无。下一步：用户 git push + 重跑（首会话 bench 复用策略生效：bench_info 只在其所在工作区；Save Version 新版本是全新工作区，每次都会重跑校准——已知成本 ~18min/会话，接受）。


- 2026-08-25 票 07 运行反馈修复八期（跨会话免重复校准 + 产物核对）：用户提供 Kaggle 输出快照核查——`Downloads/repo` = 取消时点快照（有 outputs/bench_config.yaml、无 bench_info.json/无 outputs/train/无 preview），与七期诊断一致（校准子进程在 val 评估中被取消，未写回）——**非新问题**。改进：**bench_info.json 随 checkpoint 数据集走**——① cell 7 块尾打包前 `shutil.copy2` 到 `outputs/train/bench_info.json`（模式 A `kaggle datasets version` 与模式 B zip 都携带）；② cell 3 还原时自动落入 outputs/train/；③ cell 5 来源选择 = 本会话实测 → 还原值（`kaggle.chunking.pick_bench_source` 纯函数，**+4 测试**：新优先/还原兜底/双缺 None/损坏 json fail loud——全量 159 项待跑）；④ README 说明。效果：**首会话校准一次，后续每会话省 ~18 min**（Save Version 每版本全新工作区，此前每会话重复校准）。另修复替换过程引入的预算检查 `else:` 误删（回归检查 `if total_h > 4 * 12` 块含 else 分支）。未决：无。下一步：用户 git push + 重跑 → 校准（仅首会话）→ 训练 → 块尾打包（含 bench_info）→ 下会话复用。
- 2026-08-25 票 07 运行反馈修复七期（校准「无输出」根因与修复）：用户报告校准 1 epoch 后（1096.5s）长时无输出、手动取消（1685s）——**根因**：train_kaggle.main 的 val 触发条件 `epoch == start_epoch` 恒真 → bench 校准完成后**强制跑 val 片全量评估（20000 样本、batch 100 ≈17.5min、无进度条（evaluate 为朴素 for 循环）**——期间日志仅剩 `print`（stdout 管道块缓冲）→ 表现为「卡死」；实测取消时 val 已运行 ~10min（还差 ~7.5min 完成）。**修复**：① bench 配置 `data.val_split="none"`（跳过 val——校准只测训练步速）；② 生产 `val_freq` 5→10（val 开销从每 5 epoch 17.5min 降到每 10 epoch；~7% 块占比，2026-08-25 工程权衡）；③ notebook cell 1 `PYTHONUNBUFFERED=1`（f624183，print 实时刷）。**新事实（日志节奏）**：正常节奏 = 校准 ~18min → 训练 tqdm 17.6min/epoch → 每 10 epoch 停顿 ~17.5min（val，无进度条）；取消本次运行无损失（bench ckpt 在 outputs/bench 独立目录、未写 bench_info——下个版本重测校准即可）。未决：无。下一步：用户 git push + 重跑 → 校准 18min 后直接进入训练（无中途长停顿）→ 块尾打包 → 下载 preview 图复核。
- 2026-08-25 **票 07 延伸（多数据集 + τ 对齐论文 Fig.6）**：用户目标 = ① 弱标签 IVD 阈值对齐论文 Fig.6 列 1（IVD 在该数据集的白色等值线呈现——"appropriate threshold"，论文 §4.3 明示 IVD"highly sensitive to threshold selection"）；② 按论文 §4.2 评估范式（多数据集训练 → 真实数据测试，未接触测试数据）做**多数据集联合训练 + 跨数据集留出评估**。**口径说明（诚实性）**：本实现为**时间留出近似**——各数据集自身前 60% 参与训练、后 40% 时间上未见（且覆盖 7 类不同流场，跨分布泛化观察）；**非**论文语义的严格零样本（严格零样本须按数据集留出：部分数据集完全不参与训练；用户拍板按时间 60/40，未采用按数据集留出——如未来需要严格零样本，可另配留出数据集划分）。**执行前拍板（用户确认）**：τ = **p85 逐时间片分位**、5×5 面积过滤保留、**按时间各数据集帧前 60% 训/后 40% 测（无 val）**、τ/归一化**逐数据集各自**、训练池 = **全部 7 个数据集**（6 新 + pipedcylinder2d）。
  **需求 A（τ 对齐）**：多阈值敏感性报告落地 `weak_labels.multi_tau_report` + CLI `--multi-tau-dir`（候选 = 95/90/85/80 分位 × 固定 2.5/2.0/1.5/1.0 × min_area 25/9/1；统计 = 正格占比/连通块数/正样本占比；目检图含论文风格白色等值线）。**实测结论（pipedcylinder2d，流体区）**：p95 正格 4.81% 且涡块 6.8/帧（稀疏——用户观察一致）→ **p90 = 9.90%（τ≈1.08-1.18、块 6.98——覆盖率翻倍但结构块数不变：新增标签主要"填补"涡街/拐角回流，碎片新增少）**；p85 = 14.80%（τ≈0.61-0.65、块 9.82）；p80 = 19.73%（块 12.23、碎片 38.8/帧若不过滤）；**5×5 过滤在 p85 处仅删 0.15% 正格但消掉 ~15 碎片块/帧 → 保留（未吞大涡结构）**。用户拍板 p85（更饱满）+ 保留 5×5。**实现**：τ 默认 95→85（weak_labels.DEFAULT_PERCENTILE + compute_tau/prepare_dataset/CLI 默认，HANDOFF §6 回写）；**默认已回写 → 已训练的 25-epoch 模型为旧 τ（p95）标签，新 τ 标签需重训（用户 Kaggle 执行，单数据集产出 `outputs/dataset` 已本地重生成 p85 标签：taus=0.6539/0.6512/0.6113）**；preview 三联图标题明示两语义（"IVD reference (continuous)" vs "Weak label (binary train target)"）。
  **需求 B（多数据集）**：① `dataset.fraction_slices`（按帧比例 60/40/可选 val；floor 取整、全覆盖无泄漏——守护测试）；`prepare_dataset` 增 `split_mode=abs|frac` + CLI；② dataset.py 重构：`_DatasetStore`（单数据集存储/提取/池，池判定-标签-提取一致性不变）→ `WeakLabelPathlineDataset` 包装（公开面与票 05 逐字节兼容——单数据集续训采样序不变）+ `MultiDatasetPathlineDataset`（7 roots 联合池 (ds_idx,y0,x0,frame)，50% 过采样/自然分布/sample_at(si,...)/stores 公开；组合级 rng 基含 ds_id 派生——同语义不同构）；③ train_kaggle：`_make_dataset` 接受 `data.root` 列表、`evaluate_f1` 增 **IoU**（tp/(tp+fp+fn)）、`--f1-split`（默认 val_split；多数据集无 val → `--report-f1 --f1-split test` 对留出 40% 出自然分布 F1/IoU，缺少指定片 fail loud）；④ preview_eval：`--dataset` 索引跨数据集推理（单 ckpt → 各数据集 test 片，时间留出泛化观察的简版落地——非严格零样本，见本条目口径说明）+ 标题语义标注；⑤ `prepare_multi.py` 逐数据集预处理驱动（geometry→IVD/label/τ→memmap+meta+目检图，复用票 02/05 管线）；`kaggle/prepare_dataset_a.py` 增 `--nc/--dataset-dir` 多对打包（data/<nc> + datasets/<名>/ 布局 + manifest，单数据集布局不变）；⑥ `config/pathline_transformer_multi.yaml`（7 roots、frac、val_split none、run_name 独立）。
  **验证证据**：全量 **186 passed**（159 基线 + 27 新增：多阈值报告 3、fraction_slices 6、多数据集池 6、train 集成 4、IoU 1、preview 2、packaging 2、prepare_multi 3）；真实数据 = 7 数据集预处理产物 `outputs/datasets/`（§2 盘点表；≈3.8GB）+ 多数据集池实测（train 正 78,351/负 82,757、构建 2.0s、跨数据集样本有限性/逐数据集归一化/确定性序 all ✓）+ test 片自然分布 F1/IoU 评估跑通（随机初始化小模型：n=16384, F1=0.281, IoU=0.163——管线正确性检查）+ 跨数据集预览（25-epoch 基线模型 → 未见过的 duffing 帧 350：`outputs/preview/multi_duffing_t350.png`）。
  **执行边界**：新 τ 标签的单数据集重训、多数据集 200 epoch 联合训练与留出评估级报告 = **用户 Kaggle 执行**（`--config config/pathline_transformer_multi.yaml --report-f1 --f1-split test`；Dataset A 打包 `python kaggle/prepare_dataset_a.py --nc <7 个> --dataset-dir <7 个> --out kaggle_dataset_a_multi --zip`）。
  **未决**：① 多数据集 Kaggle Notebook 已自动适配（cell 3 单/多布局探测 + 自动选 config + RUN_NAME 参数化 cells 6/8/9；用户**重新上传 notebook 一次**即可，两条线共用）；② 正式弱定量表（网格投影级）仍属票 08；③ doublegyre/jung 的"圆柱"检出声明的单格噪声点（无实质障碍物；掩膜仅 4/12 格，属预期噪声处理——HANDOFF §2 已记录判据：圆柱 = 孤立连通块、无尺寸判据）。**用户决策（2026-08-25）：只跑多数据集联合训练线**（单数据集 p85 重训暂缓，单数据集 staging 已删除）→ 下一步：用户 git push → 上传多数据集 Dataset A（`kaggle_dataset_a_multi.zip`，含 7 nc + 7 套 memmap，notebook cell 3 自动适配单/多布局）→ 多数据集 200 epoch 联合训练（`--report-f1 --f1-split test` 收尾）→ 票 08。
- 2026-08-25 票 07 延伸 运行反馈修复（嵌套挂载探测）：用户上传多数据集 `kaggle_dataset_a_multi.zip` 后 notebook cell 3 报「未找到 Dataset A」——**根因**：Kaggle Add Input 挂载为多级嵌套 `{root}/datasets/<owner>/<slug>/`（三期已见单数据集同现象；本次用户 slug = `datasets/ziyixu317/dataset-a-multi`），而 cell 3 探测仅 glob 一层 → 命中失败。**修复**：探测抽为 `kaggle/mount_probe.py::probe_layout`（**含 root 自身 + rglob 全子目录深度优先**；判据 = 目录下有 `datasets/<名>/dataset/meta.json`（**多优先**——多布局内部 `datasets/<名>` 子目录含 dataset/meta.json，先判单会误命中）或 `dataset/meta.json`；未命中 → (None, None) 走 zip 回退/fail loud）+ notebook cell 3 改 `from kaggle.mount_probe import probe_layout`；**+4 测试**（嵌套多/嵌套单/浅层单/无命中，tmp 树模拟 Kaggle 结构；全量 189→193 待跑）。未决：无。下一步：用户 git push + **重传 notebook** → Run All（cell 3 应打印「多数据集布局 → config: config/pathline_transformer_multi.yaml」+ 7 条链接）。
- 2026-08-25 票 07 延伸 运行反馈修复二期（仓库克隆快路径漏更新）：用户重传 notebook 后 cell 3 报 `ModuleNotFoundError: No module named 'kaggle.mount_probe'`——**根因**：cell 2 的「仓库已就绪」快路径（`/kaggle/working/repo` 存在且含 train_kaggle.py 即跳过克隆）——`/kaggle/working` 在同一会话内持久，用户在本会话此前跑过旧代码 → 旧仓库残留、没克隆最新，且 `kaggle` 名解析到 site-packages 的 Kaggle CLI 包（故报 mount_probe 缺失而非 kaggle 缺失）。**修复**：cell 2 改**每次 rmtree + git clone --depth 1 最新 main** + 打印 repo HEAD（自证 >= b3c381e）；cell 3 的导入加 try/except（提示「先 Run cell 2 确认 HEAD」后上抛）；README §7 补「每次 Run All 重克隆」。未决：无。下一步：用户重传 notebook（或直接在打开的 notebook 里粘贴新 cell 2）→ Run cell 2（见 repo HEAD 行）→ Run All。
- 2026-08-25 票 07 延伸 运行反馈修复三期（cell 4 自检数据根多布局硬编码）：二期的 cell 2/3 修复生效（日志：HEAD d0646e8、多数据集布局 7 条链接全成功）后，cell 4 自检报 `FileNotFoundError: 数据集元数据缺失: outputs/dataset/meta.json`——**根因**：cell 4 硬编码 `--data-root outputs/dataset`（单数据集布局目录），多数据集模式下 cell 3 明确删除该目录（数据在 outputs/datasets/<名>/dataset）。**修复**：cell 4 自检数据根随布局自适应（单 = outputs/dataset；多 = outputs/datasets/pipedcylinder2d/dataset（任一已链接数据集即可——self_check 的 split="train" 在 frac 片下存在；无链接 fail loud 提示先跑 cell 3））。未决：无。下一步：用户粘贴新 cell 4（或重传 notebook）→ Run All（cell 4 应打印 `[自检] data-root = outputs/datasets/pipedcylinder2d/dataset` 后通过自检）。
- 2026-08-25 票 07 延伸 运行反馈修复四期（批式积分域边缘越界——多数据集训练崩溃）：cell 1-6 全通（自检通过、步速实测 ~2-5s/步）后 epoch 1 第 21 步崩 `IndexError: index 200 is out of bounds for axis 0 with size 200`（extractor._integrate_batched `m2[j, i]`，jungtelziemniak2d Y=200）——**根因（实测数值）**：网格**非整跨度**（生成器舍入：jung `span/dy = 199.0000628 > 199`），上缘半格容差内（out_ok ✓）的位置经 `(y-ydim[0])/dy + 0.5` 取整可得 **j == Y**，批式积分器的掩膜索引未越界守护（标量 mask_at 有前置越界返回 False 不崩，批式缺同语义守护）；pipedcylinder 单数据集此前不崩是边界为壁面固体掩膜挡住边缘轨迹的巧合。**修复**：`in_solid[out_ok] = m2[np.clip(j,...), np.clip(i,...)]`——半格容差内一律按裁剪索引取边界格掩膜（out_ok 已排除真出域；边界格为固体时批式冻结，比标量"越界视为流体"更物理）。**+1 回归测试**（jung 类非整跨度网格字面量构造：fringe 位置族 → 不越界 + 流体边界格 n=3 不冻结 + 固体边界格 n=1 冻结；全量 194 passed）；真实数据定向验证 = 3 个确定性崩溃组合复跑全过 + 7 数据集池各抽样 200 无异常。未决：无。下一步：用户 git push（仅仓库代码变更，notebook 无需重传——cell 2 每次强制重克隆）→ Run All。
- 2026-08-25 票 07 延伸 **实测回填一（多数据集步速，运行中）**：修复四期后用户重跑 —— cell 5 校准 1 epoch ≈ **10.85 min**；训练（cell 6 分块，43 epoch/块）实测 **每 epoch 10.0-10.3 min ≈ 3.0-3.1 s/步**（epoch 1 1260.2s / epoch 2 1876.9s / epoch 3 2487.8s / epoch 4 3087.6s，连续差分 617/611/600s），loss 正常下降（0.2685 → 0.1867 → 0.1741 → 0.1618）。**对比单数据集基线 5.3 s/步 / 17.6 min/epoch → 快 ~40%**（200 epoch ≈ 33.7h ≈ 5 会话 vs 基线 59h/8 会话）。**机制（已本地定量验证）**：提取成本主项 = _interp_pair（O(迹线点数)，与网格无关，duffing 0.166ms/次 vs pipedcylinder 0.179ms/次 +8%）；pipedcylinder 独有 ① 复杂固体几何（41.8% 固体 + 双圆柱/壁面 → 重播种/短迹线重试，实测 median 53ms vs 其余 39-43ms，约 +35%）② 页故障尖峰（1.2GB memmap 工作集，max 609-939ms vs duffing 100MB 无尖峰）——基线 100% 采样 pipedcylinder，数据管线停滞 ~2.3s/步；多数据集池 pipedcylinder 占 1/7 → 去抖。**修正此前判断**：多数据集"变快"为真（非 tqdm EMA 假象），速度收益幅度高于先前保守估计（3.0 vs 预估 4-4.5 s/步）。未决：无。下一步：训练继续（5 会话）→ 块尾打包续训 → 完成后 `--report-f1 --f1-split test` 回填。
- 2026-08-25 票 07 延伸 运行反馈修复五期（cell 7 块尾打包路径硬编码——43 epoch 全成但包没打上）：首块 43 epoch 全部训练成功（loss 至 0.0873，latest 在 outputs/train_multi/），但 cell 7 块尾打包报 `FileNotFoundError: No such file or directory: 'outputs/train/bench_info.json'`——**根因**：多数据集配置 `ckpt_dir=outputs/train_multi`，而 notebook cells 3/5/6/7/8/9 **硬编码 `outputs/train`**（单数据集时代路径）：cell 7 复制 bench_info 到不存在的目录（os.makedirs 仅在 cell 3 的 ckpt 还原分支——首会话无 ckpt 从未创建过）；**连带隐患（数据丢失级）**：下会话 cell 3 还原 checkpoint 到 outputs/train/ 而训练读 outputs/train_multi/ → 漏修则续训从 0 开始。**修复**：notebook 统一引入 `CKPT_DIR = cfg0["train"]["ckpt_dir"]`（单=outputs/train；多=outputs/train_multi）——cell 3（还原目标+makedirs）、cell 5（BENCH_RESTORED）、cell 6（latest 路径）、cell 7（bench_info 复制目标[复制前 makedirs]+srcdir+模式 A 发布目录）、cell 8（F1 glob）、cell 9（预览 ckpt 路径）全部去硬编码（无残留引用）。**用户恢复路径（43 epoch 无损）**：① 失败会话文件浏览器下载 `repo/outputs/train_multi/pathline_transformer_multi_ckpt_latest.pth`（+`outputs/bench_info.json`）；② git push 本修复；③ 新建 Kaggle Dataset（放这两个文件，如 `vortex-train-ckpt-multi`）挂 input（模式 B：CKPT_DATASET_SLUG 留空）；④ 下会话 Run All → cell 3 还原（落 outputs/train_multi ✓）→ cell 6 `--resume auto` 从 43 续训（每块 43 epoch，200 epoch 共 5 块）。未决：无。下一步：用户执行 ①-④ → 续训 → 块尾打包（修复后 cell 7 应成功）→ 完成回填。
- 2026-08-29 票 07 延伸 运行反馈修复九期（cell 6 基准读取硬编码——跨会话复用路径崩溃，八期功能首个真实落地缺口）：用户按五期恢复路径 ①-④ 执行，日志 cell 1-5 全通（repo HEAD 534c72a、7 数据集链接、checkpoint 还原 outputs/train_multi/、**复用步速基准（outputs/train_multi/bench_info.json）: 1 epoch = 627.8 s**（2026-08-28 15:31 首块实测）、34.9h 预算内、计划 [43,43,43,43,28]）后 cell 6 崩 `FileNotFoundError: 'outputs/bench_info.json'`——**根因**：cell 5 的 `pick_bench_source` 走「还原值复用」分支时**不写** `outputs/bench_info.json`（该文件只在「本会话实测」分支写入），而 cell 6 开头硬编码 `json.load(open("outputs/bench_info.json"))`——读的是本会话实测路径而非还原值路径；首会话（无还原值、实测写入）能过，跨会话复用是八期引入功能后的首次真实执行，此缺口首次暴露（与五期同类：cell 间隐藏路径耦合，表现不同）。**修复**：cell 5 无论来源**恒写** `outputs/bench_info.json`（= 本会话生效基准；还原值仅来源）→ cell 6/7 读取入口语义不变、cell 7 打包闭环成立；本地模拟验证（pick_bench_source 命中还原值 → 恒写 → cell 6 同款读取一致 → cell 7 存在性断言成立）+ notebook JSON 合法性校验通过（10 cells）。**用户恢复路径**：git push → 重传 notebook（或粘贴新 cell 5）→ 重启会话 Run All → cell 5 应打印「复用步速基准（outputs/train_multi/bench_info.json）」、cell 6 应打印「[分块] 从 epoch 43 续到 86（本块 43；完整计划 [43,43,43,28]）」。未决：无。下一步：续训（5 会话 ~34h）→ 完成回填。
- 2026-08-29 第二块完成（43→86，实测回填二）+ **两项用户决策（结算 130 + duffing 移除）**：第二块日志核对——repo HEAD 132243c（九期修复生效）、cell 1-5 全通（还原/自检/复用基准 627.8s/预算内/计划 [43,43,43,43,28]）、`[分块] 从 epoch 43 续到 86`；训练 44-86 loss 0.0859→0.0811（**无回退**，43 终点 0.0873 无缝衔接）、lr 61 号起换挡 5e-6（TwoStep 语义正确）；**cell 7 打包成功**（bench_info 已随打包 + ckpt_snapshot.zip 10.3MB——五期+九期修复双双生效）；cell 9 预览 t1300（**数据集索引 0 = boussinesq**——cell 9 默认 --dataset 0，非 pipedcylinder；boussinesq 域 x[-0.5,0.5] y[-0.5,2.5]，帧 1300 ∈ test 片 ✓ 生效）：概率域 [0,1]、正格 14766（21.9%）、三联图目检高概率区与 IVD/弱标签涡结构对应良好（无结构错位、边界毛糙+轻微过分割=中途模型+投影纹理预期，管线正确性确认）。**用户决策 1（结算标准变更 + 结算指标）**：43-86 loss 已趋稳 → **200→130 epoch 结算**（86 续跑 44，config epochs 130——§1 决策 3/§3 验收 2/§5 阶段 5/§6/本节回写）；**结算指标 = test 片自然分布 Precision/Recall/F1**（用户 2026-08-29 定；阈值 0.5（evaluate_f1 固定口径）、IoU 附带记录；json 字段 precision/recall/f1 已含，无需代码改动）；最终块自动带 `--report-f1 --f1-split test` → `pathline_transformer_multi_test_f1.json`（130 epoch 结算值）。**用户决策 2（数据质量问题）**：**forceddampedduffing2d 移出训练池**（roots 7→6；§2 盘点表保留该行作记录 + 注记；config root 即训练池唯一来源——**Kaggle Dataset A 无需重传**，cell 3 多链接的目录不参与池构建；本地打包目录可留可删）。**块规划预告**：epochs=130 后 cell 5 打印总时长 ≈ 22.7h（预算内）、分块计划 [43,43,44]（3 会话）；剩余 44 epoch（86→130）→ cell 6 自动 [43,1]：下一会话 86→129（43 epoch）、再下一 129→130（1 epoch + --report-f1 自动追加）。**未决**：① duffing 问题**三项实证完成**（2026-08-29）：ParaView 1×1×1 退化 = reader 路径问题（非数据缺陷）；**时间维冻结**（512 帧 max|du|=0）；**IVD/标签退化**（全域常数 0.006、正格全为 x=±2 边界伪影、与官网双涡不符）→ 用户移出决策实质成立（§2 注记）；② **6 既有数据集补做「时间演化+结构」体检**（§7 新预案——盘点盲点暴露；用户确认后执行，只读本地产物、不动预处理产物）；③ 130 epoch 完成后回填（test F1/IoU/checkpoint 位置）→ 票 07 勾选 + 票 08 启动（正式评估；cavity2d 已作为第 8 个"严格零样本"测试集预检通过——单帧 t=3 三联图 `outputs/preview/cavity2d_t3.png`：概率域 [0,0.997]、正格 346/3.5%（保守）、高概率区正确落在主涡+顶盖/右壁剪切层（与 IVD 对应、无结构错位），管线对新数据集（等距/无固体/21 帧非定常）适配路径 = abs 全帧单片 + t_win=16，属用户故事 9 落地演示）。下一步：用户 git push → 下一会话续训（86→129）→ 130 结算回填 → 票 08。
- 2026-08-29 **票 07 延伸多数据集训练完成（130 epoch 结算回填）**：用户完成 130 epoch 训练（4 个 Kaggle 会话：0→43、43→86、86→129、129→130+结算；比原 200/5 块计划省 1 会话——duffing 移除 + 130 中止）。下载第 4 块会话快照（`results (2).zip`）核对：① **Kaggle clone HEAD = 6ba1f47**（本会话所有 commit 已 push 生效——含 cell 6 `--f1-split test` 自动补全 `6fa9131`、结算口径 `9c6c464`、duffing 定论 `6ba1f47`）；② **latest ckpt epoch=129**（=第 130 个，config epochs=130、6 roots 验证）、metrics train_loss=**0.0797** lr=5e-6（86→129 缓降 0.0811→0.0797）；③ **结算值**（`pathline_transformer_multi_test_f1.json`，6 数据集联合 test 片自然分布、阈值 0.5、n=5,120,000 迹线）：**Precision=0.4967 / Recall=0.9549 / F1=0.6535**、IoU=0.4853（tp 844,308 / fp 855,629 / fn 39,855 / tn 3,380,208）；④ bench_info 627.8 s/epoch（首块实测、跨会话复用生效）。**结算值解读**：P<0.5 + R≈0.95 = 联合过分割——主因 **boussinesq τ 跨时间片漂移**（train τ=0.0555 vs test τ=0.5955，~10×；瞬态发展段 vs 成熟段标签口径不一致→模型学宽松标准→test 严格标签上全判正）；86 epoch 逐数据集 100 样本评估印证（boussinesq P=0.305/R=0.999、cylinder2d P=0.572/R=0.966——τ 差 2×；其余 4 数据集 F1 0.88-0.96 健康）。**产物**：`outputs/_ckpt130/repo/outputs/train_multi/pathline_transformer_multi_ckpt_latest.pth`（5.6MB，130 epoch 最终权重，已解压本地）+ test_f1.json + bench_info.json + cell 9 预览图 `prob_vs_ivd_t1300.png`（boussinesq t1300）；final_ckpt.zip = ckpt_snapshot.zip（5.16MB，含 latest+bench_info+test_f1）。**缺口（非阻塞）**：87-129 会话产物（E90/E120 里程碑 + 训练 tqdm 日志）用户未下载、Kaggle 会话已结束→里程碑丢失（latest=最终权重已拿到，里程碑非必需；loss 中段轨迹缺 87-128，已知 86=0.0811/129=0.0797 端点）。**票 07 验收 2/3/4 达成**（200→130 结算口径、跨会话断点续训、checkpoint 归档 + test P/R/F1 记录）→ 票文件改 done。**下一步**：票 08 正式评估（TTA 滑窗推理→网格投影→多帧对比图/动画/弱定量表；latest 权重在 `outputs/_ckpt130/` 可直接用；cavity2d 作为第 8 个严格零样本测试集纳入；boussinesq τ 漂移在票 08 报告中按数据集拆分声明或用全局 τ 重标——用户决策）。
- 2026-08-31 票 08（推理评估 evaluate）完成：新增 `evaluate.py`（评估管线）+ `tests/test_evaluate.py`（37 项），**全量 231 passed**。覆盖规格主验收缝：滑窗推理（stride 16 全场覆盖 + **贴边补全**，spec 端到端缝"覆盖全场"——ds.patch_locations 不贴边致顶部/右缘无种子格投影=0 暗带）+ TTA 平均 → 网格投影（累积 + 计数平均消 patch 重叠）→ 对比图（模型/IVD/Q/速度模+弱标签等值线）+ mp4 动画（ffmpeg 缺失回退 gif）+ 弱定量表（F1/IoU、标签参考 `vortex_area_ratio` 与模型预测 `pred_vortex_ratio` 涡面积占比、帧间连续性）+ 推理可复现（TTA 固定种子，`test_fixed_seed_reproducible` 守护）。**实现决策 / 规格偏离（重要）**：① 展示帧模型概率面板用滑窗 `prob_sw`（patch 归一化，与训练一致），**未用加密种子 dense 路径**——spec 要求"展示帧用加密种子（每 2×2 一组十字）"，但 `infer_dense` 用**全场归一化**与模型训练 **patch 归一化**口径不同，实测输入分布偏移致模型输出退化（展示帧 Panel 几乎全为低概率，无法正确目检涡结构）；滑窗 `prob_sw` 输出涡结构正确；`infer_dense`/`_dense_seeds`/`_dense_extract` 保留为独立工具 + 单测（`test_dense_seeds_coordinate_order` 守护），不作为主流程输入。此为 spec 偏离，记录于本日志与 §4 职责行。② **修复 dense 种子坐标序 bug**：`_dense_seeds` 原构造 `[y,x]`，与 extractor 全场约定 `[x,y]`（`integrate_pathline`/`seeding_grid`）相反，dense 迹线/投影串扰；改 `[x,y]` + 回归测试（原测试用对称点 0.5 无法捕获）。③ **滑窗贴合（覆盖全场）**：`sliding_window_patches` 补最后贴边 patch。④ **`--dense-step` 弃用参数移除**（原贯穿 CLI/run/summary 却零效果）；`make_animation` 死参数 `frames_ivd`、死代码 `_resolve_data_roots`/`import sys` 移除。⑤ **多数据集帧间连续性按数据集各自算**（跨数据集边界无意义）；`compute_frame_metrics` 增 `pred_vortex_ratio`。**实测（真实数据，130-epoch 最终权重 `outputs/_ckpt130/` + `outputs/datasets/`）**：单帧展示 smoke `outputs/evaluation_smoke/comparison_t1300.png`（pipdycylinder2d test 片帧 1300）——高概率涡街（左下）+ 拐角涡（右上）与 IVD/Q 参考对应良好、固体几何正确、顶部/右缘无暗带；F1=0.6575 / IoU=0.4897 / 标签涡面积占比 0.1410 / 模型预测涡面积占比 0.0855（threshold 0.5）。动画 smoke `outputs/evaluation_anim/vortex_animation.gif`（3 帧 1296/1298/1300，本机无 ffmpeg 回退 gif，代码优先 mp4）。**事实回写**：§3 验收 1/3/4 标注交付状态；§4 evaluate.py 职责行更新（展示帧用滑窗 prob_sw、dense 保留工具、pred_vortex_ratio）。**/code-review 双轴**：Standards 硬性 0（依赖/中文 docstring/领域词汇合规；英文图题被 §11"中文标题豆腐块→英文"先例压制）；判断项处置——dense 死路径+`dense_step` 弃用（移除运行依赖，保留工具+单测，见决策①）；`_resolve_data_roots`/`import sys`/`mask2d` 死代码（删）；`make_animation` 死参数（删）；`_dense_extract` 与 `extract_pathlines_batched` 7 通道组装同源（坐标序守护 + 记录，公式一致性保留）；Feature Envy 直读 store 私有（评估需底层场，`dataset.sample_at` 返回单样本不合用，记录）。Spec 轴——滑窗"覆盖全场"（修贴边）、`--dense-step` 误导（移除）、`make_animation` 死参数（删除）、`vortex_area_ratio` 与模型无关（增 `pred_vortex_ratio`）、`frame_continuity` 全负返回 1.0（两帧无涡=一致，数学约定保留+注释）、弱标签等值线/gif 回退/CLI 覆盖参数（范围蔓延/工程实用，保留+记录）、多阈值敏感性报告（票 07 延伸已交付，非本票）。`tests/test_multidataset.py` 单行 7→6 roots（duffing 移出训练池的过时断言，配合 §1/§11 修正使全量全绿）。**未决**：正式 mp4 需在含 ffmpeg 环境产出（本机回退 gif）；最终展示帧选择与多阈值报告引用由用户复核。下一步：用户复核对比图目检（验收 1 目检）+ 视需要跑正式多帧评估（含 cavity2d 严格零样本测试集）。


