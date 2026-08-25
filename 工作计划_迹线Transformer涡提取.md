# 工作计划：迹线 Transformer 在 2D 非定常圆柱绕流数据上的涡提取

> **⚠️ 本文档已被 `HANDOFF.md` 取代（单一事实来源迁移）**：项目权威上下文与更新协议见同目录 `HANDOFF.md`，新会话一律先读它。本文件仅保留作历史考古，内容不再维护。

> 依据：VortexTransformer 论文（Zotero: T76G9Z3A，CGF 2025, DOI 10.1111/cgf.70042）+ 开源仓库 `PyflowVis-main`
> 目标数据：`C:\Users\徐子屹\Desktop\AI CFD\CFD数据集\pipedcylinder2d.nc`（ETH CGL "2D Unsteady Cylinder Flow Around Corners"）

---

## 0. 已确认的约束与决策（用户拍板）

| 事项 | 决策 |
|---|---|
| 训练标签来源 | **直接用 IVD/Q-criterion 阈值给仿真数据打弱标注**，不生成 Vatistas 数据集、不做参数拟合（明确不参考论文的数据生成管线） |
| 模型 | 论文主体：迹线 Transformer（`PathlineTransformerV0`），从头训练，**不使用**仓库内预训练权重与 demo 验证集（用户已在 Kaggle 验证过该模型可用） |
| 训练平台 | Kaggle，T4×2（约 16GB×2），**从头训练 200 epoch** |
| 评估 | 定性对比（IVD / Q-criterion 为参考），不涉及 Vatistas 验证集 |
| 代码组织 | **不依赖原始 `PyflowVis-main` 目录**：新建独立项目文件夹，迁移（复制）所需代码为自包含工程（含 Apache 2.0 LICENSE/NOTICE 署名），见阶段 0 |
| 本地环境 | 核显、torch 2.10.0+cpu（实测无 CUDA）→ 本地仅做 **CPU 正确性冒烟**（迹线/标签/单次前向），训练全在 Kaggle |
| 迹线口径 | **256 条 = 64 组 × 4 条（卫星点，不含中心），KpathlinePerGroup=4**，对齐论文正文/附录 C 与发布数据、Python 加载器 |

## 1. 已核实的技术事实（论文 + 代码逐行确认）

### 数据集（本地实测）
- `pipedcylinder2d.nc`：NetCDF-4/HDF5，**450×150 网格，1501 时间步**（dt=0.01，t∈[0,15]）
- 域 x∈[-0.5,5.5]，y∈[-0.5,1.5]；Re=160，圆柱半径 0.0625
- 变量 `tdim/xdim/ydim/u/v`（u,v 形状 (T,150,450)），与仓库 `NetCDFLoader.load_vector_field2d` 完全兼容
- u∈[-3.0,4.6]，v∈[-2.3,2.4]
- **无 NaN**（全量 1501×150×450×2 扫描为 0）
- **域内含静态固体几何（实测）**：约 41.8% 细胞近零速（每帧固定 28213 个，逐帧不变）→ 台阶管道壁面 + 圆柱 + 拐角低速死区；迹线提取必须带固体掩膜（种子排除、迹线截断、IVD 掩膜），见 §2 `geometry` 与阶段 1
- **本地坑**：netCDF4 的 C 库在中文路径下打开失败（实测），**h5py 可直接读中文路径（已实测可行）**→ 新数据集类用 h5py 读；Kaggle 上路径为 ASCII 无此问题

### 模型（`DeepUtils/models/segmentation/pathline_transformer.py`）
- 输入：迹线簇 `[B, L, K, C=7]`，C = **[px, py, t, ivd, distance(距种子点), u, v]**
- 十字采样（最终口径）：每组 4 条 = **4 个卫星点 (x±Δ,y)、(x,y±Δ)（不含中心）**，64 组 × 4 = **256 条**，L=16 步——与论文 §3.2.1/附录 C、发布训练数据、Python 加载器一致
- PSL（Pathline Sampling Layer）：组置换 + 空间下采样（64 组保留 32）+ 时间下采样（L 16→8）→ N = 32 组×4 条×8 步 = **1024 点**
- 3 层 KNN Point-Transformer（k=16，相对位置编码 MLP），全局池化（mean+max），特征传播回全部 256 条迹线，sigmoid 输出每条迹线的涡概率
- ⚠️ **KNN 在 (x,y,t) 三点坐标上暴力计算**（O(N²)，时间与空间坐标混在同一距离度量里）→ 空间/时间归一化尺度直接影响邻居选择。原训练数据 x,y∈[-2,2]（跨 4）、t∈[0,π/4≈0.785]（比值≈0.2），故引入 `t_scale` 参数（默认 0.25 复刻原比例，冒烟可调）
- ⚠️ **推理非确定性**：PSL 的时空采样在 forward 里恒为 `random=True`（训练/推理都随机）→ 评估协议：同一样本随机采样 5 次取平均概率（TTA），或临时改 `random=False` 做确定性评估
- 4/组口径下采样后 N = 32 组×4 条×8 步 = **1024 点**（与论文一致）；暴力 KNN 每样本 1024²≈1M 距离对，batch 100 时每步 ~1 亿 → 每步耗时以冒烟实测为准（T4 预计明显慢于 A100，见阶段 5）
- 损失 BCE；训练超参（论文附录 C）：AdamW(wd=1e-6)、lr=1e-4、batch=100、200 epochs、warmup+余弦（仓库 config 为 TwoStep：warmup 60、二段 5e-6）
- ⚠️ **每组迹线数口径（已核实 C++ 源码，最终决策 = 论文口径）**：C++ `generateSeedingsCross` 含中心点（64 组×5=320 条，`GridCrossSampling` 产出），但论文 §3.2.1"seed four points … resulting in four pathlines"、附录 C"64 groups, 4 pathlines per group"、随仓库发布的数据与 Python 加载器（`PathlineCount=8×8×4=256`）均为 **4 条/组 = 256 条**；`outputPathlinesPerCluster=5` 是声明后从未使用的死代码。**本项目采用 4 条/组（256 条），KpathlinePerGroup=4**，卫星点不含中心，编组为组主序（组 g 的 4 条在前），与发布加载器的 `linesPerGroup=4` 解析一致。

### 标签通道机制（必须复刻）
训练数据中每条迹线第 1 个时间步的 `distance` 通道存标签（C++ 中 `judgeVortex(seed)`），加载器 `getSegmentationofPathlines` 提取标签后把该位置清零。新数据集直接产出"迹线张量 + 每条迹线 0/1 标签"，不必复刻 `.bin/.json` 文件布局。

### 可复用组件（实测存在且可用）
- `DeepUtils/models/`（`pathline_transformer.py`、`base_seg.py`、`build.py`、`samplingLayers.py`）+ `DeepUtils/loss/`（BCELoss 已注册）+ `DeepUtils/utils/registry.py`：**导入链全纯 torch**（已逐文件核实）；原仓库的 `utils/__init__`→`config.py` 会引入 multimethod，但阶段 0 迁移时剔除 config.py，故**本项目无需 multimethod**
- ⚠️ **不要导入** `DeepUtils/MiscFunctions.py`、`train.py`、`test.py`、`DeepUtils/dataset/`、`FLowUtils/`：它们会拖入 numba、pybind/C++ 模块、GUI 等 Kaggle 上不必要的依赖（`train.py` 顶层还无条件 `import wandb`）→ 训练/评估脚本自写，仅复用上面的纯 torch 模型代码
- `FLowUtils/vortexCriteria.py` 的 `computeIVD`/`computeQcriterion`：仅作参考（纯 Python 慢循环，且 IVD 用**全片均值**）→ 本项目自行矢量化实现（局部邻域均值版）
- `test.py::pathlineSegToFieldSeg` 的投影逻辑：抄入 `evaluate.py`（种子点→最近网格累加），并加计数平均

---

## 2. 新增代码清单（独立自包含工程，不依赖原 PyflowVis-main）

项目根目录 `C:\Users\徐子屹\Desktop\AI CFD\cylinder_vortex_pipeline\`，结构：

```
cylinder_vortex_pipeline/
├── vendor/DeepUtils/            # 从 PyflowVis-main 迁移（复制）的最小纯 torch 子集
│   ├── models/                  #   pathline_transformer.py, base_seg.py, build.py,
│   │   │                        #   samplingLayers.py, layers/, segmentation/, ...
│   ├── loss/                    #   build.py（BCELoss 注册）, cross_entropy.py
│   └── utils/                   #   registry.py, ckpt_util.py, random.py + 精简 __init__.py
│                                #   （剔除 config.py→无需 multimethod 依赖）
├── LICENSE  NOTICE              # 随迁移代码保留 Apache 2.0 署名
├── geometry.py
├── extractor.py
├── weak_labels.py
├── dataset.py
├── train_kaggle.py
├── evaluate.py
└── config/pathline_transformer_cylinder.yaml
```

**迁移规则（阶段 0 执行）**：只复制上述三个目录，`utils/__init__.py` 重写为仅导出 `registry/ckpt_util/random` 中模型链用到的符号（`get_missing_parameters_message` 等位于 ckpt_util.py，已核实），从而**不引入** `config.py→multimethod`；不复制 dataset/、FLowUtils/、MiscFunctions、train.py/test.py（那些会拖入 numba/pybind/GUI/wandb）。

| 文件 | 内容 |
|---|---|
| `geometry.py` | 固体掩膜提取：\|v\|<ε 逐帧取与（静态几何）→ 连通域标记 → 定位圆柱（不与壁面相连的孤立连通块）与管道边界；输出 `mask.npy`。**数据集无关的通用实现**：对每个新数据集独立运行、生成各自的掩膜；无障碍物数据集得到空掩膜，全流程不变（见下方"多数据集泛化"说明） |
| `extractor.py` | 迹线生成：十字采样种子（**每组 4 个卫星点 (x±Δ,y)、(x,y±Δ)，不含中心**，Δ=patch×0.05）→ 并行 RK4（矢量化三线性时空插值，**用全局场积分、允许迹线离开 patch**）→ 7 通道特征 → 位置按 patch 归一化到 [-1,1]（可超界）。固体处理：种子落固体→按 C++ `JittorReSeeding` 同款重播种；迹线入固体→截断并重复末点（不引入 -1000 毒值）。**组主序编组**（组 0 的 4 条在前），匹配 `PathlineSpatialSamplingLayer` 与发布加载器 `linesPerGroup=4` 的 mask 逻辑 |
| `weak_labels.py` | 矢量化 IVD（**5×5 局部邻域均值**，与论文定义一致；固体区置 0）+ Q-criterion；阈值化 + 最小连通域过滤 → 每条迹线种子点的 0/1 标签（与 C++ `judgeVortex(seed)` 的"种子点在涡内"口径一致） |
| `dataset.py` | `WeakLabelPathlineDataset`（torch Dataset 直用，不注册 DeepUtils.dataset registry）；h5py+np.memmap 读 nc（u,v 与预计算 IVD 都 memmap 共享给多进程）；on-the-fly 生成 patch+迹线+标签；正样本过采样；train/val/test 时间划分；返回 `((dummy_field, pathlines), labels)` 以匹配模型输入；**数据源可扩展为多数据集列表**（每个数据集各自跑 geometry/IVD 预计算） |
| `train_kaggle.py` | 自写训练脚本：仅 import `vendor/DeepUtils`（纯 torch 链）；自行实现 TwoStep 调度（warmup 60 → 5e-6）、梯度裁剪、每 epoch checkpoint（含 optimizer 状态）、断点续训、val F1；可选 DataParallel / AMP |
| `evaluate.py` | 测试集滑窗推理 → TTA（随机采样 5 次平均）→ 网格投影（累积+计数平均消除 patch 重叠）→ 预测 vs IVD vs Q 对比图 + 动画 + 弱定量指标；最终展示用**加密种子**（每 2×2 输出像素一组十字）提升投影分辨率 |
| `config/pathline_transformer_cylinder.yaml` | 训练配置（KpathlinePerGroup: 4、in_channels: 7、dmodel: 144、3 层、k: 16、BCELoss、其余同论文） |

**依赖清单（最终）**：torch（本地 CPU 版已装 / Kaggle 自带）、numpy、h5py、yaml、matplotlib、tqdm —— 全部可 pip；**不需要** wandb/easydict/numba/netCDF4/multimethod/任何 C++ 编译。

### 关于 geometry.py 与多数据集泛化（用户关切，已澄清）

掩膜**只用于数据准备阶段**（决定种子点放哪、迹线何时截断、IVD 在固体区置 0），**不作为模型输入、不参与损失**，因此不会限制模型的泛化能力——模型看到的仍是 7 通道迹线特征，与论文一致。换新数据集时：
1. 每个数据集各自跑一遍 `geometry.py`（同一套 \|v\|<ε + 连通域规则，全自动）→ 各自 `mask.npy`；
2. 掩膜随数据集的 (T,Y,X) 形状与坐标各自存储，`extractor/dataset` 读取对应掩膜即可；
3. 无障碍物的数据集（如解析场、无壁流场）掩膜为空，代码路径相同。
即：几何差异被隔离在预处理层，多数据集混训时每个样本只带"自己数据集"的预处理结果，网络结构/输入格式不变。

---

## 3. 阶段计划

### 阶段 0：项目独立化（代码迁移，0.5 天）
1. 按 §2 结构创建 `cylinder_vortex_pipeline/vendor/DeepUtils/{models,loss,utils}`，从 `PyflowVis-main` **复制**对应文件（不动原仓库）
2. 重写 `vendor/DeepUtils/utils/__init__.py`：仅导出 registry/ckpt_util/random 中所需符号，**不引入 config.py（免 multimethod）**
3. 复制 `LICENSE`、`NOTICE`（Apache 2.0 署名义务）
4. 验证：`python -c "import sys; sys.path.insert(0,'.'); from vendor.DeepUtils.models import build_model_from_cfg"` 在本地 CPU 环境通过；此后**不再引用原始 PyflowVis-main**

### 阶段 1：本地 CPU 冒烟（Windows，Python 3.12，1~2 天）
> 环境已实测：核显无 CUDA，torch 2.10.0+cpu 已装。**全部冒烟在 CPU 完成**：迹线/标签/掩膜是 numpy+h5py 纯 CPU；模型单样本前向在 CPU 上秒级（模型仅 ~0.5M 参数量级，可接受）。训练不在本地做（Kaggle 承担）。
0. 写 `geometry.py`：提取固体掩膜 → 可视化台阶管道形状、**定位圆柱中心**（孤立连通块；若分离不干净则用圆拟合/人工标定），确认种子排除与截断规则
1. 写 `weak_labels.py`：读 nc（h5py）→ 计算若干时间片的 IVD/Q → 保存对比 PNG
2. 写 `extractor.py`：在 3~5 个 32×32 patch × 24 帧窗口上生成 256 条迹线 → 叠加 IVD/掩膜图目检（迹线是否被正确积分、是否被固体正确截断、特征量级是否合理）
3. 用不同阈值 τ 出标签图，**校准 τ 默认值**（见参数表）
4. 用本地 CPU torch 跑通一次前向：随机迹线 → `PathlineTransformerV0` → 输出形状正确（同时验证 registry 接受普通 dict 配置）

### 阶段 2：弱标签定义与校验（与阶段 1 并行）
- IVD：中心差分求 ω = ∂v/∂x − ∂u/∂y，IVD = |ω − 局部 5×5 均值|
- 迹线标签 = IVD(种子点, t0) ≥ τ，附加 5×5 最小面积连通域过滤（去噪点）
- 备选标签源：Q-criterion > 0（作对照实验）
- 交付：不同 τ 的标签对比图（推荐 95 分位数起）+ 正样本占比统计

### 阶段 3：数据集类与训练配置（1~2 天）
- `dataset.py`：时间划分 train t∈[0,10]、val (10,12.5]、test (12.5,15]（**无时间泄漏**）；空间全滑窗 stride 16（约 27×8≈216 patch/窗口），窗口起点按 4 帧步长采样减冗余；每 epoch 随机采样 40000 样本，50% 强制含涡（正样本过采样）
- 特征归一化：px,py → [-1,1]（patch 内，可超界）、t → [0,1]×`t_scale`（默认 0.25）、ivd 标准化、distance 用归一化坐标计算、u,v 除以全局最大速度
- 预计算 IVD 场存 memmap（1501×150×450 float32 ≈ 405MB，一次算好，多进程共享页缓存）；u,v 同样 memmap 访问
- 单样本生成目标 < 5ms（矢量化后应远低于此），num_workers=2（Kaggle 内存有限，memmap 避免每进程复制 810MB）

### 阶段 4：Kaggle 上传
- Kaggle Dataset A：`pipedcylinder2d.nc`（810MB）
- Kaggle Dataset B：整个 `cylinder_vortex_pipeline/`（含 `vendor/DeepUtils`，体积小；不含原 PyflowVis-main 的 CppProjects/Eigen 等）
- Notebook：`pip install h5py yaml matplotlib tqdm`（torch 自带；**无需 wandb/easydict/numba/multimethod**）；训练脚本自写，不 import 原仓库的 train.py

### 阶段 5：训练（Kaggle T4×2，从头 200 epoch）
- 冒烟：先跑 1 epoch，实测每步耗时 → 校准 epoch 样本数（估算：40000 样本/epoch → 400 步/epoch × 200 = 8 万步；T4 单卡预计 0.8~1.3s/步 → 18~29h，**超预算**）
- ⚠️ **Kaggle 硬限制（本轮补审新增）**：交互式 GPU 会话上限 **12h/次**（每周 GPU 配额约 30h）→ 训练必须分块：每块 ≤8h（约 60~100 epoch），每 epoch 存 checkpoint（含 optimizer/scheduler 状态），块尾打包为 Kaggle Dataset 新版本，下一会话挂载后恢复
- 提速选项（冒烟后定）：① DataParallel 用满双 T4（`train.py` 无多卡支持，自写脚本里 `nn.DataParallel`，batch 100 → 每卡 50，预计 1.6~1.9×）；② AMP 混合精度（预计 ~1.5×，注意 KNN 距离与 softmax 数值稳定）；③ 降 epoch 样本数（下探到 20000）
- 超参：BCELoss、AdamW(wd 1e-6)、lr 1e-4、TwoStep（warmup 60，二段 5e-6）、batch 100、梯度裁剪 1.0、KpathlinePerGroup=4（64 组×4 卫星=256 条/样本；采样后 32 组×4×8 步=1024 点，同论文）
- 监控：train/val loss、val 上 precision/recall/F1/IoU（对弱标签）

### 阶段 6：推理与评估（1~2 天）
- 测试时间窗滑窗推理 → 每条迹线概率（**TTA：同一样本随机 PSL 采样 5 次取平均**）→ 网格投影（累积+计数平均消除 patch 重叠）
- 交付图：速度模场底图 + 预测涡概率场 + IVD 等值线 + Q>0 区域 + 固体掩膜，多个代表性时间步 + mp4 动画；展示帧用**加密种子**（每 2×2 输出像素一组十字）提升投影分辨率
- 弱定量表：对 IVD 阈值的 F1/IoU（注明阈值相关性）、预测涡面积占比、帧间连续性

### 阶段 7：整理（半天）
- 结果目录、复现 README、参数表、checkpoint 归档

---

## 4. 默认参数表（可在冒烟阶段调整）

| 参数 | 默认值 | 说明 |
|---|---|---|
| 时间窗口 T_win | 24 帧（0.24s） | 与论文生成器 observer.timesteps=24 一致 |
| 迹线长度 L / 组数 / 每组条数 | 16 / 64 / **4（卫星点）** | 共 256 条/样本，KpathlinePerGroup=4（论文与发布数据口径） |
| 十字偏移 Δ | patch 边长 × 0.05 | 轴向十字卫星：(x±Δ,y)、(x,y±Δ)，不含中心（同 C++ `generateSeedingsCross` 去中心点） |
| RK4 子步 | 每输出步 4 子步 | 三线性时空插值 |
| patch / stride | 32×32 / 16 | 与论文推理一致；窗口起点步长 4 帧 |
| 时间尺度 t_scale | 0.25 | KNN 的 (x,y,t) 混合度量中 t 的相对权重（复刻原训练数据 t/空间≈0.2 的比例），冒烟可调 |
| IVD 阈值 τ | 95 分位数（逐时间片） | 阶段 1 校准；备选 μ+3σ |
| 最小涡面积 | 5×5 | 标签连通域过滤 |
| epoch 样本数 | 40000（正样本 50%） | 阶段 5 按实测每步耗时校准（下探下限 20000） |
| 时间划分 | 10 / 12.5 / 15 s | train/val/test |
| 评估 TTA 次数 | 5 | 平均随机 PSL 采样的概率 |

## 5. 风险与预案

| 风险 | 预案 |
|---|---|
| 弱标签阈值敏感/循环评估 | 多阈值敏感性报告；定性对比为主；备选 Q-criterion 标签做对照实验 |
| Kaggle 配额不足 | 12h 会话分块 + 每 epoch checkpoint + Dataset 版本续训；样本数/epoch 可调；DataParallel/AMP 提速 |
| 正负样本严重不平衡 | 50% 过采样；必要时 BCE pos_weight |
| 涡特征微弱（Re=160 尾流为主） | 检查标签图上涡街结构；τ 下探；观察窗加长（T_win 24→48） |
| **迹线撞固体/种子在固体** | 固体掩膜：种子重播种（JittorReSeeding 同款）、入固体截断+重复末点；冒烟目检 |
| **KNN 时空尺度失调**（t 与 x,y 混量纲） | t_scale 参数（默认 0.25），冒烟对比 t_scale∈{0.1,0.25,1} |
| **推理非确定性** | TTA 5 次平均；或改 random=False 确定性评估并固定记录 |
| 本地中文路径 | h5py 直读已解决；建议同时复制一份到 ASCII 路径（如 `C:\flowdata\`） |

## 6. 里程碑与验收

- **M1**（阶段1-2）：固体掩膜/圆柱定位完成；迹线/标签可视化目检通过，τ 校准完成
- **M2**（阶段5）：Kaggle 跑通 1 epoch，每步耗时与总预算明确
- **M3**（阶段5）：200 epoch 训练完成，val F1 记录，checkpoint 归档
- **M4**（阶段6）：测试集定性对比图 + 动画 + 定量表 + ivd 遮除消融交付
- **验收标准**：预测涡区域与 IVD/Q 参考结构一致（涡街、拐角回流区），无明显碎裂噪声；附多阈值 F1 报告

## 7. 第二轮补审新增要点（与初稿差异）

1. **数据含静态固体几何**：41.8% 细胞近零速（台阶管道壁+圆柱+死区）→ 新增 `geometry.py`（掩膜/圆柱定位）与种子排除、迹线截断、IVD 掩膜策略；无 NaN 已确认
2. **KNN 时空混合度量**：KNN 在 (x,y,t) 上暴力计算 → 引入 `t_scale=0.25` 复刻原文尺度比，冒烟对比
3. **推理非确定性**：PSL 随机采样恒开 → 评估用 TTA(5 次平均)；可选确定性模式
4. **依赖链收紧**：只迁移 DeepUtils/models+loss+utils（纯 torch；剔除 config.py 后连 multimethod 都不用装）；不 import train.py/test.py/dataset/FLowUtils（numba/pybind/GUI/wandb 陷阱）；训练与评估脚本完全自写
5. **Kaggle 硬限制**：会话 12h/次 + 周配额 → 分块训练 + 每 epoch 断点 + Dataset 版本续训；T4 单卡预计 0.8~1.3s/步（比初稿估算更保守），DataParallel/AMP/降样本数三档预案
6. **迹线在 patch 外继续积分**（用全局场），位置按 patch 归一化后可超界（与论文真实数据推理一致）；组主序编组匹配采样层 mask 逻辑；固体截断用"重复末点"而非 -1000 毒值
7. **IVD 预计算 memmap**（405MB，一次算好共享），u,v 也 memmap 访问，num_workers=2（Kaggle 内存约束）
8. **评估口径**：val F1 只反映对弱标签的拟合度，不作为"检测质量"唯一标准；IVD 是公认的高精度涡判据，模型学习其近似可接受（不做 ivd 遮除消融）
9. **可视化投影**：训练 8×8 组/patch 稀疏投影留空隙 → 展示帧加密种子（每 2×2 输出像素一组十字）；patch 重叠用计数平均
10. **val 指标口径**：同上第 8 条（已在阶段 5/6 注明）
11. 第三轮修正：①迹线口径定稿 **256 条（64 组×4 卫星，KpathlinePerGroup=4）**，对齐论文与发布代码；②新增**阶段 0 代码迁移**——自包含 `vendor/DeepUtils` 独立工程（保留 Apache 2.0 署名、免 multimethod），不依赖原始 PyflowVis-main；③本地冒烟改为**纯 CPU**（实测 torch 2.10.0+cpu、无 CUDA、核显），训练仍全在 Kaggle；④geometry.py 明确为逐数据集预处理、不进模型输入，多数据集扩展无障碍
