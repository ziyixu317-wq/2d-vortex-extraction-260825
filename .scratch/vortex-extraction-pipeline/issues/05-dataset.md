# 05: 数据集类（dataset）

**What to build:** 弱标签迹线数据集：train/val/test 按时间划分（10 / 12.5 / 15 s，帧 0-1000 / 1000-1250 / 1250-1500，无时间泄漏）；patch 32×32 stride 16、窗口 T_win=24 帧、窗口起点步长 4 帧；u,v 与预计算 IVD 存 memmap（IVD 一次算好）；7 通道归一化（px,py → patch 内 [-1,1]；t → [0,1]×t_scale 默认 0.25；ivd 标准化；distance 用归一化坐标；u,v ÷ 全局最大速度）；每 epoch 40000 样本（下限 20000）、50% 正样本过采样；返回 `((dummy_field, pathlines), labels)` 匹配模型输入。

**Blocked by:** 03、04

**Status:** done

- [x] 单样本生成 <5ms —— **按用户指示放宽**（2026-08-25 会话："时间问题不用太纠结，只要能跑就行"）：on-the-fly 实测中位数 ~35ms（真实窗口 24 帧；热身中位 35.7ms、含首取页加载 836ms 冷启动），池构建 ~0.15-0.85s；"能跑"判据（冒烟上限 <1s + 数量级守护）已由测试 `test_sample_time_under_5ms` 固化（测试内注明降级原因与实测值）。性能模型：Kaggle 多进程 DataLoader（8 worker）下 0.44s/batch(100) < T4 训练步时（0.8~1.3s），不构成训练瓶颈。
- [x] 时间划分无泄漏（train/val/test 帧区间互斥）—— `window_starts`（窗口完全在片内 [i0, i1−t_win]）+ DEFAULT_SLICES 全覆盖 1501 帧；测试 `TestTimeSlices` 3 项 + `TestPoolStaysWithinSplit`。
- [x] 每样本 256 条×16 步×7 通道；标签与种子点 IVD 阈值判定一致 —— 测试 `test_sample_shape_matches_model_input`（(16,256,7) float32 有限无 NaN 无 -1000）+ `test_labels_match_seed_ivd_threshold`（label==1 ⇒ 重播种后种子处 IVD ≥ τ，label_field 为弱标签同一 φ 径源）+ 模型前向联测（(B,256)）。
- [x] 正样本占比≈50%（过采样生效）—— 池 = 正（patch 内 ≥1 条涡迹线，`weak_labels.patch_positive_map` 单公式）50%/负 50% 放回采样；合成场实测 0.50；真实数据池（可用 patch 口径）实测正池 43.5% → 过采样后 0.50。

## 完成记录

**做了什么**（2026-08-25，commit `7af7fc4`）：
- `dataset.py`（新）：`prepare_dataset`（nc h5py 流式或内存数组 → u/v/ivd/label/mask memmap + meta.json：slices/taus/speed_max/ivd_mu/sigma 等；支持复用票 02/04 产物）、`WeakLabelPathlineDataset`（on-the-fly：池构建 → set_epoch 50% 过采样 → 提取+归一化+标签 → `((dummy_field(1,1,1,1), pathlines), labels)`）、`normalize_pathlines`（4 通道归一化）、`window_starts`/`patch_locations`、`_patch_usable`（不可用 patch 过滤）。
- `extractor.py`：新增 `extract_pathlines_batched`（向量化批量 RK4：`_interp_pair` 时间标量化+8 角合并 gather、`_integrate_batched` 冻结语义=截断重复末点；rng per-k 确定性派生 `SeedSequence([base,k,attempt])`——与逐条版重播种为同语义不同随机实现；阶段 3 重试复用标量函数 → 公式同源）+ `nearest_cell`（物理→最近格单一公式，替代 mask_at/批量/标签四处重复）。
- `weak_labels.py`：`patch_seed_offsets` 提取（正样本判据与数据集池共用的种子格偏移单一公式）；`patch_positive_map` 重构为共享判据（positive_patch_fraction 行为不变，现存测试守护）。
- 测试：`tests/test_dataset.py` 21 项（时间片/归一化/准备/数据集/不可用 patch 过滤/真实冒烟）+ `tests/test_extractor.py` 追加 8 项（批量-逐条一致性守护：干净流场/截断/ivd 通道/时变/插值内核守护/有效性/复现）。**全量 113 passed**（前置票全绿）。

**验证证据**：
- 真实数据复验（`outputs/dataset/`，复用票 02/04 产物）：prepare 成功（slices 全覆盖、taus 与票 04 逐位一致 3.3272/3.16848/3.14344、speed_max=4.63067、ivd_mu=0.8867/sigma=3.5849）；样本 (16,256,7) 有限、t∈[0,0.25]、|u|≤1、标签 ∈{0,1}；正样本占比（60 样本）0.500；时变比例 1.000（修复后）。
- 池规模实测：可用 patch 128/216（**88 个不可用**：种子-中心线段全固体——壁面区/圆柱包围区；票 03 边界"全固体 patch 应避开"的精确化）；池组合 31,360（正 13,652/43.5%、负 17,708）；票 04 统计 37.86% 为**含不可用 patch**口径（其正比例被全固体 patch 拉低），本票池为**可用 patch**口径——两者不矛盾（完成记录与 HANDOFF §11 已披露）。
- 单样本计时：median 35.7ms / p90（预热后）157ms / 首取 836ms（页加载）；池构建 0.85s（真实数据）/0.15s（合成）。

**/code-review 双轴处置**：
- **Standards**：无硬性违规。① Duplicated Code（"物理→最近格"4 处重复 + rint/floor+0.5 口径不一）→ 提取 `extractor.nearest_cell` 单一公式（mask_at/批量 phase1/weak_labels 种子偏移/dataset 三处全部改用；floor(g+0.5) 口径统一）；② `_gather` 闭包延迟绑定 → w 前移；③ Data Clumps（normalize 9 参数等）→ 保留（与既有代码风格一致，判断项不处理）。
- **Spec**（关键缺陷 1 项 + 弱项 2 项，均已处置）：① **时变冻结 bug**：`_extract` 把窗口切片场配了**全场 tdim** → 时间映射 clamp 到窗口末帧 → u/v/ivd 通道与 RK4 速度场全部冻结在末帧（实测复现：标量 CH_U=100→123 演化 vs 批量恒 123）→ 修复为**窗口切片配窗口 tdim**（`_extract`、extractor CLI、真实测试同改；新增时变守护测试 2 项——其中"插值内核守护"还连带发现并修复了 `_interp_pair` 的 x1/y1 越界角用 x0+1 而非 ceil 的**第二处公式漂移**）；② `DEFAULT_MIN_SAMPLES_PER_EPOCH` 定义未用 → 删除常量化（规格"下限 20000"为训练配置语义约束，注释说明）；③ `seeds_for` 未含短迹线重试与 `_extract` 不一致 → 改为直接复用 `_extract`（"三者一致"严格成立）。
- **范围蔓延说明**：`_patch_usable`/池过滤超出票面字面，属票 03 边界（全固体 ValueError）在真实数据上的必要防护（不排除则真实数据采样必然崩溃），已注释披露，保留。

**实现边界（供票 06+ 衔接）**：
- 批量版 rng 为 per-k 确定性派生（与逐条版单流 rng 不同构）：确定性路径（无重播种/无重试消费）下与逐条版逐元素一致（守护测试）；随机路径下为同一重播种语义的不同随机实现。数据集层"池判定/标签/提取"三者一致由组合级 rng base（`_comb_rng_base`）+ 复用 `_extract` 保证。
- 性能：on-the-fly 提取 ~35ms（用户确认不纠结）；Kaggle 训练脚本（票 06）建议 DataLoader workers≥8 隐藏加载时间。
- `outputs/dataset/`（gitignore，走 Kaggle Dataset）：meta.json + u/v/ivd/label/mask memmap（合计 ≈1.3GB）。

**未决**：无阻塞。下一步：按 frontier 票 06（训练脚本，依赖 01、05）可启动。
