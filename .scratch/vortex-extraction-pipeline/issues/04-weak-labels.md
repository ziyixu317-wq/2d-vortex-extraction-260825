# 04: 弱标签与 τ 定值（weak_labels）

**What to build:** 弱标注生成器：中心差分涡量 ω=∂v/∂x−∂u/∂y；IVD=|ω−5×5 局部邻域均值|；固体区 IVD=0；标签 = 种子点处 IVD≥τ（默认 95 分位数、逐时间片）+ 5×5 最小面积连通域过滤；输出 τ 对比图并定值回写参数表；统计正样本占比（支撑过采样设计）。附带 Q-criterion 参考图用于目检对照。

**Blocked by:** 02（固体区 IVD=0 依赖掩膜）

**Status:** done

- [x] IVD/Q 图目检：涡街与拐角回流区成连通块、非涡区干净
- [x] τ 定值写入参数表（HANDOFF §6）
- [x] 正样本占比统计输出
- [x] 固体区 IVD=0

## 完成记录（2026-08-25 票 04 会话）

新增 `weak_labels.py`（弱标注生成器）+ `tests/test_weak_labels.py`（32 项验收测试，全量 84 passed；`extractor.py` 提取出 `seeding_grid` 供正样本统计与迹线提取共用单一种子公式，`tests/test_extractor.py` 追加 2 项一致性守护）。

**功能**：ω=∂v/∂x−∂u/∂y（中心差分、边界单边一阶、等距网格守卫）→ IVD=|ω−5×5 邻域均值|（edge pad、窗口含中心 25 格）→ 固体区 IVD=0（复用票 02 掩膜）；2D Q=−(∂u/∂y)(∂v/∂x)−½[(∂u/∂x)²+(∂v/∂y)²]（参考图对照）；标签 = 逐时间片 τ 二值化 + 5×5 面积（25 格）连通域过滤（复用 geometry.label_components，8 邻接）+ 固体强制 0；τ = 流体区 IVD 第 95 分位（排除固体 0 值）；正样本占比统计（种子判据 + 窗口起点步长 4 帧）；目检图（IVD+Q>0+标签、τ 敏感性 90/95/97.5/99 并排）；CLI（h5py 直读中文路径、逐帧流式算 IVD、落盘 i vd.npy/label_field.npy/weak_label_meta.json）。

**验收证据**：
1. **IVD/Q 图目检（✓）**：`outputs/weak_labels/ivd_q_t{400,1200,1300}.png`（覆盖 train/val/test 三时间片）+ `tau_sensitivity_t{...}.png`：涡街（入口圆柱 (0,0) 下游与拐角后管道圆柱 (3,1) 下游 Kármán 街）、拐角回流区（x≈2.5 竖带）成连通块；主流动非涡区干净（黑）；标签（lime）仅在涡核 IVD 亮区；敏感性图显示 p95 为合理默认（p90 含剪切带噪声、p97.5/99 仅剩涡核）。本会话 AI 目检（读图核验），**建议用户肉眼复核**完成最后确认。
2. **τ 定值（✓，已回写 HANDOFF §6）**：train **3.3272** / val **3.16848** / test **3.14344**（95 分位、逐时间片、流体区统计）；标签正格占比 0.0280。
3. **正样本占比（✓，`outputs/weak_labels/weak_label_meta.json`）**：种子判据（64 组×4 卫星、extractor.seeding_grid 同一公式、落固体种子不计）+ 窗口起点步长 4 帧：train 0.3738（20265/54216）、val 0.3876（5275/13608）、test 0.3882（5283/13608）、global **0.3786**（30747/81216）→ 50% 过采样需 **1.321×**（支撑票 05 设计：近自然均衡，过采样负担小）。
4. **固体区 IVD=0（✓）**：合成测试（掩膜格强制 0）+ 真实数据（3 帧切片，`np.all(ivd[:, mask2d] == 0)`；掩膜格数 = 28213 与 §2 已核实事实一致）。

**实现边界与口径（供票 05 衔接）**：
- 接口：`compute_ivd / compute_tau / build_label_field / binary_label / filter_min_area / positive_patch_fraction / q_criterion / plot_ivd_q / plot_tau_sensitivity`+ CLI；`ivd.npy`（(1501,150,450) float32）+ `label_field.npy`（uint8）为票 05 dataset 输入（memmap 预计算 IVD 口径一致）。
- **时间片口径明确化（数据正确性修复）**：「帧 0-1000/1000-1250/1250-1500」按闭包读法 = train 帧 0..1000（含 t=10.0）、val 1001..1250（含 t=12.5，帧 250）、test 1251..1500（含 t=15.0）→ `DEFAULT_SLICES = {(0,1001),(1001,1251),(1251,1501)}`（Python 半开切片；全覆盖 1501 帧、无泄漏，有测试守护）。票 05 的 dataset 时间划分应沿用此口径。
- **正样本占比的相对偏差**：种子落固体时未模拟重播种（重播种含随机性），重播种后可能为正的极小情形被忽略 → 低估 ≤2%（固体种子占比×正区比），对 1.32× 过采样设计无实质影响（已在 docstring 披露）。
- **显示帧**：CLI `--frames` 默认 400（train）/1200（val）/1300（test）。
- 输出产物走 gitignore（outputs/），权威数字在 HANDOFF §6/§11 与本记录；图谱走 Kaggle Dataset 随代码可复现（一条 CLI 命令）。

**/code-review 双轴处置**（固定点 21cb4ca，并行子代理）：
- **Standards**：硬性 1 条——`compute_tau` docstring「(100−percentile) 分位」与实现及 §6「95 分位数」矛盾 → 改「第 percentile 百分位」（commit 前修复）。判断项 5：① 标签二值化+掩膜+面积过滤形状在 build_label_field/plot_tau_sensitivity 重复 → 抽 `_labeled_mask` 共享；② binary_label 逐帧 tau 数组承诺未用且广播错位 → 删（标量语义）；③ (taus,slices) 数据团 → 维持（两概念语义不同、成对性仅模块内 3 处且紧邻，合并不降复杂度）；④ frame_indices 命名含糊 + CLI 未落实 4 帧步长 → 参数改名 + CLI 用 np.arange(i0,i1,4)（与 Spec 轴 (c)1 合并解决）；⑤ `mask2d==False`/`_slice_of` 后置 → 修（`~mask2d`、前移）。
- **Spec**：(a)1 票收尾文档 → 本记录补全；(a)2 时间划分守护测试 → 新增 TestTimeSlices 2 项（全覆盖 1501 帧、边界帧归属闭包）；(b) 三处范围蔓延（流体区 τ 总体、边界差分+等距守卫、四候选敏感性图）→ 保留并记录（§6 已注明 τ 口径；保卫与敏感性图为必要工程化，票面「τ 对比图」字面兼容）；(c)1 正样本占比口径 → 重写为种子判据 + 步长 4 帧（本记录）；(c)2 compute_ivd docstring「返回 (T,Y,X)」与实际保形 → 改「返回与输入同形状」。

未决：无阻塞。下一步：按 frontier 票 05（数据集，依赖 03、04）可启动。
