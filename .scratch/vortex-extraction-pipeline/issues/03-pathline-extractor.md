# 03: 迹线提取（extractor）

**What to build:** 全局场迹线积分器：从种子点出发按 RK4 + 三线性时空插值积分（每输出步 4 子步）生成每条迹线；每样本 256 条 = 64 组 × 4 轴向卫星点（不含中心），Δ = patch 边长×0.05，组主序编组；种子落固体 → 重播种；迹线入固体 → 截断并重复末点（不引入 -1000 毒值）；位置按 patch 归一化到 [-1,1]（可超界）；全局场积分允许迹线离开 patch。

**Blocked by:** 02（掩膜处理依赖）

**Status:** done

- [x] 3~5 个 patch×24 帧窗口生成迹线目检通过（跟随流场、不穿固体）
- [x] 每样本迹线计数恒为 256、7 通道特征有限无 NaN
- [x] 固体区种子重播种生效；截断轨迹末点重复、无 -1000 毒值
- [x] 全局场积分允许迹线离开 patch

## 完成记录

- **2026-08-25 完成**（commit `85ad7e3` 代码 + 本记录所在 docs 收尾 commit；推送由用户在普通终端执行，须带 `-c http.sslBackend=openssl`）。
- **做了什么**：
  - 新增 `extractor.py`（≈470 行）：`trilinear_interp`（标量三线性时空插值，越界 clamp 到边界格，参考 flowlineIntegral.py 语义）、`velocity_at`、`mask_at`（物理坐标 → 最近格掩膜查询，3D 掩膜取第 0 帧）、`integrate_pathline`（全局场 RK4 积分：每输出步 4 子步、输出步新点出域/入固体停止且不采纳该步，参考 C++ `PathhlineIntegrationRK4v2`）、`pad_repeat_last`（截断轨迹末点重复到 L，**无 -1000 毒值**，C++ 参考用 -1000 填充而本项目决策不用）、`reseed`（种子落固体 → 仿 C++ `JittorReSeeding`：seed + shift×(center−seed)，shift~U[0.00001,0.5] 随机重试 + 线性细扫兜底 + 全固体 ValueError）、`patch_geometry`（patch 物理几何统一口径）、`interp_path`（向量化批量插值，特征采样用）、`extract_pathlines`（样本组装：64 组 × 4 轴向卫星 = 256 条、组主序编组、组中心 patch 内 [0.1,0.9] 区间 8×8 网格仿 C++ `GridCrossSampling`、Δ=patch 边长×0.05、位置按 patch 归一化 [-1,1] 可超界、7 通道 = [px,py,t,ivd,distance,u,v]、ivd 场可选（None → 第 4 通道 0，票 04 后接入）、积分太短（≤2 点，仿 C++ suc 判据 `size>2`）→ 朝 patch 中心大幅移动重试最多 3 次、默认 dt_out=(T_win−1)×dt/(L−1) 使 24 帧窗口恰覆盖 16 输出步）、`plot_pathlines`（目检图：速度模底图 + 掩膜轮廓 + 256 迹线按组着色 + 种子点，物理坐标 extent 对齐）、CLI（h5py 直读中文路径、`--mask` 支持 2D/3D 掩膜、无 mask 时逐帧流式取与兜底、目检图 + npy 落盘）。
  - 新增 `tests/test_extractor.py`（28 项测试）：切片 1 插值（常值/线性解析/时间线性/越界 clamp/向量-标量一致性守护）、切片 2 积分器（常速度直线解析、子步精度对比 1 vs 4、空间/时间出域停止、入固体停止）、切片 3 补齐（重复末点/等长原样/单点重复/无 -1000）、切片 4 重播种（朝中心方向/非固体原样/全固体 ValueError）、切片 5 组装（形状 256 恒等/组主序 ±0.1 卫星/通道语义 t·distance·u·v/ivd 可选/条带截断端到端/重播种端到端/积分太短重试兜底/离开 patch 超界）、真实数据（4 窗口属性、种子非固体、数值目检：有效点不穿固体 + 切向一致性余弦 >0.9）、目检 PNG 落盘。
- **验证证据**：
  - `python -m pytest -q` → **46 passed**（票 01 8 项 + 票 02 10 项 + 本票 28 项；本地 CPU，Python 3.12）；
  - CLI 实测：`python extractor.py <nc> --patch-yx 100,280 --frames 400,800,1200 --visualize` 与 `--patch-yx 0,60 --frames 1200 --visualize` → 4 个窗口均 (16,256,7)、无 NaN、无 -1000；目检图 `outputs/pathlines/pathlines_t*_y*_x*.png`（outputs/ 被 gitignore，走 Kaggle Dataset）；
  - **目检通过（用户肉眼复核 2026-08-25）**：迹线跟随流场、不穿固体；另附数值目检（测试断言）：有效点 0 个穿入固体、位移-速度切向一致性余弦均值 >0.9、有效点占比 >50%。
- **/code-review 结果与处置**（双轴并行评审，commit 前修复）：
  - Standards 轴：无文档化标准硬违规。判断性处置：① 插值公式双份（标量/向量）漂移风险 → 新增一致性守护测试（随机点含越界，两版必须一致）+ 注释互指；② CLI 无 --mask 兜底循环与 geometry.static_mask_from_speed 同判据 → 保留（逐帧流式读取避免全量 810MB 驻留），注释交叉引用 ε=1e-5 口径；③ Data Clumps（xdim/ydim/tdim 结伴 7 处签名）→ 不采纳（数值管线小函数打包引入抽象成本 > 收益，记录理由）；④ `off_x` 命名 → 改名 `delta_x/delta_y`（对齐 HANDOFF Δ 术语）；⑤ 可视化/CLI 归属 extractor → 保留（仿票 02 geometry.py 惯例：plot_mask+CLI 同模块；迹线目检属本票验收，evaluate.py 是模型评估图）。
  - Spec 轴：核心规格全部落地（4 子步/256 组主序/Δ=patch×0.05/重播种/截断重复无毒值/归一化可超界/允许离开 patch/属性测试）。处置：① 积分太短重试分支无测试 → 新增 `test_short_integration_retries_reseed`（u=1 条带场景：近固体种子 n<3 → 重试后无静止退化迹线）；② CLI `--mask` 传 2D 掩膜会静默取错形状 → 已修（2D/3D 兼容）；③ 目检图未自动断言 → 新增 PNG 落盘测试；④ 输出步粒度检查（子步可能穿越固体 ≤ 一个 dt_out）→ 不改（与 C++ 参考语义一致；实测一步最大位移 ≈5.3 格 < 最薄固体 7 格，实际数据不会穿越），记入本记录；⑤ 归一化边界：extractor 只做位置归一化（票 03 验收），t→[0,1]×t_scale、ivd 标准化、u/v÷max、distance 归一化口径属票 05 dataset 职责，已记入 HANDOFF §11。
- **事实回写（HANDOFF）**：无新增数据集事实（§2 不变）。§11 已追加变更日志；正文无需改动（§4 extractor 职责行与实现一致）。
- **遗留**：无阻塞。ivd 通道当前为占位（票 04 提供 IVD 场后由票 05 接入）；下一步按 frontier：票 04（弱标签，依赖 02）与 05（数据集，依赖 03、04）可启动。
