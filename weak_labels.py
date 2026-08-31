"""弱标签（weak_labels.py）——04 票：弱标注生成器（IVD/Q-criterion + τ 定值）。

领域词汇（HANDOFF §4/§6，唯一权威）：
- 涡量 ω = ∂v/∂x − ∂u/∂y（中心差分，边界单边一阶；等距网格）；
- IVD = |ω − 5×5 局部邻域均值|：邻域窗口含中心（25 格），边界 edge pad
  （与 extractor 越界 clamp 同语义）；
- 固体区 IVD=0（依赖票 02 掩膜；geometry_meta/mask.npy 逐数据集预处理，
  本模块接受 (Y,X) 或 (T,Y,X) 掩膜，取第 0 帧）；
- 标签 = 种子点处 IVD ≥ τ（默认 85 分位数、逐时间片）+ 5×5 最小面积
  （min_area=25 格）连通域过滤（复用 geometry.label_components，8 邻接）；
- τ 的分位数在**流体区**统计（排除固体 0 值，避免其 41.8% 占比污染）；
- 正样本 = patch 内存在 ≥1 条涡迹线（t0 帧标签场在 patch 内有正格）；
- 2D Q-criterion Q = ‖Ω‖²/2 − ‖S‖²/2 = −(∂u/∂y)(∂v/∂x) − ½[(∂u/∂x)² + (∂v/∂y)²]，
  仅作参考图目检对照（HANDOFF §6 风险预案：备选 Q-criterion 标签对照）；
- 多阈值敏感性报告（multi_tau_report，票 07 延伸）：对既有 IVD 场重算
  95/90/85/80 分位 + 固定阈值 × 面积过滤的标签覆盖率/连通块数/正样本占比
  统计与目检图（含论文 Fig.6 列 1 风格的 IVD 白色等值线）——HANDOFF §7
  风险预案「多阈值敏感性报告」的落地。

实现约束：h5py 直读中文路径（数据读取在 geometry.load_field）；纯 numpy/python
（遵守 §2 依赖清单，无 scipy；连通域过滤复用 geometry 的自写并查集）。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import extractor
import geometry

# --------------------------------------------------------------------------- 时间片划分（HANDOFF §6）

# 时间划分 train [0,10]s / val (10,12.5] / test (12.5,15]，帧 i 的 t = i×dt
# （tdim[0]=0, dt=0.01）。闭包口径：t=10.0（帧 1000）∈ train、t=12.5（帧 1250）∈ val、
# t=15.0（帧 1500）∈ test → 全覆盖 1501 帧且无时间泄漏（HANDOFF §4「帧
# 0-1000 / 1000-1250 / 1250-1500」的精确化）。Python 半开切片语义：
DEFAULT_SLICES = {"train": (0, 1001), "val": (1001, 1251), "test": (1251, 1501)}

# 最小涡面积（HANDOFF §6）：5×5 连通域过滤
DEFAULT_MIN_AREA = 5 * 5

# τ 默认分位（HANDOFF §6；票 07 延伸：p95 弱标签相比论文 Fig.6 列 1 捕获稀疏
# → 下探至 p85，覆盖率 4.8%→14.8% 且 5×5 过滤后结构块数 6.8→9.8/帧）
DEFAULT_PERCENTILE = 85.0


# --------------------------------------------------------------------------- 辅助

def _mask2d(mask):
    """掩膜规格化为 (Y,X) bool：接受 (Y,X) 或 (T,Y,X)（固体几何时间不变，取第 0 帧）。"""
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 3:
        m = m[0]
    elif m.ndim != 2:
        raise ValueError(f"掩膜维度需为 (Y,X) 或 (T,Y,X)，实际 {m.shape}")
    return m


def _spacing(coord, axis_name):
    """等距坐标轴格距（非等距网格不支持：中心差分需均匀间距）。

    容差按 float32 坐标轴噪声水平（nc 元数据多为 float32，diff 噪声 ~1e-6）：
    捕获显著非等距（>0.1%），不误伤浮点舍入。
    """
    c = np.asarray(coord, dtype=np.float64)
    dx = c[1] - c[0]
    diffs = np.diff(c)
    if not np.allclose(diffs, dx, rtol=1e-3, atol=1e-12):
        raise ValueError(f"{axis_name} 坐标非等距，中心差分不支持")
    return dx


# --------------------------------------------------------------------------- 涡量 ω

def _diff(field, axis, d):
    """沿 axis 的中心差分（内部），边界用单边一阶（线性场精确）；d = 格距。"""
    f = np.asarray(field, dtype=np.float64)
    out = np.zeros_like(f)
    sl = [slice(None)] * f.ndim
    sl[axis] = slice(1, -1)
    out[tuple(sl)] = (np.take(f, np.arange(2, f.shape[axis]), axis=axis)
                      - np.take(f, np.arange(0, f.shape[axis] - 2), axis=axis)) / (2.0 * d)
    lo = list(sl); lo[axis] = 0
    hi = list(sl); hi[axis] = -1
    out[tuple(lo)] = (np.take(f, 1, axis=axis) - np.take(f, 0, axis=axis)) / d
    out[tuple(hi)] = (np.take(f, -1, axis=axis) - np.take(f, -2, axis=axis)) / d
    return out


def vorticity(u, v, xdim, ydim):
    """涡量 ω = ∂v/∂x − ∂u/∂y（中心差分，边界单边一阶；HANDOFF §4）。

    返回 (T,Y,X) float64（输入可为 (Y,X)，返回同形状）。"""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    dx = _spacing(xdim, "xdim")
    dy = _spacing(ydim, "ydim")
    return _diff(v, -1, dx) - _diff(u, -2, dy)


# --------------------------------------------------------------------------- 5×5 邻域均值 / IVD

def neighborhood_mean(field, k=5):
    """k×k 局部邻域均值（窗口含中心；边界 edge pad——与 extractor clamp 语义一致）。

    输入 2D (Y,X) 或 3D (T,Y,X)，返回同形状 float64。pad p=k//2 后滑动窗口恰好
    与输入同尺寸（edge pad 下每格窗口恒 k×k = 25 格，无归一化歧义）。
    """
    f = np.asarray(field, dtype=np.float64)
    if f.ndim not in (2, 3):
        raise ValueError(f"输入维度需为 (Y,X) 或 (T,Y,X)，实际 {f.ndim}")
    p = k // 2
    pad_width = ((p, p), (p, p)) if f.ndim == 2 else ((0, 0), (p, p), (p, p))
    padded = np.pad(f, pad_width, mode="edge")
    win = sliding_window_view(padded, (k, k), axis=(-2, -1))
    return win.mean(axis=(-2, -1))


def ivd_from_vorticity(omega, k=5):
    """IVD = |ω − k×k 局部邻域均值|（HANDOFF §4：5×5 邻域均值）。"""
    omega = np.asarray(omega, dtype=np.float64)
    return np.abs(omega - neighborhood_mean(omega, k=k))


def compute_ivd(u, v, xdim, ydim, mask=None, k=5):
    """完整 IVD 场：ω → IVD → 固体区 IVD=0（验收 4）。

    mask 为 (Y,X) 或 (T,Y,X) 固体掩膜（None = 无障碍物数据集的空路径）。
    返回与输入同形状 float64（(T,Y,X) 或 (Y,X)，vorticity 保形）。
    """
    ivd = ivd_from_vorticity(vorticity(u, v, xdim, ydim), k=k)
    if mask is not None:
        m2 = _mask2d(mask)
        ivd[:, m2] = 0.0        # 固体区置零（逐帧广播）
    return ivd


# --------------------------------------------------------------------------- Q-criterion（参考对照）

def q_criterion(u, v, xdim, ydim):
    """2D Q = ‖Ω‖²/2 − ‖S‖²/2 = −(∂u/∂y)(∂v/∂x) − ½[(∂u/∂x)² + (∂v/∂y)²]。

    Q>0 = 旋转占优（涡旋），Q<0 = 应变占优。仅作参考图对照（HANDOFF §7 预案）。
    返回 (T,Y,X) float64。
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    dx = _spacing(xdim, "xdim")
    dy = _spacing(ydim, "ydim")
    ux = _diff(u, -1, dx)
    uy = _diff(u, -2, dy)
    vx = _diff(v, -1, dx)
    vy = _diff(v, -2, dy)
    return -0.5 * (ux ** 2 + vy ** 2) - uy * vx


# --------------------------------------------------------------------------- τ（逐时间片 95 分位）

def compute_tau(ivd, mask, slices, percentile=DEFAULT_PERCENTILE):
    """τ = 流体区 IVD 的第 percentile 百分位（默认 85 分位——票 07 延伸定案），逐时间片（HANDOFF §6）。

    slices: dict {name: (i0, i1)} 帧索引区间（默认 DEFAULT_SLICES 的时间划分）。
    分位数统计范围 = 非固体格（排除 IVD=0 的固体，其占比 41.8% 会污染阈值）。
    mask 为 None 时视为无固体（空掩膜数据集）。
    返回 dict {name: tau}。
    """
    ivd = np.asarray(ivd, dtype=np.float64)
    fluid = ~_mask2d(mask) if mask is not None else np.ones(ivd.shape[1:], dtype=bool)
    taus = {}
    for name, (i0, i1) in slices.items():
        vals = ivd[i0:i1][:, fluid]
        if vals.size == 0:
            raise ValueError(f"时间片 {name} ({i0}:{i1}) 无流体格")
        taus[name] = float(np.percentile(vals, float(percentile)))
    return taus


# --------------------------------------------------------------------------- 标签（二值化 + 面积过滤）

def binary_label(ivd, tau):
    """标签 = IVD ≥ τ（种子点处判据）；tau 为标量阈值（HANDOFF §4：IVD(种子,t0)≥τ）。"""
    return np.asarray(ivd) >= float(tau)


def _labeled_mask(ivd2d, tau, mask2d, min_area):
    """单帧标签 = (IVD ≥ τ) & 非固体 → 5×5 面积过滤（build_label_field 与
    plot_tau_sensitivity 的共享形状，防双份逻辑漂移）。"""
    bin_ = binary_label(ivd2d, tau) & (~mask2d)
    return filter_min_area(bin_, min_area=min_area)


def filter_min_area(label2d, min_area=DEFAULT_MIN_AREA, connectivity=8):
    """5×5 最小面积连通域过滤（HANDOFF §6 最小涡面积；复用 geometry 自写并查集）。

    返回 (Y,X) uint8：面积 < min_area 的连通块置 0，其余保留。
    connectivity ∈ {4, 8} 与 geometry.component_stats 同口径（8 邻接含对角）。
    """
    lab = np.asarray(label2d, dtype=bool)
    out = np.zeros(lab.shape, dtype=np.uint8)
    if not lab.any():
        return out
    labels, n = geometry.label_components(lab, connectivity=connectivity)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    sizes[0] = 0                     # 背景（label 0）不保留
    keep = sizes >= min_area
    return keep[labels].astype(np.uint8)


def build_label_field(ivd, mask2d, taus, slices, min_area=DEFAULT_MIN_AREA):
    """标签场 = 逐时间片 τ 二值化 + 5×5 面积过滤 + 固体区强制 0。

    返回 (T,Y,X) uint8。taus/slices 为 {name: tau}/{name: (i0,i1)} 配对；
    未被任何时间片覆盖的帧不产生正标签（tau=inf 安全缺省，调用方应覆盖全部帧）。
    """
    ivd = np.asarray(ivd, dtype=np.float64)
    m2 = _mask2d(mask2d)
    T = ivd.shape[0]
    tau_per_frame = np.full(T, np.inf, dtype=np.float64)
    for name, (i0, i1) in slices.items():
        tau_per_frame[i0:i1] = float(taus[name])
    out = np.zeros(ivd.shape, dtype=np.uint8)
    for t in range(T):
        out[t] = _labeled_mask(ivd[t], tau_per_frame[t], m2, min_area)
    return out


def label_frame_at_cfg(ivd2d, mask2d, slices, cfg, frame, min_area=DEFAULT_MIN_AREA):
    """单帧标签（τ 敏感性评估用）：按 frame 所属时间片从 cfg 取 τ 后重标单帧。

    cfg 可为 dict {时间片名: tau}（分位/μ+3σ 逐时间片——compute_tau_candidates
    口径）或标量（固定 τ）；slices 为 {时间片名: (i0, i1)}。复用 _tau_for_frame
    与 _labeled_mask 单一公式（与 build_label_field 同口令，防双份漂移）。
    frame 未被任何时间片覆盖时 fail loud（评估帧应在时间片内）。
    返回 (Y,X) uint8。
    """
    tau = _tau_for_frame(cfg, frame, slices)
    if tau is None:
        raise ValueError(f"帧 {frame} 不在任何时间片内（slices={dict(slices)}）")
    return _labeled_mask(ivd2d, tau, mask2d, min_area)


# --------------------------------------------------------------------------- 正样本占比（支撑过采样）

def patch_seed_offsets(patch_size=(32, 32), xdim=None, ydim=None,
                       groups=(8, 8), delta_frac=0.05):
    """256 种子相对 patch 原点 (0,0) 的格偏移 (off_y, off_x)，各 (K,) intp。

    种子 → 最近格（extractor.nearest_cell 单一公式：四舍五入 floor(g+0.5)，
    与 mask_at 同口径）；网格等距故偏移与 patch 绝对位置无关。正样本判定
    （patch_positive_map）与数据集池构建（票 05：不可用 patch 排除）共用该
    单一公式，防双份漂移。
    """
    seeds = extractor.seeding_grid((0, 0), patch_size, xdim, ydim,
                                   groups, delta_frac)
    ph, pw = patch_size
    off_y, off_x = extractor.nearest_cell(seeds[:, 0], seeds[:, 1], xdim, ydim)
    off_y = np.clip(off_y, 0, ph - 1)
    off_x = np.clip(off_x, 0, pw - 1)
    return off_y, off_x


def patch_positive_map(label2d, xdim, ydim, patch_size=(32, 32),
                       stride=(16, 16), groups=(8, 8), delta_frac=0.05):
    """单帧 → (nY, nX) bool：各 patch 位置是否存在 ≥1 条涡迹线（HANDOFF §4 口径）。

    涡迹线判据 = 种子（64 组 × 4 轴向卫星，与 extractor.seeding_grid 单一公式）
    处标签为 1；固体区标签恒 0（build_label_field 强制）→ 落固体的种子不计正。
    正样本占比统计（positive_patch_fraction）与数据集样本池（票 05）共用此函数，
    防判据双份漂移。返回值 (nY, nX) 与 patch_locations 同序（y 外 x 内）。
    """
    lab = np.asarray(label2d, dtype=bool)
    if lab.ndim != 2:
        raise ValueError(f"标签为单帧 (Y,X)，实际 {lab.shape}")
    ph, pw = patch_size
    sy, sx = stride
    off_y, off_x = patch_seed_offsets(patch_size, xdim, ydim, groups, delta_frac)
    H, W = lab.shape
    if H < ph or W < pw:
        raise ValueError(f"帧尺寸 {lab.shape} 小于 patch {patch_size}")
    ys = np.arange(0, H - ph + 1, sy)
    xs = np.arange(0, W - pw + 1, sx)
    seed_y = ys[:, None, None] + off_y[None, None, :]
    seed_x = xs[None, :, None] + off_x[None, None, :]
    return lab[seed_y, seed_x].any(axis=-1)


def positive_patch_fraction(label_tyx, xdim, ydim, patch_size=(32, 32),
                            stride=(16, 16), frame_indices=None,
                            groups=(8, 8), delta_frac=0.05):
    """正样本占比统计：正样本 = patch 内存在 ≥1 条涡迹线（HANDOFF §4 dataset 口径）。

    与 patch_positive_map 共享判据（单一公式）；近似说明见 patch_positive_map：
    种子落固体时未做重播种（extractor.reseed 含随机性），重播种后可能为正的
    极小情形被忽略——统计对此低估 ≤2%（固体种子占比 × 正区比例），对
    过采样设计（0.5/占比）无实质影响（此边界已在完成记录中披露）。

    统计帧由 frame_indices 显式给定（HANDOFF §6：窗口起点步长 4 帧；None = 全部帧，
    每帧等权）。返回值 {"n_patches", "n_positive", "fraction"}。
    """
    lab = np.asarray(label_tyx)
    if lab.ndim == 2:
        lab = lab[None]
    if lab.ndim != 3:
        raise ValueError(f"标签场需为 (Y,X) 或 (T,Y,X)，实际 {lab.shape}")
    frame_idx = np.arange(lab.shape[0]) if frame_indices is None else np.asarray(frame_indices)
    n_pos = n_tot = 0
    for t in frame_idx:
        pos_map = patch_positive_map(lab[t], xdim, ydim, patch_size, stride,
                                     groups, delta_frac)
        n_pos += int(pos_map.sum())
        n_tot += int(pos_map.size)
    if n_tot == 0:
        raise ValueError("无有效 patch 位置（帧尺寸不足）")
    return {"n_patches": n_tot, "n_positive": n_pos,
            "fraction": float(n_pos) / float(n_tot)}


# --------------------------------------------------------------------------- 多阈值敏感性报告（HANDOFF §7 预案）

# 候选 τ 默认集：逐时间片 95/90/85/80 分位 + 若干固定低阈值（绝对阈值仅对当前
# 数据集有效——跨数据集 IVD 量纲不同，主口径仍为分位；票 07 延伸定案 p85）。
DEFAULT_REPORT_PERCENTILES = (95.0, 90.0, 85.0, 80.0)
DEFAULT_REPORT_FIXED = (2.5, 2.0, 1.5, 1.0)
DEFAULT_REPORT_MIN_AREAS = (25, 9, 1)


def compute_tau_candidates(ivd, mask, slices, percentiles=DEFAULT_REPORT_PERCENTILES,
                           fixed_values=DEFAULT_REPORT_FIXED):
    """多阈值候选 → {name: tau_cfg}；percentile 配置 → {片名: τ}，fixed → 标量。

    与 multi_tau_report 共用（单一公式，防双份漂移）；name 格式
    "p{percentile}"（分位）/"fixed{value}"（固定值）。
    """
    cfgs = {}
    for p in percentiles:
        cfgs[f"p{int(p)}"] = compute_tau(ivd, mask, slices,
                                          percentile=float(p))
    for v in fixed_values:
        cfgs[f"fixed{v:g}"] = float(v)
    return cfgs


def _tau_for_frame(cfg, frame, slices):
    """候选配置 + 帧 → 该帧 τ（分位配置按所属时间片取；固定配置恒等；未覆盖 None）。"""
    if isinstance(cfg, dict):
        for name, (i0, i1) in slices.items():
            if i0 <= frame < i1:
                return cfg[name]
        return None
    return float(cfg)


def multi_tau_report(ivd, mask, slices, xdim, ydim, out_dir, *,
                     percentiles=DEFAULT_REPORT_PERCENTILES,
                     fixed_values=DEFAULT_REPORT_FIXED,
                     min_areas=DEFAULT_REPORT_MIN_AREAS,
                     display_frames=(400, 1200, 1300), sample_step=25,
                     frame_step=4, title="", source_nc="", save_json=True):
    """多阈值敏感性报告（HANDOFF §7 预案「多阈值敏感性报告」；票 07 延伸落地）。

    对既有 IVD 场重算多阈值标签并输出：每候选 τ 的（正格占比[流体区]、
    平均连通块数[采样帧]、正样本占比[种子判据, 逐时间片+全局]）+ 目检图：
    - multi_tau_filled_t{t}.png：标签填充对比（行 = min_area，列 = 候选 τ）；
    - multi_tau_isocontour_t{t}.png：IVD 底图 + 候选 τ 白色等值线
      （论文 Fig.6 列 1 的呈现风格：IVD 参考是**连续场**，二值标签是**训练
      目标**——两个不同语义，此处并列供用户肉眼对照）。
    - multi_tau_stats.json：stats 落盘（save_json=True 时）。
    返回 {"stats": {...}, "out_dir": str, "source_nc": str}。
    """
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ivd = np.asarray(ivd, dtype=np.float64)
    m2 = _mask2d(mask) if mask is not None else np.zeros(ivd.shape[1:], dtype=bool)
    xdim = np.asarray(xdim, dtype=np.float64)
    ydim = np.asarray(ydim, dtype=np.float64)
    T = ivd.shape[0]
    cfgs = compute_tau_candidates(ivd, m2, slices,
                                  percentiles=percentiles,
                                  fixed_values=fixed_values)

    # ---- 统计
    stats = {}
    for name, cfg in cfgs.items():
        tau_per_slice = cfg if isinstance(cfg, dict) else None
        row = {"tau": None if tau_per_slice else float(cfg),
               "taus_per_slice": tau_per_slice}
        for ma in min_areas:
            lab = build_label_field(ivd, m2, {k: cfg for k in slices} if tau_per_slice is None else cfg,
                                    slices, min_area=ma)
            row[f"pos_cell_frac_ma{ma}"] = float(lab[:, ~m2].mean())
            frames = np.arange(0, T, sample_step)
            n_comp = 0
            for t in frames:
                if lab[t].any():
                    n_comp += geometry.label_components(
                        lab[t].astype(bool), connectivity=8)[1]
            row[f"mean_n_components_ma{ma}"] = float(n_comp) / max(len(frames), 1)
            if ma == min_areas[0]:
                per_slice = {}
                for sname, (i0, i1) in slices.items():
                    per_slice[sname] = positive_patch_fraction(
                        lab, xdim, ydim, frame_indices=np.arange(i0, i1, frame_step))
                row["pos_patch_fraction_per_slice"] = per_slice
                row["pos_patch_fraction_all"] = positive_patch_fraction(
                    lab, xdim, ydim, frame_indices=np.arange(0, T, frame_step))
            del lab
        stats[name] = row

    # ---- 目检图（填充对比 + 论文风格白色等值线）
    extent = [float(xdim[0]), float(xdim[-1]), float(ydim[0]), float(ydim[-1])]
    frames = [min(int(t), T - 1) for t in display_frames if int(t) >= 0]
    for t in frames:
        ivd_t = np.asarray(ivd[t], dtype=np.float64)
        # 1) 填充标签对比（行 = min_area，列 = 候选 τ）
        fig, axes = plt.subplots(len(min_areas), len(cfgs),
                                 figsize=(3.4 * len(cfgs), 3.4 * len(min_areas)))
        axes = np.atleast_1d(axes) if len(cfgs) == 1 or len(min_areas) == 1 else axes
        axes = np.asarray(axes).reshape(len(min_areas), len(cfgs))
        for ri, ma in enumerate(min_areas):
            for ci, (name, cfg) in enumerate(cfgs.items()):
                ax = axes[ri, ci]
                tau_t = _tau_for_frame(cfg, t, slices)
                lab_t = _labeled_mask(ivd_t, tau_t, m2, ma)
                ax.imshow(lab_t, origin="lower", aspect="auto", cmap="Greys",
                          vmin=0, vmax=1, extent=extent)
                ax.set_title(f"{name} tau={tau_t:.3g} ma={ma}\npos={lab_t.mean():.3f}",
                             fontsize=8)
                ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.suptitle(title or f"frame {t}: multi-tau weak labels "
                             f"(rows: min_area, cols: tau candidates)")   # 英文：无 CJK 字体环境防豆腐块
        fig.tight_layout()
        fig.savefig(out_dir / f"multi_tau_filled_t{t}.png", dpi=110)
        plt.close(fig)
        # 2) 论文风格：IVD 底图 + 白色等值线（连续 IVD 参考）
        fig, axes = plt.subplots(1, len(cfgs), figsize=(3.4 * len(cfgs), 3.8))
        if len(cfgs) == 1:
            axes = [axes]
        vmax = float(np.percentile(ivd_t[~m2], 99.0))
        for ax, (name, cfg) in zip(axes, cfgs.items()):
            tau_t = _tau_for_frame(cfg, t, slices)
            ax.imshow(ivd_t, origin="lower", aspect="auto", cmap="gray",
                      vmin=0, vmax=max(vmax, 1e-12), extent=extent)
            if tau_t is not None:
                ax.contour(ivd_t, levels=[tau_t], colors="white", linewidths=1.0,
                           extent=extent)
            if m2.any():
                ax.contour(m2, levels=[0.5], colors="red", linewidths=0.6,
                           extent=extent)
            ax.set_title(f"{name} tau={tau_t:.3g} (white iso)", fontsize=8)
            ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.suptitle(title or f"frame {t}: IVD + white iso-contour at tau "
                             f"(paper Fig.6 style, continuous reference)")
        fig.tight_layout()
        fig.savefig(out_dir / f"multi_tau_isocontour_t{t}.png", dpi=110)
        plt.close(fig)

    report = {"stats": stats, "out_dir": str(out_dir), "source_nc": str(source_nc)}
    if save_json:
        (out_dir / "multi_tau_stats.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


# --------------------------------------------------------------------------- 可视化（验收 1：目检）

def plot_ivd_q(ivd2d, q2d, label2d, mask2d, xdim, ydim, out_png, tau,
               title="", vmax=None):
    """展示帧目检图：IVD 场底图 + Q>0 等值线（青色对照）+ 标签≥τ 填充（红）+ 掩膜轮廓。

    坐标系统一物理坐标（imshow/contour 用 extent 对齐，仿 geometry.plot_mask）。
    返回输出路径 str。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ivd2d = np.asarray(ivd2d, dtype=np.float64)
    if vmax is None:
        vmax = float(np.percentile(ivd2d[~mask2d] if mask2d is not None else ivd2d, 99.0))
    extent = [xdim[0], xdim[-1], ydim[0], ydim[-1]]
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(ivd2d, origin="lower", aspect="auto", cmap="magma",
                   vmin=0, vmax=max(vmax, 1e-12), extent=extent)
    if q2d is not None:
        ax.contour(q2d, levels=[0.0], colors="cyan", linewidths=0.8,
                   extent=extent)
    if mask2d is not None:
        ax.contour(np.asarray(mask2d, dtype=bool), levels=[0.5], colors="red",
                   linewidths=1.0, extent=extent)
    if label2d is not None:
        lab = np.asarray(label2d).astype(bool)
        if lab.any():
            ax.contour(lab, levels=[0.5], colors="lime", linewidths=1.2,
                       extent=extent)
    ax.set_title(title or f"IVD + Q>0 (cyan) + label IVD>={tau:.4g} (lime)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return str(out_png)


def plot_tau_sensitivity(ivd2d, mask2d, xdim, ydim, out_png,
                         percentiles=(90.0, 95.0, 97.5, 99.0),
                         min_area=DEFAULT_MIN_AREA, title=""):
    """τ 敏感性对比图：同一展示帧在多个分位数候选下的标签场并排（验收 1 支撑）。

    每子图标题 = 分位数值 + 该分位的 τ + 正格占比；标签应用 5×5 面积过滤。
    返回输出路径 str。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ivd2d = np.asarray(ivd2d, dtype=np.float64)
    m2 = _mask2d(mask2d) if mask2d is not None else np.zeros(ivd2d.shape, dtype=bool)
    fluid_vals = ivd2d[~m2]
    extent = [xdim[0], xdim[-1], ydim[0], ydim[-1]]
    fig, axes = plt.subplots(1, len(percentiles), figsize=(4.2 * len(percentiles), 4.5))
    if len(percentiles) == 1:
        axes = [axes]
    for ax, p in zip(axes, percentiles):
        tau = float(np.percentile(fluid_vals, p))
        lab = _labeled_mask(ivd2d, tau, m2, min_area)
        ax.imshow(lab, origin="lower", aspect="auto", cmap="Greys",
                  vmin=0, vmax=1, extent=extent)
        ax.set_title(f"p{p:g}  tau={tau:.4g}\npos frac={lab.mean():.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(title or f"tau sensitivity (percentile -> labels, min_area={min_area})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return str(out_png)


# --------------------------------------------------------------------------- 主入口（CLI）

def _slice_of(frame, slices):
    """帧 → 所属时间片名（用于展示帧的 τ 标注；未覆盖返回 None）。"""
    for name, (i0, i1) in slices.items():
        if i0 <= frame < i1:
            return name
    return None


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="弱标签：IVD（ω=∂v/∂x−∂u/∂y，5×5 邻域偏差）+ 固体置零 + "
                    "逐时间片 τ 标签（95 分位）+ 面积过滤 + 正样本占比统计 + 目检图")
    ap.add_argument("nc_path", help="nc 数据集路径（h5py 直读，支持中文路径）")
    ap.add_argument("--mask", default=None,
                    help="固体掩膜 mask.npy 路径；缺省从 nc 流式计算（逐帧取与）")
    ap.add_argument("--out-dir", default="outputs/weak_labels",
                    help="输出目录（ivd.npy / label_field.npy / weak_label_meta.json / 图）")
    ap.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE,
                    help="τ 分位数（默认 85——票 07 延伸；HANDOFF §6）")
    ap.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA,
                    help="连通域过滤最小面积（默认 5×5=25 格）")
    ap.add_argument("--frames", default="400,1200,1300",
                    help="目检图展示帧（逗号分隔；默认覆盖 train/val/test 三时间片）")
    ap.add_argument("--no-visualize", action="store_true", help="不生成目检图")
    ap.add_argument("--multi-tau-dir", default=None,
                    help="额外生成多阈值敏感性报告到该目录（HANDOFF §7 预案；"
                         "p95/90/85/80 分位 + 固定阈值 × 面积过滤 统计与目检图）")
    args = ap.parse_args(argv)

    import h5py

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(args.nc_path), "r") as f:
        xdim = f["xdim"][:].astype(np.float64)
        ydim = f["ydim"][:].astype(np.float64)
        tdim = f["tdim"][:].astype(np.float64)
        T, Y, X = len(tdim), len(ydim), len(xdim)
        if args.mask:
            mask_arr = np.load(args.mask)
            mask2d = (mask_arr[0] if mask_arr.ndim == 3 else mask_arr).astype(bool)
        else:
            # 与 geometry.static_mask_from_speed 同判据（ε=1e-5，HANDOFF §2）；
            # 逐帧流式读取，避免全量 (T,Y,X) 驻留（推荐先跑票 02 CLI 落盘传入）。
            mask2d = np.ones((Y, X), dtype=bool)
            for t in range(T):
                mask2d &= np.hypot(f["u"][t], f["v"][t]) < 1e-5

        # 时间片截断到数据集帧数（子集数据集时）
        slices = {name: (i0, min(i1, T)) for name, (i0, i1) in DEFAULT_SLICES.items()
                  if i0 < T}

        # 1) IVD 场（逐帧流式，避免 u/v 全量驻留）
        print(f"计算 IVD 场 ({T}×{Y}×{X}, 5×5 邻域, 固体区置零) ...")
        ivd = np.zeros((T, Y, X), dtype=np.float32)
        for t in range(T):
            ivd[t] = compute_ivd(f["u"][t][None], f["v"][t][None], xdim, ydim,
                                 mask=mask2d).astype(np.float32)[0]
        np.save(out_dir / "ivd.npy", ivd)
        print(f"  IVD 场已保存: {out_dir / 'ivd.npy'} "
              f"(max={ivd.max():.5g}, 固体区零={bool((ivd[:, mask2d] == 0).all())})")

        # 2) 逐时间片 τ
        taus = compute_tau(ivd, mask2d, slices, percentile=args.percentile)
        print("τ 值（逐时间片, " + f"{args.percentile:g} 分位):")
        for name, tau in taus.items():
            print(f"  {name}: {tau:.6g}")

        # 3) 标签场
        lab = build_label_field(ivd, mask2d, taus, slices, min_area=args.min_area)
        np.save(out_dir / "label_field.npy", lab)
        print(f"  标签场已保存: {out_dir / 'label_field.npy'} "
              f"(正格占比={lab.mean():.4f})")

        # 4) 正样本占比（每时间片 + 全局；窗口起点步长 4 帧——HANDOFF §6 口径）
        fracs = {}
        for name, (i0, i1) in slices.items():
            fa = positive_patch_fraction(lab, xdim, ydim,
                                         frame_indices=np.arange(i0, i1, 4))
            fracs[name] = fa
            print(f"  正样本占比[{name}]: {fa['fraction']:.4f} "
                  f"({fa['n_positive']}/{fa['n_patches']})")
        frac_global = positive_patch_fraction(
            lab, xdim, ydim, frame_indices=np.arange(0, T, 4))
        fracs["global"] = frac_global
        print(f"  正样本占比[global]: {frac_global['fraction']:.4f} "
              f"({frac_global['n_positive']}/{frac_global['n_patches']})")

        meta = {
            "source_nc": str(args.nc_path),
            "shape": [T, Y, X],
            "mask_source": args.mask if args.mask else "computed_from_speed(eps=1e-5)",
            "percentile": float(args.percentile),
            "min_area": int(args.min_area),
            "taus": taus,
            "positive_fraction": fracs,
            "ivd_max": float(ivd.max()),
            "label_positive_fraction": float(lab.mean()),
            "oversample_factor_for_50pct": float(
                0.5 / frac_global["fraction"]) if frac_global["fraction"] > 0 else None,
        }
        (out_dir / "weak_label_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  元数据已保存: {out_dir / 'weak_label_meta.json'}")

        # 5) 目检图（展示帧：IVD + Q 对照 + 标签；τ 敏感性）
        if not args.no_visualize:
            for t0f in (int(s) for s in args.frames.split(",") if s):
                if not 0 <= t0f < T:
                    print(f"  跳过展示帧 {t0f}（超出 0..{T-1}）")
                    continue
                u_t = np.asarray(f["u"][t0f], dtype=np.float64)
                v_t = np.asarray(f["v"][t0f], dtype=np.float64)
                q_t = q_criterion(u_t[None], v_t[None], xdim, ydim)[0]
                tau_t = taus.get(_slice_of(t0f, slices), 0.0)
                plot_ivd_q(ivd[t0f], q_t, lab[t0f], mask2d, xdim, ydim,
                           out_dir / f"ivd_q_t{t0f}.png", tau=tau_t,
                           title=f"t={tdim[t0f]:.2f}s frame={t0f} "
                                 f"IVD + Q>0(cyan) + label>=τ(lime)")
                plot_tau_sensitivity(ivd[t0f], mask2d, xdim, ydim,
                                     out_dir / f"tau_sensitivity_t{t0f}.png",
                                     title=f"t={tdim[t0f]:.2f}s frame={t0f} "
                                           f"tau sensitivity")
                print(f"  目检图已保存: {out_dir / f'ivd_q_t{t0f}.png'} "
                      f"与 {out_dir / f'tau_sensitivity_t{t0f}.png'}")

        # 6) 多阈值敏感性报告（HANDOFF §7 预案；票 07 延伸——τ 下探对齐论文 Fig.6）
        if args.multi_tau_dir:
            print(f"  多阈值敏感性报告 → {args.multi_tau_dir}")
            rep = multi_tau_report(ivd, mask2d, slices, xdim, ydim,
                                   args.multi_tau_dir, source_nc=str(args.nc_path))
            for name, row in rep["stats"].items():
                frac = row.get("pos_cell_frac_ma25")
                comps = row.get("mean_n_components_ma25")
                pa = row.get("pos_patch_fraction_all")
                print(f"    {name}: pos_cell(ma25)={frac:.4f} "
                      f"comps={comps:.2f} pos_patch={pa['fraction']:.4f}")
    return 0


if __name__ == "__main__":
    main()
