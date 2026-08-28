"""迹线提取（extractor.py）——03 票：全局场迹线积分器。

领域词汇（HANDOFF §4/§6，唯一权威）：
- 迹线 = 从种子点按 RK4 + 三线性时空插值积分（每输出步 4 子步）生成的轨迹；
- 每样本 256 条 = 64 组 × 4 轴向卫星点（不含中心），Δ = patch 边长×0.05，组主序编组；
- 7 通道 = [px, py, t, ivd, distance(距种子点), u, v]（HANDOFF §2 模型事实）；
- 种子落固体 → 重播种（仿 C++ JittorReSeeding：朝 patch 中心随机移动）；
- 迹线入固体 → 截断并重复末点（不引入 -1000 毒值；C++ 参考用 -1000 填充，
  本项目决策明确不用）；
- 位置按 patch 归一化到 [-1,1]（可超界）；全局场积分允许迹线离开 patch
  （离开 patch 不停止，离开场域边界才停止）。

参考实现（只读）：PyflowVis-main `CppProjects/src/VectorFieldCompute.cpp`
（PathhlineIntegrationRK4v2 / PathlineIntegrationInfoCollect2D / JittorReSeeding）
与 `FLowUtils/flowlineIntegral.py`（越界 clamp 语义）。
C++ 生成器只支持解析场（离散场 assert）→ 本模块为离散场自写（HANDOFF §2）。

实现约束：h5py 直读中文路径（数据读取在 geometry.load_field）；纯 numpy/python
（遵守 §2 依赖清单：torch、numpy、h5py、yaml、matplotlib、tqdm）。
"""

from __future__ import annotations

import pathlib

import numpy as np

# --------------------------------------------------------------------------- 通道口径

# 7 通道 = [px, py, t, ivd, distance(距种子点), u, v]（HANDOFF §2 模型事实）
CH_PX, CH_PY, CH_T, CH_IVD, CH_DIST, CH_U, CH_V = range(7)
N_CHANNELS = 7


# --------------------------------------------------------------------------- 三线性时空插值

def _float_index(value, coord):
    """物理坐标 → 浮点格索引（未 clamp）；coord 为等距坐标轴。"""
    return (float(value) - coord[0]) / (coord[1] - coord[0])


def trilinear_interp(field, x, y, t, xdim, ydim, tdim):
    """三线性时空插值：空间双线性 + 时间线性（参考 C++ trilinear_interpolate）。

    越界 clamp 到边界格（参考 flowlineIntegral.py 语义）：RK4 中间阶段点可能
    探出场域边界，clamp 保证插值始终有限（无 NaN）。
    """
    field = np.asarray(field)
    T, Y, X = field.shape
    gx = _float_index(x, xdim)
    gy = _float_index(y, ydim)
    gt = _float_index(t, tdim)
    x0 = int(np.floor(gx)); x1 = int(np.ceil(gx))
    y0 = int(np.floor(gy)); y1 = int(np.ceil(gy))
    t0 = int(np.floor(gt)); t1 = int(np.ceil(gt))
    x0 = min(max(x0, 0), X - 1); x1 = min(max(x1, 0), X - 1)
    y0 = min(max(y0, 0), Y - 1); y1 = min(max(y1, 0), Y - 1)
    t0 = min(max(t0, 0), T - 1); t1 = min(max(t1, 0), T - 1)
    wx = gx - x0; wy = gy - y0; wt = gt - t0
    # 帧 t0 与 t1 的空间双线性
    def _bilinear(fr):
        c00 = field[fr, y0, x0]
        c10 = field[fr, y0, x1]
        c01 = field[fr, y1, x0]
        c11 = field[fr, y1, x1]
        top = c00 * (1.0 - wx) + c10 * wx
        bot = c01 * (1.0 - wx) + c11 * wx
        return top * (1.0 - wy) + bot * wy
    a = _bilinear(t0)
    b = _bilinear(t1)
    return a * (1.0 - wt) + b * wt


def velocity_at(u, v, x, y, t, xdim, ydim, tdim):
    """速度场三线性时空插值 → (ux, uy)。"""
    return (trilinear_interp(u, x, y, t, xdim, ydim, tdim),
            trilinear_interp(v, x, y, t, xdim, ydim, tdim))


# --------------------------------------------------------------------------- 掩膜查询

def nearest_cell(xs, ys, xdim, ydim):
    """物理坐标 → 最近格 (j, i)（四舍五入 floor(g+0.5)；等距坐标）。

    单一公式：mask_at（种子判固体）、weak_labels.patch_seed_offsets（正样本
    判据）、dataset 的标签/种子判定共用——防"最近格"口径多份漂移。
    标量或数组输入均可；不做越界 clip（调用方按语义处理：
    mask_at 越界 → 非固体；dataset 判定 clip 到边界格）。
    """
    gx = (np.asarray(xs, dtype=np.float64) - xdim[0]) / (xdim[1] - xdim[0])
    gy = (np.asarray(ys, dtype=np.float64) - ydim[0]) / (ydim[1] - ydim[0])
    return np.floor(gy + 0.5).astype(np.intp), np.floor(gx + 0.5).astype(np.intp)


def mask_at(mask, x, y, xdim, ydim):
    """物理坐标 → 最近格（四舍五入）的掩膜值；越界视为 False（域外由积分器停止）。

    掩膜为 2D (Y,X) bool（固体几何时间不变，HANDOFF §4）；接受 (T,Y,X) 时取第 0 帧。
    """
    if mask is None:
        return False
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[0]
    Y, X = mask.shape
    j, i = nearest_cell(x, y, xdim, ydim)
    if i < 0 or i >= X or j < 0 or j >= Y:
        return False
    return bool(mask[j, i])


# --------------------------------------------------------------------------- RK4 积分器

# 积分终止状态（HANDOFF 词汇：出域 / 入固体 / 完成）
STATUS_COMPLETE = "complete"
STATUS_OUT_OF_DOMAIN = "out_of_domain"
STATUS_HIT_SOLID = "hit_solid"


def integrate_pathline(u, v, mask, seed, t0, dt_out, L, xdim, ydim, tdim,
                       n_substeps=4):
    """全局场 RK4 积分一条迹线（参考 C++ PathhlineIntegrationRK4v2）。

    每输出步 n_substeps 个子步（HANDOFF §6：RK4 子步 = 每输出步 4），
    速度用三线性时空插值；输出步新点：
      - 出域（空间格边缘外 / 时间超出 tdim 范围）→ 停止（不采纳该步）；
      - 入固体（掩膜最近格）→ 停止（不采纳该步，截断）；
      - 否则采纳，直到 L 步（complete）。

    返回 (pos (n,2), times (n,), status)：n ≤ L 为实际输出步数；
    补齐（重复末点）由调用方处理（不引入 -1000 毒值）。
    """
    u = np.asarray(u)
    v = np.asarray(v)
    xdim = np.asarray(xdim, dtype=np.float64)
    ydim = np.asarray(ydim, dtype=np.float64)
    tdim = np.asarray(tdim, dtype=np.float64)
    dx = xdim[1] - xdim[0]
    dy = ydim[1] - ydim[0]
    x_min, x_max = xdim[0] - dx / 2.0, xdim[-1] + dx / 2.0
    y_min, y_max = ydim[0] - dy / 2.0, ydim[-1] + dy / 2.0
    t_min, t_max = tdim[0], tdim[-1]

    h = float(dt_out) / n_substeps
    px, py, t = float(seed[0]), float(seed[1]), float(t0)
    pos = [(px, py)]
    times = [t]

    def _out_of_domain(x, y, t):
        return (x <= x_min or x >= x_max or y <= y_min or y >= y_max
                or t < t_min or t > t_max)

    for _ in range(L - 1):
        # 每输出步：n_substeps 个 RK4 子步（阶段点插值 clamp 保证有限）
        for _ in range(n_substeps):
            k1x, k1y = velocity_at(u, v, px, py, t, xdim, ydim, tdim)
            k2x, k2y = velocity_at(u, v, px + 0.5 * h * k1x, py + 0.5 * h * k1y,
                                   t + 0.5 * h, xdim, ydim, tdim)
            k3x, k3y = velocity_at(u, v, px + 0.5 * h * k2x, py + 0.5 * h * k2y,
                                   t + 0.5 * h, xdim, ydim, tdim)
            k4x, k4y = velocity_at(u, v, px + h * k3x, py + h * k3y,
                                   t + h, xdim, ydim, tdim)
            px += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            py += (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
            t += h
        # 输出步新点检查
        if _out_of_domain(px, py, t):
            status = STATUS_OUT_OF_DOMAIN
            break
        if mask_at(mask, px, py, xdim, ydim):
            status = STATUS_HIT_SOLID
            break
        pos.append((px, py))
        times.append(t)
    else:
        status = STATUS_COMPLETE
    return np.asarray(pos, dtype=np.float64), np.asarray(times), status


def pad_repeat_last(feat, L):
    """把 (n, C) 特征行补齐到 (L, C)：重复最后一行（截断轨迹的末点重复）。

    票 03 语义：迹线入固体 → 截断并重复末点，**不引入 -1000 毒值**
    （C++ 参考 PathlineIntegrationInfoCollect2D 用 -1000 填充，本项目决策不用）。
    n ≥ 1；n == L 时原样返回。
    """
    feat = np.asarray(feat)
    n = feat.shape[0]
    if n >= L:
        return feat[:L]
    pad = np.repeat(feat[-1:], L - n, axis=0)
    return np.concatenate([feat, pad], axis=0)


# --------------------------------------------------------------------------- 重播种

def reseed(seed, mask, center, xdim, ydim, rng=None, max_attempts=50):
    """种子落固体 → 重播种（仿 C++ GroupSeeding::JittorReSeeding）。

    语义（C++ 参考）：seed_new = seed + shift × (center − seed)，
    shift ~ U[0.00001, 0.5]（均匀随机）；重试直到不在固体。
    工程化（对 C++ 的 while 真循环加显式上限）：max_attempts 次仍失败 →
    沿 center−seed 方向线性细扫找第一个流体点；仍失败（patch 全固体或
    零方向）→ ValueError（上层采样应避开全固体 patch）。
    种子不在固体 → 原样返回（无扰动）。
    """
    if not mask_at(mask, seed[0], seed[1], xdim, ydim):
        return np.asarray(seed, dtype=np.float64)
    rng = rng if rng is not None else np.random.default_rng()
    seed = np.asarray(seed, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    direction = center - seed
    if np.hypot(direction[0], direction[1]) <= 0.0:
        raise ValueError("种子与 patch 中心重合且中心在固体中，无法重播种")
    for _ in range(max_attempts):
        shift = rng.uniform(0.00001, 0.5)
        cand = seed + shift * direction
        if not mask_at(mask, cand[0], cand[1], xdim, ydim):
            return cand
    # 细扫：沿方向找第一个流体点（覆盖随机未命中的细长/稀疏情形）
    for s in np.linspace(0.0, 1.0, 201)[1:]:
        cand = seed + s * direction
        if not mask_at(mask, cand[0], cand[1], xdim, ydim):
            return cand
    raise ValueError("patch 全固体，无法重播种")


# --------------------------------------------------------------------------- 样本组装

def patch_geometry(patch_yx, patch_size, xdim, ydim):
    """patch 物理几何（extract 与可视化共用口径，避免重复计算漂移）。

    返回 dict：cx/cy 归一化原点（patch 中心）、hx/hy 归一化半宽、
    span_x/span_y patch 物理范围（格边缘起止）、dx/dy 格距。
    """
    y0, x0 = patch_yx
    ph, pw = patch_size
    dx = xdim[1] - xdim[0]
    dy = ydim[1] - ydim[0]
    x_lo = xdim[x0] - dx / 2.0
    x_hi = xdim[x0 + pw - 1] + dx / 2.0
    y_lo = ydim[y0] - dy / 2.0
    y_hi = ydim[y0 + ph - 1] + dy / 2.0
    return {
        "cx": 0.5 * (x_lo + x_hi),
        "cy": 0.5 * (y_lo + y_hi),
        "hx": 0.5 * (x_hi - x_lo),
        "hy": 0.5 * (y_hi - y_lo),
        "x_lo": x_lo,
        "y_lo": y_lo,
        "span_x": x_hi - x_lo,
        "span_y": y_hi - y_lo,
        "dx": dx,
        "dy": dy,
    }


def interp_path(field, pos, times, xdim, ydim, tdim):
    """对一条迹线的所有点批量三线性插值 → (n,) 向量（特征采样用）。

    与标量 trilinear_interp 同语义（双线性 4 角加权 + 时间线性 + 越界 clamp），
    但一次处理整条迹线（numpy 向量化，避免标量 Python 循环）。
    """
    field = np.asarray(field)
    T, Y, X = field.shape
    gx = (pos[:, 0] - xdim[0]) / (xdim[1] - xdim[0])
    gy = (pos[:, 1] - ydim[0]) / (ydim[1] - ydim[0])
    gt = (times - tdim[0]) / (tdim[1] - tdim[0])
    x0 = np.clip(np.floor(gx).astype(np.int64), 0, X - 1)
    x1 = np.clip(np.ceil(gx).astype(np.int64), 0, X - 1)
    y0 = np.clip(np.floor(gy).astype(np.int64), 0, Y - 1)
    y1 = np.clip(np.ceil(gy).astype(np.int64), 0, Y - 1)
    t0 = np.clip(np.floor(gt).astype(np.int64), 0, T - 1)
    t1 = np.clip(np.ceil(gt).astype(np.int64), 0, T - 1)
    wx = gx - x0
    wy = gy - y0
    wt = gt - t0
    f = field[t0, y0, x0], field[t0, y0, x1], field[t0, y1, x0], field[t0, y1, x1]
    g = field[t1, y0, x0], field[t1, y0, x1], field[t1, y1, x0], field[t1, y1, x1]
    a = (f[0] * (1 - wx) * (1 - wy) + f[1] * wx * (1 - wy)
         + f[2] * (1 - wx) * wy + f[3] * wx * wy)
    b = (g[0] * (1 - wx) * (1 - wy) + g[1] * wx * (1 - wy)
         + g[2] * (1 - wx) * wy + g[3] * wx * wy)
    return (a * (1 - wt) + b * wt).ravel()


def seeding_grid(patch_yx, patch_size, xdim, ydim, groups=(8, 8), delta_frac=0.05):
    """256 条迹线的种子物理坐标 (K,2)——组主序，与 extract_pathlines 共用同一公式。

    口径（HANDOFF §1 决策 6 / §6 / 票 03）：
    - K = 64 组 × 4 轴向卫星点（不含中心）= 256 条，组主序编组（组 0 的 4 条在前）；
    - 组中心 = patch 内 [0.1,0.9] 区间等距网格（仿 C++ GridCrossSampling 边距）；
    - Δ = patch 边长 × delta_frac（x/y 分别按格距计算）。
    单一事实来源：extract_pathlines 与 weak_labels（正样本统计）均从此取种子，
    防止公式双份漂移。
    """
    geo = patch_geometry(patch_yx, patch_size, xdim, ydim)
    gy, gx = groups
    span_x, span_y = geo["span_x"], geo["span_y"]
    x_lo, y_lo = geo["x_lo"], geo["y_lo"]
    delta_x = span_x * delta_frac
    delta_y = span_y * delta_frac
    xc = x_lo + 0.1 * span_x + np.arange(gx) * (0.8 * span_x) / (gx - 1)
    yc = y_lo + 0.1 * span_y + np.arange(gy) * (0.8 * span_y) / (gy - 1)
    seeds = np.empty((gy * gx * 4, 2), dtype=np.float64)
    k = 0
    for i in range(gy):                      # y 外层（仿 C++ 行主序）
        for j in range(gx):
            for ox, oy in ((-delta_x, 0.0), (0.0, -delta_y), (delta_x, 0.0), (0.0, delta_y)):
                seeds[k] = (xc[j] + ox, yc[i] + oy)
                k += 1
    return seeds


def extract_pathlines(u, v, mask, ivd, xdim, ydim, tdim,
                      patch_yx, patch_size=(32, 32), t0=0.0, L=16,
                      groups=(8, 8), delta_frac=0.05, t_win_frames=24,
                      n_substeps=4, dt_out=None, rng=None,
                      return_seeds=False, max_integration_attempts=3):
    """生成一个样本的迹线张量 → (L, K=256, 7) float32（数据准备缝）。

    口径（HANDOFF §1 决策 6 / §6 参数表 / 票 03）：
    - K = 64 组 × 4 轴向卫星点（不含中心）= 256 条，组主序编组（组 0 的 4 条在前）；
    - 组中心 = patch 内 [0.1, 0.9] 区间 8×8 等距网格（仿 C++ GridCrossSampling 边距）；
    - Δ = patch 边长 × delta_frac（x/y 分别按格距计算）；
    - 种子落固体 → reseed（仿 C++ JittorReSeeding）；迹线入固体 → 截断重复末点；
    - 积分太短（≤2 点，仿 C++ suc 判据 pathPositions.size() > 2）→ 朝 patch 中心
      大幅移动后重试（max_integration_attempts 次，避免固体边缘退化迹线）；
    - 位置按 patch 归一化到 [-1,1]（可超界）；全局场积分允许离开 patch；
    - 7 通道 = [px, py, t, ivd, distance(距种子), u, v]；ivd=None 时第 4 通道为 0
      （票 04 提供真实 IVD 场后由票 05 接入）；
    - 默认 dt_out = (t_win_frames−1)×dt/(L−1)（窗口 24 帧覆盖 L 个输出步）。

    参数：
      u, v     (T,Y,X) 速度场；mask  (Y,X) 或 (T,Y,X) 固体掩膜（None = 无障碍物）；
      ivd      (T,Y,X) IVD 场或 None；xdim/ydim/tdim 物理坐标；
      patch_yx patch 起点格索引 (y0, x0)；patch_size 格数 (ph, pw)；
      t0       窗口起点物理时间；L 输出步数；groups 组网格 (gy, gx)；
      rng      随机源（重播种），可注入保证确定性；
      return_seeds   True 时返回 (out, seeds)，seeds 为重播种后的 (256,2) 种子
                     （供测试验证重播种生效）；
      max_integration_attempts  积分太短时的重试上限（含首次）。
    """
    u = np.asarray(u)
    v = np.asarray(v)
    xdim = np.asarray(xdim, dtype=np.float64)
    ydim = np.asarray(ydim, dtype=np.float64)
    tdim = np.asarray(tdim, dtype=np.float64)
    gy, gx = groups
    dt = tdim[1] - tdim[0]
    if dt_out is None:
        dt_out = (t_win_frames - 1) * dt / (L - 1)
    geo = patch_geometry(patch_yx, patch_size, xdim, ydim)
    cx, cy = geo["cx"], geo["cy"]
    hx, hy = geo["hx"], geo["hy"]
    # 种子图案：组中心 [0.1,0.9] 等距网格 × 4 轴向卫星（共用 seeding_grid 单一公式）
    seeds_grid = seeding_grid(patch_yx, patch_size, xdim, ydim, groups, delta_frac)
    rng = rng if rng is not None else np.random.default_rng()
    center = np.array([cx, cy])

    K = gy * gx * 4
    out = np.zeros((L, K, N_CHANNELS), dtype=np.float32)
    seeds = np.zeros((K, 2), dtype=np.float64)
    for k in range(K):                       # 组主序（与 seeding_grid 编组一致）
        seed = seeds_grid[k].copy()
        for attempt in range(max_integration_attempts):
            seed = reseed(seed, mask, center, xdim, ydim, rng=rng)
            pos, times, _status = integrate_pathline(
                u, v, mask, seed, t0, dt_out, L, xdim, ydim, tdim,
                n_substeps=n_substeps)
            if len(pos) >= 3 or attempt == max_integration_attempts - 1:
                break
            # 积分太短（仿 C++ suc 判据）→ 朝 patch 中心大幅移动后重试
            seed = seed + 0.5 * (center - seed)
        seeds[k] = seed
        n = len(pos)
        rows = np.zeros((n, N_CHANNELS), dtype=np.float64)
        rows[:, CH_PX] = (pos[:, 0] - cx) / hx
        rows[:, CH_PY] = (pos[:, 1] - cy) / hy
        rows[:, CH_T] = times
        if ivd is not None:
            rows[:, CH_IVD] = interp_path(ivd, pos, times, xdim, ydim, tdim)
        rows[:, CH_DIST] = np.hypot(pos[:, 0] - seed[0], pos[:, 1] - seed[1])
        rows[:, CH_U] = interp_path(u, pos, times, xdim, ydim, tdim)
        rows[:, CH_V] = interp_path(v, pos, times, xdim, ydim, tdim)
        out[:, k, :] = pad_repeat_last(rows, L)
    if return_seeds:
        return out, seeds
    return out


# --------------------------------------------------------------------------- 向量化批量提取（票 05）


def _interp_pair(u, v, xs, ys, ts, xdim, ydim, tdim):
    """u/v 双场共享权重的向量化三线性时空插值（与 trilinear_interp 同语义）。

    一次计算 (gx, gy, 角索引, 权重)，对 u 与 v 各做一次 8 角 gather——
    批量积分器每子步只承担一次索引计算。时间分量标量化：批量积分中所有
    迹线同步推进（同一子步时刻相同），时间帧索引为标量。
    输入 xs/ys/ts 为 (N,) 数组（ts 全同值），返回 (u_interp, v_interp) 各 (N,)。
    与标量 trilinear_interp 的双份公式由守护测试保证一致。
    """
    u = np.asarray(u)
    v = np.asarray(v)
    T, Y, X = u.shape
    ts = np.asarray(ts)
    if ts.size and not np.all(ts == ts.reshape(-1)[0]):
        raise ValueError("_interp_pair 仅支持同步时间（批量积分中所有迹线同一时刻）")
    gx = (xs - xdim[0]) / (xdim[1] - xdim[0])
    gy = (ys - ydim[0]) / (ydim[1] - ydim[0])
    gt = float(ts.reshape(-1)[0] - tdim[0]) / (tdim[1] - tdim[0])
    t0_ = min(max(int(np.floor(gt)), 0), T - 1)
    t1_ = min(int(np.ceil(gt)), T - 1)
    wt = gt - t0_
    x0 = np.clip(np.floor(gx).astype(np.int64), 0, X - 1)
    x1 = np.clip(np.ceil(gx).astype(np.int64), 0, X - 1)
    y0 = np.clip(np.floor(gy).astype(np.int64), 0, Y - 1)
    y1 = np.clip(np.ceil(gy).astype(np.int64), 0, Y - 1)
    wx = gx - x0
    wy = gy - y0
    yx = Y * X
    # 8 角索引以增量偏移构造（x/y 角差为向量、帧差为标量：t1_==t0_ 时同帧，
    # 边界格 x1==x0/y1==y0 时角差值 0 → 无越界，与标量 clamp 语义逐角一致）
    dxo = (x1 - x0).astype(np.int64)
    dyo = (y1 - y0) * X
    fo = 0 if t1_ == t0_ else yx
    base = t0_ * yx + y0 * X + x0
    offs = np.stack([dxo * 0, dxo, dyo, dxo + dyo,
                     dxo * 0 + fo, fo + dxo, fo + dyo, fo + dxo + dyo], axis=1)
    idx = (base[:, None] + offs).reshape(-1)

    # 权重 (N, 8)：4 空间角 × 2 时间帧（先在 _gather 使用前构造，避免延迟绑定歧义）
    w = np.empty((len(wx), 8), dtype=np.float64)
    a0 = (1 - wx) * (1 - wy)
    a1 = wx * (1 - wy)
    a2 = (1 - wx) * wy
    a3 = wx * wy
    w[:, 0] = a0 * (1 - wt)
    w[:, 1] = a1 * (1 - wt)
    w[:, 2] = a2 * (1 - wt)
    w[:, 3] = a3 * (1 - wt)
    w[:, 4] = a0 * wt
    w[:, 5] = a1 * wt
    w[:, 6] = a2 * wt
    w[:, 7] = a3 * wt

    def _gather(field):
        vals = np.asarray(field).reshape(-1)[idx].reshape(len(wx), 8)
        return vals * w

    return _gather(u).sum(axis=1), _gather(v).sum(axis=1)


def _integrate_batched(u, v, mask, seeds, t0, dt_out, L, xdim, ydim, tdim,
                       n_substeps=4):
    """向量化批量 RK4 积分：K 条迹线同步推进（与 integrate_pathline 同语义）。

    冻结语义 == 标量"截断并重复末点"：失效迹线（出域/入固体）不再更新，
    其后位置/时间保持上一有效采纳点（pad_repeat_last 的批量等价）。
    返回 (pos (K,L,2) float64, times (K,L), n (K,))：n = 有效输出步数（含种子点，
    与标量 pos 长度一致；完整 = L）。
    """
    u = np.asarray(u)
    v = np.asarray(v)
    xdim = np.asarray(xdim, dtype=np.float64)
    ydim = np.asarray(ydim, dtype=np.float64)
    tdim = np.asarray(tdim, dtype=np.float64)
    K = seeds.shape[0]
    dx = xdim[1] - xdim[0]
    dy = ydim[1] - ydim[0]
    x_min, x_max = xdim[0] - dx / 2.0, xdim[-1] + dx / 2.0
    y_min, y_max = ydim[0] - dy / 2.0, ydim[-1] + dy / 2.0
    t_min, t_max = tdim[0], tdim[-1]
    h = float(dt_out) / n_substeps
    m2 = None
    if mask is not None:
        m2 = np.asarray(mask, dtype=bool)
        if m2.ndim == 3:
            m2 = m2[0]
    Y, X = m2.shape if m2 is not None else (len(ydim), len(xdim))

    pos = np.empty((K, L, 2), dtype=np.float64)
    pos[:, 0] = seeds
    times = np.empty((K, L), dtype=np.float64)
    times[:, 0] = float(t0)
    active = np.ones(K, dtype=bool)
    n = np.ones(K, dtype=np.intp)
    for step in range(1, L):
        # 冻结行：上一有效点（失效迹线从本行起保持重复末点——截断语义）
        pos[:, step] = pos[:, step - 1]
        times[:, step] = times[:, step - 1]
        if not active.any():
            continue
        idx = np.nonzero(active)[0]
        px = pos[idx, step - 1, 0].copy()
        py = pos[idx, step - 1, 1].copy()
        t = times[idx, step - 1].copy()
        for _ in range(n_substeps):
            k1x, k1y = _interp_pair(u, v, px, py, t, xdim, ydim, tdim)
            k2x, k2y = _interp_pair(u, v, px + 0.5 * h * k1x, py + 0.5 * h * k1y,
                                    t + 0.5 * h, xdim, ydim, tdim)
            k3x, k3y = _interp_pair(u, v, px + 0.5 * h * k2x, py + 0.5 * h * k2y,
                                    t + 0.5 * h, xdim, ydim, tdim)
            k4x, k4y = _interp_pair(u, v, px + h * k3x, py + h * k3y,
                                    t + h, xdim, ydim, tdim)
            px += (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            py += (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
            t += h
        # 输出步检查（与标量同序：出域 → 入固）
        out_ok = ~((px <= x_min) | (px >= x_max) | (py <= y_min) | (py >= y_max)
                   | (t < t_min) | (t > t_max))
        in_solid = np.zeros(len(idx), dtype=bool)
        if m2 is not None and out_ok.any():
            gx = (px[out_ok] - xdim[0]) / dx
            gy = (py[out_ok] - ydim[0]) / dy
            i = np.floor(gx + 0.5).astype(np.intp)
            j = np.floor(gy + 0.5).astype(np.intp)
            # 半格容差内的位置一律按**裁剪索引**取边界格掩膜（out_ok 已排除真出域）：
            # 网格非整跨度（linspace/生成器舍入使 (ydim[-1]-ydim[0])/dy > Y-1，
            # 票 07 延伸四期实测 jung 199.0000628）时上缘半格内 floor 取整可为 Y →
            # 未裁剪 m2[j,i] 越界崩溃（标量 mask_at 对越界返回 False 不崩，批式补守卫）；
            # 边界格为固体时批式在此冻结——比标量（越界视为流体）更物理。
            in_solid[out_ok] = m2[np.clip(j, 0, Y - 1), np.clip(i, 0, X - 1)]
        dead = ~out_ok | in_solid
        if dead.any():
            active[idx[dead]] = False            # 失效者不采纳该步（行已冻结为 step-1）
        alive = idx[~dead]
        if alive.size:
            pos[alive, step] = np.stack([px[~dead], py[~dead]], axis=1)
            times[alive, step] = t[~dead]
            n[alive] += 1
    return pos, times, n


def extract_pathlines_batched(u, v, mask, ivd, xdim, ydim, tdim,
                              patch_yx, patch_size=(32, 32), t0=0.0, L=16,
                              groups=(8, 8), delta_frac=0.05, t_win_frames=24,
                              n_substeps=4, dt_out=None, rng=None,
                              max_integration_attempts=3, return_seeds=False):
    """向量化批量迹线提取（票 05；供 dataset 的 on-the-fly 样本生成）
    → (L, K, 7) float32（与 extract_pathlines 同口径同公式）。

    - 种子图案 / 重播种 / 截断 / 短迹线重试 / 通道语义与 extract_pathlines
      完全一致；积分批量化（K 条同步 RK4），每子步共享一次索引计算。
    - **rng 语义（与逐条版不同构）**：重播种/重试的随机源按
      SeedSequence([base_seed, k, attempt]) 逐迹线派生 → 样本级可复现
      （同 base_seed 同输入 → 同输出），无单流相位问题；
      base_seed = int(rng)（Generator 取其一个 32 位整数；None → 系统熵）。
    - 一致性守护：无随机消费路径（种子不落固体、不触发短迹线重试）时
      与逐条版逐元素一致（公式同源）；含随机路径时两者是同一个重播种
      语义（JittorReSeeding）的不同随机实现，不保证逐元素一致。
    - return_seeds=True 返回 (out, seeds)（seeds = 每条迹线的最终种子，
      重播种/重试后的实际出发位置，供标签判定与可视化）。
    """
    u = np.asarray(u)
    v = np.asarray(v)
    xdim = np.asarray(xdim, dtype=np.float64)
    ydim = np.asarray(ydim, dtype=np.float64)
    tdim = np.asarray(tdim, dtype=np.float64)
    gy, gx = groups
    dt = tdim[1] - tdim[0]
    if dt_out is None:
        dt_out = (t_win_frames - 1) * dt / (L - 1)
    geo = patch_geometry(patch_yx, patch_size, xdim, ydim)
    cx, cy, hx, hy = geo["cx"], geo["cy"], geo["hx"], geo["hy"]
    if isinstance(rng, (int, np.integer)):
        rng_base = int(rng)
    elif isinstance(rng, np.random.Generator):
        rng_base = int(rng.integers(0, 2 ** 31))
    else:
        rng_base = int(np.random.default_rng().integers(0, 2 ** 31))
    center = np.array([cx, cy])
    K = gy * gx * 4
    out = np.zeros((L, K, N_CHANNELS), dtype=np.float32)

    # 阶段 1：逐 k 首次重播种（per-k 确定性派生，样本级可复现）。
    # 惰性 rng：仅对落固体的种子构造（全流体 patch 零 rng 开销；确定性保持一致：
    # 每 k 的派生只依赖 (rng_base, k)，与消费与否无关）。
    seeds = np.array(seeding_grid(patch_yx, patch_size, xdim, ydim,
                                  groups, delta_frac), dtype=np.float64)
    mask2d = None
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        mask2d = m[0] if m.ndim == 3 else m
    if mask2d is not None:
        j, i = nearest_cell(seeds[:, 0], seeds[:, 1], xdim, ydim)
        j = np.clip(j, 0, mask2d.shape[0] - 1)
        i = np.clip(i, 0, mask2d.shape[1] - 1)
        solid_k = np.nonzero(mask2d[j, i])[0]
    else:
        solid_k = np.empty(0, dtype=np.intp)
    for k in solid_k:
        rng_k = np.random.default_rng(np.random.SeedSequence([rng_base, k]))
        seeds[k] = reseed(seeds[k], mask, center, xdim, ydim, rng=rng_k)

    # 阶段 2：批量积分（失效迹线冻结为重复末点）
    pos, times, n = _integrate_batched(
        u, v, mask, seeds, float(t0), dt_out, L, xdim, ydim, tdim,
        n_substeps=n_substeps)

    # 阶段 3：短迹线（≤2 点）按 k 序重试（仿逐条版 suc 判据；低频兜底，
    # 逐条标量函数复用 → 公式同源）
    for k in np.nonzero(n <= 2)[0]:
        seed = seeds[k]
        pos_k = times_k = None
        for attempt in range(max_integration_attempts):
            rng_k = np.random.default_rng(
                np.random.SeedSequence([rng_base, k, attempt]))
            seed = reseed(seed, mask, center, xdim, ydim, rng=rng_k)
            pos_k, times_k, _status = integrate_pathline(
                u, v, mask, seed, float(t0), dt_out, L, xdim, ydim, tdim,
                n_substeps=n_substeps)
            if len(pos_k) >= 3 or attempt == max_integration_attempts - 1:
                break
            seed = seed + 0.5 * (center - seed)     # 朝 patch 中心大幅移动后重试
        seeds[k] = seed
        n[k] = len(pos_k)
        pos[k] = pad_repeat_last(pos_k, L)
        times[k] = pad_repeat_last(times_k[:, None], L)[:, 0]

    # 通道组装（与逐条版同公式）：位置已归一化；t/ivd/dist/u/v 原始值
    out[:, :, CH_PX] = ((pos[:, :, 0] - cx) / hx).T
    out[:, :, CH_PY] = ((pos[:, :, 1] - cy) / hy).T
    out[:, :, CH_T] = times.T
    if ivd is not None:
        out[:, :, CH_IVD] = interp_path(ivd, pos.reshape(-1, 2),
                                        times.ravel(), xdim, ydim,
                                        tdim).reshape(K, L).T
    out[:, :, CH_DIST] = np.hypot(pos[:, :, 0] - seeds[:, None, 0],
                                  pos[:, :, 1] - seeds[:, None, 1]).T
    out[:, :, CH_U] = interp_path(u, pos.reshape(-1, 2), times.ravel(),
                                  xdim, ydim, tdim).reshape(K, L).T
    out[:, :, CH_V] = interp_path(v, pos.reshape(-1, 2), times.ravel(),
                                  xdim, ydim, tdim).reshape(K, L).T
    if return_seeds:
        return out, seeds
    return out


# --------------------------------------------------------------------------- 可视化（验收 1：目检）

def plot_pathlines(u_t, mask2d, pathlines_phys, seeds_phys, xdim, ydim,
                   out_png, title="", vmax=1.5):
    """目检图：速度模底图 + 掩膜轮廓 + 256 条迹线（每组 4 条同色）+ 种子点。

    pathlines_phys: (L, K, 2) 物理坐标（extract 输出为 patch 归一化坐标，
    需用 patch_geometry 的 cx/hx 反算回物理坐标）；seeds_phys: (K, 2)。
    坐标系统一物理坐标（仿 geometry.plot_mask 的 extent 对齐）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = np.hypot(u_t[0], u_t[1])
    extent = [xdim[0], xdim[-1], ydim[0], ydim[-1]]
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(sp, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=vmax, extent=extent)
    if mask2d is not None:
        ax.contour(mask2d, levels=[0.5], colors="red", linewidths=1.0,
                   extent=extent)
    cmap = plt.get_cmap("tab20")
    L, K, _ = pathlines_phys.shape
    for k in range(K):
        c = cmap((k // 4) % 20)            # 组主序：每组 4 条同色
        ax.plot(pathlines_phys[:, k, 0], pathlines_phys[:, k, 1],
                color=c, lw=0.8, alpha=0.8)
    ax.scatter(seeds_phys[:, 0], seeds_phys[:, 1], s=6, c="black", marker="+")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return str(out_png)


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="迹线提取：全局场 RK4 积分（每输出步 4 子步）→ "
                    "(L, 256, 7) 样本 + 目检图（跟随流场、不穿固体）")
    ap.add_argument("nc_path", help="nc 数据集路径（h5py 直读，支持中文路径）")
    ap.add_argument("--out-dir", default="outputs/pathlines",
                    help="输出目录（pathlines_t*.npy + 目检图）")
    ap.add_argument("--mask", default=None,
                    help="固体掩膜 mask.npy 路径；缺省从 nc 全量计算（逐帧取与）")
    ap.add_argument("--patch-yx", default="100,280",
                    help="patch 起点格索引 y0,x0（默认拐角圆柱下游涡街区）")
    ap.add_argument("--patch-size", default="32,32", help="patch 格数 ph,pw")
    ap.add_argument("--frames", default="400,800,1200",
                    help="窗口起点帧（逗号分隔；每窗口 T_win=24 帧）")
    ap.add_argument("--t-win", type=int, default=24, help="窗口帧数")
    ap.add_argument("--visualize", action="store_true", help="生成目检图")
    args = ap.parse_args(argv)

    import h5py

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_yx = tuple(int(s) for s in args.patch_yx.split(","))
    patch_size = tuple(int(s) for s in args.patch_size.split(","))
    frames = [int(s) for s in args.frames.split(",")]

    with h5py.File(str(args.nc_path), "r") as f:
        xdim = f["xdim"][:].astype(np.float64)
        ydim = f["ydim"][:].astype(np.float64)
        tdim = f["tdim"][:].astype(np.float64)
        if args.mask:
            mask_arr = np.load(args.mask)
            # 兼容 2D (Y,X) 与 3D (T,Y,X)（固体几何时间不变，取第 0 帧）
            mask2d = (mask_arr[0] if mask_arr.ndim == 3 else mask_arr).astype(bool)
        else:
            # 与 geometry.static_mask_from_speed 同判据（ε=1e-5，HANDOFF §2）；
            # 此处逐帧流式读取，避免全量 (T,Y,X) 驻留内存；可先跑票 02 CLI
            # 落盘 mask.npy 后经 --mask 传入（推荐路径）。
            mask2d = np.ones((len(ydim), len(xdim)), dtype=bool)
            for t in range(len(tdim)):
                mask2d &= np.hypot(f["u"][t], f["v"][t]) < 1e-5
        geo = patch_geometry(patch_yx, patch_size, xdim, ydim)
        py, px = patch_yx
        for t0f in frames:
            u_win = np.asarray(f["u"][t0f:t0f + args.t_win], dtype=np.float32)
            v_win = np.asarray(f["v"][t0f:t0f + args.t_win], dtype=np.float32)
            # 窗口切片场必须配窗口 tdim（时间索引相对窗口起点；传全场 tdim 会把
            # 时间映射 clamp 到窗口末帧——票 05 实测的时变冻结 bug 修复点）
            tdim_win = tdim[t0f:t0f + args.t_win]
            out, seeds = extract_pathlines(
                u_win, v_win, mask2d, None, xdim, ydim, tdim_win,
                patch_yx=patch_yx, patch_size=patch_size,
                t0=float(tdim[t0f]), L=16,
                rng=np.random.default_rng(t0f), return_seeds=True)
            stem = f"pathlines_t{t0f}_y{py}_x{px}"
            np.save(out_dir / f"{stem}.npy", out)
            if args.visualize:
                phys = np.empty_like(out[:, :, :2])
                phys[:, :, 0] = out[:, :, CH_PX] * geo["hx"] + geo["cx"]
                phys[:, :, 1] = out[:, :, CH_PY] * geo["hy"] + geo["cy"]
                plot_pathlines(
                    (u_win[0], v_win[0]), mask2d, phys, seeds,
                    xdim, ydim,
                    out_dir / f"{stem}.png",
                    title=f"pathlines t={tdim[t0f]:.2f}s "
                          f"patch=({py},{px}) "
                          f"256 lines / 64 groups")
            print(f"t0 帧 {t0f} (t={tdim[t0f]:.2f}s): "
                  f"({out.shape[0]}, {out.shape[1]}, {out.shape[2]}) "
                  f"无NaN={bool(np.isfinite(out).all())} "
                  f"无-1000={bool(not (out == -1000.0).any())} -> {out_dir}")
    return 0


if __name__ == "__main__":
    main()
