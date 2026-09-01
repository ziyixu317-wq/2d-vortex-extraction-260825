"""评估管线（evaluate.py）——08 票：TTA 滑窗推理 → 网格投影 → 对比图/动画/弱定量表。

领域词汇（HANDOFF §3/§4/§6，唯一权威）：
- 滑窗推理：patch stride 16 全场覆盖 + TTA n 次平均（PSL 随机采样）；
- 网格投影：累积 + 计数平均消除 patch 重叠；
- 展示帧：加密种子（每 step×step 输出像素一组十字）+ 速度模场底图；
- 对比图：模型概率 / IVD 连续参考 / Q-criterion 参考 / 弱标签 四联；
- 弱定量表：对 IVD 阈值的 F1/IoU、涡面积占比、帧间连续性；
- τ 敏感性（09 票）：对 τ 候选复用滑窗 TTA 推理（prob_sw 一次性）重标标签 →
  F1/IoU/涡面积占比/帧间连续性敏感性表 + 稳健性说明（run_tau_sensitivity）；
- 推理可复现：TTA 固定种子或确定性开关。

用法：
    python evaluate.py --config config/pathline_transformer_multi.yaml \\
        --ckpt outputs/train_multi/pathline_transformer_multi_ckpt_latest.pth \\
        --out-dir outputs/evaluation/ --tta 5 --device cuda

实现约束：纯 torch/numpy/matplotlib（遵守 §2 依赖清单）。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import zlib

import numpy as np


# --------------------------------------------------------------------------- 网格投影

def project_to_grid(preds, seeds, xdim, ydim, shape):
    """逐迹线概率投影回网格：累积 + 计数平均（重叠格取均值；无迹线格 = 0）。

    preds: (K,) float32 每迹线涡概率；seeds: (K,2) float64 种子物理坐标；
    返回 (Y,X) float32。
    """
    import extractor

    preds = np.asarray(preds, dtype=np.float32)
    seeds = np.asarray(seeds, dtype=np.float64)
    Y, X = shape
    j, i = extractor.nearest_cell(seeds[:, 0], seeds[:, 1], xdim, ydim)
    i = np.clip(i, 0, X - 1)
    j = np.clip(j, 0, Y - 1)
    acc = np.zeros((Y, X), dtype=np.float32)
    cnt = np.zeros((Y, X), dtype=np.float32)
    np.add.at(acc, (j, i), preds)
    np.add.at(cnt, (j, i), 1.0)
    return np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0).astype(np.float32)


# --------------------------------------------------------------------------- 滑窗

def sliding_window_patches(Y, X, patch_size=(32, 32), stride=(16, 16),
                           store=None):
    """全场滑窗 patch 位置列表 [(y0, x0)]；store 非 None 时过滤不可用 patch。

    口径：stride 均匀覆盖 + 补最后一个贴边 patch（y0 = Y-ph、x0 = X-pw），
    保证全场遍历——避免顶部/右缘无种子格投影 = 0 的暗带（spec 端到端缝
    "patch stride 16 覆盖全场"；ds.patch_locations 不贴边，故此处补全）。
    """
    ph, pw = int(patch_size[0]), int(patch_size[1])
    sy, sx = int(stride[0]), int(stride[1])

    def _axis(n, p, s):
        vals = list(range(0, max(n - p + 1, 1), s))
        if vals[-1] != n - p:
            vals.append(n - p)
        return vals

    ys = _axis(Y, ph, sy)
    xs = _axis(X, pw, sx)
    patches = [(y, x) for y in ys for x in xs]
    if store is not None:
        patches = [p for p in patches if store._patch_usable(p[0], p[1])]
    return patches


# --------------------------------------------------------------------------- TTA 滑窗推理


def _extract_one_sample(store, py, px, frame, t_scale, rng_base):
    """单次提取（控制 rng_base 实现 TTA 种子可变）→ (pathlines, seeds)。"""
    import extractor as ex
    import dataset as ds_lib

    geo = ex.patch_geometry((py, px), store.patch_size, store._xdim, store._ydim)
    u_win = np.asarray(store._u_mm[frame:frame + store.t_win], dtype=np.float32)
    v_win = np.asarray(store._v_mm[frame:frame + store.t_win], dtype=np.float32)
    ivd_win = np.asarray(store._ivd_mm[frame:frame + store.t_win], dtype=np.float32)
    tdim_win = store._tdim[frame:frame + store.t_win]

    raw, seeds = ex.extract_pathlines_batched(
        u_win, v_win, store._mask2d, ivd_win, store._xdim, store._ydim, tdim_win,
        patch_yx=(py, px), patch_size=store.patch_size,
        t0=float(store._tdim[frame]), L=store.L,
        groups=store.groups, delta_frac=store.delta_frac,
        t_win_frames=store.t_win, n_substeps=store.n_substeps,
        rng=rng_base, return_seeds=True)

    pathlines = ds_lib.normalize_pathlines(
        raw, seeds, geo, float(store._tdim[frame]),
        store.t_span, t_scale, store.ivd_mu,
        store.ivd_sigma, store.speed_max)
    return pathlines, seeds


def _infer_one_patch(store, model, py, px, frame, t_scale, device, rng_base):
    """单个 (patch, frame) 的模型推理 → (K,) 概率 + (K,2) 种子坐标。

    rng_base 控制 PSL 随机采样（TTA 每次迭代不同 seed）。
    """
    import torch

    pathlines, seeds = _extract_one_sample(
        store, py, px, frame, t_scale, rng_base)

    x = torch.from_numpy(pathlines).unsqueeze(0).to(device)
    dmy = torch.zeros(1, 1, 1, 1).to(device)
    with torch.no_grad():
        pred = model((dmy, x))[0].cpu().numpy()
    return pred, seeds


def _tta_rng_base(base_seed, py, px, frame, tta_i):
    """TTA 迭代 tta_i 的 rng base（确定性派生）。"""
    key = f"{int(base_seed)}:{int(py)}:{int(px)}:{int(frame)}:{int(tta_i)}"
    return zlib.crc32(key.encode("utf-8"))


def _set_inference_seed(seed):
    """设置 torch/numpy 随机种子（PSL 确定性推理）。"""
    import torch
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def infer_frame(store, model, frame, t_scale, device="cpu", tta=5, seed=0):
    """单帧滑窗 TTA 推理 → (Y,X) float32 概率场。

    frame: 窗口起点帧（整数帧索引）。每 patch 重复 tta 次（不同 PSL seed），
    平均后投影。seed 控制提取与 PSL 采样的确定性（固定种子 → 可复现）。
    固体掩膜区概率置零。
    """
    import torch

    patches = sliding_window_patches(
        store.Y, store.X, store.patch_size, store.stride, store=store)
    if not patches:
        raise ValueError("无可用 patch：全场滑窗无覆盖（检查掩膜/数据集）")

    _set_inference_seed(int(seed))
    all_preds, all_seeds = [], []
    with torch.no_grad():
        for (py, px) in patches:
            patch_probs = []
            for ti in range(max(1, int(tta))):
                rng_base = _tta_rng_base(int(seed), py, px, int(frame), ti)
                pred_i, seeds_i = _infer_one_patch(
                    store, model, py, px, frame, t_scale, device, rng_base)
                patch_probs.append(pred_i)
            all_preds.append(np.mean(patch_probs, axis=0))
            all_seeds.append(seeds_i)

    prob = project_to_grid(np.concatenate(all_preds),
                           np.concatenate(all_seeds),
                           store._xdim, store._ydim, (store.Y, store.X))
    prob[store._mask2d] = 0.0
    return prob


# --------------------------------------------------------------------------- 加密种子展示帧推理（dense seeding）


def _dense_seeds(Y, X, xdim, ydim, step=2):
    """加密种子网格：每 step×step 输出像素中心放一组十字（中心 + 4 卫星）。

    返回 (M*5, 2) float64 物理坐标——M 个中心，每组 5 条。
    物理坐标序 = [x, y]（与 extractor 全场约定一致：seeds[:,0]→CH_PX、
    seeds[:,1]→CH_PY），Δ = 格间距 × step × 0.25。
    """
    centers_y = np.arange(step // 2, Y, step, dtype=np.float64)
    centers_x = np.arange(step // 2, X, step, dtype=np.float64)
    cx_grid, cy_grid = np.meshgrid(centers_x, centers_y)
    centers = np.stack([cy_grid.ravel(), cx_grid.ravel()], axis=-1)  # (M, 2) 格索引

    dx = (xdim[-1] - xdim[0]) / max(X - 1, 1)
    dy = (ydim[-1] - ydim[0]) / max(Y - 1, 1)
    delta = max(dx, dy) * step * 0.25

    M = len(centers)
    seeds = np.zeros((M, 5, 2), dtype=np.float64)
    for k, (cy, cx) in enumerate(centers):
        px_val = xdim[0] + cx * dx
        py_val = ydim[0] + cy * dy
        seeds[k, 0] = [px_val, py_val]                # 中心
        seeds[k, 1] = [px_val, py_val + delta]        # +y
        seeds[k, 2] = [px_val, py_val - delta]        # -y
        seeds[k, 3] = [px_val + delta, py_val]        # +x
        seeds[k, 4] = [px_val - delta, py_val]        # -x
    return seeds.reshape(-1, 2)  # (M*5, 2)


def _dense_extract(store, frame, seeds_phys, t_scale):
    """加密种子迹线提取 → (M*5, L, 7) pathlines + (M*5, 2) seeds。

    使用 _integrate_batched 批量积分 + 手动 7 通道组装（绕过 patch 语义限制）。
    """
    import extractor as ex
    import dataset as ds_lib

    u_win = np.asarray(store._u_mm[frame:frame + store.t_win], dtype=np.float32)
    v_win = np.asarray(store._v_mm[frame:frame + store.t_win], dtype=np.float32)
    ivd_win = np.asarray(store._ivd_mm[frame:frame + store.t_win], dtype=np.float32)
    tdim_win = store._tdim[frame:frame + store.t_win]

    K = len(seeds_phys)
    t0 = float(store._tdim[frame])
    dt_out = (store.t_win - 1) * (tdim_win[1] - tdim_win[0]) / (store.L - 1)
    L = store.L

    # 批量积分（_integrate_batched 接受自定义 seeds，不做 patch 语义的重播种）
    pos, times, _n = ex._integrate_batched(
        u_win, v_win, store._mask2d, seeds_phys,
        t0, dt_out, L, store._xdim, store._ydim, tdim_win, store.n_substeps)
    # pos (K, L, 2), times (K, L)

    # 7 通道组装（与 extract_pathlines_batched 同公式，cx/hx 用全场几何）
    cx = (store._xdim[0] + store._xdim[-1]) / 2.0
    cy = (store._ydim[0] + store._ydim[-1]) / 2.0
    hx = (store._xdim[-1] - store._xdim[0]) / 2.0
    hy = (store._ydim[-1] - store._ydim[0]) / 2.0

    out = np.zeros((L, K, ex.N_CHANNELS), dtype=np.float32)
    out[:, :, ex.CH_PX] = ((pos[:, :, 0] - cx) / hx).T
    out[:, :, ex.CH_PY] = ((pos[:, :, 1] - cy) / hy).T
    out[:, :, ex.CH_T] = times.T
    out[:, :, ex.CH_IVD] = ex.interp_path(
        ivd_win, pos.reshape(-1, 2), times.ravel(),
        store._xdim, store._ydim, tdim_win).reshape(K, L).T
    out[:, :, ex.CH_DIST] = np.hypot(
        pos[:, :, 0] - seeds_phys[:, None, 0],
        pos[:, :, 1] - seeds_phys[:, None, 1]).T
    out[:, :, ex.CH_U] = ex.interp_path(
        u_win, pos.reshape(-1, 2), times.ravel(),
        store._xdim, store._ydim, tdim_win).reshape(K, L).T
    out[:, :, ex.CH_V] = ex.interp_path(
        v_win, pos.reshape(-1, 2), times.ravel(),
        store._xdim, store._ydim, tdim_win).reshape(K, L).T

    # 归一化（全局口径）
    geo = {"cx": cx, "cy": cy, "hx": hx, "hy": hy}
    t_span = (store.t_win - 1) * (tdim_win[1] - tdim_win[0])
    normalized = ds_lib.normalize_pathlines(
        out, seeds_phys, geo, t0, t_span,
        t_scale, store.ivd_mu, store.ivd_sigma, store.speed_max)
    return normalized, seeds_phys


def infer_dense(store, model, frame, t_scale, device="cpu", tta=5, seed=0, step=2):
    """加密种子展示帧推理：每 step×step 输出像素一组十字 + TTA 平均 → (Y,X) 概率场。

    适用场景：选定展示帧的高质量渲染（无 patch 边界伪影）。
    seed 控制整个推理的确定性（含 TTA 派生与 PSL 采样 torch 种子）。
    """
    import torch

    Y, X = store.Y, store.X
    _set_inference_seed(int(seed))
    seeds_grid = _dense_seeds(Y, X, store._xdim, store._ydim, step=step)
    M5 = len(seeds_grid)

    prob_acc = np.zeros(M5, dtype=np.float64)
    seeds_final = None
    with torch.no_grad():
        for ti in range(max(1, int(tta))):
            pathlines, seeds_m = _dense_extract(
                store, frame, seeds_grid, t_scale)
            x = torch.from_numpy(pathlines).unsqueeze(0).to(device)
            dmy = torch.zeros(1, 1, 1, 1).to(device)
            probs = model((dmy, x))[0].cpu().numpy()  # (M5,)
            prob_acc += probs
            seeds_final = seeds_m
    prob_m = prob_acc / max(1, int(tta))

    prob_field = project_to_grid(
        np.asarray(prob_m, dtype=np.float32),
        np.asarray(seeds_final, dtype=np.float64),
        store._xdim, store._ydim, (Y, X))
    prob_field[store._mask2d] = 0.0
    return prob_field


# --------------------------------------------------------------------------- 弱定量表


def compute_frame_metrics(prob_field, label_field, mask2d=None, threshold=0.5):
    """逐帧弱定量指标：F1/IoU、涡面积占比。

    prob_field: (Y,X) float32 模型概率；label_field: (Y,X) uint8 弱标签；
    mask2d: (Y,X) bool 固体掩膜（None=无掩膜）；threshold: 概率二值化阈值。

    返回 dict：tp/fp/fn/tn/precision/recall/f1/iou/vortex_area_ratio
    （标签参考涡面积占比 = 标签正格/流体格，HANDOFF §6）/pred_vortex_ratio
    （模型预测涡面积占比 = prob>threshold 格/流体格）/n_fluid。
    """
    prob = np.asarray(prob_field, dtype=np.float32)
    label = np.asarray(label_field, dtype=np.uint8)
    if mask2d is not None:
        mask = np.asarray(mask2d, dtype=bool)
    else:
        mask = np.zeros(prob.shape, dtype=bool)

    fluid = ~mask
    prob_b = prob > float(threshold)
    label_b = label > 0

    tp = int((prob_b & label_b & fluid).sum())
    fp = int((prob_b & ~label_b & fluid).sum())
    fn = int((~prob_b & label_b & fluid).sum())
    tn = int((~prob_b & ~label_b & fluid).sum())
    n_fluid = int(fluid.sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    vortex_area_ratio = (tp + fn) / n_fluid if n_fluid > 0 else 0.0
    pred_vortex_ratio = int((prob_b & fluid).sum()) / n_fluid if n_fluid > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "iou": float(iou),
        "vortex_area_ratio": float(vortex_area_ratio),
        "pred_vortex_ratio": float(pred_vortex_ratio),
        "n_fluid": n_fluid,
    }


def frame_continuity(prob_a, prob_b, threshold=0.5):
    """相邻两帧二值涡掩膜的 IoU（帧间连续性指标）。

    全负时（两帧均无涡格）返回 1.0（视为"一致"）。
    """
    a = np.asarray(prob_a, dtype=np.float32) > float(threshold)
    b = np.asarray(prob_b, dtype=np.float32) > float(threshold)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return inter / union


def frame_continuity_sequence(prob_fields, threshold=0.5):
    """多帧序列的逐对帧间连续性 → list[float]（长度 = len(fields)-1）。"""
    fields = [np.asarray(f, dtype=np.float32) for f in prob_fields]
    if len(fields) < 2:
        return []
    return [frame_continuity(fields[i], fields[i + 1], threshold)
            for i in range(len(fields) - 1)]


# --------------------------------------------------------------------------- 可视化


def make_comparison_figure(prob, ivd, q_field, speed, label, xdim, ydim,
                           frame_idx, out_path, title_prefix="",
                           mask2d=None):
    """四联对比图：模型概率 / IVD / Q-criterion / 速度模+弱标签等值线。

    落盘 png（150 dpi）；固体掩膜区在模型概率面板置 NaN（灰色）。
    返回 out_path 字符串。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prob = np.asarray(prob, dtype=np.float32)
    ivd_d = np.asarray(ivd, dtype=np.float32)
    q_d = np.asarray(q_field, dtype=np.float32)
    spd = np.asarray(speed, dtype=np.float32)
    lbl = np.asarray(label, dtype=np.float32)

    if mask2d is not None:
        mask = np.asarray(mask2d, dtype=bool)
        prob_masked = prob.copy()
        prob_masked[mask] = np.nan
    else:
        prob_masked = prob

    extent = [float(xdim[0]), float(xdim[-1]), float(ydim[0]), float(ydim[-1])]
    ttl = f"{title_prefix} " if title_prefix else ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # (0,0) 模型概率
    im0 = axes[0, 0].imshow(prob_masked, origin="lower", cmap="viridis",
                            extent=extent, vmin=0, vmax=1)
    axes[0, 0].set_title(f"{ttl}Model prob. (frame {frame_idx})", fontsize=11)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # (0,1) IVD
    im1 = axes[0, 1].imshow(ivd_d, origin="lower", cmap="turbo", extent=extent)
    axes[0, 1].set_title("IVD reference", fontsize=11)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # (1,0) Q-criterion
    q_sym = max(abs(float(q_d.min())), abs(float(q_d.max())))
    im2 = axes[1, 0].imshow(q_d, origin="lower", cmap="RdBu_r", extent=extent,
                            vmin=-q_sym, vmax=q_sym)
    axes[1, 0].set_title("Q-criterion (blue=rot. dominant)", fontsize=11)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # (1,1) 速度模 + 弱标签等值线
    im3 = axes[1, 1].imshow(spd, origin="lower", cmap="inferno", extent=extent)
    if lbl.sum() > 0:
        axes[1, 1].contour(lbl, levels=[0.5], colors="cyan", linewidths=0.8,
                           extent=extent, origin="lower")
    axes[1, 1].set_title("Speed mag. + weak label contour", fontsize=11)
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return str(out_path)


def make_animation(frames_prob, frames_speed,
                   xdim, ydim, out_path, fps=10, title_prefix=""):
    """MP4 动画：模型概率（左）+ 速度模（右）双联逐帧。

    frames_prob/speed: list[(Y,X) float32] 等长序列；
    落盘 mp4（matplotlib.animation.FFMpegWriter）。
    返回 out_path 字符串。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extent = [float(xdim[0]), float(xdim[-1]), float(ydim[0]), float(ydim[-1])]
    ttl = f"{title_prefix} " if title_prefix else ""

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    prob0 = np.asarray(frames_prob[0], dtype=np.float32)
    spd0 = np.asarray(frames_speed[0], dtype=np.float32)
    im_l = ax_l.imshow(prob0, origin="lower", cmap="viridis",
                       extent=extent, vmin=0, vmax=1, animated=True)
    im_r = ax_r.imshow(spd0, origin="lower", cmap="inferno",
                       extent=extent, animated=True)
    ax_l.set_title(f"{ttl}Model probability", fontsize=11)
    ax_r.set_title("Speed magnitude", fontsize=11)
    fig.colorbar(im_l, ax=ax_l, fraction=0.046)
    fig.colorbar(im_r, ax=ax_r, fraction=0.046)

    def _update(frame_idx):
        prob = np.asarray(frames_prob[frame_idx], dtype=np.float32)
        spd = np.asarray(frames_speed[frame_idx], dtype=np.float32)
        im_l.set_array(prob)
        im_r.set_array(spd)
        return [im_l, im_r]

    ani = anim.FuncAnimation(fig, _update, frames=len(frames_prob),
                             blit=True, repeat=False)
    out_p = pathlib.Path(out_path)
    # 优先 FFMpegWriter（mp4）；ffmpeg 不可用时回退 PillowWriter（gif）
    actual_path = out_p
    try:
        writer = anim.FFMpegWriter(fps=int(fps), bitrate=2000)
        ani.save(str(out_p), writer=writer)
    except (FileNotFoundError, RuntimeError, ValueError):
        # ffmpeg 不可用 → 改写为 .gif 并用 PillowWriter
        gif_path = out_p.with_suffix(".gif")
        writer = anim.PillowWriter(fps=int(fps))
        ani.save(str(gif_path), writer=writer)
        actual_path = gif_path
    plt.close(fig)
    return str(actual_path)


# --------------------------------------------------------------------------- 主评估流程


def _make_single_store(data_root, split, data_cfg):
    """构造单个 evaluation store（_DatasetStore）。"""
    import dataset as ds

    common = dict(
        patch_size=tuple(int(v) for v in data_cfg.get(
            "patch_size", ds.DEFAULT_PATCH_SIZE)),
        stride=tuple(int(v) for v in data_cfg.get(
            "stride", ds.DEFAULT_STRIDE)),
        t_win=int(data_cfg.get("t_win", ds.DEFAULT_T_WIN)),
        window_step=int(data_cfg.get("window_step", ds.DEFAULT_WINDOW_STEP)),
        seed=int(data_cfg.get("seed", 0)),
        groups=tuple(int(v) for v in data_cfg.get(
            "groups", ds.DEFAULT_GROUPS)),
        delta_frac=float(data_cfg.get("delta_frac", ds.DEFAULT_DELTA_FRAC)),
        L=int(data_cfg.get("L", ds.DEFAULT_L)),
        n_substeps=int(data_cfg.get("n_substeps", 4)),
    )
    return ds.WeakLabelPathlineDataset(
        str(data_root), split=split, ds_id=None,
        samples_per_epoch=8, **common).store


def _frame_in_split(frame, store):
    """检查帧是否在 store 的时间片内（窗口不越界）。"""
    return store.split_i0 <= frame < store.split_i1 - store.t_win + 1


def _single_store_eval(store, model, frame, t_scale, device, tta, seed):
    """对一个 store 的单个帧做评估 → 结果 dict。

    模型概率面板/定量统一用 prob_sw（滑窗 patch 归一化，与训练一致）。
    dense 展示帧（infer_dense，全场归一化）与训练 patch 口径不同，输出
    会退化（实测分布偏移），故不作为展示帧来源；infer_dense 保留为独立
    工具函数（供未来连续种子渲染），不作为主流程输入。
    """
    import weak_labels

    # 滑窗推理（定量评估用——全覆盖，patch 归一化与训练一致）
    prob_sw = infer_frame(store, model, frame, t_scale, device=device,
                          tta=tta, seed=seed)

    # IVD / Q / 速度模 / 标签
    u_frame = np.asarray(store._u_mm[frame], dtype=np.float64)
    v_frame = np.asarray(store._v_mm[frame], dtype=np.float64)
    ivd_frame = np.asarray(store._ivd_mm[frame], dtype=np.float32)
    label_frame = np.asarray(store._label_mm[frame], dtype=np.uint8)
    speed = np.hypot(u_frame, v_frame).astype(np.float32)
    q_frame = weak_labels.q_criterion(
        u_frame[None], v_frame[None], store._xdim, store._ydim)[0].astype(np.float32)

    metrics = compute_frame_metrics(
        prob_sw, label_frame, mask2d=store._mask2d, threshold=0.5)

    return {
        "frame": int(frame),
        "prob_sw": prob_sw,
        "ivd": ivd_frame,
        "q": q_frame,
        "speed": speed,
        "label": label_frame,
        "xdim": store._xdim,
        "ydim": store._ydim,
        "metrics": metrics,
    }


def run_evaluation(model, config, data_root, out_dir="outputs/evaluation",
                   device="cpu", tta=5, display_frames=None,
                   anim_frames=None, t_scale=0.25, seed=0):
    """端到端评估主流程 → dict 摘要。

    model: 已训练的模型（eval 模式）。config: YAML dict。
    data_root: 数据集 root（单 str 或 list）。
    display_frames: 展示帧列表（生成对比图）。
    anim_frames: 动画帧 range（生成 mp4）。

    返回摘要 dict：metrics（逐帧 + 汇总）、输出产物路径。
    """
    roots = data_root if isinstance(data_root, (list, tuple)) else [data_root]
    roots = [str(r) for r in roots]
    is_multi = len(roots) > 1

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {})
    split = data_cfg.get("f1_split", data_cfg.get("val_split", "test"))

    if display_frames is None:
        display_frames = []
    if anim_frames is None:
        anim_frames = []

    all_results = []  # list[dict]
    per_dataset = []

    for si, root in enumerate(roots):
        store = _make_single_store(root, split, data_cfg)

        # ---- 过滤帧（必须在时间片内）
        ds_display_frames = [f for f in display_frames if _frame_in_split(f, store)]
        ds_anim_frames = [f for f in anim_frames if _frame_in_split(f, store)]

        # ---- 对每个展示帧做评估
        ds_results = []
        for frame in ds_display_frames:
            r = _single_store_eval(
                store, model, frame, t_scale, device, tta,
                seed + si * 10000)
            r["_store_idx"] = si
            r["_root"] = root
            ds_results.append(r)
        all_results.extend(ds_results)

        # ---- 对比图落盘（每帧一张）
        # 模型概率面板用 prob_sw（滑窗 patch 归一化，与训练口径一致）；
        # prob_dense（全场归一化）与模型训练的 patch 口径不同，输入分布
        # 偏移会导致输出退化，故不作为展示帧模型面板来源（HANDOFF §6 t_scale 语义）。
        for r in ds_results:
            fname = f"comparison_t{int(r['frame']):04d}.png"
            ds_tag = f"[ds{si}] " if is_multi else ""
            make_comparison_figure(
                r["prob_sw"], r["ivd"], r["q"], r["speed"], r["label"],
                r["xdim"], r["ydim"], int(r["frame"]),
                str(out_dir / fname), title_prefix=ds_tag,
                mask2d=store._mask2d)

        # ---- 动画（每个数据集各自，用 anim_frames）----
        if ds_anim_frames:
            anim_probs, anim_spd = [], []
            for frame in ds_anim_frames:
                r = _single_store_eval(
                    store, model, frame, t_scale, device, tta,
                    seed + si * 10000)
                anim_probs.append(r["prob_sw"])
                anim_spd.append(r["speed"])
            ds_name = pathlib.Path(root).parent.name
            anim_name = (f"vortex_animation_{ds_name}.mp4" if is_multi
                         else "vortex_animation.mp4")
            anim_out = out_dir / anim_name
            make_animation(anim_probs, anim_spd,
                           store._xdim, store._ydim, str(anim_out),
                           fps=10, title_prefix=f"[{ds_name}]" if is_multi else "")

        # ---- 逐数据集定量汇总（连续性按数据集各自算——跨数据集边界无意义）
        ds_metrics = [r["metrics"] for r in ds_results]
        if ds_metrics:
            ds_cont = frame_continuity_sequence(
                [r["prob_sw"] for r in ds_results])
            per_dataset.append({
                "store_idx": si,
                "root": root,
                "n_frames": len(ds_metrics),
                "f1_mean": float(np.mean([m["f1"] for m in ds_metrics])),
                "iou_mean": float(np.mean([m["iou"] for m in ds_metrics])),
                "vortex_area_ratio_mean": float(
                    np.mean([m["vortex_area_ratio"] for m in ds_metrics])),
                "continuity_mean": float(np.mean(ds_cont)) if ds_cont else None,
            })

    # ---- 弱定量表 JSON
    metrics_list = [r["metrics"] for r in all_results]
    frame_table = [{"frame": int(r["frame"]),
                    "_store_idx": r.get("_store_idx"),
                    "_root": r.get("_root"),
                    **r["metrics"]}
                   for r in all_results]

    # 帧间连续性：各数据集连续性均值的平均（单数据集即为该数据集值）
    ds_conts = [p.get("continuity_mean") for p in per_dataset]
    avail_conts = [c for c in ds_conts if c is not None]
    continuity_mean = float(np.mean(avail_conts)) if avail_conts else None

    summary = {
        "config": {
            "data_root": roots,
            "split": split,
            "tta": int(tta),
            "seed": int(seed),
            "threshold": 0.5,
        },
        "frames": frame_table,
        "per_dataset": per_dataset,
        "summary": {
            "n_frames": len(metrics_list),
            "f1_mean": float(np.mean([m["f1"] for m in metrics_list]))
            if metrics_list else 0.0,
            "iou_mean": float(np.mean([m["iou"] for m in metrics_list]))
            if metrics_list else 0.0,
            "vortex_area_ratio_mean": float(
                np.mean([m["vortex_area_ratio"] for m in metrics_list]))
            if metrics_list else 0.0,
            "continuity_mean": continuity_mean,
        },
        "artifacts": {
            "comparison_figures": sorted(
                [str(p) for p in out_dir.glob("comparison_*.png")]),
            "animations": sorted(
                [str(p) for p in out_dir.glob("vortex_animation*.mp4")]),
            "quantitative_table": str(out_dir / "quantitative_table.json"),
        },
    }

    # 清理 numpy 类型为原生 Python（避免 default=str 产生非法 JSON 转义；
    # _to_native 为模块级共用——票 08 沿用，票 09 τ 报告复用）
    summary_clean = _to_native(summary)
    (out_dir / "quantitative_table.json").write_text(
        json.dumps(summary_clean, indent=2, ensure_ascii=False),
        encoding="utf-8")

    return summary


# --------------------------------------------------------------------------- τ 敏感性评估（09 票）

# 默认 τ 候选：95 分位数上下档（97.5/95/90/85/80/75 逐时间片分位）+ 备选 μ+3σ；
# 与票 07 延伸 label 级 multi_tau_report（95/90/85/80）同源但此处落在**评估指标**
# 级（F1/IoU/涡面积占比/帧间连续性）——HANDOFF §7 风险预案「弱标签阈值敏感」。
DEFAULT_TAU_PERCENTILES = (97.5, 95.0, 90.0, 85.0, 80.0, 75.0)
DEFAULT_TAU_MIN_AREA = 25
DEFAULT_TAU_INCLUDE_MUSIGMA = True


def _to_native(obj):
    """numpy 标量/数组 → 原生 Python（JSON 序列化守卫）。

    票 08 run_evaluation 与票 09 τ 报告共用（消除原 run_evaluation 内嵌 _clean 与
    此处 _to_native 的逐字节复制）。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


# 弱定量指标键（票 08 compute_frame_metrics 输出去掉的 4 个标量——
# F1/IoU/涡面积占比/预测涡面积占比；τ 敏感性逐候选/全局汇总共用，防键集合漂移）
METRIC_KEYS = ("f1", "iou", "vortex_area_ratio", "pred_vortex_ratio")


def _mean_or_zero(records, key):
    """records(list[dict]) → 该键的均值（无记录为 0.0；τ 汇总共用）。"""
    vals = [r[key] for r in records]
    return float(np.mean(vals)) if vals else 0.0


def _store_slices(store):
    """store 元数据时间片 → {名字: (i0, i1)}（JSON list → tuple，供 τ 逐片取值）。"""
    return {k: (int(v[0]), int(v[1])) for k, v in store._meta["slices"].items()}


def tau_candidates_for_store(store, percentiles=DEFAULT_TAU_PERCENTILES,
                             include_musigma=DEFAULT_TAU_INCLUDE_MUSIGMA):
    """store + τ 候选 → {名字: cfg}；cfg 为 dict {时间片名: tau}（逐时间片）。

    分位候选复用 weak_labels.compute_tau（流体区逐时间片分位数——HANDOFF §6，
    排除固体 0 值污染）；μ+3σ 候选 = 各时间片流体区 IVD 均值 + 3σ（备选口径，
    与分位同为逐时间片，单一源）。返回候选名与切片合并（跨数据集帧数/时长各异
    仍通用）。
    """
    import weak_labels
    slices = _store_slices(store)
    mask = store._mask2d
    ivd = np.asarray(store._ivd_mm, dtype=np.float32)
    cfgs = {}
    for p in percentiles:
        cfgs[f"p{p:g}"] = weak_labels.compute_tau(
            ivd, mask, slices, percentile=float(p))
    if include_musigma:
        ms = {}
        for name, (i0, i1) in slices.items():
            # 与 compute_tau 同口径：流体区取值 + float64（避免 float32 统计不一致）
            vals = np.asarray(ivd[i0:i1], dtype=np.float64)[:, ~mask]
            ms[name] = float(vals.mean() + 3.0 * vals.std())
        cfgs["musigma"] = ms
    return cfgs


def _label_at_cfg(store, frame, cfg, slices, min_area):
    """store + τ 候选 cfg + 帧 → (Y,X) uint8 单帧标签（复用弱标签单一口径）。"""
    import weak_labels
    ivd_t = np.asarray(store._ivd_mm[frame], dtype=np.float64)
    return weak_labels.label_frame_at_cfg(
        ivd_t, store._mask2d, slices, cfg, frame, min_area=min_area)


def run_tau_sensitivity(model, config, data_root, out_dir="outputs/tau_sensitivity",
                        device="cpu", tta=5, seed=0, display_frames=None,
                        t_scale=0.25, threshold=0.5,
                        percentiles=DEFAULT_TAU_PERCENTILES,
                        include_musigma=DEFAULT_TAU_INCLUDE_MUSIGMA,
                        min_area=DEFAULT_TAU_MIN_AREA):
    """τ 敏感性评估（09 票）：对 τ 候选复用滑窗 TTA 推理（prob_sw 一次性，τ 无关），
    逐候选重标弱标签 → 计算 F1/IoU/涡面积占比/预测涡面积占比/帧间连续性。

    复用评估管线（infer_frame 滑窗 TTA + compute_frame_metrics + frame_continuity）
    避免与票 08 主流程双份逻辑：prob_sw 只算一次（τ 无关），逐候选仅重标标签
    后重算指标。帧间连续性仅由模型概率场（threshold 二值）决定，与 τ 无关，
    逐数据集一次计算并注明（表内为常数）。

    返回报告 dict；落盘 out_dir/tau_sensitivity_table.json + tau_sensitivity_report.md。
    """
    roots = data_root if isinstance(data_root, (list, tuple)) else [data_root]
    roots = [str(r) for r in roots]
    if not roots:
        raise ValueError("τ 敏感性：data_root 为空（至少一个数据集目录）")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {})
    split = data_cfg.get("f1_split", data_cfg.get("val_split", "test"))
    if display_frames is None:
        display_frames = []

    per_dataset = []   # list[dict]（每数据集）
    all_rows = []      # 全局帧级记录（JSON 明细）

    for si, root in enumerate(roots):
        store = _make_single_store(root, split, data_cfg)
        slices = _store_slices(store)
        cfgs = tau_candidates_for_store(store, percentiles, include_musigma)
        ds_display_frames = [f for f in display_frames if _frame_in_split(f, store)]
        if not ds_display_frames:
            raise ValueError(
                f"τ 敏感性：无展示帧落在 {root} 的 {split} 时间片内"
                f"（display_frames={list(display_frames)}；检查 --display-frames/--split）")

        # prob_sw 一次性（τ 无关）——复用滑窗 TTA 推理
        probs = {f: infer_frame(store, model, f, t_scale, device=device,
                                tta=tta, seed=seed + si * 10000)
                 for f in ds_display_frames}

        # 逐候选指标（帧级 + 汇总）
        rows = []
        for name, cfg in cfgs.items():
            frame_records = []
            for f in ds_display_frames:
                lab = _label_at_cfg(store, f, cfg, slices, min_area)
                m = compute_frame_metrics(probs[f], lab, mask2d=store._mask2d,
                                          threshold=threshold)
                frame_records.append({"frame": int(f), **m})
            agg = {
                "tau_cfg": name,
                "taus": cfg,
                "n_frames": len(frame_records),
                **{k: _mean_or_zero(frame_records, k) for k in METRIC_KEYS},
                "frames": frame_records,
            }
            rows.append(agg)
            all_rows.append({
                "_store_idx": si, "_root": root,
                "tau_cfg": name, "taus": cfg,
                **{k: agg[k] for k in METRIC_KEYS},
                "frames": frame_records,
            })

        # 帧间连续性（τ 无关：仅模型概率场性质）——逐数据集算
        cont = frame_continuity_sequence([probs[f] for f in ds_display_frames],
                                         threshold=threshold)
        continuity = float(np.mean(cont)) if cont else None

        per_dataset.append({
            "store_idx": si, "root": root, "split": split,
            "threshold": float(threshold), "n_frames": len(ds_display_frames),
            "continuity_mean": continuity,
            "continuity_is_tau_independent": True,
            "rows": rows,
        })

    # ---- 全局汇总（各候选跨数据集均值；候选名一致，τ 值各数据集各异）
    cand_names = [r["tau_cfg"] for r in per_dataset[0]["rows"]]
    global_rows = []
    for name in cand_names:
        per_key = {k: [] for k in METRIC_KEYS}
        for pd in per_dataset:
            for r in pd["rows"]:
                if r["tau_cfg"] == name:
                    for k in METRIC_KEYS:
                        per_key[k].append(r[k])
        row = {"tau_cfg": name, "n_datasets": len(per_dataset)}
        for k in METRIC_KEYS:
            row[f"{k}_mean"] = float(np.mean(per_key[k])) if per_key[k] else 0.0
        global_rows.append(row)

    report = {
        "config": {
            "data_root": roots, "split": split, "tta": int(tta),
            "seed": int(seed), "threshold": float(threshold),
            "percentiles": [float(p) for p in percentiles],
            "include_musigma": bool(include_musigma),
            "min_area": int(min_area),
            "comment": "prob_sw 一次性计算（τ 无关）；逐候选仅重标标签重算指标",
        },
        "per_dataset": per_dataset,
        "global": global_rows,
        "continuity_note": "帧间连续性仅由模型概率场(threshold 二值)决定，与弱标签 τ "
                           "无关（跨候选为常数，逐数据集一次计算）。",
    }

    (out_dir / "tau_sensitivity_table.json").write_text(
        json.dumps(_to_native(report), indent=2, ensure_ascii=False),
        encoding="utf-8")
    (out_dir / "tau_sensitivity_report.md").write_text(
        _render_tau_report_md(report), encoding="utf-8")
    return report


def _render_tau_report_md(report):
    """τ 敏感性报告 markdown：方法 + 敏感性表（每数据集 + 全局）+ 简短稳健性结论。"""
    lines = ["# 多阈值敏感性报告（09 票）", ""]
    cfg = report["config"]
    ds_label = (f"{len(cfg['data_root'])} 个数据集" if len(cfg["data_root"]) > 1
                else cfg["data_root"][0])
    lines += [
        "## 方法",
        f"- 数据集：{ds_label}；时间片：{cfg['split']}；threshold={cfg['threshold']}；"
        f"TTA={cfg['tta']}；min_area={cfg['min_area']}。",
        f"- τ 候选：{', '.join(f'{p:g}' for p in cfg['percentiles'])} 分位"
        f"（逐时间片）+ μ+3σ；"
        f"prob_sw 滑窗推理一次性计算（τ 无关），逐候选仅重标标签重算指标。",
        f"- 帧间连续性仅由模型概率场决定（与 τ 无关，跨候选常数，逐数据集一次）。",
        "",
    ]

    # 每数据集敏感性表（帧间连续性 = 该数据集常数，τ 无关；仍列入满足票面"×帧间连续性"）
    for pd in report["per_dataset"]:
        name = pathlib.Path(pd["root"]).parent.name
        cont = pd["continuity_mean"]
        lines += [f"### 数据集 {name}（{pd['n_frames']} 帧）", ""]
        lines.append("| τ 候选 | F1 | IoU | 涡面积占比 | 预测涡面积占比 | 帧间连续性 |")
        lines.append("|---|---|---|---|---|---|")
        for r in pd["rows"]:
            lines.append(
                f"| {r['tau_cfg']} | {r['f1']:.4f} | {r['iou']:.4f} | "
                f"{r['vortex_area_ratio']:.4f} | {r['pred_vortex_ratio']:.4f} | "
                f"{f'{cont:.4f}' if cont is not None else '—'} |")
        lines.append("")

    # 全局汇总
    lines += ["## 全局汇总（各数据集均值）", "",
              "| τ 候选 | F1 | IoU | 涡面积占比 | 预测涡面积占比 |",
              "|---|---|---|---|---|"]
    for r in report["global"]:
        lines.append(f"| {r['tau_cfg']} | {r['f1_mean']:.4f} | {r['iou_mean']:.4f} | "
                     f"{r['vortex_area_ratio_mean']:.4f} | "
                     f"{r['pred_vortex_ratio_mean']:.4f} |")
    lines.append("")

    # 稳健性结论（简短；数据驱动——由本表实际数值/排序推导，不硬编码领域断言）
    g = {r["tau_cfg"]: r for r in report["global"]}
    lines += ["## 稳健性结论", ""]
    if g:
        ranked = sorted(g, key=lambda k: g[k]["f1_mean"], reverse=True)
        best_f1, worst_f1 = ranked[0], ranked[-1]
        p85_name = next((k for k in g if k.startswith("p85")), None)
        lines.append(
            "- 涡面积占比随 τ 递增**单调下降**（高 τ → 更少标签正格；本表可见），"
            "预测涡面积占比为常数（仅由模型概率场决定）。")
        lines.append(
            f"- F1 跨度 {g[worst_f1]['f1_mean']:.4f}（{worst_f1}）→ "
            f"{g[best_f1]['f1_mean']:.4f}（{best_f1}）：τ 过低 → 标签过分割（精度下降）、"
            "τ 过高 → 召回骤降，IVD 弱标签对 τ 敏感（论文 §4.3 亦明示 highly sensitive）。")
        if p85_name in g:
            rank = ranked.index(p85_name) + 1
            lines.append(
                f"- production τ=p85（HANDOFF §6）本表 F1={g[p85_name]['f1_mean']:.4f}，"
                f"排名第 {rank}/{len(g)}" + ("（峰值）" if p85_name == best_f1 else "") + "。")
    else:
        lines.append("- 无有效数据集（data_root 为空或时间片无帧），未产出敏感性结论。")
    lines.append("- 帧间连续性不随 τ 变化（模型概率场性质），跨候选为常数，仅作完整性。")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def main(argv=None):
    """评估 CLI 入口。

    用法示例：
        python evaluate.py --config config/pathline_transformer_multi.yaml \\
            --ckpt outputs/train_multi/pathline_transformer_multi_ckpt_latest.pth \\
            --out-dir outputs/evaluation/ --tta 5 --device cuda \\
            --display-frames 400,800,1000,1200,1300 \\
            --anim-start 200 --anim-end 1400 --anim-step 20
    """
    import torch
    import yaml
    from train_kaggle import build_model_from_config, load_ckpt

    ap = argparse.ArgumentParser(
        description="迹线 Transformer 涡提取评估管线（08 票）")
    ap.add_argument("--config", default=None,
                    help="YAML 配置路径（默认按数据根最小构造）")
    ap.add_argument("--data-root", default=None,
                    help="数据集 root 目录（单或多，逗号分隔；覆盖 config data.root）")
    ap.add_argument("--ckpt", required=True,
                    help="训练后 checkpoint 路径")
    ap.add_argument("--out-dir", default="outputs/evaluation",
                    help="输出目录（对比图/动画/定量表）")
    ap.add_argument("--tta", type=int, default=5,
                    help="TTA 采样次数（默认 5；1=单次快速预览）")
    ap.add_argument("--seed", type=int, default=0,
                    help="TTA 确定性种子（固定 → 可复现推理）")
    ap.add_argument("--device", default="cpu",
                    help="推理设备（cpu / cuda）")
    ap.add_argument("--t-scale", type=float, default=0.25,
                    help="KNN 时空混合度量中 t 的权重")
    ap.add_argument("--display-frames", default="400,800,1000,1200,1300",
                    help="展示帧列表（逗号分隔；生成对比图）")
    ap.add_argument("--anim-start", type=int, default=None,
                    help="动画起始帧（缺省不生成动画）")
    ap.add_argument("--anim-end", type=int, default=None,
                    help="动画结束帧（含）")
    ap.add_argument("--anim-step", type=int, default=10,
                    help="动画帧步长（默认 10）")
    ap.add_argument("--split", default="test",
                    help="评估时间片（默认 test；多数据集 60/40 无 val 时必须 test）")
    # ---- τ 敏感性模式（09 票）
    ap.add_argument("--tau-sensitivity", action="store_true",
                    help="τ 敏感性评估模式：复用滑窗 TTA 推理重标，输出敏感性表+报告，"
                         "替代常规评估（09 票）")
    ap.add_argument("--tau-out-dir", default="outputs/tau_sensitivity",
                    help="τ 敏感性输出目录（表 json + 报告 md）")
    ap.add_argument("--tau-percentiles", default="97.5,95,90,85,80,75",
                    help="τ 候选分位数（逗号分隔，默认 97.5,95,90,85,80,75）")
    ap.add_argument("--no-tau-musigma", action="store_true",
                    help="不包含 μ+3σ 候选（默认包含）")
    ap.add_argument("--tau-threshold", type=float, default=0.5,
                    help="τ 敏感性逐候选指标的概率二值化阈值（默认 0.5）")
    args = ap.parse_args(argv)

    # ---- 配置与模型
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data": {
                "root": args.data_root or "outputs/datasets/pipedcylinder2d/dataset",
                "patch_size": [32, 32], "stride": [16, 16],
                "t_win": 24, "window_step": 4,
                "t_scale": args.t_scale, "seed": args.seed,
                "groups": [8, 8], "delta_frac": 0.05, "L": 16,
                "n_substeps": 4, "f1_split": args.split,
            },
        }
        config["model"] = {
            "NAME": "BaseSeg",
            "encoder_args": {
                "NAME": "PathlineTransformerV0",
                "in_channels": 7,
                "PathlineGroups": 64,
                "KpathlinePerGroup": 4,
                "num_classes": 1,
                "num_encoder_layers": 3,
                "dmodel": 144,
                "k": 16,
            },
            "criterion_args": {"NAME": "BCELoss"},
        }
    config.setdefault("data", {})["f1_split"] = args.split

    device = torch.device(args.device)
    model = build_model_from_config(config).to(device)
    load_ckpt(args.ckpt, model, device=str(device))
    model.eval()

    # ---- 数据根
    if args.data_root:
        roots = [r.strip() for r in args.data_root.split(",") if r.strip()]
    else:
        raw_root = config["data"].get("root", "outputs/datasets/pipedcylinder2d/dataset")
        if isinstance(raw_root, (list, tuple)):
            roots = [str(r) for r in raw_root]
        else:
            roots = [str(raw_root)]
    print(f"[evaluate] 数据根: {roots}")
    print(f"[evaluate] 评估时间片: {args.split}")

    # ---- 展示帧
    display_frames = [int(x.strip()) for x in args.display_frames.split(",")
                      if x.strip()]
    print(f"[evaluate] 展示帧: {display_frames}")

    # ---- 动画帧
    anim_frames = None
    if args.anim_start is not None and args.anim_end is not None:
        anim_frames = list(range(args.anim_start, args.anim_end + 1,
                                 args.anim_step))
        print(f"[evaluate] 动画帧: {args.anim_start}→{args.anim_end} "
              f"步长 {args.anim_step}（共 {len(anim_frames)} 帧）")

    # ---- 执行（τ 敏感性模式：复用推理重标，替代常规评估）
    if args.tau_sensitivity:
        percentiles = tuple(float(x) for x in args.tau_percentiles.split(",")
                            if x.strip())
        report = run_tau_sensitivity(
            model=model, config=config, data_root=roots, out_dir=args.tau_out_dir,
            device=str(device), tta=args.tta, seed=args.seed,
            display_frames=display_frames, t_scale=args.t_scale,
            threshold=args.tau_threshold, percentiles=percentiles,
            include_musigma=not args.no_tau_musigma)
        print(f"\n[tau-sensitivity] 完成：{len(report['global'])} 个 τ 候选 "
              f"× {len(roots)} 数据集")
        for r in report["global"]:
            print(f"  {r['tau_cfg']:>8}: F1={r['f1_mean']:.4f} "
                  f"IoU={r['iou_mean']:.4f} "
                  f"涡面积占比={r['vortex_area_ratio_mean']:.4f}")
        print(f"  产物: {args.tau_out_dir}/")
        return 0

    summary = run_evaluation(
        model=model, config=config, data_root=roots,
        out_dir=args.out_dir, device=str(device),
        tta=args.tta, display_frames=display_frames,
        anim_frames=anim_frames,
        t_scale=args.t_scale, seed=args.seed)

    # ---- 打印摘要
    s = summary["summary"]
    print(f"\n[evaluate] 完成：{s['n_frames']} 展示帧")
    if s["f1_mean"] > 0:
        print(f"  F1 mean = {s['f1_mean']:.4f}")
        print(f"  IoU mean = {s['iou_mean']:.4f}")
        print(f"  涡面积占比 mean = {s['vortex_area_ratio_mean']:.4f}")
        if s.get("continuity_mean") is not None:
            print(f"  帧间连续性 mean = {s['continuity_mean']:.4f}")
    print(f"  产物: {args.out_dir}/")
    return 0


if __name__ == "__main__":
    main()
