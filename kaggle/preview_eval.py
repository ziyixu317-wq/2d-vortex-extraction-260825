"""单帧涡提取预览（07 票：12h 中途看模型效果的轻量入口；正式评估属票 08）。

领域词汇（HANDOFF §3/§4 与规格，唯一权威）：
- 用途：加载 latest checkpoint → 指定帧的滑窗样本（patch stride 16 全场覆盖）→
  模型逐迹线涡概率（可选 TTA n 次平均，PSL 随机采样）→ 投影回网格
  （累积 + 计数平均消除 patch 重叠，与票 08 规格同口径的简版）→ 三联图
  （模型概率场 / IVD / 弱标签）落盘。**非交付级评估**：单帧、单次/少次采样；
  定量表、动画、多帧目检属票 08 evaluate.py。
- 中途模型（12h 会话分块前段）在此目检只用于「管线正确性检查」：
  高概率区域应大致落在涡街/拐角回流区；边界毛糙、背景噪声属预期
  （论文 200 epoch 全训的早期阶段）。

用法：python kaggle/preview_eval.py --config config/pathline_transformer_cylinder.yaml \
      --ckpt outputs/train/pathline_transformer_cylinder_ckpt_latest.pth \
      --frame 1300 --out outputs/preview/prob_vs_ivd_t1300.png [--tta 3] [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# CLI 入口（python kaggle/preview_eval.py）从任意 cwd 运行时也能 import 项目模块
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np


def project_to_grid(preds, seeds, xdim, ydim, shape):
    """逐迹线概率投影回网格：累积 + 计数平均（重叠格取均值；无迹线格 = 0）。

    口径与票 08 规格一致（滑窗投影的基础元操作）；seeds 为物理坐标 (K,2)。
    返回值 (Y,X) float32。
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


def run_preview(config_path, ckpt_path, frame, out_path, tta=1, device="cpu"):
    """单帧预览主流程 → (prob_field, ivd_frame, label_frame) 并落盘三联图。

    frame 必须在某时间片内且窗口 [frame, frame+t_win) ⊆ 片区间（时间口径
    与数据集一致）；输出 png（模型概率 / IVD / 弱标签 3 联）。
    """
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml

    import dataset as ds
    from train_kaggle import build_model_from_config, load_ckpt

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = ds.load_dataset_meta(cfg["data"]["root"])
    frame = int(frame)
    # ---- 定位含 frame 的时间片（闭包口径与数据集一致）
    split = None
    for name, (i0, i1) in data_cfg["slices"].items():
        if i0 <= frame < i1:
            split = name
    if split is None:
        raise ValueError(f"帧 {frame} 不在任何时间片 {data_cfg['slices']} 内")
    t_win = int(cfg["data"]["t_win"])
    if frame + t_win > data_cfg["slices"][split][1]:
        raise ValueError(f"帧 {frame} 的窗口 [.., {frame + t_win}) 超出片 {split} 右界")

    # ---- 数据集与模型
    d = ds.WeakLabelPathlineDataset(
        cfg["data"]["root"], split=split,
        patch_size=tuple(int(v) for v in cfg["data"]["patch_size"]),
        stride=tuple(int(v) for v in cfg["data"]["stride"]),
        t_win=t_win, window_step=int(cfg["data"]["window_step"]),
        samples_per_epoch=8, seed=0)
    # 上式样本数仅占位（preview 不走 set_epoch 采样序；池构建必需）
    device = torch.device(device)
    model = build_model_from_config(cfg).to(device)
    load_ckpt(ckpt_path, model, device=str(device))
    model.eval()

    # ---- 滑窗：全场 patch（stride 16）+ 可用 patch + 窗口在片内
    y0s, x0s = [], []
    patches = ds.patch_locations(d.Y, d.X, d.patch_size, d.stride)
    usable = [p for p in patches if d._patch_usable(p[0], p[1])]
    if not usable:
        raise ValueError("可用 patch 为空：数据集无可用滑窗位置")
    all_preds, all_seeds = [], []
    with torch.no_grad():
        for (py, px) in usable:
            (dummy, path), _labels, seeds = d.sample_at(py, px, frame)
            x = torch.from_numpy(path).unsqueeze(0).to(device)
            dmy = torch.from_numpy(dummy).to(device)
            probs = []
            for _ in range(max(1, int(tta))):
                probs.append(model((dmy, x))[0].cpu().numpy())
            all_preds.append(np.mean(probs, axis=0))
            all_seeds.append(seeds)
    prob = project_to_grid(np.concatenate(all_preds),
                           np.concatenate(all_seeds),
                           d._xdim, d._ydim, (d.Y, d.X))
    ivd = np.asarray(d._ivd_mm[frame])
    label = np.asarray(d._label_mm[frame], dtype=np.float32)
    mask2d = np.asarray(np.load(Path(cfg["data"]["root"]) / "mask.npy"), dtype=bool)
    prob[mask2d] = 0.0            # 固体区不显示（与弱标签口径一致）

    # ---- 三联图（物理坐标 extent；与票 04 目检图同风格；标题用英文——
    #      Kaggle/无中文字体环境下 DejaVu Sans 缺 CJK 字形会出豆腐块）
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xdim, ydim = np.asarray(d._xdim), np.asarray(d._ydim)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, data, title, cmap in (
            (axes[0], prob, f"Model prob. (frame {frame}, tta={tta})", "viridis"),
            (axes[1], ivd, "IVD reference", "turbo"),
            (axes[2], label, "Weak label", "gray")):
        im = ax.imshow(data, origin="lower", cmap=cmap,
                       extent=[float(xdim[0]), float(xdim[-1]),
                               float(ydim[0]), float(ydim[-1])])
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return prob, ivd, label


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="单帧涡提取预览（07 票中途目检；正式评估属票 08）")
    ap.add_argument("--config", default="config/pathline_transformer_cylinder.yaml")
    ap.add_argument("--ckpt", required=True, help="checkpoint 路径（如 latest）")
    ap.add_argument("--frame", type=int, required=True, help="展示帧（须在时间片内且窗口不越界）")
    ap.add_argument("--out", default=None, help="输出 png 路径（默认 outputs/preview/）")
    ap.add_argument("--tta", type=int, default=1, help="TTA 采样次数（PSL 随机；默认 1 快）")
    ap.add_argument("--device", default="cpu", help="设备（Kaggle = cuda）")
    args = ap.parse_args(argv)

    out = args.out or f"outputs/preview/prob_vs_ivd_t{args.frame}.png"
    prob, ivd, label = run_preview(args.config, args.ckpt, args.frame, out,
                                   tta=args.tta, device=args.device)
    print(f"[preview] 帧 {args.frame}：场 {prob.shape}，概率域 "
          f"[{prob.min():.3f}, {prob.max():.3f}]，正格 {int((prob > 0.5).sum())}，图 → {out}")
    return 0


if __name__ == "__main__":
    main()
