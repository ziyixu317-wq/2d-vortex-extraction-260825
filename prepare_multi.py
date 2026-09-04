"""多数据集逐数据集预处理驱动（prepare_multi.py）——票 07 延伸（需求 B 第 1 步）。

领域词汇（HANDOFF §1 决策 8 / §2 / 票 07 延伸，唯一权威）：
- geometry 掩膜**逐数据集预处理**（决策 8：不进入模型输入；每个数据集各自跑，
  掩膜随数据集 (T,Y,X) 存储）→ IVD+label → τ 逐时间片 → memmap+meta.json；
  复用票 02（geometry CLI）/票 05（prepare_dataset CLI）的既有管线；
- 旧入口默认保留 `frac` 60/40 口径以复用阶段 0 产物；新弱监督入口必须显式
  使用 `weak_supervision`，按各数据集 frame index 生成 0/50/60/100 三段；
- τ = 各数据集逐时间片同分位（默认 85——需求 A 定案；逐数据集各自统计，
  防跨数据集 IVD 量纲差）；
- 归一化统计逐数据集（ivd μ/σ 取各数据集 train 片流体区、speed_max 逐数据集——
  prepare_dataset 内建口径，此处不覆盖）。

用法（本地 CPU，中文路径 h5py 直读）：
    python prepare_multi.py [--nc-dir ../CFD数据集] [--out-root outputs/datasets]
                            [--names a,b,...] [--split-mode weak_supervision]
                            [--label-source legacy_p85] [--frames 400,1200,1300]
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

import dataset as ds
import geometry
import weak_labels

# 单数据集统计的默认展示帧（各数据集按自身帧数截断）
DEFAULT_DISPLAY_FRAMES = (400, 1200, 1300)


def prepare_one(nc_path, out_root, *, percentile=weak_labels.DEFAULT_PERCENTILE,
                min_area=weak_labels.DEFAULT_MIN_AREA, train_frac=0.6, val_frac=0.0,
                split_mode="frac", label_source=None, sampling_source=None,
                loss_label_source=None, t_win=ds.DEFAULT_T_WIN,
                window_step=ds.DEFAULT_WINDOW_STEP,
                display_frames=DEFAULT_DISPLAY_FRAMES, eps=1e-5):
    """单个数据集：geometry 掩膜 → prepare_dataset（IVD/τ/label/memmap/meta）→ 目检图。

    返回 summary dict（写 multi_meta.json 用）。
    """
    nc_path = pathlib.Path(nc_path)
    name = nc_path.stem
    if split_mode == ds.WEAK_SUPERVISION_SPLIT_MODE \
            and name not in ds.VALID_WEAK_DATASETS:
        raise ValueError(
            f"weak_supervision 只允许六个有效数据集 {list(ds.VALID_WEAK_DATASETS)}，"
            f"不允许 {name!r}"
        )
    out_root = pathlib.Path(out_root)
    geo_dir = out_root / name / "geometry"
    ds_dir = out_root / name / "dataset"
    prev_dir = out_root / name / "previews"
    geo_dir.mkdir(parents=True, exist_ok=True)
    prev_dir.mkdir(parents=True, exist_ok=True)

    print(f"== {name}: geometry 掩膜 ==", flush=True)
    u, v, xdim, ydim, tdim = geometry.load_field(str(nc_path))
    geo_meta = geometry.build_geometry_mask(u, v, xdim, ydim, out_dir=str(geo_dir),
                                            eps=eps, min_block_cells=1)
    mask2d = np.asarray(np.load(geo_dir / "mask.npy"), dtype=bool)
    mask2d = mask2d[0] if mask2d.ndim == 3 else mask2d
    T = len(tdim)
    print(f"  固体格 {geo_meta['solid_cells']} ({geo_meta['solid_fraction']:.3%})，"
          f"块 {geo_meta['n_components']}，圆柱 {len(geo_meta['cylinders'])}",
          flush=True)

    split_desc = (f"weak_supervision 0/50/60/100"
                  if split_mode == ds.WEAK_SUPERVISION_SPLIT_MODE
                  else f"{split_mode} {train_frac:.0%}/{1 - train_frac - val_frac:.0%}")
    print(f"== {name}: prepare_dataset（p{percentile:g} 分位 τ、{split_desc}）==",
          flush=True)
    meta = ds.prepare_dataset(str(nc_path), str(ds_dir), mask=mask2d,
                              percentile=percentile, min_area=min_area,
                              split_mode=split_mode, train_frac=train_frac,
                              val_frac=val_frac, label_source=label_source,
                              sampling_source=sampling_source,
                              loss_label_source=loss_label_source,
                              t_win=t_win, window_step=window_step)
    print(f"  slices={meta['slices']} taus={ {k: round(v, 4) for k, v in meta['taus'].items()} }",
          flush=True)

    print(f"== {name}: 目检图 ==", flush=True)
    ivd_mm = np.load(ds_dir / ds.FN_IVD, mmap_mode="r")
    lab_mm = np.load(ds_dir / ds.FN_LABEL, mmap_mode="r")
    xd = np.asarray(meta["xdim"], dtype=np.float64)
    yd = np.asarray(meta["ydim"], dtype=np.float64)
    for t0 in display_frames:
        t0 = int(min(max(t0, 0), meta["shape"][0] - 1))
        tau = float(meta["taus"].get(_slice_name(meta["slices"], t0),
                                     next(iter(meta["taus"].values()))))
        weak_labels.plot_ivd_q(
            np.asarray(ivd_mm[t0]), None, np.asarray(lab_mm[t0]), mask2d,
            xd, yd, prev_dir / f"ivd_q_t{t0}.png", tau=tau,
            title=f"{name} t-f={t0} IVD + label>=tau (lime)")
    del ivd_mm, lab_mm

    lab_all = np.asarray(np.load(ds_dir / ds.FN_LABEL, mmap_mode="r"))
    lab_frac = float(lab_all[:, ~mask2d].mean())     # 流体区正格占比（固体恒 0，不影响）
    del lab_all
    return {
        "name": name,
        "nc": str(nc_path),
        "shape": meta["shape"],
        "dt": meta["dt"],
        "slices": meta["slices"],
        "split_mode": meta["split_mode"],
        "window": meta["window"],
        "feature_schema": meta["feature_schema"],
        "label_source": meta["label_source"],
        "label_provenance": meta["label_provenance"],
        "normalization_source": meta["normalization_source"],
        "normalization_frozen": meta["normalization_frozen"],
        "generation_version": meta["generation_version"],
        "generation_hash": meta["generation_hash"],
        "contract_hash": meta["contract_hash"],
        "percentile": meta["percentile"],
        "taus": meta["taus"],
        "speed_max": meta["speed_max"],
        "ivd_mu": meta["ivd_mu"],
        "ivd_sigma": meta["ivd_sigma"],
        "label_positive_fraction": lab_frac,
        "solid_cells": int(geo_meta["solid_cells"]),
        "n_components": int(geo_meta["n_components"]),
        "cylinders": geo_meta["cylinders"],
        "out_dir": str(ds_dir),
    }


def _slice_name(slices, frame):
    """帧 → 时间片名（uncovered → None）。"""
    for name, (i0, i1) in slices.items():
        if i0 <= frame < i1:
            return name
    return None


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="多数据集逐数据集预处理：geometry 掩膜 → IVD/label/τ → "
                    "memmap+meta.json（旧 frac 兼容；weak_supervision 为六数据集新契约）")
    ap.add_argument("--nc-dir", default="../CFD数据集",
                    help="nc 数据集目录（h5py 直读中英文路径；默认 ../CFD数据集）")
    ap.add_argument("--out-root", default="outputs/datasets",
                    help="输出根目录（<名>/geometry + <名>/dataset + <名>/previews）")
    ap.add_argument("--names", default=None,
                    help="可选子集（逗号分隔的 nc 文件名，含后缀；默认全部 .nc）")
    ap.add_argument("--percentile", type=float,
                    default=weak_labels.DEFAULT_PERCENTILE,
                    help="τ 分位（默认 85——票 07 延伸；HANDOFF §6）")
    ap.add_argument("--split-mode", choices=("frac", ds.WEAK_SUPERVISION_SPLIT_MODE),
                    default="frac",
                    help="旧兼容默认 frac 60/40；新 feature 必须显式选 "
                         "weak_supervision（0/50/60/100）")
    ap.add_argument("--label-source", default=None,
                    help="弱监督 label source（缺省时新 split 会 fail loudly）")
    ap.add_argument("--sampling-source", default=None,
                    help="独立记录的 sampling source（例如 legacy_p85）")
    ap.add_argument("--loss-label-source", default=None,
                    help="formal loss label source；不能隐式回退 p85")
    ap.add_argument("--train-frac", type=float, default=0.6,
                    help="训练帧比例（frac 划分口径）")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="val 帧比例（默认 0=无 val 片）")
    ap.add_argument("--frames", default="400,1200,1300",
                    help="目检图展示帧（逗号分隔；按数据集帧数截断）")
    ap.add_argument("--t-win", type=int, default=ds.DEFAULT_T_WIN,
                    help="pathline window 帧数（weak split 每段都必须容纳）")
    ap.add_argument("--window-step", type=int, default=ds.DEFAULT_WINDOW_STEP,
                    help="pathline window 起点步长")
    args = ap.parse_args(argv)

    nc_dir = pathlib.Path(args.nc_dir)
    names = [n.strip() for n in (args.names or "").split(",") if n.strip()]
    ncs = sorted(nc_dir.glob("*.nc")) if not names else \
        [nc_dir / n for n in names if (nc_dir / n).exists()]
    if not ncs:
        raise FileNotFoundError(f"--nc-dir {nc_dir} 下无 .nc 数据集"
                                + ("且 --names 指定文件不存在" if names else ""))
    if args.split_mode == ds.WEAK_SUPERVISION_SPLIT_MODE:
        invalid = sorted({nc.stem for nc in ncs
                          if nc.stem not in ds.VALID_WEAK_DATASETS})
        if invalid:
            raise ValueError(
                f"weak_supervision 只允许六个有效数据集 {list(ds.VALID_WEAK_DATASETS)}，"
                f"发现不在实验池中的数据集 {invalid}"
            )
    frames = tuple(int(s) for s in args.frames.split(",") if s.strip())

    summaries = []
    for nc in ncs:
        summaries.append(prepare_one(
            nc, args.out_root, percentile=args.percentile,
            split_mode=args.split_mode, label_source=args.label_source,
            sampling_source=args.sampling_source,
            loss_label_source=args.loss_label_source,
            train_frac=args.train_frac, val_frac=args.val_frac,
            t_win=args.t_win, window_step=args.window_step,
            display_frames=frames))
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "multi_meta.json").write_text(
        json.dumps({"datasets": summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"完成: {len(summaries)} 个数据集 → {out_root}/multi_meta.json", flush=True)
    return 0


if __name__ == "__main__":
    main()
