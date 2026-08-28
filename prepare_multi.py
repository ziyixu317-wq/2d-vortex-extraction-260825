"""多数据集逐数据集预处理驱动（prepare_multi.py）——票 07 延伸（需求 B 第 1 步）。

领域词汇（HANDOFF §1 决策 8 / §2 / 票 07 延伸，唯一权威）：
- geometry 掩膜**逐数据集预处理**（决策 8：不进入模型输入；每个数据集各自跑，
  掩膜随数据集 (T,Y,X) 存储）→ IVD+label → τ 逐时间片 → memmap+meta.json；
  复用票 02（geometry CLI）/票 05（prepare_dataset CLI）的既有管线；
- 时间划分 = 按帧比例（dataset.fraction_slices 的 frac 口径：默认前 60% 训 /
  后 40% 测、无 val——票 07 延伸用户定案；多数据集帧数/时长各异，绝对秒数
  划分（10/12.5/15s）不通用）；
- τ = 各数据集逐时间片同分位（默认 85——需求 A 定案；逐数据集各自统计，
  防跨数据集 IVD 量纲差）；
- 归一化统计逐数据集（ivd μ/σ 取各数据集 train 片流体区、speed_max 逐数据集——
  prepare_dataset 内建口径，此处不覆盖）。

用法（本地 CPU，中文路径 h5py 直读）：
    python prepare_multi.py [--nc-dir ../CFD数据集] [--out-root outputs/datasets]
                            [--names a,b,...] [--percentile 85] [--train-frac 0.6]
                            [--frames 400,1200,1300]
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
                display_frames=DEFAULT_DISPLAY_FRAMES, eps=1e-5):
    """单个数据集：geometry 掩膜 → prepare_dataset（IVD/τ/label/memmap/meta）→ 目检图。

    返回 summary dict（写 multi_meta.json 用）。
    """
    nc_path = pathlib.Path(nc_path)
    name = nc_path.stem
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

    print(f"== {name}: prepare_dataset（p{percentile:g} 分位 τ、frac "
          f"{train_frac:.0%}/{1 - train_frac - val_frac:.0%}）==", flush=True)
    meta = ds.prepare_dataset(str(nc_path), str(ds_dir), mask=mask2d,
                              percentile=percentile, min_area=min_area,
                              split_mode="frac", train_frac=train_frac,
                              val_frac=val_frac)
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
                    "memmap+meta.json（票 02/05 CLI 复用；frac 60/40、τ 逐数据集）")
    ap.add_argument("--nc-dir", default="../CFD数据集",
                    help="nc 数据集目录（h5py 直读中英文路径；默认 ../CFD数据集）")
    ap.add_argument("--out-root", default="outputs/datasets",
                    help="输出根目录（<名>/geometry + <名>/dataset + <名>/previews）")
    ap.add_argument("--names", default=None,
                    help="可选子集（逗号分隔的 nc 文件名，含后缀；默认全部 .nc）")
    ap.add_argument("--percentile", type=float,
                    default=weak_labels.DEFAULT_PERCENTILE,
                    help="τ 分位（默认 85——票 07 延伸；HANDOFF §6）")
    ap.add_argument("--train-frac", type=float, default=0.6,
                    help="训练帧比例（frac 划分口径）")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="val 帧比例（默认 0=无 val 片）")
    ap.add_argument("--frames", default="400,1200,1300",
                    help="目检图展示帧（逗号分隔；按数据集帧数截断）")
    args = ap.parse_args(argv)

    nc_dir = pathlib.Path(args.nc_dir)
    names = [n.strip() for n in (args.names or "").split(",") if n.strip()]
    ncs = sorted(nc_dir.glob("*.nc")) if not names else \
        [nc_dir / n for n in names if (nc_dir / n).exists()]
    if not ncs:
        raise FileNotFoundError(f"--nc-dir {nc_dir} 下无 .nc 数据集"
                                + ("且 --names 指定文件不存在" if names else ""))
    frames = tuple(int(s) for s in args.frames.split(",") if s.strip())

    summaries = []
    for nc in ncs:
        summaries.append(prepare_one(
            nc, args.out_root, percentile=args.percentile,
            train_frac=args.train_frac, val_frac=args.val_frac,
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
