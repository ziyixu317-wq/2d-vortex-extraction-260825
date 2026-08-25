"""固体几何掩膜（geometry.py）——02 票：固体几何掩膜。

领域词汇（HANDOFF §4，唯一权威）：
- 掩膜 = |v|<ε 逐帧取与（速度模近零）+ 连通域标记，随数据集 (T,Y,X) 存储；
  供迹线提取（种子排除/截断）与弱标签（IVD 置零）共用；
- 圆柱 = 不与壁相连的孤立连通块；无障碍物数据集输出空掩膜、代码路径不变；
- 每个新数据集各自跑一遍（逐数据集预处理，不进入模型输入）。

已核实事实（§2）：pipedcylinder2d.nc 每帧固定 28213 个近零速细胞（41.8%）= 静态固体几何。
实测（2026-08-25，本票）：ε∈[1e-6,1e-3] 时逐帧计数恒定 28213（非零速度最小 6.4e-5）；
全帧取与后 4 个连通块：两块壁面（接触域边界）+ 两个孤立圆柱（入口 (≈0,0) 与拐角后管道
(≈3,1)）。radius=0.0625 与 Re=160=U·D/ν（U=1, D=2r, ν=0.00078125）自洽；零速区为圆柱
内切区（表面约 1 格厚格被插值为非零速），物理半径估计 = 零速区面积等效半径 + 表面层 1 格。

实现约束：h5py 直读中文路径；不依赖 scipy（连通域标记为自写并查集，两遍扫描）。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np


# --------------------------------------------------------------------------- 静态掩膜

def speed_magnitude(u, v):
    """速度模 sqrt(u²+v²)，float64 精度（可视化与近零速判据共用）。"""
    return np.sqrt(np.asarray(u).astype(np.float64) ** 2
                   + np.asarray(v).astype(np.float64) ** 2)


def static_mask_from_speed(u, v, eps=1e-5):
    """|v|<ε 逐帧取与：速度模 sqrt(u²+v²) < eps 在所有帧成立的格 → 静态固体。

    返回 (Y, X) bool 掩膜（2D；时间不变，落盘时广播到 (T,Y,X)）。
    """
    u = np.asarray(u)
    v = np.asarray(v)
    if u.ndim != 3 or u.shape != v.shape:
        raise ValueError("u/v 需为 (T,Y,X) 同形状")
    solid = np.ones(u.shape[1:], dtype=bool)
    for t in range(u.shape[0]):
        solid &= speed_magnitude(u[t], v[t]) < float(eps)
    return solid


# --------------------------------------------------------------------------- 连通域标记

def label_components(mask, connectivity=8):
    """二值掩膜连通域标记（两遍扫描 + 并查集，纯 numpy/python，无 scipy 依赖）。

    返回 (labels, n)：labels 为 (Y,X) int32，背景 0，连通块编号 1..n。
    connectivity ∈ {4, 8}：8 邻接含对角（与探查用 scipy ones(3,3) 一致）。
    """
    mask = np.asarray(mask, dtype=bool)
    Y, X = mask.shape
    labels = np.zeros((Y, X), dtype=np.int32)
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    di_prev = (0, 1, -1) if connectivity == 8 else (0,)
    nxt = 1
    for j in range(Y):
        for i in np.nonzero(mask[j])[0]:
            nbr = 0
            if i > 0 and labels[j, i - 1] > 0:
                nbr = labels[j, i - 1]
            if j > 0:
                for di in di_prev:
                    ii = i + di
                    if 0 <= ii < X and labels[j - 1, ii] > 0:
                        lab = labels[j - 1, ii]
                        if nbr == 0:
                            nbr = lab
                        elif lab != nbr:
                            union(nbr, lab)
            if nbr == 0:
                labels[j, i] = nxt
                parent[nxt] = nxt
                nxt += 1
            else:
                labels[j, i] = find(nbr)

    # 重编号为 1..n（压缩路径后的根）
    final = {}
    out = np.zeros_like(labels)
    n = 0
    for j in range(Y):
        for i in np.nonzero(labels[j])[0]:
            root = find(labels[j, i])
            if root not in final:
                n += 1
                final[root] = n
            out[j, i] = final[root]
    return out, n


# --------------------------------------------------------------------------- 块统计与圆柱定位

def component_stats(mask, xdim, ydim, surface_layer_cells=1.0, connectivity=8):
    """连通域标记 + 每块统计（数据准备缝的连通性属性测试面）。

    返回 list[dict]，每块：
      id / cells / bbox([y0,y1,x0,x1]) / center_x / center_y（物理坐标，线性插值）/
      touches_border（是否接触域边界）/ r_eff（面积等效半径）/
      r_phys（r_eff + 表面层修正；圆柱表面约 1 格厚格被插值为非零速，见模块 docstring）。
    """
    xdim = np.asarray(xdim, dtype=np.float64)
    ydim = np.asarray(ydim, dtype=np.float64)
    labels, n = label_components(mask, connectivity=connectivity)
    dx = xdim[1] - xdim[0]
    dy = ydim[1] - ydim[0]
    d_avg = np.sqrt(dx * dy)
    Y, X = mask.shape
    stats = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(labels == k)
        cells = len(ys)
        r_eff = np.sqrt(cells / np.pi) * d_avg
        stats.append({
            "id": k,
            "cells": cells,
            "bbox": [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())],
            "center_x": float(np.interp(xs.mean(), np.arange(X), xdim)),
            "center_y": float(np.interp(ys.mean(), np.arange(Y), ydim)),
            "touches_border": bool(
                ys.min() == 0 or ys.max() == Y - 1
                or xs.min() == 0 or xs.max() == X - 1),
            "r_eff": float(r_eff),
            "r_phys": float(r_eff + surface_layer_cells * d_avg),
        })
    return stats


def locate_cylinders(stats, min_block_cells=1):
    """圆柱 = 不与壁相连的孤立连通块（HANDOFF §4，无尺寸判据）。

    min_block_cells 为显式收紧选项（默认 1 = 规格字面：任何孤立块都是圆柱），
    用于过滤极小数值噪声块时显式传入；该参数不是圆柱定义的组成部分。
    """
    return [c for c in stats
            if not c["touches_border"] and c["cells"] >= min_block_cells]


# --------------------------------------------------------------------------- 主入口

def build_geometry_mask(u, v, xdim, ydim, out_dir=None, eps=1e-5,
                        min_block_cells=1, surface_layer_cells=1.0):
    """从原始场生成固体几何掩膜：静态掩膜 → 连通域 → 圆柱定位。

    参数：
      u, v        (T,Y,X) float 速度场（h5py 直读结果，中文路径无碍）
      xdim, ydim  物理坐标（(X,) / (Y,)）
      out_dir     给定则落盘 mask.npy（(T,Y,X) uint8，每帧相同）+ geometry_meta.json
      eps         近零速判据（默认 1e-5：pipedcylinder2d 非零速度最小 6.4e-5，稳定分离）
      min_block_cells / surface_layer_cells  见 component_stats / locate_cylinders

    返回 meta dict（含 cylinders 列表；无障碍物时为空列表，代码路径不变）。
    """
    mask2d = static_mask_from_speed(u, v, eps=eps)
    stats = component_stats(mask2d, xdim, ydim, surface_layer_cells=surface_layer_cells)
    cylinders = locate_cylinders(stats, min_block_cells=min_block_cells)
    meta = {
        "source": "in-memory field (u,v,xdim,ydim)",
        "shape": [int(u.shape[0]), int(u.shape[1]), int(u.shape[2])],
        "eps": float(eps),
        "solid_cells": int(mask2d.sum()),
        "solid_fraction": float(mask2d.mean()),
        "n_components": len(stats),
        "components": stats,
        "cylinders": cylinders,
    }
    if out_dir is not None:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        T, Y, X = u.shape
        mask_tyx = np.broadcast_to(mask2d[None, :, :], (T, Y, X)).astype(np.uint8)
        np.save(out / "mask.npy", mask_tyx)
        (out / "geometry_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def load_field(nc_path):
    """h5py 直读 nc（中文路径可用；netCDF4 C 库打不开中文路径——HANDOFF §2）。"""
    import h5py
    with h5py.File(str(nc_path), "r") as f:
        u = f["u"][:]
        v = f["v"][:]
        xdim = f["xdim"][:].astype(np.float64)
        ydim = f["ydim"][:].astype(np.float64)
        tdim = f["tdim"][:]
    return u, v, xdim, ydim, tdim


# --------------------------------------------------------------------------- 可视化（验收 1：目检）

def plot_mask(u_t, mask2d, cylinders, xdim, ydim, out_png, vmax=1.5):
    """展示帧对比图：速度模底图 + 掩膜填充/轮廓 + 圆柱拟合圆（供人工目检）。

    坐标系统一为物理坐标：imshow/contour 用 extent 对齐 xdim/ydim，
    圆柱圆心/半径（物理坐标）直接叠加，不再错位。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    sp = speed_magnitude(u_t[0], u_t[1])
    extent = [xdim[0], xdim[-1], ydim[0], ydim[-1]]
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(sp, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=vmax, extent=extent)
    ax.contour(mask2d, levels=[0.5], colors="red", linewidths=1.0,
               extent=extent)
    for c in cylinders:
        ax.add_patch(Circle(
            (c["center_x"], c["center_y"]), c["r_phys"],
            fill=False, edgecolor="cyan", linewidth=1.5, linestyle="--"))
        ax.text(c["center_x"], c["center_y"],
                f"cyl{c['id']} r={c['r_phys']:.4f}",
                color="cyan", fontsize=8, ha="center")
    ax.set_title("speed magnitude + solid mask (red) + cylinder fit (cyan)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return str(out_png)


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="固体几何掩膜：|v|<ε 逐帧取与 + 连通域标记 → (T,Y,X) 掩膜 + 圆柱定位")
    ap.add_argument("nc_path", help="nc 数据集路径（h5py 直读，支持中文路径）")
    ap.add_argument("--out-dir", default="outputs/geometry",
                    help="掩膜输出目录（mask.npy + geometry_meta.json）")
    ap.add_argument("--eps", type=float, default=1e-5, help="近零速判据")
    ap.add_argument("--min-block-cells", type=int, default=1,
                    help="圆柱最小块格数（默认 1=规格字面；显式收紧时传入）")
    ap.add_argument("--visualize", default=None,
                    help="可选：输出目检图路径（如 outputs/geometry/mask_overview.png）")
    ap.add_argument("--frame", type=int, default=1000, help="目检图展示帧")
    args = ap.parse_args(argv)

    u, v, xdim, ydim, _ = load_field(args.nc_path)
    meta = build_geometry_mask(u, v, xdim, ydim, out_dir=args.out_dir,
                               eps=args.eps, min_block_cells=args.min_block_cells)
    print(f"固体格数: {meta['solid_cells']} ({meta['solid_fraction']:.2%})")
    print(f"连通块数: {meta['n_components']} | 圆柱数: {len(meta['cylinders'])}")
    for c in meta["cylinders"]:
        print(f"  圆柱 id={c['id']} 圆心=({c['center_x']:.4f},{c['center_y']:.4f}) "
              f"r_eff={c['r_eff']:.4f} r_phys={c['r_phys']:.4f} cells={c['cells']}")
    if args.visualize:
        mask2d = static_mask_from_speed(u, v, eps=args.eps)
        plot_mask((u[args.frame], v[args.frame]), mask2d, meta["cylinders"],
                  xdim, ydim, args.visualize)
        print(f"目检图已保存: {args.visualize}")


if __name__ == "__main__":
    main()
