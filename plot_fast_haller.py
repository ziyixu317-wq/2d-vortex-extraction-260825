"""Render one CPU fast_haller frame for visual inspection."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

import haller_anchors


def _load_frame(dataset_root: pathlib.Path, frame: int):
    meta = json.loads((dataset_root / "meta.json").read_text(encoding="utf-8"))
    u = np.asarray(
        np.load(dataset_root / "u.npy", mmap_mode="r", allow_pickle=False)[frame],
        dtype=np.float64,
    )
    v = np.asarray(
        np.load(dataset_root / "v.npy", mmap_mode="r", allow_pickle=False)[frame],
        dtype=np.float64,
    )
    mask = np.load(dataset_root / "mask.npy", mmap_mode="r", allow_pickle=False)
    if mask.ndim == 3:
        mask = mask[frame]
    return (
        u,
        v,
        np.asarray(meta["xdim"], dtype=np.float64),
        np.asarray(meta["ydim"], dtype=np.float64),
        np.asarray(mask, dtype=bool),
    )


def render_frame(
    dataset_root: pathlib.Path,
    output: pathlib.Path,
    *,
    frame: int = 600,
    global_level_count: int = 32,
) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u, v, xdim, ydim, mask = _load_frame(dataset_root, frame)
    details = haller_anchors._compute_standard_ivd_details(u, v, xdim, ydim, mask)
    peaks = haller_anchors.find_local_maxima(details.ivd, details.solid_mask)
    params = haller_anchors._resolve_parameters()
    started = time.perf_counter()
    contours, _, evaluation = haller_anchors._select_contours(
        details.ivd,
        details.solid_mask,
        xdim,
        ydim,
        peaks,
        params,
        details.dx,
        details.dy,
        backend=haller_anchors.BACKEND_FAST_HALLER,
        contour_mode=haller_anchors.CONTOUR_MODE_OPTIMIZED,
        fast_global_level_count=int(global_level_count),
        fast_refinement_iterations=7,
        fast_refinement_halo_cells=2,
        return_evaluation=True,
    )
    wall_seconds = time.perf_counter() - started

    X, Y = np.meshgrid(xdim, ydim)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7, 14),
        sharex=True,
        constrained_layout=True,
    )
    omega = np.ma.array(details.omega, mask=details.solid_mask)
    ivd = np.ma.array(details.ivd, mask=details.solid_mask)
    omega_limit = float(np.nanpercentile(np.abs(omega.compressed()), 99.5))
    omega_limit = max(omega_limit, np.finfo(float).eps)
    omega_image = axes[0].pcolormesh(
        X,
        Y,
        omega,
        shading="auto",
        cmap="RdBu_r",
        vmin=-omega_limit,
        vmax=omega_limit,
        rasterized=True,
    )
    ivd_image = axes[1].pcolormesh(
        X,
        Y,
        ivd,
        shading="auto",
        cmap="magma",
        rasterized=True,
    )
    fig.colorbar(omega_image, ax=axes[0], label="vorticity $\\omega$")
    fig.colorbar(ivd_image, ax=axes[1], label="standard IVD")

    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_ylabel("y")
        axis.grid(False)
    axes[1].set_xlabel("x")
    axes[0].set_title(f"Frame {frame}: vorticity with fast_haller boundaries")
    axes[1].set_title(
        f"IVD + {len(contours)} selected boundaries | geomspace {global_level_count} levels | "
        f"{wall_seconds:.1f} s"
    )

    for axis in axes:
        for item in contours:
            points = np.asarray(item["points_xy"], dtype=np.float64)
            if points.ndim == 2 and points.shape[1] == 2 and len(points) > 1:
                axis.plot(
                    points[:, 0],
                    points[:, 1],
                    color="#20e3c2",
                    linewidth=0.45,
                    alpha=0.68,
                )
        axis.scatter(
            [xdim[int(peak["col"])] for peak in peaks],
            [ydim[int(peak["row"])] for peak in peaks],
            s=2.0,
            c="#f7f7f7",
            alpha=0.18,
            linewidths=0,
        )
    axes[1].plot([], [], color="#20e3c2", linewidth=1.2, label="selected contour")
    axes[1].scatter([], [], s=12, color="#f7f7f7", label="all local maxima")
    axes[1].legend(loc="upper right", framealpha=0.75)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)
    return {
        "frame": int(frame),
        "shape": [int(value) for value in details.ivd.shape],
        "peaks": int(len(peaks)),
        "vortex_count": int(len(contours)),
        "global_level_count": int(global_level_count),
        "global_contour_calls": int(evaluation["global_contour_calls"]),
        "refinement_calls": int(evaluation["refinement_calls"]),
        "wall_time_seconds": float(wall_seconds),
        "output": str(output.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="outputs/datasets/boussinesq/dataset")
    parser.add_argument("--frame", type=int, default=600)
    parser.add_argument("--levels", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary = render_frame(
        pathlib.Path(args.dataset_root),
        pathlib.Path(args.output),
        frame=args.frame,
        global_level_count=args.levels,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
