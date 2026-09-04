"""Benchmark the unmodified NumbaCS 0.2.0 ``rotcohvrt`` baseline."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np

import weak_labels


UPSTREAM_VERSION = "0.2.0"
UPSTREAM_COMMIT = "c067f542543f5dd4ae3dc45fc506213e8d98b845"


def _load_frame(dataset_root: pathlib.Path, frame: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    metadata = json.loads((dataset_root / "meta.json").read_text(encoding="utf-8"))
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
        np.asarray(metadata["xdim"], dtype=np.float64),
        np.asarray(metadata["ydim"], dtype=np.float64),
        np.asarray(mask, dtype=bool),
    )


def _run_rotcohvrt(
    ivd: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
    *,
    r: float,
    convexity_deficiency: float,
    min_len: float,
    nlevs: int | None = None,
) -> tuple[list[Any], float]:
    # These are the only explicit arguments.  In particular, leave
    # min_val=-1, nlevs=20, start_level=0, and end_level=0 at the upstream
    # defaults so NumbaCS applies its own p80 and p70-to-maximum schedule.
    from numbacs.extraction.elliptic import rotcohvrt

    started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "convexity_deficiency": convexity_deficiency,
        "min_len": min_len,
    }
    if nlevs is not None:
        kwargs["nlevs"] = int(nlevs)
    result = rotcohvrt(
        np.asarray(ivd.T, dtype=np.float64),
        xdim,
        ydim,
        r,
        **kwargs,
    )
    return result, time.perf_counter() - started


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])))


def run(
    dataset_root: pathlib.Path,
    output_root: pathlib.Path,
    *,
    frame: int = 600,
    convexity_deficiency: float = 0.1,
    minimum_perimeter_factor: float = 8.0,
    nlevs: int | None = None,
) -> dict[str, Any]:
    u, v, xdim, ydim, solid = _load_frame(dataset_root, frame)
    dx = float(xdim[1] - xdim[0])
    dy = float(ydim[1] - ydim[0])
    omega = np.asarray(weak_labels.vorticity(u, v, xdim, ydim), dtype=np.float64)
    fluid = ~solid
    fluid_mean = float(omega[fluid].mean())
    ivd = np.abs(omega - fluid_mean)
    ivd[solid] = 0.0

    r = max(dx, dy)
    min_len = minimum_perimeter_factor * r
    p80 = float(np.percentile(ivd.T, 80.0))

    # Count the exact maxima used by the upstream default path.  This calls
    # the unmodified helper with the value rotcohvrt computes internally when
    # min_val=-1; it does not alter the rotcohvrt invocation below.
    from numbacs.utils import max_in_radius

    maxima_values, maxima_indices = max_in_radius(
        np.asarray(ivd.T, dtype=np.float64).copy(),
        r,
        dx,
        dy,
        min_val=p80,
    )

    first_result, cold_wall_seconds = _run_rotcohvrt(
        ivd,
        xdim,
        ydim,
        r=r,
        convexity_deficiency=convexity_deficiency,
        min_len=min_len,
        nlevs=nlevs,
    )
    second_result, warm_wall_seconds = _run_rotcohvrt(
        ivd,
        xdim,
        ydim,
        r=r,
        convexity_deficiency=convexity_deficiency,
        min_len=min_len,
        nlevs=nlevs,
    )
    if len(first_result) != len(second_result):
        raise RuntimeError("native NumbaCS frame 600 cold/warm vortex count mismatch")

    vortices = []
    for contour, center in second_result:
        points = np.asarray(contour, dtype=np.float64)
        center_array = np.asarray(center, dtype=np.float64)
        vortices.append({
            "center_xy": center_array.round(12).tolist(),
            "area": _polygon_area(points),
            "n_points": int(len(points)),
            "boundary_xy": points.round(12).tolist(),
        })

    output_root.mkdir(parents=True, exist_ok=True)
    level_suffix = "baseline" if nlevs is None else f"nlevs{int(nlevs)}"
    contour_path = output_root / (
        f"numbacs_native_{level_suffix}_frame{frame}_contours.json"
    )
    contour_path.write_text(
        json.dumps(vortices, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, Y = np.meshgrid(xdim, ydim)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 13),
        sharex=True,
        constrained_layout=True,
    )
    omega_plot = np.ma.array(omega, mask=solid)
    ivd_plot = np.ma.array(ivd, mask=solid)
    omega_limit = max(
        float(np.nanpercentile(np.abs(omega_plot.compressed()), 99.5)),
        np.finfo(float).eps,
    )
    omega_image = axes[0].pcolormesh(
        X,
        Y,
        omega_plot,
        shading="auto",
        cmap="RdBu_r",
        vmin=-omega_limit,
        vmax=omega_limit,
        rasterized=True,
    )
    ivd_image = axes[1].pcolormesh(
        X,
        Y,
        ivd_plot,
        shading="auto",
        cmap="magma",
        rasterized=True,
    )
    fig.colorbar(omega_image, ax=axes[0], label="vorticity $\\omega$")
    fig.colorbar(ivd_image, ax=axes[1], label="IVD")
    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_ylabel("y")
        for vortex in vortices:
            points = np.asarray(vortex["boundary_xy"], dtype=np.float64)
            axis.plot(
                points[:, 0],
                points[:, 1],
                color="#20e3c2",
                linewidth=0.65,
                alpha=0.8,
            )
        if vortices:
            centers = np.asarray([vortex["center_xy"] for vortex in vortices])
            axis.scatter(
                centers[:, 0],
                centers[:, 1],
                s=7,
                c="#ffffff",
                edgecolors="#111111",
                linewidths=0.25,
                alpha=0.9,
            )
    axes[0].set_title(
        f"Native NumbaCS {UPSTREAM_VERSION} rotcohvrt | frame {frame} | "
        f"{len(vortices)} vortices"
    )
    axes[1].set_title(
        f"IVD boundary | p80 maxima, p70→max, nlevs="
        f"{20 if nlevs is None else int(nlevs)} | "
        f"r={r:.6g}, convexity_deficiency={convexity_deficiency:g}, "
        f"min_len={min_len:.6g} | warm {warm_wall_seconds:.2f}s"
    )
    axes[1].set_xlabel("x")
    image_path = output_root / f"numbacs_native_{level_suffix}_frame{frame}.png"
    fig.savefig(image_path, dpi=180, facecolor="white")
    plt.close(fig)

    summary = {
        "backend": "numbacs_native_upstream",
        "upstream_version": UPSTREAM_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source": "/data/xuziyi/tmp/numbacs-0.2.0/src/numbacs/extraction/elliptic.py",
        "nlevs": 20 if nlevs is None else int(nlevs),
        "nlevs_passed_explicitly": nlevs is not None,
        "frame": int(frame),
        "shape": [int(value) for value in ivd.shape],
        "fluid_vorticity_mean": fluid_mean,
        "ivd_formula": "abs(omega - mean_fluid(omega))",
        "p80_used_in_upstream_default": p80,
        "maxima_count": int(len(maxima_values)),
        "maxima_index_shape": [int(value) for value in maxima_indices.shape],
        "vortex_count": int(len(vortices)),
        "cold_wall_seconds": float(cold_wall_seconds),
        "warm_wall_seconds": float(warm_wall_seconds),
        "r": r,
        "convexity_deficiency": float(convexity_deficiency),
        "min_len": min_len,
        "omitted_rotcohvrt_arguments": {
            "min_val": -1.0,
            "start_level": 0.0,
            "end_level": 0.0,
        },
        "image_path": str(image_path.resolve()),
        "contours_path": str(contour_path.resolve()),
    }
    summary_path = output_root / f"numbacs_native_{level_suffix}_frame{frame}.json"
    summary["summary_path"] = str(summary_path.resolve())
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="outputs/datasets/boussinesq/dataset",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frame", type=int, default=600)
    parser.add_argument("--convexity-deficiency", type=float, default=0.1)
    parser.add_argument("--minimum-perimeter-factor", type=float, default=8.0)
    parser.add_argument(
        "--nlevs",
        type=int,
        choices=(32, 64),
        default=None,
        help="explicit sensitivity value; omit for the native default nlevs=20",
    )
    args = parser.parse_args(argv)
    summary = run(
        pathlib.Path(args.dataset_root),
        pathlib.Path(args.output_root),
        frame=args.frame,
        convexity_deficiency=args.convexity_deficiency,
        minimum_perimeter_factor=args.minimum_perimeter_factor,
        nlevs=args.nlevs,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
