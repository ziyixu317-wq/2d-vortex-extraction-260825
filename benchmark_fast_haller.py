"""Frame-600 benchmark for the local ContourPy ``fast_haller`` backend."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any, Mapping

import numpy as np

import haller_anchors


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _polygon_centroid(points_xy: Any) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        return np.full((2,), np.nan, dtype=np.float64)
    closed = points if np.allclose(points[0], points[-1]) else np.vstack([points, points[0]])
    cross = (
        closed[:-1, 0] * closed[1:, 1]
        - closed[1:, 0] * closed[:-1, 1]
    )
    area_twice = float(cross.sum())
    if abs(area_twice) <= np.finfo(np.float64).eps:
        return points.mean(axis=0)
    return np.asarray([
        float(((closed[:-1, 0] + closed[1:, 0]) * cross).sum() / (3.0 * area_twice)),
        float(((closed[:-1, 1] + closed[1:, 1]) * cross).sum() / (3.0 * area_twice)),
    ])


def _hausdorff(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    from scipy.spatial.distance import directed_hausdorff
    return float(max(
        directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]
    ))


def _selection_by_peak(selection: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (
            int(item["peak"]["row"]),
            int(item["peak"]["col"]),
        ): item
        for item in selection
    }


def compare_final_boundaries(
    strict_selection: list[dict[str, Any]],
    fast_selection: list[dict[str, Any]],
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Compare only final boundaries, not intermediate candidate paths."""
    strict_by_peak = _selection_by_peak(strict_selection)
    fast_by_peak = _selection_by_peak(fast_selection)
    common = sorted(set(strict_by_peak) & set(fast_by_peak))
    centers: list[float] = []
    area_abs: list[float] = []
    area_rel: list[float] = []
    ious: list[float] = []
    hausdorffs: list[float] = []
    for key in common:
        expected = strict_by_peak[key]
        actual = fast_by_peak[key]
        expected_center = _polygon_centroid(expected["points_xy"])
        actual_center = _polygon_centroid(actual["points_xy"])
        centers.append(float(np.linalg.norm(expected_center - actual_center)))
        expected_area = float(expected["area"])
        actual_area = float(actual["area"])
        delta = abs(actual_area - expected_area)
        area_abs.append(delta)
        area_rel.append(delta / max(abs(expected_area), np.finfo(np.float64).eps))
        expected_mask = haller_anchors.grid_points_in_poly(
            shape, np.asarray(expected["points_grid"], dtype=np.float64)
        )
        actual_mask = haller_anchors.grid_points_in_poly(
            shape, np.asarray(actual["points_grid"], dtype=np.float64)
        )
        union = int(np.count_nonzero(expected_mask | actual_mask))
        intersection = int(np.count_nonzero(expected_mask & actual_mask))
        ious.append(float(intersection / union) if union else 1.0)
        hausdorffs.append(_hausdorff(expected["points_xy"], actual["points_xy"]))

    def summary(values: list[float], *, empty: float = float("nan")) -> dict[str, float]:
        if not values:
            return {"mean": empty, "max": empty, "min": empty}
        return {
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
        }

    return {
        "strict_vortex_count": len(strict_selection),
        "fast_vortex_count": len(fast_selection),
        "vortex_count_delta": len(fast_selection) - len(strict_selection),
        "matched_vortex_count": len(common),
        "strict_only_count": len(set(strict_by_peak) - set(fast_by_peak)),
        "fast_only_count": len(set(fast_by_peak) - set(strict_by_peak)),
        "center_error": summary(centers),
        "area_abs_error": summary(area_abs),
        "area_relative_error": summary(area_rel),
        "boundary_iou": summary(ious),
        "boundary_hausdorff": summary(hausdorffs),
    }


def _load_frame(
    dataset_root: pathlib.Path,
    frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta = json.loads((dataset_root / "meta.json").read_text(encoding="utf-8"))
    u = np.load(dataset_root / "u.npy", mmap_mode="r", allow_pickle=False)[frame]
    v = np.load(dataset_root / "v.npy", mmap_mode="r", allow_pickle=False)[frame]
    mask = np.load(dataset_root / "mask.npy", mmap_mode="r", allow_pickle=False)
    xdim = np.asarray(meta["xdim"], dtype=np.float64)
    ydim = np.asarray(meta["ydim"], dtype=np.float64)
    if mask.ndim == 3:
        mask = mask[frame]
    return (
        np.asarray(u, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        xdim,
        ydim,
        np.asarray(mask, dtype=bool),
    )


def benchmark_frame(
    dataset_root: pathlib.Path,
    *,
    frame: int = 600,
    global_level_counts: tuple[int, ...] = (32, 64, 96),
    reference_summary: Mapping[str, Any] | None = None,
    compute_reference: bool = True,
) -> dict[str, Any]:
    """Run one strict reference and the requested fast level-count variants."""
    u, v, xdim, ydim, mask = _load_frame(dataset_root, frame)
    prep_start = time.perf_counter()
    details = haller_anchors._compute_standard_ivd_details(u, v, xdim, ydim, mask)
    peaks = haller_anchors.find_local_maxima(details.ivd, details.solid_mask)
    params = haller_anchors._resolve_parameters()
    prep_seconds = time.perf_counter() - prep_start

    if reference_summary is None and compute_reference:
        print(
            f"strict_reference_started frame={frame} peaks={len(peaks)}",
            flush=True,
        )
        reference_start = time.perf_counter()
        strict_selection, _, strict_evaluation = haller_anchors._select_contours(
            details.ivd,
            details.solid_mask,
            xdim,
            ydim,
            peaks,
            params,
            details.dx,
            details.dy,
            backend=haller_anchors.BACKEND_NUMPY,
            contour_mode=haller_anchors.CONTOUR_MODE_REFERENCE,
            return_evaluation=True,
        )
        reference_summary = {
            "frame": frame,
            "vortex_count": len(strict_selection),
            "selection_wall_time_seconds": time.perf_counter() - reference_start,
            "contour_evaluation": strict_evaluation,
            "selection": strict_selection,
        }
        print(
            "strict_reference_finished "
            f"seconds={reference_summary['selection_wall_time_seconds']:.6f} "
            f"vortex_count={reference_summary['vortex_count']}",
            flush=True,
        )
    elif reference_summary is not None:
        strict_selection = list(reference_summary.get("selection", []))
    else:
        strict_selection = None

    results: list[dict[str, Any]] = []
    for level_count in global_level_counts:
        print(f"fast_variant_started global_level_count={level_count}", flush=True)
        started = time.perf_counter()
        fast_selection, _, fast_evaluation = haller_anchors._select_contours(
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
            fast_global_level_count=int(level_count),
            fast_refinement_iterations=7,
            fast_refinement_halo_cells=2,
            return_evaluation=True,
        )
        wall_seconds = time.perf_counter() - started
        comparison = (
            compare_final_boundaries(
                strict_selection, fast_selection, tuple(details.ivd.shape)
            )
            if strict_selection is not None
            else {"comparison": "strict_reference_not_run"}
        )
        results.append({
            "global_level_count": int(level_count),
            "wall_time_seconds": wall_seconds,
            "vortex_count": len(fast_selection),
            "global_contour_calls": fast_evaluation["global_contour_calls"],
            "refinement_calls": fast_evaluation["refinement_calls"],
            "contour_generator_count": fast_evaluation["contour_generator_count"],
            "fast_evaluation": fast_evaluation,
            **comparison,
        })
        print(
            f"fast_variant_finished global_level_count={level_count} "
            f"seconds={wall_seconds:.6f} vortex_count={len(fast_selection)}",
            flush=True,
        )
    return {
        "dataset_root": str(dataset_root.resolve()),
        "frame": int(frame),
        "shape": list(details.ivd.shape),
        "peaks": len(peaks),
        "ivd_preparation_seconds": prep_seconds,
        "strict_reference": (
            reference_summary
            if reference_summary is not None
            else {"skipped": True}
        ),
        "variants": results,
        "backend": haller_anchors.resolve_haller_backend(
            haller_anchors.BACKEND_FAST_HALLER, "cpu"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="outputs/datasets/boussinesq/dataset")
    parser.add_argument("--frame", type=int, default=600)
    parser.add_argument("--levels", nargs="+", type=int, default=[32, 64, 96])
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--reference-summary",
        default=None,
        help="可选已保存的 strict reference JSON；否则本次计算一次 oracle",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="只测 fast_haller，用于先确认吞吐；不生成误差比较",
    )
    args = parser.parse_args(argv)
    reference_summary = None
    if args.reference_summary:
        reference_summary = json.loads(
            pathlib.Path(args.reference_summary).read_text(encoding="utf-8")
        )["strict_reference"]
    report = benchmark_frame(
        pathlib.Path(args.dataset_root),
        frame=args.frame,
        global_level_counts=tuple(args.levels),
        reference_summary=reference_summary,
        compute_reference=not args.skip_reference,
    )
    output = pathlib.Path(args.output) if args.output else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(_jsonable(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(_jsonable(report), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
