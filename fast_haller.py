"""ContourPy-based fast Haller contour search.

This module is a local backend inspired by ``numbacs.extraction.elliptic``.
It does not import, patch, or otherwise modify the upstream package.  The
important structural difference from the strict Haller oracle is that one
ContourPy generator is built for the field and shared global levels are
requested in ascending order.  A coarse boundary is then refined locally
with the unchanged Haller geometry predicates.

The backend deliberately keeps the engineering Haller predicates in
``haller_anchors``: a candidate must be closed, contain the requested local
maximum, avoid solid, have positive area, satisfy convexity deficiency, and
meet the minimum perimeter.  ConvexHull is therefore used only by
``haller_anchors.contour_metrics`` to measure deficiency; the selected path
is always the original IVD contour returned by ContourPy.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np

import haller_anchors as strict


BACKEND_FAST_HALLER = "fast_haller"
BACKEND_VERSION = "fast_haller-contourpy-v1"
DEFAULT_GLOBAL_LEVEL_COUNT = 64
DEFAULT_REFINEMENT_ITERATIONS = 7
DEFAULT_REFINEMENT_HALO_CELLS = 2


def pts_in_poly_mask(points: Any, contour: Any) -> np.ndarray:
    """Return one containment boolean per point, including 0/1/>1 cases.

    NumbaCS's ``pts_in_poly`` helper returns point memberships.  This wrapper
    makes the cardinality contract explicit so a contour containing multiple
    maxima is not silently assigned to the first point in an array.
    """
    point_array = np.asarray(points, dtype=np.float64)
    contour_array = np.asarray(contour, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1:] != (2,):
        raise ValueError("points 必须是形状 (N,2) 的坐标数组")
    if contour_array.ndim != 2 or contour_array.shape[1:] != (2,):
        raise ValueError("contour 必须是形状 (M,2) 的坐标数组")
    if len(point_array) == 0:
        return np.empty((0,), dtype=bool)
    if len(contour_array) < 3:
        return np.zeros((len(point_array),), dtype=bool)
    try:
        from skimage.measure import points_in_poly
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "fast_haller requires scikit-image for the point-in-polygon predicate"
        ) from exc
    memberships = np.asarray(points_in_poly(point_array, contour_array), dtype=bool)
    if memberships.shape != (len(point_array),):
        raise RuntimeError(
            "points_in_poly 返回的 membership 形状不满足 (N,) cardinality contract"
        )
    return memberships


def _make_contour_generator(
    ivd: np.ndarray,
    fluid: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
) -> Any:
    """Create a serial ContourPy generator for the full field or one ROI."""
    try:
        from contourpy import contour_generator
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ModuleNotFoundError(
            "fast_haller requires contourpy; install it in the active CPU environment"
        ) from exc

    if roi is None:
        row0, row1, col0, col1 = 0, ivd.shape[0], 0, ivd.shape[1]
    else:
        row0, row1, col0, col1 = (int(value) for value in roi)
    values = np.asarray(ivd[row0:row1, col0:col1], dtype=np.float64)
    fluid_crop = np.asarray(fluid[row0:row1, col0:col1], dtype=bool)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError(f"ContourPy ROI 至少需要 2x2 网格，实际 {values.shape}")
    masked_values = np.ma.array(values, mask=~fluid_crop, copy=False)
    # ``z`` uses the repository's (row, col) convention.  ContourPy returns
    # physical (x, y) coordinates, which are converted back below.
    return contour_generator(
        x=np.asarray(xdim[col0:col1], dtype=np.float64),
        y=np.asarray(ydim[row0:row1], dtype=np.float64),
        z=masked_values,
        name="serial",
        thread_count=1,
    )


def _paths_to_grid(
    paths: Any,
    xdim: np.ndarray,
    ydim: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, ...]:
    """Convert ContourPy physical (x,y) paths to global (row,col) paths."""
    if roi is None:
        row0, row1, col0, col1 = 0, len(ydim), 0, len(xdim)
    else:
        row0, row1, col0, col1 = (int(value) for value in roi)
    local_x = np.asarray(xdim[col0:col1], dtype=np.float64)
    local_y = np.asarray(ydim[row0:row1], dtype=np.float64)
    grid_rows = np.arange(len(local_y), dtype=np.float64) + row0
    grid_cols = np.arange(len(local_x), dtype=np.float64) + col0
    converted: list[np.ndarray] = []
    for raw_path in paths:
        physical = np.asarray(raw_path, dtype=np.float64)
        if physical.ndim != 2 or physical.shape[1] != 2:
            continue
        converted.append(np.column_stack((
            np.interp(physical[:, 1], local_y, grid_rows),
            np.interp(physical[:, 0], local_x, grid_cols),
        )))
    return tuple(converted)


def _paths_at_level(
    generator: Any,
    level: float,
    xdim: np.ndarray,
    ydim: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, ...]:
    """Call one ContourPy generator and normalize its output paths."""
    return _paths_to_grid(generator.lines(float(level)), xdim, ydim, roi)


def _global_level_schedule(
    peaks: list[dict[str, Any]],
    params: Mapping[str, Any],
    count: int,
) -> tuple[np.ndarray, float, float]:
    """Build shared absolute levels without a percentile or radius cutoff."""
    if count < 2:
        raise ValueError("fast_haller global_level_count 必须至少为 2")
    values = np.asarray([float(peak["value"]) for peak in peaks], dtype=np.float64)
    if len(values) == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("fast_haller peaks 必须包含有限的正 IVD maxima")
    # The strict oracle's frozen 0.1..1 peak-relative envelope supplies the
    # physical range.  The discretisation is global: no p80/p70 cutoff, no
    # prominence, and no arbitrary inter-peak suppression are introduced.
    low = float(values.min() * float(params["contour_level_end"]))
    high = float(values.max() * float(params["contour_level_start"]))
    if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low < high):
        raise ValueError(f"fast_haller global level range 无效：low={low}, high={high}")
    # Frame 600 has a wide IVD dynamic range.  Geometric spacing keeps the
    # levels global/shared while still giving weak maxima a resolvable bracket;
    # using a percentile cutoff would hide those maxima rather than solve the
    # range problem.
    return np.geomspace(low, high, int(count), dtype=np.float64), low, high


def _refinement_roi(
    coarse_path: np.ndarray,
    shape: tuple[int, int],
    halo_cells: int,
) -> tuple[int, int, int, int]:
    """Return the coarse contour bbox plus a computational halo."""
    halo = int(halo_cells)
    if halo < 0:
        raise ValueError("fast_haller refinement halo_cells 不能为负")
    path = np.asarray(coarse_path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) == 0:
        raise ValueError("coarse_path 必须是非空的 (N,2) contour")
    rows, cols = (int(shape[0]), int(shape[1]))
    row0 = max(0, int(np.floor(np.min(path[:, 0]))) - halo)
    row1 = min(rows, int(np.ceil(np.max(path[:, 0]))) + halo + 1)
    col0 = max(0, int(np.floor(np.min(path[:, 1]))) - halo)
    col1 = min(cols, int(np.ceil(np.max(path[:, 1]))) + halo + 1)
    return row0, row1, col0, col1


def _find_refinement_contours(
    ivd: np.ndarray,
    fluid: np.ndarray,
    level: float,
    xdim: np.ndarray,
    ydim: np.ndarray,
    coarse_path: np.ndarray,
    *,
    global_generator: Any,
    halo_cells: int = DEFAULT_REFINEMENT_HALO_CELLS,
) -> tuple[tuple[np.ndarray, ...], tuple[int, int, int, int], bool, int]:
    """Extract a refinement level in an exact-safe ROI, or full-domain fallback.

    The same ContourPy algorithm is used in each local generator.  A crop is
    accepted only when both the high-side support and every returned path are
    interior to it.  Otherwise the ROI grows geometrically.  A domain-sized
    crop reuses the original global generator, so no boundary-touching contour
    is ever discarded as an optimization artefact.
    """
    shape = (int(ivd.shape[0]), int(ivd.shape[1]))
    domain = (0, shape[0], 0, shape[1])
    roi = _refinement_roi(coarse_path, shape, halo_cells)
    calls = 0
    while True:
        if roi == domain:
            calls += 1
            return _paths_at_level(global_generator, level, xdim, ydim), domain, True, calls

        calls += 1
        local_generator = _make_contour_generator(
            ivd, fluid, xdim, ydim, roi=roi
        )
        local_paths = _paths_at_level(local_generator, level, xdim, ydim, roi)
        row0, row1, col0, col1 = roi
        high_side = np.asarray(fluid[row0:row1, col0:col1], dtype=bool) & (
            np.asarray(ivd[row0:row1, col0:col1], dtype=np.float64) >= float(level)
        )
        support_touches = bool(
            np.any(high_side[0, :])
            or np.any(high_side[-1, :])
            or np.any(high_side[:, 0])
            or np.any(high_side[:, -1])
        )
        contour_touches = any(
            strict._path_touches_crop_boundary(path, (row1 - row0, col1 - col0))
            for path in local_paths
        )
        if not support_touches and not contour_touches:
            return local_paths, roi, False, calls

        expanded, _ = strict.expand_component_roi(
            roi,
            shape,
            margin_cells=max(
                1,
                (row1 - row0 + 1) // 2,
                (col1 - col0 + 1) // 2,
            ),
        )
        if expanded == roi or expanded == domain:
            roi = domain
        else:
            roi = expanded


@dataclass(frozen=True)
class _Candidate:
    path: np.ndarray
    level: float
    metrics: Mapping[str, Any]
    record: dict[str, Any]


def _record(
    path: np.ndarray,
    peak: Mapping[str, Any],
    level: float,
    xdim: np.ndarray,
    ydim: np.ndarray,
    *,
    reason: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    maximum_count: int | None = None,
    phase: str = "global",
    refinement_iteration: int | None = None,
) -> dict[str, Any]:
    """Build a compact, JSON-safe candidate diagnostic."""
    peak_value = float(peak["value"])
    result: dict[str, Any] = {
        "peak": {
            "row": int(peak["row"]),
            "col": int(peak["col"]),
            "value": peak_value,
        },
        "level": float(level),
        "level_fraction": float(level / peak_value),
        "closed": reason != "open_contour",
        "status": "valid" if reason is None else "rejected",
        "phase": phase,
    }
    if maximum_count is not None:
        result["maximum_count"] = int(maximum_count)
    if refinement_iteration is not None:
        result["refinement_iteration"] = int(refinement_iteration)
    if reason is not None:
        result["rejection_reason"] = str(reason)
        return result
    if metrics is None:
        raise ValueError("valid fast_haller candidate 必须包含 metrics")
    for key in ("area", "perimeter", "hull_area", "convexity_defect", "n_points"):
        result[key] = (
            int(metrics[key])
            if key == "n_points"
            else float(np.round(float(metrics[key]), decimals=10))
        )
    return result


def _evaluate_for_peak(
    paths: tuple[np.ndarray, ...],
    peak: Mapping[str, Any],
    peak_index: int,
    maximum_count: int,
    level: float,
    solid: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
    params: Mapping[str, Any],
    dx: float,
    dy: float,
    *,
    phase: str,
    refinement_iteration: int | None = None,
) -> tuple[list[_Candidate], list[dict[str, Any]]]:
    """Apply all strict geometry tests to paths containing one peak."""
    _, _, QhullError, _, _, _ = strict._haller_geometry_dependencies()
    valid: list[_Candidate] = []
    diagnostics: list[dict[str, Any]] = []
    minimum_perimeter = float(params["minimum_perimeter_factor"]) * max(dx, dy)
    peak_point = np.asarray([[float(peak["row"]), float(peak["col"])]])
    for path in paths:
        closed, canonical_path = strict._closed_path(
            path, dx, dy, float(params["closure_tolerance_factor"])
        )
        if not closed:
            diagnostics.append(_record(
                canonical_path, peak, level, xdim, ydim,
                reason="open_contour", maximum_count=maximum_count,
                phase=phase, refinement_iteration=refinement_iteration,
            ))
            continue
        if not bool(pts_in_poly_mask(peak_point, canonical_path)[0]):
            diagnostics.append(_record(
                canonical_path, peak, level, xdim, ydim,
                reason="does_not_enclose_peak", maximum_count=maximum_count,
                phase=phase, refinement_iteration=refinement_iteration,
            ))
            continue
        if strict._contour_crosses_solid(canonical_path, solid, dx, dy):
            diagnostics.append(_record(
                canonical_path, peak, level, xdim, ydim,
                reason="solid_crossing", maximum_count=maximum_count,
                phase=phase, refinement_iteration=refinement_iteration,
            ))
            continue
        try:
            metrics = strict.contour_metrics(canonical_path, xdim, ydim)
        except (ValueError, QhullError) as exc:
            diagnostics.append(_record(
                canonical_path, peak, level, xdim, ydim,
                reason=f"geometry_error:{type(exc).__name__}",
                maximum_count=maximum_count, phase=phase,
                refinement_iteration=refinement_iteration,
            ))
            continue
        if float(metrics["area"]) <= 0.0:
            reason = "zero_area"
        elif float(metrics["convexity_defect"]) > float(params["convexity_defect_max"]) + 1e-12:
            reason = "convexity_defect"
        elif float(metrics["perimeter"]) + 1e-12 < minimum_perimeter:
            reason = "minimum_perimeter"
        else:
            reason = None
        diagnostics.append(_record(
            canonical_path, peak, level, xdim, ydim,
            reason=reason, metrics=metrics if reason is None else None,
            maximum_count=maximum_count, phase=phase,
            refinement_iteration=refinement_iteration,
        ))
        if reason is None:
            valid.append(_Candidate(
                path=canonical_path,
                level=float(level),
                metrics=metrics,
                record=diagnostics[-1],
            ))
    return valid, diagnostics


def _best_candidate(candidates: list[_Candidate]) -> _Candidate | None:
    """Select the outermost candidate at one level by physical area."""
    if not candidates:
        return None
    return max(candidates, key=lambda item: (
        float(item.metrics["area"]), -float(item.level)
    ))


def _selected_record(
    candidate: _Candidate,
    peak: Mapping[str, Any],
    xdim: np.ndarray,
    ydim: np.ndarray,
) -> dict[str, Any]:
    """Return the raw IVD path, never the ConvexHull polygon."""
    selected = dict(candidate.record)
    selected["status"] = "valid"
    selected["points_grid"] = np.asarray(candidate.path, dtype=np.float64).round(12).tolist()
    selected["points_xy"] = strict._grid_path_to_xy(
        candidate.path, xdim, ydim
    ).round(12).tolist()
    return selected


def select_contours(
    ivd: Any,
    solid: Any,
    xdim: Any,
    ydim: Any,
    peaks: list[dict[str, Any]],
    params: Mapping[str, Any],
    dx: float,
    dy: float,
    *,
    global_level_count: int = DEFAULT_GLOBAL_LEVEL_COUNT,
    refinement_iterations: int = DEFAULT_REFINEMENT_ITERATIONS,
    refinement_halo_cells: int = DEFAULT_REFINEMENT_HALO_CELLS,
    return_evaluation: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Select fast Haller boundaries with shared levels and local refinement.

    Global levels are traversed from low to high.  The first level with a
    valid contour for a peak is its coarse outermost level; the nearest lower
    level without a valid contour is the outer invalid bracket for bisection.
    All global levels are still evaluated, so this routine does not rely on an
    unproved early-stop assumption.
    """
    started = time.perf_counter()
    field = np.asarray(ivd, dtype=np.float64)
    solid_array = np.asarray(solid, dtype=bool)
    x = np.asarray(xdim, dtype=np.float64)
    y = np.asarray(ydim, dtype=np.float64)
    if field.ndim != 2 or solid_array.shape != field.shape:
        raise ValueError("fast_haller ivd/solid 必须是同形状二维数组")
    if not np.all(np.isfinite(field)):
        raise ValueError("fast_haller ivd 必须是有限数组")
    count = int(global_level_count)
    iterations = int(refinement_iterations)
    if count < 2:
        raise ValueError("fast_haller global_level_count 必须至少为 2")
    if iterations < 0:
        raise ValueError("fast_haller refinement_iterations 不能为负")
    if not peaks:
        empty_evaluation = {
            "global_level_count": count,
            "global_contour_call_count": 0,
            "global_contour_calls": 0,
            "refinement_call_count": 0,
            "refinement_calls": 0,
            "refinement_iterations": iterations,
            "vortex_count": 0,
            "contour_generator_count": 0,
            "wall_time_seconds": float(time.perf_counter() - started),
        }
        if return_evaluation:
            return [], [], empty_evaluation
        return [], []
    levels, global_low, global_high = _global_level_schedule(
        peaks, params, count
    )
    fluid = ~solid_array
    generator = _make_contour_generator(field, fluid, x, y)
    generator_count = 1
    peak_points = np.asarray(
        [[float(peak["row"]), float(peak["col"])] for peak in peaks],
        dtype=np.float64,
    )
    valid_by_peak: list[list[_Candidate]] = [[] for _ in peaks]
    invalid_levels_by_peak: list[list[float]] = [[] for _ in peaks]
    diagnostics: list[dict[str, Any]] = []
    global_call_count = 0
    membership_counts = {"zero": 0, "one": 0, "multiple": 0}
    for level in levels:
        global_call_count += 1
        paths = _paths_at_level(generator, float(level), x, y)
        contained_indices: list[tuple[np.ndarray, np.ndarray]] = []
        for path in paths:
            membership = pts_in_poly_mask(peak_points, path)
            contained = np.flatnonzero(membership).astype(np.intp)
            if len(contained) == 0:
                membership_counts["zero"] += 1
            elif len(contained) == 1:
                membership_counts["one"] += 1
            else:
                membership_counts["multiple"] += 1
            contained_indices.append((path, contained))

        for peak_index, peak in enumerate(peaks):
            candidates_at_level: list[_Candidate] = []
            containing_entries = [
                (path, contained) for path, contained in contained_indices
                if peak_index in contained
            ]
            containing_paths = [path for path, _ in containing_entries]
            if containing_paths:
                maximum_count = max(
                    len(contained) for _, contained in containing_entries
                )
                valid, level_diagnostics = _evaluate_for_peak(
                    tuple(containing_paths), peak, peak_index, maximum_count,
                    float(level), solid_array, x, y, params, dx, dy,
                    phase="global",
                )
                candidates_at_level.extend(valid)
                diagnostics.extend(level_diagnostics)
            else:
                reason = "no_contour" if not paths else "does_not_enclose_peak"
                diagnostics.append(_record(
                    np.empty((0, 2), dtype=np.float64), peak, float(level), x, y,
                    reason=reason, maximum_count=0, phase="global",
                ))
            candidate = _best_candidate(candidates_at_level)
            if candidate is None:
                invalid_levels_by_peak[peak_index].append(float(level))
            else:
                valid_by_peak[peak_index].append(candidate)

    selected: list[dict[str, Any]] = []
    refinement_call_count = 0
    refinement_fallback_count = 0
    refinement_success_count = 0
    refinement_generator_count = 0
    for peak_index, peak in enumerate(peaks):
        coarse = valid_by_peak[peak_index][0] if valid_by_peak[peak_index] else None
        if coarse is None:
            continue
        outer_invalid = [
            level for level in invalid_levels_by_peak[peak_index]
            if level < coarse.level
        ]
        if not outer_invalid or iterations == 0:
            selected.append(_selected_record(coarse, peak, x, y))
            continue
        lo = max(outer_invalid)
        hi = float(coarse.level)
        best = coarse
        for iteration in range(1, iterations + 1):
            mid = float((lo + hi) * 0.5)
            paths, _, fallback, calls = _find_refinement_contours(
                field, fluid, mid, x, y, coarse.path,
                global_generator=generator,
                halo_cells=refinement_halo_cells,
            )
            refinement_call_count += int(calls)
            refinement_fallback_count += int(bool(fallback))
            # One local generator is created for every non-fallback bisection;
            # a fallback reuses the single global generator.
            refinement_generator_count += int(calls - bool(fallback))
            refinement_membership = [
                int(np.count_nonzero(pts_in_poly_mask(peak_points, path)))
                for path in paths
            ]
            valid, level_diagnostics = _evaluate_for_peak(
                paths, peak, peak_index,
                maximum_count=max(refinement_membership, default=1),
                level=mid, solid=solid_array, xdim=x, ydim=y, params=params,
                dx=dx, dy=dy, phase="refinement",
                refinement_iteration=iteration,
            )
            diagnostics.extend(level_diagnostics)
            candidate = _best_candidate(valid)
            if candidate is None:
                lo = mid
            else:
                hi = mid
                best = candidate
                refinement_success_count += 1
        selected.append(_selected_record(best, peak, x, y))

    evaluation = {
        "global_level_count": count,
        "global_level_start": global_low,
        "global_level_end": global_high,
        "global_level_schedule": "geomspace(min_peak*contour_level_end,max_peak*contour_level_start)",
        "global_contour_call_count": int(global_call_count),
        "global_contour_calls": int(global_call_count),
        "refinement_call_count": int(refinement_call_count),
        "refinement_calls": int(refinement_call_count),
        "refinement_iterations": iterations,
        "refinement_halo_cells": int(refinement_halo_cells),
        "refinement_success_count": int(refinement_success_count),
        "refinement_roi_boundary_fallback_count": int(refinement_fallback_count),
        "contour_generator_count": int(generator_count + refinement_generator_count),
        "membership_counts": membership_counts,
        "vortex_count": int(len(selected)),
        "wall_time_seconds": float(time.perf_counter() - started),
        "contour_algorithm": "contourpy_serial_contour_generator",
        "final_boundary": "original_ivd_level_contour",
        "convexity_measure": "convex_hull_area_only",
        "peak_assignment": "all_contained_maxima; zero/one/multiple explicit",
        "level_search": "global_shared_low_to_high_then_outer_invalid_valid_bisection",
        "roi_policy": "coarse_bbox_plus_halo_expand_on_support_or_path_boundary_then_full_domain",
    }
    compressed = strict._compress_contour_diagnostics(diagnostics)
    if return_evaluation:
        return selected, compressed, evaluation
    return selected, compressed


__all__ = [
    "BACKEND_FAST_HALLER",
    "BACKEND_VERSION",
    "DEFAULT_GLOBAL_LEVEL_COUNT",
    "DEFAULT_REFINEMENT_ITERATIONS",
    "DEFAULT_REFINEMENT_HALO_CELLS",
    "pts_in_poly_mask",
    "select_contours",
]
