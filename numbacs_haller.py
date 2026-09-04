"""Adapter for the upstream NumbaCS ``rotcohvrt`` Haller extractor.

The upstream checkout is loaded at runtime and is never modified by this
module.  NumbaCS consumes an already-computed IVD/LAVD field and returns
physical ``(x, y)`` contour points plus the selected maximum.  The adapter
converts those points to this repository's ``(row, col)`` artifact convention
and reuses the existing three-state rasterization and artifact writer.

Formal artifacts use the native upstream call contract: only the physical
radius ``r``, ``convexity_deficiency`` and ``min_len`` are supplied.  The
upstream defaults for ``min_val``, ``nlevs``, ``start_level`` and ``end_level``
remain authoritative.  In particular, this module deliberately contains no
replacement maximum finder, no no-JIT compatibility path and no CUDA Haller
backend.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import haller_anchors as strict


BACKEND_NUMBACS = "numbacs"
UPSTREAM_VERSION = "0.2.0"
UPSTREAM_COMMIT = "c067f542543f5dd4ae3dc45fc506213e8d98b845"
BACKEND_VERSION = f"numbacs-{UPSTREAM_VERSION}+{UPSTREAM_COMMIT[:12]}"
UPSTREAM_DEFAULT_MIN_VAL = -1.0
UPSTREAM_DEFAULT_NLEVS = 20
UPSTREAM_DEFAULT_START_LEVEL = 0.0
UPSTREAM_DEFAULT_END_LEVEL = 0.0
_UPSTREAM_ELLIPTIC_MODULE: Any | None = None


def _load_rotcohvrt() -> tuple[Any, str, str]:
    """Load the unmodified upstream function and expose its runtime identity."""
    global _UPSTREAM_ELLIPTIC_MODULE
    try:
        import numbacs

        # ``numbacs.extraction.__init__`` eagerly imports the ridge and
        # hyperbolic extractors.  Those optional modules pull in the full
        # interpolation/Numba stack even though ``rotcohvrt`` only needs
        # ``numbacs.utils``.  Load the official elliptic module directly from
        # the read-only upstream checkout while preserving its package name so
        # the module's relative ``..utils`` import remains unchanged.  This is
        # an import-boundary optimization only; the upstream source is not
        # copied or edited.
        extraction_name = "numbacs.extraction"
        elliptic_name = "numbacs.extraction.elliptic"
        injected_package = False
        if extraction_name not in sys.modules:
            extraction_path = Path(numbacs.__path__[0]) / "extraction"
            extraction_package = types.ModuleType(extraction_name)
            extraction_package.__package__ = extraction_name
            extraction_package.__path__ = [str(extraction_path)]
            extraction_package.__spec__ = importlib.util.spec_from_loader(
                extraction_name,
                loader=None,
                is_package=True,
            )
            sys.modules[extraction_name] = extraction_package
            injected_package = True
        try:
            elliptic_module = importlib.import_module(elliptic_name)
        except Exception:
            if injected_package:
                sys.modules.pop(extraction_name, None)
            raise
        rotcohvrt = elliptic_module.rotcohvrt
        _UPSTREAM_ELLIPTIC_MODULE = elliptic_module
    except ImportError as exc:  # pragma: no cover - exercised in runtime env
        raise ModuleNotFoundError(
            "numbacs backend requires the upstream numbacs source on PYTHONPATH "
            "and its runtime dependencies"
        ) from exc
    version = str(getattr(numbacs, "__version__", UPSTREAM_VERSION))
    source = str(inspect.getsourcefile(rotcohvrt) or "unknown")
    return rotcohvrt, version, source


def runtime_mode() -> str:
    """Return whether the upstream helper is running the native Numba runtime."""
    try:
        import numba
    except ImportError:
        return "unavailable"
    module_path = str(getattr(numba, "__file__", "")).replace("\\", "/")
    if module_path.endswith("/numbacs-purepy-runtime/numba.py"):
        return "python_compat_no_jit"
    return "numba_jit"


def ensure_native_runtime() -> None:
    """Reject the compatibility shim before formal native artifact generation."""
    mode = runtime_mode()
    if mode != "numba_jit":
        raise RuntimeError(
            "formal NumbaCS artifacts require the official Numba JIT runtime; "
            f"runtime_mode={mode!r} is not native"
        )


def _physical_to_grid(
    points_xy: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
) -> np.ndarray:
    """Convert upstream physical ``(x,y)`` points to global ``(row,col)``."""
    return np.column_stack((
        np.interp(
            points_xy[:, 1],
            ydim,
            np.arange(len(ydim), dtype=np.float64),
        ),
        np.interp(
            points_xy[:, 0],
            xdim,
            np.arange(len(xdim), dtype=np.float64),
        ),
    ))


def _empty_evaluation(
    *,
    radius: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "numbacs_requested_level_count": UPSTREAM_DEFAULT_NLEVS,
        "numbacs_contour_level_schedule": "upstream_linspace_percentile70_to_max",
        "numbacs_runtime_mode": runtime_mode(),
        "numbacs_radius": float(radius),
        "numbacs_native_call": True,
        "numbacs_explicit_kwargs": ["convexity_deficiency", "min_len"],
        "numbacs_omitted_kwargs": [
            "min_val", "nlevs", "start_level", "end_level"
        ],
        "numbacs_default_min_val": UPSTREAM_DEFAULT_MIN_VAL,
        "numbacs_default_nlevs": UPSTREAM_DEFAULT_NLEVS,
        "numbacs_default_start_level": UPSTREAM_DEFAULT_START_LEVEL,
        "numbacs_default_end_level": UPSTREAM_DEFAULT_END_LEVEL,
        "numbacs_return_count": 0,
        "numbacs_empty_reason": reason,
        "contour_algorithm": "numbacs.rotcohvrt",
        "contour_generator_policy": "one_upstream_contourpy_generator_per_frame",
        "final_boundary": "upstream_rotcohvrt_convex_hull",
        "convexity_measure": "upstream_convex_hull_area_deficiency",
    }


def select_contours(
    ivd: Any,
    solid: Any,
    xdim: Any,
    ydim: Any,
    params: Mapping[str, Any],
    dx: float,
    dy: float,
    *,
    return_evaluation: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Run upstream ``rotcohvrt`` and normalize its selected contours.

    ``rotcohvrt`` has no geometry-mask parameter, so solid IVD values are
    already zeroed by :func:`haller_anchors._compute_standard_ivd_details`.
    The adapter intentionally preserves upstream's global level scan,
    radius-based maximum removal, and convex-hull boundary output.  Those
    choices are recorded in the returned evaluation and artifact parameters.
    """
    field = np.asarray(ivd, dtype=np.float64)
    solid_array = np.asarray(solid, dtype=bool)
    x = np.asarray(xdim, dtype=np.float64)
    y = np.asarray(ydim, dtype=np.float64)
    if field.ndim != 2 or solid_array.shape != field.shape:
        raise ValueError("numbacs ivd/solid 必须是同形状二维数组")
    if len(x) != field.shape[1] or len(y) != field.shape[0]:
        raise ValueError("numbacs 坐标轴与 IVD shape 不匹配")

    forbidden_overrides = (
        "numbacs_min_val", "numbacs_nlevs", "numbacs_start_level",
        "numbacs_end_level",
    )
    stale = [key for key in forbidden_overrides if key in params]
    if stale:
        raise ValueError(
            "formal NumbaCS rotcohvrt 禁止覆盖 upstream defaults："
            f"{stale!r}"
        )
    radius = max(float(dx), float(dy))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("numbacs radius 必须为正")

    fluid = ~solid_array
    if not np.any(fluid):
        evaluation = _empty_evaluation(
            radius=radius,
            reason="all_solid",
        )
        return ([], [], evaluation) if return_evaluation else ([], [])
    positive = field[fluid]
    upstream_min_val = float(np.percentile(field, 80.0))
    if positive.size == 0 or float(np.max(field)) <= upstream_min_val:
        evaluation = _empty_evaluation(
            radius=radius,
            reason="no_maximum_above_min_val",
        )
        return ([], [], evaluation) if return_evaluation else ([], [])

    rotcohvrt, runtime_version, source_file = _load_rotcohvrt()
    # NumbaCS uses (nx, ny) = (x, y), while this project stores arrays as
    # (row, col) = (y, x).  Its own implementation transposes the field back
    # into the ContourPy convention, so the transpose here is intentional.
    # Keep this call intentionally native.  Omitting the four optional search
    # arguments is part of the formal artifact identity and is guarded by the
    # adapter tests.
    upstream_result = rotcohvrt(
        np.asarray(field.T, dtype=np.float64),
        x,
        y,
        radius,
        convexity_deficiency=float(params["convexity_defect_max"]),
        min_len=float(params["minimum_perimeter_factor"]) * max(float(dx), float(dy)),
    )

    selected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for item_index, item in enumerate(upstream_result):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"numbacs rotcohvrt 返回项 {item_index} 不符合 [contour, center] contract"
            )
        points_xy = np.asarray(item[0], dtype=np.float64)
        center_xy = np.asarray(item[1], dtype=np.float64)
        if points_xy.ndim != 2 or points_xy.shape[1] != 2 or len(points_xy) < 4:
            raise ValueError(f"numbacs contour {item_index} 不是有效二维路径")
        if center_xy.shape != (2,) or not np.all(np.isfinite(center_xy)):
            raise ValueError(f"numbacs contour {item_index} 的 center 无效")
        if not np.all(np.isfinite(points_xy)):
            raise ValueError(f"numbacs contour {item_index} 包含非有限坐标")
        grid_path = _physical_to_grid(points_xy, x, y)
        closed, grid_path = strict._closed_path(
            grid_path,
            float(dx),
            float(dy),
            float(params["closure_tolerance_factor"]),
        )
        if not closed:
            raise ValueError(f"numbacs contour {item_index} 不是闭合路径")
        col = int(np.argmin(np.abs(x - center_xy[0])))
        row = int(np.argmin(np.abs(y - center_xy[1])))
        peak = {
            "row": row,
            "col": col,
            "value": float(field[row, col]),
        }
        metrics = strict.contour_metrics(grid_path, x, y)
        record: dict[str, Any] = {
            "peak": peak,
            "center_xy": center_xy.round(12).tolist(),
            "level": None,
            "level_fraction": None,
            "closed": True,
            "status": "valid",
            "phase": "numbacs_rotcohvrt",
            "area": float(np.round(metrics["area"], decimals=10)),
            "perimeter": float(np.round(metrics["perimeter"], decimals=10)),
            "hull_area": float(np.round(metrics["hull_area"], decimals=10)),
            "convexity_defect": float(
                np.round(metrics["convexity_defect"], decimals=10)
            ),
            "n_points": int(metrics["n_points"]),
            "points_grid": grid_path.round(12).tolist(),
            "points_xy": points_xy.round(12).tolist(),
            "boundary_source": "numbacs_rotcohvrt_convex_hull",
        }
        selected.append(record)
        diagnostics.append(dict(record))

    evaluation = {
        "numbacs_requested_level_count": UPSTREAM_DEFAULT_NLEVS,
        "numbacs_contour_level_schedule": "upstream_linspace_percentile70_to_max",
        "numbacs_runtime_mode": runtime_mode(),
        "numbacs_radius": radius,
        "numbacs_radius_factor": 1.0,
        "numbacs_native_call": True,
        "numbacs_explicit_kwargs": ["convexity_deficiency", "min_len"],
        "numbacs_omitted_kwargs": [
            "min_val", "nlevs", "start_level", "end_level"
        ],
        "numbacs_default_min_val": UPSTREAM_DEFAULT_MIN_VAL,
        "numbacs_default_nlevs": UPSTREAM_DEFAULT_NLEVS,
        "numbacs_default_start_level": UPSTREAM_DEFAULT_START_LEVEL,
        "numbacs_default_end_level": UPSTREAM_DEFAULT_END_LEVEL,
        "numbacs_start_level": "upstream_default_percentile70",
        "numbacs_end_level": "upstream_default_maximum",
        "numbacs_min_len": float(params["minimum_perimeter_factor"]) * max(float(dx), float(dy)),
        "numbacs_maxima_storage": "upstream_max_in_radius_native_fixed_capacity",
        "numbacs_return_count": len(selected),
        "numbacs_runtime_version": runtime_version,
        "numbacs_runtime_source": source_file,
        "contour_algorithm": "numbacs.rotcohvrt",
        "contour_generator_policy": "one_upstream_contourpy_generator_per_frame",
        "final_boundary": "upstream_rotcohvrt_convex_hull",
        "convexity_measure": "upstream_convex_hull_area_deficiency",
        "peak_assignment": "upstream_pts_in_poly_first_remaining_maximum",
    }
    if return_evaluation:
        return selected, diagnostics, evaluation
    return selected, diagnostics


__all__ = [
    "BACKEND_NUMBACS",
    "BACKEND_VERSION",
    "UPSTREAM_DEFAULT_END_LEVEL",
    "UPSTREAM_DEFAULT_MIN_VAL",
    "UPSTREAM_DEFAULT_NLEVS",
    "UPSTREAM_DEFAULT_START_LEVEL",
    "UPSTREAM_COMMIT",
    "UPSTREAM_VERSION",
    "ensure_native_runtime",
    "runtime_mode",
    "select_contours",
]
