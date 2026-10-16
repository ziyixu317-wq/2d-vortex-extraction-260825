"""Sparse Haller anchor qualification and anchor--percentile relationships.

This module is the Ticket 05 seam.  It deliberately keeps the current
NumbaCS raw-contour evidence as an input source rather than promoting that
candidate to a dense label.  The public single-frame builder turns returned
centres into either ``center_cell`` or ``small_core_r1`` positive-only
evidence, while retaining support, failure, raw-contour, and provenance
records needed by the qualification and relationship reports.

The controlled qualification and six-dataset report are implemented below in
the same module so the construction choice has one auditable entry point and
cannot be selected from real-data coverage or test results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import platform
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


PROTOCOL_VERSION = "criterion-ambiguity-vortex-learning-sparse-anchor-v1"
ARTIFACT_TYPE = "haller_sparse_physical_anchor"
ARTIFACT_VERSION = "haller-sparse-physical-anchor-v1"
SAMPLE_ARTIFACT_TYPE = "haller_sparse_physical_anchor_sample"
QUALIFICATION_REPORT_TYPE = "haller_sparse_anchor_qualification_report"
RELATIONSHIP_REPORT_TYPE = "haller_anchor_percentile_relationship_report"
MANIFEST_TYPE = "haller_sparse_physical_anchor_manifest"
INTEGRITY_SIDECAR_TYPE = "haller_sparse_physical_anchor_integrity"
# Backward-compatible spelling for the first local implementation slice.
INTEGRITY_SIDEcar_TYPE = INTEGRITY_SIDECAR_TYPE

CONSTRUCTION_CENTER_CELL = "center_cell"
CONSTRUCTION_SMALL_CORE_R1 = "small_core_r1"
REGISTERED_CONSTRUCTIONS = (
    CONSTRUCTION_CENTER_CELL,
    CONSTRUCTION_SMALL_CORE_R1,
)
SELECTION_ORDER = (
    CONSTRUCTION_SMALL_CORE_R1,
    CONSTRUCTION_CENTER_CELL,
)

STATE_POSITIVE_ANCHOR = "positive_anchor"
STATE_UNCOVERED = "uncovered"
STATE_INVALID = "invalid"
STATE_EXTRACTOR_FAILURE = "extractor_failure"
STATE_SOLID = "solid"
SPARSE_STATES = (
    STATE_POSITIVE_ANCHOR,
    STATE_UNCOVERED,
    STATE_INVALID,
    STATE_EXTRACTOR_FAILURE,
    STATE_SOLID,
)

QUALIFICATION_STATUS_QUALIFIED = "qualified"
QUALIFICATION_STATUS_NO_GO = "no-go"
ARTIFACT_STATES = (
    "candidate",
    "audited",
    "failed",
    "unqualified",
    "qualified",
)

ALLOWED_REAL_SPLITS = frozenset({"train", "calibration"})
ALLOWED_BUILD_SPLITS = frozenset({
    "train",
    "calibration",
    "controlled",
    "development_audit",
})
REAL_DATASET_NAMES = (
    "boussinesq",
    "cylinder2d",
    "doublegyre2d",
    "fourcenters2d",
    "jungtelziemniak2d",
    "pipedcylinder2d",
)

ANCHOR_PRECISION_GATE = 0.95
CENTER_ERROR_GATE_MULTIPLIER = 2.0
NO_VORTEX_FALSE_ANCHOR_GATE = 0
CONTROLLED_FAILURE_RATE_GATE = 0.0

_FRAME1300_DATASET = "boussinesq"
_TEST_MARKER = re.compile(r"(?:^|[_-])(?:final[-_]?)?test(?:$|[_-])", re.IGNORECASE)


class SparseAnchorContractError(ValueError):
    """Raised when a sparse-anchor or relationship contract is malformed."""


class FinalTestAccessError(PermissionError):
    """Raised before reading a test partition or test relationship."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _json_copy(value: Any) -> Any:
    return _jsonable(copy.deepcopy(value))


def _hash_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _hash_named_arrays(values: Mapping[str, Any]) -> str:
    return _json_hash({name: _hash_array(values[name]) for name in sorted(values)})


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SparseAnchorContractError(f"{name} 必须是非空字符串")
    return value.strip()


def _exact_frame(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise SparseAnchorContractError("frame_index 必须是整数，不能是 bool")
    try:
        frame = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SparseAnchorContractError("frame_index 必须是整数") from exc
    if not np.isfinite(numeric) or numeric != frame:
        raise SparseAnchorContractError(f"frame_index 必须是整数，实际 {value!r}")
    return frame


def _validate_scope(
    dataset_name: Any,
    split_name: Any,
    frame_index: Any,
    *,
    real_relationship: bool = False,
) -> tuple[str, str, int]:
    """Validate scope before converting any array-like input."""
    dataset = _text(dataset_name, name="dataset_name")
    split = _text(split_name, name="split_name").lower()
    frame = _exact_frame(frame_index)
    if split == "test" or "final-test" in split or "final_test" in split:
        raise FinalTestAccessError(
            f"sparse-anchor development path 禁止读取 final-test split={split!r}"
        )
    if _TEST_MARKER.search(split):
        raise FinalTestAccessError(
            f"sparse-anchor development path 禁止读取 test-like split={split!r}"
        )
    if real_relationship and split not in ALLOWED_REAL_SPLITS:
        raise SparseAnchorContractError(
            f"relationship study 只允许 train/calibration，实际 split={split!r}"
        )
    if split not in ALLOWED_BUILD_SPLITS:
        raise SparseAnchorContractError(
            f"不允许的 sparse-anchor split={split!r}；允许 {sorted(ALLOWED_BUILD_SPLITS)}"
        )
    if dataset.lower() == _FRAME1300_DATASET and frame == 1300:
        raise SparseAnchorContractError(
            "Boussinesq frame 1300 是 development/audit-only，禁止进入本票 relationship/artifact"
        )
    return dataset, split, frame


def _validate_construction(construction: Any) -> str:
    value = _text(construction, name="construction")
    if value not in REGISTERED_CONSTRUCTIONS:
        raise SparseAnchorContractError(
            "本票只允许预注册 construction "
            f"{REGISTERED_CONSTRUCTIONS}，实际 {value!r}"
        )
    return value


def _coerce_axes(
    xdim: Any,
    ydim: Any,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.asarray(xdim, dtype=np.float64)
    y = np.asarray(ydim, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != shape[1] or len(y) != shape[0]:
        raise SparseAnchorContractError(
            f"coordinate axes 与 support shape={shape} 不匹配"
        )
    if len(x) < 2 or len(y) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise SparseAnchorContractError("coordinate axes 必须是有限且至少含两个点")
    dxs = np.diff(x)
    dys = np.diff(y)
    if np.any(dxs <= 0.0) or np.any(dys <= 0.0):
        raise SparseAnchorContractError("coordinate axes 必须严格递增")
    return x, y, float(np.median(dxs)), float(np.median(dys))


def _coerce_support(
    value: Any,
    shape: tuple[int, int],
    *,
    name: str,
    default: bool,
) -> np.ndarray:
    if value is None:
        return np.full(shape, default, dtype=bool)
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise SparseAnchorContractError(f"{name} shape={array.shape} != {shape}")
    return array.copy()


def _validate_support(
    solid_mask: np.ndarray,
    invalid_support: np.ndarray,
    valid_fluid: np.ndarray,
) -> None:
    if np.any(solid_mask & invalid_support):
        raise SparseAnchorContractError("solid_mask 与 invalid_support 不能重叠")
    expected = ~solid_mask & ~invalid_support
    if not np.array_equal(valid_fluid, expected):
        raise SparseAnchorContractError(
            "valid_fluid 必须严格等于 ~solid_mask & ~invalid_support"
        )


def _grid_to_physical(
    grid_rc: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
) -> np.ndarray:
    row, col = float(grid_rc[0]), float(grid_rc[1])
    return np.asarray([
        np.interp(col, np.arange(len(xdim), dtype=np.float64), xdim),
        np.interp(row, np.arange(len(ydim), dtype=np.float64), ydim),
    ], dtype=np.float64)


def _physical_to_grid(
    physical_xy: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
) -> np.ndarray:
    return np.asarray([
        np.interp(float(physical_xy[1]), ydim, np.arange(len(ydim), dtype=np.float64)),
        np.interp(float(physical_xy[0]), xdim, np.arange(len(xdim), dtype=np.float64)),
    ], dtype=np.float64)


def _center_coordinates(
    value: Any,
    *,
    xdim: np.ndarray,
    ydim: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    if isinstance(value, Mapping):
        physical = value.get("center_xy")
        if physical is None:
            physical = value.get("native_physical_xy")
        if physical is None:
            physical = value.get("center_physical")
        grid = value.get("center_grid")
        if grid is None:
            grid = value.get("native_grid_rc")
    else:
        physical = value
        grid = None

    physical_array: np.ndarray | None = None
    grid_array: np.ndarray | None = None
    if physical is not None:
        try:
            candidate = np.asarray(physical, dtype=np.float64)
            if candidate.shape == (2,):
                physical_array = candidate
            else:
                return None, None, "malformed_center_coordinates"
        except (TypeError, ValueError, OverflowError):
            return None, None, "malformed_center_coordinates"
    if grid is not None:
        try:
            candidate = np.asarray(grid, dtype=np.float64)
            if candidate.shape == (2,):
                grid_array = candidate
            else:
                return physical_array, None, "malformed_center_grid"
        except (TypeError, ValueError, OverflowError):
            return physical_array, None, "malformed_center_grid"
    if physical_array is None and grid_array is None:
        return None, None, "missing_center_coordinates"
    if physical_array is None:
        if not np.all(np.isfinite(grid_array)):
            return None, grid_array, "nonfinite_center_grid"
        physical_array = _grid_to_physical(grid_array, xdim, ydim)
    if grid_array is None and np.all(np.isfinite(physical_array)):
        grid_array = _physical_to_grid(physical_array, xdim, ydim)
    return physical_array, grid_array, None


def _normalize_centers(
    centers: Sequence[Any] | None,
    *,
    xdim: np.ndarray,
    ydim: np.ndarray,
) -> list[dict[str, Any]]:
    if centers is None:
        return []
    if isinstance(centers, Mapping):
        centers = centers.get("centers", centers.get("returned_centers", []))
    if isinstance(centers, (str, bytes)) or not isinstance(centers, Sequence):
        raise SparseAnchorContractError("centers 必须是有序 sequence")
    records: list[dict[str, Any]] = []
    for order, item in enumerate(centers):
        source = _json_copy(item)
        physical, grid, coordinate_error = _center_coordinates(
            item, xdim=xdim, ydim=ydim
        )
        if isinstance(item, Mapping):
            native_id = item.get("native_record_id", item.get("record_id"))
            raw_index = item.get("raw_contour_index", item.get("source_record_index"))
        else:
            native_id = None
            raw_index = None
        record: dict[str, Any] = {
            "record_order": int(order),
            "native_record_id": None if native_id is None else str(native_id),
            "source_record": source,
            "native_physical_xy": (
                None if physical is None else [float(physical[0]), float(physical[1])]
            ),
            "native_grid_rc": (
                None if grid is None else [float(grid[0]), float(grid[1])]
            ),
            "raw_contour_index": (
                None if raw_index is None else int(raw_index)
            ),
            "mapping_status": "pending",
        }
        if coordinate_error is not None:
            record.update({
                "mapping_status": "rejected",
                "reason": coordinate_error,
                "rejection_reason": coordinate_error,
            })
        records.append(record)
    return records


def _raw_contour_lookup(raw_contours: Sequence[Any] | None) -> dict[int, dict[str, Any]]:
    if raw_contours is None:
        return {}
    if isinstance(raw_contours, (str, bytes)) or not isinstance(raw_contours, Sequence):
        raise SparseAnchorContractError("raw_contours 必须是有序 sequence")
    return {
        index: _json_copy(record)
        for index, record in enumerate(raw_contours)
    }


def _raw_contour_mask(
    record: Mapping[str, Any],
    *,
    shape: tuple[int, int],
    xdim: np.ndarray,
    ydim: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    status = str(record.get("status", "valid"))
    if status not in {"valid", "accepted"}:
        return None, "raw_contour_not_legal"
    points = record.get("points_grid")
    if points is None and record.get("points_xy") is not None:
        try:
            points_xy = np.asarray(record["points_xy"], dtype=np.float64)
            if points_xy.ndim != 2 or points_xy.shape[1] != 2:
                return None, "raw_contour_geometry_malformed"
            points = np.column_stack((
                np.interp(points_xy[:, 1], ydim, np.arange(len(ydim))),
                np.interp(points_xy[:, 0], xdim, np.arange(len(xdim))),
            ))
        except (TypeError, ValueError, OverflowError):
            return None, "raw_contour_geometry_malformed"
    if points is None:
        return None, "upstream_raw_contour_geometry_absent"
    try:
        path = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None, "raw_contour_geometry_malformed"
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 3 or not np.all(np.isfinite(path)):
        return None, "raw_contour_geometry_malformed"
    try:
        # Use the same polygon rasterizer as the existing Haller evidence seam.
        import haller_anchors

        mask = np.asarray(haller_anchors.grid_points_in_poly(shape, path), dtype=bool)
    except Exception as exc:
        raise SparseAnchorContractError(
            "legal raw-contour clipping requires the project polygon rasterizer"
        ) from exc
    return mask, "applied"


def _nearest_valid_cell(
    physical_xy: np.ndarray,
    *,
    valid_fluid: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
) -> tuple[int, int, float] | None:
    indices = np.argwhere(valid_fluid)
    if len(indices) == 0:
        return None
    ys = ydim[indices[:, 0]]
    xs = xdim[indices[:, 1]]
    distances = (xs - physical_xy[0]) ** 2 + (ys - physical_xy[1]) ** 2
    minimum = float(np.min(distances))
    ties = np.flatnonzero(np.isclose(distances, minimum, rtol=0.0, atol=1e-15))
    # np.argwhere is row-major, so the first tied entry is the explicit
    # row-then-column tie-break.
    row, col = indices[int(ties[0])]
    return int(row), int(col), float(np.sqrt(minimum))


def _state_counts(state: np.ndarray) -> dict[str, int]:
    counts = Counter(str(item) for item in state.ravel())
    return {name: int(counts.get(name, 0)) for name in SPARSE_STATES}


def _failure_state(
    *,
    solid_mask: np.ndarray,
    invalid_support: np.ndarray,
    valid_fluid: np.ndarray,
    extractor_status: str,
) -> np.ndarray:
    state = np.full(solid_mask.shape, STATE_UNCOVERED, dtype="<U20")
    state[solid_mask] = STATE_SOLID
    state[invalid_support] = STATE_INVALID
    if extractor_status in {"extractor_failure", "invalid_input"}:
        state[valid_fluid] = (
            STATE_EXTRACTOR_FAILURE
            if extractor_status == "extractor_failure"
            else STATE_INVALID
        )
    return state


def build_sparse_anchor(
    centers: Sequence[Any] | None,
    *,
    xdim: Any,
    ydim: Any,
    valid_fluid: Any,
    solid_mask: Any = None,
    invalid_support: Any = None,
    raw_contours: Sequence[Any] | None = None,
    dataset_name: str,
    split_name: str,
    frame_index: int,
    construction: str,
    source_fingerprint: str | None = None,
    split_fingerprint: str | None = None,
    extractor_status: str = "valid",
    failure_reason: str | None = None,
    failure_count: int = 0,
    runtime_provenance: Mapping[str, Any] | None = None,
    config_provenance: Mapping[str, Any] | None = None,
    input_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one auditable positive-only sparse-anchor sample.

    The scope guard intentionally runs before any support/coordinate array is
    converted.  A real relationship caller therefore cannot accidentally
    inspect a test array before the split check rejects it.
    """
    dataset, split, frame = _validate_scope(dataset_name, split_name, frame_index)
    mode = _validate_construction(construction)
    status = _text(extractor_status, name="extractor_status")
    allowed_statuses = {
        "valid",
        "valid_empty",
        "valid_all_negative",
        "all_unknown",
        "extractor_failure",
        "invalid_input",
    }
    if status not in allowed_statuses:
        raise SparseAnchorContractError(
            f"extractor_status={status!r} 不在 {sorted(allowed_statuses)}"
        )
    shape_array = np.asarray(valid_fluid, dtype=bool)
    if shape_array.ndim != 2 or any(int(size) < 1 for size in shape_array.shape):
        raise SparseAnchorContractError("valid_fluid 必须是非空二维数组")
    shape = (int(shape_array.shape[0]), int(shape_array.shape[1]))
    x, y, dx, dy = _coerce_axes(xdim, ydim, shape)
    solid = _coerce_support(solid_mask, shape, name="solid_mask", default=False)
    invalid = _coerce_support(
        invalid_support, shape, name="invalid_support", default=False
    )
    valid = shape_array.copy()
    _validate_support(solid, invalid, valid)
    centers_normalized = _normalize_centers(centers, xdim=x, ydim=y)
    raw_records = [] if raw_contours is None else _json_copy(list(raw_contours))
    raw_lookup = _raw_contour_lookup(raw_records)
    raw_absence = {
        "provided": bool(raw_records),
        "count": int(len(raw_records)),
        "absence_reason": None if raw_records else "upstream_not_provided",
    }

    anchor_mask = np.zeros(shape, dtype=bool)
    rejected_centers: list[dict[str, Any]] = []
    delta = max(dx, dy)
    radius = delta if mode == CONSTRUCTION_SMALL_CORE_R1 else None
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    for record in centers_normalized:
        if status in {"extractor_failure", "invalid_input"}:
            record.update({
                "mapping_status": "rejected",
                "reason": status,
                "rejection_reason": status,
            })
            rejected_centers.append(_json_copy(record))
            continue
        if record.get("mapping_status") == "rejected":
            rejected_centers.append(_json_copy(record))
            continue
        physical_list = record.get("native_physical_xy")
        physical = np.asarray(physical_list, dtype=np.float64)
        if not np.all(np.isfinite(physical)):
            record.update({
                "mapping_status": "rejected",
                "reason": "nonfinite_center_coordinates",
                "rejection_reason": "nonfinite_center_coordinates",
            })
            rejected_centers.append(_json_copy(record))
            continue
        if not (
            x[0] <= physical[0] <= x[-1]
            and y[0] <= physical[1] <= y[-1]
        ):
            record.update({
                "mapping_status": "rejected",
                "reason": "center_out_of_bounds",
                "rejection_reason": "center_out_of_bounds",
            })
            rejected_centers.append(_json_copy(record))
            continue
        mapped = _nearest_valid_cell(
            physical,
            valid_fluid=valid,
            xdim=x,
            ydim=y,
        )
        if mapped is None:
            record.update({
                "mapping_status": "rejected",
                "reason": "no_valid_fluid_support",
                "rejection_reason": "no_valid_fluid_support",
            })
            rejected_centers.append(_json_copy(record))
            continue
        row, col, distance = mapped
        record.update({
            "mapping_status": "accepted",
            "mapped_grid_rc": [int(row), int(col)],
            "mapped_physical_xy": [float(x[col]), float(y[row])],
            "mapping_distance_physical": float(distance),
            "mapping_distance_over_delta": float(distance / delta),
            "support_check": {
                "mapped_cell_valid_fluid": bool(valid[row, col]),
                "mapped_cell_solid": bool(solid[row, col]),
                "mapped_cell_invalid": bool(invalid[row, col]),
            },
        })
        if mode == CONSTRUCTION_CENTER_CELL:
            contribution = np.zeros(shape, dtype=bool)
            contribution[row, col] = True
            record["construction_detail"] = {
                "radius_over_delta": None,
                "contour_clipping": {"applied": False, "reason": "not_applicable"},
            }
        else:
            contribution = (
                (grid_x - physical[0]) ** 2 + (grid_y - physical[1]) ** 2
                <= float(radius) ** 2 + 1e-15
            ) & valid
            contour_index = record.get("raw_contour_index")
            clipping: dict[str, Any]
            if contour_index is None and record["record_order"] in raw_lookup:
                contour_index = record["record_order"]
                record["raw_contour_index"] = int(contour_index)
            if contour_index is None:
                clipping = {"applied": False, "reason": "corresponding_raw_contour_absent"}
            elif int(contour_index) not in raw_lookup:
                clipping = {"applied": False, "reason": "corresponding_raw_contour_absent"}
            else:
                contour_mask, clipping_reason = _raw_contour_mask(
                    raw_lookup[int(contour_index)],
                    shape=shape,
                    xdim=x,
                    ydim=y,
                )
                if contour_mask is None:
                    clipping = {"applied": False, "reason": clipping_reason}
                else:
                    contribution &= contour_mask
                    clipping = {
                        "applied": True,
                        "raw_contour_index": int(contour_index),
                        "reason": "legal_raw_contour_interior",
                    }
            record["construction_detail"] = {
                "radius_over_delta": 1.0,
                "radius_physical": float(radius),
                "contour_clipping": clipping,
            }
        contribution &= valid
        before = int(np.count_nonzero(anchor_mask))
        contribution_count = int(np.count_nonzero(contribution))
        anchor_mask |= contribution
        after = int(np.count_nonzero(anchor_mask))
        record.update({
            "contribution_cell_count": contribution_count,
            "new_anchor_cell_count": int(after - before),
            "overlap_cell_count": int(contribution_count - (after - before)),
        })

    if status in {"extractor_failure", "invalid_input"}:
        anchor_mask.fill(False)
        for record in centers_normalized:
            if record.get("mapping_status") == "pending":
                record.update({
                    "mapping_status": "rejected",
                    "reason": status,
                    "rejection_reason": status,
                })
                rejected_centers.append(_json_copy(record))
    if status in {"valid_empty", "valid_all_negative", "all_unknown"} and centers_normalized:
        # A declared no-vortex frame cannot also claim returned centers.
        raise SparseAnchorContractError(
            f"extractor_status={status} 不能携带 returned centers"
        )

    sparse_state = _failure_state(
        solid_mask=solid,
        invalid_support=invalid,
        valid_fluid=valid,
        extractor_status=status,
    )
    sparse_state[anchor_mask] = STATE_POSITIVE_ANCHOR
    # Support priority is explicit even if a caller supplied inconsistent mask.
    sparse_state[solid] = STATE_SOLID
    sparse_state[invalid] = STATE_INVALID
    if status == "extractor_failure":
        sparse_state[valid] = STATE_EXTRACTOR_FAILURE
    elif status == "invalid_input":
        sparse_state[valid] = STATE_INVALID
    else:
        sparse_state[anchor_mask] = STATE_POSITIVE_ANCHOR

    if int(failure_count) != failure_count or int(failure_count) < 0:
        raise SparseAnchorContractError("failure_count 必须是非负整数")
    failure_count_int = int(failure_count)
    failure_reasons = [] if failure_reason is None else [_text(failure_reason, name="failure_reason")]
    if status in {"extractor_failure", "invalid_input"} and failure_count_int == 0:
        failure_count_int = 1
    if status == "valid_empty" and not failure_reasons:
        failure_reasons = ["no_vortex_evidence"]

    input_arrays = {
        "xdim": x,
        "ydim": y,
        "solid_mask": solid,
        "invalid_support": invalid,
        "valid_fluid": valid,
    }
    inferred_source = _hash_named_arrays(input_arrays)
    resolved_source = (
        inferred_source if source_fingerprint is None else _text(
            source_fingerprint, name="source_fingerprint"
        )
    )
    resolved_split = (
        _json_hash({"dataset_name": dataset, "split_name": split, "frame_index": frame})
        if split_fingerprint is None
        else _text(split_fingerprint, name="split_fingerprint")
    )
    resolved_runtime = _json_copy(runtime_provenance or {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    })
    resolved_config = _json_copy(config_provenance or {
        "construction": mode,
        "radius_factor": None if radius is None else 1.0,
        "contour_clipping": "legal_raw_contour_only_for_small_core_r1",
    })
    resolved_input = _json_copy(input_provenance or {})
    for record in centers_normalized:
        record["sample_identity"] = {
            "dataset_name": dataset,
            "split_name": split,
            "frame_index": frame,
            "source_fingerprint": resolved_source,
            "split_fingerprint": resolved_split,
        }
        record["extractor_provenance"] = {
            "status": status,
            "failure_reason": failure_reasons[0] if failure_reasons else None,
            "failure_reasons": list(failure_reasons),
            "failure_count": failure_count_int,
        }
        record["runtime_provenance"] = _json_copy(resolved_runtime)
        record["config_provenance"] = _json_copy(resolved_config)
        record["input_provenance"] = _json_copy(resolved_input)
    cell_area = float(dx * dy)
    metadata: dict[str, Any] = {
        "artifact_type": SAMPLE_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "dataset_name": dataset,
        "split_name": split,
        "frame_index": frame,
        "source_fingerprint": resolved_source,
        "split_fingerprint": resolved_split,
        "construction": mode,
        "registered_constructions": list(REGISTERED_CONSTRUCTIONS),
        "delta_definition": "max(median_dx, median_dy)",
        "delta_physical": float(delta),
        "radius_physical": None if radius is None else float(radius),
        "radius_over_delta": None if radius is None else 1.0,
        "mapping_algorithm": "nearest_valid_fluid_euclidean_physical",
        "mapping_tie_break": "first_row_then_column_in_row_major_valid_support",
        "support_contract": {
            "valid_fluid": "~solid_mask & ~invalid_support",
            "solid_state": STATE_SOLID,
            "invalid_state": STATE_INVALID,
            "positive_anchor_support": "valid_fluid_only",
        },
        "extractor_status": status,
        "failure_count": failure_count_int,
        "failure_reasons": failure_reasons,
        "center_count": int(len(centers_normalized)),
        "accepted_center_count": int(sum(
            record.get("mapping_status") == "accepted"
            for record in centers_normalized
        )),
        "rejected_center_count": int(len(rejected_centers)),
        "rejected_centers": rejected_centers,
        "anchor_count": int(np.count_nonzero(anchor_mask)),
        "anchor_area": float(np.count_nonzero(anchor_mask) * cell_area),
        "valid_fluid_count": int(np.count_nonzero(valid)),
        "valid_fluid_area": float(np.count_nonzero(valid) * cell_area),
        "anchor_coverage_valid_fluid": (
            float(np.count_nonzero(anchor_mask) / np.count_nonzero(valid))
            if np.any(valid) else None
        ),
        "state_counts": _state_counts(sparse_state),
        "geometry": {
            "coordinate_order": "x_then_y",
            "array_order": "row_y_then_col_x",
            "shape": [int(shape[0]), int(shape[1])],
            "xdim": x.tolist(),
            "ydim": y.tolist(),
            "dx": float(dx),
            "dy": float(dy),
            "xdim_fingerprint": _hash_array(x),
            "ydim_fingerprint": _hash_array(y),
        },
        "raw_contour_provenance": raw_absence,
        "runtime_provenance": resolved_runtime,
        "config_provenance": resolved_config,
        "input_provenance": resolved_input,
        "source_and_split_identity": {
            "source_fingerprint": resolved_source,
            "split_fingerprint": resolved_split,
            "dataset_name": dataset,
            "split_name": split,
            "frame_index": frame,
        },
        "state_encoding": {
            "states": list(SPARSE_STATES),
            "uncovered_is_not_negative": True,
            "hard_negative_state": None,
        },
        "final_test_relationships": "not_read",
        "boussinesq_frame_1300": "excluded",
        "array_fingerprints": {
            "solid_mask": _hash_array(solid),
            "invalid_support": _hash_array(invalid),
            "valid_fluid": _hash_array(valid),
            "anchor_mask": _hash_array(anchor_mask),
            "sparse_state": _hash_array(sparse_state),
        },
    }
    payload_for_integrity = {
        "metadata": metadata,
        "centers": centers_normalized,
        "raw_contours": raw_records,
    }
    metadata["input_fingerprint"] = {
        "arrays": {name: _hash_array(value) for name, value in input_arrays.items()},
        "combined": _hash_named_arrays(input_arrays),
    }
    metadata["raw_contours_fingerprint"] = _json_hash(raw_records)
    metadata["integrity_fingerprint"] = _json_hash(payload_for_integrity)
    return {
        "anchor_mask": anchor_mask,
        "sparse_state": sparse_state,
        "valid_fluid": valid,
        "solid_mask": solid,
        "invalid_support": invalid,
        "center_records": centers_normalized,
        "raw_contours": raw_records,
        "metadata": metadata,
    }


def build_sparse_anchor_from_evidence(
    evidence: Mapping[str, Any],
    *,
    construction: str,
    source_fingerprint: str | None = None,
    split_fingerprint: str | None = None,
    input_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapt the Ticket 03/evidence-tracer mapping to the sparse seam."""
    metadata = evidence.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise SparseAnchorContractError("evidence.metadata 必须是 mapping")
    centers = evidence.get("center_records")
    if centers is None:
        centers = [
            {**_json_copy(record), "raw_contour_index": int(index)}
            for index, record in enumerate(evidence.get("raw_contours", []))
            if isinstance(record, Mapping)
            and record.get("status") in {"valid", "accepted"}
            and (record.get("center_xy") is not None or record.get("center_physical_xy") is not None)
        ]
    return build_sparse_anchor(
        centers,
        xdim=evidence["xdim"],
        ydim=evidence["ydim"],
        valid_fluid=evidence.get("valid_fluid"),
        solid_mask=evidence.get("solid_mask"),
        invalid_support=evidence.get("invalid_support"),
        raw_contours=evidence.get("raw_contours"),
        dataset_name=metadata["dataset_name"],
        split_name=metadata["split_name"],
        frame_index=metadata["frame_index"],
        construction=construction,
        source_fingerprint=source_fingerprint or metadata.get("input_fingerprint", {}).get("combined"),
        split_fingerprint=split_fingerprint or metadata.get("protocol", {}).get("manifest_hash"),
        extractor_status=metadata.get("status", "valid"),
        failure_reason=(
            metadata.get("status_reason")
            if metadata.get("status") in {"extractor_failure", "invalid_input"}
            else None
        ),
        failure_count=metadata.get("failure_count", 0),
        runtime_provenance=metadata.get("runtime_identity"),
        config_provenance=metadata.get("parameters"),
        input_provenance=input_provenance or metadata.get("input_fingerprint"),
    )


@dataclass(frozen=True)
class ControlledFixture:
    """One exact physical case used only by the controlled panel."""

    name: str
    xdim: np.ndarray
    ydim: np.ndarray
    exact_positive_mask: np.ndarray
    vortex_masks: tuple[np.ndarray, ...]
    true_centers_xy: tuple[tuple[float, float], ...]
    centers: tuple[Mapping[str, Any], ...]
    raw_contours: tuple[Mapping[str, Any], ...]
    solid_mask: np.ndarray
    invalid_support: np.ndarray
    extractor_status: str
    failure_reason: str | None = None
    failure_count: int = 0


def _disc_mask(
    xdim: np.ndarray,
    ydim: np.ndarray,
    center_xy: tuple[float, float],
    radius: float,
) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(xdim, ydim, indexing="xy")
    return (
        (grid_x - float(center_xy[0])) ** 2
        + (grid_y - float(center_xy[1])) ** 2
        <= float(radius) ** 2 + 1e-15
    )


def _circle_polygon(
    center_xy: tuple[float, float],
    radius: float,
    *,
    points: int = 64,
) -> list[list[float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    values = np.column_stack((
        float(center_xy[0]) + float(radius) * np.cos(angles),
        float(center_xy[1]) + float(radius) * np.sin(angles),
    ))
    closed = np.vstack((values, values[0]))
    return [[float(x), float(y)] for x, y in closed]


def build_controlled_fixtures(
    resolutions: Sequence[int] = (65, 97, 129),
) -> list[ControlledFixture]:
    """Return the pre-registered exact/synthetic fixture panel.

    Physical centres and vortex radii stay fixed while the grid resolution
    changes.  ``malformed_input`` and ``forced_extractor_failure`` are
    contract cases, so their expected failures are classified rather than
    treated as unexpected panel failures.
    """
    if not resolutions or any(int(size) < 9 or int(size) != size for size in resolutions):
        raise SparseAnchorContractError("controlled resolutions 必须是至少 9 点的整数序列")
    fixtures: list[ControlledFixture] = []
    for raw_size in resolutions:
        size = int(raw_size)
        xdim = np.linspace(-8.0, 8.0, size, dtype=np.float64)
        ydim = np.linspace(-8.0, 8.0, size, dtype=np.float64)
        shape = (size, size)
        empty = np.zeros(shape, dtype=bool)

        cases = [
            ("no_vortex", (), (), (), "valid_empty", "no_vortex_evidence", 0),
            (
                "single_vortex",
                ((0.0, 0.0),),
                (2.0,),
                (2.5,),
                "valid",
                None,
                0,
            ),
            (
                "multi_vortex",
                ((-3.0, 0.0), (3.0, 0.0)),
                (1.6, 1.6),
                (2.0, 2.0),
                "valid",
                None,
                0,
            ),
            (
                "nearby_independent_vortices",
                ((-1.8, 0.0), (1.8, 0.0)),
                (1.15, 1.15),
                (1.45, 1.45),
                "valid",
                None,
                0,
            ),
            (
                "solid_intersection",
                ((0.25, 0.0),),
                (1.6,),
                (1.9,),
                "valid",
                None,
                0,
            ),
            ("malformed_input", (), (), (), "valid", None, 0),
            (
                "forced_extractor_failure",
                (),
                (),
                (),
                "extractor_failure",
                "forced_extractor_failure",
                1,
            ),
        ]
        for name, centers_xy, radii, contour_radii, status, reason, failure_count in cases:
            solid = empty.copy()
            invalid = empty.copy()
            if name == "solid_intersection":
                solid = _disc_mask(xdim, ydim, (0.0, 0.0), 0.5)
            vortex_masks = tuple(
                _disc_mask(xdim, ydim, center, radius)
                for center, radius in zip(centers_xy, radii)
            )
            exact = np.zeros(shape, dtype=bool)
            for mask in vortex_masks:
                exact |= mask
            if name == "malformed_input":
                centers: tuple[Mapping[str, Any], ...] = (
                    {"native_record_id": "missing-center"},
                )
                raw: tuple[Mapping[str, Any], ...] = ()
            else:
                centers = tuple(
                    {
                        "native_record_id": f"{name}-{index}",
                        "center_xy": [float(center[0]), float(center[1])],
                        "raw_contour_index": int(index),
                    }
                    for index, center in enumerate(centers_xy)
                )
                raw = tuple(
                    {
                        "status": "valid",
                        "center_xy": [float(center[0]), float(center[1])],
                        "points_xy": _circle_polygon(center, contour_radius),
                    }
                    for center, contour_radius in zip(centers_xy, contour_radii)
                )
            fixtures.append(ControlledFixture(
                name=name,
                xdim=xdim.copy(),
                ydim=ydim.copy(),
                exact_positive_mask=exact,
                vortex_masks=vortex_masks,
                true_centers_xy=tuple((float(x), float(y)) for x, y in centers_xy),
                centers=centers,
                raw_contours=raw,
                solid_mask=solid,
                invalid_support=invalid,
                extractor_status=status,
                failure_reason=reason,
                failure_count=failure_count,
            ))
    # Preserve the fixture order while grouping all resolutions in the report.
    return fixtures


def _fixture_names() -> list[str]:
    return [
        "no_vortex",
        "single_vortex",
        "multi_vortex",
        "nearby_independent_vortices",
        "solid_intersection",
        "malformed_input",
        "forced_extractor_failure",
    ]


def _qualification_run_metrics(
    result: Mapping[str, Any],
    fixture: ControlledFixture,
) -> dict[str, Any]:
    metadata = result["metadata"]
    anchor_mask = np.asarray(result["anchor_mask"], dtype=bool)
    valid = np.asarray(result["valid_fluid"], dtype=bool)
    exact = np.asarray(fixture.exact_positive_mask, dtype=bool) & valid
    anchor_count = int(np.count_nonzero(anchor_mask))
    exact_positive_count = int(np.count_nonzero(anchor_mask & exact))
    precision = (
        float(exact_positive_count / anchor_count)
        if anchor_count
        else (1.0 if not np.any(exact) else 0.0)
    )
    records = [
        record for record in result["center_records"]
        if record.get("mapping_status") == "accepted"
    ]
    center_errors: list[float] = []
    for record in records:
        if not fixture.true_centers_xy:
            continue
        point = np.asarray(record["native_physical_xy"], dtype=np.float64)
        center_errors.append(float(min(
            np.linalg.norm(point - np.asarray(expected, dtype=np.float64))
            for expected in fixture.true_centers_xy
        )))
    max_center_error = max(center_errors, default=None)
    per_vortex = []
    for index, (center, vortex_mask) in enumerate(
        zip(fixture.true_centers_xy, fixture.vortex_masks)
    ):
        hit_count = int(np.count_nonzero(anchor_mask & vortex_mask & valid))
        per_vortex.append({
            "vortex_index": int(index),
            "center_xy": [float(center[0]), float(center[1])],
            "anchor_hit": bool(hit_count > 0),
            "anchor_hit_count": hit_count,
            "anchor_recall": float(hit_count > 0),
        })
    no_vortex_false = int(
        np.count_nonzero(anchor_mask & ~exact)
        if fixture.name == "no_vortex" else 0
    )
    observed_status = str(metadata["extractor_status"])
    if fixture.name == "forced_extractor_failure":
        classification_pass = (
            observed_status == "extractor_failure"
            and int(metadata["failure_count"]) == fixture.failure_count
            and anchor_count == 0
        )
    elif fixture.name == "malformed_input":
        classification_pass = (
            observed_status == fixture.extractor_status
            and int(metadata["accepted_center_count"]) == 0
            and int(metadata["rejected_center_count"]) == 1
        )
    else:
        classification_pass = observed_status == fixture.extractor_status
    delta = float(metadata["delta_physical"])
    center_error_pass = (
        max_center_error is None
        or (
            bool(fixture.true_centers_xy)
            and len(records) >= len(fixture.true_centers_xy)
            and max_center_error <= CENTER_ERROR_GATE_MULTIPLIER * delta + 1e-12
        )
    )
    return {
        "fixture": fixture.name,
        "resolution": int(len(fixture.xdim)),
        "status": observed_status,
        "status_reason": metadata.get("failure_reasons", []),
        "failure_count": int(metadata.get("failure_count", 0)),
        "failure_reasons": list(metadata.get("failure_reasons", [])),
        "center_error": max_center_error,
        "center_error_over_delta": (
            None if max_center_error is None else float(max_center_error / delta)
        ),
        "anchor_precision": precision,
        "exact_positive_anchor_count": exact_positive_count,
        "anchor_count": anchor_count,
        "anchor_area": float(metadata["anchor_area"]),
        "anchor_coverage": metadata["anchor_coverage_valid_fluid"],
        "per_vortex": per_vortex,
        "per_vortex_anchor_recall": (
            float(sum(item["anchor_hit"] for item in per_vortex) / len(per_vortex))
            if per_vortex else None
        ),
        "no_vortex_false_anchor_count": no_vortex_false,
        "classification_pass": bool(classification_pass),
        "anchor_precision_gate": bool(precision >= ANCHOR_PRECISION_GATE),
        "center_error_gate": bool(center_error_pass),
        "raw_contour_count": int(len(result["raw_contours"])),
        "raw_contour_presence": bool(result["raw_contours"]),
        "construction": metadata["construction"],
    }


def _resolution_consistency(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_fixture: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        by_fixture[str(run["fixture"])].append(run)
    details: dict[str, Any] = {}
    for fixture, entries in by_fixture.items():
        ordered = sorted(entries, key=lambda item: int(item["resolution"]))
        errors = [item["center_error_over_delta"] for item in ordered if item["center_error_over_delta"] is not None]
        details[fixture] = {
            "resolutions": [int(item["resolution"]) for item in ordered],
            "statuses": [str(item["status"]) for item in ordered],
            "anchor_precision": [float(item["anchor_precision"]) for item in ordered],
            "anchor_count": [int(item["anchor_count"]) for item in ordered],
            "anchor_area": [float(item["anchor_area"]) for item in ordered],
            "center_error_over_delta": errors,
            "status_consistent": len(set(item["status"] for item in ordered)) <= 1,
            "all_center_error_gates_pass": all(bool(item["center_error_gate"]) for item in ordered),
            "definition": "same physical centres/radii reconstructed at each declared grid resolution",
        }
    return details


def _candidate_qualification_summary(
    construction: str,
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchor_runs = [run for run in runs if int(run["anchor_count"]) > 0]
    no_vortex_runs = [run for run in runs if run["fixture"] == "no_vortex"]
    failures = sum(not bool(run["classification_pass"]) for run in runs)
    minimum_precision = min(
        (float(run["anchor_precision"]) for run in anchor_runs),
        default=1.0,
    )
    maximum_center_error_over_delta = max(
        (
            float(run["center_error_over_delta"])
            for run in runs
            if run["center_error_over_delta"] is not None
        ),
        default=0.0,
    )
    gates = {
        "anchor_precision": bool(
            all(bool(run["anchor_precision_gate"]) for run in anchor_runs)
            and minimum_precision >= ANCHOR_PRECISION_GATE
        ),
        "center_error": bool(
            all(bool(run["center_error_gate"]) for run in runs)
            and maximum_center_error_over_delta <= CENTER_ERROR_GATE_MULTIPLIER + 1e-12
        ),
        "no_vortex_false_anchor_count": bool(
            all(
                int(run["no_vortex_false_anchor_count"]) == NO_VORTEX_FALSE_ANCHOR_GATE
                for run in no_vortex_runs
            )
        ),
        "controlled_fixture_failure_rate": bool(
            failures / len(runs) <= CONTROLLED_FAILURE_RATE_GATE
            if runs else False
        ),
    }
    return {
        "construction": construction,
        "status": "qualified" if all(gates.values()) else "unqualified",
        "resolution_runs": list(runs),
        "resolution_consistency": _resolution_consistency(runs),
        "gates": gates,
        "gate_thresholds": {
            "anchor_precision": ANCHOR_PRECISION_GATE,
            "center_error_over_delta": CENTER_ERROR_GATE_MULTIPLIER,
            "no_vortex_false_anchor_count": NO_VORTEX_FALSE_ANCHOR_GATE,
            "controlled_fixture_failure_rate": CONTROLLED_FAILURE_RATE_GATE,
        },
        "summary": {
            "run_count": int(len(runs)),
            "minimum_anchor_precision": float(minimum_precision),
            "maximum_center_error_over_delta": float(maximum_center_error_over_delta),
            "unexpected_classification_failure_count": int(failures),
            "controlled_fixture_failure_rate": float(failures / len(runs)) if runs else None,
            "no_vortex_false_anchor_count_max": max(
                (int(run["no_vortex_false_anchor_count"]) for run in no_vortex_runs),
                default=0,
            ),
        },
    }


def qualify_sparse_anchor_constructions(
    *,
    resolutions: Sequence[int] = (65, 97, 129),
) -> dict[str, Any]:
    """Run the controlled-only qualification and select one construction.

    The function has no real-data input by design.  Its selection order is
    fixed by the protocol: ``small_core_r1`` first, then ``center_cell``.
    """
    if isinstance(resolutions, (str, bytes)) or not isinstance(resolutions, Sequence):
        raise SparseAnchorContractError("qualification resolutions 必须是 sequence")
    normalised_resolutions = []
    for value in resolutions:
        if isinstance(value, (bool, np.bool_)):
            raise SparseAnchorContractError("qualification resolution 必须是正整数")
        try:
            resolution = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SparseAnchorContractError("qualification resolution 必须是正整数") from exc
        if resolution != value or resolution < 3:
            raise SparseAnchorContractError("qualification resolution 必须是至少 3 的整数")
        normalised_resolutions.append(resolution)
    if len(normalised_resolutions) < 3 or len(set(normalised_resolutions)) != len(normalised_resolutions):
        raise SparseAnchorContractError(
            "controlled qualification 必须在至少三个不同空间分辨率上运行"
        )
    resolutions = tuple(normalised_resolutions)
    fixtures = build_controlled_fixtures(resolutions)
    by_construction: dict[str, list[dict[str, Any]]] = {
        construction: [] for construction in REGISTERED_CONSTRUCTIONS
    }
    for construction in REGISTERED_CONSTRUCTIONS:
        for fixture in fixtures:
            result = build_sparse_anchor(
                fixture.centers,
                xdim=fixture.xdim,
                ydim=fixture.ydim,
                valid_fluid=~fixture.solid_mask & ~fixture.invalid_support,
                solid_mask=fixture.solid_mask,
                invalid_support=fixture.invalid_support,
                raw_contours=fixture.raw_contours,
                dataset_name=f"synthetic_{fixture.name}",
                split_name="controlled",
                frame_index=int(len(fixture.xdim)),
                construction=construction,
                source_fingerprint=_json_hash({
                    "fixture": fixture.name,
                    "resolution": int(len(fixture.xdim)),
                    "source": "controlled_exact_fixture",
                }),
                split_fingerprint=_json_hash({
                    "fixture": fixture.name,
                    "resolution": int(len(fixture.xdim)),
                    "split": "controlled",
                }),
                extractor_status=fixture.extractor_status,
                failure_reason=fixture.failure_reason,
                failure_count=fixture.failure_count,
                runtime_provenance={
                    "fixture_generator": "sparse_anchor_study.build_controlled_fixtures",
                    "python": sys.version,
                    "numpy": np.__version__,
                },
                config_provenance={
                    "panel": "ticket05-A-controlled-only",
                    "exact_positive_mask": "independent_disc_fixture",
                },
                input_provenance={"source": "synthetic_exact_fixture"},
            )
            by_construction[construction].append(
                _qualification_run_metrics(result, fixture)
            )
    candidates = {
        construction: _candidate_qualification_summary(construction, runs)
        for construction, runs in by_construction.items()
    }
    selected = None
    for construction in SELECTION_ORDER:
        if candidates[construction]["status"] == "qualified":
            selected = construction
            break
    report: dict[str, Any] = {
        "artifact_type": QUALIFICATION_REPORT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "status": (
            QUALIFICATION_STATUS_QUALIFIED
            if selected is not None else QUALIFICATION_STATUS_NO_GO
        ),
        "selected_construction": selected,
        "fixture_names": _fixture_names(),
        "resolutions": [int(size) for size in resolutions],
        "registered_constructions": list(REGISTERED_CONSTRUCTIONS),
        "candidates": candidates,
        "selection": {
            "order": list(SELECTION_ORDER),
            "rule": "small_core_r1-if-all-gates-pass-else-center_cell-if-all-gates-pass-else-no-go",
            "basis": "controlled_exact_synthetic_only",
            "real_data_used": False,
            "calibration_used": False,
            "test_used": False,
            "model_results_used": False,
            "selected_construction": selected,
        },
        "test_access_guard": {
            "status": "closed",
            "final_test_relationships": "not_read",
            "frame_1300_excluded": True,
        },
        "haller_semantics": {
            "states": list(SPARSE_STATES),
            "uncovered_is_not_negative": True,
            "hard_negative_state": None,
            "raw_contours": "preserved_as_input; optional clipping only",
        },
        "literature": {
            "haller_zotero_item_key": "L2PX3NQX",
            "scope": "IVD definition and conceptual extraction skeleton only",
            "project_radius_is_engineering_choice": True,
            "numbacs_zotero_item": None,
            "numbacs_claim_status": "unverified_local_zotero_entry_unavailable",
        },
    }
    report["qualification_report_fingerprint"] = _json_hash(report)
    return report


def _load_json_or_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, (str, pathlib.Path)):
        path = pathlib.Path(value)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SparseAnchorContractError(f"{name} 不是可读 JSON artifact") from exc
    else:
        loaded = value
    if not isinstance(loaded, Mapping):
        raise SparseAnchorContractError(f"{name} 必须是 JSON object/mapping")
    return _json_copy(dict(loaded))


def _validate_qualification_report(value: Any) -> dict[str, Any]:
    report = _load_json_or_mapping(value, name="qualification_report")
    if report.get("artifact_type") != QUALIFICATION_REPORT_TYPE:
        raise SparseAnchorContractError("qualification report artifact_type 不匹配")
    if report.get("protocol_version") != PROTOCOL_VERSION:
        raise SparseAnchorContractError("qualification report protocol_version 不匹配")
    declared = report.get("qualification_report_fingerprint")
    payload = _json_copy(report)
    payload.pop("qualification_report_fingerprint", None)
    if not isinstance(declared, str) or _json_hash(payload) != declared:
        raise SparseAnchorContractError("qualification report fingerprint 校验失败")
    if report.get("status") != QUALIFICATION_STATUS_QUALIFIED:
        raise SparseAnchorContractError(
            "只有 status=qualified 的 controlled qualification 才能启动 relationship phase"
        )
    selected = report.get("selected_construction")
    if selected not in REGISTERED_CONSTRUCTIONS:
        raise SparseAnchorContractError("qualification report selected_construction 无效")
    selected_candidate = report.get("candidates", {}).get(selected)
    if not isinstance(selected_candidate, Mapping):
        raise SparseAnchorContractError("qualification report 缺少 selected candidate")
    if selected_candidate.get("status") != QUALIFICATION_STATUS_QUALIFIED:
        raise SparseAnchorContractError("selected construction 的所有 controlled gates 未通过")
    gates = selected_candidate.get("gates", {})
    required_gates = (
        "anchor_precision",
        "center_error",
        "no_vortex_false_anchor_count",
        "controlled_fixture_failure_rate",
    )
    if any(gates.get(name) is not True for name in required_gates):
        raise SparseAnchorContractError("qualification report gate 状态不完整或未通过")
    selection = report.get("selection", {})
    if (
        selection.get("real_data_used") is not False
        or selection.get("calibration_used") is not False
        or selection.get("test_used") is not False
        or selection.get("model_results_used") is not False
    ):
        raise SparseAnchorContractError("qualification selection 不是 controlled-only")
    if tuple(selection.get("order", ())) != SELECTION_ORDER:
        raise SparseAnchorContractError("qualification selection order 被修改")
    access = report.get("test_access_guard", {})
    if access.get("final_test_relationships") != "not_read":
        raise FinalTestAccessError("qualification report 没有封存 final-test relationships")
    return report


def _protocol_manifest_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, pathlib.Path)):
        try:
            import experiment_protocol

            return experiment_protocol.read_protocol_manifest(value)
        except Exception as exc:
            raise SparseAnchorContractError("protocol_manifest 无法通过项目 validator") from exc
    try:
        import experiment_protocol

        return experiment_protocol.validate_protocol_manifest(value)
    except Exception as exc:
        raise SparseAnchorContractError("protocol_manifest 无法通过项目 validator") from exc


def _protocol_sample_identity(
    manifest: Mapping[str, Any] | None,
    *,
    dataset_name: str,
    split_name: str,
    frame_index: int,
) -> dict[str, Any]:
    """Validate a real sample against the frozen R0 frame manifest.

    This helper only examines identity metadata and frame ranges.  It is
    deliberately called before a percentile array is converted or loaded.
    """
    if manifest is None:
        if dataset_name in REAL_DATASET_NAMES:
            raise SparseAnchorContractError(
                "六数据集 relationship sample 必须绑定已验证的 protocol_manifest"
            )
        return {
            "source_fingerprint": None,
            "split_fingerprint": None,
            "manifest_fingerprint": None,
            "manifest_bound": False,
        }
    datasets = manifest.get("datasets", {})
    if dataset_name not in datasets:
        raise SparseAnchorContractError(
            f"dataset={dataset_name!r} 不在 protocol manifest 六数据集集合"
        )
    dataset_manifest = datasets[dataset_name]
    split_manifest = dataset_manifest["frame_partitions"].get(split_name)
    if split_manifest is None:
        raise SparseAnchorContractError(
            f"dataset={dataset_name} 缺少 split={split_name} 的 protocol partition"
        )
    ranges = split_manifest["ranges"]
    if not any(int(start) <= frame_index < int(end) for start, end in ranges):
        raise SparseAnchorContractError(
            f"dataset={dataset_name} frame={frame_index} 不在 {split_name} manifest"
        )
    source_identity = dataset_manifest["source_identity"]
    source_fingerprint = _text(
        source_identity.get("fingerprint"), name="source_identity.fingerprint"
    )
    split_fingerprint = _json_hash({
        "protocol_manifest_hash": manifest["manifest_hash"],
        "dataset_name": dataset_name,
        "split_name": split_name,
        "ranges": ranges,
        "window": dataset_manifest["window"],
    })
    return {
        "source_fingerprint": source_fingerprint,
        "split_fingerprint": split_fingerprint,
        "manifest_fingerprint": manifest["manifest_hash"],
        "manifest_bound": True,
        "split_ranges": _json_copy(ranges),
    }


def _relationship_ratio(count: int, denominator: int) -> float | None:
    return float(count / denominator) if denominator else None


def _relationship_frame(
    sample: Mapping[str, Any],
    *,
    protocol_manifest: Mapping[str, Any] | None,
    real_relationship: bool,
) -> dict[str, Any]:
    """Compute one frame row after identity guards have run."""
    if not isinstance(sample, Mapping):
        raise SparseAnchorContractError("relationship sample 必须是 mapping")
    anchor_result = sample.get("anchor_result")
    anchor_metadata = (
        anchor_result.get("metadata", {})
        if isinstance(anchor_result, Mapping)
        else {}
    )
    if not isinstance(anchor_metadata, Mapping):
        raise SparseAnchorContractError("anchor_result.metadata 必须是 mapping")
    dataset_value = sample.get("dataset_name", anchor_metadata.get("dataset_name"))
    split_value = sample.get("split_name", anchor_metadata.get("split_name"))
    frame_value = sample.get("frame_index", anchor_metadata.get("frame_index"))
    # This is the first operation that establishes scope.  In particular,
    # do not access sample["percentile_field"] before this point.
    dataset, split, frame = _validate_scope(
        dataset_value,
        split_value,
        frame_value,
        real_relationship=real_relationship,
    )
    identity = _protocol_sample_identity(
        protocol_manifest,
        dataset_name=dataset,
        split_name=split,
        frame_index=frame,
    )
    for field_name, expected in (
        ("dataset_name", dataset),
        ("split_name", split),
        ("frame_index", frame),
    ):
        if field_name in anchor_metadata and anchor_metadata[field_name] != expected:
            raise SparseAnchorContractError(
                f"anchor_result {field_name} 与 relationship sample identity 不一致"
            )
    if anchor_result is None:
        required = ("centers", "xdim", "ydim", "valid_fluid")
        missing = [name for name in required if name not in sample]
        if missing:
            raise SparseAnchorContractError(
                f"relationship sample 缺少 anchor_result 或 build 输入={missing}"
            )
        anchor_result = build_sparse_anchor(
            sample["centers"],
            xdim=sample["xdim"],
            ydim=sample["ydim"],
            valid_fluid=sample["valid_fluid"],
            solid_mask=sample.get("solid_mask"),
            invalid_support=sample.get("invalid_support"),
            raw_contours=sample.get("raw_contours"),
            dataset_name=dataset,
            split_name=split,
            frame_index=frame,
            construction=sample.get("construction", CONSTRUCTION_CENTER_CELL),
            source_fingerprint=sample.get("source_fingerprint") or identity["source_fingerprint"],
            split_fingerprint=sample.get("split_fingerprint") or identity["split_fingerprint"],
            extractor_status=sample.get("extractor_status", "valid"),
            failure_reason=sample.get("failure_reason"),
            failure_count=sample.get("failure_count", 0),
            runtime_provenance=sample.get("runtime_provenance"),
            config_provenance=sample.get("config_provenance"),
            input_provenance=sample.get("input_provenance"),
        )
        anchor_metadata = anchor_result["metadata"]
    else:
        if not isinstance(anchor_result, Mapping):
            raise SparseAnchorContractError("anchor_result 必须是 mapping")
    metadata = anchor_result.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise SparseAnchorContractError("anchor_result.metadata 必须是 mapping")
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise SparseAnchorContractError("anchor_result protocol_version 不匹配")
    construction = metadata.get("construction")
    if construction not in REGISTERED_CONSTRUCTIONS:
        raise SparseAnchorContractError("relationship anchor construction 未注册")
    if identity["source_fingerprint"] is not None:
        if metadata.get("source_fingerprint") != identity["source_fingerprint"]:
            raise SparseAnchorContractError(
                "anchor source_fingerprint 与 R0 source_identity 不一致"
            )
        if metadata.get("split_fingerprint") != identity["split_fingerprint"]:
            raise SparseAnchorContractError(
                "anchor split_fingerprint 与 R0 split partition 不一致"
            )
    source_fingerprint = sample.get("source_fingerprint", metadata.get("source_fingerprint"))
    split_fingerprint = sample.get("split_fingerprint", metadata.get("split_fingerprint"))
    if identity["source_fingerprint"] is not None:
        if source_fingerprint not in (None, identity["source_fingerprint"]):
            raise SparseAnchorContractError("sample source_fingerprint 与 R0 不一致")
        if split_fingerprint not in (None, identity["split_fingerprint"]):
            raise SparseAnchorContractError("sample split_fingerprint 与 R0 不一致")
        source_fingerprint = identity["source_fingerprint"]
        split_fingerprint = identity["split_fingerprint"]

    # Only now are array-like payloads accessed.  The test guard above is
    # therefore observable even for lazy objects whose __array__ explodes.
    arrays = _sample_arrays_from_result({
        "anchor_mask": anchor_result.get("anchor_mask"),
        "sparse_state": anchor_result.get("sparse_state"),
        "valid_fluid": anchor_result.get("valid_fluid"),
        "solid_mask": anchor_result.get("solid_mask"),
        "invalid_support": anchor_result.get("invalid_support"),
    })
    percentile_value = sample.get("percentile_field", sample.get("ivd"))
    if percentile_value is None:
        raise SparseAnchorContractError("relationship sample 缺少 percentile_field")
    percentile = np.asarray(percentile_value, dtype=np.float64)
    shape = arrays["valid_fluid"].shape
    if percentile.shape != shape:
        raise SparseAnchorContractError(
            f"percentile_field shape={percentile.shape} != anchor shape={shape}"
        )
    finite = np.isfinite(percentile)
    valid = arrays["valid_fluid"]
    solid = arrays["solid_mask"]
    invalid = arrays["invalid_support"] | (valid & ~finite)
    criterion_valid = valid & finite
    if np.any(valid & ~finite):
        # A non-finite criterion value cannot silently become percentile
        # negative or unknown; it is tracked as invalid support.
        criterion_valid = valid & finite
    values = percentile[criterion_valid]
    p90 = float(np.percentile(values, 90.0)) if values.size else None
    p60 = float(np.percentile(values, 60.0)) if values.size else None
    positive = criterion_valid & (percentile >= p90) if p90 is not None else np.zeros(shape, bool)
    negative = (
        criterion_valid & ~positive & (percentile <= p60)
        if p60 is not None else np.zeros(shape, bool)
    )
    unknown = criterion_valid & ~positive & ~negative
    anchor_mask = arrays["anchor_mask"] & valid & ~solid & ~arrays["invalid_support"]
    sparse_state = arrays["sparse_state"]
    if not np.array_equal(
        sparse_state == STATE_POSITIVE_ANCHOR,
        anchor_mask,
    ):
        raise SparseAnchorContractError(
            "anchor_mask 与 sparse_state positive_anchor 必须完全一致"
        )
    anchor_count = int(np.count_nonzero(anchor_mask))
    cell_area = float(metadata.get("geometry", {}).get("dx", 1.0)) * float(
        metadata.get("geometry", {}).get("dy", 1.0)
    )
    valid_count = int(np.count_nonzero(valid))
    total_count = int(np.prod(shape))
    failure_mask = sparse_state == STATE_EXTRACTOR_FAILURE
    agreement = anchor_mask & positive
    conflict = anchor_mask & negative
    ambiguous = anchor_mask & unknown
    if int(np.count_nonzero(agreement | conflict | ambiguous)) != anchor_count:
        raise SparseAnchorContractError(
            "每个 Haller positive anchor 必须落入 percentile positive/negative/unknown"
        )
    haller_state_counts = _state_counts(sparse_state)
    haller_failure_count = int(metadata.get("failure_count", 0))
    haller_failure_reasons = list(metadata.get("failure_reasons", []))
    positive_count = int(np.count_nonzero(positive))
    negative_count = int(np.count_nonzero(negative))
    unknown_count = int(np.count_nonzero(unknown))
    invalid_count = int(np.count_nonzero(invalid))
    solid_count = int(np.count_nonzero(solid))
    finite_valid_count = int(np.count_nonzero(criterion_valid))
    row: dict[str, Any] = {
        "sample_id": f"{dataset}__{split}__{frame}",
        "dataset_name": dataset,
        "split_name": split,
        "frame_index": frame,
        "source_fingerprint": source_fingerprint,
        "split_fingerprint": split_fingerprint,
        "protocol_manifest_fingerprint": identity["manifest_fingerprint"],
        "construction": construction,
        "percentile_definition": {
            "positive": "valid_fluid & finite & value >= frame_p90",
            "negative": "valid_fluid & finite & value < frame_p90 & value <= frame_p60",
            "unknown": "valid_fluid & finite & not positive & not negative",
            "positive_percentile": 90.0,
            "negative_percentile": 60.0,
            "positive_threshold": p90,
            "negative_threshold": p60,
            "tie_policy": "positive takes precedence when p90 and p60 coincide",
        },
        "percentile_positive_count": positive_count,
        "percentile_negative_count": negative_count,
        "percentile_unknown_count": unknown_count,
        "percentile_solid_count": solid_count,
        "percentile_invalid_count": invalid_count,
        "percentile_extractor_failure_count": int(np.count_nonzero(failure_mask)),
        "percentile_positive_area": float(positive_count * cell_area),
        "percentile_known_support_count": positive_count + negative_count,
        "percentile_known_support_area": float((positive_count + negative_count) * cell_area),
        "percentile_valid_fluid_count": valid_count,
        "percentile_finite_valid_count": finite_valid_count,
        "percentile_positive_coverage": _relationship_ratio(positive_count, valid_count),
        "percentile_negative_coverage": _relationship_ratio(negative_count, valid_count),
        "percentile_unknown_coverage": _relationship_ratio(unknown_count, valid_count),
        "percentile_solid_coverage": _relationship_ratio(solid_count, total_count),
        "percentile_invalid_coverage": _relationship_ratio(invalid_count, total_count),
        "percentile_extractor_failure_coverage": _relationship_ratio(
            int(np.count_nonzero(failure_mask)), valid_count
        ),
        "percentile_known_support_coverage": _relationship_ratio(
            positive_count + negative_count, valid_count
        ),
        "haller_anchor_count": anchor_count,
        "haller_anchor_area": float(anchor_count * cell_area),
        "haller_anchor_coverage_valid_fluid": _relationship_ratio(anchor_count, valid_count),
        "haller_state_counts": haller_state_counts,
        "haller_extractor_failure_count": haller_failure_count,
        "haller_extractor_failure_cell_count": int(np.count_nonzero(failure_mask)),
        "haller_extractor_failure_reasons": haller_failure_reasons,
        "anchor_percentile_agreement_count": int(np.count_nonzero(agreement)),
        "anchor_percentile_conflict_count": int(np.count_nonzero(conflict)),
        "anchor_percentile_uncovered_or_ambiguous_count": int(np.count_nonzero(ambiguous)),
        "anchor_percentile_denominator": anchor_count,
        "anchor_percentile_agreement_denominator": anchor_count,
        "anchor_percentile_conflict_denominator": anchor_count,
        "anchor_percentile_uncovered_or_ambiguous_denominator": anchor_count,
        "anchor_percentile_agreement_ratio": _relationship_ratio(
            int(np.count_nonzero(agreement)), anchor_count
        ),
        "anchor_percentile_conflict_ratio": _relationship_ratio(
            int(np.count_nonzero(conflict)), anchor_count
        ),
        "anchor_percentile_uncovered_or_ambiguous_ratio": _relationship_ratio(
            int(np.count_nonzero(ambiguous)), anchor_count
        ),
        "source_fingerprints": {
            "haller_anchor_input": metadata.get("source_fingerprint"),
            "percentile_field": _hash_array(percentile),
            "valid_fluid": _hash_array(valid),
            "solid_mask": _hash_array(solid),
            "invalid_support": _hash_array(arrays["invalid_support"]),
            "raw_contours": metadata.get("raw_contours_fingerprint"),
        },
        "runtime_provenance": _json_copy(
            sample.get("runtime_provenance", metadata.get("runtime_provenance", {}))
        ),
        "config_provenance": _json_copy(
            sample.get("config_provenance", metadata.get("config_provenance", {}))
        ),
        "input_provenance": _json_copy({
            **(
                metadata.get("input_provenance", {})
                if isinstance(metadata.get("input_provenance", {}), Mapping)
                else {}
            ),
            **(
                sample.get("input_provenance", {})
                if isinstance(sample.get("input_provenance", {}), Mapping)
                else {}
            ),
            "percentile_field_fingerprint": _hash_array(percentile),
        }),
        "raw_contour_provenance": _json_copy(
            metadata.get("raw_contour_provenance", {})
        ),
        "array_shape": [int(shape[0]), int(shape[1])],
        "cell_area": cell_area,
        "final_test_relationships": "not_read",
        "frame_1300_excluded": True,
    }
    return row


def _aggregate_relationship_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset_name"])].append(row)
    summary: dict[str, Any] = {}
    additive = (
        "percentile_positive_count",
        "percentile_negative_count",
        "percentile_unknown_count",
        "percentile_solid_count",
        "percentile_invalid_count",
        "percentile_extractor_failure_count",
        "percentile_known_support_count",
        "haller_anchor_count",
        "haller_extractor_failure_count",
        "haller_extractor_failure_cell_count",
        "anchor_percentile_agreement_count",
        "anchor_percentile_conflict_count",
        "anchor_percentile_uncovered_or_ambiguous_count",
    )
    for dataset, entries in grouped.items():
        entries = sorted(entries, key=lambda item: (str(item["split_name"]), int(item["frame_index"])))
        aggregate = {
            name: int(sum(int(item[name]) for item in entries)) for name in additive
        }
        valid_total = int(sum(int(item["percentile_valid_fluid_count"]) for item in entries))
        total_total = int(sum(
            int(np.prod(item["array_shape"])) for item in entries
        ))
        anchor_total = aggregate["haller_anchor_count"]
        reasons = Counter(
            reason
            for item in entries
            for reason in item.get("haller_extractor_failure_reasons", [])
        )
        summary[dataset] = {
            "dataset_name": dataset,
            "frame_count": int(len(entries)),
            "split_counts": dict(Counter(str(item["split_name"]) for item in entries)),
            "frame_indices": [int(item["frame_index"]) for item in entries],
            "frame_distributions": {
                "haller_anchor_count": [int(item["haller_anchor_count"]) for item in entries],
                "percentile_positive_coverage": [item["percentile_positive_coverage"] for item in entries],
                "percentile_negative_coverage": [item["percentile_negative_coverage"] for item in entries],
                "percentile_unknown_coverage": [item["percentile_unknown_coverage"] for item in entries],
                "anchor_percentile_agreement_ratio": [item["anchor_percentile_agreement_ratio"] for item in entries],
                "anchor_percentile_conflict_ratio": [item["anchor_percentile_conflict_ratio"] for item in entries],
                "anchor_percentile_uncovered_or_ambiguous_ratio": [
                    item["anchor_percentile_uncovered_or_ambiguous_ratio"]
                    for item in entries
                ],
            },
            **aggregate,
            "percentile_positive_area": float(sum(item["percentile_positive_area"] for item in entries)),
            "percentile_known_support_area": float(sum(item["percentile_known_support_area"] for item in entries)),
            "percentile_positive_coverage": _relationship_ratio(
                aggregate["percentile_positive_count"], valid_total
            ),
            "percentile_known_support_coverage": _relationship_ratio(
                aggregate["percentile_known_support_count"], valid_total
            ),
            "haller_anchor_area": float(sum(
                item["haller_anchor_area"] for item in entries
            )),
            "haller_anchor_coverage_valid_fluid": _relationship_ratio(anchor_total, valid_total),
            "anchor_percentile_denominator": anchor_total,
            "anchor_percentile_agreement_ratio": _relationship_ratio(
                aggregate["anchor_percentile_agreement_count"], anchor_total
            ),
            "anchor_percentile_conflict_ratio": _relationship_ratio(
                aggregate["anchor_percentile_conflict_count"], anchor_total
            ),
            "anchor_percentile_uncovered_or_ambiguous_ratio": _relationship_ratio(
                aggregate["anchor_percentile_uncovered_or_ambiguous_count"], anchor_total
            ),
            "haller_extractor_failure_reason_counts": dict(sorted(reasons.items())),
            "source_fingerprints": sorted({
                str(item["source_fingerprint"])
                for item in entries
                if item.get("source_fingerprint") is not None
            }),
            "split_fingerprints": sorted({
                str(item["split_fingerprint"])
                for item in entries
                if item.get("split_fingerprint") is not None
            }),
            "total_cell_count": total_total,
        }
    return summary


def run_anchor_percentile_relationship_study(
    samples: Sequence[Mapping[str, Any]],
    *,
    qualification_report: Any = None,
    protocol_manifest: Any = None,
    require_all_six: bool = False,
) -> dict[str, Any]:
    """Run the asymmetric train/calibration anchor relationship study.

    ``samples`` is a deliberately narrow seam: each item supplies identity,
    one p-field, and either a previously built sparse anchor or the inputs to
    build one.  Identity guards run for every item before its p-field is
    consumed.  The returned report is descriptive only and has no method,
    threshold, construction, or test-selection side effects.
    """
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise SparseAnchorContractError("samples 必须是有序 sequence")
    manifest = _protocol_manifest_mapping(protocol_manifest)
    qualification = None if qualification_report is None else _validate_qualification_report(
        qualification_report
    )
    real_names = set(REAL_DATASET_NAMES)
    real_relationship = bool(
        manifest is not None
        or any(
            isinstance(item, Mapping)
            and str(item.get("dataset_name", "")) in real_names
            for item in samples
        )
    )
    if real_relationship and qualification is None:
        raise SparseAnchorContractError(
            "六数据集 relationship phase 只能在 qualification status=qualified 后运行"
        )
    rows = [
        _relationship_frame(
            sample,
            protocol_manifest=manifest,
            real_relationship=real_relationship,
        )
        for sample in samples
    ]
    rows.sort(key=lambda item: (
        REAL_DATASET_NAMES.index(item["dataset_name"])
        if item["dataset_name"] in REAL_DATASET_NAMES
        else len(REAL_DATASET_NAMES),
        str(item["split_name"]),
        int(item["frame_index"]),
        str(item["sample_id"]),
    ))
    present = {str(row["dataset_name"]) for row in rows}
    if require_all_six and present != real_names:
        raise SparseAnchorContractError(
            f"real relationship report 必须覆盖六数据集，实际={sorted(present)}"
        )
    if require_all_six:
        observed_pairs = {
            (str(row["dataset_name"]), str(row["split_name"])) for row in rows
        }
        expected_pairs = {
            (dataset, split)
            for dataset in REAL_DATASET_NAMES
            for split in sorted(ALLOWED_REAL_SPLITS)
        }
        if observed_pairs != expected_pairs:
            raise SparseAnchorContractError(
                "real relationship report 必须为每个 dataset 提供 train/calibration sample"
            )
    if real_relationship and manifest is None:
        raise SparseAnchorContractError("real relationship report 缺少 protocol manifest")
    selected = None if qualification is None else qualification["selected_construction"]
    if qualification is not None and any(
        row["construction"] != selected for row in rows
    ):
        raise SparseAnchorContractError(
            "relationship sample construction 与 controlled qualification 选择不一致"
        )
    report: dict[str, Any] = {
        "artifact_type": RELATIONSHIP_REPORT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "status": "published" if rows else "empty",
        "selected_construction": selected,
        "qualification_report_fingerprint": (
            None if qualification is None else qualification["qualification_report_fingerprint"]
        ),
        "sample_count": int(len(rows)),
        "dataset_names": [
            *[name for name in REAL_DATASET_NAMES if name in present],
            *sorted(present - set(REAL_DATASET_NAMES)),
        ],
        "relationship_scope": {
            "allowed_splits": sorted(ALLOWED_REAL_SPLITS),
            "real_data_only": bool(real_relationship),
            "require_all_six": bool(require_all_six),
            "test_inputs_allowed": False,
            "boussinesq_frame_1300_allowed": False,
        },
        "percentile_source": {
            "field_role": "operational dense criterion source",
            "positive": "frame-local p90 on finite valid-fluid cells, inclusive",
            "negative": "frame-local p60 on finite valid-fluid cells, exclusive of positive",
            "unknown": "finite valid-fluid cells not positive and not negative",
            "solid_invalid_failure": "reported separately; never coerced to ordinary unknown",
        },
        "frames": rows,
        "dataset_summary": _aggregate_relationship_rows(rows),
        "haller_metric_contract": {
            "positive_anchor_diagnostics": "reported",
            "negative_class": "not_defined",
            "segmentation_metrics": "not_computed",
            "selective_metrics": "not_computed",
            "uncovered_semantics": "not_negative",
            "conflict_semantics": "positive_anchor_intersect_percentile_negative",
            "prohibited_outputs": [
                "negative-class agreement or coverage",
                "symmetric two-source confusion",
                "dense segmentation score from sparse anchors",
                "selective risk from sparse anchors",
            ],
        },
        "access_guard": {
            "final_test_relationships": "not_read",
            "test_labels": "not_read",
            "test_coverage": "not_read",
            "test_agreement": "not_read",
            "test_conflict": "not_read",
            "test_model_metrics": "not_read",
            "frame_1300": "excluded",
            "guard_order": "identity_and_split_checked_before_field_read",
        },
        "provenance": {
            "protocol_manifest_fingerprint": (
                None if manifest is None else manifest["manifest_hash"]
            ),
            "source_relationship_is_descriptive": True,
            "construction_selection_source": "controlled_qualification_only",
            "real_data_or_test_used_for_selection": False,
        },
    }
    report["report_fingerprint"] = _json_hash(report)
    return report


_QUALIFICATION_REPORT_FILE = "qualification_report.json"
_QUALIFICATION_INTEGRITY_FILE = "qualification_integrity.json"
_RELATIONSHIP_REPORT_FILE = "relationship_report.json"
_RELATIONSHIP_INTEGRITY_FILE = "relationship_integrity.json"
_BUNDLE_MANIFEST_FILE = "manifest.json"
_BUNDLE_INTEGRITY_FILE = "integrity.json"


def _validate_qualification_identity(value: Any) -> dict[str, Any]:
    report = _load_json_or_mapping(value, name="qualification_report")
    if report.get("artifact_type") != QUALIFICATION_REPORT_TYPE:
        raise SparseAnchorContractError("qualification report artifact_type 不匹配")
    if report.get("protocol_version") != PROTOCOL_VERSION:
        raise SparseAnchorContractError("qualification report protocol_version 不匹配")
    declared = report.get("qualification_report_fingerprint")
    payload = _json_copy(report)
    payload.pop("qualification_report_fingerprint", None)
    if not isinstance(declared, str) or _json_hash(payload) != declared:
        raise SparseAnchorContractError("qualification report fingerprint 校验失败")
    selected = report.get("selected_construction")
    if selected is not None and selected not in REGISTERED_CONSTRUCTIONS:
        raise SparseAnchorContractError("qualification report selected_construction 无效")
    status = report.get("status")
    if status not in {QUALIFICATION_STATUS_QUALIFIED, QUALIFICATION_STATUS_NO_GO}:
        raise SparseAnchorContractError("qualification report status 无效")
    return report


def _validate_relationship_report(value: Any) -> dict[str, Any]:
    report = _load_json_or_mapping(value, name="relationship_report")
    if report.get("artifact_type") != RELATIONSHIP_REPORT_TYPE:
        raise SparseAnchorContractError("relationship report artifact_type 不匹配")
    if report.get("protocol_version") != PROTOCOL_VERSION:
        raise SparseAnchorContractError("relationship report protocol_version 不匹配")
    declared = report.get("report_fingerprint")
    payload = _json_copy(report)
    payload.pop("report_fingerprint", None)
    if not isinstance(declared, str) or _json_hash(payload) != declared:
        raise SparseAnchorContractError("relationship report fingerprint 校验失败")
    if report.get("access_guard", {}).get("final_test_relationships") != "not_read":
        raise FinalTestAccessError("relationship report 没有封存 final-test relationships")
    if report.get("haller_metric_contract", {}).get("segmentation_metrics") != "not_computed":
        raise SparseAnchorContractError("relationship report Haller metric contract 不完整")
    return report


def _publish_report_directory(
    report: Mapping[str, Any],
    out_dir: str | pathlib.Path,
    *,
    report_file: str,
    integrity_file: str,
    report_fingerprint_key: str,
    integrity_type: str,
) -> dict[str, str]:
    out = pathlib.Path(out_dir)
    if out.exists():
        if any(out.iterdir()):
            raise FileExistsError(f"拒绝覆盖已有 report artifact: {out}")
        out.rmdir()
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(prefix=".report-staging-", dir=out.parent))
    try:
        report_copy = _json_copy(dict(report))
        fingerprint = report_copy.get(report_fingerprint_key)
        if not isinstance(fingerprint, str):
            raise SparseAnchorContractError(f"{report_fingerprint_key} 缺失")
        _write_json(stage / report_file, report_copy)
        file_hash = _file_sha256(stage / report_file)
        integrity_fingerprint = _json_hash({
            "report_fingerprint": fingerprint,
            "report_file": report_file,
            "report_file_hash": file_hash,
            "protocol_version": PROTOCOL_VERSION,
        })
        sidecar = {
            "artifact_type": integrity_type,
            "protocol_version": PROTOCOL_VERSION,
            "schema_version": f"{integrity_type}-v1",
            report_fingerprint_key: fingerprint,
            "report_fingerprint": fingerprint,
            "integrity_fingerprint": integrity_fingerprint,
            "file_hashes": {report_file: file_hash},
            "final_test_relationships": "not_read",
        }
        _write_json(stage / integrity_file, sidecar)
        shutil.move(str(stage), str(out))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "artifact_dir": str(out),
        "report": str(out / report_file),
        "integrity": str(out / integrity_file),
        "report_fingerprint": str(fingerprint),
        "integrity_fingerprint": str(integrity_fingerprint),
    }


def _read_report_directory(
    out_dir: str | pathlib.Path,
    *,
    report_file: str,
    integrity_file: str,
    report_fingerprint_key: str,
    expected_type: str,
    expected_integrity_type: str,
    allow_no_go: bool = False,
) -> dict[str, Any]:
    out = pathlib.Path(out_dir)
    report_path = out / report_file
    integrity_path = out / integrity_file
    if not report_path.exists() or not integrity_path.exists():
        raise SparseAnchorContractError("report 缺少 report/integrity sidecar")
    report = _load_json_or_mapping(report_path, name="report")
    if report.get("artifact_type") != expected_type:
        raise SparseAnchorContractError("report artifact_type 不匹配")
    if expected_type == QUALIFICATION_REPORT_TYPE:
        identity = _validate_qualification_identity(report)
        if not allow_no_go and identity.get("status") != QUALIFICATION_STATUS_QUALIFIED:
            raise SparseAnchorContractError("qualification report 是 no-go，不能作为 qualified input")
    else:
        identity = _validate_relationship_report(report)
    sidecar = _load_json_or_mapping(integrity_path, name="integrity")
    if sidecar.get("artifact_type") != expected_integrity_type:
        raise SparseAnchorContractError("report integrity sidecar type 不匹配")
    if sidecar.get(report_fingerprint_key) != report[report_fingerprint_key]:
        raise SparseAnchorContractError("report sidecar fingerprint 不匹配")
    expected_file_hash = sidecar.get("file_hashes", {}).get(report_file)
    actual_file_hash = _file_sha256(report_path)
    if expected_file_hash != actual_file_hash:
        raise ValueError("report file hash 校验失败")
    expected_integrity = _json_hash({
        "report_fingerprint": report[report_fingerprint_key],
        "report_file": report_file,
        "report_file_hash": actual_file_hash,
        "protocol_version": PROTOCOL_VERSION,
    })
    if sidecar.get("integrity_fingerprint") != expected_integrity:
        raise ValueError("report integrity fingerprint 校验失败")
    if sidecar.get("final_test_relationships") != "not_read":
        raise FinalTestAccessError("report sidecar 没有封存 final-test relationships")
    return {
        **identity,
        "readback": {
            "schema_verified": True,
            "integrity_verified": True,
            "file_hashes_verified": True,
            "report_fingerprint": report[report_fingerprint_key],
            "integrity_fingerprint": expected_integrity,
        },
    }


def save_qualification_report(
    report: Mapping[str, Any], out_dir: str | pathlib.Path
) -> dict[str, str]:
    validated = _validate_qualification_identity(report)
    return _publish_report_directory(
        validated,
        out_dir,
        report_file=_QUALIFICATION_REPORT_FILE,
        integrity_file=_QUALIFICATION_INTEGRITY_FILE,
        report_fingerprint_key="qualification_report_fingerprint",
        integrity_type="haller_sparse_anchor_qualification_integrity",
    )


def read_qualification_report(
    out_dir: str | pathlib.Path, *, allow_no_go: bool = False
) -> dict[str, Any]:
    return _read_report_directory(
        out_dir,
        report_file=_QUALIFICATION_REPORT_FILE,
        integrity_file=_QUALIFICATION_INTEGRITY_FILE,
        report_fingerprint_key="qualification_report_fingerprint",
        expected_type=QUALIFICATION_REPORT_TYPE,
        expected_integrity_type="haller_sparse_anchor_qualification_integrity",
        allow_no_go=allow_no_go,
    )


def save_relationship_report(
    report: Mapping[str, Any], out_dir: str | pathlib.Path
) -> dict[str, str]:
    validated = _validate_relationship_report(report)
    return _publish_report_directory(
        validated,
        out_dir,
        report_file=_RELATIONSHIP_REPORT_FILE,
        integrity_file=_RELATIONSHIP_INTEGRITY_FILE,
        report_fingerprint_key="report_fingerprint",
        integrity_type="haller_anchor_percentile_relationship_integrity",
    )


def read_relationship_report(out_dir: str | pathlib.Path) -> dict[str, Any]:
    return _read_report_directory(
        out_dir,
        report_file=_RELATIONSHIP_REPORT_FILE,
        integrity_file=_RELATIONSHIP_INTEGRITY_FILE,
        report_fingerprint_key="report_fingerprint",
        expected_type=RELATIONSHIP_REPORT_TYPE,
        expected_integrity_type="haller_anchor_percentile_relationship_integrity",
    )


def _sample_identifier(result: Mapping[str, Any]) -> str:
    metadata = result.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise SparseAnchorContractError("sparse result.metadata 必须是 mapping")
    dataset = _text(metadata.get("dataset_name"), name="metadata.dataset_name")
    split = _text(metadata.get("split_name"), name="metadata.split_name")
    frame = _exact_frame(metadata.get("frame_index"))
    return f"{dataset}__{split}__{frame}"


def _bundle_file_hashes(root: pathlib.Path, relative_paths: Sequence[str]) -> dict[str, str]:
    return {
        relative: _file_sha256(root / relative)
        for relative in sorted(relative_paths)
    }


def save_sparse_anchor_bundle(
    results: Sequence[Mapping[str, Any]],
    *,
    qualification_report: Mapping[str, Any],
    out_dir: str | pathlib.Path,
    relationship_report: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Publish qualified samples plus their qualification/report identities.

    A no-go qualification is publishable only as an explicitly unqualified
    diagnostic bundle.  It never receives the training-anchor flag.
    """
    qualification = _validate_qualification_identity(qualification_report)
    qualification_status = qualification["status"]
    if qualification_status == QUALIFICATION_STATUS_QUALIFIED:
        _validate_qualification_report(qualification)
        selected = qualification["selected_construction"]
        artifact_state = "qualified"
    else:
        selected = None
        artifact_state = "unqualified"
        if relationship_report is not None:
            raise SparseAnchorContractError("no-go qualification 不能绑定 relationship report")
    relationship = None
    if relationship_report is not None:
        relationship = _validate_relationship_report(relationship_report)
        if relationship.get("qualification_report_fingerprint") != qualification[
            "qualification_report_fingerprint"
        ]:
            raise SparseAnchorContractError("relationship report 与 qualification fingerprint 不一致")
        if relationship.get("selected_construction") != selected:
            raise SparseAnchorContractError("relationship report construction 不一致")
    out = pathlib.Path(out_dir)
    if out.exists():
        if any(out.iterdir()):
            raise FileExistsError(f"拒绝覆盖已有 sparse-anchor bundle: {out}")
        out.rmdir()
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(prefix=".bundle-staging-", dir=out.parent))
    try:
        _write_json(stage / _QUALIFICATION_REPORT_FILE, qualification)
        qualification_file_hash = _file_sha256(stage / _QUALIFICATION_REPORT_FILE)
        qualification_integrity_fingerprint = _json_hash({
            "report_fingerprint": qualification["qualification_report_fingerprint"],
            "report_file": _QUALIFICATION_REPORT_FILE,
            "report_file_hash": qualification_file_hash,
            "protocol_version": PROTOCOL_VERSION,
        })
        qualification_sidecar = {
            "artifact_type": "haller_sparse_anchor_qualification_integrity",
            "protocol_version": PROTOCOL_VERSION,
            "schema_version": "haller_sparse_anchor_qualification_integrity-v1",
            "qualification_report_fingerprint": qualification["qualification_report_fingerprint"],
            "report_fingerprint": qualification["qualification_report_fingerprint"],
            "integrity_fingerprint": qualification_integrity_fingerprint,
            "file_hashes": {_QUALIFICATION_REPORT_FILE: qualification_file_hash},
            "final_test_relationships": "not_read",
        }
        _write_json(stage / _QUALIFICATION_INTEGRITY_FILE, qualification_sidecar)
        relative_files = [_QUALIFICATION_REPORT_FILE, _QUALIFICATION_INTEGRITY_FILE]
        if relationship is not None:
            _write_json(stage / _RELATIONSHIP_REPORT_FILE, relationship)
            relationship_file_hash = _file_sha256(stage / _RELATIONSHIP_REPORT_FILE)
            relationship_integrity_fingerprint = _json_hash({
                "report_fingerprint": relationship["report_fingerprint"],
                "report_file": _RELATIONSHIP_REPORT_FILE,
                "report_file_hash": relationship_file_hash,
                "protocol_version": PROTOCOL_VERSION,
            })
            relationship_sidecar = {
                "artifact_type": "haller_anchor_percentile_relationship_integrity",
                "protocol_version": PROTOCOL_VERSION,
                "schema_version": "haller_anchor_percentile_relationship_integrity-v1",
                "report_fingerprint": relationship["report_fingerprint"],
                "integrity_fingerprint": relationship_integrity_fingerprint,
                "file_hashes": {_RELATIONSHIP_REPORT_FILE: relationship_file_hash},
                "final_test_relationships": "not_read",
            }
            _write_json(stage / _RELATIONSHIP_INTEGRITY_FILE, relationship_sidecar)
            relative_files.extend([_RELATIONSHIP_REPORT_FILE, _RELATIONSHIP_INTEGRITY_FILE])
        sample_root = stage / "samples"
        sample_root.mkdir(parents=True, exist_ok=True)
        sample_records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for result in results:
            if not isinstance(result, Mapping):
                raise SparseAnchorContractError("bundle result 必须是 mapping")
            metadata = result.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise SparseAnchorContractError("bundle result.metadata 必须是 mapping")
            sample_id = _sample_identifier(result)
            if sample_id in seen_ids:
                raise SparseAnchorContractError(f"bundle sample_id 重复: {sample_id}")
            seen_ids.add(sample_id)
            if selected is not None and metadata.get("construction") != selected:
                raise SparseAnchorContractError("bundle sample construction 与 qualification 不一致")
            sample_dir = sample_root / sample_id
            save_sparse_anchor_artifact(
                result,
                sample_dir,
                artifact_state=artifact_state,
                qualification_report_fingerprint=qualification[
                    "qualification_report_fingerprint"
                ],
            )
            sample_metadata = json.loads(
                (sample_dir / _SAMPLE_METADATA_FILE).read_text(encoding="utf-8")
            )
            relative_sample = f"samples/{sample_id}"
            relative_files.extend(
                str(pathlib.Path(relative_sample) / filename)
                for filename in (
                    *_SAMPLE_ARRAY_FILES.values(),
                    *_SAMPLE_JSON_FILES.values(),
                    _SAMPLE_METADATA_FILE,
                    _SAMPLE_INTEGRITY_FILE,
                )
            )
            sample_records.append({
                "sample_id": sample_id,
                "path": relative_sample,
                "dataset_name": sample_metadata["dataset_name"],
                "split_name": sample_metadata["split_name"],
                "frame_index": sample_metadata["frame_index"],
                "source_fingerprint": sample_metadata.get("source_fingerprint"),
                "split_fingerprint": sample_metadata.get("split_fingerprint"),
                "raw_contours_fingerprint": sample_metadata.get("raw_contours_fingerprint"),
                "construction": sample_metadata.get("construction"),
                "artifact_state": sample_metadata.get("artifact_state"),
                "artifact_fingerprint": sample_metadata.get("integrity_fingerprint"),
            })
        if relationship is not None:
            relationship_sample_ids = {
                str(row.get("sample_id")) for row in relationship.get("frames", [])
            }
            bundle_sample_ids = {
                str(record["sample_id"]) for record in sample_records
            }
            if relationship_sample_ids != bundle_sample_ids:
                raise SparseAnchorContractError(
                    "relationship report frames 与 bundle samples identity 不一致"
                )
        manifest: dict[str, Any] = {
            "artifact_type": ARTIFACT_TYPE,
            "protocol_version": PROTOCOL_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "artifact_state": artifact_state,
            "status": qualification_status,
            "allowed_for_training_anchor": artifact_state == "qualified",
            "eligible_as_dense_evaluation_reference": False,
            "trusted_haller_v2": False,
            "construction": selected,
            "registered_constructions": list(REGISTERED_CONSTRUCTIONS),
            "qualification_report_fingerprint": qualification[
                "qualification_report_fingerprint"
            ],
            "relationship_report_fingerprint": (
                None if relationship is None else relationship["report_fingerprint"]
            ),
            "sample_count": int(len(sample_records)),
            "samples": sample_records,
            "source_fingerprints": sorted({
                str(record["source_fingerprint"])
                for record in sample_records
                if record.get("source_fingerprint") is not None
            }),
            "raw_contour_fingerprints": sorted({
                str(record["raw_contours_fingerprint"])
                for record in sample_records
                if record.get("raw_contours_fingerprint") is not None
            }),
            "provenance": {
                "qualification": "controlled_exact_synthetic_only",
                "relationship": "train_calibration_descriptive_only" if relationship else None,
                "runtime_config_input": "retained_in_each_sample_metadata",
            },
            "final_test_relationships": "not_read",
            "frame_1300_excluded": True,
            "failure_or_no_go": (
                None
                if qualification_status == QUALIFICATION_STATUS_QUALIFIED
                else {
                    "status": "no-go",
                    "reason": "controlled qualification gates did not all pass",
                    "trusted_artifact_created": False,
                }
            ),
        }
        manifest["integrity_fingerprint"] = _json_hash(manifest)
        _write_json(stage / _BUNDLE_MANIFEST_FILE, manifest)
        relative_files.append(_BUNDLE_MANIFEST_FILE)
        file_hashes = _bundle_file_hashes(stage, relative_files)
        sidecar = {
            "artifact_type": "haller_sparse_physical_anchor_integrity",
            "protocol_version": PROTOCOL_VERSION,
            "schema_version": "haller-sparse-physical-anchor-bundle-integrity-v1",
            "artifact_fingerprint": manifest["integrity_fingerprint"],
            "file_hashes": file_hashes,
            "final_test_relationships": "not_read",
        }
        _write_json(stage / _BUNDLE_INTEGRITY_FILE, sidecar)
        shutil.move(str(stage), str(out))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "artifact_dir": str(out),
        "manifest": str(out / _BUNDLE_MANIFEST_FILE),
        "integrity": str(out / _BUNDLE_INTEGRITY_FILE),
        "artifact_fingerprint": str(manifest["integrity_fingerprint"]),
    }


def load_sparse_anchor_bundle(
    out_dir: str | pathlib.Path,
    *,
    require_training_anchor: bool = False,
) -> dict[str, Any]:
    """Read the bundle, its sidecar, reports, and every sample artifact."""
    out = pathlib.Path(out_dir)
    manifest = _load_json_or_mapping(out / _BUNDLE_MANIFEST_FILE, name="bundle manifest")
    sidecar = _load_json_or_mapping(out / _BUNDLE_INTEGRITY_FILE, name="bundle integrity")
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise SparseAnchorContractError("bundle artifact_type 不匹配")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise SparseAnchorContractError("bundle protocol_version 不匹配")
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise SparseAnchorContractError("bundle artifact_version 不匹配")
    state = manifest.get("artifact_state")
    if state not in ARTIFACT_STATES:
        raise SparseAnchorContractError("bundle artifact_state 无效")
    if require_training_anchor and (
        state != "qualified" or manifest.get("allowed_for_training_anchor") is not True
    ):
        raise PermissionError("只有 qualified sparse-anchor bundle 可以进入 training anchor")
    if manifest.get("eligible_as_dense_evaluation_reference") is not False:
        raise SparseAnchorContractError("bundle dense-reference guard 被解除")
    if manifest.get("trusted_haller_v2") is not False:
        raise SparseAnchorContractError("bundle trusted_haller_v2 guard 被解除")
    if manifest.get("final_test_relationships") != "not_read":
        raise FinalTestAccessError("bundle manifest 没有封存 final-test relationships")
    if manifest.get("frame_1300_excluded") is not True:
        raise SparseAnchorContractError("bundle 未声明排除 Boussinesq frame 1300")
    payload = _json_copy(manifest)
    declared = payload.pop("integrity_fingerprint", None)
    if not isinstance(declared, str) or _json_hash(payload) != declared:
        raise ValueError("bundle manifest integrity fingerprint 校验失败")
    if sidecar.get("artifact_type") != "haller_sparse_physical_anchor_integrity":
        raise SparseAnchorContractError("bundle integrity sidecar type 不匹配")
    if sidecar.get("artifact_fingerprint") != declared:
        raise ValueError("bundle sidecar fingerprint 校验失败")
    if sidecar.get("final_test_relationships") != "not_read":
        raise FinalTestAccessError("bundle sidecar 没有封存 final-test relationships")
    declared_hashes = sidecar.get("file_hashes", {})
    for relative, expected_hash in declared_hashes.items():
        if expected_hash != _file_sha256(out / relative):
            raise ValueError(f"bundle file hash 校验失败: {relative}")
    qualification = _read_report_directory(
        out,
        report_file=_QUALIFICATION_REPORT_FILE,
        integrity_file=_QUALIFICATION_INTEGRITY_FILE,
        report_fingerprint_key="qualification_report_fingerprint",
        expected_type=QUALIFICATION_REPORT_TYPE,
        expected_integrity_type="haller_sparse_anchor_qualification_integrity",
        allow_no_go=True,
    )
    if qualification.get("qualification_report_fingerprint") != manifest.get(
        "qualification_report_fingerprint"
    ):
        raise ValueError("bundle qualification fingerprint 与 manifest 不一致")
    relationship = None
    if manifest.get("relationship_report_fingerprint") is not None:
        relationship = _read_report_directory(
            out,
            report_file=_RELATIONSHIP_REPORT_FILE,
            integrity_file=_RELATIONSHIP_INTEGRITY_FILE,
            report_fingerprint_key="report_fingerprint",
            expected_type=RELATIONSHIP_REPORT_TYPE,
            expected_integrity_type="haller_anchor_percentile_relationship_integrity",
        )
        if relationship.get("report_fingerprint") != manifest.get(
            "relationship_report_fingerprint"
        ):
            raise ValueError("bundle relationship fingerprint 与 manifest 不一致")
        if {
            str(row.get("sample_id")) for row in relationship.get("frames", [])
        } != {
            str(record.get("sample_id")) for record in manifest.get("samples", [])
        }:
            raise SparseAnchorContractError(
                "bundle relationship frames 与 manifest samples identity 不一致"
            )
    if len(manifest.get("samples", [])) != len({
        str(record.get("sample_id")) for record in manifest.get("samples", [])
    }):
        raise SparseAnchorContractError("bundle manifest sample_id 重复")
    samples: dict[str, Any] = {}
    for record in manifest.get("samples", []):
        sample_id = _text(record.get("sample_id"), name="bundle sample_id")
        sample = load_sparse_anchor_artifact(
            out / record["path"],
            require_training_anchor=require_training_anchor,
        )
        metadata = sample["metadata"]
        if metadata.get("artifact_state") != state:
            raise SparseAnchorContractError("sample artifact_state 与 bundle 不一致")
        if metadata.get("construction") != manifest.get("construction"):
            raise SparseAnchorContractError("sample construction 与 bundle 不一致")
        if metadata.get("qualification_report_fingerprint") != manifest.get(
            "qualification_report_fingerprint"
        ):
            raise SparseAnchorContractError("sample qualification fingerprint 与 bundle 不一致")
        if metadata.get("integrity_fingerprint") != record.get("artifact_fingerprint"):
            raise ValueError("bundle sample fingerprint 与 manifest 不一致")
        samples[sample_id] = sample
    if len(samples) != int(manifest.get("sample_count", -1)):
        raise SparseAnchorContractError("bundle sample_count 与 readback 不一致")
    return {
        "manifest": manifest,
        "integrity": sidecar,
        "qualification_report": qualification,
        "relationship_report": relationship,
        "samples": samples,
        "readback": {
            "schema_verified": True,
            "integrity_verified": True,
            "file_hashes_verified": True,
            "sample_count": len(samples),
            "artifact_fingerprint": declared,
        },
    }


_CANDIDATE_RUN_ID = re.compile(
    r"^development__(?P<dataset>.+)__(?P<split>train|calibration)__frame(?P<frame>[0-9]+)__r1$"
)


def _candidate_manifest_path(candidate_artifact_dir: str | pathlib.Path) -> pathlib.Path:
    supplied = pathlib.Path(candidate_artifact_dir)
    if supplied.is_file() and supplied.name == "candidate_manifest.json":
        return supplied
    candidates = [supplied / "candidate_manifest.json", supplied / "candidate" / "candidate_manifest.json"]
    for path in candidates:
        if path.exists():
            return path
    raise SparseAnchorContractError(
        f"candidate artifact 缺少 candidate_manifest.json: {supplied}"
    )


def _candidate_run_path(artifact_root: pathlib.Path, relative_dir: Any) -> pathlib.Path:
    relative = pathlib.PurePosixPath(_text(relative_dir, name="relative_dir"))
    if relative.is_absolute() or ".." in relative.parts:
        raise SparseAnchorContractError("candidate relative_dir 越出 artifact root")
    return artifact_root / pathlib.Path(*relative.parts)


def _load_candidate_json(path: pathlib.Path, *, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SparseAnchorContractError(f"candidate {name} 不是可读 JSON") from exc


def load_numbacs_candidate_samples(
    candidate_artifact_dir: str | pathlib.Path,
    *,
    protocol_manifest: Any,
    construction: str = CONSTRUCTION_SMALL_CORE_R1,
) -> list[dict[str, Any]]:
    """Read the immutable Ticket 03 candidate without promoting its identity.

    The candidate manifest is metadata-only.  Every run is checked against
    the R0 protocol and its declared file hashes before arrays are consumed;
    no test-like run is ever loaded.
    """
    manifest_path = _candidate_manifest_path(candidate_artifact_dir)
    candidate = _load_candidate_json(manifest_path, name="candidate_manifest")
    if not isinstance(candidate, Mapping):
        raise SparseAnchorContractError("candidate_manifest 必须是 JSON object")
    if candidate.get("artifact_type") != "numbacs_raw_contour_candidate":
        raise SparseAnchorContractError("candidate artifact_type 不匹配")
    if candidate.get("artifact_state") != "candidate" or candidate.get("trusted") is not False:
        raise SparseAnchorContractError("输入必须是未信任的 Ticket 03 candidate")
    if candidate.get("final_test_relationships") != "not_read":
        raise FinalTestAccessError("candidate 没有封存 final-test relationships")
    if candidate.get("radius_factor") != 1.0 or candidate.get("radius_over_delta") != 1.0:
        raise SparseAnchorContractError("candidate radius provenance 不是固定 r1")
    candidate_payload = _json_copy(dict(candidate))
    candidate_fingerprint = candidate_payload.pop("integrity_fingerprint", None)
    if (
        not isinstance(candidate_fingerprint, str)
        or _json_hash(candidate_payload) != candidate_fingerprint
    ):
        raise ValueError("candidate manifest integrity fingerprint 校验失败")
    entries = candidate.get("sample_runs")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise SparseAnchorContractError("candidate sample_runs 必须是 sequence")
    if int(candidate.get("sample_run_count", -1)) != len(entries):
        raise SparseAnchorContractError("candidate sample_run_count 与 manifest 不一致")
    manifest = _protocol_manifest_mapping(protocol_manifest)
    if manifest is None:
        raise SparseAnchorContractError("six-dataset candidate loader 必须绑定 protocol manifest")
    mode = _validate_construction(construction)
    artifact_root = manifest_path.parent.parent
    parsed: list[tuple[dict[str, Any], str, str, int]] = []
    # Identity-only pass.  This occurs before any run file is opened.
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise SparseAnchorContractError("candidate sample_run entry 必须是 mapping")
        entry = _json_copy(dict(raw_entry))
        run_id = _text(entry.get("run_id"), name="candidate.run_id")
        match = _CANDIDATE_RUN_ID.fullmatch(run_id)
        if match is None:
            raise SparseAnchorContractError(f"candidate run_id 不符合 train/cal schema: {run_id}")
        dataset = match.group("dataset")
        split = match.group("split")
        frame = _exact_frame(match.group("frame"))
        _validate_scope(dataset, split, frame, real_relationship=True)
        _protocol_sample_identity(
            manifest,
            dataset_name=dataset,
            split_name=split,
            frame_index=frame,
        )
        if "test" in str(entry.get("relative_dir", "")).lower():
            raise FinalTestAccessError(f"candidate relative_dir 含 test marker: {run_id}")
        parsed.append((entry, dataset, split, frame))
    if {
        (dataset, split) for _, dataset, split, _ in parsed
    } != {
        (dataset, split)
        for dataset in REAL_DATASET_NAMES
        for split in sorted(ALLOWED_REAL_SPLITS)
    }:
        raise SparseAnchorContractError("candidate 必须提供六数据集各自 train/calibration sample")
    results: list[dict[str, Any]] = []
    for entry, dataset, split, frame in parsed:
        run_dir = _candidate_run_path(artifact_root, entry.get("relative_dir"))
        declared_hashes = entry.get("file_hashes")
        if not isinstance(declared_hashes, Mapping):
            raise SparseAnchorContractError(f"candidate {entry['run_id']} 缺少 file_hashes")
        required_files = (
            "standard_ivd.npy",
            "solid_mask.npy",
            "xdim.npy",
            "ydim.npy",
            "raw_contours.json",
            "filtering_decisions.json",
            "run_metrics.json",
        )
        missing = [name for name in required_files if name not in declared_hashes]
        if missing:
            raise SparseAnchorContractError(
                f"candidate {entry['run_id']} 缺少声明文件={missing}"
            )
        for filename in required_files:
            path = run_dir / filename
            if not path.exists():
                raise SparseAnchorContractError(f"candidate 文件不存在: {path}")
            if declared_hashes[filename] != _file_sha256(path):
                raise ValueError(f"candidate file hash 校验失败: {entry['run_id']}/{filename}")
        xdim = np.load(run_dir / "xdim.npy", allow_pickle=False)
        ydim = np.load(run_dir / "ydim.npy", allow_pickle=False)
        ivd = np.load(run_dir / "standard_ivd.npy", allow_pickle=False)
        solid = np.asarray(np.load(run_dir / "solid_mask.npy", allow_pickle=False), dtype=bool)
        raw_contours = _load_candidate_json(run_dir / "raw_contours.json", name="raw_contours")
        filtering = _load_candidate_json(
            run_dir / "filtering_decisions.json", name="filtering_decisions"
        )
        metrics = _load_candidate_json(run_dir / "run_metrics.json", name="run_metrics")
        if not isinstance(raw_contours, list) or not isinstance(metrics, Mapping):
            raise SparseAnchorContractError(f"candidate {entry['run_id']} JSON schema 无效")
        centers: list[dict[str, Any]] = []
        for index, raw_record in enumerate(raw_contours):
            if not isinstance(raw_record, Mapping):
                continue
            status = str(raw_record.get("status", "valid"))
            center = raw_record.get("center_xy")
            if status in {"valid", "accepted"} and center is not None:
                center_record = _json_copy(dict(raw_record))
                center_record.update({
                    "native_record_id": f"{entry['run_id']}::raw::{index}",
                    "raw_contour_index": int(index),
                })
                centers.append(center_record)
        runtime = metrics.get("runtime_identity", {})
        config = {
            "candidate_manifest": {
                "artifact_type": candidate.get("artifact_type"),
                "integrity_fingerprint": candidate.get("integrity_fingerprint"),
                "parameter_fingerprint": candidate.get("parameter_fingerprint"),
                "radius_factor": candidate.get("radius_factor"),
                "min_len_factor": candidate.get("min_len_factor"),
            },
            "contour_evaluation": metrics.get("contour_evaluation", {}),
            "filtering_policy": {
                "artifact_file": "filtering_decisions.json",
                "fingerprint": _json_hash(filtering),
                "record_count": len(filtering) if isinstance(filtering, Sequence) else None,
            },
            "construction": mode,
            "radius_factor": 1.0 if mode == CONSTRUCTION_SMALL_CORE_R1 else None,
        }
        identity = _protocol_sample_identity(
            manifest,
            dataset_name=dataset,
            split_name=split,
            frame_index=frame,
        )
        result = build_sparse_anchor(
            centers,
            xdim=xdim,
            ydim=ydim,
            valid_fluid=~solid,
            solid_mask=solid,
            invalid_support=np.zeros_like(solid, dtype=bool),
            raw_contours=raw_contours,
            dataset_name=dataset,
            split_name=split,
            frame_index=frame,
            construction=mode,
            source_fingerprint=identity["source_fingerprint"],
            split_fingerprint=identity["split_fingerprint"],
            extractor_status=str(metrics.get("status", "valid")),
            failure_reason=(
                metrics.get("status_reason")
                if metrics.get("status") not in {None, "valid"}
                else None
            ),
            failure_count=int(metrics.get("failure_count", 0)),
            runtime_provenance={
                "candidate_runtime_identity": runtime,
                "candidate_backend": metrics.get("backend", candidate.get("backend")),
                "candidate_run_metrics_fingerprint": _file_sha256(run_dir / "run_metrics.json"),
            },
            config_provenance=config,
            input_provenance={
                "candidate_manifest_fingerprint": _file_sha256(manifest_path),
                "candidate_manifest_integrity_fingerprint": candidate.get("integrity_fingerprint"),
                "candidate_sample_run_artifact_fingerprint": entry.get(
                    "source_run_artifact_fingerprint"
                ),
                "candidate_run_id": entry["run_id"],
                "candidate_relative_dir": entry["relative_dir"],
                "candidate_declared_file_hashes": _json_copy(dict(declared_hashes)),
                "candidate_input_hash": metrics.get("input_hash"),
                "source_identity_fingerprint": metrics.get("source_identity_fingerprint"),
                "protocol_manifest_fingerprint": manifest["manifest_hash"],
            },
        )
        result["percentile_field"] = ivd
        result["candidate_filtering_decisions"] = filtering
        result["candidate_run_metrics"] = metrics
        results.append(result)
    results.sort(key=lambda item: (
        REAL_DATASET_NAMES.index(item["metadata"]["dataset_name"]),
        (0 if item["metadata"]["split_name"] == "train" else 1),
        int(item["metadata"]["frame_index"]),
    ))
    return results


def run_ticket05(
    *,
    candidate_artifact_dir: str | pathlib.Path,
    protocol_manifest: Any,
    out_dir: str | pathlib.Path,
    controlled_resolutions: Sequence[int] = (65, 97, 129),
) -> dict[str, Any]:
    """Execute Ticket 05-A then 05-B and publish one auditable bundle."""
    manifest = _protocol_manifest_mapping(protocol_manifest)
    qualification = qualify_sparse_anchor_constructions(
        resolutions=controlled_resolutions
    )
    if qualification["status"] != QUALIFICATION_STATUS_QUALIFIED:
        bundle_paths = save_sparse_anchor_bundle(
            [], qualification_report=qualification, out_dir=out_dir
        )
        bundle = load_sparse_anchor_bundle(bundle_paths["artifact_dir"])
        return {
            "status": QUALIFICATION_STATUS_NO_GO,
            "qualification_report": qualification,
            "relationship_report": None,
            "bundle": bundle,
            "trusted_training_anchor": False,
        }
    selected = qualification["selected_construction"]
    samples = load_numbacs_candidate_samples(
        candidate_artifact_dir,
        protocol_manifest=manifest,
        construction=selected,
    )
    relationship = run_anchor_percentile_relationship_study(
        [
            {
                "anchor_result": sample,
                "percentile_field": sample["percentile_field"],
                "dataset_name": sample["metadata"]["dataset_name"],
                "split_name": sample["metadata"]["split_name"],
                "frame_index": sample["metadata"]["frame_index"],
            }
            for sample in samples
        ],
        qualification_report=qualification,
        protocol_manifest=manifest,
        require_all_six=True,
    )
    bundle_paths = save_sparse_anchor_bundle(
        samples,
        qualification_report=qualification,
        relationship_report=relationship,
        out_dir=out_dir,
    )
    bundle = load_sparse_anchor_bundle(
        bundle_paths["artifact_dir"], require_training_anchor=True
    )
    return {
        "status": "published",
        "qualification_report": qualification,
        "relationship_report": relationship,
        "bundle": bundle,
        "trusted_training_anchor": True,
        "paths": bundle_paths,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Ticket 05 sparse Haller anchor study")
    parser.add_argument("--candidate-artifact-dir", required=True)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--controlled-resolution", type=int, nargs="+", default=[65, 97, 129])
    args = parser.parse_args()
    result = run_ticket05(
        candidate_artifact_dir=args.candidate_artifact_dir,
        protocol_manifest=args.protocol_manifest,
        out_dir=args.out_dir,
        controlled_resolutions=tuple(args.controlled_resolution),
    )
    print(json.dumps({
        "status": result["status"],
        "trusted_training_anchor": result["trusted_training_anchor"],
        "bundle": result.get("paths", {}).get("artifact_dir"),
        "qualification_report_fingerprint": result["qualification_report"][
            "qualification_report_fingerprint"
        ],
        "relationship_report_fingerprint": (
            None
            if result["relationship_report"] is None
            else result["relationship_report"]["report_fingerprint"]
        ),
    }, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] in {"published", QUALIFICATION_STATUS_NO_GO} else 1


_SAMPLE_ARRAY_FILES = {
    "anchor_mask": "anchor_mask.npy",
    "sparse_state": "sparse_state.npy",
    "valid_fluid": "valid_fluid.npy",
    "solid_mask": "solid_mask.npy",
    "invalid_support": "invalid_support.npy",
}
_SAMPLE_JSON_FILES = {
    "center_records": "center_records.json",
    "raw_contours": "raw_contours.json",
}
_SAMPLE_METADATA_FILE = "sparse_anchor_meta.json"
_SAMPLE_INTEGRITY_FILE = "sparse_anchor_integrity.json"


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sample_integrity_payload(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    json_layers: Mapping[str, Any],
) -> dict[str, Any]:
    clean_metadata = _json_copy(metadata)
    clean_metadata.pop("integrity_fingerprint", None)
    return {
        "metadata": clean_metadata,
        "array_fingerprints": {
            name: _hash_array(arrays[name]) for name in sorted(arrays)
        },
        "json_fingerprints": {
            name: _json_hash(json_layers[name]) for name in sorted(json_layers)
        },
    }


def _sample_arrays_from_result(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = tuple(_SAMPLE_ARRAY_FILES)
    missing = [name for name in required if name not in result]
    if missing:
        raise SparseAnchorContractError(f"sparse sample 缺少 arrays={missing}")
    arrays = {
        name: np.asarray(result[name]).copy() for name in required
    }
    shape = arrays["anchor_mask"].shape
    if arrays["anchor_mask"].dtype != np.dtype(bool):
        arrays["anchor_mask"] = arrays["anchor_mask"].astype(bool)
    for name in ("valid_fluid", "solid_mask", "invalid_support"):
        arrays[name] = arrays[name].astype(bool)
    if any(arrays[name].shape != shape for name in arrays):
        raise SparseAnchorContractError("sparse sample arrays shape 不一致")
    arrays["sparse_state"] = arrays["sparse_state"].astype("<U20")
    if not np.all(np.isin(arrays["sparse_state"], list(SPARSE_STATES))):
        raise SparseAnchorContractError("sparse_state 含未注册状态")
    _validate_support(
        arrays["solid_mask"], arrays["invalid_support"], arrays["valid_fluid"]
    )
    if np.any(arrays["anchor_mask"] & ~arrays["valid_fluid"]):
        raise SparseAnchorContractError("positive anchor 不能超出 valid-fluid support")
    if not np.array_equal(
        arrays["sparse_state"] == STATE_POSITIVE_ANCHOR,
        arrays["anchor_mask"],
    ):
        raise SparseAnchorContractError(
            "sparse artifact integrity/schema: sparse_state positive_anchor 与 anchor_mask 不一致"
        )
    return arrays


def save_sparse_anchor_artifact(
    result: Mapping[str, Any],
    out_dir: str | pathlib.Path,
    *,
    artifact_state: str = "candidate",
    qualification_report_fingerprint: str | None = None,
) -> dict[str, str]:
    """Persist one sample with a manifest-like metadata and integrity sidecar."""
    if artifact_state not in ARTIFACT_STATES:
        raise SparseAnchorContractError(
            f"artifact_state={artifact_state!r} 不在 {ARTIFACT_STATES}"
        )
    if artifact_state == "qualified" and not qualification_report_fingerprint:
        raise SparseAnchorContractError(
            "qualified sparse artifact 必须绑定 qualification_report_fingerprint"
        )
    out = pathlib.Path(out_dir)
    if out.exists():
        if any(out.iterdir()):
            raise FileExistsError(f"拒绝覆盖已有 sparse-anchor artifact: {out}")
        # A caller may have created an empty destination directory.  Removing
        # that empty placeholder lets the fully written staging directory be
        # published as one unit without touching any material output.
        out.rmdir()
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(prefix=".sparse-anchor-staging-", dir=out.parent))
    try:
        arrays = _sample_arrays_from_result(result)
        metadata = _json_copy(result.get("metadata", {}))
        if not isinstance(metadata, dict):
            raise SparseAnchorContractError("result.metadata 必须是 mapping")
        metadata.update({
            "artifact_type": SAMPLE_ARTIFACT_TYPE,
            "protocol_version": PROTOCOL_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "artifact_state": artifact_state,
            "allowed_for_training_anchor": artifact_state == "qualified",
            "eligible_as_dense_evaluation_reference": False,
            "trusted_haller_v2": False,
            "qualification_report_fingerprint": qualification_report_fingerprint,
            "artifact_files": {
                "arrays": dict(_SAMPLE_ARRAY_FILES),
                "json": dict(_SAMPLE_JSON_FILES),
                "metadata": _SAMPLE_METADATA_FILE,
                "integrity": _SAMPLE_INTEGRITY_FILE,
            },
        })
        json_layers = {
            "center_records": _json_copy(result.get("center_records", [])),
            "raw_contours": _json_copy(result.get("raw_contours", [])),
        }
        if not isinstance(json_layers["center_records"], list):
            raise SparseAnchorContractError("center_records 必须是 list")
        if not isinstance(json_layers["raw_contours"], list):
            raise SparseAnchorContractError("raw_contours 必须是 list")
        metadata["anchor_count"] = int(np.count_nonzero(arrays["anchor_mask"]))
        metadata["state_counts"] = _state_counts(arrays["sparse_state"])
        metadata["array_fingerprints"] = {
            name: _hash_array(value) for name, value in sorted(arrays.items())
        }
        metadata["raw_contours_fingerprint"] = _json_hash(json_layers["raw_contours"])
        metadata.pop("integrity_fingerprint", None)
        metadata["integrity_fingerprint"] = _json_hash(
            _sample_integrity_payload(metadata, arrays, json_layers)
        )
        for name, filename in _SAMPLE_ARRAY_FILES.items():
            np.save(stage / filename, arrays[name], allow_pickle=False)
        for name, filename in _SAMPLE_JSON_FILES.items():
            _write_json(stage / filename, json_layers[name])
        _write_json(stage / _SAMPLE_METADATA_FILE, metadata)
        file_hashes = {
            filename: _file_sha256(stage / filename)
            for filename in (
                *_SAMPLE_ARRAY_FILES.values(),
                *_SAMPLE_JSON_FILES.values(),
                _SAMPLE_METADATA_FILE,
            )
        }
        sidecar = {
            "artifact_type": INTEGRITY_SIDECAR_TYPE,
            "protocol_version": PROTOCOL_VERSION,
            "schema_version": "haller-sparse-physical-anchor-integrity-v1",
            "artifact_fingerprint": metadata["integrity_fingerprint"],
            "metadata_fingerprint": _json_hash(metadata),
            "file_hashes": file_hashes,
            "final_test_relationships": "not_read",
        }
        _write_json(stage / _SAMPLE_INTEGRITY_FILE, sidecar)
        shutil.move(str(stage), str(out))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "artifact_dir": str(out),
        "metadata": str(out / _SAMPLE_METADATA_FILE),
        "integrity": str(out / _SAMPLE_INTEGRITY_FILE),
        "artifact_fingerprint": str(metadata["integrity_fingerprint"]),
    }


def load_sparse_anchor_artifact(
    out_dir: str | pathlib.Path,
    *,
    require_training_anchor: bool = False,
    require_dense_reference: bool = False,
) -> dict[str, Any]:
    """Read a sample and verify every persisted file before returning it."""
    out = pathlib.Path(out_dir)
    metadata_path = out / _SAMPLE_METADATA_FILE
    integrity_path = out / _SAMPLE_INTEGRITY_FILE
    if not metadata_path.exists() or not integrity_path.exists():
        raise SparseAnchorContractError("sparse artifact 缺少 metadata/integrity sidecar")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sidecar = json.loads(integrity_path.read_text(encoding="utf-8"))
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise SparseAnchorContractError("sparse artifact protocol_version 不匹配")
    if metadata.get("artifact_type") != SAMPLE_ARTIFACT_TYPE:
        raise SparseAnchorContractError("sparse artifact artifact_type 不匹配")
    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise SparseAnchorContractError("sparse artifact artifact_version 不匹配")
    state = metadata.get("artifact_state")
    if state not in ARTIFACT_STATES:
        raise SparseAnchorContractError("sparse artifact artifact_state 无效")
    if metadata.get("allowed_for_training_anchor") is not (state == "qualified"):
        raise SparseAnchorContractError("sparse artifact training-anchor guard 与 state 不一致")
    if metadata.get("eligible_as_dense_evaluation_reference") is not False:
        raise SparseAnchorContractError("sparse artifact dense-reference guard 被解除")
    if metadata.get("trusted_haller_v2") is not False:
        raise SparseAnchorContractError("sparse artifact trusted_haller_v2 guard 被解除")
    if require_training_anchor and (
        state != "qualified" or metadata.get("allowed_for_training_anchor") is not True
    ):
        raise PermissionError(
            "candidate/audited/failed/unqualified sparse artifact 不能作为 formal training anchor"
        )
    if require_dense_reference or metadata.get("eligible_as_dense_evaluation_reference") is True:
        raise PermissionError(
            "qualified sparse anchor 不是 dense segmentation reference"
        )
    if sidecar.get("artifact_type") != INTEGRITY_SIDECAR_TYPE:
        raise SparseAnchorContractError("sparse artifact integrity sidecar type 不匹配")
    if sidecar.get("protocol_version") != PROTOCOL_VERSION:
        raise SparseAnchorContractError("sparse artifact integrity sidecar protocol 不匹配")
    if sidecar.get("final_test_relationships") != "not_read":
        raise FinalTestAccessError("sparse artifact sidecar 没有封存 final-test relationships")
    arrays = {
        name: np.load(out / filename, allow_pickle=False)
        for name, filename in _SAMPLE_ARRAY_FILES.items()
    }
    arrays = _sample_arrays_from_result(arrays)
    json_layers = {
        name: json.loads((out / filename).read_text(encoding="utf-8"))
        for name, filename in _SAMPLE_JSON_FILES.items()
    }
    expected_fingerprint = _json_hash(
        _sample_integrity_payload(metadata, arrays, json_layers)
    )
    if metadata.get("integrity_fingerprint") != expected_fingerprint:
        raise ValueError("sparse artifact integrity fingerprint 校验失败")
    if sidecar.get("artifact_fingerprint") != expected_fingerprint:
        raise ValueError("sparse artifact sidecar fingerprint 校验失败")
    if sidecar.get("metadata_fingerprint") != _json_hash(metadata):
        raise ValueError("sparse artifact metadata fingerprint 校验失败")
    declared_hashes = sidecar.get("file_hashes", {})
    for filename in (
        *_SAMPLE_ARRAY_FILES.values(),
        *_SAMPLE_JSON_FILES.values(),
        _SAMPLE_METADATA_FILE,
    ):
        if declared_hashes.get(filename) != _file_sha256(out / filename):
            raise ValueError(f"sparse artifact file hash 校验失败: {filename}")
    state_counts = _state_counts(arrays["sparse_state"])
    if metadata.get("state_counts") != state_counts:
        raise ValueError("sparse artifact state_counts 与数组不一致")
    return {
        **arrays,
        **json_layers,
        "metadata": metadata,
        "readback": {
            "schema_verified": True,
            "integrity_verified": True,
            "file_hashes_verified": True,
            "artifact_fingerprint": expected_fingerprint,
        },
    }


read_sparse_anchor_artifact = load_sparse_anchor_artifact
write_sparse_anchor_artifact = save_sparse_anchor_artifact


__all__ = [
    "ANCHOR_PRECISION_GATE",
    "ARTIFACT_STATES",
    "ARTIFACT_TYPE",
    "ARTIFACT_VERSION",
    "CONSTRUCTION_CENTER_CELL",
    "CONSTRUCTION_SMALL_CORE_R1",
    "CONTROLLED_FAILURE_RATE_GATE",
    "ControlledFixture",
    "FinalTestAccessError",
    "INTEGRITY_SIDECAR_TYPE",
    "MANIFEST_TYPE",
    "PROTOCOL_VERSION",
    "QUALIFICATION_STATUS_NO_GO",
    "QUALIFICATION_STATUS_QUALIFIED",
    "QUALIFICATION_REPORT_TYPE",
    "REAL_DATASET_NAMES",
    "REGISTERED_CONSTRUCTIONS",
    "RELATIONSHIP_REPORT_TYPE",
    "SAMPLE_ARTIFACT_TYPE",
    "SPARSE_STATES",
    "STATE_EXTRACTOR_FAILURE",
    "STATE_INVALID",
    "STATE_POSITIVE_ANCHOR",
    "STATE_SOLID",
    "STATE_UNCOVERED",
    "SparseAnchorContractError",
    "build_controlled_fixtures",
    "build_sparse_anchor",
    "build_sparse_anchor_from_evidence",
    "load_sparse_anchor_artifact",
    "load_sparse_anchor_bundle",
    "load_numbacs_candidate_samples",
    "qualify_sparse_anchor_constructions",
    "read_qualification_report",
    "read_relationship_report",
    "read_sparse_anchor_artifact",
    "run_anchor_percentile_relationship_study",
    "run_ticket05",
    "save_qualification_report",
    "save_relationship_report",
    "save_sparse_anchor_artifact",
    "save_sparse_anchor_bundle",
    "write_sparse_anchor_artifact",
]


if __name__ == "__main__":
    raise SystemExit(_cli())
