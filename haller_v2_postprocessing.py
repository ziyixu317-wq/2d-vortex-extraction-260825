"""Audit and freeze post-processing for the fixed NumbaCS raw contours.

This module is deliberately downstream of :mod:`numbacs_contour_audit`.  It
never calls the native extractor.  Every configuration in an audit consumes
the same serialized raw-contour candidate and carries its raw-contour
fingerprint through the resulting evidence layers.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt

import experiment_protocol
import haller_anchors


HALLER_V2_VERSION = "haller-v2"
PHYSICAL_REGION_ARTIFACT_VERSION = "haller-v2-physical-region-v1"
POSTPROCESS_SCHEMA_VERSION = "haller-v2-postprocessing-r2-physical-region-v2"
REMEDIATION_PROTOCOL_VERSION = "haller-v2-physical-region-remediation-v1"
BAND_FACTORS = (0.0, 0.5, 1.0, 1.5, 2.0)
RASTERIZATION_POLICIES = ("cell_center", "four_sample_majority")
NEGATIVE_POLICIES = ("frame_p60_far_field", "all_far_field_negative")
GEOMETRY_POLICIES = (
    "all_contour_band",
    "physical_outer_boundary_band",
)
HALLER_V2_ARRAY_FILES = frozenset({
    "standard_ivd.npy",
    "solid_mask.npy",
    "invalid_support.npy",
    "valid_fluid.npy",
    "filled_interior.npy",
    "unknown_band.npy",
    "negative_candidates.npy",
    "evidence_state.npy",
    "xdim.npy",
    "ydim.npy",
})
HALLER_V2_JSON_FILES = frozenset({"raw_contours.json", "filtering_decisions.json"})
R1_CANDIDATE_RUN_FILES = frozenset({
    "standard_ivd.npy",
    "solid_mask.npy",
    "xdim.npy",
    "ydim.npy",
    "raw_contours.json",
    "filtering_decisions.json",
    "run_metrics.json",
})
POSITIVE_RAW_INTERIOR_MINIMUM = 0.25
FULLY_ERASED_MAXIMUM = 0.05
SELECTED_CENTER_RETENTION_MINIMUM = 0.80
MATCHED_CORE_IOU_MINIMUM = 0.50
RELATIVE_KNOWN_COVERAGE_CHANGE_MAXIMUM = 0.50
EQUIVALENT_RADIUS_CV_MAXIMUM = 0.15
# These are the registered controlled-fixture gates from the shared spec.
CONTROLLED_CORE_IOU_MINIMUM = 0.70
CONTROLLED_RADIUS_RELATIVE_ERROR_MAXIMUM = 0.20
DEVELOPMENT_KNOWN_COVERAGE_MINIMUM = 0.05
DEVELOPMENT_KNOWN_COVERAGE_MAXIMUM = 0.95
DEVELOPMENT_POSITIVE_COVERAGE_MINIMUM = 0.0001
DEVELOPMENT_POSITIVE_COVERAGE_MAXIMUM = 0.25
DEVELOPMENT_HARD_FAILURE_RATE_MAXIMUM = 0.02
CONTROLLED_DIAGNOSTIC_CONFIGURATION = {
    "band_factor": 1.0,
    "rasterization_policy": "cell_center",
    "negative_policy": "all_far_field_negative",
    "geometry_policy": "physical_outer_boundary_band",
}
IDENTITY_CATEGORIES = (
    "nested",
    "root",
    "independent_proven",
    "unresolved",
)


@dataclass(frozen=True)
class PostprocessingConfig:
    """One pre-registered geometry, rasterization, band, and negative choice."""

    band_factor: float
    rasterization_policy: str
    negative_policy: str
    geometry_policy: str = "all_contour_band"

    def __post_init__(self) -> None:
        band = float(self.band_factor)
        if not np.isfinite(band) or band not in BAND_FACTORS:
            raise ValueError(f"band_factor must be one of {BAND_FACTORS!r}")
        if self.rasterization_policy not in RASTERIZATION_POLICIES:
            raise ValueError(
                "rasterization_policy must be one of "
                f"{RASTERIZATION_POLICIES!r}"
            )
        if self.negative_policy not in NEGATIVE_POLICIES:
            raise ValueError(
                "negative_policy must be one of " f"{NEGATIVE_POLICIES!r}"
            )
        if self.geometry_policy not in GEOMETRY_POLICIES:
            raise ValueError(
                "geometry_policy must be one of "
                f"{GEOMETRY_POLICIES!r}"
            )
        object.__setattr__(self, "band_factor", band)

    def as_dict(self) -> dict[str, Any]:
        return {
            "band_factor": float(self.band_factor),
            "rasterization_policy": self.rasterization_policy,
            "negative_policy": self.negative_policy,
            "geometry_policy": self.geometry_policy,
        }


def complete_configuration_grid() -> tuple[PostprocessingConfig, ...]:
    """Return the current all-contour post-processing grid."""

    return tuple(
        PostprocessingConfig(
            band_factor=band_factor,
            rasterization_policy=rasterization_policy,
            negative_policy=negative_policy,
            geometry_policy="all_contour_band",
        )
        for rasterization_policy in RASTERIZATION_POLICIES
        for negative_policy in NEGATIVE_POLICIES
        for band_factor in BAND_FACTORS
    )


def complete_geometry_configuration_grid() -> tuple[PostprocessingConfig, ...]:
    """Return the fixed-band comparison grid for both geometry semantics."""

    return tuple(
        PostprocessingConfig(
            band_factor=band_factor,
            rasterization_policy=rasterization_policy,
            negative_policy=negative_policy,
            geometry_policy=geometry_policy,
        )
        for geometry_policy in GEOMETRY_POLICIES
        for rasterization_policy in RASTERIZATION_POLICIES
        for negative_policy in NEGATIVE_POLICIES
        for band_factor in BAND_FACTORS
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _postprocessing_runtime_identity() -> dict[str, str]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def _uniform_spacing(axis: Any, *, name: str) -> float:
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional axis")
    spacing = float(values[1] - values[0])
    if spacing <= 0.0 or not np.allclose(
        np.diff(values), spacing, rtol=1e-3, atol=1e-12
    ):
        raise ValueError(f"{name} must be uniformly increasing")
    return spacing


def _rasterize_contour(
    points_grid: Any,
    shape: tuple[int, int],
    policy: str,
) -> np.ndarray:
    points = np.asarray(points_grid, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        return np.zeros(shape, dtype=bool)
    if policy == "cell_center":
        return np.asarray(
            haller_anchors.grid_points_in_poly(shape, points), dtype=bool
        )
    if policy != "four_sample_majority":
        raise ValueError(f"unsupported rasterization policy: {policy!r}")
    # points_grid is expressed in (row, column) coordinates.  Test four
    # quarter-cell samples and require a majority; shifting the polygon is
    # equivalent to evaluating the fixed grid at the sub-cell location.
    samples = []
    for row_offset, column_offset in (
        (-0.25, -0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (0.25, 0.25),
    ):
        shifted = points - np.asarray([row_offset, column_offset])
        samples.append(
            np.asarray(
                haller_anchors.grid_points_in_poly(shape, shifted), dtype=bool
            )
        )
    return np.count_nonzero(np.stack(samples, axis=0), axis=0) >= 2


def _boundary_band(
    interior: np.ndarray,
    *,
    dx: float,
    dy: float,
    band_factor: float,
) -> np.ndarray:
    if band_factor <= 0.0 or not np.any(interior):
        return np.zeros_like(interior, dtype=bool)
    width = float(band_factor) * max(float(dx), float(dy))
    outside_distance = distance_transform_edt(~interior, sampling=(dy, dx))
    inside_distance = distance_transform_edt(interior, sampling=(dy, dx))
    return (
        (interior & (inside_distance <= width))
        | (~interior & (outside_distance <= width))
    )


def _valid_raw_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for ordinal, source in enumerate(records):
        record = dict(source)
        status = str(record.get("status", "unknown"))
        points = np.asarray(record.get("points_grid", []), dtype=np.float64)
        if status != "valid" or points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
            continue
        record.setdefault("raw_record_index", ordinal)
        valid.append(record)
    return valid


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(left & right) / union)


def _center_from_record(record: Mapping[str, Any], xdim: np.ndarray, ydim: np.ndarray) -> tuple[float, float] | None:
    value = record.get("center_xy")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        try:
            center = (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            center = None
        if center is not None and all(np.isfinite(item) for item in center):
            return center
    points = np.asarray(record.get("points_grid", []), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        return None
    rows = np.clip(np.rint(points[:, 0]).astype(int), 0, len(ydim) - 1)
    columns = np.clip(np.rint(points[:, 1]).astype(int), 0, len(xdim) - 1)
    return (float(np.mean(xdim[columns])), float(np.mean(ydim[rows])))


def _record_area(
    record: Mapping[str, Any],
    interior: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> float:
    try:
        area = float(record.get("area", float("nan")))
    except (TypeError, ValueError):
        area = float("nan")
    if np.isfinite(area) and area > 0.0:
        return area
    return float(np.count_nonzero(interior) * dx * dy)


def _upstream_peak_identity(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read an identity that was actually emitted by the native raw record.

    Geometry-derived centroids and contour order are deliberately not treated
    as peak lineage.  The native adapter currently exposes peak coordinates,
    while future adapters may expose an explicit peak/topology identifier.
    """

    for field in (
        "topology_id",
        "peak_id",
        "component_id",
        "vortex_id",
    ):
        value = record.get(field)
        if isinstance(value, Mapping):
            payload = dict(value)
            if payload:
                return {
                    "kind": f"native_{field}",
                    "value": _jsonable(payload),
                    "source_field": field,
                }
        elif value is not None and not isinstance(value, (str, bytes)):
            return {
                "kind": f"native_{field}",
                "value": _jsonable(value),
                "source_field": field,
            }
        elif isinstance(value, str) and value.strip():
            return {
                "kind": f"native_{field}",
                "value": value,
                "source_field": field,
            }

    peak = record.get("peak")
    if not isinstance(peak, Mapping):
        return None
    row = peak.get("row")
    column = peak.get("col", peak.get("column"))
    try:
        row_value = float(row)
        column_value = float(column)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(row_value) or not np.isfinite(column_value):
        return None
    if row_value != round(row_value) or column_value != round(column_value):
        return None
    return {
        "kind": "native_peak_coordinate",
        "row": int(round(row_value)),
        "col": int(round(column_value)),
        "source_field": "peak",
    }


def _identity_key(identity: Mapping[str, Any] | None) -> str | None:
    if identity is None:
        return None
    return json.dumps(
        _jsonable(identity), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _consolidate_physical_regions(
    records: Sequence[Mapping[str, Any]],
    interiors: Sequence[np.ndarray],
    *,
    xdim: np.ndarray,
    ydim: np.ndarray,
    dx: float,
    dy: float,
) -> list[dict[str, Any]]:
    """Group nested raw contours into physical regions.

    A smaller contour is assigned to the smallest larger contour whose
    rasterized interior contains the complete smaller mask, whose reported
    centers agree within the existing controlled tolerance
    ``2 * max(dx, dy)``, and whose explicit native peak/topology identities
    match.  Requiring all three proofs is conservative: containment without
    center continuity or matching upstream identity cannot prove that two
    contours belong to one physical vortex, so those contours remain separate
    regions with unresolved root attribution.
    """

    prepared: list[dict[str, Any]] = []
    for ordinal, (record, interior) in enumerate(zip(records, interiors)):
        raw_record_index = int(record.get("raw_record_index", ordinal))
        center = _center_from_record(record, xdim, ydim)
        prepared.append(
            {
                "position": ordinal,
                "raw_record_index": raw_record_index,
                "center_xy": center,
                "interior": np.asarray(interior, dtype=bool),
                "area": _record_area(record, interior, dx=dx, dy=dy),
                "upstream_identity": _upstream_peak_identity(record),
            }
        )
    ordered = sorted(
        range(len(prepared)),
        key=lambda index: (
            -float(prepared[index]["area"]),
            -int(np.count_nonzero(prepared[index]["interior"])),
            int(prepared[index]["raw_record_index"]),
        ),
    )
    parents: dict[int, int | None] = {}
    parent_evidence: dict[int, dict[str, Any]] = {}
    order_position = {index: position for position, index in enumerate(ordered)}
    for inner_index in ordered:
        inner = prepared[inner_index]
        candidates: list[int] = []
        for outer_index in ordered:
            if outer_index == inner_index:
                continue
            outer = prepared[outer_index]
            if float(outer["area"]) < float(inner["area"]):
                continue
            if (
                float(outer["area"]) == float(inner["area"])
                and order_position[outer_index] >= order_position[inner_index]
            ):
                continue
            inner_mask = np.asarray(inner["interior"], dtype=bool)
            outer_mask = np.asarray(outer["interior"], dtype=bool)
            if not np.all(~inner_mask | outer_mask):
                continue
            outer_center = outer["center_xy"]
            inner_center = inner["center_xy"]
            if outer_center is None or inner_center is None:
                continue
            center_distance = float(
                np.linalg.norm(
                    np.asarray(inner_center, dtype=np.float64)
                    - np.asarray(outer_center, dtype=np.float64)
                )
            )
            center_tolerance = 2.0 * max(dx, dy)
            if center_distance > center_tolerance:
                continue
            row = int(np.argmin(np.abs(ydim - float(inner_center[1]))))
            column = int(np.argmin(np.abs(xdim - float(inner_center[0]))))
            if not bool(outer["interior"][row, column]):
                continue
            inner_identity = inner["upstream_identity"]
            outer_identity = outer["upstream_identity"]
            if inner_identity is None or outer_identity is None:
                continue
            if _identity_key(inner_identity) != _identity_key(outer_identity):
                continue
            candidates.append(outer_index)
        parent = (
            min(
                candidates,
                key=lambda index: (
                    float(prepared[index]["area"]),
                    int(prepared[index]["raw_record_index"]),
                ),
            )
            if candidates
            else None
        )
        parents[inner_index] = parent
        if parent is not None:
            outer = prepared[parent]
            inner_center = inner["center_xy"]
            outer_center = outer["center_xy"]
            assert inner_center is not None and outer_center is not None
            center_distance = float(
                np.linalg.norm(
                    np.asarray(inner_center, dtype=np.float64)
                    - np.asarray(outer_center, dtype=np.float64)
                )
            )
            inner_identity = inner["upstream_identity"]
            outer_identity = outer["upstream_identity"]
            parent_evidence[inner_index] = {
                "child_raw_record_index": int(inner["raw_record_index"]),
                "parent_raw_record_index": int(outer["raw_record_index"]),
                "containment": {
                    "complete": True,
                    "child_interior_cells": int(np.count_nonzero(inner["interior"])),
                    "parent_interior_cells": int(np.count_nonzero(outer["interior"])),
                },
                "center_continuity": {
                    "distance": center_distance,
                    "tolerance": 2.0 * max(dx, dy),
                    "pass": True,
                },
                "topology": {
                    "status": "same_upstream_peak_or_topology_identity",
                    "child_identity": inner_identity,
                    "parent_identity": outer_identity,
                    "identities_match": True,
                },
            }

    roots: dict[int, list[int]] = {}
    for index in ordered:
        root = index
        seen: set[int] = set()
        while parents[root] is not None:
            if root in seen:
                cycle_ids = [
                    int(prepared[item]["raw_record_index"])
                    for item in sorted(seen)
                ]
                raise ValueError(
                    "physical-region parent cycle for raw contours: "
                    f"{cycle_ids}"
                )
            seen.add(root)
            root = int(parents[root])
        roots.setdefault(root, []).append(index)

    root_identity_counts: dict[str, int] = {}
    for root in roots:
        identity_key = _identity_key(prepared[root]["upstream_identity"])
        if identity_key is not None:
            root_identity_counts[identity_key] = root_identity_counts.get(identity_key, 0) + 1

    regions: list[dict[str, Any]] = []
    for region_id, root in enumerate(
        sorted(
            roots,
            key=lambda index: (
                -float(prepared[index]["area"]),
                int(prepared[index]["raw_record_index"]),
            ),
        )
    ):
        members = sorted(
            roots[root],
            key=lambda index: (
                -float(prepared[index]["area"]),
                int(prepared[index]["raw_record_index"]),
            ),
        )
        outer = prepared[members[0]]
        region_interior = np.zeros_like(outer["interior"], dtype=bool)
        for member_index in members:
            region_interior |= prepared[member_index]["interior"]
        center = outer["center_xy"]
        if center is None:
            centers = [item["center_xy"] for item in (prepared[index] for index in members)]
            known_centers = [item for item in centers if item is not None]
            if known_centers:
                center = tuple(
                    float(np.mean([item[axis] for item in known_centers]))
                    for axis in (0, 1)
                )
        root_has_children = any(
            parent == root for parent in parents.values()
        )
        root_identity = outer["upstream_identity"]
        root_identity_key = _identity_key(root_identity)
        if root_has_children:
            root_category = "root"
        elif (
            root_identity_key is not None
            and len(roots) > 1
            and root_identity_counts.get(root_identity_key, 0) == 1
        ):
            root_category = "independent_proven"
        else:
            root_category = "unresolved"
        identity_attributes: list[dict[str, Any]] = []
        for member_index in members:
            member = prepared[member_index]
            if parents[member_index] is not None:
                category = "nested"
                proof = parent_evidence[member_index]
            elif member_index == root:
                category = root_category
                proof = {
                    "basis": (
                        "physical-region root with nested child contours"
                        if root_category == "root"
                        else (
                            "unique upstream peak/topology identity across root regions"
                            if root_category == "independent_proven"
                            else "no upstream peak/topology identity proving independence"
                        )
                    ),
                    "upstream_identity": root_identity,
                    "root_has_children": root_has_children,
                }
            else:  # pragma: no cover - every non-root member has a parent
                category = "unresolved"
                proof = {"basis": "missing_parent_evidence"}
            identity_attributes.append({
                "record_position": int(member["position"]),
                "raw_record_index": int(member["raw_record_index"]),
                "category": category,
                "upstream_identity": member["upstream_identity"],
                "proof": proof,
            })
        regions.append(
            {
                "physical_region_id": region_id,
                "raw_record_indices": [
                    int(prepared[index]["raw_record_index"]) for index in members
                ],
                "outer_raw_record_index": int(outer["raw_record_index"]),
                "nested_raw_contour_count": max(0, len(members) - 1),
                "center_xy": center,
                "interior": region_interior,
                "outer_boundary": np.asarray(outer["interior"], dtype=bool),
                "identity_attributes": identity_attributes,
                "identity_basis": {
                    "parent_edges": [
                        parent_evidence[index]
                        for index in members
                        if index in parent_evidence
                    ],
                    "root_category": root_category,
                    "root_upstream_identity": root_identity,
                },
            }
        )
    return regions


def _layer_conservation(
    *,
    solid_mask: np.ndarray,
    invalid_support: np.ndarray,
    valid_fluid: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    evidence_state: np.ndarray,
) -> dict[str, Any]:
    expected_fluid = ~(solid_mask | invalid_support)
    state_values = set(np.unique(evidence_state).tolist())
    partition = positive | negative | (valid_fluid & (evidence_state == -1))
    valid = (
        np.array_equal(valid_fluid, expected_fluid)
        and np.all(~(positive & negative))
        and np.array_equal(partition, valid_fluid)
        and state_values <= {-1, 0, 1}
        and np.all(evidence_state[positive] == 1)
        and np.all(evidence_state[negative] == 0)
        and np.all(evidence_state[~valid_fluid] == -1)
    )
    return {
        "status": "pass" if valid else "fail",
        "valid_fluid_matches_mask": bool(np.array_equal(valid_fluid, expected_fluid)),
        "positive_negative_unknown_partition": bool(np.array_equal(partition, valid_fluid)),
        "state_values": sorted(int(value) for value in state_values),
        "nonfluid_state_is_unknown": bool(np.all(evidence_state[~valid_fluid] == -1)),
    }


def _region_scale_versus_erosion(
    *,
    unit_type: str,
    units: Sequence[Mapping[str, Any]],
    valid_fluid: np.ndarray,
    erosion_band: np.ndarray,
    dx: float,
    dy: float,
    erosion_width: float,
    erosion_bands: Sequence[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Describe physical scale and core retention without changing any gate."""

    if erosion_bands is not None and len(erosion_bands) != len(units):
        raise ValueError("region-scale erosion bands must match the unit count")
    unit_records: list[dict[str, Any]] = []
    raw_cells_total = 0
    core_cells_total = 0
    erased_count = 0
    for ordinal, unit in enumerate(units):
        interior = np.asarray(unit["interior"], dtype=bool)
        raw_mask = valid_fluid & interior
        unit_erosion_band = np.asarray(
            erosion_bands[ordinal] if erosion_bands is not None else erosion_band,
            dtype=bool,
        )
        if unit_erosion_band.shape != interior.shape:
            raise ValueError("region-scale erosion band shape does not match the unit")
        core = raw_mask & ~unit_erosion_band
        raw_cells = int(np.count_nonzero(raw_mask))
        core_cells = int(np.count_nonzero(core))
        raw_area = float(raw_cells * dx * dy)
        core_area = float(core_cells * dx * dy)
        raw_radius = (
            float(np.sqrt(raw_area / np.pi)) if raw_area > 0.0 else None
        )
        core_radius = (
            float(np.sqrt(core_area / np.pi)) if core_area > 0.0 else None
        )
        fully_erased = raw_cells > 0 and core_cells == 0
        raw_cells_total += raw_cells
        core_cells_total += core_cells
        erased_count += int(fully_erased)
        if unit_type == "physical_region":
            unit_id: Any = int(unit["physical_region_id"])
            raw_record_indices = [int(item) for item in unit["raw_record_indices"]]
        else:
            unit_id = int(unit.get("raw_record_index", ordinal))
            raw_record_indices = [unit_id]
        unit_records.append({
            "unit_id": unit_id,
            "raw_record_indices": raw_record_indices,
            "raw_cells": raw_cells,
            "raw_area_physical": raw_area,
            "raw_equivalent_radius": raw_radius,
            "core_cells": core_cells,
            "core_area_physical": core_area,
            "core_equivalent_radius": core_radius,
            "core_retention": (
                float(core_cells / raw_cells) if raw_cells else None
            ),
            "positive_raw_interior_ratio": (
                float(core_cells / raw_cells) if raw_cells else None
            ),
            "fully_erased": bool(fully_erased),
            "erosion_width_physical": float(erosion_width),
        })
    return {
        "unit_type": unit_type,
        "scale_definition": "equivalent_radius_from_rasterized_physical_area",
        "erosion_width_physical": float(erosion_width),
        "units": unit_records,
        "aggregate": {
            "unit_count": len(unit_records),
            "raw_cells": raw_cells_total,
            "raw_area_physical": float(raw_cells_total * dx * dy),
            "core_cells": core_cells_total,
            "core_area_physical": float(core_cells_total * dx * dy),
            "core_retention": (
                float(core_cells_total / raw_cells_total)
                if raw_cells_total else None
            ),
            "fully_erased_count": erased_count,
            "fully_erased_fraction": (
                float(erased_count / len(unit_records))
                if unit_records else None
            ),
            "positive_raw_interior_ratio": (
                float(core_cells_total / raw_cells_total)
                if raw_cells_total else None
            ),
        },
    }


def _raw_fingerprint_from_run(run: Mapping[str, Any]) -> str:
    supplied = run.get("raw_contour_fingerprint")
    if supplied:
        return str(supplied)
    return _json_hash({
        "run_id": run.get("run_id"),
        "raw_contours": run.get("raw_contours", []),
        "standard_ivd": np.asarray(run["standard_ivd"]).tolist(),
        "solid_mask": np.asarray(run["solid_mask"]).tolist(),
        "xdim": np.asarray(run["xdim"]).tolist(),
        "ydim": np.asarray(run["ydim"]).tolist(),
    })


def postprocess_raw_run(
    run: Mapping[str, Any],
    config: PostprocessingConfig,
    *,
    include_contour_layers: bool = True,
) -> dict[str, Any]:
    """Rasterize one fixed raw-contour run and return auditable layers.

    The returned mapping contains arrays for artifact publication and JSON
    metrics for audit aggregation.  No native contour extraction occurs here.
    """

    if not isinstance(config, PostprocessingConfig):
        raise TypeError("config must be a PostprocessingConfig")
    ivd = np.asarray(run["standard_ivd"], dtype=np.float64)
    solid = np.asarray(run["solid_mask"], dtype=bool)
    if ivd.ndim != 2 or solid.shape != ivd.shape:
        raise ValueError("standard_ivd and solid_mask must have matching 2-D shapes")
    y_size, x_size = ivd.shape
    xdim = np.asarray(run["xdim"], dtype=np.float64)
    ydim = np.asarray(run["ydim"], dtype=np.float64)
    if xdim.shape != (x_size,) or ydim.shape != (y_size,):
        raise ValueError("coordinate axes do not match standard_ivd")
    dx = _uniform_spacing(xdim, name="xdim")
    dy = _uniform_spacing(ydim, name="ydim")
    invalid = np.asarray(
        run.get("invalid_support", ~np.isfinite(ivd)), dtype=bool
    )
    if invalid.shape != ivd.shape:
        raise ValueError("invalid_support does not match standard_ivd")
    invalid = invalid | ~np.isfinite(ivd)
    valid_fluid = ~(solid | invalid)
    if not np.any(valid_fluid):
        raise ValueError("postprocessing requires at least one valid fluid cell")

    records = list(run.get("raw_contours", []))
    valid_records = _valid_raw_records(records)
    shape = (y_size, x_size)
    raw_interior = np.zeros(shape, dtype=bool)
    all_contour_band = np.zeros(shape, dtype=bool)
    contour_metrics: list[dict[str, Any]] = []
    contour_layers: list[dict[str, Any]] = []
    contour_signatures: list[dict[str, Any]] = []
    interiors: list[np.ndarray] = []
    contour_bands: list[np.ndarray] = []
    for ordinal, record in enumerate(valid_records):
        interior = _rasterize_contour(
            record["points_grid"], shape, config.rasterization_policy
        )
        interiors.append(interior)
        band = _boundary_band(interior, dx=dx, dy=dy, band_factor=config.band_factor)
        contour_bands.append(band)
        raw_interior |= interior
        all_contour_band |= band
        core = valid_fluid & interior & ~band
        center = _center_from_record(record, xdim, ydim)
        raw_record_index = int(record.get("raw_record_index", ordinal))
        contour_signatures.append({
            "raw_record_index": raw_record_index,
            "center_xy": center,
            "core_packed": np.packbits(core.reshape(-1)),
            "shape": shape,
        })
        if include_contour_layers:
            contour_layers.append({
                "raw_record_index": raw_record_index,
                "center_xy": center,
                "interior": interior,
                "band": band,
                "core": core,
            })
        contour_metrics.append({
            "raw_record_index": int(record.get("raw_record_index", ordinal)),
            "center_xy": None if center is None else [center[0], center[1]],
            "raw_interior_cells": int(np.count_nonzero(valid_fluid & interior)),
            "band_cells": int(np.count_nonzero(valid_fluid & interior & band)),
            "core_cells": int(np.count_nonzero(core)),
            "fully_erased": bool(not np.any(core)),
            "equivalent_radius": float(
                np.sqrt(np.count_nonzero(core) * dx * dy / np.pi)
            ) if np.any(core) else None,
        })

    erased_count = sum(1 for item in contour_metrics if item["fully_erased"])
    physical_regions = _consolidate_physical_regions(
        valid_records,
        interiors,
        xdim=xdim,
        ydim=ydim,
        dx=dx,
        dy=dy,
    )
    physical_outer_boundary_band = np.zeros(shape, dtype=bool)
    physical_region_layers: list[dict[str, Any]] = []
    physical_region_signatures: list[dict[str, Any]] = []
    physical_region_metrics: list[dict[str, Any]] = []
    physical_region_bands: list[np.ndarray] = []
    physical_region_erased_count = 0
    nested_raw_contour_count = 0
    for region in physical_regions:
        region_band = _boundary_band(
            np.asarray(region["outer_boundary"], dtype=bool),
            dx=dx,
            dy=dy,
            band_factor=config.band_factor,
        )
        physical_region_bands.append(region_band)
        region_interior = np.asarray(region["interior"], dtype=bool)
        region_core = valid_fluid & region_interior & ~region_band
        physical_outer_boundary_band |= region_band
        raw_indices = [int(item) for item in region["raw_record_indices"]]
        center = region.get("center_xy")
        nested_count = int(region["nested_raw_contour_count"])
        nested_raw_contour_count += nested_count
        fully_erased = bool(not np.any(region_core))
        physical_region_erased_count += int(fully_erased)
        physical_region_signatures.append({
            "physical_region_id": int(region["physical_region_id"]),
            "raw_record_indices": raw_indices,
            "outer_raw_record_index": int(region["outer_raw_record_index"]),
            "center_xy": center,
            "core_packed": np.packbits(region_core.reshape(-1)),
            "shape": shape,
        })
        if include_contour_layers:
            physical_region_layers.append({
                "physical_region_id": int(region["physical_region_id"]),
                "raw_record_indices": raw_indices,
                "raw_record_index": int(region["outer_raw_record_index"]),
                "outer_raw_record_index": int(region["outer_raw_record_index"]),
                "center_xy": center,
                "interior": region_interior,
                "band": region_band,
                "core": region_core,
            })
        physical_region_metrics.append({
            "physical_region_id": int(region["physical_region_id"]),
            "raw_record_indices": raw_indices,
            "outer_raw_record_index": int(region["outer_raw_record_index"]),
            "nested_raw_contour_count": nested_count,
            "center_xy": None if center is None else [center[0], center[1]],
            "raw_interior_cells": int(np.count_nonzero(valid_fluid & region_interior)),
            "raw_area_physical": float(
                np.count_nonzero(valid_fluid & region_interior) * dx * dy
            ),
            "band_cells": int(np.count_nonzero(valid_fluid & region_interior & region_band)),
            "core_cells": int(np.count_nonzero(region_core)),
            "core_area_physical": float(np.count_nonzero(region_core) * dx * dy),
            "positive_raw_interior_ratio": (
                float(
                    np.count_nonzero(region_core)
                    / np.count_nonzero(valid_fluid & region_interior)
                )
                if np.count_nonzero(valid_fluid & region_interior)
                else None
            ),
            "fully_erased": fully_erased,
            "equivalent_radius": float(
                np.sqrt(np.count_nonzero(region_core) * dx * dy / np.pi)
            ) if np.any(region_core) else None,
            "identity_attributes": region["identity_attributes"],
            "identity_basis": region["identity_basis"],
        })

    identity_attributes = sorted(
        (
            attribute
            for region in physical_regions
            for attribute in region["identity_attributes"]
        ),
        key=lambda item: int(item["record_position"]),
    )
    identity_counts = {category: 0 for category in IDENTITY_CATEGORIES}
    attributes_by_position: dict[int, Mapping[str, Any]] = {}
    for attribute in identity_attributes:
        category = str(attribute["category"])
        if category not in identity_counts:
            raise ValueError(f"unknown raw contour identity category: {category!r}")
        identity_counts[category] += 1
        position = int(attribute["record_position"])
        if position in attributes_by_position:
            raise ValueError(f"duplicate raw contour identity position: {position}")
        attributes_by_position[position] = attribute
    if len(identity_attributes) != len(valid_records):
        raise ValueError(
            "raw contour identity attribution does not conserve accepted contours"
        )
    for position, contour_metric in enumerate(contour_metrics):
        attribute = attributes_by_position[position]
        contour_metric["identity_attribution"] = {
            "category": attribute["category"],
            "upstream_identity": attribute["upstream_identity"],
            "proof": attribute["proof"],
        }

    if config.geometry_policy == "all_contour_band":
        unknown_band = all_contour_band
        geometry_erosion_unit = "raw_contour"
        geometry_unit_count = len(contour_metrics)
        fully_erased_geometry_count = erased_count
    elif config.geometry_policy == "physical_outer_boundary_band":
        unknown_band = physical_outer_boundary_band
        geometry_erosion_unit = "physical_region"
        geometry_unit_count = len(physical_region_metrics)
        fully_erased_geometry_count = physical_region_erased_count
    else:  # pragma: no cover - guarded by PostprocessingConfig
        raise ValueError(f"unsupported geometry policy: {config.geometry_policy!r}")

    if config.geometry_policy == "all_contour_band":
        scale_units = [
            {
                "raw_record_index": int(record.get("raw_record_index", ordinal)),
                "interior": interior,
                "band": contour_bands[ordinal],
            }
            for ordinal, (record, interior) in enumerate(
                zip(valid_records, interiors)
            )
        ]
        scale_unit_type = "raw_contour"
    else:
        scale_units = [
            {
                "physical_region_id": int(region["physical_region_id"]),
                "raw_record_indices": [
                    int(item) for item in region["raw_record_indices"]
                ],
                "interior": np.asarray(region["interior"], dtype=bool),
                "band": physical_region_bands[ordinal],
            }
                for ordinal, region in enumerate(physical_regions)
            ]
        scale_unit_type = "physical_region"
    scale_attribution = _region_scale_versus_erosion(
        unit_type=scale_unit_type,
        units=scale_units,
        valid_fluid=valid_fluid,
        erosion_band=unknown_band,
        dx=dx,
        dy=dy,
        erosion_width=float(config.band_factor) * max(dx, dy),
        erosion_bands=[np.asarray(unit["band"], dtype=bool) for unit in scale_units],
    )

    fluid_frame_p60 = float(np.percentile(ivd[valid_fluid], 60))
    outside_far_field = valid_fluid & ~raw_interior & ~unknown_band
    if config.negative_policy == "frame_p60_far_field":
        negative = outside_far_field & (ivd <= fluid_frame_p60)
        outside_high_ivd = outside_far_field & (ivd > fluid_frame_p60)
    elif config.negative_policy == "all_far_field_negative":
        negative = outside_far_field
        outside_high_ivd = np.zeros_like(outside_far_field, dtype=bool)
    else:  # pragma: no cover - guarded by PostprocessingConfig
        raise ValueError(f"unsupported negative policy: {config.negative_policy!r}")
    positive = valid_fluid & raw_interior & ~unknown_band
    # The order is intentional: known positive/negative are mutually
    # exclusive and every non-fluid cell stays unknown.
    evidence_state = np.full(shape, -1, dtype=np.int8)
    evidence_state[positive] = 1
    evidence_state[negative] = 0
    conservation = _layer_conservation(
        solid_mask=solid,
        invalid_support=invalid,
        valid_fluid=valid_fluid,
        positive=positive,
        negative=negative,
        evidence_state=evidence_state,
    )
    if conservation["status"] != "pass":
        raise ValueError("postprocessing layer conservation failed")

    valid_count = int(np.count_nonzero(valid_fluid))
    positive_count = int(np.count_nonzero(positive))
    negative_count = int(np.count_nonzero(negative))
    unknown_fluid = valid_fluid & (evidence_state == -1)
    unknown_count = int(np.count_nonzero(unknown_fluid))
    raw_count = int(np.count_nonzero(valid_fluid & raw_interior))
    accepted_count = len(contour_metrics)
    fully_erased_physical_region_fraction = (
        float(physical_region_erased_count / len(physical_region_metrics))
        if physical_region_metrics else None
    )
    fully_erased_geometry_fraction = (
        float(fully_erased_geometry_count / geometry_unit_count)
        if geometry_unit_count else None
    )
    cell_area_physical = float(dx * dy)
    all_contour_positive = valid_fluid & raw_interior & ~all_contour_band
    physical_positive = valid_fluid & raw_interior & ~physical_outer_boundary_band
    metrics = {
        "geometry_policy": config.geometry_policy,
        "geometry_erosion_unit": geometry_erosion_unit,
        "geometry_unit_count": geometry_unit_count,
        "fully_erased_geometry_count": fully_erased_geometry_count,
        "fully_erased_geometry_fraction": fully_erased_geometry_fraction,
        "valid_fluid_cells": valid_count,
        "cell_area_physical": cell_area_physical,
        "solid_cells": int(np.count_nonzero(solid)),
        "invalid_support_cells": int(np.count_nonzero(invalid)),
        "raw_interior_cells": raw_count,
        "raw_contour_count": accepted_count,
        "raw_interior_area_physical": float(raw_count * cell_area_physical),
        "boundary_band_cells": int(np.count_nonzero(valid_fluid & unknown_band)),
        "boundary_band_area_physical": float(
            np.count_nonzero(valid_fluid & unknown_band) * cell_area_physical
        ),
        "positive_cells": positive_count,
        "positive_area_physical": float(positive_count * cell_area_physical),
        "negative_cells": negative_count,
        "unknown_fluid_cells": unknown_count,
        "known_coverage": float((positive_count + negative_count) / valid_count),
        "unknown_coverage": float(unknown_count / valid_count),
        "positive_raw_interior_ratio": (
            float(positive_count / raw_count) if raw_count else None
        ),
        "accepted_raw_contour_count": accepted_count,
        "fully_erased_contour_count": erased_count,
        "fully_erased_contour_fraction": (
            float(erased_count / accepted_count) if accepted_count else None
        ),
        "fully_erased_physical_region_count": physical_region_erased_count,
        "fully_erased_physical_region_fraction": fully_erased_physical_region_fraction,
        "physical_region_count": len(physical_region_metrics),
        "nested_raw_contour_count": nested_raw_contour_count,
        "frame_fluid_p60": fluid_frame_p60,
        "unknown_attribution": {
            "boundary_band": int(np.count_nonzero(valid_fluid & unknown_band)),
            "outside_high_ivd": int(np.count_nonzero(outside_high_ivd)),
            "total_unknown_fluid": unknown_count,
            "solid": int(np.count_nonzero(solid)),
            "invalid_support": int(np.count_nonzero(invalid)),
        },
        "extractor_failure_count": int(
            run.get("candidate_run_metrics", {}).get("failure_count", 0)
        ),
        "contours": contour_metrics,
        "physical_regions": physical_region_metrics,
        "identity_attribution": {
            "categories": list(IDENTITY_CATEGORIES),
            "accepted_raw_contour_count": len(valid_records),
            "counts": identity_counts,
            "conservation": sum(identity_counts.values()) == len(valid_records),
            "contours": identity_attributes,
        },
        "region_scale_versus_erosion": scale_attribution,
        "physical_region_consolidation": {
            "method": (
                "rasterized_complete_containment_with_center_continuity_guard"
            ),
            "raw_contour_count": accepted_count,
            "physical_region_count": len(physical_region_metrics),
            "nested_raw_contour_count": nested_raw_contour_count,
            "regions": [
                {
                    "physical_region_id": int(item["physical_region_id"]),
                    "raw_record_indices": list(item["raw_record_indices"]),
                    "outer_raw_record_index": int(item["outer_raw_record_index"]),
                    "nested_raw_contour_count": int(item["nested_raw_contour_count"]),
                    "center_xy": item.get("center_xy"),
                    "identity_attributes": item["identity_attributes"],
                    "identity_basis": item["identity_basis"],
                }
                for item in physical_region_metrics
            ],
        },
        "geometry_semantics_comparison": {
            "all_contour_band": {
                "boundary_band_cells": int(
                    np.count_nonzero(valid_fluid & all_contour_band)
                ),
                "positive_cells": int(np.count_nonzero(all_contour_positive)),
                "positive_raw_interior_ratio": (
                    float(np.count_nonzero(all_contour_positive) / raw_count)
                    if raw_count else None
                ),
                "fully_erased_geometry_fraction": (
                    float(erased_count / accepted_count) if accepted_count else None
                ),
            },
            "physical_outer_boundary_band": {
                "boundary_band_cells": int(
                    np.count_nonzero(valid_fluid & physical_outer_boundary_band)
                ),
                "positive_cells": int(np.count_nonzero(physical_positive)),
                "positive_raw_interior_ratio": (
                    float(np.count_nonzero(physical_positive) / raw_count)
                    if raw_count else None
                ),
                "fully_erased_geometry_fraction": fully_erased_physical_region_fraction,
            },
        },
    }
    return {
        "schema_version": POSTPROCESS_SCHEMA_VERSION,
        "artifact_type": "haller_v2_postprocessed_run",
        "run_id": str(run.get("run_id", "")),
        "dataset_name": str(run.get("dataset_name", "")),
        "split_name": str(run.get("split_name", "")),
        "frame_index": int(run.get("frame_index", -1)),
        "raw_contour_fingerprint": _raw_fingerprint_from_run(run),
        "source_run_artifact_fingerprint": run.get("source_run_artifact_fingerprint"),
        "configuration": config.as_dict(),
        "metrics": metrics,
        "layer_conservation": conservation,
        "raw_contours": _jsonable(records),
        "filtering_decisions": _jsonable(run.get("filtering_decisions", [])),
        "contour_layers": contour_layers,
        "contour_signatures": contour_signatures,
        "physical_region_layers": physical_region_layers,
        "physical_region_signatures": physical_region_signatures,
        "arrays": {
            "standard_ivd": ivd,
            "solid_mask": solid,
            "invalid_support": invalid,
            "valid_fluid": valid_fluid,
            "filled_interior": raw_interior,
            "unknown_band": unknown_band,
            "negative_candidates": negative,
            "evidence_state": evidence_state,
            "xdim": xdim,
            "ydim": ydim,
        },
    }


def _verify_relative_path(root: Path, relative_path: str | Path) -> Path:
    raw_path = str(relative_path)
    if not raw_path or Path(raw_path).is_absolute() or PureWindowsPath(raw_path).is_absolute():
        raise ValueError(f"artifact path must be relative: {relative_path!r}")
    normalized = raw_path.replace("\\", "/")
    parts = tuple(part for part in normalized.split("/") if part not in {"", "."})
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"artifact path escapes artifact root: {relative_path!r}")
    root = root.resolve()
    path = (root / Path(*parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"candidate path escapes artifact root: {relative_path!r}") from exc
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"artifact path traverses a symlink: {relative_path!r}")
    return path


def _verify_file_name(root: Path, filename: str) -> Path:
    raw_name = str(filename)
    if (
        not raw_name
        or "/" in raw_name
        or "\\" in raw_name
        or Path(raw_name).name != raw_name
        or PureWindowsPath(raw_name).name != raw_name
    ):
        raise ValueError(f"artifact file name must be a single safe component: {filename!r}")
    return _verify_relative_path(root, raw_name)


def _publish_directory_no_overwrite(staging_root: Path, output_root: Path) -> None:
    """Publish a staging directory with create-only destination semantics."""

    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file_ex = kernel32.MoveFileExW
        move_file_ex.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        move_file_ex.restype = ctypes.c_int
        if not move_file_ex(str(staging_root), str(output_root), 0x00000008):
            error_code = ctypes.get_last_error()
            if error_code == 183:
                raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
            raise OSError(error_code, f"create-only publish failed: {output_root}")
        return

    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(staging_root),
            -100,
            os.fsencode(output_root),
            1,
        )
        if result == 0:
            return
        error_code = ctypes.get_errno()
        if error_code == errno.EEXIST:
            raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
        raise OSError(error_code, f"create-only publish failed: {output_root}")

    raise OSError(
        "create-only directory publication is unavailable on this platform; "
        "refusing a replace-based fallback"
    )


def _verify_candidate_manifest(candidate_root: Path) -> dict[str, Any]:
    manifest_path = candidate_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "numbacs_raw_contour_candidate":
        raise ValueError("input is not a NumbaCS raw-contour candidate")
    if manifest.get("artifact_state") != "candidate" or manifest.get("trusted") is not False:
        raise PermissionError("raw-contour input must remain an untrusted candidate")
    if manifest.get("final_test_relationships") != "not_read":
        raise PermissionError("candidate does not prove final-test isolation")
    identity = dict(manifest)
    declared_integrity = identity.pop("integrity_fingerprint", None)
    if declared_integrity != _json_hash(identity):
        raise ValueError("candidate manifest integrity fingerprint mismatch")
    expected_parameter = _json_hash({
        "backend": manifest.get("backend"),
        "radius_factor": manifest.get("radius_factor"),
        "min_len_factor": manifest.get("min_len_factor"),
        "sample_manifest_hash": manifest.get("sample_manifest_hash"),
    })
    if manifest.get("parameter_fingerprint") != expected_parameter:
        raise ValueError("candidate parameter fingerprint mismatch")
    if float(manifest.get("radius_factor", -1.0)) != 1.0:
        raise ValueError("Haller-v2 input candidate must be the fixed R1 r/Delta=1 candidate")
    return manifest


def _load_candidate_sample_manifest(candidate_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    sample_path = candidate_root.parent / "sample_manifest.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"candidate sample manifest is missing: {sample_path}")
    sample_manifest = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(sample_manifest, Mapping):
        raise ValueError(f"sample manifest must be a JSON object: {sample_path}")
    sample_payload = dict(sample_manifest)
    declared_hash = sample_payload.pop("manifest_hash", None)
    if declared_hash != _json_hash(sample_payload):
        raise ValueError("sample manifest integrity fingerprint mismatch")
    if declared_hash != manifest.get("sample_manifest_hash"):
        raise ValueError("candidate/sample manifest fingerprint mismatch")
    samples = sample_manifest.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"sample manifest samples must be a non-empty list: {sample_path}")
    if sample_manifest.get("protocol_version") != experiment_protocol.PROTOCOL_VERSION:
        raise ValueError("candidate sample manifest protocol version is not the fixed R0 version")
    if sample_manifest.get("dataset_names") != list(experiment_protocol.DATASET_NAMES):
        raise ValueError("candidate sample manifest must cover the fixed six datasets")
    protocol_manifest_hash = sample_manifest.get("protocol_manifest_hash")
    if not isinstance(protocol_manifest_hash, str) or not protocol_manifest_hash:
        raise ValueError("candidate sample manifest is missing the R0 protocol manifest hash")
    if int(sample_manifest.get("sample_count", -1)) != len(samples):
        raise ValueError("candidate sample manifest sample_count is inconsistent")
    if int(manifest.get("sample_run_count", -1)) != len(samples):
        raise ValueError("candidate sample count does not match sample manifest")
    if sample_manifest.get("final_test_relationships") != "sealed; no test data read":
        raise PermissionError("sample manifest does not prove final-test isolation")
    sample_keys: set[tuple[str, str, int]] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError(f"sample manifest entry must be an object: {sample_path}")
        split_name = str(sample.get("split_name"))
        if split_name not in {"train", "calibration"}:
            raise PermissionError("Haller-v2 development audit cannot consume test samples")
        if sample.get("eligible_for_test_relationships") is not False:
            raise PermissionError("candidate sample is not sealed from final-test relationships")
        selection_basis = sample.get("selection_basis")
        if not isinstance(selection_basis, Mapping) or selection_basis.get("rule") != (
            "midpoint_of_half_open_partition_range"
        ):
            raise ValueError("candidate sample is missing the fixed R0 selection basis")
        selection_range = selection_basis.get("range")
        if (
            not isinstance(selection_range, list)
            or len(selection_range) != 2
            or not all(isinstance(value, int) for value in selection_range)
            or not selection_range[0] <= int(sample.get("frame_index", -1)) < selection_range[1]
        ):
            raise ValueError("candidate sample frame is outside its recorded R0 range")
        sample_key = (
            str(sample.get("dataset_name", "")),
            split_name,
            int(sample.get("frame_index", -1)),
        )
        if sample_key in sample_keys:
            raise ValueError(f"duplicate candidate sample identity: {sample_key!r}")
        sample_keys.add(sample_key)
        if sample_key[0].casefold() == "boussinesq" and sample_key[2] == 1300:
            raise PermissionError("Boussinesq frame 1300 is development/audit-only")
    return sample_manifest


def _load_candidate_run(
    candidate_root: Path,
    copied: Mapping[str, Any],
    raw_contour_fingerprint: str,
) -> dict[str, Any]:
    relative_dir = str(copied["relative_dir"])
    normalized_relative_dir = relative_dir.replace("\\", "/").lstrip("/")
    candidate_prefix = f"{candidate_root.name}/"
    if not normalized_relative_dir.startswith(candidate_prefix):
        raise ValueError(
            "candidate run must remain inside the candidate artifact subtree: "
            f"{relative_dir!r}"
        )
    run_dir = _verify_relative_path(candidate_root.parent, relative_dir)
    try:
        run_dir.relative_to(candidate_root.resolve())
    except ValueError as exc:
        raise ValueError(
            "candidate run must remain inside the candidate artifact subtree: "
            f"{relative_dir!r}"
        ) from exc
    expected_hashes = copied.get("file_hashes", {})
    if not isinstance(expected_hashes, Mapping):
        raise ValueError(f"candidate file hashes missing: {relative_dir}")
    if set(expected_hashes) != R1_CANDIDATE_RUN_FILES:
        raise ValueError(
            "candidate file hash set is incomplete or contains unexpected files: "
            f"{relative_dir}"
        )
    for filename, expected_hash in expected_hashes.items():
        path = _verify_file_name(run_dir, str(filename))
        if _file_sha256(path) != str(expected_hash):
            raise ValueError(f"candidate file fingerprint mismatch: {path}")
    metrics = json.loads((run_dir / "run_metrics.json").read_text(encoding="utf-8"))
    dataset_name = str(metrics.get("dataset_name", ""))
    split_name = str(metrics.get("split_name", ""))
    frame_index = int(metrics.get("frame_index", -1))
    if split_name not in {"train", "calibration"}:
        raise PermissionError("candidate run is outside the train/calibration development scope")
    if dataset_name.casefold() == "boussinesq" and frame_index == 1300:
        raise PermissionError("Boussinesq frame 1300 cannot enter a Haller-v2 artifact")
    run_id = str(copied.get("run_id", ""))
    if run_id != str(metrics.get("run_id", run_id)):
        raise ValueError(f"candidate run identity mismatch: {run_id}")
    raw_contours = json.loads((run_dir / "raw_contours.json").read_text(encoding="utf-8"))
    filtering_decisions = json.loads(
        (run_dir / "filtering_decisions.json").read_text(encoding="utf-8")
    )
    if not isinstance(raw_contours, list) or not isinstance(filtering_decisions, list):
        raise ValueError(f"candidate contour JSON is malformed: {run_dir}")
    standard_ivd = np.asarray(
        np.load(run_dir / "standard_ivd.npy", allow_pickle=False), dtype=np.float64
    )
    solid_mask = np.asarray(
        np.load(run_dir / "solid_mask.npy", allow_pickle=False), dtype=bool
    )
    xdim = np.asarray(np.load(run_dir / "xdim.npy", allow_pickle=False), dtype=np.float64)
    ydim = np.asarray(np.load(run_dir / "ydim.npy", allow_pickle=False), dtype=np.float64)
    if standard_ivd.ndim != 2 or solid_mask.shape != standard_ivd.shape:
        raise ValueError(f"candidate array shape mismatch: {run_dir}")
    if xdim.shape != (standard_ivd.shape[1],) or ydim.shape != (standard_ivd.shape[0],):
        raise ValueError(f"candidate coordinate shape mismatch: {run_dir}")
    return {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "split_name": split_name,
        "frame_index": frame_index,
        "standard_ivd": standard_ivd,
        "solid_mask": solid_mask,
        "xdim": xdim,
        "ydim": ydim,
        "raw_contours": raw_contours,
        "filtering_decisions": filtering_decisions,
        "raw_contour_fingerprint": raw_contour_fingerprint,
        "source_run_artifact_fingerprint": copied.get("source_run_artifact_fingerprint"),
        "candidate_run_relative_dir": relative_dir,
        "candidate_file_hashes": dict(expected_hashes),
        "candidate_run_metrics": metrics,
    }


def load_raw_contour_candidate(candidate_dir: str | Path) -> dict[str, Any]:
    """Load and verify the immutable R1 candidate without running extraction."""

    candidate_root = Path(candidate_dir).resolve()
    manifest = _verify_candidate_manifest(candidate_root)
    sample_manifest = _load_candidate_sample_manifest(candidate_root, manifest)
    copied_runs = manifest.get("sample_runs", [])
    if not isinstance(copied_runs, list) or not copied_runs:
        raise ValueError("candidate has no serialized sample runs")
    if not all(isinstance(item, Mapping) for item in copied_runs):
        raise ValueError("candidate sample runs must be JSON objects")
    run_ids = [str(item.get("run_id", "")) for item in copied_runs]
    if any(not run_id for run_id in run_ids):
        raise ValueError("candidate sample runs must have non-empty run_id values")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("candidate sample runs contain duplicate run_id values")
    raw_identity = {
        "candidate_manifest_fingerprint": manifest["integrity_fingerprint"],
        "sample_runs": [
            {
                "run_id": item.get("run_id"),
                "source_run_artifact_fingerprint": item.get("source_run_artifact_fingerprint"),
                "raw_contours_sha256": item.get("file_hashes", {}).get("raw_contours.json"),
                "standard_ivd_sha256": item.get("file_hashes", {}).get("standard_ivd.npy"),
                "solid_mask_sha256": item.get("file_hashes", {}).get("solid_mask.npy"),
                "xdim_sha256": item.get("file_hashes", {}).get("xdim.npy"),
                "ydim_sha256": item.get("file_hashes", {}).get("ydim.npy"),
            }
            for item in copied_runs
        ],
    }
    raw_contour_fingerprint = _json_hash(raw_identity)
    runs = [
        _load_candidate_run(candidate_root, copied, raw_contour_fingerprint)
        for copied in copied_runs
    ]
    if len(runs) != len(sample_manifest.get("samples", [])):
        raise ValueError("candidate run count does not match sample manifest")
    expected_keys = {
        (str(sample["dataset_name"]), str(sample["split_name"]), int(sample["frame_index"]))
        for sample in sample_manifest["samples"]
    }
    actual_keys = {
        (run["dataset_name"], run["split_name"], int(run["frame_index"]))
        for run in runs
    }
    if expected_keys != actual_keys:
        raise ValueError("candidate run identities do not match its sample manifest")
    return {
        "candidate_root": str(candidate_root),
        "candidate_manifest": manifest,
        "sample_manifest": sample_manifest,
        "raw_contour_fingerprint": raw_contour_fingerprint,
        "runs": runs,
    }


def _configuration_key(config: PostprocessingConfig) -> str:
    return json.dumps(config.as_dict(), sort_keys=True, separators=(",", ":"))


def _aggregate_run_metrics(outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    integer_names = (
        "valid_fluid_cells",
        "solid_cells",
        "invalid_support_cells",
        "raw_interior_cells",
        "positive_cells",
        "negative_cells",
        "unknown_fluid_cells",
        "extractor_failure_count",
        "raw_contour_count",
        "boundary_band_cells",
        "geometry_unit_count",
        "fully_erased_geometry_count",
        "physical_region_count",
        "nested_raw_contour_count",
        "fully_erased_physical_region_count",
    )
    area_names = (
        "raw_interior_area_physical",
        "boundary_band_area_physical",
        "positive_area_physical",
    )
    totals: dict[str, int | float] = {
        **{name: 0 for name in integer_names},
        **{name: 0.0 for name in area_names},
    }
    identity_counts = {category: 0 for category in IDENTITY_CATEGORIES}
    unknown_attribution = {
        "boundary_band": 0,
        "outside_high_ivd": 0,
        "total_unknown_fluid": 0,
        "solid": 0,
        "invalid_support": 0,
    }
    accepted_count = 0
    erased_count = 0
    erosion_units: set[str] = set()
    geometry_policies: set[str] = set()
    scale_summaries: list[dict[str, Any]] = []
    for output in outputs:
        metrics = output["metrics"]
        for name in integer_names:
            totals[name] += int(metrics.get(name, 0) or 0)
        for name in area_names:
            totals[name] += float(metrics.get(name, 0.0) or 0.0)
        accepted_count += int(metrics["accepted_raw_contour_count"])
        erased_count += int(metrics["fully_erased_contour_count"])
        erosion_units.add(str(metrics.get("geometry_erosion_unit", "raw_contour")))
        geometry_policies.add(str(metrics.get("geometry_policy", "all_contour_band")))
        for category in IDENTITY_CATEGORIES:
            identity_counts[category] += int(
                metrics.get("identity_attribution", {})
                .get("counts", {})
                .get(category, 0)
            )
        scale_summary = metrics.get("region_scale_versus_erosion", {}).get(
            "aggregate"
        )
        if isinstance(scale_summary, Mapping):
            scale_summaries.append({
                "run_id": str(output.get("run_id", "")),
                **dict(scale_summary),
                "erosion_width_physical": metrics.get(
                    "region_scale_versus_erosion", {}
                ).get("erosion_width_physical"),
            })
        for name in unknown_attribution:
            unknown_attribution[name] += int(metrics["unknown_attribution"][name])
    valid_count = totals["valid_fluid_cells"]
    raw_count = totals["raw_interior_cells"]
    known_count = totals["positive_cells"] + totals["negative_cells"]
    return {
        **totals,
        "known_cells": known_count,
        "unknown_cells": totals["unknown_fluid_cells"],
        "known_coverage": float(known_count / valid_count) if valid_count else None,
        "unknown_coverage": (
            float(totals["unknown_fluid_cells"] / valid_count) if valid_count else None
        ),
        "positive_raw_interior_ratio": (
            float(totals["positive_cells"] / raw_count) if raw_count else None
        ),
        "accepted_raw_contour_count": accepted_count,
        "raw_contour_count": totals["raw_contour_count"],
        "boundary_band_cells": totals["boundary_band_cells"],
        "raw_interior_area_physical": totals["raw_interior_area_physical"],
        "boundary_band_area_physical": totals["boundary_band_area_physical"],
        "positive_area_physical": totals["positive_area_physical"],
        "fully_erased_contour_count": erased_count,
        "fully_erased_contour_fraction": (
            float(erased_count / accepted_count) if accepted_count else None
        ),
        "geometry_erosion_unit": (
            next(iter(erosion_units)) if len(erosion_units) == 1 else "mixed"
        ),
        "geometry_policy": (
            next(iter(geometry_policies)) if len(geometry_policies) == 1 else "mixed"
        ),
        "geometry_unit_count": totals["geometry_unit_count"],
        "fully_erased_geometry_count": totals["fully_erased_geometry_count"],
        "fully_erased_geometry_fraction": (
            float(totals["fully_erased_geometry_count"] / totals["geometry_unit_count"])
            if totals["geometry_unit_count"] else None
        ),
        "physical_region_count": totals["physical_region_count"],
        "nested_raw_contour_count": totals["nested_raw_contour_count"],
        "identity_attribution": {
            "categories": list(IDENTITY_CATEGORIES),
            "counts": identity_counts,
            "accepted_raw_contour_count": sum(identity_counts.values()),
            "conservation": sum(identity_counts.values()) == accepted_count,
        },
        "fully_erased_physical_region_count": totals["fully_erased_physical_region_count"],
        "fully_erased_physical_region_fraction": (
            float(
                totals["fully_erased_physical_region_count"]
                / totals["physical_region_count"]
            )
            if totals["physical_region_count"] else None
        ),
        "unknown_attribution": unknown_attribution,
        "region_scale_versus_erosion": {
            "unit_type": (
                next(iter(erosion_units)) if len(erosion_units) == 1 else "mixed"
            ),
            "per_run": scale_summaries,
        },
        "run_count": len(outputs),
        "extractor_hard_failure_rate": (
            float(totals["extractor_failure_count"] / len(outputs)) if outputs else None
        ),
        "status": "pass" if outputs else "no-data",
    }


def _dataset_metrics(
    outputs_by_run: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for output in outputs_by_run.values():
        grouped.setdefault(str(output["dataset_name"]), []).append(output)
    return {
        dataset_name: _aggregate_run_metrics(outputs)
        for dataset_name, outputs in sorted(grouped.items())
    }


def _configuration_eligible(dataset_metrics: Mapping[str, Mapping[str, Any]]) -> tuple[bool, dict[str, Any]]:
    by_dataset: dict[str, Any] = {}
    overall = True
    for dataset_name, metrics in dataset_metrics.items():
        raw_ratio = metrics.get("positive_raw_interior_ratio")
        geometry_erosion_unit = str(metrics.get("geometry_erosion_unit", "raw_contour"))
        geometry_unit_count = int(
            metrics.get(
                "geometry_unit_count",
                metrics.get("accepted_raw_contour_count", 0),
            )
        )
        raw_erased_fraction = metrics.get("fully_erased_contour_fraction")
        geometry_erased_fraction = metrics.get("fully_erased_geometry_fraction")
        if geometry_erased_fraction is None and geometry_erosion_unit == "raw_contour":
            geometry_erased_fraction = raw_erased_fraction
        no_contours = int(metrics.get("accepted_raw_contour_count", 0)) == 0
        no_geometry_units = geometry_unit_count == 0
        ratio_pass = no_contours or (
            raw_ratio is not None and raw_ratio >= POSITIVE_RAW_INTERIOR_MINIMUM
        )
        raw_erased_pass = no_contours or (
            raw_erased_fraction is not None
            and raw_erased_fraction <= FULLY_ERASED_MAXIMUM
        )
        geometry_erased_pass = no_geometry_units or (
            geometry_erased_fraction is not None
            and geometry_erased_fraction <= FULLY_ERASED_MAXIMUM
        )
        eligible = bool(ratio_pass and raw_erased_pass)
        overall &= eligible
        by_dataset[dataset_name] = {
            "positive_raw_interior_ratio": raw_ratio,
            "positive_raw_interior_ratio_threshold": POSITIVE_RAW_INTERIOR_MINIMUM,
            "positive_raw_interior_ratio_pass": bool(ratio_pass),
            "fully_erased_contour_fraction": raw_erased_fraction,
            "fully_erased_contour_fraction_threshold": FULLY_ERASED_MAXIMUM,
            "fully_erased_contour_fraction_pass": bool(raw_erased_pass),
            "geometry_erosion_unit": geometry_erosion_unit,
            "geometry_unit_count": geometry_unit_count,
            "fully_erased_geometry_fraction": geometry_erased_fraction,
            "fully_erased_geometry_fraction_threshold": FULLY_ERASED_MAXIMUM,
            "fully_erased_geometry_fraction_pass": bool(geometry_erased_pass),
            "raw_contour_erosion_gate": {
                "fraction": raw_erased_fraction,
                "threshold": FULLY_ERASED_MAXIMUM,
                "pass": bool(raw_erased_pass),
            },
            "geometry_erosion_diagnostic_gate": {
                "fraction": geometry_erased_fraction,
                "threshold": FULLY_ERASED_MAXIMUM,
                "pass": bool(geometry_erased_pass),
            },
            "no_accepted_raw_contours": no_contours,
            "no_geometry_units": no_geometry_units,
            "eligible": eligible,
        }
    return overall, by_dataset


def _development_health(
    dataset_metrics: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Apply the spec's six-real-dataset non-degeneracy bounds when present."""

    real_names = set(experiment_protocol.DATASET_NAMES)
    present_real = real_names & set(dataset_metrics)
    if not present_real:
        health = {
            dataset_name: {
                "status": "fail",
                "reason": "missing_six_dataset_development_sample",
            }
            for dataset_name in sorted(real_names)
        }
        health.update({
            dataset_name: {
                "status": "not-applicable",
                "reason": "non-real synthetic fixture is outside the six-real-dataset health panel",
            }
            for dataset_name in sorted(set(dataset_metrics) - real_names)
        })
        return False, health
    health: dict[str, Any] = {}
    overall = True
    missing = sorted(real_names - set(dataset_metrics))
    if missing:
        overall = False
    for dataset_name in sorted(real_names):
        metrics = dataset_metrics.get(dataset_name)
        if metrics is None:
            health[dataset_name] = {
                "status": "fail",
                "reason": "missing_six_dataset_development_sample",
            }
            continue
        valid_cells = int(metrics.get("valid_fluid_cells", 0))
        positive_coverage = (
            float(metrics["positive_cells"] / valid_cells) if valid_cells else None
        )
        known_coverage = metrics.get("known_coverage")
        failure_rate = metrics.get("extractor_hard_failure_rate")
        known_pass = known_coverage is not None and (
            DEVELOPMENT_KNOWN_COVERAGE_MINIMUM
            <= float(known_coverage)
            <= DEVELOPMENT_KNOWN_COVERAGE_MAXIMUM
        )
        positive_pass = positive_coverage is not None and (
            DEVELOPMENT_POSITIVE_COVERAGE_MINIMUM
            <= positive_coverage
            <= DEVELOPMENT_POSITIVE_COVERAGE_MAXIMUM
        )
        failure_pass = failure_rate is not None and (
            float(failure_rate) <= DEVELOPMENT_HARD_FAILURE_RATE_MAXIMUM
        )
        passed = bool(known_pass and positive_pass and failure_pass)
        overall &= passed
        health[dataset_name] = {
            "status": "pass" if passed else "fail",
            "known_coverage": known_coverage,
            "known_coverage_interval": [
                DEVELOPMENT_KNOWN_COVERAGE_MINIMUM,
                DEVELOPMENT_KNOWN_COVERAGE_MAXIMUM,
            ],
            "known_coverage_pass": bool(known_pass),
            "positive_core_coverage": positive_coverage,
            "positive_core_coverage_interval": [
                DEVELOPMENT_POSITIVE_COVERAGE_MINIMUM,
                DEVELOPMENT_POSITIVE_COVERAGE_MAXIMUM,
            ],
            "positive_core_coverage_pass": bool(positive_pass),
            "extractor_hard_failure_rate": failure_rate,
            "extractor_hard_failure_rate_threshold": DEVELOPMENT_HARD_FAILURE_RATE_MAXIMUM,
            "extractor_hard_failure_rate_pass": bool(failure_pass),
        }
    return overall, health


def _neighbor_configurations(config: PostprocessingConfig) -> tuple[PostprocessingConfig, ...]:
    neighbors: list[PostprocessingConfig] = []
    for band in BAND_FACTORS:
        if abs(float(band) - config.band_factor) == 0.5:
            neighbors.append(
                PostprocessingConfig(
                    band_factor=band,
                    rasterization_policy=config.rasterization_policy,
                    negative_policy=config.negative_policy,
                    geometry_policy=config.geometry_policy,
                )
            )
    other_rasterization = next(
        policy for policy in RASTERIZATION_POLICIES
        if policy != config.rasterization_policy
    )
    neighbors.append(
        PostprocessingConfig(
            band_factor=config.band_factor,
            rasterization_policy=other_rasterization,
            negative_policy=config.negative_policy,
            geometry_policy=config.geometry_policy,
        )
    )
    other_negative_policy = next(
        policy for policy in NEGATIVE_POLICIES
        if policy != config.negative_policy
    )
    neighbors.append(
        PostprocessingConfig(
            band_factor=config.band_factor,
            rasterization_policy=config.rasterization_policy,
            negative_policy=other_negative_policy,
            geometry_policy=config.geometry_policy,
        )
    )
    return tuple(neighbors)


def _matching_layers(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Materialize only the masks needed for one stability comparison."""

    configuration = output.get("configuration", {})
    geometry_policy = str(configuration.get("geometry_policy", "all_contour_band"))
    physical = geometry_policy == "physical_outer_boundary_band"
    layer_key = "physical_region_layers" if physical else "contour_layers"
    signature_key = "physical_region_signatures" if physical else "contour_signatures"
    existing = output.get(layer_key, [])
    if existing:
        return list(existing)
    shape_value = np.asarray(output["arrays"]["standard_ivd"]).shape
    shape = (int(shape_value[0]), int(shape_value[1]))
    layers: list[dict[str, Any]] = []
    for signature in output.get(signature_key, []):
        packed = np.asarray(signature["core_packed"], dtype=np.uint8)
        core = np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)
        if physical:
            raw_indices = [int(item) for item in signature["raw_record_indices"]]
            layers.append({
                "physical_region_id": int(signature["physical_region_id"]),
                "raw_record_indices": raw_indices,
                "raw_record_index": int(signature["outer_raw_record_index"]),
                "outer_raw_record_index": int(signature["outer_raw_record_index"]),
                "center_xy": signature.get("center_xy"),
                "core": core,
            })
        else:
            layers.append({
                "raw_record_index": int(signature["raw_record_index"]),
                "center_xy": signature.get("center_xy"),
                "core": core,
            })
    return layers


def _center_matches(
    base: Mapping[str, Any],
    neighbor: Mapping[str, Any],
    *,
    tolerance: float,
) -> tuple[int, int, list[float]]:
    base_layers = [
        layer for layer in _matching_layers(base)
        if np.any(layer["core"]) and layer.get("center_xy") is not None
    ]
    neighbor_layers = [
        layer for layer in _matching_layers(neighbor)
        if layer.get("center_xy") is not None
    ]
    if not base_layers:
        return 0, 0, []
    used: set[int] = set()
    retained = 0
    ious: list[float] = []
    for base_layer in base_layers:
        base_center = np.asarray(base_layer["center_xy"], dtype=np.float64)
        choices: list[tuple[float, int]] = []
        for index, target_layer in enumerate(neighbor_layers):
            if index in used:
                continue
            target_center = np.asarray(target_layer["center_xy"], dtype=np.float64)
            distance = float(np.linalg.norm(base_center - target_center))
            if distance <= tolerance:
                choices.append((distance, index))
        if not choices:
            continue
        _, target_index = min(choices)
        used.add(target_index)
        target = neighbor_layers[target_index]
        if not np.any(target["core"]):
            continue
        retained += 1
        ious.append(_iou(base_layer["core"], target["core"]))
    return len(base_layers), retained, ious


def _stability_for_configuration(
    config: PostprocessingConfig,
    outputs_by_config: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    base_key = _configuration_key(config)
    base_outputs = outputs_by_config[base_key]
    base_aggregate = _aggregate_run_metrics(list(base_outputs.values()))
    base_coverage = base_aggregate.get("known_coverage")
    neighbor_records: list[dict[str, Any]] = []
    retention_values: list[float] = []
    ious: list[float] = []
    relative_changes: list[float] = []
    for neighbor in _neighbor_configurations(config):
        neighbor_key = _configuration_key(neighbor)
        neighbor_outputs = outputs_by_config[neighbor_key]
        selected_total = 0
        retained_total = 0
        neighbor_ious: list[float] = []
        max_spacing = 0.0
        for run_id, base_output in base_outputs.items():
            neighbor_output = neighbor_outputs[run_id]
            arrays = base_output["arrays"]
            spacing = max(
                float(arrays["xdim"][1] - arrays["xdim"][0]),
                float(arrays["ydim"][1] - arrays["ydim"][0]),
            )
            selected, retained, run_ious = _center_matches(
                base_output,
                neighbor_output,
                tolerance=2.0 * spacing,
            )
            selected_total += selected
            retained_total += retained
            neighbor_ious.extend(run_ious)
        retention = (
            float(retained_total / selected_total) if selected_total else None
        )
        neighbor_aggregate = _aggregate_run_metrics(list(neighbor_outputs.values()))
        neighbor_coverage = neighbor_aggregate.get("known_coverage")
        if base_coverage is None or neighbor_coverage is None:
            relative_change = None
        elif base_coverage == 0.0:
            relative_change = 0.0 if neighbor_coverage == 0.0 else float("inf")
        else:
            relative_change = float(abs(neighbor_coverage - base_coverage) / base_coverage)
        if retention is not None:
            retention_values.append(retention)
        ious.extend(neighbor_ious)
        if relative_change is not None:
            relative_changes.append(relative_change)
        neighbor_records.append({
            "configuration": neighbor.as_dict(),
            "selected_center_count": selected_total,
            "retained_center_count": retained_total,
            "selected_center_retention": retention,
            "matched_core_iou_count": len(neighbor_ious),
            "median_matched_core_iou": float(np.median(neighbor_ious)) if neighbor_ious else 0.0,
            "relative_known_coverage_change": relative_change,
            "center_tolerance_rule": "2 * max(dx, dy) per fixed raw run",
        })
    minimum_retention = min(retention_values) if retention_values else 0.0
    median_iou = float(np.median(ious)) if ious else 0.0
    maximum_relative_change = max(relative_changes) if relative_changes else float("inf")
    pass_status = bool(
        minimum_retention >= SELECTED_CENTER_RETENTION_MINIMUM
        and median_iou >= MATCHED_CORE_IOU_MINIMUM
        and maximum_relative_change <= RELATIVE_KNOWN_COVERAGE_CHANGE_MAXIMUM
    )
    return {
        "status": "pass" if pass_status else "fail",
        "selected_center_retention": minimum_retention,
        "selected_center_retention_threshold": SELECTED_CENTER_RETENTION_MINIMUM,
        "median_matched_core_iou": median_iou,
        "median_matched_core_iou_threshold": MATCHED_CORE_IOU_MINIMUM,
        "relative_known_coverage_change": maximum_relative_change,
        "relative_known_coverage_change_threshold": RELATIVE_KNOWN_COVERAGE_CHANGE_MAXIMUM,
        "neighbor_rule": (
            "adjacent band values, alternate fixed rasterization policy, and "
            "alternate negative policy; no re-extraction"
        ),
        "neighbors": neighbor_records,
    }


def _select_configuration(
    configuration_audits: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    for key, audit in configuration_audits.items():
        if audit["eligible"] and audit["stability"]["status"] == "pass":
            eligible.append({
                "configuration_key": key,
                "configuration": audit["configuration"],
                "stability_score": {
                    "relative_known_coverage_change": audit["stability"]["relative_known_coverage_change"],
                    "one_minus_center_retention": 1.0 - audit["stability"]["selected_center_retention"],
                    "negative_median_core_iou": -audit["stability"]["median_matched_core_iou"],
                },
            })
    eligible.sort(
        key=lambda item: (
            float(item["stability_score"]["relative_known_coverage_change"]),
            float(item["stability_score"]["one_minus_center_retention"]),
            float(item["stability_score"]["negative_median_core_iou"]),
            float(item["configuration"]["band_factor"]),
            str(item["configuration"]["geometry_policy"]),
            str(item["configuration"]["rasterization_policy"]),
            str(item["configuration"]["negative_policy"]),
        )
    )
    return (eligible[0] if eligible else None), eligible


def _official_raw_gate_summary(
    configuration_audits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the unchanged raw-contour gate independently of geometry policy."""

    by_policy: dict[str, dict[str, Any]] = {}
    all_audits = list(configuration_audits.values())
    for policy in GEOMETRY_POLICIES:
        policy_audits = [
            audit
            for audit in all_audits
            if audit.get("configuration", {}).get("geometry_policy") == policy
        ]
        passed = sum(int(audit.get("raw_eligibility_pass", False)) for audit in policy_audits)
        by_policy[policy] = {
            "configuration_count": len(policy_audits),
            "raw_gate_pass_count": passed,
            "raw_gate_fail_count": len(policy_audits) - passed,
            "status": (
                "pass"
                if policy_audits and passed == len(policy_audits)
                else "fail"
                if passed == 0
                else "partial"
            ),
        }
    pass_count = sum(
        int(audit.get("raw_eligibility_pass", False)) for audit in all_audits
    )
    total = len(all_audits)
    return {
        "status": (
            "pass"
            if total and pass_count == total
            else "fail"
            if pass_count == 0
            else "partial"
        ),
        "configuration_count": total,
        "raw_gate_pass_count": pass_count,
        "raw_gate_fail_count": total - pass_count,
        "thresholds": {
            "accepted_raw_fully_erased_fraction_maximum": FULLY_ERASED_MAXIMUM,
            "positive_raw_interior_ratio_minimum": POSITIVE_RAW_INTERIOR_MINIMUM,
        },
        "by_geometry_policy": by_policy,
        "physical_policy_cannot_bypass": True,
        "gate_unit": "accepted_raw_contour",
    }


def _freeze_eligibility_summary(
    *,
    selected: Mapping[str, Any] | None,
    configuration_audits: Mapping[str, Mapping[str, Any]],
    controlled: Mapping[str, Any],
    source_audit_pass: bool,
) -> dict[str, Any]:
    selected_audit: Mapping[str, Any] | None = None
    if selected is not None:
        selected_audit = configuration_audits.get(
            str(selected.get("configuration_key"))
        )
    raw_pass = bool(selected_audit and selected_audit.get("raw_eligibility_pass"))
    development_pass = bool(
        selected_audit and selected_audit.get("development_health_pass")
    )
    stability_pass = bool(
        selected_audit
        and selected_audit.get("stability", {}).get("status") == "pass"
    )
    controlled_pass = bool(
        controlled.get("mode") == "formal"
        and controlled.get("status") == "pass"
    )
    resolution_pass = bool(
        controlled.get("mode") == "formal"
        and controlled.get("resolution", {}).get("status") == "pass"
    )
    gates = {
        "source_r1_audit": bool(source_audit_pass),
        "official_raw_gate": raw_pass,
        "development_health": development_pass,
        "stability": stability_pass,
        "controlled_validation": controlled_pass,
        "resolution": resolution_pass,
    }
    eligible = bool(all(gates.values()))
    return {
        "status": "eligible" if eligible else "ineligible",
        "eligible": eligible,
        "gates": gates,
        "selected_configuration_key": (
            None if selected is None else selected.get("configuration_key")
        ),
        "trusted_anchor_allowed": eligible,
        "reason": None if eligible else "one_or_more_r2_gates_failed_or_not_run",
    }


def _load_serialized_audit_run(
    audit_root: Path,
    run_dir: Path,
    *,
    scope: str,
) -> dict[str, Any]:
    """Load one already-written R1 run, verifying its immutable sidecars."""

    audit_root = audit_root.resolve()
    try:
        relative_dir = run_dir.relative_to(audit_root)
    except ValueError as exc:
        raise ValueError(f"serialized audit run escapes its root: {run_dir}") from exc
    run_dir = _verify_relative_path(audit_root, relative_dir)
    meta_path = _verify_file_name(run_dir, "run_meta.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    file_hashes = metadata.get("file_hashes", {})
    if not isinstance(file_hashes, Mapping):
        raise ValueError(f"run file hashes are missing: {run_dir}")
    if not R1_CANDIDATE_RUN_FILES.issubset(set(file_hashes)):
        raise ValueError(f"serialized audit run file hash set is incomplete: {run_dir}")
    for filename, expected_hash in file_hashes.items():
        path = _verify_file_name(run_dir, str(filename))
        if _file_sha256(path) != str(expected_hash):
            raise ValueError(f"source run file fingerprint mismatch: {path}")
    summary = metadata.get("summary", {})
    summary_identity = {
        "dataset_name": summary.get("dataset_name"),
        "split_name": summary.get("split_name"),
        "frame_index": summary.get("frame_index"),
        "radius_factor": summary.get("radius_factor"),
        "input_hash": summary.get("input_hash"),
        "parameter_hash": summary.get("parameter_hash"),
    }
    expected_integrity = _json_hash({
        "schema_version": metadata.get("schema_version"),
        "run_id": metadata.get("run_id"),
        "file_hashes": file_hashes,
        "summary_identity": summary_identity,
    })
    if metadata.get("integrity_fingerprint") != expected_integrity:
        raise ValueError(f"source run integrity fingerprint mismatch: {run_dir}")
    if str(metadata.get("artifact_state")) != "candidate" or metadata.get("trusted") is not False:
        raise PermissionError("source controlled run must remain an untrusted candidate")
    run_id = str(metadata.get("run_id", run_dir.name))
    if scope == "controlled" and not run_id.startswith("controlled__"):
        raise ValueError(f"non-controlled run selected for controlled validation: {run_id}")
    metrics = json.loads((run_dir / "run_metrics.json").read_text(encoding="utf-8"))
    split_name = str(metrics.get("split_name", summary.get("split_name", "")))
    frame_index = int(metrics.get("frame_index", summary.get("frame_index", -1)))
    if split_name not in {"train", "calibration", "development_audit"}:
        raise PermissionError("controlled run is outside the development scope")
    if str(metrics.get("dataset_name", summary.get("dataset_name", ""))).casefold() == "boussinesq" and frame_index == 1300:
        raise PermissionError("Boussinesq frame 1300 cannot enter Haller-v2 validation")
    raw_contours = json.loads((run_dir / "raw_contours.json").read_text(encoding="utf-8"))
    filtering_decisions = json.loads(
        (run_dir / "filtering_decisions.json").read_text(encoding="utf-8")
    )
    standard_ivd = np.asarray(np.load(run_dir / "standard_ivd.npy", allow_pickle=False), dtype=np.float64)
    solid_mask = np.asarray(np.load(run_dir / "solid_mask.npy", allow_pickle=False), dtype=bool)
    xdim = np.asarray(np.load(run_dir / "xdim.npy", allow_pickle=False), dtype=np.float64)
    ydim = np.asarray(np.load(run_dir / "ydim.npy", allow_pickle=False), dtype=np.float64)
    if standard_ivd.ndim != 2 or solid_mask.shape != standard_ivd.shape:
        raise ValueError(f"source run array shape mismatch: {run_dir}")
    return {
        "run_id": run_id,
        "dataset_name": str(metrics.get("dataset_name", summary.get("dataset_name", ""))),
        "split_name": str(metrics.get("split_name", summary.get("split_name", "development_audit"))),
        "frame_index": int(metrics.get("frame_index", summary.get("frame_index", -1))),
        "standard_ivd": standard_ivd,
        "solid_mask": solid_mask,
        "xdim": xdim,
        "ydim": ydim,
        "raw_contours": raw_contours,
        "filtering_decisions": filtering_decisions,
        "raw_contour_fingerprint": _json_hash({
            "source_run_integrity_fingerprint": metadata["integrity_fingerprint"],
            "raw_contours_sha256": file_hashes.get("raw_contours.json"),
            "standard_ivd_sha256": file_hashes.get("standard_ivd.npy"),
        }),
        "source_run_artifact_fingerprint": metadata["integrity_fingerprint"],
        "candidate_run_relative_dir": run_dir.resolve().relative_to(audit_root.resolve()).as_posix(),
        "candidate_file_hashes": dict(file_hashes),
        "candidate_run_metrics": metrics,
    }


def _controlled_source_runs(audit_root: Path) -> dict[str, dict[str, Any]]:
    audit_root = audit_root.resolve()
    runs_root = audit_root / "runs"
    if not runs_root.exists():
        return {}
    selected: dict[str, dict[str, Any]] = {}
    for path in sorted(runs_root.iterdir()):
        if not path.is_dir():
            continue
        run_dir = _verify_relative_path(
            audit_root,
            path.relative_to(audit_root),
        )
        name = run_dir.name
        if not name.startswith("controlled__") or not name.endswith("__r1p0"):
            continue
        if name == "controlled__no_vortex__r1p0":
            selected["no_vortex"] = _load_serialized_audit_run(
                audit_root, run_dir, scope="controlled"
            )
        elif name.startswith("controlled__single_vortex__"):
            selected[f"single_vortex_{name.split('__')[2]}"] = _load_serialized_audit_run(
                audit_root, run_dir, scope="controlled"
            )
        elif name == "controlled__multi_vortex__n129__r1p0":
            selected["multi_vortex"] = _load_serialized_audit_run(
                audit_root, run_dir, scope="controlled"
            )
    return selected


def _controlled_diagnostic_run(
    run_id: str,
    raw_contours: Sequence[Mapping[str, Any]],
    *,
    standard_ivd: np.ndarray | None = None,
    solid_mask: np.ndarray | None = None,
    status: str = "valid",
    failure_count: int = 0,
    failure_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    size = 65
    xdim = np.linspace(-4.0, 4.0, size, dtype=np.float64)
    ydim = np.linspace(-4.0, 4.0, size, dtype=np.float64)
    shape = (size, size)
    return {
        "run_id": run_id,
        "dataset_name": f"controlled_{run_id}",
        "split_name": "development_audit",
        "frame_index": 0,
        "standard_ivd": (
            np.ones(shape, dtype=np.float64)
            if standard_ivd is None
            else np.asarray(standard_ivd, dtype=np.float64)
        ),
        "solid_mask": (
            np.zeros(shape, dtype=bool)
            if solid_mask is None
            else np.asarray(solid_mask, dtype=bool)
        ),
        "xdim": xdim,
        "ydim": ydim,
        "raw_contours": [dict(item) for item in raw_contours],
        "filtering_decisions": [],
        "candidate_run_metrics": {
            "status": status,
            "failure_count": int(failure_count),
            "failure_reasons": list(failure_reasons),
        },
        "raw_contour_fingerprint": _json_hash({
            "run_id": run_id,
            "raw_contours": raw_contours,
            "standard_ivd": np.asarray(
                np.ones(shape, dtype=np.float64)
                if standard_ivd is None else standard_ivd
            ).tolist(),
            "solid_mask": np.asarray(
                np.zeros(shape, dtype=bool)
                if solid_mask is None else solid_mask
            ).tolist(),
        }),
    }


def _diagnostic_square(
    low_row: int,
    low_column: int,
    high_row: int,
    high_column: int,
    *,
    raw_record_index: int,
    center_xy: tuple[float, float],
    peak: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "raw_record_index": int(raw_record_index),
        "status": "valid",
        "points_grid": [
            [low_row, low_column],
            [low_row, high_column],
            [high_row, high_column],
            [high_row, low_column],
            [low_row, low_column],
        ],
        "center_xy": [float(center_xy[0]), float(center_xy[1])],
        "area": float((high_row - low_row) * (high_column - low_column)),
    }
    if peak is not None:
        record["peak"] = dict(peak)
    return record


def _controlled_diagnostic_runs() -> dict[str, dict[str, Any]]:
    """Build deterministic post-processing-only fixtures, including failure states."""

    center_peak = {"row": 32, "col": 32, "value": 1.0}
    nested = [
        _diagnostic_square(8, 8, 56, 56, raw_record_index=0, center_xy=(0.0, 0.0), peak=center_peak),
        _diagnostic_square(16, 16, 48, 48, raw_record_index=1, center_xy=(0.0, 0.0), peak=center_peak),
        _diagnostic_square(26, 26, 38, 38, raw_record_index=2, center_xy=(0.0, 0.0), peak=center_peak),
    ]
    independent = [
        _diagnostic_square(
            12, 8, 52, 30, raw_record_index=0, center_xy=(-1.7, 0.0),
            peak={"row": 32, "col": 15, "value": 1.0},
        ),
        _diagnostic_square(
            12, 34, 52, 56, raw_record_index=1, center_xy=(1.7, 0.0),
            peak={"row": 32, "col": 49, "value": 1.0},
        ),
    ]
    solid = np.zeros((65, 65), dtype=bool)
    solid[28:37, 28:37] = True
    return {
        "single_vortex": _controlled_diagnostic_run(
            "diagnostic_single_vortex",
            [_diagnostic_square(8, 8, 56, 56, raw_record_index=0, center_xy=(0.0, 0.0), peak=center_peak)],
        ),
        "nested_same_peak": _controlled_diagnostic_run(
            "diagnostic_nested_same_peak", nested
        ),
        "nearby_independent_peaks": _controlled_diagnostic_run(
            "diagnostic_nearby_independent_peaks", independent
        ),
        "no_vortex": _controlled_diagnostic_run(
            "diagnostic_no_vortex", [], standard_ivd=np.zeros((65, 65), dtype=np.float64)
        ),
        "malformed": _controlled_diagnostic_run(
            "diagnostic_malformed",
            [{
                "raw_record_index": 0,
                "status": "valid",
                "points_grid": [[1, 1], [2, 2]],
            }],
        ),
        "solid_intersection": _controlled_diagnostic_run(
            "diagnostic_solid_intersection",
            [_diagnostic_square(8, 8, 56, 56, raw_record_index=0, center_xy=(0.0, 0.0), peak=center_peak)],
            solid_mask=solid,
        ),
        "forced_extractor_failure": _controlled_diagnostic_run(
            "diagnostic_forced_extractor_failure",
            [],
            status="extractor_failure",
            failure_count=1,
            failure_reasons=("forced_extractor_failure",),
        ),
    }


def _controlled_fixture_diagnostics(
    config: PostprocessingConfig,
) -> dict[str, Any]:
    fixtures = _controlled_diagnostic_runs()
    results: dict[str, dict[str, Any]] = {}
    for name, run in fixtures.items():
        failure_count = int(
            run.get("candidate_run_metrics", {}).get("failure_count", 0) or 0
        )
        expected_reason: str | None = None
        try:
            output = postprocess_raw_run(run, config)
            metrics = output["metrics"]
            observed_status = str(
                run.get("candidate_run_metrics", {}).get("status", "valid")
            )
            if name == "single_vortex":
                passed = (
                    metrics["accepted_raw_contour_count"] == 1
                    and output["layer_conservation"]["status"] == "pass"
                )
            elif name == "nested_same_peak":
                counts = metrics["identity_attribution"]["counts"]
                passed = (
                    metrics["physical_region_count"] == 1
                    and counts["root"] == 1
                    and counts["nested"] == 2
                )
            elif name == "nearby_independent_peaks":
                counts = metrics["identity_attribution"]["counts"]
                passed = (
                    metrics["physical_region_count"] == 2
                    and counts["independent_proven"] == 2
                    and metrics["nested_raw_contour_count"] == 0
                )
            elif name == "no_vortex":
                passed = (
                    metrics["accepted_raw_contour_count"] == 0
                    and metrics["positive_cells"] == 0
                    and metrics["extractor_failure_count"] == 0
                )
            elif name == "malformed":
                expected_reason = "malformed_raw_contour_rejected"
                passed = (
                    metrics["accepted_raw_contour_count"] == 0
                    and output["layer_conservation"]["status"] == "pass"
                )
            elif name == "solid_intersection":
                solid_mask = np.asarray(run["solid_mask"], dtype=bool)
                evidence = np.asarray(output["arrays"]["evidence_state"])
                expected_reason = "solid_intersection_masked"
                passed = (
                    output["layer_conservation"]["status"] == "pass"
                    and not np.any(evidence[solid_mask] >= 0)
                )
            elif name == "forced_extractor_failure":
                expected_reason = "forced_extractor_failure"
                passed = (
                    observed_status == "extractor_failure"
                    and failure_count == 1
                    and metrics["extractor_failure_count"] == 1
                )
            else:  # pragma: no cover - fixture names are fixed above
                passed = False
            if not passed and expected_reason is None:
                expected_reason = "controlled_diagnostic_expectation_failed"
            results[name] = {
                "status": "pass" if passed else "fail",
                "failure_reason": expected_reason,
                "failure_count": failure_count,
                "observed_failure_count": failure_count,
                "fixture_run_id": run["run_id"],
                "raw_contour_fingerprint": run["raw_contour_fingerprint"],
                "dataset_name": run["dataset_name"],
                "split_name": run["split_name"],
                "frame_index": run["frame_index"],
                "observed_extractor_status": observed_status,
                "diagnostic_only": True,
                "freeze_eligible": False,
                "metrics": {
                    "accepted_raw_contour_count": metrics["accepted_raw_contour_count"],
                    "physical_region_count": metrics["physical_region_count"],
                    "nested_raw_contour_count": metrics["nested_raw_contour_count"],
                    "positive_cells": metrics["positive_cells"],
                    "layer_conservation": output["layer_conservation"],
                },
            }
        except Exception as exc:  # pragma: no cover - failure is reported, not hidden
            results[name] = {
                "status": "fail",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "failure_count": failure_count,
                "observed_failure_count": failure_count,
                "fixture_run_id": run["run_id"],
                "raw_contour_fingerprint": run["raw_contour_fingerprint"],
                "dataset_name": run["dataset_name"],
                "split_name": run["split_name"],
                "frame_index": run["frame_index"],
                "observed_extractor_status": str(
                    run.get("candidate_run_metrics", {}).get("status", "valid")
                ),
                "diagnostic_only": True,
                "freeze_eligible": False,
                "metrics": None,
            }
    failure_count = sum(int(item["status"] != "pass") for item in results.values())
    return {
        "status": "pass" if failure_count == 0 else "fail",
        "mode": "diagnostic-only",
        "diagnostic_only": True,
        "freeze_eligible": False,
        "fixture_count": len(results),
        "failure_count": failure_count,
        "failure_rate": float(failure_count / len(results)) if results else None,
        "failure_rate_threshold": 0.0,
        "diagnostic_fixtures": results,
        "configuration": config.as_dict(),
        "configuration_role": "fixed_diagnostic_configuration; not a selection input",
    }


def _best_controlled_core(
    output: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    exact_mask = np.asarray(fixture["exact_mask"], dtype=bool)
    xdim = np.asarray(fixture["xdim"], dtype=np.float64)
    ydim = np.asarray(fixture["ydim"], dtype=np.float64)
    dx = float(xdim[1] - xdim[0])
    dy = float(ydim[1] - ydim[0])
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("controlled fixture coordinate axes must be increasing")
    max_delta = max(dx, dy)
    expected_center = np.asarray(fixture["expected_center_xy"], dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    for layer in _matching_layers(output):
        core = np.asarray(layer["core"], dtype=bool)
        if not np.any(core) or layer.get("center_xy") is None:
            continue
        center = np.asarray(layer["center_xy"], dtype=np.float64)
        candidates.append({
            "raw_record_index": int(
                layer.get("outer_raw_record_index", layer["raw_record_index"])
            ),
            "physical_region_id": layer.get("physical_region_id"),
            "center_error": float(np.linalg.norm(center - expected_center)),
            "mask_iou": _iou(core, exact_mask),
            "equivalent_radius": float(
                np.sqrt(np.count_nonzero(core) * dx * dy / np.pi)
            ),
            "core": core,
        })
    if not candidates:
        return {
            "status": "fail",
            "reason": "no_nonempty_postprocessed_core",
            "center_error": None,
            "center_tolerance": 2.0 * max_delta,
            "mask_iou": 0.0,
            "equivalent_radius": None,
            "equivalent_radius_relative_error": None,
        }
    selected = max(
        candidates,
        key=lambda item: (
            float(item["mask_iou"]),
            -float(item["center_error"]),
            -abs(float(item["equivalent_radius"]) - float(fixture["expected_radius"])),
        ),
    )
    radius = float(selected["equivalent_radius"])
    expected_radius = float(fixture["expected_radius"])
    radius_error = abs(radius - expected_radius) / expected_radius
    passed = (
        selected["center_error"] <= 2.0 * max_delta
        and selected["mask_iou"] >= CONTROLLED_CORE_IOU_MINIMUM
        and radius_error <= CONTROLLED_RADIUS_RELATIVE_ERROR_MAXIMUM
    )
    return {
        "status": "pass" if passed else "fail",
        "reason": None if passed else "controlled_single_vortex_threshold_failed",
        "matched_raw_record_index": selected["raw_record_index"],
        "center_error": selected["center_error"],
        "center_tolerance": 2.0 * max_delta,
        "mask_iou": selected["mask_iou"],
        "mask_iou_threshold": CONTROLLED_CORE_IOU_MINIMUM,
        "equivalent_radius": radius,
        "expected_equivalent_radius": expected_radius,
        "equivalent_radius_relative_error": radius_error,
        "equivalent_radius_relative_error_threshold": CONTROLLED_RADIUS_RELATIVE_ERROR_MAXIMUM,
        "evaluated_nonempty_core_count": len(candidates),
    }


def _controlled_multi_metrics(output: Mapping[str, Any], fixture: Mapping[str, Any]) -> dict[str, Any]:
    import numbacs_contour_audit

    records = []
    for layer in _matching_layers(output):
        if layer.get("center_xy") is None or not np.any(layer["core"]):
            continue
        dx = float(output["arrays"]["xdim"][1] - output["arrays"]["xdim"][0])
        dy = float(output["arrays"]["ydim"][1] - output["arrays"]["ydim"][0])
        records.append({
            "raw_record_index": int(
                layer.get("outer_raw_record_index", layer["raw_record_index"])
            ),
            "center_xy": [float(item) for item in layer["center_xy"]],
            "area": float(np.count_nonzero(layer["core"]) * dx * dy),
        })
    result = {"selected_contours": records}
    metrics = numbacs_contour_audit._multi_vortex_metrics(
        result, fixture, radius_factor=1.0
    )
    metrics["postprocessed_nonempty_core_count"] = len(records)
    return metrics


def _controlled_validation(
    selected_config: PostprocessingConfig,
    audit_root: Path,
    *,
    mode: str = "formal",
    reason: str | None = None,
) -> dict[str, Any]:
    if mode not in {"formal", "diagnostic-only"}:
        raise ValueError(f"unsupported controlled validation mode: {mode!r}")
    diagnostic = _controlled_fixture_diagnostics(selected_config)
    if mode == "diagnostic-only":
        return {
            **diagnostic,
            "status": "diagnostic-only",
            "reason": reason or "formal_controlled_validation_not_eligible",
            "formal_gate_status": "not-evaluated",
            "required_run_count": 5,
            "source": (
                "deterministic in-memory controlled fixtures; no native extraction rerun"
            ),
        }
    source_runs = _controlled_source_runs(audit_root)
    required_single = {
        "single_vortex_n65",
        "single_vortex_n97",
        "single_vortex_n129",
    }
    required = required_single | {"no_vortex", "multi_vortex"}
    missing = sorted(required - set(source_runs))
    if missing:
        return {
            "status": "no-go",
            "mode": "formal",
            "diagnostic_only": False,
            "freeze_eligible": False,
            "reason": "missing_serialized_controlled_runs",
            "missing_runs": missing,
            "failure_rate": None,
            "diagnostic_fixtures": diagnostic["diagnostic_fixtures"],
            "diagnostic_failure_count": diagnostic["failure_count"],
            "diagnostic_failure_rate": diagnostic["failure_rate"],
            "formal_gate_status": "fail",
        }
    selected_outputs = {
        run["run_id"]: postprocess_raw_run(run, selected_config)
        for run in source_runs.values()
    }
    input_health: dict[str, dict[str, Any]] = {}
    fixture_failures: set[str] = set()
    for fixture_name, run in sorted(source_runs.items()):
        run_metrics = run.get("candidate_run_metrics", {})
        failure_count = int(
            run_metrics.get("failure_count", run_metrics.get("extractor_failure_count", 0))
            or 0
        )
        status = run_metrics.get("status")
        status_pass = status is True or status == "valid"
        passed = bool(status_pass and failure_count == 0)
        input_health[fixture_name] = {
            "run_id": run["run_id"],
            "status": status,
            "status_pass": status_pass,
            "failure_count": failure_count,
            "failure_count_pass": failure_count == 0,
            "pass": passed,
        }
        if not passed:
            fixture_failures.add(fixture_name)
    failures = len(fixture_failures)
    no_vortex_run = source_runs["no_vortex"]
    no_vortex_output = selected_outputs[no_vortex_run["run_id"]]
    no_vortex_positive = int(no_vortex_output["metrics"]["positive_cells"])
    no_vortex_contours = int(
        no_vortex_output["metrics"]["accepted_raw_contour_count"]
    )
    no_vortex_failures = int(
        no_vortex_output["metrics"]["extractor_failure_count"]
    )
    no_vortex = {
        "status": (
            "pass"
            if no_vortex_positive == 0
            and no_vortex_contours == 0
            and no_vortex_failures == 0
            else "fail"
        ),
        "run_id": no_vortex_run["run_id"],
        "positive_cells": no_vortex_positive,
        "accepted_raw_contour_count": no_vortex_contours,
        "extractor_failure_count": no_vortex_failures,
        "reason": (
            None
            if no_vortex_positive == 0
            and no_vortex_contours == 0
            and no_vortex_failures == 0
            else "nonempty_or_failed_no_vortex_fixture"
        ),
    }
    if no_vortex["status"] != "pass":
        fixture_failures.add("no_vortex")

    single: dict[str, Any] = {}
    radii: list[float] = []
    center_pass = True
    for key in sorted(required_single):
        run = source_runs[key]
        output = selected_outputs[run["run_id"]]
        import numbacs_contour_audit

        fixture = numbacs_contour_audit._single_vortex_fixture(int(key.rsplit("n", 1)[1]))
        result = _best_controlled_core(output, fixture)
        result["run_id"] = run["run_id"]
        result["resolution"] = int(fixture["resolution"])
        single[key] = result
        if result["status"] != "pass":
            fixture_failures.add(key)
        if result.get("equivalent_radius") is not None:
            radii.append(float(result["equivalent_radius"]))
        center_pass = center_pass and result.get("center_error") is not None and (
            result["center_error"] <= result["center_tolerance"]
        )
    mean_radius = float(np.mean(radii)) if radii else None
    radius_cv = (
        float(np.std(np.asarray(radii), ddof=0) / mean_radius)
        if mean_radius and radii else None
    )
    resolution_pass = bool(
        len(radii) == 3
        and radius_cv is not None
        and radius_cv <= EQUIVALENT_RADIUS_CV_MAXIMUM
        and center_pass
        and all(item["status"] == "pass" for item in single.values())
    )
    multi_run = source_runs["multi_vortex"]
    import numbacs_contour_audit

    multi_fixture = numbacs_contour_audit._multi_vortex_fixture()
    multi = _controlled_multi_metrics(selected_outputs[multi_run["run_id"]], multi_fixture)
    multi["run_id"] = multi_run["run_id"]
    if multi["status"] != "pass":
        fixture_failures.add("multi_vortex")
    fixture_count = 1 + len(required_single) + 1  # no-vortex, singles, multi
    failures = len(fixture_failures)
    failure_rate = float(failures / fixture_count)
    passed = failures == 0 and failure_rate == 0.0 and resolution_pass
    return {
        "status": "pass" if passed else "no-go",
        "mode": "formal",
        "diagnostic_only": False,
        "freeze_eligible": bool(passed),
        "required_run_count": 5,
        "failure_count": failures,
        "failure_rate": failure_rate,
        "failure_rate_threshold": 0.0,
        "input_health": input_health,
        "no_vortex": no_vortex,
        "single_vortex_by_resolution": single,
        "resolution": {
            "status": "pass" if resolution_pass else "fail",
            "equivalent_radius_values": radii,
            "equivalent_radius_coefficient_of_variation": radius_cv,
            "equivalent_radius_cv_threshold": EQUIVALENT_RADIUS_CV_MAXIMUM,
            "center_within_2_delta": center_pass,
            "center_tolerance_rule": "2 * max(dx, dy)",
        },
        "multi_vortex": multi,
        "diagnostic_fixtures": diagnostic["diagnostic_fixtures"],
        "diagnostic_failure_count": diagnostic["failure_count"],
        "diagnostic_failure_rate": diagnostic["failure_rate"],
        "formal_gate_status": "pass" if passed else "fail",
        "source": "serialized R1 controlled raw-contour runs; no native extraction rerun",
    }


def _artifact_fingerprint(metadata: Mapping[str, Any]) -> str:
    identity = dict(metadata)
    for field in (
        "integrity_fingerprint",
        "array_hashes",
        "json_hashes",
        "freeze_authorization",
    ):
        identity.pop(field, None)
    return _json_hash(identity)


def _literature_provenance() -> dict[str, Any]:
    return {
        "status": "metadata_and_fulltext_verified_local_zotero",
        "zotero_key": "L2PX3NQX",
        "title": "Defining coherent vortices objectively from the vorticity",
        "verified_facts": [
            "IVD(x,t)=abs(omega(x,t)-spatial_mean(omega,t))",
            "spatial mean is taken over the flow domain",
            "Eulerian candidates are nested IVD level sets with outward non-increasing values",
        ],
        "project_engineering_choices": [
            "native NumbaCS raw contours are held fixed for this audit",
            "cell-center and four-sample-majority rasterization are the registered policies",
            "band geometry and negative policy are project post-processing choices",
            "nested raw contours are optionally consolidated into physical regions; only the outer boundary is eroded under the physical_outer_boundary_band policy",
        ],
        "canonical_parameter_claim": False,
    }


def _validate_artifact_run_id(run_id: str) -> None:
    if (
        not run_id
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or Path(run_id).name != run_id
    ):
        raise ValueError(
            "Haller-v2 run_id must be a single safe path component: "
            f"{run_id!r}"
        )


def _write_haller_v2_run_artifact(
    output_root: Path,
    output: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source_context: Mapping[str, Any],
    *,
    artifact_version: str = HALLER_V2_VERSION,
) -> dict[str, Any]:
    run_id = str(output["run_id"])
    _validate_artifact_run_id(run_id)
    run_dir = output_root / "frozen_haller_v2" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Haller-v2 artifact already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    arrays = output["arrays"]
    np.save(run_dir / "standard_ivd.npy", np.asarray(arrays["standard_ivd"], dtype=np.float64))
    np.save(run_dir / "solid_mask.npy", np.asarray(arrays["solid_mask"], dtype=bool))
    np.save(run_dir / "invalid_support.npy", np.asarray(arrays["invalid_support"], dtype=bool))
    np.save(run_dir / "valid_fluid.npy", np.asarray(arrays["valid_fluid"], dtype=bool))
    np.save(run_dir / "filled_interior.npy", np.asarray(arrays["filled_interior"], dtype=bool))
    np.save(run_dir / "unknown_band.npy", np.asarray(arrays["unknown_band"], dtype=bool))
    np.save(run_dir / "negative_candidates.npy", np.asarray(arrays["negative_candidates"], dtype=bool))
    np.save(run_dir / "evidence_state.npy", np.asarray(arrays["evidence_state"], dtype=np.int8))
    np.save(run_dir / "xdim.npy", np.asarray(arrays["xdim"], dtype=np.float64))
    np.save(run_dir / "ydim.npy", np.asarray(arrays["ydim"], dtype=np.float64))
    _write_json(run_dir / "raw_contours.json", output["raw_contours"])
    _write_json(run_dir / "filtering_decisions.json", output["filtering_decisions"])
    files = sorted(
        path for path in run_dir.iterdir()
        if path.is_file() and path.name != "haller_v2_meta.json"
    )
    array_names = {
        path.name for path in files if path.suffix == ".npy"
    }
    json_names = {
        path.name for path in files if path.suffix == ".json"
    }
    metadata: dict[str, Any] = {
        "artifact_type": "haller_v2_layered_evidence",
        "schema_version": POSTPROCESS_SCHEMA_VERSION,
        "version": artifact_version,
        "artifact_state": "frozen",
        "trusted": True,
        "run_id": run_id,
        "dataset_name": str(output["dataset_name"]),
        "split_name": str(output["split_name"]),
        "frame_index": int(output["frame_index"]),
        "scope": "train/calibration_development",
        "final_test_relationships": "not_read",
        "boussinesq_frame_1300": "excluded from Haller-v2 trusted artifacts",
        "candidate_manifest_integrity_fingerprint": candidate["candidate_manifest"]["integrity_fingerprint"],
        "source_run_artifact_fingerprint": output.get("source_run_artifact_fingerprint"),
        "raw_contour_fingerprint": output["raw_contour_fingerprint"],
        "configuration": output["configuration"],
        "metrics": output["metrics"],
        "layer_conservation": output["layer_conservation"],
        "provenance": {
            "producer": "audit_haller_v2_postprocessing",
            "candidate_root": candidate["candidate_root"],
            "candidate_run_relative_dir": output.get("candidate_run_relative_dir"),
            "candidate_file_hashes": output.get("candidate_file_hashes", {}),
            "source_audit_report_sha256": source_context.get("report_sha256"),
            "native_extraction_rerun": False,
            "runtime_identity": _postprocessing_runtime_identity(),
            "r1_candidate": {
                "integrity_fingerprint": candidate["candidate_manifest"].get(
                    "integrity_fingerprint"
                ),
                "backend": candidate["candidate_manifest"].get("backend"),
                "radius_factor": candidate["candidate_manifest"].get("radius_factor"),
                "min_len_factor": candidate["candidate_manifest"].get("min_len_factor"),
                "sample_manifest_hash": candidate["candidate_manifest"].get(
                    "sample_manifest_hash"
                ),
                "parameter_fingerprint": candidate["candidate_manifest"].get(
                    "parameter_fingerprint"
                ),
            },
            "r1_source_audit": dict(source_context),
            "literature": _literature_provenance(),
        },
        "array_hashes": {
            name: _file_sha256(run_dir / name) for name in sorted(array_names)
        },
        "json_hashes": {
            name: _file_sha256(run_dir / name) for name in sorted(json_names)
        },
        "freeze_authorization": None,
    }
    metadata["integrity_fingerprint"] = _artifact_fingerprint(metadata)
    _write_json(run_dir / "haller_v2_meta.json", metadata)
    return {
        "run_id": run_id,
        "artifact_path": run_dir.relative_to(output_root).as_posix(),
        "metadata_path": (run_dir / "haller_v2_meta.json").relative_to(output_root).as_posix(),
        "artifact_fingerprint": metadata["integrity_fingerprint"],
        "raw_contour_fingerprint": output["raw_contour_fingerprint"],
    }


def _update_haller_v2_authorization(
    output_root: Path,
    entry: Mapping[str, Any],
    *,
    report_sha256: str,
    freeze_manifest_fingerprint: str,
    artifact_version: str = HALLER_V2_VERSION,
) -> None:
    metadata_path = output_root / str(entry["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["freeze_authorization"] = {
        "stage": "R2",
        "status": "pass",
        "exact_frozen_version": artifact_version,
        "audit_report_sha256": report_sha256,
        "freeze_manifest_fingerprint": freeze_manifest_fingerprint,
        "artifact_fingerprint": metadata["integrity_fingerprint"],
    }
    _write_json(metadata_path, metadata)


def _verify_freeze_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "haller_v2_freeze_manifest":
        raise ValueError("not a Haller-v2 freeze manifest")
    if manifest.get("schema_version") != POSTPROCESS_SCHEMA_VERSION:
        raise ValueError("unsupported Haller-v2 freeze manifest schema")
    if manifest.get("status") != "pass" or manifest.get("trusted") is not True:
        raise PermissionError("freeze manifest is not a passing trusted authorization")
    if manifest.get("version") not in {
        HALLER_V2_VERSION,
        PHYSICAL_REGION_ARTIFACT_VERSION,
    }:
        raise PermissionError("freeze manifest version is not a registered Haller artifact version")
    if manifest.get("artifact_state") != "frozen":
        raise PermissionError("freeze manifest does not identify frozen artifacts")
    if manifest.get("final_test_relationships") != "not_read":
        raise PermissionError("freeze manifest does not prove final-test isolation")
    identity = dict(manifest)
    declared = identity.pop("integrity_fingerprint", None)
    if declared != _json_hash(identity):
        raise ValueError("freeze manifest integrity fingerprint mismatch")
    return manifest


def load_haller_v2_artifact(
    artifact_dir: str | Path,
    *,
    require_trusted: bool = False,
    freeze_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Read and verify one layered Haller-v2 artifact."""

    artifact_root = Path(artifact_dir).resolve()
    metadata_path = _verify_file_name(artifact_root, "haller_v2_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_type") != "haller_v2_layered_evidence":
        raise ValueError("not a Haller-v2 layered evidence artifact")
    if metadata.get("schema_version") != POSTPROCESS_SCHEMA_VERSION:
        raise ValueError("unsupported Haller-v2 artifact schema")
    if metadata.get("final_test_relationships") != "not_read":
        raise PermissionError("Haller-v2 artifact does not prove final-test isolation")
    if str(metadata.get("dataset_name", "")).casefold() == "boussinesq" and metadata.get("frame_index") == 1300:
        raise PermissionError("Boussinesq frame 1300 cannot be a trusted Haller-v2 artifact")
    if set(metadata.get("array_hashes", {})) != HALLER_V2_ARRAY_FILES:
        raise ValueError("Haller-v2 artifact array hash set is incomplete")
    if set(metadata.get("json_hashes", {})) != HALLER_V2_JSON_FILES:
        raise ValueError("Haller-v2 artifact JSON hash set is incomplete")
    for filename, expected_hash in metadata.get("array_hashes", {}).items():
        path = _verify_file_name(artifact_root, str(filename))
        if _file_sha256(path) != str(expected_hash):
            raise ValueError(f"Haller-v2 array fingerprint mismatch: {filename}")
    for filename, expected_hash in metadata.get("json_hashes", {}).items():
        path = _verify_file_name(artifact_root, str(filename))
        if _file_sha256(path) != str(expected_hash):
            raise ValueError(f"Haller-v2 JSON fingerprint mismatch: {filename}")
    if metadata.get("integrity_fingerprint") != _artifact_fingerprint(metadata):
        raise ValueError("Haller-v2 artifact integrity fingerprint mismatch")
    arrays = {
        "standard_ivd": np.asarray(np.load(artifact_root / "standard_ivd.npy", allow_pickle=False), dtype=np.float64),
        "solid_mask": np.asarray(np.load(artifact_root / "solid_mask.npy", allow_pickle=False), dtype=bool),
        "invalid_support": np.asarray(np.load(artifact_root / "invalid_support.npy", allow_pickle=False), dtype=bool),
        "valid_fluid": np.asarray(np.load(artifact_root / "valid_fluid.npy", allow_pickle=False), dtype=bool),
        "filled_interior": np.asarray(np.load(artifact_root / "filled_interior.npy", allow_pickle=False), dtype=bool),
        "unknown_band": np.asarray(np.load(artifact_root / "unknown_band.npy", allow_pickle=False), dtype=bool),
        "negative_candidates": np.asarray(np.load(artifact_root / "negative_candidates.npy", allow_pickle=False), dtype=bool),
        "evidence_state": np.asarray(np.load(artifact_root / "evidence_state.npy", allow_pickle=False), dtype=np.int8),
        "xdim": np.asarray(np.load(artifact_root / "xdim.npy", allow_pickle=False), dtype=np.float64),
        "ydim": np.asarray(np.load(artifact_root / "ydim.npy", allow_pickle=False), dtype=np.float64),
    }
    if arrays["standard_ivd"].ndim != 2:
        raise ValueError("Haller-v2 standard_ivd must be two-dimensional")
    if any(np.asarray(value).shape != arrays["standard_ivd"].shape for name, value in arrays.items() if name not in {"standard_ivd", "xdim", "ydim"}):
        raise ValueError("Haller-v2 layer shape mismatch")
    if arrays["xdim"].shape != (arrays["standard_ivd"].shape[1],) or arrays["ydim"].shape != (arrays["standard_ivd"].shape[0],):
        raise ValueError("Haller-v2 coordinate shape mismatch")
    positive = arrays["evidence_state"] == 1
    negative = arrays["evidence_state"] == 0
    conservation = _layer_conservation(
        solid_mask=arrays["solid_mask"],
        invalid_support=arrays["invalid_support"],
        valid_fluid=arrays["valid_fluid"],
        positive=positive,
        negative=negative,
        evidence_state=arrays["evidence_state"],
    )
    if conservation["status"] != "pass":
        raise ValueError("Haller-v2 artifact layer conservation failed")
    if not np.array_equal(negative, arrays["negative_candidates"]):
        raise ValueError("Haller-v2 negative layer does not match evidence_state")
    if not np.array_equal(
        positive,
        arrays["valid_fluid"] & arrays["filled_interior"] & ~arrays["unknown_band"],
    ):
        raise ValueError("Haller-v2 positive layer does not match fixed layer rule")
    raw_contours = json.loads(
        _verify_file_name(artifact_root, "raw_contours.json").read_text(encoding="utf-8")
    )
    filtering_decisions = json.loads(
        _verify_file_name(artifact_root, "filtering_decisions.json").read_text(
            encoding="utf-8"
        )
    )
    if require_trusted:
        if (
            metadata.get("version") not in {
                HALLER_V2_VERSION,
                PHYSICAL_REGION_ARTIFACT_VERSION,
            }
            or metadata.get("artifact_state") != "frozen"
            or metadata.get("trusted") is not True
        ):
            raise PermissionError("only exact frozen registered Haller artifacts may be trusted")
        if not isinstance(freeze_manifest, (str, Path)):
            raise PermissionError("independent freeze manifest path is required for trust")
        manifest_path = Path(freeze_manifest).resolve()
        manifest = _verify_freeze_manifest(manifest_path)
        if manifest.get("version") != metadata.get("version"):
            raise PermissionError("freeze manifest/artifact version mismatch")
        artifact_fingerprints = manifest.get("artifact_fingerprints", {})
        if metadata["run_id"] not in artifact_fingerprints or artifact_fingerprints[metadata["run_id"]] != metadata["integrity_fingerprint"]:
            raise PermissionError("artifact is not listed in the exact freeze manifest")
        if manifest.get("raw_contour_fingerprint") != metadata.get("raw_contour_fingerprint"):
            raise PermissionError("freeze manifest/raw-contour fingerprint mismatch")
        report_path = _verify_relative_path(
            manifest_path.parent,
            str(manifest["audit_report_path"]),
        )
        if not report_path.exists():
            raise PermissionError("freeze manifest is detached from its audit report")
        if _file_sha256(report_path) != manifest.get("audit_report_sha256"):
            raise PermissionError("freeze manifest audit report fingerprint mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("artifact_type") != "haller_v2_postprocessing_audit_report":
            raise PermissionError("freeze authorization is detached from a Haller-v2 audit report")
        if report.get("schema_version") != POSTPROCESS_SCHEMA_VERSION:
            raise PermissionError("freeze authorization uses an unsupported audit schema")
        if (
            report.get("version") != manifest.get("version")
            or report.get("version") != metadata.get("version")
            or
            report.get("version") not in {
                HALLER_V2_VERSION,
                PHYSICAL_REGION_ARTIFACT_VERSION,
            }
            or report.get("status") != "pass"
        ):
            raise PermissionError("freeze authorization does not identify a passing registered Haller audit")
        if report.get("protocol", {}).get("final_test_relationships") != "not_read":
            raise PermissionError("Haller-v2 audit report does not prove final-test isolation")
        if report.get("controlled_validation", {}).get("status") != "pass":
            raise PermissionError("freeze authorization lacks passing controlled validation")
        selected = report.get("selected_configuration") or {}
        configuration_key = selected.get("configuration_key")
        selected_audit = report.get("configuration_audits", {}).get(configuration_key)
        if not isinstance(selected_audit, Mapping):
            raise PermissionError("freeze authorization lacks the selected configuration audit")
        if selected_audit.get("configuration") != metadata.get("configuration"):
            raise PermissionError("freeze authorization/configuration mismatch")
        if selected_audit.get("raw_contour_fingerprint") != metadata.get("raw_contour_fingerprint"):
            raise PermissionError("freeze authorization/raw-contour fingerprint mismatch")
        if selected_audit.get("eligible") is not True:
            raise PermissionError("freeze authorization lacks passing raw/development eligibility")
        if selected_audit.get("development_health_pass") is not True:
            raise PermissionError("freeze authorization lacks the development health audit")
        if selected_audit.get("stability", {}).get("status") != "pass":
            raise PermissionError("freeze authorization lacks passing stability audit")
        report_entries = report.get("artifact_index", [])
        if not any(
            entry.get("run_id") == metadata.get("run_id")
            and entry.get("artifact_fingerprint") == metadata.get("integrity_fingerprint")
            for entry in report_entries
            if isinstance(entry, Mapping)
        ):
            raise PermissionError("freeze authorization does not list this artifact in its audit report")
        authorization = metadata.get("freeze_authorization") or {}
        if authorization.get("status") != "pass":
            raise PermissionError("artifact lacks a passing freeze authorization")
        if authorization.get("exact_frozen_version") != metadata.get("version"):
            raise PermissionError("artifact freeze authorization has the wrong version")
        if authorization.get("artifact_fingerprint") != metadata["integrity_fingerprint"]:
            raise PermissionError("artifact freeze authorization fingerprint mismatch")
        if authorization.get("freeze_manifest_fingerprint") != manifest["integrity_fingerprint"]:
            raise PermissionError("artifact lacks the independent freeze authorization")
        if authorization.get("audit_report_sha256") != manifest.get("audit_report_sha256"):
            raise PermissionError("artifact audit-report authorization mismatch")
    return {
        "artifact_dir": str(artifact_root),
        "metadata": metadata,
        "arrays": arrays,
        "raw_contours": raw_contours,
        "filtering_decisions": filtering_decisions,
    }


def require_trusted_haller_v2(
    artifact_dir: str | Path,
    freeze_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Require the independently authorized exact frozen Haller-v2 artifact."""

    return load_haller_v2_artifact(
        artifact_dir,
        require_trusted=True,
        freeze_manifest=freeze_manifest,
    )


def _source_audit_context(candidate: Mapping[str, Any]) -> dict[str, Any]:
    candidate_root = Path(str(candidate["candidate_root"]))
    source_root = candidate_root.parent
    report_path = source_root / "audit_report.json"
    if not report_path.exists():
        return {
            "available": False,
            "report_path": None,
            "report_sha256": None,
            "source_candidate_integrity_fingerprint": None,
            "candidate_manifest_integrity_fingerprint": candidate[
                "candidate_manifest"
            ].get("integrity_fingerprint"),
            "sample_manifest_hash": candidate["sample_manifest"].get(
                "manifest_hash"
            ),
            "protocol_manifest_hash": candidate["sample_manifest"].get(
                "protocol_manifest_hash"
            ),
            "protocol_version": candidate["sample_manifest"].get(
                "protocol_version"
            ),
            "status": "not-provided",
            "eligible_for_r2": False,
        }
    # The raw audit reader verifies the source run artifacts and candidate
    # copy. Import lazily so the direct loader remains usable for unit seams;
    # formal R2 publication still requires this passing source audit context.
    import numbacs_contour_audit

    source_report = numbacs_contour_audit.read_audit_report(source_root)
    selected = source_report.get("selected_raw_contour_candidate") or {}
    if selected.get("integrity_fingerprint") != candidate["candidate_manifest"]["integrity_fingerprint"]:
        raise ValueError("candidate is not the source audit's selected raw-contour candidate")
    source_protocol = source_report.get("protocol", {})
    if source_protocol.get("final_test_relationships") != (
        "not_read; train/calibration development audit only"
    ):
        raise PermissionError("source audit does not prove final-test isolation")
    if source_protocol.get("protocol_version") != candidate["sample_manifest"].get(
        "protocol_version"
    ):
        raise ValueError("source audit and candidate sample protocol versions differ")
    if source_protocol.get("manifest_hash") != candidate["sample_manifest"].get(
        "protocol_manifest_hash"
    ):
        raise ValueError("source audit and candidate sample protocol fingerprints differ")
    return {
        "available": True,
        "report_path": str(report_path),
        "report_sha256": _file_sha256(report_path),
        "source_candidate_integrity_fingerprint": selected.get("integrity_fingerprint"),
        "candidate_manifest_integrity_fingerprint": candidate[
            "candidate_manifest"
        ].get("integrity_fingerprint"),
        "sample_manifest_hash": candidate["sample_manifest"].get("manifest_hash"),
        "status": source_report.get("status"),
        "eligible_for_r2": source_report.get("status") == "pass",
        "protocol_manifest_hash": source_protocol.get("manifest_hash"),
        "protocol_version": source_protocol.get("protocol_version"),
        "backend": source_report.get("backend"),
        "configuration": source_report.get("configuration"),
        "protocol": source_report.get("protocol"),
    }


def _configuration_audits(
    configs: Sequence[PostprocessingConfig],
    outputs_by_config: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    audits: dict[str, dict[str, Any]] = {}
    for config in configs:
        key = _configuration_key(config)
        outputs = outputs_by_config[key]
        dataset_metrics = _dataset_metrics(outputs)
        raw_eligibility_pass, eligibility_by_dataset = _configuration_eligible(dataset_metrics)
        development_health_pass, development_health = _development_health(dataset_metrics)
        eligible = bool(raw_eligibility_pass and development_health_pass)
        for dataset_name, health in development_health.items():
            eligibility_by_dataset.setdefault(dataset_name, {})[
                "development_health"
            ] = health
        raw_fingerprints = sorted({
            str(output["raw_contour_fingerprint"])
            for output in outputs.values()
        })
        if len(raw_fingerprints) != 1:
            raise ValueError("post-processing configuration consumed multiple raw fingerprints")
        audits[key] = {
            "configuration": config.as_dict(),
            "raw_contour_fingerprint": raw_fingerprints[0],
            "run_count": len(outputs),
            "by_dataset": dataset_metrics,
            "by_run": {
                run_id: output["metrics"] for run_id, output in sorted(outputs.items())
            },
            "eligibility_by_dataset": eligibility_by_dataset,
            "development_health": development_health,
            "development_health_pass": bool(development_health_pass),
            "raw_eligibility_pass": bool(raw_eligibility_pass),
            "eligible": bool(eligible),
        }
    for config in configs:
        key = _configuration_key(config)
        audits[key]["stability"] = _stability_for_configuration(config, outputs_by_config)
    return audits


def _geometry_metric_snapshot(metrics: Mapping[str, Any]) -> dict[str, Any]:
    attribution = metrics.get("unknown_attribution", {})
    identity = metrics.get("identity_attribution", {})
    return {
        "geometry_policy": metrics.get("geometry_policy"),
        "geometry_erosion_unit": metrics.get("geometry_erosion_unit"),
        "accepted_raw_contour_count": int(metrics.get("accepted_raw_contour_count", 0)),
        "raw_contour_count": int(
            metrics.get(
                "raw_contour_count",
                metrics.get("accepted_raw_contour_count", 0),
            )
        ),
        "geometry_unit_count": int(metrics.get("geometry_unit_count", 0)),
        "physical_region_count": int(metrics.get("physical_region_count", 0)),
        "nested_raw_contour_count": int(metrics.get("nested_raw_contour_count", 0)),
        "identity_attribution": {
            "counts": dict(identity.get("counts", {})),
            "accepted_raw_contour_count": int(
                identity.get("accepted_raw_contour_count", 0)
            ),
            "conservation": bool(identity.get("conservation", False)),
        },
        "boundary_band_cells": int(attribution.get("boundary_band", 0)),
        "boundary_band_area_physical": metrics.get("boundary_band_area_physical"),
        "positive_cells": int(metrics.get("positive_cells", 0)),
        "positive_area_physical": metrics.get("positive_area_physical"),
        "raw_interior_area_physical": metrics.get("raw_interior_area_physical"),
        "known_coverage": metrics.get("known_coverage"),
        "unknown_coverage": metrics.get("unknown_coverage"),
        "positive_raw_interior_ratio": metrics.get("positive_raw_interior_ratio"),
        "fully_erased_geometry_fraction": metrics.get("fully_erased_geometry_fraction"),
        "fully_erased_contour_fraction": metrics.get("fully_erased_contour_fraction"),
        "region_scale_versus_erosion": metrics.get(
            "region_scale_versus_erosion", {}
        ),
    }


def _geometry_semantics_comparison(
    configs: Sequence[PostprocessingConfig],
    audits: Mapping[str, Mapping[str, Any]],
    outputs_by_config: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compare the two fixed-raw geometry meanings at identical grid points."""

    all_configs = [
        config for config in configs if config.geometry_policy == "all_contour_band"
    ]
    comparisons: list[dict[str, Any]] = []
    for all_config in all_configs:
        physical_config = PostprocessingConfig(
            band_factor=all_config.band_factor,
            rasterization_policy=all_config.rasterization_policy,
            negative_policy=all_config.negative_policy,
            geometry_policy="physical_outer_boundary_band",
        )
        all_key = _configuration_key(all_config)
        physical_key = _configuration_key(physical_config)
        all_outputs = outputs_by_config[all_key]
        physical_outputs = outputs_by_config[physical_key]
        all_aggregate = _aggregate_run_metrics(list(all_outputs.values()))
        physical_aggregate = _aggregate_run_metrics(list(physical_outputs.values()))
        all_by_dataset = _dataset_metrics(all_outputs)
        physical_by_dataset = _dataset_metrics(physical_outputs)
        all_snapshot = _geometry_metric_snapshot(all_aggregate)
        physical_snapshot = _geometry_metric_snapshot(physical_aggregate)
        delta: dict[str, Any] = {}
        for field in (
            "geometry_unit_count",
            "physical_region_count",
            "nested_raw_contour_count",
            "boundary_band_cells",
            "boundary_band_area_physical",
            "positive_cells",
            "positive_area_physical",
            "raw_interior_area_physical",
            "known_coverage",
            "unknown_coverage",
            "positive_raw_interior_ratio",
            "fully_erased_geometry_fraction",
        ):
            left = all_snapshot.get(field)
            right = physical_snapshot.get(field)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                delta[field] = right - left
            else:
                delta[field] = None
        comparisons.append({
            "base_configuration": {
                key: value
                for key, value in all_config.as_dict().items()
                if key != "geometry_policy"
            },
            "paired_geometry_policies": list(GEOMETRY_POLICIES),
            "raw_contour_fingerprint": next(iter({
                str(output["raw_contour_fingerprint"])
                for output in all_outputs.values()
            })),
            "all_contour_band": all_snapshot,
            "physical_outer_boundary_band": physical_snapshot,
            "delta_physical_minus_all": delta,
            "by_dataset": {
                dataset_name: {
                    "all_contour_band": _geometry_metric_snapshot(
                        all_by_dataset[dataset_name]
                    ),
                    "physical_outer_boundary_band": _geometry_metric_snapshot(
                        physical_by_dataset[dataset_name]
                    ),
                }
                for dataset_name in sorted(all_by_dataset)
            },
            "audit_gates": {
                "all_contour_band": {
                    "raw_eligibility_pass": bool(
                        audits[all_key].get("raw_eligibility_pass", False)
                    ),
                    "development_health_pass": bool(
                        audits[all_key].get("development_health_pass", False)
                    ),
                    "eligible": bool(audits[all_key].get("eligible", False)),
                    "stability": audits[all_key]["stability"]["status"],
                },
                "physical_outer_boundary_band": {
                    "raw_eligibility_pass": bool(
                        audits[physical_key].get("raw_eligibility_pass", False)
                    ),
                    "development_health_pass": bool(
                        audits[physical_key].get("development_health_pass", False)
                    ),
                    "eligible": bool(audits[physical_key].get("eligible", False)),
                    "stability": audits[physical_key]["stability"]["status"],
                },
            },
        })

    policy_gate_summary: dict[str, dict[str, Any]] = {}
    for policy in GEOMETRY_POLICIES:
        policy_audits = [
            audit for audit in audits.values()
            if audit["configuration"].get("geometry_policy") == policy
        ]
        policy_gate_summary[policy] = {
            "configuration_count": len(policy_audits),
            "raw_eligibility_pass_count": sum(
                int(audit.get("raw_eligibility_pass", False))
                for audit in policy_audits
            ),
            "development_health_pass_count": sum(
                int(audit.get("development_health_pass", False))
                for audit in policy_audits
            ),
            "eligible_count": sum(
                int(audit.get("eligible", False)) for audit in policy_audits
            ),
            "stability_pass_count": sum(
                int(audit.get("stability", {}).get("status") == "pass")
                for audit in policy_audits
            ),
            "eligible_and_stable_count": sum(
                int(
                    audit.get("eligible", False)
                    and audit.get("stability", {}).get("status") == "pass"
                )
                for audit in policy_audits
            ),
        }

    strict_band_reductions = sum(
        int(
            comparison["delta_physical_minus_all"]["boundary_band_cells"] < 0
        )
        for comparison in comparisons
    )
    strict_positive_recoveries = sum(
        int(comparison["delta_physical_minus_all"]["positive_cells"] > 0)
        for comparison in comparisons
    )
    consolidation_observed = any(
        comparison["physical_outer_boundary_band"]["geometry_unit_count"]
        < comparison["all_contour_band"]["geometry_unit_count"]
        for comparison in comparisons
    )
    nested_observed = any(
        comparison["all_contour_band"]["nested_raw_contour_count"] > 0
        for comparison in comparisons
    )
    mechanism_supported = bool(
        comparisons
        and nested_observed
        and consolidation_observed
        and strict_band_reductions > 0
        and strict_positive_recoveries > 0
    )
    return {
        "status": "diagnostic",
        "question": (
            "Does treating every nested raw contour as a boundary create the "
            "Ticket 4 no-freeze geometry failure?"
        ),
        "raw_contours_fixed": True,
        "native_extraction_rerun": False,
        "policies": list(GEOMETRY_POLICIES),
        "base_configuration_count": len(all_configs),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "policy_gate_summary": policy_gate_summary,
        "hypothesis_assessment": {
            "status": "supported" if mechanism_supported else "not-supported-or-insufficient",
            "nested_raw_contours_observed": nested_observed,
            "physical_region_consolidation_observed": consolidation_observed,
            "strict_boundary_band_reductions": strict_band_reductions,
            "strict_positive_recoveries": strict_positive_recoveries,
            "basis": (
                "At identical band, rasterization, and negative-policy settings, "
                "physical outer-boundary semantics reduce the boundary band and "
                "recover positive raw-interior cells while the raw fingerprint "
                "is unchanged."
                if mechanism_supported
                else "The fixed-raw paired comparison did not establish all required "
                "geometry-mechanism observations."
            ),
            "scope": (
                "This is evidence for the geometry mechanism only; it does not "
                "bypass controlled-fixture, erosion, development-health, or "
                "stability gates required for Haller-v2 freeze."
            ),
        },
    }


def _evaluate_fixed_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    configs = complete_geometry_configuration_grid()
    outputs_by_config: dict[str, dict[str, dict[str, Any]]] = {}
    for config in configs:
        key = _configuration_key(config)
        outputs: dict[str, dict[str, Any]] = {}
        for run in candidate["runs"]:
            run_id = str(run["run_id"])
            if run_id in outputs:
                raise ValueError(
                    f"duplicate run_id in fixed-raw evaluation: {run_id}"
                )
            outputs[run_id] = postprocess_raw_run(
                run, config, include_contour_layers=False
            )
        if len(outputs) != len(candidate["runs"]):
            raise ValueError("fixed-raw evaluation lost candidate runs")
        outputs_by_config[key] = outputs
    audits = _configuration_audits(configs, outputs_by_config)
    selected, eligible_and_stable = _select_configuration(audits)
    fingerprints = sorted({
        str(run["raw_contour_fingerprint"]) for run in candidate["runs"]
    })
    if len(fingerprints) != 1 or fingerprints[0] != candidate["raw_contour_fingerprint"]:
        raise ValueError("candidate runs do not share the candidate raw-contour fingerprint")
    return {
        "configs": configs,
        "outputs_by_config": outputs_by_config,
        "configuration_audits": audits,
        "selected": selected,
        "eligible_and_stable": eligible_and_stable,
        "geometry_semantics_comparison": _geometry_semantics_comparison(
            configs, audits, outputs_by_config
        ),
        "raw_contour_fingerprint": candidate["raw_contour_fingerprint"],
    }


def _build_remediation_manifest(
    *,
    candidate: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    source_context: Mapping[str, Any],
) -> dict[str, Any]:
    sample_manifest = candidate["sample_manifest"]
    base_configuration_count = len(evaluation["configs"]) // len(GEOMETRY_POLICIES)
    configuration_count = len(evaluation["configs"])
    payload: dict[str, Any] = {
        "artifact_type": "haller_v2_remediation_manifest",
        "schema_version": POSTPROCESS_SCHEMA_VERSION,
        "protocol_version": REMEDIATION_PROTOCOL_VERSION,
        "study_slug": experiment_protocol.STUDY_SLUG,
        "artifact_state": "diagnostic",
        "raw_contours_fixed": True,
        "native_extraction_rerun": False,
        "final_test_relationships": "not_read",
        "r1_candidate": {
            "manifest_path": str(
                Path(candidate["candidate_root"]) / "candidate_manifest.json"
            ),
            "integrity_fingerprint": candidate["candidate_manifest"].get(
                "integrity_fingerprint"
            ),
            "backend": candidate["candidate_manifest"].get("backend"),
            "radius_factor": candidate["candidate_manifest"].get("radius_factor"),
            "min_len_factor": candidate["candidate_manifest"].get("min_len_factor"),
            "parameter_fingerprint": candidate["candidate_manifest"].get(
                "parameter_fingerprint"
            ),
        },
        "raw_contour_fingerprint": evaluation["raw_contour_fingerprint"],
        "r1_source_audit": dict(source_context),
        "r0_protocol": {
            "protocol_version": sample_manifest.get("protocol_version"),
            "protocol_manifest_hash": sample_manifest.get("protocol_manifest_hash"),
        },
        "sample_manifest": {
            "path": str(Path(candidate["candidate_root"]).parent / "sample_manifest.json"),
            "manifest_hash": sample_manifest.get("manifest_hash"),
            "sample_count": sample_manifest.get("sample_count"),
        },
        "protocol_identity": {
            "remediation_protocol_version": REMEDIATION_PROTOCOL_VERSION,
            "postprocessing_schema_version": POSTPROCESS_SCHEMA_VERSION,
            "geometry_policies": list(GEOMETRY_POLICIES),
            "band_factors": list(BAND_FACTORS),
            "rasterization_policies": list(RASTERIZATION_POLICIES),
            "negative_policies": list(NEGATIVE_POLICIES),
            "raw_contours_fixed": True,
            "native_extraction_rerun": False,
            "upstream_change_policy": (
                "a changed extraction identity, parameter set, or raw contour collection "
                "requires a new immutable R1 candidate and a separate remediation audit"
            ),
            "final_test_relationships": "not_read",
            "boussinesq_frame_1300": "development/audit-only and excluded",
        },
        "paired_policy_grid": {
            "base_configuration_count": base_configuration_count,
            "paired_base_configuration_count": base_configuration_count,
            "policy_evaluation_count": configuration_count,
            "expected_base_configuration_count": 20,
            "expected_policy_evaluation_count": 40,
            "pairing": "same band, rasterization, and negative policy; two geometry policies",
        },
    }
    payload["integrity_fingerprint"] = _json_hash(payload)
    return payload


def _verify_remediation_manifest(
    output_root: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    reference = report.get("remediation_manifest", {})
    manifest_path = _verify_relative_path(
        output_root, str(reference.get("path", "remediation_manifest.json"))
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "haller_v2_remediation_manifest":
        raise ValueError("not a Haller-v2 remediation manifest")
    if manifest.get("schema_version") != POSTPROCESS_SCHEMA_VERSION:
        raise ValueError("unsupported remediation manifest schema")
    if manifest.get("protocol_version") != REMEDIATION_PROTOCOL_VERSION:
        raise ValueError("unsupported remediation protocol version")
    identity = dict(manifest)
    declared = identity.pop("integrity_fingerprint", None)
    if declared != _json_hash(identity):
        raise ValueError("remediation manifest integrity fingerprint mismatch")
    if reference.get("integrity_fingerprint") != declared:
        raise ValueError("audit report/remediation manifest fingerprint mismatch")
    if manifest.get("raw_contour_fingerprint") != report.get("input", {}).get(
        "raw_contour_fingerprint"
    ):
        raise ValueError("remediation manifest/raw-contour fingerprint mismatch")
    candidate_fingerprint = manifest.get("r1_candidate", {}).get(
        "integrity_fingerprint"
    )
    if candidate_fingerprint != report.get("input", {}).get(
        "candidate_manifest_integrity_fingerprint"
    ):
        raise ValueError("remediation manifest/R1 candidate fingerprint mismatch")
    protocol = report.get("protocol", {})
    r0 = manifest.get("r0_protocol", {})
    sample = manifest.get("sample_manifest", {})
    if (
        r0.get("protocol_version") != protocol.get("r0_protocol_version")
        or r0.get("protocol_manifest_hash") != protocol.get("r0_protocol_manifest_hash")
        or sample.get("manifest_hash") != protocol.get("sample_manifest_hash")
    ):
        raise ValueError("remediation manifest/R0 or sample fingerprint mismatch")
    identity_block = manifest.get("protocol_identity", {})
    if (
        identity_block.get("raw_contours_fixed") is not True
        or identity_block.get("native_extraction_rerun") is not False
        or identity_block.get("final_test_relationships") != "not_read"
    ):
        raise PermissionError("remediation manifest does not prove fixed-raw isolation")
    grid = manifest.get("paired_policy_grid", {})
    if (
        grid.get("base_configuration_count") != 20
        or grid.get("paired_base_configuration_count") != 20
        or grid.get("policy_evaluation_count") != 40
    ):
        raise ValueError("remediation manifest paired grid is incomplete")
    return manifest


def _report_jsonable(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _report_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _report_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_report_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _audit_haller_v2_postprocessing_impl(
    candidate_dir: str | Path,
    output_dir: str | Path,
    *,
    controlled_root: str | Path | None = None,
    staging_state: dict[str, Any],
) -> dict[str, Any]:
    """Audit the complete fixed-raw grid and publish an auditable decision.

    The source candidate is loaded once and then reused for every grid point.
    A trusted artifact is published only after the development gates and the
    serialized controlled fixtures pass; otherwise the output is an explicit
    no-freeze report with no trusted artifact.
    """

    candidate = load_raw_contour_candidate(candidate_dir)
    source_context = _source_audit_context(candidate)
    source_audit_pass = bool(source_context.get("eligible_for_r2"))
    output_root = Path(output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(
            "Haller-v2 output directory must be new and absent; "
            f"refusing to overwrite: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=str(output_root.parent),
        )
    )
    staging_state["staging_root"] = str(staging_root)
    evaluation = _evaluate_fixed_candidate(candidate)
    selected = evaluation["selected"] if source_audit_pass else None
    eligible_and_stable = (
        evaluation["eligible_and_stable"] if source_audit_pass else []
    )
    validation_root = (
        Path(controlled_root).resolve()
        if controlled_root is not None
        else Path(candidate["candidate_root"]).resolve().parent
    )
    selected_outputs: Mapping[str, Mapping[str, Any]] | None = None
    selected_config: PostprocessingConfig | None = None
    if selected is not None:
        selected_key = str(selected["configuration_key"])
        selected_outputs = evaluation["outputs_by_config"][selected_key]
        selected_config = PostprocessingConfig(
            band_factor=float(selected["configuration"]["band_factor"]),
            rasterization_policy=str(selected["configuration"]["rasterization_policy"]),
            negative_policy=str(selected["configuration"]["negative_policy"]),
            geometry_policy=str(selected["configuration"]["geometry_policy"]),
        )
    diagnostic_config = PostprocessingConfig(**CONTROLLED_DIAGNOSTIC_CONFIGURATION)
    controlled_config = selected_config or diagnostic_config
    controlled = _controlled_validation(
        controlled_config,
        validation_root,
        mode="formal" if selected_config is not None else "diagnostic-only",
        reason=(
            None
            if selected_config is not None
            else (
                "source_r1_audit_not_passing"
                if not source_audit_pass
                else "no_configuration_passed_eligibility_and_stability"
            )
        ),
    )
    if selected_config is not None and not (validation_root / "runs").exists():
        controlled["controlled_root"] = str(validation_root)
    freeze_reason = (
        None
        if selected is not None and controlled.get("status") == "pass"
        else str(
            controlled.get(
                "reason",
                "no_configuration_passed_eligibility_and_stability"
                if selected is None
                else "controlled_validation_failed",
            )
        )
    )

    development_pass = selected is not None
    freeze_pass = bool(
        source_audit_pass
        and development_pass
        and controlled.get("mode") == "formal"
        and controlled.get("status") == "pass"
        and controlled.get("freeze_eligible") is True
    )
    official_raw_gate = _official_raw_gate_summary(
        evaluation["configuration_audits"]
    )
    freeze_eligibility = _freeze_eligibility_summary(
        selected=selected,
        configuration_audits=evaluation["configuration_audits"],
        controlled=controlled,
        source_audit_pass=source_audit_pass,
    )
    artifact_entries: list[dict[str, Any]] = []
    artifact_version = (
        PHYSICAL_REGION_ARTIFACT_VERSION
        if selected_config is not None
        and selected_config.geometry_policy == "physical_outer_boundary_band"
        else HALLER_V2_VERSION
    )
    if freeze_pass:
        assert selected_outputs is not None
        for run_id in sorted(selected_outputs):
            artifact_entries.append(
                _write_haller_v2_run_artifact(
                    staging_root,
                    selected_outputs[run_id],
                    candidate,
                    source_context,
                    artifact_version=artifact_version,
                )
            )
    remediation_manifest = _build_remediation_manifest(
        candidate=candidate,
        evaluation=evaluation,
        source_context=source_context,
    )
    _write_json(staging_root / "remediation_manifest.json", remediation_manifest)
    report: dict[str, Any] = {
        "artifact_type": "haller_v2_postprocessing_audit_report",
        "schema_version": POSTPROCESS_SCHEMA_VERSION,
        "producer": "audit_haller_v2_postprocessing",
        "runtime_identity": _postprocessing_runtime_identity(),
        "status": "pass" if freeze_pass else "no-freeze",
        "study_slug": "criterion-ambiguity-vortex-learning",
        "version": artifact_version,
        "remediation_manifest": {
            "path": "remediation_manifest.json",
            "integrity_fingerprint": remediation_manifest["integrity_fingerprint"],
        },
        "input": {
            "candidate_dir": str(Path(candidate_dir).resolve()),
            "candidate_manifest_integrity_fingerprint": candidate["candidate_manifest"]["integrity_fingerprint"],
            "raw_contour_fingerprint": evaluation["raw_contour_fingerprint"],
            "r1_candidate": {
                "integrity_fingerprint": candidate["candidate_manifest"].get(
                    "integrity_fingerprint"
                ),
                "backend": candidate["candidate_manifest"].get("backend"),
                "radius_factor": candidate["candidate_manifest"].get("radius_factor"),
                "min_len_factor": candidate["candidate_manifest"].get("min_len_factor"),
                "sample_manifest_hash": candidate["candidate_manifest"].get(
                    "sample_manifest_hash"
                ),
                "parameter_fingerprint": candidate["candidate_manifest"].get(
                    "parameter_fingerprint"
                ),
            },
            "source_audit": source_context,
            "native_extraction_rerun": False,
        },
        "protocol": {
            "remediation_protocol_version": REMEDIATION_PROTOCOL_VERSION,
            "r0_protocol_version": candidate["sample_manifest"].get(
                "protocol_version"
            ),
            "r0_protocol_manifest_hash": candidate["sample_manifest"].get(
                "protocol_manifest_hash"
            ),
            "sample_manifest_hash": candidate["sample_manifest"].get(
                "manifest_hash"
            ),
            "final_test_relationships": "not_read",
            "boussinesq_frame_1300": "excluded from all Haller-v2 samples; development/audit-only",
            "scope": "train/calibration development samples only",
        },
        "configuration_grid": {
            "band_factors": list(BAND_FACTORS),
            "rasterization_policies": list(RASTERIZATION_POLICIES),
            "negative_policies": list(NEGATIVE_POLICIES),
            "geometry_policies": list(GEOMETRY_POLICIES),
            "base_configuration_count": len(evaluation["configs"]) // len(GEOMETRY_POLICIES),
            "paired_base_configuration_count": len(evaluation["configs"]) // len(GEOMETRY_POLICIES),
            "configuration_count": len(evaluation["configs"]),
            "policy_evaluation_count": len(evaluation["configs"]),
            "neighbor_rule": (
                "adjacent band values, alternate fixed rasterization policy, and "
                "alternate negative policy within the same geometry policy; "
                "no re-extraction"
            ),
        },
        "configuration_audits": evaluation["configuration_audits"],
        "official_raw_gate": official_raw_gate,
        "geometry_semantics_diagnostic": evaluation["geometry_semantics_comparison"],
        "geometry_semantics_comparison": evaluation["geometry_semantics_comparison"],
        "selected_configuration": selected,
        "eligible_and_stable_configurations": eligible_and_stable,
        "source_audit_gate": {
            "available": bool(source_context.get("available")),
            "status": source_context.get("status"),
            "pass": source_audit_pass,
            "reason": None if source_audit_pass else "source_r1_audit_not_passing",
        },
        "controlled_validation": controlled,
        "freeze_eligibility": freeze_eligibility,
        "artifact_index": artifact_entries,
        "freeze_decision": {
            "status": "pass" if freeze_pass else "no-freeze",
            "reason": freeze_reason,
            "trusted": bool(freeze_pass),
            "candidate_or_audited_artifact_is_trusted": False,
            "freeze_manifest_path": "haller_v2_freeze_manifest.json" if freeze_pass else None,
        },
        "candidate_trust": {
            "artifact_state": "candidate",
            "trusted": False,
            "eligible_as_trusted_haller_anchor": False,
            "exact_frozen_version": None,
        },
        "trusted_haller_v2": {
            "artifact_state": "frozen" if freeze_pass else None,
            "trusted": bool(freeze_pass),
            "exact_version": artifact_version if freeze_pass else None,
            "artifact_count": len(artifact_entries),
        },
    }
    report_path = staging_root / "haller_v2_postprocessing_audit_report.json"
    _write_json(report_path, report)
    report_sha256 = _file_sha256(report_path)
    freeze_manifest_path: Path | None = None
    freeze_manifest_fingerprint: str | None = None
    if freeze_pass:
        freeze_manifest = {
            "artifact_type": "haller_v2_freeze_manifest",
            "schema_version": POSTPROCESS_SCHEMA_VERSION,
            "version": artifact_version,
            "artifact_state": "frozen",
            "status": "pass",
            "trusted": True,
            "audit_report_path": report_path.name,
            "audit_report_sha256": report_sha256,
            "raw_contour_fingerprint": evaluation["raw_contour_fingerprint"],
            "configuration": selected["configuration"],
            "artifact_fingerprints": {
                entry["run_id"]: entry["artifact_fingerprint"]
                for entry in artifact_entries
            },
            "artifact_paths": {
                entry["run_id"]: entry["artifact_path"] for entry in artifact_entries
            },
            "controlled_validation_fingerprint": _json_hash(controlled),
            "final_test_relationships": "not_read",
            "boussinesq_frame_1300": "excluded from Haller-v2 trusted artifacts",
            "producer": "audit_haller_v2_postprocessing",
        }
        freeze_manifest["integrity_fingerprint"] = _json_hash(freeze_manifest)
        freeze_manifest_path = staging_root / "haller_v2_freeze_manifest.json"
        _write_json(freeze_manifest_path, freeze_manifest)
        freeze_manifest_fingerprint = str(freeze_manifest["integrity_fingerprint"])
        for entry in artifact_entries:
            _update_haller_v2_authorization(
                staging_root,
                entry,
                report_sha256=report_sha256,
                freeze_manifest_fingerprint=freeze_manifest_fingerprint,
                artifact_version=artifact_version,
            )
    _write_json(staging_root / "haller_v2_postprocessing_audit_report_integrity.json", {
        "artifact_type": "haller_v2_postprocessing_audit_report_integrity",
        "report_path": report_path.name,
        "report_sha256": report_sha256,
        "report_status": report["status"],
        "remediation_manifest_path": "remediation_manifest.json",
        "remediation_manifest_integrity_fingerprint": remediation_manifest[
            "integrity_fingerprint"
        ],
    })
    _publish_directory_no_overwrite(staging_root, output_root)
    staging_state["published"] = True
    return read_haller_v2_audit_report(output_root)


def audit_haller_v2_postprocessing(
    candidate_dir: str | Path,
    output_dir: str | Path,
    *,
    controlled_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit fixed raw contours and clean unpublished staging on every failure."""

    staging_state: dict[str, Any] = {}
    try:
        return _audit_haller_v2_postprocessing_impl(
            candidate_dir,
            output_dir,
            controlled_root=controlled_root,
            staging_state=staging_state,
        )
    finally:
        if not staging_state.get("published"):
            staging_root = staging_state.get("staging_root")
            if staging_root is not None:
                shutil.rmtree(Path(str(staging_root)), ignore_errors=True)


def read_haller_v2_audit_report(output_dir: str | Path) -> dict[str, Any]:
    """Read back the published no-freeze audit report and its hash sidecar."""

    output_root = Path(output_dir)
    report_path = output_root / "haller_v2_postprocessing_audit_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    integrity = json.loads(
        (output_root / "haller_v2_postprocessing_audit_report_integrity.json").read_text(
            encoding="utf-8"
        )
    )
    if integrity.get("report_sha256") != _file_sha256(report_path):
        raise ValueError("Haller-v2 audit report fingerprint mismatch")
    if integrity.get("report_status") != report.get("status"):
        raise ValueError("Haller-v2 audit report status sidecar mismatch")
    if integrity.get("remediation_manifest_path") != report.get(
        "remediation_manifest", {}
    ).get("path") or integrity.get(
        "remediation_manifest_integrity_fingerprint"
    ) != report.get("remediation_manifest", {}).get("integrity_fingerprint"):
        raise ValueError("Haller-v2 remediation manifest sidecar mismatch")
    if report.get("schema_version") != POSTPROCESS_SCHEMA_VERSION:
        raise ValueError("unsupported Haller-v2 audit report schema")
    _verify_remediation_manifest(output_root, report)
    report_status = report.get("status")
    if report_status not in {"pass", "no-freeze", "no-go"}:
        raise ValueError("unsupported Haller-v2 audit report status")
    selected_configuration = report.get("selected_configuration")
    if selected_configuration is None:
        expected_version = HALLER_V2_VERSION
    else:
        selected_policy = (
            selected_configuration.get("configuration", {}).get("geometry_policy")
            if isinstance(selected_configuration, Mapping)
            else None
        )
        if selected_policy not in GEOMETRY_POLICIES:
            raise ValueError("audit report selected configuration has an invalid geometry policy")
        expected_version = (
            PHYSICAL_REGION_ARTIFACT_VERSION
            if selected_policy == "physical_outer_boundary_band"
            else HALLER_V2_VERSION
        )
    if report.get("version") != expected_version:
        raise ValueError("audit report version does not match its geometry semantics")
    if report.get("configuration_grid", {}).get("band_factors") != list(BAND_FACTORS):
        raise ValueError("Haller-v2 audit report band grid is incomplete")
    if report.get("configuration_grid", {}).get("geometry_policies") != list(GEOMETRY_POLICIES):
        raise ValueError("Haller-v2 audit report geometry grid is incomplete")
    if report.get("configuration_grid", {}).get("configuration_count") != 40:
        raise ValueError("Haller-v2 audit report configuration grid is incomplete")
    if (
        report.get("configuration_grid", {}).get("base_configuration_count") != 20
        or report.get("configuration_grid", {}).get("paired_base_configuration_count") != 20
        or report.get("configuration_grid", {}).get("policy_evaluation_count") != 40
    ):
        raise ValueError("Haller-v2 paired policy grid is incomplete")
    geometry_comparison = report.get(
        "geometry_semantics_diagnostic",
        report.get("geometry_semantics_comparison", {}),
    )
    if (
        geometry_comparison.get("status") != "diagnostic"
        or geometry_comparison.get("raw_contours_fixed") is not True
        or geometry_comparison.get("native_extraction_rerun") is not False
        or geometry_comparison.get("comparison_count") != 20
    ):
        raise ValueError("Haller-v2 geometry-semantics comparison is incomplete")
    if report.get("geometry_semantics_comparison") != geometry_comparison:
        raise ValueError("Haller-v2 geometry-semantics diagnostic aliases disagree")
    if report.get("official_raw_gate", {}).get("physical_policy_cannot_bypass") is not True:
        raise ValueError("Haller-v2 report lacks the official raw gate separation")
    if not isinstance(report.get("freeze_eligibility"), Mapping):
        raise ValueError("Haller-v2 report lacks freeze eligibility separation")
    freeze_eligibility = report["freeze_eligibility"]
    freeze_decision = report.get("freeze_decision", {})
    if freeze_decision.get("status") != report_status:
        raise ValueError("Haller-v2 freeze decision/status mismatch")
    if bool(freeze_eligibility.get("eligible")) != (report_status == "pass"):
        raise ValueError("Haller-v2 freeze eligibility/status mismatch")
    if freeze_eligibility.get("status") != (
        "eligible" if report_status == "pass" else "ineligible"
    ):
        raise ValueError("Haller-v2 freeze eligibility status is inconsistent")
    if bool(freeze_decision.get("trusted")) != (report_status == "pass"):
        raise ValueError("Haller-v2 freeze decision trust status is inconsistent")
    if report_status != "pass":
        if report.get("artifact_index"):
            raise ValueError("no-freeze audit report unexpectedly lists trusted artifacts")
        if report.get("trusted_haller_v2", {}).get("trusted") is not False:
            raise ValueError("no-freeze audit report unexpectedly marks Haller evidence trusted")
        if freeze_decision.get("freeze_manifest_path") is not None:
            raise ValueError("no-freeze audit report unexpectedly has a freeze manifest")
    if report.get("protocol", {}).get("final_test_relationships") != "not_read":
        raise PermissionError("Haller-v2 audit report does not prove final-test isolation")
    if report.get("status") == "pass":
        manifest_path = _verify_relative_path(
            output_root,
            str(report["freeze_decision"]["freeze_manifest_path"]),
        )
        manifest = _verify_freeze_manifest(manifest_path)
        if manifest.get("version") != report.get("version"):
            raise ValueError("freeze manifest/report version mismatch")
        if manifest.get("audit_report_sha256") != _file_sha256(report_path):
            raise ValueError("freeze manifest/report fingerprint mismatch")
        for run_id, artifact_path in manifest.get("artifact_paths", {}).items():
            entry_fingerprint = manifest.get("artifact_fingerprints", {}).get(run_id)
            artifact = load_haller_v2_artifact(
                _verify_relative_path(output_root, str(artifact_path)),
                require_trusted=True,
                freeze_manifest=manifest_path,
            )
            if artifact["metadata"].get("integrity_fingerprint") != entry_fingerprint:
                raise ValueError("freeze manifest/artifact fingerprint mismatch")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit fixed NumbaCS raw contours and freeze Haller-v2 post-processing"
    )
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--controlled-root", default=None)
    args = parser.parse_args(argv)
    report = audit_haller_v2_postprocessing(
        args.candidate_dir,
        args.out_dir,
        controlled_root=args.controlled_root,
    )
    print(json.dumps(_report_jsonable(report), ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "BAND_FACTORS",
    "CONTROLLED_CORE_IOU_MINIMUM",
    "CONTROLLED_RADIUS_RELATIVE_ERROR_MAXIMUM",
    "EQUIVALENT_RADIUS_CV_MAXIMUM",
    "FULLY_ERASED_MAXIMUM",
    "GEOMETRY_POLICIES",
    "HALLER_V2_VERSION",
    "PHYSICAL_REGION_ARTIFACT_VERSION",
    "MATCHED_CORE_IOU_MINIMUM",
    "NEGATIVE_POLICIES",
    "POSITIVE_RAW_INTERIOR_MINIMUM",
    "POSTPROCESS_SCHEMA_VERSION",
    "REMEDIATION_PROTOCOL_VERSION",
    "PostprocessingConfig",
    "RASTERIZATION_POLICIES",
    "RELATIVE_KNOWN_COVERAGE_CHANGE_MAXIMUM",
    "SELECTED_CENTER_RETENTION_MINIMUM",
    "complete_configuration_grid",
    "complete_geometry_configuration_grid",
    "audit_haller_v2_postprocessing",
    "load_haller_v2_artifact",
    "load_raw_contour_candidate",
    "main",
    "postprocess_raw_run",
    "read_haller_v2_audit_report",
    "require_trusted_haller_v2",
]


if __name__ == "__main__":
    raise SystemExit(main())
