"""Ticket 03 standard-IVD, P95/P80 target, and calibration artifacts.

This module is intentionally a new, split-safe artifact boundary.  It reads
only the raw velocity fields, geometry mask, and protocol-allowed metadata from
the existing dataset loader.  It never calls the historical weak-supervision
``prepare_all`` entry point and never uses ``ivd.npy`` or ``label_field.npy`` as
an input source.

The numerical IVD implementation is *not* duplicated here.  Every computed
frame is delegated to :mod:`standard_ivd`, which is the Ticket 02 canonical
implementation.  This module owns only the artifact orchestration, train-only
percentile statistics, target-state semantics, connected-component
post-processing, and audit/provenance records.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import experiment_protocol as protocol
import geometry
import standard_ivd
from prepare_weak_supervision_artifacts import _load_legacy_dataset


IMPLEMENTATION_VERSION = "ticket03-standard-ivd-p95p80-v1"
TARGET_SCHEMA_VERSION = "criterion-ambiguity-ticket03-target-v1"
CALIBRATION_SCHEMA_VERSION = "criterion-ambiguity-ticket03-calibration-reference-v1"
AUDIT_SCHEMA_VERSION = "criterion-ambiguity-ticket03-audit-v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "criterion-ambiguity-ticket03-artifact-manifest-v1"

# The values are deliberately distinct from the historical weak-label values.
# ``UNKNOWN`` includes positive evidence removed by the independent area filter;
# it is never silently turned into a dense negative.
NEGATIVE = 0
POSITIVE = 1
UNKNOWN = -1
SOLID = -2
INVALID = -3

DEFAULT_MIN_AREA_CELLS = 25
DEFAULT_CONNECTIVITY = 8

_FILTER_ORDER = [
    "threshold_valid_fluid_ivd",
    "connected_component_filter_positive_only",
    "assign_negative_unknown_solid_invalid_states",
]


class Ticket03BuildError(RuntimeError):
    """Raised when a Ticket 03 artifact cannot be completed safely."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_fingerprint(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert numpy scalar/array values without weakening identity fields."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _as_str_path(path: pathlib.Path) -> str:
    return str(path.resolve())


def _write_json_exclusive(path: pathlib.Path, value: Mapping[str, Any]) -> pathlib.Path:
    if path.exists():
        raise FileExistsError(f"refuse to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(_json_safe(value)) + b"\n")
    return path


def _write_npy_exclusive(path: pathlib.Path, value: Any) -> pathlib.Path:
    if path.exists():
        raise FileExistsError(f"refuse to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value), allow_pickle=False)
    return path


def _open_memmap_exclusive(
    path: pathlib.Path,
    *,
    dtype: np.dtype[Any] | type,
    shape: tuple[int, ...],
) -> np.memmap:
    if path.exists():
        raise FileExistsError(f"refuse to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def _strict_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} 必须是正整数")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return result


def _normalise_frame_mask(
    value: Any,
    *,
    frame_shape: tuple[int, int],
    frame_count: int,
    name: str,
) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.shape == frame_shape:
        return np.broadcast_to(mask, (frame_count, *frame_shape)).copy()
    if mask.shape == (frame_count, *frame_shape):
        return mask.copy()
    raise ValueError(
        f"{name} shape={mask.shape} 与 expected={frame_shape} 或 "
        f"{(frame_count, *frame_shape)} 不一致"
    )


def _normalise_single_mask(value: Any, *, shape: tuple[int, int], name: str) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} shape={mask.shape} 与 expected={shape} 不一致")
    return mask.copy()


def _validate_source_identity(identity: Mapping[str, Any]) -> None:
    if dict(identity) != dict(standard_ivd.SOURCE_IDENTITY):
        raise Ticket03BuildError(
            "standard IVD source identity drifted from standard_ivd.SOURCE_IDENTITY"
        )


@dataclass(frozen=True)
class P95P80Thresholds:
    """Frozen train-only thresholds for one dataset."""

    negative: float
    positive: float
    dataset_name: str
    train_indices: tuple[int, ...]
    train_frame_indices: tuple[int, ...]
    valid_fluid_count: int
    standard_ivd_source_identity: Mapping[str, Any]
    source_id: str = protocol.P95P80_PERCENTILE_SOURCE_ID
    negative_percentile: float = 80.0
    positive_percentile: float = 95.0
    percentile_method: str = "linear"

    def __post_init__(self) -> None:
        if self.dataset_name not in protocol.DATASET_NAMES:
            raise ValueError(f"dataset_name 不在固定六数据集内：{self.dataset_name!r}")
        if self.source_id != protocol.P95P80_PERCENTILE_SOURCE_ID:
            raise ValueError("P95/P80 threshold source_id 必须是当前 percentile source")
        if not np.isfinite(self.negative) or not np.isfinite(self.positive):
            raise ValueError("P95/P80 threshold 必须有限")
        if not self.negative < self.positive:
            raise ValueError(
                f"P80 必须严格小于 P95，实际={self.negative}/{self.positive}"
            )
        if len(self.train_indices) == 0 or len(self.train_indices) != len(
            self.train_frame_indices
        ):
            raise ValueError("train indices 不能为空且必须与实际 frame indices 对齐")
        if self.valid_fluid_count <= 0:
            raise ValueError("train valid-fluid count 必须为正")
        _validate_source_identity(self.standard_ivd_source_identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "p80": float(self.negative),
            "p95": float(self.positive),
            "negative": float(self.negative),
            "positive": float(self.positive),
            "negative_percentile": float(self.negative_percentile),
            "positive_percentile": float(self.positive_percentile),
            "percentile_method": self.percentile_method,
            "dataset_name": self.dataset_name,
            "train_indices": [int(value) for value in self.train_indices],
            "train_frame_indices": [int(value) for value in self.train_frame_indices],
            "train_frame_count": len(self.train_frame_indices),
            "valid_fluid_count": int(self.valid_fluid_count),
            "source_id": self.source_id,
            "artifact_identity": protocol.build_p95p80_artifact_identity(self.source_id),
            "standard_ivd_source_identity": _json_safe(
                self.standard_ivd_source_identity
            ),
        }


def compute_p95p80_thresholds(
    ivd: Any,
    *,
    valid_fluid_mask: Any,
    solid_mask: Any,
    invalid_mask: Any,
    train_indices: Sequence[int],
    dataset_name: str,
    standard_ivd_source_identity: Mapping[str, Any],
    train_frame_indices: Sequence[int] | None = None,
) -> P95P80Thresholds:
    """Compute P80/P95 exclusively from the supplied train frame positions.

    The function accepts frame *positions* so callers can pass a materialized
    allowed-frame artifact whose rows do not contain sealed test frames.  The
    optional ``train_frame_indices`` are the corresponding absolute frame
    numbers recorded in provenance; they do not cause another source read.
    """
    ivd_value = np.asarray(ivd, dtype=np.float64)
    if ivd_value.ndim != 3:
        raise ValueError(f"ivd 必须是 (T,Y,X)，实际 shape={ivd_value.shape}")
    frame_count, height, width = (int(size) for size in ivd_value.shape)
    frame_shape = (height, width)
    valid = _normalise_frame_mask(
        valid_fluid_mask,
        frame_shape=frame_shape,
        frame_count=frame_count,
        name="valid_fluid_mask",
    )
    solid = _normalise_frame_mask(
        solid_mask,
        frame_shape=frame_shape,
        frame_count=frame_count,
        name="solid_mask",
    )
    invalid = _normalise_frame_mask(
        invalid_mask,
        frame_shape=frame_shape,
        frame_count=frame_count,
        name="invalid_mask",
    )
    indices = tuple(int(index) for index in train_indices)
    if not indices:
        raise ValueError("train_indices 不能为空")
    if len(set(indices)) != len(indices) or any(
        index < 0 or index >= frame_count for index in indices
    ):
        raise ValueError("train_indices 必须是 materialized IVD 内不重复的 frame positions")
    actual_frames = (
        tuple(int(index) for index in train_frame_indices)
        if train_frame_indices is not None
        else indices
    )
    if len(actual_frames) != len(indices):
        raise ValueError("train_frame_indices 必须与 train_indices 等长")
    _validate_source_identity(standard_ivd_source_identity)

    values: list[np.ndarray] = []
    for position in indices:
        support = valid[position] & ~solid[position] & ~invalid[position]
        frame_values = np.asarray(ivd_value[position][support], dtype=np.float64)
        if frame_values.size == 0:
            raise ValueError(
                f"dataset={dataset_name} train frame position={position} 没有 valid-fluid support"
            )
        if not np.all(np.isfinite(frame_values)):
            raise ValueError(
                f"dataset={dataset_name} train IVD support 含非有限值，拒绝统计 threshold"
            )
        values.append(frame_values)
    train_values = np.concatenate(values)
    p80 = float(np.percentile(train_values, 80.0, method="linear"))
    p95 = float(np.percentile(train_values, 95.0, method="linear"))
    return P95P80Thresholds(
        negative=p80,
        positive=p95,
        dataset_name=dataset_name,
        train_indices=indices,
        train_frame_indices=actual_frames,
        valid_fluid_count=int(train_values.size),
        standard_ivd_source_identity=copy.deepcopy(dict(standard_ivd_source_identity)),
    )


def filter_positive_components(
    positive_mask: Any,
    *,
    min_area: int = DEFAULT_MIN_AREA_CELLS,
    connectivity: int = DEFAULT_CONNECTIVITY,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Filter positive components as an explicit post-processing operation.

    This function does not know about IVD, thresholds, negative labels, or
    geometry semantics.  Keeping that seam independent makes it impossible for
    the area threshold to become part of the canonical IVD definition.
    """
    raw = np.asarray(positive_mask, dtype=bool)
    if raw.ndim != 2:
        raise ValueError(f"positive_mask 必须是二维，实际 shape={raw.shape}")
    area = _strict_positive_int(min_area, name="min_area")
    if connectivity not in (4, 8):
        raise ValueError("connectivity 只能是 4 或 8")
    labels, component_count = geometry.label_components(raw, connectivity=connectivity)
    sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    if sizes.size:
        sizes[0] = 0
    keep = sizes >= area
    filtered = keep[labels]
    raw_cells = int(np.count_nonzero(raw))
    kept_cells = int(np.count_nonzero(filtered))
    kept_components = int(np.count_nonzero(keep[1:])) if keep.size > 1 else 0
    audit = {
        "min_area_cells": area,
        "connectivity": int(connectivity),
        "raw_component_count": int(component_count),
        "kept_component_count": kept_components,
        "filtered_component_count": int(component_count - kept_components),
        "raw_positive_cells": raw_cells,
        "post_filter_positive_cells": kept_cells,
        "filtered_positive_cells": int(raw_cells - kept_cells),
        "part_of_ivd_definition": False,
        "applies_to": "positive_components_only",
        "does_not_change_negative_rule": True,
        "operation_order": list(_FILTER_ORDER),
    }
    return filtered.astype(bool, copy=False), audit


def _class_coverage(
    count: int,
    *,
    total_cells: int,
    valid_fluid_cells: int,
    valid_fluid_class: bool,
) -> dict[str, Any]:
    return {
        "count": int(count),
        "grid_fraction": float(count / total_cells) if total_cells else 0.0,
        "valid_fluid_fraction": (
            float(count / valid_fluid_cells)
            if valid_fluid_class and valid_fluid_cells
            else None
        ),
        "grid_denominator_cells": int(total_cells),
        "valid_fluid_denominator_cells": (
            int(valid_fluid_cells) if valid_fluid_class else None
        ),
    }


@dataclass(frozen=True)
class P95P80FrameClassification:
    """Mutually exclusive state/mask view for one materialized frame."""

    state: np.ndarray
    labels: np.ndarray
    label_mask: np.ndarray
    positive_mask: np.ndarray
    negative_mask: np.ndarray
    unknown_mask: np.ndarray
    solid_mask: np.ndarray
    valid_fluid_mask: np.ndarray
    invalid_mask: np.ndarray
    counts: Mapping[str, int]
    coverage: Mapping[str, Mapping[str, Any]]
    filter_audit: Mapping[str, Any]


def classify_p95p80_frame(
    ivd: Any,
    *,
    solid_mask: Any,
    valid_fluid_mask: Any,
    invalid_mask: Any,
    thresholds: P95P80Thresholds,
    min_area: int = DEFAULT_MIN_AREA_CELLS,
    connectivity: int = DEFAULT_CONNECTIVITY,
) -> P95P80FrameClassification:
    """Classify one frame while preserving unknown/solid/invalid semantics."""
    ivd_value = np.asarray(ivd, dtype=np.float64)
    if ivd_value.ndim != 2:
        raise ValueError(f"ivd frame 必须是二维，实际 shape={ivd_value.shape}")
    shape = tuple(int(size) for size in ivd_value.shape)
    solid = _normalise_single_mask(solid_mask, shape=shape, name="solid_mask")
    valid = _normalise_single_mask(
        valid_fluid_mask, shape=shape, name="valid_fluid_mask"
    )
    invalid = _normalise_single_mask(invalid_mask, shape=shape, name="invalid_mask")
    if np.any(solid & valid) or np.any(solid & invalid) or np.any(valid & invalid):
        raise ValueError("solid/valid-fluid/invalid mask 必须互斥")
    if not np.all(solid | valid | invalid):
        raise ValueError("solid/valid-fluid/invalid mask 必须覆盖每个网格 cell")
    if np.any(valid & ~np.isfinite(ivd_value)):
        raise ValueError("valid-fluid IVD 含非有限值；应先由 canonical source 标记 invalid")

    raw_positive = valid & (ivd_value >= thresholds.positive)
    raw_negative = valid & (ivd_value <= thresholds.negative)
    raw_unknown = valid & ~raw_positive & ~raw_negative
    positive, filter_audit = filter_positive_components(
        raw_positive,
        min_area=min_area,
        connectivity=connectivity,
    )
    filtered_out_positive = raw_positive & ~positive
    unknown = raw_unknown | filtered_out_positive
    negative = raw_negative

    # State assignment is intentionally explicit.  In particular, a filtered
    # positive becomes UNKNOWN, never NEGATIVE; SOLID and INVALID are separate
    # from both dense target labels and Haller sparse evidence.
    state = np.full(shape, UNKNOWN, dtype=np.int8)
    state[negative] = NEGATIVE
    state[unknown] = UNKNOWN
    state[positive] = POSITIVE
    state[solid] = SOLID
    state[invalid] = INVALID

    partition = np.stack([positive, negative, unknown, solid, invalid], axis=0)
    if not np.all(np.sum(partition, axis=0) == 1):
        raise ValueError("P95/P80 state masks 不是互斥且完备的 partition")

    labels = np.zeros(shape, dtype=np.float32)
    labels[positive] = 1.0
    label_mask = (positive | negative).astype(np.uint8)
    total_cells = int(np.prod(shape))
    valid_count = int(np.count_nonzero(valid))
    masks = {
        "positive": positive,
        "negative": negative,
        "unknown": unknown,
        "solid": solid,
        "valid_fluid": valid,
        "invalid": invalid,
    }
    counts = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
    coverage = {
        name: _class_coverage(
            count,
            total_cells=total_cells,
            valid_fluid_cells=valid_count,
            valid_fluid_class=name in {"positive", "negative", "unknown", "valid_fluid"},
        )
        for name, count in counts.items()
    }
    filter_audit = dict(filter_audit)
    filter_audit.update(
        {
            "raw_positive_cells": int(np.count_nonzero(raw_positive)),
            "post_filter_positive_cells": int(np.count_nonzero(positive)),
            "filtered_positive_cells": int(np.count_nonzero(filtered_out_positive)),
            "raw_positive_grid_fraction": float(
                np.count_nonzero(raw_positive) / total_cells
            ),
            "post_filter_positive_grid_fraction": float(
                np.count_nonzero(positive) / total_cells
            ),
            "raw_positive_valid_fluid_fraction": (
                float(np.count_nonzero(raw_positive) / valid_count) if valid_count else None
            ),
            "post_filter_positive_valid_fluid_fraction": (
                float(np.count_nonzero(positive) / valid_count) if valid_count else None
            ),
        }
    )
    return P95P80FrameClassification(
        state=state,
        labels=labels,
        label_mask=label_mask,
        positive_mask=positive.astype(bool, copy=False),
        negative_mask=negative.astype(bool, copy=False),
        unknown_mask=unknown.astype(bool, copy=False),
        solid_mask=solid,
        valid_fluid_mask=valid,
        invalid_mask=invalid,
        counts=counts,
        coverage=coverage,
        filter_audit=filter_audit,
    )


def load_raw_dataset_source(input_root: str | pathlib.Path, name: str) -> dict[str, Any]:
    """Load the existing raw source seam without loading historical labels/IVD."""
    # The loader's implementation is shared with the existing preparation
    # pipeline, but its returned inputs are restricted here to u/v/mask/meta.
    source = _load_legacy_dataset(pathlib.Path(input_root), name)
    required = ("u", "v", "mask", "xdim", "ydim", "tdim", "meta", "root")
    missing = [key for key in required if key not in source]
    if missing:
        raise Ticket03BuildError(f"raw loader returned incomplete source: {missing!r}")
    return source


def _ranges_to_frames(ranges: Sequence[Sequence[int]]) -> tuple[int, ...]:
    frames: list[int] = []
    for pair in ranges:
        if len(pair) != 2:
            raise Ticket03BuildError(f"非法 half-open range：{pair!r}")
        start, end = int(pair[0]), int(pair[1])
        if start < 0 or end <= start:
            raise Ticket03BuildError(f"非法 half-open range：{pair!r}")
        frames.extend(range(start, end))
    return tuple(frames)


def _partition_frames(record: Mapping[str, Any], split_name: str) -> tuple[int, ...]:
    partitions = record["frame_partitions"]
    return _ranges_to_frames(partitions[split_name]["ranges"])


def _partition_window_audit(record: Mapping[str, Any]) -> dict[str, Any]:
    window = record["window"]
    t_win = int(window["t_win"])
    split_audit: dict[str, Any] = {}
    cross_split = 0
    for split_name in protocol.ALL_PARTITION_NAMES:
        partition = record["frame_partitions"][split_name]
        ranges = [tuple(int(value) for value in pair) for pair in partition["ranges"]]
        starts = [int(value) for value in partition["window_starts"]]
        invalid_windows = 0
        for start in starts:
            end = start + t_win
            if not any(range_start <= start and end <= range_end for range_start, range_end in ranges):
                invalid_windows += 1
                cross_split += 1
        split_audit[split_name] = {
            "window_count": len(starts),
            "window_starts": starts,
            "ranges": [list(pair) for pair in ranges],
            "cross_boundary_window_count": invalid_windows,
        }
    bouss_audit_frames = set(int(value) for value in record["development_audit_frames"])
    test_starts = split_audit["test"]["window_starts"]
    audit_in_test = sum(
        1
        for start in test_starts
        if any(start <= frame < start + t_win for frame in bouss_audit_frames)
    )
    return {
        "t_win": t_win,
        "window_step": int(window["window_step"]),
        "splits": split_audit,
        "cross_split_window_count": int(cross_split),
        "audit_frame_in_test_window_count": int(audit_in_test),
    }


def _coverage_from_counts(
    counts: Mapping[str, int], *, frame_count: int, frame_shape: tuple[int, int]
) -> dict[str, Any]:
    total_cells = int(frame_count * frame_shape[0] * frame_shape[1])
    valid_count = int(counts["valid_fluid"])
    return {
        name: _class_coverage(
            int(counts[name]),
            total_cells=total_cells,
            valid_fluid_cells=valid_count,
            valid_fluid_class=name in {"positive", "negative", "unknown", "valid_fluid"},
        )
        for name in ("positive", "negative", "unknown", "solid", "valid_fluid", "invalid")
    }


def _add_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for name in ("positive", "negative", "unknown", "solid", "valid_fluid", "invalid"):
        target[name] = int(target.get(name, 0) + int(source[name]))


def _build_frame_provenance(
    *,
    dataset_name: str,
    split_name: str,
    frame_index: int,
    protocol_manifest: Mapping[str, Any],
    geometry_fingerprint: str,
    input_fingerprint: str,
    valid_fluid_fingerprint: str,
    source_id: str,
) -> dict[str, Any]:
    """Build the public protocol provenance shape without revalidating per frame."""
    if source_id not in protocol.P95P80_ALLOWED_SOURCE_IDS:
        raise Ticket03BuildError(f"unsupported P95/P80 source_id={source_id!r}")
    result: dict[str, Any] = {
        "schema_version": protocol.P95P80_PROVENANCE_SCHEMA_VERSION,
        "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
        "protocol_manifest_hash": str(protocol_manifest["manifest_hash"]),
        "source_id": source_id,
        "artifact_id": protocol.build_p95p80_artifact_identity(source_id)["artifact_id"],
        "dataset_name": dataset_name,
        "split_name": split_name,
        "frame_index": int(frame_index),
        "geometry_fingerprint": str(geometry_fingerprint),
        "input_fingerprint": str(input_fingerprint),
        "valid_fluid_fingerprint": str(valid_fluid_fingerprint),
        "final_test_relationships": "not_read",
    }
    result["integrity_fingerprint"] = _json_hash(result)
    return result


def _preflight_sources(
    input_root: pathlib.Path,
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load and validate only raw source arrays before creating output files."""
    sources: dict[str, dict[str, Any]] = {}
    for name in protocol.DATASET_NAMES:
        source = load_raw_dataset_source(input_root, name)
        shape = tuple(int(size) for size in source["shape"])
        u = source["u"]
        v = source["v"]
        mask = np.asarray(source["mask"], dtype=bool)
        expected_record = manifest["datasets"][name]
        expected_shape = (int(expected_record["total_frames"]), len(source["ydim"]), len(source["xdim"]))
        if shape != expected_shape or tuple(u.shape) != shape or tuple(v.shape) != shape:
            raise Ticket03BuildError(
                f"dataset={name} loader shape mismatch: {shape}/{u.shape}/{v.shape}/{expected_shape}"
            )
        if mask.shape != shape[1:]:
            raise Ticket03BuildError(
                f"dataset={name} geometry mask shape={mask.shape} != {shape[1:]}"
            )
        xdim = np.asarray(source["xdim"], dtype=np.float64)
        ydim = np.asarray(source["ydim"], dtype=np.float64)
        tdim = np.asarray(source["tdim"], dtype=np.float64)
        if len(tdim) != shape[0] or not np.all(np.isfinite(tdim)):
            raise Ticket03BuildError(f"dataset={name} tdim shape/finite check failed")
        if len(xdim) != shape[2] or len(ydim) != shape[1]:
            raise Ticket03BuildError(f"dataset={name} coordinate lengths do not match shape")
        if len(xdim) < 2 or len(ydim) < 2:
            raise Ticket03BuildError(f"dataset={name} requires at least two grid points per axis")
        dxs = np.diff(xdim)
        dys = np.diff(ydim)
        if (
            not np.all(np.isfinite(xdim))
            or not np.all(np.isfinite(ydim))
            or np.any(dxs <= 0.0)
            or np.any(dys <= 0.0)
            or not np.allclose(dxs, dxs[0], rtol=1e-3, atol=1e-12)
            or not np.allclose(dys, dys[0], rtol=1e-3, atol=1e-12)
        ):
            raise Ticket03BuildError(f"dataset={name} coordinates are not valid uniform axes")
        partition_audit = _partition_window_audit(expected_record)
        if partition_audit["cross_split_window_count"] != 0:
            raise Ticket03BuildError(
                f"dataset={name} pathline windows cross split boundary: {partition_audit!r}"
            )
        if name == "boussinesq":
            audit_frame = protocol.BOUSSINESQ_AUDIT_FRAME
            test_frames = set(_partition_frames(expected_record, "test"))
            if audit_frame in test_frames or partition_audit["audit_frame_in_test_window_count"] != 0:
                raise Ticket03BuildError(
                    "boussinesq frame 1300 entered test manifest or test window"
                )
        sources[name] = {
            **source,
            "shape": shape,
            "mask": mask,
            "xdim": xdim,
            "ydim": ydim,
            "tdim": tdim,
            "dx": float(dxs[0]),
            "dy": float(dys[0]),
            "partition_audit": partition_audit,
            "source_read_scope": (
                "meta,u,v,geometry_mask; values only from train/calibration/"
                "development_audit frames"
            ),
            "test_values_read": False,
            "test_frame_indices_read": [],
        }
    return sources


def _artifact_instance_identity(
    *,
    base_identity: Mapping[str, Any],
    dataset_name: str,
    artifact_kind: str,
    protocol_manifest_hash: str,
    frame_indices: Sequence[int],
    input_identities: Sequence[str],
    geometry_fingerprint: str,
) -> dict[str, Any]:
    payload = {
        "base_artifact_identity": _json_safe(base_identity),
        "dataset_name": dataset_name,
        "artifact_kind": artifact_kind,
        "protocol_manifest_hash": protocol_manifest_hash,
        "frame_indices": [int(value) for value in frame_indices],
        "input_identities": [str(value) for value in input_identities],
        "geometry_fingerprint": str(geometry_fingerprint),
        "implementation_version": IMPLEMENTATION_VERSION,
    }
    return {
        "artifact_instance_id": (
            f"{base_identity['artifact_id']}:dataset:{dataset_name}:{artifact_kind}:v1"
        ),
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        **payload,
        "identity_fingerprint": _json_hash(payload),
    }


def _flush_memmap(value: np.memmap) -> None:
    value.flush()


def _materialize_target_split(
    *,
    split_name: str,
    dataset_name: str,
    rows: Sequence[int],
    actual_frames: Sequence[int],
    standard_ivd_mm: np.memmap,
    valid_mm: np.memmap,
    invalid_mm: np.memmap,
    solid_mask: np.ndarray,
    thresholds: P95P80Thresholds,
    standard_meta: Mapping[str, Any],
    output_dir: pathlib.Path,
    input_identities: Sequence[str],
    geometry_fingerprint: str,
    protocol_manifest: Mapping[str, Any],
    min_area: int,
    connectivity: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if split_name not in {"train", "calibration"}:
        raise ValueError("Ticket 03 target materialization 只接受 train/calibration")
    if len(rows) != len(actual_frames) or len(rows) == 0:
        raise ValueError("target split rows/frame indices 必须非空且等长")
    height, width = (int(size) for size in standard_ivd_mm.shape[1:])
    shape = (len(rows), height, width)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_mm = _open_memmap_exclusive(output_dir / "state.npy", dtype=np.int8, shape=shape)
    labels_mm = _open_memmap_exclusive(
        output_dir / "labels.npy", dtype=np.float32, shape=shape
    )
    label_mask_mm = _open_memmap_exclusive(
        output_dir / "label_mask.npy", dtype=np.uint8, shape=shape
    )
    positive_mm = _open_memmap_exclusive(
        output_dir / "positive_mask.npy", dtype=np.uint8, shape=shape
    )
    negative_mm = _open_memmap_exclusive(
        output_dir / "negative_mask.npy", dtype=np.uint8, shape=shape
    )
    unknown_mm = _open_memmap_exclusive(
        output_dir / "unknown_mask.npy", dtype=np.uint8, shape=shape
    )
    valid_split_mm = _open_memmap_exclusive(
        output_dir / "valid_fluid_mask.npy", dtype=np.uint8, shape=shape
    )
    invalid_split_mm = _open_memmap_exclusive(
        output_dir / "invalid_mask.npy", dtype=np.uint8, shape=shape
    )
    _write_npy_exclusive(output_dir / "solid_mask.npy", solid_mask.astype(np.uint8))
    _write_npy_exclusive(output_dir / "frame_indices.npy", np.asarray(actual_frames, dtype=np.int64))
    _write_npy_exclusive(output_dir / "standard_ivd_row_indices.npy", np.asarray(rows, dtype=np.int64))

    counts = {name: 0 for name in ("positive", "negative", "unknown", "solid", "valid_fluid", "invalid")}
    frame_audit: list[dict[str, Any]] = []
    for output_row, (standard_row, frame_index) in enumerate(zip(rows, actual_frames)):
        result = classify_p95p80_frame(
            standard_ivd_mm[int(standard_row)],
            solid_mask=solid_mask,
            valid_fluid_mask=np.asarray(valid_mm[int(standard_row)], dtype=bool),
            invalid_mask=np.asarray(invalid_mm[int(standard_row)], dtype=bool),
            thresholds=thresholds,
            min_area=min_area,
            connectivity=connectivity,
        )
        state_mm[output_row] = result.state
        labels_mm[output_row] = result.labels
        label_mask_mm[output_row] = result.label_mask
        positive_mm[output_row] = result.positive_mask
        negative_mm[output_row] = result.negative_mask
        unknown_mm[output_row] = result.unknown_mask
        valid_split_mm[output_row] = result.valid_fluid_mask
        invalid_split_mm[output_row] = result.invalid_mask
        _add_counts(counts, result.counts)
        frame_audit.append(
            {
                "frame_index": int(frame_index),
                "standard_ivd_row_index": int(standard_row),
                "counts": _json_safe(dict(result.counts)),
                "coverage": _json_safe(dict(result.coverage)),
                "component_filter": _json_safe(dict(result.filter_audit)),
            }
        )
    for value in (
        state_mm,
        labels_mm,
        label_mask_mm,
        positive_mm,
        negative_mm,
        unknown_mm,
        valid_split_mm,
        invalid_split_mm,
    ):
        _flush_memmap(value)

    data_files = {
        path.name: {
            "path": _as_str_path(path),
            "sha256": _file_sha256(path),
        }
        for path in sorted(output_dir.glob("*.npy"))
    }
    base_identity = protocol.build_p95p80_artifact_identity(
        protocol.P95P80_PERCENTILE_SOURCE_ID
    )
    instance_identity = _artifact_instance_identity(
        base_identity=base_identity,
        dataset_name=dataset_name,
        artifact_kind="train_target" if split_name == "train" else "calibration_reference",
        protocol_manifest_hash=str(protocol_manifest["manifest_hash"]),
        frame_indices=actual_frames,
        input_identities=input_identities,
        geometry_fingerprint=geometry_fingerprint,
    )
    threshold_provenance = {
        "source_id": protocol.P95P80_PERCENTILE_SOURCE_ID,
        "artifact_identity": base_identity,
        "statistics_split": "train",
        "thresholds_frozen_before_consumer_materialization": True,
        "threshold_values": thresholds.to_dict(),
        "test_source_read": False,
        "test_frame_indices_read": [],
        "calibration_values_used_for_threshold": False,
        "development_audit_values_used_for_threshold": False,
        "standard_ivd_source_identity": _json_safe(standard_ivd.SOURCE_IDENTITY),
        "standard_ivd_artifact_identity": standard_meta["artifact_identity"],
        "standard_ivd_artifact_instance_identity": standard_meta[
            "artifact_instance_identity"
        ],
        "standard_ivd_artifact_sha256": standard_meta["artifact_sha256"],
        "protocol_manifest_hash": protocol_manifest["manifest_hash"],
    }
    threshold_provenance["integrity_fingerprint"] = _json_hash(threshold_provenance)
    schema_version = (
        TARGET_SCHEMA_VERSION
        if split_name == "train"
        else CALIBRATION_SCHEMA_VERSION
    )
    metadata: dict[str, Any] = {
        "artifact_type": (
            "criterion_ambiguity_p95p80_train_target"
            if split_name == "train"
            else "criterion_ambiguity_p95p80_calibration_reference"
        ),
        "schema_version": schema_version,
        "implementation_version": IMPLEMENTATION_VERSION,
        "entry_point": "ticket03_standard_ivd_artifacts._materialize_target_split",
        "dataset_name": dataset_name,
        "split_name": split_name,
        "frame_indices": [int(value) for value in actual_frames],
        "standard_ivd_row_indices": [int(value) for value in rows],
        "source_id": protocol.P95P80_PERCENTILE_SOURCE_ID,
        "artifact_identity": base_identity,
        "artifact_instance_identity": instance_identity,
        "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
        "protocol_manifest_hash": protocol_manifest["manifest_hash"],
        "standard_ivd_source_identity": _json_safe(standard_ivd.SOURCE_IDENTITY),
        "haller_standard_ivd_source_identity": _json_safe(standard_ivd.SOURCE_IDENTITY),
        "standard_ivd_artifact": {
            "metadata_path": standard_meta["metadata_path"],
            "artifact_identity": standard_meta["artifact_identity"],
            "artifact_instance_identity": standard_meta["artifact_instance_identity"],
            "artifact_sha256": standard_meta["artifact_sha256"],
            "frame_index_mapping": standard_meta["frame_index_mapping"],
        },
        "thresholds": thresholds.to_dict(),
        "threshold_provenance": threshold_provenance,
        "label_semantics": {
            "positive": "IVD >= frozen train valid-fluid P95 after positive-only component filter",
            "negative": "IVD <= frozen train valid-fluid P80; component filter does not alter this rule",
            "unknown": "valid-fluid values between P80/P95 or raw positive components removed by post-processing",
            "solid": "geometry solid; excluded from mean and label support; not a negative",
            "invalid": "non-finite/outside valid-fluid support; excluded and counted",
            "haller_sparse_anchor": "sparse positive physics evidence only; not dense negative or dense GT",
        },
        "state_codes": {
            "positive": POSITIVE,
            "negative": NEGATIVE,
            "unknown": UNKNOWN,
            "solid": SOLID,
            "invalid": INVALID,
        },
        "connected_component_postprocessing": {
            "min_area_cells": int(min_area),
            "connectivity": int(connectivity),
            "applies_after_thresholding": True,
            "applies_to": "positive_components_only",
            "part_of_ivd_definition": False,
            "does_not_change_negative_rule": True,
            "operation_order": list(_FILTER_ORDER),
        },
        "mask_semantics": {
            "valid_fluid": "canonical standard IVD valid-fluid support",
            "solid": "geometry solid, excluded from omega_bar and label support",
            "invalid": "canonical standard IVD invalid/non-finite support, excluded and counted",
            "partition": "positive/negative/unknown/solid/invalid mutually exclusive and complete",
        },
        "input_identity": {
            "frame_input_identities": {
                str(frame): str(identity)
                for frame, identity in zip(actual_frames, input_identities)
            },
            "geometry_mask_identity": geometry_fingerprint,
            "read_scope": "train/calibration/development_audit only",
        },
        "final_test_relationships": "not_read",
        "test_source_read": False,
        "test_frame_indices_read": [],
        "legacy_ivd_read": False,
        "legacy_label_read": False,
        "legacy_artifacts_formal_source": False,
        "calibration_reference_recomputed_thresholds": False,
        "class_counts": counts,
        "coverage": _coverage_from_counts(
            counts, frame_count=len(actual_frames), frame_shape=(height, width)
        ),
        "data_files": data_files,
    }
    metadata["metadata_hash"] = _json_hash(metadata)
    metadata_path = _write_json_exclusive(output_dir / "target_meta.json", metadata)
    return {
        "metadata": _as_str_path(metadata_path),
        "frame_indices": _as_str_path(output_dir / "frame_indices.npy"),
        "standard_ivd_row_indices": _as_str_path(output_dir / "standard_ivd_row_indices.npy"),
        "directory": _as_str_path(output_dir),
        "metadata_value": metadata,
        "counts": counts,
        "frame_audit": frame_audit,
    }, frame_audit


def _build_standard_ivd_artifact(
    *,
    dataset_name: str,
    source: Mapping[str, Any],
    record: Mapping[str, Any],
    protocol_manifest: Mapping[str, Any],
    output_dir: pathlib.Path,
) -> tuple[dict[str, Any], np.memmap, np.memmap, np.memmap, list[dict[str, Any]]]:
    train_frames = _partition_frames(record, "train")
    calibration_frames = _partition_frames(record, "calibration")
    audit_frames = _partition_frames(record, protocol.AUDIT_SPLIT_NAME)
    allowed_frames = train_frames + calibration_frames + audit_frames
    if not allowed_frames:
        raise Ticket03BuildError(f"dataset={dataset_name} has no allowed frames")
    frame_to_row = {frame: row for row, frame in enumerate(allowed_frames)}
    shape = tuple(int(size) for size in source["shape"])
    height, width = shape[1:]
    standard_dir = output_dir / "standard_ivd"
    standard_dir.mkdir(parents=True, exist_ok=True)
    ivd_mm = _open_memmap_exclusive(
        standard_dir / "standard_ivd.npy",
        dtype=np.float64,
        shape=(len(allowed_frames), height, width),
    )
    valid_mm = _open_memmap_exclusive(
        standard_dir / "valid_fluid_mask.npy",
        dtype=np.uint8,
        shape=(len(allowed_frames), height, width),
    )
    invalid_mm = _open_memmap_exclusive(
        standard_dir / "invalid_mask.npy",
        dtype=np.uint8,
        shape=(len(allowed_frames), height, width),
    )
    _write_npy_exclusive(
        standard_dir / "solid_mask.npy", np.asarray(source["mask"], dtype=np.uint8)
    )
    _write_npy_exclusive(
        standard_dir / "frame_indices.npy", np.asarray(allowed_frames, dtype=np.int64)
    )
    geometry_fingerprint = _array_fingerprint(np.asarray(source["mask"], dtype=np.uint8))
    frame_records: list[dict[str, Any]] = []
    input_identities: list[str] = []
    for row, frame_index in enumerate(allowed_frames):
        if frame_index not in frame_to_row:
            raise Ticket03BuildError("internal frame mapping failure")
        # This is the only value access in the production builder.  The frame
        # is selected from train/calibration/development_audit above; test is
        # never indexed, hashed, or passed to standard_ivd.
        u_frame = np.asarray(source["u"][frame_index])
        v_frame = np.asarray(source["v"][frame_index])
        input_nonfinite = ~np.isfinite(u_frame) | ~np.isfinite(v_frame)
        split_name = (
            "train"
            if frame_index in train_frames
            else "calibration"
            if frame_index in calibration_frames
            else protocol.AUDIT_SPLIT_NAME
        )
        try:
            details = standard_ivd.compute_standard_ivd_details(
                u_frame,
                v_frame,
                source["xdim"],
                source["ydim"],
                solid_mask=source["mask"],
            )
        except Exception as exc:  # pragma: no cover - exercised by real-data audit on failure
            raise Ticket03BuildError(
                f"dataset={dataset_name} frame={frame_index} canonical standard IVD failed"
            ) from exc
        _validate_source_identity(details.metadata["source_identity"])
        if details.dx != source["dx"] or details.dy != source["dy"]:
            raise Ticket03BuildError(f"dataset={dataset_name} frame={frame_index} spacing drift")
        if not np.all(np.isfinite(details.ivd)):
            raise Ticket03BuildError(
                f"dataset={dataset_name} frame={frame_index} standard IVD output is non-finite"
            )
        if not np.array_equal(details.solid_mask, np.asarray(source["mask"], dtype=bool)):
            raise Ticket03BuildError(f"dataset={dataset_name} frame={frame_index} solid mask drift")
        ivd_mm[row] = details.ivd
        valid_mm[row] = details.valid_fluid_mask.astype(np.uint8)
        invalid_mm[row] = details.invalid_mask.astype(np.uint8)
        input_identity = str(details.metadata["input_identity"])
        input_identities.append(input_identity)
        valid_fingerprint = _array_fingerprint(details.valid_fluid_mask.astype(np.uint8))
        frame_records.append(
            {
                "frame_index": int(frame_index),
                "materialized_row_index": int(row),
                "split_name": split_name,
                "omega_bar": float(details.omega_bar),
                "grid_spacing": {"dx": float(details.dx), "dy": float(details.dy)},
                "input_identity": input_identity,
                "input_nonfinite_cell_count": int(np.count_nonzero(input_nonfinite)),
                "valid_fluid_mask_identity": valid_fingerprint,
                "valid_fluid_count": int(np.count_nonzero(details.valid_fluid_mask)),
                "invalid_count": int(np.count_nonzero(details.invalid_mask)),
                "standard_ivd_provenance": _json_safe(dict(details.metadata["provenance"])),
                "protocol_provenance": {
                    "standard_ivd": _build_frame_provenance(
                        dataset_name=dataset_name,
                        split_name=split_name,
                        frame_index=frame_index,
                        protocol_manifest=protocol_manifest,
                        geometry_fingerprint=geometry_fingerprint,
                        input_fingerprint=input_identity,
                        valid_fluid_fingerprint=valid_fingerprint,
                        source_id=protocol.P95P80_HALLER_SOURCE_ID,
                    ),
                    "percentile_target": _build_frame_provenance(
                        dataset_name=dataset_name,
                        split_name=split_name,
                        frame_index=frame_index,
                        protocol_manifest=protocol_manifest,
                        geometry_fingerprint=geometry_fingerprint,
                        input_fingerprint=input_identity,
                        valid_fluid_fingerprint=valid_fingerprint,
                        source_id=protocol.P95P80_PERCENTILE_SOURCE_ID,
                    ),
                },
            }
        )
    for value in (ivd_mm, valid_mm, invalid_mm):
        _flush_memmap(value)
    data_files = {
        path.name: {"path": _as_str_path(path), "sha256": _file_sha256(path)}
        for path in sorted(standard_dir.glob("*.npy"))
    }
    ivd_identity = protocol.build_p95p80_artifact_identity(
        protocol.P95P80_HALLER_SOURCE_ID
    )
    standard_instance_identity = _artifact_instance_identity(
        base_identity=ivd_identity,
        dataset_name=dataset_name,
        artifact_kind="standard_ivd",
        protocol_manifest_hash=str(protocol_manifest["manifest_hash"]),
        frame_indices=allowed_frames,
        input_identities=input_identities,
        geometry_fingerprint=geometry_fingerprint,
    )
    artifact_payload = {
        "artifact_type": "criterion_ambiguity_ticket03_standard_ivd",
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "entry_point": "ticket03_standard_ivd_artifacts._build_standard_ivd_artifact",
        "dataset_name": dataset_name,
        "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
        "protocol_manifest_hash": protocol_manifest["manifest_hash"],
        "artifact_identity": ivd_identity,
        "artifact_instance_identity": standard_instance_identity,
        "frame_indices": [int(value) for value in allowed_frames],
        "frame_index_mapping": {
            str(frame): int(row) for row, frame in enumerate(allowed_frames)
        },
        "computed_frame_count": len(allowed_frames),
        "train_frame_count": len(train_frames),
        "calibration_frame_count": len(calibration_frames),
        "development_audit_frame_count": len(audit_frames),
        "test_frame_count": int(record["frame_partitions"]["test"]["frame_count"]),
        "test_frame_indices_read": [],
        "test_values_read": False,
        "standard_ivd_source_identity": _json_safe(standard_ivd.SOURCE_IDENTITY),
        "formula": standard_ivd.FORMULA,
        "formula_version": standard_ivd.FORMULA_VERSION,
        "mean_scope": "declared_valid_fluid_support",
        "mean_quantity": "signed omega",
        "grid_spacing": {"dx": source["dx"], "dy": source["dy"]},
        "coordinates": {
            "xdim_identity": _array_fingerprint(source["xdim"]),
            "ydim_identity": _array_fingerprint(source["ydim"]),
            "coordinate_order": "x_then_y",
            "array_order": "row_y_then_col_x",
        },
        "input_arrays": {
            "u_path": _as_str_path(pathlib.Path(source["root"]) / "u.npy"),
            "v_path": _as_str_path(pathlib.Path(source["root"]) / "v.npy"),
            "frame_input_identities": {
                str(frame): identity
                for frame, identity in zip(allowed_frames, input_identities)
            },
            "read_scope": "train/calibration/development_audit only",
        },
        "geometry_mask_identity": geometry_fingerprint,
        "geometry_mask_path": _as_str_path(pathlib.Path(source["root"]) / "mask.npy"),
        "mask_semantics": {
            "valid_fluid": "declared valid-fluid and finite canonical omega, non-solid",
            "solid": "geometry solid; excluded from omega_bar and label support",
            "invalid": "non-finite omega/input or outside declared valid-fluid support; counted and excluded",
        },
        "omega_bar_by_frame": [record_item["omega_bar"] for record_item in frame_records],
        "frame_records": frame_records,
        "data_files": data_files,
        "legacy_ivd_read": False,
        "legacy_label_read": False,
        "legacy_artifacts_formal_source": False,
        "final_test_relationships": "not_read",
    }
    artifact_payload["artifact_sha256"] = _json_hash(artifact_payload)
    standard_meta_path = _write_json_exclusive(
        standard_dir / "standard_ivd_meta.json", artifact_payload
    )
    artifact_payload["metadata_path"] = _as_str_path(standard_meta_path)
    return artifact_payload, ivd_mm, valid_mm, invalid_mm, frame_records


def _split_summary(
    *,
    split_name: str,
    frame_indices: Sequence[int],
    window_info: Mapping[str, Any],
    counts: Mapping[str, int] | None,
    frame_shape: tuple[int, int],
    values_read: bool,
    frame_audit: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if values_read:
        assert counts is not None
        coverage: Mapping[str, Any] = {
            "status": "computed",
            "classes": _coverage_from_counts(
                counts, frame_count=len(frame_indices), frame_shape=frame_shape
            ),
        }
    else:
        coverage = {
            "status": "not_read",
            "classes": {},
            "denominator_cells": 0,
        }
    return {
        "split_name": split_name,
        "frame_count": len(frame_indices),
        "frame_ranges": _json_safe(window_info["ranges"]),
        "window_count": int(window_info["window_count"]),
        "window_starts": _json_safe(window_info["window_starts"]),
        "values_read": bool(values_read),
        "coverage": coverage,
        "frame_audit": _json_safe(list(frame_audit)),
    }


def _verify_no_test_frames(
    *,
    allowed_frames: Sequence[int],
    test_ranges: Sequence[Sequence[int]],
    artifact_name: str,
) -> None:
    test_frames = set(_ranges_to_frames(test_ranges))
    overlap = test_frames.intersection(int(value) for value in allowed_frames)
    if overlap:
        raise Ticket03BuildError(
            f"{artifact_name} unexpectedly contains sealed test frames: {sorted(overlap)[:5]}"
        )


def _build_dataset(
    *,
    dataset_name: str,
    source: Mapping[str, Any],
    record: Mapping[str, Any],
    protocol_manifest: Mapping[str, Any],
    output_root: pathlib.Path,
    min_area: int,
    connectivity: int,
) -> dict[str, Any]:
    dataset_output = output_root / "datasets" / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)
    standard_meta, ivd_mm, valid_mm, invalid_mm, standard_frame_records = (
        _build_standard_ivd_artifact(
            dataset_name=dataset_name,
            source=source,
            record=record,
            protocol_manifest=protocol_manifest,
            output_dir=dataset_output,
        )
    )
    allowed_frames = _ranges_to_frames(
        record["frame_partitions"]["train"]["ranges"]
    ) + _ranges_to_frames(record["frame_partitions"]["calibration"]["ranges"]) + _ranges_to_frames(
        record["frame_partitions"][protocol.AUDIT_SPLIT_NAME]["ranges"]
    )
    test_ranges = record["frame_partitions"]["test"]["ranges"]
    _verify_no_test_frames(
        allowed_frames=allowed_frames,
        test_ranges=test_ranges,
        artifact_name=f"{dataset_name}/standard_ivd",
    )
    frame_to_row = {int(value): index for index, value in enumerate(allowed_frames)}
    train_frames = _ranges_to_frames(record["frame_partitions"]["train"]["ranges"])
    calibration_frames = _ranges_to_frames(
        record["frame_partitions"]["calibration"]["ranges"]
    )
    audit_frames = _ranges_to_frames(
        record["frame_partitions"][protocol.AUDIT_SPLIT_NAME]["ranges"]
    )
    input_identities = [item["input_identity"] for item in standard_frame_records]
    train_rows = tuple(frame_to_row[frame] for frame in train_frames)
    thresholds = compute_p95p80_thresholds(
        ivd_mm,
        valid_fluid_mask=valid_mm,
        solid_mask=np.asarray(source["mask"], dtype=bool),
        invalid_mask=invalid_mm,
        train_indices=train_rows,
        train_frame_indices=train_frames,
        dataset_name=dataset_name,
        standard_ivd_source_identity=standard_ivd.SOURCE_IDENTITY,
    )
    threshold_provenance = {
        "source_id": protocol.P95P80_PERCENTILE_SOURCE_ID,
        "artifact_identity": protocol.build_p95p80_artifact_identity(
            protocol.P95P80_PERCENTILE_SOURCE_ID
        ),
        "statistics_split": "train",
        "percentiles": {"p80": 80.0, "p95": 95.0},
        "method": "numpy.percentile(method='linear')",
        "threshold_values": thresholds.to_dict(),
        "train_frame_ranges": _json_safe(record["frame_partitions"]["train"]["ranges"]),
        "train_frame_indices": list(train_frames),
        "train_valid_fluid_count": thresholds.valid_fluid_count,
        "calibration_values_used_for_threshold": False,
        "development_audit_values_used_for_threshold": False,
        "test_source_read": False,
        "test_frame_indices_read": [],
        "standard_ivd_source_identity": _json_safe(standard_ivd.SOURCE_IDENTITY),
        "standard_ivd_artifact_identity": standard_meta["artifact_identity"],
        "standard_ivd_artifact_instance_identity": standard_meta[
            "artifact_instance_identity"
        ],
        "standard_ivd_artifact_sha256": standard_meta["artifact_sha256"],
        "protocol_manifest_hash": protocol_manifest["manifest_hash"],
        "input_identities": list(input_identities),
    }
    threshold_provenance["integrity_fingerprint"] = _json_hash(threshold_provenance)

    train_result, train_frame_audit = _materialize_target_split(
        split_name="train",
        dataset_name=dataset_name,
        rows=train_rows,
        actual_frames=train_frames,
        standard_ivd_mm=ivd_mm,
        valid_mm=valid_mm,
        invalid_mm=invalid_mm,
        solid_mask=np.asarray(source["mask"], dtype=bool),
        thresholds=thresholds,
        standard_meta=standard_meta,
        output_dir=dataset_output / "train_target",
        input_identities=[input_identities[frame_to_row[frame]] for frame in train_frames],
        geometry_fingerprint=standard_meta["geometry_mask_identity"],
        protocol_manifest=protocol_manifest,
        min_area=min_area,
        connectivity=connectivity,
    )
    calibration_result, calibration_frame_audit = _materialize_target_split(
        split_name="calibration",
        dataset_name=dataset_name,
        rows=tuple(frame_to_row[frame] for frame in calibration_frames),
        actual_frames=calibration_frames,
        standard_ivd_mm=ivd_mm,
        valid_mm=valid_mm,
        invalid_mm=invalid_mm,
        solid_mask=np.asarray(source["mask"], dtype=bool),
        thresholds=thresholds,
        standard_meta=standard_meta,
        output_dir=dataset_output / "calibration_reference",
        input_identities=[
            input_identities[frame_to_row[frame]] for frame in calibration_frames
        ],
        geometry_fingerprint=standard_meta["geometry_mask_identity"],
        protocol_manifest=protocol_manifest,
        min_area=min_area,
        connectivity=connectivity,
    )

    audit_frame_audit: list[dict[str, Any]] = []
    audit_counts = {
        name: 0 for name in ("positive", "negative", "unknown", "solid", "valid_fluid", "invalid")
    }
    for frame in audit_frames:
        row = frame_to_row[frame]
        result = classify_p95p80_frame(
            ivd_mm[row],
            solid_mask=np.asarray(source["mask"], dtype=bool),
            valid_fluid_mask=np.asarray(valid_mm[row], dtype=bool),
            invalid_mask=np.asarray(invalid_mm[row], dtype=bool),
            thresholds=thresholds,
            min_area=min_area,
            connectivity=connectivity,
        )
        _add_counts(audit_counts, result.counts)
        audit_frame_audit.append(
            {
                "frame_index": int(frame),
                "standard_ivd_row_index": int(row),
                "counts": _json_safe(dict(result.counts)),
                "coverage": _json_safe(dict(result.coverage)),
                "component_filter": _json_safe(dict(result.filter_audit)),
            }
        )

    aggregate_counts = {
        name: 0 for name in ("positive", "negative", "unknown", "solid", "valid_fluid", "invalid")
    }
    _add_counts(aggregate_counts, train_result["counts"])
    _add_counts(aggregate_counts, calibration_result["counts"])
    _add_counts(aggregate_counts, audit_counts)
    all_frame_audit = train_frame_audit + calibration_frame_audit + audit_frame_audit
    filter_totals = {
        "raw_component_count": 0,
        "kept_component_count": 0,
        "filtered_component_count": 0,
        "raw_positive_cells": 0,
        "post_filter_positive_cells": 0,
        "filtered_positive_cells": 0,
    }
    for frame_record in all_frame_audit:
        component = frame_record["component_filter"]
        for key in filter_totals:
            filter_totals[key] += int(component[key])
    computed_frame_count = len(allowed_frames)
    computed_total_cells = computed_frame_count * int(np.prod(ivd_mm.shape[1:]))
    computed_valid_cells = aggregate_counts["valid_fluid"]
    component_filter = {
        "min_area_cells": int(min_area),
        "connectivity": int(connectivity),
        "applies_after_thresholding": True,
        "applies_to": "positive_components_only",
        "part_of_ivd_definition": False,
        "does_not_change_negative_rule": True,
        "operation_order": list(_FILTER_ORDER),
        **filter_totals,
        "pre_filter_positive_coverage": {
            "grid_fraction": float(filter_totals["raw_positive_cells"] / computed_total_cells),
            "valid_fluid_fraction": float(
                filter_totals["raw_positive_cells"] / computed_valid_cells
            )
            if computed_valid_cells
            else None,
        },
        "post_filter_positive_coverage": {
            "grid_fraction": float(
                filter_totals["post_filter_positive_cells"] / computed_total_cells
            ),
            "valid_fluid_fraction": float(
                filter_totals["post_filter_positive_cells"] / computed_valid_cells
            )
            if computed_valid_cells
            else None,
        },
        "coverage_impact": {
            "grid_fraction_delta": float(
                (filter_totals["post_filter_positive_cells"] - filter_totals["raw_positive_cells"])
                / computed_total_cells
            ),
            "valid_fluid_fraction_delta": float(
                (filter_totals["post_filter_positive_cells"] - filter_totals["raw_positive_cells"])
                / computed_valid_cells
            )
            if computed_valid_cells
            else None,
        },
    }
    window_audit = source["partition_audit"]
    split_windows = window_audit["splits"]
    split_summaries = {
        "train": _split_summary(
            split_name="train",
            frame_indices=train_frames,
            window_info=split_windows["train"],
            counts=train_result["counts"],
            frame_shape=ivd_mm.shape[1:],
            values_read=True,
            frame_audit=train_frame_audit,
        ),
        "calibration": _split_summary(
            split_name="calibration",
            frame_indices=calibration_frames,
            window_info=split_windows["calibration"],
            counts=calibration_result["counts"],
            frame_shape=ivd_mm.shape[1:],
            values_read=True,
            frame_audit=calibration_frame_audit,
        ),
        "development_audit": _split_summary(
            split_name=protocol.AUDIT_SPLIT_NAME,
            frame_indices=audit_frames,
            window_info=split_windows[protocol.AUDIT_SPLIT_NAME],
            counts=audit_counts,
            frame_shape=ivd_mm.shape[1:],
            values_read=True,
            frame_audit=audit_frame_audit,
        ),
        "test": _split_summary(
            split_name="test",
            frame_indices=_partition_frames(record, "test"),
            window_info=split_windows["test"],
            counts=None,
            frame_shape=ivd_mm.shape[1:],
            values_read=False,
        ),
    }
    test_frame_set = set(_partition_frames(record, "test"))
    if test_frame_set.intersection(audit_frames):
        raise Ticket03BuildError(f"dataset={dataset_name} audit/test frame overlap")
    if dataset_name == "boussinesq" and protocol.BOUSSINESQ_AUDIT_FRAME in test_frame_set:
        raise Ticket03BuildError("boussinesq frame 1300 is in test denominator")

    failure_counts = {
        "loader": 0,
        "preflight": 0,
        "standard_ivd": 0,
        "threshold": 0,
        "postprocess": 0,
        "artifact_integrity": 0,
        "artifact_write": 0,
    }
    audit: dict[str, Any] = {
        "artifact_type": "criterion_ambiguity_ticket03_dataset_audit",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "dataset_name": dataset_name,
        "dataset_order_index": list(protocol.DATASET_NAMES).index(dataset_name),
        "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
        "protocol_manifest_hash": protocol_manifest["manifest_hash"],
        "source_manifest_identity": _json_safe(record["source_identity"]),
        "standard_ivd_source_identity": _json_safe(standard_ivd.SOURCE_IDENTITY),
        "standard_ivd_artifact_identity": standard_meta["artifact_identity"],
        "standard_ivd_artifact_instance_identity": standard_meta[
            "artifact_instance_identity"
        ],
        "p95p80_target_artifact_identity": protocol.build_p95p80_artifact_identity(
            protocol.P95P80_PERCENTILE_SOURCE_ID
        ),
        "input_array_identity": {
            "u_path": _as_str_path(pathlib.Path(source["root"]) / "u.npy"),
            "v_path": _as_str_path(pathlib.Path(source["root"]) / "v.npy"),
            "frame_input_identities": {
                str(item["frame_index"]): item["input_identity"]
                for item in standard_frame_records
            },
            "read_scope": "train/calibration/development_audit only",
        },
        "geometry_mask_identity": standard_meta["geometry_mask_identity"],
        "geometry_mask_path": standard_meta["geometry_mask_path"],
        "standard_ivd_provenance": {
            "module": "standard_ivd",
            "implementation_version": standard_ivd.IMPLEMENTATION_VERSION,
            "entry_point": "standard_ivd.compute_standard_ivd_details",
            "formula": standard_ivd.FORMULA,
            "formula_version": standard_ivd.FORMULA_VERSION,
            "mean_scope": "declared_valid_fluid_support",
            "mean_quantity": "signed omega",
            "grid_spacing": {"dx": source["dx"], "dy": source["dy"]},
            "standard_ivd_source_identity": _json_safe(standard_ivd.SOURCE_IDENTITY),
            "implementation_source_path": _as_str_path(pathlib.Path(__file__).with_name("standard_ivd.py")),
            "implementation_source_sha256": _file_sha256(
                pathlib.Path(__file__).with_name("standard_ivd.py")
            ),
        },
        "thresholds": thresholds.to_dict(),
        "threshold_provenance": threshold_provenance,
        "class_counts": aggregate_counts,
        "coverage": _coverage_from_counts(
            aggregate_counts,
            frame_count=computed_frame_count,
            frame_shape=ivd_mm.shape[1:],
        ),
        "invalid_nonfinite_count": int(
            sum(item["input_nonfinite_cell_count"] for item in standard_frame_records)
        ),
        "invalid_count": int(aggregate_counts["invalid"]),
        "frame_counts": {
            "computed_allowed": computed_frame_count,
            "train": len(train_frames),
            "calibration": len(calibration_frames),
            "development_audit": len(audit_frames),
            "test_manifest": len(test_frame_set),
            "test_values_read": False,
        },
        "development_audit_frames": [int(value) for value in audit_frames],
        "manifest_test_ranges": _json_safe(test_ranges),
        "pathline_audit": {
            "t_win": window_audit["t_win"],
            "window_step": window_audit["window_step"],
            "per_split": _json_safe(split_windows),
            "cross_split_window_count": window_audit["cross_split_window_count"],
            "audit_frame_in_test_window_count": window_audit[
                "audit_frame_in_test_window_count"
            ],
            "test_windows_values_read": False,
        },
        "splits": split_summaries,
        "component_filter": component_filter,
        "filter_order": list(_FILTER_ORDER),
        "mask_semantics": {
            "positive_negative_unknown_solid_invalid_mutually_exclusive": True,
            "valid_fluid": "canonical standard IVD valid-fluid mask",
            "solid": "geometry solid, separate from invalid and not a negative",
            "invalid": "non-finite/outside valid-fluid support, excluded and counted",
        },
        "haller_anchor_semantics": {
            "source_identity": _json_safe(standard_ivd.SOURCE_IDENTITY),
            "positive_physics_evidence_only": True,
            "dense_negative_generated": False,
            "dense_segmentation_gt_generated": False,
            "used_to_build_p95p80_target": False,
        },
        "test_source_read": False,
        "test_frame_indices_read": [],
        "final_test_artifacts_read": False,
        "final_test_relationships": "not_read",
        "historical_ivd_read": False,
        "historical_label_read": False,
        "legacy_artifacts_formal_source": False,
        "failure_counts": failure_counts,
        "frame_audit": all_frame_audit,
        "output_paths": {
            "standard_ivd": standard_meta["metadata_path"],
            "train_target": train_result["metadata"],
            "calibration_reference": calibration_result["metadata"],
        },
    }
    audit["audit_hash"] = _json_hash(audit)
    audit_path = _write_json_exclusive(dataset_output / "dataset_audit.json", audit)
    return {
        "audit": audit,
        "audit_path": _as_str_path(audit_path),
        "standard_ivd": {
            "metadata": standard_meta["metadata_path"],
            "frame_indices": _as_str_path(dataset_output / "standard_ivd" / "frame_indices.npy"),
            "directory": _as_str_path(dataset_output / "standard_ivd"),
        },
        "train_target": train_result,
        "calibration_reference": calibration_result,
    }


def _validate_output_identity(dataset_result: Mapping[str, Any]) -> None:
    audit = dataset_result["audit"]
    if audit["test_source_read"] or audit["final_test_artifacts_read"]:
        raise Ticket03BuildError("output audit claims sealed test source was read")
    for key in ("standard_ivd", "train_target", "calibration_reference"):
        path = pathlib.Path(dataset_result[key]["metadata"] if key != "standard_ivd" else dataset_result[key]["metadata"])
        if not path.is_file():
            raise Ticket03BuildError(f"missing output metadata: {path}")
    if any(value != 0 for value in audit["failure_counts"].values()):
        raise Ticket03BuildError(f"dataset audit has failures: {audit['failure_counts']!r}")


def build_ticket03_artifacts(
    input_root: str | pathlib.Path,
    output_root: str | pathlib.Path,
    *,
    min_area: int = DEFAULT_MIN_AREA_CELLS,
    connectivity: int = DEFAULT_CONNECTIVITY,
    t_win: int = protocol.DEFAULT_T_WIN,
    window_step: int = protocol.DEFAULT_WINDOW_STEP,
    recorded_on: str | None = None,
) -> dict[str, Any]:
    """Build the six new Ticket 03 artifacts without opening final-test data."""
    input_path = pathlib.Path(input_root).resolve()
    output_path = pathlib.Path(output_root).resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"input_root 不存在或不是目录：{input_path}")
    if output_path.exists():
        raise FileExistsError(
            f"refuse to overwrite existing Ticket 03 output directory: {output_path}"
        )
    try:
        output_path.relative_to(input_path)
        raise ValueError("output_root 不能位于 input_root 内，避免覆盖 source tree")
    except ValueError as exc:
        if "不能位于" in str(exc):
            raise
    try:
        input_path.relative_to(output_path)
    except ValueError:
        pass
    else:
        raise ValueError("input_root 不能位于 output_root 内")
    area = _strict_positive_int(min_area, name="min_area")
    if connectivity not in (4, 8):
        raise ValueError("connectivity 只能是 4 或 8")
    if isinstance(t_win, bool) or int(t_win) <= 0 or isinstance(window_step, bool) or int(window_step) <= 0:
        raise ValueError("t_win/window_step 必须为正整数")

    # All protocol and source checks occur before output_path is created.  The
    # protocol builder is metadata-only; it does not load old IVD/label values.
    manifest = protocol.build_p95p80_protocol_manifest_from_data_root(
        input_path,
        t_win=int(t_win),
        window_step=int(window_step),
    )
    manifest = protocol.validate_p95p80_protocol_manifest(manifest)
    freeze = protocol.build_p95p80_freeze_record(
        manifest,
        recorded_on=recorded_on or _datetime.date.today().isoformat(),
    )
    freeze = protocol.validate_p95p80_freeze_record(freeze, protocol_manifest=manifest)
    sources = _preflight_sources(input_path, manifest)

    output_path.mkdir(parents=True, exist_ok=False)
    manifest_path = _write_json_exclusive(output_path / "p95p80_protocol_manifest.json", manifest)
    freeze_path = _write_json_exclusive(output_path / "p95p80_freeze_record.json", freeze)
    dataset_results: dict[str, Any] = {}
    for name in protocol.DATASET_NAMES:
        dataset_results[name] = _build_dataset(
            dataset_name=name,
            source=sources[name],
            record=manifest["datasets"][name],
            protocol_manifest=manifest,
            output_root=output_path,
            min_area=area,
            connectivity=int(connectivity),
        )
        _validate_output_identity(dataset_results[name])

    artifact_manifest_payload: dict[str, Any] = {
        "artifact_type": "criterion_ambiguity_ticket03_artifact_manifest",
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "ticket": "03",
        "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
        "protocol_manifest": {
            "path": _as_str_path(manifest_path),
            "manifest_hash": manifest["manifest_hash"],
        },
        "freeze_record": {
            "path": _as_str_path(freeze_path),
            "freeze_record_hash": freeze["freeze_record_hash"],
        },
        "dataset_names": list(protocol.DATASET_NAMES),
        "standard_ivd_source_identity": _json_safe(standard_ivd.SOURCE_IDENTITY),
        "p95p80_target_source_identity": protocol.build_p95p80_artifact_identity(
            protocol.P95P80_PERCENTILE_SOURCE_ID
        ),
        "datasets": {
            name: {
                "standard_ivd": result["standard_ivd"],
                "train_target": {
                    "metadata": result["train_target"]["metadata"],
                    "directory": result["train_target"]["directory"],
                },
                "calibration_reference": {
                    "metadata": result["calibration_reference"]["metadata"],
                    "directory": result["calibration_reference"]["directory"],
                },
                "audit": result["audit_path"],
            }
            for name, result in dataset_results.items()
        },
        "connected_component_postprocessing": {
            "min_area_cells": area,
            "connectivity": int(connectivity),
            "part_of_ivd_definition": False,
            "operation_order": list(_FILTER_ORDER),
        },
        "input_source_contract": {
            "loader_entry_point": "prepare_weak_supervision_artifacts._load_legacy_dataset",
            "allowed_inputs": ["meta.json", "u.npy", "v.npy", "mask.npy"],
            "historical_ivd_read": False,
            "historical_label_read": False,
            "test_source_read": False,
            "test_frame_indices_read": [],
        },
        "final_test_artifacts_read": False,
        "final_test_relationships": "not_read",
        "legacy_artifacts_formal_source": False,
    }
    artifact_manifest = {
        **artifact_manifest_payload,
        "artifact_manifest_hash": _json_hash(artifact_manifest_payload),
    }
    artifact_manifest_path = _write_json_exclusive(
        output_path / "artifact_manifest.json", artifact_manifest
    )
    report_payload = {
        "artifact_type": "criterion_ambiguity_ticket03_report",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "ticket": "03",
        "status": "pass",
        "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
        "protocol_manifest_hash": manifest["manifest_hash"],
        "freeze_record_hash": freeze["freeze_record_hash"],
        "artifact_manifest_path": _as_str_path(artifact_manifest_path),
        "dataset_names": list(protocol.DATASET_NAMES),
        "datasets": {name: result["audit"] for name, result in dataset_results.items()},
        "test_source_read": False,
        "final_test_artifacts_read": False,
        "historical_ivd_read": False,
        "historical_label_read": False,
        "final_test_relationships": "not_read",
        "execution_scope": {
            "computed_splits": ["train", "calibration", protocol.AUDIT_SPLIT_NAME],
            "test_values": "sealed/not_read",
            "boussinesq_frame_1300": "development/audit-only",
            "training_started": False,
            "final_evaluation_started": False,
            "ticket04_or_later_started": False,
        },
    }
    report = {**report_payload, "report_hash": _json_hash(report_payload)}
    report_path = _write_json_exclusive(output_path / "ticket03_report.json", report)
    return {
        "output_root": _as_str_path(output_path),
        "dataset_names": list(protocol.DATASET_NAMES),
        "protocol_manifest": manifest,
        "freeze_record": freeze,
        "protocol_manifest_path": _as_str_path(manifest_path),
        "freeze_record_path": _as_str_path(freeze_path),
        "artifact_manifest_path": _as_str_path(artifact_manifest_path),
        "report_path": _as_str_path(report_path),
        "datasets": dataset_results,
        "test_source_read": False,
        "final_test_artifacts_read": False,
        "historical_ivd_read": False,
        "historical_label_read": False,
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA_CELLS)
    parser.add_argument("--connectivity", type=int, default=DEFAULT_CONNECTIVITY)
    parser.add_argument("--t-win", type=int, default=protocol.DEFAULT_T_WIN)
    parser.add_argument("--window-step", type=int, default=protocol.DEFAULT_WINDOW_STEP)
    parser.add_argument("--recorded-on", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    result = build_ticket03_artifacts(
        args.input_root,
        args.output_root,
        min_area=args.min_area,
        connectivity=args.connectivity,
        t_win=args.t_win,
        window_step=args.window_step,
        recorded_on=args.recorded_on,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "output_root": result["output_root"],
                "report_path": result["report_path"],
                "dataset_names": result["dataset_names"],
                "test_source_read": result["test_source_read"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "IMPLEMENTATION_VERSION",
    "TARGET_SCHEMA_VERSION",
    "CALIBRATION_SCHEMA_VERSION",
    "AUDIT_SCHEMA_VERSION",
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "NEGATIVE",
    "POSITIVE",
    "UNKNOWN",
    "SOLID",
    "INVALID",
    "P95P80Thresholds",
    "P95P80FrameClassification",
    "Ticket03BuildError",
    "compute_p95p80_thresholds",
    "filter_positive_components",
    "classify_p95p80_frame",
    "load_raw_dataset_source",
    "build_ticket03_artifacts",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
