"""First-class, layered Haller evidence tracing for one development frame.

The legacy :mod:`haller_anchors` artifact is intentionally kept intact.  This
module adds the R1 label-evidence seam around the same numerical extractor:
raw standard IVD, raw/selected contour records, filtering decisions, filled
interiors, an explicit boundary band, negative candidates, and a final
three-state evidence array are published together with provenance and an
integrity fingerprint.

The tracer is development-only.  It never falls back to ``p85`` and every
artifact produced here is a candidate, not a trusted Haller anchor.  Trust is
reserved for the later R2 freeze process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

import haller_anchors as strict


EVIDENCE_FORMAT_VERSION = strict.EVIDENCE_FORMAT_VERSION
ARTIFACT_STATE_CANDIDATE = strict.ARTIFACT_STATE_CANDIDATE
ARTIFACT_STATE_AUDITED = strict.ARTIFACT_STATE_AUDITED
ARTIFACT_STATE_FROZEN = strict.ARTIFACT_STATE_FROZEN

STATUS_VALID = "valid"
STATUS_VALID_EMPTY = "valid_empty"
STATUS_VALID_ALL_NEGATIVE = "valid_all_negative"
STATUS_ALL_UNKNOWN = "all_unknown"
STATUS_INVALID_INPUT = "invalid_input"
STATUS_EXTRACTOR_FAILURE = "extractor_failure"

_VALID_STATUSES = frozenset({
    STATUS_VALID,
    STATUS_VALID_EMPTY,
    STATUS_VALID_ALL_NEGATIVE,
    STATUS_ALL_UNKNOWN,
    STATUS_INVALID_INPUT,
    STATUS_EXTRACTOR_FAILURE,
})

_DEVELOPMENT_SPLITS = frozenset({"train", "calibration", "development_audit"})
_SPLIT_TO_SOURCE = {
    "train": strict.SOURCE_TRAIN,
    "calibration": strict.SOURCE_CALIBRATION,
    "development_audit": strict.SOURCE_DEVELOPMENT_AUDIT,
}

# These names are the R0 protocol's six real datasets.  Synthetic fixtures
# may omit a manifest, but a real dataset must carry the validated protocol
# identity before a tracer can read its frame.
_PROTOCOL_DATASET_NAMES = frozenset({
    "boussinesq",
    "cylinder2d",
    "doublegyre2d",
    "fourcenters2d",
    "jungtelziemniak2d",
    "pipedcylinder2d",
})

_ARRAY_FILES = {
    "velocity_u": "velocity_u.npy",
    "velocity_v": "velocity_v.npy",
    "xdim": "xdim.npy",
    "ydim": "ydim.npy",
    "omega": "omega.npy",
    "raw_scalar_field": "raw_scalar_field.npy",
    # Keep an explicit name for consumers that call the field ``standard_ivd``.
    "standard_ivd": "standard_ivd.npy",
    "solid_mask": "solid_mask.npy",
    "invalid_support": "invalid_support.npy",
    "valid_fluid": "valid_fluid.npy",
    "filled_interior": "filled_interior.npy",
    "unknown_band": "unknown_band.npy",
    "negative_candidates": "negative_candidates.npy",
    "evidence_state": "evidence_state.npy",
}

_JSON_FILES = {
    "raw_contours": "raw_contours.json",
    "filtering_decisions": "filtering_decisions.json",
    "selected_contours": "selected_contours.json",
}

_CATEGORICAL_ARRAYS = frozenset({
    "solid_mask",
    "invalid_support",
    "valid_fluid",
    "filled_interior",
    "unknown_band",
    "negative_candidates",
    "evidence_state",
})


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_copy(value: Any) -> Any:
    """Make the JSON records independent of numpy scalar/list objects."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=True))


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _coerce_exact_frame_index(value: Any) -> int:
    """Coerce integer-like frame identities without silently truncating."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("frame_index 必须是整数，不能是 bool")
    try:
        frame = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("frame_index 必须是整数") from exc
    if not np.isfinite(numeric) or numeric != frame:
        raise ValueError(f"frame_index 必须是整数，实际 {value!r}")
    return frame


def _safe_shape(*values: Any) -> tuple[int, int]:
    """Infer a storage shape for an invalid-input artifact."""
    for value in values:
        try:
            array = np.asarray(value)
        except Exception:
            continue
        if array.ndim == 2 and all(int(size) > 0 for size in array.shape):
            return int(array.shape[0]), int(array.shape[1])
        if array.ndim == 3 and array.shape[0] == 1 and all(
            int(size) > 0 for size in array.shape[1:]
        ):
            return int(array.shape[1]), int(array.shape[2])
    return 1, 1


def _safe_storage_array(
    value: Any,
    shape: tuple[int, int],
    *,
    dtype: Any,
    fill_value: Any,
) -> np.ndarray:
    try:
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.shape == shape:
            return np.asarray(array, dtype=dtype).copy()
    except (TypeError, ValueError, OverflowError):
        pass
    return np.full(shape, fill_value, dtype=dtype)


def _safe_axis(value: Any, length: int) -> np.ndarray:
    try:
        axis = np.asarray(value, dtype=np.float64)
        if axis.ndim == 1 and len(axis) == length and np.all(np.isfinite(axis)):
            return axis.copy()
    except (TypeError, ValueError, OverflowError):
        pass
    return np.arange(length, dtype=np.float64)


def _axis_spacing(axis: np.ndarray) -> float | None:
    """Return spacing when an axis is non-degenerate, else an explicit null."""
    if len(axis) < 2:
        return None
    delta = float(axis[1] - axis[0])
    return delta if np.isfinite(delta) and delta > 0.0 else None


def _safe_hash(value: Any) -> str:
    try:
        return strict._hash_array(np.asarray(value))
    except Exception:
        return _json_hash({"repr": repr(value)})


def _input_fingerprint(
    u: Any,
    v: Any,
    xdim: Any,
    ydim: Any,
    solid: Any,
) -> dict[str, Any]:
    arrays = {
        "u": np.asarray(u),
        "v": np.asarray(v),
        "xdim": np.asarray(xdim),
        "ydim": np.asarray(ydim),
        "solid_mask": np.asarray(solid),
    }
    hashes = {name: _safe_hash(array) for name, array in arrays.items()}
    try:
        combined = strict._hash_named_arrays(arrays)
    except Exception:
        combined = _json_hash(hashes)
    return {
        "algorithm": "sha256(dtype,shape,c-order-by-name)",
        "arrays": hashes,
        "combined": combined,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_hash(path: pathlib.Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _runtime_identity(backend: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": _package_version("scipy"),
        "scikit_image_version": _package_version("scikit-image"),
        "numba_version": _package_version("numba"),
        "backend": dict(backend),
    }


def _generator_identity() -> dict[str, Any]:
    return {
        # Keep the public tracer seam stable for downstream readers while
        # recording the concrete implementation module for auditability.
        "module": "haller_anchors",
        "implementation_module": "haller_evidence",
        "version": EVIDENCE_FORMAT_VERSION,
        "source_sha256": _source_hash(pathlib.Path(__file__)),
        "haller_anchors_algorithm_version": strict.ALGORITHM_VERSION,
    }


def _literature_record() -> dict[str, Any]:
    return {
        "status": "metadata_and_fulltext_verified_local_zotero",
        "zotero_key": "L2PX3NQX",
        "title": "Defining coherent vortices objectively from the vorticity",
        "verified_facts": [
            "IVD(x,t)=abs(omega(x,t)-spatial_mean(omega,t))",
            "the spatial mean is taken over the flow domain",
            "Eulerian vortex candidates are nested IVD level sets with outward non-increasing values",
        ],
        "engineering_choices": [
            "fluid-only finite-difference vorticity on the project grid",
            "8-neighborhood local maxima",
            "project contour levels, convexity, perimeter, band, and negative policy",
        ],
        "canonical_parameter_claim": False,
    }


def _resolve_configuration(
    backend: str,
    device: Any,
    contour_mode: str,
    fast_global_level_count: int,
    fast_refinement_iterations: int,
    fast_refinement_halo_cells: int,
    numbacs_radius_factor: float,
) -> tuple[Any, str, dict[str, Any]]:
    details = strict._resolve_backend(backend, device)
    mode = strict._contour_mode_for_backend(details.resolved, contour_mode)
    params = dict(strict._resolve_parameters())
    if details.resolved == strict.BACKEND_FAST_HALLER:
        try:
            global_count = int(fast_global_level_count)
            iterations = int(fast_refinement_iterations)
            halo_cells = int(fast_refinement_halo_cells)
        except (TypeError, ValueError) as exc:
            raise ValueError("fast_haller 参数必须是整数") from exc
        if global_count < 2 or iterations < 0 or halo_cells < 0:
            raise ValueError("fast_haller 参数不满足非负/最小值约束")
        params.update({
            "fast_global_level_count": global_count,
            "fast_refinement_iterations": iterations,
            "fast_refinement_halo_cells": halo_cells,
            "fast_level_schedule": (
                "geomspace(min_peak*contour_level_end,max_peak*contour_level_start)"
            ),
            "fast_refinement": "outer_invalid_to_valid_bisection",
        })
    elif details.resolved == strict.BACKEND_NUMBACS:
        from numbacs_haller import (
            UPSTREAM_DEFAULT_END_LEVEL,
            UPSTREAM_DEFAULT_MIN_VAL,
            UPSTREAM_DEFAULT_NLEVS,
            UPSTREAM_DEFAULT_START_LEVEL,
            runtime_mode as numbacs_runtime_mode,
        )

        try:
            radius_factor = float(numbacs_radius_factor)
        except (TypeError, ValueError) as exc:
            raise ValueError("numbacs_radius_factor 必须是正数") from exc
        if not np.isfinite(radius_factor) or radius_factor <= 0.0:
            raise ValueError("numbacs_radius_factor 必须是正数")

        params.update({
            "numbacs_method": "numbacs.extraction.elliptic.rotcohvrt",
            "numbacs_upstream_version": "0.2.0",
            "numbacs_upstream_commit": "c067f542543f5dd4ae3dc45fc506213e8d98b845",
            "numbacs_runtime_mode": numbacs_runtime_mode(),
            "numbacs_contour_level_count": UPSTREAM_DEFAULT_NLEVS,
            "numbacs_nlevs_passed_explicitly": False,
            "numbacs_default_min_val": UPSTREAM_DEFAULT_MIN_VAL,
            "numbacs_default_start_level": UPSTREAM_DEFAULT_START_LEVEL,
            "numbacs_default_end_level": UPSTREAM_DEFAULT_END_LEVEL,
            "numbacs_explicit_kwargs": ["convexity_deficiency", "min_len"],
            "numbacs_omitted_kwargs": [
                "min_val", "nlevs", "start_level", "end_level",
            ],
            "numbacs_schedule_start": "upstream_default_percentile70",
            "numbacs_schedule_end": "upstream_default_maximum",
            "numbacs_radius_factor": radius_factor,
            "numbacs_convexity_method": "upstream_default_convex_hull",
            "numbacs_boundary": "upstream_convex_hull",
        })
    return details, mode, params


def _load_protocol_manifest(
    protocol_manifest: Mapping[str, Any] | str | pathlib.Path | None,
) -> dict[str, Any] | None:
    if protocol_manifest is None:
        return None
    import experiment_protocol

    if isinstance(protocol_manifest, (str, pathlib.Path)):
        return experiment_protocol.read_protocol_manifest(protocol_manifest)
    return experiment_protocol.validate_protocol_manifest(protocol_manifest)


def _load_freeze_record(
    freeze_record: Mapping[str, Any] | str | pathlib.Path | None,
) -> dict[str, Any] | None:
    if freeze_record is None:
        return None
    if isinstance(freeze_record, (str, pathlib.Path)):
        return json.loads(pathlib.Path(freeze_record).read_text(encoding="utf-8"))
    return _json_copy(freeze_record)


def _validate_scope(
    dataset_name: str,
    split_name: str,
    frame_index: Any,
    source: str | None,
    protocol_manifest: Mapping[str, Any] | str | pathlib.Path | None,
) -> tuple[str, int, dict[str, Any] | None, dict[str, Any] | None]:
    dataset = _text(dataset_name, name="dataset_name")
    split = _text(split_name, name="split_name")
    frame = _coerce_exact_frame_index(frame_index)
    if frame < 0 or split not in {"train", "calibration", "development_audit", "test"}:
        raise ValueError(f"无效 development frame identity: split={split!r}, frame={frame}")
    if split == "test":
        raise PermissionError(
            "Haller evidence tracer 只允许 development train/calibration/audit，拒绝 final-test"
        )
    expected_source = _SPLIT_TO_SOURCE[split]
    resolved_source = expected_source if source is None else _text(source, name="source")
    strict._validate_source(resolved_source)
    if resolved_source == strict.SOURCE_TEST:
        raise PermissionError("candidate Haller evidence 不能从 final-test source 生成")
    if resolved_source != expected_source:
        raise ValueError(
            f"source={resolved_source!r} 与 split={split!r} 不匹配，拒绝混用 evidence"
        )

    if protocol_manifest is None and dataset in _PROTOCOL_DATASET_NAMES:
        raise ValueError(
            f"real dataset={dataset!r} 必须提供已验证的 R0 protocol manifest，"
            "再读取任何 frame"
        )

    manifest = _load_protocol_manifest(protocol_manifest)
    if manifest is not None:
        datasets = manifest["datasets"]
        if dataset not in datasets:
            raise ValueError(f"dataset={dataset!r} 不在冻结的六数据集协议中")
        record = datasets[dataset]
        if split == "development_audit":
            allowed = set(record.get("development_audit_frames", ()))
            if frame not in allowed:
                raise ValueError(
                    f"dataset={dataset} frame={frame} 不在 development_audit manifest"
                )
        else:
            ranges = record["frame_partitions"][split]["ranges"]
            if not any(int(start) <= frame < int(end) for start, end in ranges):
                raise ValueError(
                    f"dataset={dataset} frame={frame} 不在 {split} manifest"
                )
    freeze = None
    return resolved_source, frame, manifest, freeze


def _prepare_valid_inputs(
    u: Any,
    v: Any,
    xdim: Any,
    ydim: Any,
    mask: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    u2, v2 = strict._coerce_frame(u, v)
    x, y, dx, dy = strict._validate_coords(xdim, ydim, u2.shape)
    solid = strict._coerce_solid_mask(mask, u2.shape)
    return u2, v2, x, y, solid, dx, dy


def _backend_identity(details: Any) -> dict[str, Any]:
    return {
        "requested": details.requested,
        "resolved": details.resolved,
        "backend": details.resolved,
        "device": details.device,
        "cuda_used": bool(details.cuda_used),
        "fallback_reason": details.fallback_reason,
        "backend_version": details.backend_version,
        "compute_dtype": details.compute_dtype,
    }


def _base_metadata(
    *,
    dataset_name: str,
    split_name: str,
    frame_index: int,
    source: str,
    u: np.ndarray,
    v: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
    solid: np.ndarray,
    invalid_support: np.ndarray,
    details: Any,
    contour_mode: str,
    params: Mapping[str, Any],
    status: str,
    status_reason: str,
    failure_count: int,
    failure_reasons: Sequence[str],
    contour_evaluation: Mapping[str, Any] | None,
    frame_fluid_p60: float | None,
    protocol_manifest: Mapping[str, Any] | None,
    freeze_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    backend = _backend_identity(details)
    input_fp = _input_fingerprint(u, v, xdim, ydim, solid)
    geometry = {
        "coordinate_order": "x_then_y",
        "array_order": "row_y_then_col_x",
        "shape": [int(u.shape[0]), int(u.shape[1])],
        "xdim_length": int(len(xdim)),
        "ydim_length": int(len(ydim)),
        "xdim_min": float(xdim[0]),
        "xdim_max": float(xdim[-1]),
        "ydim_min": float(ydim[0]),
        "ydim_max": float(ydim[-1]),
        "dx": _axis_spacing(xdim),
        "dy": _axis_spacing(ydim),
        "xdim_hash": _safe_hash(xdim),
        "ydim_hash": _safe_hash(ydim),
        "fluid_mask": "valid_fluid_only",
        "solid_semantics": "ignored_not_negative",
        "invalid_semantics": "ignored_and_counted",
    }
    metadata: dict[str, Any] = {
        "artifact_type": "haller_ivd_layered_evidence",
        "format_version": EVIDENCE_FORMAT_VERSION,
        "artifact_state": ARTIFACT_STATE_CANDIDATE,
        "trusted": False,
        "dataset_name": dataset_name,
        "split_name": split_name,
        "frame_index": int(frame_index),
        "source": source,
        "scope_binding": {
            "dataset_name": dataset_name,
            "split_name": split_name,
            "frame_index": int(frame_index),
            "source": source,
        },
        "label_source": source,
        "status": status,
        "status_reason": status_reason,
        "frame_valid": status not in {STATUS_INVALID_INPUT, STATUS_EXTRACTOR_FAILURE},
        "valid": status in {STATUS_VALID, STATUS_VALID_EMPTY, STATUS_VALID_ALL_NEGATIVE},
        "failure_count": int(failure_count),
        "failure_reasons": [str(reason) for reason in failure_reasons],
        "fallback_used": None,
        "legacy_p85_used": False,
        "geometry": geometry,
        "input_hash": input_fp["combined"],
        "mask_hash": _safe_hash(solid),
        "input_fingerprint": input_fp,
        "generator_identity": _generator_identity(),
        "runtime_identity": _runtime_identity(backend),
        "backend_requested": details.requested,
        "resolved": details.resolved,
        "backend": details.resolved,
        "device": details.device,
        "cuda_used": bool(details.cuda_used),
        "backend_version": details.backend_version,
        "backend_fallback_reason": details.fallback_reason,
        "contour_mode": contour_mode,
        "contour_backend": (
            "numbacs.extraction.elliptic.rotcohvrt"
            if details.resolved == strict.BACKEND_NUMBACS
            else "project.haller_anchors.contour_selector"
        ),
        "raw_contour_source": (
            "numbacs.rotcohvrt"
            if details.resolved == strict.BACKEND_NUMBACS
            else "project_contour_selector"
        ),
        "contour_evaluation": _json_copy(contour_evaluation or {}),
        "parameters": _json_copy(dict(params)),
        "parameter_hash": _json_hash(dict(params)),
        "state_encoding": {
            "positive": int(strict.POSITIVE),
            "negative": int(strict.NEGATIVE),
            "unknown": int(strict.UNKNOWN),
        },
        "raw_scalar_definition": "standard_ivd",
        "ivd_formula": "abs(omega - mean_fluid(omega))",
        "mean_scope": "entire_fluid_spatial_domain",
        "frame_fluid_p60": frame_fluid_p60,
        "solid_policy": "solid cells are always unknown; no positive or negative evidence",
        "invalid_policy": "invalid cells are ignored and counted; no evidence is emitted",
        "failure_policy": "extractor failure is unknown with no p85 fallback",
        "literature": _literature_record(),
        "protocol": None if protocol_manifest is None else {
            "protocol_version": protocol_manifest["protocol_version"],
            "schema_version": protocol_manifest["schema_version"],
            "manifest_hash": protocol_manifest["manifest_hash"],
            "dataset_name": dataset_name,
            "split_name": split_name,
            "frame_index": int(frame_index),
            "eligible_for_test_relationships": False,
            "access_state": "development_only",
        },
        "freeze_record_hash": (
            None if freeze_record is None else _json_hash(freeze_record)
        ),
        "layer_contract": "r1-layered-label-evidence-v1",
        "artifact_files": {
            "arrays": dict(_ARRAY_FILES),
            "json": dict(_JSON_FILES),
            "metadata": "evidence_meta.json",
        },
    }
    return metadata


def _contour_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the stable candidate identity shared by raw/filter/selected layers."""
    peak = record.get("peak", {})
    if not isinstance(peak, Mapping):
        peak = {}
    return (
        peak.get("row"),
        peak.get("col"),
        record.get("level"),
        record.get("level_fraction"),
    )


def _candidate_weight(record: Mapping[str, Any]) -> int:
    """Read one compressed candidate weight, treating error events as zero."""
    if record.get("status") == "error":
        return 0
    raw_weight = record.get("candidate_count", 1)
    if isinstance(raw_weight, (bool, np.bool_)) or not isinstance(
        raw_weight, (int, np.integer)
    ):
        raise ValueError("Haller evidence candidate_count 必须为正整数")
    weight = int(raw_weight)
    if weight < 1:
        raise ValueError("Haller evidence candidate_count 必须为正整数")
    return weight


def _raw_contour_records(
    selected: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain raw diagnostics and restore selected geometry when compressed."""
    selected_records = [_json_copy(item) for item in selected]

    selected_by_identity = {
        _contour_identity(item): item for item in selected_records
    }
    records: list[dict[str, Any]] = []
    for ordinal, item in enumerate(diagnostics):
        record = _json_copy(item)
        record["raw_record_index"] = ordinal
        selected_record = selected_by_identity.get(_contour_identity(record))
        if selected_record is not None and "points_grid" not in record:
            for key in ("points_grid", "points_xy", "area", "perimeter", "hull_area", "n_points"):
                if key in selected_record:
                    record[key] = _json_copy(selected_record[key])
        record["geometry_available"] = bool("points_grid" in record)
        records.append(record)
    known = {_contour_identity(item) for item in records}
    for selected_record in selected_records:
        if _contour_identity(selected_record) not in known:
            selected_record["raw_record_index"] = len(records)
            selected_record["geometry_available"] = True
            records.append(selected_record)
    return records


def _filtering_decisions(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for item in records:
        record = _json_copy(item)
        status = record.get("status")
        if status == "error":
            record["decision"] = "error"
            record["candidate_count"] = 0
        else:
            record["decision"] = "accepted" if status == "valid" else "rejected"
        record["reason"] = "accepted" if record["decision"] == "accepted" else (
            record.get("rejection_reason") or record.get("error") or "rejected"
        )
        decisions.append(record)
    return decisions


def _candidate_accounting(
    decisions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Count compressed diagnostics by candidate weight, not record count."""
    accepted = 0
    rejected = 0
    rejection_reasons: Counter[str] = Counter()
    non_candidate_events = 0
    for record in decisions:
        if record.get("decision") == "error":
            if record.get("candidate_count", 0) != 0:
                raise ValueError("Haller evidence error event 不能带 candidate_count")
            non_candidate_events += 1
            continue
        weight = _candidate_weight(record)
        if record.get("decision") == "accepted":
            accepted += weight
        else:
            rejected += weight
            rejection_reasons[str(record.get("reason"))] += weight
    return (
        {
            "total": int(accepted + rejected),
            "accepted": int(accepted),
            "rejected": int(rejected),
            "non_candidate_events": int(non_candidate_events),
        },
        dict(rejection_reasons),
    )


def _validate_candidate_accounting(
    metadata: Mapping[str, Any],
    raw_contours: Sequence[Mapping[str, Any]],
    filtering_decisions: Sequence[Mapping[str, Any]],
    selected_contours: Sequence[Mapping[str, Any]],
) -> None:
    """Recompute counts and verify raw/filter/selected layer associations."""
    raw_candidate_count = 0
    raw_by_index: dict[int, Mapping[str, Any]] = {}
    for position, record in enumerate(raw_contours):
        raw_index = record.get("raw_record_index")
        if isinstance(raw_index, (bool, np.bool_)) or not isinstance(
            raw_index, (int, np.integer)
        ) or int(raw_index) != position:
            raise ValueError("Haller evidence raw_record_index 必须连续且与 raw layer 顺序一致")
        raw_by_index[int(raw_index)] = record
        raw_candidate_count += _candidate_weight(record)
    if len(raw_by_index) != len(raw_contours):
        raise ValueError("Haller evidence raw_record_index 不能重复")

    filtering_by_index: dict[int, Mapping[str, Any]] = {}
    for decision in filtering_decisions:
        if decision.get("decision") == "error" and decision.get("raw_record_index") is None:
            if decision.get("status") != "error" or decision.get("candidate_count", 0) != 0:
                raise ValueError("Haller evidence error event candidate accounting 无效")
            continue
        raw_index = decision.get("raw_record_index")
        if isinstance(raw_index, (bool, np.bool_)) or not isinstance(
            raw_index, (int, np.integer)
        ) or int(raw_index) not in raw_by_index:
            raise ValueError("Haller evidence filtering_decisions 未绑定 raw contour")
        raw_index = int(raw_index)
        if raw_index in filtering_by_index:
            raise ValueError("Haller evidence filtering_decisions raw_record_index 不能重复")
        raw_record = raw_by_index[raw_index]
        if _contour_identity(decision) != _contour_identity(raw_record):
            raise ValueError("Haller evidence filtering decision 与 raw contour identity 不一致")
        if decision.get("status") != raw_record.get("status"):
            raise ValueError("Haller evidence filtering decision 与 raw contour status 不一致")
        if _candidate_weight(decision) != _candidate_weight(raw_record):
            raise ValueError("Haller evidence filtering decision candidate_count 与 raw 不一致")
        expected_decision = (
            "error" if raw_record.get("status") == "error" else
            "accepted" if raw_record.get("status") == "valid" else "rejected"
        )
        if decision.get("decision") != expected_decision:
            raise ValueError("Haller evidence filtering decision 与 raw status 不一致")
        filtering_by_index[raw_index] = decision
    if set(filtering_by_index) != set(raw_by_index):
        raise ValueError("Haller evidence filtering_decisions 必须覆盖全部 raw contours")

    accepted_identities = Counter(
        _contour_identity(raw_by_index[index])
        for index, decision in filtering_by_index.items()
        if decision.get("decision") == "accepted"
    )
    selected_identities: Counter[tuple[Any, ...]] = Counter()
    for selected in selected_contours:
        selected_index = selected.get("raw_record_index")
        if selected_index is not None:
            if isinstance(selected_index, (bool, np.bool_)) or not isinstance(
                selected_index, (int, np.integer)
            ) or int(selected_index) not in raw_by_index:
                raise ValueError("Haller evidence selected contour 未绑定 raw contour")
            selected_index = int(selected_index)
            raw_record = raw_by_index[selected_index]
            if _contour_identity(selected) != _contour_identity(raw_record):
                raise ValueError("Haller evidence selected contour 与 raw identity 不一致")
            if filtering_by_index[selected_index].get("decision") != "accepted":
                raise ValueError("Haller evidence selected contour 不是 accepted raw contour")
        selected_identities[_contour_identity(selected)] += 1
    if any(
        selected_identities[identity] > accepted_identities[identity]
        for identity in selected_identities
    ):
        raise ValueError("Haller evidence selected contours 不是 accepted raw contours 的子集")

    candidate_counts, rejection_histogram = _candidate_accounting(
        filtering_decisions
    )
    if metadata.get("raw_contour_count") != len(raw_contours):
        raise ValueError("Haller evidence raw_contour_count 与 records 不一致")
    if metadata.get("raw_candidate_count") != raw_candidate_count:
        raise ValueError("Haller evidence raw_candidate_count 与 records 不一致")
    if metadata.get("filtering_decision_count") != len(filtering_decisions):
        raise ValueError("Haller evidence filtering_decision_count 与 records 不一致")
    if metadata.get("selected_contour_count") != len(selected_contours):
        raise ValueError("Haller evidence selected_contour_count 与 records 不一致")
    if candidate_counts["total"] != raw_candidate_count:
        raise ValueError("Haller evidence candidate accounting 与 raw records 不一致")
    if dict(metadata.get("candidate_counts", {})) != candidate_counts:
        raise ValueError("Haller evidence candidate_counts 重算不一致")
    if dict(metadata.get("rejection_reason_histogram", {})) != rejection_histogram:
        raise ValueError("Haller evidence rejection_reason_histogram 重算不一致")
    if len(selected_contours) > candidate_counts["accepted"]:
        raise ValueError("Haller evidence selected contours 超过 accepted candidates")


def _layer_counts(arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    solid = np.asarray(arrays["solid_mask"], dtype=bool)
    invalid = np.asarray(arrays["invalid_support"], dtype=bool)
    fluid = np.asarray(arrays["valid_fluid"], dtype=bool)
    state = np.asarray(arrays["evidence_state"], dtype=np.int8)
    return {
        "total_cells": int(state.size),
        "solid_cells": int(np.count_nonzero(solid)),
        "invalid_cells": int(np.count_nonzero(invalid)),
        "valid_fluid_cells": int(np.count_nonzero(fluid)),
        "filled_interior_cells": int(np.count_nonzero(arrays["filled_interior"])),
        "unknown_band_cells": int(np.count_nonzero(arrays["unknown_band"])),
        "negative_candidate_cells": int(np.count_nonzero(arrays["negative_candidates"])),
        "positive_cells": int(np.count_nonzero((state == strict.POSITIVE) & fluid)),
        "negative_cells": int(np.count_nonzero((state == strict.NEGATIVE) & fluid)),
        "unknown_cells": int(np.count_nonzero((state == strict.UNKNOWN) & fluid)),
        "known_cells": int(np.count_nonzero(
            ((state == strict.POSITIVE) | (state == strict.NEGATIVE)) & fluid
        )),
    }


def _layer_conservation(arrays: Mapping[str, np.ndarray]) -> bool:
    solid = np.asarray(arrays["solid_mask"], dtype=bool)
    invalid = np.asarray(arrays["invalid_support"], dtype=bool)
    fluid = np.asarray(arrays["valid_fluid"], dtype=bool)
    state = np.asarray(arrays["evidence_state"], dtype=np.int8)
    if not np.array_equal(fluid, ~(solid | invalid)):
        return False
    if np.any(~np.isin(state, [strict.POSITIVE, strict.NEGATIVE, strict.UNKNOWN])):
        return False
    if np.any(state[~fluid] != strict.UNKNOWN):
        return False
    filled = np.asarray(arrays["filled_interior"], dtype=bool)
    band = np.asarray(arrays["unknown_band"], dtype=bool)
    negative = np.asarray(arrays["negative_candidates"], dtype=bool)
    if np.any(negative & (~fluid | filled | band)):
        return False
    if np.any((state == strict.POSITIVE) & (~fluid | ~filled | band)):
        return False
    if np.any((state == strict.NEGATIVE) != negative):
        return False
    if np.any((state == strict.POSITIVE) != (fluid & filled & ~band)):
        return False
    return bool(
        np.count_nonzero(fluid)
        == np.count_nonzero((state == strict.POSITIVE) & fluid)
        + np.count_nonzero((state == strict.NEGATIVE) & fluid)
        + np.count_nonzero((state == strict.UNKNOWN) & fluid)
    )


def _array_hashes(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {
        name: strict._hash_array(np.asarray(arrays[name]))
        for name in _ARRAY_FILES
    }


def _json_hashes(json_layers: Mapping[str, Any]) -> dict[str, str]:
    return {name: _json_hash(json_layers[name]) for name in _JSON_FILES}


def _verify_integrity_fields(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    json_layers: Mapping[str, Any],
) -> None:
    """Verify the complete bundle before any trust decision is considered."""
    expected_array_hashes = metadata.get("array_hashes")
    expected_json_hashes = metadata.get("json_hashes")
    if not isinstance(expected_array_hashes, Mapping) or not isinstance(
        expected_json_hashes, Mapping
    ):
        raise ValueError("incomplete Haller evidence artifact: missing array/json hashes")
    actual_array_hashes = _array_hashes(arrays)
    actual_json_hashes = _json_hashes(json_layers)
    if dict(expected_array_hashes) != actual_array_hashes:
        raise ValueError("Haller evidence array hash 校验失败")
    if dict(expected_json_hashes) != actual_json_hashes:
        raise ValueError("Haller evidence JSON hash 校验失败")
    actual_integrity = _integrity_fingerprint(metadata, arrays, json_layers)
    if metadata.get("integrity_fingerprint") != actual_integrity:
        raise ValueError("Haller evidence integrity fingerprint 校验失败")


def _round_trip_metrics(
    out: pathlib.Path,
    expected_arrays: Mapping[str, np.ndarray],
    expected_json: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Compare the pre-save/first-read bundle with an independent disk read."""
    categorical_difference_cells = 0
    floating_max_abs_difference = 0.0
    for name, filename in _ARRAY_FILES.items():
        actual = np.load(out / filename, allow_pickle=False)
        expected = np.asarray(expected_arrays[name])
        if name in _CATEGORICAL_ARRAYS:
            categorical_difference_cells = max(
                categorical_difference_cells,
                int(np.count_nonzero(expected != actual)),
            )
            continue
        if np.array_equal(expected, actual, equal_nan=True):
            continue
        finite_both = np.isfinite(expected) & np.isfinite(actual)
        if np.any(finite_both):
            floating_max_abs_difference = max(
                floating_max_abs_difference,
                float(np.max(np.abs(expected[finite_both] - actual[finite_both]))),
            )
        if np.any(np.isfinite(expected) != np.isfinite(actual)):
            floating_max_abs_difference = float("inf")
    json_layers_equal = True
    for name, filename in _JSON_FILES.items():
        actual = _read_json_records(out / filename, name=filename)
        if _canonical_json(expected_json[name]) != _canonical_json(actual):
            json_layers_equal = False
    return {
        "comparison": "pre_save_or_first_read_vs_independent_disk_readback",
        "categorical_max_difference_cells": int(categorical_difference_cells),
        "floating_max_abs_difference": float(floating_max_abs_difference),
        "json_layers_equal": bool(json_layers_equal),
    }


def _integrity_fingerprint(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    json_layers: Mapping[str, Any],
) -> str:
    clean_metadata = dict(metadata)
    clean_metadata.pop("integrity_fingerprint", None)
    clean_metadata.pop("array_hashes", None)
    clean_metadata.pop("json_hashes", None)
    # R2 authorization is detached from the artifact fingerprint.  Including
    # its own artifact_fingerprint here would create an impossible cycle.
    clean_metadata.pop("trust_authorization", None)
    payload = {
        "format_version": EVIDENCE_FORMAT_VERSION,
        "metadata": clean_metadata,
        "array_hashes": _array_hashes(arrays),
        "json_hashes": _json_hashes(json_layers),
    }
    return _json_hash(payload)


def _assemble_result(
    *,
    u: np.ndarray,
    v: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
    omega: np.ndarray,
    raw_scalar_field: np.ndarray,
    solid_mask: np.ndarray,
    invalid_support: np.ndarray,
    filled_interior: np.ndarray,
    unknown_band: np.ndarray,
    negative_candidates: np.ndarray,
    evidence_state: np.ndarray,
    raw_contours: Sequence[Mapping[str, Any]],
    filtering_decisions: Sequence[Mapping[str, Any]],
    selected_contours: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "velocity_u": np.asarray(u, dtype=np.float64).copy(),
        "velocity_v": np.asarray(v, dtype=np.float64).copy(),
        "xdim": np.asarray(xdim, dtype=np.float64).copy(),
        "ydim": np.asarray(ydim, dtype=np.float64).copy(),
        "omega": np.asarray(omega, dtype=np.float64).copy(),
        "raw_scalar_field": np.asarray(raw_scalar_field, dtype=np.float64).copy(),
        "standard_ivd": np.asarray(raw_scalar_field, dtype=np.float64).copy(),
        "solid_mask": np.asarray(solid_mask, dtype=bool).copy(),
        "invalid_support": np.asarray(invalid_support, dtype=bool).copy(),
        "valid_fluid": (~np.asarray(solid_mask, dtype=bool) &
                        ~np.asarray(invalid_support, dtype=bool)),
        "filled_interior": np.asarray(filled_interior, dtype=bool).copy(),
        "unknown_band": np.asarray(unknown_band, dtype=bool).copy(),
        "negative_candidates": np.asarray(negative_candidates, dtype=bool).copy(),
        "evidence_state": np.asarray(evidence_state, dtype=np.int8).copy(),
    }
    shape = arrays["evidence_state"].shape
    if any(
        np.asarray(arrays[name]).shape != shape
        for name in _ARRAY_FILES
        if name not in {"xdim", "ydim"}
    ) or arrays["xdim"].ndim != 1 or arrays["ydim"].ndim != 1:
        raise ValueError("Haller evidence arrays shape 不一致")
    json_layers = {
        "raw_contours": _json_copy(list(raw_contours)),
        "filtering_decisions": _json_copy(list(filtering_decisions)),
        "selected_contours": _json_copy(list(selected_contours)),
    }
    result_metadata = _json_copy(dict(metadata))
    counts = _layer_counts(arrays)
    candidate_counts, rejection_reason_histogram = _candidate_accounting(
        filtering_decisions
    )
    raw_candidate_count = int(sum(
        _candidate_weight(record) for record in json_layers["raw_contours"]
    ))
    if candidate_counts["total"] != raw_candidate_count:
        raise ValueError(
            "Haller evidence filtering candidate accounting 与 raw contour accounting 不一致"
        )
    result_metadata.update({
        "raw_contour_count": int(len(json_layers["raw_contours"])),
        "raw_candidate_count": raw_candidate_count,
        "candidate_counts": candidate_counts,
        "filtering_decision_count": int(len(json_layers["filtering_decisions"])),
        "selected_contour_count": int(len(json_layers["selected_contours"])),
        "rejection_reason_histogram": rejection_reason_histogram,
        "layer_counts": counts,
        "coverage": {
            **counts,
            "known_fraction_valid_fluid": float(
                counts["known_cells"] / counts["valid_fluid_cells"]
            ) if counts["valid_fluid_cells"] else None,
            "positive_fraction_valid_fluid": float(
                counts["positive_cells"] / counts["valid_fluid_cells"]
            ) if counts["valid_fluid_cells"] else None,
            "negative_fraction_valid_fluid": float(
                counts["negative_cells"] / counts["valid_fluid_cells"]
            ) if counts["valid_fluid_cells"] else None,
            "unknown_fraction_valid_fluid": float(
                counts["unknown_cells"] / counts["valid_fluid_cells"]
            ) if counts["valid_fluid_cells"] else None,
        },
        "layer_conservation": _layer_conservation(arrays),
        "floating_summary": {
            "raw_scalar_finite": bool(np.all(np.isfinite(arrays["raw_scalar_field"]))),
            "raw_scalar_min": (
                float(np.nanmin(arrays["raw_scalar_field"]))
                if np.any(np.isfinite(arrays["raw_scalar_field"])) else None
            ),
            "raw_scalar_max": (
                float(np.nanmax(arrays["raw_scalar_field"]))
                if np.any(np.isfinite(arrays["raw_scalar_field"])) else None
            ),
        },
    })
    result_metadata["array_hashes"] = _array_hashes(arrays)
    result_metadata["json_hashes"] = _json_hashes(json_layers)
    result_metadata["integrity_fingerprint"] = _integrity_fingerprint(
        result_metadata, arrays, json_layers
    )
    return {
        **arrays,
        **json_layers,
        "metadata": result_metadata,
        # Keep the original Python-scope binding outside persisted metadata so
        # save() can detect a caller relabeling both source and split.
        "_scope_identity": {
            "dataset_name": result_metadata["dataset_name"],
            "split_name": result_metadata["split_name"],
            "frame_index": result_metadata["frame_index"],
            "source": result_metadata["source"],
        },
    }


def _invalid_reason(exc: BaseException) -> str:
    message = str(exc) or type(exc).__name__
    tags = ["invalid_input", type(exc).__name__]
    if "有限" in message or "finite" in message.lower() or "nan" in message.lower():
        tags.append("nonfinite")
    tags.append(message)
    return ":".join(tags)


def _failure_reason(exc: BaseException, *, prefix: str = "extractor_failure") -> str:
    return f"{prefix}:{type(exc).__name__}:{str(exc) or type(exc).__name__}"


def _empty_layers(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    false = np.zeros(shape, dtype=bool)
    return false.copy(), false.copy(), false.copy(), np.full(shape, strict.UNKNOWN, dtype=np.int8)


def trace_haller_evidence(
    u: Any,
    v: Any,
    xdim: Any,
    ydim: Any,
    mask: Any = None,
    *,
    dataset_name: str,
    split_name: str,
    frame_index: int,
    source: str | None = None,
    backend: str = strict.BACKEND_NUMPY,
    device: Any = None,
    contour_mode: str = strict.CONTOUR_MODE_OPTIMIZED,
    fast_global_level_count: int = 64,
    fast_refinement_iterations: int = 7,
    fast_refinement_halo_cells: int = 2,
    numbacs_radius_factor: float = 1.0,
    protocol_manifest: Mapping[str, Any] | str | pathlib.Path | None = None,
    freeze_record: Mapping[str, Any] | str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Trace one train/calibration/development-audit frame.

    The function returns a complete in-memory bundle.  It deliberately allows
    synthetic dataset names when no protocol manifest is supplied, while a
    supplied R0 manifest validates six-dataset frame membership and the
    Boussinesq audit frame rule before numerical work starts.
    """
    resolved_source, frame, manifest, _ = _validate_scope(
        dataset_name,
        split_name,
        frame_index,
        source,
        protocol_manifest,
    )
    freeze = _load_freeze_record(freeze_record)
    details, resolved_mode, params = _resolve_configuration(
        backend,
        device,
        contour_mode,
        fast_global_level_count,
        fast_refinement_iterations,
        fast_refinement_halo_cells,
        numbacs_radius_factor,
    )

    shape = _safe_shape(u, v, mask)
    safe_u = _safe_storage_array(u, shape, dtype=np.float64, fill_value=np.nan)
    safe_v = _safe_storage_array(v, shape, dtype=np.float64, fill_value=np.nan)
    safe_x = _safe_axis(xdim, shape[1])
    safe_y = _safe_axis(ydim, shape[0])
    safe_solid = _safe_storage_array(mask, shape, dtype=bool, fill_value=False)

    try:
        u2, v2, x, y, solid, dx, dy = _prepare_valid_inputs(u, v, xdim, ydim, mask)
    except Exception as exc:
        invalid = np.ones(shape, dtype=bool)
        raw = np.full(shape, np.nan, dtype=np.float64)
        omega = np.full(shape, np.nan, dtype=np.float64)
        interior, band, negative, state = _empty_layers(shape)
        metadata = _base_metadata(
            dataset_name=_text(dataset_name, name="dataset_name"),
            split_name=_text(split_name, name="split_name"),
            frame_index=frame,
            source=resolved_source,
            u=safe_u,
            v=safe_v,
            xdim=safe_x,
            ydim=safe_y,
            solid=safe_solid,
            invalid_support=invalid,
            details=details,
            contour_mode=resolved_mode,
            params=params,
            status=STATUS_INVALID_INPUT,
            status_reason=_invalid_reason(exc),
            failure_count=1,
            failure_reasons=[_invalid_reason(exc)],
            contour_evaluation={},
            frame_fluid_p60=None,
            protocol_manifest=manifest,
            freeze_record=freeze,
        )
        return _assemble_result(
            u=safe_u, v=safe_v, xdim=safe_x, ydim=safe_y, omega=omega,
            raw_scalar_field=raw, solid_mask=safe_solid, invalid_support=invalid,
            filled_interior=interior, unknown_band=band,
            negative_candidates=negative, evidence_state=state,
            raw_contours=[], filtering_decisions=[], selected_contours=[],
            metadata=metadata,
        )

    valid_fluid = ~solid
    if not np.any(valid_fluid):
        raw = np.zeros(u2.shape, dtype=np.float64)
        omega = np.zeros_like(raw)
        interior, band, negative, state = _empty_layers(raw.shape)
        metadata = _base_metadata(
            dataset_name=_text(dataset_name, name="dataset_name"),
            split_name=_text(split_name, name="split_name"),
            frame_index=frame,
            source=resolved_source,
            u=u2, v=v2, xdim=x, ydim=y, solid=solid,
            invalid_support=np.zeros_like(solid), details=details,
            contour_mode=resolved_mode, params=params,
            status=STATUS_ALL_UNKNOWN,
            status_reason="all_support_solid",
            failure_count=0,
            failure_reasons=["all_support_solid"],
            contour_evaluation={"empty_reason": "all_support_solid"},
            frame_fluid_p60=None, protocol_manifest=manifest,
            freeze_record=freeze,
        )
        return _assemble_result(
            u=u2, v=v2, xdim=x, ydim=y, omega=omega, raw_scalar_field=raw,
            solid_mask=solid, invalid_support=np.zeros_like(solid),
            filled_interior=interior, unknown_band=band,
            negative_candidates=negative, evidence_state=state,
            raw_contours=[], filtering_decisions=[], selected_contours=[],
            metadata=metadata,
        )

    try:
        details_ivd = strict._compute_standard_ivd_details(
            u2, v2, x, y, solid, backend=details.resolved, device=details.device
        )
    except Exception as exc:
        raw = np.full(u2.shape, np.nan, dtype=np.float64)
        omega = np.full(u2.shape, np.nan, dtype=np.float64)
        interior, band, negative, state = _empty_layers(raw.shape)
        input_finite = np.isfinite(u2) & np.isfinite(v2)
        invalid_input = (~solid) & ~input_finite
        if not np.any((~solid) & input_finite):
            # The shared standard-IVD source rejects a frame with no finite
            # valid-fluid support.  Preserve that as an input audit failure,
            # rather than misclassifying it as a contour/extractor failure.
            invalid_input = ~solid
            reason = _invalid_reason(ValueError("nonfinite velocity input"))
            status = STATUS_INVALID_INPUT
        else:
            reason = _failure_reason(exc, prefix="ivd_compute_failure")
            status = STATUS_EXTRACTOR_FAILURE
        metadata = _base_metadata(
            dataset_name=_text(dataset_name, name="dataset_name"),
            split_name=_text(split_name, name="split_name"),
            frame_index=frame, source=resolved_source,
            u=u2, v=v2, xdim=x, ydim=y, solid=solid,
            invalid_support=invalid_input, details=details,
            contour_mode=resolved_mode, params=params,
            status=status, status_reason=reason,
            failure_count=1, failure_reasons=[reason],
            contour_evaluation={}, frame_fluid_p60=None,
            protocol_manifest=manifest, freeze_record=freeze,
        )
        return _assemble_result(
            u=u2, v=v2, xdim=x, ydim=y, omega=omega, raw_scalar_field=raw,
            solid_mask=solid, invalid_support=invalid_input,
            filled_interior=interior, unknown_band=band,
            negative_candidates=negative, evidence_state=state,
            raw_contours=[], filtering_decisions=[{
                "decision": "error", "status": "error", "reason": reason,
            }], selected_contours=[], metadata=metadata,
        )

    raw = np.asarray(details_ivd.ivd, dtype=np.float64).copy()
    omega = np.asarray(details_ivd.omega, dtype=np.float64).copy()
    valid_fluid = np.asarray(details_ivd.valid_fluid_mask, dtype=bool)
    invalid_support = np.asarray(details_ivd.invalid_mask, dtype=bool)
    excluded_support = solid | invalid_support | ~valid_fluid
    fluid_values = raw[valid_fluid]
    frame_p60 = float(np.percentile(fluid_values, float(params["negative_percentile"])))
    try:
        peaks = strict.find_local_maxima(
            raw, excluded_support, backend=details.resolved, device=details.device
        )
        selected, diagnostics, contour_evaluation = strict._select_contours(
            raw, excluded_support, x, y, peaks, params, dx, dy,
            backend=details.resolved,
            device=details.device,
            contour_mode=resolved_mode,
            fast_global_level_count=fast_global_level_count,
            fast_refinement_iterations=fast_refinement_iterations,
            fast_refinement_halo_cells=fast_refinement_halo_cells,
            return_evaluation=True,
        )
    except Exception as exc:
        reason = _failure_reason(exc)
        metadata = _base_metadata(
            dataset_name=_text(dataset_name, name="dataset_name"),
            split_name=_text(split_name, name="split_name"),
            frame_index=frame, source=resolved_source,
            u=u2, v=v2, xdim=x, ydim=y, solid=solid,
            invalid_support=invalid_support, details=details,
            contour_mode=resolved_mode, params=params,
            status=STATUS_EXTRACTOR_FAILURE, status_reason=reason,
            failure_count=1, failure_reasons=[reason],
            contour_evaluation={}, frame_fluid_p60=frame_p60,
            protocol_manifest=manifest, freeze_record=freeze,
        )
        return _assemble_result(
            u=u2, v=v2, xdim=x, ydim=y, omega=omega, raw_scalar_field=raw,
            solid_mask=solid, invalid_support=invalid_support,
            filled_interior=np.zeros_like(solid), unknown_band=np.zeros_like(solid),
            negative_candidates=np.zeros_like(solid),
            evidence_state=np.full(solid.shape, strict.UNKNOWN, dtype=np.int8),
            raw_contours=[], filtering_decisions=[{
                "decision": "error", "status": "error", "reason": reason,
            }], selected_contours=[], metadata=metadata,
        )

    raw_records = _raw_contour_records(selected, diagnostics)
    filtering = _filtering_decisions(raw_records)
    if selected:
        status = STATUS_VALID
        status_reason = "selected_contours"
        failure_reasons: list[str] = []
    elif np.isclose(float(np.max(fluid_values)), frame_p60, rtol=0.0, atol=1e-15):
        status = STATUS_VALID_ALL_NEGATIVE
        status_reason = "no_vortex_evidence"
        failure_reasons = ["no_vortex_evidence"]
    else:
        status = STATUS_VALID_EMPTY
        status_reason = "no_legal_contour"
        failure_reasons = ["no_legal_contour"]
        if not peaks:
            failure_reasons.append("no_local_maximum")

    try:
        if selected:
            filled, band = strict._rasterize_union(
                selected,
                raw.shape,
                dx,
                dy,
                float(params["unknown_band_factor"]) * max(dx, dy),
            )
        else:
            filled = np.zeros(raw.shape, dtype=bool)
            band = np.zeros(raw.shape, dtype=bool)
        negative = valid_fluid & ~filled & ~band & (raw <= frame_p60)
        state = np.full(raw.shape, strict.UNKNOWN, dtype=np.int8)
        state[negative] = strict.NEGATIVE
        state[valid_fluid & filled & ~band] = strict.POSITIVE
    except Exception as exc:
        reason = _failure_reason(exc, prefix="rasterization_failure")
        metadata = _base_metadata(
            dataset_name=_text(dataset_name, name="dataset_name"),
            split_name=_text(split_name, name="split_name"),
            frame_index=frame, source=resolved_source,
            u=u2, v=v2, xdim=x, ydim=y, solid=solid,
            invalid_support=invalid_support, details=details,
            contour_mode=resolved_mode, params=params,
            status=STATUS_EXTRACTOR_FAILURE, status_reason=reason,
            failure_count=1, failure_reasons=[reason],
            contour_evaluation=contour_evaluation, frame_fluid_p60=frame_p60,
            protocol_manifest=manifest, freeze_record=freeze,
        )
        return _assemble_result(
            u=u2, v=v2, xdim=x, ydim=y, omega=omega, raw_scalar_field=raw,
            solid_mask=solid, invalid_support=invalid_support,
            filled_interior=np.zeros_like(solid), unknown_band=np.zeros_like(solid),
            negative_candidates=np.zeros_like(solid),
            evidence_state=np.full(solid.shape, strict.UNKNOWN, dtype=np.int8),
            raw_contours=raw_records, filtering_decisions=filtering,
            selected_contours=selected, metadata=metadata,
        )

    metadata = _base_metadata(
        dataset_name=_text(dataset_name, name="dataset_name"),
        split_name=_text(split_name, name="split_name"),
        frame_index=frame, source=resolved_source,
        u=u2, v=v2, xdim=x, ydim=y, solid=solid,
        invalid_support=invalid_support, details=details,
        contour_mode=resolved_mode, params=params,
        status=status, status_reason=status_reason,
        failure_count=0, failure_reasons=failure_reasons,
        contour_evaluation=contour_evaluation, frame_fluid_p60=frame_p60,
        protocol_manifest=manifest, freeze_record=freeze,
    )
    return _assemble_result(
        u=u2, v=v2, xdim=x, ydim=y, omega=omega, raw_scalar_field=raw,
        solid_mask=solid, invalid_support=invalid_support,
        filled_interior=filled, unknown_band=band,
        negative_candidates=negative, evidence_state=state,
        raw_contours=raw_records, filtering_decisions=filtering,
        selected_contours=selected, metadata=metadata,
    )


def _required_metadata(metadata: Mapping[str, Any]) -> None:
    required = (
        "artifact_type", "format_version", "artifact_state", "trusted",
        "dataset_name", "split_name", "frame_index", "source", "scope_binding",
        "status",
        "geometry", "input_fingerprint", "generator_identity",
        "runtime_identity", "parameters", "parameter_hash",
        "raw_contour_count", "raw_candidate_count",
        "filtering_decision_count", "selected_contour_count",
        "candidate_counts", "rejection_reason_histogram",
        "layer_counts", "integrity_fingerprint",
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"Haller evidence metadata 不完整：{missing!r}")
    if metadata["artifact_type"] != "haller_ivd_layered_evidence":
        raise ValueError("Haller evidence artifact_type 不匹配")
    if metadata["format_version"] != EVIDENCE_FORMAT_VERSION:
        raise ValueError("Haller evidence format_version 不受支持")
    if metadata["artifact_state"] not in {
        ARTIFACT_STATE_CANDIDATE, ARTIFACT_STATE_AUDITED, ARTIFACT_STATE_FROZEN,
    }:
        raise ValueError("Haller evidence artifact_state 无效")
    if metadata["status"] not in _VALID_STATUSES:
        raise ValueError("Haller evidence status 无效")
    frame_index = _coerce_exact_frame_index(metadata["frame_index"])
    binding = metadata["scope_binding"]
    if not isinstance(binding, Mapping):
        raise ValueError("Haller evidence scope_binding 不完整")
    expected_binding = {
        "dataset_name": metadata["dataset_name"],
        "split_name": str(metadata["split_name"]),
        "frame_index": frame_index,
        "source": metadata["source"],
    }
    if any(binding.get(key) != value for key, value in expected_binding.items()):
        raise ValueError("Haller evidence scope_binding 与 metadata identity 不一致")
    split_name = str(metadata["split_name"])
    if split_name == "test" or metadata["source"] == strict.SOURCE_TEST:
        raise PermissionError("evidence loader 拒绝 final-test artifact")
    expected_source = _SPLIT_TO_SOURCE.get(split_name)
    if expected_source is None:
        raise ValueError(f"Haller evidence split_name 无效：{split_name!r}")
    strict._validate_source(str(metadata["source"]))
    if metadata["source"] != expected_source:
        raise ValueError(
            f"Haller evidence source={metadata['source']!r} 与 split={split_name!r} 不匹配"
        )
    if metadata["dataset_name"] in _PROTOCOL_DATASET_NAMES:
        protocol = metadata.get("protocol")
        if not isinstance(protocol, Mapping):
            raise ValueError(
                "real Haller evidence artifact 缺少 R0 protocol identity"
            )
        expected_identity = {
            "dataset_name": metadata["dataset_name"],
            "split_name": split_name,
            "frame_index": int(metadata["frame_index"]),
        }
        if any(protocol.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("Haller evidence protocol identity 与 dataset/split/frame 不一致")
        if protocol.get("access_state") != "development_only":
            raise ValueError("Haller evidence real dataset 不是 development-only protocol artifact")
        if protocol.get("eligible_for_test_relationships") is not False:
            raise ValueError("Haller evidence protocol artifact 不能进入 test relationships")
    if metadata["parameter_hash"] != _json_hash(metadata["parameters"]):
        raise ValueError("Haller evidence parameter_hash 校验失败")


def _validate_artifact_protocol_identity(
    metadata: Mapping[str, Any],
    protocol_manifest: Mapping[str, Any] | str | pathlib.Path | None,
) -> None:
    """Bind a real artifact read to the caller's validated R0 manifest."""
    if metadata["dataset_name"] not in _PROTOCOL_DATASET_NAMES:
        return
    if protocol_manifest is None:
        raise ValueError(
            "读取 real Haller evidence artifact 必须提供 R0 protocol manifest"
        )
    manifest = _load_protocol_manifest(protocol_manifest)
    if manifest is None or manifest["manifest_hash"] != metadata["protocol"]["manifest_hash"]:
        raise ValueError("Haller evidence artifact 与提供的 R0 manifest hash 不匹配")
    dataset_record = manifest["datasets"][metadata["dataset_name"]]
    split_name = metadata["split_name"]
    frame_index = int(metadata["frame_index"])
    if split_name == "development_audit":
        allowed = set(dataset_record.get("development_audit_frames", ()))
        if frame_index not in allowed:
            raise ValueError(
                "Haller evidence artifact frame 不在 R0 development_audit manifest"
            )
    else:
        ranges = dataset_record["frame_partitions"][split_name]["ranges"]
        if not any(int(start) <= frame_index < int(end) for start, end in ranges):
            raise ValueError("Haller evidence artifact frame 不在 R0 split manifest")


def _validate_persisted_round_trip(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the save-time measurement persisted in artifact metadata."""
    round_trip = metadata.get("round_trip")
    if not isinstance(round_trip, Mapping):
        raise ValueError("Haller evidence artifact 缺少 round_trip measurement")
    if round_trip.get("comparison") != (
        "pre_save_or_first_read_vs_independent_disk_readback"
    ):
        raise ValueError("Haller evidence round_trip comparison provenance 无效")
    if round_trip.get("categorical_max_difference_cells") != 0:
        raise ValueError("Haller evidence categorical round_trip 超出 0 cells")
    floating = round_trip.get("floating_max_abs_difference")
    if not isinstance(floating, (int, float)) or not np.isfinite(float(floating)):
        raise ValueError("Haller evidence floating round_trip measurement 无效")
    if float(floating) > 1e-6:
        raise ValueError("Haller evidence floating round_trip 超过 1e-6")
    if round_trip.get("json_layers_equal") is not True:
        raise ValueError("Haller evidence JSON round_trip 不一致")
    return dict(round_trip)


def _validate_bound_scope(result: Mapping[str, Any]) -> None:
    """Reject relabeling of a tracer result before persisting it."""
    binding = result.get("_scope_identity")
    metadata = result.get("metadata")
    if not isinstance(binding, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("Haller evidence result 缺少 immutable scope binding")
    expected = {
        "dataset_name": metadata.get("dataset_name"),
        "split_name": metadata.get("split_name"),
        "frame_index": _coerce_exact_frame_index(metadata.get("frame_index")),
        "source": metadata.get("source"),
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Haller evidence result scope binding source/split/frame 与 metadata 不一致"
        )


def _normalise_for_artifact(
    result: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, list[Any]], dict[str, Any]]:
    if not isinstance(result, Mapping) or "metadata" not in result:
        raise ValueError("Haller evidence result 缺少 metadata")
    metadata = _json_copy(dict(result["metadata"]))
    _required_metadata(metadata)
    geometry = metadata["geometry"]
    shape_value = geometry.get("shape")
    if not isinstance(shape_value, (list, tuple)) or len(shape_value) != 2:
        raise ValueError("Haller evidence metadata.geometry 缺少二维 shape")
    shape = (int(shape_value[0]), int(shape_value[1]))
    arrays: dict[str, np.ndarray] = {}
    dtype_by_name = {
        **{name: np.float64 for name in _ARRAY_FILES if name not in _CATEGORICAL_ARRAYS},
        **{name: (np.int8 if name == "evidence_state" else bool)
           for name in _CATEGORICAL_ARRAYS},
    }
    for name in _ARRAY_FILES:
        if name not in result:
            raise ValueError(f"incomplete Haller evidence result: missing {name}")
        array = np.asarray(result[name], dtype=dtype_by_name[name])
        expected_shape = (
            (shape[1],) if name == "xdim" else
            (shape[0],) if name == "ydim" else shape
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"Haller evidence {name} shape={array.shape} != {expected_shape}"
            )
        arrays[name] = array.copy()
    state = arrays["evidence_state"]
    if not np.all(np.isin(state, [strict.POSITIVE, strict.NEGATIVE, strict.UNKNOWN])):
        raise ValueError("Haller evidence evidence_state 含未知编码")
    for name in _CATEGORICAL_ARRAYS - {"evidence_state"}:
        if not np.all(np.isin(arrays[name], [False, True])):
            raise ValueError(f"Haller evidence {name} 必须是 bool")
    if metadata["status"] != STATUS_INVALID_INPUT and metadata["status"] != STATUS_EXTRACTOR_FAILURE:
        for name in ("velocity_u", "velocity_v", "xdim", "ydim", "omega", "raw_scalar_field", "standard_ivd"):
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"Haller evidence {name} 包含非有限值")
    json_layers: dict[str, list[Any]] = {}
    for name in _JSON_FILES:
        if name not in result or not isinstance(result[name], list):
            raise ValueError(f"incomplete Haller evidence result: missing {name}")
        json_layers[name] = _json_copy(result[name])
    _validate_candidate_accounting(
        metadata,
        json_layers["raw_contours"],
        json_layers["filtering_decisions"],
        json_layers["selected_contours"],
    )
    if not np.array_equal(arrays["raw_scalar_field"], arrays["standard_ivd"], equal_nan=True):
        raise ValueError("raw_scalar_field 与 standard_ivd 不一致")
    if not _layer_conservation(arrays):
        raise ValueError("Haller evidence layer conservation 失败")
    return arrays, json_layers, metadata


def save_haller_evidence_artifact(
    result: Mapping[str, Any],
    out_dir: str | pathlib.Path,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Persist a complete candidate bundle and publish metadata last.

    Artifact directories are immutable by default.  Even with ``overwrite``
    set, an existing metadata file is rejected; callers should create a new
    versioned directory instead of silently replacing evidence.
    """
    _validate_bound_scope(result)
    arrays, json_layers, metadata = _normalise_for_artifact(result)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_filenames = (*_ARRAY_FILES.values(), *_JSON_FILES.values(), "evidence_meta.json")
    if any(out.iterdir()):
        raise FileExistsError(
            f"Haller evidence artifact 目录非空或已存在；请使用新的 versioned path：{out}"
        )
    metadata["array_hashes"] = _array_hashes(arrays)
    metadata["json_hashes"] = _json_hashes(json_layers)
    staging = pathlib.Path(tempfile.mkdtemp(
        prefix=".haller-evidence-staging-",
        dir=str(out),
    ))
    published: list[pathlib.Path] = []
    round_trip: dict[str, Any]
    try:
        for name, filename in _ARRAY_FILES.items():
            strict._atomic_save_npy(staging / filename, arrays[name])
        for name, filename in _JSON_FILES.items():
            # Preserve the established layer contract: each JSON layer is a
            # bare list of records.  The loader also accepts the wrapped shape
            # for callers that provide the same current schema in that form.
            strict._atomic_write_json(staging / filename, json_layers[name])
        round_trip = _round_trip_metrics(staging, arrays, json_layers)
        metadata["round_trip"] = round_trip
        _validate_persisted_round_trip(metadata)
        metadata["integrity_fingerprint"] = _integrity_fingerprint(
            metadata, arrays, json_layers
        )
        strict._atomic_write_json(staging / "evidence_meta.json", metadata)
        for filename in final_filenames:
            staged = staging / filename
            destination = out / filename
            staged.replace(destination)
            published.append(destination)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if isinstance(result, dict):
        result["metadata"] = _json_copy(metadata)
        result["_round_trip_metrics"] = dict(round_trip)
    paths = {name: str(out / filename) for name, filename in _ARRAY_FILES.items()}
    paths.update({name: str(out / filename) for name, filename in _JSON_FILES.items()})
    paths["metadata"] = str(out / "evidence_meta.json")
    return paths


def _read_json_records(path: pathlib.Path, *, name: str) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"incomplete Haller evidence artifact: missing {name}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Haller evidence {name} JSON 无法读取") from exc
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("records"), list):
        return value["records"]
    raise ValueError(f"Haller evidence {name} 必须是 records array")


def load_haller_evidence_artifact(
    out_dir: str | pathlib.Path,
    *,
    require_trusted: bool = False,
    protocol_manifest: Mapping[str, Any] | str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Load, hash-check, and conservation-check one evidence bundle."""
    out = pathlib.Path(out_dir)
    metadata_path = out / "evidence_meta.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"incomplete Haller evidence artifact: missing evidence_meta.json") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Haller evidence metadata JSON 无法读取") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("Haller evidence metadata 必须是 JSON object")
    metadata = _json_copy(dict(metadata))
    _required_metadata(metadata)
    _validate_artifact_protocol_identity(metadata, protocol_manifest)
    persisted_round_trip = _validate_persisted_round_trip(metadata)
    shape_value = metadata["geometry"].get("shape")
    shape = (int(shape_value[0]), int(shape_value[1]))
    arrays: dict[str, np.ndarray] = {}
    dtype_by_name = {
        **{name: np.float64 for name in _ARRAY_FILES if name not in _CATEGORICAL_ARRAYS},
        **{name: (np.int8 if name == "evidence_state" else bool)
           for name in _CATEGORICAL_ARRAYS},
    }
    for name, filename in _ARRAY_FILES.items():
        path = out / filename
        try:
            array = np.load(path, allow_pickle=False)
        except FileNotFoundError as exc:
            raise ValueError(f"incomplete Haller evidence artifact: missing {filename}") from exc
        expected_shape = (
            (shape[1],) if name == "xdim" else
            (shape[0],) if name == "ydim" else shape
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"Haller evidence {name} shape={array.shape} != {expected_shape}"
            )
        if array.dtype != np.dtype(dtype_by_name[name]):
            raise ValueError(
                f"Haller evidence {name} dtype={array.dtype} != {np.dtype(dtype_by_name[name])}"
            )
        arrays[name] = array
    json_layers = {
        name: _read_json_records(out / filename, name=filename)
        for name, filename in _JSON_FILES.items()
    }
    _verify_integrity_fields(metadata, arrays, json_layers)
    result = {**arrays, **json_layers, "metadata": metadata}
    normalised_arrays, normalised_json, normalised_metadata = _normalise_for_artifact(result)
    stored_counts = normalised_metadata.get("layer_counts")
    recomputed_counts = _layer_counts(normalised_arrays)
    if not isinstance(stored_counts, Mapping) or dict(stored_counts) != recomputed_counts:
        raise ValueError("Haller evidence stored layer counts 与重算不一致")
    result["_round_trip_metrics"] = persisted_round_trip
    if require_trusted:
        require_trusted_haller_evidence(
            result, protocol_manifest=protocol_manifest
        )
    return result


def summarize_haller_evidence(
    artifact: Mapping[str, Any] | str | pathlib.Path,
    *,
    protocol_manifest: Mapping[str, Any] | str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Recompute counts and conservation from the persisted layers."""
    loaded = (
        load_haller_evidence_artifact(artifact, protocol_manifest=protocol_manifest)
        if isinstance(artifact, (str, pathlib.Path))
        else artifact
    )
    arrays, json_layers, metadata = _normalise_for_artifact(loaded)
    _validate_artifact_protocol_identity(metadata, protocol_manifest)
    counts = _layer_counts(arrays)
    stored = metadata.get("layer_counts", {})
    count_differences = {
        name: (stored.get(name), value)
        for name, value in counts.items()
        if stored.get(name) != value
    }
    round_trip = loaded.get("_round_trip_metrics", metadata.get("round_trip"))
    if not isinstance(round_trip, Mapping):
        round_trip = {
            "comparison": "not_available_for_in_memory_bundle",
            "categorical_max_difference_cells": None,
            "floating_max_abs_difference": None,
            "json_layers_equal": None,
        }
    summary = {
        "status": metadata["status"],
        "status_reason": metadata["status_reason"],
        "dataset_name": metadata["dataset_name"],
        "split_name": metadata["split_name"],
        "frame_index": metadata["frame_index"],
        "raw_contour_count": len(json_layers["raw_contours"]),
        "selected_contour_count": len(json_layers["selected_contours"]),
        "layer_counts": counts,
        "layer_conservation": _layer_conservation(arrays),
        "stored_vs_recomputed": {
            "count_differences": count_differences,
            "floating_max_abs_difference": 0.0,
        },
        "round_trip": dict(round_trip),
        "integrity_fingerprint": metadata["integrity_fingerprint"],
    }
    return summary


def require_trusted_haller_evidence(
    artifact: Mapping[str, Any] | str | pathlib.Path,
    *,
    protocol_manifest: Mapping[str, Any] | str | pathlib.Path | None = None,
    freeze_manifest: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Require the appropriate independent trust gate for Haller evidence.

    R1 emits only candidate artifacts and remains rejected by this path.  The
    exact versioned Haller-v2 producer is delegated to its own freeze-manifest
    verifier; metadata flags or a self-consistent authorization object are
    never sufficient.
    """
    if isinstance(artifact, (str, pathlib.Path)):
        artifact_path = pathlib.Path(artifact)
        if (artifact_path / "haller_v2_meta.json").exists():
            from haller_v2_postprocessing import require_trusted_haller_v2

            return require_trusted_haller_v2(artifact_path, freeze_manifest)
    if isinstance(artifact, Mapping):
        candidate_metadata = artifact.get("metadata")
        if isinstance(candidate_metadata, Mapping) and candidate_metadata.get("artifact_type") == (
            "haller_v2_layered_evidence"
        ):
            from haller_v2_postprocessing import require_trusted_haller_v2

            artifact_dir = artifact.get("artifact_dir")
            if not isinstance(artifact_dir, (str, pathlib.Path)):
                raise PermissionError(
                    "loaded Haller-v2 mapping lacks its immutable artifact path"
                )
            return require_trusted_haller_v2(artifact_dir, freeze_manifest)
    if isinstance(artifact, (str, pathlib.Path)):
        loaded = load_haller_evidence_artifact(
            artifact,
            require_trusted=False,
            protocol_manifest=protocol_manifest,
        )
    elif isinstance(artifact, Mapping):
        arrays, json_layers, metadata = _normalise_for_artifact(artifact)
        _verify_integrity_fields(metadata, arrays, json_layers)
        loaded = dict(artifact)
    else:
        raise ValueError("Haller evidence artifact 类型不受支持，不能声明 trusted")
    if not isinstance(loaded, Mapping) or not isinstance(loaded.get("metadata"), Mapping):
        raise ValueError("Haller evidence artifact 不完整，不能声明 trusted")
    metadata = loaded["metadata"]
    _validate_artifact_protocol_identity(metadata, protocol_manifest)
    if (
        metadata.get("status") not in {
            STATUS_VALID,
            STATUS_VALID_EMPTY,
            STATUS_VALID_ALL_NEGATIVE,
        }
        or metadata.get("valid") is not True
        or metadata.get("frame_valid") is not True
        or metadata.get("failure_count") != 0
    ):
        raise PermissionError(
            "invalid/extractor_failure/all_unknown Haller evidence 不能声明 trusted"
        )
    # R1 has no independent R2 freeze producer.  A caller-supplied metadata
    # mapping must never be able to self-upgrade by adding frozen/trusted flags
    # and a self-consistent authorization object.
    if metadata.get("artifact_state") != ARTIFACT_STATE_FROZEN or metadata.get("trusted") is not True:
        raise PermissionError(
            "candidate/audited Haller evidence 不能声明 trusted；需要 exact R2 frozen artifact"
        )
    raise PermissionError(
        "R1 trust gate 已关闭；只有独立的 R2 audit/freeze producer 才能启用 trusted Haller evidence"
    )


def write_haller_evidence_trace_report(
    artifact_dirs: Sequence[str | pathlib.Path],
    report_path: str | pathlib.Path,
    *,
    protocol_manifest: Mapping[str, Any] | str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Read back artifacts and write a compact R1 trace report."""
    entries: list[dict[str, Any]] = []
    overall_status = "pass"
    for artifact_dir in artifact_dirs:
        try:
            loaded = load_haller_evidence_artifact(
                artifact_dir, protocol_manifest=protocol_manifest
            )
            summary = summarize_haller_evidence(
                loaded, protocol_manifest=protocol_manifest
            )
            entries.append({
                "artifact_dir": str(pathlib.Path(artifact_dir)),
                "dataset_name": loaded["metadata"]["dataset_name"],
                "split_name": loaded["metadata"]["split_name"],
                "frame_index": loaded["metadata"]["frame_index"],
                "status": summary["status"],
                "integrity_fingerprint": loaded["metadata"]["integrity_fingerprint"],
                "summary": summary,
            })
        except Exception as exc:
            overall_status = "fail"
            entries.append({
                "artifact_dir": str(pathlib.Path(artifact_dir)),
                "status": "readback_failure",
                "reason": f"{type(exc).__name__}:{exc}",
            })
    report = {
        "artifact_type": "haller_evidence_trace_report",
        "format_version": EVIDENCE_FORMAT_VERSION,
        "status": overall_status,
        "artifact_count": len(entries),
        "artifacts": entries,
        "final_test_relationships": "not_read; development-only tracer",
    }
    strict._atomic_write_json(pathlib.Path(report_path), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trace layered Haller evidence for one development frame")
    parser.add_argument("--u", required=True, help="2D velocity-u .npy")
    parser.add_argument("--v", required=True, help="2D velocity-v .npy")
    parser.add_argument("--xdim", required=True, help="1D x-coordinate .npy")
    parser.add_argument("--ydim", required=True, help="1D y-coordinate .npy")
    parser.add_argument("--mask", default=None, help="optional 2D solid mask .npy")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-name", required=True, choices=sorted(_DEVELOPMENT_SPLITS | {"test"}))
    parser.add_argument("--frame-index", required=True, type=int)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", choices=sorted(strict.VALID_BACKENDS), default=strict.BACKEND_NUMPY)
    parser.add_argument("--protocol-manifest", default=None)
    args = parser.parse_args(argv)
    # Reject final-test/ungoverned real-dataset identities before touching any
    # frame file.  trace_haller_evidence repeats the check after coercion for
    # callers using the Python API.
    _validate_scope(
        args.dataset_name,
        args.split_name,
        args.frame_index,
        None,
        args.protocol_manifest,
    )
    u = np.load(args.u, mmap_mode="r", allow_pickle=False)
    v = np.load(args.v, mmap_mode="r", allow_pickle=False)
    xdim = np.load(args.xdim, allow_pickle=False)
    ydim = np.load(args.ydim, allow_pickle=False)
    mask = None if args.mask is None else np.load(args.mask, allow_pickle=False)
    result = trace_haller_evidence(
        u, v, xdim, ydim, mask,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        frame_index=args.frame_index,
        backend=args.backend,
        protocol_manifest=args.protocol_manifest,
    )
    paths = save_haller_evidence_artifact(result, args.out_dir)
    loaded = load_haller_evidence_artifact(
        args.out_dir, protocol_manifest=args.protocol_manifest
    )
    print(json.dumps({
        "status": loaded["metadata"]["status"],
        "integrity_fingerprint": loaded["metadata"]["integrity_fingerprint"],
        "summary": summarize_haller_evidence(
            loaded, protocol_manifest=args.protocol_manifest
        ),
        "paths": paths,
    }, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "ARTIFACT_STATE_AUDITED",
    "ARTIFACT_STATE_CANDIDATE",
    "ARTIFACT_STATE_FROZEN",
    "EVIDENCE_FORMAT_VERSION",
    "STATUS_ALL_UNKNOWN",
    "STATUS_EXTRACTOR_FAILURE",
    "STATUS_INVALID_INPUT",
    "STATUS_VALID",
    "STATUS_VALID_ALL_NEGATIVE",
    "STATUS_VALID_EMPTY",
    "load_haller_evidence_artifact",
    "require_trusted_haller_evidence",
    "save_haller_evidence_artifact",
    "summarize_haller_evidence",
    "trace_haller_evidence",
    "write_haller_evidence_trace_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
