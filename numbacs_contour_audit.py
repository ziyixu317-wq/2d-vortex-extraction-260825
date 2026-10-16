"""R1 native NumbaCS raw-contour audit seams.

This module owns the audit orchestration and its small, JSON-auditable data
contract.  It deliberately keeps raw contour candidates separate from the
later Haller-v2 post-processing/freeze ticket.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import shutil
from statistics import median
from typing import Any

import numpy as np

import experiment_protocol
import haller_anchors
import numbacs_haller


AUDIT_SCHEMA_VERSION = "numbacs-raw-contour-audit-r1-v1"
PRIMARY_RADIUS_FACTORS = (1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class AuditConfig:
    """Pre-registered R1 audit grid and execution controls."""

    radius_factors: tuple[float, ...] = PRIMARY_RADIUS_FACTORS
    min_len_factors: tuple[float, ...] | None = None
    repetitions: int = 3
    min_len_grid_opened: bool = field(init=False)

    def __post_init__(self) -> None:
        factors = tuple(float(value) for value in self.radius_factors)
        if factors != PRIMARY_RADIUS_FACTORS:
            raise ValueError(
                "R1 primary radius grid is fixed at r/Delta=[1,2,4,8]"
            )
        if any(not np.isfinite(value) or value <= 0.0 for value in factors):
            raise ValueError("radius factors must be positive finite values")
        repetitions = int(self.repetitions)
        if repetitions != self.repetitions or repetitions != 3:
            raise ValueError("R1 determinism audit requires exactly three repetitions")
        object.__setattr__(self, "radius_factors", factors)
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "min_len_grid_opened", self.min_len_factors is not None)


def _diff_float64(field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    result = np.zeros_like(field, dtype=np.float64)
    if field.shape[axis] < 2:
        raise ValueError("standard IVD requires at least two points per coordinate axis")
    interior = [slice(None)] * field.ndim
    interior[axis] = slice(1, -1)
    result[tuple(interior)] = (
        np.take(field, np.arange(2, field.shape[axis]), axis=axis)
        - np.take(field, np.arange(0, field.shape[axis] - 2), axis=axis)
    ) / (2.0 * spacing)
    lower = [slice(None)] * field.ndim
    lower[axis] = 0
    upper = [slice(None)] * field.ndim
    upper[axis] = -1
    result[tuple(lower)] = (
        np.take(field, 1, axis=axis) - np.take(field, 0, axis=axis)
    ) / spacing
    result[tuple(upper)] = (
        np.take(field, -1, axis=axis) - np.take(field, -2, axis=axis)
    ) / spacing
    return result


def independent_standard_ivd(
    u: Any,
    v: Any,
    xdim: Any,
    ydim: Any,
    solid_mask: Any = None,
) -> np.ndarray:
    """Compute the float64 standard IVD independently of the extractor."""

    u_array = np.asarray(u, dtype=np.float64)
    v_array = np.asarray(v, dtype=np.float64)
    if u_array.ndim != 2 or v_array.shape != u_array.shape:
        raise ValueError("u/v must be matching two-dimensional arrays")
    x = np.asarray(xdim, dtype=np.float64)
    y = np.asarray(ydim, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != u_array.shape[1] or len(y) != u_array.shape[0]:
        raise ValueError("coordinate axes do not match the velocity field")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ValueError("coordinate spacing must be positive and finite")
    # Dataset metadata is serialized as float32 in several sources.  Keep the
    # same 1e-3 uniform-grid tolerance as the production Haller path while
    # still rejecting materially non-uniform axes.
    if not np.allclose(np.diff(x), dx, rtol=1e-3, atol=1e-12):
        raise ValueError("xdim must be uniformly spaced")
    if not np.allclose(np.diff(y), dy, rtol=1e-3, atol=1e-12):
        raise ValueError("ydim must be uniformly spaced")
    solid = np.zeros_like(u_array, dtype=bool) if solid_mask is None else np.asarray(
        solid_mask, dtype=bool
    )
    if solid.shape != u_array.shape:
        raise ValueError("solid_mask shape does not match the velocity field")
    fluid = ~solid
    if not np.any(fluid):
        raise ValueError("standard IVD requires at least one fluid cell")
    omega = _diff_float64(v_array, axis=1, spacing=dx) - _diff_float64(
        u_array, axis=0, spacing=dy
    )
    mean = float(omega[fluid].mean(dtype=np.float64))
    ivd = np.abs(omega - mean)
    ivd[solid] = 0.0
    return ivd


def summarize_raw_contours(
    records: list[Mapping[str, Any]],
    *,
    shape: tuple[int, int],
    xdim: Any,
    ydim: Any,
    solid_mask: Any,
) -> dict[str, Any]:
    """Add equivalent-radius and raster-overlap measurements to raw records."""

    solid = np.asarray(solid_mask, dtype=bool)
    if solid.shape != tuple(int(value) for value in shape):
        raise ValueError("solid_mask shape does not match contour raster shape")
    contour_masks: list[np.ndarray | None] = []
    summary_records: list[dict[str, Any]] = []
    for ordinal, source in enumerate(records):
        record = dict(source)
        points_value = record.get("points_grid")
        mask: np.ndarray | None = None
        if points_value is not None:
            points = np.asarray(points_value, dtype=np.float64)
            if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 4:
                mask = haller_anchors.grid_points_in_poly(tuple(shape), points) & ~solid
        contour_masks.append(mask)
        area_value = record.get("area")
        area = None
        if isinstance(area_value, (int, float, np.integer, np.floating)):
            area = float(area_value)
        equivalent_radius = (
            float(np.sqrt(area / np.pi)) if area is not None and area > 0.0 else None
        )
        summary_records.append({
            "raw_record_index": int(record.get("raw_record_index", ordinal)),
            "status": str(record.get("status", "unknown")),
            "area": area,
            "equivalent_radius": equivalent_radius,
            "center_xy": record.get("center_xy"),
            "rejection_reason": record.get("rejection_reason") or record.get("reason"),
            "geometry_available": mask is not None,
        })

    pairwise_overlap_count = 0
    pairwise_max_iou = 0.0
    max_overlap_by_record = [0.0 for _ in summary_records]
    for left in range(len(contour_masks)):
        if contour_masks[left] is None:
            continue
        for right in range(left + 1, len(contour_masks)):
            if contour_masks[right] is None:
                continue
            intersection = int(np.count_nonzero(contour_masks[left] & contour_masks[right]))
            if intersection == 0:
                continue
            union = int(np.count_nonzero(contour_masks[left] | contour_masks[right]))
            iou = float(intersection / union) if union else 0.0
            pairwise_overlap_count += 1
            pairwise_max_iou = max(pairwise_max_iou, iou)
            max_overlap_by_record[left] = max(max_overlap_by_record[left], iou)
            max_overlap_by_record[right] = max(max_overlap_by_record[right], iou)
    for record, overlap in zip(summary_records, max_overlap_by_record):
        record["max_overlap_iou"] = float(overlap)

    rejection_reasons: dict[str, int] = {}
    for record in summary_records:
        reason = record.get("rejection_reason")
        if reason:
            rejection_reasons[str(reason)] = rejection_reasons.get(str(reason), 0) + 1
    return {
        "count": len(summary_records),
        "records": summary_records,
        "pairwise_overlap_count": pairwise_overlap_count,
        "max_pairwise_overlap_iou": pairwise_max_iou,
        "rejection_reason_histogram": rejection_reasons,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _split_source(split_name: str) -> str:
    if split_name == "development_audit":
        return haller_anchors.SOURCE_DEVELOPMENT_AUDIT
    return _source_for_split(split_name)


def _load_dataset_frame(data_root: str | Path, sample: Mapping[str, Any]) -> dict[str, Any]:
    """Load only velocity, geometry, and mask for one manifest-bound frame."""

    root = Path(data_root)
    dataset_name = str(sample["dataset_name"])
    frame_index = int(sample["frame_index"])
    dataset_root = root / dataset_name
    data_dir = dataset_root / "dataset"
    meta_path = data_dir / "meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    u_path = data_dir / "u.npy"
    v_path = data_dir / "v.npy"
    mask_path = data_dir / "mask.npy"
    u_all = np.load(u_path, mmap_mode="r", allow_pickle=False)
    v_all = np.load(v_path, mmap_mode="r", allow_pickle=False)
    mask = np.asarray(np.load(mask_path, mmap_mode="r", allow_pickle=False), dtype=bool)
    if mask.ndim == 3:
        mask = mask[0]
    if u_all.ndim != 3 or v_all.shape != u_all.shape:
        raise ValueError(f"{dataset_name} velocity arrays must be matching (T,Y,X)")
    if not 0 <= frame_index < u_all.shape[0]:
        raise ValueError(f"{dataset_name} frame {frame_index} is outside the velocity array")
    if mask.shape != tuple(u_all.shape[1:]):
        raise ValueError(f"{dataset_name} mask shape {mask.shape} != {u_all.shape[1:]}")
    xdim = np.asarray(metadata["xdim"], dtype=np.float64)
    ydim = np.asarray(metadata["ydim"], dtype=np.float64)
    return {
        "u": u_all[frame_index],
        "v": v_all[frame_index],
        "xdim": xdim,
        "ydim": ydim,
        "mask": mask,
        "source_files": {
            "u": str(u_path),
            "v": str(v_path),
            "mask": str(mask_path),
            "metadata": str(meta_path),
        },
    }


def _ivd_comparison(actual: Any, expected: Any) -> dict[str, Any]:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        return {
            "status": "fail",
            "max_absolute_error": None,
            "relative_l2_error": None,
            "reason": "shape_mismatch",
        }
    finite = np.isfinite(actual_array) & np.isfinite(expected_array)
    if not np.all(finite):
        return {
            "status": "fail",
            "max_absolute_error": float("inf"),
            "relative_l2_error": float("inf"),
            "reason": "nonfinite_actual_or_expected",
        }
    difference = actual_array - expected_array
    max_absolute = float(np.max(np.abs(difference))) if difference.size else 0.0
    denominator = float(np.linalg.norm(expected_array.ravel()))
    numerator = float(np.linalg.norm(difference.ravel()))
    relative_l2 = numerator / denominator if denominator > 0.0 else (
        0.0 if numerator == 0.0 else float("inf")
    )
    return {
        "status": "pass" if max_absolute <= 1e-10 and relative_l2 <= 1e-8 else "fail",
        "max_absolute_error": max_absolute,
        "relative_l2_error": float(relative_l2),
        "reason": None,
    }


def _floating_summary(
    result: Mapping[str, Any],
    ivd_check: Mapping[str, Any],
) -> dict[str, float | None]:
    metadata = result["metadata"]
    floating = metadata.get("floating_summary", {})
    values: dict[str, float | None] = {}
    for name in ("raw_scalar_min", "raw_scalar_max"):
        value = floating.get(name)
        values[name] = None if value is None else float(value)
    p60 = metadata.get("frame_fluid_p60")
    values["frame_fluid_p60"] = None if p60 is None else float(p60)
    values["ivd_max_absolute_error"] = (
        None if ivd_check.get("max_absolute_error") is None
        else float(ivd_check["max_absolute_error"])
    )
    values["ivd_relative_l2_error"] = (
        None if ivd_check.get("relative_l2_error") is None
        else float(ivd_check["relative_l2_error"])
    )
    return values


def _frame_summary(
    result: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
    radius_factor: float,
    xdim: np.ndarray,
    ydim: np.ndarray,
    expected_ivd: np.ndarray,
) -> dict[str, Any]:
    metadata = result["metadata"]
    ivd_check = _ivd_comparison(result["standard_ivd"], expected_ivd)
    contour_summary = summarize_raw_contours(
        list(result.get("raw_contours", [])),
        shape=tuple(int(value) for value in result["standard_ivd"].shape),
        xdim=xdim,
        ydim=ydim,
        solid_mask=result["solid_mask"],
    )
    coverage = dict(metadata.get("coverage", {}))
    rejection_histogram = dict(metadata.get("rejection_reason_histogram", {}))
    return {
        "dataset_name": sample["dataset_name"],
        "split_name": sample["split_name"],
        "frame_index": int(sample["frame_index"]),
        "radius_factor": float(radius_factor),
        "radius_over_delta": float(radius_factor),
        "min_len_factor": 8.0,
        "status": metadata.get("status"),
        "status_reason": metadata.get("status_reason"),
        "failure_count": int(metadata.get("failure_count", 0)),
        "failure_reasons": list(metadata.get("failure_reasons", [])),
        "raw_contour_count": len(result.get("raw_contours", [])),
        "selected_contour_count": len(result.get("selected_contours", [])),
        "candidate_counts": dict(metadata.get("candidate_counts", {})),
        "rejection_reason_histogram": rejection_histogram,
        "rejection_reasons_observable": False,
        "rejection_reasons_note": (
            "native rotcohvrt exposes returned accepted contours only; internal "
            "candidate rejection reasons are not exposed by the upstream API; "
            "adapter/extractor reasons remain recorded here"
        ),
        "coverage": coverage,
        "contours": contour_summary,
        "ivd_check": ivd_check,
        "floating_summary": _floating_summary(result, ivd_check),
        "input_hash": metadata.get("input_hash"),
        "parameter_hash": metadata.get("parameter_hash"),
        "integrity_fingerprint": metadata.get("integrity_fingerprint"),
        "runtime_identity": metadata.get("runtime_identity"),
        "contour_evaluation": metadata.get("contour_evaluation", {}),
        "artifact_state": metadata.get("artifact_state"),
        "trusted": metadata.get("trusted"),
        "eligible_for_test_relationships": (
            metadata.get("protocol", {}).get("eligible_for_test_relationships", False)
            if isinstance(metadata.get("protocol"), Mapping) else False
        ),
    }


def _category_signature(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    names = (
        "solid_mask",
        "invalid_support",
        "valid_fluid",
        "filled_interior",
        "unknown_band",
        "negative_candidates",
        "evidence_state",
    )
    return {name: np.asarray(result[name]).copy() for name in names}


def _determinism_check(
    first_result: Mapping[str, Any],
    *,
    trace_again: Any,
    repetitions: int,
    first_summary: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_categories = _category_signature(first_result)
    baseline_metadata = first_result["metadata"]
    baseline_reasons = {
        "candidate_counts": dict(baseline_metadata.get("candidate_counts", {})),
        "rejection_reason_histogram": dict(
            baseline_metadata.get("rejection_reason_histogram", {})
        ),
        "raw_contour_count": int(first_summary["raw_contour_count"]),
        "selected_contour_count": int(first_summary["selected_contour_count"]),
    }
    max_float_difference = 0.0
    categorical_identical = True
    counts_and_reasons_identical = True
    runtime_identical = True
    repeat_results: list[dict[str, Any]] = []
    baseline_floats = first_summary["floating_summary"]
    for repeat_index in range(1, repetitions):
        repeated = trace_again()
        repeated_metadata = repeated["metadata"]
        repeated_categories = _category_signature(repeated)
        categorical_identical = categorical_identical and all(
            np.array_equal(baseline_categories[name], repeated_categories[name])
            for name in baseline_categories
        )
        repeated_counts_and_reasons = {
            "candidate_counts": dict(repeated_metadata.get("candidate_counts", {})),
            "rejection_reason_histogram": dict(
                repeated_metadata.get("rejection_reason_histogram", {})
            ),
            "raw_contour_count": len(repeated.get("raw_contours", [])),
            "selected_contour_count": len(repeated.get("selected_contours", [])),
        }
        counts_and_reasons_identical = (
            counts_and_reasons_identical
            and repeated_counts_and_reasons == baseline_reasons
        )
        runtime_identical = runtime_identical and (
            repeated_metadata.get("runtime_identity")
            == baseline_metadata.get("runtime_identity")
        )
        repeated_floats = _floating_summary(
            repeated,
            _ivd_comparison(repeated["standard_ivd"], first_result["standard_ivd"]),
        )
        per_repeat_max = 0.0
        for name, baseline in baseline_floats.items():
            current = repeated_floats.get(name)
            if baseline is None or current is None:
                continue
            per_repeat_max = max(per_repeat_max, abs(float(current) - float(baseline)))
        max_float_difference = max(max_float_difference, per_repeat_max)
        repeat_results.append({
            "repeat_index": repeat_index + 1,
            "categorical_identical": bool(categorical_identical),
            "counts_and_reasons_identical": bool(counts_and_reasons_identical),
            "runtime_identical": bool(runtime_identical),
            "floating_summary_max_absolute_difference": per_repeat_max,
        })
    passed = (
        categorical_identical
        and counts_and_reasons_identical
        and runtime_identical
        and max_float_difference <= 1e-12
    )
    return {
        "repetitions": repetitions,
        "categorical_layers_identical": bool(categorical_identical),
        "counts_and_reasons_identical": bool(counts_and_reasons_identical),
        "runtime_identity_identical": bool(runtime_identical),
        "floating_summary_max_absolute_difference": max_float_difference,
        "floating_summary_tolerance": 1e-12,
        "status": "pass" if passed else "fail",
        "repeats": repeat_results,
    }


def _save_visualization(
    path: Path,
    *,
    ivd: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
    solid_mask: np.ndarray,
    contours: list[Mapping[str, Any]],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    finite = np.asarray(ivd, dtype=np.float64)
    display = np.where(np.isfinite(finite), finite, 0.0)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = axis.imshow(
        display,
        origin="lower",
        extent=(float(xdim[0]), float(xdim[-1]), float(ydim[0]), float(ydim[-1])),
        aspect="auto",
        cmap="magma",
    )
    for record in contours:
        points = record.get("points_xy")
        if points is None:
            continue
        points_array = np.asarray(points, dtype=np.float64)
        if points_array.ndim != 2 or points_array.shape[1] != 2:
            continue
        axis.plot(
            points_array[:, 0],
            points_array[:, 1],
            color="cyan",
            linewidth=0.6,
            alpha=0.7,
        )
        center = record.get("center_xy")
        if center is not None:
            center_array = np.asarray(center, dtype=np.float64)
            if center_array.shape == (2,):
                axis.plot(center_array[0], center_array[1], "wo", markersize=2)
    if np.any(solid_mask):
        axis.contour(
            np.asarray(solid_mask, dtype=np.float64),
            levels=[0.5],
            colors="white",
            linewidths=0.4,
            origin="lower",
            extent=(float(xdim[0]), float(xdim[-1]), float(ydim[0]), float(ydim[-1])),
        )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.colorbar(image, ax=axis, label="standard IVD")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _save_run_artifact(
    output_root: Path,
    run_id: str,
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    xdim: np.ndarray,
    ydim: np.ndarray,
    title: str,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"audit run artifact already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "standard_ivd.npy", np.asarray(result["standard_ivd"], dtype=np.float64))
    np.save(run_dir / "solid_mask.npy", np.asarray(result["solid_mask"], dtype=bool))
    np.save(run_dir / "xdim.npy", np.asarray(xdim, dtype=np.float64))
    np.save(run_dir / "ydim.npy", np.asarray(ydim, dtype=np.float64))
    _write_json(run_dir / "raw_contours.json", result.get("raw_contours", []))
    _write_json(run_dir / "filtering_decisions.json", result.get("filtering_decisions", []))
    _save_visualization(
        run_dir / "ivd_contours.png",
        ivd=np.asarray(result["standard_ivd"]),
        xdim=xdim,
        ydim=ydim,
        solid_mask=np.asarray(result["solid_mask"], dtype=bool),
        contours=list(result.get("raw_contours", [])),
        title=title,
    )
    _write_json(run_dir / "run_metrics.json", summary)
    file_paths = sorted(
        path for path in run_dir.iterdir() if path.is_file() and path.name != "run_meta.json"
    )
    file_hashes = {
        path.name: _file_sha256(path)
        for path in file_paths
    }
    fingerprint = _json_hash({
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": run_id,
        "file_hashes": file_hashes,
        "summary_identity": {
            "dataset_name": summary["dataset_name"],
            "split_name": summary["split_name"],
            "frame_index": summary["frame_index"],
            "radius_factor": summary["radius_factor"],
            "input_hash": summary.get("input_hash"),
            "parameter_hash": summary.get("parameter_hash"),
        },
    })
    run_meta = {
        "artifact_type": "numbacs_raw_contour_audit_run",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "valid" if summary.get("status") else "invalid",
        "trusted": False,
        "artifact_state": "candidate",
        "file_hashes": file_hashes,
        "integrity_fingerprint": fingerprint,
        "summary": summary,
    }
    _write_json(run_dir / "run_meta.json", run_meta)
    return {
        "run_id": run_id,
        "artifact_dir": _relative_path(run_dir, output_root),
        "artifact_fingerprint": fingerprint,
        "summary": _jsonable(summary),
    }


def _swirling_velocity(
    xdim: np.ndarray,
    ydim: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(ydim, xdim, indexing="ij")
    shifted_x = xx - float(center_x)
    shifted_y = yy - float(center_y)
    radius_squared = shifted_x * shifted_x + shifted_y * shifted_y
    exponent = np.exp(-radius_squared / (2.0 * sigma * sigma))
    h = np.empty_like(radius_squared, dtype=np.float64)
    nonzero = radius_squared > 1e-15
    h[nonzero] = sigma * sigma * (1.0 - exponent[nonzero]) / radius_squared[nonzero]
    h[~nonzero] = 0.5
    return -shifted_y * h, shifted_x * h


def _no_vortex_fixture(size: int = 65, extent: float = 4.0) -> dict[str, Any]:
    xdim = np.linspace(-extent, extent, size, dtype=np.float64)
    ydim = np.linspace(-extent, extent, size, dtype=np.float64)
    shape = (size, size)
    return {
        "case": "no_vortex",
        "resolution": size,
        "u": np.zeros(shape, dtype=np.float64),
        "v": np.zeros(shape, dtype=np.float64),
        "xdim": xdim,
        "ydim": ydim,
        "mask": np.zeros(shape, dtype=bool),
    }


def _single_vortex_fixture(size: int) -> dict[str, Any]:
    xdim = np.linspace(-4.0, 4.0, size, dtype=np.float64)
    ydim = np.linspace(-4.0, 4.0, size, dtype=np.float64)
    u, v = _swirling_velocity(xdim, ydim, center_x=0.0, center_y=0.0, sigma=1.0)
    yy, xx = np.meshgrid(ydim, xdim, indexing="ij")
    exact_radius = 4.0
    exact_mask = (xx * xx + yy * yy) <= exact_radius * exact_radius
    return {
        "case": "single_vortex",
        "resolution": size,
        "u": u,
        "v": v,
        "xdim": xdim,
        "ydim": ydim,
        "mask": np.zeros_like(u, dtype=bool),
        "expected_center_xy": [0.0, 0.0],
        "expected_radius": exact_radius,
        "exact_mask": exact_mask,
        "fixture_definition": {
            "velocity": "analytic Gaussian-vorticity swirl from a stream-function solution",
            "sigma": 1.0,
            "exact_region": "in-domain exact evaluation disc radius 4 around (0,0)",
        },
    }


def _multi_vortex_fixture(size: int = 129) -> dict[str, Any]:
    xdim = np.linspace(-4.5, 4.5, size, dtype=np.float64)
    ydim = np.linspace(-4.5, 4.5, size, dtype=np.float64)
    centers = [(-1.8, 0.0), (1.8, 0.0)]
    u = np.zeros((size, size), dtype=np.float64)
    v = np.zeros((size, size), dtype=np.float64)
    for center_x, center_y in centers:
        component_u, component_v = _swirling_velocity(
            xdim,
            ydim,
            center_x=center_x,
            center_y=center_y,
            sigma=0.75,
        )
        u += component_u
        v += component_v
    return {
        "case": "multi_vortex",
        "resolution": size,
        "u": u,
        "v": v,
        "xdim": xdim,
        "ydim": ydim,
        "mask": np.zeros_like(u, dtype=bool),
        "expected_centers_xy": [[float(x), float(y)] for x, y in centers],
        "fixture_definition": {
            "velocity": "sum of two separated analytic Gaussian-vorticity swirls",
            "sigma": 0.75,
            "centers": [[float(x), float(y)] for x, y in centers],
        },
    }


def _trace(
    frame: Mapping[str, Any],
    *,
    dataset_name: str,
    split_name: str,
    frame_index: int,
    radius_factor: float,
    protocol_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return haller_anchors.trace_haller_evidence(
        frame["u"],
        frame["v"],
        frame["xdim"],
        frame["ydim"],
        frame["mask"],
        dataset_name=dataset_name,
        split_name=split_name,
        frame_index=frame_index,
        source=_split_source(split_name),
        backend=haller_anchors.BACKEND_NUMBACS,
        contour_mode=haller_anchors.CONTOUR_MODE_NUMBACS,
        numbacs_radius_factor=radius_factor,
        protocol_manifest=protocol_manifest,
    )


def _single_vortex_metrics(
    result: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    records = list(result.get("selected_contours", []))
    xdim = np.asarray(fixture["xdim"], dtype=np.float64)
    ydim = np.asarray(fixture["ydim"], dtype=np.float64)
    max_delta = max(float(xdim[1] - xdim[0]), float(ydim[1] - ydim[0]))
    expected_center = np.asarray(fixture["expected_center_xy"], dtype=np.float64)
    if not records:
        return {
            "status": "fail",
            "resolution": int(fixture["resolution"]),
            "selected_contour_count": 0,
            "evaluated_candidate_count": 0,
            "center_error": None,
            "center_tolerance": 2.0 * max_delta,
            "mask_iou": 0.0,
            "equivalent_radius": None,
            "equivalent_radius_relative_error": None,
            "reason": "no_selected_contour",
        }
    exact_mask = np.asarray(fixture["exact_mask"], dtype=bool)
    candidate_metrics: list[dict[str, Any]] = []
    for candidate_index, record in enumerate(records):
        center_value = record.get("center_xy")
        points_value = record.get("points_grid")
        if center_value is None or points_value is None:
            continue
        center = np.asarray(center_value, dtype=np.float64)
        path = np.asarray(points_value, dtype=np.float64)
        if center.shape != (2,) or path.ndim != 2 or path.shape[1] != 2:
            continue
        contour_mask = haller_anchors.grid_points_in_poly(tuple(fixture["u"].shape), path)
        intersection = int(np.count_nonzero(contour_mask & exact_mask))
        union = int(np.count_nonzero(contour_mask | exact_mask))
        iou = float(intersection / union) if union else 0.0
        area = float(record.get("area", 0.0))
        equivalent_radius = float(np.sqrt(area / np.pi)) if area > 0.0 else 0.0
        radius_error = abs(equivalent_radius - float(fixture["expected_radius"])) / float(
            fixture["expected_radius"]
        )
        center_error = float(np.linalg.norm(center - expected_center))
        candidate_metrics.append({
            "candidate_index": candidate_index,
            "record": record,
            "center_error": center_error,
            "mask_iou": iou,
            "equivalent_radius": equivalent_radius,
            "equivalent_radius_relative_error": radius_error,
        })
    if not candidate_metrics:
        return {
            "status": "fail",
            "resolution": int(fixture["resolution"]),
            "selected_contour_count": len(records),
            "evaluated_candidate_count": 0,
            "center_error": None,
            "center_tolerance": 2.0 * max_delta,
            "mask_iou": 0.0,
            "equivalent_radius": None,
            "equivalent_radius_relative_error": None,
            "reason": "no_geometric_candidate",
        }
    selected_metrics = max(
        candidate_metrics,
        key=lambda item: (
            float(item["mask_iou"]),
            -float(item["center_error"]),
            -float(item["equivalent_radius_relative_error"]),
        ),
    )
    selected = selected_metrics["record"]
    center_error = float(selected_metrics["center_error"])
    iou = float(selected_metrics["mask_iou"])
    equivalent_radius = float(selected_metrics["equivalent_radius"])
    radius_error = float(selected_metrics["equivalent_radius_relative_error"])
    passed = (
        center_error <= 2.0 * max_delta
        and iou >= 0.70
        and radius_error <= 0.20
    )
    return {
        "status": "pass" if passed else "fail",
        "resolution": int(fixture["resolution"]),
        "selected_contour_count": len(records),
        "evaluated_candidate_count": len(candidate_metrics),
            "matched_record_index": int(
                selected.get("raw_record_index", selected_metrics["candidate_index"])
            ),
        "center_error": center_error,
        "center_tolerance": 2.0 * max_delta,
        "mask_iou": iou,
        "equivalent_radius": equivalent_radius,
        "expected_equivalent_radius": float(fixture["expected_radius"]),
        "equivalent_radius_relative_error": float(radius_error),
        "candidate_selection_rule": "best exact-mask IoU among raw selected contours",
        "reason": None if passed else "single_vortex_threshold_failed",
    }


def _multi_vortex_metrics(
    result: Mapping[str, Any],
    fixture: Mapping[str, Any],
    *,
    radius_factor: float = 1.0,
) -> dict[str, Any]:
    records = list(result.get("selected_contours", []))
    xdim = np.asarray(fixture["xdim"], dtype=np.float64)
    ydim = np.asarray(fixture["ydim"], dtype=np.float64)
    max_delta = max(float(xdim[1] - xdim[0]), float(ydim[1] - ydim[0]))
    tolerance = 2.0 * max_delta
    centers = [np.asarray(item, dtype=np.float64) for item in fixture["expected_centers_xy"]]
    center_records = [
        (index, np.asarray(record.get("center_xy"), dtype=np.float64))
        for index, record in enumerate(records)
        if record.get("center_xy") is not None
        and np.asarray(record.get("center_xy"), dtype=np.float64).shape == (2,)
    ]
    center_by_record_index = dict(center_records)
    # Native rotcohvrt intentionally returns nested candidates for different
    # levels around one local peak.  For this controlled physical-region
    # check, group only those nearby centers; raw candidates remain unchanged
    # in the audit artifacts and are still counted there.
    deduplication_radius = 3.0 * float(radius_factor) * max_delta
    clusters: list[list[int]] = []
    for record_index, center in center_records:
        touching = [
            cluster_index
            for cluster_index, cluster in enumerate(clusters)
            if any(
                float(np.linalg.norm(center - center_by_record_index[item]))
                <= deduplication_radius
                for item in cluster
            )
        ]
        if not touching:
            clusters.append([record_index])
            continue
        merged = [record_index]
        for cluster_index in reversed(touching):
            merged.extend(clusters.pop(cluster_index))
        clusters.append(sorted(set(merged)))
    representative_indices = [
        max(
            cluster,
            key=lambda index: float(records[index].get("area", 0.0) or 0.0),
        )
        for cluster in clusters
    ]
    predicted = [
        np.asarray(records[index]["center_xy"], dtype=np.float64)
        for index in representative_indices
    ]
    unmatched_predictions = set(range(len(predicted)))
    matched: list[dict[str, Any]] = []
    for expected_index, expected in enumerate(centers):
        candidates = [
            (float(np.linalg.norm(predicted[index] - expected)), index)
            for index in unmatched_predictions
        ]
        if not candidates:
            continue
        distance, prediction_index = min(candidates)
        if distance <= tolerance:
            unmatched_predictions.remove(prediction_index)
            matched.append({
                "expected_index": expected_index,
                "prediction_index": prediction_index,
                "center_error": distance,
            })
    recall = float(len(matched) / len(centers)) if centers else 1.0
    unmatched = len(unmatched_predictions) + max(0, len(centers) - len(matched))
    passed = recall == 1.0 and unmatched <= 1
    return {
        "status": "pass" if passed else "fail",
        "selected_contour_count": len(records),
        "deduplicated_region_count": len(predicted),
        "deduplication_radius": deduplication_radius,
        "deduplication_rule": (
            "connected center groups within 3*(r/Delta)*Delta; "
            "raw candidates are preserved"
        ),
        "center_tolerance": tolerance,
        "center_recall": recall,
        "unmatched_selected_regions": int(unmatched),
        "matches": matched,
        "reason": None if passed else "multi_vortex_threshold_failed",
    }


def _no_vortex_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = result["metadata"]
    coverage = metadata.get("coverage", {})
    positive_cells = int(coverage.get("positive_cells", 0))
    selected = len(result.get("selected_contours", []))
    failures = int(metadata.get("failure_count", 0))
    passed = selected == 0 and positive_cells == 0 and failures == 0
    return {
        "status": "pass" if passed else "fail",
        "selected_contour_count": selected,
        "positive_cells": positive_cells,
        "extractor_failures": failures,
        "reason": None if passed else "no_vortex_control_failed",
    }


def _health_by_dataset(
    summaries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for summary in summaries:
        grouped.setdefault(str(summary["dataset_name"]), []).append(summary)
    by_dataset: dict[str, Any] = {}
    for dataset_name, rows in grouped.items():
        total = len(rows)
        failure_count = sum(
            1 for row in rows
            if int(row.get("failure_count", 0)) > 0
            or row.get("status") in {"invalid_input", "extractor_failure"}
        )
        valid_fluid = sum(int(row.get("coverage", {}).get("valid_fluid_cells", 0)) for row in rows)
        known = sum(int(row.get("coverage", {}).get("known_cells", 0)) for row in rows)
        positive = sum(int(row.get("coverage", {}).get("positive_cells", 0)) for row in rows)
        known_fraction = known / valid_fluid if valid_fluid else None
        positive_fraction = positive / valid_fluid if valid_fluid else None
        passed = (
            total > 0
            and failure_count / total <= 0.02
            and known_fraction is not None
            and 0.05 <= known_fraction <= 0.95
            and positive_fraction is not None
            and 0.0001 <= positive_fraction <= 0.25
        )
        by_dataset[dataset_name] = {
            "sample_count": total,
            "hard_failure_count": failure_count,
            "hard_failure_rate": failure_count / total if total else None,
            "valid_fluid_cells": valid_fluid,
            "known_cells": known,
            "known_coverage": known_fraction,
            "positive_core_cells": positive,
            "positive_core_coverage": positive_fraction,
            "status": "pass" if passed else "fail",
        }
    all_pass = all(item["status"] == "pass" for item in by_dataset.values())
    return {
        "by_dataset": by_dataset,
        "status": "pass" if all_pass else "fail",
        "thresholds": {
            "hard_failure_rate_max": 0.02,
            "known_coverage_min": 0.05,
            "known_coverage_max": 0.95,
            "positive_core_coverage_min": 0.0001,
            "positive_core_coverage_max": 0.25,
        },
    }


def _radius_summaries(
    summaries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for summary in summaries:
        grouped.setdefault(float(summary["radius_factor"]), []).append(summary)
    output: dict[str, Any] = {}
    for radius, rows in sorted(grouped.items()):
        radii = [
            item["equivalent_radius"]
            for row in rows
            for item in row["contours"]["records"]
            if item.get("equivalent_radius") is not None
        ]
        output[str(radius)] = {
            "radius_factor": radius,
            "sample_count": len(rows),
            "raw_contour_count_total": sum(int(row["raw_contour_count"]) for row in rows),
            "raw_contour_count_median_per_frame": median(
                [int(row["raw_contour_count"]) for row in rows]
            ) if rows else 0.0,
            "equivalent_radius_median": median(radii) if radii else None,
            "max_overlap_iou": max(
                (float(row["contours"]["max_pairwise_overlap_iou"]) for row in rows),
                default=0.0,
            ),
            "hard_failure_count": sum(int(row["failure_count"]) for row in rows),
        }
    return output


def _fragmentation_judgement(radius_summary: Mapping[str, Any]) -> dict[str, Any]:
    if "1.0" not in radius_summary:
        return {"status": "inconclusive", "reason": "missing_r1"}
    r1 = radius_summary["1.0"]
    other = [
        value for key, value in radius_summary.items() if key != "1.0"
    ]
    if not other:
        return {"status": "inconclusive", "reason": "missing_radius_comparators"}
    other_counts = [float(item["raw_contour_count_median_per_frame"]) for item in other]
    other_radii = [
        float(item["equivalent_radius_median"])
        for item in other
        if item.get("equivalent_radius_median") is not None
        and float(item["equivalent_radius_median"]) > 0.0
    ]
    r1_count = float(r1["raw_contour_count_median_per_frame"])
    reference_count = median(other_counts)
    r1_radius = r1.get("equivalent_radius_median")
    reference_radius = median(other_radii) if other_radii else None
    count_ratio = float("inf") if reference_count == 0.0 and r1_count > 0 else (
        r1_count / reference_count if reference_count else 1.0
    )
    radius_ratio = (
        None if r1_radius is None or reference_radius is None or reference_radius == 0.0
        else float(r1_radius / reference_radius)
    )
    distinguishable = count_ratio >= 1.25 or (
        radius_ratio is not None and radius_ratio <= 0.85
    )
    excessive = count_ratio >= 2.0 and (
        radius_ratio is None or radius_ratio <= 0.75
    )
    return {
        "status": "distinguished" if distinguishable else "inconclusive",
        "r1_excessive_fragmentation": bool(excessive),
        "r1_count_median": r1_count,
        "other_radius_count_median": reference_count,
        "r1_to_other_count_ratio": count_ratio,
        "r1_equivalent_radius_median": r1_radius,
        "other_equivalent_radius_median": reference_radius,
        "r1_to_other_equivalent_radius_ratio": radius_ratio,
        "rule": (
            "diagnostic flag when r1 median contour count is at least 2x the "
            "median of r>=2 and its median equivalent radius is at most 0.75x; "
            "this is not a trust-freeze gate"
        ),
    }


def _controlled_grid(
    output_root: Path,
    config: AuditConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    controlled: dict[str, Any] = {
        "no_vortex_by_radius": {},
        "single_vortex_by_radius": {},
        "multi_vortex_by_radius": {},
    }
    artifact_index: list[dict[str, Any]] = []
    ivd_checks: list[Mapping[str, Any]] = []
    for radius_factor in config.radius_factors:
        radius_key = str(float(radius_factor))

        no_vortex = _no_vortex_fixture()
        no_result = _trace(
            no_vortex,
            dataset_name="synthetic_no_vortex",
            split_name="development_audit",
            frame_index=0,
            radius_factor=radius_factor,
            protocol_manifest=None,
        )
        no_expected = independent_standard_ivd(
            no_vortex["u"], no_vortex["v"], no_vortex["xdim"], no_vortex["ydim"], no_vortex["mask"]
        )
        no_summary = _frame_summary(
            no_result,
            sample={"dataset_name": "synthetic_no_vortex", "split_name": "development_audit", "frame_index": 0},
            radius_factor=radius_factor,
            xdim=no_vortex["xdim"],
            ydim=no_vortex["ydim"],
            expected_ivd=no_expected,
        )
        no_summary["controlled_metrics"] = _no_vortex_metrics(no_result)
        no_entry = _save_run_artifact(
            output_root,
            f"controlled__no_vortex__r{radius_key.replace('.', 'p')}",
            no_result,
            no_summary,
            xdim=no_vortex["xdim"],
            ydim=no_vortex["ydim"],
            title=f"No-vortex control | r/Delta={radius_factor:g}",
        )
        no_entry["scope"] = "controlled_no_vortex"
        artifact_index.append(no_entry)
        ivd_checks.append(no_summary["ivd_check"])
        controlled["no_vortex_by_radius"][radius_key] = no_summary["controlled_metrics"]

        single_results: list[dict[str, Any]] = []
        for size in (65, 97, 129):
            fixture = _single_vortex_fixture(size)
            result = _trace(
                fixture,
                dataset_name=f"synthetic_single_vortex_{size}",
                split_name="development_audit",
                frame_index=0,
                radius_factor=radius_factor,
                protocol_manifest=None,
            )
            expected = independent_standard_ivd(
                fixture["u"], fixture["v"], fixture["xdim"], fixture["ydim"], fixture["mask"]
            )
            summary = _frame_summary(
                result,
                sample={
                    "dataset_name": f"synthetic_single_vortex_{size}",
                    "split_name": "development_audit",
                    "frame_index": 0,
                },
                radius_factor=radius_factor,
                xdim=fixture["xdim"],
                ydim=fixture["ydim"],
                expected_ivd=expected,
            )
            summary["controlled_metrics"] = _single_vortex_metrics(result, fixture)
            entry = _save_run_artifact(
                output_root,
                f"controlled__single_vortex__n{size}__r{radius_key.replace('.', 'p')}",
                result,
                summary,
                xdim=fixture["xdim"],
                ydim=fixture["ydim"],
                title=f"Single-vortex control n={size} | r/Delta={radius_factor:g}",
            )
            entry["scope"] = "controlled_single_vortex"
            artifact_index.append(entry)
            ivd_checks.append(summary["ivd_check"])
            single_results.append(summary["controlled_metrics"])
        controlled["single_vortex_by_radius"][radius_key] = {
            "resolutions": single_results,
            "status": "pass" if all(item["status"] == "pass" for item in single_results) else "fail",
        }

        multi = _multi_vortex_fixture()
        multi_result = _trace(
            multi,
            dataset_name="synthetic_multi_vortex",
            split_name="development_audit",
            frame_index=0,
            radius_factor=radius_factor,
            protocol_manifest=None,
        )
        multi_expected = independent_standard_ivd(
            multi["u"], multi["v"], multi["xdim"], multi["ydim"], multi["mask"]
        )
        multi_summary = _frame_summary(
            multi_result,
            sample={"dataset_name": "synthetic_multi_vortex", "split_name": "development_audit", "frame_index": 0},
            radius_factor=radius_factor,
            xdim=multi["xdim"],
            ydim=multi["ydim"],
            expected_ivd=multi_expected,
        )
        multi_summary["controlled_metrics"] = _multi_vortex_metrics(
            multi_result,
            multi,
            radius_factor=radius_factor,
        )
        multi_entry = _save_run_artifact(
            output_root,
            f"controlled__multi_vortex__n{multi['resolution']}__r{radius_key.replace('.', 'p')}",
            multi_result,
            multi_summary,
            xdim=multi["xdim"],
            ydim=multi["ydim"],
            title=f"Multi-vortex control n={multi['resolution']} | r/Delta={radius_factor:g}",
        )
        multi_entry["scope"] = "controlled_multi_vortex"
        artifact_index.append(multi_entry)
        ivd_checks.append(multi_summary["ivd_check"])
        controlled["multi_vortex_by_radius"][radius_key] = multi_summary["controlled_metrics"]

    controlled["no_vortex_status"] = (
        "pass"
        if all(item["status"] == "pass" for item in controlled["no_vortex_by_radius"].values())
        else "fail"
    )
    return controlled, artifact_index, [dict(item) for item in ivd_checks]


def _health_by_radius(
    summaries: list[Mapping[str, Any]],
    radius_factors: tuple[float, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for radius_factor in radius_factors:
        rows = [
            row for row in summaries
            if float(row["radius_factor"]) == float(radius_factor)
        ]
        result[str(float(radius_factor))] = _health_by_dataset(rows)
    return result


def _controlled_radius_passes(controlled: Mapping[str, Any], radius_factor: float) -> bool:
    key = str(float(radius_factor))
    return (
        controlled["no_vortex_by_radius"].get(key, {}).get("status") == "pass"
        and controlled["single_vortex_by_radius"].get(key, {}).get("status") == "pass"
        and controlled["multi_vortex_by_radius"].get(key, {}).get("status") == "pass"
    )


def _raw_contour_artifact_contract(
    output_root: Path,
    artifact_index: list[Mapping[str, Any]],
) -> bool:
    """Verify that every indexed run has the raw-geometry audit sidecars."""

    if not artifact_index:
        return False
    for entry in artifact_index:
        summary = entry.get("summary")
        if not isinstance(summary, Mapping):
            return False
        contours = summary.get("contours")
        if not isinstance(contours, Mapping):
            return False
        if not isinstance(contours.get("records"), list):
            return False
        if "rejection_reason_histogram" not in summary:
            return False
        run_dir = output_root / str(entry["artifact_dir"])
        filtering_path = run_dir / "filtering_decisions.json"
        visualization_path = run_dir / "ivd_contours.png"
        if not filtering_path.is_file() or not visualization_path.is_file():
            return False
        try:
            json.loads(filtering_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
    return True


def _write_candidate_artifact(
    output_root: Path,
    *,
    selected_radius: float | None,
    sample_manifest: Mapping[str, Any],
    development_entries: list[Mapping[str, Any]],
    fragmentation: Mapping[str, Any],
) -> dict[str, Any] | None:
    if selected_radius is None:
        return None
    candidate_root = output_root / "candidate"
    sample_root = candidate_root / "sample_runs"
    sample_root.mkdir(parents=True, exist_ok=True)
    selected_entries = [
        entry for entry in development_entries
        if float(entry["summary"]["radius_factor"]) == float(selected_radius)
    ]
    copied_runs: list[dict[str, Any]] = []
    for entry in selected_entries:
        source = output_root / entry["artifact_dir"]
        destination = sample_root / entry["run_id"]
        destination.mkdir(parents=True, exist_ok=True)
        copied_hashes: dict[str, str] = {}
        for filename in (
            "standard_ivd.npy",
            "solid_mask.npy",
            "xdim.npy",
            "ydim.npy",
            "raw_contours.json",
            "filtering_decisions.json",
            "run_metrics.json",
        ):
            shutil.copy2(source / filename, destination / filename)
            copied_hashes[filename] = _file_sha256(destination / filename)
        copied_runs.append({
            "run_id": entry["run_id"],
            "source_run_artifact_fingerprint": entry["artifact_fingerprint"],
            "relative_dir": _relative_path(destination, output_root),
            "file_hashes": copied_hashes,
        })
    candidate_identity = {
        "artifact_type": "numbacs_raw_contour_candidate",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_state": "candidate",
        "trusted": False,
        "backend": "numbacs.rotcohvrt",
        "radius_factor": float(selected_radius),
        "radius_over_delta": float(selected_radius),
        "min_len_factor": 8.0,
        "sample_manifest_hash": sample_manifest["manifest_hash"],
        "sample_run_count": len(copied_runs),
        "sample_runs": copied_runs,
        "fragmentation_judgement": dict(fragmentation),
        "role": "raw contour candidate for later fixed post-processing only",
        "final_test_relationships": "not_read",
    }
    candidate_identity["parameter_fingerprint"] = _json_hash({
        "backend": candidate_identity["backend"],
        "radius_factor": candidate_identity["radius_factor"],
        "min_len_factor": candidate_identity["min_len_factor"],
        "sample_manifest_hash": candidate_identity["sample_manifest_hash"],
    })
    candidate_identity["integrity_fingerprint"] = _json_hash(candidate_identity)
    _write_json(candidate_root / "candidate_manifest.json", candidate_identity)
    return {
        "artifact_dir": _relative_path(candidate_root, output_root),
        "manifest_path": _relative_path(candidate_root / "candidate_manifest.json", output_root),
        "radius_factor": float(selected_radius),
        "parameter_fingerprint": candidate_identity["parameter_fingerprint"],
        "integrity_fingerprint": candidate_identity["integrity_fingerprint"],
        "artifact_state": "candidate",
        "trusted": False,
        "sample_run_count": len(copied_runs),
    }


def _aggregate_ivd_checks(checks: list[Mapping[str, Any]]) -> dict[str, Any]:
    absolute = [
        float(item["max_absolute_error"])
        for item in checks
        if item.get("max_absolute_error") is not None
    ]
    relative = [
        float(item["relative_l2_error"])
        for item in checks
        if item.get("relative_l2_error") is not None
    ]
    maximum_absolute = max(absolute, default=float("inf"))
    maximum_relative = max(relative, default=float("inf"))
    passed = maximum_absolute <= 1e-10 and maximum_relative <= 1e-8
    return {
        "status": "pass" if passed else "fail",
        "max_absolute_error": maximum_absolute,
        "relative_l2_error": maximum_relative,
        "max_absolute_error_threshold": 1e-10,
        "relative_l2_error_threshold": 1e-8,
        "comparison": "independent float64 finite-difference vorticity and global fluid mean",
        "checked_case_count": len(checks),
    }


def run_native_numbacs_audit(
    data_root: str | Path,
    protocol_manifest: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    *,
    config: AuditConfig | None = None,
) -> dict[str, Any]:
    """Run the R1 native contour audit and publish a readback-verifiable report."""

    audit_config = AuditConfig() if config is None else config
    if not isinstance(audit_config, AuditConfig):
        raise TypeError("config must be an AuditConfig")
    protocol = (
        experiment_protocol.read_protocol_manifest(protocol_manifest)
        if isinstance(protocol_manifest, (str, Path))
        else experiment_protocol.validate_protocol_manifest(protocol_manifest)
    )
    numbacs_haller.ensure_native_runtime()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError(f"audit output directory must be empty: {output_root}")

    sample_manifest = build_development_sample_manifest(protocol)
    _write_json(output_root / "sample_manifest.json", sample_manifest)
    development_entries: list[dict[str, Any]] = []
    development_summaries: list[dict[str, Any]] = []
    determinism_checks: list[Mapping[str, Any]] = []
    ivd_checks: list[Mapping[str, Any]] = []
    runtime_identities: list[Mapping[str, Any]] = []
    for sample in sample_manifest["samples"]:
        frame = _load_dataset_frame(data_root, sample)
        expected_ivd = independent_standard_ivd(
            frame["u"], frame["v"], frame["xdim"], frame["ydim"], frame["mask"]
        )
        source_fingerprint = protocol["datasets"][sample["dataset_name"]]["source_identity"]["fingerprint"]
        for radius_factor in audit_config.radius_factors:
            result = _trace(
                frame,
                dataset_name=sample["dataset_name"],
                split_name=sample["split_name"],
                frame_index=int(sample["frame_index"]),
                radius_factor=radius_factor,
                protocol_manifest=protocol,
            )
            summary = _frame_summary(
                result,
                sample=sample,
                radius_factor=radius_factor,
                xdim=frame["xdim"],
                ydim=frame["ydim"],
                expected_ivd=expected_ivd,
            )
            summary["source_identity_fingerprint"] = source_fingerprint
            run_id = (
                f"development__{sample['dataset_name']}__{sample['split_name']}"
                f"__frame{int(sample['frame_index'])}__r{radius_factor:g}"
            ).replace(".", "p")
            entry = _save_run_artifact(
                output_root,
                run_id,
                result,
                summary,
                xdim=frame["xdim"],
                ydim=frame["ydim"],
                title=(
                    f"{sample['dataset_name']} {sample['split_name']} frame "
                    f"{sample['frame_index']} | r/Delta={radius_factor:g}"
                ),
            )
            entry["scope"] = "development_sample"
            development_entries.append(entry)
            development_summaries.append(summary)
            ivd_checks.append(summary["ivd_check"])
            runtime_identities.append(summary.get("runtime_identity", {}))

            determinism = _determinism_check(
                result,
                trace_again=lambda: _trace(
                    frame,
                    dataset_name=sample["dataset_name"],
                    split_name=sample["split_name"],
                    frame_index=int(sample["frame_index"]),
                    radius_factor=radius_factor,
                    protocol_manifest=protocol,
                ),
                repetitions=audit_config.repetitions,
                first_summary=summary,
            )
            determinism_checks.append({
                "run_id": run_id,
                **determinism,
            })

    controlled, controlled_entries, controlled_ivd_checks = _controlled_grid(
        output_root, audit_config
    )
    ivd_checks.extend(controlled_ivd_checks)
    artifact_index = development_entries + controlled_entries
    raw_contour_artifact_pass = _raw_contour_artifact_contract(
        output_root, artifact_index
    )
    radius_summary = _radius_summaries(development_summaries)
    fragmentation = _fragmentation_judgement(radius_summary)
    health_by_radius = _health_by_radius(
        development_summaries, audit_config.radius_factors
    )
    candidate_radius: float | None = None
    eligibility: dict[str, Any] = {}
    for radius_factor in audit_config.radius_factors:
        key = str(float(radius_factor))
        health_pass = health_by_radius[key]["status"] == "pass"
        control_pass = _controlled_radius_passes(controlled, radius_factor)
        excluded_for_fragmentation = (
            bool(fragmentation.get("r1_excessive_fragmentation"))
            and float(radius_factor) == 1.0
        )
        eligible = health_pass and control_pass and not excluded_for_fragmentation
        eligibility[key] = {
            "development_health": health_pass,
            "controlled_fixtures": control_pass,
            "excluded_for_excessive_r1_fragmentation": excluded_for_fragmentation,
            "eligible": eligible,
        }
        if candidate_radius is None and eligible:
            candidate_radius = float(radius_factor)
    min_len_required = fragmentation.get("status") == "inconclusive"
    if min_len_required:
        candidate_radius = None
    candidate = _write_candidate_artifact(
        output_root,
        selected_radius=candidate_radius,
        sample_manifest=sample_manifest,
        development_entries=development_entries,
        fragmentation=fragmentation,
    )
    ivd_summary = _aggregate_ivd_checks(ivd_checks)
    determinism_pass = all(item["status"] == "pass" for item in determinism_checks)
    native_runtime_pass = bool(runtime_identities) and all(
        identity == runtime_identities[0] for identity in runtime_identities
    )
    selected_health = (
        health_by_radius[str(float(candidate_radius))]
        if candidate_radius is not None else {"status": "fail", "by_dataset": {}}
    )
    single_pass = candidate_radius is not None and all(
        item["status"] == "pass"
        for item in controlled["single_vortex_by_radius"].values()
    )
    multi_pass = candidate_radius is not None and all(
        item["status"] == "pass"
        for item in controlled["multi_vortex_by_radius"].values()
    )
    no_vortex_pass = controlled["no_vortex_status"] == "pass"
    acceptance = {
        "sample_manifest_train_calibration_only": (
            sample_manifest["sample_count"] == 12
            and all(
                sample["split_name"] in {"train", "calibration"}
                and int(sample["frame_index"]) != 1300
                and not bool(sample["eligible_for_test_relationships"])
                for sample in sample_manifest["samples"]
            )
        ),
        "primary_radius_grid_complete": tuple(audit_config.radius_factors) == PRIMARY_RADIUS_FACTORS,
        "min_len_grid_only_if_radius_inconclusive": not audit_config.min_len_grid_opened,
        "raw_contour_geometry_rejection_visualization": raw_contour_artifact_pass,
        "standard_ivd_thresholds": ivd_summary["status"] == "pass",
        "native_three_repeat_determinism": determinism_pass and native_runtime_pass,
        "no_vortex_fixture": no_vortex_pass,
        "single_vortex_fixture": single_pass,
        "multi_vortex_fixture": multi_pass,
        "six_dataset_development_health": selected_health.get("status") == "pass",
        "candidate_selected_without_test_access": candidate is not None,
        "final_test_not_read": True,
    }
    overall_status = "pass" if all(acceptance.values()) else "no-go"
    report: dict[str, Any] = {
        "artifact_type": "numbacs_raw_contour_audit_report",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": overall_status,
        "study_slug": experiment_protocol.STUDY_SLUG,
        "backend": {
            "name": "numbacs.rotcohvrt",
            "runtime_identity": runtime_identities[0] if runtime_identities else None,
            "native_runtime_required": True,
            "min_len_factor": 8.0,
        },
        "protocol": {
            "protocol_version": protocol["protocol_version"],
            "manifest_hash": protocol["manifest_hash"],
            "sample_manifest": "sample_manifest.json",
            "sample_manifest_hash": sample_manifest["manifest_hash"],
            "final_test_relationships": "not_read; train/calibration development audit only",
            "boussinesq_frame_1300": "excluded from sample; development/audit-only and never a final-test relationship",
        },
        "configuration": {
            "radius_factors": list(audit_config.radius_factors),
            "radius_grid": "r/Delta=[1,2,4,8]",
            "min_len_factor": 8.0,
            "min_len_grid_opened": audit_config.min_len_grid_opened,
            "min_len_grid_required_by_radius_result": min_len_required,
            "repetitions": audit_config.repetitions,
        },
        "radius_grid": radius_summary,
        "fragmentation_judgement": fragmentation,
        "candidate_eligibility": eligibility,
        "selected_raw_contour_candidate": candidate,
        "standard_ivd_check": ivd_summary,
        "determinism_check": {
            "status": "pass" if determinism_pass and native_runtime_pass else "fail",
            "native_runtime_identity_stable": native_runtime_pass,
            "checked_config_count": len(determinism_checks),
            "per_config": determinism_checks,
        },
        "controlled_validation": controlled,
        "development_health": {
            "selected_radius_factor": candidate_radius,
            "by_radius": health_by_radius,
            "selected": selected_health,
        },
        "acceptance": acceptance,
        "artifact_index": artifact_index,
        "rejection_reason_contract": {
            "native_internal_rejection_reasons": "not exposed by upstream rotcohvrt return contract",
            "adapter_and_extractor_reasons": "recorded per run in run_metrics.json and filtering_decisions.json",
        },
        "candidate_trust": {
            "artifact_state": "candidate",
            "trusted": False,
            "eligible_as_trusted_haller_anchor": False,
            "freeze_owner": "later R2 audit/freeze ticket",
        },
        "evidence_level": "experimental fact for this named development artifact; no test relationships read",
    }
    _write_json(output_root / "audit_report.json", report)
    report_hash = _file_sha256(output_root / "audit_report.json")
    _write_json(output_root / "audit_report_integrity.json", {
        "artifact_type": "numbacs_raw_contour_audit_report_integrity",
        "report_path": "audit_report.json",
        "report_sha256": report_hash,
        "report_status": overall_status,
    })
    return read_audit_report(output_root)


def _verify_run_artifact(output_root: Path, entry: Mapping[str, Any]) -> None:
    run_dir = output_root / str(entry["artifact_dir"])
    meta_path = run_dir / "run_meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("integrity_fingerprint") != entry.get("artifact_fingerprint"):
        raise ValueError(f"run artifact fingerprint mismatch: {run_dir}")
    for filename, expected_hash in metadata.get("file_hashes", {}).items():
        path = run_dir / filename
        if _file_sha256(path) != expected_hash:
            raise ValueError(f"run artifact file fingerprint mismatch: {path}")
    json.loads((run_dir / "raw_contours.json").read_text(encoding="utf-8"))
    json.loads((run_dir / "filtering_decisions.json").read_text(encoding="utf-8"))
    np.load(run_dir / "standard_ivd.npy", allow_pickle=False)


def read_audit_report(output_dir: str | Path) -> dict[str, Any]:
    """Read back and verify the published report and every indexed run artifact."""

    output_root = Path(output_dir)
    report_path = output_root / "audit_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    integrity = json.loads(
        (output_root / "audit_report_integrity.json").read_text(encoding="utf-8")
    )
    if integrity.get("report_sha256") != _file_sha256(report_path):
        raise ValueError("audit report fingerprint mismatch")
    sample_manifest = json.loads(
        (output_root / "sample_manifest.json").read_text(encoding="utf-8")
    )
    sample_payload = dict(sample_manifest)
    declared_hash = sample_payload.pop("manifest_hash", None)
    if declared_hash != _json_hash(sample_payload):
        raise ValueError("sample manifest fingerprint mismatch")
    if declared_hash != report["protocol"]["sample_manifest_hash"]:
        raise ValueError("report/sample manifest hash mismatch")
    for entry in report.get("artifact_index", []):
        _verify_run_artifact(output_root, entry)
    candidate = report.get("selected_raw_contour_candidate")
    if candidate is not None:
        candidate_manifest = json.loads(
            (output_root / candidate["manifest_path"]).read_text(encoding="utf-8")
        )
        if candidate_manifest.get("integrity_fingerprint") != candidate.get("integrity_fingerprint"):
            raise ValueError("candidate manifest fingerprint mismatch")
        for copied in candidate_manifest.get("sample_runs", []):
            destination = output_root / copied["relative_dir"]
            for filename, expected_hash in copied["file_hashes"].items():
                if _file_sha256(destination / filename) != expected_hash:
                    raise ValueError(f"candidate file fingerprint mismatch: {destination / filename}")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the R1 native NumbaCS raw-contour audit")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    report = run_native_numbacs_audit(
        args.data_root,
        args.protocol_manifest,
        args.out_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _source_for_split(split_name: str) -> str:
    return {
        "train": "haller_anchor_train",
        "calibration": "haller_gt_calibration",
    }[split_name]


def _midpoint_frame(range_record: list[int] | tuple[int, int]) -> int:
    start, end = (int(value) for value in range_record)
    if not start < end:
        raise ValueError(f"development sample range must be non-empty: {range_record!r}")
    return start + (end - start) // 2


def build_development_sample_manifest(
    protocol_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one deterministic midpoint from train and calibration per dataset.

    The R0 protocol currently has one contiguous range for each of those
    splits.  The explicit check prevents a future manifest shape change from
    silently changing the sampling rule.
    """

    protocol = experiment_protocol.validate_protocol_manifest(protocol_manifest)
    samples: list[dict[str, Any]] = []
    for dataset_name in experiment_protocol.DATASET_NAMES:
        dataset = protocol["datasets"][dataset_name]
        for split_name in ("train", "calibration"):
            partition = dataset["frame_partitions"][split_name]
            ranges = partition["ranges"]
            if len(ranges) != 1:
                raise ValueError(
                    "fixed development sample rule requires one contiguous "
                    f"{split_name} range for {dataset_name}, got {ranges!r}"
                )
            frame_index = _midpoint_frame(ranges[0])
            if not any(
                int(start) <= frame_index < int(end)
                for start, end in ranges
            ):
                raise AssertionError("selected frame is outside its source range")
            samples.append({
                "dataset_name": dataset_name,
                "split_name": split_name,
                "source": _source_for_split(split_name),
                "frame_index": frame_index,
                "selection_basis": {
                    "rule": "midpoint_of_half_open_partition_range",
                    "range": [int(ranges[0][0]), int(ranges[0][1])],
                },
                "eligible_for_test_relationships": False,
            })

    manifest: dict[str, Any] = {
        "artifact_type": "numbacs_raw_contour_development_sample_manifest",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "study_slug": experiment_protocol.STUDY_SLUG,
        "protocol_version": protocol["protocol_version"],
        "protocol_manifest_hash": protocol["manifest_hash"],
        "dataset_names": list(experiment_protocol.DATASET_NAMES),
        "selection_rule": "one midpoint frame per train/calibration split",
        "samples": samples,
        "sample_count": len(samples),
        "final_test_relationships": "sealed; no test data read",
        "boussinesq_frame_1300": "excluded; development/audit-only and not train/calibration",
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    return manifest


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AuditConfig",
    "PRIMARY_RADIUS_FACTORS",
    "build_development_sample_manifest",
    "independent_standard_ivd",
    "main",
    "read_audit_report",
    "run_native_numbacs_audit",
    "summarize_raw_contours",
]


if __name__ == "__main__":
    raise SystemExit(main())
