"""Materialize current Ticket 05 W1-P/W1-Hybrid calibration predictions.

This utility is deliberately separate from the historical ``evaluate.py``
entrypoint.  It consumes the Ticket 03 P95/P80 ``DatasetView`` contract,
defaults to the current 100-epoch checkpoint, writes raw prediction
arrays plus per-frame P95/P80 provenance, and never opens test artifacts or
indexes a test frame.  Train-only Haller diagnostics remain optional.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

import experiment_protocol as protocol
from prediction_support import support_summary


DATASET_ORDER = (
    "boussinesq",
    "cylinder2d",
    "doublegyre2d",
    "fourcenters2d",
    "jungtelziemniak2d",
    "pipedcylinder2d",
)

CALIBRATION_FRAMES = {
    "boussinesq": (1000, 1100, 1176),
    "cylinder2d": (750, 825, 874),
    "doublegyre2d": (256, 270, 282),
    "fourcenters2d": (256, 270, 282),
    "jungtelziemniak2d": (250, 270, 276),
    "pipedcylinder2d": (750, 825, 874),
}

# Desired display frames from the report.  The actual frame is snapped to the
# nearest current train window start because the Ticket 04 Haller artifact is
# generated only at complete train window starts.
HALLER_REFERENCE_FRAMES = {
    "boussinesq": (500, 743, 748),
    "cylinder2d": (375, 700, 748),
    "doublegyre2d": (32, 128, 224),
    "fourcenters2d": (16, 113, 219),
    "jungtelziemniak2d": (25, 125, 225),
    "pipedcylinder2d": (375, 437, 700),
}


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_relocated_calibration_artifact(
    w1_p: Any,
    root: pathlib.Path,
    protocol_manifest: Mapping[str, Any],
) -> Any:
    """Load a verified calibration artifact after a filesystem relocation.

    Ticket 03 metadata was created on Windows and the immutable ``data_files``
    entries therefore retain the original Windows absolute paths.  The array
    bytes and hashes are unchanged on SHU-server.  We keep the original
    metadata/identity/hash, validate it, verify every local array hash, and
    deliberately ignore only the stale absolute path spelling.
    """
    root = root.resolve()
    metadata = _read_json(root / "target_meta.json")
    validated = w1_p._w1p_validate_ticket03_metadata(
        metadata,
        root=root,
        expected_split="calibration",
        protocol_manifest=protocol_manifest,
    )
    data_files = validated.get("data_files")
    if not isinstance(data_files, Mapping):
        raise ValueError(f"calibration artifact data_files missing: {root}")
    arrays: dict[str, Any] = {}
    for name, (filename, dtype) in w1_p._TICKET03_TARGET_FILES.items():
        path_value = root / filename
        declared = data_files.get(filename)
        if not isinstance(declared, Mapping):
            raise ValueError(f"calibration artifact data_files missing {filename}")
        declared_path = str(declared.get("path", ""))
        declared_name = (
            pathlib.PureWindowsPath(declared_path).name
            if "\\" in declared_path
            else pathlib.PurePath(declared_path).name
        )
        if declared_name != filename:
            raise ValueError(
                f"calibration artifact declared filename mismatch for {filename}"
            )
        if not path_value.is_file():
            raise FileNotFoundError(path_value)
        array = np.load(path_value, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.dtype(dtype):
            raise ValueError(
                f"calibration artifact {filename} dtype mismatch: "
                f"{array.dtype} != {np.dtype(dtype)}"
            )
        declared_hash = str(declared.get("sha256", ""))
        if not declared_hash or w1_p._w1p_file_hash(path_value) != declared_hash:
            raise ValueError(f"calibration artifact {filename} sha256 mismatch")
        arrays[name] = array
    w1_p._w1p_validate_arrays(arrays, validated)
    return w1_p.W1PLabelArtifact(root=root, metadata=validated, **arrays)


def _load_standard_ivd_frame_provenance(
    root: pathlib.Path,
    *,
    protocol_manifest: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Load the canonical per-frame P95/P80 provenance beside the target.

    The calibration target metadata intentionally stores the compact target
    identity and arrays, while the standard-IVD manifest stores the per-frame
    valid-fluid fingerprints.  Ticket 05 needs both pieces to make each
    prediction auditable, so join them by the immutable frame index here.
    """

    metadata = _read_json(root / "standard_ivd_meta.json")
    records = metadata.get("frame_records")
    if not isinstance(records, list):
        raise ValueError(f"standard IVD frame_records missing: {root}")
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"standard IVD frame record is not an object: {root}")
        frame = int(record["frame_index"])
        protocol_records = record.get("protocol_provenance")
        if not isinstance(protocol_records, Mapping):
            raise ValueError(f"standard IVD frame provenance missing: {root}, frame={frame}")
        percentile = protocol_records.get("percentile_target")
        if not isinstance(percentile, Mapping):
            raise ValueError(
                f"standard IVD percentile provenance missing: {root}, frame={frame}"
            )
        # This validates the source/protocol/split/frame boundary before the
        # provenance is copied into the prediction artifact.
        validated = protocol.validate_p95p80_provenance(
            percentile,
            protocol_manifest=protocol_manifest,
        )
        if validated["split_name"] != "calibration":
            continue
        if frame in result:
            raise ValueError(f"duplicate standard IVD frame provenance: {root}, frame={frame}")
        result[frame] = dict(validated)
    return result


def _calibration_frame_provenance(
    calibration_artifact: Any,
    standard_frame_provenance: Mapping[str, Any],
    *,
    protocol_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the Ticket 05 calibration provenance envelope for one frame."""

    provenance = dict(standard_frame_provenance)
    protocol.validate_p95p80_provenance(
        provenance,
        protocol_manifest=protocol_manifest,
    )
    threshold_provenance = calibration_artifact.metadata.get("threshold_provenance")
    if not isinstance(threshold_provenance, Mapping):
        raise ValueError("calibration target threshold_provenance is missing")
    return {
        "split": "calibration",
        "label_source": "percentile_calibration_reference",
        "percentile_threshold_source": protocol.P95P80_PERCENTILE_SOURCE_ID,
        "threshold_provenance": dict(threshold_provenance),
        "p95p80_protocol_manifest": dict(protocol_manifest),
        "p95p80_provenance": provenance,
        "test_source_read": False,
        "test_frame_indices_read": [],
        "final_test_relationships": "not_read",
    }


def _split_contains_window(
    runner: Any,
    record: Mapping[str, Any],
    split: str,
    frame: int,
    t_win: int,
) -> bool:
    return any(
        int(start) <= int(frame) and int(frame) + int(t_win) <= int(end)
        for start, end in runner._ranges(record, split)
    )


def _snap_train_frames(
    runner: Any,
    view: Any,
    desired: tuple[int, ...],
    t_win: int,
) -> tuple[int, ...]:
    available = tuple(
        int(frame)
        for frame in view.train_starts
        if _split_contains_window(runner, view.record, "train", int(frame), t_win)
    )
    if not available:
        raise ValueError(f"dataset={view.name} has no complete train windows")
    selected: list[int] = []
    for target in desired:
        candidate = min(available, key=lambda frame: abs(frame - int(target)))
        if candidate not in selected:
            selected.append(candidate)
    if len(selected) < 3:
        for candidate in available:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == 3:
                break
    return tuple(selected[:3])


def _extract_normalized(
    runner: Any,
    extractor: Any,
    dataset: Any,
    view: Any,
    *,
    py: int,
    px: int,
    frame: int,
    patch_size: tuple[int, int],
    t_win: int,
    groups: tuple[int, int],
    delta_frac: float,
    L: int,
    n_substeps: int,
    t_scale: float,
    seed: int,
    initial_seeds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if int(frame) < 0 or int(frame) + int(t_win) > int(view.u.shape[0]):
        raise ValueError(f"incomplete frame window: dataset={view.name}, frame={frame}")
    frame_rows = [
        view.standard_row_by_frame[int(start)]
        for start in range(int(frame), int(frame) + int(t_win))
    ]
    raw_u = np.asarray(view.u[int(frame) : int(frame) + int(t_win)], dtype=np.float32)
    raw_v = np.asarray(view.v[int(frame) : int(frame) + int(t_win)], dtype=np.float32)
    ivd = np.asarray(view.standard_ivd[frame_rows], dtype=np.float32)
    tdim_window = np.asarray(
        view.tdim[int(frame) : int(frame) + int(t_win)], dtype=np.float64
    )
    if len(tdim_window) != int(t_win):
        raise ValueError(f"incomplete coordinate window: dataset={view.name}, frame={frame}")
    rng_base = dataset._comb_rng_base(
        int(seed), int(py), int(px), int(frame), ds_id=view.ds_id
    )
    raw, seeds = extractor.extract_pathlines_batched(
        raw_u,
        raw_v,
        view.mask,
        ivd,
        view.xdim,
        view.ydim,
        tdim_window,
        patch_yx=(int(py), int(px)),
        patch_size=patch_size,
        t0=float(view.tdim[int(frame)]),
        L=int(L),
        groups=groups,
        delta_frac=float(delta_frac),
        t_win_frames=int(t_win),
        n_substeps=int(n_substeps),
        dt_out=float(tdim_window[-1] - tdim_window[0]) / (int(L) - 1),
        rng=rng_base,
        return_seeds=True,
        initial_seeds=initial_seeds,
    )
    geometry = extractor.patch_geometry(
        (int(py), int(px)), patch_size, view.xdim, view.ydim
    )
    pathlines = dataset.normalize_pathlines(
        raw,
        seeds,
        geometry,
        float(view.tdim[int(frame)]),
        float(tdim_window[-1] - tdim_window[0]),
        float(t_scale),
        float(view.normalization["ivd_mu"]),
        float(view.normalization["ivd_sigma"]),
        float(view.normalization["speed_max"]),
    )
    if not np.isfinite(pathlines).all():
        raise ValueError(f"non-finite pathlines: dataset={view.name}, frame={frame}")
    seed_y, seed_x = extractor.nearest_cell(
        seeds[:, 0], seeds[:, 1], view.xdim, view.ydim
    )
    cells = np.stack(
        [
            np.clip(seed_y, 0, view.height - 1),
            np.clip(seed_x, 0, view.width - 1),
        ],
        axis=1,
    ).astype(np.int64, copy=False)
    return np.asarray(pathlines, dtype=np.float32), cells


def _infer_frame(
    runner: Any,
    dataset: Any,
    extractor: Any,
    view: Any,
    model: Any,
    *,
    frame: int,
    frame_split: str,
    patch_size: tuple[int, int],
    t_win: int,
    window_step: int,
    groups: tuple[int, int],
    delta_frac: float,
    L: int,
    n_substeps: int,
    t_scale: float,
    seed: int,
    device: Any,
    inference_batch_size: int,
    supplementary_roi: tuple[int, int, int, int] | None = None,
    deadline: float | None = None,
    existing_prediction: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    if not _split_contains_window(runner, view.record, frame_split, frame, t_win):
        raise ValueError(
            f"frame window is not contained in split={frame_split}: "
            f"dataset={view.name}, frame={frame}, t_win={t_win}"
        )
    del window_step  # patch locations are already frozen in view.usable_patches.
    patches = tuple(view.usable_patches)
    if not patches:
        raise ValueError(f"dataset={view.name} has no usable inference patches")
    acc = np.zeros((view.height, view.width), dtype=np.float64)
    count = np.zeros((view.height, view.width), dtype=np.int64)
    if existing_prediction is not None:
        existing_probability, existing_count = existing_prediction
        count = np.asarray(existing_count, dtype=np.int64).copy()
        if count.shape != acc.shape or existing_probability.shape != acc.shape:
            raise ValueError("reused prediction/reference shape mismatch")
        acc = np.where(count > 0, existing_probability, 0).astype(np.float64) * count
    failures: list[dict[str, Any]] = []
    model.eval()
    inference_patches = () if existing_prediction is not None else patches
    for start in range(0, len(inference_patches), int(inference_batch_size)):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("calibration diagnostic reached its declared time budget; preserve completed frames")
        chunk = inference_patches[start : start + int(inference_batch_size)]
        feature_rows: list[np.ndarray] = []
        cell_rows: list[np.ndarray] = []
        for py, px in chunk:
            try:
                features, cells = _extract_normalized(
                    runner,
                    extractor,
                    dataset,
                    view,
                    py=int(py),
                    px=int(px),
                    frame=int(frame),
                    patch_size=patch_size,
                    t_win=t_win,
                    groups=groups,
                    delta_frac=delta_frac,
                    L=L,
                    n_substeps=n_substeps,
                    t_scale=t_scale,
                    seed=seed,
                )
            except (ValueError, IndexError, FloatingPointError) as exc:
                # A single unusable patch is an expected partial-data case;
                # retain its coordinates/error in the artifact while keeping
                # the successful-cell count explicit.  Structural/configuration
                # and device failures remain fail-loud outside this boundary.
                failures.append(
                    {
                        "patch_yx": [int(py), int(px)],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            feature_rows.append(features)
            cell_rows.append(cells)
        if not feature_rows:
            continue
        pathlines = np.stack(feature_rows, axis=0)
        dummy = torch.zeros(
            (pathlines.shape[0], 1, 1, 1), dtype=torch.float32, device=device
        )
        x_tensor = torch.from_numpy(pathlines).to(device)
        with torch.inference_mode():
            prediction = model((dummy, x_tensor))
        prediction = np.asarray(prediction.detach().cpu().numpy(), dtype=np.float32)
        if prediction.ndim != 2 or prediction.shape[0] != len(cell_rows):
            raise ValueError(
                f"unexpected prediction shape={prediction.shape}; "
                f"dataset={view.name}, frame={frame}"
            )
        for row, cells in zip(prediction, cell_rows):
            np.add.at(acc, (cells[:, 0], cells[:, 1]), row.astype(np.float64))
            np.add.at(count, (cells[:, 0], cells[:, 1]), 1)
    base_count = count.copy()
    if supplementary_roi is not None:
        y0, y1, x0, x1 = supplementary_roi
        if not (0 <= y0 < y1 <= view.height and 0 <= x0 < x1 <= view.width):
            raise ValueError("supplementary ROI must be inside the physical grid")
        k = groups[0] * groups[1] * 4
        target_cells = np.argwhere((count[y0:y1, x0:x1] == 0) & ~view.mask[y0:y1, x0:x1])
        target_cells += np.array([y0, x0])
        # Keep the original patch normalization and model group layout; only
        # requested missing grid starts replace the regular seeding pattern.
        by_patch: dict[tuple[int, int], list[np.ndarray]] = {}
        for cell in target_cells:
            eligible = [(py, px) for py, px in patches
                        if py <= cell[0] < py + patch_size[0] and px <= cell[1] < px + patch_size[1]]
            if eligible:
                center_patch = min(eligible, key=lambda p: (p[0] + patch_size[0]/2 - cell[0])**2 + (p[1] + patch_size[1]/2 - cell[1])**2)
                by_patch.setdefault(center_patch, []).append(cell)
        for (py, px), requested in by_patch.items():
            for start in range(0, len(requested), k):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("calibration diagnostic reached its declared time budget during direct ROI completion")
                cells_requested = np.asarray(requested[start:start+k], dtype=np.int64)
                padded = cells_requested[np.arange(k) % len(cells_requested)]
                seeds = np.stack([view.xdim[padded[:, 1]], view.ydim[padded[:, 0]]], axis=1)
                features, cells = _extract_normalized(runner, extractor, dataset, view,
                    py=py, px=px, frame=frame, patch_size=patch_size, t_win=t_win,
                    groups=groups, delta_frac=delta_frac, L=L, n_substeps=n_substeps,
                    t_scale=t_scale, seed=seed, initial_seeds=seeds)
                with torch.inference_mode():
                    row = model((torch.zeros((1, 1, 1, 1), device=device),
                                 torch.from_numpy(features[None]).to(device))).detach().cpu().numpy()[0]
                np.add.at(acc, (cells[:, 0], cells[:, 1]), row.astype(np.float64))
                np.add.at(count, (cells[:, 0], cells[:, 1]), 1)
    probability = np.full((view.height, view.width), np.nan, dtype=np.float32)
    covered = count > 0
    probability[covered] = (acc[covered] / count[covered]).astype(np.float32)
    count[view.mask] = 0
    base_count[view.mask] = 0
    probability[view.mask] = np.nan
    return probability, {
        "patch_count": int(len(inference_patches)),
        "successful_patch_count": int(len(inference_patches) - len(failures)),
        "failed_patch_count": int(len(failures)),
        "failed_patches": failures,
        "projected_seed_cell_count": int(count.sum()),
        "projected_grid_coverage": float(np.mean(covered & ~view.mask)),
        "prediction_count": count,
        "base_prediction_count": base_count,
        "supplementary_roi": supplementary_roi,
        "base_predictions_reused": existing_prediction is not None,
    }


def _binary_metrics(
    probability: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    known = np.asarray(positive, dtype=bool) | np.asarray(negative, dtype=bool)
    known &= np.isfinite(probability)
    prediction = np.asarray(probability, dtype=np.float32) >= float(threshold)
    reference = np.asarray(positive, dtype=bool)
    tp = int(np.count_nonzero(prediction & reference & known))
    fp = int(np.count_nonzero(prediction & ~reference & known))
    fn = int(np.count_nonzero(~prediction & reference & known))
    tn = int(np.count_nonzero(~prediction & ~reference & known))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "status": "ok" if known.any() else "undefined",
        "precision": float(precision) if known.any() else None,
        "recall": float(recall) if known.any() else None,
        "f1": float(f1) if known.any() else None,
        "iou": float(iou) if known.any() else None,
        "n_known": int(known.sum()),
        "n_positive": int(np.asarray(positive, dtype=bool).sum()),
        "n_negative": int(np.asarray(negative, dtype=bool).sum()),
        "n_unclassified_reference": int((~(np.asarray(positive, dtype=bool) | np.asarray(negative, dtype=bool))).sum()),
        "n_missing_known": int(((np.asarray(positive, dtype=bool) | np.asarray(negative, dtype=bool)) & ~np.isfinite(probability)).sum()),
        "n_predicted_positive": int(np.count_nonzero(prediction & known)),
    }


def _haller_response(
    probability: np.ndarray,
    view: Any,
    anchor_dir: pathlib.Path,
    *,
    threshold: float,
) -> dict[str, Any]:
    raw = np.asarray(np.load(anchor_dir / "anchor_state.npy", allow_pickle=False))
    solid = np.asarray(np.load(anchor_dir / "solid_mask.npy", allow_pickle=False), dtype=bool)
    invalid = np.asarray(np.load(anchor_dir / "invalid_mask.npy", allow_pickle=False), dtype=bool)
    positive = (raw == 1) & ~solid & ~invalid & ~view.mask
    predicted = np.asarray(probability, dtype=np.float32) >= float(threshold)
    predicted &= ~view.mask
    hits = int(np.count_nonzero(predicted & positive))
    positive_count = int(positive.sum())
    predicted_count = int(predicted.sum())
    metadata = _read_json(anchor_dir / "anchor_meta.json")
    return {
        "response_semantics": "train-only sparse positive Haller evidence; no dense negative",
        "positive_anchor_recall": float(hits / positive_count) if positive_count else 0.0,
        "predicted_region_positive_anchor_ratio": float(hits / predicted_count) if predicted_count else 0.0,
        "positive_anchor_count": positive_count,
        "predicted_positive_count": predicted_count,
        "positive_anchor_hit_count": hits,
        "failure_count": int(metadata.get("failure_count", 0)),
        "haller_source_identity": metadata.get("haller_source_identity"),
        "anchor_metadata_hash": metadata.get("metadata_hash"),
        "test_source_read": False,
        "test_frame_indices_read": [],
    }


def _render_figure(
    out_path: pathlib.Path,
    dataset_name: str,
    method_label: str,
    epoch: int,
    frames: tuple[int, ...],
    probabilities: Mapping[int, np.ndarray],
    states: Mapping[int, np.ndarray],
    view: Any,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(frames), figsize=(12, 5), squeeze=False)
    for col, frame in enumerate(frames):
        probability = np.asarray(probabilities[int(frame)], dtype=np.float32)
        state = np.asarray(states[int(frame)], dtype=np.int8)
        ax_prob = axes[0, col]
        ax_ref = axes[1, col]
        masked = probability.copy()
        masked[view.mask] = np.nan
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgray")
        ax_prob.imshow(masked, origin="lower", cmap=cmap, vmin=0, vmax=1)
        ax_prob.imshow(np.ma.masked_where(~view.mask, view.mask), origin="lower", cmap="gray_r", vmin=0, vmax=1)
        ax_prob.set_title(f"frame {frame}: gray=missing, black=solid")
        ax_ref.imshow(state == 1, origin="lower", cmap="Greens", vmin=0, vmax=1)
        ax_ref.set_title("P95/P80 reference")
        for axis in (ax_prob, ax_ref):
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        f"{dataset_name} — {method_label} epoch {epoch}, P95/P80 calibration inference",
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows or any(row[key] is None for row in rows):
        return None
    return float(np.mean([float(row[key]) for row in rows]))


def normalize_calibration_only_metadata(output: dict[str, Any]) -> dict[str, Any]:
    """Mark skipped physical measurements without rerunning saved predictions."""
    if not output.get("calibration_only"):
        return output
    for method, method_record in output["methods"].items():
        for epoch, budget in method_record["budgets"].items():
            for dataset in budget["datasets"].values():
                dataset["haller_mean"] = {"status": "not_evaluated",
                    "positive_anchor_recall": None, "predicted_region_positive_anchor_ratio": None,
                    "positive_anchor_count": None, "positive_anchor_hit_count": None}
                metrics = list(dataset.get("ivd_per_frame", [])) + [record["ivd"] for record in dataset["inference_per_frame"]]
                for metric in metrics:
                    if "n_unknown" in metric:
                        metric["legacy_grid_complement_of_observed_known"] = metric.pop("n_unknown")
            for macro in (budget["macro"], output["macro"][method][epoch]):
                macro["haller_positive_anchor_recall"] = None
                macro["haller_predicted_region_positive_anchor_ratio"] = None
                macro["haller_status"] = "not_evaluated"
    output["diagnostic_status"] = {"haller": "not_evaluated", "q": "not_evaluated"}
    output.setdefault("metric_semantics", {})["ivd"] = "own direct-support known P95/P80 diagnostic; cross-method errors are recomputed on common direct support in prediction_support_report.json"
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--input-root", type=pathlib.Path, required=True)
    parser.add_argument("--ticket03-root", type=pathlib.Path, required=True)
    parser.add_argument("--run-root", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=list(DATASET_ORDER))
    parser.add_argument("--methods", nargs="+", choices=("w1_p", "w1_hybrid"), default=["w1_p", "w1_hybrid"])
    parser.add_argument("--supplementary-roi", nargs=6, metavar="ROI", help="dataset frame y0 y1 x0 x1; declared before inference")
    parser.add_argument("--max-seconds", type=float, default=3600, help="total diagnostic time budget, at most 3600 seconds")
    parser.add_argument("--reuse-prediction-json", type=pathlib.Path, help="reuse existing direct probabilities/counts; infer only requested missing ROI starts")
    parser.add_argument("--roi-reference-status", choices=("unreviewed_candidate", "human_accepted_compact_core"), default="unreviewed_candidate")
    parser.add_argument("--roi-object-id", type=int)
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[100],
        help="checkpoint epochs to materialize; all must be completed checkpoints",
    )
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="skip train-only Haller diagnostic inference",
    )
    args = parser.parse_args(argv)
    started_at = time.monotonic()
    if not 0 < args.max_seconds <= 3600:
        raise ValueError("--max-seconds must be in (0,3600]")
    progress_records = []
    roi_request = None
    if args.supplementary_roi:
        roi_request = (args.supplementary_roi[0], *map(int, args.supplementary_roi[1:]))
        if roi_request[0] not in args.datasets or roi_request[1] not in CALIBRATION_FRAMES[roi_request[0]]:
            raise ValueError("ROI must belong to a declared calibration frame")
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive")
    epochs = tuple(dict.fromkeys(int(epoch) for epoch in args.epochs))
    if not epochs or any(epoch <= 0 for epoch in epochs):
        raise ValueError("--epochs must contain positive integers")
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))

    import torch
    import dataset
    import extractor
    import standard_ivd
    import ticket04_real_training as runner
    import train_kaggle
    import weak_labels
    import weak_supervision_contract as contract
    import w1_p

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA inference but CUDA is unavailable")
    ticket03_root = args.ticket03_root.resolve()
    run_root = args.run_root.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite inference output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "run_declaration.json").write_text(json.dumps({"datasets": args.datasets,
        "methods": args.methods, "epochs": list(epochs), "supplementary_roi": roi_request,
        "max_seconds": args.max_seconds, "object_reference": args.roi_reference_status,
        "roi_object_id": args.roi_object_id,
        "reused_prediction_artifact": str(args.reuse_prediction_json) if args.reuse_prediction_json else None,
        "supplementary_sampling": "missing true grid starts; grouped and padded to original model input size; no interpolation",
        "test_source_read": False}, indent=2), encoding="utf-8")

    run_manifest = _read_json(run_root / "run_manifest.json")
    protocol_manifest = runner.protocol.read_p95p80_protocol_manifest(
        ticket03_root / "p95p80_protocol_manifest.json"
    )
    # The protocol manifest is an external, relocated artifact.  Git cannot
    # establish that the run and calibration arrays came from the same source
    # contract, so fail before model inference if their recorded identities
    # differ.
    if run_manifest.get("protocol_manifest_hash") != protocol_manifest["manifest_hash"]:
        raise ValueError("run/protocol manifest identity mismatch")
    reused_artifact = _read_json(args.reuse_prediction_json) if args.reuse_prediction_json else None
    if reused_artifact is not None and reused_artifact.get("protocol_manifest_hash") != protocol_manifest["manifest_hash"]:
        raise ValueError("reused prediction/protocol identity mismatch")
    cfg = train_kaggle.load_config(str(args.config.resolve()))
    data_cfg = dict(cfg.get("data", {}))
    patch_size_values = tuple(int(v) for v in data_cfg.get("patch_size", (32, 32)))
    if len(patch_size_values) != 2:
        raise ValueError("patch_size must contain exactly two dimensions")
    patch_size: tuple[int, int] = (patch_size_values[0], patch_size_values[1])
    stride = tuple(int(v) for v in data_cfg.get("stride", (16, 16)))
    groups_values = tuple(int(v) for v in data_cfg.get("groups", (8, 8)))
    if len(groups_values) != 2:
        raise ValueError("groups must contain exactly two dimensions")
    groups: tuple[int, int] = (groups_values[0], groups_values[1])
    t_win = int(data_cfg.get("t_win", 24))
    window_step = int(data_cfg.get("window_step", 4))
    delta_frac = float(data_cfg.get("delta_frac", 0.05))
    L = int(data_cfg.get("L", 16))
    n_substeps = int(data_cfg.get("n_substeps", 4))
    t_scale = float(data_cfg.get("t_scale", 0.25))
    training = dict(run_manifest["training"])
    base_seed = int(training["seed"])
    anchor_hash = str(run_manifest["anchor_hash"])
    dataset_config = run_manifest["dataset_config"]
    split_config = run_manifest["split_config"]
    sampling_config = run_manifest["sampling_config"]

    views = [
        runner.load_dataset_view(
            args.input_root.resolve(),
            ticket03_root,
            name,
            protocol_manifest,
            patch_size=patch_size,
            stride=stride,
            t_win=t_win,
            window_step=window_step,
            groups=groups,
            delta_frac=delta_frac,
        )
        for name in args.datasets
    ]
    calibration_artifacts = {
        view.name: _load_relocated_calibration_artifact(
            w1_p,
            ticket03_root / "datasets" / view.name / "calibration_reference",
            protocol_manifest,
        )
        for view in views
    }
    standard_frame_provenance = {
        view.name: _load_standard_ivd_frame_provenance(
            ticket03_root / "datasets" / view.name / "standard_ivd",
            protocol_manifest=protocol_manifest,
        )
        for view in views
    }

    methods = (
        ("w1_p", "W1-P", contract.MODE_W1_P),
        ("w1_hybrid", "W1-Hybrid", contract.MODE_W1_HYBRID),
    )
    all_results: dict[str, Any] = {}
    macro: dict[str, Any] = {}
    for method_index, (method_name, method_label, mode) in enumerate(methods):
        if method_name not in args.methods:
            continue
        budget_results: dict[str, Any] = {}
        for epoch in epochs:
            checkpoint = (
                run_root
                / "methods"
                / method_name
                / "checkpoints"
                / f"epoch_{epoch:04d}.pt"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing completed checkpoint: {checkpoint}")
            model = train_kaggle.build_model_from_config(cfg, mode=mode)
            model.to(device)
            expected_anchor = None if mode == contract.MODE_W1_P else anchor_hash
            load_result = contract.load_checkpoint(
                checkpoint,
                student=model,
                device=device,
                expected_mode=mode,
                expected_feature_schema=contract.FEATURE_SCHEMA_7,
                expected_dataset_config=dataset_config,
                expected_split_config=split_config,
                expected_sampling_config=sampling_config,
                expected_label_source=contract.LABEL_SOURCE_P95P80,
                expected_sampling_source=contract.LABEL_SOURCE_P95P80,
                expected_anchor_hash=expected_anchor,
                load_mode="inference",
            )
            model.eval()
            method_rows: dict[str, Any] = {}
            dataset_ivd_rows: list[dict[str, Any]] = []
            dataset_q_rows: list[dict[str, Any]] = []
            dataset_haller_rows: list[dict[str, Any]] = []
            for view in views:
                name = view.name
                cal_frames = tuple(int(frame) for frame in CALIBRATION_FRAMES[name])
                haller_frames = (
                    ()
                    if args.calibration_only
                    else _snap_train_frames(
                        runner, view, HALLER_REFERENCE_FRAMES[name], t_win
                    )
                )
                calibration_artifact = calibration_artifacts[name]
                cal_state_by_frame = {
                    int(frame): np.asarray(calibration_artifact.state[row], dtype=np.int8)
                    for row, frame in enumerate(calibration_artifact.frame_indices)
                }
                if any(frame not in cal_state_by_frame for frame in cal_frames):
                    raise ValueError(f"calibration reference missing frame for {name}")
                frame_records: list[dict[str, Any]] = []
                prediction_records: list[dict[str, Any]] = []
                probabilities: dict[int, np.ndarray] = {}
                for frame in cal_frames:
                    existing_prediction = None
                    if reused_artifact is not None:
                        reused_dataset = reused_artifact["methods"][method_name]["budgets"][str(epoch)]["datasets"][name]
                        reused_record = next(row for row in reused_dataset["inference_per_frame"] if int(row["frame"]) == frame)
                        source_paths = []
                        for key in ("prediction_path", "prediction_count_path"):
                            source_path = pathlib.Path(reused_record[key])
                            if not source_path.is_absolute():
                                source_path = args.reuse_prediction_json.parent / source_path
                            source_paths.append(source_path)
                        existing_prediction = tuple(np.load(path, allow_pickle=False) for path in source_paths)
                    probability, projection = _infer_frame(
                        runner,
                        dataset,
                        extractor,
                        view,
                        model,
                        frame=frame,
                        frame_split="calibration",
                        patch_size=patch_size,
                        t_win=t_win,
                        window_step=window_step,
                        groups=groups,
                        delta_frac=delta_frac,
                        L=L,
                        n_substeps=n_substeps,
                        t_scale=t_scale,
                        seed=base_seed + method_index,
                        device=device,
                        inference_batch_size=args.inference_batch_size,
                        supplementary_roi=(tuple(roi_request[2:]) if roi_request and roi_request[:2] == (name, frame) else None),
                        deadline=started_at + args.max_seconds,
                        existing_prediction=existing_prediction,
                    )
                    count = projection.pop("prediction_count")
                    base_count = projection.pop("base_prediction_count")
                    cal_row = list(map(int, calibration_artifact.frame_indices)).index(frame)
                    support_diagnostic = support_summary(count, cal_state_by_frame[frame],
                        calibration_artifact.valid_fluid_mask[cal_row], view.mask,
                        calibration_artifact.invalid_mask[cal_row])
                    base_diagnostic = support_summary(base_count, cal_state_by_frame[frame],
                        calibration_artifact.valid_fluid_mask[cal_row], view.mask,
                        calibration_artifact.invalid_mask[cal_row])
                    probabilities[frame] = probability
                    state = cal_state_by_frame[frame]
                    ivd_metric = _binary_metrics(
                        probability,
                        state == 1,
                        state == 0,
                        threshold=args.prediction_threshold,
                    )
                    q_metric = None
                    if not args.calibration_only:
                        row = view.standard_row_by_frame[frame]
                        valid = (
                            np.asarray(view.standard_valid[row], dtype=bool)
                            & ~np.asarray(view.standard_invalid[row], dtype=bool)
                            & ~np.asarray(view.standard_solid, dtype=bool)
                        )
                        u = np.asarray(view.u[frame], dtype=np.float64)
                        v = np.asarray(view.v[frame], dtype=np.float64)
                        q = np.asarray(
                            weak_labels.q_criterion(
                                u[None], v[None], view.xdim, view.ydim
                            )[0],
                            dtype=np.float32,
                        )
                        q_known = valid & np.isfinite(q)
                        # Q is retained only as a secondary diagnostic; its
                        # denominator is restricted to valid-fluid, finite-Q
                        # cells and is omitted by calibration-only runs.
                        q_metric = _binary_metrics(
                            np.where(q_known, probability, 0.0),
                            (q > 0.0) & q_known,
                            (q <= 0.0) & q_known,
                            threshold=args.prediction_threshold,
                        )
                    calibration_provenance = _calibration_frame_provenance(
                        calibration_artifact,
                        standard_frame_provenance[name][frame],
                        protocol_manifest=protocol_manifest,
                    )
                    prediction_path = (
                        out_dir
                        / "predictions"
                        / method_name
                        / f"epoch_{epoch:04d}"
                        / f"{name}_frame_{frame:06d}.npy"
                    )
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(prediction_path, np.asarray(probability, dtype=np.float32))
                    count_path = prediction_path.with_name(prediction_path.stem + "_count.npy")
                    support_path = prediction_path.with_name(prediction_path.stem + "_support.npy")
                    base_count_path = prediction_path.with_name(prediction_path.stem + "_base_count.npy")
                    np.save(count_path, count)
                    np.save(support_path, count > 0)
                    np.save(base_count_path, base_count)
                    prediction_record = {
                        "dataset": name,
                        "frame": int(frame),
                        "split": "calibration",
                        "prediction_path": str(prediction_path.relative_to(out_dir)),
                        "prediction_count_path": str(count_path.relative_to(out_dir)),
                        "prediction_support_path": str(support_path.relative_to(out_dir)),
                        "base_prediction_count_path": str(base_count_path.relative_to(out_dir)),
                        "prediction_support": support_diagnostic,
                        "base_prediction_support": base_diagnostic,
                        "shape": [int(size) for size in probability.shape],
                        "p95p80_provenance": calibration_provenance["p95p80_provenance"],
                    }
                    prediction_records.append(prediction_record)
                    frame_records.append(
                        {
                            **prediction_record,
                            "ivd": ivd_metric,
                            "q": q_metric,
                            "projection": projection,
                            "prediction_min": float(np.nanmin(probability)) if np.isfinite(probability).any() else None,
                            "prediction_max": float(np.nanmax(probability)) if np.isfinite(probability).any() else None,
                            "prediction_mean": float(np.nanmean(probability)) if np.isfinite(probability).any() else None,
                            "calibration_provenance": calibration_provenance,
                        }
                    )
                    progress_records.append({"method": method_name, "epoch": epoch, **frame_records[-1]})
                    (out_dir / "completed_frame_records.json").write_text(json.dumps(progress_records,
                        indent=2, allow_nan=False), encoding="utf-8")
                    print(json.dumps({"completed": [method_name, name, frame], "elapsed_seconds": time.monotonic() - started_at,
                        "valid_fluid": support_diagnostic["denominators"]["valid_fluid"],
                        "failed_patches": projection["failed_patch_count"]}), flush=True)
                figure_path = out_dir / f"method_{method_name}_p95p80_epoch{epoch:04d}_{name}.png"
                _render_figure(
                    figure_path,
                    name,
                    method_label,
                    epoch,
                    cal_frames,
                    probabilities,
                    cal_state_by_frame,
                    view,
                )
                haller_records: list[dict[str, Any]] = []
                haller_probabilities: dict[int, np.ndarray] = {}
                for frame in haller_frames:
                    probability, projection = _infer_frame(
                        runner,
                        dataset,
                        extractor,
                        view,
                        model,
                        frame=frame,
                        frame_split="train",
                        patch_size=patch_size,
                        t_win=t_win,
                        window_step=window_step,
                        groups=groups,
                        delta_frac=delta_frac,
                        L=L,
                        n_substeps=n_substeps,
                        t_scale=t_scale,
                        seed=base_seed + method_index,
                        device=device,
                        inference_batch_size=args.inference_batch_size,
                    )
                    haller_probabilities[frame] = probability
                    projection.pop("prediction_count")
                    projection.pop("base_prediction_count")
                    anchor_dir = (
                        run_root / "haller_anchor_train" / name / f"frame_{frame:06d}"
                    )
                    if not anchor_dir.is_dir():
                        # The current Ticket 04 artifact keeps the dataset name as
                        # an additional container directory.  Resolve that layout
                        # without copying or rewriting the train-only artifact.
                        nested_anchor_dir = (
                            run_root
                            / "haller_anchor_train"
                            / name
                            / name
                            / f"frame_{frame:06d}"
                        )
                        if nested_anchor_dir.is_dir():
                            anchor_dir = nested_anchor_dir
                    if not anchor_dir.is_dir():
                        raise FileNotFoundError(f"missing train Haller frame artifact: {anchor_dir}")
                    response = _haller_response(
                        probability,
                        view,
                        anchor_dir,
                        threshold=args.prediction_threshold,
                    )
                    response.update({"frame": int(frame), "split": "train", "projection": projection})
                    haller_records.append(response)
                ivd_rows = [row["ivd"] for row in frame_records]
                q_rows = [
                    row["q"]
                    for row in frame_records
                    if isinstance(row.get("q"), Mapping)
                ]
                haller_mean = {
                    "positive_anchor_recall": _mean_metric(haller_records, "positive_anchor_recall"),
                    "predicted_region_positive_anchor_ratio": _mean_metric(
                        haller_records, "predicted_region_positive_anchor_ratio"
                    ),
                    "positive_anchor_count": int(
                        sum(int(row["positive_anchor_count"]) for row in haller_records)
                    ),
                    "positive_anchor_hit_count": int(
                        sum(int(row["positive_anchor_hit_count"]) for row in haller_records)
                    ),
                }
                dataset_result = {
                    "calibration_frames": list(cal_frames),
                    "haller_train_frames": list(haller_frames),
                    "thresholds": calibration_artifact.metadata["thresholds"],
                    "ivd_mean": {
                        "f1": _mean_metric(ivd_rows, "f1"),
                        "iou": _mean_metric(ivd_rows, "iou"),
                    },
                    "q_mean": (
                        {"f1": _mean_metric(q_rows, "f1"), "iou": _mean_metric(q_rows, "iou")}
                        if q_rows
                        else None
                    ),
                    "ivd_per_frame": ivd_rows,
                    "q_per_frame": q_rows,
                    "haller_mean": haller_mean,
                    "haller_per_frame": haller_records,
                    "inference_per_frame": frame_records,
                    "prediction_records": prediction_records,
                    "p95p80_provenance": [
                        item["p95p80_provenance"] for item in prediction_records
                    ],
                    "figure": None if figure_path is None else str(figure_path),
                }
                method_rows[name] = dataset_result
                dataset_ivd_rows.append(dataset_result["ivd_mean"])
                if dataset_result["q_mean"] is not None:
                    dataset_q_rows.append(dataset_result["q_mean"])
                dataset_haller_rows.append(dataset_result["haller_mean"])
            budget_macro = {
                "label": method_label,
                "ivd_f1": _mean_metric(dataset_ivd_rows, "f1"),
                "ivd_iou": _mean_metric(dataset_ivd_rows, "iou"),
                "q_f1": _mean_metric(dataset_q_rows, "f1") if dataset_q_rows else None,
                "q_iou": _mean_metric(dataset_q_rows, "iou") if dataset_q_rows else None,
                "haller_positive_anchor_recall": _mean_metric(
                    dataset_haller_rows, "positive_anchor_recall"
                ),
                "haller_predicted_region_positive_anchor_ratio": _mean_metric(
                    dataset_haller_rows, "predicted_region_positive_anchor_ratio"
                ),
            }
            budget_results[str(epoch)] = {
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": int(load_result["epoch"]) + 1,
                "datasets": method_rows,
                "macro": budget_macro,
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        all_results[method_name] = {
            "label": method_label,
            "mode": mode,
            "budgets": budget_results,
        }
        macro[method_name] = {
            str(epoch): budget_results[str(epoch)]["macro"] for epoch in epochs
        }

    output = {
        "artifact_type": "ticket05_calibration_prediction_artifact",
        "schema_version": "criterion-ambiguity-ticket05-calibration-predictions-v1",
        "prediction_support_schema": "direct-grid-count-v1",
        "elapsed_seconds": time.monotonic() - started_at,
        "supplementary_roi_declared": roi_request,
        "roi_reference": {"status": args.roi_reference_status, "object_id": args.roi_object_id},
        "reused_prediction_artifact": str(args.reuse_prediction_json) if args.reuse_prediction_json else None,
        "missing_probability": "NaN (no direct prediction)",
        "protocol_version": protocol_manifest["protocol_version"],
        "protocol_manifest_hash": protocol_manifest["manifest_hash"],
        "standard_ivd_source_identity": standard_ivd.SOURCE_IDENTITY,
        "run_root": str(run_root),
        "epochs": list(epochs),
        "calibration_only": bool(args.calibration_only),
        "predictions_root": str((out_dir / "predictions").resolve()),
        "run_manifest_identity": {
            "execution_namespace": run_manifest.get("ticket04_execution_namespace"),
            "anchor_hash": anchor_hash,
            "training_seed": base_seed,
            "checkpoint_epochs": training.get("checkpoint_epochs"),
        },
        "frame_policy": {
            "calibration_frames": {name: list(frames) for name, frames in CALIBRATION_FRAMES.items()},
            "haller_frames_requested": {
                name: list(frames) for name, frames in HALLER_REFERENCE_FRAMES.items()
            },
            "calibration_window_policy": {
                "t_win": t_win,
                "window_step": window_step,
                "cross_split": False,
            },
            "haller_source_split": None if args.calibration_only else "train",
            "haller_frames_read": not args.calibration_only,
            "boussinesq_frame_1300_read": False,
            "test_source_read": False,
            "test_frame_indices_read": [],
            "final_test_relationships": "not_read",
        },
        "prediction_threshold": float(args.prediction_threshold),
        "inference_device": str(device),
        "inference_batch_size": int(args.inference_batch_size),
        "methods": all_results,
        "macro": macro,
        "metric_semantics": {
            "ivd": "directly observed calibration reference known P95 positive/P80 negative only; unknown, solid, invalid and missing predictions excluded; cross-method errors use the support report intersection",
            "q": "Q > 0 positive and Q <= 0 negative on valid-fluid, non-solid, finite-Q cells",
            "haller": "train-only sparse positive anchor response; no dense negative or dense Haller GT",
            "connected_component_filter": "already applied in Ticket 03 target/calibration artifact after P95 thresholding; not part of IVD",
        },
        "input_identity_policy": {
            "raw_source": "Ticket 03 loader raw u/v/geometry/coordinates",
            "standard_ivd": "Ticket 03 standard IVD artifact bound to canonical standard_ivd source",
            "calibration_reference": "Ticket 03 P95/P80 calibration reference",
            "haller_source": (
                "not_read (calibration-only run)"
                if args.calibration_only
                else "current Ticket 04 train-only Haller artifact"
            ),
            "legacy_ivd_read": False,
            "legacy_label_read": False,
        },
    }
    output_path = out_dir / "ticket05_calibration_predictions.json"
    normalize_calibration_only_metadata(output)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(out_dir), "json": str(output_path), "macro": macro}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
