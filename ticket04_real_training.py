"""Ticket 04 real W1-P/W1-Hybrid/W2 execution on the current source contract.

This runner is deliberately separate from the historical train_kaggle data
path. It consumes only the Ticket 03 raw-source seam, current standard-IVD
artifact, train target, calibration reference, and freshly generated
train-only sparse Haller artifacts. It never opens a legacy ivd.npy or
label_field.npy and never indexes a test frame.

The command is an execution harness rather than a fourth IVD implementation:
the numerical source remains standard_ivd, and the target/fusion/loss seams
remain in w1_p and minimal_training_closure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import dataset
import experiment_protocol as protocol
import extractor
import haller_anchors
import minimal_training_closure as closure
import sparse_anchor_study as sparse
import standard_ivd
import ticket03_standard_ivd_artifacts as ticket03
import train_kaggle
import w1_p
import w2
import weak_labels
import weak_supervision_contract as contract


DATASET_ORDER = (
    "boussinesq",
    "cylinder2d",
    "doublegyre2d",
    "fourcenters2d",
    "jungtelziemniak2d",
    "pipedcylinder2d",
)
if tuple(protocol.DATASET_NAMES) != DATASET_ORDER:
    raise RuntimeError(
        "Ticket 04 dataset order drifted from the frozen six-dataset protocol"
    )

CHECKPOINT_INTERVAL = 10
DEFAULT_EPOCHS = 100
DEFAULT_SAMPLES_PER_EPOCH = 20_000
DEFAULT_BATCH_SIZE = 100
DEFAULT_NUM_WORKERS = 8
DEFAULT_HALLER_WORKERS = 8
DEFAULT_VARIANCE_GATE = 0.05
DEFAULT_SEED = 0
DEFAULT_EXECUTION_NAMESPACE = "ticket04-real-w1p-w1hybrid-w2-v1"
TICKET04_HALLER_BACKEND = haller_anchors.BACKEND_NUMBACS
TICKET04_HALLER_NLEVS = haller_anchors.NUMBACS_TICKET04_NLEVS
TICKET04_HALLER_PARAMETER_VERSION = (
    haller_anchors.NUMBACS_TICKET04_PARAMETER_VERSION
)
TICKET04_HALLER_SOURCE_IDENTITY = {
    "source": haller_anchors.SOURCE_TRAIN,
    "backend": TICKET04_HALLER_BACKEND,
    "method": "numbacs.extraction.elliptic.rotcohvrt",
    "upstream_version": "0.2.0",
    "upstream_commit": "c067f542543f5dd4ae3dc45fc506213e8d98b845",
    "nlevs": TICKET04_HALLER_NLEVS,
    "parameter_version": TICKET04_HALLER_PARAMETER_VERSION,
    "backend_version": haller_anchors.NUMBACS_TICKET04_BACKEND_VERSION,
}

METHODS = (
    ("w1_p", contract.MODE_W1_P),
    ("w1_hybrid", contract.MODE_W1_HYBRID),
    ("w2", contract.MODE_W2_ON_HYBRID),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(child) for child in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(value)


def checkpoint_epochs(
    total_epochs: int, interval: int = CHECKPOINT_INTERVAL
) -> tuple[int, ...]:
    """Return one-based completed epochs at which checkpoints must be saved."""
    if isinstance(total_epochs, (bool, np.bool_)) or int(total_epochs) != total_epochs:
        raise ValueError("total_epochs must be an integer")
    if isinstance(interval, (bool, np.bool_)) or int(interval) != interval:
        raise ValueError("checkpoint interval must be an integer")
    total_epochs = int(total_epochs)
    interval = int(interval)
    if total_epochs <= 0 or interval <= 0:
        raise ValueError("total_epochs and checkpoint interval must be positive")
    result = [epoch for epoch in range(interval, total_epochs + 1, interval)]
    if total_epochs not in result:
        result.append(total_epochs)
    return tuple(result)


def _ranges(record: Mapping[str, Any], split_name: str) -> tuple[tuple[int, int], ...]:
    partitions = record.get("frame_partitions")
    if not isinstance(partitions, Mapping) or split_name not in partitions:
        raise ValueError(f"protocol record has no {split_name!r} partition")
    ranges = partitions[split_name].get("ranges")
    if not isinstance(ranges, (list, tuple)):
        raise ValueError(f"protocol record {split_name!r}.ranges is invalid")
    result = []
    for pair in ranges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"invalid {split_name} half-open range: {pair!r}")
        start, end = int(pair[0]), int(pair[1])
        if start < 0 or end <= start:
            raise ValueError(f"invalid {split_name} half-open range: {pair!r}")
        result.append((start, end))
    return tuple(result)


def validate_train_window_starts(
    record: Mapping[str, Any],
    dataset_name: str,
    starts: Sequence[int],
    *,
    t_win: int,
    window_step: int,
) -> tuple[int, ...]:
    """Validate complete train-only windows and explicitly reject audit frames."""
    if int(t_win) != t_win or int(window_step) != window_step:
        raise ValueError("t_win and window_step must be integers")
    t_win, window_step = int(t_win), int(window_step)
    if t_win <= 0 or window_step <= 0:
        raise ValueError("t_win and window_step must be positive")
    train_ranges = _ranges(record, "train")
    audit_frames = {
        int(frame)
        for frame in record.get(
            "development_audit_frames",
            record.get("development_audit_only_frames", ()),
        )
    }
    audit_ranges = _ranges(record, "development_audit")
    audit_frames.update(
        frame
        for start, end in audit_ranges
        for frame in range(start, end)
    )
    result = []
    for raw_start in starts:
        if isinstance(raw_start, (bool, np.bool_)) or int(raw_start) != raw_start:
            raise ValueError(
                f"dataset={dataset_name} train split window start is not integer: "
                f"{raw_start!r}"
            )
        start = int(raw_start)
        end = start + t_win
        if any(start <= frame < end for frame in audit_frames):
            raise ValueError(
                f"dataset={dataset_name} window [{start}, {end}) contains a "
                "development/audit frame"
            )
        containing = [
            (range_start, range_end)
            for range_start, range_end in train_ranges
            if range_start <= start and end <= range_end
        ]
        if not containing:
            raise ValueError(
                f"dataset={dataset_name} window [{start}, {end}) is not fully "
                "inside the train split"
            )
        range_start, _range_end = containing[0]
        if (start - range_start) % window_step:
            raise ValueError(
                f"dataset={dataset_name} train split window start={start} is "
                f"not aligned to window_step={window_step}"
            )
        result.append(start)
    return tuple(result)


def haller_result_to_sparse_state(
    haller_state: Any,
    *,
    solid_mask: Any,
    invalid_mask: Any,
    failure_count: int,
) -> np.ndarray:
    """Map raw Haller output to sparse positive evidence without dense negatives."""
    state = np.asarray(haller_state, dtype=np.int8)
    solid = np.asarray(solid_mask, dtype=bool)
    invalid = np.asarray(invalid_mask, dtype=bool)
    if state.ndim != 2 or solid.shape != state.shape or invalid.shape != state.shape:
        raise ValueError("Haller state and masks must be matching two-dimensional arrays")
    if np.any(solid & invalid):
        raise ValueError("Haller solid and invalid masks overlap")
    if not np.all(np.isin(state, [-1, 0, 1])):
        raise ValueError("Haller raw state contains an unsupported negative encoding")
    failure_count = int(failure_count)
    if failure_count < 0:
        raise ValueError("Haller failure_count must be non-negative")
    if np.any(state == 0):
        raise ValueError(
            "Haller sparse evidence must not emit a dense negative state"
        )
    result = np.full(state.shape, sparse.STATE_UNCOVERED, dtype="<U24")
    result[solid] = sparse.STATE_SOLID
    result[invalid] = sparse.STATE_INVALID
    result[(state == 1) & ~solid & ~invalid] = sparse.STATE_POSITIVE_ANCHOR
    if failure_count:
        # The current extractor reports frame-level failure and encodes the
        # failed cells as UNKNOWN.  Preserve any explicit positive evidence
        # if a future backend reports a partial failure; never turn all
        # fluid cells into a dense negative or erase a positive anchor.
        result[(state == -1) & (~solid) & (~invalid)] = (
            sparse.STATE_EXTRACTOR_FAILURE
        )
    return result


@dataclass
class DatasetView:
    name: str
    u: Any
    v: Any
    mask: np.ndarray
    xdim: np.ndarray
    ydim: np.ndarray
    tdim: np.ndarray
    record: dict[str, Any]
    standard_meta: dict[str, Any]
    standard_ivd: Any
    standard_valid: Any
    standard_invalid: Any
    standard_solid: np.ndarray
    standard_row_by_frame: dict[int, int]
    target: w1_p.W1PLabelArtifact
    target_meta: dict[str, Any]
    calibration_meta: dict[str, Any]
    target_row_by_frame: dict[int, int]
    train_frames: tuple[int, ...]
    train_starts: tuple[int, ...]
    normalization: dict[str, float | int]
    patches: tuple[tuple[int, int], ...]
    usable_patches: tuple[tuple[int, int], ...]
    positive_pool_by_frame: dict[int, tuple[tuple[int, int, int], ...]]
    negative_pool_by_frame: dict[int, tuple[tuple[int, int, int], ...]]
    batch_frame_choices: tuple[int, ...]
    haller_dirs: dict[int, Path] = field(default_factory=dict)
    haller_info: dict[int, dict[str, Any]] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.u.shape[1])

    @property
    def width(self) -> int:
        return int(self.u.shape[2])

    @property
    def ds_id(self) -> int:
        return DATASET_ORDER.index(self.name)


def _load_ticket03_target(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    expected_split: str,
) -> w1_p.W1PLabelArtifact:
    metadata = _read_json(root / "target_meta.json")
    w1_p._w1p_validate_ticket03_metadata(
        metadata,
        root=root.resolve(),
        expected_split=expected_split,
        protocol_manifest=manifest,
    )
    arrays: dict[str, Any] = {}
    for name, (filename, _dtype) in w1_p._TICKET03_TARGET_FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Ticket 03 {expected_split} array is missing: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.dtype(_dtype):
            raise ValueError(
                f"Ticket 03 {expected_split} {filename} dtype mismatch: "
                f"{array.dtype} != {np.dtype(_dtype)}"
            )
        arrays[name] = array
    w1_p._w1p_validate_arrays(arrays, metadata)
    return w1_p.W1PLabelArtifact(
        root=root.resolve(),
        metadata=metadata,
        **arrays,
    )


def _compute_train_normalization(view: DatasetView) -> dict[str, float | int]:
    count = 0
    ivd_sum = 0.0
    ivd_sum_sq = 0.0
    speed_max = 0.0
    for frame in view.train_frames:
        row = view.standard_row_by_frame[frame]
        ivd_frame = np.asarray(view.standard_ivd[row], dtype=np.float64)
        valid = np.asarray(view.standard_valid[row], dtype=bool)
        valid &= ~np.asarray(view.standard_invalid[row], dtype=bool)
        valid &= np.isfinite(ivd_frame)
        values = ivd_frame[valid]
        if values.size:
            count += int(values.size)
            ivd_sum += float(values.sum(dtype=np.float64))
            ivd_sum_sq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        speed = np.hypot(
            np.asarray(view.u[frame], dtype=np.float32),
            np.asarray(view.v[frame], dtype=np.float32),
        )
        speed_values = speed[valid]
        speed_values = speed_values[np.isfinite(speed_values)]
        if speed_values.size:
            speed_max = max(speed_max, float(speed_values.max()))
    if count <= 0 or speed_max <= 0.0:
        raise ValueError(
            f"dataset={view.name} has no valid train normalization support"
        )
    mean = ivd_sum / count
    variance = max(ivd_sum_sq / count - mean * mean, 0.0)
    sigma = math.sqrt(variance)
    return {
        "ivd_mu": float(mean),
        "ivd_sigma": float(sigma),
        "speed_max": float(speed_max),
        "ivd_valid_cell_count": int(count),
        "statistics_split": "train",
    }


def _patch_is_usable(
    mask: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
    patch_yx: tuple[int, int],
    *,
    patch_size: tuple[int, int],
    groups: tuple[int, int],
    delta_frac: float,
) -> bool:
    seeds = np.asarray(
        extractor.seeding_grid(
            patch_yx, patch_size, xdim, ydim, groups, delta_frac
        ),
        dtype=np.float64,
    )
    geometry = extractor.patch_geometry(patch_yx, patch_size, xdim, ydim)
    center = np.array([geometry["cx"], geometry["cy"]], dtype=np.float64)
    j, i = extractor.nearest_cell(seeds[:, 0], seeds[:, 1], xdim, ydim)
    j = np.clip(j, 0, mask.shape[0] - 1)
    i = np.clip(i, 0, mask.shape[1] - 1)
    solid_indices = np.flatnonzero(mask[j, i])
    if solid_indices.size == 0:
        return True
    fractions = np.linspace(0.0, 1.0, 201, dtype=np.float64)[:, None]
    for index in solid_indices:
        points = seeds[index][None, :] + fractions * (
            center[None, :] - seeds[index][None, :]
        )
        pj, pi = extractor.nearest_cell(points[:, 0], points[:, 1], xdim, ydim)
        pj = np.clip(pj, 0, mask.shape[0] - 1)
        pi = np.clip(pi, 0, mask.shape[1] - 1)
        if np.any(~mask[pj, pi]):
            return True
    return False


def _build_pools(
    view: DatasetView,
    *,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
    groups: tuple[int, int],
    delta_frac: float,
) -> None:
    patches = tuple(
        dataset.patch_locations(view.height, view.width, patch_size, stride)
    )
    usable = tuple(
        patch
        for patch in patches
        if _patch_is_usable(
            view.mask,
            view.xdim,
            view.ydim,
            patch,
            patch_size=patch_size,
            groups=groups,
            delta_frac=delta_frac,
        )
    )
    usable_set = set(usable)
    positive: dict[int, list[tuple[int, int, int]]] = {
        frame: [] for frame in view.train_starts
    }
    negative: dict[int, list[tuple[int, int, int]]] = {
        frame: [] for frame in view.train_starts
    }
    for row, frame in enumerate(view.target.frame_indices):
        frame = int(frame)
        if frame not in positive:
            continue
        positive_map = weak_labels.patch_positive_map(
            np.asarray(view.target.positive_mask[row], dtype=bool),
            view.xdim,
            view.ydim,
            patch_size,
            stride,
            groups,
            delta_frac,
        ).reshape(-1)
        for patch_index, patch in enumerate(patches):
            if patch not in usable_set:
                continue
            item = (int(patch[0]), int(patch[1]), frame)
            (positive if bool(positive_map[patch_index]) else negative)[frame].append(
                item
            )
    view.patches = patches
    view.usable_patches = usable
    view.positive_pool_by_frame = {
        frame: tuple(items) for frame, items in positive.items()
    }
    view.negative_pool_by_frame = {
        frame: tuple(items) for frame, items in negative.items()
    }
    view.batch_frame_choices = tuple(
        frame
        for frame in view.train_starts
        if view.positive_pool_by_frame[frame]
        and view.negative_pool_by_frame[frame]
    )
    if not view.batch_frame_choices:
        raise ValueError(
            f"dataset={view.name} has no train window with both positive and "
            "negative usable patch pools"
        )


def load_dataset_view(
    data_root: str | Path,
    ticket03_root: str | Path,
    name: str,
    manifest: Mapping[str, Any],
    *,
    patch_size: tuple[int, int] = (32, 32),
    stride: tuple[int, int] = (16, 16),
    t_win: int = 24,
    window_step: int = 4,
    groups: tuple[int, int] = (8, 8),
    delta_frac: float = 0.05,
) -> DatasetView:
    """Load only current raw/train/calibration seams for one dataset."""
    if name not in DATASET_ORDER:
        raise ValueError(f"dataset={name!r} is not in the frozen six-dataset order")
    data_root = Path(data_root)
    ticket03_root = Path(ticket03_root)
    source = ticket03.load_raw_dataset_source(data_root, name)
    record = copy.deepcopy(dict(manifest["datasets"][name]))
    shape = tuple(int(value) for value in source["shape"])
    if tuple(source["u"].shape) != shape or tuple(source["v"].shape) != shape:
        raise ValueError(f"dataset={name} raw u/v shape mismatch")
    if tuple(np.asarray(source["mask"]).shape) != shape[1:]:
        raise ValueError(f"dataset={name} raw geometry mask shape mismatch")
    xdim = np.asarray(source["xdim"], dtype=np.float64)
    ydim = np.asarray(source["ydim"], dtype=np.float64)
    tdim = np.asarray(source["tdim"], dtype=np.float64)
    if (len(tdim), len(ydim), len(xdim)) != shape:
        raise ValueError(f"dataset={name} coordinate lengths mismatch")
    dx = float(np.diff(xdim)[0])
    dy = float(np.diff(ydim)[0])

    standard_root = ticket03_root / "datasets" / name / "standard_ivd"
    standard_meta = _read_json(standard_root / "standard_ivd_meta.json")
    if standard_meta.get("standard_ivd_source_identity") != standard_ivd.SOURCE_IDENTITY:
        raise ValueError(f"dataset={name} standard IVD source identity mismatch")
    if standard_meta.get("formula") != standard_ivd.FORMULA:
        raise ValueError(f"dataset={name} standard IVD formula mismatch")
    if standard_meta.get("mean_scope") != "declared_valid_fluid_support":
        raise ValueError(f"dataset={name} standard IVD mean scope mismatch")
    declared_spacing = standard_meta.get("grid_spacing", {})
    if not np.isclose(float(declared_spacing["dx"]), dx) or not np.isclose(
        float(declared_spacing["dy"]), dy
    ):
        raise ValueError(f"dataset={name} standard IVD spacing mismatch")
    standard_frames = np.asarray(
        np.load(standard_root / "frame_indices.npy", allow_pickle=False),
        dtype=np.int64,
    )
    standard_ivd_mm = np.load(
        standard_root / "standard_ivd.npy", mmap_mode="r", allow_pickle=False
    )
    standard_valid_mm = np.load(
        standard_root / "valid_fluid_mask.npy", mmap_mode="r", allow_pickle=False
    )
    standard_invalid_mm = np.load(
        standard_root / "invalid_mask.npy", mmap_mode="r", allow_pickle=False
    )
    standard_solid = np.asarray(
        np.load(standard_root / "solid_mask.npy", allow_pickle=False), dtype=bool
    )
    if (
        standard_ivd_mm.ndim != 3
        or tuple(standard_ivd_mm.shape[1:]) != shape[1:]
        or standard_valid_mm.shape != standard_ivd_mm.shape
        or standard_invalid_mm.shape != standard_ivd_mm.shape
        or standard_solid.shape != shape[1:]
    ):
        raise ValueError(f"dataset={name} standard IVD artifact shape mismatch")
    standard_row_by_frame = {
        int(frame): int(row) for row, frame in enumerate(standard_frames)
    }

    target_root = ticket03_root / "datasets" / name / "train_target"
    calibration_root = ticket03_root / "datasets" / name / "calibration_reference"
    target = _load_ticket03_target(
        target_root, manifest=manifest, expected_split="train"
    )
    calibration_meta = _read_json(calibration_root / "target_meta.json")
    w1_p._w1p_validate_ticket03_metadata(
        calibration_meta,
        root=calibration_root.resolve(),
        expected_split="calibration",
        protocol_manifest=manifest,
    )
    target_meta = target.metadata
    target_frames = tuple(int(frame) for frame in target.frame_indices)
    train_ranges = _ranges(record, "train")
    train_frames = tuple(
        frame for start, end in train_ranges for frame in range(start, end)
    )
    if target_frames != train_frames:
        raise ValueError(f"dataset={name} train target frame list does not match manifest")
    target_row_by_frame = {
        int(frame): int(row) for row, frame in enumerate(target.frame_indices)
    }
    starts = tuple(
        int(value)
        for start, end in train_ranges
        for value in dataset.window_starts(
            start,
            end,
            t_win=t_win,
            step=window_step,
            dataset_name=name,
            split_name="train",
            T=shape[0],
        )
    )
    starts = validate_train_window_starts(
        record, name, starts, t_win=t_win, window_step=window_step
    )
    declared_starts = tuple(
        int(value) for value in record["frame_partitions"]["train"]["window_starts"]
    )
    if starts != declared_starts:
        raise ValueError(f"dataset={name} train window manifest drifted")
    for frame in train_frames:
        if frame not in standard_row_by_frame:
            raise ValueError(
                f"dataset={name} train frame {frame} has no standard IVD row"
            )
    target_input = target_meta.get("input_identity", {})
    geometry_identity = target_input.get("geometry_mask_identity")
    if not isinstance(geometry_identity, str) or not geometry_identity:
        raise ValueError(f"dataset={name} target has no geometry mask identity")
    if standard_meta.get("geometry_mask_identity") != geometry_identity:
        raise ValueError(f"dataset={name} geometry identity drifted")
    standard_input = standard_meta.get("input_arrays", {}).get(
        "frame_input_identities", {}
    )
    target_input_frames = target_input.get("frame_input_identities", {})
    for frame in train_frames:
        key = str(frame)
        if standard_input.get(key) != target_input_frames.get(key):
            raise ValueError(
                f"dataset={name} train input identity mismatch at frame={frame}"
            )

    view = DatasetView(
        name=name,
        u=source["u"],
        v=source["v"],
        mask=np.asarray(source["mask"], dtype=bool),
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        record=record,
        standard_meta=standard_meta,
        standard_ivd=standard_ivd_mm,
        standard_valid=standard_valid_mm,
        standard_invalid=standard_invalid_mm,
        standard_solid=standard_solid,
        standard_row_by_frame=standard_row_by_frame,
        target=target,
        target_meta=target_meta,
        calibration_meta=calibration_meta,
        target_row_by_frame=target_row_by_frame,
        train_frames=train_frames,
        train_starts=starts,
        normalization={},
        patches=(),
        usable_patches=(),
        positive_pool_by_frame={},
        negative_pool_by_frame={},
        batch_frame_choices=(),
    )
    view.normalization = _compute_train_normalization(view)
    _build_pools(
        view,
        patch_size=patch_size,
        stride=stride,
        groups=groups,
        delta_frac=delta_frac,
    )
    target_counts = dict(target_meta.get("class_counts", {}))
    target_coverage = dict(target_meta.get("coverage", {}))
    view.audit = {
        "dataset_name": name,
        "raw_input": {
            "root": str(Path(source["root"]).resolve()),
            "u_path": str((Path(source["root"]) / "u.npy").resolve()),
            "v_path": str((Path(source["root"]) / "v.npy").resolve()),
            "geometry_mask_path": str((Path(source["root"]) / "mask.npy").resolve()),
            "shape": list(shape),
            "coordinate_lengths": {
                "tdim": len(tdim),
                "ydim": len(ydim),
                "xdim": len(xdim),
            },
            "grid_spacing": {"dx": dx, "dy": dy},
            "input_identity": copy.deepcopy(target_input),
            "geometry_mask_identity": geometry_identity,
            "legacy_ivd_read": False,
            "legacy_label_read": False,
        },
        "standard_ivd": {
            "source_identity": copy.deepcopy(standard_ivd.SOURCE_IDENTITY),
            "artifact_instance_identity": copy.deepcopy(
                standard_meta["artifact_instance_identity"]
            ),
            "artifact_sha256": standard_meta.get("artifact_sha256"),
            "implementation_entry_point": standard_meta.get("entry_point"),
            "formula": standard_ivd.FORMULA,
            "formula_version": standard_ivd.FORMULA_VERSION,
            "mean_scope": standard_meta["mean_scope"],
            "grid_spacing": {"dx": dx, "dy": dy},
            "test_values_read": False,
            "test_frame_indices_read": [],
        },
        "target": {
            "artifact_identity": copy.deepcopy(target_meta["artifact_identity"]),
            "artifact_instance_identity": copy.deepcopy(
                target_meta["artifact_instance_identity"]
            ),
            "metadata_hash": target_meta["metadata_hash"],
            "thresholds": copy.deepcopy(target_meta["thresholds"]),
            "threshold_provenance": copy.deepcopy(
                target_meta["threshold_provenance"]
            ),
            "class_counts": target_counts,
            "coverage": target_coverage,
            "split_name": "train",
            "test_source_read": False,
            "test_frame_indices_read": [],
        },
        "frame_counts": {
            "total": int(record["total_frames"]),
            "train": sum(end - start for start, end in _ranges(record, "train")),
            "calibration": sum(
                end - start for start, end in _ranges(record, "calibration")
            ),
            "development_audit": sum(
                end - start for start, end in _ranges(record, "development_audit")
            ),
            "test": sum(
                end - start for start, end in _ranges(record, "test")
            ),
        },
        "pathline_windows": {
            "train_count": len(starts),
            "train_window_starts": list(starts),
            "cross_split_window_count": 0,
            "window_step": int(window_step),
            "t_win": int(t_win),
            "audit_frame_in_train_window_count": 0,
            "audit_frame_in_test_window_count": 0,
        },
        "pools": {
            "patch_count": len(view.patches),
            "usable_patch_count": len(view.usable_patches),
            "positive_patch_count": sum(
                len(pool) for pool in view.positive_pool_by_frame.values()
            ),
            "negative_patch_count": sum(
                len(pool) for pool in view.negative_pool_by_frame.values()
            ),
            "train_windows_with_both_classes": len(view.batch_frame_choices),
        },
        "normalization": copy.deepcopy(view.normalization),
        "development_audit_frames": list(
            int(frame) for frame in record.get("development_audit_frames", ())
        ),
        "boussinesq_frame_1300_development_audit_only": bool(
            name == "boussinesq"
            and protocol.BOUSSINESQ_AUDIT_FRAME
            in set(record.get("development_audit_frames", ()))
        ),
        "protocol_version": manifest["protocol_version"],
        "protocol_manifest_hash": manifest["manifest_hash"],
    }
    return view


def _build_haller_frame(
    view: DatasetView,
    frame: int,
    output_root: Path,
    *,
    run_namespace: str,
    protocol_manifest_hash: str,
) -> dict[str, Any]:
    row = view.standard_row_by_frame[frame]
    valid = np.asarray(view.standard_valid[row], dtype=bool)
    invalid = np.asarray(view.standard_invalid[row], dtype=bool)
    result = haller_anchors.extract_haller_anchors(
        np.asarray(view.u[frame], dtype=np.float32),
        np.asarray(view.v[frame], dtype=np.float32),
        view.xdim,
        view.ydim,
        view.mask,
        source=haller_anchors.SOURCE_TRAIN,
        frame_index=frame,
        backend=TICKET04_HALLER_BACKEND,
        device="cpu",
        numbacs_nlevs=TICKET04_HALLER_NLEVS,
        numbacs_parameter_version=TICKET04_HALLER_PARAMETER_VERSION,
        valid_fluid_mask=valid,
        invalid_mask=invalid,
    )
    expected_ivd = np.asarray(view.standard_ivd[row], dtype=np.float32)
    actual_ivd = np.asarray(result["standard_ivd"], dtype=np.float32)
    if not np.array_equal(actual_ivd, expected_ivd, equal_nan=True):
        raise ValueError(
            f"dataset={view.name} Haller frame={frame} did not reproduce "
            "the Ticket 03 current standard IVD row"
        )
    if not np.array_equal(
        np.asarray(result["solid_mask"], dtype=bool), view.standard_solid
    ):
        raise ValueError(f"dataset={view.name} Haller solid mask drifted at frame={frame}")
    if not np.array_equal(
        np.asarray(result["valid_fluid_mask"], dtype=bool), valid
    ) or not np.array_equal(np.asarray(result["invalid_mask"], dtype=bool), invalid):
        raise ValueError(f"dataset={view.name} Haller support masks drifted at frame={frame}")
    metadata = copy.deepcopy(dict(result["metadata"]))
    metadata.update(
        {
            "ticket04_execution_namespace": run_namespace,
            "dataset_name": view.name,
            "split_name": "train",
            "frame_index": int(frame),
            "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
            "protocol_manifest_hash": protocol_manifest_hash,
            "standard_ivd_source_identity": copy.deepcopy(
                standard_ivd.SOURCE_IDENTITY
            ),
            "standard_ivd_artifact": copy.deepcopy(
                view.target_meta["standard_ivd_artifact"]
            ),
            "input_identity": view.target_meta["input_identity"][
                "frame_input_identities"
            ][str(frame)],
            "geometry_mask_identity": view.target_meta["input_identity"][
                "geometry_mask_identity"
            ],
            "grid_spacing": {
                "dx": float(np.diff(view.xdim)[0]),
                "dy": float(np.diff(view.ydim)[0]),
            },
            "haller_source_identity": copy.deepcopy(TICKET04_HALLER_SOURCE_IDENTITY),
            "test_source_read": False,
            "test_frame_indices_read": [],
            "final_test_relationships": "not_read",
            "legacy_ivd_read": False,
            "legacy_label_read": False,
            "legacy_artifacts_formal_source": False,
        }
    )
    result = dict(result)
    result["metadata"] = metadata
    frame_root = output_root / view.name / f"frame_{frame:06d}"
    files = haller_anchors.save_haller_artifact(
        result, frame_root, overwrite=False
    )
    mapped = haller_result_to_sparse_state(
        result["anchor_state"],
        solid_mask=result["solid_mask"],
        invalid_mask=result["invalid_mask"],
        failure_count=int(result["failure_count"]),
    )
    valid_count = int(
        np.count_nonzero(
            ~np.asarray(result["solid_mask"], dtype=bool)
            & ~np.asarray(result["invalid_mask"], dtype=bool)
        )
    )
    total = int(mapped.size)
    counts = {
        "positive_anchor": int(np.count_nonzero(mapped == sparse.STATE_POSITIVE_ANCHOR)),
        "uncovered": int(np.count_nonzero(mapped == sparse.STATE_UNCOVERED)),
        "solid": int(np.count_nonzero(mapped == sparse.STATE_SOLID)),
        "invalid": int(np.count_nonzero(mapped == sparse.STATE_INVALID)),
        "extractor_failure": int(
            np.count_nonzero(mapped == sparse.STATE_EXTRACTOR_FAILURE)
        ),
    }
    return {
        "frame_index": int(frame),
        "path": str(frame_root),
        "relative_path": str(frame_root.relative_to(output_root)),
        "metadata_hash": _json_hash(metadata),
        "anchor_state_hash": metadata["artifact_array_hashes"]["anchor_state"],
        "failure_count": int(result["failure_count"]),
        "positive_anchor_count": counts["positive_anchor"],
        "valid_fluid_count": valid_count,
        "cell_count": total,
        "mapped_counts": counts,
        "files": files,
    }


def build_current_haller_train(
    views: Sequence[DatasetView],
    output_root: str | Path,
    *,
    workers: int = DEFAULT_HALLER_WORKERS,
    run_namespace: str = "ticket04-real-w1p-w1hybrid-w2-v1",
    protocol_manifest_hash: str,
) -> tuple[dict[str, Any], str]:
    """Generate fresh train-only Haller artifacts and one global anchor identity."""
    workers = int(workers)
    if workers <= 0:
        raise ValueError("Haller workers must be positive")
    output_root = Path(output_root)
    per_dataset: dict[str, Any] = {}
    for view in views:
        print(
            f"[ticket04] Haller train start dataset={view.name} "
            f"frames={len(view.train_starts)} workers={workers}",
            flush=True,
        )
        dataset_root = output_root / view.name
        futures = {}
        frame_info: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for frame in view.train_starts:
                futures[
                    executor.submit(
                        _build_haller_frame,
                        view,
                        int(frame),
                        dataset_root,
                        run_namespace=run_namespace,
                        protocol_manifest_hash=protocol_manifest_hash,
                    )
                ] = int(frame)
            for future in as_completed(futures):
                frame_info.append(future.result())
        frame_info.sort(key=lambda item: item["frame_index"])
        if tuple(item["frame_index"] for item in frame_info) != view.train_starts:
            raise ValueError(f"dataset={view.name} Haller train frame coverage is incomplete")
        view.haller_dirs = {
            int(item["frame_index"]): Path(item["path"]) for item in frame_info
        }
        view.haller_info = {
            int(item["frame_index"]): item for item in frame_info
        }
        payload = {
            "artifact_type": "ticket04_current_haller_train_manifest",
            "schema_version": "criterion-ambiguity-ticket04-haller-train-v1",
            "ticket04_execution_namespace": run_namespace,
            "dataset_name": view.name,
            "split_name": "train",
            "source": haller_anchors.SOURCE_TRAIN,
            "haller_source_identity": copy.deepcopy(TICKET04_HALLER_SOURCE_IDENTITY),
            "haller_backend": TICKET04_HALLER_BACKEND,
            "haller_nlevs": TICKET04_HALLER_NLEVS,
            "haller_parameter_version": TICKET04_HALLER_PARAMETER_VERSION,
            "anchor_semantics": "sparse_positive_physics_evidence_only",
            "dense_negative_generated": False,
            "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
            "protocol_manifest_hash": protocol_manifest_hash,
            "standard_ivd_source_identity": copy.deepcopy(
                standard_ivd.SOURCE_IDENTITY
            ),
            "standard_ivd_artifact": copy.deepcopy(
                view.target_meta["standard_ivd_artifact"]
            ),
            "input_identity": copy.deepcopy(view.target_meta["input_identity"]),
            "frame_indices": list(view.train_starts),
            "frame_artifacts": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"files", "path"}
                }
                for item in frame_info
            ],
            "failure_frame_count": int(
                sum(item["failure_count"] > 0 for item in frame_info)
            ),
            "failure_count": int(sum(item["failure_count"] for item in frame_info)),
            "test_source_read": False,
            "test_frame_indices_read": [],
            "final_test_relationships": "not_read",
            "legacy_ivd_read": False,
            "legacy_label_read": False,
            "legacy_artifacts_formal_source": False,
        }
        payload["manifest_hash"] = _json_hash(payload)
        _write_json(dataset_root / "haller_train_manifest.json", payload)
        per_dataset[view.name] = {
            "manifest_path": str(dataset_root / "haller_train_manifest.json"),
            "manifest_hash": payload["manifest_hash"],
            "frame_count": len(frame_info),
            "failure_frame_count": payload["failure_frame_count"],
            "failure_count": payload["failure_count"],
            "frame_artifacts": payload["frame_artifacts"],
        }
        print(
            f"[ticket04] Haller train complete dataset={view.name} "
            f"frames={len(frame_info)} failures={payload['failure_count']}",
            flush=True,
        )

    global_payload = {
        "artifact_type": "ticket04_current_haller_train_global_manifest",
        "schema_version": "criterion-ambiguity-ticket04-haller-global-v1",
        "ticket04_execution_namespace": run_namespace,
        "source": haller_anchors.SOURCE_TRAIN,
        "haller_source_identity": copy.deepcopy(TICKET04_HALLER_SOURCE_IDENTITY),
        "haller_backend": TICKET04_HALLER_BACKEND,
        "haller_nlevs": TICKET04_HALLER_NLEVS,
        "haller_parameter_version": TICKET04_HALLER_PARAMETER_VERSION,
        "anchor_semantics": "sparse_positive_physics_evidence_only",
        "dense_negative_generated": False,
        "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
        "protocol_manifest_hash": protocol_manifest_hash,
        "standard_ivd_source_identity": copy.deepcopy(standard_ivd.SOURCE_IDENTITY),
        "datasets": per_dataset,
        "dataset_order": list(DATASET_ORDER),
        "train_only": True,
        "test_source_read": False,
        "test_frame_indices_read": [],
        "final_test_relationships": "not_read",
        "legacy_ivd_read": False,
        "legacy_label_read": False,
        "legacy_artifacts_formal_source": False,
    }
    anchor_hash = _json_hash(global_payload)
    global_payload["anchor_hash"] = anchor_hash
    global_payload["manifest_hash"] = _json_hash(global_payload)
    _write_json(output_root / "haller_train_global_manifest.json", global_payload)
    return global_payload, anchor_hash


def _sparse_selected_states(
    view: DatasetView,
    frame: int,
    seed_cells: np.ndarray,
) -> np.ndarray:
    haller_dir = view.haller_dirs[frame]
    meta = _read_json(haller_dir / "anchor_meta.json")
    raw_state = np.load(
        haller_dir / "anchor_state.npy", mmap_mode="r", allow_pickle=False
    )
    solid = np.load(
        haller_dir / "solid_mask.npy", mmap_mode="r", allow_pickle=False
    )
    invalid = np.load(
        haller_dir / "invalid_mask.npy", mmap_mode="r", allow_pickle=False
    )
    rows = seed_cells[..., 0]
    cols = seed_cells[..., 1]
    raw = np.asarray(raw_state[rows, cols], dtype=np.int8)
    solid_selected = np.asarray(solid[rows, cols], dtype=bool)
    invalid_selected = np.asarray(invalid[rows, cols], dtype=bool)
    selected = np.full(raw.shape, sparse.STATE_UNCOVERED, dtype="<U24")
    selected[solid_selected] = sparse.STATE_SOLID
    selected[invalid_selected] = sparse.STATE_INVALID
    selected[(raw == 1) & ~solid_selected & ~invalid_selected] = (
        sparse.STATE_POSITIVE_ANCHOR
    )
    if int(meta["failure_count"]):
        selected[(raw == -1) & ~solid_selected & ~invalid_selected] = (
            sparse.STATE_EXTRACTOR_FAILURE
        )
    if np.any(raw == 0):
        raise ValueError("fresh Haller artifact emitted a dense negative state")
    return selected


def _align_percentile_state_with_sparse_exclusions(
    percentile_state: Any,
    sparse_state: Any,
) -> np.ndarray:
    """Mask Haller-excluded cells out of the Hybrid percentile source.

    Ticket 03 keeps the P95/P80 state independent from Haller.  A train-only
    Haller frame can nevertheless mark a cell as solid, invalid, or an
    extractor failure.  Those sparse states are explicit ignored states in
    W1-Hybrid/W2 and therefore cannot carry an ordinary percentile positive
    or negative target.  W1-P never calls this seam and continues to consume
    the unmodified Ticket 03 target.
    """
    percentile = np.asarray(percentile_state, dtype=np.int8)
    sparse_values = np.asarray(sparse_state)
    if percentile.shape != sparse_values.shape:
        raise ValueError(
            "percentile_state 与 sparse_state shape 不一致，无法对齐 excluded cells"
        )
    allowed = np.isin(
        percentile,
        (closure.PERCENTILE_POSITIVE, closure.PERCENTILE_NEGATIVE,
         closure.PERCENTILE_UNKNOWN),
    )
    if not np.all(allowed):
        raise ValueError("percentile_state 含未知状态")
    excluded = np.isin(
        sparse_values,
        (
            sparse.STATE_SOLID,
            sparse.STATE_INVALID,
            sparse.STATE_EXTRACTOR_FAILURE,
        ),
    )
    aligned = percentile.copy()
    aligned[excluded] = closure.PERCENTILE_UNKNOWN
    return aligned


def _prepare_normalized_window(view: DatasetView, frame: int, *, t_win: int):
    """Prepare one read-only stored window for same-window patch extraction.

    The advanced IVD frame selection creates a full-window copy. A mixed batch
    can share that copy while each patch keeps its original extraction RNG.
    """
    frame_rows = [view.standard_row_by_frame[start] for start in range(frame, frame + t_win)]
    raw_u = np.asarray(view.u[frame : frame + t_win], dtype=np.float32)
    raw_v = np.asarray(view.v[frame : frame + t_win], dtype=np.float32)
    ivd = np.asarray(view.standard_ivd[frame_rows], dtype=np.float32)
    tdim_window = view.tdim[frame : frame + t_win]
    if len(tdim_window) != t_win:
        raise ValueError(f"dataset={view.name} sample window is incomplete")
    t_span = float(tdim_window[-1] - tdim_window[0])
    return raw_u, raw_v, ivd, tdim_window, t_span


def _extract_one_normalized(
    view: DatasetView,
    item: tuple[int, int, int],
    *,
    patch_size: tuple[int, int],
    t_win: int,
    groups: tuple[int, int],
    delta_frac: float,
    L: int,
    n_substeps: int,
    t_scale: float,
    seed: int,
    prepared_window=None,
) -> tuple[np.ndarray, np.ndarray]:
    py, px, frame = (int(value) for value in item)
    if frame not in view.target_row_by_frame:
        raise ValueError(f"dataset={view.name} sample frame={frame} is not train")
    raw_u, raw_v, ivd, tdim_window, t_span = (
        _prepare_normalized_window(view, frame, t_win=t_win)
        if prepared_window is None else prepared_window)
    base = dataset._comb_rng_base(seed, py, px, frame, ds_id=view.ds_id)
    raw, seeds = extractor.extract_pathlines_batched(
        raw_u,
        raw_v,
        view.mask,
        ivd,
        view.xdim,
        view.ydim,
        tdim_window,
        patch_yx=(py, px),
        patch_size=patch_size,
        t0=float(view.tdim[frame]),
        L=L,
        groups=groups,
        delta_frac=delta_frac,
        t_win_frames=t_win,
        n_substeps=n_substeps,
        dt_out=t_span / (L - 1),
        rng=base,
        return_seeds=True,
    )
    return _normalize_extracted(view, (py, px, frame), raw, seeds,
                                patch_size=patch_size, t_span=t_span, t_scale=t_scale)


def _normalize_extracted(view, item, raw, seeds, *, patch_size, t_span, t_scale):
    py, px, frame = (int(value) for value in item)
    pathlines = dataset.normalize_pathlines(
        raw,
        seeds,
        extractor.patch_geometry((py, px), patch_size, view.xdim, view.ydim),
        float(view.tdim[frame]),
        t_span,
        t_scale,
        float(view.normalization["ivd_mu"]),
        float(view.normalization["ivd_sigma"]),
        float(view.normalization["speed_max"]),
    )
    if pathlines.shape[-1] != contract.FEATURE_SCHEMA_7.channel_count:
        raise ValueError("normalized pathline feature schema is not 7-channel")
    if not np.isfinite(pathlines).all():
        raise ValueError(
            f"dataset={view.name} sample frame={frame} produced non-finite pathlines"
        )
    seed_y, seed_x = extractor.nearest_cell(
        seeds[:, 0], seeds[:, 1], view.xdim, view.ydim
    )
    seed_cells = np.stack(
        [
            np.clip(seed_y, 0, view.height - 1),
            np.clip(seed_x, 0, view.width - 1),
        ],
        axis=1,
    ).astype(np.int64, copy=False)
    return np.asarray(pathlines, dtype=np.float32), seed_cells


def _extract_normalized_batch(view, items, *, patch_size, t_win, groups,
                              delta_frac, L, n_substeps, t_scale, seed):
    """Share each stored window and first RK4 pass; retain original item order."""
    grouped = {}
    for index, item in enumerate(items):
        grouped.setdefault(int(item[2]), []).append(index)
    results = [None] * len(items)
    for frame, indices in grouped.items():
        if frame not in view.target_row_by_frame:
            raise ValueError(f"dataset={view.name} sample frame={frame} is not train")
        raw_u, raw_v, ivd, tdim_window, t_span = _prepare_normalized_window(view, frame, t_win=t_win)
        patches = list(dict.fromkeys((int(items[index][0]), int(items[index][1]))
                                      for index in indices))
        bases = [dataset._comb_rng_base(seed, py, px, frame, ds_id=view.ds_id) for py, px in patches]
        rows = extractor.extract_pathlines_batched_patches(raw_u, raw_v, view.mask, ivd,
            view.xdim, view.ydim, tdim_window, patches, bases, patch_size=patch_size,
            t0=float(view.tdim[frame]), L=L, groups=groups, delta_frac=delta_frac,
            t_win_frames=t_win, n_substeps=n_substeps, dt_out=t_span / (L - 1))
        normalized = {patch: _normalize_extracted(view, (*patch, frame), raw, seeds,
                patch_size=patch_size, t_span=t_span, t_scale=t_scale)
            for patch, (raw, seeds) in zip(patches, rows)}
        for index in indices:
            results[index] = normalized[(int(items[index][0]), int(items[index][1]))]
    return results


class _CurrentSourceBatchDataset(Dataset):
    """Map-style dataset whose item is one same-window real training batch."""

    def __init__(
        self,
        views: Sequence[DatasetView],
        specs: Sequence[Mapping[str, Any]],
        *,
        patch_size: tuple[int, int],
        t_win: int,
        groups: tuple[int, int],
        delta_frac: float,
        L: int,
        n_substeps: int,
        t_scale: float,
        seed: int,
    ) -> None:
        self.views = tuple(views)
        self.specs = tuple(specs)
        self.patch_size = patch_size
        self.t_win = int(t_win)
        self.groups = groups
        self.delta_frac = float(delta_frac)
        self.L = int(L)
        self.n_substeps = int(n_substeps)
        self.t_scale = float(t_scale)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        spec = self.specs[int(index)]
        view = self.views[int(spec["dataset_index"])]
        items = tuple(tuple(int(value) for value in item) for item in spec["items"])
        pathline_rows = []
        seed_rows = []
        for item in items:
            pathlines, seed_cells = _extract_one_normalized(
                view,
                item,
                patch_size=self.patch_size,
                t_win=self.t_win,
                groups=self.groups,
                delta_frac=self.delta_frac,
                L=self.L,
                n_substeps=self.n_substeps,
                t_scale=self.t_scale,
                seed=self.seed,
            )
            pathline_rows.append(pathlines)
            seed_rows.append(seed_cells)
        return {
            "dataset_index": int(spec["dataset_index"]),
            "frame_start": int(spec["frame_start"]),
            "pathlines": np.stack(pathline_rows, axis=0),
            "frame_indices": np.full(
                len(items), int(spec["frame_start"]), dtype=np.int64
            ),
            "seed_cells": np.stack(seed_rows, axis=0),
        }


def _identity_collate(value: Any) -> Any:
    # DataLoader may wrap a map-style item in a one-element list when
    # batch_size=None is used with a custom collate function.  Keep the
    # source batch object intact in either representation.
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _build_epoch_specs(
    views: Sequence[DatasetView],
    *,
    samples_per_epoch: int,
    batch_size: int,
    positive_fraction: float,
    seed: int,
    epoch: int,
) -> list[dict[str, Any]]:
    samples_per_epoch = int(samples_per_epoch)
    batch_size = int(batch_size)
    if samples_per_epoch <= 0 or batch_size <= 0:
        raise ValueError("samples_per_epoch and batch_size must be positive")
    if not 0.0 < positive_fraction < 1.0:
        raise ValueError("positive_fraction must be in (0,1)")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    specs = []
    remaining = samples_per_epoch
    batch_index = 0
    while remaining:
        current_size = min(batch_size, remaining)
        dataset_index = batch_index % len(views)
        view = views[dataset_index]
        frame = int(
            view.batch_frame_choices[
                int(rng.integers(0, len(view.batch_frame_choices)))
            ]
        )
        positive_count = int(round(current_size * positive_fraction))
        positive_count = max(1, min(current_size - 1, positive_count))
        negative_count = current_size - positive_count
        pos_pool = view.positive_pool_by_frame[frame]
        neg_pool = view.negative_pool_by_frame[frame]
        pos_indices = rng.integers(0, len(pos_pool), size=positive_count)
        neg_indices = rng.integers(0, len(neg_pool), size=negative_count)
        items = [pos_pool[int(index)] for index in pos_indices]
        items.extend(neg_pool[int(index)] for index in neg_indices)
        rng.shuffle(items)
        specs.append(
            {
                "dataset_index": dataset_index,
                "frame_start": frame,
                "items": tuple(items),
            }
        )
        remaining -= current_size
        batch_index += 1
    return specs


def _window_provenance(
    view: DatasetView, frame: int, *, t_win: int, step: int
) -> dict[str, Any]:
    record = view.record
    train_ranges = _ranges(record, "train")
    containing = next(
        (
            pair
            for pair in train_ranges
            if pair[0] <= frame and frame + t_win <= pair[1]
        ),
        None,
    )
    if containing is None:
        raise ValueError(f"dataset={view.name} window is not train-contained")
    return {
        "dataset_name": view.name,
        "split_name": "train",
        "frame_start": int(frame),
        "frame_end": int(frame + t_win),
        "split_start": int(containing[0]),
        "split_end": int(containing[1]),
        "t_win": int(t_win),
        "window_step": int(step),
        "cross_split": False,
        "development_audit_frames": list(
            int(value) for value in record.get("development_audit_frames", ())
        ),
    }


def _batch_provenance(
    view: DatasetView,
    frame: int,
    *,
    t_win: int,
    window_step: int,
    mode: str,
    anchor_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    provenance = view.target.as_provenance()
    provenance["window"] = _window_provenance(
        view, frame, t_win=t_win, step=window_step
    )
    provenance["mode"] = contract.canonical_mode(mode)
    provenance["standard_ivd_source_identity"] = copy.deepcopy(
        standard_ivd.SOURCE_IDENTITY
    )
    provenance["standard_ivd_artifact"] = copy.deepcopy(
        view.target_meta["standard_ivd_artifact"]
    )
    provenance["input_identity"] = copy.deepcopy(view.target_meta["input_identity"])
    provenance["geometry_mask_identity"] = view.target_meta["input_identity"][
        "geometry_mask_identity"
    ]
    provenance["sampling"] = {
        "source": contract.LABEL_SOURCE_P95P80,
        "target_artifact_identity": copy.deepcopy(
            view.target_meta["artifact_identity"]
        ),
        "statistics_split": "train",
    }
    provenance["test_source_read"] = False
    provenance["test_frame_indices_read"] = []
    provenance["final_test_relationships"] = "not_read"
    provenance["legacy_ivd_read"] = False
    provenance["legacy_label_read"] = False
    provenance["legacy_artifacts_formal_source"] = False
    if anchor_metadata is not None:
        provenance["anchor"] = {
            "source": haller_anchors.SOURCE_TRAIN,
            "anchor_hash": anchor_metadata["anchor_hash"],
            "anchor_metadata_hash": anchor_metadata["manifest_hash"],
            "standard_ivd_source_identity": copy.deepcopy(
                standard_ivd.SOURCE_IDENTITY
            ),
            "anchor_semantics": "sparse_positive_physics_evidence_only",
            "dense_negative_generated": False,
            "uncovered_is_not_negative": True,
        }
    return provenance


def _make_contract_batch(
    payload: Mapping[str, Any],
    view: DatasetView,
    *,
    mode: str,
    t_win: int,
    window_step: int,
    anchor_metadata: Mapping[str, Any] | None,
    anchor_hash: str | None,
) -> Any:
    if int(payload["dataset_index"]) != view.ds_id:
        raise ValueError("batch payload dataset/view mismatch")
    frame = int(payload["frame_start"])
    frames = np.asarray(payload["frame_indices"], dtype=np.int64)
    if frames.size == 0 or not np.all(frames == frame):
        raise ValueError("one training batch must have one exact frame window")
    pathline_window = _window_provenance(
        view, frame, t_win=t_win, step=window_step
    )
    provenance = _batch_provenance(
        view,
        frame,
        t_win=t_win,
        window_step=window_step,
        mode=mode,
        anchor_metadata=anchor_metadata,
    )
    pathlines = np.asarray(payload["pathlines"], dtype=np.float32)
    cells = np.asarray(payload["seed_cells"], dtype=np.int64)
    dummy = np.zeros((pathlines.shape[0], 1, 1, 1), dtype=np.float32)
    canonical = contract.canonical_mode(mode)
    if canonical == contract.MODE_W1_P:
        return w1_p.build_w1_p_batch_from_artifact(
            pathlines,
            view.target,
            frame_indices=frames,
            seed_cells=cells,
            sampling_source=contract.LABEL_SOURCE_P95P80,
            split_name="train",
            pathline_window=pathline_window,
            provenance=provenance,
            dummy_field=dummy,
        )
    if anchor_metadata is None or not anchor_hash:
        raise ValueError("Hybrid batch construction requires the global Haller anchor")
    target_rows = np.asarray(
        [view.target_row_by_frame[int(frame)] for frame in frames], dtype=np.int64
    )
    percentile_state = np.asarray(
        view.target.state[
            target_rows[:, None], cells[:, :, 0], cells[:, :, 1]
        ],
        dtype=np.int8,
    )
    percentile_state = np.where(
        percentile_state == ticket03.POSITIVE,
        closure.PERCENTILE_POSITIVE,
        np.where(
            percentile_state == ticket03.NEGATIVE,
            closure.PERCENTILE_NEGATIVE,
            closure.PERCENTILE_UNKNOWN,
        ),
    ).astype(np.int8)
    # Every sample in this batch shares the validated frame above. Read its
    # Haller metadata and arrays once, then gather all samples' seed states.
    sparse_state_rows = _sparse_selected_states(view, frame, cells)
    percentile_state = _align_percentile_state_with_sparse_exclusions(
        percentile_state,
        sparse_state_rows,
    )
    return closure.build_w1_hybrid_batch(
        pathlines,
        percentile_state,
        sparse_state_rows,
        anchor_hash=anchor_hash,
        sampling_source=contract.LABEL_SOURCE_P95P80,
        split_name="train",
        provenance=provenance,
        dummy_field=dummy,
        anchor_metadata=anchor_metadata,
        mode=canonical,
    )


def _dataset_config(views: Sequence[DatasetView]) -> dict[str, Any]:
    return {
        "dataset_order": list(DATASET_ORDER),
        "datasets": [
            {
                "dataset_name": view.name,
                "raw_shape": list(view.u.shape),
                "raw_input_identity": copy.deepcopy(
                    view.target_meta["input_identity"]
                ),
                "geometry_mask_identity": view.target_meta["input_identity"][
                    "geometry_mask_identity"
                ],
                "standard_ivd_source_identity": copy.deepcopy(
                    standard_ivd.SOURCE_IDENTITY
                ),
                "standard_ivd_artifact": copy.deepcopy(
                    view.target_meta["standard_ivd_artifact"]
                ),
                "train_target_artifact_identity": copy.deepcopy(
                    view.target_meta["artifact_identity"]
                ),
                "train_target_artifact_instance_identity": copy.deepcopy(
                    view.target_meta["artifact_instance_identity"]
                ),
                "train_target_metadata_hash": view.target_meta["metadata_hash"],
                "calibration_reference_artifact_identity": copy.deepcopy(
                    view.calibration_meta["artifact_identity"]
                ),
                "calibration_reference_artifact_instance_identity": copy.deepcopy(
                    view.calibration_meta["artifact_instance_identity"]
                ),
                "calibration_reference_metadata_hash": view.calibration_meta[
                    "metadata_hash"
                ],
                "thresholds": copy.deepcopy(view.target_meta["thresholds"]),
            }
            for view in views
        ],
        "legacy_ivd_read": False,
        "legacy_label_read": False,
        "test_source_read": False,
        "test_frame_indices_read": [],
    }


def _split_config(
    manifest: Mapping[str, Any], views: Sequence[DatasetView]
) -> dict[str, Any]:
    return {
        "split_name": "train",
        "protocol_version": manifest["protocol_version"],
        "protocol_manifest_hash": manifest["manifest_hash"],
        "dataset_order": list(DATASET_ORDER),
        "dataset_train_ranges": {
            view.name: [list(pair) for pair in _ranges(view.record, "train")]
            for view in views
        },
        "pathline_window_policy": {
            "t_win": 24,
            "window_step": 4,
            "complete_only": True,
            "cross_split": False,
        },
        "development_audit_frames": {
            view.name: list(view.audit["development_audit_frames"]) for view in views
        },
        "boussinesq_frame_1300_in_test_denominator": False,
        "test_source_read": False,
        "test_frame_indices_read": [],
    }


def _sampling_config(
    *,
    samples_per_epoch: int,
    batch_size: int,
    positive_fraction: float,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
    t_win: int,
    window_step: int,
    groups: tuple[int, int],
    delta_frac: float,
    L: int,
    n_substeps: int,
    t_scale: float,
) -> dict[str, Any]:
    return {
        "source": contract.LABEL_SOURCE_P95P80,
        "statistics_split": "train",
        "samples_per_epoch": int(samples_per_epoch),
        "batch_size": int(batch_size),
        "positive_fraction": float(positive_fraction),
        "patch_size": list(patch_size),
        "stride": list(stride),
        "t_win": int(t_win),
        "window_step": int(window_step),
        "groups": list(groups),
        "delta_frac": float(delta_frac),
        "L": int(L),
        "n_substeps": int(n_substeps),
        "t_scale": float(t_scale),
        "pool_definition": "current Ticket 03 train positive_mask patch membership",
        "same_frame_batch_window": True,
        "connected_component_filter_is_outside_ivd": True,
    }


def _calibration_policy(
    view: DatasetView,
    manifest: Mapping[str, Any],
    *,
    variance_gate: float | None = None,
) -> dict[str, Any]:
    policy = {
        "source_id": contract.LABEL_SOURCE_P95P80,
        "source": contract.LABEL_SOURCE_P95P80,
        "protocol_version": manifest["protocol_version"],
        "protocol_manifest_hash": manifest["manifest_hash"],
        "statistics_split": "train",
        "threshold_provenance": copy.deepcopy(
            view.target_meta["threshold_provenance"]
        ),
        "calibration_reference_artifact": copy.deepcopy(
            view.calibration_meta["artifact_instance_identity"]
        ),
        "threshold_values": copy.deepcopy(view.target_meta["thresholds"]),
        "test_source_read": False,
        "test_frame_indices_read": [],
        "final_test_relationships": "not_read",
        "standard_ivd_source_identity": copy.deepcopy(
            standard_ivd.SOURCE_IDENTITY
        ),
    }
    if variance_gate is not None:
        policy["variance_gate"] = float(variance_gate)
        policy["variance_gate_status"] = "development_explicit_not_final_calibration"
    return policy


def _make_trainer(
    cfg: Mapping[str, Any],
    *,
    mode: str,
    views: Sequence[DatasetView],
    manifest: Mapping[str, Any],
    anchor_metadata: Mapping[str, Any],
    anchor_hash: str,
    seed: int,
    variance_gate: float,
    device: str | torch.device | None = None,
) -> tuple[Any, Any]:
    device = torch.device(device if device is not None else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    # The runner owns the method seed so model initialization, EMA creation,
    # and any torch-side sampling are reproducible for each method.
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = train_kaggle.build_model_from_config(cfg, mode=mode)
    train_cfg = dict(cfg.get("train", {}))
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = train_kaggle.TwoStepLR(
        optimizer,
        lr,
        float(train_cfg.get("second_lr", 5e-6)),
        int(train_cfg.get("warmup_epochs", 60)),
    )
    if mode == contract.MODE_W1_P:
        trainer = w1_p.W1PTrainer(
            model,
            optimizer,
            sampling_source=contract.LABEL_SOURCE_P95P80,
            config=w1_p.W1PConfig(ramp_up_epochs=12),
            scheduler=scheduler,
            seed=seed,
            # Each W1-P batch binds its exact dataset-specific Ticket 03
            # target provenance.  A copied single-dataset metadata object
            # cannot truthfully represent six dataset roots and would be
            # rejected by the Ticket 03 validator.  The joint lineage is
            # carried by dataset_config/calibration_policy and the batch
            # provenance below.
            target_metadata=None,
            calibration_metadata=None,
            grad_clip_norm=float(train_cfg.get("grad_clip", 1.0)),
            teacher_device=device,
        )
    elif mode == contract.MODE_W1_HYBRID:
        trainer = closure.W1HybridTrainer(
            model,
            optimizer,
            sampling_source=contract.LABEL_SOURCE_P95P80,
            anchor_hash=anchor_hash,
            anchor_metadata=anchor_metadata,
            scheduler=scheduler,
            config=w1_p.W1PConfig(ramp_up_epochs=12),
            seed=seed,
            grad_clip_norm=float(train_cfg.get("grad_clip", 1.0)),
        )
    elif mode == contract.MODE_W2_ON_HYBRID:
        trainer = closure.W2OnHybridTrainer(
            model,
            optimizer,
            sampling_source=contract.LABEL_SOURCE_P95P80,
            anchor_hash=anchor_hash,
            anchor_metadata=anchor_metadata,
            scheduler=scheduler,
            config=w2.W2Config(ramp_up_epochs=12, variance_gate=variance_gate),
            seed=seed,
            grad_clip_norm=float(train_cfg.get("grad_clip", 1.0)),
        )
    else:
        raise ValueError(f"unsupported Ticket 04 mode: {mode}")
    return trainer, model


def _aggregate_policy(
    views: Sequence[DatasetView],
    manifest: Mapping[str, Any],
    variance_gate: float | None,
) -> dict[str, Any]:
    result = {
        "source_id": contract.LABEL_SOURCE_P95P80,
        "source": contract.LABEL_SOURCE_P95P80,
        "protocol_version": manifest["protocol_version"],
        "protocol_manifest_hash": manifest["manifest_hash"],
        "statistics_split": "train",
        "datasets": [
            _calibration_policy(view, manifest, variance_gate=variance_gate)
            for view in views
        ],
        "threshold_provenance": [
            copy.deepcopy(view.target_meta["threshold_provenance"])
            for view in views
        ],
        "test_source_read": False,
        "test_frame_indices_read": [],
        "final_test_relationships": "not_read",
    }
    return result


def _save_checkpoint(
    trainer: Any,
    *,
    mode: str,
    path: Path,
    completed_epoch: int,
    dataset_config: Mapping[str, Any],
    split_config: Mapping[str, Any],
    sampling_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    calibration_policy: Mapping[str, Any],
    anchor_metadata: Mapping[str, Any],
    execution_namespace: str,
) -> None:
    kwargs = dict(
        path=path,
        epoch=int(completed_epoch) - 1,
        dataset_config=dataset_config,
        split_config=split_config,
        sampling_config=sampling_config,
        metrics=dict(metrics),
        calibration_policy=dict(calibration_policy),
        extra_metadata={
            "ticket": execution_namespace,
            "execution_ticket": execution_namespace,
            "execution_namespace": execution_namespace,
            "completed_epoch": int(completed_epoch),
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
            "protocol_manifest_hash": split_config["protocol_manifest_hash"],
            "standard_ivd_source_identity": copy.deepcopy(
                standard_ivd.SOURCE_IDENTITY
            ),
            "haller_anchor_identity": (
                copy.deepcopy(anchor_metadata)
                if mode != contract.MODE_W1_P
                else None
            ),
            "test_source_read": False,
            "test_frame_indices_read": [],
            "final_test_relationships": "not_read",
            "legacy_ivd_read": False,
            "legacy_label_read": False,
        },
    )
    trainer.save_checkpoint(**kwargs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reuse_completed_w1p(
    previous_root: str | Path,
    output_root: Path,
    *,
    manifest: Mapping[str, Any],
    epochs: int,
    samples_per_epoch: int,
    batch_size: int,
    num_workers: int,
    haller_workers: int,
    seed: int,
    execution_namespace: str,
) -> dict[str, Any]:
    """Copy only a fully completed W1-P stage into a new run namespace.

    A failed downstream stage must not force a second multi-hour W1-P run.
    This continuation seam is deliberately narrow: it validates the prior
    Ticket 03/protocol and training contract, copies the immutable W1-P
    checkpoint artifacts into a fresh output namespace, and leaves
    W1-Hybrid/W2 to train under the corrected consumer code.
    """
    source_root = Path(previous_root).resolve()
    output_root = Path(output_root).resolve()
    if source_root == output_root:
        raise ValueError("reuse_completed_run 必须指向不同的历史输出 namespace")
    source_manifest = _read_json(source_root / "run_manifest.json")
    if source_manifest.get("protocol_manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("复用的 W1-P run protocol manifest identity 不一致")
    if source_manifest.get("dataset_order") != list(DATASET_ORDER):
        raise ValueError("复用的 W1-P run dataset order 不一致")
    if source_manifest.get("haller_source_identity") != TICKET04_HALLER_SOURCE_IDENTITY:
        raise ValueError("复用的 W1-P run Haller source identity 不一致")
    if source_manifest.get("standard_ivd_provenance") != standard_ivd.SOURCE_IDENTITY:
        raise ValueError("复用的 W1-P run standard IVD source identity 不一致")
    training = source_manifest.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("复用的 W1-P run 缺少 training contract")
    expected_training = {
        "epochs": int(epochs),
        "samples_per_epoch": int(samples_per_epoch),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "haller_workers": int(haller_workers),
        "seed": int(seed),
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(
                f"复用的 W1-P run training.{key}={training.get(key)!r} "
                f"与当前要求 {expected!r} 不一致"
            )
    source_method = source_root / "methods" / "w1_p"
    source_report_path = source_method / "training_report.json"
    source_report = _read_json(source_report_path)
    if source_report.get("method_name") != "w1_p" or source_report.get(
        "mode"
    ) != contract.MODE_W1_P:
        raise ValueError("复用的 W1-P report method/mode 不一致")
    if source_report.get("epochs_completed") != int(epochs):
        raise ValueError("复用的 W1-P run 尚未完成请求的全部 epoch")
    expected_checkpoints = checkpoint_epochs(epochs)
    if tuple(source_report.get("checkpoint_epochs", ())) != expected_checkpoints:
        raise ValueError("复用的 W1-P checkpoint epoch 列表不完整")
    checkpoint_source_paths = {
        epoch: source_method / "checkpoints" / f"epoch_{epoch:04d}.pt"
        for epoch in expected_checkpoints
    }
    missing = [str(path) for path in checkpoint_source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "复用的 W1-P checkpoint 缺失：" + ", ".join(missing)
        )
    if source_report.get("test_source_read") is not False:
        raise ValueError("复用的 W1-P report 没有明确声明 test source 未读取")
    destination_method = output_root / "methods" / "w1_p"
    destination_method.parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source_method, destination_method)
    reused_report = _read_json(destination_method / "training_report.json")
    reused_report.update(
        {
            "output_directory": str(destination_method),
            "checkpoint_directory": str(destination_method / "checkpoints"),
            "reused_completed_stage": True,
            "reused_from_output_directory": str(source_root),
            "reused_from_execution_namespace": source_report.get(
                "execution_namespace"
            ),
            "reused_into_execution_namespace": execution_namespace,
            "test_source_read": False,
            "test_frame_indices_read": [],
            "final_test_relationships": "not_read",
        }
    )
    _write_json(destination_method / "training_report.json", reused_report)
    return {
        "method_name": "w1_p",
        "source_output_directory": str(source_root),
        "source_execution_namespace": source_report.get("execution_namespace"),
        "destination_output_directory": str(destination_method),
        "checkpoint_epochs": list(expected_checkpoints),
        "checkpoint_sha256": {
            str(epoch): _sha256_file(path)
            for epoch, path in checkpoint_source_paths.items()
        },
        "report": reused_report,
    }


def _reuse_haller_train(
    previous_root: str | Path,
    output_root: Path,
    *,
    views: Sequence[DatasetView],
    manifest: Mapping[str, Any],
    execution_namespace: str,
) -> dict[str, Any]:
    """Reuse a complete, identity-validated train-only Haller artifact.

    Haller extraction is independent of model training.  A new Ticket 04
    continuation may therefore copy a prior complete artifact, but only
    after checking the current protocol, standard-IVD source, six-dataset
    order, raw-input identities, frame coverage, and every frame metadata
    boundary.  The source global manifest and anchor hash remain unchanged;
    the new run records the reuse in its own run manifest.
    """
    source_root = Path(previous_root).resolve()
    output_root = Path(output_root).resolve()
    source_haller = source_root / "haller_anchor_train"
    destination_haller = output_root / "haller_anchor_train"
    if source_root == output_root:
        raise ValueError("reuse_haller_run 必须指向不同的历史输出 namespace")
    source_preflight = _read_json(source_root / "preflight.json")
    source_run_manifest = _read_json(source_root / "run_manifest.json")
    current_protocol_hash = str(manifest["manifest_hash"])
    if source_preflight.get("protocol_manifest_hash") != current_protocol_hash:
        raise ValueError("复用的 Haller source preflight protocol identity 不一致")
    if source_run_manifest.get("protocol_manifest_hash") != current_protocol_hash:
        raise ValueError("复用的 Haller source run protocol identity 不一致")
    if source_run_manifest.get("standard_ivd_provenance") != standard_ivd.SOURCE_IDENTITY:
        raise ValueError("复用的 Haller source standard IVD identity 不一致")
    if source_run_manifest.get("test_source_read") is not False:
        raise ValueError("复用的 Haller source 读取了 test source")
    if source_run_manifest.get("test_frame_indices_read") != []:
        raise ValueError("复用的 Haller source 包含 test frame")
    global_path = source_haller / "haller_train_global_manifest.json"
    global_manifest = _read_json(global_path)
    historical_frame_manifest_hashes = {
        str(view.record["manifest_hash"]) for view in views
    }
    required_global = {
        "artifact_type": "ticket04_current_haller_train_global_manifest",
        "source": haller_anchors.SOURCE_TRAIN,
        "haller_source_identity": TICKET04_HALLER_SOURCE_IDENTITY,
        "haller_backend": TICKET04_HALLER_BACKEND,
        "haller_nlevs": TICKET04_HALLER_NLEVS,
        "haller_parameter_version": TICKET04_HALLER_PARAMETER_VERSION,
        "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
        "standard_ivd_source_identity": standard_ivd.SOURCE_IDENTITY,
        "dataset_order": list(DATASET_ORDER),
        "train_only": True,
        "test_source_read": False,
        "test_frame_indices_read": [],
        "legacy_ivd_read": False,
        "legacy_label_read": False,
        "legacy_artifacts_formal_source": False,
    }
    for key, expected in required_global.items():
        if global_manifest.get(key) != expected:
            raise ValueError(
                f"复用的 Haller global manifest {key} identity 不一致"
            )
    source_global_protocol_hash = global_manifest.get("protocol_manifest_hash")
    if source_global_protocol_hash not in {
        current_protocol_hash,
        *historical_frame_manifest_hashes,
    }:
        raise ValueError("复用的 Haller global manifest protocol identity 不一致")
    anchor_hash = global_manifest.get("anchor_hash")
    manifest_hash = global_manifest.get("manifest_hash")
    if not isinstance(anchor_hash, str) or not anchor_hash:
        raise ValueError("复用的 Haller global manifest 缺少 anchor_hash")
    if not isinstance(manifest_hash, str) or not manifest_hash:
        raise ValueError("复用的 Haller global manifest 缺少 manifest_hash")
    source_datasets = global_manifest.get("datasets")
    if not isinstance(source_datasets, Mapping):
        raise ValueError("复用的 Haller global manifest 缺少 datasets")

    per_dataset_info: dict[str, dict[str, Any]] = {}
    for view in views:
        dataset_root = source_haller / view.name
        dataset_manifest_path = dataset_root / "haller_train_manifest.json"
        dataset_manifest = _read_json(dataset_manifest_path)
        required_dataset = {
            "artifact_type": "ticket04_current_haller_train_manifest",
            "dataset_name": view.name,
            "split_name": "train",
            "source": haller_anchors.SOURCE_TRAIN,
            "haller_source_identity": TICKET04_HALLER_SOURCE_IDENTITY,
            "haller_backend": TICKET04_HALLER_BACKEND,
            "haller_nlevs": TICKET04_HALLER_NLEVS,
            "haller_parameter_version": TICKET04_HALLER_PARAMETER_VERSION,
            "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
            "standard_ivd_source_identity": standard_ivd.SOURCE_IDENTITY,
            "standard_ivd_artifact": view.target_meta["standard_ivd_artifact"],
            "input_identity": view.target_meta["input_identity"],
            "dense_negative_generated": False,
            "test_source_read": False,
            "test_frame_indices_read": [],
            "legacy_ivd_read": False,
            "legacy_label_read": False,
            "legacy_artifacts_formal_source": False,
        }
        for key, expected in required_dataset.items():
            if dataset_manifest.get(key) != expected:
                raise ValueError(
                    f"复用的 Haller dataset={view.name} manifest {key} identity 不一致"
                )
        if dataset_manifest.get("protocol_manifest_hash") not in {
            current_protocol_hash,
            str(view.record["manifest_hash"]),
        }:
            raise ValueError(
                f"复用的 Haller dataset={view.name} protocol identity 不一致"
            )
        expected_frames = tuple(int(frame) for frame in view.train_starts)
        frame_indices = tuple(int(frame) for frame in dataset_manifest.get("frame_indices", ()))
        if frame_indices != expected_frames:
            raise ValueError(
                f"复用的 Haller dataset={view.name} frame coverage 不完整"
            )
        frame_artifacts = dataset_manifest.get("frame_artifacts")
        if not isinstance(frame_artifacts, list) or len(frame_artifacts) != len(expected_frames):
            raise ValueError(
                f"复用的 Haller dataset={view.name} frame artifact 列表不完整"
            )
        frame_by_index = {
            int(item["frame_index"]): item
            for item in frame_artifacts
            if isinstance(item, Mapping) and "frame_index" in item
        }
        if tuple(sorted(frame_by_index)) != expected_frames:
            raise ValueError(
                f"复用的 Haller dataset={view.name} frame artifact index 不一致"
            )
        source_dirs: dict[int, Path] = {}
        for frame in expected_frames:
            item = frame_by_index[frame]
            relative_path = f"{view.name}/frame_{frame:06d}"
            if item.get("relative_path") != relative_path:
                raise ValueError(
                    f"复用的 Haller dataset={view.name} frame={frame} path identity 不一致"
                )
            frame_root = dataset_root / relative_path
            required_files = (
                "anchor_meta.json",
                "anchor_state.npy",
                "haller_gt.npy",
                "anchor_confidence.npy",
                "standard_ivd.npy",
                "omega.npy",
                "solid_mask.npy",
                "valid_fluid_mask.npy",
                "invalid_mask.npy",
            )
            missing = [name for name in required_files if not (frame_root / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"复用的 Haller dataset={view.name} frame={frame} 缺少 {missing}"
                )
            frame_meta = _read_json(frame_root / "anchor_meta.json")
            expected_frame_meta = {
                "dataset_name": view.name,
                "split_name": "train",
                "frame_index": frame,
                "source": haller_anchors.SOURCE_TRAIN,
                "haller_source_identity": TICKET04_HALLER_SOURCE_IDENTITY,
                "standard_ivd_source_identity": standard_ivd.SOURCE_IDENTITY,
                "standard_ivd_artifact": view.target_meta["standard_ivd_artifact"],
                "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
                "input_identity": view.target_meta["input_identity"][
                    "frame_input_identities"
                ][str(frame)],
                "geometry_mask_identity": view.target_meta["input_identity"][
                    "geometry_mask_identity"
                ],
                "dense_negative_generated": False,
                "test_source_read": False,
                "test_frame_indices_read": [],
                "legacy_ivd_read": False,
                "legacy_label_read": False,
                "legacy_artifacts_formal_source": False,
            }
            for key, expected in expected_frame_meta.items():
                if frame_meta.get(key) != expected:
                    raise ValueError(
                        f"复用的 Haller dataset={view.name} frame={frame} "
                        f"metadata {key} identity 不一致"
                    )
            if frame_meta.get("protocol_manifest_hash") not in {
                current_protocol_hash,
                str(view.record["manifest_hash"]),
            }:
                raise ValueError(
                    f"复用的 Haller dataset={view.name} frame={frame} "
                    "protocol identity 不一致"
                )
            source_dirs[frame] = frame_root
        source_datasets_entry = source_datasets.get(view.name)
        if not isinstance(source_datasets_entry, Mapping):
            raise ValueError(f"复用的 Haller global manifest 缺少 dataset={view.name}")
        if source_datasets_entry.get("manifest_hash") != dataset_manifest.get("manifest_hash"):
            raise ValueError(f"复用的 Haller dataset={view.name} manifest hash 不一致")
        per_dataset_info[view.name] = {
            "manifest_path": str(dataset_manifest_path),
            "manifest_hash": dataset_manifest["manifest_hash"],
            "frame_count": len(expected_frames),
            "failure_frame_count": int(dataset_manifest.get("failure_frame_count", 0)),
            "failure_count": int(dataset_manifest.get("failure_count", 0)),
        }

    if destination_haller.exists():
        raise FileExistsError(
            f"Haller reuse destination already exists; refusing overwrite: {destination_haller}"
        )
    shutil.copytree(source_haller, destination_haller)
    destination_dataset_entries: dict[str, dict[str, Any]] = {}
    for view in views:
        destination_dataset_root = destination_haller / view.name
        dataset_manifest = _read_json(
            destination_dataset_root / "haller_train_manifest.json"
        )
        dataset_manifest["ticket04_execution_namespace"] = execution_namespace
        dataset_manifest["protocol_manifest_hash"] = current_protocol_hash
        rebound_frame_artifacts = []
        for item in dataset_manifest["frame_artifacts"]:
            rebound_item = dict(item)
            frame = int(rebound_item["frame_index"])
            frame_meta_path = (
                destination_dataset_root
                / str(rebound_item["relative_path"])
                / "anchor_meta.json"
            )
            frame_meta = _read_json(frame_meta_path)
            frame_meta["ticket04_execution_namespace"] = execution_namespace
            frame_meta["protocol_manifest_hash"] = current_protocol_hash
            _write_json(frame_meta_path, frame_meta)
            rebound_item["metadata_hash"] = _json_hash(frame_meta)
            rebound_frame_artifacts.append(rebound_item)
        dataset_manifest["frame_artifacts"] = rebound_frame_artifacts
        dataset_manifest.pop("manifest_hash", None)
        dataset_manifest["manifest_hash"] = _json_hash(dataset_manifest)
        _write_json(
            destination_dataset_root / "haller_train_manifest.json",
            dataset_manifest,
        )
        destination_dataset_entries[view.name] = {
            **dict(global_manifest["datasets"][view.name]),
            "manifest_path": str(
                destination_dataset_root / "haller_train_manifest.json"
            ),
            "manifest_hash": dataset_manifest["manifest_hash"],
            "frame_artifacts": rebound_frame_artifacts,
        }
        frame_artifacts = rebound_frame_artifacts
        view.haller_dirs = {
            int(item["frame_index"]): destination_haller
            / view.name
            / str(item["relative_path"])
            for item in frame_artifacts
        }
        view.haller_info = {
            int(item["frame_index"]): dict(item) for item in frame_artifacts
        }
    destination_global_manifest = copy.deepcopy(global_manifest)
    destination_global_manifest["ticket04_execution_namespace"] = execution_namespace
    destination_global_manifest["protocol_manifest_hash"] = current_protocol_hash
    destination_global_manifest["datasets"] = destination_dataset_entries
    destination_global_manifest.pop("anchor_hash", None)
    destination_global_manifest.pop("manifest_hash", None)
    destination_anchor_hash = _json_hash(destination_global_manifest)
    destination_global_manifest["anchor_hash"] = destination_anchor_hash
    destination_global_manifest["manifest_hash"] = _json_hash(
        destination_global_manifest
    )
    _write_json(
        destination_haller / "haller_train_global_manifest.json",
        destination_global_manifest,
    )
    return {
        "source_output_directory": str(source_root),
        "source_haller_directory": str(source_haller),
        "source_global_manifest_hash": manifest_hash,
        "source_anchor_hash": anchor_hash,
        "anchor_hash": destination_anchor_hash,
        "protocol_manifest_hash": current_protocol_hash,
        "protocol_metadata_rebound": True,
        "dataset_frame_counts": {
            name: int(info["frame_count"]) for name, info in per_dataset_info.items()
        },
        "dataset_failure_counts": {
            name: int(info["failure_count"]) for name, info in per_dataset_info.items()
        },
        "reused": True,
        "test_source_read": False,
        "test_frame_indices_read": [],
        "final_test_relationships": "not_read",
        "global_manifest": destination_global_manifest,
    }


def _load_trainer_resume(trainer, checkpoint, *, mode, dataset_config,
                         split_config, sampling_config):
    """Call the existing W1-P and Hybrid trainer resume APIs explicitly."""
    if mode == contract.MODE_W1_P:
        return trainer.load_checkpoint(checkpoint,
            expected_dataset_config=dataset_config,
            expected_split_config=split_config,
            expected_sampling_config=sampling_config)
    return trainer.load_checkpoint(checkpoint, dataset_config=dataset_config,
        split_config=split_config, sampling_config=sampling_config)


def run_method(
    *,
    method_name: str,
    mode: str,
    cfg: Mapping[str, Any],
    views: Sequence[DatasetView],
    manifest: Mapping[str, Any],
    anchor_metadata: Mapping[str, Any],
    anchor_hash: str,
    output_root: Path,
    epochs: int,
    samples_per_epoch: int,
    batch_size: int,
    positive_fraction: float,
    num_workers: int,
    seed: int,
    variance_gate: float,
    sampling_config: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    split_config: Mapping[str, Any],
    t_win: int,
    window_step: int,
    patch_size: tuple[int, int],
    groups: tuple[int, int],
    delta_frac: float,
    L: int,
    n_substeps: int,
    t_scale: float,
    execution_namespace: str,
    resume_checkpoint: Path | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    device = torch.device(device if device is not None else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    method_root = output_root / "methods" / method_name
    method_root.mkdir(parents=True, exist_ok=resume_checkpoint is not None)
    trainer, _model = _make_trainer(
        cfg,
        mode=mode,
        views=views,
        manifest=manifest,
        anchor_metadata=anchor_metadata,
        anchor_hash=anchor_hash,
        seed=seed,
        variance_gate=variance_gate,
        device=device,
    )
    start_epoch = 0
    if resume_checkpoint is not None:
        restored = _load_trainer_resume(
            trainer, resume_checkpoint, mode=mode, dataset_config=dataset_config,
            split_config=split_config, sampling_config=sampling_config)
        start_epoch = int(restored["epoch"]) + 1
        print(f"[ticket04] resumed method={method_name} completed={start_epoch} checkpoint={resume_checkpoint}", flush=True)
    if resume_checkpoint is not None:
        trainer._move_models(device)
        for state in trainer.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(device)
    history = []
    checkpoint_set = set(checkpoint_epochs(epochs))
    method_start = time.time()
    for epoch_zero in range(start_epoch, epochs):
        specs = _build_epoch_specs(
            views,
            samples_per_epoch=samples_per_epoch,
            batch_size=batch_size,
            positive_fraction=positive_fraction,
            seed=seed,
            epoch=epoch_zero,
        )
        source_dataset = _CurrentSourceBatchDataset(
            views,
            specs,
            patch_size=patch_size,
            t_win=t_win,
            groups=groups,
            delta_frac=delta_frac,
            L=L,
            n_substeps=n_substeps,
            t_scale=t_scale,
            seed=seed,
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": source_dataset,
            "batch_size": None,
            "shuffle": False,
            "num_workers": int(num_workers),
            "collate_fn": _identity_collate,
            "pin_memory": False,
        }
        if num_workers:
            loader_kwargs["prefetch_factor"] = 1
        loader = DataLoader(**loader_kwargs)

        def contract_batches():
            for payload in loader:
                view = views[int(payload["dataset_index"])]
                yield _make_contract_batch(
                    payload,
                    view,
                    mode=mode,
                    t_win=t_win,
                    window_step=window_step,
                    anchor_metadata=(
                        None if mode == contract.MODE_W1_P else anchor_metadata
                    ),
                    anchor_hash=None if mode == contract.MODE_W1_P else anchor_hash,
                )

        if mode == contract.MODE_W1_P:
            epoch_log = trainer.run_epoch(
                contract_batches(), epoch=epoch_zero, device=device
            )
        else:
            epoch_log = trainer.run_epoch(
                contract_batches(), epoch=epoch_zero, device=device
            )
        trainer.scheduler.step(epoch_zero)
        completed = epoch_zero + 1
        epoch_log = dict(epoch_log)
        epoch_log["epoch_completed"] = completed
        epoch_log["method_name"] = method_name
        epoch_log["elapsed_seconds"] = float(time.time() - method_start)
        history.append(epoch_log)
        print(
            f"[ticket04] method={method_name} epoch={completed}/{epochs} "
            f"batches={len(specs)} elapsed={epoch_log['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if completed in checkpoint_set:
            checkpoint_path = (
                method_root / "checkpoints" / f"epoch_{completed:04d}.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"[ticket04] checkpoint method={method_name} "
                f"epoch={completed} path={checkpoint_path}",
                flush=True,
            )
            _save_checkpoint(
                trainer,
                mode=mode,
                path=checkpoint_path,
                completed_epoch=completed,
                dataset_config=dataset_config,
                split_config=split_config,
                sampling_config=sampling_config,
                metrics=epoch_log,
                calibration_policy=_aggregate_policy(
                    views,
                    manifest,
                    variance_gate
                    if mode == contract.MODE_W2_ON_HYBRID
                    else None,
                ),
                anchor_metadata=anchor_metadata,
                execution_namespace=execution_namespace,
            )
        del loader
    report = {
        "method_name": method_name,
        "mode": mode,
        "execution_namespace": execution_namespace,
        "epochs_completed": int(epochs),
        "checkpoint_epochs": sorted(checkpoint_set),
        "history": history,
        "resumed_from_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
        "history_start_epoch": start_epoch + 1,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "checkpoint_directory": str(method_root / "checkpoints"),
        "output_directory": str(method_root),
        "device": str(device),
        "test_source_read": False,
        "test_frame_indices_read": [],
        "final_test_relationships": "not_read",
        "legacy_ivd_read": False,
        "legacy_label_read": False,
    }
    _write_json(method_root / "training_report.json", report)
    return report


def guarded_final_test_preflight(
    manifest: Mapping[str, Any], freeze_record_path: Path | None
) -> dict[str, Any]:
    """Try only the protocol guard; do not construct a test-value producer."""
    freeze = None
    if freeze_record_path is not None and freeze_record_path.is_file():
        freeze = _read_json(freeze_record_path)
    guard = protocol.FinalTestAccessGuard(
        protocol_manifest=manifest,
        freeze_record=freeze,
    )
    sealed = guard.sealed_view()
    try:
        authorized = guard.authorize_relationships()
    except protocol.FinalTestAccessError as exc:
        return {
            "status": "blocked",
            "reason": str(exc),
            "sealed_view": sealed,
            "freeze_record_path": (
                None if freeze_record_path is None else str(freeze_record_path)
            ),
            "test_values_read": False,
            "test_source_read": False,
            "final_test_relationships": "sealed",
        }
    return {
        "status": "authorized_preflight_only",
        "authorization": authorized,
        "sealed_view": sealed,
        "freeze_record_path": (
            None if freeze_record_path is None else str(freeze_record_path)
        ),
        "test_values_read": False,
        "test_source_read": False,
        "final_test_relationships": "authorized_but_not_read",
    }


def run(
    *,
    input_root: str | Path,
    ticket03_root: str | Path,
    output_root: str | Path,
    config_path: str | Path = "config/pathline_transformer_multi.yaml",
    epochs: int = DEFAULT_EPOCHS,
    samples_per_epoch: int = DEFAULT_SAMPLES_PER_EPOCH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    positive_fraction: float = 0.5,
    num_workers: int = DEFAULT_NUM_WORKERS,
    haller_workers: int = DEFAULT_HALLER_WORKERS,
    seed: int = DEFAULT_SEED,
    device: str = "cuda:0",
    variance_gate: float = DEFAULT_VARIANCE_GATE,
    freeze_record: str | Path | None = None,
    execution_namespace: str = DEFAULT_EXECUTION_NAMESPACE,
    reuse_completed_run: str | Path | None = None,
    reuse_haller_run: str | Path | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    if method is not None and method not in dict(METHODS):
        raise ValueError(f"unsupported Ticket 04 method: {method}")
    selected_methods = METHODS if method is None else ((method, dict(METHODS)[method]),)
    if reuse_completed_run is not None and method not in (None, "w1_p"):
        raise ValueError("reuse_completed_run requires W1-P in the selected methods")
    device = torch.device(device)
    execution_namespace = str(execution_namespace).strip()
    if not execution_namespace or any(char in execution_namespace for char in "\r\n"):
        raise ValueError("execution_namespace must be a non-empty single-line string")
    output_root = Path(output_root).resolve()
    input_root = Path(input_root).resolve()
    ticket03_root = Path(ticket03_root).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Ticket 04 output namespace already exists; refusing overwrite: {output_root}"
        )
    if output_root == input_root or output_root.is_relative_to(input_root):
        raise ValueError("Ticket 04 output must not overlap raw input root")
    if output_root == ticket03_root or output_root.is_relative_to(ticket03_root):
        raise ValueError("Ticket 04 output must not overlap Ticket 03 artifact root")
    if epochs <= 0 or samples_per_epoch <= 0 or batch_size <= 0:
        raise ValueError("epochs, samples_per_epoch, and batch_size must be positive")
    if num_workers < 0 or haller_workers <= 0:
        raise ValueError("worker counts are invalid")
    variance_gate = float(variance_gate)
    if not np.isfinite(variance_gate) or not 0.0 <= variance_gate <= 1.0:
        raise ValueError("variance_gate must be finite and in [0,1]")
    manifest_path = ticket03_root / "p95p80_protocol_manifest.json"
    manifest = protocol.read_p95p80_protocol_manifest(manifest_path)
    cfg = train_kaggle.load_config(config_path)
    data_cfg = dict(cfg.get("data", {}))
    patch_size = tuple(int(value) for value in data_cfg.get("patch_size", (32, 32)))
    stride = tuple(int(value) for value in data_cfg.get("stride", (16, 16)))
    groups = tuple(int(value) for value in data_cfg.get("groups", (8, 8)))
    t_win = int(data_cfg.get("t_win", 24))
    window_step = int(data_cfg.get("window_step", 4))
    delta_frac = float(data_cfg.get("delta_frac", 0.05))
    L = int(data_cfg.get("L", 16))
    n_substeps = int(data_cfg.get("n_substeps", 4))
    t_scale = float(data_cfg.get("t_scale", 0.25))
    positive_fraction = float(data_cfg.get("positive_fraction", positive_fraction))
    output_root.mkdir(parents=True, exist_ok=False)
    views = [
        load_dataset_view(
            input_root,
            ticket03_root,
            name,
            manifest,
            patch_size=patch_size,
            stride=stride,
            t_win=t_win,
            window_step=window_step,
            groups=groups,
            delta_frac=delta_frac,
        )
        for name in DATASET_ORDER
    ]
    preflight = {
        "artifact_type": "ticket04_real_training_preflight",
        "schema_version": "criterion-ambiguity-ticket04-real-training-v1",
        "ticket04_execution_namespace": execution_namespace,
        "protocol_version": manifest["protocol_version"],
        "protocol_manifest_hash": manifest["manifest_hash"],
        "dataset_order": list(DATASET_ORDER),
        "datasets": {view.name: view.audit for view in views},
        "input_root": str(input_root),
        "ticket03_root": str(ticket03_root),
        "output_root": str(output_root),
        "config_path": str(Path(config_path).resolve()),
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "haller_source_identity": copy.deepcopy(TICKET04_HALLER_SOURCE_IDENTITY),
        "final_test_artifacts_read": False,
        "test_source_read": False,
        "test_frame_indices_read": [],
        "final_test_relationships": "not_read",
    }
    _write_json(output_root / "preflight.json", preflight)
    if reuse_haller_run is None:
        anchor_manifest, anchor_hash = build_current_haller_train(
            views,
            output_root / "haller_anchor_train",
            workers=haller_workers,
            run_namespace=execution_namespace,
            protocol_manifest_hash=manifest["manifest_hash"],
        )
        haller_reuse: dict[str, Any] | None = None
    else:
        haller_reuse = _reuse_haller_train(
            reuse_haller_run,
            output_root,
            views=views,
            manifest=manifest,
            execution_namespace=execution_namespace,
        )
        anchor_manifest = copy.deepcopy(haller_reuse["global_manifest"])
        anchor_hash = str(haller_reuse["anchor_hash"])
    anchor_metadata = copy.deepcopy(anchor_manifest)
    anchor_metadata["anchor_hash"] = anchor_hash
    _write_json(output_root / "anchor_metadata.json", anchor_metadata)
    dataset_config = _dataset_config(views)
    split_config = _split_config(manifest, views)
    split_config["pathline_window_policy"]["t_win"] = t_win
    split_config["pathline_window_policy"]["window_step"] = window_step
    sampling_config = _sampling_config(
        samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        positive_fraction=positive_fraction,
        patch_size=patch_size,
        stride=stride,
        t_win=t_win,
        window_step=window_step,
        groups=groups,
        delta_frac=delta_frac,
        L=L,
        n_substeps=n_substeps,
        t_scale=t_scale,
    )
    run_manifest = {
        **preflight,
        "anchor_hash": anchor_hash,
        "anchor_manifest_hash": anchor_manifest["manifest_hash"],
        "anchor_metadata_path": str(output_root / "anchor_metadata.json"),
        "dataset_config": dataset_config,
        "split_config": split_config,
        "sampling_config": sampling_config,
        "training": {
            "epochs": int(epochs),
            "samples_per_epoch": int(samples_per_epoch),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "haller_workers": int(haller_workers),
            "seed": int(seed),
            "device": str(device),
            "variance_gate": float(variance_gate),
            "execution_namespace": execution_namespace,
            "checkpoint_epochs": list(checkpoint_epochs(epochs)),
            "methods": [name for name, _mode in selected_methods],
            "method_seeds": {
                name: int(seed) + index if method is None else int(seed)
                for index, (name, _mode) in enumerate(selected_methods)
            },
        },
        "standard_ivd_provenance": copy.deepcopy(standard_ivd.SOURCE_IDENTITY),
        "reuse_haller_run": (
            None
            if haller_reuse is None
            else {
                key: value
                for key, value in haller_reuse.items()
                if key != "global_manifest"
            }
        ),
        "threshold_provenance": {
            view.name: copy.deepcopy(view.target_meta["threshold_provenance"])
            for view in views
        },
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    reused_methods: dict[str, dict[str, Any]] = {}
    if reuse_completed_run is not None:
        reused = _reuse_completed_w1p(
            reuse_completed_run,
            output_root,
            manifest=manifest,
            epochs=int(epochs),
            samples_per_epoch=int(samples_per_epoch),
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            haller_workers=int(haller_workers),
            seed=int(seed),
            execution_namespace=execution_namespace,
        )
        reused_methods[reused["method_name"]] = reused
        run_manifest["reuse_completed_run"] = reused
        run_manifest["training"]["reused_methods"] = list(reused_methods)
        _write_json(output_root / "run_manifest.json", run_manifest)
    method_reports = {}
    for method_index, (method_name, mode) in enumerate(selected_methods):
        if method_name in reused_methods:
            method_reports[method_name] = reused_methods[method_name]["report"]
            print(
                f"[ticket04] method={method_name} reused completed stage from "
                f"{reused_methods[method_name]['source_output_directory']}",
                flush=True,
            )
            continue
        method_reports[method_name] = run_method(
            method_name=method_name,
            mode=mode,
            cfg=cfg,
            views=views,
            manifest=manifest,
            anchor_metadata=anchor_metadata,
            anchor_hash=anchor_hash,
            output_root=output_root,
            epochs=int(epochs),
            samples_per_epoch=int(samples_per_epoch),
            batch_size=int(batch_size),
            positive_fraction=positive_fraction,
            num_workers=int(num_workers),
            seed=int(seed) + method_index if method is None else int(seed),
            device=device,
            variance_gate=variance_gate,
            sampling_config=sampling_config,
            dataset_config=dataset_config,
            split_config=split_config,
            t_win=t_win,
            window_step=window_step,
            patch_size=patch_size,
            groups=groups,
            delta_frac=delta_frac,
            L=L,
            n_substeps=n_substeps,
            t_scale=t_scale,
            execution_namespace=execution_namespace,
        )
    freeze_path = None if freeze_record is None else Path(freeze_record).resolve()
    if freeze_path is None:
        candidate = ticket03_root / "p95p80_freeze_record.json"
        freeze_path = candidate if candidate.is_file() else None
    final_test_preflight = guarded_final_test_preflight(manifest, freeze_path)
    result = {
        "run_manifest_path": str(output_root / "run_manifest.json"),
        "preflight_path": str(output_root / "preflight.json"),
        "anchor_metadata_path": str(output_root / "anchor_metadata.json"),
        "method_reports": method_reports,
        "reused_methods": {
            name: {
                key: value
                for key, value in info.items()
                if key != "report"
            }
            for name, info in reused_methods.items()
        },
        "final_test_preflight": final_test_preflight,
        "final_test_evaluation": {
            "status": "not_run",
            "reason": (
                "No test values were read; final relationships remain guarded "
                "by the independent final protocol freeze."
            ),
        },
        "test_source_read": False,
        "test_frame_indices_read": [],
        "final_test_artifacts_read": False,
        "final_test_relationships": final_test_preflight[
            "final_test_relationships"
        ],
    }
    _write_json(output_root / "ticket04_report.json", result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real current-source W1-P, W1-Hybrid, and W2-on-Hybrid "
            "training with ten-epoch checkpoints."
        )
    )
    parser.add_argument("--input-root", default="outputs/datasets")
    parser.add_argument(
        "--ticket03-root",
        default=(
            ".scratch/criterion-ambiguity-vortex-learning-haller-ivd-p95p80-v1/"
            "artifacts/ticket03-standard-ivd-p95p80-v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new output namespace; existing directories are never overwritten",
    )
    parser.add_argument(
        "--config", default="config/pathline_transformer_multi.yaml"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--samples-per-epoch", type=int, default=DEFAULT_SAMPLES_PER_EPOCH
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--haller-workers", type=int, default=DEFAULT_HALLER_WORKERS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--method", choices=tuple(name for name, _mode in METHODS),
        help="train only this method using the exact --seed and fresh weights unless --reuse-completed-run is supplied; omitted runs the historical three-method sequence",
    )
    parser.add_argument("--device", default="cuda:0", help="student and EMA device (within CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--variance-gate", type=float, default=DEFAULT_VARIANCE_GATE)
    parser.add_argument("--freeze-record", default=None)
    parser.add_argument(
        "--execution-namespace",
        default=DEFAULT_EXECUTION_NAMESPACE,
        help="unique provenance namespace for this non-overwriting run",
    )
    parser.add_argument(
        "--reuse-completed-run",
        default=None,
        help=(
            "explicitly reuse a fully completed W1-P stage from a prior Ticket 04 run; "
            "in default multi-method mode W1-Hybrid and W2 still train in the new namespace"
        ),
    )
    parser.add_argument(
        "--reuse-haller-run",
        default=None,
        help=(
            "reuse a complete identity-validated train-only Haller artifact "
            "from a prior Ticket 04 run"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run(
        input_root=args.input_root,
        ticket03_root=args.ticket03_root,
        output_root=args.output_dir,
        config_path=args.config,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        haller_workers=args.haller_workers,
        seed=args.seed,
        method=args.method,
        device=args.device,
        variance_gate=args.variance_gate,
        freeze_record=args.freeze_record,
        execution_namespace=args.execution_namespace,
        reuse_completed_run=args.reuse_completed_run,
        reuse_haller_run=args.reuse_haller_run,
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
