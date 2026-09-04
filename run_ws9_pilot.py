"""Run the ticket-09 pilot against the prepared weak-supervision artifacts.

This module is the real-data adapter around :mod:`e2e_weak_supervision`.
It deliberately keeps the experiment seam small and explicit:

* the six weak datasets are loaded from the new ``weak_supervision`` namespace;
* legacy p85 is used only by the B0/B1 sampling contract (and by the same
  deterministic patch pool for W1 methods), never as W1 formal labels;
* train Haller anchors, calibration Haller GT, and test Haller GT are separate
  source-specific paths;
* the complete artifact manifest is checked before a model is instantiated;
* test artifacts are only projected by the final ``evaluate_test`` callbacks.

The Haller extractor itself remains NumPy/skimage CPU-only for this ticket.
The ``--device`` option controls model training/evaluation, not artifact
generation, and does not enable a CUDA Haller backend.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

import dataset as dataset_module
import e2e_weak_supervision as e2e
import evaluation_report
import extractor
import haller_anchors
import prepare_weak_supervision_artifacts as artifact_preparer
import w1_h
import w1_p
import w2
import w3
import weak_supervision_contract as contract


DATASETS = tuple(artifact_preparer.VALID_DATASETS)
TRAIN_SOURCE = haller_anchors.SOURCE_TRAIN
CALIBRATION_SOURCE = haller_anchors.SOURCE_CALIBRATION
TEST_SOURCE = haller_anchors.SOURCE_TEST
SOURCE_SPLITS = {
    TRAIN_SOURCE: "train",
    CALIBRATION_SOURCE: "calibration",
    TEST_SOURCE: "test",
}
# Calibration may choose a single global gate from this pre-registered grid.
# Training never consumes that selection: both W2 and W3 use the fixed
# train-time gate below, so calibration cannot change pseudo-label rules or
# model weights after training has started.
PILOT_VARIANCE_CANDIDATES = (0.005, 0.01, 0.02, 0.05)
PILOT_TRAINING_VARIANCE_GATE = 0.01


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 JSON artifact：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact 必须是 object：{path}")
    return value


def _json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _strict_int(value: Any, *, name: str, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是整数")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if result != value or (result <= 0 if positive else result < 0):
        adjective = "正整数" if positive else "非负整数"
        raise ValueError(f"{name} 必须是{adjective}")
    return result


def _configure_adapter_devices(
    adapter: contract.ChannelSelectingAdapter,
    *,
    mode: str,
    device: str,
    parallel_devices: Sequence[str] | None,
    wrap_data_parallel: bool = True,
) -> tuple[str, ...]:
    """Place one adapter on a primary CUDA device, optionally using DataParallel.

    The ticket fixes the training batch and optimizer settings.  This seam is
    therefore deliberately a device-only control: it does not split a batch
    in the caller, change the batch size, or alter any loss/sampling
    parameter.  ``DataParallel`` scatters the prescribed batch across the
    requested devices and gathers the prediction on the first (primary)
    device, which is the device used by the existing trainer loss seams.

    W1-P may set ``wrap_data_parallel=False`` when it owns a separate EMA
    teacher device.  In that case the full device group is still validated,
    but the student adapter stays on the primary device and is not replicated;
    W1PTrainer places the teacher on the second device.  This keeps the fixed
    global batch and loss semantics while avoiding two model replicas on every
    GPU in a student/teacher pair.
    """
    if not isinstance(adapter, contract.ChannelSelectingAdapter):
        raise TypeError("pilot model 必须是 ChannelSelectingAdapter")
    mode = contract.canonical_mode(mode)
    primary = torch.device(str(device))
    group = tuple(
        str(item).strip()
        for item in (parallel_devices if parallel_devices is not None else (str(device),))
        if str(item).strip()
    )
    if not group:
        group = (str(device),)
    if group[0] != str(device):
        raise ValueError(
            f"{mode} parallel device group 的 primary 必须等于 device："
            f"device={device!r} group={group!r}"
        )
    if len(set(group)) != len(group):
        raise ValueError(f"{mode} parallel device group 不能重复：{group!r}")
    if mode == contract.MODE_W3 and len(group) > 1:
        raise ValueError("W3 按 ticket 09 必须保持 single GPU")
    if primary.type != "cuda" and len(group) > 1:
        raise ValueError(
            f"{mode} 的 multi-GPU group 必须使用 CUDA：device={device!r} group={group!r}"
        )
    if primary.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"请求 {mode} CUDA device={device}，但 CUDA 不可用")
        count = int(torch.cuda.device_count())
        indices = []
        for item in group:
            parsed = torch.device(item)
            if parsed.type != "cuda" or parsed.index is None:
                raise ValueError(
                    f"{mode} parallel device 必须是带 index 的 cuda:N：{item!r}"
                )
            if int(parsed.index) >= count:
                raise ValueError(
                    f"{mode} parallel device 超出 device_count={count}：{item!r}"
                )
            indices.append(int(parsed.index))
        adapter.to(primary)
        if wrap_data_parallel and len(indices) > 1:
            if isinstance(adapter.model, torch.nn.DataParallel):
                raise ValueError(f"{mode} adapter 不应重复包裹 DataParallel")
            adapter.model = torch.nn.DataParallel(
                adapter.model,
                device_ids=indices,
                output_device=indices[0],
            )
    else:
        adapter.to(primary)
    return group


def _compare_artifact_summary(
    loaded: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    source: str,
    name: str,
    frame: int,
) -> None:
    metadata = loaded.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}/{name}/frame{frame} metadata 不是 object")
    fields = (
        "frame_index", "algorithm_version", "parameter_hash", "input_hash",
        "mask_hash", "failure_count", "valid", "artifact_array_hashes",
    )
    for field in fields:
        if metadata.get(field) != summary.get(field):
            raise ValueError(
                f"{source}/{name}/frame{frame} 的 {field} 与 manifest 不一致"
            )
    if metadata.get("source") != source or metadata.get("label_source") != source:
        raise ValueError(f"{source}/{name}/frame{frame} source metadata 不一致")
    if metadata.get("backend") not in haller_anchors.VALID_BACKENDS:
        raise ValueError("WS-9 artifact manifest 含未知 Haller backend")
    if metadata.get("resolved", metadata.get("backend")) != metadata.get("backend"):
        raise ValueError("WS-9 artifact manifest resolved backend 与 backend 不一致")
    if bool(metadata.get("cuda_used")):
        raise ValueError("WS-9 artifact manifest 含 CUDA Haller artifact")
    if metadata.get("literature", {}).get("status") != "pending_verification":
        raise ValueError("Haller artifact literature 必须保持 pending_verification")
    if metadata.get("backend") == haller_anchors.BACKEND_NUMBACS:
        parameters = metadata.get("parameters")
        evaluation = metadata.get("contour_evaluation")
        if not isinstance(parameters, Mapping) or not isinstance(evaluation, Mapping):
            raise ValueError("NumbaCS artifact 缺少 parameters/contour_evaluation")
        if parameters.get("numbacs_upstream_commit") != (
            "c067f542543f5dd4ae3dc45fc506213e8d98b845"
        ):
            raise ValueError("NumbaCS artifact upstream commit 不匹配")
        if parameters.get("numbacs_contour_level_count") != 20 or evaluation.get(
            "numbacs_default_nlevs"
        ) != 20:
            raise ValueError("NumbaCS artifact 必须使用 upstream default nlevs=20")
        if parameters.get("numbacs_nlevs_passed_explicitly") is not False:
            raise ValueError("NumbaCS artifact 不得显式传入 nlevs")
        if "numbacs_min_val" in parameters:
            raise ValueError("NumbaCS artifact 不得携带旧的 numbacs_min_val override")
        expected_explicit = ["convexity_deficiency", "min_len"]
        expected_omitted = ["min_val", "nlevs", "start_level", "end_level"]
        if parameters.get("numbacs_explicit_kwargs") != expected_explicit:
            raise ValueError("NumbaCS artifact explicit kwargs contract 不匹配")
        if parameters.get("numbacs_omitted_kwargs") != expected_omitted:
            raise ValueError("NumbaCS artifact omitted kwargs contract 不匹配")
        if parameters.get("numbacs_runtime_mode") != "numba_jit":
            raise ValueError("NumbaCS artifact 必须来自 native Numba JIT runtime")
        if evaluation.get("numbacs_native_call") is not True:
            raise ValueError("NumbaCS artifact contour evaluation 不是 native call")
        if evaluation.get("numbacs_omitted_kwargs") != expected_omitted:
            raise ValueError("NumbaCS contour evaluation omitted kwargs 不匹配")


def _read_haller_metadata_only(
    path: pathlib.Path, *, expected_source: str
) -> dict[str, Any]:
    """Validate test artifact declarations without reading test label arrays.

    Ticket 09 permits test GT reads only after model, threshold, gate, epoch,
    method, and seed are frozen.  Preflight therefore checks the manifest,
    metadata contract, and required file presence, while the final
    ``test_records`` path calls ``load_haller_artifact`` and verifies array
    hashes at evaluation time.
    """
    meta_path = path / "anchor_meta.json"
    metadata = _read_json(meta_path)
    if metadata.get("source") != expected_source:
        raise ValueError(
            f"Haller metadata source 不匹配：expected={expected_source!r} "
            f"actual={metadata.get('source')!r}"
        )
    if metadata.get("label_source") != expected_source:
        raise ValueError("Haller metadata label_source 与 test source 不一致")
    parameters = metadata.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"Haller metadata 缺少 parameters：{meta_path}")
    if metadata.get("parameter_hash") != _json_hash(parameters):
        raise ValueError(f"Haller metadata parameter_hash 校验失败：{meta_path}")
    if not isinstance(metadata.get("artifact_array_hashes"), Mapping):
        raise ValueError(f"Haller metadata 缺少 artifact_array_hashes：{meta_path}")
    required = (
        "haller_gt.npy", "anchor_state.npy", "anchor_confidence.npy",
        "standard_ivd.npy", "omega.npy", "solid_mask.npy",
    )
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Haller test artifact 文件不完整：{path} missing={missing!r}"
        )
    return metadata


def validate_prepared_artifacts(
    weak_root: str | pathlib.Path,
    haller_root: str | pathlib.Path,
    *,
    dataset_names: tuple[str, ...] = DATASETS,
    t_win: int | None = None,
    window_step: int | None = None,
) -> dict[str, Any]:
    """Validate every six-dataset artifact, frame hash, and source boundary.

    This is an acceptance gate, not a model-selection operation.  It reads
    arrays to verify their declared hashes, but never uses test labels to
    construct a training batch, threshold, variance gate, or method choice.
    """
    weak_path = pathlib.Path(weak_root)
    haller_path = pathlib.Path(haller_root)
    names = tuple(dataset_names)
    if names != DATASETS:
        raise ValueError(f"WS-9 只接受固定六个数据集：{DATASETS!r}")
    total_manifest = _read_json(weak_path / "artifact_manifest.json")
    declared_hash = total_manifest.get("manifest_hash")
    if not isinstance(declared_hash, str) or not declared_hash:
        raise ValueError("artifact_manifest.json 缺少 manifest_hash")
    total_payload = copy.deepcopy(total_manifest)
    total_payload.pop("manifest_hash", None)
    if _json_hash(total_payload) != declared_hash:
        raise ValueError("artifact_manifest.json manifest_hash 校验失败")
    if total_manifest.get("artifact_type") != "weak_supervision_pilot_inputs":
        raise ValueError("artifact manifest artifact_type 不匹配")
    if tuple(total_manifest.get("datasets", ())) != names:
        raise ValueError("artifact manifest dataset 顺序/集合不匹配")
    if total_manifest.get("split_mode") != dataset_module.WEAK_SUPERVISION_SPLIT_MODE:
        raise ValueError("artifact manifest 必须使用 split_mode=weak_supervision")
    manifest_t_win = _strict_int(total_manifest.get("t_win"), name="manifest.t_win", positive=True)
    manifest_step = _strict_int(
        total_manifest.get("window_step"), name="manifest.window_step", positive=True
    )
    if t_win is not None and manifest_t_win != _strict_int(t_win, name="t_win", positive=True):
        raise ValueError("requested t_win 与 artifact manifest 不一致")
    if window_step is not None and manifest_step != _strict_int(
        window_step, name="window_step", positive=True
    ):
        raise ValueError("requested window_step 与 artifact manifest 不一致")
    backend = total_manifest.get("haller_backend")
    if not isinstance(backend, Mapping):
        raise ValueError("artifact manifest 缺少 haller_backend")
    resolved_backend = backend.get("resolved", backend.get("backend"))
    if resolved_backend not in haller_anchors.VALID_BACKENDS or backend.get("cuda_used"):
        raise ValueError(
            "WS-9 artifact contract 只允许 CPU NumPy、fast_haller 或 numbacs Haller"
        )
    if backend.get("backend") != resolved_backend:
        raise ValueError("WS-9 artifact manifest backend/resolved 字段不一致")
    if total_manifest.get("haller_literature_status") != "pending_verification":
        raise ValueError("artifact manifest literature status 必须是 pending_verification")

    weak_contracts: dict[str, Any] = {}
    target_contracts: dict[str, Any] = {}
    for name in names:
        root = weak_path / "datasets" / name / "dataset"
        meta = _read_json(root / dataset_module.FN_META)
        shape = tuple(_strict_int(value, name=f"{name}.shape") for value in meta.get("shape", ()))
        if len(shape) != 3:
            raise ValueError(f"{name} weak dataset shape 必须是三维")
        dataset_module._validate_weak_contract_metadata(
            meta, dataset_name=name, total_frames=shape[0]
        )
        if meta.get("window", {}).get("t_win") != manifest_t_win:
            raise ValueError(f"{name} weak dataset t_win 与总 manifest 不一致")
        if meta.get("window", {}).get("window_step") != manifest_step:
            raise ValueError(f"{name} weak dataset window_step 与总 manifest 不一致")
        required = (
            dataset_module.FN_U, dataset_module.FN_V, dataset_module.FN_IVD,
            dataset_module.FN_LABEL, dataset_module.FN_MASK,
        )
        if any(not (root / filename).exists() for filename in required):
            raise FileNotFoundError(f"{name} weak dataset 文件不完整：{root}")
        manifest_contract = total_manifest.get("weak_dataset_contracts", {}).get(name, {})
        if manifest_contract.get("contract_hash") != meta.get("contract_hash"):
            raise ValueError(f"{name} weak contract hash 与总 manifest 不一致")
        for hash_field in ("input_hash", "input_array_hashes", "array_hashes"):
            if manifest_contract.get(hash_field) != meta.get(hash_field):
                raise ValueError(f"{name} weak {hash_field} 与总 manifest 不一致")

        target_root = weak_path / "w1_p_targets" / name
        target_meta = _read_json(target_root / "target_meta.json")
        if target_meta.get("source") != "w1_p_train":
            raise ValueError(f"{name} W1-P target source 不匹配")
        if target_meta.get("label_source") != contract.LABEL_SOURCE_LOCAL_P90_P60:
            raise ValueError(f"{name} W1-P target label source 不匹配")
        if target_meta.get("split_name") != "train":
            raise ValueError(f"{name} W1-P target 必须是 train-only")
        if target_meta.get("input_dataset_contract_hash") != meta.get("contract_hash"):
            raise ValueError(f"{name} W1-P target input contract hash 不匹配")
        manifest_target = total_manifest.get("w1_p_targets", {}).get(name, {})
        for hash_field in (
            "input_ivd_hash", "input_mask_hash", "target_array_hashes"
        ):
            if manifest_target.get(hash_field) != target_meta.get(hash_field):
                raise ValueError(f"{name} W1-P target {hash_field} 与总 manifest 不一致")
        target_files = {
            "anchor_state.npy": np.int8,
            "labels.npy": np.float32,
            "label_mask.npy": np.uint8,
            "unknown_mask.npy": np.uint8,
        }
        for filename, dtype in target_files.items():
            path = target_root / filename
            if not path.exists():
                raise FileNotFoundError(f"{name} W1-P target 缺少 {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if tuple(array.shape) != shape or array.dtype != np.dtype(dtype):
                raise ValueError(f"{name} W1-P target {filename} shape/dtype 不匹配")
            target_key = pathlib.Path(filename).stem
            declared_target_hashes = target_meta.get("target_array_hashes")
            if not isinstance(declared_target_hashes, Mapping):
                raise ValueError(f"{name} W1-P target 缺少 target_array_hashes")
            actual_hash = artifact_preparer._array_hash(array)
            if declared_target_hashes.get(target_key) != actual_hash:
                raise ValueError(f"{name} W1-P target {filename} hash 校验失败")
        weak_contracts[name] = {
            "contract_hash": meta["contract_hash"],
            "generation_hash": meta["generation_hash"],
            "split_ranges": copy.deepcopy(meta["split_ranges"]),
        }
        target_contracts[name] = {
            "path": str(target_root),
            "positive_threshold": target_meta["positive_threshold"],
            "negative_threshold": target_meta["negative_threshold"],
            "input_dataset_contract_hash": target_meta["input_dataset_contract_hash"],
        }

    haller_manifests = total_manifest.get("haller_manifests")
    if not isinstance(haller_manifests, Mapping):
        raise ValueError("artifact manifest 缺少 haller_manifests")
    haller_hashes: dict[str, dict[str, str]] = {}
    frame_counts: dict[str, dict[str, int]] = {}
    for source, split_name in SOURCE_SPLITS.items():
        source_manifests = haller_manifests.get(source)
        if not isinstance(source_manifests, Mapping):
            raise ValueError(f"artifact manifest 缺少 source={source}")
        haller_hashes[source] = {}
        frame_counts[source] = {}
        for name in names:
            manifest = _read_json(
                haller_path / source / name / "anchor_manifest.json"
            )
            if manifest != source_manifests.get(name):
                raise ValueError(f"{source}/{name} anchor_manifest 与总 manifest 不一致")
            manifest_hash = manifest.get("manifest_hash")
            payload = copy.deepcopy(manifest)
            payload.pop("manifest_hash", None)
            if not isinstance(manifest_hash, str) or _json_hash(payload) != manifest_hash:
                raise ValueError(f"{source}/{name} manifest_hash 校验失败")
            if manifest.get("source") != source or manifest.get("split_name") != split_name:
                raise ValueError(f"{source}/{name} source/split 不匹配")
            expected_range = tuple(
                int(value)
                for value in weak_contracts[name]["split_ranges"][split_name]
            )
            if tuple(manifest.get("frame_range", ())) != expected_range:
                raise ValueError(f"{source}/{name} frame_range 不匹配")
            expected_window = {
                "split_name": split_name,
                "frame_range": list(expected_range),
                "t_win": manifest_t_win,
                "window_step": manifest_step,
                "complete_windows_only": True,
            }
            if manifest.get("window") != expected_window:
                raise ValueError(f"{source}/{name} window contract 不匹配")
            records = manifest.get("frame_artifacts")
            if not isinstance(records, list):
                raise ValueError(f"{source}/{name} frame_artifacts 必须是 list")
            expected_frames = list(range(*expected_range))
            if [record.get("frame_index") for record in records] != expected_frames:
                raise ValueError(f"{source}/{name} frame_artifacts 不完整或顺序漂移")
            if manifest.get("frame_count") != len(records):
                raise ValueError(f"{source}/{name} frame_count 不一致")
            if manifest.get("input_hash") != _hash_strings(
                [str(record.get("input_hash")) for record in records]
            ):
                raise ValueError(f"{source}/{name} aggregate input_hash 校验失败")
            if manifest.get("valid_frame_count") != sum(
                int(bool(record.get("valid"))) for record in records
            ):
                raise ValueError(f"{source}/{name} valid_frame_count 不一致")
            if manifest.get("invalid_frame_count") != sum(
                int(not bool(record.get("valid"))) for record in records
            ):
                raise ValueError(f"{source}/{name} invalid_frame_count 不一致")
            if manifest.get("failure_count") != sum(
                int(record.get("failure_count", 0)) for record in records
            ):
                raise ValueError(f"{source}/{name} failure_count 不一致")
            for record in records:
                frame = _strict_int(record.get("frame_index"), name="frame_index")
                frame_dir = haller_path / source / name / f"frame{frame}"
                if source == TEST_SOURCE:
                    metadata = _read_haller_metadata_only(
                        frame_dir, expected_source=source
                    )
                    loaded = {"metadata": metadata}
                else:
                    loaded = haller_anchors.load_haller_artifact(
                        frame_dir, expected_source=source
                    )
                    metadata = loaded["metadata"]
                if metadata.get("split_name") != split_name:
                    raise ValueError(f"{source}/{name}/frame{frame} split metadata 不匹配")
                if metadata.get("window") != expected_window:
                    raise ValueError(f"{source}/{name}/frame{frame} window metadata 不匹配")
                _compare_artifact_summary(
                    loaded, record, source=source, name=name, frame=frame
                )
                if metadata.get("mask_hash") != manifest.get("mask_hash"):
                    raise ValueError(f"{source}/{name}/frame{frame} mask hash 漂移")
            haller_hashes[source][name] = manifest_hash
            frame_counts[source][name] = len(records)
    return {
        "manifest_path": str(weak_path / "artifact_manifest.json"),
        "manifest_hash": declared_hash,
        "weak_contracts": weak_contracts,
        "w1_p_targets": target_contracts,
        "haller_manifest_hashes": haller_hashes,
        "haller_frame_counts": frame_counts,
        "haller_root": str(haller_path),
        "weak_root": str(weak_path),
        "t_win": manifest_t_win,
        "window_step": manifest_step,
        "datasets": list(names),
        "test_labels_integrity_checked": True,
        "haller_backend": copy.deepcopy(dict(backend)),
        "haller_contour_mode": total_manifest.get("haller_contour_mode"),
        "haller_literature_status": "pending_verification",
    }


class TwoStepPilotScheduler:
    """Checkpointable epoch scheduler compatible with all dependency trainers."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        lr: float,
        second_lr: float,
        warmup_epochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.lr = float(lr)
        self.second_lr = float(second_lr)
        self.warmup_epochs = _strict_int(warmup_epochs, name="warmup_epochs", positive=True)
        self.epoch = 0
        self._set_lr(self.lr)

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)

    def step(self, epoch: int | None = None) -> float:
        self.epoch = self.epoch + 1 if epoch is None else _strict_int(
            epoch, name="scheduler.epoch"
        )
        value = self.lr if self.epoch < self.warmup_epochs else self.second_lr
        self._set_lr(value)
        return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "lr": self.lr,
            "second_lr": self.second_lr,
            "warmup_epochs": self.warmup_epochs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("pilot scheduler state 必须是 object")
        if float(state.get("lr")) != self.lr or float(state.get("second_lr")) != self.second_lr:
            raise ValueError("pilot scheduler lr contract 不一致")
        if _strict_int(state.get("warmup_epochs"), name="scheduler.warmup_epochs", positive=True) != self.warmup_epochs:
            raise ValueError("pilot scheduler warmup contract 不一致")
        self.epoch = _strict_int(state.get("epoch"), name="scheduler.epoch")
        value = self.lr if self.epoch < self.warmup_epochs else self.second_lr
        self._set_lr(value)


class _IndexedBaseDataset(Dataset):
    """Expose deterministic multi-dataset order plus explicit window metadata."""

    def __init__(self, base: dataset_module.MultiDatasetPathlineDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.base._order is None:
            raise RuntimeError("先调用 base.set_epoch(epoch)")
        order = tuple(self.base._order[int(index)])
        if len(order) != 4:
            raise ValueError(f"multi dataset order 必须是 (si,py,px,frame)，实际 {order!r}")
        si, py, px, frame = (_strict_int(value, name="sampling_order") for value in order)
        (dummy, pathlines), labels, seeds = self.base.sample_at(si, py, px, frame)
        return {
            "dummy": np.asarray(dummy, dtype=np.float32),
            "pathlines": np.asarray(pathlines, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.float32),
            "seeds": np.asarray(seeds, dtype=np.float64),
            "window": self.base.window_metadata(si, frame),
            "dataset_index": si,
            "frame": frame,
        }


class _IndexedW1PDataset(_IndexedBaseDataset):
    def __init__(
        self,
        base: dataset_module.MultiDatasetPathlineDataset,
        target_states: Mapping[str, Any],
    ) -> None:
        super().__init__(base)
        self.target_states = target_states

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        si = int(item["dataset_index"])
        frame = int(item["frame"])
        store = self.base.stores[si]
        target = self.target_states[store.dataset_name]
        seeds = item["seeds"]
        rows, cols = extractor.nearest_cell(
            seeds[:, 0], seeds[:, 1], store._xdim, store._ydim
        )
        rows = np.clip(rows, 0, store.Y - 1)
        cols = np.clip(cols, 0, store.X - 1)
        state = np.asarray(target[frame, rows, cols], dtype=np.int8)
        known = state >= 0
        item["labels"] = (state == 1).astype(np.float32)
        item["labels"][~known] = 0.0
        item["label_mask"] = known
        item["unknown_mask"] = ~known
        item["solid_mask"] = np.asarray(store._mask2d[rows, cols], dtype=bool)
        return item


def _collate_indexed_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("pilot DataLoader 收到空 batch")
    return {
        "dummy": np.concatenate([item["dummy"] for item in items], axis=0),
        "pathlines": np.stack([item["pathlines"] for item in items], axis=0),
        "labels": np.stack([item["labels"] for item in items], axis=0),
        "seeds": np.stack([item["seeds"] for item in items], axis=0),
        "windows": [copy.deepcopy(dict(item["window"])) for item in items],
        "dataset_indices": [int(item["dataset_index"]) for item in items],
        "frames": [int(item["frame"]) for item in items],
        **({
            "label_mask": np.stack([item["label_mask"] for item in items], axis=0),
            "unknown_mask": np.stack([item["unknown_mask"] for item in items], axis=0),
            "solid_mask": np.stack([item["solid_mask"] for item in items], axis=0),
        } if "label_mask" in items[0] else {}),
    }


def _collate_haller_items(items: list[w1_h.W1HBatch]) -> w1_h.W1HBatch:
    return w1_h.collate_w1_h_batches(items)


class RealPilotData:
    """Shared real-data stores, train streams, artifact projection, and models."""

    def __init__(
        self,
        *,
        weak_root: pathlib.Path,
        haller_root: pathlib.Path,
        model_config: Mapping[str, Any],
        manifest_summary: Mapping[str, Any],
        batch_size: int,
        samples_per_epoch: int,
        data_workers: int,
        eval_samples_per_dataset: int,
        eval_batch_size: int,
        seed: int,
        device: str,
        ramp_up_epochs: int,
        parallel_devices: Sequence[str] | None = None,
    ) -> None:
        self.weak_root = weak_root
        self.haller_root = haller_root
        self.model_config = copy.deepcopy(dict(model_config))
        self.manifest_summary = copy.deepcopy(dict(manifest_summary))
        self.batch_size = _strict_int(batch_size, name="batch_size", positive=True)
        self.samples_per_epoch = _strict_int(
            samples_per_epoch, name="samples_per_epoch", positive=True
        )
        self.data_workers = _strict_int(data_workers, name="data_workers")
        self.eval_samples_per_dataset = _strict_int(
            eval_samples_per_dataset, name="eval_samples_per_dataset", positive=True
        )
        self.eval_batch_size = _strict_int(
            eval_batch_size, name="eval_batch_size", positive=True
        )
        self.seed = _strict_int(seed, name="seed")
        self.device = str(device)
        self.parallel_devices = tuple(
            str(item).strip()
            for item in (parallel_devices if parallel_devices is not None else (self.device,))
            if str(item).strip()
        ) or (self.device,)
        if self.parallel_devices[0] != self.device:
            raise ValueError(
                "RealPilotData.parallel_devices 的 primary 必须等于 device："
                f"device={self.device!r} parallel_devices={self.parallel_devices!r}"
            )
        self.ramp_up_epochs = _strict_int(
            ramp_up_epochs, name="ramp_up_epochs", positive=True
        )
        self.t_win = int(manifest_summary["t_win"])
        self.window_step = int(manifest_summary["window_step"])
        data_config = self.model_config.get("data", {})
        self.data_config = dict(data_config) if isinstance(data_config, Mapping) else {}
        self.train_dataset = self._make_dataset("train", "train", self.samples_per_epoch)
        target_paths = {
            name: weak_root / "w1_p_targets" / name / "anchor_state.npy"
            for name in DATASETS
        }
        self.target_states = {
            name: np.load(path, mmap_mode="r", allow_pickle=False)
            for name, path in target_paths.items()
        }
        train_roots = {
            name: haller_root / TRAIN_SOURCE / name
            for name in DATASETS
        }
        self.train_haller_dataset = w1_h.HallerAnchorPathlineDataset(
            self.train_dataset,
            train_roots,
            sampling_source=contract.LABEL_SOURCE_LEGACY_P85,
            anchor_hash=self.global_anchor_hash,
        )
        self.anchor_metadata = self._build_anchor_metadata()
        target_metadata = {
            "source": "w1_p_train",
            "label_source": contract.LABEL_SOURCE_LOCAL_P90_P60,
            "datasets": copy.deepcopy(self.manifest_summary["w1_p_targets"]),
            "legacy_p85_used": False,
        }
        self.target_metadata = target_metadata
        self._eval_datasets: dict[str, dataset_module.MultiDatasetPathlineDataset] = {}
        self._eval_specs: dict[str, dict[str, list[tuple[int, int, int, int]]]] = {}
        self._eval_samples: dict[tuple[str, str], dict[str, Any]] = {}
        self._artifact_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    @property
    def global_anchor_hash(self) -> str:
        hashes = self.manifest_summary["haller_manifest_hashes"][TRAIN_SOURCE]
        return _json_hash({
            "source": TRAIN_SOURCE,
            "datasets": list(DATASETS),
            "manifest_hashes": {name: hashes[name] for name in DATASETS},
        })

    def _data_value(self, name: str, default: Any) -> Any:
        return self.data_config.get(name, default)

    def _make_dataset(
        self, split: str, consumer: str, samples_per_epoch: int
    ) -> dataset_module.MultiDatasetPathlineDataset:
        return dataset_module.MultiDatasetPathlineDataset(
            [self.weak_root / "datasets" / name / "dataset" for name in DATASETS],
            split=split,
            patch_size=(32, 32),
            stride=(16, 16),
            t_win=self.t_win,
            window_step=self.window_step,
            samples_per_epoch=samples_per_epoch,
            positive_fraction=float(self._data_value("positive_fraction", 0.5)),
            t_scale=float(self._data_value("t_scale", 0.25)),
            seed=self.seed,
            groups=(8, 8),
            delta_frac=float(self._data_value("delta_frac", 0.05)),
            L=int(self._data_value("L", 16)),
            n_substeps=int(self._data_value("n_substeps", 4)),
            consumer=consumer,
            label_source=contract.LABEL_SOURCE_LEGACY_P85,
        )

    def _build_anchor_metadata(self) -> dict[str, Any]:
        first_name = DATASETS[0]
        first_path = self.haller_root / TRAIN_SOURCE / first_name / "frame0"
        loaded = haller_anchors.load_haller_artifact(
            first_path, expected_source=TRAIN_SOURCE
        )
        metadata = copy.deepcopy(dict(loaded["metadata"]))
        metadata["anchor_hash"] = self.global_anchor_hash
        metadata["manifest_hashes"] = copy.deepcopy(
            self.manifest_summary["haller_manifest_hashes"][TRAIN_SOURCE]
        )
        return metadata

    def _loader(self, source: Dataset, collate_fn: Any) -> DataLoader:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "drop_last": False,
            "num_workers": self.data_workers,
            "collate_fn": collate_fn,
            "pin_memory": self.device.startswith("cuda"),
        }
        return DataLoader(source, **kwargs)

    def train_batches(self, mode: str):
        mode = contract.canonical_mode(mode)
        if mode in {contract.MODE_B0, contract.MODE_B1, contract.MODE_W1_P}:
            indexed = (
                _IndexedW1PDataset(self.train_dataset, self.target_states)
                if mode == contract.MODE_W1_P
                else _IndexedBaseDataset(self.train_dataset)
            )

            def stream(epoch: int):
                self.train_dataset.set_epoch(epoch)
                loader = self._loader(indexed, _collate_indexed_items)
                adapter = getattr(self, f"_adapter_{mode.replace('-', '_')}", None)
                for raw in loader:
                    provenance = {
                        "windows": copy.deepcopy(raw["windows"]),
                        "sampling": {"source": contract.LABEL_SOURCE_LEGACY_P85},
                    }
                    if mode == contract.MODE_W1_P:
                        yield w1_p.build_w1_p_batch(
                            raw["pathlines"], raw["labels"], raw["label_mask"],
                            raw["unknown_mask"], raw["solid_mask"],
                            sampling_source=contract.LABEL_SOURCE_LEGACY_P85,
                            split_name="train", provenance=provenance,
                            dummy_field=raw["dummy"],
                        )
                    else:
                        if adapter is None:
                            raise RuntimeError(
                                f"{mode} train stream 尚未绑定 channel adapter"
                            )
                        model_pathlines = adapter.adapt(
                            raw["pathlines"], input_schema=contract.FEATURE_SCHEMA_7
                        )
                        yield contract.WeakSupervisionBatch(
                            pathlines=model_pathlines,
                            labels=raw["labels"],
                            label_source=contract.LABEL_SOURCE_LEGACY_P85,
                            split_name="train",
                            feature_schema=contract.feature_schema_for_mode(mode),
                            label_mask=np.ones_like(raw["labels"], dtype=bool),
                            unknown_mask=np.zeros_like(raw["labels"], dtype=bool),
                            sampling_source=contract.LABEL_SOURCE_LEGACY_P85,
                            provenance=provenance,
                            mode=mode,
                            input_schema=contract.FEATURE_SCHEMA_7,
                        )

            return stream

        if mode not in {contract.MODE_W1_H, contract.MODE_W2, contract.MODE_W3}:
            raise ValueError(f"unsupported real pilot mode={mode}")

        def haller_stream(epoch: int):
            self.train_haller_dataset.set_epoch(epoch)
            loader = self._loader(self.train_haller_dataset, _collate_haller_items)
            for batch in loader:
                if mode == contract.MODE_W1_H:
                    yield batch
                elif mode == contract.MODE_W2:
                    yield w2.build_w2_batch_from_w1_h(batch)
                else:
                    yield w3.build_w3_batch_from_w1_h(batch)

        return haller_stream

    def bind_adapter(self, mode: str, adapter: contract.ChannelSelectingAdapter) -> None:
        mode = contract.canonical_mode(mode)
        setattr(self, f"_adapter_{mode.replace('-', '_')}", adapter)

    def _eval_dataset(self, split: str, consumer: str) -> dataset_module.MultiDatasetPathlineDataset:
        if split not in self._eval_datasets:
            self._eval_datasets[split] = self._make_dataset(split, consumer, 1)
        return self._eval_datasets[split]

    def _specs(self, split: str) -> dict[str, list[tuple[int, int, int, int]]]:
        if split in self._eval_specs:
            return self._eval_specs[split]
        consumer = "calibration" if split == "calibration" else "evaluation"
        eval_dataset = self._eval_dataset(split, consumer)
        result: dict[str, list[tuple[int, int, int, int]]] = {}
        for si, store in enumerate(eval_dataset.stores):
            starts = dataset_module.window_starts(
                store.split_i0, store.split_i1, self.t_win, self.window_step,
                dataset_name=store.dataset_name, split_name=split, T=store.T,
            )
            patches = list(store._usable_patches)
            if len(starts) == 0 or not patches:
                raise ValueError(f"{split}/{store.dataset_name} 没有完整可采样 window/patch")
            total = len(starts) * len(patches)
            count = min(self.eval_samples_per_dataset, total)
            flat_indices = np.linspace(0, total - 1, count, dtype=np.int64)
            specs = []
            for flat in flat_indices:
                patch_index = int(flat % len(patches))
                frame_index = int(starts[int(flat // len(patches))])
                py, px = patches[patch_index]
                specs.append((si, int(py), int(px), frame_index))
            result[store.dataset_name] = specs
        self._eval_specs[split] = result
        return result

    def _samples(self, split: str, name: str) -> dict[str, Any]:
        key = (split, name)
        if key in self._eval_samples:
            return self._eval_samples[key]
        eval_dataset = self._eval_dataset(
            split, "calibration" if split == "calibration" else "evaluation"
        )
        store = next(store for store in eval_dataset.stores if store.dataset_name == name)
        specs = self._specs(split)[name]
        raw = []
        seeds = []
        windows = []
        frames = []
        for si, py, px, frame in specs:
            (dummy, pathlines), _sampling_labels, sample_seeds = eval_dataset.sample_at(
                si, py, px, frame
            )
            raw.append(np.asarray(pathlines, dtype=np.float32))
            seeds.append(np.asarray(sample_seeds, dtype=np.float64))
            windows.append(eval_dataset.window_metadata(si, frame))
            frames.append(frame)
        value = {
            "raw": np.stack(raw, axis=0),
            "seeds": np.stack(seeds, axis=0),
            "windows": windows,
            "frames": frames,
            "store": store,
        }
        self._eval_samples[key] = value
        return value

    @contextmanager
    def _seeded_model_rng(self, seed: int):
        devices: list[int] = []
        if self.device.startswith("cuda"):
            devices = sorted({
                int(parsed.index)
                for parsed in (torch.device(item) for item in self.parallel_devices)
                if parsed.type == "cuda" and parsed.index is not None
            })
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            yield

    def _predict_once(
        self,
        model: torch.nn.Module,
        mode: str,
        raw: np.ndarray,
        *,
        seed: int,
        consumer: str,
    ) -> np.ndarray:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device(self.device)
        pathlines = torch.as_tensor(raw, dtype=torch.float32, device=device)
        dummy = torch.zeros((pathlines.shape[0], 1, 1, 1), device=device)
        was_training = bool(model.training)
        model.eval()
        try:
            with self._seeded_model_rng(seed), torch.no_grad():
                if isinstance(model, contract.ChannelSelectingAdapter):
                    prediction = model(
                        (dummy, pathlines),
                        input_schema=contract.FEATURE_SCHEMA_7,
                        consumer=consumer,
                    )
                elif isinstance(model, w3.TrajectoryEmbeddingAdapter):
                    prediction = model((dummy, pathlines), consumer=consumer)
                else:
                    prediction = model((dummy, pathlines))
        finally:
            model.train(was_training)
        result = prediction.detach().float().cpu().numpy()
        if result.shape != (raw.shape[0], raw.shape[2]):
            raise ValueError(
                f"{mode} prediction shape={result.shape} 与 expected={(raw.shape[0], raw.shape[2])} 不一致"
            )
        if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
            raise ValueError(f"{mode} prediction 必须是有限 [0,1] probability")
        return result

    def _predict_dataset(
        self,
        model: torch.nn.Module,
        mode: str,
        split: str,
        name: str,
        *,
        stochastic_model: torch.nn.Module | None = None,
        views: int = 1,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        sample = self._samples(split, name)
        raw = sample["raw"]
        predictions = []
        for start in range(0, raw.shape[0], self.eval_batch_size):
            chunk = raw[start:start + self.eval_batch_size]
            model_for_view = model if stochastic_model is None else stochastic_model
            chunk_views = [
                self._predict_once(
                    model_for_view, mode, chunk,
                    seed=self.seed + 100_003 * (start + 1) + 10_007 * view,
                    consumer="calibration" if split == "calibration" else "evaluation",
                )
                for view in range(views)
            ]
            predictions.append(np.stack(chunk_views, axis=0))
        stacked = np.concatenate(predictions, axis=1)
        if views == 1:
            return stacked[0], None
        return stacked.mean(axis=0), stacked.var(axis=0)

    def _artifact(
        self, source: str, name: str, frame: int
    ) -> dict[str, Any]:
        key = (source, name, int(frame))
        if key not in self._artifact_cache:
            path = self.haller_root / source / name / f"frame{int(frame)}"
            self._artifact_cache[key] = haller_anchors.load_haller_artifact(
                path, expected_source=source
            )
        return self._artifact_cache[key]

    def _project_haller(
        self, split: str, source: str, name: str
    ) -> dict[str, Any]:
        sample = self._samples(split, name)
        store = sample["store"]
        labels = []
        known = []
        unknown = []
        solid = []
        invalid = []
        frame_metadata: dict[int, Mapping[str, Any]] = {}
        for frame, seeds in zip(sample["frames"], sample["seeds"]):
            artifact = self._artifact(source, name, frame)
            metadata = artifact["metadata"]
            frame_metadata[int(frame)] = metadata
            rows, cols = extractor.nearest_cell(
                seeds[:, 0], seeds[:, 1], store._xdim, store._ydim
            )
            rows = np.clip(rows, 0, store.Y - 1)
            cols = np.clip(cols, 0, store.X - 1)
            state = np.asarray(artifact["anchor_state"][rows, cols], dtype=np.int8)
            frame_invalid = not bool(metadata.get("valid", False))
            row_invalid = np.full(state.shape, frame_invalid, dtype=bool)
            row_solid = np.asarray(artifact["solid_mask"][rows, cols], dtype=bool)
            row_known = (state >= 0) & ~row_invalid
            row_unknown = ~row_known & ~row_invalid
            row_labels = state.astype(np.float64)
            row_labels[row_invalid] = -1.0
            labels.append(row_labels)
            known.append(row_known)
            unknown.append(row_unknown)
            solid.append(row_solid & ~row_invalid)
            invalid.append(row_invalid)
        first = dict(next(iter(frame_metadata.values())))
        ordered_frames = sorted(frame_metadata)
        invalid_frames = [
            frame for frame in ordered_frames
            if not bool(frame_metadata[frame].get("valid", False))
        ]
        provenance = {
            "source": source,
            "label_source": source,
            "algorithm_version": first["algorithm_version"],
            "parameter_hash": first["parameter_hash"],
            "input_hash": _hash_strings(
                [str(frame_metadata[frame]["input_hash"]) for frame in ordered_frames]
            ),
            "mask_hash": first["mask_hash"],
            "parameters": copy.deepcopy(first["parameters"]),
            "failure_count": int(sum(
                int(frame_metadata[frame]["failure_count"]) for frame in ordered_frames
            )),
            "coverage": copy.deepcopy(first["coverage"]),
            "literature": copy.deepcopy(first["literature"]),
            "split_name": split,
            "window": {
                "t_win": self.t_win,
                "window_step": self.window_step,
                "split_name": split,
            },
            "manifest_hash": self.manifest_summary["haller_manifest_hashes"][source][name],
            "frame_count": len(ordered_frames),
            "invalid_frame_count": len(invalid_frames),
            "frame_artifacts": [
                {
                    "frame_index": frame,
                    "input_hash": frame_metadata[frame]["input_hash"],
                    "artifact_array_hashes": copy.deepcopy(
                        frame_metadata[frame]["artifact_array_hashes"]
                    ),
                    "failure_count": int(frame_metadata[frame]["failure_count"]),
                    "valid": bool(frame_metadata[frame]["valid"]),
                }
                for frame in ordered_frames
            ],
        }
        return {
            "labels": np.stack(labels, axis=0),
            "known": np.stack(known, axis=0),
            "unknown": np.stack(unknown, axis=0),
            "solid": np.stack(solid, axis=0),
            "invalid": np.stack(invalid, axis=0),
            "frames": np.asarray(sample["frames"], dtype=np.int64),
            "provenance": provenance,
        }

    def calibration_records(self, mode: str, trainer: Any):
        mode = contract.canonical_mode(mode)
        records = []
        for name in DATASETS:
            projected = self._project_haller(
                "calibration", CALIBRATION_SOURCE, name
            )
            model = trainer.student
            stochastic = trainer.teacher if mode == contract.MODE_W2 else None
            prediction, variance = self._predict_dataset(
                model, mode, "calibration", name,
                stochastic_model=stochastic, views=3 if mode == contract.MODE_W2 else 1,
            )
            if mode == contract.MODE_W2:
                records.append(w2.W2CalibrationRecord(
                    dataset_name=name,
                    mean_probability=prediction,
                    predictive_variance=variance,
                    labels=projected["labels"],
                    known_mask=projected["known"],
                    split_name="calibration",
                    label_source=CALIBRATION_SOURCE,
                    provenance=projected["provenance"],
                ))
            else:
                records.append(evaluation_report.CalibrationPredictionRecord(
                    dataset_name=name,
                    prediction=prediction,
                    labels=projected["labels"],
                    known_mask=projected["known"],
                    split_name="calibration",
                    label_source=CALIBRATION_SOURCE,
                    provenance=projected["provenance"],
                ))
        return records

    def test_records(
        self, mode: str, trainer: Any, *, variance_gate: float | None
    ) -> list[evaluation_report.TestEvaluationRecord]:
        mode = contract.canonical_mode(mode)
        records = []
        for name in DATASETS:
            projected = self._project_haller("test", TEST_SOURCE, name)
            stochastic = trainer.teacher if mode in {contract.MODE_W2, contract.MODE_W3} else None
            prediction, variance = self._predict_dataset(
                trainer.student, mode, "test", name,
                stochastic_model=stochastic,
                views=3 if stochastic is not None else 1,
            )
            # Keep one evaluation record per sampled frame.  This preserves
            # valid frames in the denominator when another frame in the same
            # dataset has an invalid Haller artifact.
            provenance = projected["provenance"]
            sampled_frames = projected["frames"]
            for frame in sorted(set(int(value) for value in sampled_frames)):
                indices = np.flatnonzero(sampled_frames == frame)
                frame_metadata = next(
                    item for item in provenance["frame_artifacts"]
                    if int(item["frame_index"]) == frame
                )
                frame_provenance = copy.deepcopy(provenance)
                frame_provenance["frame_count"] = 1
                frame_provenance["invalid_frame_count"] = int(
                    not bool(frame_metadata["valid"])
                )
                frame_provenance["failure_count"] = int(
                    frame_metadata["failure_count"]
                )
                frame_provenance["frame_artifacts"] = [copy.deepcopy(frame_metadata)]
                frame_prediction = prediction[indices]
                frame_variance = None if variance is None else variance[indices]
                records.append(evaluation_report.TestEvaluationRecord(
                    dataset_name=name,
                    prediction=frame_prediction,
                    labels=projected["labels"][indices],
                    known_mask=projected["known"][indices],
                    unknown_mask=projected["unknown"][indices],
                    solid_mask=projected["solid"][indices],
                    invalid_mask=projected["invalid"][indices],
                    predictive_variance=frame_variance,
                    split_name="test",
                    label_source=TEST_SOURCE,
                    provenance=frame_provenance,
                    frame_count=1,
                    invalid_frame_count=frame_provenance["invalid_frame_count"],
                    failure_count=frame_provenance["failure_count"],
                    sample_count=int(frame_prediction.size),
                    frame_valid=bool(frame_metadata["valid"]),
                ))
        return records

    def _model_config_for_mode(self, mode: str) -> dict[str, Any]:
        config = copy.deepcopy(self.model_config)
        encoder = config.get("model", {}).get("encoder_args")
        if not isinstance(encoder, dict):
            raise ValueError("pilot model config 缺少 model.encoder_args object")
        encoder["in_channels"] = contract.feature_schema_for_mode(mode).channel_count
        return config

    def _build_trainer(
        self,
        mode: str,
        *,
        pilot_config: e2e.PilotConfig,
        selection: e2e.PilotSelection | None = None,
        model_device: str | None = None,
        parallel_devices: Sequence[str] | None = None,
        lr: float,
        second_lr: float,
        warmup_epochs: int,
        weight_decay: float,
        grad_clip: float,
    ) -> tuple[Any, contract.ChannelSelectingAdapter | w3.TrajectoryEmbeddingAdapter]:
        mode = contract.canonical_mode(mode)
        model_device = self.device if model_device is None else str(model_device)
        parallel_devices = (
            self.parallel_devices
            if parallel_devices is None
            else tuple(str(item) for item in parallel_devices)
        )
        config = self._model_config_for_mode(mode)
        adapter = contract.build_model_for_mode(config, mode)
        _configure_adapter_devices(
            adapter,
            mode=mode,
            device=model_device,
            parallel_devices=parallel_devices,
            wrap_data_parallel=not (
                mode == contract.MODE_W1_P and len(parallel_devices) > 1
            ),
        )
        optimizer_parameters = list(adapter.parameters())
        if mode == contract.MODE_W3:
            model: Any = w3.TrajectoryEmbeddingAdapter(adapter)
            projection = w3.TrajectoryProjectionHead(model.embedding_dim, projection_dim=64)
            optimizer_parameters += list(projection.parameters())
        else:
            model = adapter
            projection = None
        optimizer = torch.optim.AdamW(
            optimizer_parameters, lr=float(lr), weight_decay=float(weight_decay)
        )
        scheduler = TwoStepPilotScheduler(
            optimizer, lr=lr, second_lr=second_lr, warmup_epochs=warmup_epochs
        )
        common = {
            "sampling_source": contract.LABEL_SOURCE_LEGACY_P85,
            "scheduler": scheduler,
            "seed": self.seed,
            "grad_clip_norm": grad_clip,
        }
        if mode in {contract.MODE_B0, contract.MODE_B1}:
            from vendor.DeepUtils.loss import build_criterion_from_cfg

            criterion = contract.ModeAwareLoss(
                mode,
                build_criterion_from_cfg(config["model"]["criterion_args"]),
            )
            trainer = e2e.ContractTrainer(
                model, optimizer, criterion, mode=mode, **common
            )
            self.bind_adapter(mode, adapter)
        elif mode == contract.MODE_W1_P:
            trainer = w1_p.W1PTrainer(
                model, optimizer, target_metadata=self.target_metadata,
                config=w1_p.W1PConfig(ramp_up_epochs=self.ramp_up_epochs),
                teacher_device=(
                    parallel_devices[1] if len(parallel_devices) > 1 else model_device
                ),
                **common
            )
        elif mode == contract.MODE_W1_H:
            trainer = w1_h.W1HTrainer(
                model, optimizer, anchor_hash=self.global_anchor_hash,
                anchor_metadata=self.anchor_metadata,
                config=w1_h.W1HConfig(ramp_up_epochs=self.ramp_up_epochs), **common
            )
        elif mode == contract.MODE_W2:
            trainer = w2.W2Trainer(
                model, optimizer, anchor_hash=self.global_anchor_hash,
                anchor_metadata=self.anchor_metadata,
                config=w2.W2Config(
                    ramp_up_epochs=self.ramp_up_epochs,
                    variance_gate=PILOT_TRAINING_VARIANCE_GATE,
                ), **common
            )
        elif mode == contract.MODE_W3:
            trainer = w3.W3Trainer(
                model, optimizer, projection_head=projection,
                anchor_hash=self.global_anchor_hash,
                anchor_metadata=self.anchor_metadata,
                config=w3.W3Config(
                    ramp_up_epochs=self.ramp_up_epochs,
                    variance_gate=PILOT_TRAINING_VARIANCE_GATE,
                ), **common
            )
        else:
            raise ValueError(f"unsupported real pilot mode={mode}")
        return trainer, model

    def build_methods(
        self,
        *,
        pilot_config: e2e.PilotConfig,
        lr: float,
        second_lr: float,
        warmup_epochs: int,
        weight_decay: float,
        grad_clip: float,
        method_devices: Mapping[str, str] | None = None,
        method_parallel_devices: Mapping[str, Sequence[str]] | None = None,
        modes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        method_devices = dict(method_devices or {})
        method_parallel_devices = dict(method_parallel_devices or {})
        requested_modes = {
            contract.canonical_mode(mode)
            for mode in (
                (
                    contract.MODE_B0, contract.MODE_B1, contract.MODE_W1_P,
                    contract.MODE_W1_H, contract.MODE_W2, contract.MODE_W3,
                )
                if modes is None else modes
            )
        }

        def resources(mode: str) -> tuple[str, tuple[str, ...]]:
            canonical = contract.canonical_mode(mode)
            device = str(method_devices.get(canonical, self.device))
            group = tuple(
                str(item)
                for item in method_parallel_devices.get(
                    canonical, (device,) if canonical == contract.MODE_W3
                    else self.parallel_devices
                )
            )
            if not group:
                group = (device,)
            if group[0] != device:
                raise ValueError(
                    f"{canonical} method resource primary 不一致："
                    f"device={device!r} group={group!r}"
                )
            return device, group

        methods: dict[str, Any] = {}
        for mode in (
            contract.MODE_B0, contract.MODE_B1, contract.MODE_W1_P,
            contract.MODE_W1_H, contract.MODE_W2,
        ):
            if mode not in requested_modes:
                continue
            mode_device, mode_parallel_devices = resources(mode)
            trainer, model = self._build_trainer(
                mode, pilot_config=pilot_config, lr=lr, second_lr=second_lr,
                warmup_epochs=warmup_epochs, weight_decay=weight_decay,
                grad_clip=grad_clip, model_device=mode_device,
                parallel_devices=mode_parallel_devices,
            )
            methods[mode] = e2e.PilotMethod.from_trainer(
                mode=mode,
                trainer=trainer,
                train_batches=self.train_batches(mode),
                calibration_records=lambda trainer=trainer, mode=mode: self.calibration_records(mode, trainer),
                evaluate_test=lambda prediction_threshold, variance_gate, trainer=trainer, mode=mode: self.test_records(
                    mode, trainer, variance_gate=variance_gate
                ),
                dataset_config=pilot_config.dataset_config,
                split_config=pilot_config.split_config,
                sampling_config=pilot_config.sampling_config,
                device=mode_device,
                max_steps=pilot_config.max_steps,
                role="diagnostic" if mode == contract.MODE_B1 else "headline_candidate",
            )

        if contract.MODE_W3 not in requested_modes:
            return methods

        def w3_factory(selection: e2e.PilotSelection):
            mode_device, mode_parallel_devices = resources(contract.MODE_W3)
            trainer, model = self._build_trainer(
                contract.MODE_W3, pilot_config=pilot_config, selection=selection,
                lr=lr, second_lr=second_lr, warmup_epochs=warmup_epochs,
                weight_decay=weight_decay, grad_clip=grad_clip,
                model_device=mode_device,
                parallel_devices=mode_parallel_devices,
            )
            return e2e.PilotMethod.from_trainer(
                mode=contract.MODE_W3,
                trainer=trainer,
                train_batches=self.train_batches(contract.MODE_W3),
                calibration_records=lambda trainer=trainer: self.calibration_records(
                    contract.MODE_W3, trainer
                ),
                evaluate_test=lambda prediction_threshold, variance_gate, trainer=trainer: self.test_records(
                    contract.MODE_W3, trainer, variance_gate=variance_gate
                ),
                dataset_config=pilot_config.dataset_config,
                split_config=pilot_config.split_config,
                sampling_config=pilot_config.sampling_config,
                device=mode_device,
                max_steps=pilot_config.max_steps,
            )

        methods[contract.MODE_W3] = w3_factory
        return methods


def _load_model_config(path: pathlib.Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("model"), dict):
        raise ValueError(f"model config 必须包含 object model：{path}")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ticket-09 real weak-supervision pilot")
    parser.add_argument(
        "--weak-root", default="outputs/weak_supervision_numbacs_native"
    )
    parser.add_argument(
        "--haller-root", default="outputs/haller_artifacts_numbacs_native"
    )
    parser.add_argument("--model-config", default="config/pathline_transformer_b1.yaml")
    parser.add_argument("--output-dir", default="outputs/pilot_numbacs_native")
    parser.add_argument("--epochs", type=int, default=e2e.PILOT_EPOCHS)
    parser.add_argument("--seed", type=int, default=e2e.PILOT_SEED)
    parser.add_argument("--samples-per-epoch", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--data-workers", type=int, default=8)
    parser.add_argument("--eval-samples-per-dataset", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--parallel-devices",
        default=None,
        help=(
            "独立前置 method 的 CUDA device 列表；按相邻两张卡组成一个 "
            "multi-GPU worker group，例如 cuda:0,cuda:1,cuda:2,cuda:3；"
            "W3 仍 single GPU"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "从 outputs/<pilot>/training_progress 下最近的完整 10-epoch "
            "checkpoint 继续；所有 run/data/device/training 参数必须完全一致"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.epochs != e2e.PILOT_EPOCHS:
        raise ValueError(
            f"ticket 09 real pilot 必须固定 {e2e.PILOT_EPOCHS} epochs，"
            f"实际收到 {args.epochs}"
        )
    if args.seed != e2e.PILOT_SEED:
        raise ValueError(
            f"ticket 09 real pilot 必须固定 seed={e2e.PILOT_SEED}，"
            f"实际收到 {args.seed}"
        )
    if args.max_steps is not None:
        raise ValueError(
            "ticket 09 real pilot 不允许 --max-steps；smoke 必须使用独立测试入口"
        )
    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"请求 pilot device={device} 但当前 CUDA 不可用")
    parallel_devices = None
    if args.parallel_devices is not None:
        parallel_devices = tuple(
            item.strip() for item in str(args.parallel_devices).split(",") if item.strip()
        )
        if len(parallel_devices) < 4 or len(parallel_devices) % 2 != 0:
            raise ValueError(
                "--parallel-devices 需要偶数个、至少四个 CUDA device；"
                "每个独立 worker 使用两张卡以保持 ticket 参数不变"
            )
        if not device.startswith("cuda"):
            raise ValueError("使用 --parallel-devices 时 --device 必须指定 W3 的 cuda:N")
    if args.resume and parallel_devices is None:
        raise ValueError("--resume 当前只支持 multi-GPU parallel pilot")
    model_config_path = pathlib.Path(args.model_config)
    model_config = _load_model_config(model_config_path)
    manifest_summary = validate_prepared_artifacts(
        args.weak_root, args.haller_root,
        t_win=model_config.get("data", {}).get("t_win", None),
        window_step=model_config.get("data", {}).get("window_step", None),
    )
    split_ranges = {
        name: list(manifest_summary["weak_contracts"][name]["split_ranges"]["train"])
        for name in DATASETS
    }
    dataset_config = {
        "datasets": list(DATASETS),
        "split_mode": dataset_module.WEAK_SUPERVISION_SPLIT_MODE,
        "weak_root": str(pathlib.Path(args.weak_root).resolve()),
        "artifact_manifest_hash": manifest_summary["manifest_hash"],
    }
    split_config = {
        "split_name": "train",
        "split_ranges": split_ranges,
        "t_win": manifest_summary["t_win"],
        "window_step": manifest_summary["window_step"],
    }
    sampling_config = {
        "source": contract.LABEL_SOURCE_LEGACY_P85,
        "t_win": manifest_summary["t_win"],
        "window_step": manifest_summary["window_step"],
        "samples_per_epoch": args.samples_per_epoch,
        "batch_size": args.batch_size,
    }
    pilot_config = e2e.PilotConfig(
        dataset_config=dataset_config,
        split_config=split_config,
        sampling_config=sampling_config,
        epochs=args.epochs,
        seed=args.seed,
        device=device,
        max_steps=args.max_steps,
        variance_candidates=PILOT_VARIANCE_CANDIDATES,
    )
    train_config = model_config.get("train", {})
    if not isinstance(train_config, Mapping):
        raise ValueError("model config train 必须是 object")
    if parallel_devices is not None:
        from parallel_pilot import run_parallel_pilot

        report = run_parallel_pilot(
            weak_root=pathlib.Path(args.weak_root),
            haller_root=pathlib.Path(args.haller_root),
            model_config=model_config,
            manifest_summary=manifest_summary,
            pilot_config=pilot_config,
            output_dir=pathlib.Path(args.output_dir),
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch,
            data_workers=args.data_workers,
            eval_samples_per_dataset=args.eval_samples_per_dataset,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            ramp_up_epochs=pilot_config.ramp_up_epochs,
            train_config={
                "lr": float(train_config.get("lr", 1e-4)),
                "second_lr": float(train_config.get("second_lr", 5e-6)),
                "warmup_epochs": int(train_config.get("warmup_epochs", 60)),
                "weight_decay": float(train_config.get("weight_decay", 1e-6)),
                "grad_clip": float(train_config.get("grad_clip", 1.0)),
            },
            devices=parallel_devices,
            resume=bool(args.resume),
        )
    else:
        data = RealPilotData(
            weak_root=pathlib.Path(args.weak_root),
            haller_root=pathlib.Path(args.haller_root),
            model_config=model_config,
            manifest_summary=manifest_summary,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch,
            data_workers=args.data_workers,
            eval_samples_per_dataset=args.eval_samples_per_dataset,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            device=device,
            ramp_up_epochs=pilot_config.ramp_up_epochs,
        )
        methods = data.build_methods(
            pilot_config=pilot_config,
            lr=float(train_config.get("lr", 1e-4)),
            second_lr=float(train_config.get("second_lr", 5e-6)),
            warmup_epochs=int(train_config.get("warmup_epochs", 60)),
            weight_decay=float(train_config.get("weight_decay", 1e-6)),
            grad_clip=float(train_config.get("grad_clip", 1.0)),
        )
        report = e2e.run_pilot(
            methods, config=pilot_config, output_dir=pathlib.Path(args.output_dir)
        )
    report["pilot"].update({
        "artifact_manifest": manifest_summary["manifest_path"],
        "artifact_manifest_hash": manifest_summary["manifest_hash"],
        "haller_root": str(pathlib.Path(args.haller_root).resolve()),
        "weak_root": str(pathlib.Path(args.weak_root).resolve()),
        "w1_p_target_root": str(pathlib.Path(args.weak_root).resolve() / "w1_p_targets"),
        "haller_manifest_hashes": manifest_summary["haller_manifest_hashes"],
        "haller_frame_counts": manifest_summary["haller_frame_counts"],
        "evaluation_samples_per_dataset": args.eval_samples_per_dataset,
        "evaluation_batch_size": args.eval_batch_size,
        "data_workers": args.data_workers,
        "test_artifacts_manifest_checked_before_pilot": True,
        "test_artifacts_integrity_checked_before_pilot": False,
        "haller_backend": manifest_summary["haller_backend"],
        "haller_backend_scope": "cpu_only_cuda_out_of_scope",
    })
    output_report = pathlib.Path(args.output_dir) / "pilot_report.json"
    output_report.write_text(
        json.dumps(e2e._jsonable(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(output_report),
        "artifact_manifest": manifest_summary["manifest_path"],
        "artifact_manifest_hash": manifest_summary["manifest_hash"],
        "checkpoint_paths": {
            mode: report["methods"][mode]["checkpoint"]
            for mode in e2e.PILOT_METHOD_ORDER
        },
        "selection": report["selection"],
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
