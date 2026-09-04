"""Prepare the explicit WS-9 weak-supervision input artifacts.

本工具只负责把已有的六个历史 memmap 输入转换成 WS-9 所需的显式
``weak_supervision`` 三段 metadata、W1-P train target，以及按 source 分离的
Haller train/calibration/test artifacts。它不改变 Haller 提取算法，也不把
test Haller 标签传给训练路径；所有工程参数和文献状态都由现有
``haller_anchors`` 实现写入并保留 ``pending_verification``。

历史 ``outputs/datasets/<dataset>/dataset`` 只作为原始速度/IVD/p85 sampling
输入读取。新产物写入独立 namespace，避免覆盖阶段 0 baseline。
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import pathlib
import re
import sys
from typing import Any, Mapping

import numpy as np

import dataset
import haller_anchors
import w1_p
import weak_supervision_contract as contract


VALID_DATASETS = (
    "boussinesq",
    "cylinder2d",
    "doublegyre2d",
    "fourcenters2d",
    "jungtelziemniak2d",
    "pipedcylinder2d",
)

DEFAULT_T_WIN = 24
DEFAULT_WINDOW_STEP = 4
DEFAULT_PATCH_SIZE = (32, 32)
DEFAULT_STRIDE = (16, 16)

SOURCE_RANGES = {
    haller_anchors.SOURCE_TRAIN: "train",
    haller_anchors.SOURCE_CALIBRATION: "calibration",
    haller_anchors.SOURCE_TEST: "test",
}


def _normalize_contour_mode(contour_mode: str) -> str:
    """Validate the contour implementation as part of the artifact contract."""
    normalized = str(contour_mode).lower()
    if normalized not in haller_anchors.VALID_CONTOUR_MODES:
        allowed = ", ".join(sorted(haller_anchors.VALID_CONTOUR_MODES))
        raise ValueError(
            f"未知 Haller contour_mode={contour_mode!r}；必须是 {allowed}"
        )
    return normalized


def _resolve_cpu_haller_backend(backend: str, device: Any = None) -> dict[str, Any]:
    """Resolve the artifact backend under the explicit WS-9 CPU boundary.

    Ticket 09 has CPU NumPy, the local CPU ``fast_haller`` backend, and the
    upstream CPU ``numbacs.rotcohvrt`` backend; none of these paths uses CUDA.
    Rejecting every other backend before input
    loading keeps this boundary fail-loud and prevents a partial run from
    being mislabeled as CPU output.
    """
    requested = str(backend).lower()
    if requested not in haller_anchors.VALID_BACKENDS:
        raise ValueError(
            "WS-9 当前只支持 CPU NumPy、CPU fast_haller 或 CPU numbacs artifact；"
            "CUDA Haller backend 未实现，"
            f"不接受 backend={backend!r}"
        )
    if device is not None and str(device).lower() != "cpu":
        raise ValueError(
            "WS-9 CPU Haller artifact 只接受 device=None 或 'cpu'，"
            f"实际收到 {device!r}"
        )
    metadata = haller_anchors.resolve_haller_backend(
        requested, "cpu"
    )
    if metadata["resolved"] == haller_anchors.BACKEND_NUMBACS:
        # Formal artifacts must never be produced through the historical
        # compatibility runtime.  Fail before loading datasets or creating
        # any output directories.
        from numbacs_haller import ensure_native_runtime

        ensure_native_runtime()
    return metadata


def _json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _assert_non_overlapping_roots(
    input_root: pathlib.Path,
    output_root: pathlib.Path,
    haller_root: pathlib.Path,
) -> None:
    """Reject input/output namespace overlap before any artifact is read."""
    roots = {
        "input_root": input_root.resolve(strict=False),
        "output_root": output_root.resolve(strict=False),
        "haller_root": haller_root.resolve(strict=False),
    }
    items = tuple(roots.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1:]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError(
                    "artifact roots 不能重叠，拒绝覆盖输入或其他 namespace："
                    f"{left_name}={left} {right_name}={right}"
                )


def _array_hash(value: Any, *, chunk_frames: int = 8) -> str:
    """按 dtype/shape/内容 hash 大数组，避免一次性复制整个 memmap。"""
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    if array.ndim == 0:
        digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()
    for start in range(0, int(array.shape[0]), chunk_frames):
        chunk = np.ascontiguousarray(array[start : start + chunk_frames])
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def _read_json(
    path: pathlib.Path, *, allow_legacy_escape_repair: bool = False
) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 JSON artifact：{path}")
    text = path.read_text(encoding="utf-8")
    repaired = False
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        if not allow_legacy_escape_repair:
            raise
        # Old Windows-path metadata was emitted with some single backslashes
        # (for example ``\Desktop``). Repair only invalid JSON escapes, never
        # overwrite the old file, and expose the repair to the caller.
        repaired_text = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
        value = json.loads(repaired_text)
        repaired = True
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact 必须是 object：{path}")
    return value, repaired


def _legacy_paths(input_root: pathlib.Path, name: str) -> pathlib.Path:
    root = input_root / name / "dataset"
    if not root.is_dir():
        raise FileNotFoundError(
            f"缺少 dataset={name!r} 的历史输入目录：{root}"
        )
    return root


def _load_legacy_dataset(input_root: pathlib.Path, name: str) -> dict[str, Any]:
    """加载旧 frac memmap，并拒绝把已有 weak metadata 当作输入回退。"""
    root = _legacy_paths(input_root, name)
    meta, metadata_repaired = _read_json(
        root / dataset.FN_META, allow_legacy_escape_repair=True
    )
    metadata_name = meta.get("dataset_name")
    if metadata_name not in (None, name):
        raise ValueError(
            f"历史 dataset metadata name 不匹配：expected={name!r} "
            f"actual={metadata_name!r}"
        )
    if meta.get("split_mode") != "frac":
        raise ValueError(
            f"artifact preparer 只接受已确认的历史 frac 输入，"
            f"dataset={name!r} split_mode={meta.get('split_mode')!r}"
        )
    shape = tuple(int(value) for value in meta.get("shape", ()))
    if len(shape) != 3:
        raise ValueError(f"历史 dataset={name!r} shape 非三维：{shape!r}")
    paths = {
        "u": root / dataset.FN_U,
        "v": root / dataset.FN_V,
        "ivd": root / dataset.FN_IVD,
        "labels": root / dataset.FN_LABEL,
        "mask": root / dataset.FN_MASK,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"dataset={name!r} 缺少历史输入文件：{missing!r}"
        )
    arrays = {
        key: np.load(path, mmap_mode="r" if key != "mask" else None, allow_pickle=False)
        for key, path in paths.items()
    }
    for key in ("u", "v", "ivd", "labels"):
        if tuple(arrays[key].shape) != shape:
            raise ValueError(
                f"dataset={name!r} {key} shape={arrays[key].shape} != {shape}"
            )
    if tuple(arrays["mask"].shape) != shape[1:]:
        raise ValueError(
            f"dataset={name!r} mask shape={arrays['mask'].shape} != {shape[1:]}"
        )
    coordinates = {
        key: np.asarray(meta[key], dtype=np.float64)
        for key in ("xdim", "ydim", "tdim")
    }
    if tuple(map(len, (coordinates["tdim"], coordinates["ydim"], coordinates["xdim"]))) != shape:
        raise ValueError(
            f"dataset={name!r} 坐标长度与 shape 不一致："
            f"{tuple(map(len, (coordinates['tdim'], coordinates['ydim'], coordinates['xdim'])))}"
            f" != {shape}"
        )
    return {
        "name": name,
        "root": root,
        "meta": meta,
        "legacy_metadata_escape_repaired": metadata_repaired,
        "legacy_metadata_name_inferred": metadata_name is None,
        "shape": shape,
        **arrays,
        **coordinates,
    }


def _weak_dataset_is_ready(
    root: pathlib.Path,
    *,
    name: str,
    shape: tuple[int, int, int],
    t_win: int,
    window_step: int,
    expected_input_hashes: Mapping[str, str],
) -> bool:
    """只在 metadata 和所有 memmap 都存在且核心 contract 完整时复用。"""
    meta_path = root / dataset.FN_META
    if not meta_path.exists():
        return False
    meta, _ = _read_json(meta_path)
    required_files = (
        dataset.FN_U,
        dataset.FN_V,
        dataset.FN_IVD,
        dataset.FN_LABEL,
        dataset.FN_MASK,
    )
    if any(not (root / filename).exists() for filename in required_files):
        return False
    if meta.get("input_array_hashes") != dict(expected_input_hashes):
        return False
    expected_output_hashes = meta.get("array_hashes")
    if not isinstance(expected_output_hashes, Mapping):
        return False
    try:
        actual_output_hashes = {
            "u": _array_hash(np.load(root / dataset.FN_U, mmap_mode="r", allow_pickle=False)),
            "v": _array_hash(np.load(root / dataset.FN_V, mmap_mode="r", allow_pickle=False)),
            "ivd": _array_hash(np.load(root / dataset.FN_IVD, mmap_mode="r", allow_pickle=False)),
            "labels": _array_hash(np.load(root / dataset.FN_LABEL, mmap_mode="r", allow_pickle=False)),
            "mask": _array_hash(np.asarray(np.load(root / dataset.FN_MASK, allow_pickle=False), dtype=np.uint8)),
        }
    except (OSError, ValueError, TypeError):
        return False
    if dict(expected_output_hashes) != actual_output_hashes:
        return False
    return (
        meta.get("dataset_name") == name
        and meta.get("shape") == list(shape)
        and meta.get("split_mode") == dataset.WEAK_SUPERVISION_SPLIT_MODE
        and meta.get("label_source") == contract.LABEL_SOURCE_LEGACY_P85
        and meta.get("sampling_source") == contract.LABEL_SOURCE_LEGACY_P85
        and meta.get("loss_label_source") == contract.LABEL_SOURCE_LEGACY_P85
        and meta.get("normalization_source") == dataset.NORMALIZATION_SOURCE
        and meta.get("normalization_frozen") is True
        and meta.get("window", {}).get("t_win") == t_win
        and meta.get("window", {}).get("window_step") == window_step
    )


def prepare_weak_dataset(
    source: Mapping[str, Any],
    output_root: pathlib.Path,
    *,
    t_win: int,
    window_step: int,
) -> dict[str, Any]:
    """从历史输入构造新 weak split，保留 p85 仅为 sampling/diagnostic source。"""
    name = str(source["name"])
    out = output_root / "datasets" / name / "dataset"
    out.parent.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(value) for value in source["shape"])
    input_hashes = {
        "u": _array_hash(np.asarray(source["u"], dtype=np.float32)),
        "v": _array_hash(np.asarray(source["v"], dtype=np.float32)),
        "ivd": _array_hash(np.asarray(source["ivd"], dtype=np.float32)),
        "labels": _array_hash(np.asarray(source["labels"], dtype=np.uint8)),
        "mask": _array_hash(np.asarray(source["mask"], dtype=np.uint8)),
        "xdim": _array_hash(np.asarray(source["xdim"], dtype=np.float64)),
        "ydim": _array_hash(np.asarray(source["ydim"], dtype=np.float64)),
        "tdim": _array_hash(np.asarray(source["tdim"], dtype=np.float64)),
    }
    if _weak_dataset_is_ready(
        out, name=name, shape=shape, t_win=t_win, window_step=window_step,
        expected_input_hashes=input_hashes,
    ):
        meta, _ = _read_json(out / dataset.FN_META)
        return meta
    meta = dataset.prepare_dataset(
        out_dir=out,
        u=source["u"],
        v=source["v"],
        xdim=source["xdim"],
        ydim=source["ydim"],
        tdim=source["tdim"],
        mask=source["mask"],
        ivd=source["ivd"],
        labels=source["labels"],
        split_mode=dataset.WEAK_SUPERVISION_SPLIT_MODE,
        dataset_name=name,
        label_source=contract.LABEL_SOURCE_LEGACY_P85,
        sampling_source=contract.LABEL_SOURCE_LEGACY_P85,
        loss_label_source=contract.LABEL_SOURCE_LEGACY_P85,
        patch_size=DEFAULT_PATCH_SIZE,
        stride=DEFAULT_STRIDE,
        t_win=t_win,
        window_step=window_step,
    )
    output_arrays = {
        "u": np.load(out / dataset.FN_U, mmap_mode="r", allow_pickle=False),
        "v": np.load(out / dataset.FN_V, mmap_mode="r", allow_pickle=False),
        "ivd": np.load(out / dataset.FN_IVD, mmap_mode="r", allow_pickle=False),
        "labels": np.load(out / dataset.FN_LABEL, mmap_mode="r", allow_pickle=False),
        "mask": np.asarray(np.load(out / dataset.FN_MASK, allow_pickle=False), dtype=np.uint8),
    }
    meta = copy.deepcopy(meta)
    meta["input_array_hashes"] = input_hashes
    meta["input_hash"] = _json_hash(input_hashes)
    meta["input_mask_hash"] = input_hashes["mask"]
    meta["array_hashes"] = {
        key: _array_hash(value) for key, value in output_arrays.items()
    }
    (out / dataset.FN_META).write_text(
        json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if not _weak_dataset_is_ready(
        out, name=name, shape=shape, t_win=t_win, window_step=window_step,
        expected_input_hashes=input_hashes,
    ):
        raise RuntimeError(f"weak dataset prepare 后 contract 校验失败：{out}")
    return meta


def prepare_w1_p_target(
    weak_root: pathlib.Path,
    output_root: pathlib.Path,
    *,
    name: str,
) -> dict[str, Any]:
    """生成 train-only p90/p60/unknown target，不写 calibration/test labels。"""
    root = weak_root / "datasets" / name / "dataset"
    meta, _ = _read_json(root / dataset.FN_META)
    start, end = (int(value) for value in meta["split_ranges"]["train"])
    ivd = np.load(root / dataset.FN_IVD, mmap_mode="r", allow_pickle=False)
    solid = np.load(root / dataset.FN_MASK, allow_pickle=False).astype(bool)
    out = output_root / "w1_p_targets" / name
    input_ivd_hash = _array_hash(ivd)
    input_mask_hash = _array_hash(solid)
    target_files = {
        "anchor_state": (np.int8, ("anchor_state.npy",)),
        "labels": (np.float32, ("labels.npy",)),
        "label_mask": (np.uint8, ("label_mask.npy",)),
        "unknown_mask": (np.uint8, ("unknown_mask.npy",)),
    }
    target_meta_path = out / "target_meta.json"
    if target_meta_path.exists() and all(
        (out / filename).exists()
        for _, (_, filenames) in target_files.items()
        for filename in filenames
    ):
        try:
            existing, _ = _read_json(target_meta_path)
            identity_matches = (
                existing.get("artifact_type") == "w1_p_train_target"
                and existing.get("dataset_name") == name
                and existing.get("split_name") == "train"
                and existing.get("frame_range") == [start, end]
                and existing.get("input_dataset_contract_hash") == meta["contract_hash"]
                and existing.get("input_ivd_hash") == input_ivd_hash
                and existing.get("input_mask_hash") == input_mask_hash
                and existing.get("legacy_p85_used") is False
                and existing.get("calibration_test_frames_written") is False
            )
            arrays_match = identity_matches
            declared_target_hashes = existing.get("target_array_hashes")
            if not isinstance(declared_target_hashes, Mapping):
                arrays_match = False
            for key, (dtype, filenames) in target_files.items():
                if not arrays_match:
                    break
                array = np.load(out / filenames[0], mmap_mode="r", allow_pickle=False)
                arrays_match = (
                    tuple(array.shape) == tuple(ivd.shape)
                    and array.dtype == np.dtype(dtype)
                    and declared_target_hashes.get(key) == _array_hash(array)
                )
            if arrays_match:
                return existing
        except PermissionError:
            raise
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    thresholds = w1_p.compute_w1_p_thresholds(
        ivd,
        solid,
        train_frame_range=(start, end),
        dataset_name=name,
        split_name="train",
    )
    target = w1_p.build_w1_p_target_field(
        ivd,
        solid,
        thresholds,
        train_frame_range=(start, end),
        dataset_name=name,
        split_name="train",
    )
    out.mkdir(parents=True, exist_ok=True)
    metadata = copy.deepcopy(target.metadata)
    metadata.update({
        "artifact_type": "w1_p_train_target",
        "input_dataset_contract_hash": meta["contract_hash"],
        "input_ivd_hash": _array_hash(ivd),
        "input_mask_hash": _array_hash(solid),
        "legacy_p85_used": False,
        "calibration_test_frames_written": False,
        "source": "w1_p_train",
        "window": copy.deepcopy(meta.get("window", {})),
    })
    target_arrays = {
        "anchor_state": target.anchor_state.astype(np.int8),
        "labels": target.labels.astype(np.float32),
        "label_mask": target.label_mask.astype(np.uint8),
        "unknown_mask": target.unknown_mask.astype(np.uint8),
    }
    metadata["target_array_hashes"] = {
        key: _array_hash(array) for key, array in target_arrays.items()
    }
    np.save(out / "anchor_state.npy", target_arrays["anchor_state"])
    np.save(out / "labels.npy", target_arrays["labels"])
    np.save(out / "label_mask.npy", target_arrays["label_mask"])
    np.save(out / "unknown_mask.npy", target_arrays["unknown_mask"])
    (out / "target_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _slim_haller_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """去掉逐帧 contour 点的重复 debug payload，保留完整 contract metadata。"""
    slim = copy.deepcopy(dict(metadata))
    omitted = []
    for key in ("selected_contours", "contour_diagnostics"):
        if key in slim:
            slim.pop(key)
            omitted.append(key)
    if omitted:
        slim["storage"] = {
            "omitted_debug_fields": omitted,
            "arrays_and_contract_metadata_retained": True,
        }
    return slim


def _coverage_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合 frame coverage，fraction 字段由总计重新计算。"""
    count_fields = (
        "solid_cells", "fluid_cells", "positive_cells", "negative_cells",
        "unknown_cells", "total_unknown_cells_including_solid", "known_cells",
    )
    totals = {field: 0 for field in count_fields}
    for record in records:
        coverage = record["coverage"]
        for field in count_fields:
            totals[field] += int(coverage.get(field, 0))
    fluid = totals["fluid_cells"]
    denominator = float(fluid) if fluid else 1.0
    known = totals["known_cells"]
    totals.update({
        "positive_fraction_fluid": totals["positive_cells"] / denominator,
        "negative_fraction_fluid": totals["negative_cells"] / denominator,
        "unknown_fraction_fluid": totals["unknown_cells"] / denominator,
        "known_fraction_fluid": known / denominator,
    })
    return totals


def _manifest_for(
    *,
    name: str,
    source: str,
    split_name: str,
    frame_range: tuple[int, int],
    window: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"dataset={name} source={source} 没有 frame artifact")
    first = records[0]
    for record in records:
        if record["algorithm_version"] != first["algorithm_version"]:
            raise ValueError("同一 Haller manifest 的 algorithm_version 发生漂移")
        if record["parameter_hash"] != first["parameter_hash"]:
            raise ValueError("同一 Haller manifest 的 parameter_hash 发生漂移")
        if record["mask_hash"] != first["mask_hash"]:
            raise ValueError("同一 Haller manifest 的 mask_hash 发生漂移")
        for key in (
            "backend_requested",
            "resolved",
            "backend",
            "device",
            "backend_version",
            "compute_dtype",
            "backend_fallback_reason",
            "cuda_used",
            "contour_mode",
        ):
            if record[key] != first[key]:
                raise ValueError(f"同一 Haller manifest 的 {key} 发生漂移")
        if record["literature"].get("status") != "pending_verification":
            raise ValueError("Haller manifest literature 必须保持 pending_verification")
        if record.get("split_name") != split_name:
            raise ValueError("同一 Haller manifest 的 split_name 发生漂移")
        if record.get("window") != dict(window):
            raise ValueError("同一 Haller manifest 的 window contract 发生漂移")
    payload: dict[str, Any] = {
        "artifact_type": "haller_ivd_manifest",
        "dataset_name": name,
        "source": source,
        "split_name": split_name,
        "label_source": source,
        "window": copy.deepcopy(dict(window)),
        "algorithm_version": first["algorithm_version"],
        "parameters": copy.deepcopy(first["parameters"]),
        "parameter_hash": first["parameter_hash"],
        "backend_requested": first["backend_requested"],
        "resolved": first["resolved"],
        "backend": first["backend"],
        "device": first["device"],
        "backend_version": first["backend_version"],
        "compute_dtype": first["compute_dtype"],
        "backend_fallback_reason": first["backend_fallback_reason"],
        "cuda_used": first["cuda_used"],
        "contour_mode": first["contour_mode"],
        "input_hash": _hash_strings([record["input_hash"] for record in records]),
        "mask_hash": first["mask_hash"],
        "frame_range": list(frame_range),
        "frame_count": len(records),
        "valid_frame_count": sum(int(record["valid"]) for record in records),
        "invalid_frame_count": sum(int(not record["valid"]) for record in records),
        "failure_count": sum(int(record["failure_count"]) for record in records),
        "coverage": _coverage_totals(records),
        "literature": copy.deepcopy(first["literature"]),
        "legacy_p85_used": False,
        "fallback_used": None,
        "artifact_pattern": "frame{frame}",
        "frame_artifacts": [
            {
                "frame_index": int(record["frame_index"]),
                "relative_dir": f"frame{int(record['frame_index'])}",
                "algorithm_version": record["algorithm_version"],
                "parameter_hash": record["parameter_hash"],
                "input_hash": record["input_hash"],
                "mask_hash": record["mask_hash"],
                "artifact_array_hashes": copy.deepcopy(record["artifact_array_hashes"]),
                "failure_count": int(record["failure_count"]),
                "valid": bool(record["valid"]),
            }
            for record in records
        ],
    }
    payload["manifest_hash"] = _json_hash(payload)
    return payload


def _prepare_haller_frame(
    source_data: Mapping[str, Any],
    output_dir: pathlib.Path,
    *,
    source: str,
    frame: int,
    resume: bool,
    backend: str,
    device: Any,
    contour_mode: str,
    expected_backend: Mapping[str, Any],
    split_name: str,
    window: Mapping[str, Any],
) -> dict[str, Any]:
    """准备单帧并返回 manifest 所需的稳定记录。"""
    u = source_data["u"]
    v = source_data["v"]
    xdim = source_data["xdim"]
    ydim = source_data["ydim"]
    mask = source_data["mask"]
    frame_dir = output_dir / f"frame{frame}"
    metadata_path = frame_dir / "anchor_meta.json"
    expected_input_hash = haller_anchors._hash_named_arrays({
        "u": np.asarray(u[frame], dtype=np.float64),
        "v": np.asarray(v[frame], dtype=np.float64),
        "xdim": np.asarray(xdim, dtype=np.float64),
        "ydim": np.asarray(ydim, dtype=np.float64),
    })
    expected_mask_hash = haller_anchors._hash_array(
        np.asarray(mask, dtype=bool)
    )
    result: dict[str, Any]
    if resume and metadata_path.exists():
        try:
            loaded = haller_anchors.load_haller_artifact(
                frame_dir, expected_source=source
            )
            metadata = loaded["metadata"]
            if int(metadata.get("frame_index", -1)) != frame:
                raise ValueError("frame_index mismatch")
            for key, expected in expected_backend.items():
                if metadata.get(key) != expected:
                    raise ValueError(f"已有 Haller artifact 的 {key} 不匹配")
            if metadata.get("input_hash") != expected_input_hash:
                raise ValueError("已有 Haller artifact 的 input_hash 不匹配")
            if metadata.get("mask_hash") != expected_mask_hash:
                raise ValueError("已有 Haller artifact 的 mask_hash 不匹配")
            if metadata.get("split_name") != split_name:
                raise ValueError("已有 Haller artifact 的 split_name 不匹配")
            if metadata.get("window") != dict(window):
                raise ValueError("已有 Haller artifact 的 window contract 不匹配")
            result = loaded
        except PermissionError:
            raise
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            result = haller_anchors.extract_haller_anchors(
                u[frame], v[frame], xdim, ydim, mask,
                source=source,
                frame_index=frame,
                backend=backend,
                device=device,
                contour_mode=contour_mode,
            )
            result["metadata"] = _slim_haller_metadata(result["metadata"])
            result["metadata"].update({
                "split_name": split_name,
                "window": copy.deepcopy(dict(window)),
            })
            haller_anchors.save_haller_artifact(result, frame_dir, overwrite=True)
    else:
        result = haller_anchors.extract_haller_anchors(
            u[frame], v[frame], xdim, ydim, mask,
            source=source,
            frame_index=frame,
            backend=backend,
            device=device,
            contour_mode=contour_mode,
        )
        result["metadata"] = _slim_haller_metadata(result["metadata"])
        result["metadata"].update({
            "split_name": split_name,
            "window": copy.deepcopy(dict(window)),
        })
        haller_anchors.save_haller_artifact(result, frame_dir, overwrite=True)
    metadata = result["metadata"]
    return {
        "frame_index": frame,
        "algorithm_version": metadata["algorithm_version"],
        "parameters": metadata["parameters"],
        "parameter_hash": metadata["parameter_hash"],
        "backend_requested": metadata["backend_requested"],
        "resolved": metadata.get("resolved", metadata["backend"]),
        "backend": metadata["backend"],
        "device": metadata["device"],
        "backend_version": metadata["backend_version"],
        "compute_dtype": metadata["compute_dtype"],
        "backend_fallback_reason": metadata["backend_fallback_reason"],
        "cuda_used": bool(metadata["cuda_used"]),
        "contour_mode": metadata["contour_mode"],
        "input_hash": metadata["input_hash"],
        "mask_hash": metadata["mask_hash"],
        "split_name": metadata["split_name"],
        "window": copy.deepcopy(metadata["window"]),
        "failure_count": int(metadata["failure_count"]),
        "valid": bool(metadata["valid"]),
        "coverage": metadata["coverage"],
        "literature": metadata["literature"],
        "artifact_array_hashes": metadata["artifact_array_hashes"],
    }


_HALLER_WORKER_CONTEXT: tuple[
    Mapping[str, Any], pathlib.Path, str, bool, str, Any, str,
    Mapping[str, Any], str, Mapping[str, Any]
] | None = None


def _init_haller_worker(
    source_data: Mapping[str, Any],
    output_dir: pathlib.Path,
    source: str,
    resume: bool,
    backend: str,
    device: Any,
    contour_mode: str,
    expected_backend: Mapping[str, Any],
    split_name: str,
    window: Mapping[str, Any],
) -> None:
    """初始化 POSIX worker，避免每个 frame 重复传输 memmap context。"""
    global _HALLER_WORKER_CONTEXT
    _HALLER_WORKER_CONTEXT = (
        source_data,
        output_dir,
        source,
        resume,
        backend,
        device,
        contour_mode,
        expected_backend,
        split_name,
        window,
    )
    haller_anchors._haller_geometry_dependencies()


def _prepare_haller_frame_worker(frame: int) -> dict[str, Any]:
    """ProcessPoolExecutor 的可 pickle frame worker。"""
    if _HALLER_WORKER_CONTEXT is None:
        raise RuntimeError("Haller worker context 尚未初始化")
    source_data, output_dir, source, resume, backend, device, contour_mode, expected_backend, split_name, window = (
        _HALLER_WORKER_CONTEXT
    )
    return _prepare_haller_frame(
        source_data,
        output_dir,
        source=source,
        frame=frame,
        resume=resume,
        backend=backend,
        device=device,
        contour_mode=contour_mode,
        expected_backend=expected_backend,
        split_name=split_name,
        window=window,
    )


def prepare_haller_source(
    source_data: Mapping[str, Any],
    output_root: pathlib.Path,
    *,
    source: str,
    resume: bool = True,
    workers: int = 1,
    backend: str = haller_anchors.BACKEND_NUMPY,
    device: Any = None,
    contour_mode: str = haller_anchors.CONTOUR_MODE_OPTIMIZED,
) -> dict[str, Any]:
    """按 source 的显式 split 范围生成/恢复逐帧 Haller artifacts。"""
    workers = int(workers)
    if workers <= 0:
        raise ValueError(f"Haller workers 必须为正整数，实际 {workers}")
    if source not in SOURCE_RANGES:
        raise ValueError(f"未知 Haller source={source!r}")
    contour_mode = _normalize_contour_mode(contour_mode)
    backend_metadata = _resolve_cpu_haller_backend(backend, device)
    contour_mode = haller_anchors._contour_mode_for_backend(
        backend_metadata["resolved"], contour_mode
    )
    expected_haller_metadata = dict(backend_metadata)
    expected_haller_metadata["contour_mode"] = contour_mode
    name = str(source_data["name"])
    weak_meta, _ = _read_json(
        pathlib.Path(source_data["weak_root"]) / "meta.json"
    )
    split_name = SOURCE_RANGES[source]
    frame_range = tuple(int(value) for value in weak_meta["split_ranges"][split_name])
    window_metadata = weak_meta.get("window")
    if not isinstance(window_metadata, Mapping):
        raise ValueError(
            f"dataset={name} weak metadata 缺少显式 window contract，拒绝生成 Haller artifact"
        )
    t_win = int(window_metadata.get("t_win", 0))
    window_step = int(window_metadata.get("window_step", 0))
    if t_win <= 0 or window_step <= 0:
        raise ValueError(
            f"dataset={name} window contract 必须包含正 t_win/window_step"
        )
    window = {
        "split_name": split_name,
        "frame_range": list(frame_range),
        "t_win": t_win,
        "window_step": window_step,
        "complete_windows_only": True,
    }
    expected_haller_metadata["algorithm_version"] = haller_anchors.ALGORITHM_VERSION
    out = output_root / source / name
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "anchor_manifest.json"
    # Do not short-circuit on the top-level manifest alone.  Each frame is
    # loaded through ``load_haller_artifact`` below, so valid frames are
    # reused while missing/tampered frames are recomputed and the manifest is
    # regenerated deterministically.  This makes resume fail-safe without a
    # broad delete or a trust gap between manifest hashes and arrays.

    records: list[dict[str, Any]] = []
    total = frame_range[1] - frame_range[0]
    frame_indices = list(range(*frame_range))

    def prepare_frame(frame: int) -> dict[str, Any]:
        return _prepare_haller_frame(
            source_data,
            out,
            source=source,
            frame=frame,
            resume=resume,
            backend=backend,
            device=device,
            contour_mode=contour_mode,
            expected_backend=expected_haller_metadata,
            split_name=split_name,
            window=window,
        )

    # POSIX runs use processes because contour screening includes Python work
    # and each worker can call the official native NumbaCS implementation
    # independently. Windows uses threads so local unit tests do not need a
    # multiprocessing main guard. Ordered map results keep the manifest
    # deterministic in either case.
    if workers > 1:
        use_processes = os.name == "posix"
        if use_processes:
            mp_context = multiprocessing.get_context("fork")
            executor_context = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_context,
                initializer=_init_haller_worker,
                initargs=(
                    dict(source_data),
                    out,
                    source,
                    resume,
                    backend,
                    device,
                    contour_mode,
                    expected_haller_metadata,
                    split_name,
                    window,
                ),
            )
        else:
            executor_context = ThreadPoolExecutor(max_workers=workers)
        with executor_context as executor:
            frame_records = executor.map(
                _prepare_haller_frame_worker if use_processes else prepare_frame,
                frame_indices,
            )
            for ordinal, (frame, record) in enumerate(
                zip(frame_indices, frame_records), start=1
            ):
                records.append(record)
                if ordinal == 1 or ordinal == total or ordinal % 25 == 0:
                    print(
                        f"[haller] dataset={name} source={source} "
                        f"frame={frame} progress={ordinal}/{total}",
                        flush=True,
                    )
    else:
        for ordinal, frame in enumerate(frame_indices, start=1):
            records.append(prepare_frame(frame))
            if ordinal == 1 or ordinal == total or ordinal % 25 == 0:
                print(
                    f"[haller] dataset={name} source={source} "
                    f"frame={frame} progress={ordinal}/{total}",
                    flush=True,
                )
    manifest = _manifest_for(
        name=name,
        source=source,
        split_name=split_name,
        frame_range=frame_range,
        window=window,
        records=records,
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare_all(
    *,
    input_root: str | pathlib.Path,
    output_root: str | pathlib.Path,
    haller_root: str | pathlib.Path,
    datasets: tuple[str, ...] = VALID_DATASETS,
    t_win: int = DEFAULT_T_WIN,
    window_step: int = DEFAULT_WINDOW_STEP,
    resume: bool = True,
    haller_workers: int = 1,
    haller_backend: str = haller_anchors.BACKEND_NUMPY,
    haller_device: Any = None,
    haller_contour_mode: str = haller_anchors.CONTOUR_MODE_OPTIMIZED,
) -> dict[str, Any]:
    """准备六数据集全部 artifacts，并写一份总 manifest。"""
    input_path = pathlib.Path(input_root)
    output_path = pathlib.Path(output_root)
    haller_path = pathlib.Path(haller_root)
    _assert_non_overlapping_roots(input_path, output_path, haller_path)
    if tuple(datasets) != VALID_DATASETS:
        raise ValueError(
            f"WS-9 只允许六个有效 dataset 且顺序固定：{VALID_DATASETS!r}"
        )
    if int(t_win) <= 0 or int(window_step) <= 0:
        raise ValueError("t_win/window_step 必须为正整数")
    if int(haller_workers) <= 0:
        raise ValueError("haller_workers 必须为正整数")
    haller_backend_metadata = _resolve_cpu_haller_backend(
        haller_backend, haller_device
    )
    haller_contour_mode = _normalize_contour_mode(haller_contour_mode)
    haller_contour_mode = haller_anchors._contour_mode_for_backend(
        haller_backend_metadata["resolved"], haller_contour_mode
    )

    sources = {name: _load_legacy_dataset(input_path, name) for name in datasets}
    weak_metas: dict[str, Any] = {}
    target_metas: dict[str, Any] = {}
    for name, source in sources.items():
        weak_metas[name] = prepare_weak_dataset(
            source, output_path, t_win=t_win, window_step=window_step
        )
        target_metas[name] = prepare_w1_p_target(
            output_path, output_path, name=name
        )
        source["weak_root"] = output_path / "datasets" / name / "dataset"

    haller_manifests: dict[str, dict[str, Any]] = {
        haller_source: {}
        for haller_source in (
            haller_anchors.SOURCE_TRAIN,
            haller_anchors.SOURCE_CALIBRATION,
            haller_anchors.SOURCE_TEST,
        )
    }
    for haller_source in haller_manifests:
        for name, source in sources.items():
            haller_manifests[haller_source][name] = prepare_haller_source(
                source,
                haller_path,
                source=haller_source,
                resume=resume,
                workers=haller_workers,
                backend=haller_backend,
                device=haller_device,
                contour_mode=haller_contour_mode,
            )
    total_manifest = {
        "artifact_type": "weak_supervision_pilot_inputs",
        "generation_version": "ws9-artifact-preparer-v3",
        "datasets": list(datasets),
        "split_mode": dataset.WEAK_SUPERVISION_SPLIT_MODE,
        "t_win": int(t_win),
        "window_step": int(window_step),
        "patch_size": list(DEFAULT_PATCH_SIZE),
        "stride": list(DEFAULT_STRIDE),
        "weak_dataset_contracts": {
            name: {
                "contract_hash": weak_metas[name]["contract_hash"],
                "split_ranges": weak_metas[name]["split_ranges"],
                "generation_hash": weak_metas[name]["generation_hash"],
                "input_hash": weak_metas[name]["input_hash"],
                "input_array_hashes": copy.deepcopy(
                    weak_metas[name]["input_array_hashes"]
                ),
                "array_hashes": copy.deepcopy(weak_metas[name]["array_hashes"]),
            }
            for name in datasets
        },
        "w1_p_targets": {
            name: {
                "positive_threshold": target_metas[name]["positive_threshold"],
                "negative_threshold": target_metas[name]["negative_threshold"],
                "input_dataset_contract_hash": target_metas[name]["input_dataset_contract_hash"],
                "input_ivd_hash": target_metas[name]["input_ivd_hash"],
                "input_mask_hash": target_metas[name]["input_mask_hash"],
                "target_array_hashes": copy.deepcopy(
                    target_metas[name]["target_array_hashes"]
                ),
            }
            for name in datasets
        },
        "haller_manifests": haller_manifests,
        "haller_backend": haller_backend_metadata,
        "haller_contour_mode": haller_contour_mode,
        "haller_literature_status": "pending_verification",
    }
    total_manifest["manifest_hash"] = _json_hash(total_manifest)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "artifact_manifest.json").write_text(
        json.dumps(total_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return total_manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WS-9 weak-supervision artifacts")
    parser.add_argument("--input-root", default="outputs/datasets")
    parser.add_argument(
        "--output-root", default="outputs/weak_supervision_numbacs_native"
    )
    parser.add_argument(
        "--haller-root", default="outputs/haller_artifacts_numbacs_native"
    )
    parser.add_argument("--t-win", type=int, default=DEFAULT_T_WIN)
    parser.add_argument("--window-step", type=int, default=DEFAULT_WINDOW_STEP)
    parser.add_argument(
        "--haller-workers",
        type=int,
        default=1,
        help="并行处理独立 frame 的 worker 数；默认 1，manifest 仍按 frame 顺序生成",
    )
    parser.add_argument(
        "--haller-backend",
        choices=sorted(haller_anchors.VALID_BACKENDS),
        default=haller_anchors.BACKEND_NUMPY,
        help="Haller 数值后端；当前均为 CPU，CUDA Haller backend 未实现",
    )
    parser.add_argument(
        "--haller-device",
        default=None,
        help="CPU device，只接受 cpu；不指定时使用 cpu",
    )
    parser.add_argument(
        "--haller-contour-mode",
        choices=sorted(haller_anchors.VALID_CONTOUR_MODES),
        default=haller_anchors.CONTOUR_MODE_OPTIMIZED,
        help="contour 实现；numbacs 使用 upstream rotcohvrt，reference 仅供 NumPy oracle",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不复用已有同 source/frame artifact；仍只写独立输出 namespace",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = prepare_all(
        input_root=args.input_root,
        output_root=args.output_root,
        haller_root=args.haller_root,
        t_win=args.t_win,
        window_step=args.window_step,
        resume=not args.no_resume,
        haller_workers=args.haller_workers,
        haller_backend=args.haller_backend,
        haller_device=args.haller_device,
        haller_contour_mode=args.haller_contour_mode,
    )
    print(json.dumps({
        "manifest": str(pathlib.Path(args.output_root) / "artifact_manifest.json"),
        "manifest_hash": manifest["manifest_hash"],
        "datasets": manifest["datasets"],
        "haller_literature_status": manifest["haller_literature_status"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
