"""W1-P P95/P80/unknown 弱监督基础设施。

本模块实现 W1-P 的 train-only P95/P80 三态监督、masked/pseudo/
consistency loss、EMA teacher、Ticket 03 target/calibration artifact
consumer 和可恢复训练 seam。历史 ``local_p90_p60`` 只保留在公共
contract 的历史登记中，不能作为当前 W1-P 的 formal 或 sampling source。
W1-H 的 Haller physics anchor、W2 uncertainty gate 与 W3 contrastive head
由各自模块负责。
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import experiment_protocol as protocol
import standard_ivd
import weak_labels
import weak_supervision_contract as contract


W1P_GENERATION_VERSION = "w1-p-standard-ivd-p95-p80-v1"
W1P_LABEL_SOURCE = contract.LABEL_SOURCE_P95P80
W1P_DEFAULT_POSITIVE_PERCENTILE = 95.0
W1P_DEFAULT_NEGATIVE_PERCENTILE = 80.0
W1P_DEFAULT_PSEUDO_HIGH = 0.90
W1P_DEFAULT_PSEUDO_LOW = 0.10
W1P_DEFAULT_EMA_DECAY = 0.99
W1P_DEFAULT_RAMP_UP_EPOCHS = 12
W1P_DEFAULT_PSEUDO_WEIGHT = 1.0
W1P_DEFAULT_CONSISTENCY_WEIGHT = 1.0


def _as_ivd_3d(ivd: Any) -> np.ndarray:
    """将 prepared standard IVD 统一为 (T, Y, X) 并拒绝非有限值。"""
    array = np.asarray(ivd, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"W1-P IVD 必须是 (T,Y,X)，实际 shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("W1-P IVD 含有非有限值")
    return array


def _as_solid_mask(mask: Any, shape: tuple[int, int, int]) -> np.ndarray:
    """将 2D/3D 固体掩膜规格化到 IVD shape；None 表示无固体。"""
    if mask is None:
        return np.zeros(shape, dtype=bool)
    array = np.asarray(mask, dtype=bool)
    if array.ndim == 2:
        if tuple(array.shape) != shape[1:]:
            raise ValueError(
                f"W1-P 2D solid mask shape={array.shape} 与 IVD={shape} 不匹配"
            )
        return np.broadcast_to(array, shape)
    if array.ndim == 3 and tuple(array.shape) == shape:
        return array
    raise ValueError(
        f"W1-P solid mask 必须是 (Y,X) 或 (T,Y,X)，实际 shape={array.shape}"
    )


def _as_support_mask(
    mask: Any,
    shape: tuple[int, int, int],
    *,
    name: str,
    default: bool,
) -> np.ndarray:
    """将 valid-fluid/invalid mask 规格化到 IVD shape。"""
    if mask is None:
        return np.full(shape, default, dtype=bool)
    array = np.asarray(mask, dtype=bool)
    if array.ndim == 2:
        if tuple(array.shape) != shape[1:]:
            raise ValueError(
                f"W1-P 2D {name} shape={array.shape} 与 IVD={shape} 不匹配"
            )
        return np.broadcast_to(array, shape)
    if array.ndim == 3 and tuple(array.shape) == shape:
        return array
    raise ValueError(
        f"W1-P {name} 必须是 (Y,X) 或 (T,Y,X)，实际 shape={array.shape}"
    )


def _resolve_support_masks(
    solid_mask: Any,
    valid_fluid_mask: Any,
    invalid_mask: Any,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 solid、有效 fluid support 与非 solid invalid support。"""
    solid = _as_solid_mask(solid_mask, shape)
    declared_valid = _as_support_mask(
        valid_fluid_mask, shape, name="valid_fluid_mask", default=True
    )
    explicit_invalid = _as_support_mask(
        invalid_mask, shape, name="invalid_mask", default=False
    )
    valid_support = (~solid) & declared_valid & ~explicit_invalid
    invalid_support = (~solid) & (~declared_valid | explicit_invalid)
    return solid, valid_support, invalid_support


def _frame_range(
    value: Any,
    total_frames: int,
    *,
    dataset_name: str,
    name: str = "train_frame_range",
) -> tuple[int, int]:
    """校验半开 frame range，拒绝隐式截断和非整数边界。"""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            f"数据集 {dataset_name!r} 的 {name} 必须是二元半开区间，实际 {value!r}"
        )
    bounds = []
    for bound in value:
        if isinstance(bound, (bool, np.bool_)):
            raise ValueError(f"数据集 {dataset_name!r} 的 {name} 边界必须是整数")
        try:
            integer = int(bound)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"数据集 {dataset_name!r} 的 {name} 边界必须是整数"
            ) from exc
        if integer != bound:
            raise ValueError(
                f"数据集 {dataset_name!r} 的 {name} 边界必须是整数，实际 {value!r}"
            )
        bounds.append(integer)
    start, end = bounds
    if not (0 <= start < end <= total_frames):
        raise ValueError(
            f"数据集 {dataset_name!r} 的 {name}={tuple(bounds)!r} "
            f"超出 [0, {total_frames}) 或为空"
        )
    return start, end


@dataclass(frozen=True)
class W1PThresholds:
    """只由 train frame valid-fluid cells 计算的 P80/P95 阈值。

    ``source_id`` and the optional identity fields make a threshold object
    carry the same source boundary as the Ticket 03 artifacts.  The optional
    fields keep the small synthetic/public numerical seam usable without
    requiring a full six-dataset manifest.
    """

    negative: float
    positive: float
    train_frame_range: tuple[int, int]
    dataset_name: str = "dataset"
    source_split: str = "train"
    source_id: str = W1P_LABEL_SOURCE
    standard_ivd_source_identity: Mapping[str, Any] | None = None
    protocol_manifest_hash: str | None = None
    threshold_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.source_split != "train":
            raise ValueError(
                "W1-P threshold source_split 必须是 train，不能消费 calibration/test"
            )
        negative = float(self.negative)
        positive = float(self.positive)
        if not np.isfinite(negative) or not np.isfinite(positive):
            raise ValueError("W1-P P80/P95 threshold 必须是有限数")
        if not negative < positive:
            raise ValueError(
                f"W1-P threshold 必须满足 P80 < P95，实际 "
                f"P80={negative} P95={positive}"
            )
        if self.source_id != W1P_LABEL_SOURCE:
            raise ValueError(
                "W1-P threshold source_id 必须是当前 Ticket 03 P95/P80 source，"
                f"实际 {self.source_id!r}"
            )
        if self.standard_ivd_source_identity is not None:
            if dict(self.standard_ivd_source_identity) != dict(
                standard_ivd.SOURCE_IDENTITY
            ):
                raise ValueError(
                    "W1-P threshold standard IVD source identity 与 canonical source 不一致"
                )
            object.__setattr__(
                self,
                "standard_ivd_source_identity",
                copy.deepcopy(dict(self.standard_ivd_source_identity)),
            )
        if self.protocol_manifest_hash is not None:
            if (not isinstance(self.protocol_manifest_hash, str)
                    or not self.protocol_manifest_hash.strip()):
                raise ValueError("W1-P protocol_manifest_hash 必须是非空字符串或 None")
            object.__setattr__(
                self, "protocol_manifest_hash", self.protocol_manifest_hash.strip()
            )
        if self.threshold_provenance is not None:
            if not isinstance(self.threshold_provenance, Mapping):
                raise ValueError("W1-P threshold_provenance 必须是 object 或 None")
            object.__setattr__(
                self,
                "threshold_provenance",
                copy.deepcopy(dict(self.threshold_provenance)),
            )
        if (not isinstance(self.train_frame_range, (tuple, list))
                or len(self.train_frame_range) != 2):
            raise ValueError("W1-P train_frame_range 必须是二元半开区间")
        bounds = []
        for bound in self.train_frame_range:
            if isinstance(bound, (bool, np.bool_)):
                raise ValueError("W1-P train_frame_range 边界必须是整数")
            try:
                integer = int(bound)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "W1-P train_frame_range 边界必须是整数"
                ) from exc
            if integer != bound:
                raise ValueError("W1-P train_frame_range 边界必须是整数")
            bounds.append(integer)
        if not (0 <= bounds[0] < bounds[1]):
            raise ValueError(
                f"W1-P train_frame_range={tuple(bounds)!r} 必须是非空半开区间"
            )
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "positive", positive)
        object.__setattr__(
            self,
            "train_frame_range",
            (bounds[0], bounds[1]),
        )

    def as_dict(self) -> dict[str, Any]:
        """返回可写入 batch/checkpoint metadata 的稳定阈值记录。"""
        result: dict[str, Any] = {
            "negative_percentile": W1P_DEFAULT_NEGATIVE_PERCENTILE,
            "positive_percentile": W1P_DEFAULT_POSITIVE_PERCENTILE,
            "p80": float(self.negative),
            "p95": float(self.positive),
            "negative": float(self.negative),
            "positive": float(self.positive),
            "train_frame_range": list(self.train_frame_range),
            "dataset_name": self.dataset_name,
            "source_split": self.source_split,
            "source_id": self.source_id,
            "artifact_identity": protocol.build_p95p80_artifact_identity(
                protocol.P95P80_PERCENTILE_SOURCE_ID
            ),
            "standard_ivd_source_identity": copy.deepcopy(
                dict(self.standard_ivd_source_identity
                     if self.standard_ivd_source_identity is not None
                     else standard_ivd.SOURCE_IDENTITY)
            ),
            "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
        }
        if self.protocol_manifest_hash is not None:
            result["protocol_manifest_hash"] = self.protocol_manifest_hash
        if self.threshold_provenance is not None:
            result["threshold_provenance"] = copy.deepcopy(
                dict(self.threshold_provenance)
            )
        return result


def compute_w1_p_thresholds(
    ivd: Any,
    solid_mask: Any,
    *,
    train_frame_range: tuple[int, int] | list[int],
    dataset_name: str = "dataset",
    split_name: str = "train",
    valid_fluid_mask: Any = None,
    invalid_mask: Any = None,
    protocol_manifest_hash: str | None = None,
    threshold_provenance: Mapping[str, Any] | None = None,
) -> W1PThresholds:
    """从显式 train frame range 的 valid-fluid standard IVD 计算 P80/P95。

    train_frame_range 是必需参数；函数不会把完整 IVD（尤其 test）
    静默当作统计来源。
    """
    if split_name != "train":
        raise ValueError(
            f"W1-P thresholds 只能从 split=train 计算，实际 split={split_name!r}"
        )
    array = _as_ivd_3d(ivd)
    frame_range = _frame_range(
        train_frame_range, array.shape[0], dataset_name=dataset_name
    )
    _solid, valid_support, _invalid_support = _resolve_support_masks(
        solid_mask, valid_fluid_mask, invalid_mask, array.shape
    )
    start, end = frame_range
    values = array[start:end][valid_support[start:end]]
    if values.size == 0:
        raise ValueError(
            f"数据集 {dataset_name!r} 的 train valid-fluid cells 为空，"
            "无法计算 W1-P P80/P95"
        )
    negative = float(np.percentile(values, W1P_DEFAULT_NEGATIVE_PERCENTILE))
    positive = float(np.percentile(values, W1P_DEFAULT_POSITIVE_PERCENTILE))
    return W1PThresholds(
        negative=negative,
        positive=positive,
        train_frame_range=frame_range,
        dataset_name=str(dataset_name),
        standard_ivd_source_identity=standard_ivd.SOURCE_IDENTITY,
        protocol_manifest_hash=protocol_manifest_hash,
        threshold_provenance=threshold_provenance,
    )


@dataclass
class W1PTargetField:
    """W1-P 三态 target field：1=positive、0=negative、-1=unknown。"""

    anchor_state: np.ndarray
    solid_mask: np.ndarray
    metadata: dict[str, Any]
    valid_fluid_mask: np.ndarray | None = None
    invalid_mask: np.ndarray | None = None

    @property
    def labels(self) -> np.ndarray:
        """将 positive/negative/unknown state 转为 BCE target（unknown=0 占位）。"""
        return (self.anchor_state == 1).astype(np.float32)

    @property
    def label_mask(self) -> np.ndarray:
        """只标记已知 positive/negative，solid 和 unknown 均为 False。"""
        return self.anchor_state >= 0

    @property
    def unknown_mask(self) -> np.ndarray:
        """三态中的 unknown（包含 solid，训练 loss 会额外报告 solid）。"""
        return self.anchor_state < 0


def build_w1_p_target_field(
    ivd: Any,
    solid_mask: Any,
    thresholds: W1PThresholds,
    *,
    train_frame_range: tuple[int, int] | list[int],
    min_area: int = weak_labels.DEFAULT_MIN_AREA,
    dataset_name: str = "dataset",
    split_name: str = "train",
    valid_fluid_mask: Any = None,
    invalid_mask: Any = None,
) -> W1PTargetField:
    """构造只覆盖 train 的 P95 positive/P80 negative 三态 field。

    P95 positive 使用独立的连通域面积 post-processing；过滤掉的高值小块
    保持 unknown。train 之外的 frame 全部保持 unknown，避免生成可被误用
    的 calibration/test supervision。
    """
    if split_name != "train":
        raise ValueError(
            f"W1-P target field 只能写入 split=train，实际 split={split_name!r}"
        )
    array = _as_ivd_3d(ivd)
    frame_range = _frame_range(
        train_frame_range, array.shape[0], dataset_name=dataset_name
    )
    if tuple(thresholds.train_frame_range) != frame_range:
        raise ValueError(
            f"数据集 {dataset_name!r} 的 threshold train range="
            f"{thresholds.train_frame_range!r} 与 target range={frame_range!r} 不一致"
        )
    if thresholds.source_split != "train":
        raise ValueError("W1-P target 禁止消费非 train threshold")
    if thresholds.source_id != W1P_LABEL_SOURCE:
        raise ValueError("W1-P target 只能消费当前 Ticket 03 P95/P80 threshold")
    if isinstance(min_area, (bool, np.bool_)) or int(min_area) != min_area:
        raise ValueError(f"W1-P min_area 必须是正整数，实际 {min_area!r}")
    min_area = int(min_area)
    if min_area <= 0:
        raise ValueError(f"W1-P min_area 必须是正整数，实际 {min_area!r}")
    solid, valid_support, invalid_support = _resolve_support_masks(
        solid_mask, valid_fluid_mask, invalid_mask, array.shape
    )
    state = np.full(array.shape, -1, dtype=np.int8)
    start, end = frame_range
    for frame in range(start, end):
        positive = (array[frame] >= thresholds.positive) & valid_support[frame]
        positive = weak_labels.filter_min_area(positive, min_area=min_area).astype(bool)
        negative = (
            (array[frame] <= thresholds.negative)
            & valid_support[frame]
            & ~positive
        )
        state[frame, positive] = 1
        state[frame, negative] = 0
    train_state = state[start:end]
    fluid = valid_support[start:end]
    invalid = invalid_support[start:end]
    positive_count = int(np.count_nonzero(train_state == 1))
    negative_count = int(np.count_nonzero(train_state == 0))
    unknown_fluid_count = int(np.count_nonzero((train_state < 0) & fluid))
    solid_count = int(np.count_nonzero(solid[start:end]))
    fluid_count = int(np.count_nonzero(fluid))
    invalid_count = int(np.count_nonzero(invalid))
    metadata = {
        "generation_version": W1P_GENERATION_VERSION,
        "label_source": W1P_LABEL_SOURCE,
        "source_id": W1P_LABEL_SOURCE,
        "split_name": "train",
        "frame_range": [start, end],
        "dataset_name": str(dataset_name),
        "positive_percentile": W1P_DEFAULT_POSITIVE_PERCENTILE,
        "negative_percentile": W1P_DEFAULT_NEGATIVE_PERCENTILE,
        "p95": float(thresholds.positive),
        "p80": float(thresholds.negative),
        "positive_threshold": float(thresholds.positive),
        "negative_threshold": float(thresholds.negative),
        "artifact_identity": protocol.build_p95p80_artifact_identity(
            protocol.P95P80_PERCENTILE_SOURCE_ID
        ),
        "standard_ivd_source_identity": copy.deepcopy(
            dict(thresholds.standard_ivd_source_identity
                 if thresholds.standard_ivd_source_identity is not None
                 else standard_ivd.SOURCE_IDENTITY)
        ),
        "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
        "min_area": min_area,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "unknown_fluid_count": unknown_fluid_count,
        "solid_count": solid_count,
        "fluid_count": fluid_count,
        "valid_fluid_count": fluid_count,
        "invalid_count": invalid_count,
        "solid_mask_semantics": (
            "solid cells are excluded from threshold statistics and retained as "
            "unknown; solid is not a negative"
        ),
        "valid_fluid_mask_semantics": (
            "declared valid-fluid support after excluding solid and invalid cells"
        ),
        "invalid_mask_semantics": (
            "non-solid cells outside declared valid-fluid support or explicitly "
            "invalid; retained as unknown"
        ),
        "known_coverage": (
            float((positive_count + negative_count) / fluid_count)
            if fluid_count else 0.0
        ),
        "unknown_coverage": (
            float(unknown_fluid_count / fluid_count) if fluid_count else 0.0
        ),
    }
    if thresholds.protocol_manifest_hash is not None:
        metadata["protocol_manifest_hash"] = thresholds.protocol_manifest_hash
    if thresholds.threshold_provenance is not None:
        metadata["threshold_provenance"] = copy.deepcopy(
            dict(thresholds.threshold_provenance)
        )
    return W1PTargetField(
        anchor_state=state,
        solid_mask=solid,
        metadata=metadata,
        valid_fluid_mask=valid_support,
        invalid_mask=invalid_support,
    )


_TICKET03_TARGET_FILES = {
    "state": ("state.npy", np.int8),
    "labels": ("labels.npy", np.float32),
    "label_mask": ("label_mask.npy", np.uint8),
    "positive_mask": ("positive_mask.npy", np.uint8),
    "negative_mask": ("negative_mask.npy", np.uint8),
    "unknown_mask": ("unknown_mask.npy", np.uint8),
    "valid_fluid_mask": ("valid_fluid_mask.npy", np.uint8),
    "invalid_mask": ("invalid_mask.npy", np.uint8),
    "solid_mask": ("solid_mask.npy", np.uint8),
    "frame_indices": ("frame_indices.npy", np.int64),
    "standard_ivd_row_indices": ("standard_ivd_row_indices.npy", np.int64),
}
_TICKET03_TARGET_SCHEMA_BY_SPLIT = {
    "train": (
        "criterion_ambiguity_p95p80_train_target",
        "criterion-ambiguity-ticket03-target-v1",
    ),
    "calibration": (
        "criterion_ambiguity_p95p80_calibration_reference",
        "criterion-ambiguity-ticket03-calibration-reference-v1",
    ),
}


def _w1p_json_hash(value: Any) -> str:
    """Hash JSON metadata using the same canonical form as Ticket 03."""
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _w1p_file_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _w1p_nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _w1p_bool_flag(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是 bool")
    return bool(value)


def _w1p_int_list(value: Any, *, name: str, nonempty: bool = True) -> list[int]:
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ValueError(f"{name} 必须是非空 integer list")
    result: list[int] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{name} 只能包含整数")
        try:
            integer = int(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} 只能包含整数") from exc
        if integer != item or integer < 0:
            raise ValueError(f"{name} 只能包含非负整数")
        result.append(integer)
    if len(set(result)) != len(result) or result != sorted(result):
        raise ValueError(f"{name} 必须严格递增且无重复")
    return result


def _w1p_validate_identity_pair(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    name: str,
) -> None:
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise ValueError(f"{name} 与 canonical Ticket 03 identity 不一致")


def _w1p_validate_ticket03_metadata(
    metadata: Mapping[str, Any],
    *,
    root: pathlib.Path,
    expected_split: str,
    protocol_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if expected_split not in _TICKET03_TARGET_SCHEMA_BY_SPLIT:
        raise ValueError(
            f"W1-P artifact expected_split={expected_split!r} 非法；"
            "只能是 train 或 calibration"
        )
    candidate = copy.deepcopy(dict(metadata))
    artifact_type, schema_version = _TICKET03_TARGET_SCHEMA_BY_SPLIT[expected_split]
    if candidate.get("artifact_type") != artifact_type:
        raise ValueError(
            f"W1-P artifact_type 必须是 {artifact_type!r}，"
            f"实际 {candidate.get('artifact_type')!r}"
        )
    if candidate.get("schema_version") != schema_version:
        raise ValueError("W1-P Ticket 03 target schema_version 不匹配")
    if candidate.get("split_name") != expected_split:
        raise ValueError(
            f"W1-P target split={candidate.get('split_name')!r} 与 expected_split="
            f"{expected_split!r} 不一致"
        )
    if candidate.get("source_id") != W1P_LABEL_SOURCE:
        raise ValueError(
            "W1-P target 只能使用当前 Ticket 03 P95/P80 source；"
            f"实际 {candidate.get('source_id')!r}"
        )
    if "label_source" in candidate and candidate["label_source"] != W1P_LABEL_SOURCE:
        raise ValueError("W1-P target label_source 不能是历史 P90/P60 source")
    if candidate.get("protocol_version") != protocol.P95P80_PROTOCOL_VERSION:
        raise ValueError("W1-P target 不能复用历史 protocol")
    manifest_hash = _w1p_nonempty_text(
        candidate.get("protocol_manifest_hash"),
        name="target.protocol_manifest_hash",
    )
    if protocol_manifest is not None:
        manifest = protocol.validate_p95p80_protocol_manifest(protocol_manifest)
        if manifest_hash != manifest["manifest_hash"]:
            raise ValueError("W1-P target protocol_manifest_hash 不匹配")
    identity = candidate.get("artifact_identity")
    expected_identity = protocol.build_p95p80_artifact_identity(
        protocol.P95P80_PERCENTILE_SOURCE_ID
    )
    try:
        validated_identity = protocol.validate_p95p80_artifact_identity(identity)
    except Exception as exc:
        raise ValueError("W1-P target artifact_identity 无效") from exc
    if validated_identity != expected_identity:
        raise ValueError("W1-P target artifact_identity 不是当前 P95/P80 identity")
    _w1p_validate_identity_pair(
        candidate.get("standard_ivd_source_identity"),
        standard_ivd.SOURCE_IDENTITY,
        name="target.standard_ivd_source_identity",
    )
    _w1p_validate_identity_pair(
        candidate.get("haller_standard_ivd_source_identity"),
        standard_ivd.SOURCE_IDENTITY,
        name="target.haller_standard_ivd_source_identity",
    )
    if candidate.get("final_test_relationships") != "not_read":
        raise ValueError("W1-P target 必须声明 final_test_relationships=not_read")
    if _w1p_bool_flag(candidate.get("test_source_read"), name="target.test_source_read"):
        raise ValueError("W1-P target 已读取 test source")
    if candidate.get("test_frame_indices_read") != []:
        raise ValueError("W1-P target 不得含有 test frame indices")
    if _w1p_bool_flag(candidate.get("legacy_ivd_read"), name="target.legacy_ivd_read"):
        raise ValueError("W1-P target 不得读取 legacy IVD")
    if _w1p_bool_flag(candidate.get("legacy_label_read"), name="target.legacy_label_read"):
        raise ValueError("W1-P target 不得读取 legacy label")
    if _w1p_bool_flag(
        candidate.get("legacy_artifacts_formal_source"),
        name="target.legacy_artifacts_formal_source",
    ):
        raise ValueError("W1-P target 不得把历史 artifact 作为 formal source")
    if _w1p_bool_flag(
        candidate.get("calibration_reference_recomputed_thresholds"),
        name="target.calibration_reference_recomputed_thresholds",
    ):
        raise ValueError("W1-P calibration reference 不得重算 threshold")

    frame_indices = _w1p_int_list(
        candidate.get("frame_indices"), name="target.frame_indices"
    )
    rows = _w1p_int_list(
        candidate.get("standard_ivd_row_indices"),
        name="target.standard_ivd_row_indices",
    )
    if len(frame_indices) != len(rows):
        raise ValueError("W1-P target frame_indices/row_indices 长度不一致")
    if "boussinesq" == candidate.get("dataset_name") and protocol.BOUSSINESQ_AUDIT_FRAME in frame_indices:
        raise ValueError("Boussinesq frame 1300 不能进入 W1-P train/calibration artifact")
    dataset_name = _w1p_nonempty_text(
        candidate.get("dataset_name"), name="target.dataset_name"
    )
    if dataset_name not in protocol.DATASET_NAMES:
        raise ValueError("W1-P target dataset 不在固定六数据集内")
    if protocol_manifest is not None:
        record = protocol_manifest["datasets"][dataset_name]
        ranges = record["frame_partitions"][expected_split]["ranges"]
        if any(
            not any(int(start) <= frame < int(end) for start, end in ranges)
            for frame in frame_indices
        ):
            raise ValueError("W1-P target frame 不在 protocol manifest 的目标 split")

    thresholds = candidate.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("W1-P target thresholds 必须是 object")
    if thresholds.get("source_id") != W1P_LABEL_SOURCE:
        raise ValueError("W1-P thresholds source_id 不匹配")
    if thresholds.get("artifact_identity") != expected_identity:
        raise ValueError("W1-P thresholds artifact_identity 不匹配")
    if thresholds.get("standard_ivd_source_identity") != standard_ivd.SOURCE_IDENTITY:
        raise ValueError("W1-P thresholds standard IVD source identity 不匹配")
    if thresholds.get("negative_percentile") != 80.0 or thresholds.get("positive_percentile") != 95.0:
        raise ValueError("W1-P thresholds 必须冻结为 P80/P95")
    if thresholds.get("p80") != thresholds.get("negative") or thresholds.get("p95") != thresholds.get("positive"):
        raise ValueError("W1-P thresholds p80/p95 与 negative/positive 不一致")
    if not np.isfinite(float(thresholds.get("p80"))) or not np.isfinite(float(thresholds.get("p95"))):
        raise ValueError("W1-P thresholds 必须是有限数")
    if not float(thresholds["p80"]) < float(thresholds["p95"]):
        raise ValueError("W1-P thresholds 必须满足 P80 < P95")
    threshold_train_frames = _w1p_int_list(
        thresholds.get("train_frame_indices"),
        name="thresholds.train_frame_indices",
    )
    if thresholds.get("train_frame_count") != len(threshold_train_frames):
        raise ValueError("W1-P thresholds train_frame_count 不一致")
    if thresholds.get("train_indices") != list(range(len(threshold_train_frames))):
        raise ValueError("W1-P thresholds train_indices 必须对应 materialized train rows")
    if thresholds.get("dataset_name") != dataset_name:
        raise ValueError("W1-P thresholds dataset_name 不一致")

    threshold_provenance = candidate.get("threshold_provenance")
    if not isinstance(threshold_provenance, Mapping):
        raise ValueError("W1-P threshold_provenance 必须是 object")
    if threshold_provenance.get("source_id") != W1P_LABEL_SOURCE:
        raise ValueError("W1-P threshold_provenance source_id 不匹配")
    if threshold_provenance.get("artifact_identity") != expected_identity:
        raise ValueError("W1-P threshold_provenance artifact_identity 不匹配")
    if threshold_provenance.get("statistics_split") != "train":
        raise ValueError("W1-P threshold provenance 统计来源必须是 train")
    if threshold_provenance.get("thresholds_frozen_before_consumer_materialization") is not True:
        raise ValueError("W1-P threshold 必须在 consumer materialization 前冻结")
    if threshold_provenance.get("test_source_read") is not False or threshold_provenance.get("test_frame_indices_read") != []:
        raise ValueError("W1-P threshold provenance 不得读取 test source")
    if threshold_provenance.get("calibration_values_used_for_threshold") is not False:
        raise ValueError("W1-P threshold 不得使用 calibration values")
    if threshold_provenance.get("development_audit_values_used_for_threshold") is not False:
        raise ValueError("W1-P threshold 不得使用 development/audit values")
    if threshold_provenance.get("protocol_manifest_hash") != manifest_hash:
        raise ValueError("W1-P threshold provenance protocol hash 不匹配")
    if threshold_provenance.get("standard_ivd_source_identity") != standard_ivd.SOURCE_IDENTITY:
        raise ValueError("W1-P threshold provenance standard IVD source 不匹配")
    if threshold_provenance.get("threshold_values") != thresholds:
        raise ValueError("W1-P threshold_values 与 thresholds 不一致")
    provenance_payload = copy.deepcopy(dict(threshold_provenance))
    provenance_fingerprint = provenance_payload.pop("integrity_fingerprint", None)
    if (not isinstance(provenance_fingerprint, str)
            or _w1p_json_hash(provenance_payload) != provenance_fingerprint):
        raise ValueError("W1-P threshold provenance integrity fingerprint 校验失败")

    ccp = candidate.get("connected_component_postprocessing")
    if not isinstance(ccp, Mapping):
        raise ValueError("W1-P target 缺少 connected-component post-processing provenance")
    if (
        ccp.get("part_of_ivd_definition") is not False
        or ccp.get("applies_after_thresholding") is not True
        or ccp.get("applies_to") != "positive_components_only"
        or ccp.get("does_not_change_negative_rule") is not True
    ):
        raise ValueError("W1-P 连通域过滤不得写入 IVD 定义或 negative rule")
    min_area = ccp.get("min_area_cells")
    connectivity = ccp.get("connectivity")
    if isinstance(min_area, bool) or not isinstance(min_area, (int, np.integer)) or int(min_area) <= 0:
        raise ValueError("W1-P 连通域 min_area_cells 必须是正整数")
    if connectivity != 8:
        raise ValueError("W1-P 连通域 connectivity 必须是 8")
    required_operation_order = [
        "threshold_valid_fluid_ivd",
        "connected_component_filter_positive_only",
        "assign_negative_unknown_solid_invalid_states",
    ]
    if ccp.get("operation_order") != required_operation_order:
        raise ValueError("W1-P 连通域过滤作用顺序不匹配")

    state_codes = candidate.get("state_codes")
    if state_codes != {"positive": 1, "negative": 0, "unknown": -1, "solid": -2, "invalid": -3}:
        raise ValueError("W1-P target state_codes 不匹配")
    mask_semantics = candidate.get("mask_semantics")
    if not isinstance(mask_semantics, Mapping) or mask_semantics.get("partition") != "positive/negative/unknown/solid/invalid mutually exclusive and complete":
        raise ValueError("W1-P target mask semantics 不完整")
    label_semantics = candidate.get("label_semantics")
    if (
        not isinstance(label_semantics, Mapping)
        or "sparse positive physics evidence only" not in str(
            label_semantics.get("haller_sparse_anchor", "")
        )
        or "not dense negative" not in str(label_semantics.get("haller_sparse_anchor", ""))
    ):
        raise ValueError("W1-P target 必须声明 Haller 只提供 sparse positive evidence")
    input_identity = candidate.get("input_identity")
    if not isinstance(input_identity, Mapping) or not input_identity.get("frame_input_identities"):
        raise ValueError("W1-P target 缺少 frame input identity")
    _w1p_nonempty_text(
        input_identity.get("geometry_mask_identity"),
        name="target.input_identity.geometry_mask_identity",
    )
    if input_identity.get("read_scope") != "train/calibration/development_audit only":
        raise ValueError("W1-P target input read scope 不允许 test")

    declared_hash = candidate.get("metadata_hash")
    payload = copy.deepcopy(candidate)
    payload.pop("metadata_hash", None)
    if not isinstance(declared_hash, str) or _w1p_json_hash(payload) != declared_hash:
        raise ValueError("W1-P target metadata_hash 校验失败")
    if root != pathlib.Path(root).resolve():
        raise AssertionError("internal path normalization failure")
    return candidate


def _w1p_validate_arrays(
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    state = arrays["state"]
    if len(state.shape) != 3 or any(int(size) <= 0 for size in state.shape):
        raise ValueError("W1-P target state 必须是非空 (N,Y,X)")
    n, height, width = (int(size) for size in state.shape)
    expected_3d = (n, height, width)
    for name in (
        "labels", "label_mask", "positive_mask", "negative_mask",
        "unknown_mask", "valid_fluid_mask", "invalid_mask",
    ):
        if tuple(arrays[name].shape) != expected_3d:
            raise ValueError(f"W1-P target {name} shape 不匹配")
    if tuple(arrays["solid_mask"].shape) != (height, width):
        raise ValueError("W1-P target solid_mask 必须是二维 geometry mask")
    for name in ("frame_indices", "standard_ivd_row_indices"):
        if tuple(arrays[name].shape) != (n,):
            raise ValueError(f"W1-P target {name} shape 不匹配")
    frames = np.asarray(arrays["frame_indices"])
    rows = np.asarray(arrays["standard_ivd_row_indices"])
    if np.any(frames < 0) or np.any(rows < 0):
        raise ValueError("W1-P target frame/row indices 必须非负")
    if len(np.unique(frames)) != n or np.any(np.diff(frames) <= 0):
        raise ValueError("W1-P target frame_indices 必须严格递增且无重复")
    if len(np.unique(rows)) != n or np.any(np.diff(rows) <= 0):
        raise ValueError("W1-P target standard_ivd_row_indices 必须严格递增且无重复")
    if metadata["dataset_name"] == "boussinesq" and np.any(
        frames == protocol.BOUSSINESQ_AUDIT_FRAME
    ):
        raise ValueError("Boussinesq frame 1300 不能进入 W1-P target")

    solid_2d = np.asarray(arrays["solid_mask"], dtype=bool)
    counts = {name: 0 for name in (
        "positive", "negative", "unknown", "solid", "valid_fluid", "invalid"
    )}
    chunk_size = 32
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        state_chunk = np.asarray(state[start:stop])
        labels = np.asarray(arrays["labels"][start:stop])
        masks = {
            name: np.asarray(arrays[name][start:stop], dtype=bool)
            for name in (
                "label_mask", "positive_mask", "negative_mask", "unknown_mask",
                "valid_fluid_mask", "invalid_mask",
            )
        }
        if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError("W1-P target labels 必须是有限 0/1")
        for name, value in masks.items():
            raw = np.asarray(arrays[name][start:stop])
            if not np.isin(raw, [0, 1]).all():
                raise ValueError(f"W1-P target {name} 必须是 0/1 mask")
        positive = masks["positive_mask"]
        negative = masks["negative_mask"]
        unknown = masks["unknown_mask"]
        solid = np.broadcast_to(solid_2d, state_chunk.shape)
        invalid = masks["invalid_mask"]
        valid = masks["valid_fluid_mask"]
        partition = positive.astype(np.int8) + negative + unknown + solid + invalid
        if not np.all(partition == 1):
            raise ValueError("W1-P target state masks 不是互斥且完备的 partition")
        if not np.array_equal(state_chunk, np.where(
            positive, 1, np.where(negative, 0, np.where(unknown, -1, np.where(solid, -2, -3)))
        ).astype(np.int8)):
            raise ValueError("W1-P state 与 masks 不一致")
        if not np.array_equal(masks["label_mask"], positive | negative):
            raise ValueError("W1-P label_mask 与 positive/negative 不一致")
        if not np.array_equal(labels, positive.astype(np.float32)):
            raise ValueError("W1-P labels 与 positive_mask 不一致")
        if np.any(valid & (solid | invalid)) or not np.array_equal(
            valid, (~solid) & ~invalid
        ):
            raise ValueError("W1-P valid_fluid/solid/invalid mask 语义不一致")
        counts["positive"] += int(positive.sum())
        counts["negative"] += int(negative.sum())
        counts["unknown"] += int(unknown.sum())
        counts["solid"] += int(solid.sum())
        counts["valid_fluid"] += int(valid.sum())
        counts["invalid"] += int(invalid.sum())
    declared_counts = metadata.get("class_counts")
    if not isinstance(declared_counts, Mapping):
        raise ValueError("W1-P target class_counts 缺失")
    for name, count in counts.items():
        if int(declared_counts.get(name, -1)) != count:
            raise ValueError(
                f"W1-P target class_counts.{name} 不一致："
                f"declared={declared_counts.get(name)!r} actual={count}"
            )


@dataclass
class W1PLabelArtifact:
    """Memory-mapped Ticket 03 P95/P80 train or calibration artifact."""

    root: pathlib.Path
    metadata: dict[str, Any]
    state: Any
    labels: Any
    label_mask: Any
    positive_mask: Any
    negative_mask: Any
    unknown_mask: Any
    valid_fluid_mask: Any
    invalid_mask: Any
    solid_mask: Any
    frame_indices: Any
    standard_ivd_row_indices: Any

    @property
    def source_id(self) -> str:
        return str(self.metadata["source_id"])

    @property
    def split_name(self) -> str:
        return str(self.metadata["split_name"])

    @property
    def dataset_name(self) -> str:
        return str(self.metadata["dataset_name"])

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def as_provenance(self) -> dict[str, Any]:
        """Return a bounded provenance record suitable for a train checkpoint."""
        metadata = self.metadata
        return {
            "source": W1P_LABEL_SOURCE,
            "source_id": W1P_LABEL_SOURCE,
            "label_source": W1P_LABEL_SOURCE,
            "protocol_version": metadata["protocol_version"],
            "protocol_manifest_hash": metadata["protocol_manifest_hash"],
            "dataset_name": metadata["dataset_name"],
            "split_name": metadata["split_name"],
            "frame_indices": copy.deepcopy(metadata["frame_indices"]),
            "standard_ivd_row_indices": copy.deepcopy(
                metadata["standard_ivd_row_indices"]
            ),
            "artifact_identity": copy.deepcopy(metadata["artifact_identity"]),
            "artifact_instance_identity": copy.deepcopy(
                metadata["artifact_instance_identity"]
            ),
            "metadata_hash": metadata["metadata_hash"],
            "thresholds": copy.deepcopy(metadata["thresholds"]),
            "threshold_provenance": copy.deepcopy(
                metadata["threshold_provenance"]
            ),
            "standard_ivd_source_identity": copy.deepcopy(
                metadata["standard_ivd_source_identity"]
            ),
            "standard_ivd_artifact": copy.deepcopy(
                metadata["standard_ivd_artifact"]
            ),
            "input_identity": copy.deepcopy(metadata["input_identity"]),
            "geometry_mask_identity": metadata["input_identity"][
                "geometry_mask_identity"
            ],
            "mask_semantics": copy.deepcopy(metadata["mask_semantics"]),
            "label_semantics": copy.deepcopy(metadata["label_semantics"]),
            "connected_component_postprocessing": copy.deepcopy(
                metadata["connected_component_postprocessing"]
            ),
            "test_source_read": False,
            "test_frame_indices_read": [],
            "final_test_relationships": "not_read",
            "legacy_ivd_read": False,
            "legacy_label_read": False,
            "legacy_artifacts_formal_source": False,
        }


def load_p95p80_label_artifact(
    path: str | pathlib.Path,
    *,
    expected_split: str = "train",
    protocol_manifest: Mapping[str, Any] | str | pathlib.Path | None = None,
    verify_integrity: bool = True,
) -> W1PLabelArtifact:
    """Load only a Ticket 03 train/calibration target, never a legacy target.

    The loader is deliberately explicit about ``expected_split``.  It opens
    only the target arrays under ``path``; it never searches sibling output
    directories and never opens a test source or result.
    """
    if not isinstance(verify_integrity, bool):
        raise ValueError("verify_integrity 必须是 bool")
    root = pathlib.Path(path)
    if root.is_file():
        if root.name != "target_meta.json":
            raise ValueError("W1-P artifact path 必须是目录或 target_meta.json")
        root = root.parent
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"W1-P Ticket 03 artifact directory 不存在：{root}")
    metadata_path = root / "target_meta.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"W1-P Ticket 03 artifact 缺少 {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"W1-P target metadata 无法读取：{metadata_path}") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("W1-P target metadata 必须是 object")
    manifest_value = protocol_manifest
    if isinstance(manifest_value, (str, pathlib.Path)):
        manifest_path = pathlib.Path(manifest_value).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"P95/P80 protocol manifest 不存在：{manifest_path}")
        try:
            manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("P95/P80 protocol manifest 无法读取") from exc
    validated_metadata = _w1p_validate_ticket03_metadata(
        metadata,
        root=root,
        expected_split=expected_split,
        protocol_manifest=(None if manifest_value is None else manifest_value),
    )
    arrays: dict[str, Any] = {}
    data_files = validated_metadata.get("data_files")
    if not isinstance(data_files, Mapping):
        raise ValueError("W1-P target data_files 缺失")
    for name, (filename, dtype) in _TICKET03_TARGET_FILES.items():
        path_value = root / filename
        declared = data_files.get(filename)
        if not isinstance(declared, Mapping):
            raise ValueError(f"W1-P target data_files 缺少 {filename}")
        declared_path = _w1p_nonempty_text(
            declared.get("path"), name=f"data_files.{filename}.path"
        )
        if pathlib.Path(declared_path).resolve() != path_value.resolve():
            raise ValueError(f"W1-P target data_files.{filename}.path 越界或不匹配")
        if not path_value.is_file():
            raise FileNotFoundError(f"W1-P target 缺少 {path_value}")
        try:
            array = np.load(path_value, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"W1-P target array 无法安全读取：{path_value}") from exc
        if array.dtype != np.dtype(dtype):
            raise ValueError(
                f"W1-P target {filename} dtype 不匹配："
                f"expected={np.dtype(dtype)} actual={array.dtype}"
            )
        declared_hash = _w1p_nonempty_text(
            declared.get("sha256"), name=f"data_files.{filename}.sha256"
        )
        if verify_integrity and _w1p_file_hash(path_value) != declared_hash:
            raise ValueError(f"W1-P target {filename} sha256 校验失败")
        arrays[name] = array
    _w1p_validate_arrays(arrays, validated_metadata)
    return W1PLabelArtifact(root=root, metadata=validated_metadata, **arrays)


def load_w1_p_train_target_artifact(
    path: str | pathlib.Path,
    *,
    protocol_manifest: Mapping[str, Any] | str | pathlib.Path | None = None,
    verify_integrity: bool = True,
) -> W1PLabelArtifact:
    """Load a Ticket 03 train-only P95/P80 target artifact."""
    return load_p95p80_label_artifact(
        path,
        expected_split="train",
        protocol_manifest=protocol_manifest,
        verify_integrity=verify_integrity,
    )


def load_w1_p_calibration_reference_artifact(
    path: str | pathlib.Path,
    *,
    protocol_manifest: Mapping[str, Any] | str | pathlib.Path | None = None,
    verify_integrity: bool = True,
) -> W1PLabelArtifact:
    """Load a Ticket 03 calibration reference without recomputing thresholds."""
    return load_p95p80_label_artifact(
        path,
        expected_split="calibration",
        protocol_manifest=protocol_manifest,
        verify_integrity=verify_integrity,
    )


# Explicit aliases keep the source name visible at call sites without creating
# another implementation or another artifact format.
load_p95p80_train_target_artifact = load_w1_p_train_target_artifact
load_p95p80_calibration_reference_artifact = load_w1_p_calibration_reference_artifact


@dataclass(frozen=True)
class W1PConfig:
    """W1-P 的预注册阈值、EMA、loss 权重和 ramp-up 配置。"""

    positive_percentile: float = W1P_DEFAULT_POSITIVE_PERCENTILE
    negative_percentile: float = W1P_DEFAULT_NEGATIVE_PERCENTILE
    pseudo_high: float = W1P_DEFAULT_PSEUDO_HIGH
    pseudo_low: float = W1P_DEFAULT_PSEUDO_LOW
    ema_decay: float = W1P_DEFAULT_EMA_DECAY
    ramp_up_epochs: int = W1P_DEFAULT_RAMP_UP_EPOCHS
    pseudo_weight: float = W1P_DEFAULT_PSEUDO_WEIGHT
    consistency_weight: float = W1P_DEFAULT_CONSISTENCY_WEIGHT
    min_area: int = weak_labels.DEFAULT_MIN_AREA

    def __post_init__(self) -> None:
        positive = float(self.positive_percentile)
        negative = float(self.negative_percentile)
        if not (0.0 <= negative < positive <= 100.0):
            raise ValueError(
                f"W1-P percentile 必须满足 0 <= P80 < P95 <= 100，"
                f"实际 P80={negative} P95={positive}"
            )
        if (positive != W1P_DEFAULT_POSITIVE_PERCENTILE
                or negative != W1P_DEFAULT_NEGATIVE_PERCENTILE):
            raise ValueError(
                "W1-P 当前票据冻结使用 P95 positive / P80 negative；"
                f"实际 P{positive:g}/P{negative:g}"
            )
        low = float(self.pseudo_low)
        high = float(self.pseudo_high)
        if not (0.0 <= low < 0.5 < high <= 1.0):
            raise ValueError(
                f"W1-P pseudo confidence 必须满足 0 <= low < 0.5 < high <= 1，"
                f"实际 low={low} high={high}"
            )
        decay = float(self.ema_decay)
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"W1-P ema_decay 必须位于 [0,1)，实际 {decay}")
        if isinstance(self.ramp_up_epochs, (bool, np.bool_)):
            raise ValueError("W1-P ramp_up_epochs 必须是正整数")
        ramp = int(self.ramp_up_epochs)
        if ramp != self.ramp_up_epochs or ramp <= 0:
            raise ValueError(
                f"W1-P ramp_up_epochs 必须是正整数，实际 {self.ramp_up_epochs!r}"
            )
        for name, weight in (
            ("pseudo_weight", self.pseudo_weight),
            ("consistency_weight", self.consistency_weight),
        ):
            weight = float(weight)
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(f"W1-P {name} 必须是非负有限数，实际 {weight}")
        if isinstance(self.min_area, (bool, np.bool_)):
            raise ValueError("W1-P min_area 必须是正整数")
        area = int(self.min_area)
        if area != self.min_area or area <= 0:
            raise ValueError(f"W1-P min_area 必须是正整数，实际 {self.min_area!r}")
        object.__setattr__(self, "positive_percentile", positive)
        object.__setattr__(self, "negative_percentile", negative)
        object.__setattr__(self, "pseudo_low", low)
        object.__setattr__(self, "pseudo_high", high)
        object.__setattr__(self, "ema_decay", decay)
        object.__setattr__(self, "ramp_up_epochs", ramp)
        object.__setattr__(self, "pseudo_weight", float(self.pseudo_weight))
        object.__setattr__(
            self, "consistency_weight", float(self.consistency_weight)
        )
        object.__setattr__(self, "min_area", area)

    def as_dict(self) -> dict[str, Any]:
        """返回 checkpoint/config 可复现的普通 dict。"""
        return {
            "generation_version": W1P_GENERATION_VERSION,
            "positive_percentile": self.positive_percentile,
            "negative_percentile": self.negative_percentile,
            "pseudo_high": self.pseudo_high,
            "pseudo_low": self.pseudo_low,
            "ema_decay": self.ema_decay,
            "ramp_up_epochs": self.ramp_up_epochs,
            "pseudo_weight": self.pseudo_weight,
            "consistency_weight": self.consistency_weight,
            "min_area": self.min_area,
            "feature_schema": contract.FEATURE_SCHEMA_7.as_dict(),
            "label_source": W1P_LABEL_SOURCE,
            "source_id": W1P_LABEL_SOURCE,
            "protocol_version": protocol.P95P80_PROTOCOL_VERSION,
            "artifact_identity": protocol.build_p95p80_artifact_identity(
                protocol.P95P80_PERCENTILE_SOURCE_ID
            ),
        }


def ramp_up_weight(epoch: int, ramp_up_epochs: int = W1P_DEFAULT_RAMP_UP_EPOCHS) -> float:
    """返回线性 unsupervised ramp：epoch 0 为 0，达到窗口后为 1。"""
    if isinstance(epoch, (bool, np.bool_)) or int(epoch) != epoch or int(epoch) < 0:
        raise ValueError(f"W1-P epoch 必须是非负整数，实际 {epoch!r}")
    if isinstance(ramp_up_epochs, (bool, np.bool_)):
        raise ValueError("W1-P ramp_up_epochs 必须是正整数")
    ramp = int(ramp_up_epochs)
    if ramp != ramp_up_epochs or ramp <= 0:
        raise ValueError(f"W1-P ramp_up_epochs 必须是正整数，实际 {ramp_up_epochs!r}")
    return float(min(int(epoch) / ramp, 1.0))


@dataclass
class W1PBatch:
    """携带 W1-P state mask、solid/invalid ignore 和公共 batch contract。"""

    contract_batch: contract.WeakSupervisionBatch
    solid_mask: Any
    dummy_field: Any | None = None
    invalid_mask: Any | None = None

    def __post_init__(self) -> None:
        batch = contract.validate_training_batch(
            self.contract_batch, contract.MODE_W1_P)
        if batch.label_source != W1P_LABEL_SOURCE:
            raise ValueError(
                f"W1-P formal loss source 必须是 {W1P_LABEL_SOURCE!r}，"
                f"实际 {batch.label_source!r}"
            )
        _validate_w1_p_sampling_source(batch.sampling_source)
        shape = tuple(int(v) for v in batch.labels.shape)
        if isinstance(batch.label_mask, torch.Tensor):
            solid_mask = (
                self.solid_mask.to(device=batch.label_mask.device)
                if isinstance(self.solid_mask, torch.Tensor)
                else torch.as_tensor(
                    self.solid_mask, device=batch.label_mask.device)
            )
            if tuple(solid_mask.shape) != shape:
                raise ValueError(
                    f"W1-P solid_mask shape={tuple(solid_mask.shape)} "
                    f"与 labels={shape} 不一致"
                )
            self.solid_mask = solid_mask.to(dtype=torch.bool)
            overlap = self.solid_mask & batch.label_mask
            outside = self.solid_mask & ~batch.unknown_mask
            if bool(overlap.any()) or bool(outside.any()):
                raise ValueError("W1-P solid_mask 必须只落在 unknown/ignored 区域")
        else:
            self.solid_mask = (
                self.solid_mask.detach().cpu().numpy().astype(bool)
                if isinstance(self.solid_mask, torch.Tensor)
                else np.asarray(self.solid_mask, dtype=bool)
            )
            if tuple(self.solid_mask.shape) != shape:
                raise ValueError(
                    f"W1-P solid_mask shape={self.solid_mask.shape} "
                    f"与 labels={shape} 不一致"
                )
            if np.any(self.solid_mask & batch.label_mask) or np.any(
                self.solid_mask & ~batch.unknown_mask
            ):
                raise ValueError("W1-P solid_mask 必须只落在 unknown/ignored 区域")
        if self.invalid_mask is None:
            self.invalid_mask = (
                torch.zeros(shape, dtype=torch.bool, device=batch.label_mask.device)
                if isinstance(batch.label_mask, torch.Tensor)
                else np.zeros(shape, dtype=bool)
            )
        elif isinstance(batch.label_mask, torch.Tensor):
            invalid_mask = (
                self.invalid_mask.to(device=batch.label_mask.device)
                if isinstance(self.invalid_mask, torch.Tensor)
                else torch.as_tensor(
                    self.invalid_mask, device=batch.label_mask.device
                )
            )
            if tuple(invalid_mask.shape) != shape:
                raise ValueError(
                    f"W1-P invalid_mask shape={tuple(invalid_mask.shape)} "
                    f"与 labels={shape} 不一致"
                )
            self.invalid_mask = invalid_mask.to(dtype=torch.bool)
            overlap = self.invalid_mask & batch.label_mask
            outside = self.invalid_mask & ~batch.unknown_mask
            if bool(overlap.any()) or bool(outside.any()):
                raise ValueError("W1-P invalid_mask 必须只落在 unknown/ignored 区域")
        else:
            self.invalid_mask = (
                self.invalid_mask.detach().cpu().numpy().astype(bool)
                if isinstance(self.invalid_mask, torch.Tensor)
                else np.asarray(self.invalid_mask, dtype=bool)
            )
            if tuple(self.invalid_mask.shape) != shape:
                raise ValueError(
                    f"W1-P invalid_mask shape={self.invalid_mask.shape} "
                    f"与 labels={shape} 不一致"
                )
            if np.any(self.invalid_mask & batch.label_mask) or np.any(
                self.invalid_mask & ~batch.unknown_mask
            ):
                raise ValueError("W1-P invalid_mask 必须只落在 unknown/ignored 区域")
        if self.dummy_field is None:
            if isinstance(batch.pathlines, torch.Tensor):
                self.dummy_field = batch.pathlines.new_zeros(
                    (shape[0], 1, 1, 1))
            else:
                self.dummy_field = np.zeros((shape[0], 1, 1, 1), dtype=np.float32)
        dummy_shape = getattr(self.dummy_field, "shape", ())
        if len(dummy_shape) < 1 or int(dummy_shape[0]) != shape[0]:
            raise ValueError(
                f"W1-P dummy_field batch 维度与 labels 不一致："
                f"dummy={dummy_shape} labels={shape}"
            )

    @property
    def pathlines(self) -> Any:
        """model-facing 7-channel pathlines。"""
        return self.contract_batch.pathlines

    @property
    def labels(self) -> Any:
        """BCE target；unknown 值仅作占位，必须配合 label_mask 使用。"""
        return self.contract_batch.labels

    @property
    def label_mask(self) -> Any:
        """known positive/negative mask。"""
        return self.contract_batch.label_mask

    @property
    def unknown_mask(self) -> Any:
        """unknown mask（包含 solid/invalid；loss 会分别排除）。"""
        return self.contract_batch.unknown_mask

    @property
    def invalid(self) -> Any:
        """canonical standard-IVD invalid/non-finite mask。"""
        return self.invalid_mask

    @property
    def sampling_source(self) -> str | None:
        """采样池来源，与 formal loss source 分离。"""
        return self.contract_batch.sampling_source

    @property
    def label_source(self) -> str:
        """formal W1-P loss source。"""
        return self.contract_batch.label_source

    def as_dict(self) -> dict[str, Any]:
        """返回不含大数组的日志摘要。"""
        result = self.contract_batch.as_dict()
        result["solid_count"] = int(
            self.solid_mask.sum().item()
            if isinstance(self.solid_mask, torch.Tensor)
            else np.asarray(self.solid_mask).sum()
        )
        result["invalid_count"] = int(
            self.invalid_mask.sum().item()
            if isinstance(self.invalid_mask, torch.Tensor)
            else np.asarray(self.invalid_mask).sum()
        )
        return result

    def to(self, device: str | torch.device) -> "W1PBatch":
        """将 batch 数组搬到指定 torch device，保留 provenance。"""
        batch = self.contract_batch
        converted = contract.WeakSupervisionBatch(
            pathlines=torch.as_tensor(batch.pathlines, device=device),
            labels=torch.as_tensor(batch.labels, device=device).float(),
            label_source=batch.label_source,
            split_name=batch.split_name,
            feature_schema=batch.feature_schema,
            label_mask=torch.as_tensor(batch.label_mask, device=device),
            unknown_mask=torch.as_tensor(batch.unknown_mask, device=device),
            sampling_source=batch.sampling_source,
            provenance=batch.provenance,
            mode=batch.mode,
            input_schema=batch.input_schema,
        )
        return W1PBatch(
            converted,
            torch.as_tensor(self.solid_mask, device=device),
            torch.as_tensor(self.dummy_field, device=device),
            torch.as_tensor(self.invalid_mask, device=device),
        )


def build_w1_p_batch(
    pathlines: Any,
    labels: Any,
    label_mask: Any,
    unknown_mask: Any,
    solid_mask: Any,
    *,
    sampling_source: str,
    split_name: str = "train",
    provenance: Mapping[str, Any] | None = None,
    dummy_field: Any | None = None,
    invalid_mask: Any | None = None,
) -> W1PBatch:
    """构造带 source/mask 追踪的 W1-P batch。

    sampling_source 必须显式提供；它只进入 sampling_source，不会进入
    W1-P formal label_source。invalid_mask 与 solid_mask 保持独立语义。
    """
    sampling_source = _validate_w1_p_sampling_source(sampling_source)
    sources = dict(provenance or {})
    sources.setdefault("anchor", {"source": W1P_LABEL_SOURCE})
    sources.setdefault("sampling", {"source": sampling_source})
    base = contract.WeakSupervisionBatch(
        pathlines=pathlines,
        labels=labels,
        label_source=W1P_LABEL_SOURCE,
        split_name=split_name,
        feature_schema=contract.FEATURE_SCHEMA_7,
        label_mask=label_mask,
        unknown_mask=unknown_mask,
        sampling_source=sampling_source,
        provenance=sources,
        mode=contract.MODE_W1_P,
        input_schema=contract.FEATURE_SCHEMA_7,
    )
    return W1PBatch(
        base, solid_mask, dummy_field, invalid_mask
    )


def _w1p_pathline_window(
    value: Mapping[str, Any],
    *,
    artifact: W1PLabelArtifact,
) -> dict[str, Any]:
    """Validate one complete, same-split pathline window for an artifact batch."""
    if not isinstance(value, Mapping):
        raise TypeError("W1-P pathline_window 必须是 object")
    required = (
        "split_name", "frame_start", "frame_end", "split_start", "split_end",
        "t_win", "window_step",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"W1-P pathline_window 缺少字段：{missing!r}")
    if value["split_name"] != artifact.split_name:
        raise ValueError(
            "W1-P pathline window 不得跨 split："
            f"window={value['split_name']!r} artifact={artifact.split_name!r}"
        )
    if artifact.split_name != "train":
        raise ValueError("W1-P artifact batch 只能来自 split=train")

    integers: dict[str, int] = {}
    for field in (
        "frame_start", "frame_end", "split_start", "split_end", "t_win",
        "window_step",
    ):
        raw = value[field]
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"W1-P pathline_window.{field} 必须是整数")
        try:
            integer = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"W1-P pathline_window.{field} 必须是整数"
            ) from exc
        if integer != raw:
            raise ValueError(f"W1-P pathline_window.{field} 必须是整数")
        integers[field] = integer
    if (
        integers["split_start"] < 0
        or integers["split_start"] >= integers["split_end"]
        or integers["t_win"] <= 0
        or integers["window_step"] <= 0
    ):
        raise ValueError("W1-P pathline_window split/t_win/window_step 非法")
    if integers["frame_end"] - integers["frame_start"] != integers["t_win"]:
        raise ValueError("W1-P pathline_window frame_end-frame_start 必须等于 t_win")
    if not (
        integers["split_start"] <= integers["frame_start"]
        and integers["frame_start"] < integers["frame_end"] <= integers["split_end"]
    ):
        raise ValueError("W1-P pathline window 超出声明的 split 边界")
    artifact_frames = np.asarray(artifact.frame_indices, dtype=np.int64)
    covered_frames = np.arange(
        integers["frame_start"], integers["frame_end"], dtype=np.int64
    )
    if not np.isin(covered_frames, artifact_frames).all():
        raise ValueError(
            "W1-P pathline window 包含 artifact 未提供的 frame；"
            "拒绝以邻近 split frame 补齐窗口"
        )
    dataset_name = value.get("dataset_name")
    if dataset_name is not None and dataset_name != artifact.dataset_name:
        raise ValueError("W1-P pathline_window dataset_name 与 artifact 不一致")
    normalized = copy.deepcopy(dict(value))
    normalized.update(integers)
    normalized["dataset_name"] = artifact.dataset_name
    normalized["artifact_split_name"] = artifact.split_name
    normalized["cross_split"] = False
    return normalized


def _w1p_numpy(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"W1-P {name} 无法转换为 ndarray") from exc


def _w1p_integer_vector(value: Any, *, name: str, length: int) -> np.ndarray:
    array = _w1p_numpy(value, name=name)
    if array.ndim != 1 or tuple(array.shape) != (length,):
        raise ValueError(f"W1-P {name} 必须是 shape=({length},) 的整数数组")
    if np.issubdtype(array.dtype, np.bool_) or not np.isfinite(array).all():
        raise ValueError(f"W1-P {name} 必须是有限整数数组")
    integer = array.astype(np.int64)
    if not np.array_equal(array, integer) or np.any(integer < 0):
        raise ValueError(f"W1-P {name} 必须是非负整数数组")
    return integer


def _w1p_seed_cells(value: Any, *, batch_size: int, point_count: int) -> np.ndarray:
    cells = _w1p_numpy(value, name="seed_cells")
    if cells.ndim == 2 and batch_size == 1:
        cells = cells[None, ...]
    if tuple(cells.shape) != (batch_size, point_count, 2):
        raise ValueError(
            "W1-P seed_cells 必须是 (batch,trajectory,2)，"
            f"实际 shape={cells.shape}"
        )
    if not np.issubdtype(cells.dtype, np.number) or not np.isfinite(cells).all():
        raise ValueError("W1-P seed_cells 必须是有限数值坐标")
    integer = cells.astype(np.int64)
    if not np.array_equal(cells, integer):
        raise ValueError("W1-P seed_cells 必须是整数网格坐标")
    return integer


def build_w1_p_batch_from_artifact(
    pathlines: Any,
    artifact: W1PLabelArtifact,
    *,
    frame_indices: Any,
    seed_cells: Any,
    sampling_source: str = W1P_LABEL_SOURCE,
    split_name: str = "train",
    pathline_window: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    dummy_field: Any | None = None,
) -> W1PBatch:
    """Build a W1-P batch by indexing a validated Ticket 03 target artifact.

    ``frame_indices`` names the absolute frame used for each batch row and
    ``seed_cells`` names the corresponding ``(y, x)`` cells.  The builder
    never opens a sibling output, recomputes thresholds, or turns solid,
    invalid, failure, or unknown cells into dense negatives.
    """
    if not isinstance(artifact, W1PLabelArtifact):
        raise TypeError("W1-P artifact batch 必须消费 W1PLabelArtifact")
    if artifact.split_name != "train" or split_name != "train":
        raise ValueError("W1-P artifact batch 只能消费 split=train")
    sampling_source = _validate_w1_p_sampling_source(sampling_source)
    if sampling_source != artifact.source_id or artifact.source_id != W1P_LABEL_SOURCE:
        raise ValueError(
            "W1-P artifact batch sampling_source 必须是当前 Ticket 03 P95/P80 source"
        )
    window = _w1p_pathline_window(pathline_window, artifact=artifact)

    shape = getattr(pathlines, "shape", None)
    if shape is None:
        pathlines = _w1p_numpy(pathlines, name="pathlines")
        shape = pathlines.shape
    if len(shape) == 3:
        if isinstance(pathlines, torch.Tensor):
            pathlines = pathlines.unsqueeze(0)
        else:
            pathlines = np.expand_dims(np.asarray(pathlines), axis=0)
        shape = pathlines.shape
    if len(shape) != 4 or int(shape[0]) <= 0 or int(shape[2]) <= 0:
        raise ValueError("W1-P pathlines 必须是 (batch,time,trajectory,7)")
    if int(shape[-1]) != contract.FEATURE_SCHEMA_7.channel_count:
        raise ValueError("W1-P artifact batch 必须使用 7-channel pathlines")
    if isinstance(pathlines, torch.Tensor):
        if not bool(torch.isfinite(pathlines).all()):
            raise ValueError("W1-P pathlines 含有非有限值")
    elif not np.isfinite(np.asarray(pathlines)).all():
        raise ValueError("W1-P pathlines 含有非有限值")
    batch_size = int(shape[0])
    point_count = int(shape[2])
    frames = _w1p_integer_vector(
        frame_indices, name="frame_indices", length=batch_size
    )
    cells = _w1p_seed_cells(
        seed_cells, batch_size=batch_size, point_count=point_count
    )
    height = int(artifact.state.shape[1])
    width = int(artifact.state.shape[2])
    if np.any(cells[..., 0] >= height) or np.any(cells[..., 1] >= width):
        raise ValueError("W1-P seed_cells 超出 artifact geometry mask")

    artifact_frames = np.asarray(artifact.frame_indices, dtype=np.int64)
    positions = np.searchsorted(artifact_frames, frames)
    if np.any(positions >= len(artifact_frames)) or not np.array_equal(
        artifact_frames[positions], frames
    ):
        raise ValueError(
            "W1-P frame_indices 不在 target artifact 中；"
            "拒绝以其他 split 或历史 artifact 补齐"
        )
    selected_state = np.empty((batch_size, point_count), dtype=np.int8)
    for batch_index, position in enumerate(positions):
        selected_state[batch_index] = np.asarray(
            artifact.state[position, cells[batch_index, :, 0], cells[batch_index, :, 1]],
            dtype=np.int8,
        )
    labels = (selected_state == 1).astype(np.float32)
    known = (selected_state == 0) | (selected_state == 1)
    unknown = selected_state < 0
    solid = selected_state == -2
    invalid = selected_state == -3

    sources = artifact.as_provenance()
    supplied = copy.deepcopy(dict(provenance or {}))
    for key in (
        "source", "source_id", "label_source", "protocol_version",
        "protocol_manifest_hash", "artifact_identity", "metadata_hash",
        "standard_ivd_source_identity", "standard_ivd_artifact",
        "input_identity", "geometry_mask_identity", "mask_semantics",
        "label_semantics", "connected_component_postprocessing",
        "test_source_read", "test_frame_indices_read", "final_test_relationships",
        "legacy_ivd_read", "legacy_label_read", "legacy_artifacts_formal_source",
    ):
        if key in supplied and supplied[key] != sources[key]:
            raise ValueError(f"W1-P artifact provenance.{key} 与 artifact 不一致")
    sources.update(supplied)
    sampling = dict(sources.get("sampling", {}))
    if sampling.get("source") not in (None, sampling_source):
        raise ValueError("W1-P sampling provenance source 与 artifact 不一致")
    sampling["source"] = sampling_source
    sampling["artifact_source_id"] = artifact.source_id
    sources["sampling"] = sampling
    sources["window"] = window
    sources["selected_frame_indices"] = frames.tolist()
    sources["selected_seed_cells"] = cells.tolist()
    sources["state_selection"] = {
        "positive": "state == 1",
        "negative": "state == 0",
        "unknown": "state == -1",
        "solid": "state == -2",
        "invalid": "state == -3",
        "haller_sparse_anchor_is_not_dense_negative": True,
    }
    return build_w1_p_batch(
        pathlines,
        labels,
        known,
        unknown,
        solid,
        sampling_source=sampling_source,
        split_name=split_name,
        provenance=sources,
        dummy_field=dummy_field,
        invalid_mask=invalid,
    )


# Explicit source-aware name for downstream callers; it is an alias rather
# than a second implementation or artifact format.
build_p95p80_batch_from_artifact = build_w1_p_batch_from_artifact


def _as_prediction_tensor(value: Any, expected_shape: tuple[int, ...], name: str) -> torch.Tensor:
    """将模型概率统一为 tensor，并校验 shape、有限性和 [0,1] 范围。"""
    prediction = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tuple(prediction.shape) != expected_shape:
        raise ValueError(
            f"W1-P {name} shape={tuple(prediction.shape)} 与 labels="
            f"{expected_shape} 不一致"
        )
    if not prediction.is_floating_point():
        raise ValueError(f"W1-P {name} 必须是浮点 sigmoid probability")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError(f"W1-P {name} 含有非有限值")
    if bool((prediction < 0).any()) or bool((prediction > 1).any()):
        raise ValueError(f"W1-P {name} 必须是 [0,1] 内的 sigmoid probability")
    return prediction


def _masked_bce(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """只在 mask=True 的元素上计算 BCE；空 mask 返回带梯度的零。"""
    if not bool(mask.any()):
        return predictions.sum() * 0.0
    return F.binary_cross_entropy(predictions[mask], targets[mask])


def compute_w1_p_loss(
    student_predictions: Any,
    teacher_predictions: Any,
    batch: W1PBatch,
    *,
    config: W1PConfig | None = None,
    epoch: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """计算 W1-P 的 masked anchor、confident pseudo 和 consistency loss。

    anchor BCE 只消费 known positive/negative；unknown 的 pseudo-label 仅在
    teacher probability 达到 0.90/0.10 时接受；consistency 仅作用于 valid
    unknown。solid 和 invalid 永远不进入 pseudo/consistency loss。返回 loss
    与可直接写入日志/checkpoint 的统计。
    """
    if not isinstance(batch, W1PBatch):
        raise TypeError("W1-P loss 必须消费 W1PBatch")
    config = W1PConfig() if config is None else config
    if not isinstance(config, W1PConfig):
        raise TypeError("W1-P config 必须是 W1PConfig")
    contract.validate_training_batch(batch.contract_batch, contract.MODE_W1_P)
    expected_shape = tuple(int(v) for v in batch.labels.shape)
    student = _as_prediction_tensor(
        student_predictions, expected_shape, "student_predictions")
    teacher = _as_prediction_tensor(
        teacher_predictions, expected_shape, "teacher_predictions").detach()
    if teacher.device != student.device:
        teacher = teacher.to(student.device)
    labels = torch.as_tensor(batch.labels, device=student.device).float()
    known = torch.as_tensor(batch.label_mask, device=student.device, dtype=torch.bool)
    unknown = torch.as_tensor(batch.unknown_mask, device=student.device, dtype=torch.bool)
    solid = torch.as_tensor(batch.solid_mask, device=student.device, dtype=torch.bool)
    invalid = torch.as_tensor(batch.invalid_mask, device=student.device, dtype=torch.bool)
    if tuple(labels.shape) != expected_shape:
        raise ValueError("W1-P labels shape 在 loss 中发生漂移")
    if bool((solid & known).any()) or bool((solid & ~unknown).any()):
        raise ValueError("W1-P solid mask 必须只落在 unknown/ignored 区域")
    if bool((invalid & known).any()) or bool((invalid & ~unknown).any()):
        raise ValueError("W1-P invalid mask 必须只落在 unknown/ignored 区域")
    if bool((solid & invalid).any()):
        raise ValueError("W1-P solid 与 invalid mask 不能重叠")

    pseudo_eligible = unknown & ~solid & ~invalid
    # 将阈值转换到 teacher 的 dtype，避免 float32 中的 0.90 因二进制表示
    # 略低于 Python float 0.90 而被错误拒绝。
    high = torch.as_tensor(
        config.pseudo_high, device=teacher.device, dtype=teacher.dtype)
    low = torch.as_tensor(
        config.pseudo_low, device=teacher.device, dtype=teacher.dtype)
    confident = (teacher >= high) | (teacher <= low)
    pseudo_mask = pseudo_eligible & confident
    pseudo_targets = (teacher >= 0.5).float()
    consistency_mask = pseudo_eligible
    anchor_loss = _masked_bce(student, labels, known)
    pseudo_loss = _masked_bce(student, pseudo_targets, pseudo_mask)
    consistency_loss = (
        F.mse_loss(student[consistency_mask], teacher[consistency_mask])
        if bool(consistency_mask.any()) else student.sum() * 0.0
    )
    ramp = ramp_up_weight(epoch, config.ramp_up_epochs)
    total = (
        anchor_loss
        + ramp * config.pseudo_weight * pseudo_loss
        + ramp * config.consistency_weight * consistency_loss
    )

    total_count = int(labels.numel())
    anchor_count = int(known.sum().item())
    unknown_count = int(unknown.sum().item())
    solid_count = int(solid.sum().item())
    invalid_count = int(invalid.sum().item())
    eligible_count = int(pseudo_eligible.sum().item())
    accepted_count = int(pseudo_mask.sum().item())
    positive_count = int((pseudo_mask & (pseudo_targets >= 0.5)).sum().item())
    negative_count = accepted_count - positive_count
    disagreement = (
        float(torch.abs(student[consistency_mask] - teacher[consistency_mask]).mean().detach())
        if bool(consistency_mask.any()) else 0.0
    )
    stats = {
        "loss": float(total.detach().cpu()),
        "anchor_loss": float(anchor_loss.detach().cpu()),
        "pseudo_loss": float(pseudo_loss.detach().cpu()),
        "consistency_loss": float(consistency_loss.detach().cpu()),
        "ramp_weight": ramp,
        "anchor_count": anchor_count,
        "anchor_positive_count": int(((labels >= 0.5) & known).sum().item()),
        "anchor_negative_count": int(((labels < 0.5) & known).sum().item()),
        "unknown_count": unknown_count,
        "solid_count": solid_count,
        "invalid_count": invalid_count,
        "pseudo_eligible_count": eligible_count,
        "pseudo_accepted_count": accepted_count,
        "pseudo_positive_count": positive_count,
        "pseudo_negative_count": negative_count,
        "anchor_coverage": anchor_count / total_count if total_count else 0.0,
        "unknown_coverage": unknown_count / total_count if total_count else 0.0,
        "solid_coverage": solid_count / total_count if total_count else 0.0,
        "invalid_coverage": invalid_count / total_count if total_count else 0.0,
        "pseudo_acceptance": accepted_count / eligible_count if eligible_count else 0.0,
        "teacher_student_disagreement": disagreement,
        "sampling_source": batch.sampling_source,
        "loss_source": batch.label_source,
    }
    return total, stats


def _strict_nonnegative_int(value: Any, *, name: str) -> int:
    """将 epoch/seed/step 规范为非负整数，拒绝字符串和隐式截断。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是非负整数，实际 {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是非负整数，实际 {value!r}") from exc
    try:
        integral = converted == value
    except (TypeError, ValueError):
        integral = False
    if not integral or converted < 0:
        raise ValueError(f"{name} 必须是非负整数，实际 {value!r}")
    return converted


def _validate_w1_p_sampling_source(source: Any) -> str:
    """W1-P 只允许当前 P95/P80 或 legacy sampling，禁止历史 P90/P60。"""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("W1-P 必须显式提供 sampling_source")
    source = contract.validate_label_source(source.strip())
    allowed = {
        contract.LABEL_SOURCE_P95P80,
        contract.LABEL_SOURCE_LEGACY_P85,
    }
    if source not in allowed:
        raise ValueError(
            f"W1-P sampling_source 只能是 {sorted(allowed)!r}，实际 {source!r}"
        )
    return source


def _w1p_consumer_metadata(
    value: Mapping[str, Any] | None,
    *,
    expected_split: str,
    name: str,
) -> dict[str, Any] | None:
    """Validate trainer-bound target/calibration metadata without reading arrays."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"W1-P {name} 必须是 object")
    metadata = copy.deepcopy(dict(value))
    source = metadata.get("source_id", metadata.get("label_source"))
    if source != W1P_LABEL_SOURCE:
        raise ValueError(
            f"W1-P {name} 只能绑定当前 P95/P80 source，实际 {source!r}"
        )
    if metadata.get("label_source", W1P_LABEL_SOURCE) != W1P_LABEL_SOURCE:
        raise ValueError(f"W1-P {name} 不能使用历史 P90/P60 label source")
    if metadata.get("protocol_version") not in (None, protocol.P95P80_PROTOCOL_VERSION):
        raise ValueError(f"W1-P {name} 不能绑定历史 protocol")
    split = metadata.get("split_name")
    if split not in (None, expected_split):
        raise ValueError(
            f"W1-P {name} split={split!r} 与 expected_split={expected_split!r} 不一致"
        )
    artifact_type = metadata.get("artifact_type")
    if artifact_type is not None:
        _w1p_validate_ticket03_metadata(
            metadata,
            root=pathlib.Path.cwd().resolve(),
            expected_split=expected_split,
            protocol_manifest=None,
        )
    return metadata


def _w1p_validate_calibration_policy(
    policy: Mapping[str, Any] | None,
    *,
    target_metadata: Mapping[str, Any] | None,
    calibration_metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Validate the calibration lineage attached to a P95/P80 checkpoint."""
    if target_metadata is None:
        return policy
    if policy is None or not isinstance(policy, Mapping):
        raise ValueError(
            "W1-P 当前 target 必须显式提供 P95/P80 calibration_policy"
        )
    expected_hash = target_metadata.get("protocol_manifest_hash")
    if (
        policy.get("source_id", policy.get("source")) != W1P_LABEL_SOURCE
        or policy.get("protocol_version") != protocol.P95P80_PROTOCOL_VERSION
        or policy.get("protocol_manifest_hash") != expected_hash
    ):
        raise ValueError(
            "W1-P calibration_policy 必须绑定当前 P95/P80 protocol manifest"
        )
    if policy.get("statistics_split") not in (None, "train"):
        raise ValueError("W1-P calibration_policy threshold statistics 只能来自 train")
    if policy.get("threshold_provenance") != target_metadata.get(
        "threshold_provenance"
    ):
        raise ValueError(
            "W1-P calibration_policy 必须携带冻结 train threshold provenance"
        )
    if policy.get("test_source_read") is True:
        raise ValueError("W1-P calibration_policy 不得读取 test source")
    if calibration_metadata is not None:
        if calibration_metadata.get("artifact_identity") != target_metadata.get(
            "artifact_identity"
        ):
            raise ValueError("W1-P calibration reference 与 target artifact identity 不一致")
        if calibration_metadata.get("protocol_manifest_hash") != expected_hash:
            raise ValueError("W1-P calibration reference 与 target protocol hash 不一致")
        if calibration_metadata.get("thresholds") != target_metadata.get("thresholds"):
            raise ValueError("W1-P calibration reference 与 target thresholds 不一致")
    return copy.deepcopy(dict(policy))


def _w1p_checkpoint_lineage(
    extra_metadata: Mapping[str, Any],
    *,
    target_metadata: Mapping[str, Any] | None,
    calibration_metadata: Mapping[str, Any] | None,
) -> None:
    """Reject resume when the bound P95/P80 target/reference lineage drifts."""
    checkpoint_target = extra_metadata.get("target_field")
    checkpoint_calibration = extra_metadata.get("calibration_reference")
    if target_metadata is None:
        if checkpoint_target is not None or checkpoint_calibration is not None:
            raise ValueError(
                "W1-P checkpoint 已绑定 target/calibration provenance，"
                "resume trainer 不能省略它"
            )
        return
    if checkpoint_target != dict(target_metadata):
        raise ValueError("W1-P checkpoint target provenance 与 trainer 不一致")
    if calibration_metadata is not None:
        if checkpoint_calibration != dict(calibration_metadata):
            raise ValueError("W1-P checkpoint calibration provenance 与 trainer 不一致")
    elif checkpoint_calibration is not None:
        raise ValueError("W1-P trainer 缺少 checkpoint 要求的 calibration provenance")
    if extra_metadata.get("formal_loss_source") != W1P_LABEL_SOURCE:
        raise ValueError("W1-P checkpoint formal loss source 不是当前 P95/P80 source")


def _prepare_ema_teacher(teacher: nn.Module) -> nn.Module:
    """冻结并切换 EMA teacher 到 evaluation mode。"""
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    return teacher


def clone_ema_teacher(student: nn.Module) -> nn.Module:
    """深拷贝 student 作为同初始状态、不可反传的 EMA teacher。"""
    if not isinstance(student, nn.Module):
        raise TypeError("W1-P student 必须是 torch.nn.Module")
    return _prepare_ema_teacher(copy.deepcopy(student))


@torch.no_grad()
def update_ema_teacher(
    student: nn.Module,
    teacher: nn.Module,
    *,
    decay: float = W1P_DEFAULT_EMA_DECAY,
) -> None:
    """在 student optimizer step 后按 ``decay=.99`` 更新 teacher state。"""
    if not isinstance(student, nn.Module) or not isinstance(teacher, nn.Module):
        raise TypeError("W1-P EMA student/teacher 必须是 torch.nn.Module")
    if student is teacher:
        raise ValueError("W1-P EMA teacher 不能与 student 共享同一 module")
    decay = float(decay)
    if not np.isfinite(decay) or not (0.0 <= decay < 1.0):
        raise ValueError(f"W1-P EMA decay 必须位于 [0,1)，实际 {decay}")
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    if tuple(student_state) != tuple(teacher_state):
        raise ValueError("W1-P student/teacher state_dict keys 不一致")
    for name, target in teacher_state.items():
        source = student_state[name]
        if not isinstance(target, torch.Tensor) or not isinstance(source, torch.Tensor):
            raise TypeError(f"W1-P EMA state {name!r} 必须是 tensor")
        if tuple(target.shape) != tuple(source.shape):
            raise ValueError(f"W1-P EMA state {name!r} shape 不一致")
        source = source.detach().to(device=target.device, dtype=target.dtype)
        if target.is_floating_point() or target.is_complex():
            target.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            # 非浮点 buffer（如 BatchNorm 的 num_batches_tracked）没有有意义
            # 的凸组合，直接同步以保证 teacher state 可恢复。
            target.copy_(source)
    _prepare_ema_teacher(teacher)


def _forward_w1_p_model(model: nn.Module, batch: W1PBatch) -> Any:
    """调用现有 (dummy_field, 7-channel pathlines) seam。"""
    if isinstance(model, contract.ChannelSelectingAdapter):
        return model.forward_batch(
            batch.contract_batch, dummy_field=batch.dummy_field, consumer="train")
    return model((batch.dummy_field, batch.pathlines))


class W1PTrainer:
    """W1-P 的最小可恢复训练 seam，不扩展到下游 W1-H/W2/W3。"""

    def __init__(
        self,
        student: nn.Module,
        optimizer: Any,
        *,
        sampling_source: str,
        config: W1PConfig | None = None,
        teacher: nn.Module | None = None,
        scheduler: Any | None = None,
        seed: int = 0,
        target_metadata: Mapping[str, Any] | None = None,
        calibration_metadata: Mapping[str, Any] | None = None,
        grad_clip_norm: float | None = None,
        teacher_device: str | torch.device | None = None,
    ) -> None:
        if not isinstance(student, nn.Module):
            raise TypeError("W1-P student 必须是 torch.nn.Module")
        if optimizer is None or not all(
            hasattr(optimizer, attribute)
            for attribute in ("zero_grad", "step", "state_dict")
        ):
            raise TypeError("W1-P optimizer 必须提供 zero_grad()/step()/state_dict()")
        if teacher is not None and not isinstance(teacher, nn.Module):
            raise TypeError("W1-P teacher 必须是 torch.nn.Module")
        self.student = student
        self.teacher = clone_ema_teacher(student) if teacher is None else teacher
        if self.teacher is self.student:
            raise ValueError("W1-P teacher 不能与 student 共享同一 module")
        if teacher_device is not None:
            try:
                self.teacher_device = torch.device(teacher_device)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"W1-P teacher_device 必须是合法 torch device：{teacher_device!r}"
                ) from exc
            self.teacher.to(self.teacher_device)
        else:
            self.teacher_device = None
        _prepare_ema_teacher(self.teacher)
        student_keys = tuple(self.student.state_dict())
        teacher_keys = tuple(self.teacher.state_dict())
        if student_keys != teacher_keys:
            raise ValueError("W1-P student/teacher state_dict keys 不一致")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = W1PConfig() if config is None else config
        if not isinstance(self.config, W1PConfig):
            raise TypeError("W1-P config 必须是 W1PConfig")
        self.sampling_source = _validate_w1_p_sampling_source(sampling_source)
        self.seed = _strict_nonnegative_int(seed, name="seed")
        self.target_metadata = _w1p_consumer_metadata(
            target_metadata, expected_split="train", name="target_metadata"
        )
        self.calibration_metadata = _w1p_consumer_metadata(
            calibration_metadata,
            expected_split="calibration",
            name="calibration_metadata",
        )
        if (
            self.target_metadata is not None
            and self.target_metadata.get("artifact_type")
            == _TICKET03_TARGET_SCHEMA_BY_SPLIT["train"][0]
            and self.calibration_metadata is None
        ):
            raise ValueError(
                "W1-P Ticket 03 train target 必须同时绑定 calibration reference"
            )
        if (
            self.target_metadata is not None
            and self.target_metadata.get("artifact_type")
            == _TICKET03_TARGET_SCHEMA_BY_SPLIT["train"][0]
            and self.calibration_metadata is not None
            and self.calibration_metadata.get("artifact_type")
            != _TICKET03_TARGET_SCHEMA_BY_SPLIT["calibration"][0]
        ):
            raise ValueError("W1-P calibration_metadata 必须是 Ticket 03 calibration reference")
        if grad_clip_norm is not None:
            grad_clip_norm = float(grad_clip_norm)
            if not np.isfinite(grad_clip_norm) or grad_clip_norm <= 0.0:
                raise ValueError("W1-P grad_clip_norm 必须是正的有限数或 None")
        self.grad_clip_norm = grad_clip_norm
        self.global_step = 0

    def _move_models(self, device: str | torch.device) -> None:
        student_device = torch.device(device)
        teacher_device = self.teacher_device or student_device
        self.student.to(student_device)
        self.teacher.to(teacher_device)
        _prepare_ema_teacher(self.teacher)

    def train_step(
        self,
        batch: W1PBatch,
        *,
        epoch: int,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """执行一批：forward → W1-P loss → optimizer → EMA update。"""
        if not isinstance(batch, W1PBatch):
            raise TypeError("W1-P train_step 必须消费 W1PBatch")
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        if batch.sampling_source != self.sampling_source:
            raise ValueError(
                f"W1-P sampling_source 不匹配：trainer={self.sampling_source!r} "
                f"batch={batch.sampling_source!r}"
            )
        self._move_models(device)
        student_device = torch.device(device)
        teacher_device = self.teacher_device or student_device
        student_batch = batch.to(student_device)
        teacher_batch = (
            student_batch
            if teacher_device == student_device
            else batch.to(teacher_device)
        )
        self.student.train()
        _prepare_ema_teacher(self.teacher)
        self.optimizer.zero_grad(set_to_none=True)
        student_predictions = _forward_w1_p_model(self.student, student_batch)
        with torch.no_grad():
            teacher_predictions = _forward_w1_p_model(self.teacher, teacher_batch)
        loss, stats = compute_w1_p_loss(
            student_predictions, teacher_predictions, student_batch,
            config=self.config, epoch=epoch,
        )
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        update_ema_teacher(
            self.student, self.teacher, decay=self.config.ema_decay)
        self.global_step += 1
        stats = dict(stats)
        stats.update({"epoch": epoch, "global_step": self.global_step})
        return stats

    def run_epoch(
        self,
        batches: Any,
        *,
        epoch: int,
        device: str | torch.device = "cpu",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """运行一个有明确非空 guard 的 epoch，并返回平均 loss/计数日志。"""
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        if max_steps is not None:
            max_steps = _strict_nonnegative_int(max_steps, name="max_steps")
            if max_steps <= 0:
                raise ValueError("W1-P max_steps 必须是正整数或 None")
        logs = []
        for batch in batches:
            if max_steps is not None and len(logs) >= max_steps:
                break
            logs.append(self.train_step(batch, epoch=epoch, device=device))
        if not logs:
            raise ValueError("W1-P batches 为空：训练循环无样本可跑")
        average_keys = {
            "loss", "anchor_loss", "pseudo_loss", "consistency_loss",
            "ramp_weight", "anchor_coverage", "unknown_coverage",
            "solid_coverage", "invalid_coverage", "pseudo_acceptance",
            "teacher_student_disagreement",
        }
        count_keys = {
            "anchor_count", "anchor_positive_count", "anchor_negative_count",
            "unknown_count", "solid_count", "pseudo_eligible_count",
            "invalid_count",
            "pseudo_accepted_count", "pseudo_positive_count",
            "pseudo_negative_count",
        }
        summary = {}
        for key in average_keys:
            summary[key] = float(np.mean([float(log[key]) for log in logs]))
        for key in count_keys:
            summary[key] = int(sum(int(log[key]) for log in logs))
        sources = {log["sampling_source"] for log in logs}
        loss_sources = {log["loss_source"] for log in logs}
        if len(sources) != 1 or len(loss_sources) != 1:
            raise ValueError("W1-P 一个 epoch 内 source 发生漂移")
        summary.update({
            "sampling_source": next(iter(sources)),
            "loss_source": next(iter(loss_sources)),
            "epoch": epoch,
            "steps": len(logs),
            "global_step": self.global_step,
        })
        return summary

    def _checkpoint_extra_metadata(
        self,
        extra_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """构造保留 formal source 语义的 W1-P checkpoint metadata。"""
        if extra_metadata is not None and not isinstance(extra_metadata, Mapping):
            raise TypeError("W1-P extra_metadata 必须是 object")
        extra = dict(extra_metadata or {})
        reserved = {
            "generation_version": W1P_GENERATION_VERSION,
            "formal_loss_source": W1P_LABEL_SOURCE,
            "w1_p_config": self.config.as_dict(),
        }
        if self.target_metadata is not None:
            reserved["target_field"] = self.target_metadata
        if self.calibration_metadata is not None:
            reserved["calibration_reference"] = self.calibration_metadata
        for key, expected in reserved.items():
            if key in extra and extra[key] != expected:
                raise ValueError(
                    f"W1-P checkpoint extra_metadata.{key} 与 trainer 语义不一致"
                )
            extra[key] = expected
        return extra

    def save_checkpoint(
        self,
        path: Any,
        *,
        epoch: int,
        dataset_config: Mapping[str, Any],
        split_config: Any,
        sampling_config: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
        calibration_policy: Mapping[str, Any] | None = None,
        anchor_hash: str | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """保存可 resume 的 W1-P checkpoint；数据/split/window 均必须显式给出。"""
        if self.scheduler is None:
            raise ValueError(
                "W1-P resume checkpoint 必须提供 scheduler，禁止保存不可恢复的空 scheduler"
            )
        if anchor_hash is not None:
            raise ValueError("W1-P checkpoint 不应携带 Haller anchor_hash")
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        calibration_policy = _w1p_validate_calibration_policy(
            calibration_policy,
            target_metadata=self.target_metadata,
            calibration_metadata=self.calibration_metadata,
        )
        return contract.save_checkpoint(
            path, self.student, self.optimizer, self.scheduler,
            mode=contract.MODE_W1_P,
            feature_schema=contract.FEATURE_SCHEMA_7,
            adapter_input_schema=contract.FEATURE_SCHEMA_7,
            dataset_config=dataset_config,
            split_config=split_config,
            sampling_config=sampling_config,
            label_source=W1P_LABEL_SOURCE,
            sampling_source=self.sampling_source,
            teacher=self.teacher,
            epoch=epoch,
            global_step=self.global_step,
            metrics=metrics,
            seed=self.seed,
            calibration_policy=(
                {"source": "none"}
                if calibration_policy is None else calibration_policy
            ),
            anchor_hash=anchor_hash,
            extra_metadata=self._checkpoint_extra_metadata(extra_metadata),
        )

    def load_checkpoint(
        self,
        path: Any,
        *,
        expected_dataset_config: Mapping[str, Any],
        expected_split_config: Any,
        expected_sampling_config: Mapping[str, Any],
        device: str | torch.device = "cpu",
        load_mode: str = "resume",
        restore_rng: bool | None = None,
        strict_cuda_rng: bool = False,
    ) -> dict[str, Any]:
        """按 W1-P mode/source/schema/split contract 恢复 student 与 EMA。"""
        self._move_models(device)
        result = contract.load_checkpoint(
            path, self.student, self.optimizer, self.scheduler,
            teacher=self.teacher, device=device,
            expected_mode=contract.MODE_W1_P,
            expected_feature_schema=contract.FEATURE_SCHEMA_7,
            expected_dataset_config=expected_dataset_config,
            expected_split_config=expected_split_config,
            expected_sampling_config=expected_sampling_config,
            expected_label_source=W1P_LABEL_SOURCE,
            expected_sampling_source=self.sampling_source,
            restore_rng=restore_rng,
            strict_cuda_rng=strict_cuda_rng,
            load_mode=load_mode,
        )
        if result["anchor_hash"] is not None:
            raise ValueError("W1-P checkpoint 不应携带 Haller anchor_hash")
        _w1p_checkpoint_lineage(
            result.get("extra_metadata", {}),
            target_metadata=self.target_metadata,
            calibration_metadata=self.calibration_metadata,
        )
        self.global_step = _strict_nonnegative_int(
            result["global_step"], name="checkpoint global_step")
        self.seed = _strict_nonnegative_int(result["seed"], name="checkpoint seed")
        _prepare_ema_teacher(self.teacher)
        return result
