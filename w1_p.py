"""W1-P p90/p60/unknown 弱监督基础设施。

本模块只实现 W1-P 的 train-only local-IVD 三态监督、masked/pseudo/
consistency loss、EMA teacher 和可恢复训练 seam。W1-H 的 Haller physics
anchor、W2 uncertainty gate 与 W3 contrastive head 留给后续票据。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import weak_labels
import weak_supervision_contract as contract


W1P_GENERATION_VERSION = "w1-p-local-p90-p60-v1"
W1P_LABEL_SOURCE = contract.LABEL_SOURCE_LOCAL_P90_P60
W1P_DEFAULT_POSITIVE_PERCENTILE = 90.0
W1P_DEFAULT_NEGATIVE_PERCENTILE = 60.0
W1P_DEFAULT_PSEUDO_HIGH = 0.90
W1P_DEFAULT_PSEUDO_LOW = 0.10
W1P_DEFAULT_EMA_DECAY = 0.99
W1P_DEFAULT_RAMP_UP_EPOCHS = 12
W1P_DEFAULT_PSEUDO_WEIGHT = 1.0
W1P_DEFAULT_CONSISTENCY_WEIGHT = 1.0


def _as_ivd_3d(ivd: Any) -> np.ndarray:
    """将 local-IVD 统一为 (T, Y, X) 并拒绝非有限值。"""
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
    """只由 train frame fluid cells 计算的 p60/p90 阈值。"""

    negative: float
    positive: float
    train_frame_range: tuple[int, int]
    dataset_name: str = "dataset"
    source_split: str = "train"

    def __post_init__(self) -> None:
        if self.source_split != "train":
            raise ValueError(
                "W1-P threshold source_split 必须是 train，不能消费 calibration/test"
            )
        negative = float(self.negative)
        positive = float(self.positive)
        if not np.isfinite(negative) or not np.isfinite(positive):
            raise ValueError("W1-P p60/p90 threshold 必须是有限数")
        if not negative < positive:
            raise ValueError(
                f"W1-P threshold 必须满足 p60 < p90，实际 "
                f"p60={negative} p90={positive}"
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
        return {
            "negative_percentile": W1P_DEFAULT_NEGATIVE_PERCENTILE,
            "positive_percentile": W1P_DEFAULT_POSITIVE_PERCENTILE,
            "negative": float(self.negative),
            "positive": float(self.positive),
            "train_frame_range": list(self.train_frame_range),
            "dataset_name": self.dataset_name,
            "source_split": self.source_split,
        }


def compute_w1_p_thresholds(
    ivd: Any,
    solid_mask: Any,
    *,
    train_frame_range: tuple[int, int] | list[int],
    dataset_name: str = "dataset",
    split_name: str = "train",
) -> W1PThresholds:
    """从显式 train frame range 的 fluid local-IVD 计算 p60/p90。

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
    solid = _as_solid_mask(solid_mask, array.shape)
    start, end = frame_range
    values = array[start:end][~solid[start:end]]
    if values.size == 0:
        raise ValueError(
            f"数据集 {dataset_name!r} 的 train fluid cells 为空，无法计算 W1-P p60/p90"
        )
    negative = float(np.percentile(values, W1P_DEFAULT_NEGATIVE_PERCENTILE))
    positive = float(np.percentile(values, W1P_DEFAULT_POSITIVE_PERCENTILE))
    return W1PThresholds(
        negative=negative,
        positive=positive,
        train_frame_range=frame_range,
        dataset_name=str(dataset_name),
    )


@dataclass
class W1PTargetField:
    """W1-P 三态 target field：1=positive、0=negative、-1=unknown。"""

    anchor_state: np.ndarray
    solid_mask: np.ndarray
    metadata: dict[str, Any]

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
) -> W1PTargetField:
    """构造只覆盖 train 的 p90 positive/p60 negative 三态 field。

    p90 positive 复用既有 5×5 连通域面积过滤；过滤掉的高值小块保持
    unknown。train 之外的 frame 全部保持 unknown，避免生成可被误用的
    calibration/test supervision。
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
    if isinstance(min_area, (bool, np.bool_)) or int(min_area) != min_area:
        raise ValueError(f"W1-P min_area 必须是正整数，实际 {min_area!r}")
    min_area = int(min_area)
    if min_area <= 0:
        raise ValueError(f"W1-P min_area 必须是正整数，实际 {min_area!r}")
    solid = _as_solid_mask(solid_mask, array.shape)
    state = np.full(array.shape, -1, dtype=np.int8)
    start, end = frame_range
    for frame in range(start, end):
        positive = (array[frame] >= thresholds.positive) & ~solid[frame]
        positive = weak_labels.filter_min_area(positive, min_area=min_area).astype(bool)
        negative = (
            (array[frame] <= thresholds.negative)
            & ~solid[frame]
            & ~positive
        )
        state[frame, positive] = 1
        state[frame, negative] = 0
    train_state = state[start:end]
    fluid = ~solid[start:end]
    positive_count = int(np.count_nonzero(train_state == 1))
    negative_count = int(np.count_nonzero(train_state == 0))
    unknown_fluid_count = int(np.count_nonzero((train_state < 0) & fluid))
    solid_count = int(np.count_nonzero(solid[start:end]))
    fluid_count = int(np.count_nonzero(fluid))
    metadata = {
        "generation_version": W1P_GENERATION_VERSION,
        "label_source": W1P_LABEL_SOURCE,
        "split_name": "train",
        "frame_range": [start, end],
        "dataset_name": str(dataset_name),
        "positive_percentile": W1P_DEFAULT_POSITIVE_PERCENTILE,
        "negative_percentile": W1P_DEFAULT_NEGATIVE_PERCENTILE,
        "positive_threshold": float(thresholds.positive),
        "negative_threshold": float(thresholds.negative),
        "min_area": min_area,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "unknown_fluid_count": unknown_fluid_count,
        "solid_count": solid_count,
        "fluid_count": fluid_count,
        "known_coverage": (
            float((positive_count + negative_count) / fluid_count)
            if fluid_count else 0.0
        ),
        "unknown_coverage": (
            float(unknown_fluid_count / fluid_count) if fluid_count else 0.0
        ),
    }
    return W1PTargetField(anchor_state=state, solid_mask=solid, metadata=metadata)


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
                f"W1-P percentile 必须满足 0 <= p60 < p90 <= 100，"
                f"实际 p60={negative} p90={positive}"
            )
        if (positive != W1P_DEFAULT_POSITIVE_PERCENTILE
                or negative != W1P_DEFAULT_NEGATIVE_PERCENTILE):
            raise ValueError(
                "W1-P 当前票据冻结使用 p90 positive / p60 negative；"
                f"实际 p{positive:g}/p{negative:g}"
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
    """携带 W1-P 三态 mask、solid ignore 和公共 batch contract 的批次。"""

    contract_batch: contract.WeakSupervisionBatch
    solid_mask: Any
    dummy_field: Any | None = None

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
        """unknown mask（包含 solid；solid_mask 会在 loss 中再次排除）。"""
        return self.contract_batch.unknown_mask

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
) -> W1PBatch:
    """构造带 source/mask 追踪的 W1-P batch。

    sampling_source 必须显式提供；典型值是 legacy_p85，但它永远只进入
    sampling_source，不会进入 W1-P formal label_source。
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
    return W1PBatch(base, solid_mask, dummy_field)


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
    teacher probability 达到 0.90/0.10 时接受；consistency 仅作用于非 solid
    unknown。返回 loss 与可直接写入日志/checkpoint 的统计。
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
    if tuple(labels.shape) != expected_shape:
        raise ValueError("W1-P labels shape 在 loss 中发生漂移")
    if bool((solid & known).any()) or bool((solid & ~unknown).any()):
        raise ValueError("W1-P solid mask 必须只落在 unknown/ignored 区域")

    pseudo_eligible = unknown & ~solid
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
        "pseudo_eligible_count": eligible_count,
        "pseudo_accepted_count": accepted_count,
        "pseudo_positive_count": positive_count,
        "pseudo_negative_count": negative_count,
        "anchor_coverage": anchor_count / total_count if total_count else 0.0,
        "unknown_coverage": unknown_count / total_count if total_count else 0.0,
        "solid_coverage": solid_count / total_count if total_count else 0.0,
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
    """W1-P 只允许 local/legacy sampling，禁止把 Haller source 当采样池。"""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("W1-P 必须显式提供 sampling_source")
    source = contract.validate_label_source(source.strip())
    allowed = {
        contract.LABEL_SOURCE_LOCAL_P90_P60,
        contract.LABEL_SOURCE_LEGACY_P85,
    }
    if source not in allowed:
        raise ValueError(
            f"W1-P sampling_source 只能是 {sorted(allowed)!r}，实际 {source!r}"
        )
    return source


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
        if target_metadata is not None and not isinstance(target_metadata, Mapping):
            raise TypeError("W1-P target_metadata 必须是 object")
        if target_metadata is not None:
            target_source = target_metadata.get("label_source")
            if target_source is not None and target_source != W1P_LABEL_SOURCE:
                raise ValueError(
                    "W1-P target_metadata.label_source 必须是 local_p90_p60"
                )
            self.target_metadata = copy.deepcopy(dict(target_metadata))
        else:
            self.target_metadata = None
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
            "solid_coverage", "pseudo_acceptance",
            "teacher_student_disagreement",
        }
        count_keys = {
            "anchor_count", "anchor_positive_count", "anchor_negative_count",
            "unknown_count", "solid_count", "pseudo_eligible_count",
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
        self.global_step = _strict_nonnegative_int(
            result["global_step"], name="checkpoint global_step")
        self.seed = _strict_nonnegative_int(result["seed"], name="checkpoint seed")
        _prepare_ema_teacher(self.teacher)
        return result
