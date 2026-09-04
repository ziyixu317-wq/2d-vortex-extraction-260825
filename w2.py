"""W2 不确定性门控伪标签训练 seam。

W2 消费 W1-H 的 train anchor batch，并继续使用包含 5×5 local-IVD 特征的
现有七通道 pathline 输入。Teacher 对每个 batch 固定执行三次 stochastic
view，使用 mean probability 和 population variance 驱动伪标签门控，
Bernoulli entropy 仅作为诊断。Calibration 只能读取显式声明的 calibration
Haller GT，并为所有 dataset 产生一个 global variance gate。

Haller 文献的 Zotero 候选 key 是 ``L2PX3NQX``。本模块不实现或重新解释
Haller contour 算法，因此相关 source metadata 继续遵守 W1-H 的
``pending_verification`` 合同。
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import weak_supervision_contract as contract
from w1_h import W1HBatch, _validate_train_window_metadata
from w1_p import (
    W1PConfig,
    _as_prediction_tensor,
    _masked_bce,
    _prepare_ema_teacher,
    _strict_nonnegative_int,
    clone_ema_teacher,
    ramp_up_weight,
    update_ema_teacher,
)


W2_GENERATION_VERSION = "w2-uncertainty-gate-v1"
W2_LABEL_SOURCE = contract.LABEL_SOURCE_HALLER_TRAIN
W2_CALIBRATION_SOURCE = contract.LABEL_SOURCE_HALLER_CALIBRATION
W2_TEST_SOURCE = contract.LABEL_SOURCE_HALLER_TEST
W2_DEFAULT_VIEW_COUNT = 3
W2_DEFAULT_PSEUDO_HIGH = 0.90
W2_DEFAULT_PSEUDO_LOW = 0.10
W2_ALLOWED_SAMPLING_SOURCES = frozenset({
    contract.LABEL_SOURCE_LEGACY_P85,
    contract.LABEL_SOURCE_LOCAL_P90_P60,
})


def _nonempty_text(value: Any, *, name: str) -> str:
    """校验必须显式提供的非空文本字段。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _nonempty_hash(value: Any, *, name: str) -> str:
    """校验内容/manifest hash，拒绝隐式空 hash。"""
    return _nonempty_text(value, name=name)


def _nonempty_text_sequence(
    value: Any,
    *,
    name: str,
    hash_values: bool = False,
) -> tuple[str, ...]:
    """校验 provenance 名称/hash 序列，拒绝字符串被拆成字符列表。"""
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} 必须是非空字符串序列")
    if not value:
        raise ValueError(f"{name} 必须是非空字符串序列")
    validator = _nonempty_hash if hash_values else _nonempty_text
    return tuple(
        validator(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _source_names(value: Any):
    """递归提取已注册 supervision source，供泄漏 guard 使用。"""
    if isinstance(value, str):
        if value in contract.VALID_LABEL_SOURCES:
            yield value
        return
    if isinstance(value, np.ndarray):
        yield from _source_names(value.tolist())
        return
    if isinstance(value, np.generic):
        yield from _source_names(value.item())
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _source_names(key)
            yield from _source_names(child)
    elif isinstance(value, (tuple, list, set)):
        for child in value:
            yield from _source_names(child)


_TEST_ONLY_KEYS = frozenset({
    "test",
    "test_gt",
    "test_label",
    "test_labels",
    "test_metric",
    "test_metrics",
    "test_prediction",
    "test_predictions",
    "test_result",
    "test_results",
})
_TEST_ONLY_KEY_EXCEPTIONS = frozenset({
    # These are audit flags/parameter names describing a guarded boundary;
    # they do not carry test labels or metrics into a train/calibration seam.
    "failure_fallback_calibration_test",
    "haller_gt_test_artifact_read",
})

_W2_CHECKPOINT_REQUIRED_METRICS = frozenset({
    "view_count",
    "variance_gate",
    "pseudo_accepted_count",
    "pseudo_acceptance",
    "pseudo_positive_ratio",
    "pseudo_negative_ratio",
    "mean_probability_mean",
    "mean_probability_std",
    "mean_probability_min",
    "mean_probability_max",
    "predictive_variance_mean",
    "predictive_variance_std",
    "predictive_variance_min",
    "predictive_variance_max",
    "entropy_mean",
    "entropy_std",
    "entropy_min",
    "entropy_max",
    "teacher_student_disagreement",
})


def _test_only_key_names(value: Any):
    """递归找出 test label/metric/result 的字段名或 marker。

    训练和 calibration provenance 不使用可扩展的 test 指标字段，因此除了
    历史显式键名，还拒绝由 ``test`` token 组成的别名（如 ``test_f1``、
    ``label_test``）。这样新加的 test-only 指标不会因为没有加入固定枚举而
    静默穿过 W2 gate。
    """
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(".", "_")
        test_tokens = {token for token in normalized.split("_") if token}
        if (normalized not in _TEST_ONLY_KEY_EXCEPTIONS
                and (normalized in _TEST_ONLY_KEYS
                     or normalized.startswith("test_")
                     or normalized.endswith("_test")
                     or {"gt", "test"}.issubset(test_tokens)
                     or {"label", "test"}.issubset(test_tokens)
                     or {"metric", "test"}.issubset(test_tokens))):
            yield normalized
        return
    if isinstance(value, np.ndarray):
        yield from _test_only_key_names(value.tolist())
        return
    if isinstance(value, np.generic):
        yield from _test_only_key_names(value.item())
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(".", "_")
            test_tokens = {token for token in normalized.split("_") if token}
            if (normalized not in _TEST_ONLY_KEY_EXCEPTIONS
                    and (normalized in _TEST_ONLY_KEYS
                         or normalized.startswith("test_")
                         or normalized.endswith("_test")
                         or {"gt", "test"}.issubset(test_tokens)
                         or {"label", "test"}.issubset(test_tokens)
                         or {"metric", "test"}.issubset(test_tokens))):
                yield normalized
            if normalized in {"split", "split_name"} and str(child).strip().lower() == "test":
                yield normalized
            yield from _test_only_key_names(child)
    elif isinstance(value, (tuple, list, set)):
        for child in value:
            yield from _test_only_key_names(child)


def _reject_test_source(value: Any, *, context: str) -> None:
    """拒绝 test Haller GT 出现在训练/calibration 输入或 metadata 中。"""
    if W2_TEST_SOURCE in set(_source_names(value)):
        raise ValueError(
            f"{context} 禁止出现 {W2_TEST_SOURCE}；test Haller GT 只能用于最终 evaluation"
        )
    test_keys = sorted(set(_test_only_key_names(value)))
    if test_keys:
        raise ValueError(
            f"{context} 禁止访问 test-only label/metric/result 字段：{test_keys!r}"
        )


def _validate_w2_checkpoint_metrics(
    metrics: Any,
    *,
    variance_gate: float,
) -> dict[str, Any]:
    """校验 checkpoint 必须携带的 W2 gate/view/acceptance 诊断。"""
    if not isinstance(metrics, Mapping):
        raise TypeError("W2 checkpoint metrics 必须是 object，且必须记录 gate/acceptance")
    result = copy.deepcopy(dict(metrics))
    _reject_test_source(result, context="W2 checkpoint metrics")
    missing = sorted(_W2_CHECKPOINT_REQUIRED_METRICS.difference(result))
    if missing:
        raise ValueError(
            "W2 checkpoint metrics 缺少 gate/view/acceptance 诊断字段："
            f"{missing!r}"
        )
    try:
        view_count = int(result["view_count"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("W2 checkpoint metrics.view_count 必须固定为 3") from exc
    if view_count != W2_DEFAULT_VIEW_COUNT or view_count != result["view_count"]:
        raise ValueError("W2 checkpoint metrics.view_count 必须固定为 3")
    metric_gate = _strict_gate(
        result["variance_gate"], name="W2 checkpoint metrics.variance_gate")
    if metric_gate != variance_gate:
        raise ValueError(
            "W2 checkpoint metrics.variance_gate 与 trainer global variance_gate 不一致"
        )
    numeric_fields = _W2_CHECKPOINT_REQUIRED_METRICS - {"view_count", "variance_gate"}
    for field in numeric_fields:
        try:
            value = float(result[field])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"W2 checkpoint metrics.{field} 必须是有限数") from exc
        if not np.isfinite(value):
            raise ValueError(f"W2 checkpoint metrics.{field} 必须是有限数")
        result[field] = value
    bounded_ranges = {}
    for field in result:
        field_name = str(field)
        if field_name.startswith("mean_probability"):
            bounded_ranges[field_name] = (0.0, 1.0, "[0,1]")
        elif field_name.startswith("predictive_variance"):
            bounded_ranges[field_name] = (0.0, 0.25, "[0,0.25]")
        elif field_name.startswith("entropy"):
            bounded_ranges[field_name] = (0.0, float(np.log(2.0)) + 1e-6, "[0,log(2)]")
        elif "disagreement" in field_name:
            bounded_ranges[field_name] = (0.0, 1.0, "[0,1]")
    for field, (lower, upper, interval) in bounded_ranges.items():
        value = float(result[field])
        if not lower <= value <= upper:
            raise ValueError(
                f"W2 checkpoint metrics.{field} 必须位于 {interval} 内"
            )
    for field, raw_value in result.items():
        if field == "view_count":
            continue
        if str(field).endswith("_count"):
            result[field] = _strict_nonnegative_int(
                raw_value, name=f"W2 checkpoint metrics.{field}")
        elif (str(field).endswith("_ratio")
              or str(field).endswith("_acceptance")
              or str(field).endswith("_coverage")):
            try:
                value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"W2 checkpoint metrics.{field} 必须是 [0,1] 内的有限数"
                ) from exc
            if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"W2 checkpoint metrics.{field} 必须是 [0,1] 内的有限数"
                )
            result[field] = value
    result["view_count"] = view_count
    result["variance_gate"] = metric_gate
    return result


def _as_float_tensor(
    value: Any,
    *,
    name: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """将概率/方差输入转为有限 floating tensor，并保留 device。"""
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 numeric tensor/array") from exc
    if not torch.is_floating_point(tensor):
        tensor = tensor.float()
    if device is not None:
        tensor = tensor.to(device=device)
    if tensor.ndim == 0:
        raise ValueError(f"{name} 必须带有 sample 维度")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} 必须全部有限")
    return tensor


def _as_probability_tensor(
    value: Any,
    *,
    name: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    """将预测转为并校验 [0,1] probability tensor。"""
    tensor = _as_float_tensor(value, name=name, device=device)
    if not bool(((tensor >= 0.0) & (tensor <= 1.0)).all()):
        raise ValueError(f"{name} 必须位于 [0, 1]")
    return tensor


def _as_bool_mask(
    value: Any,
    shape: tuple[int, ...],
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    """把 numpy/torch mask 规格化为给定 device 上的 bool tensor。"""
    try:
        mask = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 bool/0-1 mask") from exc
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} shape={tuple(mask.shape)} 与 expected={shape} 不一致")
    if mask.dtype != torch.bool:
        if not bool(torch.all((mask == 0) | (mask == 1))):
            raise ValueError(f"{name} 必须只包含 0/1")
        mask = mask.to(dtype=torch.bool)
    return mask.to(device=device)


def _mask_count(mask: torch.Tensor) -> int:
    """统计 bool mask 的 true 数量。"""
    return int(mask.sum().item())


def _strict_gate(value: Any, *, name: str = "variance_gate") -> float:
    """校验 uncertainty threshold，拒绝 None、NaN 和隐式负值。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是 [0,1] 内的有限数")
    try:
        gate = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是 [0,1] 内的有限数") from exc
    if not np.isfinite(gate) or not (0.0 <= gate <= 1.0):
        raise ValueError(f"{name} 必须是 [0,1] 内的有限数，实际 {value!r}")
    return gate


def _strict_probability_threshold(value: Any, *, name: str) -> float:
    """校验 calibration 使用的全局 prediction threshold。"""
    return _strict_gate(value, name=name)


def _distribution(tensor: torch.Tensor) -> dict[str, float]:
    """生成可直接写入日志的 min/max/mean/std 分布摘要。"""
    values = tensor.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean().cpu()),
        "std": float(values.std(unbiased=False).cpu()),
        "min": float(values.min().cpu()),
        "max": float(values.max().cpu()),
    }


@dataclass(frozen=True)
class W2Config(W1PConfig):
    """W2 的冻结三视图、confidence 和显式 global variance gate 配置。"""

    view_count: int = W2_DEFAULT_VIEW_COUNT
    variance_gate: float | None = None

    def __post_init__(self) -> None:
        W1PConfig.__post_init__(self)
        if isinstance(self.view_count, (bool, np.bool_)):
            raise ValueError("W2 view_count 必须固定为 3")
        view_count = int(self.view_count)
        if view_count != self.view_count or view_count != W2_DEFAULT_VIEW_COUNT:
            raise ValueError(
                f"W2 view_count 必须固定为 {W2_DEFAULT_VIEW_COUNT}，实际 {self.view_count!r}"
            )
        if (float(self.pseudo_high) != W2_DEFAULT_PSEUDO_HIGH
                or float(self.pseudo_low) != W2_DEFAULT_PSEUDO_LOW):
            raise ValueError(
                "W2 confidence gate 必须固定为 positive=0.90/negative=0.10"
            )
        gate = None if self.variance_gate is None else _strict_gate(self.variance_gate)
        object.__setattr__(self, "view_count", view_count)
        object.__setattr__(self, "variance_gate", gate)

    @property
    def uncertainty_gate(self) -> float | None:
        """variance gate 的语义别名，便于日志/调用方阅读。"""
        return self.variance_gate

    def as_dict(self) -> dict[str, Any]:
        """返回可写入 checkpoint 的 W2 配置，不包含数组。"""
        result = W1PConfig.as_dict(self)
        result.update({
            "generation_version": W2_GENERATION_VERSION,
            "view_count": self.view_count,
            "pseudo_high": W2_DEFAULT_PSEUDO_HIGH,
            "pseudo_low": W2_DEFAULT_PSEUDO_LOW,
            "variance_gate": self.variance_gate,
            "variance_is_primary_gate": True,
            "entropy_is_diagnostic_only": True,
            "label_source": W2_LABEL_SOURCE,
        })
        return result


@dataclass(frozen=True)
class W2Statistics:
    """三次 stochastic teacher prediction 的统计量。"""

    mean_probability: torch.Tensor
    predictive_variance: torch.Tensor
    entropy: torch.Tensor
    view_count: int = W2_DEFAULT_VIEW_COUNT

    def __post_init__(self) -> None:
        mean = _as_probability_tensor(self.mean_probability, name="mean_probability")
        variance = _as_float_tensor(
            self.predictive_variance, name="predictive_variance", device=mean.device)
        entropy = _as_float_tensor(self.entropy, name="entropy", device=mean.device)
        if tuple(variance.shape) != tuple(mean.shape) or tuple(entropy.shape) != tuple(mean.shape):
            raise ValueError("W2 mean/variance/entropy shape 必须一致")
        if bool((variance < 0.0).any()) or bool((variance > 0.25).any()):
            raise ValueError("W2 predictive_variance 必须位于 [0, 0.25]")
        entropy_upper = float(np.log(2.0)) + 1e-6
        if bool((entropy < 0.0).any()) or bool((entropy > entropy_upper).any()):
            raise ValueError("W2 Bernoulli entropy 必须位于 [0, log(2)]")
        if self.view_count != W2_DEFAULT_VIEW_COUNT:
            raise ValueError("W2 statistics view_count 必须固定为 3")
        object.__setattr__(self, "mean_probability", mean)
        object.__setattr__(self, "predictive_variance", variance)
        object.__setattr__(self, "entropy", entropy)
        object.__setattr__(self, "view_count", int(self.view_count))

    @property
    def mean(self) -> torch.Tensor:
        """mean probability 的简短别名。"""
        return self.mean_probability

    @property
    def variance(self) -> torch.Tensor:
        """predictive variance 的简短别名。"""
        return self.predictive_variance

    def __getitem__(self, key: str) -> Any:
        """允许 mapping 风格读取常用统计字段。"""
        aliases = {
            "mean": "mean_probability",
            "variance": "predictive_variance",
        }
        key = aliases.get(key, key)
        if key == "view_count":
            return self.view_count
        if key not in {"mean_probability", "predictive_variance", "entropy"}:
            raise KeyError(key)
        return getattr(self, key)

    def as_dict(self) -> dict[str, Any]:
        """返回张量与 view count，供 loss seam 使用。"""
        return {
            "mean_probability": self.mean_probability,
            "predictive_variance": self.predictive_variance,
            "entropy": self.entropy,
            "view_count": self.view_count,
        }


def _stack_teacher_views(view_predictions: Any) -> torch.Tensor:
    """将 (3,B,K) 或三个 (B,K) prediction 组成统一 tensor。"""
    if isinstance(view_predictions, torch.Tensor):
        views = view_predictions
        if views.ndim != 3:
            raise ValueError(
                f"W2 teacher views 必须是 (view,batch,trajectory)，实际 {tuple(views.shape)}"
            )
        if views.shape[0] != W2_DEFAULT_VIEW_COUNT:
            raise ValueError("W2 teacher views 必须恰好为 3 次 stochastic view")
        return _as_probability_tensor(views, name="teacher_views")
    if isinstance(view_predictions, (str, bytes)):
        raise TypeError("W2 teacher views 必须是 tensor 或 3 个 prediction 的序列")
    try:
        values = list(view_predictions)
    except TypeError as exc:
        raise TypeError("W2 teacher views 必须是 tensor 或 3 个 prediction 的序列") from exc
    if len(values) != W2_DEFAULT_VIEW_COUNT:
        raise ValueError("W2 teacher views 必须恰好为 3 次 stochastic view")
    tensors = [_as_probability_tensor(value, name=f"teacher_view[{index}]")
               for index, value in enumerate(values)]
    first_shape = tuple(tensors[0].shape)
    first_device = tensors[0].device
    if any(tuple(value.shape) != first_shape for value in tensors[1:]):
        raise ValueError("W2 teacher views 的 batch/trajectory shape 必须一致")
    if any(value.device != first_device for value in tensors[1:]):
        raise ValueError("W2 teacher views 的 device 必须一致")
    return torch.stack(tensors, dim=0)


def _bernoulli_entropy(probability: torch.Tensor) -> torch.Tensor:
    """按 predictive mean 计算稳定的 Bernoulli entropy。"""
    eps = torch.finfo(probability.dtype).eps
    p = probability.clamp(min=eps, max=1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p))


def compute_w2_statistics(view_predictions: Any) -> W2Statistics:
    """计算三视图 mean probability、population variance 和 Bernoulli entropy。"""
    views = _stack_teacher_views(view_predictions)
    mean = views.mean(dim=0)
    variance = views.var(dim=0, unbiased=False)
    entropy = _bernoulli_entropy(mean)
    return W2Statistics(mean, variance, entropy, view_count=int(views.shape[0]))


@dataclass(frozen=True)
class W2GateResult:
    """W2 双门控结果；未被接受的 unknown 不会产生 pseudo target。"""

    pseudo_mask: torch.Tensor
    pseudo_labels: torch.Tensor
    candidate_mask: torch.Tensor
    confidence_mask: torch.Tensor
    low_uncertainty_mask: torch.Tensor
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    variance_gate: float

    @property
    def accepted_mask(self) -> torch.Tensor:
        """pseudo_mask 的语义别名。"""
        return self.pseudo_mask

    @property
    def unknown_mask(self) -> torch.Tensor:
        """仍保持 unknown 的候选区域。"""
        return self.candidate_mask & ~self.pseudo_mask

    @property
    def accepted_count(self) -> int:
        return _mask_count(self.pseudo_mask)

    @property
    def positive_count(self) -> int:
        return _mask_count(self.positive_mask)

    @property
    def negative_count(self) -> int:
        return _mask_count(self.negative_mask)

    @property
    def unknown_count(self) -> int:
        return _mask_count(self.unknown_mask)

    def __getitem__(self, key: str) -> Any:
        """允许调用方用常见字段名读取 gate 结果。"""
        aliases = {"mask": "pseudo_mask", "accepted": "pseudo_mask"}
        key = aliases.get(key, key)
        if key in {
            "pseudo_mask", "pseudo_labels", "candidate_mask", "confidence_mask",
            "low_uncertainty_mask", "positive_mask", "negative_mask",
        }:
            return getattr(self, key)
        if key == "variance_gate":
            return self.variance_gate
        if key == "accepted_count":
            return self.accepted_count
        if key == "positive_count":
            return self.positive_count
        if key == "negative_count":
            return self.negative_count
        if key == "unknown_count":
            return self.unknown_count
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        """返回不含张量的大门控计数摘要。"""
        candidate_count = _mask_count(self.candidate_mask)
        return {
            "variance_gate": self.variance_gate,
            "candidate_count": candidate_count,
            "accepted_count": self.accepted_count,
            "unknown_count": self.unknown_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "acceptance": (
                self.accepted_count / candidate_count if candidate_count else 0.0
            ),
        }


def apply_w2_uncertainty_gate(
    mean_probability: Any,
    predictive_variance: Any,
    unknown_mask: Any,
    solid_mask: Any,
    *,
    failed_frame_mask: Any | None = None,
    variance_gate: float,
    pseudo_high: float = W2_DEFAULT_PSEUDO_HIGH,
    pseudo_low: float = W2_DEFAULT_PSEUDO_LOW,
) -> W2GateResult:
    """应用固定 confidence 与单一 global variance 双门控。"""
    mean = _as_probability_tensor(mean_probability, name="mean_probability")
    variance = _as_float_tensor(
        predictive_variance, name="predictive_variance", device=mean.device)
    if tuple(variance.shape) != tuple(mean.shape):
        raise ValueError("mean_probability 与 predictive_variance shape 必须一致")
    if bool((variance < 0.0).any()) or bool((variance > 0.25).any()):
        raise ValueError("predictive_variance 必须位于 [0,0.25]")
    shape = tuple(int(size) for size in mean.shape)
    unknown = _as_bool_mask(unknown_mask, shape, name="unknown_mask", device=mean.device)
    solid = _as_bool_mask(solid_mask, shape, name="solid_mask", device=mean.device)
    if failed_frame_mask is None:
        failed = torch.zeros_like(unknown)
    else:
        failed = _as_bool_mask(
            failed_frame_mask, shape, name="failed_frame_mask", device=mean.device)
    if bool((solid & failed).any()):
        raise ValueError("W2 solid_mask 与 failed_frame_mask 不能重叠")
    if bool((solid & ~unknown).any()) or bool((failed & ~unknown).any()):
        raise ValueError("W2 solid/failed mask 必须位于 unknown 区域")
    high = _strict_probability_threshold(pseudo_high, name="pseudo_high")
    low = _strict_probability_threshold(pseudo_low, name="pseudo_low")
    if not (0.0 <= low < 0.5 < high <= 1.0):
        raise ValueError("W2 pseudo confidence 必须满足 0 <= low < 0.5 < high <= 1")
    if high != W2_DEFAULT_PSEUDO_HIGH or low != W2_DEFAULT_PSEUDO_LOW:
        raise ValueError("W2 confidence gate 必须固定为 positive=0.90/negative=0.10")
    gate = _strict_gate(variance_gate)
    candidate = unknown & ~solid & ~failed
    confidence = (mean >= high) | (mean <= low)
    low_uncertainty = variance <= gate
    pseudo = candidate & confidence & low_uncertainty
    pseudo_labels = torch.where(
        pseudo, (mean >= 0.5).to(dtype=mean.dtype), torch.zeros_like(mean))
    positive = pseudo & (mean >= 0.5)
    negative = pseudo & ~positive
    return W2GateResult(
        pseudo_mask=pseudo,
        pseudo_labels=pseudo_labels,
        candidate_mask=candidate,
        confidence_mask=candidate & confidence,
        low_uncertainty_mask=candidate & low_uncertainty,
        positive_mask=positive,
        negative_mask=negative,
        variance_gate=gate,
    )


def _as_numpy_float(value: Any, *, name: str) -> np.ndarray:
    """把 calibration 数组转为独立 float64 数组。"""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise ValueError
        array = array.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 numeric array") from exc
    if array.ndim == 0:
        raise ValueError(f"{name} 必须带有 sample 维度")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 必须全部有限")
    return array


def _as_numpy_bool(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """把 calibration mask 转为 bool 并校验 shape/0-1 语义。"""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} shape={array.shape} 与 expected={shape} 不一致")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} 必须只包含 0/1")
    return array.astype(bool, copy=True)


def _array_hash(array: np.ndarray) -> str:
    """计算 calibration 输入数组的稳定 hash。"""
    contiguous = np.ascontiguousarray(array)
    payload = json.dumps({"shape": list(contiguous.shape), "dtype": str(contiguous.dtype)},
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload + contiguous.tobytes()).hexdigest()


@dataclass(frozen=True)
class W2CalibrationRecord:
    """单个 dataset calibration Haller GT 与 W2 prediction 记录。"""

    dataset_name: str
    mean_probability: Any
    predictive_variance: Any
    labels: Any
    known_mask: Any
    split_name: str = "calibration"
    label_source: str = W2_CALIBRATION_SOURCE
    provenance: Mapping[str, Any] | None = None
    view_count: int = W2_DEFAULT_VIEW_COUNT

    def __post_init__(self) -> None:
        dataset_name = _nonempty_text(self.dataset_name, name="dataset_name")
        view_count = _strict_nonnegative_int(
            self.view_count, name="calibration view_count")
        if view_count != W2_DEFAULT_VIEW_COUNT or view_count != self.view_count:
            raise ValueError(
                "W2 calibration record view_count 必须固定为 3"
            )
        if self.split_name != "calibration":
            raise ValueError(
                "W2 calibration record 只能来自 split=calibration，test GT 不能用于 gate selection"
            )
        if self.label_source != W2_CALIBRATION_SOURCE:
            raise ValueError(
                "W2 calibration record 必须显式使用 haller_gt_calibration source"
            )
        _reject_test_source(self.provenance, context="W2 calibration provenance")
        mean = _as_numpy_float(self.mean_probability, name="mean_probability")
        variance = _as_numpy_float(
            self.predictive_variance, name="predictive_variance")
        labels = _as_numpy_float(self.labels, name="labels")
        if tuple(variance.shape) != tuple(mean.shape) or tuple(labels.shape) != tuple(mean.shape):
            raise ValueError("W2 calibration mean/variance/labels shape 必须一致")
        if not np.all((mean >= 0.0) & (mean <= 1.0)):
            raise ValueError("W2 calibration mean_probability 必须位于 [0,1]")
        known = _as_numpy_bool(self.known_mask, name="known_mask", shape=tuple(mean.shape))
        if not bool(known.any()):
            raise ValueError(f"W2 calibration dataset={dataset_name!r} 没有 known Haller cells")
        if not np.all((variance >= 0.0) & (variance <= 0.25)):
            raise ValueError("W2 calibration predictive_variance 必须位于 [0,0.25]")
        if not np.all((labels == -1.0) | (labels == 0.0) | (labels == 1.0)):
            raise ValueError("W2 calibration Haller labels 必须是 -1/0/1 三态")
        if not np.all((labels[known] == 0.0) | (labels[known] == 1.0)):
            raise ValueError("W2 calibration known Haller labels 必须为 0/1")
        object.__setattr__(self, "dataset_name", dataset_name)
        object.__setattr__(self, "mean_probability", mean)
        object.__setattr__(self, "predictive_variance", variance)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "known_mask", known)
        object.__setattr__(self, "provenance", copy.deepcopy(dict(self.provenance or {})))
        object.__setattr__(self, "view_count", view_count)

    @property
    def source(self) -> str:
        """label_source 的简短别名。"""
        return self.label_source

    @property
    def record_hash(self) -> str:
        """返回 prediction/GT/mask 联合 hash，便于复现 calibration 选择。"""
        digest = hashlib.sha256()
        digest.update(self.dataset_name.encode("utf-8"))
        digest.update(str(self.view_count).encode("ascii"))
        for array in (
            self.mean_probability,
            self.predictive_variance,
            self.labels,
            self.known_mask,
        ):
            digest.update(_array_hash(array).encode("ascii"))
        return digest.hexdigest()


def _mapping_value(mapping: Mapping[str, Any], keys: tuple[str, ...], *, name: str) -> Any:
    """从 calibration mapping 的兼容字段中取一个显式值。"""
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise ValueError(f"W2 calibration record 缺少显式 {name}")


def _coerce_calibration_record(value: Any) -> W2CalibrationRecord:
    """把 mapping/record 统一为带 calibration source guard 的记录。"""
    if isinstance(value, W2CalibrationRecord):
        # The dataclass is frozen only at the outer level; numpy arrays and
        # the provenance mapping remain mutable. Reconstruct it so a record
        # mutated after construction cannot bypass source/alias validation.
        return W2CalibrationRecord(
            dataset_name=value.dataset_name,
            mean_probability=value.mean_probability,
            predictive_variance=value.predictive_variance,
            labels=value.labels,
            known_mask=value.known_mask,
            split_name=value.split_name,
            label_source=value.label_source,
            provenance=value.provenance,
            view_count=value.view_count,
        )
    if not isinstance(value, Mapping):
        raise TypeError("W2 calibration records 必须是 W2CalibrationRecord 或 object")
    _reject_test_source(value, context="W2 calibration record")
    for forbidden in ("variance_gate", "uncertainty_gate", "dataset_gate"):
        if forbidden in value:
            raise ValueError(
                "W2 calibration record 不允许携带 per-dataset gate；gate 必须全局选择"
            )
    source = _mapping_value(value, ("label_source", "source"), name="label_source")
    split = _mapping_value(value, ("split_name", "split"), name="split_name")
    mean = _mapping_value(
        value, ("mean_probability", "mean", "prediction"), name="mean_probability")
    variance = _mapping_value(
        value, ("predictive_variance", "variance"), name="predictive_variance")
    labels = _mapping_value(
        value, ("labels", "haller_labels", "target"), name="labels")
    known = _mapping_value(
        value, ("known_mask", "valid_mask"), name="known_mask")
    return W2CalibrationRecord(
        dataset_name=_mapping_value(value, ("dataset_name", "dataset"), name="dataset_name"),
        mean_probability=mean,
        predictive_variance=variance,
        labels=labels,
        known_mask=known,
        split_name=split,
        label_source=source,
        provenance=value.get("provenance"),
        view_count=value.get("view_count", W2_DEFAULT_VIEW_COUNT),
    )


@dataclass(frozen=True)
class W2CalibrationSelection:
    """可复现的单一 global prediction threshold/variance gate 选择结果。"""

    prediction_threshold: float
    variance_gate: float
    objective_value: float
    dataset_names: tuple[str, ...]
    record_hashes: tuple[str, ...]
    candidate_count: int
    selection_hash: str
    objective: str = "f1"
    source: str = W2_CALIBRATION_SOURCE

    @property
    def uncertainty_gate(self) -> float:
        """variance_gate 的语义别名。"""
        return self.variance_gate

    def __post_init__(self) -> None:
        prediction_threshold = _strict_probability_threshold(
            self.prediction_threshold, name="prediction_threshold")
        variance_gate = _strict_gate(self.variance_gate)
        if self.source != W2_CALIBRATION_SOURCE:
            raise ValueError("W2 calibration selection source 必须是 haller_gt_calibration")
        objective = _nonempty_text(self.objective, name="objective")
        dataset_names = _nonempty_text_sequence(
            self.dataset_names, name="dataset_names")
        record_hashes = _nonempty_text_sequence(
            self.record_hashes, name="record_hashes", hash_values=True)
        candidate_count = _strict_nonnegative_int(
            self.candidate_count, name="candidate_count")
        if candidate_count <= 0:
            raise ValueError("W2 calibration candidate_count 必须为正")
        objective_value = float(self.objective_value)
        if not np.isfinite(objective_value):
            raise ValueError("W2 calibration objective_value 必须有限")
        selection_hash = _nonempty_hash(self.selection_hash, name="selection_hash")
        object.__setattr__(self, "prediction_threshold", prediction_threshold)
        object.__setattr__(self, "variance_gate", variance_gate)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "objective_value", objective_value)
        object.__setattr__(self, "dataset_names", dataset_names)
        object.__setattr__(self, "record_hashes", record_hashes)
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "selection_hash", selection_hash)

    def __getitem__(self, key: str) -> Any:
        """允许 mapping 风格读取选择结果。"""
        aliases = {"gate": "variance_gate", "uncertainty_threshold": "variance_gate"}
        key = aliases.get(key, key)
        if key == "dataset_gate_count":
            return 1
        if key == "dataset_count":
            return len(self.dataset_names)
        if key in {
            "prediction_threshold", "variance_gate", "objective_value",
            "dataset_names", "record_hashes", "candidate_count", "selection_hash",
            "objective", "source",
        }:
            return getattr(self, key)
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        """返回 checkpoint/calibration metadata 的 JSON-friendly 记录。"""
        return {
            "source": self.source,
            "objective": self.objective,
            "prediction_threshold": float(self.prediction_threshold),
            "variance_gate": float(self.variance_gate),
            "objective_value": float(self.objective_value),
            "dataset_names": list(self.dataset_names),
            "dataset_count": len(self.dataset_names),
            "dataset_gate_count": 1,
            "record_hashes": list(self.record_hashes),
            "candidate_count": int(self.candidate_count),
            "selection_hash": self.selection_hash,
        }


def _candidate_values(
    values: Any,
    *,
    name: str,
    default: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    """校验并去重排序 calibration candidate grid。"""
    if values is None:
        if default is None:
            raise ValueError(f"{name} 必须显式提供")
        values = default
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} 必须是 numeric sequence")
    try:
        sequence = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} 必须是 numeric sequence") from exc
    if not sequence:
        raise ValueError(f"{name} 不能为空")
    result = []
    for value in sequence:
        result.append(
            _strict_probability_threshold(value, name=name)
            if name == "prediction_thresholds"
            else _strict_gate(value, name=name)
        )
    return tuple(sorted(set(result)))


def _default_variance_candidates(records: list[W2CalibrationRecord]) -> tuple[float, ...]:
    """从所有 dataset 的 calibration variance 生成有限的 global candidate grid。"""
    values = np.concatenate([
        record.predictive_variance[record.known_mask].reshape(-1)
        for record in records
    ])
    if values.size == 0:
        raise ValueError("W2 calibration 没有 known variance samples")
    if values.size > 128:
        values = np.quantile(values, np.linspace(0.0, 1.0, 33))
    values = np.concatenate(([0.0], values, [float(values.max())]))
    return tuple(sorted(set(float(value) for value in values)))


def _calibration_metrics(
    records: list[W2CalibrationRecord],
    *,
    prediction_threshold: float,
    variance_gate: float,
) -> dict[str, float | int]:
    """在拼接后的所有 calibration known cells 上计算 global F1。"""
    true_positive = false_positive = false_negative = true_negative = 0
    for record in records:
        known = record.known_mask
        target = record.labels[known] >= 0.5
        prediction = (
            (record.mean_probability[known] >= prediction_threshold)
            & (record.predictive_variance[known] <= variance_gate)
        )
        true_positive += int(np.count_nonzero(prediction & target))
        false_positive += int(np.count_nonzero(prediction & ~target))
        false_negative += int(np.count_nonzero(~prediction & target))
        true_negative += int(np.count_nonzero(~prediction & ~target))
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2.0 * true_positive / denominator if denominator else 0.0
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def calibrate_w2_gate(
    records: Any,
    *,
    prediction_thresholds: Any = (0.5,),
    variance_candidates: Any | None = None,
) -> W2CalibrationSelection:
    """只用 calibration Haller GT 选择一个 global prediction/variance gate。

    所有记录先跨 dataset 合并再评分；接口不接受 per-dataset gate，且显式
    拒绝 ``haller_gt_test``。该函数纯计算，不持有 model/optimizer，也不会
    产生 optimizer step。
    """
    if isinstance(records, Mapping):
        records = [records]
    if isinstance(records, (str, bytes)):
        raise TypeError("W2 calibration records 必须是 record sequence")
    try:
        values = list(records)
    except TypeError as exc:
        raise TypeError("W2 calibration records 必须是 record sequence") from exc
    if not values:
        raise ValueError("W2 calibration records 不能为空")
    normalized = [_coerce_calibration_record(value) for value in values]
    normalized.sort(key=lambda record: (record.dataset_name, record.record_hash))
    thresholds = _candidate_values(
        prediction_thresholds, name="prediction_thresholds", default=(0.5,))
    gates = _candidate_values(
        _default_variance_candidates(normalized)
        if variance_candidates is None else variance_candidates,
        name="variance_candidates",
    )
    rows = []
    for threshold in thresholds:
        for gate in gates:
            metrics = _calibration_metrics(
                normalized,
                prediction_threshold=threshold,
                variance_gate=gate,
            )
            rows.append({
                "prediction_threshold": threshold,
                "variance_gate": gate,
                **metrics,
            })
    # 先最大化 global F1；并列时优先较小 variance gate，再优先接近 0.5
    # 的 prediction threshold，排序规则固定以便跨运行复现。
    best = min(rows, key=lambda row: (
        -float(row["f1"]),
        float(row["variance_gate"]),
        abs(float(row["prediction_threshold"]) - 0.5),
        float(row["prediction_threshold"]),
    ))
    dataset_names = tuple(sorted({record.dataset_name for record in normalized}))
    record_hashes = tuple(record.record_hash for record in normalized)
    payload = {
        "source": W2_CALIBRATION_SOURCE,
        "objective": "f1",
        "prediction_threshold": best["prediction_threshold"],
        "variance_gate": best["variance_gate"],
        "objective_value": best["f1"],
        "dataset_names": list(dataset_names),
        "record_hashes": list(record_hashes),
        "candidate_count": len(rows),
        "metrics": {
            key: best[key]
            for key in (
                "precision", "recall", "true_positive", "false_positive",
                "false_negative", "true_negative",
            )
        },
    }
    selection_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return W2CalibrationSelection(
        prediction_threshold=float(best["prediction_threshold"]),
        variance_gate=float(best["variance_gate"]),
        objective_value=float(best["f1"]),
        dataset_names=dataset_names,
        record_hashes=record_hashes,
        candidate_count=len(rows),
        selection_hash=selection_hash,
    )


# Explicit alias used by callers that name the selected value rather than the
# operation; both names retain the same single-global-gate contract.
select_global_w2_gate = calibrate_w2_gate


@dataclass
class W2Batch:
    """W2 training batch：W1-H Haller train anchors + 7-channel features。"""

    contract_batch: contract.WeakSupervisionBatch
    solid_mask: Any
    failed_frame_mask: Any
    anchor_hash: str
    dummy_field: Any | None = None
    anchor_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        batch = contract.validate_training_batch(
            self.contract_batch, contract.MODE_W2)
        if batch.label_source != W2_LABEL_SOURCE:
            raise ValueError("W2 formal loss source 必须是 haller_anchor_train")
        sampling_source = batch.sampling_source
        if sampling_source not in W2_ALLOWED_SAMPLING_SOURCES:
            raise ValueError(
                f"W2 sampling_source 必须是 {sorted(W2_ALLOWED_SAMPLING_SOURCES)!r}"
            )
        window = batch.provenance.get("window")
        if window is not None:
            batch.provenance["window"] = _validate_w2_window_provenance(
                window,
                split_name=batch.split_name,
                sampling_source=sampling_source,
            )
        else:
            batch.provenance["windows"] = _validate_w2_collated_windows(
                batch.provenance,
                split_name=batch.split_name,
                sampling_source=sampling_source,
            )
        self.anchor_hash = _nonempty_hash(self.anchor_hash, name="anchor_hash")
        shape = tuple(int(value) for value in batch.labels.shape)
        device = batch.labels.device if isinstance(batch.labels, torch.Tensor) else torch.device("cpu")
        self.solid_mask = _as_bool_mask(
            self.solid_mask, shape, name="solid_mask", device=device)
        self.failed_frame_mask = _as_bool_mask(
            self.failed_frame_mask, shape, name="failed_frame_mask", device=device)
        known = _as_bool_mask(batch.label_mask, shape, name="label_mask", device=device)
        unknown = _as_bool_mask(batch.unknown_mask, shape, name="unknown_mask", device=device)
        if bool((self.solid_mask & known).any()) or bool((self.solid_mask & ~unknown).any()):
            raise ValueError("W2 solid_mask 必须只落在 unknown/ignored 区域")
        if bool((self.failed_frame_mask & known).any()) or bool((self.failed_frame_mask & ~unknown).any()):
            raise ValueError("W2 failed_frame_mask 必须只落在 unknown/ignored 区域")
        if bool((self.failed_frame_mask & self.solid_mask).any()):
            raise ValueError("W2 failed_frame_mask 必须与 solid_mask 分离")
        anchor_provenance = dict(batch.provenance.get("anchor", {}))
        if anchor_provenance.get("source") not in (None, W2_LABEL_SOURCE):
            raise ValueError("W2 anchor provenance source 必须是 haller_anchor_train")
        if anchor_provenance.get("anchor_hash") not in (None, self.anchor_hash):
            raise ValueError("W2 anchor provenance hash 与 batch anchor_hash 不一致")
        sampling_provenance = batch.provenance.get("sampling", {})
        if not isinstance(sampling_provenance, Mapping):
            raise ValueError("W2 sampling provenance 必须是 object")
        if sampling_provenance.get("source") not in (None, sampling_source):
            raise ValueError(
                "W2 sampling provenance source 与 batch sampling_source 不一致"
            )
        _reject_test_source(batch.provenance, context="W2 training provenance")
        self.anchor_metadata = _validate_checkpoint_anchor_metadata(
            self.anchor_metadata
        )
        if self.dummy_field is None:
            if isinstance(batch.pathlines, torch.Tensor):
                self.dummy_field = batch.pathlines.new_zeros((shape[0], 1, 1, 1))
            else:
                self.dummy_field = np.zeros((shape[0], 1, 1, 1), dtype=np.float32)
        dummy_shape = getattr(self.dummy_field, "shape", ())
        if len(dummy_shape) < 1 or int(dummy_shape[0]) != shape[0]:
            raise ValueError("W2 dummy_field batch 维度与 labels 不一致")

    @classmethod
    def from_w1_h_batch(cls, batch: W1HBatch) -> "W2Batch":
        """显式把已通过 W1-H train guard 的 batch升级为 W2 mode。"""
        if not isinstance(batch, W1HBatch):
            raise TypeError("W2.from_w1_h_batch 必须消费 W1HBatch")
        base = batch.contract_batch
        converted = contract.WeakSupervisionBatch(
            pathlines=base.pathlines,
            labels=base.labels,
            label_source=base.label_source,
            split_name=base.split_name,
            feature_schema=base.feature_schema,
            label_mask=base.label_mask,
            unknown_mask=base.unknown_mask,
            sampling_source=base.sampling_source,
            provenance=base.provenance,
            mode=contract.MODE_W2,
            input_schema=base.input_schema,
        )
        return cls(
            converted,
            batch.solid_mask,
            batch.failed_frame_mask,
            batch.anchor_hash,
            dummy_field=batch.dummy_field,
            anchor_metadata=batch.anchor_metadata,
        )

    @property
    def pathlines(self) -> Any:
        """model-facing 7-channel local-IVD pathlines。"""
        return self.contract_batch.pathlines

    @property
    def labels(self) -> Any:
        """Haller anchor labels；unknown 区域仅为 BCE 占位。"""
        return self.contract_batch.labels

    @property
    def label_mask(self) -> Any:
        """known Haller positive/negative mask。"""
        return self.contract_batch.label_mask

    @property
    def unknown_mask(self) -> Any:
        """Haller boundary/solid/failed unknown mask。"""
        return self.contract_batch.unknown_mask

    @property
    def sampling_source(self) -> str:
        """patch sampling source，与 formal Haller source 分离。"""
        return str(self.contract_batch.sampling_source)

    @property
    def label_source(self) -> str:
        """formal W2 loss source。"""
        return self.contract_batch.label_source

    def as_dict(self) -> dict[str, Any]:
        """返回不含大数组的 batch provenance 摘要。"""
        result = self.contract_batch.as_dict()
        total = int(np.prod(self.labels.shape))
        result.update({
            "mode": contract.MODE_W2,
            "anchor_hash": self.anchor_hash,
            "solid_count": _mask_count(self.solid_mask),
            "failed_cell_count": _mask_count(self.failed_frame_mask),
            "anchor_coverage": (
                _mask_count(self.label_mask) / total if total else 0.0
            ),
            "unknown_coverage": (
                _mask_count(self.unknown_mask) / total if total else 0.0
            ),
            "uncertainty_views": W2_DEFAULT_VIEW_COUNT,
        })
        if self.anchor_metadata is not None:
            result["anchor_metadata"] = copy.deepcopy(dict(self.anchor_metadata))
        return result

    def to(self, device: str | torch.device) -> "W2Batch":
        """把 batch 数组搬到 device，保留 split/source/provenance。"""
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
            mode=contract.MODE_W2,
            input_schema=batch.input_schema,
        )
        return W2Batch(
            converted,
            torch.as_tensor(self.solid_mask, device=device),
            torch.as_tensor(self.failed_frame_mask, device=device),
            self.anchor_hash,
            torch.as_tensor(self.dummy_field, device=device),
            self.anchor_metadata,
        )


def build_w2_batch(
    pathlines: Any,
    labels: Any,
    label_mask: Any,
    unknown_mask: Any,
    solid_mask: Any,
    *,
    failed_frame_mask: Any | None = None,
    sampling_source: str,
    split_name: str = "train",
    anchor_hash: str,
    provenance: Mapping[str, Any] | None = None,
    anchor_metadata: Mapping[str, Any] | None = None,
    dummy_field: Any | None = None,
) -> W2Batch:
    """构造显式 W2 train batch，formal source 固定为 Haller train。"""
    sampling_source = contract.validate_label_source(sampling_source)
    if sampling_source not in W2_ALLOWED_SAMPLING_SOURCES:
        raise ValueError(
            f"W2 sampling_source 必须是 {sorted(W2_ALLOWED_SAMPLING_SOURCES)!r}"
        )
    anchor_hash = _nonempty_hash(anchor_hash, name="anchor_hash")
    if failed_frame_mask is None:
        if isinstance(labels, torch.Tensor):
            failed_frame_mask = torch.zeros_like(labels, dtype=torch.bool)
        else:
            failed_frame_mask = np.zeros_like(np.asarray(labels), dtype=bool)
    sources = copy.deepcopy(dict(provenance or {}))
    anchor = dict(sources.get("anchor", {}))
    anchor.setdefault("source", W2_LABEL_SOURCE)
    anchor.setdefault("anchor_hash", anchor_hash)
    sources["anchor"] = anchor
    sampling = dict(sources.get("sampling", {}))
    sampling.setdefault("source", sampling_source)
    sources["sampling"] = sampling
    base = contract.WeakSupervisionBatch(
        pathlines=pathlines,
        labels=labels,
        label_source=W2_LABEL_SOURCE,
        split_name=split_name,
        feature_schema=contract.FEATURE_SCHEMA_7,
        label_mask=label_mask,
        unknown_mask=unknown_mask,
        sampling_source=sampling_source,
        provenance=sources,
        mode=contract.MODE_W2,
        input_schema=contract.FEATURE_SCHEMA_7,
    )
    return W2Batch(
        base,
        solid_mask,
        failed_frame_mask,
        anchor_hash,
        dummy_field=dummy_field,
        anchor_metadata=anchor_metadata,
    )


def build_w2_batch_from_w1_h(batch: W1HBatch) -> W2Batch:
    """显式 W1-H → W2 batch adapter。"""
    return W2Batch.from_w1_h_batch(batch)


def _coerce_w2_batch(batch: Any) -> W2Batch:
    """让 W2 seam 接受显式 W2 batch 或明确升级的 W1-H batch。"""
    if isinstance(batch, W2Batch):
        return batch
    if isinstance(batch, W1HBatch):
        return W2Batch.from_w1_h_batch(batch)
    raise TypeError("W2 loss/trainer 必须消费 W2Batch 或 W1HBatch")


def _coerce_w2_config(config: W2Config | None) -> W2Config:
    """校验 W2 config，并拒绝没有 calibration/显式 gate 的隐式默认。"""
    config = W2Config() if config is None else config
    if not isinstance(config, W2Config):
        raise TypeError("W2 config 必须是 W2Config")
    if config.variance_gate is None:
        raise ValueError(
            "W2 必须显式提供 calibration-selected global variance_gate；禁止隐式默认"
        )
    return config


def _coerce_statistics(
    teacher_predictions: Any,
    *,
    expected_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> W2Statistics:
    """把三视图或预计算统计搬到 student 的 shape/device/dtype。"""
    statistics = (
        teacher_predictions
        if isinstance(teacher_predictions, W2Statistics)
        else compute_w2_statistics(teacher_predictions)
    )
    if tuple(statistics.mean_probability.shape) != expected_shape:
        raise ValueError(
            "W2 teacher statistics shape 不匹配："
            f"expected={expected_shape} actual={tuple(statistics.mean_probability.shape)}"
        )
    return W2Statistics(
        statistics.mean_probability.to(device=device, dtype=dtype),
        statistics.predictive_variance.to(device=device, dtype=dtype),
        statistics.entropy.to(device=device, dtype=dtype),
        view_count=statistics.view_count,
    )


def compute_w2_loss(
    student_predictions: Any,
    teacher_predictions: Any,
    batch: Any,
    *,
    config: W2Config | None = None,
    epoch: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """计算 Haller anchor + uncertainty-gated pseudo/consistency loss。"""
    batch = _coerce_w2_batch(batch)
    config = _coerce_w2_config(config)
    epoch = _strict_nonnegative_int(epoch, name="epoch")
    expected_shape = tuple(int(value) for value in batch.labels.shape)
    student = _as_prediction_tensor(
        student_predictions, expected_shape, "student_predictions")
    statistics = _coerce_statistics(
        teacher_predictions,
        expected_shape=expected_shape,
        device=student.device,
        dtype=student.dtype,
    )
    teacher_mean = statistics.mean_probability.detach()
    gate = apply_w2_uncertainty_gate(
        teacher_mean,
        statistics.predictive_variance.detach(),
        batch.unknown_mask,
        batch.solid_mask,
        failed_frame_mask=batch.failed_frame_mask,
        variance_gate=config.variance_gate,
    )
    labels = torch.as_tensor(batch.labels, device=student.device).float()
    known = torch.as_tensor(batch.label_mask, device=student.device, dtype=torch.bool)
    unknown = torch.as_tensor(batch.unknown_mask, device=student.device, dtype=torch.bool)
    solid = torch.as_tensor(batch.solid_mask, device=student.device, dtype=torch.bool)
    failed = torch.as_tensor(
        batch.failed_frame_mask, device=student.device, dtype=torch.bool)
    contract.validate_training_batch(batch.contract_batch, contract.MODE_W2)
    if bool((solid & known).any()) or bool((failed & known).any()):
        raise ValueError("W2 solid/failed mask 不能贡献 known anchor BCE")
    anchor_loss = _masked_bce(student, labels, known)
    pseudo_loss = _masked_bce(student, gate.pseudo_labels, gate.pseudo_mask)
    # 不确定的 unknown 只用于诊断；consistency 也只消费已通过双门控的
    # accepted pseudo-label，从而不绕过 variance gate 影响训练。
    consistency_loss = (
        F.mse_loss(student[gate.pseudo_mask], teacher_mean[gate.pseudo_mask])
        if bool(gate.pseudo_mask.any()) else student.sum() * 0.0
    )
    ramp = ramp_up_weight(epoch, config.ramp_up_epochs)
    total = (
        anchor_loss
        + ramp * config.pseudo_weight * pseudo_loss
        + ramp * config.consistency_weight * consistency_loss
    )
    candidate = gate.candidate_mask
    candidate_count = _mask_count(candidate)
    accepted_count = gate.accepted_count
    total_count = int(labels.numel())
    mean_distribution = _distribution(statistics.mean_probability)
    variance_distribution = _distribution(statistics.predictive_variance)
    entropy_distribution = _distribution(statistics.entropy)
    disagreement = (
        float(torch.abs(student[candidate] - teacher_mean[candidate]).mean().detach().cpu())
        if bool(candidate.any()) else 0.0
    )
    accepted_disagreement = (
        float(torch.abs(student[gate.pseudo_mask] - teacher_mean[gate.pseudo_mask]).mean().detach().cpu())
        if bool(gate.pseudo_mask.any()) else 0.0
    )
    anchor_metadata = dict(batch.anchor_metadata or {})
    anchor_provenance = dict(batch.contract_batch.provenance.get("anchor", {}))
    artifact_failure_count = _strict_nonnegative_int(
        anchor_metadata.get(
            "failure_count", anchor_provenance.get("failure_count", 0)),
        name="artifact_failure_count",
    )
    stats = {
        "loss": float(total.detach().cpu()),
        "anchor_loss": float(anchor_loss.detach().cpu()),
        "pseudo_loss": float(pseudo_loss.detach().cpu()),
        "consistency_loss": float(consistency_loss.detach().cpu()),
        "ramp_weight": ramp,
        "view_count": W2_DEFAULT_VIEW_COUNT,
        "variance_gate": float(config.variance_gate),
        "pseudo_high": W2_DEFAULT_PSEUDO_HIGH,
        "pseudo_low": W2_DEFAULT_PSEUDO_LOW,
        "candidate_count": candidate_count,
        "pseudo_eligible_count": candidate_count,
        "pseudo_accepted_count": accepted_count,
        "pseudo_positive_count": gate.positive_count,
        "pseudo_negative_count": gate.negative_count,
        "unknown_count": gate.unknown_count,
        "pseudo_acceptance": (
            accepted_count / candidate_count if candidate_count else 0.0
        ),
        "gate_acceptance": (
            accepted_count / candidate_count if candidate_count else 0.0
        ),
        "pseudo_positive_ratio": (
            gate.positive_count / accepted_count if accepted_count else 0.0
        ),
        "pseudo_negative_ratio": (
            gate.negative_count / accepted_count if accepted_count else 0.0
        ),
        "anchor_count": _mask_count(known),
        "anchor_positive_count": _mask_count(known & (labels >= 0.5)),
        "anchor_negative_count": _mask_count(known & (labels < 0.5)),
        "unknown_cell_count": _mask_count(unknown),
        "solid_count": _mask_count(solid),
        "failed_cell_count": _mask_count(failed),
        "artifact_failure_count": artifact_failure_count,
        "anchor_coverage": _mask_count(known) / total_count if total_count else 0.0,
        "unknown_coverage": _mask_count(unknown) / total_count if total_count else 0.0,
        "solid_coverage": _mask_count(solid) / total_count if total_count else 0.0,
        "failed_cell_coverage": _mask_count(failed) / total_count if total_count else 0.0,
        "teacher_student_disagreement": disagreement,
        "accepted_teacher_student_disagreement": accepted_disagreement,
        "mean_probability_mean": mean_distribution["mean"],
        "mean_probability_std": mean_distribution["std"],
        "mean_probability_min": mean_distribution["min"],
        "mean_probability_max": mean_distribution["max"],
        "predictive_variance_mean": variance_distribution["mean"],
        "predictive_variance_std": variance_distribution["std"],
        "predictive_variance_min": variance_distribution["min"],
        "predictive_variance_max": variance_distribution["max"],
        "entropy_mean": entropy_distribution["mean"],
        "entropy_std": entropy_distribution["std"],
        "entropy_min": entropy_distribution["min"],
        "entropy_max": entropy_distribution["max"],
        "sampling_source": batch.sampling_source,
        "loss_source": batch.label_source,
        "anchor_hash": batch.anchor_hash,
        "anchor_algorithm_version": anchor_metadata.get(
            "algorithm_version", anchor_provenance.get("algorithm_version")),
        "anchor_parameter_hash": anchor_metadata.get(
            "parameter_hash", anchor_provenance.get("parameter_hash")),
        "anchor_input_hash": anchor_metadata.get(
            "input_hash", anchor_provenance.get("input_hash")),
        "anchor_mask_hash": anchor_metadata.get(
            "mask_hash", anchor_provenance.get("mask_hash")),
    }
    return total, stats


def _forward_w2_model(model: nn.Module, batch: W2Batch) -> Any:
    """调用现有 (dummy_field, 7-channel pathlines) model seam。"""
    if isinstance(model, contract.ChannelSelectingAdapter):
        return model.forward_batch(
            batch.contract_batch, dummy_field=batch.dummy_field, consumer="train")
    return model((batch.dummy_field, batch.pathlines))


def _rng_devices_for_batch(batch: W2Batch) -> list[int]:
    """返回需要由 fork_rng 保存/恢复的 CUDA generator 列表。"""
    pathlines = batch.pathlines
    if not isinstance(pathlines, torch.Tensor) or pathlines.device.type != "cuda":
        return []
    return list(range(torch.cuda.device_count()))


def _policy_as_dict(
    policy: W2CalibrationSelection | Mapping[str, Any] | None,
    *,
    variance_gate: float,
) -> dict[str, Any]:
    """规范化并校验 checkpoint calibration policy 的单 global gate。"""
    if isinstance(policy, W2CalibrationSelection):
        result = policy.as_dict()
    elif policy is None:
        raise ValueError(
            "W2 checkpoint 必须显式提供 haller_gt_calibration policy；"
            "禁止使用 source=none 绕过 global gate calibration"
        )
    elif isinstance(policy, Mapping):
        result = copy.deepcopy(dict(policy))
    else:
        raise TypeError("W2 calibration_policy 必须是 W2CalibrationSelection 或 object")
    _reject_test_source(result, context="W2 calibration_policy")
    if result.get("source") != W2_CALIBRATION_SOURCE:
        raise ValueError(
            "W2 calibration_policy source 必须是 haller_gt_calibration；"
            "test/none/其他 source 不能用于 global gate"
        )
    if "variance_gate" not in result:
        raise ValueError("W2 calibration_policy 必须显式记录 variance_gate")
    policy_gate = _strict_gate(result["variance_gate"], name="calibration_policy.variance_gate")
    if policy_gate != variance_gate:
        raise ValueError(
            "W2 calibration_policy.variance_gate 与 trainer global variance_gate 不一致"
        )
    if result.get("dataset_gate_count", 1) != 1:
        raise ValueError("W2 calibration_policy 只能包含一个 global gate")
    if "dataset_gates" in result or "per_dataset_gate" in result:
        raise ValueError("W2 calibration_policy 禁止 per-dataset gate")
    if "prediction_threshold" not in result:
        raise ValueError(
            "W2 calibration_policy 必须显式记录 global prediction_threshold"
        )
    _strict_probability_threshold(
        result["prediction_threshold"], name="calibration_policy.prediction_threshold")
    required_provenance = (
        "dataset_names", "record_hashes", "candidate_count", "selection_hash")
    missing_provenance = [
        field for field in required_provenance if field not in result]
    if missing_provenance:
        raise ValueError(
            "W2 calibration_policy 必须包含可复现 provenance："
            f"{missing_provenance!r}"
        )
    dataset_names = _nonempty_text_sequence(
        result["dataset_names"], name="calibration_policy.dataset_names")
    record_hashes = _nonempty_text_sequence(
        result["record_hashes"],
        name="calibration_policy.record_hashes",
        hash_values=True,
    )
    candidate_count = _strict_nonnegative_int(
        result["candidate_count"], name="calibration_policy.candidate_count")
    if candidate_count <= 0:
        raise ValueError("W2 calibration_policy.candidate_count 必须为正")
    selection_hash = _nonempty_hash(
        result["selection_hash"], name="calibration_policy.selection_hash")
    result["source"] = W2_CALIBRATION_SOURCE
    result["variance_gate"] = policy_gate
    result["dataset_gate_count"] = 1
    result["dataset_names"] = list(dataset_names)
    result["dataset_count"] = len(dataset_names)
    result["record_hashes"] = list(record_hashes)
    result["candidate_count"] = candidate_count
    result["selection_hash"] = selection_hash
    return result


def _policy_gate(policy: W2CalibrationSelection | Mapping[str, Any] | None) -> float:
    """Read the selected calibration gate without conflating it with train config."""
    if isinstance(policy, W2CalibrationSelection):
        return _strict_gate(policy.variance_gate, name="calibration_policy.variance_gate")
    if isinstance(policy, Mapping) and "variance_gate" in policy:
        return _strict_gate(policy["variance_gate"], name="calibration_policy.variance_gate")
    raise ValueError(
        "W2 checkpoint 必须显式提供 calibration_policy.variance_gate"
    )


def _validate_w2_window_provenance(
    window: Any,
    *,
    split_name: str,
    sampling_source: str,
) -> dict[str, Any]:
    """Require a complete, train-contained window before W2 consumes a batch."""
    if not isinstance(window, Mapping):
        raise ValueError(
            "W2 batch 必须显式携带 provenance.window；禁止隐式 split/window fallback"
        )
    required = (
        "dataset_name", "split_name", "frame_start", "frame_end",
        "split_start", "split_end", "t_win", "window_step",
        "generation_version", "generation_hash", "contract_hash",
        "feature_schema", "label_source",
    )
    missing = [field for field in required if field not in window]
    if missing:
        raise ValueError(f"W2 window provenance 缺少字段：{missing!r}")
    if split_name != "train" or window["split_name"] != split_name:
        raise ValueError(
            "W2 training window 必须是 split=train，且与 batch split_name 一致："
            f"batch={split_name!r} window={window['split_name']!r}"
        )
    frame_start = _strict_nonnegative_int(
        window["frame_start"], name="W2 window.frame_start")
    validated = _validate_train_window_metadata(window, frame=frame_start)
    if validated["t_win"] <= 0:
        raise ValueError("W2 window.t_win 必须是正整数")
    _nonempty_text(window["dataset_name"], name="W2 window.dataset_name")
    for field in ("generation_version", "generation_hash", "contract_hash"):
        _nonempty_text(window[field], name=f"W2 window.{field}")
    try:
        contract.validate_feature_schema(
            window["feature_schema"], contract.FEATURE_SCHEMA_7
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "W2 window feature schema 必须是当前 5×5 local-IVD 的 canonical 7-channel schema"
        ) from exc
    window_source = contract.validate_label_source(window["label_source"])
    if window_source != sampling_source:
        raise ValueError(
            "W2 window.label_source 必须与 sampling_source 一致："
            f"window={window_source!r} sampling={sampling_source!r}"
        )
    if window_source not in W2_ALLOWED_SAMPLING_SOURCES:
        raise ValueError(
            "W2 window.label_source 不能是 formal/calibration/test Haller source："
            f"{window_source!r}"
        )
    return validated


def _validate_w2_collated_windows(
    provenance: Mapping[str, Any],
    *,
    split_name: str,
    sampling_source: str,
) -> list[dict[str, Any]]:
    """Validate every explicit window retained by W1-H's collated batch seam."""
    batches = provenance.get("batches")
    if not isinstance(batches, (list, tuple)) or not batches:
        raise ValueError(
            "W2 batch 必须显式携带 provenance.window 或逐 batch windows；"
            "禁止隐式 split/window fallback"
        )
    validated = []
    for index, item in enumerate(batches):
        if not isinstance(item, Mapping) or "window" not in item:
            raise ValueError(
                f"W2 collated provenance.batches[{index}] 缺少显式 window metadata"
            )
        validated.append(
            _validate_w2_window_provenance(
                item["window"],
                split_name=split_name,
                sampling_source=sampling_source,
            )
        )
    return validated


def _validate_checkpoint_anchor_metadata(
    metadata: Any,
) -> dict[str, Any]:
    """校验 W2 checkpoint 必须保存的 Haller train artifact provenance。"""
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "W2 checkpoint 必须显式提供 Haller anchor metadata；"
            "metadata 必须是 object"
        )
    result = copy.deepcopy(dict(metadata))
    if result.get("source") != W2_LABEL_SOURCE:
        raise ValueError(
            "W2 checkpoint Haller anchor metadata source 必须是 haller_anchor_train"
        )
    for field in ("algorithm_version", "parameter_hash", "input_hash", "mask_hash"):
        result[field] = _nonempty_text(
            result.get(field), name=f"anchor_metadata.{field}")
    result["failure_count"] = _strict_nonnegative_int(
        result.get("failure_count"), name="anchor_metadata.failure_count")
    coverage = result.get("coverage")
    if isinstance(coverage, Mapping):
        coverage = copy.deepcopy(dict(coverage))
        count_fields = (
            "fluid_cells", "known_cells", "negative_cells", "positive_cells",
            "solid_cells", "total_unknown_cells_including_solid", "unknown_cells",
        )
        fraction_fields = (
            "known_fraction_fluid", "negative_fraction_fluid",
            "positive_fraction_fluid", "unknown_fraction_fluid",
        )
        missing = [
            field for field in (*count_fields, *fraction_fields)
            if field not in coverage
        ]
        if missing:
            raise ValueError(
                "anchor_metadata.coverage mapping 缺少 canonical 字段："
                f"{missing!r}"
            )
        for field in count_fields:
            coverage[field] = _strict_nonnegative_int(
                coverage[field], name=f"anchor_metadata.coverage.{field}")
        for field in fraction_fields:
            try:
                value = float(coverage[field])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"anchor_metadata.coverage.{field} 必须是 [0,1] 内的有限数"
                ) from exc
            if not np.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"anchor_metadata.coverage.{field} 必须是 [0,1] 内的有限数"
                )
            coverage[field] = value
        if coverage["known_cells"] + coverage["unknown_cells"] != coverage["fluid_cells"]:
            raise ValueError(
                "anchor_metadata.coverage known/unknown cells 与 fluid_cells 不一致"
            )
        if coverage["positive_cells"] + coverage["negative_cells"] != coverage["known_cells"]:
            raise ValueError(
                "anchor_metadata.coverage positive/negative cells 与 known_cells 不一致"
            )
        if (coverage["unknown_cells"] + coverage["solid_cells"]
                != coverage["total_unknown_cells_including_solid"]):
            raise ValueError(
                "anchor_metadata.coverage unknown/solid cells 与 total_unknown 不一致"
            )
    else:
        try:
            coverage = float(coverage)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("anchor_metadata.coverage 必须是 [0,1] 内的有限数或 canonical mapping") from exc
        if not np.isfinite(coverage) or not (0.0 <= coverage <= 1.0):
            raise ValueError("anchor_metadata.coverage 必须是 [0,1] 内的有限数")
    result["coverage"] = coverage
    literature = result.get("literature")
    if not isinstance(literature, Mapping) or literature.get("status") != "pending_verification":
        raise ValueError(
            "W2 Haller anchor metadata literature.status 必须保留 pending_verification"
        )
    if result.get("legacy_p85_used") is not False:
        raise ValueError("W2 Haller anchor metadata 禁止 legacy_p85 fallback")
    if result.get("fallback_used") is not None:
        raise ValueError("W2 Haller anchor metadata 不允许 fallback")
    _reject_test_source(result, context="W2 checkpoint anchor_metadata")
    return result


def _anchor_metadata_identity(metadata: Mapping[str, Any]) -> tuple[Any, ...]:
    """返回跨 frame anchor manifest 可比较的、非 frame-specific 身份。"""
    normalized = _validate_checkpoint_anchor_metadata(metadata)
    literature = normalized["literature"]
    return (
        normalized["source"],
        normalized["algorithm_version"],
        normalized["parameter_hash"],
        json.dumps(literature, sort_keys=True, separators=(",", ":")),
        normalized["legacy_p85_used"],
        normalized["fallback_used"],
    )


class W2Trainer:
    """W2 可恢复 teacher trainer：每批固定 3 个 stochastic teacher views。"""

    def __init__(
        self,
        student: nn.Module,
        optimizer: Any,
        *,
        sampling_source: str,
        anchor_hash: str,
        config: W2Config | None = None,
        teacher: nn.Module | None = None,
        scheduler: Any | None = None,
        seed: int = 0,
        anchor_metadata: Mapping[str, Any] | None = None,
        calibration_selection: W2CalibrationSelection | Mapping[str, Any] | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        if not isinstance(student, nn.Module):
            raise TypeError("W2 student 必须是 torch.nn.Module")
        if optimizer is None or not all(
            hasattr(optimizer, attr) for attr in ("zero_grad", "step", "state_dict")
        ):
            raise TypeError("W2 optimizer 必须提供 zero_grad()/step()/state_dict()")
        if teacher is not None and not isinstance(teacher, nn.Module):
            raise TypeError("W2 teacher 必须是 torch.nn.Module")
        self.config = _coerce_w2_config(config)
        sampling_source = contract.validate_label_source(sampling_source)
        if sampling_source not in W2_ALLOWED_SAMPLING_SOURCES:
            raise ValueError(
                f"W2 sampling_source 必须是 {sorted(W2_ALLOWED_SAMPLING_SOURCES)!r}"
            )
        self.student = student
        self.teacher = clone_ema_teacher(student) if teacher is None else teacher
        if self.teacher is self.student:
            raise ValueError("W2 teacher 不能与 student 共享同一 module")
        _prepare_ema_teacher(self.teacher)
        if tuple(self.student.state_dict()) != tuple(self.teacher.state_dict()):
            raise ValueError("W2 student/teacher state_dict keys 不一致")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sampling_source = sampling_source
        self.anchor_hash = _nonempty_hash(anchor_hash, name="anchor_hash")
        self.anchor_metadata = (
            None if anchor_metadata is None
            else copy.deepcopy(dict(anchor_metadata))
        )
        if self.anchor_metadata is not None:
            if self.anchor_metadata.get("source") not in (None, W2_LABEL_SOURCE):
                raise ValueError("W2 anchor_metadata source 必须是 haller_anchor_train")
            for field in ("algorithm_version", "parameter_hash", "input_hash", "mask_hash"):
                if field not in self.anchor_metadata:
                    raise ValueError(f"W2 anchor_metadata 缺少 {field}")
            _reject_test_source(self.anchor_metadata, context="W2 anchor_metadata")
        if calibration_selection is not None:
            if isinstance(calibration_selection, W2CalibrationSelection):
                selection = calibration_selection
            elif isinstance(calibration_selection, Mapping):
                selection = dict(calibration_selection)
            else:
                raise TypeError("W2 calibration_selection 必须是 selection 或 object")
            selection_policy = _policy_as_dict(
                selection, variance_gate=_policy_gate(selection)
            )
            self.calibration_selection = (
                copy.deepcopy(selection)
                if isinstance(selection, W2CalibrationSelection)
                else selection_policy
            )
        else:
            self.calibration_selection = None
        self.seed = _strict_nonnegative_int(seed, name="seed")
        if grad_clip_norm is not None:
            grad_clip_norm = float(grad_clip_norm)
            if not np.isfinite(grad_clip_norm) or grad_clip_norm <= 0.0:
                raise ValueError("W2 grad_clip_norm 必须是正的有限数或 None")
        self.grad_clip_norm = grad_clip_norm
        self.global_step = 0
        self.last_metrics: dict[str, Any] | None = None

    def _move_models(self, device: str | torch.device) -> None:
        self.student.to(device)
        self.teacher.to(device)
        _prepare_ema_teacher(self.teacher)

    def _validate_batch_identity(self, batch: W2Batch) -> None:
        """校验 batch 与 trainer 的 sampling/anchor contract 不发生漂移。"""
        if batch.sampling_source != self.sampling_source:
            raise ValueError(
                f"W2 sampling_source 不匹配：trainer={self.sampling_source!r} "
                f"batch={batch.sampling_source!r}"
            )
        if batch.anchor_hash != self.anchor_hash:
            raise ValueError(
                f"W2 anchor_hash 不匹配：trainer={self.anchor_hash!r} "
                f"batch={batch.anchor_hash!r}"
            )
        if self.anchor_metadata is None:
            self.anchor_metadata = copy.deepcopy(batch.anchor_metadata)
        elif _anchor_metadata_identity(self.anchor_metadata) != _anchor_metadata_identity(
                batch.anchor_metadata):
            raise ValueError(
                "W2 batch Haller anchor manifest metadata 与 trainer metadata 不一致"
            )

    def predict_teacher_views(
        self,
        batch: Any,
        *,
        device: str | torch.device = "cpu",
    ) -> list[torch.Tensor]:
        """生成固定三次、逐 view 可复现且显式持有 RNG 的 teacher prediction。"""
        batch = _coerce_w2_batch(batch)
        self._validate_batch_identity(batch)
        self._move_models(device)
        model_batch = batch.to(device)
        _prepare_ema_teacher(self.teacher)
        self.teacher.eval()
        expected_shape = tuple(int(value) for value in model_batch.labels.shape)
        rng_devices = _rng_devices_for_batch(model_batch)
        views: list[torch.Tensor] = []
        for view_index in range(self.config.view_count):
            seed = int(
                (self.seed
                 + 1_000_003 * (self.global_step + 1)
                 + 10_007 * (view_index + 1))
                % (2**63 - 1)
            )
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(seed)
                with torch.no_grad():
                    prediction = _forward_w2_model(self.teacher, model_batch)
            views.append(
                _as_prediction_tensor(
                    prediction, expected_shape, f"teacher_view[{view_index}]"
                ).detach()
            )
        return views

    def train_step(
        self,
        batch: Any,
        *,
        epoch: int,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """执行 student forward → 3-view teacher gate → loss → EMA update。"""
        batch = _coerce_w2_batch(batch)
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        self._validate_batch_identity(batch)
        self._move_models(device)
        model_batch = batch.to(device)
        self.student.train()
        _prepare_ema_teacher(self.teacher)
        self.optimizer.zero_grad(set_to_none=True)
        student_predictions = _forward_w2_model(self.student, model_batch)
        teacher_views = self.predict_teacher_views(model_batch, device=device)
        loss, stats = compute_w2_loss(
            student_predictions,
            teacher_views,
            model_batch,
            config=self.config,
            epoch=epoch,
        )
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        update_ema_teacher(self.student, self.teacher, decay=self.config.ema_decay)
        self.global_step += 1
        stats = dict(stats)
        stats.update({"epoch": epoch, "global_step": self.global_step})
        self.last_metrics = copy.deepcopy(stats)
        return stats

    def run_epoch(
        self,
        batches: Any,
        *,
        epoch: int,
        device: str | torch.device = "cpu",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """运行非空 W2 epoch，并聚合 gate/statistics diagnostics。"""
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        if max_steps is not None:
            max_steps = _strict_nonnegative_int(max_steps, name="max_steps")
            if max_steps <= 0:
                raise ValueError("W2 max_steps 必须是正整数或 None")
        logs = []
        for batch in batches:
            if max_steps is not None and len(logs) >= max_steps:
                break
            logs.append(self.train_step(batch, epoch=epoch, device=device))
        if not logs:
            raise ValueError("W2 batches 为空：训练循环无样本可跑")
        average_keys = {
            "loss", "anchor_loss", "pseudo_loss", "consistency_loss", "ramp_weight",
            "variance_gate", "pseudo_acceptance", "gate_acceptance",
            "pseudo_positive_ratio", "pseudo_negative_ratio", "anchor_coverage",
            "unknown_coverage", "solid_coverage", "failed_cell_coverage",
            "teacher_student_disagreement", "accepted_teacher_student_disagreement",
            "mean_probability_mean", "mean_probability_std", "mean_probability_min",
            "mean_probability_max", "predictive_variance_mean", "predictive_variance_std",
            "predictive_variance_min", "predictive_variance_max", "entropy_mean",
            "entropy_std", "entropy_min", "entropy_max",
        }
        count_keys = {
            "candidate_count", "pseudo_eligible_count", "pseudo_accepted_count",
            "pseudo_positive_count", "pseudo_negative_count", "unknown_count",
            "anchor_count", "anchor_positive_count", "anchor_negative_count",
            "unknown_cell_count", "solid_count", "failed_cell_count",
            "artifact_failure_count",
        }
        summary = {
            key: float(np.mean([float(log[key]) for log in logs]))
            for key in average_keys
        }
        summary.update({
            key: int(sum(int(log[key]) for log in logs)) for key in count_keys
        })
        sources = {log["sampling_source"] for log in logs}
        loss_sources = {log["loss_source"] for log in logs}
        anchor_hashes = {log["anchor_hash"] for log in logs}
        view_counts = {log["view_count"] for log in logs}
        gates = {float(log["variance_gate"]) for log in logs}
        if (len(sources) != 1 or len(loss_sources) != 1 or len(anchor_hashes) != 1
                or view_counts != {self.config.view_count}
                or gates != {self.config.variance_gate}):
            raise ValueError("W2 一个 epoch 内 source/hash/view/gate 发生漂移")
        summary.update({
            "view_count": self.config.view_count,
            "variance_gate": self.config.variance_gate,
            "sampling_source": next(iter(sources)),
            "loss_source": next(iter(loss_sources)),
            "anchor_hash": next(iter(anchor_hashes)),
            "epoch": epoch,
            "steps": len(logs),
            "global_step": self.global_step,
        })
        self.last_metrics = copy.deepcopy(summary)
        return summary

    def _checkpoint_extra_metadata(
        self,
        extra_metadata: Mapping[str, Any] | None,
        *,
        metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """构造包含 gate/view 配置、但不含 test GT 的 checkpoint metadata。"""
        if extra_metadata is not None and not isinstance(extra_metadata, Mapping):
            raise TypeError("W2 extra_metadata 必须是 object")
        extra = copy.deepcopy(dict(extra_metadata or {}))
        _reject_test_source(extra, context="W2 checkpoint extra_metadata")
        reserved = {
            "generation_version": W2_GENERATION_VERSION,
            "formal_loss_source": W2_LABEL_SOURCE,
            "w2_config": self.config.as_dict(),
            "uncertainty_gate": {
                "view_count": W2_DEFAULT_VIEW_COUNT,
                "positive_confidence": W2_DEFAULT_PSEUDO_HIGH,
                "negative_confidence": W2_DEFAULT_PSEUDO_LOW,
                "variance_gate": self.config.variance_gate,
                "variance_is_primary": True,
                "entropy_diagnostic_only": True,
                "dataset_gate_count": 1,
            },
        }
        if self.anchor_metadata is not None:
            reserved["haller_anchor"] = copy.deepcopy(dict(self.anchor_metadata))
        if metrics is not None:
            reserved["w2_metrics"] = copy.deepcopy(dict(metrics))
        for key, expected in reserved.items():
            if key in extra and extra[key] != expected:
                raise ValueError(f"W2 checkpoint extra_metadata.{key} 与 trainer 语义不一致")
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
        calibration_policy: W2CalibrationSelection | Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """保存可 resume 的 W2 checkpoint，并持久化 global gate/view contract。"""
        if self.scheduler is None:
            raise ValueError("W2 resume checkpoint 必须提供 scheduler")
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        self.anchor_metadata = _validate_checkpoint_anchor_metadata(
            self.anchor_metadata)
        selected_policy = (
            self.calibration_selection
            if calibration_policy is None
            else calibration_policy
        )
        policy = _policy_as_dict(
            selected_policy, variance_gate=_policy_gate(selected_policy)
        )
        checkpoint_metrics = self.last_metrics if metrics is None else metrics
        if checkpoint_metrics is None:
            raise ValueError(
                "W2 checkpoint 必须显式提供 gate/view/acceptance metrics；"
                "请先运行 train_step/run_epoch 或传入 metrics"
            )
        checkpoint_metrics = _validate_w2_checkpoint_metrics(
            checkpoint_metrics, variance_gate=float(self.config.variance_gate)
        )
        return contract.save_checkpoint(
            path,
            self.student,
            self.optimizer,
            self.scheduler,
            mode=contract.MODE_W2,
            feature_schema=contract.FEATURE_SCHEMA_7,
            adapter_input_schema=contract.FEATURE_SCHEMA_7,
            dataset_config=dataset_config,
            split_config=split_config,
            sampling_config=sampling_config,
            label_source=W2_LABEL_SOURCE,
            sampling_source=self.sampling_source,
            teacher=self.teacher,
            epoch=epoch,
            global_step=self.global_step,
            metrics=checkpoint_metrics,
            seed=self.seed,
            anchor_hash=self.anchor_hash,
            calibration_policy=policy,
            extra_metadata=self._checkpoint_extra_metadata(
                extra_metadata, metrics=checkpoint_metrics
            ),
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
        strict_cuda_rng: bool = True,
    ) -> dict[str, Any]:
        """按 W2 mode/source/schema/split/hash/gate 恢复 student 与 EMA teacher。"""
        self._move_models(device)
        result = contract.load_checkpoint(
            path,
            self.student,
            self.optimizer,
            self.scheduler,
            teacher=self.teacher,
            device=device,
            expected_mode=contract.MODE_W2,
            expected_feature_schema=contract.FEATURE_SCHEMA_7,
            expected_dataset_config=expected_dataset_config,
            expected_split_config=expected_split_config,
            expected_sampling_config=expected_sampling_config,
            expected_label_source=W2_LABEL_SOURCE,
            expected_sampling_source=self.sampling_source,
            expected_anchor_hash=self.anchor_hash,
            restore_rng=restore_rng,
            strict_cuda_rng=strict_cuda_rng,
            load_mode=load_mode,
        )
        extra_metadata = result.get("extra_metadata")
        if not isinstance(extra_metadata, Mapping):
            raise ValueError("W2 checkpoint 缺少可恢复的 extra_metadata")
        loaded_anchor_metadata = _validate_checkpoint_anchor_metadata(
            extra_metadata.get("haller_anchor")
        )
        if self.anchor_metadata is not None:
            current_anchor_metadata = _validate_checkpoint_anchor_metadata(
                self.anchor_metadata
            )
            if _anchor_metadata_identity(current_anchor_metadata) != _anchor_metadata_identity(
                    loaded_anchor_metadata):
                raise ValueError(
                    "W2 checkpoint Haller anchor manifest metadata 与 trainer metadata 不一致"
                )
        self.anchor_metadata = loaded_anchor_metadata
        loaded_policy = result.get("calibration_policy")
        policy = _policy_as_dict(
            loaded_policy, variance_gate=_policy_gate(loaded_policy)
        )
        loaded_gate = policy["variance_gate"]
        result["calibration_policy"] = policy
        loaded_metrics = _validate_w2_checkpoint_metrics(
            result.get("metrics"), variance_gate=float(self.config.variance_gate)
        )
        stored_w2_config = extra_metadata.get("w2_config")
        if stored_w2_config != self.config.as_dict():
            raise ValueError(
                "W2 checkpoint w2_config 与当前冻结 trainer config 不一致"
            )
        expected_uncertainty_gate = {
            "view_count": W2_DEFAULT_VIEW_COUNT,
            "positive_confidence": W2_DEFAULT_PSEUDO_HIGH,
            "negative_confidence": W2_DEFAULT_PSEUDO_LOW,
            "variance_gate": float(self.config.variance_gate),
            "variance_is_primary": True,
            "entropy_diagnostic_only": True,
            "dataset_gate_count": 1,
        }
        if extra_metadata.get("uncertainty_gate") != expected_uncertainty_gate:
            raise ValueError(
                "W2 checkpoint uncertainty_gate 与当前冻结 gate 配置不一致"
            )
        stored_w2_metrics = _validate_w2_checkpoint_metrics(
            extra_metadata.get("w2_metrics"),
            variance_gate=float(self.config.variance_gate),
        )
        if stored_w2_metrics != loaded_metrics:
            raise ValueError(
                "W2 checkpoint w2_metrics 与顶层 metrics 不一致"
            )
        self.last_metrics = copy.deepcopy(loaded_metrics)
        # 将持久化的 global calibration selection 回填到 trainer，保证
        # load → 续训 → save 不会丢失 policy 或退回 source=none。
        self.calibration_selection = copy.deepcopy(policy)
        self.global_step = _strict_nonnegative_int(
            result["global_step"], name="checkpoint global_step")
        self.seed = _strict_nonnegative_int(result["seed"], name="checkpoint seed")
        _prepare_ema_teacher(self.teacher)
        result = dict(result)
        result.update({
            "view_count": W2_DEFAULT_VIEW_COUNT,
            "variance_gate": loaded_gate,
            "uncertainty_gate": {
                "view_count": W2_DEFAULT_VIEW_COUNT,
                "variance_gate": loaded_gate,
                "positive_confidence": W2_DEFAULT_PSEUDO_HIGH,
                "negative_confidence": W2_DEFAULT_PSEUDO_LOW,
            },
        })
        return result


__all__ = [
    "W2_GENERATION_VERSION",
    "W2_LABEL_SOURCE",
    "W2_CALIBRATION_SOURCE",
    "W2_TEST_SOURCE",
    "W2_DEFAULT_VIEW_COUNT",
    "W2_DEFAULT_PSEUDO_HIGH",
    "W2_DEFAULT_PSEUDO_LOW",
    "W2Config",
    "W2Statistics",
    "W2GateResult",
    "W2CalibrationRecord",
    "W2CalibrationSelection",
    "W2Batch",
    "W2Trainer",
    "compute_w2_statistics",
    "apply_w2_uncertainty_gate",
    "calibrate_w2_gate",
    "select_global_w2_gate",
    "compute_w2_loss",
    "build_w2_batch",
    "build_w2_batch_from_w1_h",
]
