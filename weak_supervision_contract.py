"""弱监督方法的公共 mode、feature、batch 与 checkpoint 契约。

本模块位于训练/评价 seam 的外部 adapter 层，目的是把研究语义（方法 mode、
输入 channel、监督来源和可恢复状态）变成可检查的显式接口。它不修改
``vendor/DeepUtils``，也不实现 W1/W2/W3 的具体损失；后续票据通过本模块
取得统一的输入与 checkpoint 语义。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import platform
import pickle
import random
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn


class MethodMode(str, Enum):
    """弱监督 feature 注册的 canonical method mode。"""

    B0 = "B0"
    B1 = "B1"
    W1 = "W1"
    W1_P = "W1-P"
    W1_H = "W1-H"
    W2 = "W2"
    W3 = "W3"


MODE_B0 = MethodMode.B0.value
MODE_B1 = MethodMode.B1.value
MODE_W1 = MethodMode.W1.value
MODE_W1_P = MethodMode.W1_P.value
MODE_W1_H = MethodMode.W1_H.value
MODE_W2 = MethodMode.W2.value
MODE_W3 = MethodMode.W3.value


@dataclass(frozen=True)
class FeatureSchema:
    """pathline feature 的可审计 schema（顺序是语义的一部分）。"""

    name: str
    version: str
    channels: tuple[str, ...]
    channel_count: int
    local_ivd_channel: int | None

    def __post_init__(self) -> None:
        channels = tuple(str(channel) for channel in self.channels)
        object.__setattr__(self, "channels", channels)
        if isinstance(self.channel_count, (bool, np.bool_)) or not isinstance(
                self.channel_count, (int, np.integer)):
            raise ValueError("feature schema channel_count 必须是整数")
        channel_count = int(self.channel_count)
        object.__setattr__(self, "channel_count", channel_count)
        if channel_count != len(channels):
            raise ValueError(
                f"feature schema channel_count={channel_count} 与 channels="
                f"{channels!r} 不一致"
            )
        if self.local_ivd_channel is not None:
            if isinstance(self.local_ivd_channel, (bool, np.bool_)) or not isinstance(
                    self.local_ivd_channel, (int, np.integer)):
                raise ValueError(
                    "feature schema local_ivd_channel 必须是整数或 null"
                )
            ivd_channel = int(self.local_ivd_channel)
            object.__setattr__(self, "local_ivd_channel", ivd_channel)
            if not (
                0 <= ivd_channel < channel_count
                and channels[ivd_channel] == "ivd"
            ):
                raise ValueError(
                    "feature schema local_ivd_channel 必须指向当前 schema 的 ivd channel"
                )

    def as_dict(self) -> dict[str, Any]:
        """转为 checkpoint/metadata 可保存的普通 dict。"""
        return {
            "name": self.name,
            "version": self.version,
            "channels": list(self.channels),
            "channel_count": self.channel_count,
            "local_ivd_channel": self.local_ivd_channel,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureSchema":
        """从 metadata 读取 schema，并拒绝缺字段或非 object 输入。"""
        if not isinstance(value, Mapping):
            raise ValueError("feature schema 必须是 object")
        required = ("name", "version", "channels", "channel_count",
                    "local_ivd_channel")
        missing = [field for field in required if field not in value]
        if missing:
            raise ValueError(f"feature schema 缺少字段 {missing}")
        channels = value["channels"]
        if not isinstance(channels, (list, tuple)):
            raise ValueError("feature schema channels 必须是 list")
        return cls(
            name=str(value["name"]),
            version=str(value["version"]),
            channels=tuple(str(channel) for channel in channels),
            channel_count=value["channel_count"],
            local_ivd_channel=value["local_ivd_channel"],
        )


FEATURE_SCHEMA_7 = FeatureSchema(
    name="pathline_7ch",
    version="v1",
    channels=("px", "py", "t", "ivd", "distance", "u", "v"),
    channel_count=7,
    local_ivd_channel=3,
)
FEATURE_SCHEMA_6 = FeatureSchema(
    name="pathline_6ch",
    version="v1",
    channels=("px", "py", "t", "distance", "u", "v"),
    channel_count=6,
    local_ivd_channel=None,
)


LABEL_SOURCE_LEGACY_P85 = "legacy_p85"
LABEL_SOURCE_LOCAL_P90_P60 = "local_p90_p60"
LABEL_SOURCE_HALLER_TRAIN = "haller_anchor_train"
LABEL_SOURCE_HALLER_CALIBRATION = "haller_gt_calibration"
LABEL_SOURCE_HALLER_TEST = "haller_gt_test"
VALID_LABEL_SOURCES = (
    LABEL_SOURCE_LEGACY_P85,
    LABEL_SOURCE_LOCAL_P90_P60,
    LABEL_SOURCE_HALLER_TRAIN,
    LABEL_SOURCE_HALLER_CALIBRATION,
    LABEL_SOURCE_HALLER_TEST,
)
VALID_CONSUMERS = ("train", "calibration", "evaluation")
VALID_SPLIT_NAMES = ("train", "calibration", "test")


@dataclass(frozen=True)
class ModeSpec:
    """一个方法 mode 的输入/输出 schema 和监督组件能力。"""

    mode: str
    feature_schema: FeatureSchema
    adapter_input_schema: FeatureSchema
    formal_label_sources: tuple[str, ...]
    requires_teacher: bool
    supports_projection_head: bool


_MODE_ALIASES = {
    "b0": MODE_B0,
    "b1": MODE_B1,
    "w1": MODE_W1,
    "w1-p": MODE_W1_P,
    "w1_p": MODE_W1_P,
    "w1p": MODE_W1_P,
    "w1-h": MODE_W1_H,
    "w1_h": MODE_W1_H,
    "w1h": MODE_W1_H,
    "w2": MODE_W2,
    "w3": MODE_W3,
}


_MODE_SPECS = {
    MODE_B0: ModeSpec(
        MODE_B0, FEATURE_SCHEMA_7, FEATURE_SCHEMA_7, ("legacy_p85",),
        False, False,
    ),
    MODE_B1: ModeSpec(
        MODE_B1, FEATURE_SCHEMA_6, FEATURE_SCHEMA_7, ("legacy_p85",),
        False, False,
    ),
    MODE_W1: ModeSpec(
        MODE_W1, FEATURE_SCHEMA_7, FEATURE_SCHEMA_7,
        ("local_p90_p60", "haller_anchor_train"), True, False,
    ),
    MODE_W1_P: ModeSpec(
        MODE_W1_P, FEATURE_SCHEMA_7, FEATURE_SCHEMA_7, ("local_p90_p60",),
        True, False,
    ),
    MODE_W1_H: ModeSpec(
        MODE_W1_H, FEATURE_SCHEMA_7, FEATURE_SCHEMA_7,
        ("haller_anchor_train",), True, False,
    ),
    MODE_W2: ModeSpec(
        MODE_W2, FEATURE_SCHEMA_7, FEATURE_SCHEMA_7,
        ("haller_anchor_train",), True, False,
    ),
    MODE_W3: ModeSpec(
        MODE_W3, FEATURE_SCHEMA_7, FEATURE_SCHEMA_7,
        ("haller_anchor_train",), True, True,
    ),
}


def canonical_mode(mode: str | MethodMode) -> str:
    """返回 canonical mode；未知或空 mode 立即失败。"""
    if isinstance(mode, MethodMode):
        return mode.value
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(f"mode 必须是已注册的字符串，实际 {mode!r}")
    key = mode.strip()
    canonical = _MODE_ALIASES.get(key.lower())
    if canonical is None:
        raise ValueError(
            f"未知 method mode {mode!r}；允许值为 {list(_MODE_SPECS)}"
        )
    return canonical


def mode_spec(mode: str | MethodMode) -> ModeSpec:
    """读取一个 mode 的公共能力描述。"""
    return _MODE_SPECS[canonical_mode(mode)]


def feature_schema_for_mode(mode: str | MethodMode) -> FeatureSchema:
    """返回 model 实际接收的 feature schema。"""
    return mode_spec(mode).feature_schema


def _coerce_schema(value: FeatureSchema | Mapping[str, Any]) -> FeatureSchema:
    if isinstance(value, FeatureSchema):
        return value
    return FeatureSchema.from_mapping(value)


def validate_feature_schema(
    actual: FeatureSchema | Mapping[str, Any],
    expected: str | MethodMode | FeatureSchema | Mapping[str, Any],
) -> FeatureSchema:
    """校验 schema 的名称、版本、数量和完整 channel order。"""
    actual_schema = _coerce_schema(actual)
    if isinstance(expected, (str, MethodMode)):
        expected_schema = feature_schema_for_mode(expected)
    else:
        expected_schema = _coerce_schema(expected)
    if actual_schema != expected_schema:
        raise ValueError(
            "feature schema/channel order 不匹配："
            f"expected={expected_schema.as_dict()} actual={actual_schema.as_dict()}"
        )
    return actual_schema


def validate_feature_values(
    values: Any,
    schema: FeatureSchema | Mapping[str, Any],
) -> Any:
    """校验 pathline tensor/array 的末维是否与显式 schema 一致。"""
    schema = _coerce_schema(schema)
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) < 1:
        raise ValueError("pathline features 必须带有 channel 维度")
    if int(shape[-1]) != schema.channel_count:
        raise ValueError(
            "pathline channel count 不匹配："
            f"expected={schema.channel_count} actual={int(shape[-1])} "
            f"schema={schema.channels!r}"
        )
    if isinstance(values, torch.Tensor):
        if values.is_floating_point() and not bool(torch.isfinite(values).all()):
            raise ValueError("pathline features 含有非有限值")
    else:
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError("pathline features 含有非有限值")
    return values


class ChannelSelectingAdapter(nn.Module):
    """在 vendor 外部按 mode 选择 pathline channels 的 adapter。

    调用方始终可以提供完整 7-channel pathline；B1 在这里移除 local-IVD，
    其余 mode 保持原顺序。``model`` 只接收 adapter 输出，因而无需修改
    ``vendor/DeepUtils``。
    """

    def __init__(self, model: nn.Module, mode: str | MethodMode):
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("ChannelSelectingAdapter 需要 torch.nn.Module model")
        self.mode = canonical_mode(mode)
        spec = mode_spec(self.mode)
        self.model = model
        self.input_schema = spec.adapter_input_schema
        self.feature_schema = spec.feature_schema
        self._indices = tuple(
            self.input_schema.channels.index(channel)
            for channel in self.feature_schema.channels
        )

    @property
    def device(self) -> torch.device:
        """Return the device of the wrapped model's primary parameters.

        The adapter is the contract-facing module used by W1-P/W1-H/W2.
        Those loss seams intentionally query ``student.device`` when they
        materialize masks and labels.  A plain ``nn.Module`` has no such
        attribute, and the omission becomes observable as soon as the real
        pilot executes one of those methods.  Returning the first parameter's
        device also keeps the seam correct when ``self.model`` is a
        ``DataParallel`` module: its first device is the output/primary
        device, while replicas remain implementation details of the wrapped
        vendor model.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def adapt(
        self,
        pathlines: Any,
        *,
        input_schema: FeatureSchema | Mapping[str, Any] | None = None,
    ) -> Any:
        """校验输入 schema 后返回 model schema 的 pathline features。"""
        schema = self.input_schema if input_schema is None else input_schema
        schema = _coerce_schema(schema)
        # batch seam 可能已经携带 model-facing features；这不是一次隐式
        # schema fallback，而是由调用方通过 input_schema 明确声明的 pass-through。
        if schema == self.feature_schema:
            return validate_feature_values(pathlines, self.feature_schema)
        validate_feature_schema(schema, self.input_schema)
        validate_feature_values(pathlines, self.input_schema)
        selected = pathlines[..., list(self._indices)]
        return validate_feature_values(selected, self.feature_schema)

    def adapt_batch(
        self,
        batch: "WeakSupervisionBatch",
        *,
        consumer: str = "train",
    ) -> Any:
        """将已通过指定 consumer contract 的 batch 转成 model pathline。"""
        validate_batch(batch, self.mode, consumer=consumer)
        return self.adapt(batch.pathlines, input_schema=batch.feature_schema)

    def forward_batch(
        self,
        batch: "WeakSupervisionBatch",
        *,
        dummy_field: Any | None = None,
        consumer: str = "train",
    ) -> Any:
        """显式组合 batch/loss seam 与 adapter seam，避免 B1 schema 混淆。"""
        pathlines = self.adapt_batch(batch, consumer=consumer)
        if dummy_field is None:
            if isinstance(pathlines, torch.Tensor):
                dummy_field = pathlines.new_zeros((pathlines.shape[0], 1, 1, 1))
            else:
                dummy_field = np.zeros((pathlines.shape[0], 1, 1, 1), dtype=np.float32)
        return self.model((dummy_field, pathlines))

    def forward(
        self,
        data: Any,
        *,
        input_schema: FeatureSchema | Mapping[str, Any] | None = None,
        consumer: str = "train",
    ) -> Any:
        """适配现有 ``(dummy_field, pathlines)`` 输入并调用 wrapped model。"""
        if isinstance(data, WeakSupervisionBatch):
            return self.forward_batch(data, consumer=consumer)
        if isinstance(data, (tuple, list)):
            if len(data) != 2:
                raise ValueError("model input 必须是 (dummy_field, pathlines) 二元组")
            dummy, pathlines = data
            return self.model((dummy, self.adapt(pathlines, input_schema=input_schema)))
        return self.model(self.adapt(data, input_schema=input_schema))


def _extract_model_config(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """取出完整 model config 与其 encoder config（内部 dispatch 辅助）。"""
    if not isinstance(config, Mapping):
        raise ValueError("model config 必须是 object")
    model_config = config.get("model", config)
    if not isinstance(model_config, Mapping):
        raise ValueError("model config 的 model 段必须是 object")
    encoder_config = model_config.get("encoder_args", model_config)
    if not isinstance(encoder_config, Mapping):
        raise ValueError("model config 的 encoder_args 必须是 object")
    # 测试/后续 adapter 也可只提供 ``{"encoder_args": ...}``；没有完整
    # vendor wrapper 的 NAME 时，直接构造 encoder，而不是送一个无 NAME 的
    # 半配置给 Registry。
    builder_config = (model_config if "NAME" in model_config else encoder_config)
    return builder_config, encoder_config


def build_model_for_mode(
    config: Mapping[str, Any],
    mode: str | MethodMode,
    *,
    model_builder: Any | None = None,
) -> ChannelSelectingAdapter:
    """按 mode 构造外部 channel adapter 与 vendor model。

    ``config`` 可以是完整训练配置（含 ``model.encoder_args``）或直接的
    vendor model config。配置中的 ``in_channels`` 必须已经与 mode schema
    一致；函数不会偷偷改写配置来掩盖 B1/B0 的输入差异。
    """
    canonical = canonical_mode(mode)
    spec = mode_spec(canonical)
    model_config, encoder_config = _extract_model_config(config)
    if "in_channels" not in encoder_config:
        raise ValueError(
            f"mode={canonical} 的 model config 必须显式声明 in_channels="
            f"{spec.feature_schema.channel_count}"
        )
    raw_channels = encoder_config["in_channels"]
    if isinstance(raw_channels, (bool, np.bool_)) or not isinstance(
            raw_channels, (int, np.integer)):
        raise ValueError("model config in_channels 必须是整数")
    configured_channels = int(raw_channels)
    if configured_channels != spec.feature_schema.channel_count:
        raise ValueError(
            f"mode={canonical} 的 feature schema/channel count 不匹配："
            f"expected={spec.feature_schema.channel_count} "
            f"configured={configured_channels}"
        )
    if model_builder is None:
        from vendor.DeepUtils.models import build_model_from_cfg
        model_builder = build_model_from_cfg
    # 只把配置交给 builder；adapter 的 channel 选择留在 vendor 外部。
    model = model_builder(model_config)
    return ChannelSelectingAdapter(model, canonical)


def validate_label_source(source: str) -> str:
    """校验弱监督标签来源；sampling source 必须另存，不能替代它。"""
    if source not in VALID_LABEL_SOURCES:
        raise ValueError(
            f"未知 label source {source!r}；允许值为 {list(VALID_LABEL_SOURCES)}"
        )
    return str(source)


def _shape_of(value: Any, *, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        raise ValueError(f"{name} 必须是带 shape 的 tensor/array")
    return tuple(int(size) for size in shape)


def _default_mask_like(labels: Any, *, value: bool) -> Any:
    if isinstance(labels, torch.Tensor):
        return torch.full_like(labels, value, dtype=torch.bool)
    return np.full_like(np.asarray(labels), value, dtype=bool)


def _coerce_mask(
    mask: Any,
    labels: Any,
    *,
    name: str,
    default_value: bool,
) -> Any:
    shape = _shape_of(labels, name="labels")
    if mask is None:
        return _default_mask_like(labels, value=default_value)
    if _shape_of(mask, name=name) != shape:
        raise ValueError(f"{name} shape 与 labels 不一致：{_shape_of(mask, name=name)} != {shape}")
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            normalized = mask
        elif not bool(torch.is_floating_point(mask) or mask.dtype in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError(f"{name} 必须是 bool/0-1 mask")
        elif not bool(torch.all((mask == 0) | (mask == 1))):
            raise ValueError(f"{name} 必须只包含 0/1")
        else:
            normalized = mask.to(dtype=torch.bool)
        if isinstance(labels, torch.Tensor):
            return normalized.to(device=labels.device)
        return normalized.detach().cpu().numpy()
    array = np.asarray(mask)
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} 必须只包含 0/1")
    normalized = array.astype(bool, copy=False)
    if isinstance(labels, torch.Tensor):
        return torch.as_tensor(normalized, dtype=torch.bool, device=labels.device)
    return normalized


def _validate_label_values(labels: Any) -> None:
    if isinstance(labels, torch.Tensor):
        if not (labels.dtype == torch.bool or labels.dtype in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                torch.float16, torch.float32, torch.float64)):
            raise ValueError("labels 必须是 numeric/bool tensor")
        if not bool(torch.isfinite(labels).all()):
            raise ValueError("labels 含有非有限值")
        if not bool(((labels >= 0) & (labels <= 1)).all()):
            raise ValueError("labels 必须位于 [0, 1]")
        return
    array = np.asarray(labels)
    if not (np.issubdtype(array.dtype, np.bool_)
            or np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)):
        raise ValueError("labels 必须是 numeric/bool array")
    if not np.isfinite(array).all():
        raise ValueError("labels 含有非有限值")
    if not np.all((array >= 0) & (array <= 1)):
        raise ValueError("labels 必须位于 [0, 1]")


def _provenance_sources(value: Any):
    """递归提取已注册来源名，用于阻止 test GT 藏入 provenance。"""
    if isinstance(value, str):
        if value in VALID_LABEL_SOURCES:
            yield value
        return
    if isinstance(value, np.ndarray):
        yield from _provenance_sources(value.tolist())
        return
    if isinstance(value, np.generic):
        yield from _provenance_sources(value.item())
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _provenance_sources(key)
            yield from _provenance_sources(child)
    elif isinstance(value, (tuple, list, set)):
        for child in value:
            yield from _provenance_sources(child)


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


def _test_only_key_names(value: Any, *, _parents: tuple[str, ...] = ()):
    """递归识别 test label/metric/result 的显式键名或 marker。"""
    if isinstance(value, np.ndarray):
        yield from _test_only_key_names(value.tolist(), _parents=_parents)
        return
    if isinstance(value, np.generic):
        yield from _test_only_key_names(value.item(), _parents=_parents)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(".", "_")
            test_tokens = {token for token in normalized.split("_") if token}
            structural_split_key = (
                normalized == "test"
                and ("split_ranges" in _parents or "taus" in _parents)
            )
            if (not structural_split_key
                    and normalized not in _TEST_ONLY_KEY_EXCEPTIONS
                    and (normalized in _TEST_ONLY_KEYS
                         or normalized.startswith("test_")
                         or normalized.endswith("_test")
                         or {"gt", "test"}.issubset(test_tokens)
                         or {"label", "test"}.issubset(test_tokens)
                         or {"metric", "test"}.issubset(test_tokens))):
                yield normalized
            if normalized in {"split", "split_name"} and str(child).strip().lower() == "test":
                yield normalized
            yield from _test_only_key_names(child, _parents=(*_parents, normalized))
    elif isinstance(value, (tuple, list, set)):
        for child in value:
            yield from _test_only_key_names(child, _parents=_parents)


def _reject_test_only_keys(value: Any, *, context: str) -> None:
    """拒绝训练/calibration metadata 中可扩展的 test-only 字段别名。"""
    names = sorted(set(_test_only_key_names(value)))
    if names:
        raise ValueError(
            f"{context} 禁止访问 test-only label/metric/result 字段：{names!r}"
        )


def _validate_calibration_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """校验 calibration policy 不携带 test-only Haller GT 来源。"""
    normalized = _jsonable(policy)
    _reject_test_only_keys(normalized, context="calibration_policy")
    if LABEL_SOURCE_HALLER_TEST in set(_provenance_sources(normalized)):
        raise ValueError(
            "calibration_policy 禁止使用 haller_gt_test；test GT 只能进入最终 evaluation"
        )
    return normalized


def _validate_source_split(source: str | None, split_name: str, *, context: str) -> None:
    """执行 Haller source 与 checkpoint/batch split 的一一对应 guard。"""
    required_split = {
        LABEL_SOURCE_LEGACY_P85: "train",
        LABEL_SOURCE_LOCAL_P90_P60: "train",
        LABEL_SOURCE_HALLER_TRAIN: "train",
        LABEL_SOURCE_HALLER_CALIBRATION: "calibration",
        LABEL_SOURCE_HALLER_TEST: "test",
    }.get(source)
    if required_split is not None and split_name != required_split:
        raise ValueError(
            f"{context}={source!r} 只能与 split={required_split!r} 一起使用，"
            f"实际 split={split_name!r}"
        )


@dataclass
class WeakSupervisionBatch:
    """训练/评价 seam 的 batch contract。

    ``label_source`` 是 formal loss 的来源；``sampling_source`` 只记录采样池
    或 patch membership 的来源。``label_mask``/``unknown_mask`` 保留三态语义，
    后续 W1/W2 loss 可在同一接口上消费，而不把 unknown 静默当作 negative。
    """

    pathlines: Any
    labels: Any
    label_source: str
    split_name: str
    feature_schema: FeatureSchema | Mapping[str, Any] = FEATURE_SCHEMA_7
    label_mask: Any | None = None
    unknown_mask: Any | None = None
    sampling_source: str | None = None
    provenance: Mapping[str, Any] | None = None
    mode: str | MethodMode | None = None
    # model-facing pathlines 使用 feature_schema；B1 的 raw adapter input
    # schema 另存于此，避免把 7→6 选择误解成监督 batch 的 schema 漂移。
    input_schema: FeatureSchema | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.feature_schema = _coerce_schema(self.feature_schema)
        self.label_source = validate_label_source(self.label_source)
        if self.sampling_source is not None:
            self.sampling_source = validate_label_source(self.sampling_source)
        self.split_name = str(self.split_name)
        if self.split_name not in ("train", "calibration", "test"):
            raise ValueError(
                f"batch split_name={self.split_name!r} 非法；必须是 train/calibration/test"
            )
        if self.mode is not None:
            self.mode = canonical_mode(self.mode)
        if self.input_schema is None:
            self.input_schema = (
                mode_spec(self.mode).adapter_input_schema
                if self.mode is not None else self.feature_schema
            )
        else:
            self.input_schema = _coerce_schema(self.input_schema)
        validate_feature_values(self.pathlines, self.feature_schema)
        pathline_shape = _shape_of(self.pathlines, name="pathlines")
        label_shape = _shape_of(self.labels, name="labels")
        if pathline_shape[0] != label_shape[0]:
            raise ValueError(
                "pathlines 与 labels 的 batch shape 不一致："
                f"pathlines[0]={pathline_shape[0]} labels[0]={label_shape[0]}"
            )
        _validate_label_values(self.labels)
        self.label_mask = _coerce_mask(
            self.label_mask, self.labels, name="label_mask", default_value=True)
        self.unknown_mask = _coerce_mask(
            self.unknown_mask, self.labels, name="unknown_mask", default_value=False)
        if isinstance(self.label_mask, torch.Tensor):
            overlap = self.label_mask & self.unknown_mask
            if bool(overlap.any()):
                raise ValueError("label_mask 与 unknown_mask 不能重叠")
        elif np.any(self.label_mask & self.unknown_mask):
            raise ValueError("label_mask 与 unknown_mask 不能重叠")
        self.provenance = copy.deepcopy(dict(self.provenance or {}))

    def as_dict(self) -> dict[str, Any]:
        """返回不含大数组的 provenance 摘要，便于日志和 checkpoint metadata。"""
        return {
            "mode": self.mode,
            "split_name": self.split_name,
            "feature_schema": self.feature_schema.as_dict(),
            "input_schema": self.input_schema.as_dict(),
            "label_source": self.label_source,
            "sampling_source": self.sampling_source,
            "provenance": copy.deepcopy(dict(self.provenance or {})),
            "label_mask_known": int(self.label_mask.sum().item()
                                     if isinstance(self.label_mask, torch.Tensor)
                                     else np.asarray(self.label_mask).sum()),
            "unknown_mask_count": int(self.unknown_mask.sum().item()
                                       if isinstance(self.unknown_mask, torch.Tensor)
                                       else np.asarray(self.unknown_mask).sum()),
        }


def validate_batch(
    batch: WeakSupervisionBatch,
    mode: str | MethodMode | None = None,
    *,
    consumer: str = "train",
) -> WeakSupervisionBatch:
    """按 consumer/mode 校验 batch，并返回原 batch 供 loss 使用。"""
    if not isinstance(batch, WeakSupervisionBatch):
        raise TypeError("训练/评价 batch 必须是 WeakSupervisionBatch")
    if consumer not in VALID_CONSUMERS:
        raise ValueError(f"未知 batch consumer={consumer!r}；允许值为 {list(VALID_CONSUMERS)}")
    canonical = None if mode is None else canonical_mode(mode)
    if canonical is not None:
        if batch.mode is not None and batch.mode != canonical:
            raise ValueError(
                f"batch mode={batch.mode!r} 与 requested mode={canonical!r} 不匹配"
            )
        validate_feature_schema(batch.feature_schema, canonical)
        spec = mode_spec(canonical)
        if batch.input_schema != spec.adapter_input_schema:
            # 已适配的 model-facing batch 可以在未携带 raw 7-channel 数据时
            # 声明 model schema；raw input schema 仍会由 mode-aware 构造器标注。
            validate_feature_schema(batch.input_schema, batch.feature_schema)
        if consumer == "train" and batch.label_source not in spec.formal_label_sources:
            raise ValueError(
                f"mode={canonical} 的 formal label source 不匹配："
                f"expected={spec.formal_label_sources} actual={batch.label_source!r}"
            )
    if consumer == "train":
        if batch.split_name != "train":
            raise ValueError(
                f"train consumer 只能读取 split=train，实际 {batch.split_name!r}"
            )
        forbidden = {LABEL_SOURCE_HALLER_CALIBRATION, LABEL_SOURCE_HALLER_TEST}
        if batch.label_source in forbidden:
            raise ValueError(
                f"train consumer 禁止读取 Haller source {batch.label_source!r}"
            )
    elif consumer == "calibration":
        if batch.split_name != "calibration":
            raise ValueError(
                f"calibration consumer 只能读取 split=calibration，实际 {batch.split_name!r}"
            )
        if batch.label_source != LABEL_SOURCE_HALLER_CALIBRATION:
            raise ValueError(
                "calibration consumer 必须显式使用 haller_gt_calibration"
            )
    else:
        if batch.split_name != "test":
            raise ValueError(
                f"evaluation consumer 只能读取 split=test，实际 {batch.split_name!r}"
            )
        if batch.label_source != LABEL_SOURCE_HALLER_TEST:
            raise ValueError(
                "evaluation consumer 必须显式使用 haller_gt_test"
            )
    all_sources = [batch.label_source]
    if batch.sampling_source is not None:
        all_sources.append(batch.sampling_source)
    all_sources.extend(_provenance_sources(batch.provenance))
    if consumer != "evaluation":
        _reject_test_only_keys(batch.provenance, context=f"{consumer} batch provenance")
    if consumer == "train":
        forbidden = {
            LABEL_SOURCE_HALLER_CALIBRATION,
            LABEL_SOURCE_HALLER_TEST,
        }
        forbidden_source = next(
            (source for source in all_sources if source in forbidden), None
        )
        if forbidden_source is not None:
            raise ValueError(
                f"train consumer 禁止读取 Haller source {forbidden_source!r}；"
                "calibration/test GT 不能进入训练 provenance 或 sampling"
            )
    if consumer != "evaluation" and LABEL_SOURCE_HALLER_TEST in all_sources:
        raise ValueError(
            f"{consumer} consumer 禁止 provenance 中出现 haller_gt_test"
        )
    return batch


def validate_training_batch(
    batch: WeakSupervisionBatch,
    mode: str | MethodMode | None = None,
) -> WeakSupervisionBatch:
    """训练入口的显式 batch guard。"""
    return validate_batch(batch, mode, consumer="train")


def validate_calibration_batch(
    batch: WeakSupervisionBatch,
    mode: str | MethodMode | None = None,
) -> WeakSupervisionBatch:
    """calibration 入口的显式 Haller GT guard。"""
    return validate_batch(batch, mode, consumer="calibration")


def validate_evaluation_batch(
    batch: WeakSupervisionBatch,
    mode: str | MethodMode | None = None,
) -> WeakSupervisionBatch:
    """test evaluation 入口的显式 Haller GT guard。"""
    return validate_batch(batch, mode, consumer="evaluation")


class ModeAwareLoss(nn.Module):
    """给现有 criterion 加上 mode/batch contract 校验的轻量 loss adapter。

    只有声明 ``accepts_weak_supervision_batch = True`` 的后续 criterion 才能
    消费带 unknown/未标注 mask 的 batch；普通 label-only criterion 会被拒绝，
    从而不把 unknown 静默当成负样本。具体 masked/pseudo/consistency loss 仍由
    后续方法票实现。
    """

    def __init__(self, mode: str | MethodMode, criterion: Any):
        super().__init__()
        if not callable(criterion):
            raise TypeError("criterion 必须可调用")
        self.mode = canonical_mode(mode)
        self.criterion = criterion

    def forward(self, predictions: Any, batch: WeakSupervisionBatch) -> Any:
        """先执行 train guard，再按 criterion 能力保留 batch mask 语义。"""
        validated = validate_training_batch(batch, self.mode)
        if _shape_of(predictions, name="predictions") != _shape_of(
                validated.labels, name="labels"):
            raise ValueError(
                "predictions shape 与 labels 不一致："
                f"{_shape_of(predictions, name='predictions')} != "
                f"{_shape_of(validated.labels, name='labels')}"
            )
        declared_batch_aware = getattr(
            self.criterion, "accepts_weak_supervision_batch", False
        )
        if not isinstance(declared_batch_aware, bool):
            raise ValueError(
                "criterion.accepts_weak_supervision_batch 必须是 bool"
            )
        batch_aware = declared_batch_aware
        mask_has_unobserved = (
            bool(torch.any(~validated.label_mask | validated.unknown_mask))
            if isinstance(validated.label_mask, torch.Tensor)
            else bool(np.any(~validated.label_mask | validated.unknown_mask))
        )
        if mask_has_unobserved and not batch_aware:
            raise ValueError(
                "unknown/未标注 mask 需要 mask-aware criterion；"
                "label-only criterion 禁止静默消费"
            )
        if batch_aware:
            return self.criterion(predictions, validated)
        return self.criterion(predictions, validated.labels)


def build_loss_for_mode(
    mode: str | MethodMode,
    criterion: Any,
    *,
    criterion_builder: Any | None = None,
) -> ModeAwareLoss:
    """构造 mode-aware loss dispatch，不在本票据实现具体 W1/W2/W3 loss。"""
    if isinstance(criterion, Mapping):
        if criterion_builder is None:
            from vendor.DeepUtils.loss import build_criterion_from_cfg
            criterion_builder = build_criterion_from_cfg
        criterion = criterion_builder(criterion)
    return ModeAwareLoss(mode, criterion)


# --------------------------------------------------------------------------- checkpoint contract

CHECKPOINT_FORMAT_VERSION = "weak-supervision-checkpoint-v1"
_MISSING = object()


def _jsonable(value: Any) -> Any:
    """把 contract 配置转换为稳定 JSON 值；不可审计值直接失败。"""
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("contract metadata 不允许 NaN/Inf")
        return value
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    raise TypeError(
        f"contract metadata 含不可序列化值：{type(value).__name__}"
    )


def _mapping_or_empty(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是 object")
    return _jsonable(value)


def _normalize_split_config(split_config: Any) -> dict[str, Any]:
    if isinstance(split_config, str):
        if not split_config.strip():
            raise ValueError("split_config.split_name 不能为空")
        normalized = {"split_name": split_config.strip()}
    else:
        normalized = _mapping_or_empty(split_config, name="split_config")
    split_name = normalized.get("split_name")
    if not isinstance(split_name, str) or not split_name.strip():
        raise ValueError(
            "split_config 必须显式提供 split_name，禁止隐式 split fallback"
        )
    split_name = split_name.strip()
    if split_name not in VALID_SPLIT_NAMES:
        raise ValueError(
            f"split_config.split_name={split_name!r} 非法；允许值为 "
            f"{list(VALID_SPLIT_NAMES)}"
        )
    normalized["split_name"] = split_name
    return normalized


def _positive_int(value: Any, *, name: str) -> int:
    """将窗口/计数参数严格转为正整数，不截断 bool 或浮点值。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是正整数，实际 {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是正整数，实际 {value!r}") from exc
    try:
        integral = converted == value
    except (TypeError, ValueError):
        integral = False
    if not integral or converted <= 0:
        raise ValueError(f"{name} 必须是正整数，实际 {value!r}")
    return converted


def _nonnegative_int(value: Any, *, name: str) -> int:
    """将 seed 等可取 0 的计数参数严格转为非负整数。"""
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


def _normalize_sampling_config(
    sampling_config: Mapping[str, Any] | None,
    t_win: int | None,
) -> dict[str, Any]:
    """保存可复现窗口配置，并拒绝缺失 t_win 的隐式采样。"""
    sampling = _mapping_or_empty(sampling_config, name="sampling_config")
    stored_t_win = sampling.get("t_win", _MISSING)
    if stored_t_win is _MISSING and t_win is None:
        raise ValueError(
            "checkpoint 必须显式提供正整数 t_win（或 sampling_config.t_win），"
            "禁止隐式窗口配置"
        )
    explicit_t_win = None if t_win is None else _positive_int(t_win, name="t_win")
    if stored_t_win is not _MISSING:
        normalized_t_win = _positive_int(stored_t_win, name="sampling_config.t_win")
        if explicit_t_win is not None and normalized_t_win != explicit_t_win:
            raise ValueError(
                f"t_win={explicit_t_win} 与 sampling_config.t_win="
                f"{normalized_t_win} 不一致"
            )
    else:
        normalized_t_win = explicit_t_win
    sampling["t_win"] = normalized_t_win
    return sampling


def _state_copy(value: Any, *, name: str) -> Any:
    """复制 module/state dict 到 CPU，避免 checkpoint 引用活动参数。"""
    if isinstance(value, nn.Module):
        value = value.state_dict()
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是 torch.nn.Module 或 state_dict object")
    copied = {}
    for key, child in value.items():
        if isinstance(child, torch.Tensor):
            copied[key] = child.detach().cpu().clone()
        elif isinstance(child, Mapping):
            copied[key] = _state_copy(child, name=f"{name}.{key}")
        elif isinstance(child, (tuple, list)):
            copied[key] = type(child)(
                _state_copy({str(i): item}, name=f"{name}.{key}[{i}]")[str(i)]
                if isinstance(item, Mapping) else
                (item.detach().cpu().clone() if isinstance(item, torch.Tensor)
                 else copy.deepcopy(item))
                for i, item in enumerate(child)
            )
        else:
            copied[key] = copy.deepcopy(child)
    return copied


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (list(left.keys()) == list(right.keys())
                and all(_state_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right))
    return left == right


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    keys = list(state_dict)
    if keys and all(str(key).startswith("module.") for key in keys):
        return {str(key)[len("module."):]: value
                for key, value in state_dict.items()}
    return dict(state_dict)


def _load_state(target: nn.Module, state: Mapping[str, Any], *, name: str) -> None:
    if not isinstance(target, nn.Module):
        raise TypeError(f"{name} target 必须是 torch.nn.Module")
    # save_checkpoint 统一保存无 ``module.`` 前缀；DataParallel target 要把
    # state 加回 inner module，保持单卡/多卡 checkpoint 可互换。
    load_target = target.module if isinstance(target, nn.DataParallel) else target
    load_target.load_state_dict(_strip_module_prefix(state), strict=True)


def capture_rng_state() -> dict[str, Any]:
    """捕获 Python、NumPy、Torch CPU/CUDA 的随机状态。"""
    cuda_states = []
    if torch.cuda.is_available():
        cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        # 使用 tensor + 基础标量，保持 checkpoint 可由 weights_only loader 读取。
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.as_tensor(numpy_state[1], dtype=torch.uint32).cpu().clone(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state().cpu().clone(),
        "torch_cuda": cuda_states,
    }


def _numpy_rng_tuple(value: Any) -> tuple[Any, ...]:
    """将新旧两种 NumPy RNG 表示规范化为 ``set_state`` 所需 tuple。"""
    if isinstance(value, Mapping):
        required = ("bit_generator", "state", "pos", "has_gauss", "cached_gaussian")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"checkpoint rng_state.numpy 缺少字段 {missing}")
        return (
            str(value["bit_generator"]),
            np.asarray(value["state"], dtype=np.uint32),
            int(value["pos"]),
            int(value["has_gauss"]),
            float(value["cached_gaussian"]),
        )
    if isinstance(value, (tuple, list)) and len(value) == 5:
        # 兼容尚未采用 safe-loader 的早期本票 checkpoint 表示。
        return (
            str(value[0]),
            np.asarray(value[1], dtype=np.uint32),
            int(value[2]),
            int(value[3]),
            float(value[4]),
        )
    raise ValueError("checkpoint rng_state.numpy 必须是 object 或五元组")


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    """在保存/metadata 读取阶段验证 RNG contract 的四个必需部分。"""
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint rng_state 必须是 object")
    required = ("python", "numpy", "torch", "torch_cuda")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"checkpoint rng_state 缺少字段 {missing}")
    try:
        probe = random.Random()
        probe.setstate(state["python"])
        np_state = _numpy_rng_tuple(state["numpy"])
        np.random.RandomState().set_state(np_state)
        torch_state = torch.as_tensor(state["torch"])
        if torch_state.dtype != torch.uint8:
            raise ValueError("checkpoint rng_state.torch 必须是 uint8 tensor")
        if torch_state.numel() != torch.get_rng_state().numel():
            raise ValueError("checkpoint rng_state.torch 长度不匹配当前 Torch RNG")
        cuda_states = state["torch_cuda"]
        if not isinstance(cuda_states, (tuple, list)):
            raise ValueError("checkpoint rng_state.torch_cuda 必须是 list")
        for index, cuda_state in enumerate(cuda_states):
            cuda_tensor = torch.as_tensor(cuda_state)
            if cuda_tensor.dtype != torch.uint8:
                raise ValueError(
                    f"checkpoint rng_state.torch_cuda[{index}] 必须是 uint8 tensor"
                )
    except (TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("checkpoint"):
            raise
        raise ValueError("checkpoint rng_state 结构无效") from exc


def _normalize_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """将外部 RNG mapping 转成 safe-loader 可验证的 CPU 表示。"""
    _validate_rng_state(state)
    numpy_state = _numpy_rng_tuple(state["numpy"])
    return {
        "python": copy.deepcopy(state["python"]),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.as_tensor(numpy_state[1], dtype=torch.uint32).cpu().clone(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.as_tensor(state["torch"], dtype=torch.uint8).cpu().clone(),
        "torch_cuda": [
            torch.as_tensor(cuda_state, dtype=torch.uint8).cpu().clone()
            for cuda_state in state["torch_cuda"]
        ],
    }


def restore_rng_state(
    state: Mapping[str, Any], *, strict_cuda: bool = False
) -> bool:
    """恢复 checkpoint 中的随机状态并返回 CUDA RNG 是否实际恢复。

    Python/NumPy/Torch CPU RNG 在 CPU-only 运行时始终恢复。checkpoint 若带
    CUDA RNG、但当前运行时没有 CUDA，默认跳过该部分并由调用方通过返回值显式
    感知降级；``strict_cuda=True`` 则把这种环境不匹配升级为错误。
    """
    _validate_rng_state(state)
    try:
        random.setstate(state["python"])
        np.random.set_state(_numpy_rng_tuple(state["numpy"]))
        torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("checkpoint rng_state 无法恢复") from exc
    cuda_states = state["torch_cuda"]
    if cuda_states and not torch.cuda.is_available():
        if strict_cuda:
            raise ValueError(
                "checkpoint 含 CUDA rng_state，但当前运行时没有可用 CUDA，无法恢复"
            )
        return False
    if cuda_states:
        runtime_device_count = int(torch.cuda.device_count())
        if runtime_device_count != len(cuda_states):
            if strict_cuda:
                raise ValueError(
                    "checkpoint CUDA rng_state device 数量与当前运行时不匹配："
                    f"saved={len(cuda_states)} runtime={runtime_device_count}"
                )
            return False
        try:
            expected_lengths = [
                int(torch.cuda.get_rng_state(device=index).numel())
                for index in range(runtime_device_count)
            ]
        except (TypeError, ValueError, RuntimeError) as exc:
            if strict_cuda:
                raise ValueError(
                    "无法读取当前 CUDA RNG state 长度，拒绝恢复"
                ) from exc
            return False
        if any(int(torch.as_tensor(state).numel()) != expected
               for state, expected in zip(cuda_states, expected_lengths)):
            if strict_cuda:
                raise ValueError(
                    "checkpoint CUDA rng_state 长度与当前设备不匹配"
                )
            return False
        try:
            torch.cuda.set_rng_state_all(cuda_states)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("checkpoint CUDA rng_state 无法恢复") from exc
        return True
    return False


def _runtime_metadata() -> dict[str, Any]:
    """记录可复现性所需的运行时版本，不参与契约语义 hash。"""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (None if torch.version.cuda is None
                                else str(torch.version.cuda)),
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy_version": np.__version__,
    }


def _contract_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        _jsonable(payload), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _component_state(component: Any, *, name: str) -> dict[str, Any] | None:
    if component is None:
        return None
    state = _state_copy(component, name=name)
    return _strip_module_prefix(state)


def _validate_component_contract(
    component: Any,
    mode: str | MethodMode,
    *,
    name: str,
    include_feature_schema: bool = True,
) -> None:
    """校验 mode-aware component 的声明，不猜测普通 vendor module 的语义。"""
    if component is None or isinstance(component, Mapping):
        return
    canonical = canonical_mode(mode)
    declared_mode = getattr(component, "mode", None)
    if declared_mode is not None:
        try:
            declared_mode = canonical_mode(declared_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 声明了非法 mode={declared_mode!r}") from exc
        if declared_mode != canonical:
            raise ValueError(
                f"{name} mode 不匹配：expected={canonical!r} actual={declared_mode!r}"
            )
    if not include_feature_schema:
        return
    declared_schema = getattr(component, "feature_schema", None)
    if declared_schema is not None:
        validate_feature_schema(declared_schema, canonical)
    declared_input_schema = getattr(component, "input_schema", None)
    if declared_input_schema is not None:
        validate_feature_schema(
            declared_input_schema, mode_spec(canonical).adapter_input_schema
        )


def _make_checkpoint_contract(
    *,
    mode: str | MethodMode,
    feature_schema: FeatureSchema | Mapping[str, Any] | None,
    adapter_input_schema: FeatureSchema | Mapping[str, Any] | None,
    dataset_config: Mapping[str, Any] | None,
    split_config: Any,
    sampling_config: Mapping[str, Any] | None,
    t_win: int | None,
    label_source: str | None,
    sampling_source: str | None,
    anchor_hash: str | None,
    calibration_policy: Mapping[str, Any] | None,
    warm_start_aux: bool,
) -> dict[str, Any]:
    canonical = canonical_mode(mode)
    spec = mode_spec(canonical)
    schema = spec.feature_schema if feature_schema is None else _coerce_schema(feature_schema)
    validate_feature_schema(schema, spec.feature_schema)
    adapter_schema = (spec.adapter_input_schema if adapter_input_schema is None
                      else _coerce_schema(adapter_input_schema))
    validate_feature_schema(adapter_schema, spec.adapter_input_schema)
    if not isinstance(warm_start_aux, bool):
        raise ValueError("warm_start_aux 必须是 bool")
    if warm_start_aux and canonical != MODE_B0:
        raise ValueError(
            f"warm_start_aux 只允许显式用于 {MODE_B0} auxiliary mode，实际 {canonical}"
        )
    if anchor_hash is not None:
        if not isinstance(anchor_hash, str) or not anchor_hash.strip():
            raise ValueError("anchor_hash 必须是非空字符串或 None")
        anchor_hash = anchor_hash.strip()
    if label_source is None:
        raise ValueError(
            "checkpoint 必须显式提供 label_source，禁止回退到旧默认监督来源"
        )
    label_source = validate_label_source(label_source)
    if label_source not in spec.formal_label_sources:
        raise ValueError(
            f"mode={canonical} 的 checkpoint label source 不匹配："
            f"expected={spec.formal_label_sources} actual={label_source!r}"
        )
    if label_source == LABEL_SOURCE_HALLER_TRAIN and anchor_hash is None:
        raise ValueError(
            f"mode={canonical} 的 Haller anchor checkpoint 必须显式提供 anchor_hash"
        )
    if sampling_source is not None:
        sampling_source = validate_label_source(sampling_source)
        if sampling_source in (LABEL_SOURCE_HALLER_CALIBRATION,
                               LABEL_SOURCE_HALLER_TEST):
            raise ValueError(
                f"checkpoint sampling_source 不能读取 {sampling_source!r}"
            )
    split = _normalize_split_config(split_config)
    _validate_source_split(
        label_source, split["split_name"], context="checkpoint label_source"
    )
    _validate_source_split(
        sampling_source, split["split_name"], context="checkpoint sampling_source"
    )
    sampling = _normalize_sampling_config(sampling_config, t_win)
    dataset = _mapping_or_empty(dataset_config, name="dataset_config")
    if not dataset:
        raise ValueError(
            "checkpoint 必须显式提供非空 dataset_config，禁止隐式数据集语义"
        )
    calibration = _validate_calibration_policy(
        _mapping_or_empty(calibration_policy, name="calibration_policy")
    )
    contract = {
        "mode": canonical,
        "feature_schema": schema.as_dict(),
        "adapter_input_schema": adapter_schema.as_dict(),
        "dataset_config": dataset,
        "split_config": split,
        "sampling_config": sampling,
        "label_source": label_source,
        "sampling_source": sampling_source,
        "anchor_hash": anchor_hash,
        "calibration_policy": calibration,
        "warm_start_aux": warm_start_aux,
    }
    return {**contract, "contract_hash": _contract_hash(contract)}


def save_checkpoint(
    path: str | pathlib.Path,
    student: nn.Module | Mapping[str, Any],
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    *,
    mode: str | MethodMode,
    feature_schema: FeatureSchema | Mapping[str, Any] | None = None,
    adapter_input_schema: FeatureSchema | Mapping[str, Any] | None = None,
    dataset_config: Mapping[str, Any] | None = None,
    split_config: Any = None,
    sampling_config: Mapping[str, Any] | None = None,
    t_win: int | None = None,
    label_source: str | None = None,
    sampling_source: str | None = None,
    teacher: nn.Module | Mapping[str, Any] | None = None,
    ema_teacher: nn.Module | Mapping[str, Any] | None = None,
    projection_head: nn.Module | Mapping[str, Any] | None = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Mapping[str, Any] | None = None,
    seed: int | None = None,
    rng_state: Mapping[str, Any] | None = None,
    anchor_hash: str | None = None,
    calibration_policy: Mapping[str, Any] | None = None,
    warm_start_aux: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> pathlib.Path:
    """保存带 mode/schema/source/reproducibility contract 的 checkpoint。

    ``student`` 是唯一必需模型；W1/W2/W3 按 mode 要求 EMA teacher，W3
    还要求 projection head。旧 B0 checkpoint 不通过此入口生成，加载旧文件
    必须显式声明 auxiliary warm-start。
    """
    if student is None:
        raise TypeError("checkpoint 必须提供 student model/state")
    if split_config is None:
        raise ValueError("checkpoint 必须显式提供 split_config，禁止隐式 split")
    if sampling_config is None and t_win is None:
        raise ValueError(
            "checkpoint 必须显式提供 sampling_config 或 t_win，禁止隐式窗口配置"
        )
    canonical = canonical_mode(mode)
    spec = mode_spec(canonical)
    if teacher is not None and ema_teacher is not None:
        if not _state_equal(_component_state(teacher, name="teacher"),
                            _component_state(ema_teacher, name="ema_teacher")):
            raise ValueError("teacher 与 ema_teacher 同时提供但状态不一致")
    _validate_component_contract(student, canonical, name="student")
    _validate_component_contract(teacher, canonical, name="teacher")
    _validate_component_contract(ema_teacher, canonical, name="ema_teacher")
    _validate_component_contract(
        projection_head, canonical, name="projection_head", include_feature_schema=False
    )
    teacher = teacher if teacher is not None else ema_teacher
    if spec.requires_teacher and teacher is None:
        raise ValueError(f"mode={canonical} checkpoint 必须保存 EMA teacher")
    if teacher is not None and not spec.requires_teacher:
        raise ValueError(f"mode={canonical} 不允许保存 teacher state")
    if projection_head is not None and not spec.supports_projection_head:
        raise ValueError(f"mode={canonical} 不支持 projection_head")
    if canonical == MODE_W3 and projection_head is None:
        raise ValueError("mode=W3 checkpoint 必须保存 projection_head")
    epoch = _nonnegative_int(epoch, name="epoch")
    global_step = _nonnegative_int(global_step, name="global_step")
    if seed is None:
        raise ValueError("checkpoint 必须显式提供 seed，禁止丢失可复现性信息")
    seed = _nonnegative_int(seed, name="seed")
    contract = _make_checkpoint_contract(
        mode=canonical, feature_schema=feature_schema,
        adapter_input_schema=adapter_input_schema, dataset_config=dataset_config,
        split_config=split_config, sampling_config=sampling_config, t_win=t_win,
        label_source=label_source, sampling_source=sampling_source,
        anchor_hash=anchor_hash, calibration_policy=calibration_policy,
        warm_start_aux=warm_start_aux,
    )
    extra = _mapping_or_empty(extra_metadata, name="extra_metadata")
    _reject_test_only_keys(extra, context="checkpoint extra_metadata")
    forbidden_extra_sources = {
        LABEL_SOURCE_HALLER_CALIBRATION,
        LABEL_SOURCE_HALLER_TEST,
    }
    leaked_extra_source = next(
        (source for source in _provenance_sources(extra)
         if source in forbidden_extra_sources),
        None,
    )
    if leaked_extra_source is not None:
        raise ValueError(
            f"checkpoint extra_metadata 禁止携带 {leaked_extra_source!r}；"
            "calibration/test GT 必须通过专用 contract seam 管理"
        )
    rng = _normalize_rng_state(rng_state) if rng_state is not None else capture_rng_state()
    blob = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "mode": canonical,
        "feature_schema": contract["feature_schema"],
        "channel_order": list(contract["feature_schema"]["channels"]),
        "adapter_input_schema": contract["adapter_input_schema"],
        "dataset_config": contract["dataset_config"],
        "split_config": contract["split_config"],
        "split": contract["split_config"].get("split_name"),
        "sampling_config": contract["sampling_config"],
        "t_win": contract["sampling_config"].get("t_win"),
        "label_source": contract["label_source"],
        "sampling_source": contract["sampling_source"],
        "student": _component_state(student, name="student"),
        "teacher": _component_state(teacher, name="teacher"),
        "ema_teacher": _component_state(teacher, name="ema_teacher") if teacher is not None else None,
        "projection_head": _component_state(projection_head, name="projection_head"),
        "optimizer": optimizer.state_dict() if optimizer is not None else {},
        "scheduler": scheduler.state_dict() if scheduler is not None else {},
        "epoch": epoch,
        "global_step": global_step,
        "metrics": _mapping_or_empty(metrics, name="metrics"),
        "runtime": _runtime_metadata(),
        "seed": seed,
        "rng_state": rng,
        "anchor_hash": contract["anchor_hash"],
        "calibration_policy": contract["calibration_policy"],
        "warm_start_aux": warm_start_aux,
        "contract": contract,
        "contract_hash": contract["contract_hash"],
        "extra_metadata": extra,
    }
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(blob, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        try:
            directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                os.close(directory_descriptor)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return path


def _torch_load(path: str | pathlib.Path, device: str | torch.device) -> Any:
    """Load the contract blob on CPU before any target-device transfer.

    RNG state is part of the metadata contract and must remain CPU-backed while
    validation runs.  Model and optimizer states are copied to their targets by
    ``load_checkpoint`` after the contract has passed validation.
    """
    del device
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ValueError(
            "checkpoint 未通过 safe weights_only 加载；拒绝回退到不安全 pickle"
        ) from exc
    except (pickle.UnpicklingError, RuntimeError) as exc:
        raise ValueError(
            "checkpoint 未通过安全 weights-only 加载；拒绝回退到不安全 pickle"
        ) from exc


def _validate_saved_contract(blob: Mapping[str, Any]) -> dict[str, Any]:
    if blob.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"不支持 checkpoint format_version={blob.get('format_version')!r}"
        )
    required = (
        "mode", "feature_schema", "channel_order", "adapter_input_schema",
        "dataset_config", "split_config", "split", "sampling_config", "t_win",
        "student",
        "label_source", "sampling_source",
        "teacher", "projection_head", "optimizer", "scheduler", "epoch",
        "global_step", "metrics", "runtime", "seed", "rng_state",
        "anchor_hash", "calibration_policy", "warm_start_aux", "contract",
        "contract_hash",
    )
    missing = [key for key in required if key not in blob]
    if missing:
        raise ValueError(f"checkpoint contract 缺少字段 {missing}")
    contract = blob["contract"]
    if not isinstance(contract, Mapping):
        raise ValueError("checkpoint contract 必须是 object")
    contract = dict(contract)
    expected_hash = _contract_hash({key: contract[key] for key in (
        "mode", "feature_schema", "adapter_input_schema", "dataset_config",
        "split_config", "sampling_config", "label_source", "sampling_source",
        "anchor_hash",
        "calibration_policy", "warm_start_aux")}) if all(
            key in contract for key in (
                "mode", "feature_schema", "adapter_input_schema", "dataset_config",
                "split_config", "sampling_config", "label_source", "sampling_source",
                "anchor_hash",
                "calibration_policy", "warm_start_aux")) else None
    if expected_hash is None or contract.get("contract_hash") != expected_hash:
        raise ValueError("checkpoint contract_hash 校验失败")
    if blob.get("contract_hash") != contract.get("contract_hash"):
        raise ValueError("checkpoint top-level contract_hash 与 contract 不一致")
    canonical = canonical_mode(blob["mode"])
    spec = mode_spec(canonical)
    if contract.get("mode") != canonical:
        raise ValueError("checkpoint mode 与 contract 不一致")
    validate_feature_schema(blob["feature_schema"], canonical)
    validate_feature_schema(contract["feature_schema"], canonical)
    validate_feature_schema(blob["adapter_input_schema"], mode_spec(canonical).adapter_input_schema)
    if blob["channel_order"] != blob["feature_schema"]["channels"]:
        raise ValueError("checkpoint channel_order 与 feature_schema 不一致")
    for key in ("feature_schema", "adapter_input_schema", "dataset_config",
                "split_config", "sampling_config", "label_source", "sampling_source",
                "anchor_hash",
                "calibration_policy", "warm_start_aux"):
        if blob[key] != contract[key]:
            raise ValueError(f"checkpoint {key} 与 contract 不一致")
    if not isinstance(blob["student"], Mapping):
        raise ValueError("checkpoint 缺少 student state")
    if not isinstance(blob["warm_start_aux"], bool):
        raise ValueError("checkpoint warm_start_aux 必须是 bool")
    label_source = blob["label_source"]
    if label_source is None:
        raise ValueError(
            "checkpoint 缺少显式 label_source，禁止回退到旧默认监督来源"
        )
    label_source = validate_label_source(label_source)
    if label_source not in spec.formal_label_sources:
        raise ValueError(
            f"mode={canonical} 的 checkpoint label source 不匹配："
            f"expected={spec.formal_label_sources} actual={label_source!r}"
        )
    sampling_source = blob["sampling_source"]
    if sampling_source is not None:
        sampling_source = validate_label_source(sampling_source)
        if sampling_source in (LABEL_SOURCE_HALLER_CALIBRATION,
                               LABEL_SOURCE_HALLER_TEST):
            raise ValueError(
                f"checkpoint sampling_source 不能读取 {sampling_source!r}"
            )
    if label_source == LABEL_SOURCE_HALLER_TRAIN and blob["anchor_hash"] is None:
        raise ValueError(
            f"mode={canonical} 的 Haller anchor checkpoint 缺少 anchor_hash"
        )
    if blob["anchor_hash"] is not None and (
            not isinstance(blob["anchor_hash"], str)
            or not blob["anchor_hash"].strip()
    ):
        raise ValueError("checkpoint anchor_hash 必须是非空字符串或 None")
    if blob["warm_start_aux"] and canonical != MODE_B0:
        raise ValueError(
            f"warm_start_aux 只允许用于 {MODE_B0} auxiliary checkpoint，实际 {canonical}"
        )
    for key in ("dataset_config", "split_config", "sampling_config",
                "calibration_policy", "metrics", "runtime", "rng_state"):
        if not isinstance(blob[key], Mapping):
            raise ValueError(f"checkpoint {key} 必须是 object")
    if blob["split"] != blob["split_config"].get("split_name"):
        raise ValueError("checkpoint split 与 split_config.split_name 不一致")
    if blob["t_win"] != blob["sampling_config"].get("t_win"):
        raise ValueError("checkpoint t_win 与 sampling_config.t_win 不一致")
    if not blob["dataset_config"]:
        raise ValueError("checkpoint dataset_config 不能为空")
    _validate_rng_state(blob["rng_state"])
    _validate_calibration_policy(blob["calibration_policy"])
    normalized_split = _normalize_split_config(blob["split_config"])
    if normalized_split != dict(blob["split_config"]):
        raise ValueError("checkpoint split_config 未通过显式 split 规范化校验")
    _validate_source_split(
        label_source, normalized_split["split_name"],
        context="checkpoint label_source"
    )
    _validate_source_split(
        sampling_source, normalized_split["split_name"],
        context="checkpoint sampling_source"
    )
    if _normalize_sampling_config(blob["sampling_config"], None) != dict(blob["sampling_config"]):
        raise ValueError("checkpoint sampling_config 未通过显式窗口规范化校验")
    for key in ("teacher", "ema_teacher", "projection_head"):
        if blob.get(key) is not None and not isinstance(blob[key], Mapping):
            raise ValueError(f"checkpoint {key} state 必须是 object")
    stored_teacher = blob.get("teacher")
    stored_ema_teacher = blob.get("ema_teacher")
    if (stored_teacher is not None and stored_ema_teacher is not None
            and not _state_equal(stored_teacher, stored_ema_teacher)):
        raise ValueError("checkpoint teacher 与 ema_teacher state 不一致")
    stored_teacher = (stored_teacher if stored_teacher is not None
                      else stored_ema_teacher)
    if spec.requires_teacher and stored_teacher is None:
        raise ValueError(f"mode={canonical} checkpoint 缺少 EMA teacher")
    if not spec.requires_teacher and stored_teacher is not None:
        raise ValueError(f"mode={canonical} checkpoint 不应包含 teacher state")
    if spec.supports_projection_head and blob.get("projection_head") is None:
        raise ValueError(f"mode={canonical} checkpoint 缺少 projection_head")
    if (not spec.supports_projection_head
            and blob.get("projection_head") is not None):
        raise ValueError(f"mode={canonical} checkpoint 不应包含 projection_head")
    extra = blob.get("extra_metadata", {})
    if not isinstance(extra, Mapping):
        raise ValueError("checkpoint extra_metadata 必须是 object")
    _reject_test_only_keys(extra, context="checkpoint extra_metadata")
    leaked_extra_source = next(
        (source for source in _provenance_sources(extra)
         if source in (LABEL_SOURCE_HALLER_CALIBRATION, LABEL_SOURCE_HALLER_TEST)),
        None,
    )
    if leaked_extra_source is not None:
        raise ValueError(
            f"checkpoint extra_metadata 禁止携带 {leaked_extra_source!r}"
        )
    _nonnegative_int(blob["epoch"], name="checkpoint epoch")
    _nonnegative_int(blob["global_step"], name="checkpoint global_step")
    _nonnegative_int(blob["seed"], name="checkpoint seed")
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "mode": canonical,
        "feature_schema": dict(blob["feature_schema"]),
        "adapter_input_schema": dict(blob["adapter_input_schema"]),
        "dataset_config": dict(blob["dataset_config"]),
        "split_config": dict(blob["split_config"]),
        "sampling_config": dict(blob["sampling_config"]),
        "label_source": label_source,
        "sampling_source": sampling_source,
        "anchor_hash": blob["anchor_hash"],
        "calibration_policy": dict(blob["calibration_policy"]),
        "extra_metadata": copy.deepcopy(dict(extra)),
        "warm_start_aux": blob["warm_start_aux"],
        "contract_hash": contract["contract_hash"],
    }


def _check_expected_contract(
    blob: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    expected_mode: str | MethodMode | None,
    expected_feature_schema: FeatureSchema | Mapping[str, Any] | None,
    expected_dataset_config: Mapping[str, Any] | None,
    expected_split_config: Any,
    expected_split: str | None,
    expected_sampling_config: Mapping[str, Any] | None,
    expected_label_source: str | None,
    expected_sampling_source: str | None,
    expected_anchor_hash: Any,
) -> None:
    if expected_mode is not None and metadata["mode"] != canonical_mode(expected_mode):
        raise ValueError(
            f"checkpoint mode 不匹配：expected={canonical_mode(expected_mode)!r} "
            f"actual={metadata['mode']!r}"
        )
    if expected_feature_schema is not None:
        validate_feature_schema(metadata["feature_schema"], expected_feature_schema)
    if expected_dataset_config is not None:
        actual = _mapping_or_empty(metadata["dataset_config"], name="dataset_config")
        expected = _mapping_or_empty(expected_dataset_config, name="expected_dataset_config")
        if actual != expected:
            raise ValueError(
                f"checkpoint dataset_config 不匹配：expected={expected} actual={actual}"
            )
    if expected_split_config is not None:
        expected = _normalize_split_config(expected_split_config)
        if metadata["split_config"] != expected:
            raise ValueError(
                f"checkpoint split_config 不匹配：expected={expected} "
                f"actual={metadata['split_config']}"
            )
    if expected_split is not None:
        actual_split = metadata["split_config"].get("split_name")
        if actual_split != str(expected_split):
            raise ValueError(
                f"checkpoint split 不匹配：expected={expected_split!r} actual={actual_split!r}"
            )
    if expected_sampling_config is not None:
        expected = _mapping_or_empty(expected_sampling_config, name="expected_sampling_config")
        if metadata["sampling_config"] != expected:
            raise ValueError(
                f"checkpoint sampling_config 不匹配：expected={expected} "
                f"actual={metadata['sampling_config']}"
            )
    if expected_label_source is not None:
        expected = validate_label_source(expected_label_source)
        if metadata["label_source"] != expected:
            raise ValueError(
                f"checkpoint label_source 不匹配：expected={expected!r} "
                f"actual={metadata['label_source']!r}"
            )
    if expected_sampling_source is not None:
        expected = validate_label_source(expected_sampling_source)
        if metadata["sampling_source"] != expected:
            raise ValueError(
                f"checkpoint sampling_source 不匹配：expected={expected!r} "
                f"actual={metadata['sampling_source']!r}"
            )
    if expected_anchor_hash is not _MISSING and metadata["anchor_hash"] != expected_anchor_hash:
        raise ValueError(
            f"checkpoint anchor_hash 不匹配：expected={expected_anchor_hash!r} "
            f"actual={metadata['anchor_hash']!r}"
        )


def _infer_target_contract(
    student: nn.Module | None,
    teacher: nn.Module | None,
    ema_teacher: nn.Module | None,
) -> tuple[str | None, FeatureSchema | None]:
    """从 mode-aware target 推导默认 mode/schema，避免目标语义被静默忽略。"""
    inferred_mode = None
    inferred_schema = None
    for target in (student, teacher, ema_teacher):
        if target is None:
            continue
        target_mode = getattr(target, "mode", None)
        canonical = None if target_mode is None else canonical_mode(target_mode)
        if canonical is not None:
            if inferred_mode is not None and inferred_mode != canonical:
                raise ValueError(
                    "student/teacher target 声明了互相冲突的 mode："
                    f"{inferred_mode!r} 与 {canonical!r}"
                )
            inferred_mode = canonical
        target_schema = getattr(target, "feature_schema", None)
        if target_schema is not None:
            schema = _coerce_schema(target_schema)
            if inferred_schema is not None and inferred_schema != schema:
                raise ValueError("student/teacher target 声明了互相冲突的 feature schema")
            inferred_schema = schema
    return inferred_mode, inferred_schema


def load_checkpoint(
    path: str | pathlib.Path,
    student: nn.Module | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    *,
    teacher: nn.Module | None = None,
    ema_teacher: nn.Module | None = None,
    projection_head: nn.Module | None = None,
    device: str | torch.device = "cpu",
    expected_mode: str | MethodMode | None = None,
    expected_feature_schema: FeatureSchema | Mapping[str, Any] | None = None,
    expected_dataset_config: Mapping[str, Any] | None = None,
    expected_split_config: Any = None,
    expected_split: str | None = None,
    expected_sampling_config: Mapping[str, Any] | None = None,
    expected_label_source: str | None = None,
    expected_sampling_source: str | None = None,
    expected_anchor_hash: Any = _MISSING,
    restore_rng: bool | None = None,
    strict_cuda_rng: bool = False,
    load_mode: str = "resume",
    warm_start_aux: bool = False,
    allow_legacy_b0: bool = False,
) -> dict[str, Any]:
    """加载并验证 contract；不兼容 mode/schema/split/hash 立即拒绝。"""
    if load_mode not in ("resume", "inference"):
        raise ValueError(
            f"load_mode={load_mode!r} 非法；必须是 'resume' 或 'inference'"
        )
    if restore_rng is None:
        restore_rng = load_mode == "resume"
    for name, value in (
        ("restore_rng", restore_rng),
        ("strict_cuda_rng", strict_cuda_rng),
        ("warm_start_aux", warm_start_aux),
        ("allow_legacy_b0", allow_legacy_b0),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} 必须是 bool")
    blob = _torch_load(path, device)
    if not isinstance(blob, Mapping):
        raise ValueError("checkpoint 顶层必须是 object")
    format_version = blob.get("format_version", _MISSING)
    if format_version != CHECKPOINT_FORMAT_VERSION:
        if format_version is not _MISSING:
            raise ValueError(
                f"不支持 checkpoint format_version={format_version!r}；"
                "仅缺少 format_version 的 blob 可按 legacy B0 处理"
            )
        # 阶段 0 的旧 B0 blob 只有在调用方显式声明 auxiliary 时可读。
        explicit_aux = warm_start_aux or allow_legacy_b0
        requested_mode = None if expected_mode is None else canonical_mode(expected_mode)
        if not explicit_aux or requested_mode != MODE_B0:
            raise ValueError(
                "legacy B0 checkpoint 只能在显式 warm_start_aux=True 的 auxiliary mode 下加载"
            )
        if (expected_feature_schema is not None
                or expected_dataset_config is not None
                or expected_split_config is not None
                or expected_split is not None
                or expected_sampling_config is not None
                or expected_label_source is not None
                or expected_sampling_source is not None
                or expected_anchor_hash is not _MISSING):
            raise ValueError(
                "legacy B0 checkpoint 没有新 contract，不能静默忽略 expected "
                "schema/split/source/hash；请使用新的 checkpoint"
            )
        if "model" not in blob or not isinstance(blob["model"], Mapping):
            raise ValueError("legacy B0 checkpoint 缺少 model state")
        _validate_component_contract(student, MODE_B0, name="student")
        if student is not None:
            _load_state(student, blob["model"], name="student")
        if optimizer is not None and blob.get("optimizer"):
            optimizer.load_state_dict(blob["optimizer"])
        if scheduler is not None and blob.get("scheduler"):
            scheduler.load_state_dict(blob["scheduler"])
        return {
            "format_version": None,
            "mode": MODE_B0,
            "feature_schema": None,
            "warm_start_aux": True,
            "legacy": True,
            "epoch": int(blob.get("epoch", -1)),
            "start_epoch": int(blob.get("epoch", -1)) + 1,
            "global_step": int(blob.get("global_step", 0)),
            "metrics": dict(blob.get("metrics", {})),
            "rng_restored": False,
            "cuda_rng_restored": False,
            "load_mode": load_mode,
        }
    metadata = _validate_saved_contract(blob)
    inferred_mode, inferred_schema = _infer_target_contract(
        student, teacher, ema_teacher
    )
    effective_mode = expected_mode if expected_mode is not None else inferred_mode
    if effective_mode is None:
        raise ValueError(
            "load_checkpoint 必须显式提供 expected_mode，或传入带 mode 属性的 target"
        )
    if expected_split_config is None and expected_split is None:
        raise ValueError(
            "load_checkpoint 必须显式提供 expected_split 或 expected_split_config"
        )
    if (metadata["label_source"] == LABEL_SOURCE_HALLER_TRAIN
            and expected_anchor_hash is _MISSING):
        raise ValueError(
            "Haller checkpoint load 必须显式提供 expected_anchor_hash"
        )
    _check_expected_contract(
        blob, metadata,
        expected_mode=effective_mode,
        expected_feature_schema=(expected_feature_schema
                                 if expected_feature_schema is not None
                                 else inferred_schema),
        expected_dataset_config=expected_dataset_config,
        expected_split_config=expected_split_config,
        expected_split=expected_split,
        expected_sampling_config=expected_sampling_config,
        expected_label_source=expected_label_source,
        expected_sampling_source=expected_sampling_source,
        expected_anchor_hash=expected_anchor_hash,
    )
    if metadata["warm_start_aux"] and not warm_start_aux:
        raise ValueError(
            "checkpoint 标记 warm_start_aux=True；加载 auxiliary checkpoint 必须显式传 warm_start_aux=True"
        )
    spec = mode_spec(metadata["mode"])
    _validate_component_contract(student, metadata["mode"], name="student")
    _validate_component_contract(teacher, metadata["mode"], name="teacher")
    _validate_component_contract(ema_teacher, metadata["mode"], name="ema_teacher")
    _validate_component_contract(
        projection_head,
        metadata["mode"],
        name="projection_head",
        include_feature_schema=False,
    )
    stored_teacher = blob.get("teacher")
    stored_alias = blob.get("ema_teacher")
    if stored_teacher is not None and stored_alias is not None and not _state_equal(
            stored_teacher, stored_alias):
        raise ValueError("checkpoint teacher 与 ema_teacher state 不一致")
    stored_teacher = stored_teacher if stored_teacher is not None else stored_alias
    has_state_target = any(
        target is not None
        for target in (student, teacher, ema_teacher, projection_head, optimizer, scheduler)
    )
    if load_mode == "resume" and has_state_target:
        if student is None:
            raise ValueError("resume 必须提供 student target")
        if stored_teacher is not None and teacher is None and ema_teacher is None:
            raise ValueError("resume 不能静默丢失 checkpoint 的 EMA teacher target")
        if blob.get("projection_head") is not None and projection_head is None:
            raise ValueError("resume 不能静默丢失 checkpoint 的 projection_head target")
        if not blob.get("optimizer"):
            raise ValueError("resume checkpoint 缺少 optimizer state")
        if not blob.get("scheduler"):
            raise ValueError("resume checkpoint 缺少 scheduler state")
        if optimizer is None:
            raise ValueError("resume 不能静默丢失 checkpoint 的 optimizer state")
        if scheduler is None:
            raise ValueError("resume 不能静默丢失 checkpoint 的 scheduler state")
    if spec.requires_teacher and stored_teacher is None:
        raise ValueError(f"mode={metadata['mode']} checkpoint 缺少 EMA teacher")
    if metadata["mode"] == MODE_W3 and blob.get("projection_head") is None:
        raise ValueError("mode=W3 checkpoint 缺少 projection_head")
    if student is not None:
        _load_state(student, blob["student"], name="student")
    if stored_teacher is not None:
        if teacher is not None and ema_teacher is not None and teacher is not ema_teacher:
            raise ValueError("teacher 与 ema_teacher target 不能同时指向不同对象")
        teacher_target = teacher if teacher is not None else ema_teacher
        if teacher_target is not None:
            _load_state(teacher_target, stored_teacher, name="teacher")
    elif teacher is not None or ema_teacher is not None:
        raise ValueError("target 要求 EMA teacher，但 checkpoint 未保存")
    if blob.get("projection_head") is not None:
        if projection_head is not None:
            _load_state(projection_head, blob["projection_head"], name="projection_head")
    elif projection_head is not None:
        raise ValueError("target 要求 projection_head，但 checkpoint 未保存")
    if optimizer is not None:
        if not blob.get("optimizer"):
            raise ValueError("target optimizer 存在，但 checkpoint 没有 optimizer state")
        optimizer.load_state_dict(blob["optimizer"])
    if scheduler is not None:
        if not blob.get("scheduler"):
            raise ValueError("target scheduler 存在，但 checkpoint 没有 scheduler state")
        scheduler.load_state_dict(blob["scheduler"])
    rng_restored = False
    cuda_rng_restored = False
    if restore_rng:
        cuda_rng_restored = restore_rng_state(
            blob["rng_state"], strict_cuda=strict_cuda_rng
        )
        rng_restored = True
    result = dict(metadata)
    result.update({
        "epoch": int(blob["epoch"]),
        "start_epoch": int(blob["epoch"]) + 1,
        "global_step": int(blob["global_step"]),
        "metrics": dict(blob["metrics"]),
        "seed": blob["seed"],
        "label_source": blob["label_source"],
        "sampling_source": blob["sampling_source"],
        "rng_restored": rng_restored,
        "cuda_rng_restored": cuda_rng_restored,
        "load_mode": load_mode,
        "student": blob["student"],
        "teacher": stored_teacher,
        "projection_head": blob.get("projection_head"),
        "runtime": dict(blob["runtime"]),
    })
    return result


def checkpoint_metadata(
    path: str | pathlib.Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """读取 checkpoint contract metadata，不加载到模型也不改变 RNG。"""
    blob = _torch_load(path, device)
    if not isinstance(blob, Mapping) or blob.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("只有新的 weak-supervision checkpoint 支持 metadata 读取")
    metadata = _validate_saved_contract(blob)
    return {**metadata, "epoch": int(blob["epoch"]), "global_step": int(blob["global_step"]),
            "metrics": dict(blob["metrics"]), "seed": blob["seed"]}


# 更短的 dispatch 名称供后续票据使用；实现仍集中在本模块，避免调用方复制契约逻辑。
dispatch_model = build_model_for_mode
dispatch_loss = build_loss_for_mode
save_mode_checkpoint = save_checkpoint
load_mode_checkpoint = load_checkpoint
