"""W3 trajectory-level contrastive training seam。

本模块只在 ``vendor/DeepUtils`` 外部增加 W3 adapter、projection head 和
in-batch contrastive 组件。vendor backbone 仍只负责原有的逐迹线概率输出；
adapter 在 classifier 的输入边界取得 pre-classifier trajectory embedding，
因此不会把 vendor 内部实现改成 W3 专用分支。
"""

from __future__ import annotations

import copy
import os
import pathlib
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import weak_supervision_contract as contract
import w2
from w1_h import W1HBatch
from w1_p import (
    W1PConfig,
    _as_prediction_tensor,
    _prepare_ema_teacher,
    _strict_nonnegative_int,
    clone_ema_teacher,
    ramp_up_weight,
    update_ema_teacher,
)


class TrajectoryEmbeddingAdapter(nn.Module):
    """在 W3 model seam 同时暴露 classification probability 和 trajectory embedding。

    对现有 ``PathlineTransformerV0``，``fc`` 的 pre-hook 输入就是形状为
    ``[B, K, D]`` 的逐迹线 embedding。hook 只存在于一次 forward 期间，
    不修改或持久化 vendor module 的行为。
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        if isinstance(model, contract.ChannelSelectingAdapter):
            if model.mode != contract.MODE_W3:
                raise ValueError(
                    "TrajectoryEmbeddingAdapter 需要 mode=W3 的 channel adapter"
                )
            channel_adapter = model
        else:
            channel_adapter = contract.ChannelSelectingAdapter(model, contract.MODE_W3)
        self.channel_adapter = channel_adapter
        self.mode = contract.MODE_W3
        self.feature_schema = contract.FEATURE_SCHEMA_7
        self.input_schema = contract.FEATURE_SCHEMA_7

        classifier = getattr(self.channel_adapter.model, "fc", None)
        if not isinstance(classifier, nn.Module):
            raise ValueError(
                "W3 local adapter 需要 wrapped model 暴露 nn.Module classifier fc"
            )
        embedding_dim = getattr(classifier, "in_features", None)
        if embedding_dim is None:
            embedding_dim = getattr(self.channel_adapter.model, "dim", None)
        if embedding_dim is None:
            raise ValueError(
                "W3 local adapter 无法推断 trajectory embedding dimension；"
                "wrapped model 必须暴露 fc.in_features 或 dim"
            )
        if isinstance(embedding_dim, bool) or int(embedding_dim) != embedding_dim:
            raise ValueError("W3 trajectory embedding dimension 必须是整数")
        self.embedding_dim = int(embedding_dim)

    def _forward_classification(self, data: Any, *, consumer: str) -> Any:
        if hasattr(data, "contract_batch"):
            return self.channel_adapter.forward_batch(
                data.contract_batch,
                dummy_field=data.dummy_field,
                consumer=consumer,
            )
        if isinstance(data, contract.WeakSupervisionBatch):
            return self.channel_adapter.forward_batch(data, consumer=consumer)
        if isinstance(data, (tuple, list)):
            if len(data) != 2:
                raise ValueError("W3 model input 必须是 (dummy_field, pathlines) 二元组")
            return self.channel_adapter(data, consumer=consumer)
        return self.channel_adapter(data, consumer=consumer)

    def forward(self, data: Any, *, consumer: str = "train") -> Any:
        """保持原有 model seam，只返回逐迹线 classification probability。"""
        return self._forward_classification(data, consumer=consumer)

    def forward_with_embedding(
        self,
        data: Any,
        *,
        consumer: str = "train",
    ) -> tuple[Any, torch.Tensor]:
        """执行一次 forward，返回 ``(probability, [B,K,D] embedding)``。"""
        captured: list[torch.Tensor] = []

        def capture_input(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("W3 classifier 未提供 tensor pre-classifier embedding")
            captured.append(args[0])

        classifier = getattr(self.channel_adapter.model, "fc", None)
        if not isinstance(classifier, nn.Module):
            raise ValueError(
                "W3 local adapter 需要 wrapped model 暴露 nn.Module classifier fc"
            )
        handle = classifier.register_forward_pre_hook(capture_input)
        try:
            probability = self._forward_classification(data, consumer=consumer)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                "W3 classifier pre-hook 必须恰好捕获一次 trajectory embedding，"
                f"实际 {len(captured)} 次"
            )
        embedding = captured[0]
        if embedding.ndim != 3 or int(embedding.shape[-1]) != self.embedding_dim:
            raise ValueError(
                "W3 trajectory embedding 必须是 [B,K,D] 且 D 与 classifier 一致，"
                f"实际 shape={tuple(embedding.shape)} D={self.embedding_dim}"
            )
        return probability, embedding


class TrajectoryProjectionHead(nn.Module):
    """把 backbone 的逐迹线 embedding 投影到固定的 64 维空间。"""

    def __init__(self, input_dim: int, projection_dim: int = 64):
        super().__init__()
        if isinstance(input_dim, bool) or int(input_dim) != input_dim or int(input_dim) <= 0:
            raise ValueError("W3 projection head input_dim 必须是正整数")
        if projection_dim != 64:
            raise ValueError("W3 projection dimension 必须固定为 64")
        self.input_dim = int(input_dim)
        self.projection_dim = 64
        self.projection = nn.Linear(self.input_dim, self.projection_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """保持前导 batch/trajectory 维度，只改变最后的 feature 维度。"""
        if not isinstance(embedding, torch.Tensor) or embedding.ndim < 2:
            raise ValueError("W3 projection head 输入必须至少是二维 tensor")
        if int(embedding.shape[-1]) != self.input_dim:
            raise ValueError(
                "W3 projection head 输入维度不匹配："
                f"expected={self.input_dim} actual={int(embedding.shape[-1])}"
            )
        return self.projection(embedding)


W3_DEFAULT_VIEW_COUNT = 2
W3_DEFAULT_MAX_EMBEDDINGS = 512
W3_DEFAULT_TEMPERATURE = 0.1
W3_PAIR_STATS_BATCH = "batch"
W3_PAIR_STATS_EPOCH = "epoch"


def _coerce_two_views(view_embeddings: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """统一两视图输入，并拒绝缺视图、shape 漂移和非有限 embedding。"""
    if isinstance(view_embeddings, torch.Tensor):
        if view_embeddings.ndim < 3 or int(view_embeddings.shape[0]) != 2:
            raise ValueError(
                "W3 view embeddings tensor 必须是 [2,...,D]，"
                f"实际 shape={tuple(view_embeddings.shape)}"
            )
        views = (view_embeddings[0], view_embeddings[1])
    else:
        if isinstance(view_embeddings, (str, bytes)):
            raise TypeError("W3 view embeddings 必须是两个 tensor")
        try:
            values = tuple(view_embeddings)
        except TypeError as exc:
            raise TypeError("W3 view embeddings 必须是两个 tensor") from exc
        if len(values) != W3_DEFAULT_VIEW_COUNT:
            raise ValueError("W3 stochastic view count 必须固定为 2")
        views = values
    if any(not isinstance(view, torch.Tensor) for view in views):
        raise TypeError("W3 每个 view embedding 必须是 torch.Tensor")
    if views[0].ndim < 2 or views[1].ndim < 2:
        raise ValueError("W3 view embedding 必须至少是二维 tensor")
    if tuple(views[0].shape) != tuple(views[1].shape):
        raise ValueError(
            "W3 两个 stochastic view 的 embedding shape 必须一致："
            f"{tuple(views[0].shape)} != {tuple(views[1].shape)}"
        )
    if views[0].device != views[1].device:
        raise ValueError("W3 两个 stochastic view 必须位于同一 device")
    if not all(torch.is_floating_point(view) for view in views):
        raise TypeError("W3 view embedding 必须是浮点 tensor")
    if not all(bool(torch.isfinite(view).all()) for view in views):
        raise ValueError("W3 view embedding 含有非有限值")
    return views


def _coerce_identity_mask(mask: Any, shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    """把 known/reliable trajectory mask 规格化为和单视图前导维同形。"""
    result = torch.as_tensor(mask, device=device)
    if tuple(result.shape) != shape:
        raise ValueError(
            "W3 reliable identity mask shape 不匹配："
            f"expected={shape} actual={tuple(result.shape)}"
        )
    if result.dtype != torch.bool and not bool(((result == 0) | (result == 1)).all()):
        raise ValueError("W3 reliable identity mask 必须是 bool 或只包含 0/1")
    return result.to(dtype=torch.bool)


@dataclass
class TrajectoryPairSelection:
    """两视图 identity pair 选择结果，供 loss 和 checkpoint diagnostics 共用。"""

    embeddings: torch.Tensor
    positive_pairs: torch.Tensor
    negative_pairs: torch.Tensor
    identity_indices: torch.Tensor
    candidate_identity_count: int
    valid_identity_count: int
    unknown_exclusion_count: int
    cap_exclusion_count: int
    max_embeddings: int = W3_DEFAULT_MAX_EMBEDDINGS

    @property
    def effective_embedding_count(self) -> int:
        """实际送入 contrastive objective 的两视图 embedding 数。"""
        return int(self.embeddings.shape[0])

    @property
    def positive_pair_count(self) -> int:
        return int(self.positive_pairs.shape[0])

    @property
    def negative_pair_count(self) -> int:
        return int(self.negative_pairs.shape[0])

    @property
    def pair_count(self) -> int:
        return self.positive_pair_count


def _validate_embedding_cap(max_embeddings: int) -> int:
    if (
        isinstance(max_embeddings, bool)
        or int(max_embeddings) != max_embeddings
        or int(max_embeddings) != W3_DEFAULT_MAX_EMBEDDINGS
    ):
        raise ValueError("W3 max_embeddings 必须固定为两视图合计 512")
    return W3_DEFAULT_MAX_EMBEDDINGS


def build_trajectory_pairs(
    view_embeddings: Any,
    reliable_identity_mask: Any,
    *,
    max_embeddings: int = W3_DEFAULT_MAX_EMBEDDINGS,
) -> TrajectoryPairSelection:
    """构造确定性的 two-view positive 与 in-batch negative pair。

    ``reliable_identity_mask`` 只应把 Haller known anchor 或 W2 双门控通过的
    pseudo-label 标为 true；unknown、solid、invalid 和不可靠 pseudo-label
    在进入此公共 seam 前必须为 false。超过 cap 时按 flatten 后的 identity
    顺序截断，保证复现且不引入 memory bank。
    """
    cap = _validate_embedding_cap(max_embeddings)
    first, second = _coerce_two_views(view_embeddings)
    identity_shape = tuple(int(value) for value in first.shape[:-1])
    mask = _coerce_identity_mask(
        reliable_identity_mask,
        identity_shape,
        device=first.device,
    )
    first_flat = first.reshape(-1, first.shape[-1])
    second_flat = second.reshape(-1, second.shape[-1])
    candidate_count = int(mask.numel())
    all_reliable_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    max_identities = cap // W3_DEFAULT_VIEW_COUNT
    selected = all_reliable_indices[:max_identities]
    identity_count = int(selected.numel())
    embeddings = torch.cat((first_flat[selected], second_flat[selected]), dim=0)

    if identity_count:
        left = torch.arange(identity_count, device=first.device, dtype=torch.long)
        positive_pairs = torch.stack(
            (left, left + identity_count), dim=1
        )
        anchor_indices = torch.arange(
            2 * identity_count, device=first.device, dtype=torch.long
        )
        positive_index = torch.cat((left + identity_count, left), dim=0)
        all_other_indices = torch.arange(
            2 * identity_count, device=first.device, dtype=torch.long
        ).expand(2 * identity_count, -1)
        all_anchor_indices = anchor_indices[:, None].expand_as(all_other_indices)
        all_positive_indices = positive_index[:, None].expand_as(all_other_indices)
        # Remove self and the paired view from each anchor.  The remaining
        # entries are exactly the in-batch negatives, in deterministic order.
        keep = (
            (all_other_indices != all_anchor_indices)
            & (all_other_indices != all_positive_indices)
        )
        negative_pairs = torch.stack(
            (all_anchor_indices[keep], all_other_indices[keep]), dim=1
        )
    else:
        positive_pairs = torch.empty((0, 2), device=first.device, dtype=torch.long)
        negative_pairs = torch.empty((0, 2), device=first.device, dtype=torch.long)

    return TrajectoryPairSelection(
        embeddings=embeddings,
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        identity_indices=selected,
        candidate_identity_count=candidate_count,
        valid_identity_count=identity_count,
        unknown_exclusion_count=candidate_count - int(all_reliable_indices.numel()),
        cap_exclusion_count=int(all_reliable_indices.numel()) - identity_count,
        max_embeddings=cap,
    )


def _validate_temperature(temperature: float) -> float:
    try:
        value = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("W3 temperature 必须固定为 0.1") from exc
    if value != W3_DEFAULT_TEMPERATURE:
        raise ValueError("W3 temperature 必须固定为 0.1")
    return value


def compute_trajectory_contrastive_loss(
    view_embeddings: Any,
    reliable_identity_mask: Any,
    *,
    temperature: float = W3_DEFAULT_TEMPERATURE,
    max_embeddings: int = W3_DEFAULT_MAX_EMBEDDINGS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """计算无 memory bank 的 two-view in-batch InfoNCE。"""
    temperature = _validate_temperature(temperature)
    selection = build_trajectory_pairs(
        view_embeddings,
        reliable_identity_mask,
        max_embeddings=max_embeddings,
    )
    if selection.valid_identity_count == 0:
        loss = selection.embeddings.sum() * 0.0
    else:
        normalized = F.normalize(selection.embeddings, p=2.0, dim=-1)
        logits = torch.matmul(normalized, normalized.transpose(0, 1)) / temperature
        logits.fill_diagonal_(float("-inf"))
        identity_count = selection.valid_identity_count
        targets = torch.cat(
            (
                torch.arange(identity_count, 2 * identity_count, device=logits.device),
                torch.arange(identity_count, device=logits.device),
            )
        )
        loss = F.cross_entropy(logits, targets)
    stats = {
        "contrastive_loss": float(loss.detach().cpu()),
        "effective_embedding_count": selection.effective_embedding_count,
        "candidate_identity_count": selection.candidate_identity_count,
        "valid_identity_count": selection.valid_identity_count,
        "positive_pair_count": selection.positive_pair_count,
        "negative_pair_count": selection.negative_pair_count,
        "pair_count": selection.pair_count,
        "unknown_exclusion_count": selection.unknown_exclusion_count,
        "cap_exclusion_count": selection.cap_exclusion_count,
        "view_count": W3_DEFAULT_VIEW_COUNT,
        "pair_stats_scope": W3_PAIR_STATS_BATCH,
        "pair_batch_count": 1,
        "temperature": temperature,
        "max_embeddings": W3_DEFAULT_MAX_EMBEDDINGS,
        "memory_bank_used": False,
        "cross_gpu_gather": False,
        "single_gpu": True,
    }
    return loss, stats


W3_GENERATION_VERSION = "w3-trajectory-contrastive-v1"
W3_LABEL_SOURCE = contract.LABEL_SOURCE_HALLER_TRAIN
W3_ALLOWED_SAMPLING_SOURCES = w2.W2_ALLOWED_SAMPLING_SOURCES


@dataclass(frozen=True)
class W3Config(W1PConfig):
    """W3 的冻结 W2 gate、两视图、投影和资源配置。"""

    variance_gate: float | None = None
    view_count: int = W3_DEFAULT_VIEW_COUNT
    teacher_view_count: int = w2.W2_DEFAULT_VIEW_COUNT
    projection_dim: int = 64
    temperature: float = W3_DEFAULT_TEMPERATURE
    max_embeddings: int = W3_DEFAULT_MAX_EMBEDDINGS
    contrastive_weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        view_count = int(self.view_count)
        teacher_view_count = int(self.teacher_view_count)
        projection_dim = int(self.projection_dim)
        max_embeddings = int(self.max_embeddings)
        temperature = float(self.temperature)
        if isinstance(self.view_count, (bool, np.bool_)) or view_count != self.view_count:
            raise ValueError("W3 view_count 必须是整数")
        if view_count != W3_DEFAULT_VIEW_COUNT:
            raise ValueError("W3 stochastic view_count 必须固定为 2")
        if (
            isinstance(self.teacher_view_count, (bool, np.bool_))
            or teacher_view_count != self.teacher_view_count
            or teacher_view_count != w2.W2_DEFAULT_VIEW_COUNT
        ):
            raise ValueError("W3 W2 teacher_view_count 必须固定为 3")
        if (
            isinstance(self.projection_dim, (bool, np.bool_))
            or projection_dim != self.projection_dim
            or projection_dim != 64
        ):
            raise ValueError("W3 projection_dim 必须固定为 64")
        if (
            isinstance(self.max_embeddings, (bool, np.bool_))
            or max_embeddings != self.max_embeddings
            or max_embeddings != W3_DEFAULT_MAX_EMBEDDINGS
        ):
            raise ValueError("W3 max_embeddings 必须固定为两视图合计 512")
        if temperature != W3_DEFAULT_TEMPERATURE:
            raise ValueError("W3 temperature 必须固定为 0.1")
        contrastive_weight = float(self.contrastive_weight)
        if not np.isfinite(contrastive_weight) or contrastive_weight < 0.0:
            raise ValueError("W3 contrastive_weight 必须是非负有限数")
        if self.variance_gate is not None:
            gate = float(self.variance_gate)
            if not np.isfinite(gate) or not (0.0 <= gate <= 1.0):
                raise ValueError("W3 variance_gate 必须位于 [0,1] 或为 None")
            object.__setattr__(self, "variance_gate", gate)
        object.__setattr__(self, "view_count", view_count)
        object.__setattr__(self, "teacher_view_count", teacher_view_count)
        object.__setattr__(self, "projection_dim", projection_dim)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "max_embeddings", max_embeddings)
        object.__setattr__(self, "contrastive_weight", contrastive_weight)

    def as_w2_config(self) -> w2.W2Config:
        """把冻结 W3 配置映射为 W2 的三视图监督配置。"""
        return w2.W2Config(
            positive_percentile=self.positive_percentile,
            negative_percentile=self.negative_percentile,
            pseudo_high=self.pseudo_high,
            pseudo_low=self.pseudo_low,
            ema_decay=self.ema_decay,
            ramp_up_epochs=self.ramp_up_epochs,
            pseudo_weight=self.pseudo_weight,
            consistency_weight=self.consistency_weight,
            min_area=self.min_area,
            view_count=self.teacher_view_count,
            variance_gate=self.variance_gate,
        )

    def as_dict(self) -> dict[str, Any]:
        """返回 W3 checkpoint/config 的 JSON-friendly 冻结语义。"""
        result = super().as_dict()
        result.update({
            "generation_version": W3_GENERATION_VERSION,
            "label_source": W3_LABEL_SOURCE,
            "feature_schema": contract.FEATURE_SCHEMA_7.as_dict(),
            "view_count": self.view_count,
            "teacher_view_count": self.teacher_view_count,
            "projection_dim": self.projection_dim,
            "temperature": self.temperature,
            "max_embeddings": self.max_embeddings,
            "contrastive_weight": self.contrastive_weight,
            "variance_gate": self.variance_gate,
            "single_gpu": True,
            "memory_bank": False,
            "cross_gpu_gather": False,
            "w2_config": self.as_w2_config().as_dict()
            if self.variance_gate is not None else None,
        })
        return result


def _coerce_w3_config(config: W3Config | None) -> W3Config:
    """确保 W3 使用 calibration-selected global variance gate。"""
    config = W3Config() if config is None else config
    if not isinstance(config, W3Config):
        raise TypeError("W3 config 必须是 W3Config")
    if config.variance_gate is None:
        raise ValueError(
            "W3 必须显式提供 W2 calibration-selected global variance_gate；"
            "禁止隐式默认"
        )
    return config


def _calibration_policy_gate(policy: Any) -> float:
    """Read a selected calibration gate independently of the train-time gate."""
    if isinstance(policy, w2.W2CalibrationSelection):
        return w2._strict_gate(
            policy.variance_gate, name="calibration_policy.variance_gate"
        )
    if isinstance(policy, Mapping) and "variance_gate" in policy:
        return w2._strict_gate(
            policy["variance_gate"], name="calibration_policy.variance_gate"
        )
    raise ValueError(
        "W3 checkpoint 必须显式提供 calibration_policy.variance_gate"
    )


def _as_bool_mask(value: Any, shape: tuple[int, ...], *, name: str) -> Any:
    """将固体/失败 mask 规格化为 bool 并检查与 trajectory identity 对齐。"""
    if isinstance(value, torch.Tensor):
        result = value
        if tuple(result.shape) != shape:
            raise ValueError(
                f"W3 {name} shape 不匹配：expected={shape} actual={tuple(result.shape)}"
            )
        if result.dtype != torch.bool and not bool(
                ((result == 0) | (result == 1)).all()):
            raise ValueError(f"W3 {name} 必须是 bool 或只包含 0/1")
        return result.to(dtype=torch.bool)
    result = np.asarray(value)
    if tuple(result.shape) != shape:
        raise ValueError(
            f"W3 {name} shape 不匹配：expected={shape} actual={tuple(result.shape)}"
        )
    if result.dtype != np.bool_ and not np.all((result == 0) | (result == 1)):
        raise ValueError(f"W3 {name} 必须是 bool 或只包含 0/1")
    return result.astype(bool, copy=False)


def _mask_count(value: Any) -> int:
    return int(value.sum().item() if isinstance(value, torch.Tensor) else np.asarray(value).sum())


def _mask_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    result = torch.as_tensor(value, device=device)
    if result.dtype != torch.bool and not bool(((result == 0) | (result == 1)).all()):
        raise ValueError("W3 mask 必须是 bool 或只包含 0/1")
    return result.to(dtype=torch.bool)


def _anchor_provenance_records(
    provenance: Mapping[str, Any],
    anchor_metadata: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    """展开 direct/collated anchor records 以绑定同一份 Haller metadata。"""
    anchor = provenance.get("anchor", {})
    if not isinstance(anchor, Mapping):
        raise ValueError("W3 anchor provenance 必须是 object")
    records = [("anchor", anchor, anchor_metadata)]
    batches = provenance.get("batches")
    if batches is None:
        return records
    artifacts = anchor_metadata.get("batch_artifacts")
    if not isinstance(artifacts, (list, tuple)) or len(artifacts) != len(batches):
        raise ValueError(
            "W3 collated Haller provenance 必须与 anchor_metadata.batch_artifacts 一一对应"
        )
    for index, item in enumerate(batches):
        if not isinstance(item, Mapping):
            raise ValueError(f"W3 provenance.batches[{index}] 必须是 object")
        item_anchor = item.get("anchor", {})
        if not isinstance(item_anchor, Mapping):
            raise ValueError(
                f"W3 provenance.batches[{index}].anchor 必须是 object"
            )
        artifact = artifacts[index]
        if not isinstance(artifact, Mapping):
            raise ValueError(
                f"W3 anchor_metadata.batch_artifacts[{index}] 必须是 object"
            )
        records.append((f"batches[{index}].anchor", item_anchor, artifact))
    return records


def _validate_anchor_provenance_binding(
    provenance: Mapping[str, Any],
    anchor_metadata: Mapping[str, Any],
    *,
    anchor_hash: str,
) -> None:
    """拒绝 batch、collated artifact 和 checkpoint anchor 身份的漂移。"""
    fields = (
        "source", "algorithm_version", "parameter_hash", "input_hash",
        "mask_hash", "failure_count", "coverage", "literature",
        "legacy_p85_used", "fallback_used",
    )
    metadata_anchor_hash = anchor_metadata.get("anchor_hash")
    if metadata_anchor_hash not in (None, anchor_hash):
        raise ValueError("W3 anchor_metadata.anchor_hash 与 batch anchor_hash 不一致")
    for name, observed, expected in _anchor_provenance_records(
        provenance, anchor_metadata
    ):
        for field in fields:
            if field in observed and observed[field] != expected.get(field):
                raise ValueError(
                    f"W3 {name}.{field} 与 anchor_metadata 不一致"
                )
        observed_anchor_hash = observed.get("anchor_hash")
        if observed_anchor_hash not in (None, anchor_hash):
            raise ValueError(
                f"W3 {name}.anchor_hash 与 batch anchor_hash 不一致"
            )
        if "artifact_hash" in observed:
            expected_artifact_hash = expected.get("artifact_hash")
            if (expected_artifact_hash is not None
                    and observed["artifact_hash"] != expected_artifact_hash):
                raise ValueError(
                    f"W3 {name}.artifact_hash 与 anchor metadata 不一致"
                )
    artifact_hashes = dict(provenance.get("anchor", {})).get("artifact_hashes")
    batches = provenance.get("batches")
    if artifact_hashes is not None:
        if not isinstance(artifact_hashes, (list, tuple)):
            raise ValueError("W3 anchor.artifact_hashes 必须是 list")
        if batches is None or len(artifact_hashes) != len(batches):
            raise ValueError(
                "W3 anchor.artifact_hashes 必须与 collated batches 对齐"
            )
        for index, (item, artifact_hash) in enumerate(
            zip(batches, artifact_hashes)
        ):
            observed = dict(item.get("anchor", {}))
            if observed.get("artifact_hash") not in (None, artifact_hash):
                raise ValueError(
                    f"W3 provenance.batches[{index}] artifact hash 不一致"
                )


@dataclass
class W3Batch:
    """W3 batch：保持 7-channel local-IVD 输入并携带 W2/Haller provenance。"""

    contract_batch: contract.WeakSupervisionBatch
    solid_mask: Any
    failed_frame_mask: Any
    anchor_hash: str
    dummy_field: Any | None = None
    anchor_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        batch = contract.validate_training_batch(
            self.contract_batch, contract.MODE_W3)
        if batch.mode != contract.MODE_W3:
            raise ValueError("W3 batch 必须显式声明 mode=W3，禁止 mode=None")
        if batch.label_source != W3_LABEL_SOURCE:
            raise ValueError("W3 formal loss source 必须是 haller_anchor_train")
        if not isinstance(self.anchor_hash, str) or not self.anchor_hash.strip():
            raise ValueError("W3 anchor_hash 必须是非空字符串")
        self.anchor_hash = self.anchor_hash.strip()
        if batch.sampling_source not in W3_ALLOWED_SAMPLING_SOURCES:
            raise ValueError(
                f"W3 sampling_source 必须是 {sorted(W3_ALLOWED_SAMPLING_SOURCES)!r}"
            )
        has_window = "window" in batch.provenance
        has_batches = "batches" in batch.provenance
        has_windows = "windows" in batch.provenance
        if has_window and (has_batches or has_windows):
            raise ValueError(
                "W3 provenance 不能同时携带 window 与 collated windows"
            )
        if has_batches and has_windows:
            normalized_windows = w2._validate_w2_collated_windows(
                batch.provenance,
                split_name=batch.split_name,
                sampling_source=batch.sampling_source,
            )
            if batch.provenance.get("windows") != normalized_windows:
                raise ValueError(
                    "W3 provenance.batches 与 provenance.windows 不一致"
                )
            batch.provenance["windows"] = normalized_windows
        window = batch.provenance.get("window")
        if has_window:
            batch.provenance["window"] = w2._validate_w2_window_provenance(
                window,
                split_name=batch.split_name,
                sampling_source=batch.sampling_source,
            )
        else:
            w2._validate_w2_collated_windows(
                batch.provenance,
                split_name=batch.split_name,
                sampling_source=batch.sampling_source,
            )
        sampling_provenance = batch.provenance.get("sampling", {})
        if not isinstance(sampling_provenance, Mapping):
            raise ValueError("W3 sampling provenance 必须是 object")
        if sampling_provenance.get("source") not in (None, batch.sampling_source):
            raise ValueError(
                "W3 sampling provenance source 与 batch sampling_source 不一致"
            )
        shape = tuple(int(value) for value in batch.labels.shape)
        self.solid_mask = _as_bool_mask(self.solid_mask, shape, name="solid_mask")
        self.failed_frame_mask = _as_bool_mask(
            self.failed_frame_mask, shape, name="failed_frame_mask")
        known = _mask_tensor(batch.label_mask, device=(
            batch.labels.device if isinstance(batch.labels, torch.Tensor)
            else torch.device("cpu")))
        unknown = _mask_tensor(batch.unknown_mask, device=known.device)
        solid = _mask_tensor(self.solid_mask, device=known.device)
        failed = _mask_tensor(self.failed_frame_mask, device=known.device)
        if bool((solid & known).any()) or bool((solid & ~unknown).any()):
            raise ValueError("W3 solid_mask 必须只落在 unknown/ignored 区域")
        if bool((failed & known).any()) or bool((failed & ~unknown).any()):
            raise ValueError("W3 failed_frame_mask 必须只落在 unknown/ignored 区域")
        if bool((failed & solid).any()):
            raise ValueError("W3 failed_frame_mask 必须与 solid_mask 分离")
        anchor = dict(batch.provenance.get("anchor", {}))
        if anchor.get("source") not in (None, W3_LABEL_SOURCE):
            raise ValueError("W3 anchor provenance source 必须是 haller_anchor_train")
        if anchor.get("anchor_hash") not in (None, self.anchor_hash):
            raise ValueError("W3 anchor provenance hash 与 batch anchor_hash 不一致")
        self.anchor_metadata = w2._validate_checkpoint_anchor_metadata(
            self.anchor_metadata
        )
        _validate_anchor_provenance_binding(
            batch.provenance,
            self.anchor_metadata,
            anchor_hash=self.anchor_hash,
        )
        if self.dummy_field is None:
            if isinstance(batch.pathlines, torch.Tensor):
                self.dummy_field = batch.pathlines.new_zeros((shape[0], 1, 1, 1))
            else:
                self.dummy_field = np.zeros((shape[0], 1, 1, 1), dtype=np.float32)
        dummy_shape = getattr(self.dummy_field, "shape", ())
        if len(dummy_shape) < 1 or int(dummy_shape[0]) != shape[0]:
            raise ValueError("W3 dummy_field batch 维度与 labels 不一致")

    @property
    def pathlines(self) -> Any:
        """W3 model-facing 7-channel pathlines。"""
        return self.contract_batch.pathlines

    @property
    def labels(self) -> Any:
        return self.contract_batch.labels

    @property
    def label_mask(self) -> Any:
        return self.contract_batch.label_mask

    @property
    def unknown_mask(self) -> Any:
        return self.contract_batch.unknown_mask

    @property
    def sampling_source(self) -> str:
        return str(self.contract_batch.sampling_source)

    @property
    def label_source(self) -> str:
        return self.contract_batch.label_source

    def as_dict(self) -> dict[str, Any]:
        """返回 W3 batch provenance 摘要，不复制大规模 pathline 数组。"""
        result = self.contract_batch.as_dict()
        total = int(np.prod(self.labels.shape))
        result.update({
            "mode": contract.MODE_W3,
            "anchor_hash": self.anchor_hash,
            "solid_count": _mask_count(self.solid_mask),
            "failed_cell_count": _mask_count(self.failed_frame_mask),
            "anchor_coverage": (
                _mask_count(self.label_mask) / total if total else 0.0
            ),
            "unknown_coverage": (
                _mask_count(self.unknown_mask) / total if total else 0.0
            ),
            "contrastive_view_count": W3_DEFAULT_VIEW_COUNT,
            "teacher_view_count": w2.W2_DEFAULT_VIEW_COUNT,
        })
        result["anchor_metadata"] = copy.deepcopy(dict(self.anchor_metadata))
        return result

    def as_w2_batch(self) -> w2.W2Batch:
        """显式转换为 W2 loss 所需的同数据 view，保留 W3 输入不变。"""
        base = contract.WeakSupervisionBatch(
            pathlines=self.pathlines,
            labels=self.labels,
            label_source=self.label_source,
            split_name=self.contract_batch.split_name,
            feature_schema=contract.FEATURE_SCHEMA_7,
            label_mask=self.label_mask,
            unknown_mask=self.unknown_mask,
            sampling_source=self.sampling_source,
            provenance=self.contract_batch.provenance,
            mode=contract.MODE_W2,
            input_schema=contract.FEATURE_SCHEMA_7,
        )
        return w2.W2Batch(
            base,
            self.solid_mask,
            self.failed_frame_mask,
            self.anchor_hash,
            self.dummy_field,
            self.anchor_metadata,
        )

    def to(self, device: str | torch.device) -> "W3Batch":
        """搬运 batch 数组到 device，不丢失 source/split/provenance。"""
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
            mode=contract.MODE_W3,
            input_schema=batch.input_schema,
        )
        return W3Batch(
            converted,
            torch.as_tensor(self.solid_mask, device=device),
            torch.as_tensor(self.failed_frame_mask, device=device),
            self.anchor_hash,
            torch.as_tensor(self.dummy_field, device=device),
            self.anchor_metadata,
        )


def build_w3_batch_from_w2(batch: w2.W2Batch) -> W3Batch:
    """将已通过 W2 train guard 的 batch 显式升级为 W3 mode。"""
    if not isinstance(batch, w2.W2Batch):
        raise TypeError("W3 batch builder 必须消费 W2Batch")
    return W3Batch(
        contract.WeakSupervisionBatch(
            pathlines=batch.pathlines,
            labels=batch.labels,
            label_source=W3_LABEL_SOURCE,
            split_name=batch.contract_batch.split_name,
            feature_schema=contract.FEATURE_SCHEMA_7,
            label_mask=batch.label_mask,
            unknown_mask=batch.unknown_mask,
            sampling_source=batch.sampling_source,
            provenance=batch.contract_batch.provenance,
            mode=contract.MODE_W3,
            input_schema=contract.FEATURE_SCHEMA_7,
        ),
        batch.solid_mask,
        batch.failed_frame_mask,
        batch.anchor_hash,
        batch.dummy_field,
        batch.anchor_metadata,
    )


def build_w3_batch_from_w1_h(batch: W1HBatch) -> W3Batch:
    """从 W1-H train anchor batch 经过 W2 contract 再升级为 W3。"""
    return build_w3_batch_from_w2(w2.build_w2_batch_from_w1_h(batch))


def _reliable_identity_mask(
    batch: W3Batch,
    teacher_predictions: Any,
    *,
    config: W3Config,
) -> tuple[torch.Tensor, dict[str, int]]:
    """只允许 Haller known 或 W2 双门控 pseudo identity 进入语义 pair。"""
    teacher_statistics = (
        teacher_predictions
        if isinstance(teacher_predictions, w2.W2Statistics)
        else w2.compute_w2_statistics(teacher_predictions)
    )
    device = teacher_statistics.mean_probability.device
    solid = _mask_tensor(batch.solid_mask, device=device)
    failed = _mask_tensor(batch.failed_frame_mask, device=device)
    known = _mask_tensor(batch.label_mask, device=device) & ~solid & ~failed
    gate = w2.apply_w2_uncertainty_gate(
        teacher_statistics.mean_probability,
        teacher_statistics.predictive_variance,
        _mask_tensor(batch.unknown_mask, device=device),
        solid,
        failed_frame_mask=failed,
        variance_gate=float(config.variance_gate),
    )
    reliable = known | gate.pseudo_mask
    unknown_excluded = (
        _mask_tensor(batch.unknown_mask, device=device)
        & ~solid
        & ~failed
        & ~gate.pseudo_mask
    )
    exclusions = {
        "unknown_exclusion_count": _mask_count(unknown_excluded),
        "solid_exclusion_count": _mask_count(solid),
        "invalid_exclusion_count": _mask_count(failed),
        "excluded_identity_count": _mask_count(~reliable),
    }
    return reliable, exclusions


def compute_w3_loss(
    student_predictions: Any,
    teacher_predictions: Any,
    projected_views: Any,
    batch: W3Batch,
    *,
    config: W3Config | None = None,
    epoch: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """计算 W2 supervised path 加两视图 trajectory contrastive loss。"""
    if not isinstance(batch, W3Batch):
        raise TypeError("W3 loss 必须消费 W3Batch")
    config = _coerce_w3_config(config)
    epoch = _strict_nonnegative_int(epoch, name="epoch")
    w2_loss, stats = w2.compute_w2_loss(
        student_predictions,
        teacher_predictions,
        batch.as_w2_batch(),
        config=config.as_w2_config(),
        epoch=epoch,
    )
    projected = _coerce_two_views(projected_views)
    if any(int(view.shape[-1]) != config.projection_dim for view in projected):
        raise ValueError("W3 projected embedding 最后一维必须固定为 64")
    reliable, exclusions = _reliable_identity_mask(
        batch, teacher_predictions, config=config)
    contrastive_loss, contrastive_stats = compute_trajectory_contrastive_loss(
        projected,
        reliable,
        temperature=config.temperature,
        max_embeddings=config.max_embeddings,
    )
    ramp = ramp_up_weight(epoch, config.ramp_up_epochs)
    total = w2_loss + ramp * config.contrastive_weight * contrastive_loss
    stats = dict(stats)
    stats.update(contrastive_stats)
    stats.update(exclusions)
    stats.update({
        "loss": float(total.detach().cpu()),
        "total_identity_count": int(batch.labels.numel()),
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
        "contrastive_ramp_weight": ramp,
        "contrastive_weight": config.contrastive_weight,
        "teacher_view_count": config.teacher_view_count,
        "view_count": config.view_count,
        "projection_dim": config.projection_dim,
        "temperature": config.temperature,
        "max_embeddings": config.max_embeddings,
        "single_gpu": True,
        "memory_bank_used": False,
        "cross_gpu_gather": False,
    })
    return total, stats


def _coerce_w3_batch(batch: Any) -> W3Batch:
    """接受 W3 batch，并对前序 W2/W1-H batch 做显式升级。"""
    if isinstance(batch, W3Batch):
        return batch
    if isinstance(batch, w2.W2Batch):
        return build_w3_batch_from_w2(batch)
    if isinstance(batch, W1HBatch):
        return build_w3_batch_from_w1_h(batch)
    raise TypeError("W3 trainer 必须消费 W3Batch、W2Batch 或 W1HBatch")


@contextmanager
def _seeded_view_rng(seed: int, device: torch.device):
    """只在当前 CPU/target CUDA generator 上运行一个可复现 view。"""
    cpu_state = torch.get_rng_state()
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    target_state = None
    target_generator = None
    if device.type == "cuda":
        target_state = torch.cuda.get_rng_state(device)
        target_generator = torch.Generator(device=device).manual_seed(seed)
    try:
        torch.set_rng_state(cpu_generator.get_state())
        if target_generator is not None:
            torch.cuda.set_rng_state(target_generator.get_state(), device)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if target_state is not None:
            torch.cuda.set_rng_state(target_state, device)


def _validate_w3_checkpoint_metrics(
    metrics: Mapping[str, Any],
    *,
    variance_gate: float,
) -> dict[str, Any]:
    """校验 W3 checkpoint 同时含 W2 诊断和 pair/resource 诊断。"""
    if not isinstance(metrics, Mapping):
        raise TypeError("W3 checkpoint metrics 必须是 object")
    result = dict(metrics)
    w2_metrics = dict(result)
    # W3 对外的 view_count 是 contrastive 的 2；W2 validator 只校验
    # supervised teacher 的固定三视图，因此在副本中使用明确的字段名映射。
    w2_metrics["view_count"] = w2.W2_DEFAULT_VIEW_COUNT
    w2._validate_w2_checkpoint_metrics(w2_metrics, variance_gate=variance_gate)
    pair_stats_scope = result.get("pair_stats_scope", W3_PAIR_STATS_BATCH)
    if pair_stats_scope not in (W3_PAIR_STATS_BATCH, W3_PAIR_STATS_EPOCH):
        raise ValueError(
            "W3 checkpoint metrics.pair_stats_scope 必须是 'batch' 或 'epoch'"
        )
    pair_batch_count = result.get("pair_batch_count", 1)
    if (
        isinstance(pair_batch_count, (bool, np.bool_))
        or not isinstance(pair_batch_count, (int, np.integer))
        or int(pair_batch_count) <= 0
    ):
        raise ValueError("W3 checkpoint metrics.pair_batch_count 必须是正整数")
    pair_batch_count = int(pair_batch_count)
    if pair_stats_scope == W3_PAIR_STATS_BATCH and pair_batch_count != 1:
        raise ValueError("W3 batch pair statistics 的 pair_batch_count 必须是 1")
    result["pair_stats_scope"] = pair_stats_scope
    result["pair_batch_count"] = pair_batch_count
    required = (
        "contrastive_loss", "effective_embedding_count",
        "candidate_identity_count", "valid_identity_count",
        "positive_pair_count", "negative_pair_count", "pair_count",
        "unknown_exclusion_count", "solid_exclusion_count",
        "invalid_exclusion_count", "excluded_identity_count",
        "cap_exclusion_count", "contrastive_ramp_weight",
        "teacher_view_count",
    )
    missing = [field for field in required if field not in result]
    if missing:
        raise ValueError(f"W3 checkpoint metrics 缺少 pair/resource 诊断：{missing!r}")
    for field in (
        "effective_embedding_count", "candidate_identity_count",
        "valid_identity_count", "positive_pair_count", "negative_pair_count",
        "pair_count", "unknown_exclusion_count", "solid_exclusion_count",
        "invalid_exclusion_count", "excluded_identity_count", "cap_exclusion_count",
    ):
        value = result[field]
        if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 0:
            raise ValueError(f"W3 checkpoint metrics.{field} 必须是非负整数")
        result[field] = int(value)
    if pair_stats_scope == W3_PAIR_STATS_BATCH:
        if result["effective_embedding_count"] > W3_DEFAULT_MAX_EMBEDDINGS:
            raise ValueError("W3 checkpoint effective_embedding_count 超过 512 cap")
    else:
        max_fields = (
            "max_effective_embedding_count",
            "max_valid_identity_count",
            "max_negative_pair_count",
        )
        missing_max = [field for field in max_fields if field not in result]
        if missing_max:
            raise ValueError(
                "W3 epoch checkpoint metrics 缺少 per-batch cap diagnostics："
                f"{missing_max!r}"
            )
        for field in max_fields:
            value = result[field]
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise ValueError(
                    f"W3 checkpoint metrics.{field} 必须是非负整数"
                )
            result[field] = int(value)
        if result["max_effective_embedding_count"] > W3_DEFAULT_MAX_EMBEDDINGS:
            raise ValueError("W3 per-batch effective embedding count 超过 512 cap")
        if result["max_valid_identity_count"] > W3_DEFAULT_MAX_EMBEDDINGS // 2:
            raise ValueError("W3 per-batch valid identity count 超过 256 cap")
        if result["max_effective_embedding_count"] != (
                2 * result["max_valid_identity_count"]):
            raise ValueError(
                "W3 epoch max embedding/identity counts 不一致"
            )
        expected_max_negative_pairs = (
            2 * result["max_valid_identity_count"]
            * max(0, 2 * result["max_valid_identity_count"] - 2)
        )
        if result["max_negative_pair_count"] != expected_max_negative_pairs:
            raise ValueError(
                "W3 epoch max negative pair count 不符合 two-view contract"
            )
        if result["effective_embedding_count"] > (
                pair_batch_count * W3_DEFAULT_MAX_EMBEDDINGS):
            raise ValueError("W3 epoch aggregate embeddings 超过 per-batch 512 cap")
    if result["effective_embedding_count"] != 2 * result["valid_identity_count"]:
        raise ValueError("W3 checkpoint embedding count 必须等于 2 * valid identity count")
    if result["positive_pair_count"] != result["valid_identity_count"]:
        raise ValueError("W3 checkpoint positive_pair_count 与 valid identity 不一致")
    if result["pair_count"] != result["positive_pair_count"]:
        raise ValueError("W3 checkpoint pair_count 与 positive_pair_count 不一致")
    if result["excluded_identity_count"] != (
            result["unknown_exclusion_count"]
            + result["solid_exclusion_count"]
            + result["invalid_exclusion_count"]):
        raise ValueError(
            "W3 checkpoint excluded identity 必须等于 unknown/solid/invalid exclusions"
        )
    if result["valid_identity_count"] + result["cap_exclusion_count"] != (
            result["candidate_identity_count"]
            - result["unknown_exclusion_count"]
            - result["solid_exclusion_count"]
            - result["invalid_exclusion_count"]):
        raise ValueError(
            "W3 checkpoint candidate/valid/cap identity counts 不一致"
        )
    expected_negative_pairs = (
        2 * result["valid_identity_count"]
        * max(0, 2 * result["valid_identity_count"] - 2)
    )
    if pair_stats_scope == W3_PAIR_STATS_BATCH:
        if result["negative_pair_count"] != expected_negative_pairs:
            raise ValueError(
                "W3 checkpoint negative_pair_count 不符合 two-view in-batch contract"
            )
    else:
        max_negative_pairs = (
            2 * result["max_valid_identity_count"]
            * max(0, 2 * result["max_valid_identity_count"] - 2)
        )
        if result["negative_pair_count"] > pair_batch_count * max_negative_pairs:
            raise ValueError(
                "W3 epoch negative_pair_count 超过每 batch in-batch 上限"
            )
    if result["unknown_exclusion_count"] > result["candidate_identity_count"]:
        raise ValueError("W3 checkpoint unknown exclusion 超过 candidate identity 数")
    if result["cap_exclusion_count"] > result["candidate_identity_count"]:
        raise ValueError("W3 checkpoint cap exclusion 超过 candidate identity 数")
    numeric = ("contrastive_loss", "contrastive_ramp_weight")
    for field in numeric:
        try:
            value = float(result[field])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"W3 checkpoint metrics.{field} 必须是有限数") from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"W3 checkpoint metrics.{field} 必须是非负有限数")
        result[field] = value
    for field, expected in (
        ("view_count", W3_DEFAULT_VIEW_COUNT),
        ("teacher_view_count", w2.W2_DEFAULT_VIEW_COUNT),
        ("projection_dim", 64),
        ("max_embeddings", W3_DEFAULT_MAX_EMBEDDINGS),
    ):
        value = result.get(field)
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) != value):
            raise ValueError(f"W3 checkpoint metrics.{field} 必须是整数 {expected}")
        if result.get(field) != expected:
            raise ValueError(f"W3 checkpoint metrics.{field} 必须固定为 {expected}")
    if result.get("temperature") != W3_DEFAULT_TEMPERATURE:
        raise ValueError("W3 checkpoint metrics.temperature 必须固定为 0.1")
    for field in ("memory_bank_used", "cross_gpu_gather", "single_gpu"):
        if result.get(field) is not False and field != "single_gpu":
            raise ValueError(f"W3 checkpoint metrics.{field} 必须是 False")
    if result.get("single_gpu") is not True:
        raise ValueError("W3 checkpoint metrics.single_gpu 必须是 True")
    result["view_count"] = W3_DEFAULT_VIEW_COUNT
    result["projection_dim"] = 64
    result["temperature"] = W3_DEFAULT_TEMPERATURE
    result["max_embeddings"] = W3_DEFAULT_MAX_EMBEDDINGS
    return result


class W3Trainer:
    """W3 可恢复 trainer：W2 三视图监督加 student 两视图 contrastive。"""

    def __init__(
        self,
        student: nn.Module,
        optimizer: Any,
        *,
        projection_head: TrajectoryProjectionHead,
        sampling_source: str,
        anchor_hash: str,
        config: W3Config | None = None,
        teacher: nn.Module | None = None,
        scheduler: Any | None = None,
        seed: int = 0,
        anchor_metadata: Mapping[str, Any] | None = None,
        calibration_selection: Any | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        self.config = _coerce_w3_config(config)
        if not isinstance(student, TrajectoryEmbeddingAdapter):
            student = TrajectoryEmbeddingAdapter(student)
        if not isinstance(student, TrajectoryEmbeddingAdapter):
            raise TypeError("W3 student 必须是 TrajectoryEmbeddingAdapter")
        if teacher is not None and not isinstance(teacher, TrajectoryEmbeddingAdapter):
            teacher = TrajectoryEmbeddingAdapter(teacher)
        student_backbone = student.channel_adapter.model
        teacher_backbone = (
            None if teacher is None else teacher.channel_adapter.model
        )
        if teacher is not None and student_backbone is teacher_backbone:
            raise ValueError(
                "W3 student 与 teacher 不能共享同一底层 vendor model"
            )
        if teacher is not None:
            student_parameter_ids = {id(parameter) for parameter in student.parameters()}
            teacher_parameter_ids = {id(parameter) for parameter in teacher.parameters()}
            if student_parameter_ids & teacher_parameter_ids:
                raise ValueError(
                    "W3 student 与 teacher 不能共享参数，即使底层 vendor 对象不同"
                )
        if not isinstance(projection_head, TrajectoryProjectionHead):
            raise TypeError("W3 projection_head 必须是 TrajectoryProjectionHead")
        if projection_head.input_dim != student.embedding_dim:
            raise ValueError(
                "W3 projection_head input_dim 必须与 student embedding_dim 一致"
            )
        if optimizer is None or not all(
            hasattr(optimizer, attr)
            for attr in ("zero_grad", "step", "state_dict", "param_groups")
        ):
            raise TypeError(
                "W3 optimizer 必须提供 zero_grad()/step()/state_dict()/param_groups()"
            )
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group.get("params", ())
        }
        trainable_parameters = (
            list(student.parameters()) + list(projection_head.parameters())
        )
        if any(id(parameter) not in optimizer_parameter_ids
               for parameter in trainable_parameters):
            raise ValueError(
                "W3 optimizer 必须包含 student 与 projection_head 参数"
            )
        if not isinstance(anchor_hash, str) or not anchor_hash.strip():
            raise ValueError("W3 anchor_hash 必须是非空字符串")
        sampling_source = contract.validate_label_source(sampling_source)
        if sampling_source not in W3_ALLOWED_SAMPLING_SOURCES:
            raise ValueError(
                f"W3 sampling_source 必须是 {sorted(W3_ALLOWED_SAMPLING_SOURCES)!r}"
            )
        if teacher is not None and teacher is student:
            raise ValueError("W3 teacher 不能与 student 共享同一 module")
        self.student = student
        self.teacher = clone_ema_teacher(student) if teacher is None else teacher
        _prepare_ema_teacher(self.teacher)
        if tuple(self.student.state_dict()) != tuple(self.teacher.state_dict()):
            raise ValueError("W3 student/teacher state_dict keys 不一致")
        self.projection_head = projection_head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sampling_source = sampling_source
        self.anchor_hash = anchor_hash.strip()
        self.anchor_metadata = (
            None if anchor_metadata is None
            else w2._validate_checkpoint_anchor_metadata(anchor_metadata)
        )
        self.calibration_selection = None
        if calibration_selection is not None:
            selected_gate = _calibration_policy_gate(calibration_selection)
            self.calibration_selection = w2._policy_as_dict(
                calibration_selection,
                variance_gate=selected_gate,
            )
        self.seed = _strict_nonnegative_int(seed, name="seed")
        if grad_clip_norm is not None:
            grad_clip_norm = float(grad_clip_norm)
            if not np.isfinite(grad_clip_norm) or grad_clip_norm <= 0.0:
                raise ValueError("W3 grad_clip_norm 必须是正的有限数或 None")
        self.grad_clip_norm = grad_clip_norm
        self.global_step = 0
        self.last_metrics: dict[str, Any] | None = None

    def _move_models(self, device: str | torch.device) -> None:
        self.student.to(device)
        self.teacher.to(device)
        self.projection_head.to(device)
        _prepare_ema_teacher(self.teacher)

    def _validate_batch_identity(self, batch: W3Batch) -> None:
        if batch.sampling_source != self.sampling_source:
            raise ValueError(
                f"W3 sampling_source 不匹配：trainer={self.sampling_source!r} "
                f"batch={batch.sampling_source!r}"
            )
        if batch.anchor_hash != self.anchor_hash:
            raise ValueError(
                f"W3 anchor_hash 不匹配：trainer={self.anchor_hash!r} "
                f"batch={batch.anchor_hash!r}"
            )
        if self.anchor_metadata is None:
            self.anchor_metadata = None if batch.anchor_metadata is None else dict(
                batch.anchor_metadata)
        elif batch.anchor_metadata is not None:
            current = w2._anchor_metadata_identity(self.anchor_metadata)
            incoming = w2._anchor_metadata_identity(batch.anchor_metadata)
            if current != incoming:
                raise ValueError("W3 batch Haller anchor metadata 与 trainer 不一致")

    def predict_contrastive_views(
        self,
        batch: Any,
        *,
        device: str | torch.device = "cpu",
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """为每条 trajectory 生成恰好两次、可复现且不同 RNG 的 student view。"""
        batch = _coerce_w3_batch(batch)
        self._validate_batch_identity(batch)
        self._move_models(device)
        model_batch = batch.to(device)
        self.student.train()
        expected_shape = tuple(int(value) for value in model_batch.labels.shape)
        views: list[tuple[torch.Tensor, torch.Tensor]] = []
        for view_index in range(W3_DEFAULT_VIEW_COUNT):
            seed = int(
                (self.seed
                 + 1_000_003 * (self.global_step + 1)
                 + 20_011 * (view_index + 1))
                % (2**63 - 1)
            )
            with _seeded_view_rng(seed, model_batch.pathlines.device):
                probability, embedding = self.student.forward_with_embedding(
                    model_batch, consumer="train")
            probability = _as_prediction_tensor(
                probability, expected_shape, f"student_view[{view_index}]"
            )
            views.append((probability, embedding))
        return views

    # Alias makes the semantic distinction from W2 teacher views explicit while
    # allowing callers that call the stream "student views" to use the same seam.
    predict_student_views = predict_contrastive_views

    def predict_teacher_views(
        self,
        batch: Any,
        *,
        device: str | torch.device = "cpu",
    ) -> list[torch.Tensor]:
        """复用 W2 语义，为监督路径生成固定三次 teacher view。"""
        batch = _coerce_w3_batch(batch)
        self._validate_batch_identity(batch)
        self._move_models(device)
        model_batch = batch.to(device)
        self.teacher.eval()
        expected_shape = tuple(int(value) for value in model_batch.labels.shape)
        views: list[torch.Tensor] = []
        for view_index in range(w2.W2_DEFAULT_VIEW_COUNT):
            seed = int(
                (self.seed
                 + 1_000_003 * (self.global_step + 1)
                 + 30_017 * (view_index + 1))
                % (2**63 - 1)
            )
            with _seeded_view_rng(seed, model_batch.pathlines.device):
                with torch.no_grad():
                    probability = self.teacher(model_batch, consumer="train")
            views.append(_as_prediction_tensor(
                probability, expected_shape, f"teacher_view[{view_index}]"
            ).detach())
        return views

    def train_step(
        self,
        batch: Any,
        *,
        epoch: int,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """执行 student two-view embedding、W2 teacher loss 与 EMA 更新。"""
        batch = _coerce_w3_batch(batch)
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        self._validate_batch_identity(batch)
        self._move_models(device)
        model_batch = batch.to(device)
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        student_views = self.predict_contrastive_views(model_batch, device=device)
        teacher_views = self.predict_teacher_views(model_batch, device=device)
        student_predictions = student_views[0][0]
        projected_views = [
            self.projection_head(embedding) for _probability, embedding in student_views
        ]
        loss, stats = compute_w3_loss(
            student_predictions,
            teacher_views,
            projected_views,
            model_batch,
            config=self.config,
            epoch=epoch,
        )
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.projection_head.parameters()),
                self.grad_clip_norm,
            )
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
        """运行非空 W3 epoch，并聚合 pair/resource diagnostics。"""
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        if max_steps is not None:
            max_steps = _strict_nonnegative_int(max_steps, name="max_steps")
            if max_steps <= 0:
                raise ValueError("W3 max_steps 必须是正整数或 None")
        logs = []
        for batch in batches:
            if max_steps is not None and len(logs) >= max_steps:
                break
            logs.append(self.train_step(batch, epoch=epoch, device=device))
        if not logs:
            raise ValueError("W3 batches 为空：训练循环无样本可跑")
        count_keys = {
            "candidate_count", "pseudo_eligible_count", "pseudo_accepted_count",
            "pseudo_positive_count", "pseudo_negative_count", "unknown_count",
            "anchor_count", "anchor_positive_count", "anchor_negative_count",
            "unknown_cell_count", "solid_count", "failed_cell_count",
            "artifact_failure_count", "candidate_identity_count",
            "valid_identity_count", "positive_pair_count", "negative_pair_count",
            "pair_count", "effective_embedding_count", "unknown_exclusion_count",
            "solid_exclusion_count", "invalid_exclusion_count",
            "excluded_identity_count", "cap_exclusion_count",
            "total_identity_count",
        }
        fixed_keys = {
            "view_count", "teacher_view_count", "projection_dim", "temperature",
            "max_embeddings", "memory_bank_used", "cross_gpu_gather", "single_gpu",
            "variance_gate",
            "sampling_source", "loss_source", "anchor_hash",
            "anchor_algorithm_version", "anchor_parameter_hash", "anchor_input_hash",
            "anchor_mask_hash",
        }
        ratio_weights = {
            "pseudo_acceptance": "candidate_count",
            "gate_acceptance": "candidate_count",
            "pseudo_positive_ratio": "pseudo_accepted_count",
            "pseudo_negative_ratio": "pseudo_accepted_count",
            "teacher_student_disagreement": "candidate_count",
            "accepted_teacher_student_disagreement": "pseudo_accepted_count",
        }
        distribution_prefixes = (
            "mean_probability", "predictive_variance", "entropy"
        )
        total_weight = sum(
            float(log["total_identity_count"]) for log in logs
        )
        if total_weight <= 0.0:
            raise ValueError("W3 epoch total_identity_count 必须为正")
        summary: dict[str, Any] = {}
        for key, value in logs[0].items():
            if key in ("epoch", "global_step"):
                continue
            if key in count_keys:
                summary[key] = int(sum(int(log[key]) for log in logs))
            elif key in fixed_keys:
                summary[key] = value
            elif key in ratio_weights:
                denominator = sum(
                    float(log[ratio_weights[key]]) for log in logs
                )
                summary[key] = (
                    sum(float(log[key]) * float(log[ratio_weights[key]])
                        for log in logs) / denominator
                    if denominator > 0.0 else 0.0
                )
            elif any(key.startswith(prefix + "_")
                     for prefix in distribution_prefixes):
                # Reconstruct pooled population moments below instead of
                # taking an unweighted mean of per-batch extrema/stds.
                continue
            elif isinstance(value, (int, float, np.integer, np.floating)):
                summary[key] = sum(
                    float(log[key]) * float(log["total_identity_count"])
                    for log in logs
                ) / total_weight
        for prefix in distribution_prefixes:
            mean_key = f"{prefix}_mean"
            std_key = f"{prefix}_std"
            min_key = f"{prefix}_min"
            max_key = f"{prefix}_max"
            pooled_mean = sum(
                float(log[mean_key]) * float(log["total_identity_count"])
                for log in logs
            ) / total_weight
            pooled_variance = sum(
                float(log["total_identity_count"])
                * (float(log[std_key]) ** 2
                   + (float(log[mean_key]) - pooled_mean) ** 2)
                for log in logs
            ) / total_weight
            summary[mean_key] = pooled_mean
            summary[std_key] = float(np.sqrt(max(0.0, pooled_variance)))
            summary[min_key] = min(float(log[min_key]) for log in logs)
            summary[max_key] = max(float(log[max_key]) for log in logs)
        sources = {log["sampling_source"] for log in logs}
        loss_sources = {log["loss_source"] for log in logs}
        anchor_hashes = {log["anchor_hash"] for log in logs}
        fixed_values = {
            key: logs[0][key]
            for key in fixed_keys
            if key in logs[0]
        }
        fixed_drift = any(
            any(log.get(key) != value for log in logs[1:])
            for key, value in fixed_values.items()
        )
        if (len(sources) != 1 or len(loss_sources) != 1
                or len(anchor_hashes) != 1 or fixed_drift):
            raise ValueError("W3 一个 epoch 内 source/loss/anchor hash 发生漂移")
        summary.update({
            "sampling_source": next(iter(sources)),
            "loss_source": next(iter(loss_sources)),
            "anchor_hash": next(iter(anchor_hashes)),
            "view_count": W3_DEFAULT_VIEW_COUNT,
            "teacher_view_count": w2.W2_DEFAULT_VIEW_COUNT,
            "projection_dim": self.config.projection_dim,
            "temperature": self.config.temperature,
            "max_embeddings": self.config.max_embeddings,
            "memory_bank_used": False,
            "cross_gpu_gather": False,
            "single_gpu": True,
            "epoch": epoch,
            "steps": len(logs),
            "global_step": self.global_step,
            # The count fields above are epoch aggregates.  Keep explicit
            # per-batch maxima so checkpoint validation can still enforce the
            # 512-embedding resource cap without pretending batches share one
            # in-batch negative pool.
            "pair_stats_scope": W3_PAIR_STATS_EPOCH,
            "pair_batch_count": len(logs),
            "max_effective_embedding_count": max(
                int(log["effective_embedding_count"]) for log in logs
            ),
            "max_valid_identity_count": max(
                int(log["valid_identity_count"]) for log in logs
            ),
            "max_negative_pair_count": max(
                int(log["negative_pair_count"]) for log in logs
            ),
        })
        self.last_metrics = copy.deepcopy(summary)
        return summary

    def _checkpoint_extra_metadata(
        self,
        extra_metadata: Mapping[str, Any] | None,
        *,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        """构造包含 W3 resource/pair contract 且不含 test GT 的 metadata。"""
        if extra_metadata is not None and not isinstance(extra_metadata, Mapping):
            raise TypeError("W3 extra_metadata 必须是 object")
        extra = dict(extra_metadata or {})
        w2._reject_test_source(extra, context="W3 checkpoint extra_metadata")
        reserved = {
            "generation_version": W3_GENERATION_VERSION,
            "formal_loss_source": W3_LABEL_SOURCE,
            "w3_config": self.config.as_dict(),
            "w2_config": self.config.as_w2_config().as_dict(),
            "trajectory_contrastive": {
                "view_count": W3_DEFAULT_VIEW_COUNT,
                "teacher_view_count": w2.W2_DEFAULT_VIEW_COUNT,
                "projection_dim": 64,
                "temperature": W3_DEFAULT_TEMPERATURE,
                "max_embeddings": W3_DEFAULT_MAX_EMBEDDINGS,
                "single_gpu": True,
                "memory_bank": False,
                "cross_gpu_gather": False,
            },
            "uncertainty_gate": {
                "view_count": w2.W2_DEFAULT_VIEW_COUNT,
                "teacher_view_count": w2.W2_DEFAULT_VIEW_COUNT,
                "variance_gate": float(self.config.variance_gate),
                "positive_confidence": w2.W2_DEFAULT_PSEUDO_HIGH,
                "negative_confidence": w2.W2_DEFAULT_PSEUDO_LOW,
                "variance_is_primary": True,
                "entropy_diagnostic_only": True,
            },
            "w3_metrics": copy.deepcopy(dict(metrics)),
        }
        if self.anchor_metadata is not None:
            reserved["haller_anchor"] = copy.deepcopy(dict(self.anchor_metadata))
        for key, expected in reserved.items():
            if key in extra and extra[key] != expected:
                raise ValueError(f"W3 checkpoint extra_metadata.{key} 与 trainer 语义不一致")
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
        calibration_policy: Any | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """保存 student/teacher/projection、pair stats 和 global gate。"""
        if self.scheduler is None:
            raise ValueError("W3 resume checkpoint 必须提供 scheduler")
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        self.anchor_metadata = w2._validate_checkpoint_anchor_metadata(
            self.anchor_metadata)
        checkpoint_metrics = self.last_metrics if metrics is None else metrics
        if checkpoint_metrics is None:
            raise ValueError("W3 checkpoint 必须显式提供 W2/pair diagnostics")
        checkpoint_metrics = _validate_w3_checkpoint_metrics(
            checkpoint_metrics,
            variance_gate=float(self.config.variance_gate),
        )
        selected_policy = (
            self.calibration_selection
            if calibration_policy is None
            else calibration_policy
        )
        policy = w2._policy_as_dict(
            selected_policy,
            variance_gate=_calibration_policy_gate(selected_policy),
        )
        destination = pathlib.Path(path)
        temporary_path = destination.with_name(
            f".{destination.name}.w3-tmp-{uuid.uuid4().hex}"
        )
        try:
            saved_path = pathlib.Path(contract.save_checkpoint(
                temporary_path,
                self.student,
                self.optimizer,
                self.scheduler,
                mode=contract.MODE_W3,
                feature_schema=contract.FEATURE_SCHEMA_7,
                adapter_input_schema=contract.FEATURE_SCHEMA_7,
                dataset_config=dataset_config,
                split_config=split_config,
                sampling_config=sampling_config,
                label_source=W3_LABEL_SOURCE,
                sampling_source=self.sampling_source,
                teacher=self.teacher,
                projection_head=self.projection_head,
                epoch=epoch,
                global_step=self.global_step,
                metrics=checkpoint_metrics,
                seed=self.seed,
                anchor_hash=self.anchor_hash,
                calibration_policy=policy,
                extra_metadata=self._checkpoint_extra_metadata(
                    extra_metadata, metrics=checkpoint_metrics
                ),
            ))
            with saved_path.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(saved_path, destination)
            return destination
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _validate_checkpoint_semantics(
        self,
        result: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """在任何 target state load 前校验 W3 专用 metadata。"""
        if not isinstance(result, Mapping):
            raise ValueError("W3 checkpoint 顶层必须是 object")
        extra = result.get("extra_metadata")
        if not isinstance(extra, Mapping):
            raise ValueError("W3 checkpoint 缺少 extra_metadata")
        if extra.get("generation_version") != W3_GENERATION_VERSION:
            raise ValueError("W3 checkpoint generation_version 不匹配")
        if extra.get("formal_loss_source") != W3_LABEL_SOURCE:
            raise ValueError("W3 checkpoint formal_loss_source 不匹配")
        loaded_anchor_metadata = w2._validate_checkpoint_anchor_metadata(
            extra.get("haller_anchor"))
        if self.anchor_metadata is not None:
            current = w2._validate_checkpoint_anchor_metadata(self.anchor_metadata)
            if w2._anchor_metadata_identity(current) != w2._anchor_metadata_identity(
                    loaded_anchor_metadata):
                raise ValueError("W3 checkpoint Haller anchor metadata 与 trainer 不一致")
        loaded_policy = result.get("calibration_policy")
        policy = w2._policy_as_dict(
            loaded_policy,
            variance_gate=_calibration_policy_gate(loaded_policy),
        )
        loaded_metrics = _validate_w3_checkpoint_metrics(
            result.get("metrics"),
            variance_gate=float(self.config.variance_gate),
        )
        if extra.get("w3_config") != self.config.as_dict():
            raise ValueError("W3 checkpoint w3_config 与当前冻结 trainer config 不一致")
        if extra.get("w2_config") != self.config.as_w2_config().as_dict():
            raise ValueError("W3 checkpoint w2_config 与当前冻结 trainer config 不一致")
        expected_contrastive = {
            "view_count": W3_DEFAULT_VIEW_COUNT,
            "teacher_view_count": w2.W2_DEFAULT_VIEW_COUNT,
            "projection_dim": 64,
            "temperature": W3_DEFAULT_TEMPERATURE,
            "max_embeddings": W3_DEFAULT_MAX_EMBEDDINGS,
            "single_gpu": True,
            "memory_bank": False,
            "cross_gpu_gather": False,
        }
        if extra.get("trajectory_contrastive") != expected_contrastive:
            raise ValueError("W3 checkpoint trajectory_contrastive contract 不一致")
        expected_uncertainty_gate = {
            "view_count": w2.W2_DEFAULT_VIEW_COUNT,
            "teacher_view_count": w2.W2_DEFAULT_VIEW_COUNT,
            "variance_gate": float(self.config.variance_gate),
            "positive_confidence": w2.W2_DEFAULT_PSEUDO_HIGH,
            "negative_confidence": w2.W2_DEFAULT_PSEUDO_LOW,
            "variance_is_primary": True,
            "entropy_diagnostic_only": True,
        }
        if extra.get("uncertainty_gate") != expected_uncertainty_gate:
            raise ValueError("W3 checkpoint uncertainty_gate contract 不一致")
        if _validate_w3_checkpoint_metrics(
                extra.get("w3_metrics"),
                variance_gate=float(self.config.variance_gate)) != loaded_metrics:
            raise ValueError("W3 checkpoint w3_metrics 与顶层 metrics 不一致")
        return extra, loaded_anchor_metadata, policy, loaded_metrics

    @staticmethod
    def _checkpoint_module_device(module: nn.Module) -> torch.device | None:
        """取得单设备 module 的原位置，供失败回滚使用。"""
        tensors = [*module.parameters(), *module.buffers()]
        if not tensors:
            return None
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1:
            raise ValueError(
                "W3 checkpoint transaction 要求 student/teacher/head 各自位于单一 device"
            )
        return tensors[0].device

    @classmethod
    def _snapshot_checkpoint_module(cls, module: nn.Module) -> dict[str, Any]:
        """捕获 module state、device、training 和 requires_grad 语义。"""
        return {
            "state": copy.deepcopy(module.state_dict()),
            "device": cls._checkpoint_module_device(module),
            "training": {
                name: bool(child.training)
                for name, child in module.named_modules()
            },
            "requires_grad": {
                name: bool(parameter.requires_grad)
                for name, parameter in module.named_parameters()
            },
        }

    @staticmethod
    def _module_subobject(module: nn.Module, name: str) -> nn.Module:
        return module if name == "" else module.get_submodule(name)

    @classmethod
    def _restore_checkpoint_module(
        cls,
        module: nn.Module,
        snapshot: Mapping[str, Any],
        *,
        name: str,
    ) -> None:
        """恢复一个 module 的全部可变训练状态。"""
        device = snapshot["device"]
        if device is not None:
            module.to(device)
        module.load_state_dict(snapshot["state"], strict=True)
        for parameter_name, requires_grad in snapshot["requires_grad"].items():
            module.get_parameter(parameter_name).requires_grad_(requires_grad)
        for module_name, training in snapshot["training"].items():
            cls._module_subobject(module, module_name).train(training)

    def _snapshot_checkpoint_state(self) -> dict[str, Any]:
        """捕获 load_checkpoint 可能触碰的所有本地状态。"""
        return {
            "student": self._snapshot_checkpoint_module(self.student),
            "teacher": self._snapshot_checkpoint_module(self.teacher),
            "projection_head": self._snapshot_checkpoint_module(
                self.projection_head
            ),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": (
                None if self.scheduler is None
                else copy.deepcopy(self.scheduler.state_dict())
            ),
            "rng": contract.capture_rng_state(),
            "anchor_metadata": copy.deepcopy(self.anchor_metadata),
            "calibration_selection": copy.deepcopy(self.calibration_selection),
            "last_metrics": copy.deepcopy(self.last_metrics),
            "global_step": self.global_step,
            "seed": self.seed,
        }

    def _restore_checkpoint_state(self, snapshot: Mapping[str, Any]) -> None:
        """在 checkpoint 任一阶段失败时原子恢复 trainer。"""
        self._restore_checkpoint_module(
            self.student, snapshot["student"], name="student"
        )
        self._restore_checkpoint_module(
            self.teacher, snapshot["teacher"], name="teacher"
        )
        self._restore_checkpoint_module(
            self.projection_head,
            snapshot["projection_head"],
            name="projection_head",
        )
        self.optimizer.load_state_dict(snapshot["optimizer"])
        if self.scheduler is not None and snapshot["scheduler"] is not None:
            self.scheduler.load_state_dict(snapshot["scheduler"])
        contract.restore_rng_state(snapshot["rng"], strict_cuda=False)
        self.anchor_metadata = copy.deepcopy(snapshot["anchor_metadata"])
        self.calibration_selection = copy.deepcopy(
            snapshot["calibration_selection"]
        )
        self.last_metrics = copy.deepcopy(snapshot["last_metrics"])
        self.global_step = snapshot["global_step"]
        self.seed = snapshot["seed"]

    def _load_checkpoint_impl(
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
        """严格恢复 W3 mode、projection head、pair contract 与 RNG。"""
        # The shared loader validates the top-level contract before loading
        # component states.  Preflight the W3-only metadata as well, so a
        # resource/source/config drift cannot partially mutate this trainer
        # before the specialized guard rejects it.
        preflight = contract._torch_load(path, "cpu")
        self._validate_checkpoint_semantics(preflight)
        self._move_models(device)
        result = contract.load_checkpoint(
            path,
            self.student,
            self.optimizer,
            self.scheduler,
            teacher=self.teacher,
            projection_head=self.projection_head,
            device=device,
            expected_mode=contract.MODE_W3,
            expected_feature_schema=contract.FEATURE_SCHEMA_7,
            expected_dataset_config=expected_dataset_config,
            expected_split_config=expected_split_config,
            expected_sampling_config=expected_sampling_config,
            expected_label_source=W3_LABEL_SOURCE,
            expected_sampling_source=self.sampling_source,
            expected_anchor_hash=self.anchor_hash,
            restore_rng=restore_rng,
            strict_cuda_rng=strict_cuda_rng,
            load_mode=load_mode,
        )
        extra, loaded_anchor_metadata, policy, loaded_metrics = (
            self._validate_checkpoint_semantics(result)
        )
        self.anchor_metadata = loaded_anchor_metadata
        self.calibration_selection = copy.deepcopy(policy)
        self.last_metrics = copy.deepcopy(loaded_metrics)
        self.global_step = _strict_nonnegative_int(
            result["global_step"], name="checkpoint global_step")
        self.seed = _strict_nonnegative_int(result["seed"], name="checkpoint seed")
        _prepare_ema_teacher(self.teacher)
        result = dict(result)
        result.update({
            "view_count": W3_DEFAULT_VIEW_COUNT,
            "teacher_view_count": w2.W2_DEFAULT_VIEW_COUNT,
            "projection_dim": 64,
            "temperature": W3_DEFAULT_TEMPERATURE,
            "max_embeddings": W3_DEFAULT_MAX_EMBEDDINGS,
            "single_gpu": True,
            "memory_bank_used": False,
            "cross_gpu_gather": False,
            "calibration_policy": policy,
        })
        return result

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
        """原子恢复 W3 checkpoint；任一校验/加载失败都回滚本地状态。"""
        snapshot = self._snapshot_checkpoint_state()
        try:
            return self._load_checkpoint_impl(
                path,
                expected_dataset_config=expected_dataset_config,
                expected_split_config=expected_split_config,
                expected_sampling_config=expected_sampling_config,
                device=device,
                load_mode=load_mode,
                restore_rng=restore_rng,
                strict_cuda_rng=strict_cuda_rng,
            )
        except Exception:
            try:
                self._restore_checkpoint_state(snapshot)
            except Exception as rollback_error:
                raise RuntimeError(
                    "W3 checkpoint load 失败，且本地状态回滚失败"
                ) from rollback_error
            raise


__all__ = [
    "TrajectoryEmbeddingAdapter",
    "TrajectoryProjectionHead",
    "TrajectoryPairSelection",
    "W3_GENERATION_VERSION",
    "W3_LABEL_SOURCE",
    "W3Config",
    "W3Batch",
    "W3Trainer",
    "W3_DEFAULT_VIEW_COUNT",
    "W3_DEFAULT_MAX_EMBEDDINGS",
    "W3_DEFAULT_TEMPERATURE",
    "W3_PAIR_STATS_BATCH",
    "W3_PAIR_STATS_EPOCH",
    "build_trajectory_pairs",
    "compute_trajectory_contrastive_loss",
    "build_w3_batch_from_w2",
    "build_w3_batch_from_w1_h",
    "compute_w3_loss",
]
