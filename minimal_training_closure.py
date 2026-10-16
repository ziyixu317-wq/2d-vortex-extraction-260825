"""Ticket 06: a source-aware minimal training closure.

This module is intentionally smaller than the historical pilot runners.  It
only wires the current sparse-anchor protocol into three closure methods:
W1-P, W1-Hybrid, and W2-on-Hybrid.  The public seams are:

* :func:`fuse_w1_hybrid_targets` for the positive-only Haller/percentile
  fusion contract;
* :func:`evaluate_w2_selective` for abstention-aware calibration evaluation;
* :func:`run_minimal_training_closure` for a real CPU train/save/resume/
  calibration tracer-bullet.

The closure never opens a final-test input.  The qualified bundle loader is
called before synthetic batches are built, and its identity is carried into
every hybrid checkpoint/report.  This is a semantic closure artifact, not a
performance study.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sparse_anchor_study as sparse
import weak_supervision_contract as contract
import w1_p
import w2


TICKET06_SCHEMA_VERSION = "criterion-ambiguity-vortex-learning-ticket06-closure-v1"
CLOSURE_ARTIFACT_TYPE = "minimal_training_closure"
CLOSURE_INTEGRITY_TYPE = "minimal_training_closure_integrity"

PERCENTILE_POSITIVE = 1
PERCENTILE_NEGATIVE = 0
PERCENTILE_UNKNOWN = -1

W1_P_METHOD = contract.MODE_W1_P
W1_HYBRID_METHOD = contract.MODE_W1_HYBRID
W2_ON_HYBRID_METHOD = contract.MODE_W2_ON_HYBRID
CLOSURE_METHODS = (W1_P_METHOD, W1_HYBRID_METHOD, W2_ON_HYBRID_METHOD)

_REGISTERED_SPARSE_STATES = frozenset(sparse.SPARSE_STATES)
_EXCLUDED_SPARSE_STATES = frozenset({
    sparse.STATE_SOLID,
    sparse.STATE_INVALID,
    sparse.STATE_EXTRACTOR_FAILURE,
})
_HARD_COMPONENTS = (
    "percentile_positive",
    "percentile_negative",
    "haller_anchor",
    "agreement",
    "conflict",
)


def _jsonable(value: Any) -> Any:
    """Convert the small metadata objects used by this module to JSON."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(child) for child in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _jsonable(value), ensure_ascii=False, indent=2,
            sort_keys=True, allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是非负整数")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是非负整数") from exc
    if result != value or result < 0:
        raise ValueError(f"{name} 必须是非负整数")
    return result


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是 [0,1] 内的有限数")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是 [0,1] 内的有限数") from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} 必须是 [0,1] 内的有限数")
    return result


def _as_numpy_state(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    raw = np.asarray(value)
    if raw.ndim != 2:
        raise ValueError(f"{name} 必须是 (batch,trajectory) 二维数组")
    if raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError(f"{name} 不能为空")
    if raw.dtype.kind in "fc":
        if not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all():
            raise ValueError(f"{name} 必须只包含整数 state 编码")
    try:
        array = raw.astype(np.int8, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是 percentile state 数组") from exc
    if not np.isin(array, [
        PERCENTILE_POSITIVE, PERCENTILE_NEGATIVE, PERCENTILE_UNKNOWN,
    ]).all():
        raise ValueError(
            f"{name} 只允许 positive=1、negative=0、unknown=-1"
        )
    return array


def _as_numpy_sparse_state(value: Any, *, shape: tuple[int, int]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if tuple(array.shape) != shape:
        raise ValueError(
            f"sparse_state shape={tuple(array.shape)} 与 expected={shape} 不一致"
        )
    try:
        array = array.astype("<U32", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sparse_state 必须是 registered state 数组") from exc
    if not np.isin(array, list(_REGISTERED_SPARSE_STATES)).all():
        raise ValueError(
            "sparse_state 只允许 positive_anchor/uncovered/invalid/"
            "extractor_failure/solid；不允许 Haller negative"
        )
    return array


def _mask_count(mask: np.ndarray | torch.Tensor) -> int:
    if isinstance(mask, torch.Tensor):
        return int(mask.detach().sum().item())
    return int(np.count_nonzero(mask))


def _mask_hash(mask: Any) -> str:
    array = np.ascontiguousarray(
        mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    )
    return _json_hash({
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": array.tobytes().hex(),
    })


@dataclass(frozen=True)
class HybridFusionResult:
    """The complete W1-Hybrid fusion result and its non-overlapping audit masks."""

    labels: np.ndarray
    label_mask: np.ndarray
    unknown_mask: np.ndarray
    weights: np.ndarray
    percentile_state: np.ndarray
    sparse_state: np.ndarray
    category_masks: Mapping[str, np.ndarray]
    category_counts: Mapping[str, int]
    category_stats: Mapping[str, Mapping[str, float | int]]
    haller_negative_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "category_counts": dict(self.category_counts),
            "category_stats": {
                key: dict(value) for key, value in self.category_stats.items()
            },
            "haller_negative_count": int(self.haller_negative_count),
            "weight_values": sorted({float(value) for value in self.weights.reshape(-1)}),
            "uncovered_is_not_negative": True,
            "haller_hard_negative": None,
        }


def _empty_component_stats(mask: np.ndarray, weights: np.ndarray) -> dict[str, float | int]:
    count = _mask_count(mask)
    accepted = _mask_count(mask & (weights > 0.0))
    return {
        "eligible_count": count,
        "accepted_count": accepted,
        "mean_weight": float(weights[mask].mean()) if count else 0.0,
        "contribution": 0.0,
    }


def fuse_w1_hybrid_targets(
    percentile_state: Any,
    sparse_state: Any,
) -> HybridFusionResult:
    """Apply the frozen W1-Hybrid table, including conflict abstention.

    Haller has one positive anchor state and four non-positive states.  In
    particular, ``uncovered`` never becomes a negative target.  Invalid,
    solid, and extractor-failure cells remain explicit ignored states and are
    not ordinary unknown cells eligible for pseudo labels.
    """
    percentile = _as_numpy_state(percentile_state, name="percentile_state")
    sparse_values = _as_numpy_sparse_state(
        sparse_state, shape=tuple(int(value) for value in percentile.shape)
    )
    positive_anchor = sparse_values == sparse.STATE_POSITIVE_ANCHOR
    solid = sparse_values == sparse.STATE_SOLID
    invalid = sparse_values == sparse.STATE_INVALID
    extractor_failure = sparse_values == sparse.STATE_EXTRACTOR_FAILURE
    excluded = solid | invalid | extractor_failure
    if np.any(excluded & (percentile != PERCENTILE_UNKNOWN)):
        raise ValueError(
            "solid/invalid/extractor_failure 只能携带 percentile unknown，"
            "不能伪装成普通正负样本"
        )

    masks: dict[str, np.ndarray] = {
        "percentile_positive": (
            (percentile == PERCENTILE_POSITIVE) & ~positive_anchor & ~excluded
        ),
        "percentile_negative": (
            (percentile == PERCENTILE_NEGATIVE) & ~positive_anchor & ~excluded
        ),
        "haller_anchor": (
            (percentile == PERCENTILE_UNKNOWN) & positive_anchor & ~excluded
        ),
        "agreement": (
            (percentile == PERCENTILE_POSITIVE) & positive_anchor & ~excluded
        ),
        "conflict": (
            (percentile == PERCENTILE_NEGATIVE) & positive_anchor & ~excluded
        ),
        "unknown_unlabeled": (
            (percentile == PERCENTILE_UNKNOWN) & ~positive_anchor & ~excluded
        ),
        "solid": solid,
        "invalid": invalid,
        "extractor_failure": extractor_failure,
    }
    covered = np.zeros_like(percentile, dtype=bool)
    for mask in masks.values():
        if np.any(covered & mask):
            raise AssertionError("W1-Hybrid audit categories must be disjoint")
        covered |= mask
    if not np.array_equal(covered, np.ones_like(covered, dtype=bool)):
        raise AssertionError("W1-Hybrid audit categories must cover every cell")

    labels = np.zeros_like(percentile, dtype=np.float32)
    labels[masks["haller_anchor"] | masks["agreement"] | masks["percentile_positive"]] = 1.0
    label_mask = (
        masks["percentile_positive"]
        | masks["percentile_negative"]
        | masks["haller_anchor"]
        | masks["agreement"]
    )
    unknown_mask = ~label_mask
    weights = np.zeros_like(percentile, dtype=np.float32)
    weights[masks["percentile_positive"] | masks["percentile_negative"]] = 0.5
    weights[masks["haller_anchor"] | masks["agreement"]] = 1.0
    category_counts = {
        key: _mask_count(mask) for key, mask in masks.items()
    }
    category_stats = {
        key: _empty_component_stats(mask, weights)
        for key, mask in masks.items()
    }
    # A negative Haller state cannot enter this result because the state
    # registry is closed; retaining this explicit field makes the absence
    # auditable in reports and checkpoints.
    return HybridFusionResult(
        labels=labels,
        label_mask=label_mask,
        unknown_mask=unknown_mask,
        weights=weights,
        percentile_state=percentile,
        sparse_state=sparse_values,
        category_masks=masks,
        category_counts=category_counts,
        category_stats=category_stats,
        haller_negative_count=0,
    )


def _copy_fusion_stats(fusion: HybridFusionResult) -> dict[str, dict[str, float | int]]:
    return {
        key: dict(value) for key, value in fusion.category_stats.items()
    }


def _as_prediction_tensor(
    value: Any,
    expected_shape: tuple[int, ...],
    *,
    name: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} shape={tuple(tensor.shape)} 与 expected={expected_shape} 不一致"
        )
    if not tensor.is_floating_point():
        raise ValueError(f"{name} 必须是浮点 sigmoid probability")
    tensor = tensor if device is None else tensor.to(device=device)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} 含有非有限值")
    if bool((tensor < 0.0).any()) or bool((tensor > 1.0).any()):
        raise ValueError(f"{name} 必须位于 [0,1]")
    return tensor


def _tensor_mask(value: Any, shape: tuple[int, ...], *, name: str, device: torch.device) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape={tuple(tensor.shape)} 与 expected={shape} 不一致")
    if tensor.dtype != torch.bool:
        if not bool(torch.all((tensor == 0) | (tensor == 1))):
            raise ValueError(f"{name} 必须是 bool/0-1 mask")
        tensor = tensor.to(dtype=torch.bool)
    return tensor.to(device=device)


def _weighted_bce(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = weights > 0.0
    if not bool(active.any()):
        return predictions.sum() * 0.0, torch.zeros_like(predictions)
    element = F.binary_cross_entropy(
        predictions, targets, reduction="none"
    )
    denominator = weights[active].sum().clamp_min(torch.finfo(weights.dtype).eps)
    return (element[active] * weights[active]).sum() / denominator, element


def _set_hard_component_contributions(
    fusion: HybridFusionResult,
    element_loss: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    stats = _copy_fusion_stats(fusion)
    active = weights > 0.0
    denominator = weights[active].sum() if bool(active.any()) else None
    for key in _HARD_COMPONENTS:
        mask = torch.as_tensor(
            fusion.category_masks[key], device=element_loss.device, dtype=torch.bool
        )
        count = int(mask.sum().item())
        accepted = int((mask & active).sum().item())
        component_weights = weights[mask]
        contribution = 0.0
        if denominator is not None and accepted:
            contribution = float(
                (element_loss[mask] * weights[mask]).sum().detach().cpu()
                / denominator.detach().cpu()
            )
        stats[key] = {
            "eligible_count": count,
            "accepted_count": accepted,
            "mean_weight": float(component_weights.mean().detach().cpu()) if count else 0.0,
            "contribution": contribution,
        }
    return stats


def _loss_component(
    *,
    eligible: torch.Tensor,
    accepted: torch.Tensor,
    mean_weight: float,
    contribution: torch.Tensor | float,
) -> dict[str, float | int]:
    value = (
        float(contribution.detach().cpu())
        if isinstance(contribution, torch.Tensor)
        else float(contribution)
    )
    return {
        "eligible_count": int(eligible.sum().item()),
        "accepted_count": int(accepted.sum().item()),
        "mean_weight": float(mean_weight),
        "contribution": value,
    }


def _base_hybrid_loss(
    student_predictions: Any,
    batch: "HybridBatch",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not isinstance(batch, HybridBatch):
        raise TypeError("Hybrid loss 必须消费 HybridBatch")
    expected_shape = tuple(int(value) for value in batch.labels.shape)
    student = _as_prediction_tensor(
        student_predictions, expected_shape, name="student_predictions"
    )
    device = student.device
    labels = torch.as_tensor(batch.labels, device=device, dtype=student.dtype)
    weights = torch.as_tensor(batch.fusion.weights, device=device, dtype=student.dtype)
    known = torch.as_tensor(batch.fusion.label_mask, device=device, dtype=torch.bool)
    if not torch.equal(known, weights > 0.0):
        raise ValueError("Hybrid label_mask 必须与正权重 hard target 一致")
    hard_loss, element_loss = _weighted_bce(student, labels, weights)
    components = _set_hard_component_contributions(
        batch.fusion, element_loss, weights
    )
    return hard_loss, student, labels, weights, {
        "device": device,
        "known": known,
        "components": components,
    }


def _finalize_hybrid_stats(
    *,
    batch: "HybridBatch",
    total: torch.Tensor,
    hard_loss: torch.Tensor,
    pseudo_loss: torch.Tensor,
    consistency_loss: torch.Tensor,
    components: dict[str, dict[str, float | int]],
    ramp: float,
    pseudo_eligible: torch.Tensor,
    pseudo_mask: torch.Tensor,
    pseudo_positive: torch.Tensor,
    pseudo_negative: torch.Tensor,
    consistency_eligible: torch.Tensor,
    consistency_accepted: torch.Tensor,
    mean_probability: torch.Tensor | None = None,
    predictive_variance: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    view_count: int | None = None,
    variance_gate: float | None = None,
) -> dict[str, Any]:
    components = copy.deepcopy(components)
    pseudo_weight = float(ramp)
    components["unknown_unlabeled"] = {
        **components.get("unknown_unlabeled", {}),
        "contribution": 0.0,
    }
    components["pseudo_accepted"] = _loss_component(
        eligible=pseudo_eligible,
        accepted=pseudo_mask,
        mean_weight=(pseudo_weight if bool(pseudo_mask.any()) else 0.0),
        contribution=ramp * batch.pseudo_weight * pseudo_loss,
    )
    components["consistency"] = _loss_component(
        eligible=consistency_eligible,
        accepted=consistency_accepted,
        mean_weight=(ramp if bool(consistency_accepted.any()) else 0.0),
        contribution=ramp * batch.consistency_weight * consistency_loss,
    )
    stats: dict[str, Any] = {
        "loss": float(total.detach().cpu()),
        "hard_loss": float(hard_loss.detach().cpu()),
        "pseudo_loss": float(pseudo_loss.detach().cpu()),
        "consistency_loss": float(consistency_loss.detach().cpu()),
        "ramp_weight": float(ramp),
        "loss_components": components,
        "fusion": batch.fusion.as_dict(),
        "haller_negative_count": int(batch.fusion.haller_negative_count),
        "conflict_count": int(batch.fusion.category_counts["conflict"]),
        "unknown_unlabeled_count": int(batch.fusion.category_counts["unknown_unlabeled"]),
        "pseudo_eligible_count": int(pseudo_eligible.sum().item()),
        "pseudo_accepted_count": int(pseudo_mask.sum().item()),
        "pseudo_positive_count": int(pseudo_positive.sum().item()),
        "pseudo_negative_count": int(pseudo_negative.sum().item()),
        "pseudo_acceptance": (
            float(pseudo_mask.sum().item() / pseudo_eligible.sum().item())
            if bool(pseudo_eligible.any()) else 0.0
        ),
        "consistency_eligible_count": int(consistency_eligible.sum().item()),
        "consistency_accepted_count": int(consistency_accepted.sum().item()),
        "sampling_source": batch.sampling_source,
        "loss_source": batch.label_source,
        "anchor_hash": batch.anchor_hash,
        "protocol_version": sparse.PROTOCOL_VERSION,
        "construction": batch.anchor_metadata.get("construction") if batch.anchor_metadata else None,
        "source_fingerprint": batch.anchor_metadata.get("source_fingerprints") if batch.anchor_metadata else None,
    }
    if mean_probability is not None:
        stats["mean_probability"] = _distribution(mean_probability)
    if predictive_variance is not None:
        stats["predictive_variance"] = _distribution(predictive_variance)
    if entropy is not None:
        stats["entropy"] = _distribution(entropy)
    if view_count is not None:
        stats["view_count"] = int(view_count)
    if variance_gate is not None:
        stats["variance_gate"] = float(variance_gate)
    return stats


def _distribution(value: torch.Tensor) -> dict[str, float]:
    flattened = value.detach().float().reshape(-1)
    if flattened.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(flattened.mean().cpu()),
        "std": float(flattened.std(unbiased=False).cpu()),
        "min": float(flattened.min().cpu()),
        "max": float(flattened.max().cpu()),
    }


def compute_w1_hybrid_loss(
    student_predictions: Any,
    teacher_predictions: Any,
    batch: "HybridBatch",
    *,
    config: w1_p.W1PConfig | None = None,
    epoch: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute weighted fusion BCE plus W1 pseudo/consistency losses."""
    config = w1_p.W1PConfig() if config is None else config
    if not isinstance(config, w1_p.W1PConfig):
        raise TypeError("W1-Hybrid config 必须是 W1PConfig")
    epoch = _nonnegative_int(epoch, name="epoch")
    batch = batch.for_mode(W1_HYBRID_METHOD)
    hard_loss, student, _labels, _weights, base = _base_hybrid_loss(
        student_predictions, batch
    )
    teacher = _as_prediction_tensor(
        teacher_predictions, tuple(student.shape), name="teacher_predictions",
        device=student.device,
    ).detach()
    pseudo_eligible = torch.as_tensor(
        batch.fusion.category_masks["unknown_unlabeled"],
        device=student.device, dtype=torch.bool,
    )
    high = torch.as_tensor(config.pseudo_high, device=student.device, dtype=student.dtype)
    low = torch.as_tensor(config.pseudo_low, device=student.device, dtype=student.dtype)
    pseudo_mask = pseudo_eligible & ((teacher >= high) | (teacher <= low))
    pseudo_targets = (teacher >= 0.5).to(dtype=student.dtype)
    consistency_eligible = pseudo_eligible
    consistency_accepted = pseudo_eligible
    pseudo_loss, _ = _weighted_bce(
        student, pseudo_targets, pseudo_mask.to(dtype=student.dtype)
    )
    consistency_loss = (
        F.mse_loss(student[consistency_eligible], teacher[consistency_eligible])
        if bool(consistency_eligible.any()) else student.sum() * 0.0
    )
    ramp = w1_p.ramp_up_weight(epoch, config.ramp_up_epochs)
    total = (
        hard_loss
        + ramp * config.pseudo_weight * pseudo_loss
        + ramp * config.consistency_weight * consistency_loss
    )
    components = base["components"]
    stats = _finalize_hybrid_stats(
        batch=batch,
        total=total,
        hard_loss=hard_loss,
        pseudo_loss=pseudo_loss,
        consistency_loss=consistency_loss,
        components=components,
        ramp=ramp,
        pseudo_eligible=pseudo_eligible,
        pseudo_mask=pseudo_mask,
        pseudo_positive=pseudo_mask & (pseudo_targets >= 0.5),
        pseudo_negative=pseudo_mask & (pseudo_targets < 0.5),
        consistency_eligible=consistency_eligible,
        consistency_accepted=consistency_accepted,
    )
    return total, stats


def compute_w2_on_hybrid_loss(
    student_predictions: Any,
    teacher_views: Any,
    batch: "HybridBatch",
    *,
    config: w2.W2Config,
    epoch: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute W2's three-view gate on top of the exact Hybrid fusion."""
    if not isinstance(config, w2.W2Config):
        raise TypeError("W2-on-Hybrid config 必须是 W2Config")
    if config.variance_gate is None:
        raise ValueError("W2-on-Hybrid 必须显式提供 variance_gate")
    epoch = _nonnegative_int(epoch, name="epoch")
    batch = batch.for_mode(W2_ON_HYBRID_METHOD)
    hard_loss, student, _labels, _weights, base = _base_hybrid_loss(
        student_predictions, batch
    )
    statistics = w2.compute_w2_statistics(teacher_views)
    mean = statistics.mean_probability.to(device=student.device, dtype=student.dtype)
    variance = statistics.predictive_variance.to(device=student.device, dtype=student.dtype)
    entropy = statistics.entropy.to(device=student.device, dtype=student.dtype)
    if tuple(mean.shape) != tuple(student.shape):
        raise ValueError("W2-on-Hybrid teacher statistics shape 不匹配")
    pseudo_eligible = torch.as_tensor(
        batch.fusion.category_masks["unknown_unlabeled"],
        device=student.device, dtype=torch.bool,
    )
    solid = torch.as_tensor(batch.solid_mask, device=student.device, dtype=torch.bool)
    failed = torch.as_tensor(batch.failed_frame_mask, device=student.device, dtype=torch.bool)
    conflict = torch.as_tensor(
        batch.fusion.category_masks["conflict"],
        device=student.device, dtype=torch.bool,
    )
    # The existing W2 gate has explicit solid/failed exclusion inputs but no
    # generic ignored mask.  Include conflict in that closed gate domain so
    # it is rejected before candidate construction; it still remains a
    # conflict with zero Hybrid loss weight, never a physical solid state.
    gate_unknown = pseudo_eligible | solid | failed | conflict
    gate = w2.apply_w2_uncertainty_gate(
        mean,
        variance,
        gate_unknown,
        solid | conflict,
        failed_frame_mask=failed,
        variance_gate=config.variance_gate,
    )
    pseudo_mask = gate.pseudo_mask
    pseudo_targets = gate.pseudo_labels.to(dtype=student.dtype)
    pseudo_loss, _ = _weighted_bce(
        student, pseudo_targets, pseudo_mask.to(dtype=student.dtype)
    )
    # W2 consistency only consumes accepted low-uncertainty pseudo labels.
    consistency_eligible = pseudo_eligible
    consistency_accepted = pseudo_mask
    consistency_loss = (
        F.mse_loss(student[pseudo_mask], mean[pseudo_mask])
        if bool(pseudo_mask.any()) else student.sum() * 0.0
    )
    ramp = w1_p.ramp_up_weight(epoch, config.ramp_up_epochs)
    total = (
        hard_loss
        + ramp * config.pseudo_weight * pseudo_loss
        + ramp * config.consistency_weight * consistency_loss
    )
    components = base["components"]
    stats = _finalize_hybrid_stats(
        batch=batch,
        total=total,
        hard_loss=hard_loss,
        pseudo_loss=pseudo_loss,
        consistency_loss=consistency_loss,
        components=components,
        ramp=ramp,
        pseudo_eligible=pseudo_eligible,
        pseudo_mask=pseudo_mask,
        pseudo_positive=gate.positive_mask,
        pseudo_negative=gate.negative_mask,
        consistency_eligible=consistency_eligible,
        consistency_accepted=consistency_accepted,
        mean_probability=mean,
        predictive_variance=variance,
        entropy=entropy,
        view_count=statistics.view_count,
        variance_gate=config.variance_gate,
    )
    stats.update({
        "candidate_count": int(gate.candidate_mask.sum().item()),
        "low_uncertainty_count": int(gate.low_uncertainty_mask.sum().item()),
        "confidence_count": int(gate.confidence_mask.sum().item()),
        "rejected_uncertain_count": int(
            (gate.candidate_mask & ~gate.low_uncertainty_mask).sum().item()
        ),
        "rejected_not_negative": True,
        "entropy_is_diagnostic_only": True,
    })
    return total, stats


@dataclass
class HybridBatch:
    """A train-only batch retaining raw percentile and sparse state provenance."""

    contract_batch: contract.WeakSupervisionBatch
    fusion: HybridFusionResult
    solid_mask: Any
    invalid_support: Any
    failed_frame_mask: Any
    anchor_hash: str
    dummy_field: Any | None = None
    anchor_metadata: Mapping[str, Any] | None = None
    mode: str = W1_HYBRID_METHOD
    pseudo_weight: float = 1.0
    consistency_weight: float = 1.0

    def __post_init__(self) -> None:
        self.mode = contract.canonical_mode(self.mode)
        if self.mode not in {W1_HYBRID_METHOD, W2_ON_HYBRID_METHOD}:
            raise ValueError("HybridBatch mode 必须是 W1-Hybrid 或 W2-on-Hybrid")
        contract.validate_training_batch(self.contract_batch, self.mode)
        if self.contract_batch.label_source != contract.LABEL_SOURCE_P95P80:
            raise ValueError("Hybrid formal checkpoint/loss source 必须是 P95/P80")
        shape = tuple(int(value) for value in self.fusion.labels.shape)
        if tuple(self.contract_batch.labels.shape) != shape:
            raise ValueError("Hybrid fusion 与 contract labels shape 不一致")
        if not np.array_equal(
            np.asarray(self.contract_batch.labels.detach().cpu() if isinstance(self.contract_batch.labels, torch.Tensor) else self.contract_batch.labels),
            self.fusion.labels,
        ):
            raise ValueError("Hybrid contract labels 与 fusion labels 不一致")
        for name, value in (
            ("solid_mask", self.solid_mask),
            ("invalid_support", self.invalid_support),
            ("failed_frame_mask", self.failed_frame_mask),
        ):
            array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            if tuple(array.shape) != shape:
                raise ValueError(f"{name} shape 与 fusion labels 不一致")
            if not np.isin(array, [False, True]).all():
                raise ValueError(f"{name} 必须是 bool mask")
            setattr(self, name, array.astype(bool, copy=True))
        if not np.array_equal(self.solid_mask, self.fusion.sparse_state == sparse.STATE_SOLID):
            raise ValueError("solid_mask 与 sparse_state 不一致")
        if not np.array_equal(self.invalid_support, self.fusion.sparse_state == sparse.STATE_INVALID):
            raise ValueError("invalid_support 与 sparse_state 不一致")
        if not np.array_equal(self.failed_frame_mask, self.fusion.sparse_state == sparse.STATE_EXTRACTOR_FAILURE):
            raise ValueError("failed_frame_mask 与 sparse_state 不一致")
        self.anchor_hash = _nonempty_text(self.anchor_hash, name="anchor_hash")
        if self.anchor_metadata is not None and not isinstance(self.anchor_metadata, Mapping):
            raise TypeError("anchor_metadata 必须是 object")
        self.anchor_metadata = copy.deepcopy(dict(self.anchor_metadata or {}))
        self.pseudo_weight = float(self.pseudo_weight)
        self.consistency_weight = float(self.consistency_weight)
        if not np.isfinite(self.pseudo_weight) or self.pseudo_weight < 0.0:
            raise ValueError("pseudo_weight 必须是非负有限数")
        if not np.isfinite(self.consistency_weight) or self.consistency_weight < 0.0:
            raise ValueError("consistency_weight 必须是非负有限数")
        if self.dummy_field is None:
            if isinstance(self.pathlines, torch.Tensor):
                self.dummy_field = self.pathlines.new_zeros((shape[0], 1, 1, 1))
            else:
                self.dummy_field = np.zeros((shape[0], 1, 1, 1), dtype=np.float32)
        if getattr(self.dummy_field, "shape", ())[0] != shape[0]:
            raise ValueError("dummy_field batch dimension 与 fusion labels 不一致")

    @property
    def pathlines(self) -> Any:
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

    def for_mode(self, mode: str) -> "HybridBatch":
        canonical = contract.canonical_mode(mode)
        if canonical not in {W1_HYBRID_METHOD, W2_ON_HYBRID_METHOD}:
            raise ValueError("Hybrid batch 只能切换到 W1-Hybrid/W2-on-Hybrid")
        if canonical == self.mode:
            return self
        base = self.contract_batch
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
            mode=canonical,
            input_schema=base.input_schema,
        )
        return HybridBatch(
            converted, self.fusion, self.solid_mask, self.invalid_support,
            self.failed_frame_mask, self.anchor_hash, self.dummy_field,
            self.anchor_metadata, canonical, self.pseudo_weight,
            self.consistency_weight,
        )

    def to(self, device: str | torch.device, *, mode: str | None = None) -> "HybridBatch":
        canonical = self.mode if mode is None else contract.canonical_mode(mode)
        base = self.contract_batch
        converted = contract.WeakSupervisionBatch(
            pathlines=torch.as_tensor(base.pathlines, device=device),
            labels=torch.as_tensor(base.labels, device=device, dtype=torch.float32),
            label_source=base.label_source,
            split_name=base.split_name,
            feature_schema=base.feature_schema,
            label_mask=torch.as_tensor(base.label_mask, device=device),
            unknown_mask=torch.as_tensor(base.unknown_mask, device=device),
            sampling_source=base.sampling_source,
            provenance=base.provenance,
            mode=canonical,
            input_schema=base.input_schema,
        )
        return HybridBatch(
            converted, self.fusion, self.solid_mask, self.invalid_support,
            self.failed_frame_mask, self.anchor_hash,
            torch.as_tensor(self.dummy_field, device=device),
            self.anchor_metadata, canonical, self.pseudo_weight,
            self.consistency_weight,
        )

    def to_w1_p_batch(self) -> w1_p.W1PBatch:
        """Expose only percentile states to W1-P; Haller is not its loss source."""
        percentile = self.fusion.percentile_state
        labels = (percentile == PERCENTILE_POSITIVE).astype(np.float32)
        known = percentile != PERCENTILE_UNKNOWN
        unknown = ~known
        provenance = {
            "percentile": {
                "source": contract.LABEL_SOURCE_P95P80,
                "states": ["positive", "negative", "unknown"],
            },
            "sampling": {
                "source": self.sampling_source,
            },
        }
        window = self.contract_batch.provenance.get("window")
        if window is not None:
            provenance["window"] = copy.deepcopy(dict(window))
        return w1_p.build_w1_p_batch(
            self.pathlines,
            labels,
            known,
            unknown,
            self.solid_mask | self.invalid_support | self.failed_frame_mask,
            sampling_source=self.sampling_source,
            split_name="train",
            provenance=provenance,
            dummy_field=self.dummy_field,
        )


def build_w1_hybrid_batch(
    pathlines: Any,
    percentile_state: Any,
    sparse_state: Any,
    *,
    anchor_hash: str,
    sampling_source: str,
    split_name: str = "train",
    provenance: Mapping[str, Any] | None = None,
    dummy_field: Any | None = None,
    anchor_metadata: Mapping[str, Any] | None = None,
    mode: str = W1_HYBRID_METHOD,
) -> HybridBatch:
    """Construct a train-only Hybrid batch with explicit source provenance."""
    canonical = contract.canonical_mode(mode)
    if canonical not in {W1_HYBRID_METHOD, W2_ON_HYBRID_METHOD}:
        raise ValueError("build_w1_hybrid_batch mode 必须是 W1-Hybrid/W2-on-Hybrid")
    if split_name != "train":
        raise ValueError("Hybrid training batch 只能来自 split=train")
    sampling_source = contract.validate_label_source(sampling_source)
    if sampling_source not in {
        contract.LABEL_SOURCE_P95P80,
        contract.LABEL_SOURCE_LEGACY_P85,
    }:
        raise ValueError("Hybrid sampling_source 只能是 P95/P80 或 legacy_p85")
    fusion = fuse_w1_hybrid_targets(percentile_state, sparse_state)
    sources = copy.deepcopy(dict(provenance or {}))
    anchor = dict(sources.get("anchor", {}))
    anchor.setdefault("source", contract.LABEL_SOURCE_HALLER_TRAIN)
    anchor.setdefault("anchor_hash", anchor_hash)
    anchor.setdefault("states", list(sparse.SPARSE_STATES))
    anchor.setdefault("uncovered_is_not_negative", True)
    anchor.setdefault("hard_negative_state", None)
    sources["anchor"] = anchor
    sampling = dict(sources.get("sampling", {}))
    sampling.setdefault("source", sampling_source)
    sources["sampling"] = sampling
    base = contract.WeakSupervisionBatch(
        pathlines=pathlines,
        labels=fusion.labels,
        label_source=contract.LABEL_SOURCE_P95P80,
        split_name="train",
        feature_schema=contract.FEATURE_SCHEMA_7,
        label_mask=fusion.label_mask,
        unknown_mask=fusion.unknown_mask,
        sampling_source=sampling_source,
        provenance=sources,
        mode=canonical,
        input_schema=contract.FEATURE_SCHEMA_7,
    )
    return HybridBatch(
        base,
        fusion,
        fusion.sparse_state == sparse.STATE_SOLID,
        fusion.sparse_state == sparse.STATE_INVALID,
        fusion.sparse_state == sparse.STATE_EXTRACTOR_FAILURE,
        anchor_hash,
        dummy_field,
        anchor_metadata,
        canonical,
    )


def _validate_calibration_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} shape={tuple(array.shape)} 与 expected={shape} 不一致")
    if name == "labels":
        array = array.astype(np.float32, copy=True)
        if not np.isfinite(array).all() or not np.isin(array, [0.0, 1.0]).all():
            raise ValueError("calibration labels 必须是有限 0/1")
        return array
    if not np.isin(array, [False, True]).all():
        raise ValueError(f"{name} 必须是 bool mask")
    return array.astype(bool, copy=True)


@dataclass(frozen=True)
class CalibrationBatch:
    """Calibration-only percentile reference batch; no final-test accessor."""

    pathlines: Any
    labels: Any
    known_mask: Any
    valid_fluid: Any
    dataset_name: str = "ticket06_fixture"
    split_name: str = "calibration"
    dummy_field: Any | None = None

    def __post_init__(self) -> None:
        if self.split_name != "calibration":
            raise PermissionError("CalibrationBatch 只能使用 split=calibration")
        pathlines = torch.as_tensor(self.pathlines)
        if pathlines.ndim != 4 or pathlines.shape[0] < 1 or pathlines.shape[-1] != 7:
            raise ValueError("calibration pathlines 必须是 (batch,time,trajectory,7)")
        shape = (int(pathlines.shape[0]), int(pathlines.shape[2]))
        labels = _validate_calibration_array(self.labels, shape=shape, name="labels")
        known = _validate_calibration_array(self.known_mask, shape=shape, name="known_mask")
        valid = _validate_calibration_array(self.valid_fluid, shape=shape, name="valid_fluid")
        if not bool((known & valid).any()):
            raise ValueError("calibration 必须包含 known valid-fluid cells")
        object.__setattr__(self, "pathlines", pathlines.float())
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "known_mask", known)
        object.__setattr__(self, "valid_fluid", valid)
        object.__setattr__(self, "dataset_name", _nonempty_text(self.dataset_name, name="dataset_name"))
        if self.dummy_field is None:
            object.__setattr__(
                self, "dummy_field", torch.zeros((shape[0], 1, 1, 1), dtype=torch.float32)
            )
        else:
            dummy = torch.as_tensor(self.dummy_field)
            if dummy.shape[0] != shape[0]:
                raise ValueError("calibration dummy_field batch dimension 不一致")
            object.__setattr__(self, "dummy_field", dummy.float())


def _calibration_predicate_mask(
    mean_probability: np.ndarray,
    predictive_variance: np.ndarray,
    labels: np.ndarray,
    known_mask: np.ndarray,
    valid_fluid: np.ndarray,
    *,
    prediction_threshold: float,
    variance_gate: float,
) -> dict[str, Any]:
    threshold = _probability(prediction_threshold, name="prediction_threshold")
    gate = _probability(variance_gate, name="variance_gate")
    known_valid = known_mask & valid_fluid
    confidence = (mean_probability >= 0.90) | (mean_probability <= 0.10)
    low_uncertainty = predictive_variance <= gate
    accepted = known_valid & confidence & low_uncertainty
    positive = accepted & (mean_probability >= threshold)
    negative = accepted & ~positive
    target_positive = labels >= 0.5
    true_positive = int(np.count_nonzero(positive & target_positive))
    false_positive = int(np.count_nonzero(positive & ~target_positive))
    false_negative = int(np.count_nonzero(negative & target_positive))
    true_negative = int(np.count_nonzero(negative & ~target_positive))
    denominator = int(np.count_nonzero(accepted))
    numerator = false_positive + false_negative
    risk = float(numerator / denominator) if denominator else None
    return {
        "prediction_threshold": threshold,
        "variance_gate": gate,
        "known_valid_fluid_count": int(np.count_nonzero(known_valid)),
        "accepted_count": denominator,
        "accepted_positive_count": int(np.count_nonzero(positive)),
        "accepted_negative_count": int(np.count_nonzero(negative)),
        "rejected_count": int(np.count_nonzero(known_valid & ~accepted)),
        "rejected_high_uncertainty_count": int(np.count_nonzero(known_valid & ~low_uncertainty)),
        "rejected_low_confidence_count": int(np.count_nonzero(known_valid & ~confidence)),
        "accepted_true_positive": true_positive,
        "accepted_false_positive": false_positive,
        "accepted_false_negative": false_negative,
        "accepted_true_negative": true_negative,
        "selective_risk_numerator": int(numerator),
        "selective_risk_denominator": denominator,
        "selective_risk": risk,
        "selective_risk_status": "defined" if risk is not None else "undefined",
        "operating_point_pass": bool(denominator > 0),
        "rejected_excluded_from_confusion": True,
        "rejected_not_negative": True,
        "coverage": (
            float(denominator / np.count_nonzero(known_valid))
            if np.any(known_valid) else None
        ),
    }


def evaluate_w2_selective(
    *,
    mean_probability: Any,
    predictive_variance: Any,
    labels: Any,
    known_mask: Any,
    valid_fluid: Any,
    prediction_threshold: float,
    variance_gate: float,
    split_name: str = "calibration",
) -> dict[str, Any]:
    """Evaluate W2 with confidence/uncertainty abstention semantics.

    The selective-risk denominator is exactly the accepted known valid-fluid
    count.  An empty denominator is ``undefined`` and fails the operating
    point; it is never converted to a zero risk.
    """
    if split_name not in {"train", "calibration"}:
        raise PermissionError("W2 selective evaluation 禁止读取 final-test split")
    mean = np.asarray(mean_probability, dtype=np.float64)
    variance = np.asarray(predictive_variance, dtype=np.float64)
    if mean.ndim == 0 or tuple(variance.shape) != tuple(mean.shape):
        raise ValueError("mean_probability/predictive_variance 必须是同 shape 数组")
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ValueError("W2 selective input 必须全部有限")
    if not np.all((mean >= 0.0) & (mean <= 1.0)):
        raise ValueError("mean_probability 必须位于 [0,1]")
    if not np.all((variance >= 0.0) & (variance <= 0.25)):
        raise ValueError("predictive_variance 必须位于 [0,0.25]")
    shape = tuple(int(value) for value in mean.shape)
    labels_array = _validate_calibration_array(labels, shape=shape, name="labels")
    known = _validate_calibration_array(known_mask, shape=shape, name="known_mask")
    valid = _validate_calibration_array(valid_fluid, shape=shape, name="valid_fluid")
    return {
        "split_name": split_name,
        "source": "percentile_calibration_reference",
        **_calibration_predicate_mask(
            mean, variance, labels_array, known, valid,
            prediction_threshold=prediction_threshold,
            variance_gate=variance_gate,
        ),
    }


def _anchor_identity(bundle: Mapping[str, Any]) -> dict[str, Any]:
    manifest = bundle["manifest"]
    integrity = bundle["integrity"]
    qualification = bundle["qualification_report"]
    required = {
        "protocol_version": sparse.PROTOCOL_VERSION,
        "artifact_type": sparse.ARTIFACT_TYPE,
        "artifact_version": sparse.ARTIFACT_VERSION,
        "artifact_state": "qualified",
        "status": "qualified",
        "allowed_for_training_anchor": True,
        "eligible_as_dense_evaluation_reference": False,
        "trusted_haller_v2": False,
        "construction": manifest.get("construction"),
        "artifact_fingerprint": manifest.get("integrity_fingerprint"),
        "qualification_report_fingerprint": manifest.get("qualification_report_fingerprint"),
        "source_fingerprints": manifest.get("source_fingerprints"),
        "sample_count": manifest.get("sample_count"),
        "final_test_relationships": "not_read",
        "frame_1300": "excluded",
    }
    if manifest.get("artifact_state") != "qualified" or manifest.get("status") != "qualified":
        raise PermissionError("本票只接受 status/artifact_state=qualified 的 sparse-anchor bundle")
    if manifest.get("allowed_for_training_anchor") is not True:
        raise PermissionError("sparse-anchor bundle 没有 allowed_for_training_anchor=true")
    if manifest.get("eligible_as_dense_evaluation_reference") is not False:
        raise ValueError("sparse-anchor bundle 不能升级为 dense evaluation reference")
    if manifest.get("trusted_haller_v2") is not False:
        raise ValueError("sparse-anchor bundle trusted_haller_v2 guard 被解除")
    if manifest.get("final_test_relationships") != "not_read":
        raise PermissionError("sparse-anchor bundle final-test relationship guard 未封存")
    if manifest.get("frame_1300_excluded") is not True:
        raise ValueError("sparse-anchor bundle 未排除 Boussinesq frame 1300")
    if not isinstance(required["artifact_fingerprint"], str) or not required["artifact_fingerprint"]:
        raise ValueError("sparse-anchor bundle 缺少 artifact fingerprint")
    if not isinstance(required["qualification_report_fingerprint"], str) or not required["qualification_report_fingerprint"]:
        raise ValueError("sparse-anchor bundle 缺少 qualification fingerprint")
    if not isinstance(required["source_fingerprints"], list) or not required["source_fingerprints"]:
        raise ValueError("sparse-anchor bundle 缺少 source fingerprints")
    if integrity.get("artifact_fingerprint") != required["artifact_fingerprint"]:
        raise ValueError("sparse-anchor bundle sidecar fingerprint 不一致")
    if qualification.get("status") != "qualified":
        raise PermissionError("qualification report 未通过 qualified gate")
    return _jsonable(required)


def load_qualified_anchor_identity(bundle_dir: str | pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the current protocol bundle before batch construction."""
    bundle = sparse.load_sparse_anchor_bundle(
        bundle_dir, require_training_anchor=True
    )
    return _anchor_identity(bundle), bundle


def _first_index(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return None if indices.size == 0 else int(indices[0])


def build_cpu_closure_inputs(
    bundle_dir: str | pathlib.Path,
    *,
    seed: int = 17,
) -> tuple[dict[str, Any], list[HybridBatch], list[CalibrationBatch]]:
    """Derive small split-contained CPU inputs from one qualified sample.

    This is the CLI smoke/closure fixture.  It reads only the qualified
    bundle's already validated train/calibration-safe sample arrays and never
    requests an evaluation/test sample.
    """
    identity, bundle = load_qualified_anchor_identity(bundle_dir)
    samples = bundle.get("samples", {})
    if not samples:
        raise ValueError("qualified sparse-anchor bundle 必须包含至少一个 sample")
    sample_values = [
        value for value in samples.values()
        if value.get("metadata", {}).get("split_name") == "train"
    ]
    if not sample_values:
        raise PermissionError(
            "closure training fixture 必须从 qualified bundle 的 split=train sample 构造"
        )
    sample = next(
        (value for value in sample_values if np.asarray(value["anchor_mask"]).any()),
        sample_values[0],
    )
    sparse_flat = np.asarray(sample["sparse_state"]).reshape(-1)
    if sparse_flat.size == 0:
        raise ValueError("qualified sparse-anchor sample 为空")
    groups = [
        _first_index(sparse_flat == sparse.STATE_POSITIVE_ANCHOR),
        _first_index(sparse_flat == sparse.STATE_UNCOVERED),
        _first_index(sparse_flat == sparse.STATE_INVALID),
        _first_index(sparse_flat == sparse.STATE_EXTRACTOR_FAILURE),
        _first_index(sparse_flat == sparse.STATE_SOLID),
    ]
    selected = [index for index in groups if index is not None]
    selected.extend(int(index) for index in range(sparse_flat.size) if index not in selected)
    selected = selected[:12]
    if len(selected) < 4:
        raise ValueError("qualified sample 没有足够的 sparse state fixture cells")
    states = sparse_flat[selected]
    percentile = np.full((1, len(selected)), PERCENTILE_UNKNOWN, dtype=np.int8)
    anchor_seen = 0
    uncovered_seen = 0
    for column, state in enumerate(states):
        if state == sparse.STATE_POSITIVE_ANCHOR:
            if anchor_seen == 0:
                percentile[0, column] = PERCENTILE_UNKNOWN
            elif anchor_seen % 3 == 1:
                percentile[0, column] = PERCENTILE_POSITIVE
            else:
                percentile[0, column] = PERCENTILE_NEGATIVE
            anchor_seen += 1
        elif state == sparse.STATE_UNCOVERED:
            if uncovered_seen == 0:
                percentile[0, column] = PERCENTILE_POSITIVE
            elif uncovered_seen % 2 == 1:
                percentile[0, column] = PERCENTILE_NEGATIVE
            uncovered_seen += 1
    if not np.any(percentile == PERCENTILE_POSITIVE):
        percentile[0, 0] = PERCENTILE_POSITIVE
    if not np.any(percentile == PERCENTILE_NEGATIVE):
        percentile[0, min(1, len(selected) - 1)] = PERCENTILE_NEGATIVE
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_nonnegative_int(seed, name="seed"))
    pathlines = torch.randn(
        (1, 3, len(selected), 7), generator=generator, dtype=torch.float32
    )
    signal = torch.as_tensor(
        np.where(percentile[0] == PERCENTILE_POSITIVE, 1.5,
                 np.where(percentile[0] == PERCENTILE_NEGATIVE, -1.5, 0.0)),
        dtype=pathlines.dtype,
    )
    pathlines[..., 0] += signal.view(1, 1, -1)
    sample_metadata = sample["metadata"]
    source_fingerprint = sample_metadata.get("source_fingerprint")
    provenance = {
        "anchor": {
            "source": contract.LABEL_SOURCE_HALLER_TRAIN,
            "anchor_hash": identity["artifact_fingerprint"],
            "artifact_fingerprint": identity["artifact_fingerprint"],
            "qualification_report_fingerprint": identity["qualification_report_fingerprint"],
            "construction": identity["construction"],
            "states": list(sparse.SPARSE_STATES),
        },
        "sampling": {"source": contract.LABEL_SOURCE_P95P80},
        "window": {
            "dataset_name": "ticket06_cpu_fixture",
            "split_name": "train",
            "frame_range": [0, 2],
            "t_win": 3,
            "window_step": 1,
            "sampling_source": contract.LABEL_SOURCE_P95P80,
        },
    }
    if source_fingerprint is not None:
        provenance["anchor"]["source_fingerprint"] = source_fingerprint
    batch = build_w1_hybrid_batch(
        pathlines,
        percentile,
        states.reshape(1, -1),
        anchor_hash=identity["artifact_fingerprint"],
        sampling_source=contract.LABEL_SOURCE_P95P80,
        provenance=provenance,
        anchor_metadata=identity,
    )
    valid = np.asarray([
        state not in _EXCLUDED_SPARSE_STATES for state in states
    ], dtype=bool).reshape(1, -1)
    known = (percentile >= 0) & valid
    calibration = CalibrationBatch(
        pathlines=pathlines + 0.02,
        labels=(percentile == PERCENTILE_POSITIVE).astype(np.float32),
        known_mask=known,
        valid_fluid=valid,
        dataset_name="ticket06_cpu_fixture",
    )
    return identity, [batch], [calibration]


class _DefaultClosureModel(nn.Module):
    """Small stochastic CPU model used only by the closure CLI."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(7, 8)
        self.dropout = nn.Dropout(p=0.35)
        self.output = nn.Linear(8, 1)

    def forward(self, data: Any) -> torch.Tensor:
        _dummy, pathlines = data
        hidden = torch.tanh(self.hidden(pathlines.mean(dim=1)))
        return torch.sigmoid(self.output(self.dropout(hidden)).squeeze(-1))


def _forward_model(model: nn.Module, batch: Any) -> torch.Tensor:
    if isinstance(model, contract.ChannelSelectingAdapter):
        if isinstance(batch, HybridBatch):
            return model.forward_batch(
                batch.contract_batch, dummy_field=batch.dummy_field, consumer="train"
            )
        return model.forward_batch(
            batch.contract_batch, dummy_field=batch.dummy_field, consumer="train"
        )
    return model((batch.dummy_field, batch.pathlines))


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


def _aggregate_epoch_logs(logs: list[dict[str, Any]], *, epoch: int, global_step: int) -> dict[str, Any]:
    if not logs:
        raise ValueError("训练 epoch 为空")
    result: dict[str, Any] = {
        "epoch": int(epoch),
        "steps": len(logs),
        "global_step": int(global_step),
    }
    for key in (
        "loss", "hard_loss", "pseudo_loss", "consistency_loss", "ramp_weight",
        "pseudo_acceptance", "teacher_student_disagreement",
    ):
        values = [float(log[key]) for log in logs if key in log]
        if values:
            result[key] = float(np.mean(values))
    for key in (
        "pseudo_eligible_count", "pseudo_accepted_count", "pseudo_positive_count",
        "pseudo_negative_count", "conflict_count", "unknown_unlabeled_count",
        "consistency_eligible_count", "consistency_accepted_count",
        "candidate_count", "low_uncertainty_count", "confidence_count",
        "rejected_uncertain_count",
    ):
        values = [int(log[key]) for log in logs if key in log]
        if values:
            result[key] = int(sum(values))
    component_keys = set()
    for log in logs:
        component_keys.update(log.get("loss_components", {}))
    components: dict[str, dict[str, float | int]] = {}
    for key in sorted(component_keys):
        rows = [log["loss_components"][key] for log in logs if key in log.get("loss_components", {})]
        components[key] = {
            "eligible_count": int(sum(int(row.get("eligible_count", 0)) for row in rows)),
            "accepted_count": int(sum(int(row.get("accepted_count", 0)) for row in rows)),
            "mean_weight": float(np.mean([float(row.get("mean_weight", 0.0)) for row in rows])),
            "contribution": float(sum(float(row.get("contribution", 0.0)) for row in rows)),
        }
    result["loss_components"] = components
    for key in ("sampling_source", "loss_source", "anchor_hash", "protocol_version", "construction"):
        values = {log.get(key) for log in logs}
        if len(values) == 1:
            result[key] = next(iter(values))
    if any("view_count" in log for log in logs):
        result["view_count"] = 3
        result["variance_gate"] = float(np.mean([log["variance_gate"] for log in logs]))
        for name in ("mean_probability", "predictive_variance", "entropy"):
            rows = [log[name] for log in logs if name in log]
            if rows:
                result[name] = {
                    metric: float(np.mean([row[metric] for row in rows]))
                    for metric in ("mean", "std", "min", "max")
                }
    return result


class _HybridTrainer:
    """Shared actual optimizer/EMA/checkpoint implementation for two modes."""

    def __init__(
        self,
        student: nn.Module,
        optimizer: Any,
        *,
        mode: str,
        sampling_source: str,
        anchor_hash: str,
        anchor_metadata: Mapping[str, Any],
        scheduler: Any,
        config: w1_p.W1PConfig | w2.W2Config,
        seed: int,
        grad_clip_norm: float | None = None,
        teacher: nn.Module | None = None,
    ) -> None:
        if not isinstance(student, nn.Module):
            raise TypeError("Hybrid student 必须是 torch.nn.Module")
        if teacher is not None and not isinstance(teacher, nn.Module):
            raise TypeError("Hybrid teacher 必须是 torch.nn.Module")
        if teacher is student:
            raise ValueError("Hybrid teacher 不能与 student 共享同一 module")
        self.mode = contract.canonical_mode(mode)
        if self.mode not in {W1_HYBRID_METHOD, W2_ON_HYBRID_METHOD}:
            raise ValueError("HybridTrainer mode 非法")
        if optimizer is None or not hasattr(optimizer, "step"):
            raise TypeError("Hybrid optimizer 必须提供 step")
        if scheduler is None or not hasattr(scheduler, "state_dict"):
            raise TypeError("Hybrid resume closure 必须提供 scheduler")
        if self.mode == W2_ON_HYBRID_METHOD and not isinstance(config, w2.W2Config):
            raise TypeError("W2-on-Hybrid 必须使用 W2Config")
        if self.mode == W1_HYBRID_METHOD and not isinstance(config, w1_p.W1PConfig):
            raise TypeError("W1-Hybrid 必须使用 W1PConfig")
        self.student = student
        self.teacher = (
            w1_p.clone_ema_teacher(student) if teacher is None else teacher
        )
        w1_p._prepare_ema_teacher(self.teacher)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.sampling_source = contract.validate_label_source(sampling_source)
        self.anchor_hash = _nonempty_text(anchor_hash, name="anchor_hash")
        self.anchor_metadata = copy.deepcopy(dict(anchor_metadata))
        self.seed = _nonnegative_int(seed, name="seed")
        if grad_clip_norm is not None:
            try:
                grad_clip_norm = float(grad_clip_norm)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Hybrid grad_clip_norm 必须是正的有限数或 None") from exc
            if not np.isfinite(grad_clip_norm) or grad_clip_norm <= 0.0:
                raise ValueError("Hybrid grad_clip_norm 必须是正的有限数或 None")
        self.grad_clip_norm = grad_clip_norm
        self.global_step = 0

    def _move_models(self, device: str | torch.device) -> None:
        self.student.to(device)
        self.teacher.to(device)

    def _teacher_predictions(self, batch: HybridBatch) -> Any:
        if self.mode == W1_HYBRID_METHOD:
            self.teacher.eval()
            with torch.no_grad():
                return _forward_model(self.teacher, batch)
        # Dropout/stochastic modules are enabled for each of exactly three
        # teacher views while their parameters remain frozen.
        self.teacher.train()
        with torch.no_grad():
            return tuple(_forward_model(self.teacher, batch) for _ in range(3))

    def train_step(
        self,
        batch: HybridBatch,
        *,
        epoch: int,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        epoch = _nonnegative_int(epoch, name="epoch")
        if not isinstance(batch, HybridBatch):
            raise TypeError("Hybrid train_step 必须消费 HybridBatch")
        if batch.anchor_hash != self.anchor_hash:
            raise ValueError("Hybrid batch/checkpoint anchor fingerprint 不一致")
        if batch.sampling_source != self.sampling_source:
            raise ValueError("Hybrid sampling_source 与 trainer 不一致")
        self._move_models(device)
        train_batch = batch.to(device, mode=self.mode)
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        student_predictions = _forward_model(self.student, train_batch)
        teacher_predictions = self._teacher_predictions(train_batch)
        if self.mode == W1_HYBRID_METHOD:
            total, stats = compute_w1_hybrid_loss(
                student_predictions, teacher_predictions, train_batch,
                config=self.config, epoch=epoch,
            )
        else:
            total, stats = compute_w2_on_hybrid_loss(
                student_predictions, teacher_predictions, train_batch,
                config=self.config, epoch=epoch,
            )
        total.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.grad_clip_norm
            )
        self.optimizer.step()
        w1_p.update_ema_teacher(
            self.student, self.teacher, decay=self.config.ema_decay
        )
        self.global_step += 1
        stats = dict(stats)
        stats.update({"method": self.mode, "epoch": epoch, "global_step": self.global_step})
        return stats

    def run_epoch(
        self,
        batches: Sequence[HybridBatch],
        *,
        epoch: int,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        # Accept an iterable as well as a materialized sequence.  Real
        # Ticket 04 batches are generated lazily from the current raw source;
        # materializing an entire epoch would needlessly retain every
        # pathline tensor in memory before the first optimizer step.
        logs = []
        for batch in batches:
            logs.append(self.train_step(batch, epoch=epoch, device=device))
        return _aggregate_epoch_logs(logs, epoch=epoch, global_step=self.global_step)

    def _extra_metadata(self, extra: Mapping[str, Any] | None) -> dict[str, Any]:
        if extra is not None and not isinstance(extra, Mapping):
            raise TypeError("Hybrid checkpoint extra_metadata 必须是 object")
        metadata = copy.deepcopy(dict(extra or {}))
        # Keep the historical defaults for standalone Ticket 06 closure
        # runs, but allow the current execution namespace to bind the
        # already-frozen Ticket 01/02/03 protocol explicitly.  The previous
        # unconditional assignment silently downgraded a current Ticket 04
        # checkpoint to the sparse-anchor protocol label.
        metadata.setdefault("ticket", "06-minimal-training-closure")
        metadata.setdefault("protocol_version", sparse.PROTOCOL_VERSION)
        metadata.update({
            "method_mode": self.mode,
            "method_role": "closure_only_no_performance_claim",
            "sparse_anchor_identity": self.anchor_metadata,
            "grad_clip_norm": self.grad_clip_norm,
            "source_weight_policy": {
                "percentile_positive": 0.5,
                "percentile_negative": 0.5,
                "haller_positive_anchor": 1.0,
                "agreement": 1.0,
                "conflict": 0.0,
                "uncovered_is_not_negative": True,
                "haller_hard_negative_state": None,
            },
            "uncertainty_settings": (
                {
                    "view_count": 3,
                    "variance_gate": self.config.variance_gate,
                    "pseudo_high": 0.90,
                    "pseudo_low": 0.10,
                    "entropy_is_diagnostic_only": True,
                }
                if self.mode == W2_ON_HYBRID_METHOD else {
                    "view_count": 1,
                    "variance_gate": None,
                    "pseudo_high": self.config.pseudo_high,
                    "pseudo_low": self.config.pseudo_low,
                    "entropy_is_diagnostic_only": None,
                }
            ),
        })
        return metadata

    def save_checkpoint(
        self,
        path: pathlib.Path,
        *,
        epoch: int,
        dataset_config: Mapping[str, Any],
        split_config: Mapping[str, Any],
        sampling_config: Mapping[str, Any],
        metrics: Mapping[str, Any],
        calibration_policy: Mapping[str, Any],
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> pathlib.Path:
        return contract.save_checkpoint(
            path,
            self.student,
            self.optimizer,
            self.scheduler,
            mode=self.mode,
            feature_schema=contract.FEATURE_SCHEMA_7,
            adapter_input_schema=contract.FEATURE_SCHEMA_7,
            dataset_config=dataset_config,
            split_config=split_config,
            sampling_config=sampling_config,
            label_source=contract.LABEL_SOURCE_P95P80,
            sampling_source=self.sampling_source,
            teacher=self.teacher,
            epoch=_nonnegative_int(epoch, name="epoch"),
            global_step=self.global_step,
            metrics=metrics,
            seed=self.seed,
            anchor_hash=self.anchor_hash,
            calibration_policy=calibration_policy,
            extra_metadata=self._extra_metadata(extra_metadata),
        )

    def load_checkpoint(
        self,
        path: pathlib.Path,
        *,
        dataset_config: Mapping[str, Any],
        split_config: Mapping[str, Any],
        sampling_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._move_models("cpu")
        result = contract.load_checkpoint(
            path,
            self.student,
            self.optimizer,
            self.scheduler,
            teacher=self.teacher,
            expected_mode=self.mode,
            expected_feature_schema=contract.FEATURE_SCHEMA_7,
            expected_dataset_config=dataset_config,
            expected_split_config=split_config,
            expected_sampling_config=sampling_config,
            expected_label_source=contract.LABEL_SOURCE_P95P80,
            expected_sampling_source=self.sampling_source,
            expected_anchor_hash=self.anchor_hash,
            restore_rng=True,
            load_mode="resume",
        )
        if result["anchor_hash"] != self.anchor_hash:
            raise ValueError("Hybrid checkpoint anchor hash 与 qualified artifact 不一致")
        extra = result.get("extra_metadata", {})
        if extra.get("method_mode") != self.mode:
            raise ValueError("Hybrid checkpoint method_mode 不一致")
        self.global_step = _nonnegative_int(result["global_step"], name="global_step")
        self.seed = _nonnegative_int(result["seed"], name="seed")
        self.teacher.eval()
        return result


class W1HybridTrainer(_HybridTrainer):
    """Public W1-Hybrid trainer facade."""

    def __init__(self, student: nn.Module, optimizer: Any, **kwargs: Any) -> None:
        super().__init__(
            student, optimizer, mode=W1_HYBRID_METHOD, **kwargs
        )


class W2OnHybridTrainer(_HybridTrainer):
    """Public W2-on-Hybrid trainer facade."""

    def __init__(self, student: nn.Module, optimizer: Any, **kwargs: Any) -> None:
        super().__init__(
            student, optimizer, mode=W2_ON_HYBRID_METHOD, **kwargs
        )


def _checkpoint_readback(
    path: pathlib.Path,
    *,
    expected_mode: str,
    anchor_hash: str | None,
) -> dict[str, Any]:
    metadata = contract.checkpoint_metadata(path)
    blob = torch.load(str(path), map_location="cpu", weights_only=True)
    if metadata["mode"] != contract.canonical_mode(expected_mode):
        raise ValueError("closure checkpoint mode readback 不一致")
    if not isinstance(blob.get("student"), Mapping):
        raise ValueError("closure checkpoint 缺少 student state")
    if not isinstance(blob.get("teacher"), Mapping) or not isinstance(blob.get("ema_teacher"), Mapping):
        raise ValueError("closure checkpoint 缺少 student/EMA teacher state")
    if not isinstance(blob.get("optimizer"), Mapping) or not blob["optimizer"]:
        raise ValueError("closure checkpoint 缺少 optimizer state")
    if not isinstance(blob.get("scheduler"), Mapping) or not blob["scheduler"]:
        raise ValueError("closure checkpoint 缺少 scheduler state")
    if anchor_hash is not None and metadata.get("anchor_hash") != anchor_hash:
        raise ValueError("closure checkpoint anchor hash readback 不一致")
    extra = metadata.get("extra_metadata", {})
    required_extra = (
        "method_mode", "protocol_version", "sparse_anchor_identity",
        "source_weight_policy", "uncertainty_settings",
    )
    missing = [key for key in required_extra if key not in extra]
    if missing:
        raise ValueError(f"closure checkpoint extra metadata 缺少 {missing}")
    return {
        "student": True,
        "ema_teacher": True,
        "optimizer": True,
        "scheduler": True,
        "epoch": int(metadata["epoch"]),
        "global_step": int(metadata["global_step"]),
        "seed": int(metadata["seed"]),
        "mode": metadata["mode"],
        "protocol_version": extra["protocol_version"],
        "sparse_anchor_identity": copy.deepcopy(extra["sparse_anchor_identity"]),
        "source_weight_policy": copy.deepcopy(extra["source_weight_policy"]),
        "uncertainty_settings": copy.deepcopy(extra["uncertainty_settings"]),
    }


def _evaluate_calibration_model(
    model: nn.Module,
    calibration_batches: Sequence[CalibrationBatch],
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in calibration_batches:
            predictions = _forward_model(model, batch)
            expected = tuple(int(value) for value in batch.labels.shape)
            predictions = _as_prediction_tensor(
                predictions, expected, name="calibration_predictions"
            ).detach().cpu().numpy()
            known_valid = batch.known_mask & batch.valid_fluid
            rows.append({
                "dataset_name": batch.dataset_name,
                "split_name": "calibration",
                "source": "percentile_calibration_reference",
                "known_valid_fluid_count": int(np.count_nonzero(known_valid)),
                "predicted_positive_count": int(np.count_nonzero(predictions[known_valid] >= 0.5)),
                "record_fingerprint": _json_hash({
                    "dataset_name": batch.dataset_name,
                    "labels": batch.labels,
                    "known_mask": batch.known_mask,
                    "valid_fluid": batch.valid_fluid,
                    "predictions": predictions,
                }),
            })
    return rows


def _evaluate_w2_calibration_model(
    trainer: _HybridTrainer,
    calibration_batches: Sequence[CalibrationBatch],
    *,
    variance_gate: float,
) -> dict[str, Any]:
    trainer.teacher.train()
    records = []
    for batch in calibration_batches:
        with torch.no_grad():
            views = tuple(_forward_model(trainer.teacher, batch) for _ in range(3))
        statistics = w2.compute_w2_statistics(views)
        mean = statistics.mean_probability.detach().cpu().numpy()
        variance = statistics.predictive_variance.detach().cpu().numpy()
        records.append(evaluate_w2_selective(
            mean_probability=mean,
            predictive_variance=variance,
            labels=batch.labels,
            known_mask=batch.known_mask,
            valid_fluid=batch.valid_fluid,
            prediction_threshold=0.5,
            variance_gate=variance_gate,
            split_name="calibration",
        ) | {
            "dataset_name": batch.dataset_name,
            "view_count": 3,
            "record_fingerprint": _json_hash({
                "dataset_name": batch.dataset_name,
                "labels": batch.labels,
                "known_mask": batch.known_mask,
                "valid_fluid": batch.valid_fluid,
                "mean_probability": mean,
                "predictive_variance": variance,
            }),
        })
    trainer.teacher.eval()
    aggregate_keys = (
        "known_valid_fluid_count", "accepted_count", "accepted_positive_count",
        "accepted_negative_count", "rejected_count", "accepted_true_positive",
        "accepted_false_positive", "accepted_false_negative", "accepted_true_negative",
    )
    aggregate = {
        key: int(sum(int(row[key]) for row in records)) for key in aggregate_keys
    }
    denominator = aggregate["accepted_count"]
    numerator = aggregate["accepted_false_positive"] + aggregate["accepted_false_negative"]
    aggregate.update({
        "split_name": "calibration",
        "source": "percentile_calibration_reference",
        "view_count": 3,
        "prediction_threshold": 0.5,
        "variance_gate": float(variance_gate),
        "selective_risk_numerator": numerator,
        "selective_risk_denominator": denominator,
        "selective_risk": float(numerator / denominator) if denominator else None,
        "selective_risk_status": "defined" if denominator else "undefined",
        "operating_point_pass": bool(denominator),
        "rejected_excluded_from_confusion": True,
        "rejected_not_negative": True,
        "records": records,
    })
    return aggregate


def _method_metadata(
    identity: Mapping[str, Any],
    *,
    method: str,
    epochs: int,
    checkpoint_path: pathlib.Path,
    resume: Mapping[str, Any],
    checkpoint_readback: Mapping[str, Any],
    train_history: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "method": method,
        "method_role": "closure_only_no_performance_claim",
        "epochs": int(epochs),
        "train_history": [dict(row) for row in train_history],
        "checkpoint": str(checkpoint_path.name),
        "checkpoint_readback": dict(checkpoint_readback),
        "resume": dict(resume),
        "calibration": dict(calibration),
        "evaluation_scope": "calibration",
        "qualified_anchor": {
            "artifact_fingerprint": identity["artifact_fingerprint"],
            "qualification_report_fingerprint": identity["qualification_report_fingerprint"],
            "source_fingerprints": list(identity["source_fingerprints"]),
            "protocol_version": identity["protocol_version"],
            "construction": identity["construction"],
        },
    }


def _w1p_component_snapshot(
    trainer: w1_p.W1PTrainer,
    hybrid_batches: Sequence[HybridBatch],
    w1p_batches: Sequence[w1_p.W1PBatch],
    *,
    epoch: int,
) -> dict[str, dict[str, float | int]]:
    """Capture W1-P's source-aware loss decomposition at epoch end.

    ``W1PTrainer`` is the existing, public training seam and deliberately
    reports its historical scalar statistics.  Ticket 06 additionally needs
    the common eight-component audit schema.  This snapshot replays the same
    masked BCE, confidence-pseudo, and consistency masks against the actual
    post-step student/EMA state without changing the W1-P trainer contract.
    Haller categories are present as explicit zero-consumption rows because
    Haller is provenance only for W1-P, never its formal loss source.
    """
    epoch = _nonnegative_int(epoch, name="epoch")
    if len(hybrid_batches) != len(w1p_batches) or not hybrid_batches:
        raise ValueError("W1-P component snapshot 的 batch 对必须非空且等长")
    if not isinstance(trainer.config, w1_p.W1PConfig):
        raise TypeError("W1-P component snapshot 必须使用 W1PConfig")

    component_names = (
        "percentile_positive", "percentile_negative", "haller_anchor",
        "agreement", "conflict", "unknown_unlabeled", "pseudo_accepted",
        "consistency", "solid", "invalid", "extractor_failure",
    )
    accum = {
        name: {
            "eligible_count": 0,
            "accepted_count": 0,
            "weight_sum": 0.0,
            "contribution_sum": 0.0,
        }
        for name in component_names
    }
    hard_numerator = {
        "percentile_positive": 0.0,
        "percentile_negative": 0.0,
    }
    hard_denominator = 0
    batch_count = len(w1p_batches)
    trainer.student.eval()
    trainer.teacher.eval()

    with torch.no_grad():
        for hybrid_batch, w1p_batch in zip(hybrid_batches, w1p_batches):
            batch = w1p_batch.to("cpu")
            expected_shape = tuple(int(value) for value in batch.labels.shape)
            student = _as_prediction_tensor(
                _forward_model(trainer.student, batch), expected_shape,
                name="W1-P closure student predictions",
            )
            teacher = _as_prediction_tensor(
                _forward_model(trainer.teacher, batch), expected_shape,
                name="W1-P closure teacher predictions",
            )
            labels = torch.as_tensor(batch.labels, dtype=student.dtype)
            known = torch.as_tensor(batch.label_mask, dtype=torch.bool)
            unknown = torch.as_tensor(batch.unknown_mask, dtype=torch.bool)
            excluded = torch.as_tensor(batch.solid_mask, dtype=torch.bool)
            if bool((excluded & known).any()) or bool((excluded & ~unknown).any()):
                raise ValueError("W1-P component snapshot 的 excluded mask 非法")
            valid_unknown = unknown & ~excluded
            percentile_positive = known & (labels >= 0.5)
            percentile_negative = known & ~percentile_positive
            element_loss = F.binary_cross_entropy(
                student, labels, reduction="none"
            )
            hard_count = int(known.sum().item())
            hard_denominator += hard_count
            for name, mask in (
                ("percentile_positive", percentile_positive),
                ("percentile_negative", percentile_negative),
            ):
                count = int(mask.sum().item())
                accum[name]["eligible_count"] += count
                accum[name]["accepted_count"] += count
                accum[name]["weight_sum"] += float(count)
                hard_numerator[name] += float(element_loss[mask].sum().item())
            unknown_count = int(valid_unknown.sum().item())
            accum["unknown_unlabeled"]["eligible_count"] += unknown_count
            for name, mask in (
                ("solid", torch.as_tensor(hybrid_batch.solid_mask)),
                ("invalid", torch.as_tensor(hybrid_batch.invalid_support)),
                ("extractor_failure", torch.as_tensor(hybrid_batch.failed_frame_mask)),
            ):
                count = int(mask.sum().item())
                accum[name]["eligible_count"] += count

            high = torch.as_tensor(
                trainer.config.pseudo_high, dtype=teacher.dtype
            )
            low = torch.as_tensor(
                trainer.config.pseudo_low, dtype=teacher.dtype
            )
            confident = (teacher >= high) | (teacher <= low)
            pseudo_mask = valid_unknown & confident
            pseudo_targets = (teacher >= 0.5).to(dtype=student.dtype)
            pseudo_count = int(pseudo_mask.sum().item())
            accum["pseudo_accepted"]["eligible_count"] += unknown_count
            accum["pseudo_accepted"]["accepted_count"] += pseudo_count
            consistency_count = unknown_count
            accum["consistency"]["eligible_count"] += consistency_count
            accum["consistency"]["accepted_count"] += consistency_count
            ramp = w1_p.ramp_up_weight(
                epoch, trainer.config.ramp_up_epochs
            )
            if pseudo_count:
                pseudo_loss = F.binary_cross_entropy(
                    student[pseudo_mask], pseudo_targets[pseudo_mask]
                )
                pseudo_coefficient = ramp * trainer.config.pseudo_weight
                accum["pseudo_accepted"]["weight_sum"] += pseudo_coefficient
                accum["pseudo_accepted"]["contribution_sum"] += float(
                    pseudo_coefficient * pseudo_loss.item()
                )
            if consistency_count:
                consistency_loss = F.mse_loss(
                    student[valid_unknown], teacher[valid_unknown]
                )
                consistency_coefficient = ramp * trainer.config.consistency_weight
                accum["consistency"]["weight_sum"] += consistency_coefficient
                accum["consistency"]["contribution_sum"] += float(
                    consistency_coefficient * consistency_loss.item()
                )

    result: dict[str, dict[str, float | int]] = {}
    for name in component_names:
        row = accum[name]
        eligible = int(row["eligible_count"])
        accepted = int(row["accepted_count"])
        if name in hard_numerator:
            contribution = (
                hard_numerator[name] / hard_denominator
                if hard_denominator else 0.0
            )
            mean_weight = 1.0 if eligible else 0.0
        elif name in {"pseudo_accepted", "consistency"}:
            contribution = float(row["contribution_sum"] / batch_count)
            mean_weight = float(row["weight_sum"] / batch_count)
        else:
            contribution = 0.0
            mean_weight = 0.0
        result[name] = {
            "eligible_count": eligible,
            "accepted_count": accepted,
            "mean_weight": mean_weight,
            "contribution": contribution,
        }
    return result


def _run_one_method(
    method: str,
    *,
    identity: Mapping[str, Any],
    train_batches: Sequence[HybridBatch],
    calibration_batches: Sequence[CalibrationBatch],
    output_dir: pathlib.Path,
    model_factory: Callable[[], nn.Module],
    epochs: int,
    seed: int,
    sampling_source: str,
    variance_gate: float,
) -> dict[str, Any]:
    method_seed = seed + CLOSURE_METHODS.index(method)
    _seed_everything(method_seed)
    dataset_config = {
        "dataset_name": "ticket06_cpu_fixture",
        "qualified_anchor_artifact_fingerprint": identity["artifact_fingerprint"],
        "qualified_anchor_source_fingerprints": list(identity["source_fingerprints"]),
    }
    split_config = {
        "split_name": "train",
        "frame_range": [0, 2],
        "calibration_scope": "separate_calibration_batches_only",
    }
    sampling_config = {
        "t_win": 3,
        "window_step": 1,
        "sampling_source": sampling_source,
    }
    calibration_policy = {
        "source": "percentile_calibration_reference",
        "scope": "calibration_only",
        "global_gate": method == W2_ON_HYBRID_METHOD,
    }
    model = model_factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    if method == W1_P_METHOD:
        trainer: Any = w1_p.W1PTrainer(
            model, optimizer, sampling_source=sampling_source,
            config=w1_p.W1PConfig(ramp_up_epochs=2),
            scheduler=scheduler, seed=method_seed,
        )
        method_batches = [batch.to_w1_p_batch() for batch in train_batches]
    else:
        config = (
            w2.W2Config(ramp_up_epochs=2, variance_gate=variance_gate)
            if method == W2_ON_HYBRID_METHOD
            else w1_p.W1PConfig(ramp_up_epochs=2)
        )
        trainer = (
            W2OnHybridTrainer if method == W2_ON_HYBRID_METHOD else W1HybridTrainer
        )(
            model,
            optimizer,
            sampling_source=sampling_source,
            anchor_hash=identity["artifact_fingerprint"],
            anchor_metadata=identity,
            scheduler=scheduler,
            config=config,
            seed=method_seed,
        )
        method_batches = list(train_batches)
    split_epoch = max(1, epochs // 2)
    history: list[dict[str, Any]] = []
    for epoch in range(split_epoch):
        epoch_stats = trainer.run_epoch(
            method_batches, epoch=epoch, device="cpu"
        )
        if method == W1_P_METHOD:
            epoch_stats["loss_components"] = _w1p_component_snapshot(
                trainer, train_batches, method_batches, epoch=epoch
            )
        history.append(epoch_stats)
        scheduler.step()
    checkpoint_path = output_dir / f"{method.lower().replace('-', '_')}.pth"
    extra = {
        "ticket": "06-minimal-training-closure",
        "method_mode": method,
        "protocol_version": sparse.PROTOCOL_VERSION,
        "sparse_anchor_identity": dict(identity),
        "percentile_evidence_fingerprint": _json_hash({
            "states": [batch.fusion.percentile_state for batch in train_batches],
            "source": contract.LABEL_SOURCE_P95P80,
        }),
        "source_weight_policy": {
            "percentile_positive": 0.5,
            "percentile_negative": 0.5,
            "haller_positive_anchor": 1.0,
            "agreement": 1.0,
            "conflict": 0.0,
            "uncovered_is_not_negative": True,
            "haller_hard_negative_state": None,
        },
        "uncertainty_settings": (
            {
                "view_count": 3,
                "variance_gate": variance_gate,
                "pseudo_high": 0.90,
                "pseudo_low": 0.10,
                "entropy_is_diagnostic_only": True,
            }
            if method == W2_ON_HYBRID_METHOD else {
                "view_count": 1,
                "variance_gate": None,
                "pseudo_high": 0.90,
                "pseudo_low": 0.10,
                "entropy_is_diagnostic_only": None,
            }
        ),
    }
    trainer.save_checkpoint(
        checkpoint_path,
        epoch=split_epoch - 1,
        dataset_config=dataset_config,
        split_config=split_config,
        sampling_config=sampling_config,
        metrics=history[-1],
        calibration_policy=calibration_policy,
        extra_metadata=extra,
    )
    resume_model = model_factory()
    resume_optimizer = torch.optim.AdamW(resume_model.parameters(), lr=0.02)
    resume_scheduler = torch.optim.lr_scheduler.StepLR(
        resume_optimizer, step_size=1, gamma=0.95
    )
    if method == W1_P_METHOD:
        resumed: Any = w1_p.W1PTrainer(
            resume_model, resume_optimizer, sampling_source=sampling_source,
            config=w1_p.W1PConfig(ramp_up_epochs=2),
            scheduler=resume_scheduler, seed=method_seed,
        )
        resume_result = resumed.load_checkpoint(
            checkpoint_path,
            expected_dataset_config=dataset_config,
            expected_split_config=split_config,
            expected_sampling_config=sampling_config,
            device="cpu",
            restore_rng=True,
        )
    else:
        config = (
            w2.W2Config(ramp_up_epochs=2, variance_gate=variance_gate)
            if method == W2_ON_HYBRID_METHOD
            else w1_p.W1PConfig(ramp_up_epochs=2)
        )
        resumed = (
            W2OnHybridTrainer if method == W2_ON_HYBRID_METHOD else W1HybridTrainer
        )(
            resume_model,
            resume_optimizer,
            sampling_source=sampling_source,
            anchor_hash=identity["artifact_fingerprint"],
            anchor_metadata=identity,
            scheduler=resume_scheduler,
            config=config,
            seed=method_seed,
        )
        resume_result = resumed.load_checkpoint(
            checkpoint_path,
            dataset_config=dataset_config,
            split_config=split_config,
            sampling_config=sampling_config,
        )
    resume_history = []
    for epoch in range(int(resume_result["start_epoch"]), epochs):
        epoch_stats = resumed.run_epoch(
            method_batches, epoch=epoch, device="cpu"
        )
        if method == W1_P_METHOD:
            epoch_stats["loss_components"] = _w1p_component_snapshot(
                resumed, train_batches, method_batches, epoch=epoch
            )
        resume_history.append(epoch_stats)
        resume_scheduler.step()
    history.extend(resume_history)
    resumed.save_checkpoint(
        checkpoint_path,
        epoch=epochs - 1,
        dataset_config=dataset_config,
        split_config=split_config,
        sampling_config=sampling_config,
        metrics=history[-1],
        calibration_policy=calibration_policy,
        extra_metadata=extra,
    )
    if method == W2_ON_HYBRID_METHOD:
        calibration = _evaluate_w2_calibration_model(
            resumed, calibration_batches, variance_gate=variance_gate
        )
    else:
        calibration = {
            "split_name": "calibration",
            "source": "percentile_calibration_reference",
            "records": _evaluate_calibration_model(resumed.student, calibration_batches),
        }
    checkpoint_readback = _checkpoint_readback(
        checkpoint_path,
        expected_mode=method,
        anchor_hash=(None if method == W1_P_METHOD else identity["artifact_fingerprint"]),
    )
    checkpoint_readback["percentile_evidence_fingerprint"] = extra[
        "percentile_evidence_fingerprint"
    ]
    return _method_metadata(
        identity,
        method=method,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        resume={
            "resumed": True,
            "checkpoint_epoch": int(resume_result["epoch"]),
            "start_epoch": int(resume_result["start_epoch"]),
            "global_step_after_load": int(resume_result["global_step"]),
            "rng_restored": bool(resume_result.get("rng_restored", True)),
        },
        checkpoint_readback=checkpoint_readback,
        train_history=history,
        calibration=calibration,
    )


def run_minimal_training_closure(
    qualified_bundle_dir: str | pathlib.Path,
    *,
    output_dir: str | pathlib.Path,
    train_batches: Sequence[HybridBatch] | None = None,
    calibration_batches: Sequence[CalibrationBatch] | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    epochs: int = 2,
    seed: int = 17,
    sampling_source: str = contract.LABEL_SOURCE_P95P80,
    variance_gate: float = 0.05,
) -> dict[str, Any]:
    """Run W1-P/W1-Hybrid/W2-on-Hybrid train/save/resume/calibration closure."""
    epochs = _nonnegative_int(epochs, name="epochs")
    if not 2 <= epochs <= 5:
        raise ValueError("Ticket 06 closure epochs 必须位于 [2,5]")
    seed = _nonnegative_int(seed, name="seed")
    sampling_source = contract.validate_label_source(sampling_source)
    if sampling_source not in {
        contract.LABEL_SOURCE_P95P80,
        contract.LABEL_SOURCE_LEGACY_P85,
    }:
        raise ValueError("closure sampling_source 不在 train-only allowed set")
    variance_gate = _probability(variance_gate, name="variance_gate")
    identity, _bundle = load_qualified_anchor_identity(qualified_bundle_dir)
    if train_batches is None or calibration_batches is None:
        fixture_identity, train_batches, calibration_batches = build_cpu_closure_inputs(
            qualified_bundle_dir, seed=seed
        )
        if fixture_identity != identity:
            raise ValueError("qualified bundle identity 在 fixture construction 中发生漂移")
    if not train_batches or not calibration_batches:
        raise ValueError("closure train/calibration batches 不能为空")
    train_batches = list(train_batches)
    calibration_batches = list(calibration_batches)
    for batch in train_batches:
        if not isinstance(batch, HybridBatch):
            raise TypeError("closure train_batches 必须全部是 HybridBatch")
        if batch.anchor_hash != identity["artifact_fingerprint"]:
            raise ValueError("train batch 未绑定当前 qualified sparse-anchor fingerprint")
        if batch.contract_batch.split_name != "train":
            raise ValueError("closure train batch split 不是 train")
    for batch in calibration_batches:
        if not isinstance(batch, CalibrationBatch):
            raise TypeError("closure calibration_batches 必须全部是 CalibrationBatch")
        if batch.split_name != "calibration":
            raise PermissionError("closure calibration batch split 不是 calibration")
    factory = _DefaultClosureModel if model_factory is None else model_factory
    if not callable(factory):
        raise TypeError("model_factory 必须是 callable")
    out = pathlib.Path(output_dir)
    if out.exists():
        raise FileExistsError(f"拒绝覆盖已有 Ticket 06 closure artifact: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(prefix=".ticket06-closure-", dir=out.parent))
    try:
        method_rows = []
        for method in CLOSURE_METHODS:
            method_rows.append(_run_one_method(
                method,
                identity=identity,
                train_batches=train_batches,
                calibration_batches=calibration_batches,
                output_dir=stage,
                model_factory=factory,
                epochs=epochs,
                seed=seed,
                sampling_source=sampling_source,
                variance_gate=variance_gate,
            ))
        report: dict[str, Any] = {
            "artifact_type": CLOSURE_ARTIFACT_TYPE,
            "schema_version": TICKET06_SCHEMA_VERSION,
            "status": "closure_pass",
            "evidence_role": "closure_smoke_semantic_only",
            "performance_claim": False,
            "protocol_version": sparse.PROTOCOL_VERSION,
            "methods": method_rows,
            "qualified_anchor": identity,
            "train_calibration_boundary": {
                "training_split": "train",
                "calibration_split": "calibration",
                "final_test_access": "not_read",
                "final_test_labels": "not_read",
                "final_test_coverage": "not_read",
                "final_test_agreement": "not_read",
                "final_test_conflict": "not_read",
                "final_test_model_metrics": "not_read",
                "frame_1300": "excluded",
            },
            "access_guard": {
                "final_test_relationships": "not_read",
                "frame_1300": "excluded",
                "guard_order": "qualified_identity_checked_before_fixture_field_read",
            },
            "closure_config": {
                "epochs": epochs,
                "seed": seed,
                "sampling_source": sampling_source,
                "variance_gate": variance_gate,
                "method_order": list(CLOSURE_METHODS),
            },
            "no_ranking": True,
            "no_final_performance": True,
        }
        report["report_fingerprint"] = _json_hash(report)
        report_path = stage / "closure_report.json"
        integrity_path = stage / "integrity.json"
        _write_json(report_path, report)
        files = {
            "closure_report.json": _file_sha256(report_path),
        }
        for method in method_rows:
            checkpoint = stage / method["checkpoint"]
            files[method["checkpoint"]] = _file_sha256(checkpoint)
        integrity = {
            "artifact_type": CLOSURE_INTEGRITY_TYPE,
            "schema_version": f"{CLOSURE_INTEGRITY_TYPE}-v1",
            "protocol_version": sparse.PROTOCOL_VERSION,
            "report_fingerprint": report["report_fingerprint"],
            "file_hashes": files,
            "final_test_relationships": "not_read",
            "frame_1300": "excluded",
        }
        integrity["integrity_fingerprint"] = _json_hash(integrity)
        _write_json(integrity_path, integrity)
        shutil.move(str(stage), str(out))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return read_closure_artifact(out)["report"]


def read_closure_artifact(output_dir: str | pathlib.Path) -> dict[str, Any]:
    """Read back report, sidecar, checkpoints, fingerprints, and guards."""
    out = pathlib.Path(output_dir)
    report_path = out / "closure_report.json"
    integrity_path = out / "integrity.json"
    if not report_path.exists() or not integrity_path.exists():
        raise ValueError("closure artifact 缺少 closure_report.json/integrity.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if report.get("artifact_type") != CLOSURE_ARTIFACT_TYPE:
        raise ValueError("closure report artifact_type 不匹配")
    if report.get("schema_version") != TICKET06_SCHEMA_VERSION:
        raise ValueError("closure report schema_version 不匹配")
    if report.get("status") != "closure_pass":
        raise ValueError("closure report 未标记为 closure_pass")
    if report.get("performance_claim") is not False:
        raise ValueError("closure report 不得包含 performance claim")
    if report.get("no_ranking") is not True or report.get("no_final_performance") is not True:
        raise ValueError("closure report 必须明确禁止 ranking/final performance")
    fingerprint = report.get("report_fingerprint")
    payload = dict(report)
    payload.pop("report_fingerprint", None)
    if not isinstance(fingerprint, str) or _json_hash(payload) != fingerprint:
        raise ValueError("closure report fingerprint 校验失败")
    if integrity.get("report_fingerprint") != fingerprint:
        raise ValueError("closure integrity/report fingerprint 不一致")
    integrity_payload = dict(integrity)
    integrity_declared = integrity_payload.pop("integrity_fingerprint", None)
    if not isinstance(integrity_declared, str) or _json_hash(integrity_payload) != integrity_declared:
        raise ValueError("closure integrity fingerprint 校验失败")
    if integrity.get("final_test_relationships") != "not_read":
        raise PermissionError("closure integrity final-test guard 未封存")
    if integrity.get("frame_1300") != "excluded":
        raise ValueError("closure integrity 未排除 frame 1300")
    for relative, expected_hash in integrity.get("file_hashes", {}).items():
        path = out / relative
        if not path.exists() or _file_sha256(path) != expected_hash:
            raise ValueError(f"closure file hash 校验失败: {relative}")
    if report.get("qualified_anchor", {}).get("artifact_state") != "qualified":
        raise PermissionError("closure report 未绑定 qualified artifact")
    if report.get("access_guard", {}).get("final_test_relationships") != "not_read":
        raise PermissionError("closure report final-test guard 未封存")
    if report.get("access_guard", {}).get("frame_1300") != "excluded":
        raise ValueError("closure report 未排除 frame 1300")
    boundary = report.get("train_calibration_boundary", {})
    required_boundary = {
        "training_split": "train",
        "calibration_split": "calibration",
        "final_test_access": "not_read",
        "final_test_labels": "not_read",
        "final_test_coverage": "not_read",
        "final_test_agreement": "not_read",
        "final_test_conflict": "not_read",
        "final_test_model_metrics": "not_read",
        "frame_1300": "excluded",
    }
    if any(boundary.get(key) != value for key, value in required_boundary.items()):
        raise PermissionError("closure report train/calibration boundary 未封存")
    methods = report.get("methods", [])
    if tuple(row.get("method") for row in methods) != CLOSURE_METHODS:
        raise ValueError("closure methods 顺序/集合不完整")
    required_components = {
        "percentile_positive", "percentile_negative", "haller_anchor",
        "agreement", "conflict", "unknown_unlabeled", "pseudo_accepted",
        "consistency",
    }
    required_component_fields = {
        "eligible_count", "accepted_count", "mean_weight", "contribution",
    }
    checkpoint_readbacks = {}
    for row in methods:
        method = row["method"]
        epochs = _nonnegative_int(row.get("epochs"), name=f"{method}.epochs")
        if not 2 <= epochs <= 5:
            raise ValueError(f"{method} closure epochs 不在 [2,5]")
        history = row.get("train_history", [])
        if not isinstance(history, list) or len(history) != epochs:
            raise ValueError(f"{method} closure train_history 不完整")
        for epoch_row in history:
            components = epoch_row.get("loss_components")
            if not isinstance(components, Mapping) or not required_components.issubset(components):
                raise ValueError(f"{method} loss_components 缺少 Ticket 06 类别")
            for component in required_components:
                value = components[component]
                if not isinstance(value, Mapping) or not required_component_fields.issubset(value):
                    raise ValueError(f"{method} loss_components.{component} schema 不完整")
                if any(int(value[field]) < 0 for field in ("eligible_count", "accepted_count")):
                    raise ValueError(f"{method} loss_components.{component} count 非法")
            conflict = components["conflict"]
            if (
                float(conflict["accepted_count"]) != 0.0
                or float(conflict["mean_weight"]) != 0.0
                or float(conflict["contribution"]) != 0.0
            ):
                raise ValueError(f"{method} conflict 必须保持零 accepted/weight/contribution")
        if row.get("checkpoint_readback", {}).get("source_weight_policy", {}).get("conflict") != 0.0:
            raise ValueError(f"{method} checkpoint conflict policy 不是 0")
        checkpoint_path = out / row["checkpoint"]
        checkpoint_readbacks[method] = _checkpoint_readback(
            checkpoint_path,
            expected_mode=method,
            anchor_hash=(
                None if method == W1_P_METHOD
                else report["qualified_anchor"]["artifact_fingerprint"]
            ),
        )
        if not row.get("resume", {}).get("resumed"):
            raise ValueError(f"{method} closure 没有 resume evidence")
        if row.get("evaluation_scope") != "calibration":
            raise ValueError(f"{method} closure evaluation scope 不是 calibration")
        if method == W2_ON_HYBRID_METHOD:
            calibration = row.get("calibration", {})
            denominator = int(calibration.get("selective_risk_denominator", -1))
            numerator = int(calibration.get("selective_risk_numerator", -1))
            if denominator < 0 or numerator < 0:
                raise ValueError("W2 calibration selective-risk counts 缺失")
            if denominator == 0:
                if (
                    calibration.get("selective_risk") is not None
                    or calibration.get("selective_risk_status") != "undefined"
                    or calibration.get("operating_point_pass") is not False
                ):
                    raise ValueError("W2 零分母必须是 undefined 且 operating point fail")
            elif calibration.get("selective_risk") != numerator / denominator:
                raise ValueError("W2 selective risk 未按 accepted FP+FN/accepted known valid-fluid 计算")
            if (
                calibration.get("rejected_excluded_from_confusion") is not True
                or calibration.get("rejected_not_negative") is not True
            ):
                raise ValueError("W2 rejected 计数语义未封存")
    return {
        "schema_verified": True,
        "integrity_verified": True,
        "file_hashes_verified": True,
        "report": report,
        "integrity": integrity,
        "checkpoint_readbacks": checkpoint_readbacks,
    }


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Ticket 06 minimal training closure")
    parser.add_argument("--qualified-bundle", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--variance-gate", type=float, default=0.05)
    args = parser.parse_args()
    report = run_minimal_training_closure(
        args.qualified_bundle,
        output_dir=args.out_dir,
        epochs=args.epochs,
        seed=args.seed,
        variance_gate=args.variance_gate,
    )
    print(json.dumps({
        "status": report["status"],
        "schema_version": report["schema_version"],
        "report_fingerprint": report["report_fingerprint"],
        "qualified_anchor_fingerprint": report["qualified_anchor"]["artifact_fingerprint"],
        "methods": [row["method"] for row in report["methods"]],
        "evaluation_scope": sorted({row["evaluation_scope"] for row in report["methods"]}),
    }, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "CalibrationBatch",
    "CLOSURE_METHODS",
    "HybridBatch",
    "HybridFusionResult",
    "PERCENTILE_NEGATIVE",
    "PERCENTILE_POSITIVE",
    "PERCENTILE_UNKNOWN",
    "W1HybridTrainer",
    "W2OnHybridTrainer",
    "build_cpu_closure_inputs",
    "build_w1_hybrid_batch",
    "compute_w1_hybrid_loss",
    "compute_w2_on_hybrid_loss",
    "evaluate_w2_selective",
    "fuse_w1_hybrid_targets",
    "load_qualified_anchor_identity",
    "read_closure_artifact",
    "run_minimal_training_closure",
]


if __name__ == "__main__":
    raise SystemExit(_cli())
