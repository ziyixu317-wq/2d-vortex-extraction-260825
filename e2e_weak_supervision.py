"""Primary end-to-end pilot orchestration for the weak-supervision feature.

The preceding tickets own the method-specific batch/loss/trainers.  This module
owns only their shared experiment seam: six methods are instantiated from
scratch, trained with explicit train batches, calibrated on calibration Haller
GT, checkpointed, round-tripped, and evaluated once on explicit test Haller GT.

The public :class:`PilotMethod` adapter keeps the orchestration independent of
the vendor model.  ``PilotMethod.from_trainer`` connects the W1-H/W1-P/W2/W3
trainer APIs without changing them; B0/B1 callers can provide the same five
callbacks around their own model/criterion loop.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import pathlib
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

import evaluation_report
import weak_supervision_contract as contract


PILOT_METHOD_ORDER = (
    contract.MODE_B0,
    contract.MODE_B1,
    contract.MODE_W1_P,
    contract.MODE_W1_H,
    contract.MODE_W2,
    contract.MODE_W3,
)
PILOT_DATASET_NAMES = evaluation_report.VALID_DATASET_NAMES
PILOT_RAMP_UP_EPOCHS = 12
PILOT_EPOCHS = 50
PILOT_SEED = 0
HALLER_TRAIN_MODES = frozenset({
    contract.MODE_W1_H,
    contract.MODE_W2,
    contract.MODE_W3,
})
FORMAL_LABEL_SOURCES = {
    contract.MODE_B0: contract.LABEL_SOURCE_LEGACY_P85,
    contract.MODE_B1: contract.LABEL_SOURCE_LEGACY_P85,
    contract.MODE_W1_P: contract.LABEL_SOURCE_LOCAL_P90_P60,
    contract.MODE_W1_H: contract.LABEL_SOURCE_HALLER_TRAIN,
    contract.MODE_W2: contract.LABEL_SOURCE_HALLER_TRAIN,
    contract.MODE_W3: contract.LABEL_SOURCE_HALLER_TRAIN,
}


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _strict_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是非负整数")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是非负整数") from exc
    if converted != value or converted < 0:
        raise ValueError(f"{name} 必须是非负整数")
    return converted


def _strict_positive_int(value: Any, *, name: str) -> int:
    converted = _strict_nonnegative_int(value, name=name)
    if converted <= 0:
        raise ValueError(f"{name} 必须是正整数")
    return converted


def _strict_probability(value: Any, *, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是 [0,1] 内的有限数") from exc
    if not np.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} 必须是 [0,1] 内的有限数")
    return converted


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(child) for child in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return _jsonable(value.detach().cpu().numpy())
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _contains_exact(value: Any, target: str) -> bool:
    if isinstance(value, str):
        return value.strip() == target
    if isinstance(value, np.ndarray):
        return _contains_exact(value.tolist(), target)
    if isinstance(value, np.generic):
        return _contains_exact(value.item(), target)
    if isinstance(value, Mapping):
        return any(
            _contains_exact(key, target) or _contains_exact(child, target)
            for key, child in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_exact(child, target) for child in value)
    return False


def _test_only_keys(value: Any) -> set[str]:
    # This names an audit-policy parameter for invalid calibration/test frames;
    # it does not carry test labels, metrics, or predictions.  Keep this
    # exception aligned with the contract and W2 provenance guards.
    test_only_key_exceptions = frozenset({
        "failure_fallback_calibration_test",
        "haller_gt_test_artifact_read",
    })
    found: set[str] = set()
    if isinstance(value, np.ndarray):
        return _test_only_keys(value.tolist())
    if isinstance(value, np.generic):
        return _test_only_keys(value.item())
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(".", "_")
            tokens = {token for token in normalized.split("_") if token}
            if (
                normalized not in test_only_key_exceptions
                and (normalized in {"test", "test_gt", "test_label", "test_labels",
                                   "test_metric", "test_metrics", "test_prediction",
                                   "test_predictions", "test_result", "test_results",
                                   "gt_test", "label_test", "metric_test"}
                or normalized.startswith("test_")
                or normalized.endswith("_test")
                or {"gt", "test"}.issubset(tokens)
                or {"label", "test"}.issubset(tokens)
                or {"metric", "test"}.issubset(tokens))
            ):
                found.add(normalized)
            found.update(_test_only_keys(child))
    elif isinstance(value, (tuple, list, set, frozenset)):
        for child in value:
            found.update(_test_only_keys(child))
    return found


def _reject_test_data(
    value: Any,
    *,
    context: str,
    forbidden_sources: Sequence[str] = (contract.LABEL_SOURCE_HALLER_TEST,),
) -> None:
    for source in forbidden_sources:
        if _contains_exact(value, source):
            raise ValueError(
                f"{context} 禁止出现 {source}；"
                "calibration/test Haller GT 不能进入当前 seam"
            )
    names = sorted(_test_only_keys(value))
    if names:
        raise ValueError(f"{context} 禁止 test-only 字段：{names!r}")


def _mapping_or_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _validate_window_provenance(provenance: Any, *, mode: str) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{mode} training batch 必须携带 provenance.window")
    required = (
        "split_name", "frame_start", "frame_end", "split_start", "split_end",
        "t_win", "window_step",
    )

    def validate_one(window: Any, *, ordinal: int) -> dict[str, Any]:
        if not isinstance(window, Mapping):
            raise ValueError(
                f"{mode} window provenance[{ordinal}] 必须是显式 object"
            )
        missing = [field for field in required if field not in window]
        if missing:
            raise ValueError(
                f"{mode} window provenance[{ordinal}] 缺少字段：{missing!r}"
            )
        if window["split_name"] != "train":
            raise ValueError(
                f"{mode} training window 必须属于 split=train，实际 "
                f"{window['split_name']!r}"
            )
        values = {
            field: _strict_nonnegative_int(
                window[field], name=f"window[{ordinal}].{field}"
            )
            for field in (
                "frame_start", "frame_end", "split_start", "split_end",
                "t_win", "window_step",
            )
        }
        if values["t_win"] <= 0 or values["window_step"] <= 0:
            raise ValueError(f"{mode} window t_win/window_step 必须为正整数")
        if values["split_end"] <= values["split_start"]:
            raise ValueError(f"{mode} window split_end 必须大于 split_start")
        if values["frame_start"] < values["split_start"]:
            raise ValueError(f"{mode} window frame_start 越过 train split 起点")
        if values["frame_end"] != values["frame_start"] + values["t_win"]:
            raise ValueError(f"{mode} window 必须满足 frame_end=frame_start+t_win")
        if values["frame_end"] > values["split_end"]:
            raise ValueError(f"{mode} window 跨越 train split 右边界")
        normalized = dict(window)
        normalized.update(values)
        return normalized

    single_window = provenance.get("window")
    if isinstance(single_window, Mapping):
        return validate_one(single_window, ordinal=0)

    # Collated W1-H batches carry one window per source sample.  Keeping the
    # list explicit prevents a mixed-dataset batch from smuggling a scalar
    # split bound over all stores.  ``batches`` is accepted as a compatibility
    # fallback, but every nested item still has to carry its own window.
    windows = provenance.get("windows")
    if windows is None:
        nested = provenance.get("batches")
        if isinstance(nested, (list, tuple)):
            windows = [
                item.get("window") if isinstance(item, Mapping) else None
                for item in nested
            ]
    if not isinstance(windows, (list, tuple)) or not windows:
        raise ValueError(
            f"{mode} training batch 必须携带显式 window provenance；"
            "collated batch 应提供 provenance.windows"
        )
    normalized_windows = [
        validate_one(window, ordinal=index)
        for index, window in enumerate(windows)
    ]
    return {"collated": True, "windows": normalized_windows}


def _validate_haller_anchor_metadata(metadata: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{context} 必须携带 Haller train anchor metadata")
    required = (
        "source", "algorithm_version", "parameter_hash", "input_hash",
        "mask_hash", "parameters", "failure_count", "coverage", "literature",
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"{context} Haller metadata 缺少字段：{missing!r}")
    if metadata["source"] != contract.LABEL_SOURCE_HALLER_TRAIN:
        raise ValueError(f"{context} Haller metadata source 必须是 haller_anchor_train")
    for field in ("algorithm_version", "parameter_hash", "input_hash", "mask_hash"):
        _nonempty_text(metadata[field], name=f"{context}.{field}")
    if not isinstance(metadata["parameters"], Mapping) or not metadata["parameters"]:
        raise ValueError(f"{context}.parameters 必须是非空 object")
    failure_count = _strict_nonnegative_int(metadata["failure_count"], name=f"{context}.failure_count")
    raw_coverage = metadata["coverage"]
    coverage: Any
    if isinstance(raw_coverage, Mapping):
        if not raw_coverage:
            raise ValueError(f"{context}.coverage object 不能为空")
        coverage = copy.deepcopy(dict(raw_coverage))
        # haller_anchors stores the full fluid/known/unknown accounting object;
        # accept that artifact shape while still rejecting malformed numeric
        # counters and fractions at the pilot boundary.
        for field, value in coverage.items():
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{context}.coverage.{field} 不能是 bool")
            if isinstance(value, (int, float, np.integer, np.floating)):
                if str(field).endswith(("_fraction", "_fraction_fluid")):
                    coverage[field] = _strict_probability(
                        value, name=f"{context}.coverage.{field}"
                    )
                else:
                    coverage[field] = _strict_nonnegative_int(
                        value, name=f"{context}.coverage.{field}"
                    )
    else:
        try:
            coverage = float(raw_coverage)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{context}.coverage 必须是 [0,1] 数值或 coverage object"
            ) from exc
        if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError(f"{context}.coverage 必须是 [0,1] 内的有限数")
    literature = metadata["literature"]
    if not isinstance(literature, Mapping) or literature.get("status") != "pending_verification":
        raise ValueError(
            f"{context}.literature 必须保留 pending_verification；Haller 依据待核实"
        )
    if metadata.get("legacy_p85_used") is not False:
        raise ValueError(f"{context} Haller anchor 禁止 legacy_p85 fallback")
    if metadata.get("fallback_used") is not None:
        raise ValueError(f"{context} Haller anchor 不允许 fallback")
    normalized = copy.deepcopy(dict(metadata))
    normalized["failure_count"] = failure_count
    normalized["coverage"] = coverage
    return normalized


def _validate_train_batch(batch: Any, *, mode: str) -> Any:
    """Validate a typed dependency batch or an explicit callback fixture."""
    contract_batch = batch if isinstance(batch, contract.WeakSupervisionBatch) else _mapping_or_attr(
        batch, "contract_batch"
    )
    if isinstance(contract_batch, contract.WeakSupervisionBatch):
        contract.validate_training_batch(contract_batch, mode)
        provenance = contract_batch.provenance
        sampling_source = contract_batch.sampling_source
        label_source = contract_batch.label_source
        if sampling_source is None:
            raise ValueError(f"{mode} training batch 必须显式携带 sampling_source")
    elif isinstance(batch, Mapping):
        required = (
            "mode", "split_name", "label_source", "sampling_source",
            "feature_schema", "input_schema", "provenance",
        )
        missing = [field for field in required if field not in batch]
        if missing:
            raise ValueError(
                f"{mode} training batch 缺少显式 schema/split/source/provenance：{missing!r}"
            )
        if contract.canonical_mode(batch["mode"]) != mode:
            raise ValueError(f"training batch mode 与 requested mode={mode!r} 不匹配")
        contract.validate_feature_schema(batch["feature_schema"], mode)
        contract.validate_feature_schema(
            batch["input_schema"], contract.mode_spec(mode).adapter_input_schema
        )
        split_name = batch["split_name"]
        label_source = batch["label_source"]
        sampling_source = batch["sampling_source"]
        provenance = batch["provenance"]
        if split_name != "train":
            raise ValueError(f"{mode} train consumer 只能读取 split=train")
        if label_source != FORMAL_LABEL_SOURCES[mode]:
            raise ValueError(
                f"{mode} formal label_source={FORMAL_LABEL_SOURCES[mode]!r}，"
                f"实际 {label_source!r}"
            )
        contract.validate_label_source(label_source)
        contract.validate_label_source(sampling_source)
        if sampling_source in {
            contract.LABEL_SOURCE_HALLER_CALIBRATION,
            contract.LABEL_SOURCE_HALLER_TEST,
        }:
            raise ValueError(f"{mode} training sampling_source 禁止 Haller calibration/test")
    else:
        raise TypeError(
            f"{mode} training batch 必须是带 WeakSupervisionBatch contract 的对象；"
            "禁止把旧 tuple batch 静默接入 pilot"
        )
    if isinstance(contract_batch, contract.WeakSupervisionBatch):
        contract.validate_feature_schema(contract_batch.feature_schema, mode)
        contract.validate_feature_schema(
            contract_batch.input_schema,
            contract.mode_spec(mode).adapter_input_schema,
        )
    _reject_test_data(
        {"label_source": label_source, "sampling_source": sampling_source,
         "provenance": provenance},
        context=f"{mode} training batch",
        forbidden_sources=(
            contract.LABEL_SOURCE_HALLER_CALIBRATION,
            contract.LABEL_SOURCE_HALLER_TEST,
        ),
    )
    _validate_window_provenance(provenance, mode=mode)
    if mode in HALLER_TRAIN_MODES:
        _validate_haller_anchor_metadata(
            provenance.get("anchor") if isinstance(provenance, Mapping) else None,
            context=f"{mode} training batch",
        )
    return batch


def _move_contract_batch(batch: contract.WeakSupervisionBatch, device: str | torch.device) -> contract.WeakSupervisionBatch:
    """Move a model-facing contract batch without changing its provenance."""
    if not isinstance(batch, contract.WeakSupervisionBatch):
        raise TypeError("ContractTrainer 必须消费 WeakSupervisionBatch")
    tensor = lambda value: value.to(device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)
    return contract.WeakSupervisionBatch(
        pathlines=tensor(batch.pathlines),
        labels=tensor(batch.labels).float(),
        label_source=batch.label_source,
        split_name=batch.split_name,
        feature_schema=batch.feature_schema,
        label_mask=tensor(batch.label_mask).bool(),
        unknown_mask=tensor(batch.unknown_mask).bool(),
        sampling_source=batch.sampling_source,
        provenance=copy.deepcopy(dict(batch.provenance)),
        mode=batch.mode,
        input_schema=batch.input_schema,
    )


class ContractTrainer:
    """最小可恢复 B0/B1 trainer，供 pilot 连接真实 batch→step seam。

    B1 的 raw 7→model 6 选择仍由 :class:`ChannelSelectingAdapter` 完成；本
    trainer 只消费已经显式声明 schema/source 的 model-facing batch。B0 也
    经过同一 contract checkpoint 入口，但不会读取 teacher/Haller state。
    """

    GENERATION_VERSION = "e2e-contract-trainer-v1"

    def __init__(
        self,
        student: torch.nn.Module,
        optimizer: Any,
        criterion: Any,
        *,
        mode: str,
        sampling_source: str,
        scheduler: Any,
        seed: int = 0,
        grad_clip_norm: float | None = 1.0,
    ) -> None:
        canonical = contract.canonical_mode(mode)
        if canonical not in {contract.MODE_B0, contract.MODE_B1}:
            raise ValueError("ContractTrainer 只实现 B0/B1；W1/W2/W3 使用依赖票据 trainer")
        if not isinstance(student, contract.ChannelSelectingAdapter):
            raise TypeError("ContractTrainer student 必须是外部 ChannelSelectingAdapter")
        if student.mode != canonical:
            raise ValueError(f"ContractTrainer student mode={student.mode!r} 与 {canonical!r} 不一致")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("ContractTrainer optimizer 必须是 torch optimizer")
        if not isinstance(criterion, contract.ModeAwareLoss):
            raise TypeError("ContractTrainer criterion 必须是 ModeAwareLoss")
        if criterion.mode != canonical:
            raise ValueError("ContractTrainer criterion mode 与 trainer mode 不一致")
        if scheduler is None or not all(hasattr(scheduler, field) for field in ("step", "state_dict", "load_state_dict")):
            raise TypeError("ContractTrainer scheduler 必须提供 step/state_dict/load_state_dict")
        sampling_source = contract.validate_label_source(sampling_source)
        if sampling_source in {
            contract.LABEL_SOURCE_HALLER_CALIBRATION,
            contract.LABEL_SOURCE_HALLER_TEST,
        }:
            raise ValueError("ContractTrainer sampling_source 禁止 calibration/test Haller GT")
        self.student = student
        self.optimizer = optimizer
        self.criterion = criterion
        self.mode = canonical
        self.sampling_source = sampling_source
        self.scheduler = scheduler
        self.seed = _strict_nonnegative_int(seed, name="seed")
        if grad_clip_norm is not None:
            try:
                grad_clip_norm = float(grad_clip_norm)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("grad_clip_norm 必须是正的有限数或 None") from exc
            if not np.isfinite(grad_clip_norm) or grad_clip_norm <= 0.0:
                raise ValueError("grad_clip_norm 必须是正的有限数或 None")
        self.grad_clip_norm = grad_clip_norm
        self.global_step = 0
        self.last_metrics: dict[str, Any] | None = None

    def train_step(
        self,
        batch: contract.WeakSupervisionBatch,
        *,
        epoch: int,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        epoch = _strict_positive_int(epoch, name="epoch")
        contract.validate_training_batch(batch, self.mode)
        if batch.sampling_source != self.sampling_source:
            raise ValueError(
                f"{self.mode} sampling_source 不匹配：trainer={self.sampling_source!r} "
                f"batch={batch.sampling_source!r}"
            )
        # 与 W1/W2/W3 trainer 保持同一 device seam；ContractTrainer 也可能
        # 在 pilot 中首次以 CUDA 执行，不能只迁移 batch 而把 student 留在 CPU。
        self.student.to(device)
        moved = _move_contract_batch(batch, device)
        dummy = moved.pathlines.new_zeros((moved.pathlines.shape[0], 1, 1, 1))
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.student.forward_batch(
            moved, dummy_field=dummy, consumer="train"
        )
        loss = self.criterion(prediction, moved)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError(f"{self.mode} loss 必须是有限 scalar tensor")
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.global_step += 1
        metrics = {
            "loss": float(loss.detach().cpu()),
            "steps": 1,
            "label_count": int(moved.labels.numel()),
            "known_label_count": int(moved.label_mask.sum().detach().cpu()),
            "unknown_label_count": int(moved.unknown_mask.sum().detach().cpu()),
            "mode": self.mode,
            "label_source": moved.label_source,
            "sampling_source": moved.sampling_source,
            "epoch": epoch,
            "global_step": self.global_step,
        }
        self.last_metrics = copy.deepcopy(metrics)
        return metrics

    def run_epoch(
        self,
        batches: Iterable[contract.WeakSupervisionBatch],
        *,
        epoch: int,
        device: str | torch.device = "cpu",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        epoch = _strict_positive_int(epoch, name="epoch")
        if max_steps is not None:
            max_steps = _strict_positive_int(max_steps, name="max_steps")
        logs: list[dict[str, Any]] = []
        for batch in batches:
            if max_steps is not None and len(logs) >= max_steps:
                break
            logs.append(self.train_step(batch, epoch=epoch, device=device))
        if not logs:
            raise ValueError(f"{self.mode} batches 为空：训练循环无样本可跑")
        sources = {log["sampling_source"] for log in logs}
        if sources != {self.sampling_source}:
            raise ValueError(f"{self.mode} 一个 epoch 内 sampling_source 发生漂移")
        summary = {
            "loss": float(np.mean([log["loss"] for log in logs])),
            "steps": len(logs),
            "label_count": int(sum(log["label_count"] for log in logs)),
            "known_label_count": int(sum(log["known_label_count"] for log in logs)),
            "unknown_label_count": int(sum(log["unknown_label_count"] for log in logs)),
            "mode": self.mode,
            "label_source": FORMAL_LABEL_SOURCES[self.mode],
            "sampling_source": self.sampling_source,
            "epoch": epoch,
            "global_step": self.global_step,
        }
        self.last_metrics = copy.deepcopy(summary)
        return summary

    def save_checkpoint(
        self,
        path: Any,
        *,
        epoch: int,
        dataset_config: Mapping[str, Any],
        split_config: Mapping[str, Any],
        sampling_config: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
        calibration_policy: Mapping[str, Any] | None = None,
    ) -> pathlib.Path:
        if calibration_policy is None:
            raise ValueError("ContractTrainer checkpoint 必须显式提供 calibration_policy")
        checkpoint_metrics = self.last_metrics if metrics is None else metrics
        if checkpoint_metrics is None:
            raise ValueError("ContractTrainer checkpoint 必须显式提供 metrics")
        return contract.save_checkpoint(
            path,
            self.student,
            self.optimizer,
            self.scheduler,
            mode=self.mode,
            feature_schema=contract.feature_schema_for_mode(self.mode),
            adapter_input_schema=contract.mode_spec(self.mode).adapter_input_schema,
            dataset_config=dataset_config,
            split_config=split_config,
            sampling_config=sampling_config,
            label_source=FORMAL_LABEL_SOURCES[self.mode],
            sampling_source=self.sampling_source,
            epoch=_strict_positive_int(epoch, name="epoch"),
            global_step=self.global_step,
            metrics=checkpoint_metrics,
            seed=self.seed,
            calibration_policy=calibration_policy,
            extra_metadata={
                "generation_version": self.GENERATION_VERSION,
                "formal_loss_source": FORMAL_LABEL_SOURCES[self.mode],
                "headline_eligible": self.mode != contract.MODE_B1,
                "warm_start_aux": False,
            },
        )

    def load_checkpoint(
        self,
        path: Any,
        *,
        expected_dataset_config: Mapping[str, Any],
        expected_split_config: Mapping[str, Any],
        expected_sampling_config: Mapping[str, Any],
        device: str | torch.device = "cpu",
        load_mode: str = "inference",
        restore_rng: bool = False,
        strict_cuda_rng: bool = False,
    ) -> dict[str, Any]:
        result = contract.load_checkpoint(
            path,
            self.student,
            self.optimizer,
            self.scheduler,
            device=device,
            expected_mode=self.mode,
            expected_feature_schema=contract.feature_schema_for_mode(self.mode),
            expected_dataset_config=expected_dataset_config,
            expected_split_config=expected_split_config,
            expected_sampling_config=expected_sampling_config,
            expected_label_source=FORMAL_LABEL_SOURCES[self.mode],
            expected_sampling_source=self.sampling_source,
            restore_rng=restore_rng,
            strict_cuda_rng=strict_cuda_rng,
            load_mode=load_mode,
        )
        self.global_step = _strict_nonnegative_int(result["global_step"], name="global_step")
        self.seed = _strict_nonnegative_int(result["seed"], name="seed")
        return result


@dataclass(frozen=True)
class PilotConfig:
    """Frozen pilot-level settings shared by all six methods."""

    dataset_config: Mapping[str, Any]
    split_config: Mapping[str, Any]
    sampling_config: Mapping[str, Any]
    epochs: int = PILOT_EPOCHS
    seeds: tuple[int, ...] = (PILOT_SEED,)
    seed: int | None = None
    dataset_names: tuple[str, ...] = PILOT_DATASET_NAMES
    ramp_up_epochs: int = PILOT_RAMP_UP_EPOCHS
    threshold_candidates: tuple[float, ...] = evaluation_report.DEFAULT_THRESHOLD_CANDIDATES
    variance_candidates: tuple[float, ...] | None = None
    device: str = "cpu"
    max_steps: int | None = None

    def __post_init__(self) -> None:
        dataset_config = dict(self.dataset_config)
        split_config = dict(self.split_config)
        sampling_config = dict(self.sampling_config)
        if not dataset_config or dataset_config.get("split_mode") != "weak_supervision":
            raise ValueError(
                "pilot dataset_config 必须显式声明 split_mode=weak_supervision；"
                "历史 frac 数据不能静默回退"
            )
        declared_datasets = dataset_config.get("datasets")
        if declared_datasets is not None and set(declared_datasets) != set(PILOT_DATASET_NAMES):
            raise ValueError(
                "pilot dataset_config.datasets 必须恰好覆盖六个有效 dataset"
            )
        if split_config.get("split_name") != "train":
            raise ValueError("pilot split_config 必须显式声明 split_name=train")
        if (
            "t_win" not in split_config
            or "t_win" not in sampling_config
            or "window_step" not in split_config
            or "window_step" not in sampling_config
        ):
            raise ValueError(
                "pilot split/window config 必须显式提供 t_win 和 window_step"
            )
        raw_sampling_source = sampling_config.get("source")
        if not isinstance(raw_sampling_source, str) or not raw_sampling_source.strip():
            raise ValueError(
                "pilot sampling_config 必须显式提供 sampling source"
            )
        sampling_source = contract.validate_label_source(raw_sampling_source.strip())
        if sampling_source in {
            contract.LABEL_SOURCE_HALLER_CALIBRATION,
            contract.LABEL_SOURCE_HALLER_TEST,
        }:
            raise ValueError(
                "pilot sampling_config.source 禁止 calibration/test Haller GT"
            )
        t_win = _strict_positive_int(split_config["t_win"], name="split_config.t_win")
        if _strict_positive_int(sampling_config["t_win"], name="sampling_config.t_win") != t_win:
            raise ValueError("pilot split_config.t_win 与 sampling_config.t_win 不一致")
        window_step = _strict_positive_int(
            split_config["window_step"], name="split_config.window_step"
        )
        if _strict_positive_int(sampling_config["window_step"], name="sampling_config.window_step") != window_step:
            raise ValueError("pilot split_config.window_step 与 sampling_config.window_step 不一致")
        split_ranges = split_config.get("split_ranges")
        if split_ranges is None:
            split_fields = ("split_start", "split_end")
            missing_split = [field for field in split_fields if field not in split_config]
            if missing_split:
                raise ValueError(
                    f"pilot split_config 必须显式提供 split bounds：{missing_split!r}"
                )
            split_start = _strict_nonnegative_int(
                split_config["split_start"], name="split_config.split_start"
            )
            split_end = _strict_nonnegative_int(
                split_config["split_end"], name="split_config.split_end"
            )
            if split_end <= split_start:
                raise ValueError("pilot split_config.split_end 必须大于 split_start")
            if split_end - split_start < t_win:
                raise ValueError(
                    "pilot train split 长度不足以容纳一个完整 pathline window"
                )
        else:
            if not isinstance(split_ranges, Mapping):
                raise ValueError(
                    "pilot split_config.split_ranges 必须是 dataset -> [start,end] object"
                )
            if set(split_ranges) != set(PILOT_DATASET_NAMES):
                raise ValueError(
                    "pilot split_config.split_ranges 必须恰好覆盖六个有效 dataset"
                )
            normalized_ranges: dict[str, list[int]] = {}
            for name in PILOT_DATASET_NAMES:
                raw_range = split_ranges[name]
                if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                    raise ValueError(
                        f"pilot split_ranges[{name!r}] 必须是 [split_start, split_end]"
                    )
                range_start = _strict_nonnegative_int(
                    raw_range[0], name=f"split_ranges[{name}].start"
                )
                range_end = _strict_nonnegative_int(
                    raw_range[1], name=f"split_ranges[{name}].end"
                )
                if range_end <= range_start:
                    raise ValueError(
                        f"pilot split_ranges[{name!r}] end 必须大于 start"
                    )
                if range_end - range_start < t_win:
                    raise ValueError(
                        f"pilot split_ranges[{name!r}] 长度不足以容纳一个完整 pathline window"
                    )
                normalized_ranges[name] = [range_start, range_end]
            split_config["split_ranges"] = normalized_ranges
        epochs = _strict_positive_int(self.epochs, name="epochs")
        ramp = _strict_positive_int(self.ramp_up_epochs, name="ramp_up_epochs")
        seeds = tuple(_strict_nonnegative_int(seed, name="seed") for seed in self.seeds)
        if self.seed is not None:
            explicit_seed = _strict_nonnegative_int(self.seed, name="seed")
            if seeds != (PILOT_SEED,) and seeds != (explicit_seed,):
                raise ValueError("PilotConfig.seed 与 PilotConfig.seeds 不一致")
            seeds = (explicit_seed,)
        if len(seeds) != 1:
            raise ValueError("pilot 当前票据固定为 1 seed")
        names = tuple(_nonempty_text(name, name="dataset_name") for name in self.dataset_names)
        if names != PILOT_DATASET_NAMES:
            raise ValueError(
                f"pilot 必须覆盖六个有效 dataset，顺序固定为 {PILOT_DATASET_NAMES!r}"
            )
        if str(self.device).split(":", 1)[0] not in {"cpu", "cuda"}:
            raise ValueError("pilot device 必须是 cpu 或 cuda[:index]")
        if self.max_steps is not None:
            _strict_positive_int(self.max_steps, name="max_steps")
        thresholds = tuple(sorted({_strict_probability(value, name="threshold") for value in self.threshold_candidates}))
        if not thresholds:
            raise ValueError("threshold_candidates 不能为空")
        gates = None
        if self.variance_candidates is not None:
            normalized_gates = []
            for value in self.variance_candidates:
                try:
                    gate = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("variance_candidates 必须是 [0,0.25] 数值") from exc
                if not np.isfinite(gate) or not 0.0 <= gate <= 0.25:
                    raise ValueError("variance_candidates 必须位于 [0,0.25]")
                normalized_gates.append(gate)
            gates = tuple(sorted(set(normalized_gates)))
            if not gates:
                raise ValueError("variance_candidates 不能为空")
        object.__setattr__(self, "dataset_config", copy.deepcopy(dataset_config))
        split_config["t_win"] = t_win
        split_config["window_step"] = window_step
        sampling_config["source"] = sampling_source
        sampling_config["t_win"] = t_win
        sampling_config["window_step"] = window_step
        object.__setattr__(self, "split_config", copy.deepcopy(split_config))
        object.__setattr__(self, "sampling_config", copy.deepcopy(sampling_config))
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "ramp_up_epochs", ramp)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "seed", seeds[0])
        object.__setattr__(self, "dataset_names", names)
        object.__setattr__(self, "threshold_candidates", thresholds)
        object.__setattr__(self, "variance_candidates", gates)
        object.__setattr__(self, "device", str(self.device))

@dataclass
class PilotMethod:
    """Callbacks needed by the shared six-method pilot seam.

    ``train_batches`` must return batches carrying explicit train split/window
    provenance.  ``calibration_records`` and ``evaluate_test`` are kept as
    separate callbacks so a test Haller artifact cannot be accidentally passed
    to calibration.  ``save_checkpoint``/``load_checkpoint`` are required for
    the round-trip acceptance check.
    """

    mode: str
    train_batches: Callable[[int], Iterable[Any]]
    train_epoch: Callable[[Iterable[Any], int], Mapping[str, Any]]
    calibration_records: Callable[[], Any]
    evaluate_test: Callable[..., Any]
    save_checkpoint: Callable[..., Any]
    load_checkpoint: Callable[[Any], Mapping[str, Any]]
    role: str = "headline_candidate"
    anchor_hash: str | None = None
    anchor_metadata: Mapping[str, Any] | None = None
    warm_start_aux: bool = False
    variance_gate: float | None = None
    metadata: Mapping[str, Any] | None = None
    # Optional concrete trainer handle used by the multi-GPU pilot's
    # training-progress checkpoint seam.  Keeping it out of the public
    # callback contract preserves the lightweight test/fake-method adapter.
    trainer: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.mode = contract.canonical_mode(self.mode)
        if self.mode not in PILOT_METHOD_ORDER:
            raise ValueError(f"pilot 不支持 method mode={self.mode!r}")
        for name in (
            "train_batches", "train_epoch", "calibration_records",
            "evaluate_test", "save_checkpoint", "load_checkpoint",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"PilotMethod.{name} 必须可调用")
        if not isinstance(self.warm_start_aux, bool):
            raise ValueError("pilot warm_start_aux 必须是 bool")
        if self.warm_start_aux:
            raise ValueError("当前 pilot 所有方法必须从头初始化，warm_start_aux 必须为 False")
        if self.mode == contract.MODE_B1:
            self.role = "diagnostic"
        if self.role not in {"diagnostic", "headline_candidate"}:
            raise ValueError("PilotMethod.role 必须是 diagnostic/headline_candidate")
        if self.mode in HALLER_TRAIN_MODES:
            if not isinstance(self.anchor_hash, str) or not self.anchor_hash.strip():
                raise ValueError(f"{self.mode} 必须显式携带 anchor_hash")
            self.anchor_hash = self.anchor_hash.strip()
            self.anchor_metadata = _validate_haller_anchor_metadata(
                self.anchor_metadata, context=f"{self.mode} PilotMethod"
            )
        elif self.anchor_hash is not None:
            raise ValueError(f"{self.mode} 不应携带 Haller anchor_hash")
        if self.variance_gate is not None:
            try:
                gate = float(self.variance_gate)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("variance_gate 必须是 [0,0.25] 内的有限数") from exc
            if not np.isfinite(gate) or not 0.0 <= gate <= 0.25:
                raise ValueError("variance_gate 必须是 [0,0.25] 内的有限数")
            self.variance_gate = gate
        self.metadata = copy.deepcopy(dict(self.metadata or {}))
        _reject_test_data(self.metadata, context=f"{self.mode} method metadata")

    @classmethod
    def from_trainer(
        cls,
        *,
        mode: str,
        trainer: Any,
        train_batches: Callable[[int], Iterable[Any]],
        calibration_records: Callable[[], Any],
        evaluate_test: Callable[..., Any],
        dataset_config: Mapping[str, Any],
        split_config: Mapping[str, Any],
        sampling_config: Mapping[str, Any],
        device: str = "cpu",
        max_steps: int | None = None,
        role: str = "headline_candidate",
    ) -> "PilotMethod":
        """Adapt W1-P/W1-H/W2/W3 trainer methods to the pilot callbacks."""
        canonical = contract.canonical_mode(mode)
        if not hasattr(trainer, "run_epoch") or not hasattr(trainer, "save_checkpoint"):
            raise TypeError("from_trainer 需要公开 run_epoch/save_checkpoint seam")

        def train_epoch(batches: Iterable[Any], epoch: int) -> Mapping[str, Any]:
            kwargs: dict[str, Any] = {"epoch": epoch, "device": device}
            if max_steps is not None:
                kwargs["max_steps"] = max_steps
            result = trainer.run_epoch(batches, **kwargs)
            if not isinstance(result, Mapping):
                raise TypeError(f"{canonical} trainer.run_epoch 必须返回 metrics object")
            scheduler = getattr(trainer, "scheduler", None)
            if scheduler is not None and hasattr(scheduler, "step"):
                scheduler.step()
            return dict(result)

        def save_checkpoint(
            path: Any,
            *,
            epoch: int,
            metrics: Mapping[str, Any],
            calibration_policy: Mapping[str, Any],
        ) -> Any:
            return trainer.save_checkpoint(
                path,
                epoch=epoch,
                dataset_config=dataset_config,
                split_config=split_config,
                sampling_config=sampling_config,
                metrics=metrics,
                calibration_policy=calibration_policy,
            )

        def load_checkpoint(path: Any) -> Mapping[str, Any]:
            if not hasattr(trainer, "load_checkpoint"):
                raise TypeError(f"{canonical} trainer 缺少 load_checkpoint seam")
            return trainer.load_checkpoint(
                path,
                expected_dataset_config=dataset_config,
                expected_split_config=split_config,
                expected_sampling_config=sampling_config,
                device=device,
                load_mode="inference",
                restore_rng=False,
                strict_cuda_rng=False,
            )

        trainer_config = getattr(trainer, "config", None)
        variance_gate = getattr(trainer_config, "variance_gate", None)
        metadata = {}
        if canonical in {contract.MODE_W2, contract.MODE_W3}:
            metadata["variance_gate_source"] = (
                "calibration_selection"
                if getattr(trainer, "calibration_selection", None) is not None
                else "pre_registered_training_config"
            )
        return cls(
            mode=canonical,
            train_batches=train_batches,
            train_epoch=train_epoch,
            calibration_records=calibration_records,
            evaluate_test=evaluate_test,
            save_checkpoint=save_checkpoint,
            load_checkpoint=load_checkpoint,
            role=role,
            anchor_hash=getattr(trainer, "anchor_hash", None),
            anchor_metadata=getattr(trainer, "anchor_metadata", None),
            variance_gate=variance_gate,
            metadata=metadata,
            trainer=trainer,
        )


@dataclass(frozen=True)
class PilotSelection:
    """Frozen calibration decisions passed to the W3 factory and checkpoints."""

    thresholds: Mapping[str, float]
    threshold_selections: Mapping[str, Any]
    w2_selection: Any
    best_baseline: str
    proposed_method: str = contract.MODE_W3

    def as_dict(self) -> dict[str, Any]:
        return {
            "thresholds": {key: float(value) for key, value in self.thresholds.items()},
            "threshold_selections": {
                key: _jsonable(value) for key, value in self.threshold_selections.items()
            },
            "w2": _jsonable(self.w2_selection),
            "best_baseline": self.best_baseline,
            "proposed_method": self.proposed_method,
            "headline_methods": [contract.MODE_B0, self.best_baseline, contract.MODE_W3],
        }


def _call_method_factory(factory: Any, selection: PilotSelection | None) -> PilotMethod:
    if isinstance(factory, PilotMethod):
        return factory
    if not callable(factory):
        raise TypeError("pilot method entry 必须是 PilotMethod 或 factory")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        positional = [
            parameter for parameter in signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(
            parameter.kind == parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        if not positional and not has_varargs:
            result = factory()
        else:
            result = factory(selection)
    else:
        result = factory(selection)
    if not isinstance(result, PilotMethod):
        raise TypeError("pilot method factory 必须返回 PilotMethod")
    return result


def _normalize_method_entries(
    entries: Mapping[str, Any],
    *,
    selection: PilotSelection | None = None,
    required_modes: Sequence[str] = PILOT_METHOD_ORDER,
) -> dict[str, PilotMethod]:
    if not isinstance(entries, Mapping):
        raise TypeError("methods 必须是 mode -> PilotMethod/factory mapping")
    normalized: dict[str, PilotMethod] = {}
    for raw_mode, factory in entries.items():
        mode = contract.canonical_mode(raw_mode)
        if mode in normalized:
            raise ValueError(f"methods 中 mode 重复：{mode!r}")
        method = _call_method_factory(factory, selection)
        if method.mode != mode:
            raise ValueError(f"methods key={mode!r} 与 PilotMethod.mode={method.mode!r} 不一致")
        normalized[mode] = method
    required = tuple(contract.canonical_mode(mode) for mode in required_modes)
    missing = [mode for mode in required if mode not in normalized]
    extra = [mode for mode in normalized if mode not in required]
    if missing or extra:
        raise ValueError(f"pilot methods 集合不完整：missing={missing!r} extra={extra!r}")
    return normalized


class _GuardedBatchIterable:
    """Lazy train iterable with an observable non-empty guard."""

    def __init__(self, method: PilotMethod, epoch: int) -> None:
        raw_batches = method.train_batches(epoch)
        if raw_batches is None or isinstance(raw_batches, (str, bytes)):
            raise TypeError(f"{method.mode} train_batches 必须返回 batch iterable")
        try:
            self._iterator = iter(raw_batches)
        except TypeError as exc:
            raise TypeError(f"{method.mode} train_batches 必须返回 batch iterable") from exc
        self._method = method
        self._epoch = epoch
        self.seen = False

    def __iter__(self) -> Iterator[Any]:
        for batch in self._iterator:
            self.seen = True
            yield _validate_train_batch(batch, mode=self._method.mode)


def _guarded_batches(method: PilotMethod, epoch: int) -> _GuardedBatchIterable:
    return _GuardedBatchIterable(method, epoch)


def _ensure_metrics(value: Any, *, mode: str, epoch: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{mode} epoch={epoch} train_epoch 必须返回 metrics object")
    _reject_test_data(value, context=f"{mode} training metrics")
    result = copy.deepcopy(dict(value))
    result.setdefault("epoch", epoch)
    return result


def _coerce_w2_records(values: Any) -> list[Any]:
    import w2

    if isinstance(values, Mapping):
        values = [values]
    if isinstance(values, (str, bytes)):
        raise TypeError("W2 calibration records 必须是 sequence")
    try:
        records = list(values)
    except TypeError as exc:
        raise TypeError("W2 calibration records 必须是 sequence") from exc
    if not records:
        raise ValueError("W2 calibration records 不能为空")
    normalized = []
    for record in records:
        normalized.append(w2._coerce_calibration_record(record))
    return normalized


def _validate_calibration_dataset_set(
    records: Sequence[Any],
    *,
    mode: str,
    config: PilotConfig,
) -> None:
    names = [str(getattr(record, "dataset_name", "")) for record in records]
    actual = set(names)
    expected = set(config.dataset_names)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"{mode} calibration 必须覆盖六个有效 dataset："
            f"missing={missing!r} unexpected={unexpected!r}"
        )


def _calibrate_method(
    method: PilotMethod,
    config: PilotConfig,
) -> tuple[float, Any]:
    values = method.calibration_records()
    if values is None:
        raise ValueError(f"{method.mode} calibration_records 不能为空")
    if method.mode == contract.MODE_W2:
        import w2

        records = _coerce_w2_records(values)
        _validate_calibration_dataset_set(records, mode=method.mode, config=config)
        selected = w2.calibrate_w2_gate(
            records,
            prediction_thresholds=config.threshold_candidates,
            variance_candidates=config.variance_candidates,
        )
        return float(selected.prediction_threshold), selected
    records = list(evaluation_report.normalize_calibration_records(values))
    _validate_calibration_dataset_set(records, mode=method.mode, config=config)
    selected = evaluation_report.select_global_threshold(
        records, thresholds=config.threshold_candidates
    )
    return float(selected.threshold), selected


def _make_policy(
    *,
    method: str,
    threshold: float,
    gate: float | None,
    calibration_selection: Any,
) -> dict[str, Any]:
    if hasattr(calibration_selection, "as_dict"):
        base = dict(calibration_selection.as_dict())
    elif isinstance(calibration_selection, Mapping):
        base = copy.deepcopy(dict(calibration_selection))
    else:
        raise TypeError("calibration_selection 必须是 typed selection 或 object")
    base.update({
        "source": evaluation_report.CALIBRATION_SOURCE,
        "method": method,
        "prediction_threshold": float(threshold),
        "dataset_gate_count": 1,
    })
    if gate is not None:
        base["variance_gate"] = float(gate)
    else:
        base.pop("variance_gate", None)
    # ThresholdSelection already owns a reproducible selection hash.  For a
    # method whose final threshold differs from a W2 upstream threshold (W3),
    # derive a new method-specific hash instead of pretending the decisions are
    # the same.
    payload = {
        "source": base["source"],
        "method": method,
        "prediction_threshold": base["prediction_threshold"],
        "variance_gate": base.get("variance_gate"),
        "dataset_names": base.get("dataset_names", []),
        "record_hashes": base.get("record_hashes", []),
        "candidate_count": base.get("candidate_count", 0),
    }
    base["selection_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not base.get("dataset_names") or not base.get("record_hashes"):
        raise ValueError("calibration policy 必须保留 dataset_names/record_hashes provenance")
    base["dataset_count"] = len(base["dataset_names"])
    return base


def _validate_roundtrip(
    loaded: Any,
    *,
    method: PilotMethod,
    config: PilotConfig,
    threshold: float,
    gate: float | None,
) -> dict[str, Any]:
    if not isinstance(loaded, Mapping):
        raise TypeError(f"{method.mode} checkpoint load 必须返回 metadata object")
    loaded = dict(loaded)
    _reject_test_data(
        loaded,
        context=f"{method.mode} checkpoint metadata",
        forbidden_sources=(evaluation_report.TEST_SOURCE,),
    )
    if loaded.get("mode") != method.mode:
        raise ValueError(f"{method.mode} checkpoint round-trip mode 不一致")
    if loaded.get("epoch") != config.epochs:
        raise ValueError(
            f"{method.mode} checkpoint epoch={loaded.get('epoch')!r} 不等于 pilot epochs={config.epochs}"
        )
    if loaded.get("seed") != config.seed:
        raise ValueError(f"{method.mode} checkpoint seed/RNG metadata 不一致")
    loaded_split = loaded.get("split_config")
    if not isinstance(loaded_split, Mapping) or dict(loaded_split) != dict(config.split_config):
        raise ValueError(f"{method.mode} checkpoint split_config round-trip 不一致")
    if method.anchor_hash is not None and loaded.get("anchor_hash") != method.anchor_hash:
        raise ValueError(f"{method.mode} checkpoint anchor_hash round-trip 不一致")
    if not isinstance(loaded.get("metrics"), Mapping):
        raise ValueError(f"{method.mode} checkpoint 缺少 metrics schema")
    policy = loaded.get("calibration_policy")
    if not isinstance(policy, Mapping):
        raise ValueError(f"{method.mode} checkpoint 缺少 calibration_policy")
    if policy.get("source") != evaluation_report.CALIBRATION_SOURCE:
        raise ValueError(f"{method.mode} checkpoint calibration source 不一致")
    stored_threshold = policy.get("prediction_threshold")
    if stored_threshold is None or float(stored_threshold) != float(threshold):
        raise ValueError(f"{method.mode} checkpoint threshold round-trip 不一致")
    if gate is not None:
        stored_gate = policy.get("variance_gate")
        if stored_gate is None or float(stored_gate) != float(gate):
            raise ValueError(f"{method.mode} checkpoint variance gate round-trip 不一致")
    if not ("rng_state" in loaded or "rng_restored" in loaded or "runtime" in loaded):
        raise ValueError(f"{method.mode} checkpoint 缺少 RNG metadata")
    return {
        "mode": loaded["mode"],
        "epoch": loaded["epoch"],
        "seed": loaded["seed"],
        "split_config": copy.deepcopy(dict(loaded_split)),
        "anchor_hash": loaded.get("anchor_hash"),
        "calibration_policy": copy.deepcopy(dict(policy)),
        "metrics_schema": sorted(str(key) for key in loaded["metrics"]),
        "rng_metadata_present": True,
    }


def _write_json(path: pathlib.Path, value: Any) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def run_pilot(
    methods: Mapping[str, Any],
    *,
    config: PilotConfig,
    output_dir: str | pathlib.Path,
) -> dict[str, Any]:
    """Run all six registered methods and write one auditable pilot report.

    W3 is constructed with its pre-registered train-time uncertainty gate;
    W2 calibration is recorded only after training and is used for final
    evaluation/checkpoint policy.  No test callback is invoked until every
    method has a frozen calibration decision and calibrated checkpoint.
    """
    if not isinstance(config, PilotConfig):
        raise TypeError("run_pilot config 必须是 PilotConfig")
    destination = pathlib.Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pilot_seed = _strict_nonnegative_int(config.seed, name="config.seed")
    raw_entries = dict(methods)
    pre_entries = {key: value for key, value in raw_entries.items()
                   if contract.canonical_mode(key) != contract.MODE_W3}
    pre_methods = _normalize_method_entries(
        pre_entries,
        required_modes=(
            contract.MODE_B0, contract.MODE_B1, contract.MODE_W1_P,
            contract.MODE_W1_H, contract.MODE_W2,
        ),
    )
    # Resolve W3 separately so its factory can consume the W2 decision.
    if contract.MODE_W3 in pre_methods:
        raise ValueError("W3 must be resolved after W2 calibration")

    histories: dict[str, list[dict[str, Any]]] = {}
    final_metrics: dict[str, dict[str, Any]] = {}
    calibrations: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    methods_by_mode: dict[str, PilotMethod] = {}
    for mode in (
        contract.MODE_B0, contract.MODE_B1, contract.MODE_W1_P,
        contract.MODE_W1_H, contract.MODE_W2,
    ):
        method = pre_methods[mode]
        if mode == contract.MODE_W2 and method.metadata.get(
            "variance_gate_source"
        ) not in (None, "pre_registered_training_config"):
            raise ValueError(
                "W2 training gate 不能来自 calibration selection；"
                "必须在训练开始前固定"
            )
        methods_by_mode[mode] = method
        _seed_everything(pilot_seed)
        history: list[dict[str, Any]] = []
        last: dict[str, Any] | None = None
        for epoch in range(1, config.epochs + 1):
            batches = _guarded_batches(method, epoch)
            stats = _ensure_metrics(
                method.train_epoch(batches, epoch), mode=mode, epoch=epoch
            )
            if not batches.seen:
                raise ValueError(f"{mode} epoch={epoch} train_batches 为空")
            history.append(stats)
            last = stats
            print(json.dumps({
                "event": "epoch_complete",
                "mode": mode,
                "epoch": epoch,
                "epochs": config.epochs,
                "steps": stats.get("steps"),
                "loss": stats.get("loss"),
                "global_step": stats.get("global_step"),
            }, ensure_ascii=False, sort_keys=True), flush=True)
        if last is None:
            raise ValueError(f"{mode} 没有完成任何 epoch")
        histories[mode] = history
        final_metrics[mode] = last
        threshold, selected = _calibrate_method(method, config)
        if mode == contract.MODE_W2 and method.variance_gate is not None:
            if float(method.variance_gate) != float(selected.variance_gate):
                raise ValueError(
                    "W2 trainer 的冻结 variance_gate 与 calibration-selected global gate 不一致；"
                    "必须显式重建 trainer，不能静默改写"
                )
        thresholds[mode] = threshold
        calibrations[mode] = selected

    w2_selection = calibrations[contract.MODE_W2]
    if not hasattr(w2_selection, "variance_gate"):
        raise ValueError("W2 calibration 必须返回 global variance_gate")
    w2_gate = float(w2_selection.variance_gate)
    baseline_candidates = (
        (float(calibrations[contract.MODE_W1_H].objective_value), contract.MODE_W1_H),
        (float(calibrations[contract.MODE_W2].objective_value), contract.MODE_W2),
    )
    best_baseline = min(baseline_candidates, key=lambda item: (-item[0], item[1]))[1]
    provisional_selection = PilotSelection(
        thresholds=thresholds,
        threshold_selections=calibrations,
        w2_selection=w2_selection,
        best_baseline=best_baseline,
    )

    w3_factory = None
    for raw_mode, entry in raw_entries.items():
        if contract.canonical_mode(raw_mode) == contract.MODE_W3:
            w3_factory = entry
            break
    if w3_factory is None:
        raise ValueError("pilot methods 缺少 W3 factory/method")
    _seed_everything(pilot_seed)
    w3_method = _call_method_factory(w3_factory, provisional_selection)
    if w3_method.mode != contract.MODE_W3:
        raise ValueError("W3 factory 必须返回 mode=W3 PilotMethod")
    if w3_method.variance_gate is None:
        raise ValueError("W3 必须显式记录预注册的 train-time variance_gate")
    if w3_method.metadata.get("variance_gate_source") not in (
        None, "pre_registered_training_config"
    ):
        raise ValueError(
            "W3 training gate 不能来自 calibration selection；"
            "必须在训练开始前固定"
        )
    methods_by_mode[contract.MODE_W3] = w3_method
    history = []
    last = None
    for epoch in range(1, config.epochs + 1):
        batches = _guarded_batches(w3_method, epoch)
        stats = _ensure_metrics(
            w3_method.train_epoch(batches, epoch),
            mode=contract.MODE_W3,
            epoch=epoch,
        )
        if not batches.seen:
            raise ValueError(f"W3 epoch={epoch} train_batches 为空")
        history.append(stats)
        last = stats
        print(json.dumps({
            "event": "epoch_complete",
            "mode": contract.MODE_W3,
            "epoch": epoch,
            "epochs": config.epochs,
            "steps": stats.get("steps"),
            "loss": stats.get("loss"),
            "global_step": stats.get("global_step"),
        }, ensure_ascii=False, sort_keys=True), flush=True)
    if last is None:
        raise ValueError("W3 没有完成任何 epoch")
    histories[contract.MODE_W3] = history
    final_metrics[contract.MODE_W3] = last
    w3_threshold, w3_calibration = _calibrate_method(w3_method, config)
    thresholds[contract.MODE_W3] = w3_threshold
    calibrations[contract.MODE_W3] = w3_calibration
    selection = PilotSelection(
        thresholds=thresholds,
        threshold_selections=calibrations,
        w2_selection=w2_selection,
        best_baseline=best_baseline,
    )

    method_reports: dict[str, Any] = {}
    for mode in PILOT_METHOD_ORDER:
        method = methods_by_mode[mode]
        gate = w2_gate if mode in {contract.MODE_W2, contract.MODE_W3} else None
        policy = _make_policy(
            method=mode,
            threshold=thresholds[mode],
            gate=gate,
            calibration_selection=calibrations[mode],
        )
        checkpoint_path = destination / f"{mode.lower().replace('-', '_')}_pilot.pt"
        saved = method.save_checkpoint(
            checkpoint_path,
            epoch=config.epochs,
            metrics=final_metrics[mode],
            calibration_policy=policy,
        )
        saved_path = checkpoint_path if saved is None else pathlib.Path(saved)
        if not saved_path.exists():
            raise FileNotFoundError(f"{mode} save_checkpoint 未生成文件：{saved_path}")
        loaded = method.load_checkpoint(saved_path)
        roundtrip = _validate_roundtrip(
            loaded,
            method=method,
            config=config,
            threshold=thresholds[mode],
            gate=gate,
        )
        test_records = method.evaluate_test(
            prediction_threshold=thresholds[mode],
            variance_gate=gate,
        )
        test_report = evaluation_report.build_evaluation_report(
            test_records,
            prediction_threshold=thresholds[mode],
            variance_gate=gate,
            method=mode,
            dataset_names=config.dataset_names,
            checkpoint_epoch=config.epochs,
        )
        method_reports[mode] = {
            "mode": mode,
            "role": method.role,
            "headline_eligible": method.role == "headline_candidate" and mode != contract.MODE_B1,
            "warm_start_aux": method.warm_start_aux,
            "training_variance_gate": method.variance_gate,
            "epochs": config.epochs,
            "seed": config.seed,
            "history": histories[mode],
            "final_metrics": final_metrics[mode],
            "calibration": _jsonable(calibrations[mode]),
            "calibration_policy": policy,
            "checkpoint": str(saved_path),
            "checkpoint_roundtrip": roundtrip,
            "haller_anchor": None if method.anchor_metadata is None else copy.deepcopy(dict(method.anchor_metadata)),
            "test": test_report,
        }

    report = {
        "schema_version": "weak-supervision-pilot-report-v1",
        "pilot": {
            "epochs": config.epochs,
            "seeds": list(config.seeds),
            "ramp_up_epochs": config.ramp_up_epochs,
            "dataset_names": list(config.dataset_names),
            "device": config.device,
            "from_scratch": True,
            "warm_start_aux": False,
            "dataset_config": copy.deepcopy(dict(config.dataset_config)),
            "split_config": copy.deepcopy(dict(config.split_config)),
            "sampling_config": copy.deepcopy(dict(config.sampling_config)),
        },
        "selection": selection.as_dict(),
        "methods": method_reports,
        "test_label_source": evaluation_report.TEST_SOURCE,
        "haller_literature_status": "pending_verification",
    }
    _write_json(destination / "pilot_report.json", report)
    return report


__all__ = [
    "PILOT_METHOD_ORDER",
    "PILOT_DATASET_NAMES",
    "ContractTrainer",
    "PilotConfig",
    "PilotMethod",
    "PilotSelection",
    "run_pilot",
]
