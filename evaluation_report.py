"""Explicit calibration/test evaluation for the weak-supervision pilot.

This module deliberately sits beside :mod:`evaluate` rather than changing the
stage-0 evaluator.  The pilot has a different contract: calibration may read
only ``haller_gt_calibration`` and the final evaluator may read only
``haller_gt_test``.  The records below make that boundary executable and keep
unknown Haller cells out of the confusion denominator.

The Haller algorithm itself is not implemented here.  Report provenance keeps
the source-specific artifact metadata (including the current
``pending_verification`` literature status) so that a later run can be
audited without presenting engineering parameters as canonical paper values.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import weak_supervision_contract as contract


VALID_DATASET_NAMES = (
    "boussinesq",
    "cylinder2d",
    "doublegyre2d",
    "fourcenters2d",
    "jungtelziemniak2d",
    "pipedcylinder2d",
)
DEFAULT_THRESHOLD_CANDIDATES = tuple(float(value) for value in np.linspace(0.1, 0.9, 9))
CALIBRATION_SOURCE = contract.LABEL_SOURCE_HALLER_CALIBRATION
TEST_SOURCE = contract.LABEL_SOURCE_HALLER_TEST


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _strict_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是非负整数")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是非负整数") from exc
    if integer != value or integer < 0:
        raise ValueError(f"{name} 必须是非负整数")
    return integer


def _strict_probability(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是 [0,1] 内的有限数") from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} 必须是 [0,1] 内的有限数")
    return result


def _strict_probability_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 numeric array") from exc
    if array.ndim == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须是非空有限 array")
    if not np.all((array >= 0.0) & (array <= 1.0)):
        raise ValueError(f"{name} 必须位于 [0,1]")
    return array.copy()


def _bool_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=bool)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 bool mask") from exc
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} shape={array.shape} 与 prediction={shape} 不一致")
    return array.copy()


def _label_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 numeric array") from exc
    if array.ndim == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须是非空有限 array")
    if not np.all(np.isin(array, (-1.0, 0.0, 1.0))):
        raise ValueError(f"{name} 必须使用 Haller 三态 -1/0/1")
    return array.copy()


def _validate_artifact_provenance(
    provenance: Any,
    *,
    expected_source: str,
    context: str,
) -> None:
    if provenance is None:
        return
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{context} provenance 必须是 object")
    for field in ("source", "label_source"):
        if field in provenance and provenance[field] != expected_source:
            raise ValueError(
                f"{context} provenance.{field} 必须是 {expected_source}"
            )
    literature = provenance.get("literature")
    if literature is not None and (
        not isinstance(literature, Mapping)
        or literature.get("status") != "pending_verification"
    ):
        raise ValueError(
            f"{context} literature 必须保留 pending_verification；Haller 依据待核实"
        )


def _contains_exact(value: Any, target: str) -> bool:
    """递归查找 source marker，而不是把普通文字 ``test`` 当作泄漏。"""
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


def _test_only_field_names(value: Any) -> set[str]:
    """找出 calibration record 中可扩展的 test-only 字段别名。"""
    # This is an audit-policy parameter, not a test label/metric/result.  The
    # same exception is enforced by the weak-supervision contract and W2 gate;
    # keep the calibration-report guard aligned so authoritative Haller
    # metadata can pass through without weakening the actual leakage checks.
    test_only_key_exceptions = frozenset({
        "failure_fallback_calibration_test",
        "haller_gt_test_artifact_read",
    })
    found: set[str] = set()
    if isinstance(value, np.ndarray):
        return _test_only_field_names(value.tolist())
    if isinstance(value, np.generic):
        return _test_only_field_names(value.item())
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
            found.update(_test_only_field_names(child))
    elif isinstance(value, (tuple, list, set, frozenset)):
        for child in value:
            found.update(_test_only_field_names(child))
    return found


def _reject_calibration_test_data(value: Any, *, context: str) -> None:
    if _contains_exact(value, TEST_SOURCE):
        raise ValueError(
            f"{context} 禁止出现 {TEST_SOURCE}；test Haller GT 只能用于最终 evaluation"
        )
    field_names = sorted(_test_only_field_names(value))
    if field_names:
        raise ValueError(
            f"{context} 禁止访问 test-only label/metric/result 字段：{field_names!r}"
        )


def _reject_per_dataset_selection(value: Any, *, context: str) -> None:
    forbidden = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if (
                normalized in {"dataset_threshold", "dataset_gate", "dataset_gates",
                               "per_dataset_threshold", "per_dataset_gate",
                               "per_dataset_gates"}
                or ("dataset" in normalized and ("threshold" in normalized or "gate" in normalized))
            ):
                forbidden.append(normalized)
            try:
                _reject_per_dataset_selection(child, context=context)
            except ValueError as exc:
                raise exc
    elif isinstance(value, (tuple, list, set, frozenset)):
        for child in value:
            _reject_per_dataset_selection(child, context=context)
    if forbidden:
        raise ValueError(
            f"{context} 禁止 per-dataset threshold/gate：{sorted(set(forbidden))!r}；"
            "calibration 必须选择一个 global value"
        )


def _record_hash(*arrays: np.ndarray, dataset_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(dataset_name.encode("utf-8"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationPredictionRecord:
    """一个 dataset 的 calibration Haller GT 与模型概率。

    ``known_mask`` 是唯一参与 global threshold 的 mask；unknown labels 可
    仍保留为 ``-1``，但不会进入 confusion denominator。record 必须显式声明
    calibration split/source，不能从 test record 推断或回退。
    """

    dataset_name: str
    prediction: Any
    labels: Any
    known_mask: Any
    split_name: str = "calibration"
    label_source: str = CALIBRATION_SOURCE
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        dataset_name = _nonempty_text(self.dataset_name, name="dataset_name")
        if self.split_name != "calibration":
            raise ValueError(
                "calibration prediction record 只能来自 split=calibration"
            )
        if self.label_source != CALIBRATION_SOURCE:
            raise ValueError(
                "calibration prediction record 必须显式使用 haller_gt_calibration"
            )
        _reject_calibration_test_data(self.provenance, context="calibration provenance")
        _reject_per_dataset_selection(self.provenance, context="calibration provenance")
        _validate_artifact_provenance(
            self.provenance,
            expected_source=CALIBRATION_SOURCE,
            context="calibration",
        )
        prediction = _strict_probability_array(self.prediction, name="prediction")
        labels = _label_array(self.labels, name="labels")
        if tuple(labels.shape) != tuple(prediction.shape):
            raise ValueError("calibration prediction/labels shape 必须一致")
        known = _bool_array(self.known_mask, shape=tuple(prediction.shape), name="known_mask")
        if not bool(known.any()):
            raise ValueError(f"calibration dataset={dataset_name!r} 没有 known Haller cells")
        if not np.all((labels[known] == 0.0) | (labels[known] == 1.0)):
            raise ValueError("calibration known Haller labels 必须为 0/1")
        object.__setattr__(self, "dataset_name", dataset_name)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "known_mask", known)
        object.__setattr__(self, "provenance", copy.deepcopy(dict(self.provenance or {})))

    @property
    def source(self) -> str:
        return self.label_source

    @property
    def record_hash(self) -> str:
        return _record_hash(
            self.prediction, self.labels, self.known_mask,
            dataset_name=self.dataset_name,
        )


@dataclass(frozen=True)
class TestEvaluationRecord:
    """一个 dataset/frame 的 frozen-model prediction 与 test Haller GT。

    ``unknown_mask`` 和 ``solid_mask`` 可覆盖 Haller 三态与 geometry policy；
    unknown/solid/invalid 均不进入 confusion denominator。``predictive_variance``
    只在调用方显式提供一个 global ``variance_gate`` 时参与 prediction gate。
    """

    dataset_name: str
    prediction: Any
    labels: Any
    known_mask: Any
    unknown_mask: Any | None = None
    solid_mask: Any | None = None
    invalid_mask: Any | None = None
    predictive_variance: Any | None = None
    split_name: str = "test"
    label_source: str = TEST_SOURCE
    provenance: Mapping[str, Any] | None = None
    frame_count: int = 1
    invalid_frame_count: int = 0
    failure_count: int = 0
    sample_count: int | None = None
    frame_valid: bool = True

    def __post_init__(self) -> None:
        dataset_name = _nonempty_text(self.dataset_name, name="dataset_name")
        if self.split_name != "test":
            raise ValueError("test evaluation record 只能来自 split=test")
        if self.label_source != TEST_SOURCE:
            raise ValueError(
                "test evaluation record 必须显式使用 haller_gt_test；"
                "不能回退到 calibration/legacy labels"
            )
        _validate_artifact_provenance(
            self.provenance,
            expected_source=TEST_SOURCE,
            context="test",
        )
        prediction = _strict_probability_array(self.prediction, name="prediction")
        labels = _label_array(self.labels, name="labels")
        if tuple(labels.shape) != tuple(prediction.shape):
            raise ValueError("test prediction/labels shape 必须一致")
        known = _bool_array(self.known_mask, shape=tuple(prediction.shape), name="known_mask")
        unknown = (
            ~known if self.unknown_mask is None
            else _bool_array(self.unknown_mask, shape=tuple(prediction.shape), name="unknown_mask")
        )
        solid = (
            np.zeros(prediction.shape, dtype=bool) if self.solid_mask is None
            else _bool_array(self.solid_mask, shape=tuple(prediction.shape), name="solid_mask")
        )
        invalid = (
            np.zeros(prediction.shape, dtype=bool) if self.invalid_mask is None
            else _bool_array(self.invalid_mask, shape=tuple(prediction.shape), name="invalid_mask")
        )
        if np.any(known & unknown):
            raise ValueError("test known_mask 与 unknown_mask 不能重叠")
        if np.any(known & solid):
            raise ValueError("test solid cells 必须保持 unknown")
        if np.any(known & invalid):
            raise ValueError("test invalid cells 不能进入 known mask")
        if np.any(~(known | unknown | invalid)):
            raise ValueError("test 每个 cell 必须显式属于 known/unknown/invalid")
        if not np.all((labels[known] == 0.0) | (labels[known] == 1.0)):
            raise ValueError("test known Haller labels 必须为 0/1")
        if self.predictive_variance is None:
            variance = None
        else:
            variance = np.asarray(self.predictive_variance, dtype=np.float64)
            if tuple(variance.shape) != tuple(prediction.shape):
                raise ValueError("predictive_variance shape 必须与 prediction 一致")
            if not np.all(np.isfinite(variance)) or not np.all(
                (variance >= 0.0) & (variance <= 0.25)
            ):
                raise ValueError("predictive_variance 必须位于 [0,0.25]")
            variance = variance.copy()
        frame_count = _strict_nonnegative_int(self.frame_count, name="frame_count")
        if frame_count <= 0:
            raise ValueError("frame_count 必须为正整数")
        invalid_frame_count = _strict_nonnegative_int(
            self.invalid_frame_count, name="invalid_frame_count")
        if invalid_frame_count > frame_count:
            raise ValueError("invalid_frame_count 不能大于 frame_count")
        if not isinstance(self.frame_valid, (bool, np.bool_)):
            raise ValueError("frame_valid 必须是 bool")
        frame_valid = bool(self.frame_valid)
        if not frame_valid and invalid_frame_count == 0:
            raise ValueError("frame_valid=False 时必须记录 invalid_frame_count")
        failure_count = _strict_nonnegative_int(self.failure_count, name="failure_count")
        sample_count = (
            int(prediction.size) if self.sample_count is None
            else _strict_nonnegative_int(self.sample_count, name="sample_count")
        )
        object.__setattr__(self, "dataset_name", dataset_name)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "known_mask", known)
        object.__setattr__(self, "unknown_mask", unknown)
        object.__setattr__(self, "solid_mask", solid)
        object.__setattr__(self, "invalid_mask", invalid)
        object.__setattr__(self, "predictive_variance", variance)
        object.__setattr__(self, "provenance", copy.deepcopy(dict(self.provenance or {})))
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "invalid_frame_count", invalid_frame_count)
        object.__setattr__(self, "failure_count", failure_count)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "frame_valid", frame_valid)

    @property
    def source(self) -> str:
        return self.label_source

    @property
    def record_hash(self) -> str:
        arrays = [
            self.prediction, self.labels, self.known_mask,
            self.unknown_mask, self.solid_mask, self.invalid_mask,
        ]
        if self.predictive_variance is not None:
            arrays.append(self.predictive_variance)
        return _record_hash(*arrays, dataset_name=self.dataset_name)


# Shorter aliases are useful to callers that do not want to encode the source
# in a class name.  Keep the explicit names above as the canonical API.
EvaluationRecord = TestEvaluationRecord


@dataclass(frozen=True)
class ThresholdSelection:
    """One reproducible global threshold selected on calibration only."""

    threshold: float
    objective_value: float
    dataset_names: tuple[str, ...]
    record_hashes: tuple[str, ...]
    candidate_count: int
    metrics: Mapping[str, Any]
    objective: str = "f1"
    source: str = CALIBRATION_SOURCE

    def __post_init__(self) -> None:
        threshold = _strict_probability(self.threshold, name="threshold")
        if self.source != CALIBRATION_SOURCE:
            raise ValueError("threshold selection source 必须是 haller_gt_calibration")
        objective_value = float(self.objective_value)
        if not np.isfinite(objective_value):
            raise ValueError("threshold selection objective_value 必须有限")
        names = tuple(_nonempty_text(value, name="dataset_name") for value in self.dataset_names)
        hashes = tuple(_nonempty_text(value, name="record_hash") for value in self.record_hashes)
        candidate_count = _strict_nonnegative_int(self.candidate_count, name="candidate_count")
        if not names or not hashes or candidate_count <= 0:
            raise ValueError("threshold selection provenance 不能为空")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("threshold selection metrics 必须是 object")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "objective_value", objective_value)
        object.__setattr__(self, "dataset_names", names)
        object.__setattr__(self, "record_hashes", hashes)
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "metrics", copy.deepcopy(dict(self.metrics)))

    @property
    def prediction_threshold(self) -> float:
        return self.threshold

    @property
    def selection_hash(self) -> str:
        payload = {
            "source": self.source,
            "objective": self.objective,
            "threshold": self.threshold,
            "objective_value": self.objective_value,
            "dataset_names": list(self.dataset_names),
            "record_hashes": list(self.record_hashes),
            "candidate_count": self.candidate_count,
            "metrics": dict(self.metrics),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "objective": self.objective,
            "prediction_threshold": float(self.threshold),
            "threshold": float(self.threshold),
            "objective_value": float(self.objective_value),
            "dataset_names": list(self.dataset_names),
            "dataset_count": len(self.dataset_names),
            "record_hashes": list(self.record_hashes),
            "candidate_count": int(self.candidate_count),
            "selection_hash": self.selection_hash,
            "dataset_threshold_count": 0,
            "metrics": copy.deepcopy(dict(self.metrics)),
        }


def _confusion(
    labels: np.ndarray,
    prediction: np.ndarray,
    known_mask: np.ndarray,
    *,
    threshold: float,
    solid_mask: np.ndarray | None = None,
    invalid_mask: np.ndarray | None = None,
    predictive_variance: np.ndarray | None = None,
    variance_gate: float | None = None,
) -> dict[str, int]:
    active = known_mask.copy()
    if solid_mask is not None:
        active &= ~solid_mask
    if invalid_mask is not None:
        active &= ~invalid_mask
    if variance_gate is not None:
        if predictive_variance is None:
            raise ValueError(
                "variance_gate 已显式提供，但 test record 缺少 predictive_variance"
            )
        predicted = (prediction >= threshold) & (predictive_variance <= variance_gate)
    else:
        predicted = prediction >= threshold
    target = labels >= 0.5
    return {
        "true_positive": int(np.count_nonzero(active & predicted & target)),
        "false_positive": int(np.count_nonzero(active & predicted & ~target)),
        "false_negative": int(np.count_nonzero(active & ~predicted & target)),
        "true_negative": int(np.count_nonzero(active & ~predicted & ~target)),
    }


def _metrics_from_confusion(confusion: Mapping[str, int]) -> dict[str, float | int]:
    tp = int(confusion["true_positive"])
    fp = int(confusion["false_positive"])
    fn = int(confusion["false_negative"])
    tn = int(confusion["true_negative"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def _coerce_calibration_record(value: Any) -> CalibrationPredictionRecord:
    if isinstance(value, CalibrationPredictionRecord):
        # Reconstruct mutable nested provenance so a caller cannot mutate a
        # previously validated record around the source guard.
        return CalibrationPredictionRecord(
            dataset_name=value.dataset_name,
            prediction=value.prediction,
            labels=value.labels,
            known_mask=value.known_mask,
            split_name=value.split_name,
            label_source=value.label_source,
            provenance=value.provenance,
        )
    if not isinstance(value, Mapping):
        raise TypeError(
            "calibration records 必须是 CalibrationPredictionRecord 或 object"
        )
    _reject_calibration_test_data(value, context="calibration record")
    _reject_per_dataset_selection(value, context="calibration record")
    required = ("dataset_name", "prediction", "labels", "known_mask", "split_name", "label_source")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"calibration record 缺少显式字段：{missing!r}")
    return CalibrationPredictionRecord(
        dataset_name=value["dataset_name"],
        prediction=value["prediction"],
        labels=value["labels"],
        known_mask=value["known_mask"],
        split_name=value["split_name"],
        label_source=value["label_source"],
        provenance=value.get("provenance"),
    )


def _coerce_test_record(value: Any) -> TestEvaluationRecord:
    if isinstance(value, TestEvaluationRecord):
        return TestEvaluationRecord(
            dataset_name=value.dataset_name,
            prediction=value.prediction,
            labels=value.labels,
            known_mask=value.known_mask,
            unknown_mask=value.unknown_mask,
            solid_mask=value.solid_mask,
            invalid_mask=value.invalid_mask,
            predictive_variance=value.predictive_variance,
            split_name=value.split_name,
            label_source=value.label_source,
            provenance=value.provenance,
            frame_count=value.frame_count,
            invalid_frame_count=value.invalid_frame_count,
            failure_count=value.failure_count,
            sample_count=value.sample_count,
            frame_valid=value.frame_valid,
        )
    if not isinstance(value, Mapping):
        raise TypeError("test records 必须是 TestEvaluationRecord 或 object")
    required = ("dataset_name", "prediction", "labels", "known_mask", "split_name", "label_source")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"test evaluation record 缺少显式字段：{missing!r}")
    return TestEvaluationRecord(
        dataset_name=value["dataset_name"],
        prediction=value["prediction"],
        labels=value["labels"],
        known_mask=value["known_mask"],
        unknown_mask=value.get("unknown_mask"),
        solid_mask=value.get("solid_mask"),
        invalid_mask=value.get("invalid_mask"),
        predictive_variance=value.get("predictive_variance"),
        split_name=value["split_name"],
        label_source=value["label_source"],
        provenance=value.get("provenance"),
        frame_count=value.get("frame_count", 1),
        invalid_frame_count=value.get("invalid_frame_count", 0),
        failure_count=value.get("failure_count", 0),
        sample_count=value.get("sample_count"),
        frame_valid=value.get("frame_valid", True),
    )


def _as_records(values: Any, *, coerce: Any, name: str) -> list[Any]:
    if isinstance(values, Mapping):
        # A single record mapping is accepted only when it identifies its own
        # dataset.  A dataset->records mapping is also supported, but missing
        # dataset_name remains an error rather than an old default.
        if "prediction" in values or "mean_probability" in values:
            return [coerce(values)]
        flattened = []
        for dataset_name, child in values.items():
            children = [child] if isinstance(child, Mapping) else list(child)
            for record in children:
                if isinstance(record, Mapping) and "dataset_name" not in record:
                    record = {**record, "dataset_name": dataset_name}
                flattened.append(coerce(record))
        if not flattened:
            raise ValueError(f"{name} 不能为空")
        return flattened
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} 必须是 record sequence")
    result = [coerce(value) for value in values]
    if not result:
        raise ValueError(f"{name} 不能为空")
    return result


def _candidate_thresholds(values: Any | None) -> tuple[float, ...]:
    if values is None:
        values = DEFAULT_THRESHOLD_CANDIDATES
    if isinstance(values, (str, bytes)):
        raise TypeError("thresholds 必须是 numeric sequence")
    try:
        candidates = tuple(sorted({_strict_probability(value, name="threshold") for value in values}))
    except TypeError as exc:
        raise TypeError("thresholds 必须是 numeric sequence") from exc
    if not candidates:
        raise ValueError("thresholds 不能为空")
    return candidates


def select_global_threshold(
    records: Any,
    *,
    thresholds: Any | None = None,
) -> ThresholdSelection:
    """用所有 calibration datasets 合并后的 F1 选择一个 global threshold。"""
    normalized = _as_records(
        records, coerce=_coerce_calibration_record, name="calibration records"
    )
    candidates = _candidate_thresholds(thresholds)
    rows = []
    for threshold in candidates:
        confusion = {key: 0 for key in ("true_positive", "false_positive", "false_negative", "true_negative")}
        for record in normalized:
            row = _confusion(
                record.labels, record.prediction, record.known_mask,
                threshold=threshold,
            )
            for key, value in row.items():
                confusion[key] += value
        rows.append({"threshold": threshold, **_metrics_from_confusion(confusion)})
    best = min(rows, key=lambda row: (
        -float(row["f1"]),
        abs(float(row["threshold"]) - 0.5),
        float(row["threshold"]),
    ))
    dataset_names = tuple(sorted({record.dataset_name for record in normalized}))
    record_hashes = tuple(
        record.record_hash for record in sorted(
            normalized, key=lambda record: (record.dataset_name, record.record_hash)
        )
    )
    metrics = {
        key: best[key]
        for key in (
            "true_positive", "false_positive", "false_negative", "true_negative",
            "precision", "recall", "f1", "iou",
        )
    }
    return ThresholdSelection(
        threshold=float(best["threshold"]),
        objective_value=float(best["f1"]),
        dataset_names=dataset_names,
        record_hashes=record_hashes,
        candidate_count=len(rows),
        metrics=metrics,
    )


def _report_row(
    records: Sequence[TestEvaluationRecord],
    *,
    threshold: float,
    variance_gate: float | None,
) -> dict[str, Any]:
    confusion = {key: 0 for key in ("true_positive", "false_positive", "false_negative", "true_negative")}
    known_count = unknown_count = solid_count = invalid_cell_count = 0
    frame_count = effective_frame_count = sample_count = effective_sample_count = 0
    failure_count = invalid_frame_count = 0
    for record in records:
        invalid_frame_count += record.invalid_frame_count
        frame_count += record.frame_count
        effective_frame_count += record.frame_count - record.invalid_frame_count
        if record.sample_count is None:
            raise ValueError("test evaluation record sample_count 未规范化")
        sample_count += int(record.sample_count)
        failure_count += record.failure_count
        invalid_mask = np.asarray(record.invalid_mask, dtype=bool)
        solid_mask = np.asarray(record.solid_mask, dtype=bool)
        known_mask = np.asarray(record.known_mask, dtype=bool)
        unknown_mask = np.asarray(record.unknown_mask, dtype=bool)
        invalid_cell_count += int(np.count_nonzero(invalid_mask))
        solid_count += int(np.count_nonzero(solid_mask & ~invalid_mask))
        active = known_mask & ~solid_mask & ~invalid_mask
        known_count += int(np.count_nonzero(active))
        unknown_count += int(np.count_nonzero(
            unknown_mask & ~solid_mask & ~invalid_mask
        ))
        effective_sample_count += int(np.count_nonzero(active))
        # Invalid frames are represented cell-wise by invalid_mask.  Do not
        # blanket-zero an entire dataset record: a record may contain valid
        # and invalid sampled frames, and only the invalid frame's cells may
        # leave the metric denominator.
        row = _confusion(
            record.labels, record.prediction, record.known_mask,
            threshold=threshold,
            solid_mask=record.solid_mask,
            invalid_mask=record.invalid_mask,
            predictive_variance=record.predictive_variance,
            variance_gate=variance_gate,
        )
        for key, value in row.items():
            confusion[key] += value
    metrics = _metrics_from_confusion(confusion)
    coverage_denominator = known_count + unknown_count
    coverage_known = known_count / coverage_denominator if coverage_denominator else 0.0
    coverage_unknown = unknown_count / coverage_denominator if coverage_denominator else 0.0
    denominator = known_count if known_count else 0
    predicted_positive = confusion["true_positive"] + confusion["false_positive"]
    target_positive = confusion["true_positive"] + confusion["false_negative"]
    return {
        **metrics,
        "dataset_name": records[0].dataset_name,
        "record_count": len(records),
        "frame_count": frame_count,
        "effective_frame_count": effective_frame_count,
        "invalid_frame_count": invalid_frame_count,
        "sample_count": sample_count,
        "effective_cell_count": known_count,
        "effective_sample_count": effective_sample_count,
        "known_count": known_count,
        "unknown_count": unknown_count,
        "solid_count": solid_count,
        "invalid_cell_count": invalid_cell_count,
        "invalid_count": invalid_frame_count + invalid_cell_count,
        "failure_count": failure_count,
        "haller_known_coverage": float(coverage_known),
        "haller_unknown_coverage": float(coverage_unknown),
        "coverage": float(coverage_known),
        "predicted_area_ratio": float(predicted_positive / denominator) if denominator else 0.0,
        "ground_truth_area_ratio": float(target_positive / denominator) if denominator else 0.0,
    }


def _macro(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("evaluation report 至少需要一个 dataset row")
    result = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in (
            "precision", "recall", "f1", "iou", "haller_known_coverage",
            "haller_unknown_coverage", "coverage", "predicted_area_ratio",
            "ground_truth_area_ratio",
        )
    }
    result.update({
        "dataset_count": len(rows),
        "frame_count": int(sum(int(row["frame_count"]) for row in rows)),
        "effective_frame_count": int(sum(int(row["effective_frame_count"]) for row in rows)),
        "sample_count": int(sum(int(row["sample_count"]) for row in rows)),
        "effective_cell_count": int(sum(int(row["effective_cell_count"]) for row in rows)),
        "effective_sample_count": int(sum(int(row["effective_sample_count"]) for row in rows)),
        "invalid_frame_count": int(sum(int(row["invalid_frame_count"]) for row in rows)),
        "invalid_cell_count": int(sum(int(row["invalid_cell_count"]) for row in rows)),
        "invalid_count": int(sum(int(row["invalid_count"]) for row in rows)),
        "failure_count": int(sum(int(row["failure_count"]) for row in rows)),
    })
    return result


def build_evaluation_report(
    records: Any,
    *,
    prediction_threshold: Any | None = None,
    threshold: Any | None = None,
    method: str,
    variance_gate: Any | None = None,
    dataset_names: Sequence[str] | None = None,
    checkpoint_epoch: int | None = None,
) -> dict[str, Any]:
    """生成 per-dataset 与 equal-weight macro 的 frozen test report。"""
    if prediction_threshold is not None and threshold is not None:
        if _strict_probability(prediction_threshold, name="prediction_threshold") != _strict_probability(
            threshold, name="threshold"
        ):
            raise ValueError("prediction_threshold 与 threshold 不一致")
    selected_threshold = _strict_probability(
        prediction_threshold if prediction_threshold is not None else threshold,
        name="prediction_threshold",
    )
    method = _nonempty_text(method, name="method")
    if variance_gate is None:
        selected_gate = None
    else:
        try:
            selected_gate = float(variance_gate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("variance_gate 必须是 [0,0.25] 内的有限数") from exc
        if not np.isfinite(selected_gate) or not 0.0 <= selected_gate <= 0.25:
            raise ValueError("variance_gate 必须是 [0,0.25] 内的有限数")
    normalized = _as_records(records, coerce=_coerce_test_record, name="test records")
    grouped: dict[str, list[TestEvaluationRecord]] = {}
    for record in normalized:
        grouped.setdefault(record.dataset_name, []).append(record)
    if dataset_names is None:
        ordered_names = tuple(sorted(grouped))
    else:
        ordered_names = tuple(_nonempty_text(name, name="dataset_name") for name in dataset_names)
        if len(set(ordered_names)) != len(ordered_names):
            raise ValueError("dataset_names 不能重复")
        missing = sorted(set(ordered_names) - set(grouped))
        unexpected = sorted(set(grouped) - set(ordered_names))
        if missing or unexpected:
            raise ValueError(
                f"test report dataset 集合不一致：missing={missing!r} unexpected={unexpected!r}"
            )
    rows = {
        name: _report_row(grouped[name], threshold=selected_threshold, variance_gate=selected_gate)
        for name in ordered_names
    }
    report = {
        "schema_version": "weak-supervision-evaluation-report-v1",
        "method": method,
        "split_name": "test",
        "label_source": TEST_SOURCE,
        "prediction_threshold": selected_threshold,
        "variance_gate": selected_gate,
        "checkpoint_epoch": None if checkpoint_epoch is None else _strict_nonnegative_int(
            checkpoint_epoch, name="checkpoint_epoch"
        ),
        "per_dataset": rows,
        "macro": _macro(tuple(rows.values())),
        "boussinesq_stress": copy.deepcopy(rows.get("boussinesq", {
            "dataset_name": "boussinesq",
            "available": False,
        })),
        "record_hashes": {
            name: [record.record_hash for record in grouped[name]]
            for name in ordered_names
        },
        "test_artifacts": {
            name: [_artifact_provenance_summary(record.provenance) for record in grouped[name]]
            for name in ordered_names
        },
    }
    return report


# Public aliases used by different callers while retaining one implementation.
aggregate_evaluation = build_evaluation_report
evaluate_predictions = build_evaluation_report


def write_evaluation_report(path: str | pathlib.Path, report: Mapping[str, Any]) -> pathlib.Path:
    """Write a JSON-friendly report after validating its top-level shape."""
    if not isinstance(report, Mapping):
        raise TypeError("report 必须是 object")
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def normalize_calibration_records(records: Any) -> tuple[CalibrationPredictionRecord, ...]:
    """Normalize calibration records while retaining the source guard."""
    return tuple(_as_records(
        records, coerce=_coerce_calibration_record, name="calibration records"
    ))


def normalize_test_records(records: Any) -> tuple[TestEvaluationRecord, ...]:
    """Normalize explicit test records while retaining the test source guard."""
    return tuple(_as_records(records, coerce=_coerce_test_record, name="test records"))


def _artifact_provenance_summary(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        return {}
    fields = (
        "source", "algorithm_version", "parameter_hash", "input_hash",
        "mask_hash", "failure_count", "coverage", "frame_index", "artifact_id",
        "literature",
    )
    return {
        field: _jsonable(provenance[field])
        for field in fields
        if field in provenance
    }


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
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError("report 不能写入 NaN/Inf")
    return value


__all__ = [
    "VALID_DATASET_NAMES",
    "DEFAULT_THRESHOLD_CANDIDATES",
    "CALIBRATION_SOURCE",
    "TEST_SOURCE",
    "CalibrationPredictionRecord",
    "TestEvaluationRecord",
    "EvaluationRecord",
    "ThresholdSelection",
    "select_global_threshold",
    "build_evaluation_report",
    "aggregate_evaluation",
    "evaluate_predictions",
    "write_evaluation_report",
    "normalize_calibration_records",
    "normalize_test_records",
]
