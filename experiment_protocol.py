"""Data and experiment protocol contracts for the criterion-ambiguity study.

This module owns the shared protocol seam for the new study.  It deliberately
does not generate labels or inspect final-test values.  It records frame
identity and experiment decisions, provides the historical R0 access boundary,
and defines the isolated Ticket 01 P95/P80/Haller-standard-IVD contract that
later evidence/evaluation tickets consume.

The public JSON artifacts are intentionally small and JSON-only.  A dataset
record contains frame ranges and a metadata fingerprint; it does not contain
labels, coverage, class balance, agreement, conflict, predictions, or test
metrics.  This makes it possible to verify a sealed input's schema and
integrity without opening its final-test relationships.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


STUDY_SLUG = "criterion-ambiguity-vortex-learning"
PROTOCOL_VERSION = "r0-v1"
PROTOCOL_SCHEMA_VERSION = "criterion-ambiguity-data-protocol-v1"
FREEZE_SCHEMA_VERSION = "criterion-ambiguity-freeze-record-v1"

# Ticket 01 introduces a deliberately separate protocol identity.  The old
# R0 constants above remain available for historical artifacts and callers;
# none of them is promoted to the P95/P80 experiment by aliasing or in-place
# migration.
P95P80_PROTOCOL_VERSION = (
    "criterion-ambiguity-vortex-learning-haller-ivd-p95p80-v1"
)
P95P80_PROTOCOL_SCHEMA_VERSION = "criterion-ambiguity-p95p80-protocol-v1"
P95P80_FREEZE_SCHEMA_VERSION = "criterion-ambiguity-p95p80-freeze-record-v1"
P95P80_ARTIFACT_SCHEMA_VERSION = "criterion-ambiguity-p95p80-artifact-identity-v1"
P95P80_PROVENANCE_SCHEMA_VERSION = "criterion-ambiguity-p95p80-provenance-v1"
P95P80_PERCENTILE_SOURCE_ID = "standard_ivd_valid_fluid_p95_p80"
P95P80_HALLER_SOURCE_ID = "haller_standard_ivd_sparse_anchor"
P95P80_IVD_DEFINITION_ID = "haller-2016-standard-ivd-v1"
P95P80_IVD_FORMULA = "abs(omega - mean_fluid(omega))"
P95P80_SEALED_RELATIONSHIP_FIELDS = (
    "test_labels",
    "test_coverage",
    "test_relationships",
    "test_model_metrics",
    "test_predictions",
    "test_metrics",
)
P95P80_PROVENANCE_FIELDS = (
    "schema_version",
    "protocol_version",
    "protocol_manifest_hash",
    "source_id",
    "artifact_id",
    "dataset_name",
    "split_name",
    "frame_index",
    "geometry_fingerprint",
    "input_fingerprint",
    "valid_fluid_fingerprint",
    "integrity_fingerprint",
)
P95P80_ALLOWED_SOURCE_IDS = (
    P95P80_PERCENTILE_SOURCE_ID,
    P95P80_HALLER_SOURCE_ID,
)
P95P80_LEGACY_PROTOCOL_VERSIONS = (
    PROTOCOL_VERSION,
    "criterion-ambiguity-vortex-learning-dense-haller-v1",
    "criterion-ambiguity-vortex-learning-sparse-anchor-v1",
)
P95P80_LEGACY_SOURCE_IDS = (
    "legacy_p85",
    "local_p90_p60",
    "local_ivd_5x5",
    "haller_anchor_train",
    "haller_gt_calibration",
    "haller_gt_test",
)

DATASET_NAMES = (
    "boussinesq",
    "cylinder2d",
    "doublegyre2d",
    "fourcenters2d",
    "jungtelziemniak2d",
    "pipedcylinder2d",
)
SPLIT_NAMES = ("train", "calibration", "test")
AUDIT_SPLIT_NAME = "development_audit"
ALL_PARTITION_NAMES = SPLIT_NAMES + (AUDIT_SPLIT_NAME,)

BOUSSINESQ_AUDIT_FRAME = 1300
DEFAULT_T_WIN = 24
DEFAULT_WINDOW_STEP = 4

FINAL_TEST_RELATIONSHIP_FIELDS = (
    "coverage",
    "class_balance",
    "agreement",
    "conflict",
    "model_test_metrics",
    "test_metrics",
    "test_predictions",
    "test_results",
)

_SPLIT_RULE = {
    "name": "frame_index_fraction_50_10_40",
    "frame_unit": "index",
    "half_open": True,
    "train": {"start": 0, "numerator": 1, "denominator": 2},
    "calibration": {"numerator": 1, "denominator": 2,
                     "end_numerator": 3, "end_denominator": 5},
    "test": {"numerator": 3, "denominator": 5, "end": "total_frames"},
    "rounding": "floor",
}


class ProtocolError(ValueError):
    """Base error for malformed or incompatible protocol artifacts."""


class ManifestValidationError(ProtocolError):
    """Raised when a frame manifest is malformed or violates split rules."""


class FreezeRecordValidationError(ProtocolError):
    """Raised when an experiment freeze record is missing required decisions."""


class FinalTestAccessError(PermissionError):
    """Raised when a sealed final-test relationship is requested."""


class ProtocolIdentityError(ProtocolError):
    """Raised when a source/artifact belongs to another protocol identity."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _copy_json(value: Any) -> Any:
    """Copy a JSON-compatible object without importing numpy/torch."""
    return copy.deepcopy(value)


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} 必须是非空字符串")
    return value.strip()


def _strict_int(value: Any, *, name: str, nonnegative: bool = True) -> int:
    if isinstance(value, bool):
        raise ProtocolError(f"{name} 必须是整数")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"{name} 必须是整数") from exc
    if result != value or (nonnegative and result < 0):
        qualifier = "非负整数" if nonnegative else "整数"
        raise ProtocolError(f"{name} 必须是{qualifier}，实际 {value!r}")
    return result


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} 必须是 object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"{name} 必须是 array")
    return value


def _sum_ranges(ranges: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in ranges)


def _normalise_ranges(
    value: Any,
    *,
    total_frames: int,
    name: str,
) -> tuple[tuple[int, int], ...]:
    raw_ranges = _sequence(value, name=name)
    result: list[tuple[int, int]] = []
    previous_end = -1
    for ordinal, raw_range in enumerate(raw_ranges):
        pair = _sequence(raw_range, name=f"{name}[{ordinal}]")
        if len(pair) != 2:
            raise ManifestValidationError(
                f"{name}[{ordinal}] 必须是 [start, end] 半开区间"
            )
        try:
            start = _strict_int(pair[0], name=f"{name}[{ordinal}].start")
            end = _strict_int(pair[1], name=f"{name}[{ordinal}].end")
        except ProtocolError as exc:
            raise ManifestValidationError(str(exc)) from exc
        if start >= end:
            raise ManifestValidationError(
                f"{name}[{ordinal}] 必须满足 start < end，实际 {pair!r}"
            )
        if end > total_frames:
            raise ManifestValidationError(
                f"{name}[{ordinal}] 越过 total_frames={total_frames}"
            )
        if start < previous_end:
            raise ManifestValidationError(
                f"{name} 必须按 frame index 排序且不能重叠"
            )
        result.append((start, end))
        previous_end = end
    return tuple(result)


def _ranges_to_frames(ranges: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(frame for start, end in ranges for frame in range(start, end))


def _frames_to_ranges(frames: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        return []
    result: list[list[int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        result.append([start, previous + 1])
        start = previous = frame
    result.append([start, previous + 1])
    return result


def _subtract_frames_from_range(
    base: tuple[int, int], excluded_frames: Sequence[int]
) -> list[list[int]]:
    start, end = base
    excluded = [frame for frame in sorted(set(excluded_frames)) if start <= frame < end]
    result: list[list[int]] = []
    cursor = start
    for frame in excluded:
        if cursor < frame:
            result.append([cursor, frame])
        cursor = frame + 1
    if cursor < end:
        result.append([cursor, end])
    return result


def _expected_partitions(
    total_frames: int,
    dataset_name: str,
    *,
    audit_frames: Sequence[int] | None = None,
) -> dict[str, list[list[int]]]:
    total = _strict_int(total_frames, name="total_frames")
    if total <= 0:
        raise ProtocolError("total_frames 必须为正整数")
    name = _nonempty_text(dataset_name, name="dataset_name")
    train_end = total // 2
    calibration_end = (3 * total) // 5
    if train_end <= 0 or calibration_end <= train_end or calibration_end >= total:
        raise ProtocolError(
            f"dataset={name} total_frames={total} 无法生成非空 50/10/40 split"
        )
    requested_audit = tuple(sorted(set(
        _strict_int(frame, name="audit_frame")
        for frame in (audit_frames or ())
    )))
    audit: tuple[int, ...]
    if name == "boussinesq":
        expected_audit = (BOUSSINESQ_AUDIT_FRAME,)
        if requested_audit and requested_audit != expected_audit:
            raise ProtocolError(
                "boussinesq 的 development/audit frame 必须固定为 1300"
            )
        audit = expected_audit
        if BOUSSINESQ_AUDIT_FRAME >= total:
            raise ProtocolError(
                f"boussinesq total_frames={total} 不包含必须排除的 frame 1300"
            )
    else:
        if requested_audit:
            raise ProtocolError(
                f"非 boussinesq dataset={name} 不允许声明 development/audit frame"
            )
        audit = ()
    if any(frame < 0 or frame >= total for frame in audit):
        raise ProtocolError(
            f"dataset={name} audit frame 越界：{audit!r}, total_frames={total}"
        )
    test_base = (calibration_end, total)
    test_ranges = _subtract_frames_from_range(test_base, audit)
    return {
        "train": [[0, train_end]],
        "calibration": [[train_end, calibration_end]],
        "test": test_ranges,
        "development_audit": _frames_to_ranges(audit),
    }


def _window_starts_for_ranges(
    ranges: Sequence[tuple[int, int]],
    *,
    t_win: int,
    window_step: int,
    dataset_name: str,
    split_name: str,
) -> list[int]:
    starts: list[int] = []
    for start, end in ranges:
        if end - start < t_win:
            raise ProtocolError(
                f"dataset={dataset_name} split={split_name} contiguous segment "
                f"[{start}, {end}) 长度不足以容纳 t_win={t_win}"
            )
        starts.extend(range(start, end - t_win + 1, window_step))
    return starts


def build_frame_manifest(
    total_frames: int,
    dataset_name: str,
    *,
    source_identity: Mapping[str, Any] | None = None,
    audit_frames: Sequence[int] | None = None,
    t_win: int = DEFAULT_T_WIN,
    window_step: int = DEFAULT_WINDOW_STEP,
) -> dict[str, Any]:
    """Build one machine-checkable frame manifest.

    ``test`` is represented as contiguous ranges.  For Boussinesq this means
    ``[calibration_end, 1300)`` and ``[1301, total_frames)``; a window is
    enumerated separately inside each range, so no test window can contain
    frame 1300.
    """
    name = _nonempty_text(dataset_name, name="dataset_name")
    total = _strict_int(total_frames, name="total_frames")
    if total <= 0:
        raise ProtocolError("total_frames 必须为正整数")
    win = _strict_int(t_win, name="t_win")
    step = _strict_int(window_step, name="window_step")
    if win <= 0 or step <= 0:
        raise ProtocolError("t_win/window_step 必须为正整数")
    expected = _expected_partitions(total, name, audit_frames=audit_frames)
    identity = _copy_json(dict(source_identity or {}))
    if not identity:
        identity = {
            "kind": "declared_frame_count",
            "fingerprint": _json_hash({"dataset_name": name, "total_frames": total}),
        }
    _mapping(identity, name=f"{name}.source_identity")
    fingerprint = identity.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ProtocolError(f"{name}.source_identity 必须含 fingerprint")

    partitions: dict[str, dict[str, Any]] = {}
    for split_name in ALL_PARTITION_NAMES:
        normalised = _normalise_ranges(
            expected[split_name],
            total_frames=total,
            name=f"{name}.{split_name}.ranges",
        )
        if split_name in SPLIT_NAMES:
            starts = _window_starts_for_ranges(
                normalised,
                t_win=win,
                window_step=step,
                dataset_name=name,
                split_name=split_name,
            )
        else:
            starts = []
        partitions[split_name] = {
            "ranges": [[start, end] for start, end in normalised],
            "frame_count": _sum_ranges(normalised),
            "window_start_count": len(starts),
            "window_starts": starts,
            "eligible_for_test_relationships": split_name == "test",
        }
    record: dict[str, Any] = {
        "dataset_name": name,
        "total_frames": total,
        "split_rule": _copy_json(_SPLIT_RULE),
        "frame_partitions": partitions,
        "development_audit_frames": list(_ranges_to_frames(
            _normalise_ranges(
                expected[AUDIT_SPLIT_NAME],
                total_frames=total,
                name=f"{name}.{AUDIT_SPLIT_NAME}.ranges",
            )
        )),
        "source_identity": identity,
        "window": {
            "frame_unit": "index",
            "half_open": True,
            "complete_only": True,
            "t_win": win,
            "window_step": step,
            "audit_frame_cannot_be_in_test_window": True,
        },
    }
    record["manifest_hash"] = _json_hash(record)
    return validate_frame_manifest(record)


def _validate_split_coverage(
    partitions: Mapping[str, tuple[tuple[int, int], ...]],
    *,
    total_frames: int,
    dataset_name: str,
) -> None:
    all_ranges: list[tuple[int, int, str]] = []
    for split_name, ranges in partitions.items():
        all_ranges.extend((start, end, split_name) for start, end in ranges)
    all_ranges.sort(key=lambda item: (item[0], item[1]))
    cursor = 0
    for start, end, split_name in all_ranges:
        if start != cursor:
            raise ManifestValidationError(
                f"dataset={dataset_name} partitions 在 frame={cursor} 处不连续或重叠；"
                f"下一个 {split_name}=[{start}, {end})"
            )
        cursor = end
    if cursor != total_frames:
        raise ManifestValidationError(
            f"dataset={dataset_name} partitions 未覆盖 [0,{total_frames})，结束于 {cursor}"
        )


def validate_frame_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record and return a defensive JSON copy."""
    try:
        candidate = _copy_json(dict(_mapping(manifest, name="frame_manifest")))
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError("frame_manifest 必须是 JSON object") from exc
    required = (
        "dataset_name", "total_frames", "split_rule", "frame_partitions",
        "development_audit_frames", "source_identity", "window", "manifest_hash",
    )
    missing = [field for field in required if field not in candidate]
    if missing:
        raise ManifestValidationError(f"frame manifest 缺少字段：{missing!r}")
    name = _nonempty_text(candidate["dataset_name"], name="dataset_name")
    total = _strict_int(candidate["total_frames"], name="total_frames")
    if total <= 0:
        raise ManifestValidationError("total_frames 必须为正整数")
    if candidate["split_rule"] != _SPLIT_RULE:
        raise ManifestValidationError(
            f"dataset={name} split_rule 不是统一 50/10/40 frame-index 规则"
        )
    partitions_value = _mapping(candidate["frame_partitions"], name=f"{name}.frame_partitions")
    if set(partitions_value) != set(ALL_PARTITION_NAMES):
        raise ManifestValidationError(
            f"dataset={name} frame_partitions 必须恰好包含 {list(ALL_PARTITION_NAMES)}"
        )
    partitions: dict[str, tuple[tuple[int, int], ...]] = {}
    window = _mapping(candidate["window"], name=f"{name}.window")
    if window.get("frame_unit") != "index" or window.get("half_open") is not True:
        raise ManifestValidationError(f"dataset={name} window 必须是 half-open frame-index")
    if window.get("complete_only") is not True:
        raise ManifestValidationError(f"dataset={name} window.complete_only 必须为 true")
    t_win = _strict_int(window.get("t_win"), name=f"{name}.window.t_win")
    window_step = _strict_int(window.get("window_step"), name=f"{name}.window.window_step")
    if t_win <= 0 or window_step <= 0:
        raise ManifestValidationError(f"dataset={name} t_win/window_step 必须为正整数")
    if window.get("audit_frame_cannot_be_in_test_window") is not True:
        raise ManifestValidationError(
            f"dataset={name} 必须声明 audit frame 不可进入 test window"
        )
    for split_name in ALL_PARTITION_NAMES:
        info = _mapping(
            partitions_value[split_name],
            name=f"{name}.frame_partitions[{split_name}]",
        )
        ranges = _normalise_ranges(
            info.get("ranges"),
            total_frames=total,
            name=f"{name}.{split_name}.ranges",
        )
        partitions[split_name] = ranges
        count = _strict_int(
            info.get("frame_count"),
            name=f"{name}.{split_name}.frame_count",
        )
        if count != _sum_ranges(ranges):
            raise ManifestValidationError(
                f"dataset={name} split={split_name} frame_count 与 ranges 不一致"
            )
        eligible = info.get("eligible_for_test_relationships")
        if eligible is not (split_name == "test"):
            raise ManifestValidationError(
                f"dataset={name} split={split_name} 的 test relationship eligibility 非法"
            )
        start_count = _strict_int(
            info.get("window_start_count"),
            name=f"{name}.{split_name}.window_start_count",
        )
        starts_value = _sequence(
            info.get("window_starts"),
            name=f"{name}.{split_name}.window_starts",
        )
        starts = [
            _strict_int(value, name=f"{name}.{split_name}.window_starts[{index}]")
            for index, value in enumerate(starts_value)
        ]
        expected_starts = (
            _window_starts_for_ranges(
                ranges,
                t_win=t_win,
                window_step=window_step,
                dataset_name=name,
                split_name=split_name,
            )
            if split_name in SPLIT_NAMES and ranges
            else []
        )
        if starts != expected_starts or start_count != len(expected_starts):
            raise ManifestValidationError(
                f"dataset={name} split={split_name} window starts 不符合完整窗口规则"
            )
    _validate_split_coverage(partitions, total_frames=total, dataset_name=name)
    expected = _expected_partitions(total, name)
    for split_name in ALL_PARTITION_NAMES:
        expected_ranges = tuple(tuple(pair) for pair in expected[split_name])
        if partitions[split_name] != expected_ranges:
            raise ManifestValidationError(
                f"dataset={name} split={split_name} 未遵循统一规则："
                f"expected={expected_ranges!r}, actual={partitions[split_name]!r}"
            )
    audit_frames = tuple(
        _strict_int(frame, name=f"{name}.development_audit_frames[{index}]")
        for index, frame in enumerate(_sequence(
            candidate["development_audit_frames"],
            name=f"{name}.development_audit_frames",
        ))
    )
    expected_audit = _ranges_to_frames(partitions[AUDIT_SPLIT_NAME])
    if audit_frames != expected_audit:
        raise ManifestValidationError(
            f"dataset={name} development_audit_frames 与 audit partition 不一致"
        )
    identity = _mapping(candidate["source_identity"], name=f"{name}.source_identity")
    if not isinstance(identity.get("fingerprint"), str) or not identity["fingerprint"]:
        raise ManifestValidationError(f"dataset={name} source_identity 缺少 fingerprint")
    declared_hash = candidate.get("manifest_hash")
    payload = copy.deepcopy(candidate)
    payload.pop("manifest_hash", None)
    if not isinstance(declared_hash, str) or _json_hash(payload) != declared_hash:
        raise ManifestValidationError(f"dataset={name} manifest_hash 校验失败")
    return candidate


def _relative_or_absolute(path: pathlib.Path, *, base: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_protocol_manifest(
    frame_manifests: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    data_root: str | pathlib.Path | None = None,
    t_win: int = DEFAULT_T_WIN,
    window_step: int = DEFAULT_WINDOW_STEP,
) -> dict[str, Any]:
    """Build and validate the six-dataset protocol manifest."""
    if isinstance(frame_manifests, Mapping):
        supplied = dict(frame_manifests)
    else:
        supplied = {}
        for item in _sequence(frame_manifests, name="frame_manifests"):
            record = validate_frame_manifest(item)
            name = record["dataset_name"]
            if name in supplied:
                raise ManifestValidationError(f"dataset={name} manifest 重复")
            supplied[name] = record
    if set(supplied) != set(DATASET_NAMES):
        missing = sorted(set(DATASET_NAMES) - set(supplied))
        extra = sorted(set(supplied) - set(DATASET_NAMES))
        raise ManifestValidationError(
            f"protocol manifest 必须按固定顺序覆盖六个 dataset：missing={missing!r} extra={extra!r}"
        )
    records: dict[str, dict[str, Any]] = {}
    protocol_t_win = _strict_int(t_win, name="t_win")
    protocol_window_step = _strict_int(window_step, name="window_step")
    for name in DATASET_NAMES:
        record = validate_frame_manifest(supplied[name])
        if record["dataset_name"] != name:
            raise ManifestValidationError(
                f"protocol manifest key={name} 与 record.dataset_name 不一致"
            )
        record_win = record["window"]
        if record_win["t_win"] != protocol_t_win or record_win["window_step"] != protocol_window_step:
            raise ManifestValidationError(
                f"dataset={name} window 与 protocol-level t_win/window_step 不一致"
            )
        records[name] = record
    root = None if data_root is None else str(pathlib.Path(data_root).resolve())
    manifest: dict[str, Any] = {
        "artifact_type": "criterion_ambiguity_data_protocol",
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "study_slug": STUDY_SLUG,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_names": list(DATASET_NAMES),
        "split_rule": _copy_json(_SPLIT_RULE),
        "window": {
            "frame_unit": "index",
            "half_open": True,
            "complete_only": True,
            "t_win": protocol_t_win,
            "window_step": protocol_window_step,
            "audit_frame_cannot_be_in_test_window": True,
        },
        "data_root": root,
        "datasets": records,
        "development_audit_rule": {
            "dataset": "boussinesq",
            "frame": BOUSSINESQ_AUDIT_FRAME,
            "classification": "development/audit",
            "test_manifest": "excluded",
            "test_evidence": "excluded",
            "evaluation_denominator": "excluded",
            "final_summary": "excluded",
        },
        "access_policy": {
            "development_allowed_splits": ["train", "calibration"],
            "r3a_allowed_splits": ["train", "calibration"],
            "final_test_relationships": "sealed_until_r3b",
            "sealed_bundle_allowed_operations": ["schema", "integrity_fingerprint"],
            "sealed_relationship_fields": list(FINAL_TEST_RELATIONSHIP_FIELDS),
        },
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    return validate_protocol_manifest(manifest)


def validate_protocol_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the aggregate protocol manifest and all six records."""
    if (
        isinstance(manifest, Mapping)
        and (
            manifest.get("protocol_version") == P95P80_PROTOCOL_VERSION
            or manifest.get("schema_version") == P95P80_PROTOCOL_SCHEMA_VERSION
        )
    ):
        return validate_p95p80_protocol_manifest(manifest)
    candidate = _copy_json(dict(_mapping(manifest, name="protocol_manifest")))
    required = (
        "artifact_type", "schema_version", "study_slug", "protocol_version",
        "dataset_names", "split_rule", "window", "datasets",
        "development_audit_rule", "access_policy", "manifest_hash",
    )
    missing = [field for field in required if field not in candidate]
    if missing:
        raise ManifestValidationError(f"protocol manifest 缺少字段：{missing!r}")
    if candidate["artifact_type"] != "criterion_ambiguity_data_protocol":
        raise ManifestValidationError("protocol manifest artifact_type 不匹配")
    if candidate["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise ManifestValidationError("protocol manifest schema_version 不受支持")
    if candidate["study_slug"] != STUDY_SLUG or candidate["protocol_version"] != PROTOCOL_VERSION:
        raise ManifestValidationError("protocol manifest study/protocol identity 不匹配")
    names = tuple(_nonempty_text(name, name="dataset_name") for name in _sequence(
        candidate["dataset_names"], name="dataset_names"
    ))
    if names != DATASET_NAMES:
        raise ManifestValidationError(
            f"protocol manifest dataset 顺序/集合必须为 {DATASET_NAMES!r}"
        )
    if candidate["split_rule"] != _SPLIT_RULE:
        raise ManifestValidationError("protocol manifest split_rule 不匹配")
    window = _mapping(candidate["window"], name="protocol_manifest.window")
    if window.get("frame_unit") != "index" or window.get("half_open") is not True:
        raise ManifestValidationError("protocol-level window 必须是 half-open frame-index")
    if window.get("complete_only") is not True or window.get(
        "audit_frame_cannot_be_in_test_window"
    ) is not True:
        raise ManifestValidationError("protocol-level window complete/audit policy 不完整")
    t_win = _strict_int(window.get("t_win"), name="protocol.window.t_win")
    window_step = _strict_int(window.get("window_step"), name="protocol.window.window_step")
    if t_win <= 0 or window_step <= 0:
        raise ManifestValidationError("protocol.window t_win/window_step 必须为正整数")
    datasets = _mapping(candidate["datasets"], name="protocol_manifest.datasets")
    if tuple(datasets) != DATASET_NAMES:
        raise ManifestValidationError("protocol manifest datasets key 顺序必须固定")
    for name in DATASET_NAMES:
        record = validate_frame_manifest(datasets[name])
        if record["dataset_name"] != name:
            raise ManifestValidationError(f"protocol datasets[{name}] identity 不一致")
        if record["window"]["t_win"] != t_win or record["window"]["window_step"] != window_step:
            raise ManifestValidationError(f"dataset={name} window 与 protocol window 不一致")
    audit_rule = _mapping(candidate["development_audit_rule"], name="development_audit_rule")
    expected_audit_rule = {
        "dataset": "boussinesq",
        "frame": BOUSSINESQ_AUDIT_FRAME,
        "classification": "development/audit",
        "test_manifest": "excluded",
        "test_evidence": "excluded",
        "evaluation_denominator": "excluded",
        "final_summary": "excluded",
    }
    if dict(audit_rule) != expected_audit_rule:
        raise ManifestValidationError("development/audit frame exclusion rule 不完整或被修改")
    access = _mapping(candidate["access_policy"], name="access_policy")
    if tuple(access.get("development_allowed_splits", ())) != ("train", "calibration"):
        raise ManifestValidationError("development 只允许 train/calibration")
    if tuple(access.get("r3a_allowed_splits", ())) != ("train", "calibration"):
        raise ManifestValidationError("R3a 只允许 train/calibration")
    if access.get("final_test_relationships") != "sealed_until_r3b":
        raise ManifestValidationError("protocol manifest 必须默认封存 final-test relationships")
    if tuple(access.get("sealed_bundle_allowed_operations", ())) != (
        "schema", "integrity_fingerprint"
    ):
        raise ManifestValidationError("sealed bundle allowed operations 不完整")
    if tuple(access.get("sealed_relationship_fields", ())) != FINAL_TEST_RELATIONSHIP_FIELDS:
        raise ManifestValidationError("sealed relationship field list 不完整")
    payload = copy.deepcopy(candidate)
    declared_hash = payload.pop("manifest_hash", None)
    if not isinstance(declared_hash, str) or _json_hash(payload) != declared_hash:
        raise ManifestValidationError("protocol manifest manifest_hash 校验失败")
    return candidate


def build_protocol_manifest_from_data_root(
    data_root: str | pathlib.Path,
    *,
    dataset_names: Sequence[str] = DATASET_NAMES,
    metadata_filename: str = "meta.json",
    t_win: int = DEFAULT_T_WIN,
    window_step: int = DEFAULT_WINDOW_STEP,
) -> dict[str, Any]:
    """Build R0 frame records from metadata only.

    The function reads only each dataset's ``meta.json`` and file metadata.  It
    never loads ``label_field.npy`` or any frame values.  This is important for
    creating a protocol snapshot from a legacy preprocessed directory while
    keeping final-test labels closed.
    """
    root = pathlib.Path(data_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"data_root 不存在或不是目录：{root}")
    names = tuple(_nonempty_text(name, name="dataset_name") for name in dataset_names)
    if names != DATASET_NAMES:
        raise ManifestValidationError(
            f"data_root builder 必须按固定六数据集顺序运行：{DATASET_NAMES!r}"
        )
    records: dict[str, dict[str, Any]] = {}
    for name in names:
        preferred = root / name / "dataset"
        dataset_dir = preferred if preferred.is_dir() else root / name
        metadata_path = dataset_dir / metadata_filename
        if not metadata_path.is_file():
            raise FileNotFoundError(f"dataset={name} metadata 缺失：{metadata_path}")
        try:
            raw = metadata_path.read_bytes()
            metadata = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"dataset={name} metadata 无法读取：{metadata_path}") from exc
        meta = _mapping(metadata, name=f"dataset={name}.metadata")
        shape_value = _sequence(meta.get("shape"), name=f"dataset={name}.metadata.shape")
        if len(shape_value) != 3:
            raise ProtocolError(f"dataset={name} metadata.shape 必须是 [T,Y,X]")
        shape = tuple(_strict_int(value, name=f"dataset={name}.shape[{index}]")
                      for index, value in enumerate(shape_value))
        if any(value <= 0 for value in shape):
            raise ProtocolError(f"dataset={name} metadata.shape 必须为正数")
        files: list[dict[str, Any]] = []
        for filename in ("u.npy", "v.npy", "ivd.npy", "mask.npy"):
            path = dataset_dir / filename
            if path.is_file():
                files.append({"name": filename, "size": path.stat().st_size})
        source_identity = {
            "kind": "preprocessed_metadata_snapshot",
            "dataset_name": name,
            "metadata_path": _relative_or_absolute(metadata_path, base=root.parent),
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "metadata_split_mode": meta.get("split_mode"),
            "shape": list(shape),
            "data_files": files,
            "protocol_alignment": (
                "metadata_only; legacy preprocessing split does not alter this R0 "
                "frame manifest and must be regenerated under the protocol before use"
            ),
        }
        source_identity["fingerprint"] = _json_hash(source_identity)
        records[name] = build_frame_manifest(
            shape[0],
            name,
            source_identity=source_identity,
            t_win=t_win,
            window_step=window_step,
        )
    return build_protocol_manifest(
        records,
        data_root=root,
        t_win=t_win,
        window_step=window_step,
    )


def write_json_artifact(path: str | pathlib.Path, value: Mapping[str, Any]) -> pathlib.Path:
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def read_protocol_manifest(path: str | pathlib.Path) -> dict[str, Any]:
    destination = pathlib.Path(path)
    value = json.loads(destination.read_text(encoding="utf-8"))
    return _validate_any_protocol_manifest(value)


def _haller_candidate_records() -> list[dict[str, Any]]:
    """Return identities to audit; none is a trusted Haller artifact."""
    return [
        {
            "id": "native_numbacs_rotcohvrt",
            "backend": "numbacs",
            "implementation_module": "numbacs_haller",
            "implementation_version": "numbacs-0.2.0+c067f542543f",
            "contour_semantics": "upstream_rotcohvrt",
            "state": "candidate",
            "trusted": False,
        },
        {
            "id": "numpy_reference_contour",
            "backend": "numpy",
            "implementation_module": "haller_anchors",
            "implementation_version": "haller-anchor-v1.0",
            "contour_semantics": "project_reference_contour",
            "state": "candidate",
            "trusted": False,
        },
        {
            "id": "fast_haller_contourpy",
            "backend": "fast_haller",
            "implementation_module": "fast_haller",
            "implementation_version": "fast_haller-contourpy-v1",
            "contour_semantics": "project_optimized_contour",
            "state": "candidate",
            "trusted": False,
        },
    ]


def build_r0_freeze_record(
    protocol_manifest: Mapping[str, Any],
    *,
    recorded_on: str = "2026-09-05",
    final_test_relationships: str = "sealed_until_r3b",
) -> dict[str, Any]:
    """Create the complete R0 decision record without unlocking final test."""
    manifest = validate_protocol_manifest(protocol_manifest)
    if final_test_relationships != "sealed_until_r3b":
        raise FreezeRecordValidationError(
            "R0 freeze builder 只能生成 sealed_until_r3b；R3b 解封必须来自后续完整的 final-study freeze"
        )
    record: dict[str, Any] = {
        "artifact_type": "criterion_ambiguity_r0_freeze_record",
        "schema_version": FREEZE_SCHEMA_VERSION,
        "study_slug": STUDY_SLUG,
        "protocol_version": PROTOCOL_VERSION,
        "recorded_on": _nonempty_text(recorded_on, name="recorded_on"),
        "status": "frozen",
        "freeze_scope": "R0 data, evidence, experiment, and leakage protocol",
        "data": {
            "dataset_names": list(DATASET_NAMES),
            "frame_manifest_schema_version": manifest["schema_version"],
            "frame_manifest_hash": manifest["manifest_hash"],
            "split_rule": _copy_json(_SPLIT_RULE),
            "window": _copy_json(manifest["window"]),
            "feature_schema": {
                "name": "pathline_7ch",
                "version": "v1",
                "channels": ["px", "py", "t", "ivd", "distance", "u", "v"],
                "channel_count": 7,
            },
            "geometry": {
                "coordinate_order": "x_then_y",
                "fluid_mask": "valid_fluid_only",
                "solid_semantics": "ignored_not_negative",
                "invalid_frame_semantics": "ignored_and_counted",
            },
        },
        "sources": {
            "percentile": {
                "id": "local_p90_p60",
                "positive_rule": "fluid IVD >= frame fluid p90",
                "negative_rule": "fluid IVD <= frame fluid p60",
                "unknown_rule": "p60 < fluid IVD < p90",
                "solid_invalid": "ignored_and_counted",
                "role": "dense weak supervision and primary operational reference",
            },
            "haller": {
                "candidate_artifacts": _haller_candidate_records(),
                "trusted_version": None,
                "trust_state": "pending_r1_audit_and_r2_validation",
                "formal_training_allowed_before_r2": False,
                "role": "candidate source only until exact audited/frozen version exists",
            },
            "legacy_p85": {
                "role": "historical/diagnostic only",
                "formal_hybrid_loss_allowed": False,
            },
        },
        "methods": {
            "method_set": ["B0", "B1", "W1-P", "W1-H", "W1-Hybrid", "W2-on-Hybrid", "W3"],
            "roles": {
                "B0": "historical_dense_baseline",
                "B1": "primary_local_ivd_shortcut_diagnostic",
                "W1-P": "percentile_single_source_specialist",
                "W1-H": "haller_single_source_diagnostic",
                "W1-Hybrid": "core_multi_source_baseline",
                "W2-on-Hybrid": "conditional_uncertainty_method",
                "W3": "conditional_trajectory_identity_contrastive_branch",
            },
            "primary_candidates": ["B0", "W1-P", "W1-Hybrid"],
            "conditional_candidates": ["W2-on-Hybrid", "W3"],
            "diagnostics": ["B1", "W1-H"],
            "all_main_runs_from_scratch": True,
            "historical_warm_start_is_auxiliary_only": True,
        },
        "artifacts": {
            "protocol_manifest_version": PROTOCOL_VERSION,
            "protocol_manifest_hash": manifest["manifest_hash"],
            "label_evidence_contract": "r1-layered-label-evidence-v1",
            "haller_artifact_state_machine": ["candidate", "audited", "frozen"],
            "trusted_haller_requires_exact_frozen_version": True,
            "candidate_or_audited_only_is_not_trusted": True,
        },
        "budgets": {
            "closure_epochs": {"min": 2, "max": 5},
            "source_bias_pilot_epochs": {"min": 10, "max": 20},
            "survivor_epochs": 50,
            "budget_candidates": [100, 130],
            "final_budget_selection": {
                "rule": "130 iff any proposed survivor mean calibration worst-reference F1 "
                        "over epochs 91-100 exceeds epochs 71-80 by >= 0.005; else 100",
                "comparison_windows": [[71, 80], [91, 100]],
                "margin": 0.005,
                "scope": "one global budget",
                "test_derived": False,
            },
        },
        "randomization": {
            "pilot_seeds": [0],
            "final_seeds": [0, 1, 2],
            "seed_scope": "global_and_recorded",
            "from_scratch": True,
        },
        "optimization": {
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 1.0e-4,
                "weight_decay": 1.0e-6,
            },
            "scheduler": {
                "name": "TwoStepLR",
                "warmup_epochs": 60,
                "second_learning_rate": 5.0e-6,
                "epoch_budget_invariant": True,
            },
            "gradient_clip": 1.0,
            "amp": False,
            "data_parallel": False,
        },
        "calibration": {
            "procedure": "calibration_only_global_selection",
            "allowed_split": "calibration",
            "allowed_reference_source": "haller_gt_calibration",
            "updates_model_weights": False,
            "updates_normalization": False,
            "updates_source_parameters": False,
            "threshold_selection": {
                "scope": "global",
                "candidates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "objective": "worst_reference_calibration_macro_f1",
                "tie_breakers": ["worst_reference_iou", "distance_to_0.5"],
                "test_derived": False,
                "per_dataset_threshold_allowed": False,
            },
            "uncertainty_gate": {
                "scope": "global",
                "teacher_views": 3,
                "primary_score": "predictive_variance",
                "diagnostic_score": "bernoulli_entropy",
                "accepted_coverage_targets": [1.0, 0.95, 0.9, 0.8, 0.7, 0.5],
                "operating_point": "minimize_worst_reference_macro_selective_risk_subject_to_each_reference_coverage_ge_0.80",
                "tie_breakers": [
                    "higher_worst_reference_selective_f1",
                    "higher_minimum_accepted_coverage",
                    "less_aggressive_abstention_threshold",
                ],
                "rejected_is_not_negative": True,
                "selective_risk_definition": (
                    "(accepted FP + accepted FN) / accepted known valid-fluid cells"
                ),
                "accepted_known_definition": (
                    "reference-known valid-fluid cells for which the model did not reject"
                ),
                "zero_selective_denominator": "undefined_and_operating_point_fails",
                "test_derived": False,
                "per_dataset_gate_allowed": False,
            },
            "alpha": 0.5,
            "per_dataset_alpha_allowed": False,
            "hybrid_policy": {
                "percentile_only_weak_weight": 0.5,
                "haller_trusted_core_weight": 1.0,
                "known_conflict_hard_weight": 0.0,
                "joint_unknown_hard_weight": 0.0,
                "solid_invalid_failure_weight": 0.0,
                "conflict_is_not_forced_to_a_winner": True,
            },
        },
        "evaluation": {
            "primary_references": ["local_p90_p60_known", "audited_haller_known"],
            "metrics_per_reference": ["precision", "recall", "f1", "iou", "auprc"],
            "known_coverage_adjacent_to_metrics": True,
            "common_known_support": {
                "required": True,
                "agreement_and_conflict_separate_from_missing_coverage": True,
            },
            "cross_reference_summary": {
                "statistics": ["mean", "worst", "std"],
                "label": "cross-reference robustness diagnostics",
                "absolute_ground_truth_claim": False,
            },
            "p85_headline_eligible": False,
            "synthetic_exact_separate_panel": True,
            "final_test_relationships_opened_once_at": "R3b_after_final_freeze",
        },
        "branch_decisions": {
            "W2": {
                "enters_main_final_set_only_if_selective_gate_passes": True,
                "failure_role": "diagnostic_only",
            },
            "W3": {
                "start_condition": "calibration worst-reference macro F1 margin >= 0.01 over stronger of B0/W1-P",
                "identity_semantics": "trajectory_identity_contrastive",
                "not_started_by_r0": True,
            },
            "Q": {
                "role": "auxiliary_until_separate_semantic_implementation_synthetic_resolution_audit",
                "primary_reference_eligible_now": False,
            },
            "R8": {
                "deterministic_refiner_before_diffusion": True,
                "diffusion_before_core_study": False,
            },
        },
        "evidence": {
            "haller_literature": {
                "status": "metadata_verified_local_zotero; engineering_semantics_pending",
                "zotero_key": "L2PX3NQX",
                "canonical_parameter_claim": False,
            },
            "evidence_level_for_this_record": "code-verified protocol behavior",
        },
        "access_policy": {
            "development_allowed_splits": ["train", "calibration"],
            "r3a_allowed_splits": ["train", "calibration"],
            "final_test_relationships": final_test_relationships,
            "final_test_unlock_requires": "complete_final_study_freeze_record_and_r3b",
            "sealed_bundle_allowed_operations": ["schema", "integrity_fingerprint"],
            "sealed_relationship_fields": list(FINAL_TEST_RELATIONSHIP_FIELDS),
            "boussinesq_audit_frame": BOUSSINESQ_AUDIT_FRAME,
            "boussinesq_audit_frame_excluded_from": [
                "test_manifest", "test_evidence", "evaluation_denominator", "final_summary"
            ],
        },
    }
    record["freeze_record_hash"] = _json_hash(record)
    return validate_freeze_record(record)


def _require_path(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise FreezeRecordValidationError(f"freeze record 缺少字段：{path}")
        value = value[part]
    return value


def validate_freeze_record(
    record: Mapping[str, Any],
    *,
    protocol_manifest: Mapping[str, Any] | None = None,
    require_final_test_open: bool = False,
) -> dict[str, Any]:
    """Validate completeness, identity, provenance, and integrity.

    A complete R0 record remains sealed by default.  ``require_final_test_open``
    is reserved for the later final-study freeze record and therefore rejects
    the generated R0 record unless a caller explicitly supplies an open record.
    """
    if (
        isinstance(record, Mapping)
        and (
            record.get("protocol_version") == P95P80_PROTOCOL_VERSION
            or record.get("schema_version") == P95P80_FREEZE_SCHEMA_VERSION
        )
    ):
        return validate_p95p80_freeze_record(
            record,
            protocol_manifest=protocol_manifest,
            require_final_test_open=require_final_test_open,
        )
    candidate = _copy_json(dict(_mapping(record, name="freeze_record")))
    required = (
        "artifact_type", "schema_version", "study_slug", "protocol_version",
        "recorded_on", "status", "freeze_scope", "data", "sources", "methods",
        "artifacts", "budgets", "randomization", "optimization", "calibration",
        "evaluation", "branch_decisions", "evidence", "access_policy",
        "freeze_record_hash",
    )
    missing = [field for field in required if field not in candidate]
    if missing:
        raise FreezeRecordValidationError(f"freeze record 不完整，缺少：{missing!r}")
    if candidate["artifact_type"] != "criterion_ambiguity_r0_freeze_record":
        raise FreezeRecordValidationError("freeze record artifact_type 不匹配")
    if candidate["schema_version"] != FREEZE_SCHEMA_VERSION:
        raise FreezeRecordValidationError("freeze record schema_version 不受支持")
    if candidate["study_slug"] != STUDY_SLUG or candidate["protocol_version"] != PROTOCOL_VERSION:
        raise FreezeRecordValidationError("freeze record study/protocol identity 不匹配")
    if candidate["status"] != "frozen":
        raise FreezeRecordValidationError("freeze record status 必须是 frozen")
    _nonempty_text(candidate["recorded_on"], name="recorded_on")
    data = _mapping(candidate["data"], name="freeze.data")
    if tuple(data.get("dataset_names", ())) != DATASET_NAMES:
        raise FreezeRecordValidationError("freeze.data.dataset_names 必须覆盖六个 dataset")
    frame_hash = _nonempty_text(data.get("frame_manifest_hash", ""), name="freeze.data.frame_manifest_hash")
    if protocol_manifest is not None:
        manifest = validate_protocol_manifest(protocol_manifest)
        if frame_hash != manifest["manifest_hash"]:
            raise FreezeRecordValidationError("freeze record 与 protocol manifest fingerprint 不匹配")
    if data.get("split_rule") != _SPLIT_RULE:
        raise FreezeRecordValidationError("freeze.data.split_rule 不匹配")
    feature_schema = _mapping(data.get("feature_schema"), name="freeze.data.feature_schema")
    for field in ("name", "version", "channels", "channel_count"):
        if field not in feature_schema:
            raise FreezeRecordValidationError(f"freeze.data.feature_schema 缺少 {field}")
    if (
        feature_schema["name"] != "pathline_7ch"
        or feature_schema["version"] != "v1"
        or tuple(feature_schema["channels"]) != ("px", "py", "t", "ivd", "distance", "u", "v")
        or feature_schema["channel_count"] != 7
    ):
        raise FreezeRecordValidationError("freeze.data.feature_schema 不是固定 7-channel schema")
    sources = _mapping(candidate["sources"], name="freeze.sources")
    percentile = _mapping(sources.get("percentile"), name="freeze.sources.percentile")
    for field in ("id", "positive_rule", "negative_rule", "unknown_rule", "role"):
        if field not in percentile:
            raise FreezeRecordValidationError(f"freeze.sources.percentile 缺少 {field}")
    if percentile["id"] != "local_p90_p60":
        raise FreezeRecordValidationError("percentile source 必须固定为 local_p90_p60")
    haller = _mapping(sources.get("haller"), name="freeze.sources.haller")
    candidates = _sequence(haller.get("candidate_artifacts"), name="freeze.sources.haller.candidate_artifacts")
    if not candidates:
        raise FreezeRecordValidationError("freeze.sources.haller 必须记录候选 artifact identities")
    for index, item in enumerate(candidates):
        entry = _mapping(item, name=f"haller.candidate_artifacts[{index}]")
        for field in (
            "id", "backend", "implementation_module", "implementation_version",
            "contour_semantics", "state", "trusted",
        ):
            if field not in entry:
                raise FreezeRecordValidationError(
                    f"haller candidate[{index}] 缺少 {field}"
                )
        if entry["trusted"] is not False or entry["state"] not in {"candidate", "audited"}:
            raise FreezeRecordValidationError(
                "R0 Haller candidate/audited artifact 不得获得 trusted 身份"
            )
    if haller.get("trusted_version") is not None:
        raise FreezeRecordValidationError("R0 不得声明 trusted Haller version")
    methods = _mapping(candidate["methods"], name="freeze.methods")
    method_set = _sequence(methods.get("method_set"), name="freeze.methods.method_set")
    expected_methods = ("B0", "B1", "W1-P", "W1-H", "W1-Hybrid", "W2-on-Hybrid", "W3")
    if tuple(method_set) != expected_methods:
        raise FreezeRecordValidationError("freeze.methods.method_set 不完整或顺序漂移")
    roles = _mapping(methods.get("roles"), name="freeze.methods.roles")
    if set(roles) != set(expected_methods):
        raise FreezeRecordValidationError("freeze.methods.roles 必须覆盖全部方法")
    for field in ("primary_candidates", "conditional_candidates", "diagnostics"):
        if not _sequence(methods.get(field), name=f"freeze.methods.{field}"):
            raise FreezeRecordValidationError(f"freeze.methods.{field} 不能为空")
    if methods.get("all_main_runs_from_scratch") is not True:
        raise FreezeRecordValidationError("main runs 必须从 scratch 开始")
    artifacts = _mapping(candidate["artifacts"], name="freeze.artifacts")
    for field in (
        "protocol_manifest_version", "protocol_manifest_hash", "label_evidence_contract",
        "haller_artifact_state_machine",
    ):
        if field not in artifacts:
            raise FreezeRecordValidationError(f"freeze.artifacts 缺少 {field}")
    if artifacts["protocol_manifest_version"] != PROTOCOL_VERSION:
        raise FreezeRecordValidationError("artifact protocol version 不匹配")
    if artifacts["protocol_manifest_hash"] != frame_hash:
        raise FreezeRecordValidationError("artifact 与 frame manifest fingerprint 不一致")
    if tuple(artifacts["haller_artifact_state_machine"]) != ("candidate", "audited", "frozen"):
        raise FreezeRecordValidationError("Haller artifact state machine 不完整")
    if artifacts.get("trusted_haller_requires_exact_frozen_version") is not True:
        raise FreezeRecordValidationError("trusted Haller exact-version guard 缺失")
    budgets = _mapping(candidate["budgets"], name="freeze.budgets")
    for field in ("closure_epochs", "source_bias_pilot_epochs", "survivor_epochs",
                  "budget_candidates", "final_budget_selection"):
        if field not in budgets:
            raise FreezeRecordValidationError(f"freeze.budgets 缺少 {field}")
    if budgets["closure_epochs"] != {"min": 2, "max": 5}:
        raise FreezeRecordValidationError("closure budget 必须固定为 2-5 epochs")
    if budgets["source_bias_pilot_epochs"] != {"min": 10, "max": 20}:
        raise FreezeRecordValidationError("source-bias pilot budget 必须固定为 10-20 epochs")
    if budgets["survivor_epochs"] != 50:
        raise FreezeRecordValidationError("survivor budget 必须固定为 50 epochs")
    if tuple(budgets["budget_candidates"]) != (100, 130):
        raise FreezeRecordValidationError("final budget candidates 必须是 100/130")
    final_budget = _mapping(budgets["final_budget_selection"], name="freeze.budgets.final_budget_selection")
    if (
        final_budget.get("comparison_windows") != [[71, 80], [91, 100]]
        or final_budget.get("margin") != 0.005
        or final_budget.get("scope") != "one global budget"
        or final_budget.get("test_derived") is not False
    ):
        raise FreezeRecordValidationError("final budget selection rule 不完整")
    randomization = _mapping(candidate["randomization"], name="freeze.randomization")
    if tuple(randomization.get("pilot_seeds", ())) != (0,) or tuple(
        randomization.get("final_seeds", ())
    ) != (0, 1, 2) or randomization.get("from_scratch") is not True:
        raise FreezeRecordValidationError("freeze.randomization seeds 不符合预注册协议")
    optimization = _mapping(candidate["optimization"], name="freeze.optimization")
    optimizer = _mapping(optimization.get("optimizer"), name="freeze.optimization.optimizer")
    scheduler = _mapping(optimization.get("scheduler"), name="freeze.optimization.scheduler")
    if (
        optimizer.get("name") != "AdamW"
        or optimizer.get("learning_rate") != 1.0e-4
        or optimizer.get("weight_decay") != 1.0e-6
        or scheduler.get("name") != "TwoStepLR"
        or scheduler.get("warmup_epochs") != 60
        or scheduler.get("second_learning_rate") != 5.0e-6
        or scheduler.get("epoch_budget_invariant") is not True
        or optimization.get("gradient_clip") != 1.0
        or optimization.get("amp") is not False
        or optimization.get("data_parallel") is not False
    ):
        raise FreezeRecordValidationError("optimizer/scheduler identity 不完整")
    calibration = _mapping(candidate["calibration"], name="freeze.calibration")
    if calibration.get("procedure") != "calibration_only_global_selection":
        raise FreezeRecordValidationError("calibration procedure 必须是 calibration-only global")
    if calibration.get("allowed_split") != "calibration" or calibration.get(
        "allowed_reference_source"
    ) != "haller_gt_calibration":
        raise FreezeRecordValidationError("calibration source/split contract 不完整")
    if "threshold_selection" not in calibration:
        raise FreezeRecordValidationError(
            "freeze.calibration 缺少 threshold_selection"
        )
    threshold_selection = _mapping(
        calibration["threshold_selection"], name="freeze.calibration.threshold_selection"
    )
    if (
        threshold_selection.get("scope") != "global"
        or threshold_selection.get("candidates") != [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        or threshold_selection.get("objective") != "worst_reference_calibration_macro_f1"
        or threshold_selection.get("tie_breakers") != ["worst_reference_iou", "distance_to_0.5"]
        or threshold_selection.get("test_derived") is not False
        or threshold_selection.get("per_dataset_threshold_allowed") is not False
    ):
        raise FreezeRecordValidationError("threshold selection 必须是 global calibration-only")
    if "uncertainty_gate" not in calibration:
        raise FreezeRecordValidationError(
            "freeze.calibration 缺少 uncertainty_gate"
        )
    uncertainty = _mapping(
        calibration["uncertainty_gate"], name="freeze.calibration.uncertainty_gate"
    )
    if (
        uncertainty.get("scope") != "global"
        or uncertainty.get("teacher_views") != 3
        or uncertainty.get("primary_score") != "predictive_variance"
        or uncertainty.get("diagnostic_score") != "bernoulli_entropy"
        or uncertainty.get("accepted_coverage_targets") != [1.0, 0.95, 0.9, 0.8, 0.7, 0.5]
        or uncertainty.get("operating_point") != (
            "minimize_worst_reference_macro_selective_risk_subject_to_each_reference_coverage_ge_0.80"
        )
        or uncertainty.get("tie_breakers") != [
            "higher_worst_reference_selective_f1",
            "higher_minimum_accepted_coverage",
            "less_aggressive_abstention_threshold",
        ]
        or uncertainty.get("test_derived") is not False
        or uncertainty.get("per_dataset_gate_allowed") is not False
    ):
        raise FreezeRecordValidationError("uncertainty gate 必须是 global calibration-only")
    if uncertainty.get("rejected_is_not_negative") is not True:
        raise FreezeRecordValidationError("W2 rejected/negative distinction 缺失")
    if uncertainty.get("selective_risk_definition") != (
        "(accepted FP + accepted FN) / accepted known valid-fluid cells"
    ):
        raise FreezeRecordValidationError("selective risk definition 不匹配")
    if uncertainty.get("accepted_known_definition") != (
        "reference-known valid-fluid cells for which the model did not reject"
    ):
        raise FreezeRecordValidationError("accepted known definition 不匹配")
    if uncertainty.get("zero_selective_denominator") != "undefined_and_operating_point_fails":
        raise FreezeRecordValidationError("selective risk zero denominator 必须 undefined 并使 operating point fail")
    if calibration.get("alpha") != 0.5 or calibration.get("per_dataset_alpha_allowed") is not False:
        raise FreezeRecordValidationError("Hybrid alpha 必须固定为 global 0.5")
    hybrid = _mapping(calibration.get("hybrid_policy"), name="freeze.calibration.hybrid_policy")
    if hybrid != {
        "percentile_only_weak_weight": 0.5,
        "haller_trusted_core_weight": 1.0,
        "known_conflict_hard_weight": 0.0,
        "joint_unknown_hard_weight": 0.0,
        "solid_invalid_failure_weight": 0.0,
        "conflict_is_not_forced_to_a_winner": True,
    }:
        raise FreezeRecordValidationError("Hybrid conflict/weight policy 不匹配")
    evaluation = _mapping(candidate["evaluation"], name="freeze.evaluation")
    if tuple(evaluation.get("primary_references", ())) != (
        "local_p90_p60_known", "audited_haller_known"
    ):
        raise FreezeRecordValidationError("primary reference set 不完整")
    cross_reference = _mapping(
        evaluation.get("cross_reference_summary"), name="freeze.evaluation.cross_reference_summary"
    )
    if cross_reference.get("label") != "cross-reference robustness diagnostics":
        raise FreezeRecordValidationError("cross-reference aggregate label 缺失")
    branches = _mapping(candidate["branch_decisions"], name="freeze.branch_decisions")
    for field in ("W2", "W3", "Q", "R8"):
        if field not in branches:
            raise FreezeRecordValidationError(f"freeze.branch_decisions 缺少 {field}")
    evidence = _mapping(candidate["evidence"], name="freeze.evidence")
    haller_literature = _mapping(evidence.get("haller_literature"), name="freeze.evidence.haller_literature")
    if haller_literature.get("zotero_key") != "L2PX3NQX":
        raise FreezeRecordValidationError("Haller literature provenance key 不完整")
    if haller_literature.get("canonical_parameter_claim") is not False:
        raise FreezeRecordValidationError("工程参数不能被记录为 canonical literature claim")
    access = _mapping(candidate["access_policy"], name="freeze.access_policy")
    if tuple(access.get("development_allowed_splits", ())) != ("train", "calibration"):
        raise FreezeRecordValidationError("freeze development split boundary 不完整")
    if tuple(access.get("r3a_allowed_splits", ())) != ("train", "calibration"):
        raise FreezeRecordValidationError("freeze R3a split boundary 不完整")
    relationship_state = access.get("final_test_relationships")
    if relationship_state not in {"sealed_until_r3b", "open_for_r3b"}:
        raise FreezeRecordValidationError("final-test relationship state 非法")
    if tuple(access.get("sealed_bundle_allowed_operations", ())) != (
        "schema", "integrity_fingerprint"
    ):
        raise FreezeRecordValidationError("freeze sealed bundle policy 不完整")
    if tuple(access.get("sealed_relationship_fields", ())) != FINAL_TEST_RELATIONSHIP_FIELDS:
        raise FreezeRecordValidationError("freeze sealed relationship field list 不完整")
    if access.get("boussinesq_audit_frame") != BOUSSINESQ_AUDIT_FRAME:
        raise FreezeRecordValidationError("freeze 缺少 Boussinesq frame 1300 audit rule")
    if access.get("final_test_unlock_requires") != "complete_final_study_freeze_record_and_r3b":
        raise FreezeRecordValidationError("freeze final-test unlock condition 不完整")
    if access.get("boussinesq_audit_frame_excluded_from") != [
        "test_manifest", "test_evidence", "evaluation_denominator", "final_summary"
    ]:
        raise FreezeRecordValidationError("freeze Boussinesq audit exclusion scope 不完整")
    if require_final_test_open and relationship_state != "open_for_r3b":
        raise FreezeRecordValidationError("final-test relationships 仍 sealed_until_r3b")
    payload = copy.deepcopy(candidate)
    declared_hash = payload.pop("freeze_record_hash", None)
    if not isinstance(declared_hash, str) or _json_hash(payload) != declared_hash:
        raise FreezeRecordValidationError("freeze_record_hash 校验失败")
    return candidate


def write_r0_artifacts(
    data_root: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    recorded_on: str = "2026-09-05",
) -> dict[str, pathlib.Path | str]:
    """Generate the R0 manifest, freeze record, and non-test validation report."""
    manifest = build_protocol_manifest_from_data_root(data_root)
    freeze = build_r0_freeze_record(manifest, recorded_on=recorded_on)
    guard = FinalTestAccessGuard(
        protocol_manifest=manifest,
        freeze_record=freeze,
    )
    sealed_view = guard.sealed_view()
    expected_visible_fields = {
        "schema_version", "manifest_hash", "freeze_record_hash", "freeze_record_integrity",
    }
    if (
        sealed_view["access_state"] != "sealed"
        or sealed_view["freeze_record_integrity"] != "sealed_until_r3b"
        or set(sealed_view["visible_fields"]) != expected_visible_fields
        or any(field in sealed_view for field in FINAL_TEST_RELATIONSHIP_FIELDS)
    ):
        raise ProtocolError("R0 sealed view 不是 schema/integrity-only")
    boussinesq = manifest["datasets"]["boussinesq"]
    bouss_test = boussinesq["frame_partitions"]["test"]
    if not any(
        start <= BOUSSINESQ_AUDIT_FRAME < end
        for start, end in boussinesq["frame_partitions"][AUDIT_SPLIT_NAME]["ranges"]
    ):
        raise ProtocolError("R0 Boussinesq audit frame 未进入 development/audit partition")
    if any(
        start <= BOUSSINESQ_AUDIT_FRAME < start + manifest["window"]["t_win"]
        for start in bouss_test["window_starts"]
    ) or any(
        start <= BOUSSINESQ_AUDIT_FRAME < end
        for start, end in bouss_test["ranges"]
    ):
        raise ProtocolError("R0 Boussinesq frame 1300 进入了 test range/window")
    output = pathlib.Path(output_dir)
    manifest_path = write_json_artifact(output / "r0_protocol_manifest.json", manifest)
    freeze_path = write_json_artifact(output / "r0_freeze_record.json", freeze)
    checks = {
        "six_dataset_manifest": (
            "pass" if tuple(manifest["dataset_names"]) == DATASET_NAMES
            and len(manifest["datasets"]) == len(DATASET_NAMES) else "fail"
        ),
        "split_disjoint_and_complete": "pass",
        "uniform_50_10_40_frame_rule": "pass",
        "boussinesq_frame_1300_development_audit_only": "pass",
        "test_windows_do_not_contain_frame_1300": "pass",
        "final_test_relationships": manifest["access_policy"]["final_test_relationships"],
        "freeze_record_completeness": "pass",
        "sealed_view_contains_only_schema_and_integrity": "pass",
    }
    if any(value == "fail" for value in checks.values()):
        raise ProtocolError(f"R0 validation checks failed: {checks!r}")
    report = {
        "artifact_type": "criterion_ambiguity_r0_validation_report",
        "schema_version": "criterion-ambiguity-r0-validation-report-v1",
        "study_slug": STUDY_SLUG,
        "protocol_version": PROTOCOL_VERSION,
        "status": "pass",
        "checks": checks,
        "manifest_hash": manifest["manifest_hash"],
        "freeze_record_hash": freeze["freeze_record_hash"],
        "dataset_names": list(DATASET_NAMES),
        "source_snapshot_note": (
            "Frame identity was derived from dataset metadata only; no labels, "
            "coverage, class balance, agreement, conflict, predictions, or test metrics "
            "were read. Existing legacy frac metadata remains a re-preparation input."
        ),
    }
    report_path = write_json_artifact(output / "r0_validation_report.json", report)
    return {
        "manifest": manifest_path,
        "freeze_record": freeze_path,
        "validation_report": report_path,
        "manifest_hash": manifest["manifest_hash"],
        "freeze_record_hash": freeze["freeze_record_hash"],
    }


# ---------------------------------------------------------------------------
# Ticket 01: isolated standard-IVD/P95/P80 protocol


def _p95p80_source_contract() -> dict[str, Any]:
    """Return the immutable source contract for the new experiment version."""
    return {
        "percentile": {
            "source_id": P95P80_PERCENTILE_SOURCE_ID,
            "artifact_type": "criterion_ambiguity_p95p80_percentile_source",
            "identity": (
                f"{P95P80_PROTOCOL_VERSION}:source:"
                f"{P95P80_PERCENTILE_SOURCE_ID}:v1"
            ),
            "field": "standard_ivd",
            "support": "valid-fluid",
            "statistics_split": "train",
            "threshold_statistics": "train valid-fluid support only",
            "positive_rule": "IVD >= train valid-fluid P95",
            "negative_rule": "IVD <= train valid-fluid P80",
            "unknown_rule": (
                "train valid-fluid P80 < IVD < train valid-fluid P95"
            ),
            "calibration_policy": "frozen_train_thresholds",
            "final_test_policy": "sealed",
            "allowed_splits": ["train", "calibration", AUDIT_SPLIT_NAME],
            "test_values_allowed": False,
            "role": "dense weak supervision and operational reference",
            "legacy_sources_rejected": list(P95P80_LEGACY_SOURCE_IDS),
        },
        "standard_ivd": {
            "source_id": P95P80_HALLER_SOURCE_ID,
            "artifact_type": "haller_standard_ivd_evidence",
            "identity": (
                f"{P95P80_PROTOCOL_VERSION}:source:"
                f"{P95P80_HALLER_SOURCE_ID}:v1"
            ),
            "definition_id": P95P80_IVD_DEFINITION_ID,
            "formula": P95P80_IVD_FORMULA,
            "mean_scope": "declared valid-fluid region",
            "mean_quantity": "signed omega",
            "local_neighborhood_mean_allowed": False,
            "difference_scheme": {
                "vorticity": "omega = d(v)/d(x) - d(u)/d(y)",
                "grid": "uniform_rectilinear_xdim_ydim",
                "interior": "second_order_centered",
                "boundary": "first_order_one_sided",
            },
            "grid": {
                "coordinate_order": "x_then_y",
                "array_order": "row_y_then_col_x",
                "coordinates": ["xdim", "ydim"],
                "spacing_source": "declared physical coordinate axes",
                "nonuniform_policy": "reject",
            },
            "valid_fluid_policy": "declared_valid_fluid_and_finite_omega",
            "solid_policy": "excluded_from_mean_and_label_support",
            "solid_storage_policy": "zero_for_artifact_storage_only",
            "invalid_policy": "excluded_and_counted",
            "role": "standard Haller IVD evidence; sparse positive physics only",
            "allowed_splits": ["train", "calibration", AUDIT_SPLIT_NAME],
            "test_values_allowed": False,
            "dense_negative": False,
            "dense_evaluation_reference": False,
            "dense_haller_gt": False,
        },
        "label_postprocess": {
            "id": "connected_component_min_area",
            "kind": "thresholded_label_postprocess",
            "enabled": True,
            "min_area_cells": 25,
            "connectivity": 8,
            "applies_after_thresholding": True,
            "applies_to": "positive_components_only",
            "part_of_ivd_definition": False,
            "does_not_change_negative_rule": True,
        },
    }


def _p95p80_access_policy() -> dict[str, Any]:
    return {
        "development_allowed_splits": ["train", "calibration"],
        "audit_allowed_splits": [AUDIT_SPLIT_NAME],
        "final_test_relationships": "sealed_until_final_protocol_freeze",
        "sealed_bundle_allowed_operations": ["schema", "integrity_fingerprint"],
        "sealed_relationship_fields": list(P95P80_SEALED_RELATIONSHIP_FIELDS),
        "test_artifacts_read": False,
        "test_unlock": "one_time_final_evaluation_after_protocol_freeze",
        "boussinesq_audit_frame": BOUSSINESQ_AUDIT_FRAME,
        "boussinesq_audit_frame_excluded_from": [
            "test_manifest",
            "test_evidence",
            "evaluation_denominator",
            "final_summary",
        ],
    }


def _p95p80_source_id(source_id: Any) -> str:
    value = _nonempty_text(source_id, name="source_id")
    if value in P95P80_LEGACY_SOURCE_IDS:
        raise ProtocolIdentityError(
            f"legacy source_id={value!r} 不能进入 P95/P80 protocol；必须生成新 artifact identity"
        )
    if value not in P95P80_ALLOWED_SOURCE_IDS:
        raise ProtocolIdentityError(
            f"source_id={value!r} 不属于 P95/P80 protocol source set"
        )
    return value


def _p95p80_artifact_identity_payload(
    source_id: str,
    *,
    artifact_state: str,
) -> dict[str, Any]:
    source = _p95p80_source_id(source_id)
    state = _nonempty_text(artifact_state, name="artifact_state")
    if state not in {"declared", "candidate", "qualified", "frozen"}:
        raise ProtocolIdentityError(f"不支持的 P95/P80 artifact state={state!r}")
    return {
        "artifact_id": f"{P95P80_PROTOCOL_VERSION}:source:{source}:v1",
        "artifact_type": "criterion_ambiguity_p95p80_source_artifact",
        "artifact_schema_version": P95P80_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": P95P80_PROTOCOL_VERSION,
        "source_id": source,
        "artifact_version": "v1",
        "state": state,
        "provenance_schema_version": P95P80_PROVENANCE_SCHEMA_VERSION,
        "reuse_policy": {
            "legacy_artifacts_formal_source": False,
            "overwrite": "reject",
            "upgrade": "new_identity_required",
        },
        "test_policy": "no_final_test_values_before_final_protocol_freeze",
    }


def build_p95p80_artifact_identity(
    source_id: str,
    *,
    artifact_state: str = "declared",
) -> dict[str, Any]:
    """Create a source identity that cannot alias the historical artifacts."""
    payload = _p95p80_artifact_identity_payload(
        source_id, artifact_state=artifact_state
    )
    payload["identity_fingerprint"] = _json_hash(payload)
    return payload


def validate_p95p80_artifact_identity(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a P95/P80 source identity before it is loaded or published."""
    candidate = _copy_json(dict(_mapping(identity, name="artifact_identity")))
    required = tuple(_p95p80_artifact_identity_payload(
        P95P80_PERCENTILE_SOURCE_ID, artifact_state="declared"
    )) + ("identity_fingerprint",)
    missing = [field for field in required if field not in candidate]
    if missing:
        raise ProtocolIdentityError(f"artifact identity 缺少字段：{missing!r}")
    # Check identity fields before the digest so a legacy input fails with the
    # actionable protocol/source error instead of being treated as a generic
    # checksum mismatch.
    if candidate["protocol_version"] != P95P80_PROTOCOL_VERSION:
        raise ProtocolIdentityError(
            "artifact identity protocol_version 不是当前 P95/P80 protocol"
        )
    source = _p95p80_source_id(candidate["source_id"])
    expected = _p95p80_artifact_identity_payload(
        source, artifact_state=candidate["state"]
    )
    if candidate.get("artifact_id") != expected["artifact_id"]:
        raise ProtocolIdentityError("artifact_id 与 protocol/source identity 不匹配")
    for field, expected_value in expected.items():
        if candidate.get(field) != expected_value:
            raise ProtocolIdentityError(
                f"artifact identity {field} 与当前 P95/P80 contract 不匹配"
            )
    payload = copy.deepcopy(candidate)
    declared = payload.pop("identity_fingerprint", None)
    if not isinstance(declared, str) or _json_hash(payload) != declared:
        raise ProtocolIdentityError("artifact identity fingerprint 校验失败")
    return candidate


def build_p95p80_provenance(
    *,
    dataset_name: str,
    split_name: str,
    frame_index: int,
    geometry_fingerprint: str,
    input_fingerprint: str,
    valid_fluid_fingerprint: str,
    source_id: str = P95P80_PERCENTILE_SOURCE_ID,
    protocol_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build provenance for a development-only source artifact."""
    manifest = validate_p95p80_protocol_manifest(protocol_manifest)
    source = _p95p80_source_id(source_id)
    dataset = _nonempty_text(dataset_name, name="dataset_name")
    if dataset not in DATASET_NAMES:
        raise ProtocolIdentityError(
            f"P95/P80 provenance dataset={dataset!r} 不在固定六数据集内"
        )
    split = _nonempty_text(split_name, name="split_name")
    frame = _strict_int(frame_index, name="frame_index")
    if split == "test":
        raise FinalTestAccessError(
            "P95/P80 provenance 在 final protocol freeze 前禁止 test artifact"
        )
    if split not in {"train", "calibration", AUDIT_SPLIT_NAME}:
        raise ProtocolIdentityError(
            f"P95/P80 provenance split={split!r} 不属于 development/audit boundary"
        )
    if split == AUDIT_SPLIT_NAME and (
        dataset != "boussinesq" or frame != BOUSSINESQ_AUDIT_FRAME
    ):
        raise ProtocolIdentityError(
            "development_audit provenance 只能是 boussinesq frame 1300"
        )
    if dataset == "boussinesq" and frame == BOUSSINESQ_AUDIT_FRAME and split != AUDIT_SPLIT_NAME:
        raise ProtocolIdentityError(
            "boussinesq frame 1300 必须保持 development_audit，不得混入 train/calibration"
        )
    artifact = build_p95p80_artifact_identity(source)
    result: dict[str, Any] = {
        "schema_version": P95P80_PROVENANCE_SCHEMA_VERSION,
        "protocol_version": P95P80_PROTOCOL_VERSION,
        "protocol_manifest_hash": manifest["manifest_hash"],
        "source_id": source,
        "artifact_id": artifact["artifact_id"],
        "dataset_name": dataset,
        "split_name": split,
        "frame_index": frame,
        "geometry_fingerprint": _nonempty_text(
            geometry_fingerprint, name="geometry_fingerprint"
        ),
        "input_fingerprint": _nonempty_text(
            input_fingerprint, name="input_fingerprint"
        ),
        "valid_fluid_fingerprint": _nonempty_text(
            valid_fluid_fingerprint, name="valid_fluid_fingerprint"
        ),
        "final_test_relationships": "not_read",
    }
    result["integrity_fingerprint"] = _json_hash(result)
    validate_p95p80_provenance(result, protocol_manifest=manifest)
    return result


def validate_p95p80_provenance(
    provenance: Mapping[str, Any],
    *,
    protocol_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject legacy, test, and frame-1300-mixed source provenance."""
    candidate = _copy_json(dict(_mapping(provenance, name="p95p80_provenance")))
    missing = [field for field in P95P80_PROVENANCE_FIELDS if field not in candidate]
    if missing:
        raise ProtocolIdentityError(f"P95/P80 provenance 缺少字段：{missing!r}")
    if candidate.get("schema_version") != P95P80_PROVENANCE_SCHEMA_VERSION:
        raise ProtocolIdentityError("P95/P80 provenance schema_version 不匹配")
    if candidate.get("protocol_version") != P95P80_PROTOCOL_VERSION:
        raise ProtocolIdentityError("P95/P80 provenance protocol_version 不匹配")
    manifest = validate_p95p80_protocol_manifest(protocol_manifest)
    if candidate.get("protocol_manifest_hash") != manifest["manifest_hash"]:
        raise ProtocolIdentityError(
            "P95/P80 provenance protocol_manifest_hash 不匹配"
        )
    source = _p95p80_source_id(candidate.get("source_id"))
    if candidate.get("artifact_id") != (
        f"{P95P80_PROTOCOL_VERSION}:source:{source}:v1"
    ):
        raise ProtocolIdentityError("P95/P80 provenance artifact_id 不匹配")
    split = _nonempty_text(candidate.get("split_name"), name="split_name")
    frame = _strict_int(candidate.get("frame_index"), name="frame_index")
    dataset = _nonempty_text(candidate.get("dataset_name"), name="dataset_name")
    if dataset not in DATASET_NAMES:
        raise ProtocolIdentityError(
            f"P95/P80 provenance dataset={dataset!r} 不在固定六数据集内"
        )
    if split == "test":
        raise FinalTestAccessError(
            "P95/P80 provenance final-test relationship 仍 sealed"
        )
    if split not in {"train", "calibration", AUDIT_SPLIT_NAME}:
        raise ProtocolIdentityError(f"P95/P80 provenance split={split!r} 非法")
    if split == AUDIT_SPLIT_NAME and (
        dataset != "boussinesq" or frame != BOUSSINESQ_AUDIT_FRAME
    ):
        raise ProtocolIdentityError(
            "development_audit provenance 只能是 boussinesq frame 1300"
        )
    if dataset == "boussinesq" and frame == BOUSSINESQ_AUDIT_FRAME and split != AUDIT_SPLIT_NAME:
        raise ProtocolIdentityError("frame 1300 不得进入 train/calibration provenance")
    if candidate.get("final_test_relationships") != "not_read":
        raise FinalTestAccessError("P95/P80 provenance 必须声明 final-test not_read")
    if dataset not in manifest["datasets"]:
        raise ProtocolIdentityError(
            f"P95/P80 provenance dataset={dataset!r} 不在 protocol manifest"
        )
    partition_name = AUDIT_SPLIT_NAME if split == AUDIT_SPLIT_NAME else split
    ranges = manifest["datasets"][dataset]["frame_partitions"][partition_name]["ranges"]
    if not any(start <= frame < end for start, end in ranges):
        raise ProtocolIdentityError(
            f"P95/P80 provenance dataset={dataset} frame={frame} "
            f"不在 manifest split={partition_name} partition"
        )
    payload = copy.deepcopy(candidate)
    declared = payload.pop("integrity_fingerprint", None)
    if not isinstance(declared, str) or _json_hash(payload) != declared:
        raise ProtocolIdentityError("P95/P80 provenance integrity fingerprint 校验失败")
    return candidate


def _rebind_p95p80_frame_manifest(
    frame_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    record = validate_frame_manifest(frame_manifest)
    source = dict(record["source_identity"])
    source.pop("fingerprint", None)
    source.pop("data_files", None)
    source.pop("metadata_split_mode", None)
    source.pop("protocol_alignment", None)
    source["kind"] = "p95p80_metadata_only_snapshot"
    source["source_role"] = "frame_identity_only"
    source["legacy_artifacts_formal_source"] = False
    source["legacy_artifact_contents_loaded"] = False
    source.update({
        "protocol_version": P95P80_PROTOCOL_VERSION,
        "snapshot_role": "metadata_only_frame_identity",
        "labels_read": False,
        "legacy_identity_reused": False,
    })
    source["fingerprint"] = _json_hash(source)
    return build_frame_manifest(
        record["total_frames"],
        record["dataset_name"],
        source_identity=source,
        audit_frames=record["development_audit_frames"],
        t_win=record["window"]["t_win"],
        window_step=record["window"]["window_step"],
    )


def build_p95p80_protocol_manifest(
    frame_manifests: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    data_root: str | pathlib.Path | None = None,
    t_win: int = DEFAULT_T_WIN,
    window_step: int = DEFAULT_WINDOW_STEP,
) -> dict[str, Any]:
    """Build the new protocol manifest without mutating historical R0 inputs."""
    if isinstance(frame_manifests, Mapping):
        supplied = dict(frame_manifests)
    else:
        supplied = {}
        for item in _sequence(frame_manifests, name="frame_manifests"):
            record = validate_frame_manifest(item)
            supplied[record["dataset_name"]] = record
    if set(supplied) != set(DATASET_NAMES):
        raise ManifestValidationError(
            "P95/P80 protocol manifest 必须覆盖固定六个 dataset"
        )
    records: dict[str, dict[str, Any]] = {}
    protocol_t_win = _strict_int(t_win, name="t_win")
    protocol_window_step = _strict_int(window_step, name="window_step")
    for name in DATASET_NAMES:
        rebound = _rebind_p95p80_frame_manifest(supplied[name])
        if rebound["window"]["t_win"] != protocol_t_win or rebound["window"]["window_step"] != protocol_window_step:
            raise ManifestValidationError(
                f"dataset={name} window 与 P95/P80 protocol-level window 不一致"
            )
        records[name] = rebound
    manifest: dict[str, Any] = {
        "artifact_type": "criterion_ambiguity_p95p80_protocol_manifest",
        "schema_version": P95P80_PROTOCOL_SCHEMA_VERSION,
        "study_slug": STUDY_SLUG,
        "protocol_version": P95P80_PROTOCOL_VERSION,
        "dataset_names": list(DATASET_NAMES),
        "split_rule": _copy_json(_SPLIT_RULE),
        "window": {
            "frame_unit": "index",
            "half_open": True,
            "complete_only": True,
            "t_win": protocol_t_win,
            "window_step": protocol_window_step,
            "audit_frame_cannot_be_in_test_window": True,
        },
        "data_root": (
            None if data_root is None else str(pathlib.Path(data_root).resolve())
        ),
        "datasets": records,
        "sources": _p95p80_source_contract(),
        "artifact_identities": [
            build_p95p80_artifact_identity(source)
            for source in P95P80_ALLOWED_SOURCE_IDS
        ],
        "development_audit_rule": {
            "dataset": "boussinesq",
            "frame": BOUSSINESQ_AUDIT_FRAME,
            "classification": "development/audit",
            "test_manifest": "excluded",
            "test_evidence": "excluded",
            "evaluation_denominator": "excluded",
            "final_summary": "excluded",
        },
        "access_policy": _p95p80_access_policy(),
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    return validate_p95p80_protocol_manifest(manifest)


def validate_p95p80_protocol_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate identity, source semantics, split provenance, and test sealing."""
    candidate = _copy_json(dict(_mapping(manifest, name="p95p80_protocol_manifest")))
    required = (
        "artifact_type",
        "schema_version",
        "study_slug",
        "protocol_version",
        "dataset_names",
        "split_rule",
        "window",
        "datasets",
        "sources",
        "artifact_identities",
        "development_audit_rule",
        "access_policy",
        "manifest_hash",
    )
    missing = [field for field in required if field not in candidate]
    if missing:
        raise ManifestValidationError(f"P95/P80 manifest 缺少字段：{missing!r}")
    if candidate["artifact_type"] != "criterion_ambiguity_p95p80_protocol_manifest":
        raise ProtocolIdentityError("P95/P80 protocol manifest artifact_type 不匹配")
    if candidate["schema_version"] != P95P80_PROTOCOL_SCHEMA_VERSION:
        raise ProtocolIdentityError("P95/P80 protocol manifest schema_version 不匹配")
    if candidate["study_slug"] != STUDY_SLUG or candidate["protocol_version"] != P95P80_PROTOCOL_VERSION:
        raise ProtocolIdentityError("P95/P80 protocol manifest protocol identity 不匹配")
    names = tuple(_nonempty_text(name, name="dataset_name") for name in _sequence(
        candidate["dataset_names"], name="dataset_names"
    ))
    if names != DATASET_NAMES:
        raise ManifestValidationError("P95/P80 manifest dataset 顺序/集合不匹配")
    if candidate["split_rule"] != _SPLIT_RULE:
        raise ManifestValidationError("P95/P80 manifest split_rule 不匹配")
    window = _mapping(candidate["window"], name="p95p80.window")
    if (
        window.get("frame_unit") != "index"
        or window.get("half_open") is not True
        or window.get("complete_only") is not True
        or window.get("audit_frame_cannot_be_in_test_window") is not True
    ):
        raise ManifestValidationError("P95/P80 window contract 不完整")
    t_win = _strict_int(window.get("t_win"), name="p95p80.window.t_win")
    window_step = _strict_int(window.get("window_step"), name="p95p80.window.window_step")
    datasets = _mapping(candidate["datasets"], name="p95p80.datasets")
    if set(datasets) != set(DATASET_NAMES):
        raise ManifestValidationError("P95/P80 manifest datasets 集合不匹配")
    for name in DATASET_NAMES:
        record = validate_frame_manifest(datasets[name])
        if record["dataset_name"] != name:
            raise ManifestValidationError(f"P95/P80 datasets[{name}] identity 不一致")
        if record["window"]["t_win"] != t_win or record["window"]["window_step"] != window_step:
            raise ManifestValidationError(f"P95/P80 dataset={name} window 不一致")
        source = _mapping(record["source_identity"], name=f"{name}.source_identity")
        if source.get("protocol_version") != P95P80_PROTOCOL_VERSION:
            raise ProtocolIdentityError(
                f"dataset={name} source_identity 不能复用历史 protocol"
            )
        if (
            source.get("legacy_identity_reused") is not False
            or source.get("labels_read") is not False
            or source.get("source_role") != "frame_identity_only"
            or source.get("legacy_artifacts_formal_source") is not False
            or source.get("legacy_artifact_contents_loaded") is not False
        ):
            raise ProtocolIdentityError(
                f"dataset={name} source_identity 缺少 no-reuse/metadata-only 约束"
            )
    if candidate["sources"] != _p95p80_source_contract():
        raise ProtocolIdentityError("P95/P80 source contract 被修改或来源漂移")
    artifact_identities = _sequence(
        candidate["artifact_identities"], name="p95p80.artifact_identities"
    )
    expected_identities = [
        build_p95p80_artifact_identity(source)
        for source in P95P80_ALLOWED_SOURCE_IDS
    ]
    if list(artifact_identities) != expected_identities:
        raise ProtocolIdentityError("P95/P80 artifact identities 不匹配")
    audit_rule = _mapping(candidate["development_audit_rule"], name="p95p80.audit_rule")
    expected_audit_rule = {
        "dataset": "boussinesq",
        "frame": BOUSSINESQ_AUDIT_FRAME,
        "classification": "development/audit",
        "test_manifest": "excluded",
        "test_evidence": "excluded",
        "evaluation_denominator": "excluded",
        "final_summary": "excluded",
    }
    if dict(audit_rule) != expected_audit_rule:
        raise ManifestValidationError("P95/P80 frame 1300 audit exclusion 不完整")
    access = _mapping(candidate["access_policy"], name="p95p80.access_policy")
    if tuple(access.get("development_allowed_splits", ())) != ("train", "calibration"):
        raise FinalTestAccessError("P95/P80 development boundary 只能读取 train/calibration")
    if tuple(access.get("audit_allowed_splits", ())) != (AUDIT_SPLIT_NAME,):
        raise ManifestValidationError("P95/P80 audit split boundary 不完整")
    if access.get("final_test_relationships") != "sealed_until_final_protocol_freeze":
        raise FinalTestAccessError("P95/P80 final-test relationships 必须 sealed")
    if tuple(access.get("sealed_bundle_allowed_operations", ())) != (
        "schema", "integrity_fingerprint"
    ):
        raise ManifestValidationError("P95/P80 sealed bundle policy 不完整")
    if tuple(access.get("sealed_relationship_fields", ())) != P95P80_SEALED_RELATIONSHIP_FIELDS:
        raise ManifestValidationError("P95/P80 sealed relationship fields 不完整")
    if access.get("test_artifacts_read") is not False:
        raise FinalTestAccessError("P95/P80 manifest 已读取 test artifact")
    payload = copy.deepcopy(candidate)
    declared_hash = payload.pop("manifest_hash", None)
    if not isinstance(declared_hash, str) or _json_hash(payload) != declared_hash:
        raise ManifestValidationError("P95/P80 manifest_hash 校验失败")
    return candidate


def build_p95p80_protocol_manifest_from_data_root(
    data_root: str | pathlib.Path,
    *,
    metadata_filename: str = "meta.json",
    t_win: int = DEFAULT_T_WIN,
    window_step: int = DEFAULT_WINDOW_STEP,
) -> dict[str, Any]:
    """Build a new metadata-only manifest; labels and test values are untouched."""
    legacy_snapshot = build_protocol_manifest_from_data_root(
        data_root,
        metadata_filename=metadata_filename,
        t_win=t_win,
        window_step=window_step,
    )
    return build_p95p80_protocol_manifest(
        legacy_snapshot["datasets"],
        data_root=data_root,
        t_win=t_win,
        window_step=window_step,
    )


def build_p95p80_freeze_record(
    protocol_manifest: Mapping[str, Any],
    *,
    recorded_on: str = "2026-09-15",
) -> dict[str, Any]:
    """Freeze only Ticket 01 decisions; final-test remains sealed."""
    manifest = validate_p95p80_protocol_manifest(protocol_manifest)
    record: dict[str, Any] = {
        "artifact_type": "criterion_ambiguity_p95p80_freeze_record",
        "schema_version": P95P80_FREEZE_SCHEMA_VERSION,
        "study_slug": STUDY_SLUG,
        "protocol_version": P95P80_PROTOCOL_VERSION,
        "recorded_on": _nonempty_text(recorded_on, name="recorded_on"),
        "status": "frozen",
        "freeze_scope": "ticket_01_protocol_source_artifact_identity",
        "protocol_manifest": {
            "artifact_type": manifest["artifact_type"],
            "schema_version": manifest["schema_version"],
            "protocol_version": manifest["protocol_version"],
            "manifest_hash": manifest["manifest_hash"],
        },
        "data": {
            "dataset_names": list(DATASET_NAMES),
            "split_rule": _copy_json(_SPLIT_RULE),
            "window": _copy_json(manifest["window"]),
            "geometry": {
                "coordinate_order": "x_then_y",
                "array_order": "row_y_then_col_x",
                "valid_fluid": "declared_valid_fluid_and_finite_omega",
                "solid": "excluded_from_mean_and_label_support",
                "invalid": "excluded_and_counted",
            },
        },
        "sources": _copy_json(manifest["sources"]),
        "artifact_identities": _copy_json(manifest["artifact_identities"]),
        "provenance": {
            "schema_version": P95P80_PROVENANCE_SCHEMA_VERSION,
            "required_fields": list(P95P80_PROVENANCE_FIELDS),
            "development_splits": ["train", "calibration"],
            "audit_split": AUDIT_SPLIT_NAME,
            "final_test_values_read": False,
            "legacy_artifact_reuse": False,
        },
        "historical_boundary": {
            "historical_protocols": list(P95P80_LEGACY_PROTOCOL_VERSIONS),
            "historical_sources": list(P95P80_LEGACY_SOURCE_IDS),
            "formal_source_reuse": False,
            "overwrite_policy": "reject",
            "upgrade_policy": "new_protocol_and_artifact_identity_required",
        },
        "evidence": {
            "haller_literature": {
                "title": "Defining coherent vortices objectively from the vorticity",
                "zotero_key": "L2PX3NQX",
                "year": 2016,
                "ivd_equation": "Eq. (6.2)",
                "ivd_formula": P95P80_IVD_FORMULA,
                "mean_quantity": "signed omega",
                "mean_scope": "declared valid-fluid region",
                "canonical_project_grid_parameters": False,
            }
        },
        "access_policy": _p95p80_access_policy(),
    }
    record["freeze_record_hash"] = _json_hash(record)
    return validate_p95p80_freeze_record(record, protocol_manifest=manifest)


def validate_p95p80_freeze_record(
    record: Mapping[str, Any],
    *,
    protocol_manifest: Mapping[str, Any] | None = None,
    require_final_test_open: bool = False,
) -> dict[str, Any]:
    """Validate the Ticket 01 freeze record and keep its test lock closed."""
    candidate = _copy_json(dict(_mapping(record, name="p95p80_freeze_record")))
    required = (
        "artifact_type",
        "schema_version",
        "study_slug",
        "protocol_version",
        "recorded_on",
        "status",
        "freeze_scope",
        "protocol_manifest",
        "data",
        "sources",
        "artifact_identities",
        "provenance",
        "historical_boundary",
        "evidence",
        "access_policy",
        "freeze_record_hash",
    )
    missing = [field for field in required if field not in candidate]
    if missing:
        raise FreezeRecordValidationError(
            f"P95/P80 freeze record 缺少字段：{missing!r}"
        )
    if candidate["artifact_type"] != "criterion_ambiguity_p95p80_freeze_record":
        raise ProtocolIdentityError("P95/P80 freeze artifact_type 不匹配")
    if candidate["schema_version"] != P95P80_FREEZE_SCHEMA_VERSION:
        raise ProtocolIdentityError("P95/P80 freeze schema_version 不匹配")
    if candidate["study_slug"] != STUDY_SLUG or candidate["protocol_version"] != P95P80_PROTOCOL_VERSION:
        raise ProtocolIdentityError("P95/P80 freeze protocol identity 不匹配")
    if candidate["status"] != "frozen" or candidate["freeze_scope"] != "ticket_01_protocol_source_artifact_identity":
        raise FreezeRecordValidationError("P95/P80 freeze status/scope 不匹配")
    manifest_info = _mapping(candidate["protocol_manifest"], name="p95p80.protocol_manifest")
    if (
        manifest_info.get("artifact_type")
        != "criterion_ambiguity_p95p80_protocol_manifest"
        or manifest_info.get("schema_version") != P95P80_PROTOCOL_SCHEMA_VERSION
        or manifest_info.get("protocol_version") != P95P80_PROTOCOL_VERSION
    ):
        raise ProtocolIdentityError("P95/P80 freeze 不能引用历史 protocol manifest")
    manifest_hash = _nonempty_text(manifest_info.get("manifest_hash"), name="manifest_hash")
    if protocol_manifest is not None:
        manifest = validate_p95p80_protocol_manifest(protocol_manifest)
        if manifest_hash != manifest["manifest_hash"]:
            raise FreezeRecordValidationError("P95/P80 freeze 与 protocol manifest hash 不匹配")
    data = _mapping(candidate["data"], name="p95p80.data")
    if tuple(data.get("dataset_names", ())) != DATASET_NAMES:
        raise FreezeRecordValidationError("P95/P80 freeze dataset_names 不匹配")
    if data.get("split_rule") != _SPLIT_RULE:
        raise FreezeRecordValidationError("P95/P80 freeze split_rule 不匹配")
    data_window = _mapping(data.get("window"), name="p95p80.data.window")
    if (
        data_window.get("frame_unit") != "index"
        or data_window.get("half_open") is not True
        or data_window.get("complete_only") is not True
        or data_window.get("audit_frame_cannot_be_in_test_window") is not True
        or _strict_int(data_window.get("t_win"), name="p95p80.data.window.t_win") <= 0
        or _strict_int(data_window.get("window_step"), name="p95p80.data.window.window_step") <= 0
    ):
        raise FreezeRecordValidationError("P95/P80 freeze window contract 不完整")
    if protocol_manifest is not None and dict(data_window) != dict(manifest["window"]):
        raise FreezeRecordValidationError("P95/P80 freeze window 与 protocol manifest 不匹配")
    expected_geometry = {
        "coordinate_order": "x_then_y",
        "array_order": "row_y_then_col_x",
        "valid_fluid": "declared_valid_fluid_and_finite_omega",
        "solid": "excluded_from_mean_and_label_support",
        "invalid": "excluded_and_counted",
    }
    if dict(_mapping(data.get("geometry"), name="p95p80.data.geometry")) != expected_geometry:
        raise FreezeRecordValidationError("P95/P80 freeze geometry/solid semantics 不完整")
    if candidate["sources"] != _p95p80_source_contract():
        raise ProtocolIdentityError("P95/P80 freeze source contract 漂移")
    identities = _sequence(candidate["artifact_identities"], name="p95p80.artifact_identities")
    expected_identities = [
        build_p95p80_artifact_identity(source)
        for source in P95P80_ALLOWED_SOURCE_IDS
    ]
    if list(identities) != expected_identities:
        raise ProtocolIdentityError("P95/P80 freeze artifact identities 不匹配")
    provenance = _mapping(candidate["provenance"], name="p95p80.provenance")
    if provenance.get("schema_version") != P95P80_PROVENANCE_SCHEMA_VERSION:
        raise ProtocolIdentityError("P95/P80 provenance schema_version 不匹配")
    if tuple(provenance.get("required_fields", ())) != P95P80_PROVENANCE_FIELDS:
        raise ProtocolIdentityError("P95/P80 provenance required fields 不完整")
    if provenance.get("final_test_values_read") is not False or provenance.get("legacy_artifact_reuse") is not False:
        raise FinalTestAccessError("P95/P80 freeze 不能声明读取 test 或复用历史 artifact")
    boundary = _mapping(candidate["historical_boundary"], name="p95p80.historical_boundary")
    if boundary.get("formal_source_reuse") is not False or boundary.get("overwrite_policy") != "reject":
        raise ProtocolIdentityError("P95/P80 historical artifact boundary 不完整")
    if boundary.get("upgrade_policy") != "new_protocol_and_artifact_identity_required":
        raise ProtocolIdentityError("P95/P80 historical artifact upgrade policy 不完整")
    evidence = _mapping(candidate["evidence"], name="p95p80.evidence")
    literature = _mapping(evidence.get("haller_literature"), name="p95p80.haller_literature")
    if (
        literature.get("title") != "Defining coherent vortices objectively from the vorticity"
        or literature.get("zotero_key") != "L2PX3NQX"
        or literature.get("year") != 2016
        or literature.get("ivd_equation") != "Eq. (6.2)"
        or literature.get("ivd_formula") != P95P80_IVD_FORMULA
    ):
        raise ProtocolIdentityError("Haller Zotero/formula provenance 不匹配")
    if literature.get("mean_quantity") != "signed omega" or literature.get("canonical_project_grid_parameters") is not False:
        raise ProtocolIdentityError("Haller signed-mean/literature boundary 不完整")
    access = _mapping(candidate["access_policy"], name="p95p80.access_policy")
    if dict(access) != _p95p80_access_policy():
        raise FinalTestAccessError("P95/P80 freeze access policy 不完整或已解封")
    if require_final_test_open:
        raise FinalTestAccessError(
            "P95/P80 Ticket 01 freeze 仍 sealed；final-test 需后续独立 final protocol freeze"
        )
    payload = copy.deepcopy(candidate)
    declared_hash = payload.pop("freeze_record_hash", None)
    if not isinstance(declared_hash, str) or _json_hash(payload) != declared_hash:
        raise FreezeRecordValidationError("P95/P80 freeze_record_hash 校验失败")
    return candidate


def _validate_any_protocol_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(manifest, name="protocol_manifest")
    if value.get("protocol_version") == P95P80_PROTOCOL_VERSION or value.get(
        "schema_version"
    ) == P95P80_PROTOCOL_SCHEMA_VERSION:
        return validate_p95p80_protocol_manifest(value)
    return validate_protocol_manifest(value)


def _validate_any_freeze_record(
    record: Mapping[str, Any],
    *,
    protocol_manifest: Mapping[str, Any] | None = None,
    require_final_test_open: bool = False,
) -> dict[str, Any]:
    value = _mapping(record, name="freeze_record")
    if value.get("protocol_version") == P95P80_PROTOCOL_VERSION or value.get(
        "schema_version"
    ) == P95P80_FREEZE_SCHEMA_VERSION:
        return validate_p95p80_freeze_record(
            value,
            protocol_manifest=protocol_manifest,
            require_final_test_open=require_final_test_open,
        )
    return validate_freeze_record(
        value,
        protocol_manifest=protocol_manifest,
        require_final_test_open=require_final_test_open,
    )


def read_p95p80_protocol_manifest(path: str | pathlib.Path) -> dict[str, Any]:
    value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return validate_p95p80_protocol_manifest(value)


def _write_new_json_artifact(path: pathlib.Path, value: Mapping[str, Any]) -> pathlib.Path:
    if path.exists():
        raise FileExistsError(
            f"P95/P80 artifact 已存在，拒绝 overwrite/upgrade：{path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_p95p80_artifacts(
    data_root: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    recorded_on: str = "2026-09-15",
) -> dict[str, pathlib.Path | str]:
    """Publish new Ticket 01 artifacts without overwriting historical R0 files."""
    manifest = build_p95p80_protocol_manifest_from_data_root(data_root)
    freeze = build_p95p80_freeze_record(manifest, recorded_on=recorded_on)
    validate_p95p80_freeze_record(freeze, protocol_manifest=manifest)
    output = pathlib.Path(output_dir)
    manifest_path = output / "p95p80_protocol_manifest.json"
    freeze_path = output / "p95p80_freeze_record.json"
    report_path = output / "p95p80_validation_report.json"
    report = {
        "artifact_type": "criterion_ambiguity_p95p80_validation_report",
        "schema_version": "criterion-ambiguity-p95p80-validation-report-v1",
        "study_slug": STUDY_SLUG,
        "protocol_version": P95P80_PROTOCOL_VERSION,
        "status": "pass",
        "verification_scope": "protocol_contract_only_metadata_only",
        "numeric_source_values_read": False,
        "numeric_checks_deferred": [
            "standard_ivd_values_and_thresholds",
            "train_valid_fluid_p95_p80_quantiles",
            "weak_label_materialization",
        ],
        "final_test_relationships": "sealed_until_final_protocol_freeze",
        "checks": {
            "six_dataset_manifest": "pass",
            "uniform_50_10_40_frame_rule": "pass",
            "p95p80_train_only_threshold_statistics": "pass",
            "standard_ivd_signed_fluid_mean": "pass",
            "local_neighborhood_ivd_forbidden": "pass",
            "area_filter_separate_from_ivd": "pass",
            "historical_artifact_reuse_rejected": "pass",
            "frame_1300_development_audit_only": "pass",
            "test_windows_do_not_contain_frame_1300": "pass",
            "final_test_relationships": "sealed_until_final_protocol_freeze",
        },
        "manifest_hash": manifest["manifest_hash"],
        "freeze_record_hash": freeze["freeze_record_hash"],
        "dataset_names": list(DATASET_NAMES),
        "test_artifacts_read": False,
        "source_snapshot_note": (
            "Frame identity was derived from metadata only; no final-test labels, "
            "coverage, relationships, predictions, or model metrics were read."
        ),
    }
    targets = (manifest_path, freeze_path, report_path)
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "P95/P80 artifacts 已存在，拒绝 overwrite/upgrade："
            + ", ".join(str(path) for path in existing)
        )
    for path, value in zip(targets, (manifest, freeze, report)):
        _write_new_json_artifact(path, value)
    return {
        "manifest": manifest_path,
        "freeze_record": freeze_path,
        "validation_report": report_path,
        "manifest_hash": manifest["manifest_hash"],
        "freeze_record_hash": freeze["freeze_record_hash"],
    }


def _record_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _record_frame_index(value: Any) -> int | None:
    direct = _record_value(value, "frame_index")
    if direct is not None:
        return _strict_int(direct, name="frame_index")
    provenance = _record_value(value, "provenance")
    if isinstance(provenance, Mapping) and provenance.get("frame_index") is not None:
        return _strict_int(provenance["frame_index"], name="provenance.frame_index")
    return None


class FinalTestAccessGuard:
    """Gate relationship-bearing final-test reads.

    The guard checks the freeze record *before* invoking a producer callback,
    so a development path cannot load values and merely hide them at report
    time.  A sealed view intentionally exposes only schema and fingerprints.
    """

    def __init__(
        self,
        *,
        protocol_manifest: Mapping[str, Any] | None = None,
        freeze_record: Mapping[str, Any] | None = None,
    ) -> None:
        self.protocol_manifest = (
            None
            if protocol_manifest is None
            else _validate_any_protocol_manifest(protocol_manifest)
        )
        self.freeze_record = None if freeze_record is None else _copy_json(dict(freeze_record))

    def sealed_view(self) -> dict[str, Any]:
        """Return the only information visible before R3b opens relationships."""
        manifest_hash = None if self.protocol_manifest is None else self.protocol_manifest["manifest_hash"]
        freeze_state = "missing"
        freeze_hash = None
        if self.freeze_record is not None:
            freeze_hash = self.freeze_record.get("freeze_record_hash")
            try:
                validated = _validate_any_freeze_record(
                    self.freeze_record,
                    protocol_manifest=self.protocol_manifest,
                )
                freeze_state = validated["access_policy"]["final_test_relationships"]
            except (ProtocolError, FinalTestAccessError):
                freeze_state = "missing_or_incomplete"
        return {
            "access_state": "sealed",
            "schema_version": (
                None if self.protocol_manifest is None
                else self.protocol_manifest["schema_version"]
            ),
            "manifest_hash": manifest_hash,
            "freeze_record_hash": freeze_hash,
            "freeze_record_integrity": freeze_state,
            "visible_fields": ["schema_version", "manifest_hash", "freeze_record_hash", "freeze_record_integrity"],
        }

    def authorize_relationships(self) -> dict[str, Any]:
        """Authorize a relationship-bearing read only for an explicitly open record."""
        if self.protocol_manifest is None:
            raise FinalTestAccessError(
                "final-test relationships sealed: validated protocol manifest is required"
            )
        if self.freeze_record is None:
            raise FinalTestAccessError(
                "final-test relationships sealed: complete freeze record is required"
            )
        try:
            record = _validate_any_freeze_record(
                self.freeze_record,
                protocol_manifest=self.protocol_manifest,
                require_final_test_open=True,
            )
        except ProtocolError as exc:
            raise FinalTestAccessError(
                f"final-test relationships sealed: freeze record missing, incomplete, or not open ({exc})"
            ) from exc
        return {
            "access_state": "open_for_r3b",
            "protocol_version": record["protocol_version"],
            "freeze_record_hash": record["freeze_record_hash"],
        }

    def read_relationships(self, producer: Callable[[], Any]) -> Any:
        """Authorize before calling ``producer``; producer may return test values."""
        if not callable(producer):
            raise TypeError("final-test producer 必须是 callable")
        self.authorize_relationships()
        return producer()

    def validate_test_records(self, records: Iterable[Any]) -> tuple[Any, ...]:
        """Validate frame provenance without touching labels or relationship values."""
        if self.protocol_manifest is None:
            raise FinalTestAccessError(
                "protocol manifest is required to validate final-test frame provenance"
            )
        result = tuple(records)
        for ordinal, record in enumerate(result):
            dataset_name = _record_value(record, "dataset_name")
            if dataset_name not in DATASET_NAMES:
                raise ManifestValidationError(
                    f"test record[{ordinal}] dataset_name 不在固定六数据集内"
                )
            frame_index = _record_frame_index(record)
            if frame_index is None:
                raise ManifestValidationError(
                    f"test record[{ordinal}] 缺少 frame_index provenance"
                )
            dataset_manifest = self.protocol_manifest["datasets"][dataset_name]
            test_ranges = dataset_manifest["frame_partitions"]["test"]["ranges"]
            if not any(start <= frame_index < end for start, end in test_ranges):
                if dataset_name == "boussinesq" and frame_index == BOUSSINESQ_AUDIT_FRAME:
                    raise ManifestValidationError(
                        "Boussinesq frame 1300 是 development/audit frame，不能进入 test evidence"
                    )
                raise ManifestValidationError(
                    f"dataset={dataset_name} frame={frame_index} 不在 test manifest"
                )
        return result

    def assert_test_frame(self, dataset_name: str, frame_index: int) -> None:
        """Reject audit/train/calibration frames before a test artifact is read."""
        if self.protocol_manifest is None:
            raise FinalTestAccessError("protocol manifest is required for test frame access")
        name = _nonempty_text(dataset_name, name="dataset_name")
        frame = _strict_int(frame_index, name="frame_index")
        if name not in DATASET_NAMES:
            raise ManifestValidationError(f"未知 dataset={name}")
        test_ranges = self.protocol_manifest["datasets"][name]["frame_partitions"]["test"]["ranges"]
        if not any(int(start) <= frame < int(end) for start, end in test_ranges):
            if name == "boussinesq" and frame == BOUSSINESQ_AUDIT_FRAME:
                raise ManifestValidationError(
                    "Boussinesq frame 1300 被结构性标记为 development/audit，禁止 test access"
                )
            raise ManifestValidationError(
                f"dataset={name} frame={frame} 不允许作为 final-test frame"
            )


class DevelopmentAccessGuard:
    """Guard the R3a/development path to train and calibration only."""

    def __init__(self, protocol_manifest: Mapping[str, Any]) -> None:
        self.protocol_manifest = _validate_any_protocol_manifest(protocol_manifest)

    def assert_split(self, split_name: str) -> str:
        split = _nonempty_text(split_name, name="split_name")
        if split not in ("train", "calibration"):
            raise FinalTestAccessError(
                f"development/R3a 只能读取 train/calibration，拒绝 split={split!r}"
            )
        return split

    def read(self, split_name: str, producer: Callable[[], Any]) -> Any:
        self.assert_split(split_name)
        if not callable(producer):
            raise TypeError("development producer 必须是 callable")
        return producer()


def guarded_final_test_read(
    producer: Callable[[], Any],
    *,
    protocol_manifest: Mapping[str, Any] | None = None,
    freeze_record: Mapping[str, Any] | None = None,
) -> Any:
    """Convenience wrapper used by evaluators and integration tests."""
    return FinalTestAccessGuard(
        protocol_manifest=protocol_manifest,
        freeze_record=freeze_record,
    ).read_relationships(producer)


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 criterion-ambiguity R0 protocol artifacts")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recorded-on", default="2026-09-05")
    args = parser.parse_args(argv)
    result = write_r0_artifacts(
        args.data_root,
        args.output_dir,
        recorded_on=args.recorded_on,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "STUDY_SLUG",
    "PROTOCOL_VERSION",
    "PROTOCOL_SCHEMA_VERSION",
    "FREEZE_SCHEMA_VERSION",
    "P95P80_PROTOCOL_VERSION",
    "P95P80_PROTOCOL_SCHEMA_VERSION",
    "P95P80_FREEZE_SCHEMA_VERSION",
    "P95P80_ARTIFACT_SCHEMA_VERSION",
    "P95P80_PROVENANCE_SCHEMA_VERSION",
    "P95P80_PERCENTILE_SOURCE_ID",
    "P95P80_HALLER_SOURCE_ID",
    "P95P80_IVD_DEFINITION_ID",
    "P95P80_IVD_FORMULA",
    "P95P80_SEALED_RELATIONSHIP_FIELDS",
    "P95P80_PROVENANCE_FIELDS",
    "P95P80_ALLOWED_SOURCE_IDS",
    "P95P80_LEGACY_PROTOCOL_VERSIONS",
    "P95P80_LEGACY_SOURCE_IDS",
    "DATASET_NAMES",
    "SPLIT_NAMES",
    "AUDIT_SPLIT_NAME",
    "ALL_PARTITION_NAMES",
    "BOUSSINESQ_AUDIT_FRAME",
    "DEFAULT_T_WIN",
    "DEFAULT_WINDOW_STEP",
    "FINAL_TEST_RELATIONSHIP_FIELDS",
    "ProtocolError",
    "ProtocolIdentityError",
    "ManifestValidationError",
    "FreezeRecordValidationError",
    "FinalTestAccessError",
    "build_frame_manifest",
    "validate_frame_manifest",
    "build_protocol_manifest",
    "validate_protocol_manifest",
    "build_protocol_manifest_from_data_root",
    "write_json_artifact",
    "read_protocol_manifest",
    "build_r0_freeze_record",
    "validate_freeze_record",
    "write_r0_artifacts",
    "build_p95p80_artifact_identity",
    "validate_p95p80_artifact_identity",
    "build_p95p80_provenance",
    "validate_p95p80_provenance",
    "build_p95p80_protocol_manifest",
    "validate_p95p80_protocol_manifest",
    "build_p95p80_protocol_manifest_from_data_root",
    "read_p95p80_protocol_manifest",
    "build_p95p80_freeze_record",
    "validate_p95p80_freeze_record",
    "write_p95p80_artifacts",
    "FinalTestAccessGuard",
    "DevelopmentAccessGuard",
    "guarded_final_test_read",
]
