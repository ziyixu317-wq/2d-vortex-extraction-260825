"""W1-H Haller train-anchor 与 teacher 弱监督训练 seam。

本模块把票 02 生成的 source-specific Haller train artifact 接到票 05
已经验证过的 7-channel、masked anchor BCE、EMA teacher、confident
pseudo-label、consistency 和 ramp-up 训练基础设施上。standard Haller-IVD
只在 artifact/anchor 侧出现；模型仍接收当前 5x5 local-IVD 的
``[px, py, t, ivd, distance, u, v]`` 输入。

Haller 原始文献的 Zotero 候选为 ``L2PX3NQX``，当前工程只确认公式和二维
流程骨架；离散工程参数仍是 ``pending_verification``，不能在这里表述为
canonical paper 参数。训练只允许读取 ``haller_anchor_train``，calibration
和 test GT 必须经各自的专用消费者 seam 读取。
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import string
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import extractor
import haller_anchors
import weak_supervision_contract as contract
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


W1H_GENERATION_VERSION = "w1-h-haller-anchor-v1"
W1H_LABEL_SOURCE = contract.LABEL_SOURCE_HALLER_TRAIN
W1H_DEFAULT_PSEUDO_HIGH = 0.90
W1H_DEFAULT_PSEUDO_LOW = 0.10
W1H_DEFAULT_EMA_DECAY = 0.99
W1H_DEFAULT_RAMP_UP_EPOCHS = 12
W1H_ALLOWED_SAMPLING_SOURCES = frozenset({
    contract.LABEL_SOURCE_LEGACY_P85,
    contract.LABEL_SOURCE_LOCAL_P90_P60,
})


def _nonempty_hash(value: Any, *, name: str) -> str:
    """校验输入/参数/anchor hash，拒绝空值和隐式默认。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _artifact_hash(metadata: Mapping[str, Any]) -> str:
    """从已校验 metadata 生成稳定的 source-specific anchor hash。"""
    payload = json.dumps(
        dict(metadata), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_bool_mask(value: Any, shape: tuple[int, ...], *, name: str) -> Any:
    """将 batch mask 规格化为 bool，并保留 torch device。"""
    if isinstance(value, torch.Tensor):
        result = value.to(dtype=torch.bool)
        if tuple(result.shape) != shape:
            raise ValueError(f"{name} shape={tuple(result.shape)} 与 labels={shape} 不一致")
        return result
    result = np.asarray(value, dtype=bool)
    if tuple(result.shape) != shape:
        raise ValueError(f"{name} shape={result.shape} 与 labels={shape} 不一致")
    return result


def _mask_any(value: Any) -> bool:
    """返回 numpy/torch mask 是否存在至少一个 True。"""
    return bool(value.any().item() if isinstance(value, torch.Tensor) else np.any(value))


def _mask_count(value: Any) -> int:
    """返回 numpy/torch mask 的 True 数量。"""
    return int(value.sum().item() if isinstance(value, torch.Tensor) else np.asarray(value).sum())


def _mask_row_count(value: Any) -> int:
    """按 batch 行统计 failed frame 数，而不是把 K 条轨迹重复计数。"""
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 0:
        return int(bool(array))
    return int(np.count_nonzero(np.any(array, axis=tuple(range(1, array.ndim)))))


@dataclass(frozen=True)
class HallerTrainAnchor:
    """一个已落盘并经过 source/hash 校验的 Haller train anchor。"""

    source: str
    frame_index: int
    anchor_state: np.ndarray
    solid_mask: np.ndarray
    failed_frame_mask: np.ndarray
    anchor_confidence: np.ndarray
    metadata: Mapping[str, Any]
    anchor_hash: str
    valid: bool
    failure_count: int

    def __post_init__(self) -> None:
        if self.source != haller_anchors.SOURCE_TRAIN:
            raise ValueError(
                "W1-H HallerTrainAnchor 只能是 haller_anchor_train source"
            )
        frame = _strict_nonnegative_int(self.frame_index, name="frame_index")
        state = np.asarray(self.anchor_state, dtype=np.int8)
        solid = np.asarray(self.solid_mask, dtype=bool)
        failed = np.asarray(self.failed_frame_mask, dtype=bool)
        confidence = np.asarray(self.anchor_confidence, dtype=np.float32)
        if state.ndim != 2:
            raise ValueError(f"W1-H anchor_state 必须是二维 field，实际 {state.shape}")
        if any(array.shape != state.shape for array in (solid, failed, confidence)):
            raise ValueError("W1-H Haller anchor arrays shape 必须一致")
        if not np.all(np.isin(state, [haller_anchors.UNKNOWN,
                                      haller_anchors.NEGATIVE,
                                      haller_anchors.POSITIVE])):
            raise ValueError("W1-H anchor_state 含有未知三态编码")
        if np.any(solid & (state != haller_anchors.UNKNOWN)):
            raise ValueError("W1-H solid cells 必须保持 unknown")
        if np.any(failed & ~solid & (state != haller_anchors.UNKNOWN)):
            raise ValueError("W1-H failed-frame cells 必须保持 unknown")
        if not np.all(np.isfinite(confidence)):
            raise ValueError("W1-H anchor_confidence 必须有限")
        metadata = dict(self.metadata)
        for field in ("algorithm_version", "parameter_hash", "input_hash",
                      "mask_hash", "failure_count", "coverage"):
            if field not in metadata:
                raise ValueError(f"W1-H Haller metadata 缺少 {field}")
        if metadata.get("source") != haller_anchors.SOURCE_TRAIN:
            raise ValueError("W1-H Haller metadata source 必须是 haller_anchor_train")
        literature = metadata.get("literature")
        if (not isinstance(literature, Mapping)
                or literature.get("status") != "pending_verification"):
            raise ValueError("W1-H Haller literature 必须保留 pending_verification")
        if metadata.get("legacy_p85_used") is not False:
            raise ValueError("W1-H Haller artifact 禁止 legacy_p85 fallback")
        if metadata.get("fallback_used") is not None:
            raise ValueError("W1-H Haller artifact 不允许 fallback")
        expected_failure_count = _strict_nonnegative_int(
            metadata["failure_count"], name="metadata.failure_count")
        failure_count = _strict_nonnegative_int(
            self.failure_count, name="failure_count")
        if expected_failure_count != failure_count:
            raise ValueError("W1-H failure_count 与 artifact metadata 不一致")
        object.__setattr__(self, "frame_index", frame)
        object.__setattr__(self, "anchor_state", state.copy())
        object.__setattr__(self, "solid_mask", solid.copy())
        object.__setattr__(self, "failed_frame_mask", failed.copy())
        object.__setattr__(self, "anchor_confidence", confidence.copy())
        object.__setattr__(self, "metadata", copy.deepcopy(metadata))
        object.__setattr__(self, "anchor_hash", _nonempty_hash(self.anchor_hash,
                                                                  name="anchor_hash"))
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "failure_count", failure_count)

    @property
    def shape(self) -> tuple[int, int]:
        """返回二维 Haller field shape。"""
        return tuple(int(value) for value in self.anchor_state.shape)

    def targets_for_seeds(
        self,
        seeds: Any,
        xdim: Any,
        ydim: Any,
    ) -> dict[str, np.ndarray]:
        """按 extractor 同一最近网格公式把 Haller field 投影到 seed cells。"""
        points = np.asarray(seeds, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("W1-H seeds 必须是有限的 (K,2) 物理坐标")
        x = np.asarray(xdim, dtype=np.float64)
        y = np.asarray(ydim, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or (len(y), len(x)) != self.shape:
            raise ValueError("W1-H Haller 坐标轴与 anchor field shape 不一致")
        rows, cols = extractor.nearest_cell(points[:, 0], points[:, 1], x, y)
        rows = np.clip(rows, 0, self.shape[0] - 1)
        cols = np.clip(cols, 0, self.shape[1] - 1)
        state = self.anchor_state[rows, cols]
        solid = self.solid_mask[rows, cols]
        failed = self.failed_frame_mask[rows, cols]
        known = (state >= 0) & ~solid & ~failed
        unknown = ~known
        labels = (state == haller_anchors.POSITIVE).astype(np.float32)
        labels[unknown] = 0.0
        return {
            "labels": labels,
            "label_mask": known,
            "unknown_mask": unknown,
            "solid_mask": solid,
            "failed_frame_mask": failed,
        }


def load_haller_train_artifact(
    artifact_dir: str,
    *,
    expected_frame_index: int | None = None,
) -> HallerTrainAnchor:
    """读取一个独立 train Haller artifact，并拒绝 calibration/test source。"""
    # expected_source 是训练隔离的关键 seam；不允许调用方把 source 省略。
    loaded = haller_anchors.load_haller_artifact(
        artifact_dir, expected_source=haller_anchors.SOURCE_TRAIN)
    metadata = dict(loaded["metadata"])
    frame_index = metadata.get("frame_index")
    if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, (int, np.integer)):
        raise ValueError("W1-H Haller metadata.frame_index 必须是整数")
    frame_index = int(frame_index)
    if expected_frame_index is not None:
        expected_frame = _strict_nonnegative_int(
            expected_frame_index, name="expected_frame_index")
        if frame_index != expected_frame:
            raise ValueError(
                f"W1-H artifact frame_index={frame_index} 与 expected={expected_frame} 不一致"
            )
    if "solid_mask" not in loaded:
        raise ValueError(
            "W1-H train Haller artifact 必须保存 solid_mask.npy，禁止丢失 geometry policy"
        )
    solid = np.asarray(loaded["solid_mask"], dtype=bool)
    state = np.asarray(loaded["anchor_state"], dtype=np.int8)
    raw_failure_count = metadata.get("failure_count", None)
    if raw_failure_count is None:
        raise ValueError("W1-H Haller metadata.failure_count 必须显式提供")
    failure_count = _strict_nonnegative_int(
        raw_failure_count, name="failure_count")
    failed = np.full(state.shape, failure_count > 0, dtype=bool) & ~solid
    anchor_hash = _artifact_hash(metadata)
    return HallerTrainAnchor(
        source=haller_anchors.SOURCE_TRAIN,
        frame_index=frame_index,
        anchor_state=state,
        solid_mask=solid,
        failed_frame_mask=failed,
        anchor_confidence=np.asarray(loaded["anchor_confidence"], dtype=np.float32),
        metadata=metadata,
        anchor_hash=anchor_hash,
        valid=bool(metadata.get("valid", False)),
        failure_count=failure_count,
    )


class HallerTrainArtifactResolver:
    """按显式 frame pattern 读取 train-only Haller artifacts。

    不做目录 glob、source 猜测或旧 label fallback。pattern 必须显式包含
    ``{frame}`` 或 ``{frame_index}``，这样缺帧和错误 source 都会在实际取样时
    fail loudly，而不是默默复用另一帧或旧 p85 标签。
    """

    def __init__(
        self,
        artifact_root: str | pathlib.Path,
        *,
        artifact_pattern: str = "frame{frame}",
        anchor_hash: str | None = None,
    ) -> None:
        root = pathlib.Path(artifact_root)
        if not str(root).strip():
            raise ValueError("W1-H artifact_root 必须是非空路径")
        if not root.exists():
            raise FileNotFoundError(f"W1-H Haller artifact_root 不存在：{root}")
        if not root.is_dir():
            raise ValueError(f"W1-H artifact_root 必须是目录：{root}")
        if not isinstance(artifact_pattern, str) or not artifact_pattern.strip():
            raise ValueError("W1-H artifact_pattern 必须是非空字符串")
        fields = {
            field_name.split(".", 1)[0]
            for _, field_name, _, _ in string.Formatter().parse(artifact_pattern)
            if field_name is not None
        }
        allowed = {"frame", "frame_index"}
        if not fields & allowed:
            raise ValueError(
                "W1-H artifact_pattern 必须显式包含 {frame} 或 {frame_index}"
            )
        unexpected = fields - allowed
        if unexpected:
            raise ValueError(
                f"W1-H artifact_pattern 只允许 frame 占位符，实际包含 {sorted(unexpected)!r}"
            )
        try:
            artifact_pattern.format(frame=0, frame_index=0)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                "W1-H artifact_pattern 必须能用整数 frame 格式化"
            ) from exc
        self.root = root
        self.artifact_pattern = artifact_pattern
        self._anchor_hash_override = (
            None if anchor_hash is None else _nonempty_hash(anchor_hash, name="anchor_hash")
        )
        self._cache: dict[int, HallerTrainAnchor] = {}

    @property
    def anchor_hash(self) -> str | None:
        """返回可选的跨 frame manifest hash；未声明时保持每 artifact hash。"""
        return self._anchor_hash_override

    def path_for(self, frame_index: Any) -> pathlib.Path:
        """把非负整数 frame 映射到 root 内的唯一 artifact 目录。"""
        frame = _strict_nonnegative_int(frame_index, name="frame_index")
        try:
            relative_text = self.artifact_pattern.format(
                frame=frame, frame_index=frame)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"W1-H artifact_pattern 无法格式化 frame={frame}"
            ) from exc
        relative = pathlib.Path(relative_text)
        if relative.is_absolute():
            raise ValueError("W1-H artifact_pattern 不得生成 artifact_root 之外的绝对路径")
        root_resolved = self.root.resolve()
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"W1-H artifact_pattern 不能越出 artifact_root：{relative_text!r}"
            ) from exc
        return candidate

    def load(self, frame_index: Any) -> HallerTrainAnchor:
        """读取并缓存 frame 对应的 train artifact，source/frame 都显式校验。"""
        frame = _strict_nonnegative_int(frame_index, name="frame_index")
        if frame not in self._cache:
            path = self.path_for(frame)
            if not path.is_dir():
                raise FileNotFoundError(
                    f"缺少 W1-H haller_anchor_train artifact：frame={frame} path={path}"
                )
            self._cache[frame] = load_haller_train_artifact(
                path, expected_frame_index=frame)
        return self._cache[frame]

    def hash_for(self, anchor: HallerTrainAnchor) -> str:
        """返回 batch 使用的 hash；保留每帧 artifact hash 另存于 provenance。"""
        if not isinstance(anchor, HallerTrainAnchor):
            raise TypeError("W1-H resolver.hash_for 必须消费 HallerTrainAnchor")
        return self._anchor_hash_override or anchor.anchor_hash


def _dataset_stores(dataset: Any) -> tuple[list[Any], bool]:
    """取得单/多数据集的 store，并拒绝不完整的 wrapper seam。"""
    for field in ("sample_at", "window_metadata", "set_epoch", "set_epoch_natural"):
        if not hasattr(dataset, field):
            raise TypeError(f"W1-H dataset 缺少公开 {field} seam")
    if hasattr(dataset, "stores"):
        stores = list(dataset.stores)
        multi = True
    elif hasattr(dataset, "store"):
        stores = [dataset.store]
        multi = False
    else:
        raise TypeError(
            "W1-H dataset 必须是 WeakLabelPathlineDataset 或 MultiDatasetPathlineDataset"
        )
    if not stores:
        raise ValueError("W1-H dataset stores 不能为空")
    for store in stores:
        for field in ("sample_at", "window_metadata", "_xdim", "_ydim"):
            if not hasattr(store, field):
                raise TypeError(f"W1-H dataset store 缺少 {field} seam")
    return stores, multi


def _validate_train_window_metadata(window: Any, *, frame: int) -> dict[str, Any]:
    """验证 W1-H 取样窗口的 train split 和半开 frame 边界。"""
    if not isinstance(window, Mapping):
        raise ValueError("W1-H dataset.window_metadata 必须返回 object")
    required = (
        "split_name", "frame_start", "frame_end", "split_start", "split_end",
        "t_win", "window_step",
    )
    missing = [field for field in required if field not in window]
    if missing:
        raise ValueError(f"W1-H window metadata 缺少字段：{missing!r}")
    if window["split_name"] != "train":
        raise ValueError(
            f"W1-H window split_name 必须为 'train'，实际 {window['split_name']!r}"
        )
    frame_start = _strict_nonnegative_int(window["frame_start"], name="frame_start")
    frame_end = _strict_nonnegative_int(window["frame_end"], name="frame_end")
    split_start = _strict_nonnegative_int(window["split_start"], name="split_start")
    split_end = _strict_nonnegative_int(window["split_end"], name="split_end")
    t_win = _strict_nonnegative_int(window["t_win"], name="t_win")
    window_step = _strict_nonnegative_int(window["window_step"], name="window_step")
    if frame_start != frame or frame_end - frame_start != t_win:
        raise ValueError(
            "W1-H window frame_start/frame_end 与实际 frame/t_win 不一致："
            f"window=({frame_start}, {frame_end}, {t_win}) actual_frame={frame}"
        )
    if not (split_start < split_end and split_start <= frame_start < frame_end <= split_end):
        raise ValueError(
            "W1-H window 越过 train split："
            f"window=({frame_start}, {frame_end}) split=({split_start}, {split_end})"
        )
    if window_step <= 0:
        raise ValueError("W1-H window_step 必须是正整数")
    return copy.deepcopy(dict(window))


def _artifact_roots_for_dataset(
    stores: list[Any], artifact_root: Any,
) -> list[pathlib.Path]:
    """将单 root/list/mapping 显式对齐到 dataset stores。"""
    if isinstance(artifact_root, Mapping):
        names = [str(getattr(store, "dataset_name", "")) for store in stores]
        missing = [name for name in names if name not in artifact_root]
        if missing:
            raise ValueError(
                f"W1-H artifact_root mapping 缺少数据集目录：{missing!r}"
            )
        return [pathlib.Path(artifact_root[name]) for name in names]
    if isinstance(artifact_root, (str, pathlib.Path)):
        if len(stores) != 1:
            raise ValueError(
                "W1-H 多数据集必须显式提供与 stores 一一对应的 artifact_root list/mapping"
            )
        return [pathlib.Path(artifact_root)]
    if isinstance(artifact_root, (list, tuple)):
        if len(artifact_root) != len(stores):
            raise ValueError(
                f"W1-H artifact_root 数量={len(artifact_root)} 与 stores={len(stores)} 不一致"
            )
        return [pathlib.Path(value) for value in artifact_root]
    raise TypeError("W1-H artifact_root 必须是路径、路径 list 或 dataset-name mapping")


class HallerAnchorPathlineDataset:
    """把当前 5×5 local-IVD pathline window 换成 W1-H train anchor batch。

    ``base_dataset`` 仍负责 split-safe window、重播种和 7-channel pathline 提取；
    其 label field 只参与既有 sampling pool。formal labels 完全来自显式
    ``haller_anchor_train`` artifact，因而不会把 legacy p85 或 test GT 混入
    W1-H loss。
    """

    def __init__(
        self,
        base_dataset: Any,
        artifact_root: Any,
        *,
        sampling_source: str,
        artifact_pattern: str = "frame{frame}",
        anchor_hash: str | None = None,
    ) -> None:
        stores, multi = _dataset_stores(base_dataset)
        split_values = {
            str(getattr(store, "split", getattr(base_dataset, "split", "")))
            for store in stores
        }
        if split_values != {"train"}:
            raise ValueError(
                f"W1-H Haller train dataset 只能使用 split=train，实际 {sorted(split_values)!r}"
            )
        consumer_values = {
            str(getattr(store, "consumer", getattr(base_dataset, "consumer", "")))
            for store in stores
        }
        if consumer_values != {"train"}:
            raise ValueError(
                f"W1-H Haller dataset consumer 必须是 train，实际 {sorted(consumer_values)!r}"
            )
        weak_flags = [
            bool(getattr(store, "is_weak_supervision",
                        getattr(base_dataset, "is_weak_supervision", False)))
            for store in stores
        ]
        if not all(weak_flags):
            raise ValueError(
                "W1-H Haller dataset 必须来自 split_mode='weak_supervision' metadata"
            )
        if not isinstance(sampling_source, str) or not sampling_source.strip():
            raise ValueError("W1-H 必须显式提供 sampling_source")
        sampling_source = contract.validate_label_source(sampling_source.strip())
        if sampling_source not in W1H_ALLOWED_SAMPLING_SOURCES:
            raise ValueError(
                f"W1-H sampling_source 只能是 {sorted(W1H_ALLOWED_SAMPLING_SOURCES)!r}"
            )
        forbidden = {
            contract.LABEL_SOURCE_HALLER_CALIBRATION,
            contract.LABEL_SOURCE_HALLER_TEST,
        }
        base_sources = {
            str(getattr(store, "label_source", "")) for store in stores
        }
        if base_sources & forbidden:
            raise ValueError(
                "W1-H sampling dataset 禁止 calibration/test Haller source："
                f"{sorted(base_sources & forbidden)!r}"
            )
        roots = _artifact_roots_for_dataset(stores, artifact_root)
        self.base_dataset = base_dataset
        self.sampling_source = sampling_source
        self.label_source = W1H_LABEL_SOURCE
        self.split = "train"
        self.consumer = "train"
        self.is_weak_supervision = True
        self._stores = stores
        self._multi = multi
        self.resolvers = [
            HallerTrainArtifactResolver(
                root, artifact_pattern=artifact_pattern, anchor_hash=anchor_hash)
            for root in roots
        ]
        self.artifact_pattern = artifact_pattern
        self.anchor_hash = anchor_hash
        self._order = None
        self._epoch = None

    def set_epoch(self, epoch: Any):
        """沿用 base dataset 的确定性正/负 sampling order。"""
        self._epoch = _strict_nonnegative_int(epoch, name="epoch")
        self._order = self.base_dataset.set_epoch(self._epoch)
        if self._order is None:
            raise RuntimeError("W1-H base dataset.set_epoch 未返回 sampling order")
        return self._order

    def set_epoch_natural(self, epoch: Any = 0):
        """显式透传自然分布采样；训练入口通常只调用 set_epoch。"""
        self._epoch = _strict_nonnegative_int(epoch, name="epoch")
        self._order = self.base_dataset.set_epoch_natural(self._epoch)
        if self._order is None:
            raise RuntimeError("W1-H base dataset.set_epoch_natural 未返回 sampling order")
        return self._order

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _parse_order(self, order: Any) -> tuple[int, int, int, int]:
        values = tuple(order)
        if self._multi:
            if len(values) != 4:
                raise ValueError(f"W1-H multi dataset order 必须是 (si,py,px,frame)，实际 {values!r}")
            si, py, px, frame = values
        else:
            if len(values) != 3:
                raise ValueError(f"W1-H dataset order 必须是 (py,px,frame)，实际 {values!r}")
            si = 0
            py, px, frame = values
        return (
            _strict_nonnegative_int(si, name="dataset_index"),
            _strict_nonnegative_int(py, name="patch_y"),
            _strict_nonnegative_int(px, name="patch_x"),
            _strict_nonnegative_int(frame, name="frame_index"),
        )

    def _sample(self, si: Any, py: Any, px: Any, frame: Any) -> W1HBatch:
        si = _strict_nonnegative_int(si, name="dataset_index")
        py = _strict_nonnegative_int(py, name="patch_y")
        px = _strict_nonnegative_int(px, name="patch_x")
        frame = _strict_nonnegative_int(frame, name="frame_index")
        if not self._multi and si != 0:
            raise ValueError(f"W1-H single dataset 的 dataset_index 必须为 0，实际 {si}")
        if si >= len(self._stores):
            raise IndexError(f"W1-H dataset_index={si} 超出 stores={len(self._stores)}")
        store = self._stores[si]
        if self._multi:
            sample = self.base_dataset.sample_at(si, py, px, frame)
            window = self.base_dataset.window_metadata(si, frame)
        else:
            sample = self.base_dataset.sample_at(py, px, frame)
            window = self.base_dataset.window_metadata(frame)
        if (not isinstance(sample, (tuple, list)) or len(sample) != 3
                or not isinstance(sample[0], (tuple, list)) or len(sample[0]) != 2):
            raise ValueError(
                "W1-H base dataset.sample_at 必须返回 ((dummy, pathlines), labels, seeds)"
            )
        (dummy, pathlines), _sampling_labels, seeds = sample
        pathlines = np.asarray(pathlines, dtype=np.float32)
        if pathlines.ndim != 3 or pathlines.shape[-1] != contract.FEATURE_SCHEMA_7.channel_count:
            raise ValueError(
                "W1-H base pathlines 必须是未 batching 的 (L,K,7) local-IVD features，"
                f"实际 shape={pathlines.shape}"
            )
        seeds = np.asarray(seeds, dtype=np.float64)
        if seeds.ndim != 2 or seeds.shape != (pathlines.shape[1], 2):
            raise ValueError(
                f"W1-H base seeds shape={seeds.shape} 与 pathlines K={pathlines.shape[1]} 不一致"
            )
        resolver = self.resolvers[si]
        anchor = resolver.load(frame)
        window = _validate_train_window_metadata(window, frame=frame)
        provenance = {
            "window": copy.deepcopy(dict(window)),
            "sampling": {
                "source": self.sampling_source,
                "base_label_source": str(getattr(store, "label_source", "")),
            },
            "dataset_index": si,
        }
        return build_w1_h_batch_from_anchor(
            pathlines[None, ...], seeds, anchor, store._xdim, store._ydim,
            sampling_source=self.sampling_source,
            split_name="train",
            provenance=provenance,
            dummy_field=np.asarray(dummy, dtype=np.float32),
            anchor_hash_override=resolver.anchor_hash,
        )

    def sample_at(self, *indices: Any) -> W1HBatch:
        """直接读取一个 train window；单/多数据集参数顺序与 base 保持一致。"""
        if self._multi:
            if len(indices) != 4:
                raise TypeError("W1-H multi sample_at 需要 (dataset_index, py, px, frame)")
            return self._sample(*indices)
        if len(indices) != 3:
            raise TypeError("W1-H sample_at 需要 (py, px, frame)")
        return self._sample(0, *indices)

    def __getitem__(self, index: Any) -> W1HBatch:
        if self._order is None:
            raise RuntimeError("先调用 set_epoch(epoch) 再采样 W1-H dataset")
        index = _strict_nonnegative_int(index, name="dataset index")
        if index >= len(self._order):
            raise IndexError(index)
        return self._sample(*self._parse_order(self._order[index]))


def _concat_batch_values(values: list[Any], *, name: str) -> Any:
    """沿 batch 维拼接 numpy/torch 值，并阻止隐式 device 漂移。"""
    if not values:
        raise ValueError(f"W1-H {name} 不能是空 list")
    if any(isinstance(value, torch.Tensor) for value in values):
        first = values[0] if isinstance(values[0], torch.Tensor) else torch.as_tensor(values[0])
        converted = [
            value if isinstance(value, torch.Tensor) else torch.as_tensor(
                value, device=first.device)
            for value in values
        ]
        if any(value.device != first.device for value in converted):
            raise ValueError(f"W1-H {name} batch device 不一致")
        return torch.cat(converted, dim=0)
    return np.concatenate([np.asarray(value) for value in values], axis=0)


def collate_w1_h_batches(batches: Any) -> W1HBatch:
    """DataLoader collate：保持同一 anchor manifest/source 的 batch provenance。"""
    batches = list(batches)
    if not batches:
        raise ValueError("W1-H collate 收到空 batch")
    if not all(isinstance(batch, W1HBatch) for batch in batches):
        raise TypeError("W1-H collate 只能拼接 W1HBatch")
    hashes = {batch.anchor_hash for batch in batches}
    if len(hashes) != 1:
        raise ValueError(
            "W1-H collate 检测到 anchor_hash 漂移；请声明同一 manifest hash"
        )
    sources = {batch.sampling_source for batch in batches}
    if len(sources) != 1:
        raise ValueError("W1-H collate 检测到 sampling_source 漂移")
    split_names = {batch.contract_batch.split_name for batch in batches}
    if split_names != {"train"}:
        raise ValueError("W1-H collate 只能拼接 split=train batch")
    first = batches[0]
    anchor_items = []
    windows = []
    for batch in batches:
        item = copy.deepcopy(dict(batch.anchor_metadata or {}))
        item.setdefault(
            "artifact_hash",
            dict(batch.contract_batch.provenance.get("anchor", {})).get(
                "artifact_hash", batch.anchor_hash),
        )
        anchor_items.append(item)
        batch_window = dict(batch.contract_batch.provenance).get("window")
        if not isinstance(batch_window, Mapping):
            raise ValueError(
                "W1-H collate 要求每个 batch 携带显式 provenance.window"
            )
        windows.append(copy.deepcopy(dict(batch_window)))
    combined_metadata = copy.deepcopy(dict(first.anchor_metadata or {}))
    combined_metadata["batch_artifacts"] = anchor_items
    combined_metadata["anchor_hash"] = first.anchor_hash
    combined_anchor = copy.deepcopy(dict(first.anchor_metadata or {}))
    combined_anchor.update({
        "anchor_hash": first.anchor_hash,
        "artifact_hashes": [
            str(item.get("artifact_hash", batch.anchor_hash))
            for item, batch in zip(anchor_items, batches)
        ],
        "batch_artifacts": anchor_items,
    })
    provenance = {
        "anchor": combined_anchor,
        "windows": windows,
        "batches": [copy.deepcopy(dict(batch.contract_batch.provenance))
                    for batch in batches],
        "sampling": {"source": first.sampling_source},
    }
    return build_w1_h_batch(
        _concat_batch_values([batch.pathlines for batch in batches], name="pathlines"),
        _concat_batch_values([batch.labels for batch in batches], name="labels"),
        _concat_batch_values([batch.label_mask for batch in batches], name="label_mask"),
        _concat_batch_values([batch.unknown_mask for batch in batches], name="unknown_mask"),
        _concat_batch_values([batch.solid_mask for batch in batches], name="solid_mask"),
        failed_frame_mask=_concat_batch_values(
            [batch.failed_frame_mask for batch in batches], name="failed_frame_mask"),
        sampling_source=first.sampling_source,
        split_name="train",
        anchor_hash=first.anchor_hash,
        provenance=provenance,
        anchor_metadata=combined_metadata,
        dummy_field=_concat_batch_values(
            [batch.dummy_field for batch in batches], name="dummy_field"),
    )


@dataclass
class W1HBatch:
    """携带 Haller known/unknown/solid/failed mask 的正式 W1-H batch。"""

    contract_batch: contract.WeakSupervisionBatch
    solid_mask: Any
    failed_frame_mask: Any
    anchor_hash: str
    dummy_field: Any | None = None
    anchor_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        batch = contract.validate_training_batch(
            self.contract_batch, contract.MODE_W1_H)
        if batch.label_source != W1H_LABEL_SOURCE:
            raise ValueError(
                f"W1-H formal loss source 必须是 {W1H_LABEL_SOURCE!r}，"
                f"实际 {batch.label_source!r}"
            )
        self.anchor_hash = _nonempty_hash(self.anchor_hash, name="anchor_hash")
        if not isinstance(batch.sampling_source, str):
            raise ValueError("W1-H 必须显式提供 sampling_source")
        sampling_source = contract.validate_label_source(batch.sampling_source)
        if sampling_source not in W1H_ALLOWED_SAMPLING_SOURCES:
            raise ValueError(
                f"W1-H sampling_source 不能是 Haller source：{sampling_source!r}"
            )
        shape = tuple(int(value) for value in batch.labels.shape)
        self.solid_mask = _as_bool_mask(self.solid_mask, shape, name="solid_mask")
        self.failed_frame_mask = _as_bool_mask(
            self.failed_frame_mask, shape, name="failed_frame_mask")
        if isinstance(batch.label_mask, torch.Tensor):
            known = batch.label_mask
            unknown = batch.unknown_mask
        else:
            known = np.asarray(batch.label_mask, dtype=bool)
            unknown = np.asarray(batch.unknown_mask, dtype=bool)
        if _mask_any(self.solid_mask & known) or _mask_any(self.solid_mask & ~unknown):
            raise ValueError("W1-H solid_mask 必须只落在 unknown/ignored 区域")
        if (_mask_any(self.failed_frame_mask & known)
                or _mask_any(self.failed_frame_mask & ~unknown)):
            raise ValueError("W1-H failed_frame_mask 必须只落在 unknown/ignored 区域")
        if _mask_any(self.failed_frame_mask & self.solid_mask):
            raise ValueError("W1-H failed_frame_mask 必须与 solid_mask 分离")
        anchor_provenance = dict(batch.provenance.get("anchor", {}))
        if anchor_provenance.get("source") not in (None, W1H_LABEL_SOURCE):
            raise ValueError("W1-H batch anchor provenance source 必须是 haller_anchor_train")
        if anchor_provenance.get("anchor_hash") not in (None, self.anchor_hash):
            raise ValueError("W1-H batch anchor provenance hash 与 batch anchor_hash 不一致")
        sampling_provenance = dict(batch.provenance.get("sampling", {}))
        if sampling_provenance.get("source") not in (None, sampling_source):
            raise ValueError("W1-H batch sampling provenance source 与 sampling_source 不一致")
        if self.anchor_metadata is not None:
            if not isinstance(self.anchor_metadata, Mapping):
                raise TypeError("W1-H anchor_metadata 必须是 object")
            if self.anchor_metadata.get("source") not in (None, W1H_LABEL_SOURCE):
                raise ValueError("W1-H anchor_metadata source 必须是 haller_anchor_train")
            self.anchor_metadata = copy.deepcopy(dict(self.anchor_metadata))
        if self.dummy_field is None:
            if isinstance(batch.pathlines, torch.Tensor):
                self.dummy_field = batch.pathlines.new_zeros((shape[0], 1, 1, 1))
            else:
                self.dummy_field = np.zeros((shape[0], 1, 1, 1), dtype=np.float32)
        dummy_shape = getattr(self.dummy_field, "shape", ())
        if len(dummy_shape) < 1 or int(dummy_shape[0]) != shape[0]:
            raise ValueError("W1-H dummy_field batch 维度与 labels 不一致")

    @property
    def pathlines(self) -> Any:
        """model-facing 7-channel local-IVD pathlines。"""
        return self.contract_batch.pathlines

    @property
    def labels(self) -> Any:
        """Haller known cell target；unknown 仅为 BCE 占位。"""
        return self.contract_batch.labels

    @property
    def label_mask(self) -> Any:
        """Haller known positive/negative mask。"""
        return self.contract_batch.label_mask

    @property
    def unknown_mask(self) -> Any:
        """Haller unknown mask，包含 boundary、solid 和 failed frame。"""
        return self.contract_batch.unknown_mask

    @property
    def sampling_source(self) -> str:
        """采样来源；不等同于 formal Haller loss source。"""
        return str(self.contract_batch.sampling_source)

    @property
    def label_source(self) -> str:
        """formal W1-H loss source。"""
        return self.contract_batch.label_source

    def as_dict(self) -> dict[str, Any]:
        """返回不含大数组的 batch 诊断摘要。"""
        result = self.contract_batch.as_dict()
        total = int(np.prod(self.labels.shape))
        known_count = _mask_count(self.label_mask)
        unknown_count = _mask_count(self.unknown_mask)
        solid_count = _mask_count(self.solid_mask)
        failed_count = _mask_count(self.failed_frame_mask)
        result.update({
            "anchor_hash": self.anchor_hash,
            "anchor_coverage": known_count / total if total else 0.0,
            "unknown_coverage": unknown_count / total if total else 0.0,
            "solid_coverage": solid_count / total if total else 0.0,
            "failed_cell_coverage": failed_count / total if total else 0.0,
            "solid_count": solid_count,
            "failed_cell_count": failed_count,
            "failed_frame_count": _mask_row_count(self.failed_frame_mask),
        })
        if self.anchor_metadata is not None:
            result["anchor_metadata"] = copy.deepcopy(dict(self.anchor_metadata))
        return result

    def to(self, device: str | torch.device) -> "W1HBatch":
        """将 batch 数组搬到指定 torch device，保留 Haller provenance。"""
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
        return W1HBatch(
            converted,
            torch.as_tensor(self.solid_mask, device=device),
            torch.as_tensor(self.failed_frame_mask, device=device),
            self.anchor_hash,
            torch.as_tensor(self.dummy_field, device=device),
            self.anchor_metadata,
        )


def build_w1_h_batch(
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
) -> W1HBatch:
    """构造 formal W1-H batch，强制携带 Haller train source/hash。"""
    anchor_hash = _nonempty_hash(anchor_hash, name="anchor_hash")
    sampling_source = contract.validate_label_source(sampling_source)
    if sampling_source not in W1H_ALLOWED_SAMPLING_SOURCES:
        raise ValueError(
            f"W1-H sampling_source 必须是 {sorted(W1H_ALLOWED_SAMPLING_SOURCES)!r}"
        )
    if failed_frame_mask is None:
        if isinstance(labels, torch.Tensor):
            failed_frame_mask = torch.zeros_like(labels, dtype=torch.bool)
        else:
            failed_frame_mask = np.zeros_like(np.asarray(labels), dtype=bool)
    sources = copy.deepcopy(dict(provenance or {}))
    anchor_provenance = dict(sources.get("anchor", {}))
    anchor_provenance.setdefault("source", W1H_LABEL_SOURCE)
    anchor_provenance.setdefault("anchor_hash", anchor_hash)
    sources["anchor"] = anchor_provenance
    sampling_provenance = dict(sources.get("sampling", {}))
    sampling_provenance.setdefault("source", sampling_source)
    sources["sampling"] = sampling_provenance
    base = contract.WeakSupervisionBatch(
        pathlines=pathlines,
        labels=labels,
        label_source=W1H_LABEL_SOURCE,
        split_name=split_name,
        feature_schema=contract.FEATURE_SCHEMA_7,
        label_mask=label_mask,
        unknown_mask=unknown_mask,
        sampling_source=sampling_source,
        provenance=sources,
        mode=contract.MODE_W1_H,
        input_schema=contract.FEATURE_SCHEMA_7,
    )
    return W1HBatch(
        base, solid_mask, failed_frame_mask, anchor_hash,
        dummy_field=dummy_field, anchor_metadata=anchor_metadata,
    )


def build_w1_h_batch_from_anchor(
    pathlines: Any,
    seeds: Any,
    anchor: HallerTrainAnchor,
    xdim: Any,
    ydim: Any,
    *,
    sampling_source: str,
    split_name: str = "train",
    provenance: Mapping[str, Any] | None = None,
    dummy_field: Any | None = None,
    anchor_hash_override: str | None = None,
) -> W1HBatch:
    """从 Haller field 和重播种后的实际 seeds 构造训练 batch。

    ``anchor_hash_override`` 只用于调用方已经声明并校验跨 frame manifest
    的情形；单 artifact 默认使用该 artifact 的 content-derived hash。
    """
    if not isinstance(anchor, HallerTrainAnchor):
        raise TypeError("W1-H batch builder 必须消费 HallerTrainAnchor")
    target = anchor.targets_for_seeds(seeds, xdim, ydim)
    pathline_shape = getattr(pathlines, "shape", ())
    if len(pathline_shape) == 4 and target["labels"].ndim == 1:
        # dataset.sample_at 返回单 window 的 seeds，而 model-facing pathlines
        # 已经显式加了 batch 维；保持所有 target mask 与它同形。
        target = {key: value[None, ...] for key, value in target.items()}
    anchor_provenance = copy.deepcopy(dict(provenance or {}))
    anchor_provenance.setdefault("anchor", copy.deepcopy(dict(anchor.metadata)))
    effective_hash = (
        anchor.anchor_hash
        if anchor_hash_override is None
        else _nonempty_hash(anchor_hash_override, name="anchor_hash_override")
    )
    anchor_provenance["anchor"]["artifact_hash"] = anchor.anchor_hash
    anchor_provenance["anchor"]["anchor_hash"] = effective_hash
    return build_w1_h_batch(
        pathlines,
        target["labels"],
        target["label_mask"],
        target["unknown_mask"],
        target["solid_mask"],
        failed_frame_mask=target["failed_frame_mask"],
        sampling_source=sampling_source,
        split_name=split_name,
        anchor_hash=effective_hash,
        provenance=anchor_provenance,
        anchor_metadata=anchor.metadata,
        dummy_field=dummy_field,
    )


def _validate_w1_h_config(config: W1PConfig | None) -> W1PConfig:
    """W1-H 与 W1-P 共用 pseudo/EMA/ramp 配置验证。"""
    config = W1HConfig() if config is None else config
    if not isinstance(config, W1PConfig):
        raise TypeError("W1-H config 必须是 W1HConfig/W1PConfig")
    return config


def compute_w1_h_loss(
    student_predictions: Any,
    teacher_predictions: Any,
    batch: W1HBatch,
    *,
    config: W1PConfig | None = None,
    epoch: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """计算 Haller known anchor、pseudo 和 consistency loss。

    Haller unknown boundary、solid 和 failed frame 永不贡献 anchor BCE；其中
    failed frame 也不进入 pseudo/consistency eligible 集合，避免没有合法
    physics contour 的帧被 teacher 伪标签悄悄重新标注。
    """
    if not isinstance(batch, W1HBatch):
        raise TypeError("W1-H loss 必须消费 W1HBatch")
    config = _validate_w1_h_config(config)
    epoch = _strict_nonnegative_int(epoch, name="epoch")
    contract.validate_training_batch(batch.contract_batch, contract.MODE_W1_H)
    expected_shape = tuple(int(value) for value in batch.labels.shape)
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
    failed = torch.as_tensor(
        batch.failed_frame_mask, device=student.device, dtype=torch.bool)
    if bool((solid & known).any()) or bool((solid & ~unknown).any()):
        raise ValueError("W1-H solid mask 必须只落在 unknown/ignored 区域")
    if bool((failed & known).any()) or bool((failed & ~unknown).any()):
        raise ValueError("W1-H failed frame mask 必须只落在 unknown/ignored 区域")
    if bool((failed & solid).any()):
        raise ValueError("W1-H failed frame mask 必须与 solid mask 分离")

    pseudo_eligible = unknown & ~solid & ~failed
    high = torch.as_tensor(config.pseudo_high, device=teacher.device, dtype=teacher.dtype)
    low = torch.as_tensor(config.pseudo_low, device=teacher.device, dtype=teacher.dtype)
    confident = (teacher >= high) | (teacher <= low)
    pseudo_mask = pseudo_eligible & confident
    pseudo_targets = (teacher >= 0.5).float()
    anchor_loss = _masked_bce(student, labels, known)
    pseudo_loss = _masked_bce(student, pseudo_targets, pseudo_mask)
    consistency_loss = (
        F.mse_loss(student[pseudo_eligible], teacher[pseudo_eligible])
        if bool(pseudo_eligible.any()) else student.sum() * 0.0
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
    failed_cell_count = int(failed.sum().item())
    eligible_count = int(pseudo_eligible.sum().item())
    accepted_count = int(pseudo_mask.sum().item())
    positive_count = int((pseudo_mask & (pseudo_targets >= 0.5)).sum().item())
    disagreement = (
        float(torch.abs(student[pseudo_eligible] - teacher[pseudo_eligible]).mean().detach())
        if bool(pseudo_eligible.any()) else 0.0
    )
    anchor_metadata = dict(batch.anchor_metadata or {})
    anchor_provenance = dict(batch.contract_batch.provenance.get("anchor", {}))
    artifact_failure_count = _strict_nonnegative_int(
        anchor_metadata.get("failure_count", anchor_provenance.get("failure_count", 0)),
        name="artifact_failure_count")
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
        "failed_cell_count": failed_cell_count,
        "failed_frame_count": _mask_row_count(failed),
        "artifact_failure_count": artifact_failure_count,
        "pseudo_eligible_count": eligible_count,
        "pseudo_accepted_count": accepted_count,
        "pseudo_positive_count": positive_count,
        "pseudo_negative_count": accepted_count - positive_count,
        "anchor_coverage": anchor_count / total_count if total_count else 0.0,
        "unknown_coverage": unknown_count / total_count if total_count else 0.0,
        "solid_coverage": solid_count / total_count if total_count else 0.0,
        "failed_cell_coverage": failed_cell_count / total_count if total_count else 0.0,
        "pseudo_acceptance": accepted_count / eligible_count if eligible_count else 0.0,
        "teacher_student_disagreement": disagreement,
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


class W1HConfig(W1PConfig):
    """W1-H 的冻结 Haller source/config metadata（数值字段复用 W1-P）。"""

    def as_dict(self) -> dict[str, Any]:
        result = super().as_dict()
        result.update({
            "generation_version": W1H_GENERATION_VERSION,
            "label_source": W1H_LABEL_SOURCE,
            "anchor_algorithm_version": haller_anchors.ALGORITHM_VERSION,
            "literature_status": "pending_verification",
            "literature_zotero_key": "L2PX3NQX",
        })
        return result


class W1HTrainer:
    """W1-H 的可恢复 teacher training seam，不实现 W2/W3 下游逻辑。"""

    def __init__(
        self,
        student: nn.Module,
        optimizer: Any,
        *,
        sampling_source: str,
        anchor_hash: str,
        config: W1HConfig | None = None,
        teacher: nn.Module | None = None,
        scheduler: Any | None = None,
        seed: int = 0,
        anchor_metadata: Mapping[str, Any] | None = None,
        grad_clip_norm: float | None = None,
    ) -> None:
        if not isinstance(student, nn.Module):
            raise TypeError("W1-H student 必须是 torch.nn.Module")
        if optimizer is None or not all(
            hasattr(optimizer, attr) for attr in ("zero_grad", "step", "state_dict")
        ):
            raise TypeError("W1-H optimizer 必须提供 zero_grad()/step()/state_dict()")
        if teacher is not None and not isinstance(teacher, nn.Module):
            raise TypeError("W1-H teacher 必须是 torch.nn.Module")
        sampling_source = contract.validate_label_source(sampling_source)
        if sampling_source not in W1H_ALLOWED_SAMPLING_SOURCES:
            raise ValueError(
                f"W1-H sampling_source 必须是 {sorted(W1H_ALLOWED_SAMPLING_SOURCES)!r}"
            )
        self.student = student
        self.teacher = clone_ema_teacher(student) if teacher is None else teacher
        if self.teacher is self.student:
            raise ValueError("W1-H teacher 不能与 student 共享同一 module")
        _prepare_ema_teacher(self.teacher)
        if tuple(self.student.state_dict()) != tuple(self.teacher.state_dict()):
            raise ValueError("W1-H student/teacher state_dict keys 不一致")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = W1HConfig() if config is None else config
        if not isinstance(self.config, W1PConfig):
            raise TypeError("W1-H config 必须是 W1HConfig/W1PConfig")
        self.sampling_source = sampling_source
        self.anchor_hash = _nonempty_hash(anchor_hash, name="anchor_hash")
        self.anchor_metadata = (
            None if anchor_metadata is None
            else copy.deepcopy(dict(anchor_metadata))
        )
        if self.anchor_metadata is not None:
            if self.anchor_metadata.get("source") not in (None, W1H_LABEL_SOURCE):
                raise ValueError("W1-H anchor_metadata source 必须是 haller_anchor_train")
            for field in ("algorithm_version", "parameter_hash", "input_hash", "mask_hash"):
                if field not in self.anchor_metadata:
                    raise ValueError(f"W1-H anchor_metadata 缺少 {field}")
        self.seed = _strict_nonnegative_int(seed, name="seed")
        if grad_clip_norm is not None:
            grad_clip_norm = float(grad_clip_norm)
            if not np.isfinite(grad_clip_norm) or grad_clip_norm <= 0.0:
                raise ValueError("W1-H grad_clip_norm 必须是正的有限数或 None")
        self.grad_clip_norm = grad_clip_norm
        self.global_step = 0

    def _move_models(self, device: str | torch.device) -> None:
        self.student.to(device)
        self.teacher.to(device)
        _prepare_ema_teacher(self.teacher)

    @staticmethod
    def _forward(model: nn.Module, batch: W1HBatch) -> Any:
        """调用现有 (dummy_field, 7-channel pathlines) model seam。"""
        if isinstance(model, contract.ChannelSelectingAdapter):
            return model.forward_batch(
                batch.contract_batch, dummy_field=batch.dummy_field, consumer="train")
        return model((batch.dummy_field, batch.pathlines))

    def train_step(
        self,
        batch: W1HBatch,
        *,
        epoch: int,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """执行一批 Haller anchor → pseudo/consistency → EMA 更新。"""
        if not isinstance(batch, W1HBatch):
            raise TypeError("W1-H train_step 必须消费 W1HBatch")
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        if batch.sampling_source != self.sampling_source:
            raise ValueError(
                f"W1-H sampling_source 不匹配：trainer={self.sampling_source!r} "
                f"batch={batch.sampling_source!r}"
            )
        if batch.anchor_hash != self.anchor_hash:
            raise ValueError(
                f"W1-H anchor_hash 不匹配：trainer={self.anchor_hash!r} "
                f"batch={batch.anchor_hash!r}"
            )
        self._move_models(device)
        model_batch = batch.to(device)
        self.student.train()
        _prepare_ema_teacher(self.teacher)
        self.optimizer.zero_grad(set_to_none=True)
        student_predictions = self._forward(self.student, model_batch)
        with torch.no_grad():
            teacher_predictions = self._forward(self.teacher, model_batch)
        loss, stats = compute_w1_h_loss(
            student_predictions, teacher_predictions, model_batch,
            config=self.config, epoch=epoch,
        )
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        update_ema_teacher(self.student, self.teacher, decay=self.config.ema_decay)
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
        """运行一个非空 guard 的 W1-H epoch，并聚合 anchor diagnostics。"""
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        if max_steps is not None:
            max_steps = _strict_nonnegative_int(max_steps, name="max_steps")
            if max_steps <= 0:
                raise ValueError("W1-H max_steps 必须是正整数或 None")
        logs = []
        for batch in batches:
            if max_steps is not None and len(logs) >= max_steps:
                break
            logs.append(self.train_step(batch, epoch=epoch, device=device))
        if not logs:
            raise ValueError("W1-H batches 为空：训练循环无样本可跑")
        average_keys = {
            "loss", "anchor_loss", "pseudo_loss", "consistency_loss",
            "ramp_weight", "anchor_coverage", "unknown_coverage",
            "solid_coverage", "failed_cell_coverage", "pseudo_acceptance",
            "teacher_student_disagreement", "artifact_failure_count",
        }
        count_keys = {
            "anchor_count", "anchor_positive_count", "anchor_negative_count",
            "unknown_count", "solid_count", "failed_cell_count", "failed_frame_count",
            "pseudo_eligible_count", "pseudo_accepted_count", "pseudo_positive_count",
            "pseudo_negative_count",
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
        if len(sources) != 1 or len(loss_sources) != 1 or len(anchor_hashes) != 1:
            raise ValueError("W1-H 一个 epoch 内 source/loss/anchor hash 发生漂移")
        summary.update({
            "sampling_source": next(iter(sources)),
            "loss_source": next(iter(loss_sources)),
            "anchor_hash": next(iter(anchor_hashes)),
            "epoch": epoch,
            "steps": len(logs),
            "global_step": self.global_step,
        })
        return summary

    def _checkpoint_extra_metadata(
        self,
        extra_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """构造不含 test GT、且 formal source 明确的 checkpoint metadata。"""
        if extra_metadata is not None and not isinstance(extra_metadata, Mapping):
            raise TypeError("W1-H extra_metadata 必须是 object")
        extra = dict(extra_metadata or {})
        reserved = {
            "generation_version": W1H_GENERATION_VERSION,
            "formal_loss_source": W1H_LABEL_SOURCE,
            "w1_h_config": self.config.as_dict(),
        }
        if self.anchor_metadata is not None:
            reserved["haller_anchor"] = copy.deepcopy(dict(self.anchor_metadata))
        for key, expected in reserved.items():
            if key in extra and extra[key] != expected:
                raise ValueError(f"W1-H checkpoint extra_metadata.{key} 与 trainer 语义不一致")
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
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """保存带 W1-H Haller source/hash 的可 resume checkpoint。"""
        if self.scheduler is None:
            raise ValueError("W1-H resume checkpoint 必须提供 scheduler")
        epoch = _strict_nonnegative_int(epoch, name="epoch")
        return contract.save_checkpoint(
            path, self.student, self.optimizer, self.scheduler,
            mode=contract.MODE_W1_H,
            feature_schema=contract.FEATURE_SCHEMA_7,
            adapter_input_schema=contract.FEATURE_SCHEMA_7,
            dataset_config=dataset_config,
            split_config=split_config,
            sampling_config=sampling_config,
            label_source=W1H_LABEL_SOURCE,
            sampling_source=self.sampling_source,
            teacher=self.teacher,
            epoch=epoch,
            global_step=self.global_step,
            metrics=metrics,
            seed=self.seed,
            anchor_hash=self.anchor_hash,
            calibration_policy=(
                {"source": "none"} if calibration_policy is None else calibration_policy
            ),
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
        """按 W1-H mode/source/schema/split/anchor hash 恢复 student 与 EMA。"""
        self._move_models(device)
        result = contract.load_checkpoint(
            path, self.student, self.optimizer, self.scheduler,
            teacher=self.teacher, device=device,
            expected_mode=contract.MODE_W1_H,
            expected_feature_schema=contract.FEATURE_SCHEMA_7,
            expected_dataset_config=expected_dataset_config,
            expected_split_config=expected_split_config,
            expected_sampling_config=expected_sampling_config,
            expected_label_source=W1H_LABEL_SOURCE,
            expected_sampling_source=self.sampling_source,
            expected_anchor_hash=self.anchor_hash,
            restore_rng=restore_rng,
            strict_cuda_rng=strict_cuda_rng,
            load_mode=load_mode,
        )
        self.global_step = _strict_nonnegative_int(
            result["global_step"], name="checkpoint global_step")
        self.seed = _strict_nonnegative_int(result["seed"], name="checkpoint seed")
        _prepare_ema_teacher(self.teacher)
        return result


__all__ = [
    "W1H_GENERATION_VERSION",
    "W1H_LABEL_SOURCE",
    "HallerTrainAnchor",
    "W1HBatch",
    "W1HConfig",
    "W1HTrainer",
    "HallerTrainArtifactResolver",
    "HallerAnchorPathlineDataset",
    "collate_w1_h_batches",
    "load_haller_train_artifact",
    "build_w1_h_batch",
    "build_w1_h_batch_from_anchor",
    "compute_w1_h_loss",
]
