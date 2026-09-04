"""弱标签迹线数据集（dataset.py）——旧 B0 兼容与弱监督 split 契约。

领域词汇（HANDOFF §4/§6，唯一权威）：
- 旧 B0 产物继续支持绝对时间片与 frac 60/40；新弱监督产物使用每个数据集
  自己的 frame-index 半开区间 train/calibration/test = 0/50/60/100；
- patch 32×32 stride 16、窗口 T_win=24 帧、窗口起点步长 4 帧；
- u,v 与预计算 IVD 存 memmap（IVD 一次算好；≈405MB @ float32）；
- 7 通道 = [px, py, t, ivd, distance(距种子), u, v]（extractor.N_CHANNELS 口径）；
  归一化：px,py → patch 内 [-1,1]（extractor 已做）；t → [0,1]×t_scale（默认 0.25）；
  ivd 标准化（z-score，μ/σ 取 train 片流体区，避免固体 0 值污染与时间泄漏）；
  distance 用归一化坐标（hypot 归一化重构）；u,v ÷ 冻结的 train 流体最大速度；
- 返回 ((dummy_field, pathlines), labels) 匹配模型输入（PathlineTransformerV0 取
  data[1] = pathline_src (B, L, K, C)；dummy_field = zeros((1,1,1,1)) 参考口径）；
- 标签 = 重播种后种子格处 label_field 值（label_field 含 5×5 面积过滤与固体强制 0）；
- 每 epoch 40000 样本（下限 20000）、50% 正样本过采样（正样本 = patch 内存在
  ≥1 条涡迹线；池判据与 weak_labels.patch_positive_map 单公式共用）；
- 多数据集（票 07 延伸，HANDOFF §1 决策 8 落实）：MultiDatasetPathlineDataset
  合并多个 prepare_dataset 产物做采样；旧产物保持 frac 口径，新弱监督产物
  必须由显式 weak_supervision split/source metadata 驱动。

性能说明（验收记录披露）：on-the-fly 提取 = extract_pathlines_batched（真实窗口
实测 ~35ms；服务器多进程 DataLoader 可隐藏大部分加载时间，不构成训练瓶颈；
本地 <5ms 预算未达成，用户已确认不纠结——仅要求能跑）。

实现约束：h5py 直读中文路径（prepare_dataset 的 nc_path 分支）；纯 python/numpy
（遵守 §2 依赖清单：torch、numpy、h5py、yaml、matplotlib、tqdm）。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import zlib

import numpy as np

import extractor
import weak_labels

# --------------------------------------------------------------------------- 默认参数（HANDOFF §6）

DEFAULT_PATCH_SIZE = (32, 32)
DEFAULT_STRIDE = (16, 16)
DEFAULT_T_WIN = 24
DEFAULT_WINDOW_STEP = 4
DEFAULT_SAMPLES_PER_EPOCH = 40000
# 每 epoch 样本数规格下限 20000（HANDOFF §6：默认 40000、下限 20000）——
# 训练配置的语义约束（训练脚本选用 ≥20000），不作为运行时钳制
#（合成/测试可用更小值快速迭代）。
DEFAULT_POSITIVE_FRACTION = 0.5
DEFAULT_T_SCALE = 0.25
DEFAULT_L = 16
DEFAULT_GROUPS = (8, 8)
DEFAULT_DELTA_FRAC = 0.05

# 弱监督 feature 的 split 合同。阶段 0 的 ``abs``/``frac`` 口径仍保留给
# 旧 B0 产物；新 feature 必须显式使用下面的三段 frame-index 合同。
WEAK_SUPERVISION_SPLIT_MODE = "weak_supervision"
WEAK_SUPERVISION_SPLITS = ("train", "calibration", "test")
VALID_WEAK_DATASETS = (
    "boussinesq",
    "cylinder2d",
    "doublegyre2d",
    "fourcenters2d",
    "jungtelziemniak2d",
    "pipedcylinder2d",
)
VALID_LABEL_SOURCES = (
    "legacy_p85",
    "local_p90_p60",
    "haller_anchor_train",
    "haller_gt_calibration",
    "haller_gt_test",
)
NORMALIZATION_SOURCE = "train"
GENERATION_VERSION = "b1-w1-w2-w3-split-label-v1"
VALID_CONSUMERS = ("train", "calibration", "evaluation")
SOURCE_SPLIT_REQUIREMENTS = {
    "haller_anchor_train": "train",
    "haller_gt_calibration": "calibration",
    "haller_gt_test": "test",
}
FEATURE_SCHEMA = {
    "name": "pathline_7ch",
    "version": "v1",
    "channels": ["px", "py", "t", "ivd", "distance", "u", "v"],
    "channel_count": 7,
    "local_ivd_channel": 3,
}

# 存储文件名（prepare_dataset 与 WeakLabelPathlineDataset 共用，防路径漂移）
FN_U = "u.npy"
FN_V = "v.npy"
FN_IVD = "ivd.npy"
FN_LABEL = "label_field.npy"
FN_MASK = "mask.npy"
FN_META = "meta.json"


# --------------------------------------------------------------------------- 采样几何（时间划分 / patch 位置）

def weak_supervision_slices(T, *, dataset_name="dataset"):
    """为弱监督 feature 生成严格的 train/calibration/test 半开区间。

    每个数据集都按自己的 frame 数 ``T`` 计算 ``floor(0.50*T)`` 和
    ``floor(0.60*T)``，不使用绝对物理时间，也不沿用阶段 0 的旧默认切分。
    三个区间必须非空且连续覆盖 ``[0, T)``；无法满足时立即报错。
    """
    try:
        total = _as_int(T, name="T")
    except ValueError as exc:
        raise ValueError(f"数据集 {dataset_name!r} 的 T 必须是正整数，实际 {T!r}") from exc
    if total <= 0:
        raise ValueError(f"数据集 {dataset_name!r} 的 T 必须是正整数，实际 {T!r}")
    # 用整数算术表达 floor(0.50*T) / floor(0.60*T)，避免浮点边界误差。
    s50 = total // 2
    s60 = (3 * total) // 5
    if s50 <= 0 or s60 <= s50 or total <= s60:
        raise ValueError(
            f"数据集 {dataset_name!r} 的 T={total} 无法生成非空弱监督 split："
            f"boundary=[0, {s50}, {s60}, {total}]"
        )
    return {
        "train": (0, s50),
        "calibration": (s50, s60),
        "test": (s60, total),
    }


def validate_label_source(source):
    """校验并返回一个受注册的 label source 名称。"""
    if source not in VALID_LABEL_SOURCES:
        raise ValueError(
            f"未知 label source {source!r}；允许值为 {list(VALID_LABEL_SOURCES)}"
        )
    return str(source)


def _json_hash(value):
    """对可 JSON 序列化的契约字段生成稳定 SHA-256。"""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_int(value, *, name):
    """将 frame/window 参数转为整数；拒绝 bool 与非整数数值的截断。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必须是整数，实际 {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是整数，实际 {value!r}") from exc
    try:
        integral = bool(converted == value)
    except (TypeError, ValueError):
        integral = False
    if not integral:
        raise ValueError(f"{name} 必须是整数，实际 {value!r}")
    return converted


def _validate_consumer(consumer):
    """校验数据源消费者角色，避免训练/评价语义隐式混用。"""
    if consumer not in VALID_CONSUMERS:
        raise ValueError(
            f"未知数据消费者 {consumer!r}；允许值为 {list(VALID_CONSUMERS)}"
        )
    return str(consumer)


def _validate_label_provenance(actual_source, *, consumer="train",
                               declared_source=None, additional_sources=()):
    """按消费者角色执行标签来源隔离，并返回实际 source。

    ``haller_gt_test`` 只有 evaluation 消费者可以读取，而且必须在调用方
    显式声明同名 source；训练和 calibration 统一拒绝它。训练也拒绝
    ``haller_gt_calibration``。旧 metadata 缺少
    provenance 字段时仅按阶段 0 兼容口径视为 ``legacy_p85``，新弱监督
    metadata 会在写入时完整记录来源。
    """
    actual_source = validate_label_source(actual_source)
    role = _validate_consumer(consumer)
    if declared_source is not None:
        declared_source = validate_label_source(declared_source)
        if declared_source != actual_source:
            raise ValueError(
                f"声明的 label source {declared_source!r} 与 metadata source "
                f"{actual_source!r} 不一致"
            )
    sources = (actual_source, *tuple(
        validate_label_source(source) for source in additional_sources))
    forbidden_for_train = {"haller_gt_calibration", "haller_gt_test"}
    forbidden_for_calibration = {"haller_gt_test"}
    forbidden = None
    if role == "train":
        forbidden = next(
            (source for source in sources if source in forbidden_for_train), None)
    elif role == "calibration":
        forbidden = next(
            (source for source in sources if source in forbidden_for_calibration), None)
    if forbidden is not None:
        raise ValueError(
            f"消费者 {role} 禁止读取 {forbidden}：Haller calibration/test source "
            "不能进入训练或 calibration consumer"
        )
    if "haller_gt_test" in sources:
        if role == "evaluation" and declared_source != "haller_gt_test":
            raise ValueError(
                "evaluation 读取 haller_gt_test requires explicit declaration: "
                "label_source='haller_gt_test'"
            )
        if role != "evaluation":
            raise ValueError(
                f"消费者 {role} 禁止读取 haller_gt_test：test Haller GT 只能由 "
                "evaluation 显式传入"
            )
    return actual_source


def _validate_source_split(source, split, *, dataset_name):
    """校验 Haller source 与实际读取 split 一一对应。"""
    required_split = SOURCE_SPLIT_REQUIREMENTS.get(source)
    if required_split is not None and split != required_split:
        raise ValueError(
            f"数据集 {dataset_name} 的 label source {source!r} 只能读取 "
            f"split={required_split!r}，实际为 {split!r}"
        )


def _weak_contract_payload(meta):
    """返回弱监督 metadata 中参与 hash 的不可变契约字段。"""
    return {
        "dataset_name": meta["dataset_name"],
        "generation_version": meta["generation_version"],
        "split_mode": meta["split_mode"],
        "split_ranges": meta["split_ranges"],
        "split_metadata": meta["split_metadata"],
        "window": meta["window"],
        "feature_schema": meta["feature_schema"],
        "label_provenance": meta["label_provenance"],
        "normalization_source": meta["normalization_source"],
        "normalization": meta["normalization"],
    }


def _metadata_dict(meta, field, *, dataset_name):
    """取出 dict metadata 字段，并将篡改/损坏统一转换为 ValueError。"""
    value = meta.get(field)
    if not isinstance(value, dict):
        raise ValueError(
            f"数据集 {dataset_name} 的 weak supervision metadata 字段 "
            f"{field!r} 必须为 object"
        )
    return value


def _validate_weak_contract_metadata(meta, *, dataset_name, total_frames):
    """校验弱监督 metadata 的不可变契约和生成 hash。"""
    required = (
        "dataset_name", "split_mode", "slices", "split_ranges", "split_metadata",
        "window", "feature_schema", "label_provenance", "generation_version",
        "generation_hash", "contract_hash", "normalization_source",
        "normalization_frozen",
    )
    missing = [name for name in required if name not in meta]
    if missing:
        raise ValueError(
            f"数据集 {dataset_name} 的 weak supervision metadata 缺少字段 {missing}"
        )
    if str(meta["dataset_name"]) != str(dataset_name):
        raise ValueError(
            f"数据集 {dataset_name} 的 metadata dataset_name 不一致："
            f"{meta['dataset_name']!r}"
        )
    if meta["split_mode"] != WEAK_SUPERVISION_SPLIT_MODE:
        raise ValueError(
            f"数据集 {dataset_name} 的 split_mode 不支持：{meta['split_mode']!r}"
        )
    expected_slices = weak_supervision_slices(
        total_frames, dataset_name=dataset_name)

    def _ranges(field):
        value = _metadata_dict(meta, field, dataset_name=dataset_name)
        if set(value) != set(expected_slices):
            raise ValueError(
                f"数据集 {dataset_name} 的 {field} keys 不匹配："
                f"期望 {sorted(expected_slices)}，收到 {sorted(value)}"
            )
        result = {}
        for name in WEAK_SUPERVISION_SPLITS:
            bounds = value[name]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(
                    f"数据集 {dataset_name} 的 {field}[{name!r}] 必须是二元边界"
                )
            try:
                result[name] = (
                    _as_int(bounds[0], name=f"{field}[{name}].start"),
                    _as_int(bounds[1], name=f"{field}[{name}].end"),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"数据集 {dataset_name} 的 {field}[{name!r}] 边界非法：{bounds!r}"
                ) from exc
        return result

    actual_ranges = _ranges("split_ranges")
    actual_slices = _ranges("slices")
    if actual_ranges != expected_slices or actual_slices != expected_slices:
        raise ValueError(
            f"数据集 {dataset_name} 的 split_ranges/slices 不匹配："
            f"期望 {expected_slices}，收到 ranges={actual_ranges}, "
            f"slices={actual_slices}"
        )

    window = _metadata_dict(meta, "window", dataset_name=dataset_name)
    try:
        t_win = _as_int(window["t_win"], name="window.t_win")
        window_step = _as_int(window["window_step"], name="window.window_step")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"数据集 {dataset_name} 的 window metadata 缺少有效 t_win/window_step"
        ) from exc
    if (t_win <= 0 or window_step <= 0 or window.get("frame_unit") != "index"
            or window.get("complete_only") is not True):
        raise ValueError(
            f"数据集 {dataset_name} 的 window metadata 必须是正整数、frame-index、"
            "complete_only=true"
        )
    counts = window.get("counts_by_split")
    if not isinstance(counts, dict) or set(counts) != set(WEAK_SUPERVISION_SPLITS):
        raise ValueError(
            f"数据集 {dataset_name} 的 window counts_by_split 不完整"
        )
    split_metadata = _metadata_dict(meta, "split_metadata", dataset_name=dataset_name)
    if set(split_metadata) != set(WEAK_SUPERVISION_SPLITS):
        raise ValueError(f"数据集 {dataset_name} 的 split_metadata 不完整")
    for split_name in WEAK_SUPERVISION_SPLITS:
        info = split_metadata[split_name]
        if not isinstance(info, dict):
            raise ValueError(
                f"数据集 {dataset_name} 的 split_metadata[{split_name!r}] 非 object"
            )
        try:
            info_range = (
                _as_int(info["frame_start"], name="split_metadata.frame_start"),
                _as_int(info["frame_end"], name="split_metadata.frame_end"),
            )
            info_count = _as_int(info["window_count"], name="split_metadata.window_count")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"数据集 {dataset_name} 的 split_metadata[{split_name!r}] 非法"
            ) from exc
        if (info.get("split_name") != split_name
                or info.get("half_open") is not True
                or info_range != actual_ranges[split_name]):
            raise ValueError(
                f"数据集 {dataset_name} 的 split_metadata[{split_name!r}] 边界非法"
            )
        expected_count = len(window_starts(
            *actual_ranges[split_name], t_win=t_win, step=window_step,
            dataset_name=dataset_name, split_name=split_name, T=total_frames,
        ))
        try:
            count = _as_int(
                counts[split_name], name=f"window.counts_by_split[{split_name}]"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"数据集 {dataset_name} 的 window counts_by_split[{split_name!r}] 非法"
            ) from exc
        if info_count != expected_count or count != expected_count:
            raise ValueError(
                f"数据集 {dataset_name} 的 {split_name} window_count 不一致"
            )

    if meta["normalization_source"] != NORMALIZATION_SOURCE:
        raise ValueError(
            f"数据集 {dataset_name} 的 normalization source 必须为 "
            f"{NORMALIZATION_SOURCE!r}"
        )
    if meta["normalization_frozen"] is not True:
        raise ValueError(f"数据集 {dataset_name} 的 normalization 必须冻结")
    normalization = _metadata_dict(meta, "normalization", dataset_name=dataset_name)
    if (normalization.get("source_split") != NORMALIZATION_SOURCE
            or normalization.get("frozen") is not True):
        raise ValueError(
            f"数据集 {dataset_name} 的 normalization metadata 必须冻结于 train"
        )
    for stat_name in ("ivd_mu", "ivd_sigma", "speed_max"):
        try:
            metadata_stat = float(meta[stat_name])
            frozen_stat = float(normalization[stat_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"数据集 {dataset_name} 的 normalization 缺少 {stat_name}"
            ) from exc
        if (not np.isfinite(metadata_stat) or not np.isfinite(frozen_stat)
                or metadata_stat != frozen_stat
                or (stat_name == "ivd_sigma" and frozen_stat < 0)
                or (stat_name == "speed_max" and frozen_stat <= 0)):
            raise ValueError(
                f"数据集 {dataset_name} 的 normalization {stat_name} 不一致"
            )

    schema = _metadata_dict(meta, "feature_schema", dataset_name=dataset_name)
    try:
        schema_channel_count = _as_int(
            schema.get("channel_count"), name="feature_schema.channel_count")
        schema_ivd_channel = _as_int(
            schema.get("local_ivd_channel"), name="feature_schema.local_ivd_channel")
    except ValueError as exc:
        raise ValueError(
            f"数据集 {dataset_name} 的 feature schema 数值字段非法"
        ) from exc
    if (schema.get("name") != FEATURE_SCHEMA["name"]
            or schema.get("version") != FEATURE_SCHEMA["version"]
            or schema.get("channels") != FEATURE_SCHEMA["channels"]
            or schema_channel_count != FEATURE_SCHEMA["channel_count"]
            or schema_ivd_channel != FEATURE_SCHEMA["local_ivd_channel"]):
        raise ValueError(
            f"数据集 {dataset_name} 的 feature schema 与当前 7 通道契约不一致"
        )
    provenance = _metadata_dict(meta, "label_provenance", dataset_name=dataset_name)
    if set(provenance) != {"field_source", "sampling_source", "loss_source"}:
        raise ValueError(f"数据集 {dataset_name} 的 label provenance 字段不完整")
    for source in provenance.values():
        validate_label_source(source)
    if (meta.get("label_source") != provenance["field_source"]
            or meta.get("sampling_source") != provenance["sampling_source"]
            or meta.get("loss_label_source") != provenance["loss_source"]):
        raise ValueError(f"数据集 {dataset_name} 的 label provenance 字段不一致")
    if meta["generation_version"] != GENERATION_VERSION:
        raise ValueError(
            f"数据集 {dataset_name} 的 generation version 不受支持："
            f"{meta['generation_version']!r}"
        )
    payload = _weak_contract_payload(meta)
    expected_hash = _json_hash(payload)
    if (not isinstance(meta["generation_hash"], str)
            or not isinstance(meta["contract_hash"], str)
            or meta["generation_hash"] != expected_hash
            or meta["contract_hash"] != expected_hash):
        raise ValueError(
            f"数据集 {dataset_name} 的 generation hash/contract hash 校验失败"
        )


def _validate_weak_consumer_split(consumer, split, *, dataset_name):
    """将弱监督消费者限制到其声明的 split，阻止隐式 test/calibration 读取。"""
    expected = {"train": "train", "calibration": "calibration",
                "evaluation": "test"}[consumer]
    if split != expected:
        raise ValueError(
            f"数据集 {dataset_name} 的 consumer={consumer!r} 只能读取 "
            f"split={expected!r}，实际为 {split!r}"
        )


def validate_window_start(start, *, split_start, split_end, t_win,
                          dataset_name="dataset", split_name="split", T=None):
    """验证一个窗口起点完整落在单一半开 split 内，并返回整数起点。

    ``start + t_win == split_end`` 是合法的最后一个窗口；越界和负长度都
    fail loudly，错误消息保留 dataset、总帧数、split boundary 与窗口长度，
    便于定位数据准备配置错误。
    """
    try:
        start = _as_int(start, name="start")
        split_start = _as_int(split_start, name="split_start")
        split_end = _as_int(split_end, name="split_end")
        t_win = _as_int(t_win, name="t_win")
    except ValueError as exc:
        raise ValueError(
            f"数据集 {dataset_name!r} split={split_name!r} window parameters "
            "must be integer："
            f"start={start!r}, boundary=[{split_start!r}, {split_end!r}), "
            f"t_win={t_win!r}"
        ) from exc
    try:
        total = split_end if T is None else _as_int(T, name="T")
    except ValueError as exc:
        raise ValueError(
            f"数据集 {dataset_name!r} split={split_name!r} 的 T 必须是整数，实际 {T!r}"
        ) from exc
    boundary = f"[{split_start}, {split_end})"
    if split_start < 0 or split_end < split_start or t_win <= 0:
        raise ValueError(
            f"数据集 {dataset_name} T={total} split={split_name} boundary={boundary} "
            f"参数非法：start={start}, t_win={t_win}"
        )
    if not (split_start <= start and start + t_win <= split_end):
        raise ValueError(
            f"数据集 {dataset_name} T={total} split={split_name} boundary={boundary} "
            f"窗口越界：start={start}, t_win={t_win}"
        )
    return start


def window_starts(i0, i1, t_win=DEFAULT_T_WIN, step=DEFAULT_WINDOW_STEP, *,
                  dataset_name="dataset", split_name="split", T=None):
    """枚举完全落在半开时间片 ``[i0, i1)`` 内的窗口起点。

    返回按 ``step`` 递增的 ``np.ndarray``。split 长度不足以容纳一个完整
    ``t_win`` 时不会返回空数组，而是带上下文 fail loudly；这条严格语义
    是弱监督新 split 的防泄漏守卫。
    """
    try:
        i0 = _as_int(i0, name="split_start")
        i1 = _as_int(i1, name="split_end")
        t_win = _as_int(t_win, name="t_win")
        step = _as_int(step, name="window_step")
    except ValueError as exc:
        raise ValueError(
            f"数据集 {dataset_name!r} split={split_name!r} window parameters "
            "must be integer："
            f"boundary=[{i0!r}, {i1!r}), t_win={t_win!r}, step={step!r}"
        ) from exc
    try:
        total = i1 if T is None else _as_int(T, name="T")
    except ValueError as exc:
        raise ValueError(
            f"数据集 {dataset_name!r} split={split_name!r} 的 T 必须是整数，实际 {T!r}"
        ) from exc
    if step <= 0:
        raise ValueError(
            f"数据集 {dataset_name} T={total} split={split_name} "
            f"boundary=[{i0}, {i1}) 的 window step 必须为正，实际 {step}"
        )
    if i1 - i0 < t_win:
        raise ValueError(
            f"数据集 {dataset_name} T={total} split={split_name} "
            f"boundary=[{i0}, {i1}) 长度不足以容纳完整窗口，t_win={t_win}"
        )
    starts = np.arange(i0, i1 - t_win + 1, step, dtype=np.intp)
    # 保留显式 validator 作为单一边界断言，防止未来枚举逻辑放宽约束。
    for start in starts:
        validate_window_start(
            int(start), split_start=i0, split_end=i1, t_win=t_win,
            dataset_name=dataset_name, split_name=split_name, T=total,
        )
    return starts


def fraction_slices(T, train_frac=0.6, val_frac=0.0):
    """按帧比例的时间片划分（票 07 延伸：多数据集按时间 60/40，无 val 时仅 train/test）。

    DEFAULT_SLICES 为绝对秒数口径（仅适用 1501 帧/15s 的 pipedcylinder2d）；
    多数据集帧数/时长各异（512~2001 帧、t∈[0,20]，jung telziemniak 的 t
    从 1.107 起）→ 按帧比例划分才通用。返回 {name: (i0, i1)}：
    train [0, i1)、val [i1, i2)（val_frac>0 时）、test [i2, T)；累积取整
    （正数 floor = int()）→ 三片（或两片）全覆盖、无时间泄漏（与
    DEFAULT_SLICES 同闭包语义）。train_frac ∈ (0,1)（严格）、
    val_frac ∈ [0, 1−train_frac)（留出非空），违规 fail loud。
    """
    T, train_frac, val_frac = int(T), float(train_frac), float(val_frac)
    if T < 2:
        raise ValueError(f"T 过小无法划分: {T}")
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac 必须在 (0,1) 内，实际 {train_frac}")
    if val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError(f"val_frac 必须在 [0, 1−train_frac) 内，实际 {val_frac}")
    i1 = int(T * train_frac)
    i2 = int(T * (train_frac + val_frac))
    if i1 <= 0 or i2 < i1:
        raise ValueError(f"时间片划分过窄: train_end={i1} val_end={i2}")
    out = {"train": (0, i1)}
    if val_frac > 0:
        out["val"] = (i1, i2)
    out["test"] = (i2, T)
    return out


def patch_locations(H, W, patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE):
    """patch 起点格网格（y 外 x 内；与 weak_labels.patch_positive_map 的行列序一致）。

    返回 list[(y0, x0)]，全部满足 (y0 ≤ H−ph) 且 (x0 ≤ W−pw)。
    """
    ph, pw = patch_size
    sy, sx = stride
    return [(int(y0), int(x0))
            for y0 in range(0, H - ph + 1, sy)
            for x0 in range(0, W - pw + 1, sx)]


# --------------------------------------------------------------------------- 7 通道归一化

def normalize_pathlines(raw, seeds, geo, t0, t_span, t_scale, ivd_mu, ivd_sigma,
                        speed_max):
    """extractor 输出 → 全归一化样本 (L, K=256, 7) float32。

    口径（spec Implementation Decisions）：
    - CH_PX/CH_PY：保持 extractor 的 patch 内归一化 [-1,1]（可超界）；
    - CH_T：t → [0,1]×t_scale（t_span = 窗口物理时长 = (T_win−1)×dt）；
    - CH_IVD：(ivd − ivd_mu) / ivd_sigma（z-score；ivd_sigma≤0 时置 0 通道防御）；
    - CH_DIST：重算为归一化坐标下距（重播种后）种子的距离
      hypot(px − sx_n, py − sy_n)，sx_n/sy_n = (seed − center) / half；
    - CH_U/CH_V：÷ speed_max（全局最大速度）。
    raw: (L, K, 7) float32；seeds: (K, 2) 物理坐标（重播种后）；geo: patch_geometry。
    """
    raw = np.asarray(raw, dtype=np.float32)
    out = raw.copy()
    out[:, :, extractor.CH_T] = (
        (raw[:, :, extractor.CH_T] - float(t0)) / float(t_span) * float(t_scale))
    if ivd_sigma is not None and float(ivd_sigma) > 0:
        out[:, :, extractor.CH_IVD] = (
            (raw[:, :, extractor.CH_IVD] - float(ivd_mu)) / float(ivd_sigma))
    else:
        out[:, :, extractor.CH_IVD] = 0.0
    seed_px = (seeds[:, 0] - geo["cx"]) / geo["hx"]
    seed_py = (seeds[:, 1] - geo["cy"]) / geo["hy"]
    out[:, :, extractor.CH_DIST] = np.hypot(
        out[:, :, extractor.CH_PX] - seed_px[None, :],
        out[:, :, extractor.CH_PY] - seed_py[None, :])
    out[:, :, extractor.CH_U] = raw[:, :, extractor.CH_U] / float(speed_max)
    out[:, :, extractor.CH_V] = raw[:, :, extractor.CH_V] / float(speed_max)
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------- 元数据

def load_dataset_meta(data_root):
    """读取 data_root/meta.json → dict（含 shape/坐标/slices/taus/speed_max/IVD 统计）。"""
    root = pathlib.Path(data_root)
    meta_path = root / FN_META
    if not meta_path.exists():
        raise FileNotFoundError(f"数据集元数据缺失: {meta_path}（先运行 prepare_dataset）")
    return json.loads(meta_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 预计算（memmap 落盘）

def _mask_2d(mask, Y, X):
    """掩膜规格化为 (Y,X) bool：None → 空掩膜（无障碍物数据集路径不变）。"""
    if mask is None:
        return np.zeros((Y, X), dtype=bool)
    if isinstance(mask, (str, pathlib.Path)):
        m = np.load(str(mask))
    else:
        m = np.asarray(mask)
    m = m.astype(bool)
    if m.ndim == 3:
        m = m[0]
    if m.shape != (Y, X):
        raise ValueError(f"掩膜形状 {m.shape} ≠ (Y,X)={(Y, X)}")
    return m


def _fit_slices(slices, T):
    """时间片截断到数据集帧数（子集数据集时：i1 截断到 T；i0 ≥ T 的片剔除）。

    与 weak_labels CLI 的截断口径一致（覆盖全部帧的无泄漏划分）。
    """
    out = {}
    for name, (i0, i1) in slices.items():
        if i0 < T:
            out[name] = (int(i0), min(int(i1), T))
    if not out:
        raise ValueError("时间片全部超出数据集帧数")
    return out


def prepare_dataset(nc_path=None, out_dir="outputs/dataset", *,
                    u=None, v=None, xdim=None, ydim=None, tdim=None,
                    mask=None, ivd=None, labels=None, min_area=weak_labels.DEFAULT_MIN_AREA,
                    percentile=weak_labels.DEFAULT_PERCENTILE, taus=None, slices=None,
                    split_mode="abs", train_frac=0.6, val_frac=0.0,
                    speed_max=None, ivd_stats_slice="train", ivd_mu=None, ivd_sigma=None,
                    dataset_name=None, label_source=None, sampling_source=None,
                    loss_label_source=None,
                    patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE,
                    t_win=DEFAULT_T_WIN, window_step=DEFAULT_WINDOW_STEP):
    """数据准备（memmap 预计算）：u/v/ivd/label/mask 落盘 + meta.json（返回 meta dict）。

    输入：nc_path（h5py 直读，中文路径可用；u/v 逐帧流式，不全量驻留）
    或内存数组 (u, v, xdim, ydim, tdim)（合成场/测试）。
    复用/覆盖：mask（None=无固体；(Y,X) 或 (T,Y,X) 数组/路径）、ivd（None=自算，
    数组/路径=复用票 04 产物）、labels（None=build_label_field）、taus（None=按
    percentile 在流体区逐时间片统计——弱标签口径）。legacy ``abs``/``frac``
    保留旧 B0 口径；``weak_supervision`` 必须显式传 label source，并固定
    train/calibration/test 三段与 train-only normalization。

    归一化统计（写 meta）：ivd_mu/ivd_sigma = normalization source 片内流体区 IVD 的
    均值/标准差；新弱监督模式固定为 train，且 speed_max 同样只从 train 流体格生成。
    σ=0 时写 0（normalize 防除零）。
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_mode = str(split_mode)
    valid_split_modes = ("abs", "frac", WEAK_SUPERVISION_SPLIT_MODE)
    if split_mode not in valid_split_modes:
        raise ValueError(
            f"未知 split_mode {split_mode!r}；允许值为 {list(valid_split_modes)}"
        )
    if dataset_name is None:
        dataset_name = (pathlib.Path(nc_path).stem if nc_path is not None
                        else out_dir.name)
    dataset_name = str(dataset_name)
    if not dataset_name:
        raise ValueError("dataset_name 不能为空")

    # 旧 prepare_dataset 调用继续按阶段 0 兼容口径生成 legacy p85；新弱监督
    # 模式必须显式声明来源，避免悄悄回退到旧标签或旧 split。
    if label_source is None:
        if split_mode == WEAK_SUPERVISION_SPLIT_MODE:
            raise ValueError(
                f"数据集 {dataset_name} 的弱监督 prepare 必须显式指定 label_source；"
                f"不能回退到 legacy_p85"
            )
        label_source = "legacy_p85"
    label_source = validate_label_source(label_source)
    sampling_source = validate_label_source(
        label_source if sampling_source is None else sampling_source)
    loss_label_source = validate_label_source(
        label_source if loss_label_source is None else loss_label_source)
    if split_mode == WEAK_SUPERVISION_SPLIT_MODE:
        if loss_label_source == "legacy_p85" and label_source != "legacy_p85":
            raise ValueError(
                "legacy_p85 不能作为非 legacy label source 的 formal loss source"
            )
        if ivd_stats_slice != NORMALIZATION_SOURCE:
            raise ValueError(
                "weak supervision normalization source 只允许 train，"
                f"不能使用 {ivd_stats_slice!r}"
            )
        if speed_max is not None or ivd_mu is not None or ivd_sigma is not None:
            raise ValueError(
                "weak supervision normalization 必须由 train fluid data 生成，"
                "不能注入外部 speed_max/ivd_mu/ivd_sigma"
            )
        if labels is None and label_source != "legacy_p85":
            raise ValueError(
                f"数据集 {dataset_name} 的 label_source={label_source!r} "
                "必须显式提供 labels，不能静默生成 legacy_p85 标签"
            )

    # ---- 坐标与形状
    if nc_path is not None:
        import h5py
        with h5py.File(str(nc_path), "r") as f:
            xdim = f["xdim"][:].astype(np.float64)
            ydim = f["ydim"][:].astype(np.float64)
            tdim = f["tdim"][:].astype(np.float64)
            T = len(tdim)
    xdim = np.asarray(xdim, dtype=np.float64)
    ydim = np.asarray(ydim, dtype=np.float64)
    tdim = np.asarray(tdim, dtype=np.float64)
    T, Y, X = len(tdim), len(ydim), len(xdim)
    if split_mode == WEAK_SUPERVISION_SPLIT_MODE:
        expected_slices = weak_supervision_slices(
            T, dataset_name=dataset_name)
        if slices is not None:
            supplied_slices = {}
            try:
                for name, bounds in slices.items():
                    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                        raise ValueError(
                            f"{name!r} 必须提供二元 frame boundary"
                        )
                    supplied_slices[str(name)] = (
                        _as_int(bounds[0], name=f"{name}.start"),
                        _as_int(bounds[1], name=f"{name}.end"),
                    )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"数据集 {dataset_name} 的 weak supervision split 边界必须是整数："
                    f"{slices!r}"
                ) from exc
            if supplied_slices != expected_slices:
                raise ValueError(
                    f"数据集 {dataset_name} 的 weak supervision split 必须使用 "
                    f"{expected_slices}，收到 {supplied_slices}"
                )
        slices = expected_slices
        # prepare 阶段就验证每个 split 能容纳完整窗口；真正枚举时 _DatasetStore
        # 仍会再次验证，防止 metadata 被手工篡改后静默放宽边界。
        for split_name, (i0, i1) in slices.items():
            window_starts(
                i0, i1, t_win=t_win, step=window_step,
                dataset_name=dataset_name, split_name=split_name, T=T,
            )
    if u is not None:
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        if u.shape != (T, Y, X) or v.shape != (T, Y, X):
            raise ValueError(f"u/v 形状需为 (T,Y,X)={(T, Y, X)}，实际 {u.shape}")

    # ---- 掩膜（先规格化：流式 IVD 计算需要）
    mask2d = _mask_2d(mask, Y, X)
    np.save(out_dir / FN_MASK, mask2d.astype(np.uint8))

    speed_max_given = speed_max is not None

    # ---- u/v/IVD：nc 流式 或 内存数组
    if nc_path is not None:
        import h5py
        with h5py.File(str(nc_path), "r") as f:
            umm = np.lib.format.open_memmap(out_dir / FN_U, mode="w+",
                                            dtype=np.float32, shape=(T, Y, X))
            vmm = np.lib.format.open_memmap(out_dir / FN_V, mode="w+",
                                            dtype=np.float32, shape=(T, Y, X))
            reuse_ivd = ivd is not None and isinstance(ivd, (str, pathlib.Path))
            if not reuse_ivd:
                ivd_mm = np.lib.format.open_memmap(out_dir / FN_IVD, mode="w+",
                                                   dtype=np.float32, shape=(T, Y, X))
            sp = 0.0
            for t in range(T):
                ut = np.asarray(f["u"][t], dtype=np.float32)
                vt = np.asarray(f["v"][t], dtype=np.float32)
                umm[t] = ut
                vmm[t] = vt
                sp = max(sp, float(np.hypot(ut, vt).max()))
                if not reuse_ivd:
                    ivd_mm[t] = weak_labels.compute_ivd(
                        ut[None], vt[None], xdim, ydim, mask=mask2d).astype(np.float32)[0]
            umm.flush(); vmm.flush()
            del umm, vmm
            if not reuse_ivd:
                ivd_mm.flush()
                del ivd_mm
        speed_max = sp if speed_max is None else float(speed_max)
        ivd_source = "computed_streaming" if not reuse_ivd else "provided"
    else:
        umm = np.lib.format.open_memmap(out_dir / FN_U, mode="w+",
                                        dtype=np.float32, shape=(T, Y, X))
        vmm = np.lib.format.open_memmap(out_dir / FN_V, mode="w+",
                                        dtype=np.float32, shape=(T, Y, X))
        umm[:] = u
        vmm[:] = v
        umm.flush(); vmm.flush()
        del umm, vmm
        speed_max = float(np.hypot(u, v).max()) if speed_max is None else float(speed_max)
        ivd_source = "computed_inmemory"
        if ivd is not None:
            ivd_arr = np.asarray(ivd, dtype=np.float32)[:T]
            np.save(out_dir / FN_IVD, ivd_arr)
        else:
            np.save(out_dir / FN_IVD,
                    weak_labels.compute_ivd(u, v, xdim, ydim, mask=mask2d).astype(np.float32))

    # ---- IVD 复用路径（票 04 产物）
    if ivd is not None and isinstance(ivd, (str, pathlib.Path)):
        ivd_arr = np.asarray(np.load(str(ivd)), dtype=np.float32)[:T]
        np.save(out_dir / FN_IVD, ivd_arr)
        ivd_source = "provided"

    # ---- 时间片与 τ
    if split_mode == "frac":
        # 票 07 延伸：按帧比例划分（60/40，可带 val）——多数据集通用口径
        slices = fraction_slices(T, train_frac=train_frac, val_frac=val_frac)
    slices = _fit_slices(slices or weak_labels.DEFAULT_SLICES, T)

    # 新合同的速度归一化统计也只能来自 train 流体格。旧 ``abs``/``frac``
    # 产物保留阶段 0 的全场 speed_max 口径，避免重写历史 baseline。
    if split_mode == WEAK_SUPERVISION_SPLIT_MODE and not speed_max_given:
        u_mm = np.load(out_dir / FN_U, mmap_mode="r")
        v_mm = np.load(out_dir / FN_V, mmap_mode="r")
        i0, i1 = slices[NORMALIZATION_SOURCE]
        train_speed_max = 0.0
        for frame in range(i0, i1):
            speed = np.hypot(np.asarray(u_mm[frame]), np.asarray(v_mm[frame]))
            fluid_speed = speed[~mask2d]
            if fluid_speed.size:
                train_speed_max = max(train_speed_max, float(fluid_speed.max()))
        del u_mm, v_mm
        if train_speed_max <= 0.0:
            raise ValueError(
                f"数据集 {dataset_name} train normalization 无有效正速度："
                f"T={T}, boundary=[{i0}, {i1})"
            )
        speed_max = train_speed_max
    if speed_max is None or not np.isfinite(float(speed_max)) or float(speed_max) <= 0:
        raise ValueError(f"数据集 {dataset_name} speed_max 必须为有限正数，实际 {speed_max!r}")

    if split_mode == WEAK_SUPERVISION_SPLIT_MODE and NORMALIZATION_SOURCE not in slices:
        raise ValueError(
            f"数据集 {dataset_name} 缺少 normalization source {NORMALIZATION_SOURCE!r}"
        )
    ivd_mm = np.load(out_dir / FN_IVD, mmap_mode="r")
    if taus is None:
        # 流体区逐时间片分位数（弱标签口径：排除固体 0 值污染）
        taus = weak_labels.compute_tau(ivd_mm, mask2d, slices, percentile=percentile)
    taus = {k: float(val) for k, val in taus.items()}

    # ---- 标签场（面积过滤 + 固体强制 0；复用 weak_labels 单一口径）
    if labels is not None and isinstance(labels, (str, pathlib.Path)):
        lab = np.asarray(np.load(str(labels)), dtype=np.uint8)[:T]
        np.save(out_dir / FN_LABEL, lab)
    else:
        if labels is not None:
            np.save(out_dir / FN_LABEL, np.asarray(labels, dtype=np.uint8)[:T])
        else:
            lab = weak_labels.build_label_field(ivd_mm, mask2d, taus, slices,
                                                min_area=min_area)
            np.save(out_dir / FN_LABEL, lab)

    # ---- 归一化统计（IVD z-score：train 片流体区，σ=0 防护）
    if ivd_mu is None or ivd_sigma is None:
        if split_mode == WEAK_SUPERVISION_SPLIT_MODE:
            i0, i1 = slices[NORMALIZATION_SOURCE]
        else:
            i0, i1 = slices.get(ivd_stats_slice, slices[next(iter(slices))])
        vals = np.asarray(ivd_mm[i0:i1])[:, ~mask2d]
        if vals.size == 0:
            raise ValueError(f"统计片 {ivd_stats_slice} 无流体格")
        mu = float(vals.mean()) if ivd_mu is None else float(ivd_mu)
        sg = float(vals.std()) if ivd_sigma is None else float(ivd_sigma)
        if sg <= 0:
            sg = 0.0
    else:
        mu, sg = float(ivd_mu), float(ivd_sigma)
    del ivd_mm

    # ---- 可审计的 split/window/provenance 合同
    split_ranges = {k: [int(a), int(b)] for k, (a, b) in slices.items()}
    window_meta = {
        "t_win": int(t_win),
        "window_step": int(window_step),
        "frame_unit": "index",
        "complete_only": True,
    }
    split_meta = {}
    if split_mode == WEAK_SUPERVISION_SPLIT_MODE:
        for split_name, (i0, i1) in slices.items():
            starts = window_starts(
                i0, i1, t_win=t_win, step=window_step,
                dataset_name=dataset_name, split_name=split_name, T=T,
            )
            split_meta[split_name] = {
                "split_name": split_name,
                "frame_start": int(i0),
                "frame_end": int(i1),
                "half_open": True,
                "window_count": int(len(starts)),
            }
        window_meta["counts_by_split"] = {
            name: int(info["window_count"])
            for name, info in split_meta.items()
        }

    feature_schema = {
        **FEATURE_SCHEMA,
        "channels": list(FEATURE_SCHEMA["channels"]),
    }
    label_provenance = {
        "field_source": label_source,
        "sampling_source": sampling_source,
        "loss_source": loss_label_source,
    }
    contract_payload = {
        "dataset_name": dataset_name,
        "generation_version": GENERATION_VERSION,
        "split_mode": split_mode,
        "split_ranges": split_ranges,
        "split_metadata": split_meta,
        "window": window_meta,
        "feature_schema": feature_schema,
        "label_provenance": label_provenance,
        "normalization_source": NORMALIZATION_SOURCE,
        "normalization": {
            "source_split": NORMALIZATION_SOURCE,
            "frozen": True,
            "ivd_mu": mu,
            "ivd_sigma": sg,
            "speed_max": float(speed_max),
        },
    }
    contract_hash = _json_hash(contract_payload)

    # ---- meta.json
    meta = {
        "dataset_name": dataset_name,
        "source_nc": str(nc_path) if nc_path is not None else "in-memory field",
        "shape": [T, Y, X],
        "xdim": [float(x) for x in xdim],
        "ydim": [float(y) for y in ydim],
        "tdim": [float(t) for t in tdim],
        "dt": float(tdim[1] - tdim[0]),
        "slices": {k: [int(a), int(b)] for k, (a, b) in slices.items()},
        "split_mode": split_mode,
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "split_ranges": split_ranges,
        "split_metadata": split_meta,
        "window": window_meta,
        "feature_schema": feature_schema,
        "label_source": label_source,
        "sampling_source": sampling_source,
        "loss_label_source": loss_label_source,
        "label_provenance": label_provenance,
        "generation_version": GENERATION_VERSION,
        "generation_hash": contract_hash,
        "contract_hash": contract_hash,
        "taus": taus,
        "percentile": float(percentile),
        "min_area": int(min_area),
        "speed_max": float(speed_max),
        "ivd_mu": mu,
        "ivd_sigma": sg,
        "ivd_stats_slice": ivd_stats_slice,
        "normalization_source": (NORMALIZATION_SOURCE
                                  if split_mode == WEAK_SUPERVISION_SPLIT_MODE
                                  else ivd_stats_slice),
        "normalization_frozen": True,
        "normalization": {
            "source_split": (NORMALIZATION_SOURCE
                              if split_mode == WEAK_SUPERVISION_SPLIT_MODE
                              else ivd_stats_slice),
            "frozen": True,
            "ivd_mu": mu,
            "ivd_sigma": sg,
            "speed_max": float(speed_max),
        },
        "ivd_source": ivd_source,
        "mask_source": "none" if not mask2d.any() else "provided_or_computed",
        "mask_solid_cells": int(mask2d.sum()),
        "params": {
            "patch_size": [int(p) for p in patch_size],
            "stride": [int(s) for s in stride],
            "t_win": int(t_win),
            "window_step": int(window_step),
        },
    }
    (out_dir / FN_META).write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- 数据集类

def _comb_rng_base(seed, py, px, frame, ds_id=None):
    """组合级确定性随机基：同 (ds_id, seed, patch, 帧) → 同 base（跨会话稳定）。

    供批量提取的 per-k 重播种派生（SeedSequence([base, k])）与池判定共用，
    保证"池判定 / 标签判定 / 提取"三者一致（可复现）。ds_id 标识数据集
    归属（多数据集池；None = 单数据集——字节兼容旧口径 f"{seed}:{py}:{px}:{frame}"，
    避免升级后单数据集续训采样序漂移）。
    """
    if ds_id is None:
        key = f"{int(seed)}:{int(py)}:{int(px)}:{int(frame)}"
    else:
        key = f"{int(seed)}:{int(ds_id)}:{int(py)}:{int(px)}:{int(frame)}"
    return zlib.crc32(key.encode("utf-8"))


class _DatasetStore:
    """单数据集准备产物的存储与提取（一个 prepare_dataset 输出目录）。

    弱标签口径（HANDOFF §1 决策 8 / 票 05）：池判定 = patch_positive_map
    （weak_labels 单一公式）；标签 = 重播种后种子格 label_field；归一化统计
    取本数据集 meta.json（IVD z-score 的 μ/σ 与 speed_max 逐数据集各自——
    票 07 延伸定案：跨数据集输入尺度一致化）；组合级确定性 rng 基 =
    _comb_rng_base(seed, py, px, frame, ds_id)。

    采样/过采样（epoch 序、50% 正样本）不属于 store——由
    WeakLabelPathlineDataset（单数据集）与 MultiDatasetPathlineDataset
    （多数据集池）各自实现；sample_at 为公开入口（任意组合直接取，预览/滑窗用）。
    """

    def __init__(self, data_root, split="train", *,
                 patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE,
                 t_win=DEFAULT_T_WIN, window_step=DEFAULT_WINDOW_STEP,
                 seed=0, groups=DEFAULT_GROUPS, delta_frac=DEFAULT_DELTA_FRAC,
                 L=DEFAULT_L, n_substeps=4, ds_id=None,
                 consumer="train", label_source=None):
        root = pathlib.Path(data_root)
        self._root = root
        self._meta = load_dataset_meta(root)
        self.consumer = _validate_consumer(consumer)
        self.dataset_name = str(self._meta.get(
            "dataset_name", root.parent.name if root.name == "dataset" else root.name))
        actual_label_source = self._meta.get("label_source", "legacy_p85")
        additional_sources = tuple(
            validate_label_source(self._meta[name])
            for name in ("sampling_source", "loss_label_source")
            if name in self._meta
        )
        self.label_source = _validate_label_provenance(
            actual_label_source, consumer=self.consumer,
            declared_source=label_source, additional_sources=additional_sources)
        shape = self._meta["shape"]
        self.T, self.Y, self.X = shape
        self.is_weak_supervision = (
            self._meta.get("split_mode") == WEAK_SUPERVISION_SPLIT_MODE
        )
        if self.is_weak_supervision:
            _validate_weak_contract_metadata(
                self._meta, dataset_name=self.dataset_name, total_frames=self.T)
        self._xdim = np.asarray(self._meta["xdim"], dtype=np.float64)
        self._ydim = np.asarray(self._meta["ydim"], dtype=np.float64)
        self._tdim = np.asarray(self._meta["tdim"], dtype=np.float64)
        slices = {k: (int(a), int(b)) for k, (a, b) in self._meta["slices"].items()}
        if split not in slices:
            raise ValueError(f"split {split!r} 不在时间片 {sorted(slices)} 内")
        if self.is_weak_supervision:
            _validate_weak_consumer_split(
                self.consumer, split, dataset_name=self.dataset_name)
            _validate_source_split(
                self.label_source, split, dataset_name=self.dataset_name)
            expected = weak_supervision_slices(
                self.T, dataset_name=self.dataset_name)
            if slices != expected:
                raise ValueError(
                    f"数据集 {self.dataset_name} 的 weak supervision metadata split "
                    f"不匹配：期望 {expected}，收到 {slices}"
                )
            if self._meta.get("normalization_source") != NORMALIZATION_SOURCE:
                raise ValueError(
                    f"数据集 {self.dataset_name} 的 normalization source 必须为 "
                    f"{NORMALIZATION_SOURCE!r}"
                )
            if self._meta.get("normalization_frozen") is not True:
                raise ValueError(
                    f"数据集 {self.dataset_name} 的 train normalization 必须冻结"
                )
        self.split = split
        self.split_i0, self.split_i1 = slices[split]
        self._i0, self._i1 = self.split_i0, self.split_i1
        self.patch_size = tuple(int(s) for s in patch_size)
        self.stride = tuple(int(s) for s in stride)
        self.t_win = int(t_win)
        self.window_step = int(window_step)
        self.seed = int(seed)
        self.groups = tuple(int(g) for g in groups)
        self.delta_frac = float(delta_frac)
        self.L = int(L)
        self.n_substeps = int(n_substeps)
        self.ds_id = ds_id
        if self.is_weak_supervision:
            window_meta = self._meta["window"]
            if (int(window_meta.get("t_win", -1)) != self.t_win
                    or int(window_meta.get("window_step", -1)) != self.window_step):
                raise ValueError(
                    f"数据集 {self.dataset_name} window metadata t_win/window_step "
                    f"与调用参数不一致：metadata="
                    f"({window_meta.get('t_win')}, {window_meta.get('window_step')})，"
                    f"requested=({self.t_win}, {self.window_step})"
                )
            params = self._meta.get("params", {})
            meta_patch = tuple(int(value) for value in params.get("patch_size", ()))
            meta_stride = tuple(int(value) for value in params.get("stride", ()))
            if meta_patch and meta_patch != self.patch_size:
                raise ValueError(
                    f"数据集 {self.dataset_name} window metadata patch_size "
                    f"与调用参数不一致：metadata={meta_patch}，requested={self.patch_size}"
                )
            if meta_stride and meta_stride != self.stride:
                raise ValueError(
                    f"数据集 {self.dataset_name} window metadata stride "
                    f"与调用参数不一致：metadata={meta_stride}，requested={self.stride}"
                )
        self.speed_max = float(self._meta["speed_max"])
        self.ivd_mu = float(self._meta["ivd_mu"])
        self.ivd_sigma = float(self._meta["ivd_sigma"])
        self.normalization_source = self._meta.get("normalization_source")
        self.normalization_frozen = self._meta.get("normalization_frozen")
        self.t_span = (self.t_win - 1) * (self._tdim[1] - self._tdim[0])

        self._u_mm = np.load(root / FN_U, mmap_mode="r")
        self._v_mm = np.load(root / FN_V, mmap_mode="r")
        self._ivd_mm = np.load(root / FN_IVD, mmap_mode="r")
        self._label_mm = np.load(root / FN_LABEL, mmap_mode="r")
        self._mask2d = np.asarray(np.load(root / FN_MASK), dtype=bool)
        self._patches = patch_locations(self.Y, self.X, self.patch_size, self.stride)
        self.pool_positive, self.pool_negative = self._build_pools()

    # ---------------- 池构建（正样本判据：与 weak_labels 单公式共用）

    def _patch_usable(self, y0, x0):
        """patch 位置可否用于提取（种子重播种可行性，静态精确判定）。

        票 03 边界："patch 全固体 ValueError（上层采样应避开全固体 patch）"；
        实测几何（pipedcylinder2d）存在**非全固体但种子-中心线段全固体**的
        patch（patch 中心在壁面/圆柱内）→ 重播种必然失败。
        判据：对每个落固体的种子，沿 seed→patch 中心线段采样 201 点
        （与 reseed 细扫同密度），存在流体格 → 可用；否则不可用。
        种子全流体 → 无需重播种 → 恒可用。
        """
        seeds = extractor.seeding_grid(
            (y0, x0), self.patch_size, self._xdim, self._ydim,
            self.groups, self.delta_frac)
        geo = extractor.patch_geometry((y0, x0), self.patch_size, self._xdim, self._ydim)
        center = np.array([geo["cx"], geo["cy"]])
        solid = self._solid_seeds(np.asarray(seeds, dtype=np.float64))
        if len(solid) == 0:
            return True
        s = np.linspace(0.0, 1.0, 201)[:, None]
        for k in solid:
            pts = seeds[k] + s * (center - seeds[k][None, :])
            j, i = extractor.nearest_cell(pts[:, 0], pts[:, 1], self._xdim, self._ydim)
            i = np.clip(i, 0, self.X - 1)
            j = np.clip(j, 0, self.Y - 1)
            if not self._mask2d[j, i].all():
                return True
        return False

    def _build_pools(self):
        """正/负样本池 = (patch 位置, 窗口起点帧) 组合；判定用 patch_positive_map。

        不可用 patch 排除（_patch_usable：种子全固体或种子-中心线段全固体）——
        该类 patch 提取必然失败（票 03 ValueError 语义），不入池。
        """
        usable_idx = [i for i, (y0, x0) in enumerate(self._patches)
                      if self._patch_usable(y0, x0)]
        self._usable_patches = [self._patches[i] for i in usable_idx]
        pos, neg = [], []
        if self.is_weak_supervision:
            starts = window_starts(
                self._i0, self._i1, self.t_win, self.window_step,
                dataset_name=self.dataset_name, split_name=self.split, T=self.T)
        else:
            # 阶段 0 的旧 frac/abs 评估允许用部分尾窗做 preview；严格的
            # fail-loud 完整窗口合同只对新 weak_supervision metadata 生效。
            starts = (np.arange(self._i0, self._i1 - self.t_win + 1,
                                self.window_step, dtype=np.intp)
                      if self._i1 - self._i0 >= self.t_win else np.empty(0, dtype=np.intp))
        for frame in starts:
            pm = weak_labels.patch_positive_map(
                self._label_mm[frame], self._xdim, self._ydim,
                self.patch_size, self.stride, self.groups, self.delta_frac)
            pm_flat = pm.reshape(-1)
            for i in usable_idx:
                (y0, x0) = self._patches[i]
                (pos if pm_flat[i] else neg).append((y0, x0, int(frame)))
        return pos, neg

    # ---------------- 确定性种子（重播种后；__getitem__/标签判定的判据）

    def _solid_seeds(self, seeds):
        """种子格在固体掩膜中的 k 索引（向量化检查；extractor.nearest_cell 单一公式）。"""
        j, i = extractor.nearest_cell(seeds[:, 0], seeds[:, 1], self._xdim, self._ydim)
        i = np.clip(i, 0, self.X - 1)
        j = np.clip(j, 0, self.Y - 1)
        return np.nonzero(self._mask2d[j, i])[0]

    def seeds_for(self, py, px, frame):
        """(py, px, frame) → 重播种后 256 个种子物理坐标 (K,2)（确定性）。

        与 __getitem__ 使用同一组合级 rng 派生（_comb_rng_base）与同一
        _extract 路径（含短迹线重试）→ 池判定/标签判定/提取三者严格一致。
        仅用于测试与诊断（提取完整样本以取种子，成本与 __getitem__ 同）。
        """
        _raw, seeds, _geo = self._extract(py, px, frame)
        return seeds

    # ---------------- 样本生成（on-the-fly 提取 + 归一化 + 标签）

    def _extract(self, py, px, frame):
        """组合 → (raw (L,K,7), seeds (K,2), geo)：批量提取（组合级确定性 rng）。

        时变语义：u/v/ivd 窗口切片（T_win 帧）必须配**窗口 tdim**（时间索引
        相对窗口起点；传全场 tdim 会把时间映射 clamp 到窗口末帧——Spec 审查
        实测复现的时变冻结 bug，此处为单一修复点）。
        """
        if self.is_weak_supervision:
            frame = validate_window_start(
                frame, split_start=self.split_i0, split_end=self.split_i1,
                t_win=self.t_win, dataset_name=self.dataset_name,
                split_name=self.split, T=self.T)
        else:
            frame = int(frame)
        geo = extractor.patch_geometry((py, px), self.patch_size, self._xdim, self._ydim)
        base = _comb_rng_base(self.seed, py, px, frame, ds_id=self.ds_id)
        u_win = np.asarray(self._u_mm[frame:frame + self.t_win], dtype=np.float32)
        v_win = np.asarray(self._v_mm[frame:frame + self.t_win], dtype=np.float32)
        ivd_win = np.asarray(self._ivd_mm[frame:frame + self.t_win], dtype=np.float32)
        tdim_win = self._tdim[frame:frame + self.t_win]
        raw, seeds = extractor.extract_pathlines_batched(
            u_win, v_win, self._mask2d, ivd_win, self._xdim, self._ydim, tdim_win,
            patch_yx=(py, px), patch_size=self.patch_size,
            t0=float(self._tdim[frame]), L=self.L,
            groups=self.groups, delta_frac=self.delta_frac,
            t_win_frames=self.t_win, n_substeps=self.n_substeps,
            rng=base, return_seeds=True)
        return raw, seeds, geo

    def _labels_for(self, seeds, frame):
        """重播种后种子最近格（单一公式 nearest_cell）→ label_field 值 (K,)。"""
        j, i = extractor.nearest_cell(seeds[:, 0], seeds[:, 1], self._xdim, self._ydim)
        i = np.clip(i, 0, self.X - 1)
        j = np.clip(j, 0, self.Y - 1)
        return self._label_mm[frame][j, i].astype(np.float32)

    def sample_at(self, py, px, frame, t_scale=DEFAULT_T_SCALE):
        """指定 (patch 位置 y0,x0, 窗口起点帧) 的完整样本——预览/诊断公开入口。

        与 __getitem__ 同路径（_extract + normalize_pathlines + 标签判定），
        返回 ((dummy_field, pathlines), labels, seeds)；不依赖 set_epoch/采样序
        （任意 (patch, 帧) 组合可直接取——票 07 预览、票 08 滑窗评估的基础）。
        """
        raw, seeds, geo = self._extract(py, px, frame)
        pathlines = normalize_pathlines(raw, seeds, geo, float(self._tdim[frame]),
                                        self.t_span, t_scale, self.ivd_mu,
                                        self.ivd_sigma, self.speed_max)
        labels = self._labels_for(seeds, frame)
        return (np.zeros((1, 1, 1, 1), dtype=np.float32), pathlines), labels, seeds

    def window_metadata(self, frame):
        """返回一个 split-contained window 的可审计 provenance metadata。"""
        if self.is_weak_supervision:
            frame = validate_window_start(
                frame, split_start=self.split_i0, split_end=self.split_i1,
                t_win=self.t_win, dataset_name=self.dataset_name,
                split_name=self.split, T=self.T)
        else:
            frame = int(frame)
        return {
            "dataset_name": self.dataset_name,
            "split_name": self.split,
            "frame_start": int(frame),
            "frame_end": int(frame + self.t_win),
            "split_start": int(self.split_i0),
            "split_end": int(self.split_i1),
            "t_win": int(self.t_win),
            "window_step": int(self.window_step),
            "generation_version": self._meta.get("generation_version"),
            "generation_hash": self._meta.get("generation_hash"),
            "contract_hash": self._meta.get("contract_hash"),
            "feature_schema": self._meta.get("feature_schema"),
            "label_source": self.label_source,
            "label_provenance": self._meta.get("label_provenance"),
            "normalization_source": self._meta.get("normalization_source"),
            "normalization_frozen": self._meta.get("normalization_frozen"),
        }


# --------------------------------------------------------------------------- 单数据集包装（票 05 公开面）

def _mixed_order(pool_positive, pool_negative, samples_per_epoch, positive_fraction,
                 seed, epoch):
    """50% 正池（放回）+ 50% 负池（放回）后打乱的采样序——单一公式（确定性 (seed, epoch)）。

    单数据集（WeakLabelPathlineDataset，池编码 (y0,x0,frame)）与多数据集
    （MultiDatasetPathlineDataset，编码 (si,y0,x0,frame)）共用：行数/行宽不同，
    rng 语义一致（同 seed+epoch → 同序）。
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    n_pos = int(round(int(samples_per_epoch) * float(positive_fraction)))
    n_neg = int(samples_per_epoch) - n_pos
    if not pool_positive:
        raise ValueError(
            "正样本池为空：无正 patch（检查 τ/标签场/时间片；多数据集时为联合池）")
    if n_neg > 0 and not pool_negative:
        raise ValueError("负样本池为空（样本池不完整）")
    pick_p = np.asarray(pool_positive, dtype=np.int64)
    pidx = pick_p[rng.integers(0, len(pool_positive), size=n_pos)]
    if n_neg:
        pick_n = np.asarray(pool_negative, dtype=np.int64)
        nidx = pick_n[rng.integers(0, len(pool_negative), size=n_neg)]
        order = np.concatenate([pidx, nidx])
    else:
        order = pidx
    rng.shuffle(order)
    return [tuple(int(x) for x in c) for c in order]


class WeakLabelPathlineDataset:
    """弱标签迹线数据集（on-the-fly；h5py+memmap）——单数据集包装（票 05 口径）。

    构造：data_root 为 prepare_dataset 的输出目录（meta.json + memmap）。
    set_epoch(epoch) 重建 50% 正样本过采样的采样序（每 epoch 调用一次；
    首次使用前必须调用；同 (seed, epoch) → 同序，字节兼容票 05 实现）。
    __getitem__(idx) 返回 ((dummy_field, pathlines), labels)。

    样本池：正 = patch 内存在 ≥1 条涡迹线（weak_labels.patch_positive_map 判据，
    与票 04 正样本统计单公式共用）；负 = 其余。标签 = 重播种后种子格处
    label_field 值（与输入迹线的实际出发位置自洽；正池零误差、负池掺正 ≤2%
    为票 04 已披露近似，不影响过采样设计）。存储/提取委托 _DatasetStore
    （ds_id=None → 组合级 rng 基与旧实现逐字节一致；跨数据集预览传
    dataset_idx 作 ds_id——与多数据集池同构）。
    """

    def __init__(self, data_root, split="train", *,
                 patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE,
                 t_win=DEFAULT_T_WIN, window_step=DEFAULT_WINDOW_STEP,
                 samples_per_epoch=DEFAULT_SAMPLES_PER_EPOCH,
                 positive_fraction=DEFAULT_POSITIVE_FRACTION,
                 t_scale=DEFAULT_T_SCALE, seed=0,
                 groups=DEFAULT_GROUPS, delta_frac=DEFAULT_DELTA_FRAC,
                 L=DEFAULT_L, n_substeps=4, ds_id=None,
                 consumer="train", label_source=None):
        self._store = _DatasetStore(data_root, split, patch_size=patch_size,
                                    stride=stride, t_win=t_win,
                                    window_step=window_step, seed=seed,
                                    groups=groups, delta_frac=delta_frac,
                                    L=L, n_substeps=n_substeps, ds_id=ds_id,
                                    consumer=consumer, label_source=label_source)
        # 公开别名（票 05 池名/测试/预览引用不变；委托存储）
        self.pool_positive = self._store.pool_positive
        self.pool_negative = self._store.pool_negative
        self._patch_usable = self._store._patch_usable
        self.seeds_for = self._store.seeds_for
        self.T, self.Y, self.X = self._store.T, self._store.Y, self._store.X
        self.patch_size = self._store.patch_size
        self.stride = self._store.stride
        self.split = split
        self.consumer = self._store.consumer
        self.label_source = self._store.label_source
        self._root = self._store._root
        self.t_win = self._store.t_win
        self.window_step = self._store.window_step
        self.groups = self._store.groups
        self.delta_frac = self._store.delta_frac
        self.L = self._store.L
        self.n_substeps = self._store.n_substeps
        self.samples_per_epoch = int(samples_per_epoch)
        self.positive_fraction = float(positive_fraction)
        self.t_scale = float(t_scale)
        self.seed = int(seed)
        self.speed_max = self._store.speed_max
        self.ivd_mu = self._store.ivd_mu
        self.ivd_sigma = self._store.ivd_sigma
        self.t_span = self._store.t_span
        self._xdim = self._store._xdim
        self._ydim = self._store._ydim
        self._tdim = self._store._tdim
        self._label_mm = self._store._label_mm
        self._ivd_mm = self._store._ivd_mm
        self._u_mm = self._store._u_mm
        self._v_mm = self._store._v_mm
        self._mask2d = self._store._mask2d
        self._order = None
        self._epoch = None

    # ---------------- epoch 采样（50% 正样本过采样；与票 05 同序口径）

    def set_epoch(self, epoch):
        """重建采样序：50% 正池（放回）+ 50% 负池（放回）后打乱。

        每 epoch 调用一次；同 (seed, epoch) → 同序（确定性可复现，_mixed_order
        单一公式与多数据集共用）。
        """
        self._epoch = int(epoch)
        self._order = _mixed_order(self.pool_positive, self.pool_negative,
                                   self.samples_per_epoch, self.positive_fraction,
                                   self.seed, self._epoch)
        return self._order

    def set_epoch_natural(self, epoch=0):
        """按池自然比例（正/负池大小比）重建采样序——自然分布评估口径。

        训练监控用 50% 平衡（set_epoch）；自然分布（真实正负占比）用于训练
        收尾的 val F1 记录（票 07 验收 4；正式弱定量表属票 08）。
        """
        n_pos = len(self.pool_positive)
        n_neg = len(self.pool_negative)
        total = n_pos + n_neg
        self.positive_fraction = n_pos / total if total > 0 else 0.5
        return self.set_epoch(int(epoch))

    def sample_at(self, py, px, frame):
        """指定 (patch 位置 y0,x0, 窗口起点帧) 的完整样本——预览/诊断公开入口。

        委托 store（与 __getitem__ 同路径）；返回 ((dummy_field, pathlines),
        labels, seeds)。
        """
        return self._store.sample_at(py, px, frame, self.t_scale)

    def window_metadata(self, frame):
        """返回指定窗口的 split/label/normalization provenance。"""
        return self._store.window_metadata(frame)

    @property
    def store(self):
        """底层 _DatasetStore（预览/滑窗按数据集取样本的委托接缝）。"""
        return self._store

    # ---------------- __getitem__

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if self._order is None:
            raise RuntimeError("先调用 set_epoch(epoch) 再采样（每 epoch 一次）")
        py, px, frame = self._order[idx]
        return self._store.sample_at(py, px, frame, self.t_scale)[:2]


# --------------------------------------------------------------------------- 多数据集联合池（票 07 延伸）

class MultiDatasetPathlineDataset:
    """多数据集联合采样池（旧 frac 与新 weak split 均显式受 metadata 守护）。

    池 = 各数据集 store（_DatasetStore）的组合并集，组合编码 =
    (store_idx, y0, x0, frame)；set_epoch(epoch) 重建 50% 正样本过采样序
    （同 (seed, epoch) 确定性）；set_epoch_natural 按联合池自然比例；
    sample_at(si, y0, x0, frame) 公开入口（预览/票 08 滑窗按数据集取样本）。
    τ 与归一化逐数据集（各 store 自身 meta 统计：ivd z-score、u/v÷speed_max、
    px/py 为 patch 内归一化——跨数据集输入尺度一致）；组合级 rng 基含
    ds_id 派生（同语义、与单数据集不同构）。
    """

    def __init__(self, roots, split="train", *,
                 patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE,
                 t_win=DEFAULT_T_WIN, window_step=DEFAULT_WINDOW_STEP,
                 samples_per_epoch=DEFAULT_SAMPLES_PER_EPOCH,
                 positive_fraction=DEFAULT_POSITIVE_FRACTION,
                 t_scale=DEFAULT_T_SCALE, seed=0,
                 groups=DEFAULT_GROUPS, delta_frac=DEFAULT_DELTA_FRAC,
                 L=DEFAULT_L, n_substeps=4,
                 consumer="train", label_source=None):
        roots = [pathlib.Path(r) for r in roots]
        if not roots:
            raise ValueError("roots 为空：至少一个数据集目录")
        self._stores = [_DatasetStore(
            r, split, patch_size=patch_size, stride=stride, t_win=t_win,
            window_step=window_step, seed=seed, groups=groups,
            delta_frac=delta_frac, L=L, n_substeps=n_substeps, ds_id=i,
            consumer=consumer, label_source=label_source)
            for i, r in enumerate(roots)]
        sources = {store.label_source for store in self._stores}
        if len(sources) > 1:
            raise ValueError(
                f"多数据集 {split!r} 的 label source 必须一致，实际为 "
                f"{sorted(sources)}"
            )
        self.pool_positive = [(i, *combo) for i, s in enumerate(self._stores)
                              for combo in s.pool_positive]
        self.pool_negative = [(i, *combo) for i, s in enumerate(self._stores)
                              for combo in s.pool_negative]
        self.samples_per_epoch = int(samples_per_epoch)
        self.positive_fraction = float(positive_fraction)
        self.t_scale = float(t_scale)
        self.seed = int(seed)
        self.consumer = _validate_consumer(consumer)
        self.label_source = None
        if self._stores:
            self.label_source = self._stores[0].label_source
        self._order = None
        self._epoch = None

    @property
    def stores(self):
        """各数据集 store（预览/滑窗按数据集取样本与 patch 位置）。"""
        return self._stores

    def set_epoch(self, epoch):
        """重建采样序：联合池 50% 正样本过采样（同 (seed, epoch) → 同序）。"""
        self._epoch = int(epoch)
        self._order = _mixed_order(self.pool_positive, self.pool_negative,
                                   self.samples_per_epoch, self.positive_fraction,
                                   self.seed, self._epoch)
        return self._order

    def set_epoch_natural(self, epoch=0):
        """按联合池自然比例（正/负池大小比）重建采样序（留出评估口径）。"""
        n_pos = len(self.pool_positive)
        n_neg = len(self.pool_negative)
        total = n_pos + n_neg
        self.positive_fraction = n_pos / total if total > 0 else 0.5
        return self.set_epoch(int(epoch))

    def sample_at(self, si, y0, x0, frame):
        """指定 (数据集索引, patch 位置, 窗口起点帧) 的完整样本——公开入口。

        返回 ((dummy_field, pathlines), labels, seeds)；归一化取该数据集
        store 自己的统计。
        """
        return self._stores[int(si)].sample_at(y0, x0, frame, self.t_scale)

    def window_metadata(self, si, frame):
        """返回一个数据集窗口的 split/label/normalization provenance。"""
        return self._stores[int(si)].window_metadata(frame)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if self._order is None:
            raise RuntimeError("先调用 set_epoch(epoch) 再采样（每 epoch 一次）")
        si, py, px, frame = self._order[idx]
        return self._stores[si].sample_at(py, px, frame, self.t_scale)[:2]


# --------------------------------------------------------------------------- CLI（prepare_dataset 入口）

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="数据集准备：u/v/ivd/label/mask memmap + meta.json "
                    "（弱标签迹线数据集；时间划分/τ 与 weak_labels 口径一致）")
    ap.add_argument("nc_path", nargs="?", default=None,
                    help="nc 数据集路径（h5py 直读，支持中文路径；缺省用内存数组参数）")
    ap.add_argument("--out-dir", default="outputs/dataset",
                    help="数据集输出目录（meta.json + memmap）")
    ap.add_argument("--mask", default=None,
                    help="固体掩膜 mask.npy 路径（缺省从速度场计算或空掩膜）")
    ap.add_argument("--ivd", default=None, help="复用票 04 产物 ivd.npy 路径")
    ap.add_argument("--labels", default=None, help="复用票 04 产物 label_field.npy 路径")
    ap.add_argument("--percentile", type=float, default=weak_labels.DEFAULT_PERCENTILE,
                    help="τ 分位数（默认 85——票 07 延伸；HANDOFF §6）")
    ap.add_argument("--split-mode", choices=("abs", "frac", WEAK_SUPERVISION_SPLIT_MODE),
                    default="abs",
                    help="时间片划分口径：abs=绝对秒数 DEFAULT_SLICES（单数据集默认）；"
                         "frac=旧多数据集 60/40；weak_supervision=每数据集 "
                         "0/50/60/100 新契约")
    ap.add_argument("--dataset-name", default=None,
                    help="新弱监督 metadata 的数据集名（缺省从输入/输出路径推断）")
    ap.add_argument("--label-source", default=None,
                    help="label source；weak_supervision 必须显式指定")
    ap.add_argument("--sampling-source", default=None,
                    help="独立记录的 sampling source（例如 legacy_p85）")
    ap.add_argument("--loss-label-source", default=None,
                    help="formal loss label source；不能将 p85 混入 W1 formal loss")
    ap.add_argument("--train-frac", type=float, default=0.6,
                    help="frac 口径的训练帧比例（默认 0.6）")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="frac 口径的 val 帧比例（默认 0=无 val 片）")
    ap.add_argument("--t-win", type=int, default=DEFAULT_T_WIN,
                    help="pathline window 帧数（weak split 每段都必须容纳）")
    ap.add_argument("--window-step", type=int, default=DEFAULT_WINDOW_STEP,
                    help="pathline window 起点步长")
    args = ap.parse_args(argv)

    meta = prepare_dataset(args.nc_path, args.out_dir, mask=args.mask,
                           ivd=args.ivd, labels=args.labels,
                           percentile=args.percentile,
                           split_mode=args.split_mode,
                           dataset_name=args.dataset_name,
                           label_source=args.label_source,
                           sampling_source=args.sampling_source,
                           loss_label_source=args.loss_label_source,
                           train_frac=args.train_frac, val_frac=args.val_frac,
                           t_win=args.t_win, window_step=args.window_step)
    print(f"数据集已准备: {args.out_dir}")
    print(f"  shape={meta['shape']} slices={meta['slices']}")
    print(f"  taus={meta['taus']}")
    print(f"  speed_max={meta['speed_max']:.6g} ivd_mu={meta['ivd_mu']:.6g} "
          f"ivd_sigma={meta['ivd_sigma']:.6g}")
    return 0


if __name__ == "__main__":
    main()
