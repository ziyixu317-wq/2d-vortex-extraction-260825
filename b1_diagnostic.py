"""B1 诊断性 local-IVD 输入消融的训练 seam。

B1 只在 vendor 外部把 raw 7-channel pathline 适配为
``[px, py, t, distance, u, v]`` 六通道。数据 split、legacy_p85 监督、采样和
归一化由新弱监督 dataset contract 校验；Haller anchor/GT 不属于本模块的输入。
"""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Any, Mapping

import numpy as np
import torch

import dataset as ds
import weak_labels
import weak_supervision_contract as contract


B1_MODE = contract.MODE_B1
B1_LABEL_SOURCE = contract.LABEL_SOURCE_LEGACY_P85
B1_ARTIFACT_ROLE = "diagnostic"
B1_ARTIFACT_NAMESPACE = "b1_diagnostic"
B1_DATASET_SCOPE_VALID = "valid_six_datasets"
B1_DATASET_SCOPE_SYNTHETIC = "synthetic_fixture"

_B1_DATA_FIELDS = (
    "root",
    "dataset_scope",
    "split_mode",
    "split",
    "val_split",
    "label_source",
    "sampling_source",
    "loss_label_source",
    "seed",
    "batch_size",
    "num_workers",
    "samples_per_epoch",
    "positive_fraction",
    "patch_size",
    "stride",
    "t_win",
    "window_step",
    "t_scale",
    "groups",
    "delta_frac",
    "L",
    "n_substeps",
)
_B1_TRAIN_FIELDS = (
    "mode",
    "epochs",
    "lr",
    "weight_decay",
    "warmup_epochs",
    "second_lr",
    "grad_clip",
    "save_freq",
    "seed",
    "device",
    "amp",
    "data_parallel",
    "ckpt_dir",
    "run_name",
    "warm_start_aux",
)


def _missing_fields(mapping: Mapping[str, Any], fields: tuple[str, ...], *, name: str):
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(
            f"B1 {name} 必须显式提供字段 {missing}；禁止回退旧默认配置"
        )


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"B1 {name} 必须是正整数，实际 {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"B1 {name} 必须是正整数，实际 {value!r}") from exc
    if converted != value or converted <= 0:
        raise ValueError(f"B1 {name} 必须是正整数，实际 {value!r}")
    return converted


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"B1 {name} 必须是非负整数，实际 {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"B1 {name} 必须是非负整数，实际 {value!r}") from exc
    if converted != value or converted < 0:
        raise ValueError(f"B1 {name} 必须是非负整数，实际 {value!r}")
    return converted


def _root_list(data_config: Mapping[str, Any]) -> list[str]:
    roots = data_config["root"]
    if isinstance(roots, (str, pathlib.Path)):
        roots = [roots]
    if not isinstance(roots, (list, tuple)) or not roots:
        raise ValueError("B1 data.root 必须是非空路径或路径列表")
    normalized = [str(root) for root in roots]
    if any(not root.strip() for root in normalized):
        raise ValueError("B1 data.root 不能包含空路径")
    return normalized


def validate_b1_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """校验 B1 的显式 mode、六通道模型和新 split/source 配置。

    这里只检查配置形状，不读取数据目录；数据 metadata 的 split/window/归一化
    校验由 :func:`validate_b1_dataset_contract` 在训练入口中执行。
    """
    if not isinstance(config, Mapping):
        raise ValueError("B1 config 必须是 object")
    if not isinstance(config.get("data"), Mapping):
        raise ValueError("B1 config 缺少 data object")
    if not isinstance(config.get("model"), Mapping):
        raise ValueError("B1 config 缺少 model object")
    if not isinstance(config.get("train"), Mapping):
        raise ValueError("B1 config 缺少 train object")
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    _missing_fields(data_config, _B1_DATA_FIELDS, name="data")
    _missing_fields(train_config, _B1_TRAIN_FIELDS, name="train")

    mode = contract.canonical_mode(train_config["mode"])
    if mode != B1_MODE:
        raise ValueError(
            f"当前票只实现 B1 diagnostic mode，实际 mode={mode!r}"
        )
    if data_config["split_mode"] != ds.WEAK_SUPERVISION_SPLIT_MODE:
        raise ValueError(
            "B1 必须使用 split_mode='weak_supervision' 的新三段 split，"
            f"实际为 {data_config['split_mode']!r}"
        )
    if data_config["split"] != "train":
        raise ValueError(
            f"B1 training data.split 必须是 'train'，实际为 {data_config['split']!r}"
        )
    if data_config["val_split"] != "none":
        raise ValueError(
            "B1 诊断训练只消费 train split；data.val_split 必须显式为 'none'，"
            f"实际为 {data_config['val_split']!r}"
        )
    for field in ("label_source", "sampling_source", "loss_label_source"):
        if data_config[field] != B1_LABEL_SOURCE:
            raise ValueError(
                f"B1 {field} 必须显式为 {B1_LABEL_SOURCE!r}，"
                f"实际为 {data_config[field]!r}"
            )
    _root_list(data_config)
    if data_config["dataset_scope"] not in (
            B1_DATASET_SCOPE_VALID, B1_DATASET_SCOPE_SYNTHETIC):
        raise ValueError(
            "B1 data.dataset_scope 必须是 'valid_six_datasets' 或 "
            f"'synthetic_fixture'，实际为 {data_config['dataset_scope']!r}"
        )

    for field in ("batch_size", "samples_per_epoch", "t_win", "window_step",
                  "L", "n_substeps"):
        _positive_int(data_config[field], name=f"data.{field}")
    _nonnegative_int(data_config["num_workers"], name="data.num_workers")
    _nonnegative_int(data_config["seed"], name="data.seed")
    for field in ("patch_size", "stride", "groups"):
        values = data_config[field]
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"B1 data.{field} 必须是二元正整数列表")
        for index, value in enumerate(values):
            _positive_int(value, name=f"data.{field}[{index}]")
    if not (0.0 <= float(data_config["positive_fraction"]) <= 1.0):
        raise ValueError("B1 data.positive_fraction 必须在 [0, 1] 内")
    if float(data_config["t_scale"]) <= 0 or not np.isfinite(float(data_config["t_scale"])):
        raise ValueError("B1 data.t_scale 必须是正的有限数")

    encoder_config = model_config.get("encoder_args")
    if not isinstance(encoder_config, Mapping):
        raise ValueError("B1 model.encoder_args 必须是 object")
    if "in_channels" not in encoder_config:
        raise ValueError("B1 model.encoder_args 必须显式提供 in_channels=6")
    configured_channels = encoder_config["in_channels"]
    if (isinstance(configured_channels, (bool, np.bool_))
            or not isinstance(configured_channels, (int, np.integer))
            or int(configured_channels) != contract.FEATURE_SCHEMA_6.channel_count):
        raise ValueError(
            "B1 model.encoder_args.in_channels 必须是 6；"
            f"实际为 {configured_channels!r}"
        )
    if not isinstance(model_config.get("criterion_args"), Mapping):
        raise ValueError("B1 model.criterion_args 必须显式提供")

    if train_config["warm_start_aux"] is not False:
        raise ValueError(
            "B1 主诊断运行必须 warm_start_aux=false；不得 warm-start 旧 B0 checkpoint"
        )
    _positive_int(train_config["epochs"], name="train.epochs")
    _nonnegative_int(train_config["warmup_epochs"], name="train.warmup_epochs")
    _nonnegative_int(train_config["seed"], name="train.seed")
    _nonnegative_int(train_config["save_freq"], name="train.save_freq")
    for field in ("amp", "data_parallel"):
        if not isinstance(train_config[field], bool):
            raise ValueError(f"B1 train.{field} 必须是 bool")
    if int(data_config["seed"]) != int(train_config["seed"]):
        raise ValueError(
            "B1 data.seed 必须与 train.seed 相同；禁止训练随机状态与采样序漂移"
        )
    if train_config["amp"] or train_config["data_parallel"]:
        raise ValueError(
            "B1 当前诊断 runner 只支持单设备 FP32；"
            "train.amp/data_parallel 必须显式为 false"
        )
    for field in ("ckpt_dir", "run_name"):
        if not isinstance(train_config[field], (str, pathlib.Path)) or not str(
                train_config[field]).strip():
            raise ValueError(f"B1 train.{field} 必须是非空字符串")

    # 配置中的注册 source 若出现任何 Haller train/calibration/test source，
    # 直接阻断；B1 只使用 legacy_p85，避免通过旁路把 Haller GT 带入训练。
    haller_sources = {
        contract.LABEL_SOURCE_HALLER_TRAIN,
        contract.LABEL_SOURCE_HALLER_CALIBRATION,
        contract.LABEL_SOURCE_HALLER_TEST,
    }

    def _registered_sources(value):
        if isinstance(value, str):
            if value in contract.VALID_LABEL_SOURCES:
                yield value
        elif isinstance(value, Mapping):
            for key, child in value.items():
                yield from _registered_sources(key)
                yield from _registered_sources(child)
        elif isinstance(value, (list, tuple, set)):
            for child in value:
                yield from _registered_sources(child)

    leaked = next((source for source in _registered_sources(config)
                   if source in haller_sources), None)
    if leaked is not None:
        raise ValueError(
            f"B1 config 禁止出现 Haller source {leaked!r}；B1 不读取 Haller artifact"
        )
    return {"mode": B1_MODE, "feature_schema": contract.FEATURE_SCHEMA_6,
            "adapter_input_schema": contract.FEATURE_SCHEMA_7}


def _dataset_stores(dataset):
    """读取 dataset 的公开 store seam，兼容单数据集和联合数据集。"""
    stores = getattr(dataset, "stores", None)
    if stores is not None:
        return list(stores)
    store = getattr(dataset, "store", None)
    if store is None:
        raise ValueError("B1 dataset 必须暴露 store/stores metadata seam")
    return [store]


def _audit_legacy_p85_arrays(root, meta, name, ranges):
    """重算 p85 τ 并分块核对 label_field，防止声明式 source 伪装。"""
    try:
        ivd = np.load(pathlib.Path(root) / ds.FN_IVD, mmap_mode="r")
        labels = np.load(pathlib.Path(root) / ds.FN_LABEL, mmap_mode="r")
        mask = np.asarray(np.load(pathlib.Path(root) / ds.FN_MASK), dtype=bool)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(
            f"B1 dataset {name} 缺少可审计的 ivd/label/mask artifact"
        ) from exc
    expected_shape = tuple(int(value) for value in meta.get("shape", ()))
    if len(expected_shape) != 3 or ivd.shape != expected_shape or labels.shape != expected_shape:
        raise ValueError(
            f"B1 dataset {name} 的 ivd/label shape 与 metadata 不一致："
            f"ivd={ivd.shape}, label={labels.shape}, expected={expected_shape}"
        )
    if mask.shape != expected_shape[1:]:
        raise ValueError(f"B1 dataset {name} 的 IVD/mask shape 或数值非法")

    actual_taus = meta.get("taus")
    if not isinstance(actual_taus, Mapping) or set(actual_taus) != set(ranges):
        raise ValueError(f"B1 dataset {name} 的 legacy_p85 taus 不完整")
    audited_taus = {}
    for split_name, (i0, i1) in ranges.items():
        values = np.asarray(ivd[i0:i1][:, ~mask], dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(
                f"B1 dataset {name} split={split_name} 的 IVD 流体值非法"
            )
        expected_tau = float(np.percentile(values, 85.0))
        try:
            declared_tau = float(actual_taus[split_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"B1 dataset {name} split={split_name} 的 legacy_p85 tau 非法"
            ) from exc
        if not np.isfinite(declared_tau) or not np.isclose(
                declared_tau, expected_tau, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"B1 dataset {name} split={split_name} 的 tau 不是实际 p85："
                f"declared={declared_tau!r}, expected={expected_tau!r}"
            )
        audited_taus[split_name] = expected_tau
        del values

    try:
        min_area = int(meta["min_area"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"B1 dataset {name} 缺少合法 min_area") from exc
    if min_area != weak_labels.DEFAULT_MIN_AREA:
        raise ValueError(
            f"B1 dataset {name} 的 min_area 必须为 5x5={weak_labels.DEFAULT_MIN_AREA}，"
            f"实际为 {min_area!r}"
        )
    for split_name, (i0, i1) in ranges.items():
        for chunk_start in range(int(i0), int(i1), 128):
            chunk_end = min(chunk_start + 128, int(i1))
            ivd_chunk = np.asarray(ivd[chunk_start:chunk_end])
            label_chunk = np.asarray(labels[chunk_start:chunk_end])
            if (not np.isfinite(ivd_chunk).all()
                    or not np.all((label_chunk == 0) | (label_chunk == 1))):
                raise ValueError(
                    f"B1 dataset {name} split={split_name} 的 IVD/label 数值非法"
                )
            chunk_slices = {split_name: (0, chunk_end - chunk_start)}
            expected_labels = weak_labels.build_label_field(
                ivd_chunk,
                mask,
                {split_name: audited_taus[split_name]},
                chunk_slices,
                min_area=min_area,
            )
            if not np.array_equal(label_chunk, expected_labels):
                raise ValueError(
                    f"B1 dataset {name} split={split_name} 的 label_field "
                    "与 canonical legacy_p85 重建结果不一致"
                )
    return {"percentile": 85.0, "min_area": weak_labels.DEFAULT_MIN_AREA,
            "taus": audited_taus}


def validate_b1_dataset_contract(
    dataset,
    data_config: Mapping[str, Any],
) -> dict[str, Any]:
    """验证训练 dataset 确实是新 split、legacy_p85 和 train-only normalization。"""
    roots = _root_list(data_config)
    stores = _dataset_stores(dataset)
    if len(stores) != len(roots):
        raise ValueError(
            f"B1 dataset root 数量与 store 数量不一致：roots={len(roots)} stores={len(stores)}"
        )
    dataset_names = []
    split_ranges = {}
    generation_hashes = {}
    label_audits = {}
    haller_sources_read = set()
    dataset_scope = data_config["dataset_scope"]
    for root, store in zip(roots, stores):
        meta = ds.load_dataset_meta(root)
        name = str(meta.get("dataset_name", pathlib.Path(root).name))
        root_path = pathlib.Path(root)
        if name not in {root_path.name, root_path.parent.name}:
            raise ValueError(
                f"B1 dataset root 与 metadata dataset_name 不一致："
                f"root={root!r}, dataset_name={name!r}"
            )
        if meta.get("split_mode") != ds.WEAK_SUPERVISION_SPLIT_MODE:
            raise ValueError(
                f"B1 dataset {name} 必须使用 weak_supervision metadata，"
                f"实际 {meta.get('split_mode')!r}"
            )
        if store.split != "train" or store.consumer != "train":
            raise ValueError(
                f"B1 dataset {name} 只能以 train consumer 读取，"
                f"实际 split={store.split!r}, consumer={store.consumer!r}"
            )
        if store.label_source != B1_LABEL_SOURCE:
            raise ValueError(
                f"B1 dataset {name} label source 必须为 {B1_LABEL_SOURCE!r}，"
                f"实际 {store.label_source!r}"
            )
        provenance = meta.get("label_provenance", {})
        haller_sources_read.update(
            source for source in (
                meta.get("label_source"), meta.get("sampling_source"),
                meta.get("loss_label_source"),
                provenance.get("field_source"),
                provenance.get("sampling_source"),
                provenance.get("loss_source"),
            ) if source in {
                contract.LABEL_SOURCE_HALLER_TRAIN,
                contract.LABEL_SOURCE_HALLER_CALIBRATION,
                contract.LABEL_SOURCE_HALLER_TEST,
            }
        )
        if (meta.get("label_source") != B1_LABEL_SOURCE
                or meta.get("sampling_source") != B1_LABEL_SOURCE
                or meta.get("loss_label_source") != B1_LABEL_SOURCE
                or provenance.get("field_source") != B1_LABEL_SOURCE
                or provenance.get("sampling_source") != B1_LABEL_SOURCE
                or provenance.get("loss_source") != B1_LABEL_SOURCE):
            raise ValueError(
                f"B1 dataset {name} 的 label/sampling/loss provenance 必须全部是 legacy_p85"
            )
        try:
            percentile = float(meta["percentile"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"B1 dataset {name} 必须记录 legacy_p85 的 percentile=85"
            ) from exc
        if not np.isfinite(percentile) or percentile != 85.0:
            raise ValueError(
                f"B1 dataset {name} 的 label source=legacy_p85 但 percentile="
                f"{percentile!r}，需要 85"
            )
        if (meta.get("normalization_source") != ds.NORMALIZATION_SOURCE
                or meta.get("normalization_frozen") is not True
                or meta.get("normalization", {}).get("source_split") != "train"
                or meta.get("normalization", {}).get("frozen") is not True):
            raise ValueError(
                f"B1 dataset {name} 必须消费冻结的 train-only normalization"
            )
        window = meta.get("window", {})
        expected_window = {
            "t_win": _positive_int(data_config["t_win"], name="data.t_win"),
            "window_step": _positive_int(data_config["window_step"],
                                          name="data.window_step"),
        }
        if (window.get("t_win") != expected_window["t_win"]
                or window.get("window_step") != expected_window["window_step"]
                or window.get("frame_unit") != "index"
                or window.get("complete_only") is not True):
            raise ValueError(
                f"B1 dataset {name} 的 window contract 与 config 不一致："
                f"metadata={window!r}, expected={expected_window!r}"
            )
        ranges = meta.get("split_ranges")
        if not isinstance(ranges, Mapping) or set(ranges) != set(ds.WEAK_SUPERVISION_SPLITS):
            raise ValueError(f"B1 dataset {name} 的三段 split_ranges 不完整")
        normalized_ranges = {
            split: (int(bounds[0]), int(bounds[1]))
            for split, bounds in ranges.items()
        }
        label_audits[name] = _audit_legacy_p85_arrays(
            root, meta, name, normalized_ranges
        )
        dataset_names.append(name)
        split_ranges[name] = {
            split: [int(bounds[0]), int(bounds[1])]
            for split, bounds in normalized_ranges.items()
        }
        generation_hashes[name] = str(meta.get("generation_hash"))

    if haller_sources_read:
        raise ValueError(
            "B1 dataset provenance 发现 Haller source，禁止进入 B1："
            f"{sorted(haller_sources_read)}"
        )
    if dataset_scope == B1_DATASET_SCOPE_VALID:
        expected_names = set(ds.VALID_WEAK_DATASETS)
        actual_names = set(dataset_names)
        if actual_names != expected_names or len(dataset_names) != len(expected_names):
            raise ValueError(
                "B1 valid_six_datasets 必须恰好包含六个有效数据集："
                f"expected={sorted(expected_names)} actual={dataset_names}"
            )

    return {
        "roots": roots,
        "root": roots[0] if len(roots) == 1 else roots,
        "dataset_names": dataset_names,
        "dataset_scope": dataset_scope,
        "split_mode": ds.WEAK_SUPERVISION_SPLIT_MODE,
        "split_name": "train",
        "split_ranges": split_ranges,
        "window": {
            "t_win": int(data_config["t_win"]),
            "window_step": int(data_config["window_step"]),
            "frame_unit": "index",
            "complete_only": True,
        },
        "normalization_source": ds.NORMALIZATION_SOURCE,
        "normalization_frozen": True,
        "label_source": B1_LABEL_SOURCE,
        "sampling_source": B1_LABEL_SOURCE,
        "loss_label_source": B1_LABEL_SOURCE,
        "generation_hashes": generation_hashes,
        "label_audits": label_audits,
        # 由已验证的 label/sampling/loss provenance 计算；B1 没有 Haller
        # artifact loader，因此该列表必须为空而非由调用方传入。
        "haller_artifacts_read": sorted(haller_sources_read),
    }


def _b1_sampling_config(data_config: Mapping[str, Any]) -> dict[str, Any]:
    """把 B1 实际使用的窗口/采样参数写成 checkpoint contract。"""
    return {
        "patch_size": [int(value) for value in data_config["patch_size"]],
        "stride": [int(value) for value in data_config["stride"]],
        "t_win": int(data_config["t_win"]),
        "window_step": int(data_config["window_step"]),
        "t_scale": float(data_config["t_scale"]),
        "groups": [int(value) for value in data_config["groups"]],
        "delta_frac": float(data_config["delta_frac"]),
        "L": int(data_config["L"]),
        "n_substeps": int(data_config["n_substeps"]),
        "batch_size": int(data_config["batch_size"]),
        "samples_per_epoch": int(data_config["samples_per_epoch"]),
        "positive_fraction": float(data_config["positive_fraction"]),
        "seed": int(data_config["seed"]),
        "sampling_source": B1_LABEL_SOURCE,
    }


def _b1_split_config(dataset_contract: Mapping[str, Any]) -> dict[str, Any]:
    """生成 B1 训练使用的 train-only split contract。"""
    return {
        "split_name": "train",
        "split_mode": ds.WEAK_SUPERVISION_SPLIT_MODE,
        "frame_unit": "index",
        "complete_only": True,
        "split_ranges": copy.deepcopy(dataset_contract["split_ranges"]),
    }


def _b1_extra_metadata(dataset_contract: Mapping[str, Any]) -> dict[str, Any]:
    """生成不含 Haller GT source 的 B1 审计 metadata。"""
    haller_read_summary = _haller_read_summary(dataset_contract)
    return {
        "artifact_namespace": B1_ARTIFACT_NAMESPACE,
        "artifact_role": B1_ARTIFACT_ROLE,
        "headline_eligible": False,
        "label_source": B1_LABEL_SOURCE,
        "formal_loss_source": B1_LABEL_SOURCE,
        "sampling_source": B1_LABEL_SOURCE,
        "removed_input_channels": ["ivd"],
        "model_input_channel_count": 6,
        **haller_read_summary,
        "warm_start_aux": False,
        "dataset_contract": copy.deepcopy(dataset_contract),
    }


def _haller_read_summary(dataset_contract: Mapping[str, Any]) -> dict[str, Any]:
    """从已验证 provenance 生成 Haller artifact 读取审计字段。"""
    sources = set(dataset_contract.get("haller_artifacts_read", ()))
    return {
        "haller_artifacts_read": sorted(sources),
        "haller_train_artifact_read": (
            contract.LABEL_SOURCE_HALLER_TRAIN in sources
        ),
        "haller_gt_test_artifact_read": (
            contract.LABEL_SOURCE_HALLER_TEST in sources
        ),
        "haller_gt_calibration_artifact_read": (
            contract.LABEL_SOURCE_HALLER_CALIBRATION in sources
        ),
    }


def _require_b1_adapter(model):
    """确认模型是 vendor 外部的 B1 raw-7-to-model-6 adapter。"""
    if not isinstance(model, contract.ChannelSelectingAdapter):
        raise TypeError(
            "B1 必须使用 vendor 外部 ChannelSelectingAdapter；禁止直接构造 6-channel vendor model"
        )
    if model.mode != B1_MODE:
        raise ValueError(f"B1 adapter mode 不匹配：实际 {model.mode!r}")
    contract.validate_feature_schema(model.input_schema, contract.FEATURE_SCHEMA_7)
    contract.validate_feature_schema(model.feature_schema, contract.FEATURE_SCHEMA_6)
    return model


def prepare_b1_batch(
    loader_batch,
    *,
    model,
    split_name: str | None = None,
    label_source: str | None = None,
    sampling_source: str | None = None,
    sampling_config: Mapping[str, Any] | None = None,
):
    """把 DataLoader 的 raw batch 转为 B1 model-facing batch。

    返回 ``(dummy_field, WeakSupervisionBatch)``。raw 输入必须是完整 7 通道；
    只有 adapter 选择出的六通道会进入 model 和 BCE，``ivd`` 不会被静默重排。
    """
    adapter = _require_b1_adapter(model)
    if split_name != "train":
        raise ValueError(
            f"B1 training batch 必须显式 split_name='train'，实际 {split_name!r}"
        )
    if label_source != B1_LABEL_SOURCE:
        raise ValueError(
            f"B1 batch label_source 必须显式为 {B1_LABEL_SOURCE!r}，实际 {label_source!r}"
        )
    if sampling_source != B1_LABEL_SOURCE:
        raise ValueError(
            f"B1 batch sampling_source 必须显式为 {B1_LABEL_SOURCE!r}，"
            f"实际 {sampling_source!r}"
        )
    if not isinstance(loader_batch, (tuple, list)) or len(loader_batch) != 2:
        raise ValueError("B1 DataLoader batch 必须是 ((dummy_field, raw_pathlines), labels)")
    inputs, labels = loader_batch
    if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
        raise ValueError("B1 DataLoader batch 输入必须是 (dummy_field, raw_pathlines)")
    dummy, raw_pathlines = inputs
    raw_pathlines = (raw_pathlines if isinstance(raw_pathlines, torch.Tensor)
                     else torch.as_tensor(raw_pathlines))
    labels = labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels)
    dummy = dummy if isinstance(dummy, torch.Tensor) else torch.as_tensor(dummy)
    raw_pathlines = raw_pathlines.float()
    labels = labels.float()
    selected = adapter.adapt(
        raw_pathlines, input_schema=contract.FEATURE_SCHEMA_7
    )
    provenance = {
        "feature_ablation": {
            "adapter": "external_channel_selector",
            "raw_schema": contract.FEATURE_SCHEMA_7.as_dict(),
            "model_schema": contract.FEATURE_SCHEMA_6.as_dict(),
            "removed_input_channels": ["ivd"],
        },
        "sampling": {"source": B1_LABEL_SOURCE},
    }
    if sampling_config is not None:
        provenance["window"] = copy.deepcopy(dict(sampling_config))
    batch = contract.WeakSupervisionBatch(
        pathlines=selected,
        labels=labels,
        label_source=B1_LABEL_SOURCE,
        split_name="train",
        feature_schema=contract.FEATURE_SCHEMA_6,
        input_schema=contract.FEATURE_SCHEMA_7,
        sampling_source=B1_LABEL_SOURCE,
        provenance=provenance,
        mode=B1_MODE,
    )
    return dummy, batch


def _b1_forward_loss_details(
    model,
    criterion,
    loader_batch,
    *,
    split_name: str,
    label_source: str,
    sampling_source: str,
    sampling_config: Mapping[str, Any] | None = None,
):
    """执行 B1 batch 适配、前向和 mode-aware loss，并返回审计对象。"""
    adapter = _require_b1_adapter(model)
    if not isinstance(criterion, contract.ModeAwareLoss):
        raise TypeError("B1 loss 必须是 mode-aware loss；不得绕过 batch/source guard")
    if criterion.mode != B1_MODE:
        raise ValueError(f"B1 loss mode 不匹配：实际 {criterion.mode!r}")
    dummy, batch = prepare_b1_batch(
        loader_batch,
        model=adapter,
        split_name=split_name,
        label_source=label_source,
        sampling_source=sampling_source,
        sampling_config=sampling_config,
    )
    pred = adapter.forward_batch(batch, dummy_field=dummy, consumer="train")
    loss = criterion(pred, batch)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("B1 criterion 必须返回 scalar tensor loss")
    if not bool(torch.isfinite(loss).all()):
        raise ValueError("B1 loss 产生非有限值")
    return loss, batch, pred


def b1_forward_loss(
    model,
    criterion,
    loader_batch,
    *,
    split_name: str | None = None,
    label_source: str | None = None,
    sampling_source: str | None = None,
    sampling_config: Mapping[str, Any] | None = None,
):
    """执行一次 B1 adapter → model → contract loss 的公开单批 seam。"""
    loss, _batch, _pred = _b1_forward_loss_details(
        model,
        criterion,
        loader_batch,
        split_name=split_name,
        label_source=label_source,
        sampling_source=sampling_source,
        sampling_config=sampling_config,
    )
    return loss


def run_b1_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    *,
    grad_clip: float = 1.0,
    max_steps: int | None = None,
    sampling_config: Mapping[str, Any] | None = None,
):
    """运行一个 B1 训练 epoch，并返回 loss 与可审计 batch 统计。"""
    model.train()
    total_loss = 0.0
    batch_count = 0
    label_count = 0
    positive_count = 0
    for loader_batch in loader:
        if max_steps is not None and batch_count >= int(max_steps):
            break
        inputs, labels = loader_batch
        dummy, raw_pathlines = inputs
        loader_batch = (
            (dummy.to(device), raw_pathlines.to(device)), labels.to(device)
        )
        optimizer.zero_grad()
        loss, batch, _pred = _b1_forward_loss_details(
            model,
            criterion,
            loader_batch,
            split_name="train",
            label_source=B1_LABEL_SOURCE,
            sampling_source=B1_LABEL_SOURCE,
            sampling_config=sampling_config,
        )
        loss.backward()
        if grad_clip and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batch_count += 1
        label_count += int(batch.labels.numel())
        positive_count += int((batch.labels == 1).sum().item())
    if batch_count == 0:
        raise ValueError("B1 loader 为空或 max_steps=0：训练无样本可跑")
    return total_loss / batch_count, {
        "batches": batch_count,
        "label_count": label_count,
        "positive_label_count": positive_count,
        "known_label_count": label_count,
        "unknown_label_count": 0,
        "input_channel_count": contract.FEATURE_SCHEMA_6.channel_count,
    }


def _artifact_paths(train_config: Mapping[str, Any]):
    """为 B1 强制建立独立 namespace，避免覆盖 B0/W 方法目录。"""
    base = pathlib.Path(train_config["ckpt_dir"])
    artifact_dir = (base if base.name.lower() == B1_ARTIFACT_NAMESPACE
                    else base / B1_ARTIFACT_NAMESPACE)
    configured_name = str(train_config["run_name"]).strip()
    run_name = (configured_name if "b1" in configured_name.lower()
                else f"b1_{configured_name}")
    return artifact_dir, run_name


def _save_b1_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
    checkpoint_dataset_config,
    split_config,
    sampling_config,
    epoch,
    global_step,
    metrics,
    seed,
    dataset_contract,
):
    """保存 B1 的同一份 checkpoint contract，避免 latest/milestone 漂移。"""
    contract.save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        mode=B1_MODE,
        feature_schema=contract.FEATURE_SCHEMA_6,
        adapter_input_schema=contract.FEATURE_SCHEMA_7,
        dataset_config=checkpoint_dataset_config,
        split_config=split_config,
        sampling_config=sampling_config,
        label_source=B1_LABEL_SOURCE,
        sampling_source=B1_LABEL_SOURCE,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics,
        seed=int(seed),
        anchor_hash=None,
        calibration_policy={
            "policy": "not_used_for_B1_diagnostic",
            "threshold_source": "not_used",
            "gate_source": "not_used",
            "method_selection_source": "not_used",
            "haller_artifacts_read": False,
        },
        warm_start_aux=False,
        extra_metadata=_b1_extra_metadata(dataset_contract),
    )


def run_b1_training(
    config: Mapping[str, Any] | str | pathlib.Path,
    *,
    resume: str | pathlib.Path = "none",
    epochs: int | None = None,
    max_steps: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """执行 B1 训练 smoke/pilot，并写独立 contract checkpoint/report。

    ``resume='none'`` 从头开始；``auto`` 只查找本 B1 namespace 的 latest checkpoint。
    任何旧 B0 checkpoint 都交给公共 contract 显式拒绝，不会被当作 warm-start。
    """
    if isinstance(config, (str, pathlib.Path)):
        from train_kaggle import load_config
        config = load_config(config)
    config = copy.deepcopy(dict(config))
    if epochs is not None:
        config.setdefault("train", {})["epochs"] = epochs
    validate_b1_config(config)
    data_config = config["data"]
    train_config = config["train"]
    if device is not None:
        train_config["device"] = str(device)

    from train_kaggle import (
        TwoStepLR,
        _make_dataset,
        _make_loader,
        _resolve_device,
        build_criterion_from_config,
        build_model_from_config,
        enable_tf32,
    )
    from vendor.DeepUtils.utils.random import set_random_seed

    set_random_seed(int(train_config["seed"]))
    enable_tf32()
    resolved_device = _resolve_device(train_config["device"])
    model = build_model_from_config(config, mode=B1_MODE).to(resolved_device)
    criterion = build_criterion_from_config(config, mode=B1_MODE).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = TwoStepLR(
        optimizer,
        lr=float(train_config["lr"]),
        second_lr=float(train_config["second_lr"]),
        warmup_epochs=int(train_config["warmup_epochs"]),
    )

    train_ds = _make_dataset(data_config, "train")
    dataset_contract = validate_b1_dataset_contract(train_ds, data_config)
    sampling_config = _b1_sampling_config(data_config)
    split_config = _b1_split_config(dataset_contract)
    checkpoint_dataset_config = {
        **dataset_contract,
        "patch_size": sampling_config["patch_size"],
        "stride": sampling_config["stride"],
    }
    train_loader = _make_loader(
        train_ds,
        data_config["batch_size"],
        data_config["num_workers"],
        resolved_device,
    )

    artifact_dir, run_name = _artifact_paths(train_config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    latest_path = artifact_dir / f"{run_name}_ckpt_latest.pth"
    report_path = artifact_dir / f"{run_name}_diagnostic.json"
    resume_text = str(resume)
    start_epoch = 0
    global_step = 0
    resumed_from = None
    if resume_text.lower() == "none":
        print("[B1] from_scratch=true; warm_start_aux=false")
    else:
        resume_path = latest_path if resume_text.lower() == "auto" else pathlib.Path(resume)
        if resume_path.exists():
            loaded = contract.load_checkpoint(
                resume_path,
                model,
                optimizer,
                scheduler,
                expected_mode=B1_MODE,
                expected_feature_schema=contract.FEATURE_SCHEMA_6,
                expected_dataset_config=checkpoint_dataset_config,
                expected_split_config=split_config,
                expected_sampling_config=sampling_config,
                expected_label_source=B1_LABEL_SOURCE,
                expected_sampling_source=B1_LABEL_SOURCE,
                expected_anchor_hash=None,
                device=resolved_device,
                restore_rng=True,
                load_mode="resume",
            )
            start_epoch = int(loaded["start_epoch"])
            global_step = int(loaded["global_step"])
            resumed_from = str(resume_path)
            print(f"[B1] resume={resume_path} -> start_epoch={start_epoch}")
        elif resume_text.lower() != "auto":
            raise FileNotFoundError(f"B1 resume checkpoint 不存在: {resume_path}")
        else:
            print("[B1] resume=auto but no B1 checkpoint; from_scratch=true")

    haller_read_summary = _haller_read_summary(dataset_contract)
    print(
        f"[B1] mode=B1 role=diagnostic label_source={B1_LABEL_SOURCE} "
        "sampling_source=legacy_p85 split=train split_mode=weak_supervision "
        "model_input_channels=6 removed=ivd "
        f"Haller_artifacts_read={haller_read_summary['haller_artifacts_read']}"
    )
    history = []
    target_epochs = int(train_config["epochs"])
    for epoch in range(start_epoch, target_epochs):
        train_ds.set_epoch(epoch)
        lr = scheduler.step(epoch)
        train_loss, stats = run_b1_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            resolved_device,
            grad_clip=float(train_config["grad_clip"]),
            max_steps=max_steps,
            sampling_config=sampling_config,
        )
        global_step += int(stats["batches"])
        metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "lr": float(lr),
            "label_source": B1_LABEL_SOURCE,
            "sampling_source": B1_LABEL_SOURCE,
            "input_channel_count": 6,
            **haller_read_summary,
            **stats,
        }
        history.append(metrics)
        _save_b1_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dataset_config=checkpoint_dataset_config,
            split_config=split_config,
            sampling_config=sampling_config,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            seed=train_config["seed"],
            dataset_contract=dataset_contract,
        )
        save_freq = int(train_config["save_freq"])
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            _save_b1_checkpoint(
                artifact_dir / f"{run_name}_E{epoch + 1}.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_dataset_config=checkpoint_dataset_config,
                split_config=split_config,
                sampling_config=sampling_config,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                seed=train_config["seed"],
                dataset_contract=dataset_contract,
            )
        print(
            f"[B1] epoch={epoch + 1}/{target_epochs} loss={train_loss:.6f} "
            f"lr={lr:g} label_source=legacy_p85 "
            f"Haller_artifacts_read={haller_read_summary['haller_artifacts_read']}"
        )

    report = {
        "format_version": "b1-diagnostic-v1",
        "mode": B1_MODE,
        "artifact_namespace": B1_ARTIFACT_NAMESPACE,
        "artifact_role": B1_ARTIFACT_ROLE,
        "headline_eligible": False,
        "label_source": B1_LABEL_SOURCE,
        "formal_loss_source": B1_LABEL_SOURCE,
        "sampling_source": B1_LABEL_SOURCE,
        "dataset_scope": dataset_contract["dataset_scope"],
        "label_percentile": 85.0,
        "removed_input_channels": ["ivd"],
        "model_input_channel_count": 6,
        "feature_schema": contract.FEATURE_SCHEMA_6.as_dict(),
        "adapter_input_schema": contract.FEATURE_SCHEMA_7.as_dict(),
        "split_config": split_config,
        "dataset_contract": dataset_contract,
        "sampling_config": sampling_config,
        "warm_start_aux": False,
        "from_scratch": resumed_from is None,
        "resumed_from": resumed_from,
        **haller_read_summary,
        "start_epoch": start_epoch,
        "epochs_requested": target_epochs,
        "epochs_completed": max(start_epoch, target_epochs),
        "global_step": global_step,
        "history": history,
        "checkpoint_path": str(latest_path),
        "diagnostic_report_path": str(report_path),
        "device": str(resolved_device),
        "seed": int(train_config["seed"]),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "mode": B1_MODE,
        "checkpoint_path": str(latest_path),
        "diagnostic_report": str(report_path),
        "artifact_dir": str(artifact_dir),
        "run_name": run_name,
        "start_epoch": start_epoch,
        "epochs_completed": max(start_epoch, target_epochs),
        "global_step": global_step,
        "history": history,
    }
