"""B1 诊断输入消融的训练/评价接缝测试。

测试只观察公开的 B1 adapter、训练入口和 artifact contract：raw 7 通道输入
必须在 vendor 外部变成 6 通道，监督仍明确来自 legacy_p85，并且新 split 的
checkpoint/report 与其他方法独立。不会读取或生成任何 Haller artifact。
"""

import copy
import json
import pathlib

import numpy as np
import pytest
import torch
import torch.nn as nn

import dataset as ds
import test_dataset as tds
import weak_supervision_contract as contract


class _RecordingModel(nn.Module):
    """测试用 model seam：记录真正收到的 model-facing channel 数和值。"""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.last_x = None

    def forward(self, data):
        _dummy, pathlines = data
        self.last_x = pathlines.detach().clone()
        # 输出 (B, K)，保留对参数和输入的可微路径。
        return torch.sigmoid(pathlines[..., 0].mean(dim=1) + self.bias)


def _raw_loader_batch(batch_size=2, length=3, trajectories=4):
    raw = torch.empty(batch_size, length, trajectories, 7)
    for channel in range(7):
        raw[..., channel] = float(channel + 1)
    labels = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )[:batch_size, :trajectories]
    return ((torch.zeros(batch_size, 1, 1, 1), raw), labels)


def _small_b1_config(root, ckpt_dir, *, epochs=5):
    """返回显式声明全部 B1 split/window/source 参数的 CPU smoke config。"""
    return {
        "data": {
            "root": str(root),
            "dataset_scope": "synthetic_fixture",
            "split_mode": ds.WEAK_SUPERVISION_SPLIT_MODE,
            "split": "train",
            "val_split": "none",
            "label_source": contract.LABEL_SOURCE_LEGACY_P85,
            "sampling_source": contract.LABEL_SOURCE_LEGACY_P85,
            "loss_label_source": contract.LABEL_SOURCE_LEGACY_P85,
            "seed": 0,
            "batch_size": 2,
            "num_workers": 0,
            "samples_per_epoch": 4,
            "positive_fraction": 0.5,
            "patch_size": [32, 32],
            "stride": [16, 16],
            "t_win": 4,
            "window_step": 2,
            "t_scale": 0.25,
            "groups": [4, 4],
            "delta_frac": 0.05,
            "L": 4,
            "n_substeps": 1,
        },
        "model": {
            "NAME": "BaseSeg",
            "encoder_args": {
                "NAME": "PathlineTransformerV0",
                "in_channels": 6,
                "PathlineGroups": 16,
                "KpathlinePerGroup": 4,
                "num_classes": 1,
                "num_encoder_layers": 1,
                "dmodel": 32,
                "k": 8,
            },
            "criterion_args": {"NAME": "BCELoss"},
        },
        "train": {
            "mode": contract.MODE_B1,
            "epochs": epochs,
            "lr": 1e-4,
            "weight_decay": 1e-6,
            "warmup_epochs": 2,
            "second_lr": 5e-6,
            "grad_clip": 1.0,
            "save_freq": 1,
            "seed": 0,
            "device": "cpu",
            "amp": False,
            "data_parallel": False,
            "ckpt_dir": str(ckpt_dir),
            "run_name": "b1_smoke",
            "warm_start_aux": False,
        },
    }


@pytest.fixture
def weak_b1_root(tmp_path):
    """小型新三段 split 数据集；label source 明确为 legacy_p85。"""
    u, v, xdim, ydim, tdim = tds.synth_prepared(tmp_path / "input", T=40)
    root = tmp_path / "weak_dataset"
    ds.prepare_dataset(
        None,
        str(root),
        u=u,
        v=v,
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
        label_source=contract.LABEL_SOURCE_LEGACY_P85,
        sampling_source=contract.LABEL_SOURCE_LEGACY_P85,
        loss_label_source=contract.LABEL_SOURCE_LEGACY_P85,
        t_win=4,
        window_step=2,
    )
    return root


def test_b1_public_step_removes_only_local_ivd_and_uses_contract_loss():
    """B1 单批前向/loss：模型收到 6 通道，且剩余顺序保持不变。"""
    from b1_diagnostic import b1_forward_loss, prepare_b1_batch

    recorder = _RecordingModel()
    adapter = contract.ChannelSelectingAdapter(recorder, contract.MODE_B1)
    loader_batch = _raw_loader_batch()

    _dummy, b1_batch = prepare_b1_batch(
        loader_batch,
        model=adapter,
        split_name="train",
        label_source=contract.LABEL_SOURCE_LEGACY_P85,
        sampling_source=contract.LABEL_SOURCE_LEGACY_P85,
    )
    assert b1_batch.feature_schema == contract.FEATURE_SCHEMA_6
    assert b1_batch.input_schema == contract.FEATURE_SCHEMA_7
    expected = loader_batch[0][1][..., [0, 1, 2, 4, 5, 6]]
    assert torch.equal(b1_batch.pathlines, expected)

    criterion = contract.ModeAwareLoss(contract.MODE_B1, nn.BCELoss())
    loss = b1_forward_loss(
        adapter,
        criterion,
        loader_batch,
        split_name="train",
        label_source=contract.LABEL_SOURCE_LEGACY_P85,
        sampling_source=contract.LABEL_SOURCE_LEGACY_P85,
    )
    assert torch.isfinite(loss)
    assert recorder.last_x is not None
    assert recorder.last_x.shape[-1] == 6
    assert torch.equal(recorder.last_x, expected)


def test_b1_batch_rejects_non_raw_seven_channel_schema_loudly():
    """输入若已被错误地缩成 6 通道，B1 raw adapter 不得静默接受。"""
    from b1_diagnostic import prepare_b1_batch

    recorder = _RecordingModel()
    adapter = contract.ChannelSelectingAdapter(recorder, contract.MODE_B1)
    ((dummy, raw), labels) = _raw_loader_batch()
    with pytest.raises(ValueError, match=r"channel|schema|通道"):
        prepare_b1_batch(
            ((dummy, raw[..., :6]), labels),
            model=adapter,
            split_name="train",
            label_source=contract.LABEL_SOURCE_LEGACY_P85,
            sampling_source=contract.LABEL_SOURCE_LEGACY_P85,
        )


def test_b1_config_requires_explicit_new_split_source_and_six_channels(tmp_path):
    """B1 配置缺少 split/source 或错误 channel 时必须 fail loudly。"""
    from b1_diagnostic import validate_b1_config

    config = _small_b1_config(tmp_path / "dataset", tmp_path / "ckpt")
    assert validate_b1_config(config)["mode"] == contract.MODE_B1

    missing_split = copy.deepcopy(config)
    del missing_split["data"]["split_mode"]
    with pytest.raises(ValueError, match=r"split_mode|weak_supervision|split"):
        validate_b1_config(missing_split)

    wrong_source = copy.deepcopy(config)
    wrong_source["data"]["label_source"] = contract.LABEL_SOURCE_LOCAL_P90_P60
    with pytest.raises(ValueError, match=r"legacy_p85|source|监督"):
        validate_b1_config(wrong_source)

    calibration_validation = copy.deepcopy(config)
    calibration_validation["data"]["val_split"] = "calibration"
    with pytest.raises(ValueError, match=r"val_split|train|calibration"):
        validate_b1_config(calibration_validation)

    wrong_channels = copy.deepcopy(config)
    wrong_channels["model"]["encoder_args"]["in_channels"] = 7
    with pytest.raises(ValueError, match=r"6|channel|通道"):
        validate_b1_config(wrong_channels)

    mismatched_seed = copy.deepcopy(config)
    mismatched_seed["data"]["seed"] = 1
    with pytest.raises(ValueError, match=r"seed|随机"):
        validate_b1_config(mismatched_seed)

    unsupported_amp = copy.deepcopy(config)
    unsupported_amp["train"]["amp"] = True
    with pytest.raises(ValueError, match=r"FP32|amp|data_parallel"):
        validate_b1_config(unsupported_amp)


def test_b1_smoke_writes_independent_contract_checkpoint_and_report(
    weak_b1_root, tmp_path, capsys
):
    """5 epoch CPU smoke：checkpoint 可恢复，report 明确 B1/legacy/no-Haller。"""
    from b1_diagnostic import run_b1_training
    from weak_supervision_contract import checkpoint_metadata

    config = _small_b1_config(weak_b1_root, tmp_path / "shared_outputs")
    result = run_b1_training(config, resume="none", max_steps=1, device="cpu")

    checkpoint = pathlib.Path(result["checkpoint_path"])
    report_path = pathlib.Path(result["diagnostic_report"])
    assert checkpoint.exists()
    assert report_path.exists()
    assert "b1_diagnostic" in str(checkpoint.parent).lower()

    metadata = checkpoint_metadata(checkpoint)
    assert metadata["mode"] == contract.MODE_B1
    assert metadata["feature_schema"] == contract.FEATURE_SCHEMA_6.as_dict()
    assert metadata["adapter_input_schema"] == contract.FEATURE_SCHEMA_7.as_dict()
    assert metadata["split_config"]["split_name"] == "train"
    assert metadata["split_config"]["split_mode"] == ds.WEAK_SUPERVISION_SPLIT_MODE
    assert metadata["label_source"] == contract.LABEL_SOURCE_LEGACY_P85
    assert metadata["sampling_source"] == contract.LABEL_SOURCE_LEGACY_P85
    assert metadata["sampling_config"]["seed"] == 0
    assert metadata["anchor_hash"] is None
    assert metadata["warm_start_aux"] is False

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == contract.MODE_B1
    assert report["artifact_role"] == "diagnostic"
    assert report["label_source"] == contract.LABEL_SOURCE_LEGACY_P85
    assert report["formal_loss_source"] == contract.LABEL_SOURCE_LEGACY_P85
    assert report["removed_input_channels"] == ["ivd"]
    assert report["model_input_channel_count"] == 6
    assert report["haller_artifacts_read"] == []
    assert report["haller_train_artifact_read"] is False
    assert report["haller_gt_test_artifact_read"] is False
    assert report["warm_start_aux"] is False
    assert report["headline_eligible"] is False
    assert report["dataset_scope"] == "synthetic_fixture"
    assert report["label_percentile"] == 85.0
    assert report["dataset_contract"]["haller_artifacts_read"] == []
    assert report["epochs_completed"] == 5
    assert len(report["history"]) == 5

    captured = capsys.readouterr().out
    assert "mode=B1" in captured
    assert "legacy_p85" in captured
    assert "Haller" in captured

    # auto resume 只允许继续 B1 contract；不得把同目录中的 B0 旧产物当作输入。
    resumed_config = copy.deepcopy(config)
    resumed_config["train"]["epochs"] = 6
    resumed = run_b1_training(
        resumed_config, resume="auto", max_steps=1, device="cpu"
    )
    assert resumed["start_epoch"] == 5
    resumed_metadata = checkpoint_metadata(resumed["checkpoint_path"])
    assert resumed_metadata["mode"] == contract.MODE_B1
    assert resumed_metadata["epoch"] == 5


def test_b1_dataset_binds_legacy_p85_to_percentile_85(weak_b1_root):
    """只声明 legacy_p85 但改成别的 percentile 时，B1 contract 必须阻断。"""
    from b1_diagnostic import validate_b1_dataset_contract

    meta_path = pathlib.Path(weak_b1_root) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["percentile"] = 90.0
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    dataset = ds.WeakLabelPathlineDataset(
        str(weak_b1_root),
        split="train",
        patch_size=(32, 32),
        stride=(16, 16),
        t_win=4,
        window_step=2,
        label_source=contract.LABEL_SOURCE_LEGACY_P85,
    )
    config = _small_b1_config(weak_b1_root, weak_b1_root / "unused")
    with pytest.raises(ValueError, match=r"percentile|85"):
        validate_b1_dataset_contract(dataset, config["data"])


def test_b1_valid_scope_requires_exact_six_dataset_names(weak_b1_root):
    """正式 valid_six_datasets scope 不能用单个或未知数据集静默替代。"""
    from b1_diagnostic import validate_b1_dataset_contract

    dataset = ds.WeakLabelPathlineDataset(
        str(weak_b1_root),
        split="train",
        patch_size=(32, 32),
        stride=(16, 16),
        t_win=4,
        window_step=2,
        label_source=contract.LABEL_SOURCE_LEGACY_P85,
    )
    config = _small_b1_config(weak_b1_root, weak_b1_root / "unused")
    config["data"]["dataset_scope"] = "valid_six_datasets"
    with pytest.raises(ValueError, match=r"六个|six|有效数据集"):
        validate_b1_dataset_contract(dataset, config["data"])


def test_b1_rejects_tampered_legacy_p85_taus(weak_b1_root):
    """B1 必须核对 metadata tau 与实际 IVD 的 p85，而非只信 source 字符串。"""
    from b1_diagnostic import validate_b1_dataset_contract

    meta_path = pathlib.Path(weak_b1_root) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["taus"]["train"] += 1.0
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    dataset = ds.WeakLabelPathlineDataset(
        str(weak_b1_root),
        split="train",
        patch_size=(32, 32),
        stride=(16, 16),
        t_win=4,
        window_step=2,
        label_source=contract.LABEL_SOURCE_LEGACY_P85,
    )
    config = _small_b1_config(weak_b1_root, weak_b1_root / "unused")
    with pytest.raises(ValueError, match=r"实际 p85|tau"):
        validate_b1_dataset_contract(dataset, config["data"])


def test_b1_rejects_tampered_legacy_p85_labels(weak_b1_root):
    """B1 必须重建并核对 label_field，防止 custom labels 冒充 legacy_p85。"""
    from b1_diagnostic import validate_b1_dataset_contract

    label_path = pathlib.Path(weak_b1_root) / "label_field.npy"
    labels = np.asarray(np.load(label_path)).copy()
    labels[0, 0, 0] = np.uint8(1 - int(labels[0, 0, 0]))
    np.save(label_path, labels)
    dataset = ds.WeakLabelPathlineDataset(
        str(weak_b1_root),
        split="train",
        patch_size=(32, 32),
        stride=(16, 16),
        t_win=4,
        window_step=2,
        label_source=contract.LABEL_SOURCE_LEGACY_P85,
    )
    config = _small_b1_config(weak_b1_root, weak_b1_root / "unused")
    with pytest.raises(ValueError, match=r"canonical legacy_p85|label_field"):
        validate_b1_dataset_contract(dataset, config["data"])


def test_b1_old_legacy_b0_checkpoint_is_not_warm_started(weak_b1_root, tmp_path):
    """B1 从头训练契约显式拒绝阶段 0 的 legacy checkpoint。"""
    from b1_diagnostic import run_b1_training

    config = _small_b1_config(weak_b1_root, tmp_path / "outputs", epochs=1)
    legacy = tmp_path / "old_b0.pth"
    torch.save({"model": {}}, legacy)
    with pytest.raises(ValueError, match=r"legacy|B0|mode|warm_start"):
        run_b1_training(config, resume=str(legacy), max_steps=1, device="cpu")


def test_train_cli_dispatches_configured_b1_mode_without_b0_fallback(tmp_path, monkeypatch):
    """train_kaggle 的显式 B1 mode 走 B1 runner，不落回旧 B0 主循环。"""
    import b1_diagnostic
    import train_kaggle
    import yaml

    config = _small_b1_config(tmp_path / "unused", tmp_path / "ckpt", epochs=5)
    config_path = tmp_path / "b1.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    observed = {}

    def fake_runner(received, **kwargs):
        observed["config"] = received
        observed["kwargs"] = kwargs
        return {"mode": contract.MODE_B1}

    monkeypatch.setattr(b1_diagnostic, "run_b1_training", fake_runner)
    assert train_kaggle.main([
        "--config", str(config_path), "--mode", "B1", "--resume", "none",
        "--epochs", "7", "--max-steps", "1",
    ]) == {"mode": contract.MODE_B1}
    assert observed["config"]["train"]["mode"] == contract.MODE_B1
    assert observed["kwargs"] == {
        "resume": "none", "epochs": 7, "max_steps": 1,
    }

    # 显式 --mode B1 可以补齐缺失的 mode；其余 split/source 字段仍由
    # validate_b1_config 强制要求，不能回退到 B0 默认值。
    missing_mode = copy.deepcopy(config)
    del missing_mode["train"]["mode"]
    config_path.write_text(yaml.safe_dump(missing_mode), encoding="utf-8")
    assert train_kaggle.main([
        "--config", str(config_path), "--mode", "B1", "--resume", "none",
    ]) == {"mode": contract.MODE_B1}
    assert observed["config"]["train"]["mode"] == contract.MODE_B1

    # 也支持只由配置中的 train.mode=B1 触发，不会落回 B0 主循环。
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert train_kaggle.main([
        "--config", str(config_path), "--resume", "none",
    ]) == {"mode": contract.MODE_B1}
    assert observed["kwargs"] == {
        "resume": "none", "epochs": None, "max_steps": None,
    }
