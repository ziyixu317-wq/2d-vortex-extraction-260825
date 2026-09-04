"""票 03：弱监督公共 mode/schema/batch/checkpoint 契约测试。

测试只通过公共 contract seam 观察行为；不触及 vendor/DeepUtils 的内部实现。
"""

import numpy as np
import random

import pytest
import torch
import torch.nn as nn

import weak_supervision_contract as contract


def _assert_nested_equal(left, right):
    """递归比较含 tensor 的 optimizer/scheduler state。"""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert list(left) == list(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


@pytest.mark.parametrize(
    ("mode", "channels"),
    [
        (contract.MODE_B0, ("px", "py", "t", "ivd", "distance", "u", "v")),
        (contract.MODE_B1, ("px", "py", "t", "distance", "u", "v")),
        (contract.MODE_W1, ("px", "py", "t", "ivd", "distance", "u", "v")),
        (contract.MODE_W1_P, ("px", "py", "t", "ivd", "distance", "u", "v")),
        (contract.MODE_W1_H, ("px", "py", "t", "ivd", "distance", "u", "v")),
        (contract.MODE_W2, ("px", "py", "t", "ivd", "distance", "u", "v")),
        (contract.MODE_W3, ("px", "py", "t", "ivd", "distance", "u", "v")),
    ],
)
def test_mode_schema_has_explicit_channel_order(mode, channels):
    """每个 mode 都公开固定 channel 顺序和数量。"""
    schema = contract.feature_schema_for_mode(mode)

    assert schema.channels == channels
    assert schema.channel_count == len(channels)
    assert schema.as_dict()["channels"] == list(channels)
    assert contract.mode_spec(mode).feature_schema == schema


def test_wrong_channel_order_is_rejected_loudly():
    """同样的通道数但顺序错误也不能通过 schema 校验。"""
    schema = contract.feature_schema_for_mode(contract.MODE_B0)
    wrong = schema.as_dict()
    wrong["channels"] = ["px", "py", "t", "distance", "ivd", "u", "v"]

    with pytest.raises(ValueError, match=r"feature schema|channel|通道"):
        contract.validate_feature_schema(wrong, contract.MODE_B0)


@pytest.mark.parametrize("field, value", [("channel_count", 7.9),
                                           ("local_ivd_channel", 3.9)])
def test_schema_rejects_non_integral_numeric_fields(field, value):
    """schema 的数值字段不能被 int() 静默截断。"""
    malformed = contract.FEATURE_SCHEMA_7.as_dict()
    malformed[field] = value

    with pytest.raises(ValueError, match=r"integer|整数|schema"):
        contract.FeatureSchema.from_mapping(malformed)


class _EchoModel(nn.Module):
    """只回传 pathline 的 fixture model，用于观察 adapter 的公共行为。"""

    def forward(self, data):
        return data[1]


class _ModeLinear(nn.Linear):
    """带公共 mode 属性的最小 target，用于验证 load-time 推导。"""

    def __init__(self, mode):
        super().__init__(3, 2)
        self.mode = mode


def test_b1_adapter_removes_only_local_ivd_channel():
    """B1 只去掉 ivd，位置/时间/distance/u/v 顺序保持不变。"""
    values = torch.arange(2 * 3 * 4 * 7, dtype=torch.float32).reshape(2, 3, 4, 7)
    adapter = contract.ChannelSelectingAdapter(_EchoModel(), contract.MODE_B1)

    got = adapter((torch.zeros(2, 1, 1, 1), values))

    assert got.shape == (2, 3, 4, 6)
    assert torch.equal(got, values[..., [0, 1, 2, 4, 5, 6]])


def test_adapter_rejects_wrong_input_schema_and_channel_count():
    """adapter 不接受错误的 schema 或无法匹配 schema 的 tensor。"""
    adapter = contract.ChannelSelectingAdapter(_EchoModel(), contract.MODE_B1)
    values = np.zeros((1, 2, 3, 7), dtype=np.float32)

    with pytest.raises(ValueError, match=r"feature schema|channel|通道"):
        adapter.adapt(values, input_schema=contract.FEATURE_SCHEMA_6)
    with pytest.raises(ValueError, match=r"channel|通道"):
        adapter.adapt(values[..., :6])


def _small_encoder_config(mode):
    """构造两个输入 schema 都能消费的最小真实 vendor model 配置。"""
    return {
        "NAME": "PathlineTransformerV0",
        "in_channels": contract.feature_schema_for_mode(mode).channel_count,
        "PathlineGroups": 4,
        "KpathlinePerGroup": 4,
        "num_classes": 1,
        "num_encoder_layers": 1,
        "dmodel": 32,
        "k": 4,
    }


@pytest.mark.parametrize("mode", [contract.MODE_B0, contract.MODE_B1])
def test_mode_model_dispatch_uses_external_adapter(mode):
    """mode dispatch 让 vendor model 只看到声明的 model schema。"""
    model = contract.build_model_for_mode(_small_encoder_config(mode), mode)
    values = torch.randn(1, 4, 16, 7)

    output = model((torch.zeros(1, 1, 1, 1), values))

    assert model.mode == contract.canonical_mode(mode)
    assert model.feature_schema == contract.feature_schema_for_mode(mode)
    assert output.shape == (1, 16)
    assert bool(torch.isfinite(output).all())
    assert bool(((output >= 0) & (output <= 1)).all())


def test_model_dispatch_rejects_config_channel_drift():
    """model config 的 in_channels 不能静默覆盖 mode schema。"""
    config = _small_encoder_config(contract.MODE_B1)
    config["in_channels"] = 7

    with pytest.raises(ValueError, match=r"mode|feature schema|channel|通道"):
        contract.build_model_for_mode(config, contract.MODE_B1)


def test_model_dispatch_rejects_non_integral_channel_count():
    """model config 的 in_channels 不能被 int() 截断。"""
    config = _small_encoder_config(contract.MODE_B1)
    config["in_channels"] = 6.9

    with pytest.raises(ValueError, match=r"integer|整数|in_channels"):
        contract.build_model_for_mode(config, contract.MODE_B1)


def _supervision_batch(mode=contract.MODE_B0, label_source="legacy_p85"):
    """小型 train batch fixture，明确分离 loss source 与 sampling source。"""
    channels = contract.feature_schema_for_mode(mode).channel_count
    return contract.WeakSupervisionBatch(
        pathlines=torch.zeros(2, 4, 8, channels),
        labels=torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]] * 2),
        label_mask=torch.tensor([[1, 1, 1, 0, 1, 0, 1, 1]] * 2, dtype=torch.bool),
        unknown_mask=torch.tensor([[0, 0, 0, 1, 0, 1, 0, 0]] * 2, dtype=torch.bool),
        feature_schema=contract.feature_schema_for_mode(mode),
        label_source=label_source,
        sampling_source="legacy_p85",
        split_name="train",
        mode=mode,
        provenance={
            "anchor": {"source": label_source},
            "pseudo": {"source": "teacher_probability"},
        },
    )


def test_training_batch_keeps_masks_and_source_provenance_separate():
    """batch seam 同时携带 known/unknown mask、loss source 和 anchor/pseudo provenance。"""
    batch = _supervision_batch()

    validated = contract.validate_training_batch(batch, contract.MODE_B0)

    assert validated is batch
    assert validated.label_mask.shape == validated.labels.shape
    assert not bool(torch.any(validated.label_mask & validated.unknown_mask))
    assert validated.label_source == "legacy_p85"
    assert validated.sampling_source == "legacy_p85"
    assert validated.provenance["anchor"]["source"] == "legacy_p85"
    assert validated.provenance["pseudo"]["source"] == "teacher_probability"


def test_b1_batch_uses_six_channel_model_schema():
    """B1 进入 loss seam 时是显式 6-channel batch。"""
    batch = _supervision_batch(mode=contract.MODE_B1)

    validated = contract.validate_training_batch(batch, contract.MODE_B1)

    assert validated.feature_schema == contract.FEATURE_SCHEMA_6
    assert validated.pathlines.shape[-1] == 6


def test_b1_batch_and_adapter_distinguish_raw_and_model_schemas():
    """B1 batch 保留 6-channel model schema，同时声明 raw 7-channel input。"""
    batch = _supervision_batch(mode=contract.MODE_B1)
    adapter = contract.ChannelSelectingAdapter(_EchoModel(), contract.MODE_B1)

    assert batch.input_schema == contract.FEATURE_SCHEMA_7
    adapted = adapter(batch)

    assert adapted.shape == batch.pathlines.shape
    assert torch.equal(adapted, batch.pathlines)


def test_adapter_evaluation_batch_uses_evaluation_source_guard():
    """test batch 经 adapter 前向时必须显式走 evaluation consumer。"""
    batch = _supervision_batch(
        mode=contract.MODE_W1_H,
        label_source="haller_gt_test",
    )
    batch.split_name = "test"
    adapter = contract.ChannelSelectingAdapter(_EchoModel(), contract.MODE_W1_H)

    output = adapter.forward_batch(batch, consumer="evaluation")

    assert output.shape == batch.pathlines.shape


def test_batch_defaults_are_known_labels_without_unknown_overlap():
    """省略 mask 时，默认是全 known，而不是两个互相重叠的全 1 mask。"""
    batch = contract.WeakSupervisionBatch(
        pathlines=torch.zeros(1, 1, 1, 7),
        labels=torch.tensor([[1.0]]),
        label_source="legacy_p85",
        split_name="train",
    )

    assert bool(batch.label_mask.all())
    assert not bool(batch.unknown_mask.any())


def test_batch_rejects_non_numeric_labels_and_batch_shape_drift():
    """公共 batch seam 在进入模型/loss 前拒绝坏 labels 和 batch 维度漂移。"""
    with pytest.raises(ValueError, match=r"batch|shape|labels"):
        contract.WeakSupervisionBatch(
            pathlines=torch.zeros(3, 1, 1, 7),
            labels=torch.zeros(2, 1),
            label_source="legacy_p85",
            split_name="train",
        )
    with pytest.raises(ValueError, match=r"numeric|数值|labels"):
        contract.WeakSupervisionBatch(
            pathlines=np.zeros((1, 1, 1, 7), dtype=np.float32),
            labels=np.array([["unknown"]], dtype=object),
            label_source="legacy_p85",
            split_name="train",
        )


@pytest.mark.parametrize(
    ("mode", "source"),
    [
        (contract.MODE_B0, "legacy_p85"),
        (contract.MODE_W1, "local_p90_p60"),
        (contract.MODE_W1_P, "local_p90_p60"),
        (contract.MODE_W1_H, "haller_anchor_train"),
        (contract.MODE_W2, "haller_anchor_train"),
        (contract.MODE_W3, "haller_anchor_train"),
    ],
)
def test_registered_seven_channel_training_modes_accept_contract_batch(mode, source):
    """B0/W1/W2/W3 family all retain the formal seven-channel batch seam."""
    batch = _supervision_batch(mode=mode, label_source=source)

    validated = contract.validate_training_batch(batch, mode)

    assert validated is batch
    assert validated.feature_schema == contract.FEATURE_SCHEMA_7
    assert validated.pathlines.shape[-1] == 7


@pytest.mark.parametrize(
    "source",
    [
        "legacy_p85",
        "local_p90_p60",
        "haller_anchor_train",
        "haller_gt_calibration",
        "haller_gt_test",
    ],
)
def test_label_source_values_are_explicitly_registered(source):
    """五种来源均可被识别，来源名不会退化为 sampling membership。"""
    assert contract.validate_label_source(source) == source


def test_training_rejects_calibration_and_test_haller_sources():
    """train seam 不能读取 calibration/test Haller GT，即使它藏在 provenance 中。"""
    for source in ("haller_gt_calibration", "haller_gt_test"):
        batch = _supervision_batch(mode=contract.MODE_W1_H, label_source=source)
        with pytest.raises(ValueError, match=r"train|training|禁止|Haller"):
            contract.validate_training_batch(batch, contract.MODE_W1_H)

    nested = _supervision_batch(mode=contract.MODE_W1_P,
                                label_source="local_p90_p60")
    nested.provenance["pseudo"]["source"] = "haller_gt_test"
    with pytest.raises(ValueError, match=r"test|haller_gt_test|Haller"):
        contract.validate_training_batch(nested, contract.MODE_W1_P)

    key_contaminated = _supervision_batch(
        mode=contract.MODE_W1_P,
        label_source="local_p90_p60",
    )
    key_contaminated.provenance = {"haller_gt_test": {"used": True}}
    with pytest.raises(ValueError, match=r"test|haller_gt_test|Haller"):
        contract.validate_training_batch(key_contaminated, contract.MODE_W1_P)

    array_contaminated = _supervision_batch(
        mode=contract.MODE_W1_P,
        label_source="local_p90_p60",
    )
    array_contaminated.provenance = {
        "source": np.asarray("haller_gt_test"),
    }
    with pytest.raises(ValueError, match=r"test|haller_gt_test|Haller"):
        contract.validate_training_batch(array_contaminated, contract.MODE_W1_P)

    for location in ("sampling_source", "provenance"):
        contaminated = _supervision_batch(
            mode=contract.MODE_W1_P,
            label_source="local_p90_p60",
        )
        if location == "sampling_source":
            contaminated.sampling_source = "haller_gt_calibration"
        else:
            contaminated.provenance["calibration"] = "haller_gt_calibration"
        with pytest.raises(ValueError, match=r"calibration|Haller|禁止"):
            contract.validate_training_batch(contaminated, contract.MODE_W1_P)


def test_formal_w1_training_source_cannot_fallback_to_legacy_p85():
    """W1 formal loss source 只能是 p90/p60 或 Haller train anchor。"""
    batch = _supervision_batch(
        mode=contract.MODE_W1_P,
        label_source="legacy_p85",
    )

    with pytest.raises(ValueError, match=r"formal|source|legacy_p85"):
        contract.validate_training_batch(batch, contract.MODE_W1_P)


def test_calibration_batch_requires_calibration_haller_source():
    """calibration seam 只读取 calibration Haller GT。"""
    batch = _supervision_batch(
        mode=contract.MODE_W1_H,
        label_source="haller_gt_calibration",
    )
    batch.split_name = "calibration"

    assert contract.validate_calibration_batch(batch, contract.MODE_W1_H) is batch


def test_evaluation_requires_test_split_and_explicit_test_haller_source():
    """evaluation seam 只接受 test split 上显式的 haller_gt_test。"""
    batch = _supervision_batch(mode=contract.MODE_W1_H,
                               label_source="haller_gt_test")
    batch.split_name = "test"
    validated = contract.validate_evaluation_batch(batch, contract.MODE_W1_H)
    assert validated.label_source == "haller_gt_test"

    bad = _supervision_batch(mode=contract.MODE_W1_H,
                             label_source="haller_gt_calibration")
    bad.split_name = "test"
    with pytest.raises(ValueError, match=r"haller_gt_test|evaluation|test"):
        contract.validate_evaluation_batch(bad, contract.MODE_W1_H)


def test_mode_aware_loss_receives_contract_batch():
    """loss adapter 先验证 mode/batch contract，再调用注入的 criterion。"""
    batch = contract.WeakSupervisionBatch(
        pathlines=torch.zeros(1, 1, 1, 7),
        labels=torch.tensor([[1.0]]),
        label_source="legacy_p85",
        split_name="train",
        mode=contract.MODE_B0,
    )
    loss = contract.build_loss_for_mode(contract.MODE_B0, torch.nn.BCELoss())
    predictions = torch.full_like(batch.labels, 0.5)

    value = loss(predictions, batch)

    assert value.ndim == 0
    assert bool(torch.isfinite(value))


def test_mode_aware_loss_rejects_unknown_mask_for_label_only_criterion():
    """普通 label-only BCE 不能把 unknown 静默当作负样本。"""
    batch = _supervision_batch(
        mode=contract.MODE_W1_P,
        label_source="local_p90_p60",
    )
    loss = contract.build_loss_for_mode(contract.MODE_W1_P, torch.nn.BCELoss())

    with pytest.raises(ValueError, match=r"unknown|mask|未标注"):
        loss(torch.full_like(batch.labels, 0.5), batch)


def test_mode_aware_loss_rejects_non_boolean_batch_aware_declaration():
    """criterion 能力声明不能靠字符串 truthiness 绕过 unknown guard。"""
    class _BadCriterion:
        accepts_weak_supervision_batch = "false"

        def __call__(self, predictions, labels):
            return predictions.mean()

    batch = _supervision_batch(
        mode=contract.MODE_W1_P,
        label_source="local_p90_p60",
    )
    loss = contract.build_loss_for_mode(contract.MODE_W1_P, _BadCriterion())

    with pytest.raises(ValueError, match=r"bool|accepts_weak_supervision_batch"):
        loss(torch.full_like(batch.labels, 0.5), batch)


def _checkpoint_kwargs(mode=contract.MODE_B0, **overrides):
    """所有 checkpoint seam 测试共用的最小显式 contract。"""
    default_source = {
        contract.MODE_B0: "legacy_p85",
        contract.MODE_B1: "legacy_p85",
        contract.MODE_W1: "local_p90_p60",
        contract.MODE_W1_P: "local_p90_p60",
        contract.MODE_W1_H: "haller_anchor_train",
        contract.MODE_W2: "haller_anchor_train",
        contract.MODE_W3: "haller_anchor_train",
    }[contract.canonical_mode(mode)]
    values = {
        "mode": mode,
        "feature_schema": contract.feature_schema_for_mode(mode),
        "dataset_config": {"dataset_name": "fixture"},
        "split_config": {"split_name": "train", "frame_range": [0, 8]},
        "sampling_config": {"t_win": 4, "window_step": 2},
        "anchor_hash": None,
        "calibration_policy": {"source": "none"},
        "label_source": default_source,
        "sampling_source": "legacy_p85",
        "seed": 7,
    }
    values.update(overrides)
    return values


def test_student_only_checkpoint_roundtrip_restores_contract_and_rng(tmp_path):
    """student-only checkpoint 能恢复输出、optimizer/scheduler、元数据和 RNG。"""
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    x = torch.tensor([[1.0, -2.0, 0.5]])
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_output = model(x).detach().clone()
    expected_rng = contract.capture_rng_state()
    checkpoint = tmp_path / "student_only.pth"

    contract.save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        epoch=3,
        global_step=12,
        metrics={"loss": 0.25},
        rng_state=expected_rng,
        **_checkpoint_kwargs(),
    )

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for field in (
        "format_version", "mode", "feature_schema", "dataset_config",
        "split_config", "sampling_config", "student", "teacher",
        "projection_head", "optimizer", "scheduler", "epoch",
        "global_step", "metrics", "seed", "rng_state", "anchor_hash",
        "calibration_policy", "warm_start_aux",
    ):
        assert field in blob
    assert blob["mode"] == contract.MODE_B0
    assert blob["feature_schema"] == contract.FEATURE_SCHEMA_7.as_dict()
    assert blob["label_source"] == "legacy_p85"
    assert blob["sampling_source"] == "legacy_p85"
    assert blob["warm_start_aux"] is False

    restored = nn.Linear(3, 2)
    optimizer_restored = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    scheduler_restored = torch.optim.lr_scheduler.StepLR(
        optimizer_restored, step_size=2, gamma=0.5)
    result = contract.load_checkpoint(
        checkpoint,
        restored,
        optimizer_restored,
        scheduler_restored,
        expected_mode=contract.MODE_B0,
        expected_feature_schema=contract.FEATURE_SCHEMA_7,
        expected_split_config={"split_name": "train", "frame_range": [0, 8]},
    )

    assert result["epoch"] == 3
    assert result["start_epoch"] == 4
    assert result["global_step"] == 12
    assert result["metrics"] == {"loss": 0.25}
    assert torch.equal(restored(x), expected_output)
    assert optimizer_restored.state_dict()["state"]
    assert scheduler_restored.state_dict() == scheduler.state_dict()

    expected_random = (random.random(), float(np.random.rand()), torch.rand(3))
    # load_checkpoint 已恢复 save 时的 RNG state；重新读出的序列必须相同。
    contract.load_checkpoint(
        checkpoint,
        restored,
        expected_mode=contract.MODE_B0,
        expected_split="train",
        load_mode="inference",
        restore_rng=True,
    )
    actual_random = (random.random(), float(np.random.rand()), torch.rand(3))
    assert actual_random[0] == expected_random[0]
    assert actual_random[1] == expected_random[1]
    assert torch.equal(actual_random[2], expected_random[2])


def test_cpu_load_reports_cuda_rng_degradation_explicitly(tmp_path):
    """CPU 可恢复通用 RNG；CUDA RNG 不可用时显式报告而非伪造恢复。"""
    checkpoint = tmp_path / "cuda-rng.pth"
    rng_state = contract.capture_rng_state()
    rng_state["torch_cuda"] = [torch.zeros(4, dtype=torch.uint8)]
    model = nn.Linear(3, 2)
    contract.save_checkpoint(
        checkpoint,
        model,
        rng_state=rng_state,
        **_checkpoint_kwargs(),
    )

    result = contract.load_checkpoint(
        checkpoint,
        nn.Linear(3, 2),
        expected_mode=contract.MODE_B0,
        expected_split="train",
        restore_rng=True,
        load_mode="inference",
    )

    assert result["rng_restored"] is True
    assert result["cuda_rng_restored"] is False
    with pytest.raises(ValueError, match=r"CUDA|cuda|rng"):
        contract.load_checkpoint(
            checkpoint,
            expected_mode=contract.MODE_B0,
            expected_split="train",
            restore_rng=True,
            strict_cuda_rng=True,
            load_mode="inference",
        )


def test_checkpoint_unknown_format_is_not_treated_as_legacy_b0(tmp_path):
    """未来/未知格式必须拒绝，不能借 auxiliary 标志绕过新 contract。"""
    model = nn.Linear(3, 2)
    checkpoint = tmp_path / "unknown-format.pth"
    torch.save({"format_version": "weak-supervision-checkpoint-v2",
                "model": model.state_dict()}, checkpoint)

    with pytest.raises(ValueError, match=r"format_version|格式|unsupported"):
        contract.load_checkpoint(
            checkpoint,
            expected_mode=contract.MODE_B0,
            warm_start_aux=True,
            restore_rng=False,
        )


def _save_w3_checkpoint(path):
    """保存带 teacher/projection 的 W3 contract fixture。"""
    student = nn.Linear(3, 2)
    teacher = nn.Linear(3, 2)
    projection = nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projection.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    loss = projection(student(torch.ones(1, 3))).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    contract.save_checkpoint(
        path,
        student,
        optimizer,
        scheduler,
        teacher=teacher,
        projection_head=projection,
        epoch=5,
        global_step=19,
        **_checkpoint_kwargs(
            contract.MODE_W3,
            feature_schema=contract.FEATURE_SCHEMA_7,
            anchor_hash="anchor-hash-v1",
        ),
    )
    return student, teacher, projection, optimizer, scheduler


def test_teacher_projection_checkpoint_roundtrip(tmp_path):
    """W3 checkpoint 能往返 student、EMA teacher 和 projection head。"""
    checkpoint = tmp_path / "w3.pth"
    student, teacher, projection, optimizer, scheduler = _save_w3_checkpoint(checkpoint)
    restored_student = nn.Linear(3, 2)
    restored_teacher = nn.Linear(3, 2)
    restored_projection = nn.Linear(2, 4)
    restored_optimizer = torch.optim.AdamW(
        list(restored_student.parameters()) + list(restored_projection.parameters()), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=2)

    result = contract.load_checkpoint(
        checkpoint,
        restored_student,
        restored_optimizer,
        restored_scheduler,
        teacher=restored_teacher,
        projection_head=restored_projection,
        expected_mode=contract.MODE_W3,
        expected_split="train",
        expected_anchor_hash="anchor-hash-v1",
        restore_rng=True,
    )

    assert result["mode"] == contract.MODE_W3
    assert result["feature_schema"] == contract.FEATURE_SCHEMA_7.as_dict()
    for left, right in zip(student.parameters(), restored_student.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(teacher.parameters(), restored_teacher.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(projection.parameters(), restored_projection.parameters()):
        assert torch.equal(left, right)
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
    _assert_nested_equal(restored_scheduler.state_dict(), scheduler.state_dict())

    expected_w3_random = (random.random(), float(np.random.rand()), torch.rand(3))
    contract.load_checkpoint(
        checkpoint,
        restored_student,
        restored_optimizer,
        restored_scheduler,
        teacher=restored_teacher,
        projection_head=restored_projection,
        expected_mode=contract.MODE_W3,
        expected_split="train",
        expected_anchor_hash="anchor-hash-v1",
        restore_rng=True,
    )
    actual_w3_random = (random.random(), float(np.random.rand()), torch.rand(3))
    assert actual_w3_random[0] == expected_w3_random[0]
    assert actual_w3_random[1] == expected_w3_random[1]
    assert torch.equal(actual_w3_random[2], expected_w3_random[2])


def test_resume_rejects_partial_teacher_projection_or_optimizer_state(tmp_path):
    """resume 不能静默丢失 checkpoint 中的训练状态；inference 必须显式声明。"""
    checkpoint = tmp_path / "w3-partial.pth"
    _save_w3_checkpoint(checkpoint)

    with pytest.raises(ValueError, match=r"teacher|EMA|resume"):
        contract.load_checkpoint(
            checkpoint,
            nn.Linear(3, 2),
            expected_mode=contract.MODE_W3,
            expected_split="train",
            expected_anchor_hash="anchor-hash-v1",
            restore_rng=False,
        )

    restored = nn.Linear(3, 2)
    result = contract.load_checkpoint(
        checkpoint,
        restored,
        expected_mode=contract.MODE_W3,
        expected_split="train",
        expected_anchor_hash="anchor-hash-v1",
        load_mode="inference",
    )

    assert result["load_mode"] == "inference"
    assert result["rng_restored"] is False


def test_teacher_only_checkpoint_roundtrip(tmp_path):
    """W1-H checkpoint 能恢复 student 与 EMA teacher，但不强制 projection head。"""
    checkpoint = tmp_path / "teacher-only.pth"
    student = nn.Linear(3, 2)
    teacher = nn.Linear(3, 2)
    contract.save_checkpoint(
        checkpoint,
        student,
        teacher=teacher,
        **_checkpoint_kwargs(contract.MODE_W1_H, anchor_hash="anchor-hash-v1"),
    )

    restored_student = nn.Linear(3, 2)
    restored_teacher = nn.Linear(3, 2)
    result = contract.load_checkpoint(
        checkpoint,
        restored_student,
        ema_teacher=restored_teacher,
        expected_mode=contract.MODE_W1_H,
        expected_split="train",
        expected_anchor_hash="anchor-hash-v1",
        restore_rng=False,
        load_mode="inference",
    )

    assert result["teacher"] is not None
    for left, right in zip(student.parameters(), restored_student.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(teacher.parameters(), restored_teacher.parameters()):
        assert torch.equal(left, right)


def test_checkpoint_rejects_implicit_split_window_or_label_source(tmp_path):
    """checkpoint 不能靠旧默认值补齐 split、窗口或监督来源。"""
    model = nn.Linear(3, 2)
    checkpoint = tmp_path / "invalid-contract.pth"

    with pytest.raises(ValueError, match=r"split"):
        params = _checkpoint_kwargs(label_source="legacy_p85")
        params.update(split_config={}, sampling_config={})
        contract.save_checkpoint(checkpoint, model, **params)
    with pytest.raises(ValueError, match=r"t_win|window|sampling"):
        params = _checkpoint_kwargs(sampling_config={})
        params["split_config"] = {"split_name": "train"}
        contract.save_checkpoint(checkpoint, model, **params)
    with pytest.raises(ValueError, match=r"label.source|source"):
        contract.save_checkpoint(
            checkpoint,
            model,
            **_checkpoint_kwargs(label_source=None),
        )


def test_checkpoint_rejects_test_haller_source_in_calibration_policy(tmp_path):
    """calibration policy 不能借字段名把 test Haller GT 带进训练 checkpoint。"""
    model = nn.Linear(3, 2)

    with pytest.raises(ValueError, match=r"test|haller_gt_test|calibration"):
        contract.save_checkpoint(
            tmp_path / "test-policy.pth",
            model,
            **_checkpoint_kwargs(
                calibration_policy={"threshold_source": "haller_gt_test"}
            ),
        )

    with pytest.raises(ValueError, match=r"test|haller_gt_test|metadata"):
        contract.save_checkpoint(
            tmp_path / "test-extra-metadata.pth",
            model,
            **_checkpoint_kwargs(
                extra_metadata={"haller_gt_test": {"used": True}}
            ),
        )


def test_checkpoint_requires_dataset_config_and_seed(tmp_path):
    """checkpoint 必须能定位数据语义并记录可复现 seed。"""
    model = nn.Linear(3, 2)

    with pytest.raises(ValueError, match=r"dataset|数据集"):
        contract.save_checkpoint(
            tmp_path / "missing-dataset.pth",
            model,
            **_checkpoint_kwargs(dataset_config={}),
        )
    with pytest.raises(ValueError, match=r"seed"):
        contract.save_checkpoint(
            tmp_path / "missing-seed.pth",
            model,
            **_checkpoint_kwargs(seed=None),
        )
    with pytest.raises(ValueError, match=r"source|split|Haller"):
        params = _checkpoint_kwargs(
            contract.MODE_W3,
            anchor_hash="anchor-hash-v1",
        )
        params["split_config"] = {"split_name": "calibration"}
        contract.save_checkpoint(
            tmp_path / "source-split.pth",
            model,
            teacher=nn.Linear(3, 2),
            projection_head=nn.Linear(2, 4),
            **params,
        )

    with pytest.raises(ValueError, match=r"split|train"):
        params = _checkpoint_kwargs()
        params["split_config"] = {"split_name": "test"}
        contract.save_checkpoint(tmp_path / "legacy-test-split.pth", model, **params)

    with pytest.raises(ValueError, match=r"rng_state|RNG"):
        contract.save_checkpoint(
            tmp_path / "missing-rng-state.pth",
            model,
            rng_state={},
            **_checkpoint_kwargs(),
        )


def test_resume_rejects_checkpoint_without_optimizer_or_scheduler(tmp_path):
    """resume 不能用新建 optimizer/scheduler 静默替代缺失的训练状态。"""
    checkpoint = tmp_path / "student-only-no-optimizer.pth"
    contract.save_checkpoint(
        checkpoint,
        nn.Linear(3, 2),
        **_checkpoint_kwargs(),
    )

    with pytest.raises(ValueError, match=r"optimizer|scheduler|resume"):
        contract.load_checkpoint(
            checkpoint,
            nn.Linear(3, 2),
            expected_mode=contract.MODE_B0,
            expected_split="train",
            restore_rng=False,
        )


@pytest.mark.parametrize("field, value", [("epoch", 1.5),
                                           ("global_step", "3")])
def test_checkpoint_rejects_non_integral_progress_fields(tmp_path, field, value):
    """epoch/global_step 不能由 int() 静默截断或接受字符串。"""
    params = _checkpoint_kwargs()
    params[field] = value

    with pytest.raises(ValueError, match=r"epoch|global_step|integer|整数"):
        contract.save_checkpoint(tmp_path / f"bad-{field}.pth", nn.Linear(3, 2), **params)


@pytest.mark.parametrize("field", ["teacher", "projection_head"])
def test_checkpoint_metadata_rejects_missing_required_w3_state(tmp_path, field):
    """metadata-only 读取与实际加载共享 W3 teacher/projection 完整性 guard。"""
    checkpoint = tmp_path / f"missing-{field}.pth"
    _save_w3_checkpoint(checkpoint)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if field == "teacher":
        blob["teacher"] = None
        blob["ema_teacher"] = None
    else:
        blob[field] = None
    torch.save(blob, checkpoint)

    with pytest.raises(ValueError, match=r"teacher|EMA|projection|W3"):
        contract.checkpoint_metadata(checkpoint)


@pytest.mark.parametrize("field, value", [("split", "test"), ("t_win", 99)])
def test_checkpoint_metadata_rejects_inconsistent_top_level_aliases(tmp_path, field, value):
    """顶层 split/t_win 别名必须与嵌套 contract 一致。"""
    checkpoint = tmp_path / f"inconsistent-{field}.pth"
    contract.save_checkpoint(checkpoint, nn.Linear(3, 2), **_checkpoint_kwargs())
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    blob[field] = value
    torch.save(blob, checkpoint)

    with pytest.raises(ValueError, match=r"split|t_win|sampling"):
        contract.checkpoint_metadata(checkpoint)


def test_checkpoint_rejects_mode_mismatched_components(tmp_path):
    """mode-aware component 声明与 checkpoint mode 不一致时立即拒绝。"""
    with pytest.raises(ValueError, match=r"student|mode"):
        contract.save_checkpoint(
            tmp_path / "mismatched-save.pth",
            _ModeLinear(contract.MODE_B0),
            teacher=_ModeLinear(contract.MODE_W3),
            projection_head=nn.Linear(2, 4),
            **_checkpoint_kwargs(contract.MODE_W3, anchor_hash="anchor-hash-v1"),
        )

    checkpoint = tmp_path / "mismatched-load.pth"
    _save_w3_checkpoint(checkpoint)
    with pytest.raises(ValueError, match=r"teacher|mode"):
        contract.load_checkpoint(
            checkpoint,
            nn.Linear(3, 2),
            teacher=_ModeLinear(contract.MODE_B0),
            projection_head=nn.Linear(2, 4),
            expected_mode=contract.MODE_W3,
            expected_split="train",
            expected_anchor_hash="anchor-hash-v1",
            restore_rng=False,
        )


def test_checkpoint_rejects_incompatible_mode_schema_split_and_anchor_hash(tmp_path):
    """checkpoint 的四类研究语义 drift 都必须 fail loudly。"""
    checkpoint = tmp_path / "w3_contract.pth"
    _save_w3_checkpoint(checkpoint)
    cases = [
        ({"expected_mode": contract.MODE_B0}, r"mode"),
        ({"expected_feature_schema": contract.FEATURE_SCHEMA_6}, r"feature schema|channel"),
        ({"expected_split": "calibration"}, r"split"),
        ({"expected_anchor_hash": "different-hash"}, r"anchor_hash"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            load_kwargs = {
                "expected_mode": contract.MODE_W3,
                "expected_split": "train",
                "expected_anchor_hash": "anchor-hash-v1",
                "restore_rng": False,
            }
            load_kwargs.update(kwargs)
            contract.load_checkpoint(
                checkpoint,
                **load_kwargs,
            )


def test_checkpoint_infers_mode_from_mode_aware_target(tmp_path):
    """mode-aware target 未显式传 expected_mode 时也不能接收错误 mode。"""
    checkpoint = tmp_path / "w3-target-mode.pth"
    _save_w3_checkpoint(checkpoint)

    with pytest.raises(ValueError, match=r"mode"):
        contract.load_checkpoint(
            checkpoint,
            _ModeLinear(contract.MODE_B0),
            expected_split="train",
            expected_anchor_hash="anchor-hash-v1",
            restore_rng=False,
        )


def test_checkpoint_load_requires_mode_and_split_expectations(tmp_path):
    """新 checkpoint 恢复必须声明要恢复的 mode 和 split 语义。"""
    checkpoint = tmp_path / "required-expectations.pth"
    contract.save_checkpoint(
        checkpoint,
        nn.Linear(3, 2),
        **_checkpoint_kwargs(),
    )

    with pytest.raises(ValueError, match=r"expected_mode|mode"):
        contract.load_checkpoint(
            checkpoint,
            expected_split="train",
            restore_rng=False,
        )
    with pytest.raises(ValueError, match=r"expected_split|split"):
        contract.load_checkpoint(
            checkpoint,
            expected_mode=contract.MODE_B0,
            restore_rng=False,
        )


def test_legacy_b0_checkpoint_requires_explicit_auxiliary_load(tmp_path):
    """旧 B0 blob 不能被新主实验默认吸收，只有显式 auxiliary 才能加载。"""
    model = nn.Linear(3, 2)
    legacy = tmp_path / "legacy_b0.pth"
    torch.save({"model": model.state_dict(), "epoch": 8}, legacy)
    restored = nn.Linear(3, 2)

    with pytest.raises(ValueError, match=r"legacy B0|warm_start_aux|auxiliary"):
        contract.load_checkpoint(legacy, restored, expected_mode=contract.MODE_B0,
                                 restore_rng=False)
    with pytest.raises(ValueError, match=r"legacy B0|warm_start_aux|auxiliary"):
        contract.load_checkpoint(legacy, restored, restore_rng=False,
                                 warm_start_aux=True)
    with pytest.raises(ValueError, match=r"bool|warm_start_aux"):
        contract.load_checkpoint(legacy, restored, expected_mode=contract.MODE_B0,
                                 restore_rng=False, warm_start_aux="false")
    with pytest.raises(ValueError, match=r"student|mode|B0"):
        contract.load_checkpoint(
            legacy,
            _ModeLinear(contract.MODE_B1),
            expected_mode=contract.MODE_B0,
            warm_start_aux=True,
            restore_rng=False,
        )

    result = contract.load_checkpoint(
        legacy,
        restored,
        expected_mode=contract.MODE_B0,
        warm_start_aux=True,
        restore_rng=False,
    )
    assert result["legacy"] is True
    assert result["warm_start_aux"] is True
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)


def test_legacy_b0_checkpoint_rejects_unverifiable_contract_expectations(tmp_path):
    """旧 checkpoint 不得静默忽略新 split/source/hash 约束。"""
    model = nn.Linear(3, 2)
    legacy = tmp_path / "legacy-b0-contract.pth"
    torch.save({"model": model.state_dict(), "epoch": 2}, legacy)

    with pytest.raises(ValueError, match=r"legacy|verify|校验|contract"):
        contract.load_checkpoint(
            legacy,
            expected_mode=contract.MODE_B0,
            expected_split="calibration",
            warm_start_aux=True,
            restore_rng=False,
        )


def test_training_entrypoint_exposes_explicit_mode_dispatch():
    """训练入口提供公共 mode-aware model/loss adapter，旧默认路径保持独立。"""
    import train_kaggle

    model = train_kaggle.build_model_from_config(
        {"model": {"encoder_args": _small_encoder_config(contract.MODE_B1)}},
        mode=contract.MODE_B1,
    )
    loss = train_kaggle.build_criterion_from_config(
        {"model": {"criterion_args": {"NAME": "BCELoss"}}},
        mode=contract.MODE_B0,
    )

    assert isinstance(model, contract.ChannelSelectingAdapter)
    assert model.mode == contract.MODE_B1
    assert isinstance(loss, contract.ModeAwareLoss)
    assert loss.mode == contract.MODE_B0
    assert train_kaggle.save_contract_checkpoint is contract.save_checkpoint
    assert train_kaggle.load_contract_checkpoint is contract.load_checkpoint
