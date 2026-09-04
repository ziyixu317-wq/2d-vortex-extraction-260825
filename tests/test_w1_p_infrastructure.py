"""05 票：W1-P p90/p60/unknown teacher infrastructure 的行为测试。"""

import numpy as np
import pytest


def test_train_only_p90_p60_targets_keep_middle_and_nontrain_unknown():
    """p90 positive / p60 negative；中间值与非 train 帧保持 unknown。"""
    from w1_p import build_w1_p_target_field, compute_w1_p_thresholds

    # static solid mask removes values 9 and 19 from the two train frames:
    # remaining fluid values are 0..8 and 10..18, so p60=11.2、p90=16.3。
    ivd = np.arange(40, dtype=np.float32).reshape(4, 2, 5)
    ivd[2:] = 42.0
    solid = np.zeros((2, 5), dtype=bool)
    solid[1, 4] = True  # train value 19 is excluded from percentile statistics.

    thresholds = compute_w1_p_thresholds(
        ivd, solid, train_frame_range=(0, 2), dataset_name="fixture")
    assert thresholds.negative == pytest.approx(11.2)
    assert thresholds.positive == pytest.approx(16.3)
    assert thresholds.source_split == "train"

    targets = build_w1_p_target_field(
        ivd, solid, thresholds, train_frame_range=(0, 2),
        min_area=1, dataset_name="fixture")

    assert targets.anchor_state[1, 1].tolist() == [-1, -1, 1, 1, -1]
    assert targets.anchor_state[1, 0].tolist() == [0, 0, -1, -1, -1]
    assert targets.solid_mask[1, 1, 4]
    assert targets.anchor_state[1, 1, 4] == -1
    assert np.all(targets.anchor_state[2:] == -1)
    assert targets.metadata["label_source"] == "local_p90_p60"
    assert targets.metadata["split_name"] == "train"


def test_w1_p_loss_masks_unknown_and_solid_and_accepts_only_confident_pseudo():
    """anchor BCE 只看 known；pseudo/consistency 不消费 solid 或不确定 teacher。"""
    import torch

    from w1_p import W1PConfig, build_w1_p_batch, compute_w1_p_loss

    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
    label_mask = torch.tensor([[1, 1, 0, 0],
                               [1, 1, 0, 0]], dtype=torch.bool)
    unknown_mask = ~label_mask
    solid_mask = torch.tensor([[0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=torch.bool)
    batch = build_w1_p_batch(
        torch.zeros(2, 3, 4, 7),
        labels,
        label_mask,
        unknown_mask,
        solid_mask,
        sampling_source="legacy_p85",
        split_name="train",
    )

    student = torch.tensor([[0.8, 0.2, 0.7, 0.9],
                            [0.1, 0.6, 0.4, 0.8]], requires_grad=True)
    teacher = torch.tensor([[0.8, 0.2, 0.95, 0.95],
                            [0.1, 0.9, 0.5, 0.2]])
    loss, stats = compute_w1_p_loss(
        student,
        teacher,
        batch,
        config=W1PConfig(),
        epoch=12,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert stats["anchor_count"] == 4
    assert stats["unknown_count"] == 4
    assert stats["solid_count"] == 2
    assert stats["pseudo_eligible_count"] == 2
    assert stats["pseudo_accepted_count"] == 1
    assert stats["pseudo_positive_count"] == 1
    assert stats["pseudo_negative_count"] == 0
    assert stats["ramp_weight"] == pytest.approx(1.0)
    assert stats["sampling_source"] == "legacy_p85"
    assert stats["loss_source"] == "local_p90_p60"
    loss.backward()
    assert student.grad is not None


def _tiny_w1_p_model():
    """返回一个保持现有 (dummy, pathlines) 输入 seam 的 CPU fixture model。"""
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(7, 1)

        def forward(self, data):
            _dummy, pathlines = data
            logits = self.projection(pathlines.mean(dim=1)).squeeze(-1)
            return torch.sigmoid(logits)

    return TinyModel()


def _w1_p_smoke_batch(source="legacy_p85"):
    import torch

    from w1_p import build_w1_p_batch

    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
    label_mask = torch.tensor([[1, 1, 0, 0],
                               [1, 1, 0, 0]], dtype=torch.bool)
    solid_mask = torch.tensor([[0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=torch.bool)
    return build_w1_p_batch(
        torch.randn(2, 3, 4, 7), labels, label_mask, ~label_mask, solid_mask,
        sampling_source=source, split_name="train",
    )


def test_ema_teacher_updates_after_student_optimizer_step_and_ramp_is_linear():
    """student/EMA 初始相同；optimizer 后 teacher 按 0.99 更新，ramp 不越界。"""
    import torch

    from w1_p import W1PConfig, W1PTrainer, ramp_up_weight

    torch.manual_seed(17)
    student = _tiny_w1_p_model()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.25)
    trainer = W1PTrainer(
        student, optimizer, config=W1PConfig(), sampling_source="legacy_p85",
        seed=17,
    )
    before_student = {
        name: value.detach().clone()
        for name, value in student.state_dict().items()
    }
    before_teacher = {
        name: value.detach().clone()
        for name, value in trainer.teacher.state_dict().items()
    }

    stats = trainer.train_step(_w1_p_smoke_batch(), epoch=0, device="cpu")
    assert stats["ramp_weight"] == pytest.approx(0.0)
    assert ramp_up_weight(0) == pytest.approx(0.0)
    assert ramp_up_weight(6) == pytest.approx(0.5)
    assert ramp_up_weight(12) == pytest.approx(1.0)
    assert any(
        not torch.equal(before_student[name], value)
        for name, value in student.state_dict().items()
    )
    for name, teacher_value in trainer.teacher.state_dict().items():
        expected = (before_teacher[name] * 0.99
                    + student.state_dict()[name].detach() * 0.01)
        assert torch.allclose(teacher_value, expected, atol=1e-6)
        assert teacher_value.data_ptr() != student.state_dict()[name].data_ptr()
    assert all(not parameter.requires_grad
               for parameter in trainer.teacher.parameters())


def test_w1_p_cpu_smoke_runs_five_epochs_with_finite_logs():
    """CPU synthetic smoke 覆盖五个 epoch 的 student/teacher training seam。"""
    import torch

    from w1_p import W1PConfig, W1PTrainer

    torch.manual_seed(23)
    student = _tiny_w1_p_model()
    trainer = W1PTrainer(
        student, torch.optim.AdamW(student.parameters(), lr=0.01),
        config=W1PConfig(), sampling_source="legacy_p85", seed=23,
    )
    logs = [
        trainer.run_epoch([_w1_p_smoke_batch()], epoch=epoch, device="cpu")
        for epoch in range(5)
    ]
    assert len(logs) == 5
    assert trainer.global_step == 5
    assert logs[0]["ramp_weight"] == pytest.approx(0.0)
    assert logs[-1]["ramp_weight"] == pytest.approx(4.0 / 12.0)
    assert all(np.isfinite(log["loss"]) for log in logs)


def test_w1_p_can_place_student_and_teacher_on_distinct_gpus():
    """Two-card W1-P keeps one student/teacher replica per GPU."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    from w1_p import W1PConfig, W1PTrainer

    student = _tiny_w1_p_model().to("cuda:0")
    trainer = W1PTrainer(
        student,
        torch.optim.SGD(student.parameters(), lr=0.01),
        config=W1PConfig(),
        sampling_source="legacy_p85",
        teacher_device="cuda:1",
    )
    stats = trainer.train_step(_w1_p_smoke_batch(), epoch=1, device="cuda:0")

    assert np.isfinite(stats["loss"])
    assert next(trainer.student.parameters()).device == torch.device("cuda:0")
    assert next(trainer.teacher.parameters()).device == torch.device("cuda:1")


def test_w1_p_checkpoint_round_trip_preserves_teacher_optimizer_and_contract(tmp_path):
    """checkpoint round-trip 恢复 student、EMA、optimizer/scheduler 与 provenance。"""
    import torch

    import weak_supervision_contract as contract
    from w1_p import W1PConfig, W1PTrainer

    torch.manual_seed(31)
    dataset_config = {"dataset_name": "fixture", "normalization": "train_only"}
    split_config = {"split_name": "train", "frame_range": [0, 2]}
    sampling_config = {"t_win": 3, "window_step": 1}
    student = _tiny_w1_p_model()
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    trainer = W1PTrainer(
        student, optimizer, scheduler=scheduler, config=W1PConfig(),
        sampling_source="legacy_p85", seed=31,
    )
    stats = trainer.train_step(_w1_p_smoke_batch(), epoch=1, device="cpu")
    checkpoint_path = trainer.save_checkpoint(
        tmp_path / "w1p.pt", epoch=1, metrics=stats,
        dataset_config=dataset_config, split_config=split_config,
        sampling_config=sampling_config,
    )
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert blob["label_source"] == contract.LABEL_SOURCE_LOCAL_P90_P60
    assert blob["sampling_source"] == "legacy_p85"
    assert blob["extra_metadata"]["formal_loss_source"] == "local_p90_p60"
    assert blob["extra_metadata"]["w1_p_config"]["label_source"] == "local_p90_p60"
    assert "legacy_p85" not in blob["extra_metadata"]["w1_p_config"]

    restored_student = _tiny_w1_p_model()
    restored_optimizer = torch.optim.AdamW(restored_student.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=2)
    restored = W1PTrainer(
        restored_student, restored_optimizer, scheduler=restored_scheduler,
        config=W1PConfig(), sampling_source="legacy_p85", seed=999,
    )
    result = restored.load_checkpoint(
        checkpoint_path, device="cpu", expected_dataset_config=dataset_config,
        expected_split_config=split_config, expected_sampling_config=sampling_config,
    )
    assert result["epoch"] == 1
    assert result["global_step"] == 1
    assert restored.global_step == 1
    assert restored.seed == 31
    for name, value in trainer.student.state_dict().items():
        assert torch.equal(value, restored.student.state_dict()[name])
    for name, value in trainer.teacher.state_dict().items():
        assert torch.equal(value, restored.teacher.state_dict()[name])
    original_optimizer_state = trainer.optimizer.state_dict()
    restored_optimizer_state = restored.optimizer.state_dict()
    assert restored_optimizer_state["param_groups"] == original_optimizer_state["param_groups"]
    assert restored_optimizer_state["state"].keys() == original_optimizer_state["state"].keys()
    for parameter_id, original_state in original_optimizer_state["state"].items():
        for state_name, original_value in original_state.items():
            restored_value = restored_optimizer_state["state"][parameter_id][state_name]
            if isinstance(original_value, torch.Tensor):
                assert torch.equal(restored_value, original_value)
            else:
                assert restored_value == original_value
    assert restored.scheduler.state_dict() == trainer.scheduler.state_dict()


def test_w1_p_rejects_nontrain_thresholds_and_haller_test_sampling():
    """split/source guard 必须 fail loudly，不能把 test GT 混入 W1-P train。"""
    from w1_p import compute_w1_p_thresholds

    ivd = np.ones((3, 2, 2), dtype=np.float32)
    solid = np.zeros((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="split=train"):
        compute_w1_p_thresholds(
            ivd, solid, train_frame_range=(0, 2), split_name="test")
    with pytest.raises(ValueError, match="haller_gt_test"):
        _w1_p_smoke_batch(source="haller_gt_test")
