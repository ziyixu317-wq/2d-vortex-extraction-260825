"""WS-9 real-pilot runner 的小型 contract seam 测试。"""

import pytest
import torch

import run_ws9_pilot


def test_two_step_pilot_scheduler_round_trip_preserves_epoch_and_lr():
    """pilot scheduler 的 warmup 边界和 checkpoint state 必须可恢复。"""
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = run_ws9_pilot.TwoStepPilotScheduler(
        optimizer, lr=1e-4, second_lr=5e-6, warmup_epochs=3
    )

    assert scheduler.step(1) == pytest.approx(1e-4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    state = scheduler.state_dict()
    assert scheduler.step(3) == pytest.approx(5e-6)

    restored_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=9.0)
    restored = run_ws9_pilot.TwoStepPilotScheduler(
        restored_optimizer, lr=1e-4, second_lr=5e-6, warmup_epochs=3
    )
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_real_pilot_artifact_gate_rejects_non_six_dataset_request():
    """runner 不能绕过固定六数据集 artifact/leakage gate。"""
    with pytest.raises(ValueError, match="固定六个数据集"):
        run_ws9_pilot.validate_prepared_artifacts(
            "missing-weak-root",
            "missing-haller-root",
            dataset_names=("boussinesq",),
        )
