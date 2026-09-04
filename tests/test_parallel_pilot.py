"""Multi-GPU pilot progress-checkpoint seam tests."""

from types import SimpleNamespace

import torch

import e2e_weak_supervision as e2e
import parallel_pilot
import weak_supervision_contract as contract


def _progress_fixture():
    student = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    batch = torch.ones((1, 2))
    loss = student(batch).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    trainer = SimpleNamespace(
        student=student,
        teacher=None,
        projection_head=None,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=1,
        seed=0,
        last_metrics={"mode": contract.MODE_B0, "loss": 1.0},
        anchor_hash=None,
        anchor_metadata=None,
        calibration_selection=None,
    )
    method = SimpleNamespace(mode=contract.MODE_B0, trainer=trainer)
    context = {
        "format_version": parallel_pilot.TRAINING_PROGRESS_FORMAT,
        "mode": contract.MODE_B0,
        "device_group": ["cpu"],
        "seed": 0,
    }
    return method, context


def test_training_progress_checkpoint_round_trip(tmp_path):
    method, context = _progress_fixture()
    path = (
        tmp_path
        / "training_progress"
        / "b0"
        / "epoch_010.pt"
    )
    saved = parallel_pilot._save_training_progress(
        method,
        epoch=10,
        history=[method.trainer.last_metrics],
        metrics=method.trainer.last_metrics,
        path=path,
        context=context,
    )
    assert saved == path

    expected_weight = method.trainer.student.weight.detach().clone()
    with torch.no_grad():
        method.trainer.student.weight.zero_()
    loaded = parallel_pilot._load_training_progress(
        method,
        path=path,
        expected_context=context,
        device="cpu",
    )

    assert loaded["epoch"] == 10
    assert loaded["global_step"] == 1
    assert torch.equal(method.trainer.student.weight, expected_weight)


def test_latest_training_progress_uses_epoch_order(tmp_path):
    mode_dir = parallel_pilot._progress_directory(tmp_path, contract.MODE_B0)
    mode_dir.mkdir(parents=True)
    (mode_dir / "epoch_010.pt").touch()
    (mode_dir / "epoch_020.pt").touch()

    assert parallel_pilot._latest_progress_path(
        tmp_path, contract.MODE_B0
    ).name == "epoch_020.pt"
