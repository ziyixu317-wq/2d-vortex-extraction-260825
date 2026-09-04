"""06 票：W1-H Haller train-anchor 到 teacher 训练 seam 的行为测试。"""

import numpy as np
import pytest

import haller_anchors


def _swirling_field(size=81, extent=4.0, sigma=1.1):
    coords = np.linspace(-extent, extent, size, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radial = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return -yy * radial, xx * radial, coords, coords


def _saved_train_artifact(tmp_path):
    u, v, xdim, ydim = _swirling_field()
    result = haller_anchors.extract_haller_anchors(
        u, v, xdim, ydim, np.zeros_like(u, dtype=bool),
        source=haller_anchors.SOURCE_TRAIN, frame_index=12,
    )
    artifact_dir = tmp_path / "haller_anchor_train" / "frame12"
    haller_anchors.save_haller_artifact(result, artifact_dir)
    return artifact_dir, result


def test_w1_h_reads_independent_train_artifact_and_rejects_test_artifact(tmp_path):
    """W1-H 只消费独立 train artifact，test GT 不能借 loader 进入训练。"""
    import w1_h

    artifact_dir, result = _saved_train_artifact(tmp_path)
    anchor = w1_h.load_haller_train_artifact(artifact_dir)

    assert anchor.source == haller_anchors.SOURCE_TRAIN
    assert anchor.frame_index == 12
    assert anchor.anchor_state.shape == result["anchor_state"].shape
    assert anchor.metadata["parameter_hash"]
    assert anchor.metadata["input_hash"]
    assert anchor.metadata["mask_hash"]
    assert anchor.metadata["failure_count"] == 0
    assert anchor.anchor_hash

    test_result = haller_anchors.extract_haller_anchors(
        *_swirling_field(), np.zeros((81, 81), dtype=bool),
        source=haller_anchors.SOURCE_TEST,
    )
    test_dir = tmp_path / "haller_gt_test" / "frame0000"
    haller_anchors.save_haller_artifact(test_result, test_dir)
    with pytest.raises(ValueError, match="expected_source|haller_anchor_train"):
        w1_h.load_haller_train_artifact(test_dir)


class _FakeTrainStore:
    """提供 dataset.sample_at 所需的最小真实 window seam。"""

    dataset_name = "synthetic"
    label_source = "legacy_p85"

    def __init__(self):
        self._xdim = np.linspace(-4.0, 4.0, 81)
        self._ydim = np.linspace(-4.0, 4.0, 81)

    def sample_at(self, py, px, frame, t_scale=0.25):
        del py, px, frame, t_scale
        seeds = np.asarray([
            [-3.0, -3.0], [-2.0, -2.0], [0.0, 0.0],
            [2.0, 2.0], [3.0, 3.0],
        ], dtype=np.float64)
        pathlines = np.zeros((3, len(seeds), 7), dtype=np.float32)
        pathlines[..., 0] = seeds[None, :, 0]
        pathlines[..., 1] = seeds[None, :, 1]
        return (np.zeros((1, 1, 1, 1), dtype=np.float32), pathlines), np.zeros(
            len(seeds), dtype=np.float32), seeds

    def window_metadata(self, frame):
        return {
            "dataset_name": self.dataset_name,
            "split_name": "train",
            "frame_start": int(frame),
            "frame_end": int(frame) + 3,
            "split_start": 0,
            "split_end": 50,
            "t_win": 3,
            "window_step": 1,
            "label_source": self.label_source,
        }


class _FakeTrainDataset:
    """只模拟已通过 dataset contract 的单数据集 wrapper。"""

    is_weak_supervision = True
    split = "train"
    consumer = "train"
    label_source = "legacy_p85"

    def __init__(self):
        self.store = _FakeTrainStore()
        self._order = None

    def set_epoch(self, epoch):
        assert epoch == 0
        self._order = [(0, 0, 12)]
        return self._order

    def set_epoch_natural(self, epoch=0):
        return self.set_epoch(epoch)

    def __len__(self):
        return 1

    def sample_at(self, py, px, frame):
        return self.store.sample_at(py, px, frame, 0.25)

    def window_metadata(self, frame):
        return self.store.window_metadata(frame)


def test_w1_h_dataset_adapter_replaces_sampling_labels_with_train_anchor(tmp_path):
    """真实 dataset window seam 只把独立 train Haller artifact 接入 formal loss。"""
    import torch

    import w1_h

    artifact_dir, _result = _saved_train_artifact(tmp_path)
    base = _FakeTrainDataset()
    adapter = w1_h.HallerAnchorPathlineDataset(
        base,
        tmp_path / "haller_anchor_train",
        sampling_source="legacy_p85",
    )

    batch = adapter.sample_at(0, 0, 12)
    assert isinstance(batch, w1_h.W1HBatch)
    assert tuple(batch.pathlines.shape) == (1, 3, 5, 7)
    assert batch.label_source == "haller_anchor_train"
    assert batch.sampling_source == "legacy_p85"
    assert batch.contract_batch.provenance["window"]["split_name"] == "train"
    assert batch.contract_batch.provenance["anchor"]["source"] == "haller_anchor_train"
    assert batch.contract_batch.provenance["anchor"]["artifact_hash"] == batch.anchor_hash
    assert batch.as_dict()["anchor_coverage"] >= 0.0

    student = _tiny_model()
    trainer = w1_h.W1HTrainer(
        student,
        torch.optim.AdamW(student.parameters(), lr=0.01),
        config=w1_h.W1HConfig(),
        sampling_source=batch.sampling_source,
        anchor_hash=batch.anchor_hash,
        anchor_metadata=batch.anchor_metadata,
        seed=23,
    )
    logs = [trainer.run_epoch([batch], epoch=epoch) for epoch in range(5)]
    assert trainer.global_step == 5
    assert all(log["loss_source"] == "haller_anchor_train" for log in logs)
    assert all(log["artifact_failure_count"] == 0 for log in logs)

    adapter.set_epoch(0)
    indexed = adapter[0]
    assert isinstance(indexed, w1_h.W1HBatch)
    assert indexed.anchor_hash == batch.anchor_hash


def test_w1_h_collate_rejects_anchor_hash_drift_and_preserves_masks(tmp_path):
    """批内 artifact hash 漂移必须显式处理，不能拼成无审计来源的 batch。"""
    import torch

    import w1_h

    first = _batch()
    collated = w1_h.collate_w1_h_batches([first, first])
    assert tuple(collated.labels.shape) == (2, 6)
    assert tuple(collated.pathlines.shape) == (2, 3, 6, 7)
    assert int(collated.solid_mask.sum()) == 2
    assert int(collated.failed_frame_mask.sum()) == 2

    second = w1_h.build_w1_h_batch(
        torch.zeros(1, 3, 5, 7), first.labels, first.label_mask,
        first.unknown_mask, first.solid_mask,
        failed_frame_mask=first.failed_frame_mask,
        sampling_source="legacy_p85", split_name="train",
        anchor_hash="different-anchor-hash",
    )
    with pytest.raises(ValueError, match="anchor_hash"):
        w1_h.collate_w1_h_batches([first, second])


def _batch():
    import torch

    from w1_h import build_w1_h_batch

    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    label_mask = torch.tensor([[1, 1, 0, 0, 1, 0]], dtype=torch.bool)
    unknown_mask = ~label_mask
    solid_mask = torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.bool)
    failed_frame_mask = torch.tensor([[0, 0, 0, 1, 0, 0]], dtype=torch.bool)
    return build_w1_h_batch(
        torch.zeros(1, 3, 6, 7), labels, label_mask, unknown_mask,
        solid_mask, failed_frame_mask=failed_frame_mask,
        sampling_source="legacy_p85", split_name="train",
        anchor_hash="haller-artifact-hash-v1",
        provenance={
            "anchor": {
                "source": "haller_anchor_train",
                "algorithm_version": "haller-anchor-v1.0",
                "parameter_hash": "parameter-hash-v1",
                "input_hash": "input-hash-v1",
                "mask_hash": "mask-hash-v1",
                "failure_count": 0,
            },
            "window": {
                "dataset_name": "synthetic",
                "split_name": "train",
                "frame_start": 0,
                "frame_end": 3,
                "split_start": 0,
                "split_end": 50,
                "t_win": 3,
                "window_step": 1,
            },
            "sampling": {"source": "legacy_p85"},
        },
    )


def test_w1_h_loss_uses_haller_known_cells_and_ignores_unknown_solid_failed():
    """formal loss source 是 Haller；unknown/solid/failed 不贡献 anchor BCE。"""
    import torch

    from w1_h import W1HConfig, compute_w1_h_loss

    batch = _batch()
    student = torch.tensor([[0.8, 0.2, 0.9, 0.95, 0.6, 0.4]], requires_grad=True)
    teacher = torch.tensor([[0.8, 0.2, 0.95, 0.95, 0.5, 0.95]])
    loss, stats = compute_w1_h_loss(
        student, teacher, batch, config=W1HConfig(), epoch=12,
    )

    assert torch.isfinite(loss)
    assert stats["loss_source"] == "haller_anchor_train"
    assert stats["sampling_source"] == "legacy_p85"
    assert stats["anchor_count"] == 3
    assert stats["anchor_positive_count"] == 2
    assert stats["anchor_negative_count"] == 1
    assert stats["unknown_count"] == 3
    assert stats["solid_count"] == 1
    assert stats["failed_frame_count"] == 1
    assert stats["pseudo_eligible_count"] == 1
    assert stats["pseudo_accepted_count"] == 1
    assert "legacy_p85" not in stats["loss_source"]
    loss.backward()
    assert student.grad is not None


def _tiny_model():
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


def test_w1_h_cpu_smoke_runs_five_epochs_with_ema_and_diagnostics():
    """W1-H 5 epoch CPU smoke：7ch、EMA、ramp 与 Haller diagnostics 可追踪。"""
    import torch

    from w1_h import W1HConfig, W1HTrainer

    torch.manual_seed(23)
    student = _tiny_model()
    trainer = W1HTrainer(
        student, torch.optim.AdamW(student.parameters(), lr=0.01),
        config=W1HConfig(), sampling_source="legacy_p85", seed=23,
        anchor_hash="haller-artifact-hash-v1",
    )
    logs = [trainer.run_epoch([_batch()], epoch=epoch) for epoch in range(5)]

    assert trainer.global_step == 5
    assert len(logs) == 5
    assert logs[0]["ramp_weight"] == pytest.approx(0.0)
    assert logs[-1]["ramp_weight"] == pytest.approx(4.0 / 12.0)
    assert all(np.isfinite(log["loss"]) for log in logs)
    assert all(log["loss_source"] == "haller_anchor_train" for log in logs)
    assert all(log["failed_frame_count"] == 1 for log in logs)


def test_w1_h_checkpoint_round_trip_keeps_source_hash_and_teacher(tmp_path):
    """W1-H checkpoint 必须保存 Haller source/hash、teacher、split 与窗口契约。"""
    import torch

    import weak_supervision_contract as contract
    from w1_h import W1HConfig, W1HTrainer

    student = _tiny_model()
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    trainer = W1HTrainer(
        student, optimizer, scheduler=scheduler, config=W1HConfig(),
        sampling_source="legacy_p85", seed=31,
        anchor_hash="haller-artifact-hash-v1",
        anchor_metadata={
            "source": "haller_anchor_train",
            "algorithm_version": "haller-anchor-v1.0",
            "parameter_hash": "parameter-hash-v1",
            "input_hash": "input-hash-v1",
            "mask_hash": "mask-hash-v1",
            "failure_count": 0,
        },
    )
    trainer.train_step(_batch(), epoch=1)
    dataset_config = {"dataset_name": "fixture", "normalization": "train_only"}
    split_config = {"split_name": "train", "frame_range": [0, 2]}
    sampling_config = {"t_win": 3, "window_step": 1}
    checkpoint = trainer.save_checkpoint(
        tmp_path / "w1h.pt", epoch=1, metrics={"loss": 0.5},
        dataset_config=dataset_config, split_config=split_config,
        sampling_config=sampling_config,
    )
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert blob["mode"] == contract.MODE_W1_H
    assert blob["label_source"] == contract.LABEL_SOURCE_HALLER_TRAIN
    assert blob["anchor_hash"] == "haller-artifact-hash-v1"
    assert blob["extra_metadata"]["formal_loss_source"] == "haller_anchor_train"
    assert blob["extra_metadata"]["haller_anchor"]["parameter_hash"] == "parameter-hash-v1"
    assert "legacy_p85" not in blob["extra_metadata"]["formal_loss_source"]
    assert blob["teacher"] is not None

    restored_student = _tiny_model()
    restored_optimizer = torch.optim.AdamW(restored_student.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=2)
    restored = W1HTrainer(
        restored_student, restored_optimizer, scheduler=restored_scheduler,
        config=W1HConfig(), sampling_source="legacy_p85", seed=999,
        anchor_hash="haller-artifact-hash-v1",
    )
    result = restored.load_checkpoint(
        checkpoint, device="cpu", expected_dataset_config=dataset_config,
        expected_split_config=split_config,
        expected_sampling_config=sampling_config,
    )
    assert result["mode"] == contract.MODE_W1_H
    assert result["anchor_hash"] == "haller-artifact-hash-v1"
    assert restored.global_step == 1
    for name, value in trainer.student.state_dict().items():
        assert torch.equal(value, restored.student.state_dict()[name])
    for name, value in trainer.teacher.state_dict().items():
        assert torch.equal(value, restored.teacher.state_dict()[name])


def test_w1_h_failed_train_artifact_stays_unknown_and_cannot_be_test_source(tmp_path):
    """失败 train frame 保持 fluid unknown；test source 不能伪装成 train anchor。"""
    import w1_h

    u = np.zeros((24, 32), dtype=np.float64)
    v = np.zeros_like(u)
    coords_x = np.linspace(-1.0, 1.0, u.shape[1])
    coords_y = np.linspace(-1.0, 1.0, u.shape[0])
    mask = np.zeros_like(u, dtype=bool)
    result = haller_anchors.extract_haller_anchors(
        u, v, coords_x, coords_y, mask,
        source=haller_anchors.SOURCE_TRAIN, frame_index=4,
    )
    artifact_dir = tmp_path / "failed-train"
    haller_anchors.save_haller_artifact(result, artifact_dir)
    anchor = w1_h.load_haller_train_artifact(artifact_dir)
    assert anchor.failure_count == 1
    assert not anchor.valid
    assert np.all(anchor.anchor_state == haller_anchors.UNKNOWN)
    assert np.all(anchor.failed_frame_mask)
    assert anchor.metadata["coverage"]["known_cells"] == 0
