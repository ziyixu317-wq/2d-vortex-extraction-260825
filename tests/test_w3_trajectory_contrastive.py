"""08 票：W3 trajectory embedding 与 in-batch contrastive seam。"""

import copy

import pytest
import torch


def _anchor_provenance():
    return {
        "anchor": {
            "source": "haller_anchor_train",
            "algorithm_version": "haller-anchor-v1.0",
            "parameter_hash": "parameter-hash-v1",
            "input_hash": "input-hash-v1",
            "mask_hash": "mask-hash-v1",
            "failure_count": 0,
            "coverage": 0.75,
            "literature": {"status": "pending_verification", "zotero_key": "L2PX3NQX"},
            "legacy_p85_used": False,
            "fallback_used": None,
        },
        "window": {
            "dataset_name": "fixture",
            "split_name": "train",
            "frame_start": 0,
            "frame_end": 24,
            "split_start": 0,
            "split_end": 50,
            "t_win": 24,
            "window_step": 1,
            "generation_version": "fixture-generation-v1",
            "generation_hash": "fixture-generation-hash-v1",
            "contract_hash": "fixture-contract-hash-v1",
            "feature_schema": {
                "name": "pathline_7ch",
                "version": "v1",
                "channels": ["px", "py", "t", "ivd", "distance", "u", "v"],
                "channel_count": 7,
                "local_ivd_channel": 3,
            },
            "label_source": "legacy_p85",
        },
        "sampling": {"source": "legacy_p85"},
    }


def _w2_batch():
    import w2

    provenance = _anchor_provenance()
    pathlines = torch.zeros(1, 3, 6, 7)
    labels = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    label_mask = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    unknown_mask = ~label_mask
    solid_mask = torch.tensor([[0, 0, 0, 0, 1, 0]], dtype=torch.bool)
    failed_frame_mask = torch.tensor([[0, 0, 0, 0, 0, 1]], dtype=torch.bool)
    return w2.build_w2_batch(
        pathlines,
        labels,
        label_mask,
        unknown_mask,
        solid_mask,
        failed_frame_mask=failed_frame_mask,
        sampling_source="legacy_p85",
        split_name="train",
        anchor_hash="haller-artifact-hash-v1",
        provenance=provenance,
        anchor_metadata=provenance["anchor"],
    )


class _FakeVendorPathlineModel(torch.nn.Module):
    """模拟 vendor 的概率输出边界，classifier 前输入是逐迹线 embedding。"""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.fc.weight.fill_(0.25)

    def forward(self, data):
        _dummy, pathlines = data
        embedding = pathlines[..., 3:].mean(dim=1)
        return torch.sigmoid(self.fc(embedding).squeeze(-1))


class _StochasticFakeVendorPathlineModel(_FakeVendorPathlineModel):
    """带显式随机 view 的 fake vendor，用于验证 W3 RNG seam。"""

    def forward(self, data):
        _dummy, pathlines = data
        embedding = pathlines[..., 3:].mean(dim=1)
        embedding = embedding + 0.01 * torch.rand_like(embedding)
        return torch.sigmoid(self.fc(embedding).squeeze(-1))


def _calibration_selection():
    import w2

    return w2.W2CalibrationSelection(
        prediction_threshold=0.5,
        variance_gate=0.01,
        objective_value=1.0,
        dataset_names=("fixture",),
        record_hashes=("record-hash",),
        candidate_count=1,
        selection_hash="selection-hash",
    )


def test_local_adapter_returns_probability_and_aligned_trajectory_embedding():
    import w3

    vendor_model = _FakeVendorPathlineModel()
    adapter = w3.TrajectoryEmbeddingAdapter(vendor_model)
    pathlines = torch.arange(2 * 3 * 4 * 7, dtype=torch.float32).reshape(2, 3, 4, 7)
    dummy_field = torch.zeros(2, 1, 1, 1)

    probability, embedding = adapter.forward_with_embedding(
        (dummy_field, pathlines),
    )

    assert probability.shape == (2, 4)
    assert embedding.shape == (2, 4, 4)
    expected_first_identity = pathlines[0, :, 0, 3:].mean(dim=0)
    assert torch.equal(embedding[0, 0], expected_first_identity)
    assert torch.all((probability >= 0.0) & (probability <= 1.0))


def test_projection_head_is_fixed_at_64_dimensions():
    import w3

    head = w3.TrajectoryProjectionHead(8)
    projected = head(torch.zeros(2, 5, 8))

    assert projected.shape == (2, 5, 64)
    assert head.projection_dim == 64


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("view_count", 3),
        ("teacher_view_count", 2),
        ("projection_dim", 32),
        ("temperature", 0.2),
        ("max_embeddings", 256),
    ),
)
def test_w3_freezes_resource_and_contrastive_hyperparameters(field, value):
    import w3

    with pytest.raises(ValueError, match="W3"):
        w3.W3Config(variance_gate=0.01, **{field: value})


def test_two_view_pair_builder_keeps_identity_positive_and_other_batch_negatives():
    import w3

    first = torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    second = first + 100.0
    result = w3.build_trajectory_pairs(
        [first, second],
        torch.tensor([[True, False, True, True]]),
    )

    assert result.effective_embedding_count == 6
    assert result.valid_identity_count == 3
    assert result.positive_pairs.tolist() == [[0, 3], [1, 4], [2, 5]]
    assert result.negative_pair_count == 24
    assert result.unknown_exclusion_count == 1


def test_w3_masks_and_window_provenance_fail_loudly():
    import w3

    first = torch.zeros(1, 1, 4)
    with pytest.raises(ValueError, match="0/1"):
        w3.build_trajectory_pairs(
            [first, first.clone()],
            torch.tensor([[0.5]]),
        )

    valid_w2 = _w2_batch()
    valid_w3 = w3.build_w3_batch_from_w2(valid_w2)
    with pytest.raises(ValueError, match="0/1"):
        w3.W3Batch(
            valid_w3.contract_batch,
            torch.tensor([[0.5, 0, 0, 0, 0, 0]]),
            valid_w3.failed_frame_mask,
            valid_w3.anchor_hash,
            valid_w3.dummy_field,
            valid_w3.anchor_metadata,
        )

    provenance = copy.deepcopy(valid_w3.contract_batch.provenance)
    del provenance["window"]
    import weak_supervision_contract as contract

    invalid_contract = contract.WeakSupervisionBatch(
        pathlines=valid_w3.pathlines,
        labels=valid_w3.labels,
        label_source=valid_w3.label_source,
        split_name="train",
        feature_schema=contract.FEATURE_SCHEMA_7,
        label_mask=valid_w3.label_mask,
        unknown_mask=valid_w3.unknown_mask,
        sampling_source=valid_w3.sampling_source,
        provenance=provenance,
        mode=contract.MODE_W3,
        input_schema=contract.FEATURE_SCHEMA_7,
    )
    with pytest.raises(ValueError, match="window|windows"):
        w3.W3Batch(
            invalid_contract,
            valid_w3.solid_mask,
            valid_w3.failed_frame_mask,
            valid_w3.anchor_hash,
            valid_w3.dummy_field,
            valid_w3.anchor_metadata,
        )

    ambiguous = copy.deepcopy(valid_w3.contract_batch.provenance)
    ambiguous["batches"] = [{"window": copy.deepcopy(ambiguous["window"])}]
    ambiguous_contract = contract.WeakSupervisionBatch(
        pathlines=valid_w3.pathlines,
        labels=valid_w3.labels,
        label_source=valid_w3.label_source,
        split_name="train",
        feature_schema=contract.FEATURE_SCHEMA_7,
        label_mask=valid_w3.label_mask,
        unknown_mask=valid_w3.unknown_mask,
        sampling_source=valid_w3.sampling_source,
        provenance=ambiguous,
        mode=contract.MODE_W3,
        input_schema=contract.FEATURE_SCHEMA_7,
    )
    with pytest.raises(ValueError, match="同时携带"):
        w3.W3Batch(
            ambiguous_contract,
            valid_w3.solid_mask,
            valid_w3.failed_frame_mask,
            valid_w3.anchor_hash,
            valid_w3.dummy_field,
            valid_w3.anchor_metadata,
        )


def test_w3_binds_haller_anchor_provenance_to_batch_metadata():
    import weak_supervision_contract as contract
    import w3

    valid_w3 = w3.build_w3_batch_from_w2(_w2_batch())
    provenance = copy.deepcopy(valid_w3.contract_batch.provenance)
    provenance["anchor"]["parameter_hash"] = "different-parameter-hash"
    mismatched = contract.WeakSupervisionBatch(
        pathlines=valid_w3.pathlines,
        labels=valid_w3.labels,
        label_source=valid_w3.label_source,
        split_name=valid_w3.contract_batch.split_name,
        feature_schema=contract.FEATURE_SCHEMA_7,
        label_mask=valid_w3.label_mask,
        unknown_mask=valid_w3.unknown_mask,
        sampling_source=valid_w3.sampling_source,
        provenance=provenance,
        mode=contract.MODE_W3,
        input_schema=contract.FEATURE_SCHEMA_7,
    )

    with pytest.raises(ValueError, match="parameter_hash"):
        w3.W3Batch(
            mismatched,
            valid_w3.solid_mask,
            valid_w3.failed_frame_mask,
            valid_w3.anchor_hash,
            valid_w3.dummy_field,
            valid_w3.anchor_metadata,
        )


def test_w3_accepts_collated_w1_h_windows_after_explicit_w2_normalization():
    import weak_supervision_contract as contract
    import w1_h
    import w3

    valid_w3 = w3.build_w3_batch_from_w2(_w2_batch())
    base = valid_w3.contract_batch
    w1_contract = contract.WeakSupervisionBatch(
        pathlines=base.pathlines,
        labels=base.labels,
        label_source=base.label_source,
        split_name=base.split_name,
        feature_schema=contract.FEATURE_SCHEMA_7,
        label_mask=base.label_mask,
        unknown_mask=base.unknown_mask,
        sampling_source=base.sampling_source,
        provenance=copy.deepcopy(base.provenance),
        mode=contract.MODE_W1_H,
        input_schema=contract.FEATURE_SCHEMA_7,
    )
    first = w1_h.W1HBatch(
        w1_contract,
        valid_w3.solid_mask,
        valid_w3.failed_frame_mask,
        valid_w3.anchor_hash,
        valid_w3.dummy_field,
        valid_w3.anchor_metadata,
    )
    collated = w1_h.collate_w1_h_batches([first, first])

    upgraded = w3.build_w3_batch_from_w1_h(collated)

    assert tuple(upgraded.labels.shape) == (2, 6)
    assert upgraded.contract_batch.provenance["windows"]
    assert len(upgraded.anchor_metadata["batch_artifacts"]) == 2


def test_w3_requires_explicit_contract_mode():
    import weak_supervision_contract as contract
    import w3

    valid_w3 = w3.build_w3_batch_from_w2(_w2_batch())
    mode_less = contract.WeakSupervisionBatch(
        pathlines=valid_w3.pathlines,
        labels=valid_w3.labels,
        label_source=valid_w3.label_source,
        split_name=valid_w3.contract_batch.split_name,
        feature_schema=contract.FEATURE_SCHEMA_7,
        label_mask=valid_w3.label_mask,
        unknown_mask=valid_w3.unknown_mask,
        sampling_source=valid_w3.sampling_source,
        provenance=copy.deepcopy(valid_w3.contract_batch.provenance),
        mode=None,
        input_schema=contract.FEATURE_SCHEMA_7,
    )

    with pytest.raises(ValueError, match="mode=W3|mode=None"):
        w3.W3Batch(
            mode_less,
            valid_w3.solid_mask,
            valid_w3.failed_frame_mask,
            valid_w3.anchor_hash,
            valid_w3.dummy_field,
            valid_w3.anchor_metadata,
        )


def test_contrastive_loss_caps_two_view_embeddings_at_512_deterministically():
    import w3

    first = torch.randn(1, 300, 8)
    second = torch.randn(1, 300, 8)
    loss, stats = w3.compute_trajectory_contrastive_loss(
        [first, second],
        torch.ones(1, 300, dtype=torch.bool),
    )

    assert torch.isfinite(loss)
    assert stats["effective_embedding_count"] == 512
    assert stats["valid_identity_count"] == 256
    assert stats["cap_exclusion_count"] == 44
    assert stats["positive_pair_count"] == 256


def test_w3_local_adapter_can_wrap_real_vendor_without_editing_vendor_files():
    from vendor.DeepUtils.models.segmentation.pathline_transformer import (
        PathlineTransformerV0,
    )
    import w3

    model = PathlineTransformerV0(
        in_channels=7,
        PathlineGroups=4,
        KpathlinePerGroup=1,
        num_encoder_layers=1,
        dmodel=16,
        k=2,
    )
    adapter = w3.TrajectoryEmbeddingAdapter(model)
    pathlines = torch.randn(1, 4, 4, 7)
    dummy_field = torch.zeros(1, 1, 1, 1)

    with torch.no_grad():
        probability, embedding = adapter.forward_with_embedding(
            (dummy_field, pathlines),
        )

    assert probability.shape == (1, 4)
    assert embedding.shape == (1, 4, 16)
    assert torch.isfinite(embedding).all()


def test_w3_config_and_batch_keep_w2_schema_and_frozen_contrastive_contract():
    import w3

    config = w3.W3Config(variance_gate=0.05)
    batch = w3.build_w3_batch_from_w2(_w2_batch())

    assert config.view_count == 2
    assert config.teacher_view_count == 3
    assert config.projection_dim == 64
    assert config.temperature == 0.1
    assert config.max_embeddings == 512
    assert config.as_dict()["feature_schema"]["name"] == "pathline_7ch"
    assert batch.contract_batch.mode == "W3"
    assert batch.contract_batch.feature_schema.channels[3] == "ivd"


def test_w3_loss_uses_known_and_gated_pseudo_identities_only():
    import w3

    batch = w3.build_w3_batch_from_w2(_w2_batch())
    student = torch.tensor(
        [[0.90, 0.10, 0.95, 0.05, 0.95, 0.95]],
        requires_grad=True,
    )
    teacher_views = torch.tensor([
        [[0.95, 0.05, 0.96, 0.50, 0.95, 0.95]],
        [[0.95, 0.05, 0.94, 0.60, 0.95, 0.95]],
        [[0.95, 0.05, 0.95, 0.40, 0.95, 0.95]],
    ])
    projected_views = [
        torch.randn(1, 6, 64, requires_grad=True),
        torch.randn(1, 6, 64, requires_grad=True),
    ]

    loss, stats = w3.compute_w3_loss(
        student,
        teacher_views,
        projected_views,
        batch,
        config=w3.W3Config(variance_gate=0.01),
        epoch=12,
    )

    assert torch.isfinite(loss)
    assert stats["valid_identity_count"] == 3
    assert stats["effective_embedding_count"] == 6
    assert stats["positive_pair_count"] == 3
    assert stats["unknown_exclusion_count"] == 1
    assert stats["solid_exclusion_count"] == 1
    assert stats["invalid_exclusion_count"] == 1
    assert stats["view_count"] == 2
    assert stats["teacher_view_count"] == 3
    loss.backward()
    assert student.grad is not None
    assert projected_views[0].grad is not None


def test_w3_rejects_test_split_or_test_label_provenance_at_training_boundary():
    import weak_supervision_contract as contract
    import w2
    import w3

    common = dict(
        pathlines=torch.zeros(1, 3, 6, 7),
        labels=torch.zeros(1, 6),
        label_mask=torch.ones(1, 6, dtype=torch.bool),
        unknown_mask=torch.zeros(1, 6, dtype=torch.bool),
        solid_mask=torch.zeros(1, 6, dtype=torch.bool),
        failed_frame_mask=torch.zeros(1, 6, dtype=torch.bool),
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        provenance=_anchor_provenance(),
    )
    with pytest.raises(ValueError, match="split|train|test"):
        w2.build_w2_batch(**common, split_name="test")

    leaked = copy.deepcopy(_anchor_provenance())
    leaked["test_metrics"] = {"accuracy": 1.0}
    with pytest.raises(ValueError, match="test"):
        w2.build_w2_batch(**{**common, "provenance": leaked})

    valid_w3 = w3.build_w3_batch_from_w2(_w2_batch())
    leaked_w3_provenance = copy.deepcopy(valid_w3.contract_batch.provenance)
    leaked_w3_provenance["test_metrics"] = {"accuracy": 1.0}
    leaked_w3_contract = contract.WeakSupervisionBatch(
        pathlines=valid_w3.pathlines,
        labels=valid_w3.labels,
        label_source=valid_w3.label_source,
        split_name=valid_w3.contract_batch.split_name,
        feature_schema=contract.FEATURE_SCHEMA_7,
        label_mask=valid_w3.label_mask,
        unknown_mask=valid_w3.unknown_mask,
        sampling_source=valid_w3.sampling_source,
        provenance=leaked_w3_provenance,
        mode=contract.MODE_W3,
        input_schema=contract.FEATURE_SCHEMA_7,
    )
    with pytest.raises(ValueError, match="test"):
        w3.W3Batch(
            leaked_w3_contract,
            valid_w3.solid_mask,
            valid_w3.failed_frame_mask,
            valid_w3.anchor_hash,
            valid_w3.dummy_field,
            valid_w3.anchor_metadata,
        )


def test_w3_trainer_generates_two_reproducible_stochastic_embedding_views():
    import w3

    adapter = w3.TrajectoryEmbeddingAdapter(_StochasticFakeVendorPathlineModel())
    head = w3.TrajectoryProjectionHead(adapter.embedding_dim)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(head.parameters()), lr=0.01)
    trainer = w3.W3Trainer(
        adapter,
        optimizer,
        projection_head=head,
        config=w3.W3Config(variance_gate=0.01),
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=23,
        anchor_metadata=_anchor_provenance()["anchor"],
        calibration_selection=_calibration_selection(),
    )
    batch = w3.build_w3_batch_from_w2(_w2_batch())

    first = trainer.predict_contrastive_views(batch)
    repeated = trainer.predict_contrastive_views(batch)

    assert len(first) == 2
    assert all(prob.shape == (1, 6) for prob, _embedding in first)
    assert all(embedding.shape == (1, 6, adapter.embedding_dim)
               for _prob, embedding in first)
    assert not torch.equal(first[0][1], first[1][1])
    assert all(torch.equal(left[0], right[0]) and torch.equal(left[1], right[1])
               for left, right in zip(first, repeated))


def test_w3_trainer_rejects_shared_raw_student_teacher_backbone():
    import w3

    model = _FakeVendorPathlineModel()
    head = w3.TrajectoryProjectionHead(model.fc.in_features)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=0.01)

    with pytest.raises(ValueError, match="共享"):
        w3.W3Trainer(
            model,
            optimizer,
            teacher=model,
            projection_head=head,
            config=w3.W3Config(variance_gate=0.01),
            sampling_source="legacy_p85",
            anchor_hash="haller-artifact-hash-v1",
        )


def test_w3_trainer_rejects_shared_parameters_from_distinct_raw_models():
    import w3

    student_model = _FakeVendorPathlineModel()
    teacher_model = _FakeVendorPathlineModel()
    teacher_model.fc = student_model.fc
    student_adapter = w3.TrajectoryEmbeddingAdapter(student_model)
    head = w3.TrajectoryProjectionHead(student_adapter.embedding_dim)
    optimizer = torch.optim.AdamW(
        list(student_adapter.parameters()) + list(head.parameters()), lr=0.01)

    with pytest.raises(ValueError, match="共享参数"):
        w3.W3Trainer(
            student_adapter,
            optimizer,
            teacher=teacher_model,
            projection_head=head,
            config=w3.W3Config(variance_gate=0.01),
            sampling_source="legacy_p85",
            anchor_hash="haller-artifact-hash-v1",
        )


def test_w3_trainer_step_records_w2_and_contrastive_statistics():
    import w3

    adapter = w3.TrajectoryEmbeddingAdapter(_StochasticFakeVendorPathlineModel())
    head = w3.TrajectoryProjectionHead(adapter.embedding_dim)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(head.parameters()), lr=0.01)
    trainer = w3.W3Trainer(
        adapter,
        optimizer,
        projection_head=head,
        config=w3.W3Config(variance_gate=0.01),
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=23,
        anchor_metadata=_anchor_provenance()["anchor"],
        calibration_selection=_calibration_selection(),
    )

    stats = trainer.train_step(
        w3.build_w3_batch_from_w2(_w2_batch()), epoch=12,
    )

    assert trainer.global_step == 1
    assert stats["view_count"] == 2
    assert stats["teacher_view_count"] == 3
    assert stats["projection_dim"] == 64
    assert stats["temperature"] == 0.1
    assert stats["effective_embedding_count"] <= 512
    assert stats["positive_pair_count"] == stats["valid_identity_count"]
    assert stats["memory_bank_used"] is False
    assert stats["cross_gpu_gather"] is False
    assert torch.isfinite(torch.tensor(stats["loss"]))


def test_w3_five_epoch_smoke_stays_on_fixed_in_batch_resource_contract():
    import w3

    adapter = w3.TrajectoryEmbeddingAdapter(_StochasticFakeVendorPathlineModel())
    head = w3.TrajectoryProjectionHead(adapter.embedding_dim)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(head.parameters()), lr=0.01)
    trainer = w3.W3Trainer(
        adapter,
        optimizer,
        projection_head=head,
        config=w3.W3Config(variance_gate=0.01),
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=23,
        anchor_metadata=_anchor_provenance()["anchor"],
        calibration_selection=_calibration_selection(),
    )
    batch = w3.build_w3_batch_from_w2(_w2_batch())

    summaries = [
        trainer.run_epoch([batch], epoch=epoch, device="cpu")
        for epoch in range(5)
    ]

    assert trainer.global_step == 5
    assert all(summary["steps"] == 1 for summary in summaries)
    assert all(summary["view_count"] == 2 for summary in summaries)
    assert all(summary["teacher_view_count"] == 3 for summary in summaries)
    assert all(4 <= summary["effective_embedding_count"] <= 6
               for summary in summaries)
    assert all(summary["memory_bank_used"] is False for summary in summaries)
    assert all(summary["cross_gpu_gather"] is False for summary in summaries)
    assert all(torch.isfinite(torch.tensor(summary["loss"])) for summary in summaries)


def test_w3_checkpoint_roundtrip_restores_projection_pair_contract_and_teacher(
    tmp_path, monkeypatch
):
    import weak_supervision_contract as contract
    import w3

    config = w3.W3Config(variance_gate=0.01)
    adapter = w3.TrajectoryEmbeddingAdapter(_StochasticFakeVendorPathlineModel())
    head = w3.TrajectoryProjectionHead(adapter.embedding_dim)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(head.parameters()), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    trainer = w3.W3Trainer(
        adapter,
        optimizer,
        projection_head=head,
        scheduler=scheduler,
        config=config,
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=31,
        anchor_metadata=_anchor_provenance()["anchor"],
        calibration_selection=_calibration_selection(),
    )
    batch = w3.build_w3_batch_from_w2(_w2_batch())
    trainer.train_step(batch, epoch=12)
    dataset_config = {"dataset_name": "fixture", "normalization": "train_only"}
    split_config = {"split_name": "train", "frame_range": [0, 20]}
    sampling_config = {"t_win": 3, "window_step": 1}
    checkpoint = trainer.save_checkpoint(
        tmp_path / "w3.pt",
        epoch=1,
        dataset_config=dataset_config,
        split_config=split_config,
        sampling_config=sampling_config,
    )
    checkpoint_bytes = checkpoint.read_bytes()
    original_save_checkpoint = contract.save_checkpoint

    def interrupted_save(path, *args, **kwargs):
        original_save_checkpoint(path, *args, **kwargs)
        raise RuntimeError("simulated checkpoint interruption")

    monkeypatch.setattr(contract, "save_checkpoint", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        trainer.save_checkpoint(
            checkpoint,
            epoch=1,
            dataset_config=dataset_config,
            split_config=split_config,
            sampling_config=sampling_config,
        )
    assert checkpoint.read_bytes() == checkpoint_bytes
    monkeypatch.undo()
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert blob["mode"] == contract.MODE_W3
    assert blob["projection_head"]
    assert blob["extra_metadata"]["trajectory_contrastive"]["view_count"] == 2
    assert blob["extra_metadata"]["trajectory_contrastive"]["max_embeddings"] == 512
    assert blob["extra_metadata"]["w3_metrics"]["positive_pair_count"] >= 1

    restored_adapter = w3.TrajectoryEmbeddingAdapter(
        _StochasticFakeVendorPathlineModel())
    restored_head = w3.TrajectoryProjectionHead(restored_adapter.embedding_dim)
    restored_optimizer = torch.optim.AdamW(
        list(restored_adapter.parameters()) + list(restored_head.parameters()), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=2)
    restored = w3.W3Trainer(
        restored_adapter,
        restored_optimizer,
        projection_head=restored_head,
        scheduler=restored_scheduler,
        config=config,
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=999,
    )
    result = restored.load_checkpoint(
        checkpoint,
        expected_dataset_config=dataset_config,
        expected_split_config=split_config,
        expected_sampling_config=sampling_config,
    )

    assert result["mode"] == contract.MODE_W3
    assert result["view_count"] == 2
    assert result["projection_dim"] == 64
    assert result["max_embeddings"] == 512
    assert restored.calibration_selection["source"] == "haller_gt_calibration"
    for left, right in zip(trainer.projection_head.parameters(),
                            restored.projection_head.parameters()):
        assert torch.equal(left, right)


def test_w3_checkpoint_requires_calibration_and_rejects_test_metadata(tmp_path):
    import w3

    adapter = w3.TrajectoryEmbeddingAdapter(_StochasticFakeVendorPathlineModel())
    head = w3.TrajectoryProjectionHead(adapter.embedding_dim)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(head.parameters()), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    batch = w3.build_w3_batch_from_w2(_w2_batch())
    trainer = w3.W3Trainer(
        adapter,
        optimizer,
        projection_head=head,
        scheduler=scheduler,
        config=w3.W3Config(variance_gate=0.01),
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=31,
        anchor_metadata=_anchor_provenance()["anchor"],
    )
    trainer.train_step(batch, epoch=12)
    checkpoint_args = dict(
        epoch=1,
        dataset_config={"dataset_name": "fixture"},
        split_config={"split_name": "train", "frame_range": [0, 20]},
        sampling_config={"t_win": 3, "window_step": 1},
    )

    with pytest.raises(ValueError, match="calibration_policy|calibration"):
        trainer.save_checkpoint(tmp_path / "missing-policy.pt", **checkpoint_args)
    with pytest.raises(ValueError, match="test"):
        trainer.save_checkpoint(
            tmp_path / "leaked-metadata.pt",
            extra_metadata={"test_metrics": {"accuracy": 1.0}},
            calibration_policy=_calibration_selection(),
            **checkpoint_args,
        )


def test_w3_checkpoint_load_rejects_resource_contract_drift(tmp_path):
    import w3

    config = w3.W3Config(variance_gate=0.01)
    adapter = w3.TrajectoryEmbeddingAdapter(_StochasticFakeVendorPathlineModel())
    head = w3.TrajectoryProjectionHead(adapter.embedding_dim)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(head.parameters()), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    trainer = w3.W3Trainer(
        adapter,
        optimizer,
        projection_head=head,
        scheduler=scheduler,
        config=config,
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=31,
        anchor_metadata=_anchor_provenance()["anchor"],
        calibration_selection=_calibration_selection(),
    )
    batch = w3.build_w3_batch_from_w2(_w2_batch())
    trainer.train_step(batch, epoch=12)
    dataset_config = {"dataset_name": "fixture"}
    split_config = {"split_name": "train", "frame_range": [0, 20]}
    sampling_config = {"t_win": 3, "window_step": 1}
    checkpoint = trainer.save_checkpoint(
        tmp_path / "valid.pt",
        epoch=1,
        dataset_config=dataset_config,
        split_config=split_config,
        sampling_config=sampling_config,
    )
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    blob["extra_metadata"]["trajectory_contrastive"]["view_count"] = 3
    drifted = tmp_path / "drifted.pt"
    torch.save(blob, drifted)

    restored_adapter = w3.TrajectoryEmbeddingAdapter(
        _StochasticFakeVendorPathlineModel())
    restored_head = w3.TrajectoryProjectionHead(restored_adapter.embedding_dim)
    restored_optimizer = torch.optim.AdamW(
        list(restored_adapter.parameters()) + list(restored_head.parameters()), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=2)
    restored = w3.W3Trainer(
        restored_adapter,
        restored_optimizer,
        projection_head=restored_head,
        scheduler=restored_scheduler,
        config=config,
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=999,
    )
    before_student = copy.deepcopy(restored.student.state_dict())
    before_projection = copy.deepcopy(restored.projection_head.state_dict())
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="trajectory_contrastive"):
        restored.load_checkpoint(
            drifted,
            expected_dataset_config=dataset_config,
            expected_split_config=split_config,
            expected_sampling_config=sampling_config,
        )
    assert all(torch.equal(before_student[key], value)
               for key, value in restored.student.state_dict().items())
    assert all(torch.equal(before_projection[key], value)
               for key, value in restored.projection_head.state_dict().items())
    assert torch.equal(before_rng, torch.get_rng_state())

    blob["extra_metadata"]["trajectory_contrastive"]["view_count"] = 2
    blob["extra_metadata"]["uncertainty_gate"]["variance_gate"] = 0.02
    gate_drifted = tmp_path / "gate-drifted.pt"
    torch.save(blob, gate_drifted)
    with pytest.raises(ValueError, match="uncertainty_gate"):
        restored.load_checkpoint(
            gate_drifted,
            expected_dataset_config=dataset_config,
            expected_split_config=split_config,
            expected_sampling_config=sampling_config,
        )

    corrupt_blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    corrupt_blob["teacher"] = dict(corrupt_blob["teacher"])
    removed_teacher_key = next(iter(corrupt_blob["teacher"]))
    corrupt_blob["teacher"].pop(removed_teacher_key)
    corrupt_blob["ema_teacher"] = dict(corrupt_blob["ema_teacher"])
    corrupt_blob["ema_teacher"].pop(removed_teacher_key)
    corrupt_checkpoint = tmp_path / "corrupt-teacher.pt"
    torch.save(corrupt_blob, corrupt_checkpoint)
    before_student = copy.deepcopy(restored.student.state_dict())
    before_teacher = copy.deepcopy(restored.teacher.state_dict())
    before_projection = copy.deepcopy(restored.projection_head.state_dict())
    before_rng = torch.get_rng_state().clone()
    before_global_step = restored.global_step
    before_seed = restored.seed
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_checkpoint(
            corrupt_checkpoint,
            expected_dataset_config=dataset_config,
            expected_split_config=split_config,
            expected_sampling_config=sampling_config,
        )
    assert all(torch.equal(before_student[key], value)
               for key, value in restored.student.state_dict().items())
    assert all(torch.equal(before_teacher[key], value)
               for key, value in restored.teacher.state_dict().items())
    assert all(torch.equal(before_projection[key], value)
               for key, value in restored.projection_head.state_dict().items())
    assert torch.equal(before_rng, torch.get_rng_state())
    assert restored.global_step == before_global_step
    assert restored.seed == before_seed


def test_w3_multi_batch_epoch_metrics_remain_checkpoint_valid(tmp_path):
    import w3

    config = w3.W3Config(variance_gate=0.01)
    adapter = w3.TrajectoryEmbeddingAdapter(_StochasticFakeVendorPathlineModel())
    head = w3.TrajectoryProjectionHead(adapter.embedding_dim)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(head.parameters()), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    trainer = w3.W3Trainer(
        adapter,
        optimizer,
        projection_head=head,
        scheduler=scheduler,
        config=config,
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=31,
        anchor_metadata=_anchor_provenance()["anchor"],
        calibration_selection=_calibration_selection(),
    )
    batch = w3.build_w3_batch_from_w2(_w2_batch())
    summary = trainer.run_epoch([batch, batch], epoch=12)

    assert summary["pair_stats_scope"] == "epoch"
    assert summary["pair_batch_count"] == 2
    assert summary["effective_embedding_count"] == 2 * summary["valid_identity_count"]
    assert summary["max_effective_embedding_count"] <= 512
    invalid_summary = copy.deepcopy(summary)
    invalid_summary["max_negative_pair_count"] += 1
    with pytest.raises(ValueError, match="negative pair count"):
        trainer.save_checkpoint(
            tmp_path / "invalid-multi-batch.pt",
            epoch=1,
            dataset_config={"dataset_name": "fixture"},
            split_config={"split_name": "train", "frame_range": [0, 20]},
            sampling_config={"t_win": 3, "window_step": 1},
            metrics=invalid_summary,
        )
    checkpoint = trainer.save_checkpoint(
        tmp_path / "multi-batch.pt",
        epoch=1,
        dataset_config={"dataset_name": "fixture"},
        split_config={"split_name": "train", "frame_range": [0, 20]},
        sampling_config={"t_win": 3, "window_step": 1},
        metrics=summary,
    )
    assert checkpoint.exists()
