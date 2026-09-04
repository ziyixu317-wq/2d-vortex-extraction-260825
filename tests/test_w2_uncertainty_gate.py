"""07 票：W2 三视图 uncertainty-gated pseudo-label seam。"""

import copy
import inspect
import pathlib
import subprocess
import sys

import numpy as np
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


def _batch(*, anchor_updates=None):
    import w2

    provenance = _anchor_provenance()
    for key, value in dict(anchor_updates or {}).items():
        provenance["anchor"][key] = value
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


def test_precomputed_haller_loader_imports_without_contour_runtime_dependencies(tmp_path):
    """W2 artifact consumption must not require SciPy/skimage extraction deps."""
    import haller_anchors

    state = np.asarray([[-1, 1], [0, -1]], dtype=np.int8)
    confidence = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    parameters = haller_anchors._resolve_parameters()
    artifact_dir = tmp_path / "precomputed"
    haller_anchors.save_haller_artifact(
        {
            "haller_gt": state,
            "anchor_state": state,
            "anchor_confidence": confidence,
            "standard_ivd": np.zeros((2, 2), dtype=np.float32),
            "omega": np.zeros((2, 2), dtype=np.float32),
            "solid_mask": np.zeros((2, 2), dtype=bool),
            "metadata": {
                "artifact_type": "haller_ivd_three_state",
                "algorithm_version": haller_anchors.ALGORITHM_VERSION,
                "artifact_id": "haller_v1_haller_anchor_train",
                "source": haller_anchors.SOURCE_TRAIN,
                "label_source": haller_anchors.SOURCE_TRAIN,
                "frame_index": 0,
                "shape": [2, 2],
                "parameters": parameters,
                "parameter_hash": haller_anchors._json_hash(parameters),
                "input_hash": "input-hash-v1",
                "mask_hash": "mask-hash-v1",
                "failure_count": 0,
                "valid": True,
            },
        },
        artifact_dir,
    )
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    script = f"""
import builtins

real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name == "scipy" or name.startswith("scipy.")
                       or name == "skimage" or name.startswith("skimage.")):
        raise ModuleNotFoundError(f"blocked optional dependency: {{name}}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import haller_anchors

assert haller_anchors.SOURCE_TRAIN == "haller_anchor_train"
loaded = haller_anchors.load_haller_artifact({str(artifact_dir)!r})
assert loaded["metadata"]["source"] == haller_anchors.SOURCE_TRAIN
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_three_view_statistics_return_probability_mean_population_variance_and_entropy():
    import w2

    views = torch.tensor([
        [[0.9, 0.1, 0.5, 0.0]],
        [[0.7, 0.2, 0.6, 1.0]],
        [[0.8, 0.3, 0.4, 0.5]],
    ])

    result = w2.compute_w2_statistics(views)

    assert result.view_count == 3
    assert tuple(result.mean_probability.shape) == (1, 4)
    assert tuple(result.predictive_variance.shape) == (1, 4)
    assert tuple(result.entropy.shape) == (1, 4)
    assert torch.allclose(
        result.mean_probability,
        torch.tensor([[0.8, 0.2, 0.5, 0.5]]),
        atol=1e-6,
    )
    assert torch.allclose(
        result.predictive_variance,
        torch.tensor([[2.0 / 300.0, 2.0 / 300.0, 2.0 / 300.0, 1.0 / 6.0]]),
        atol=1e-6,
    )
    expected_entropy = -(0.8 * np.log(0.8) + 0.2 * np.log(0.2))
    assert result.entropy[0, 0].item() == pytest.approx(expected_entropy, abs=1e-6)


def test_gate_requires_both_frozen_confidence_and_global_low_variance():
    import w2

    result = w2.apply_w2_uncertainty_gate(
        torch.tensor([[0.95, 0.05, 0.50, 0.95, 0.05]]),
        torch.tensor([[0.01, 0.01, 0.01, 0.20, 0.20]]),
        torch.ones(1, 5, dtype=torch.bool),
        torch.zeros(1, 5, dtype=torch.bool),
        variance_gate=0.05,
    )

    assert torch.equal(
        result.pseudo_mask,
        torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool),
    )
    assert torch.equal(
        result.pseudo_labels,
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    assert result.accepted_count == 2
    assert result.positive_count == 1
    assert result.negative_count == 1
    assert result.unknown_count == 3


def test_w2_config_freezes_three_views_and_confidence_thresholds():
    import w2

    config = w2.W2Config(variance_gate=0.05)
    assert config.view_count == 3
    assert config.pseudo_high == pytest.approx(0.90)
    assert config.pseudo_low == pytest.approx(0.10)

    with pytest.raises(ValueError, match="3.*view|view.*3"):
        w2.W2Config(view_count=2, variance_gate=0.05)
    with pytest.raises(ValueError, match="0.90|0.10|confidence"):
        w2.W2Config(pseudo_high=0.95, variance_gate=0.05)


def test_calibration_selects_one_reproducible_global_gate_across_datasets():
    import w2

    records = [
        w2.W2CalibrationRecord(
            dataset_name="dataset-a",
            mean_probability=torch.tensor([0.95, 0.05, 0.80, 0.20]),
            predictive_variance=torch.tensor([0.01, 0.01, 0.20, 0.20]),
            labels=torch.tensor([1.0, 0.0, 1.0, 0.0]),
            known_mask=torch.ones(4, dtype=torch.bool),
        ),
        w2.W2CalibrationRecord(
            dataset_name="dataset-b",
            mean_probability=torch.tensor([0.90, 0.10, 0.60, 0.40]),
            predictive_variance=torch.tensor([0.01, 0.01, 0.20, 0.20]),
            labels=torch.tensor([1.0, 0.0, 0.0, 1.0]),
            known_mask=torch.ones(4, dtype=torch.bool),
        ),
    ]

    selected = w2.calibrate_w2_gate(
        records,
        prediction_thresholds=(0.5,),
        variance_candidates=(0.01, 0.20),
    )
    repeated = w2.calibrate_w2_gate(
        copy.deepcopy(records),
        prediction_thresholds=(0.5,),
        variance_candidates=(0.01, 0.20),
    )

    assert selected.source == "haller_gt_calibration"
    assert selected.dataset_names == ("dataset-a", "dataset-b")
    assert selected.prediction_threshold == pytest.approx(0.5)
    assert selected.variance_gate == pytest.approx(0.01)
    assert selected.as_dict()["dataset_gate_count"] == 1
    assert selected.as_dict() == repeated.as_dict()


def test_calibration_rejects_test_haller_source_loudly():
    import w2

    with pytest.raises(ValueError, match="haller_gt_test|test|calibration"):
        w2.calibrate_w2_gate([
            {
                "dataset_name": "leak",
                "split_name": "test",
                "label_source": "haller_gt_test",
                "mean_probability": [0.9],
                "predictive_variance": [0.01],
                "labels": [1.0],
                "known_mask": [True],
            }
        ])


def test_calibration_rejects_test_metrics_even_when_label_source_is_calibration():
    import w2

    with pytest.raises(ValueError, match="test.*metric|metric.*test|test"):
        w2.calibrate_w2_gate([
            {
                "dataset_name": "leak",
                "split_name": "calibration",
                "label_source": "haller_gt_calibration",
                "mean_probability": [0.9],
                "predictive_variance": [0.01],
                "labels": [1.0],
                "known_mask": [True],
                "test_metrics": {"f1": 1.0},
            }
        ])


@pytest.mark.parametrize("forbidden_field", ["test_f1", "gt_test", "label_test", "metric_test"])
def test_calibration_rejects_test_metric_and_label_aliases(forbidden_field):
    import w2

    with pytest.raises(ValueError, match="test"):
        w2.calibrate_w2_gate([
            {
                "dataset_name": "leak",
                "split_name": "calibration",
                "label_source": "haller_gt_calibration",
                "mean_probability": [0.9],
                "predictive_variance": [0.01],
                "labels": [1.0],
                "known_mask": [True],
                forbidden_field: 1.0,
            }
        ])


@pytest.mark.parametrize(
    "contamination",
    [
        {"test_metrics": {"f1": 1.0}},
        {"metric_name": "test_f1"},
    ],
)
def test_calibration_revalidates_mutated_typed_record_test_metadata(contamination):
    import w2

    record = w2.W2CalibrationRecord(
        dataset_name="fixture",
        mean_probability=np.asarray([0.9]),
        predictive_variance=np.asarray([0.01]),
        labels=np.asarray([1.0]),
        known_mask=np.asarray([True]),
        provenance={
            "source": "haller_gt_calibration",
            "split_name": "calibration",
        },
    )
    # frozen dataclasses do not freeze nested provenance mappings; the
    # calibration seam must revalidate an already-constructed record.
    record.provenance.update(contamination)
    with pytest.raises(ValueError, match="test"):
        w2.calibrate_w2_gate([record])


def test_calibration_accepts_masked_haller_three_state_labels():
    import w2

    record = w2.W2CalibrationRecord(
        dataset_name="fixture",
        mean_probability=np.asarray([0.2, 0.9, 0.1]),
        predictive_variance=np.asarray([0.2, 0.01, 0.01]),
        labels=np.asarray([-1.0, 1.0, 0.0]),
        known_mask=np.asarray([False, True, True]),
    )

    selected = w2.calibrate_w2_gate(
        [record], prediction_thresholds=(0.5,), variance_candidates=(0.01,)
    )

    assert selected.objective_value == pytest.approx(1.0)


def test_calibration_records_require_three_view_provenance():
    import w2

    with pytest.raises(ValueError, match="view.*3|3.*view"):
        w2.calibrate_w2_gate([
            {
                "dataset_name": "fixture",
                "split_name": "calibration",
                "label_source": "haller_gt_calibration",
                "mean_probability": [0.9],
                "predictive_variance": [0.01],
                "labels": [1.0],
                "known_mask": [True],
                "view_count": 2,
            }
        ])


def test_calibration_selection_requires_typed_nonempty_provenance():
    import w2

    with pytest.raises(ValueError, match="dataset_names"):
        w2.W2CalibrationSelection(
            prediction_threshold=0.5,
            variance_gate=0.01,
            objective_value=1.0,
            dataset_names="fixture",
            record_hashes=("record-hash",),
            candidate_count=1,
            selection_hash="selection-hash",
        )
    with pytest.raises(ValueError, match="record_hashes"):
        w2.W2CalibrationSelection(
            prediction_threshold=0.5,
            variance_gate=0.01,
            objective_value=1.0,
            dataset_names=("fixture",),
            record_hashes="record-hash",
            candidate_count=1,
            selection_hash="selection-hash",
        )
    with pytest.raises(ValueError, match="candidate_count"):
        w2.W2CalibrationSelection(
            prediction_threshold=0.5,
            variance_gate=0.01,
            objective_value=1.0,
            dataset_names=("fixture",),
            record_hashes=("record-hash",),
            candidate_count=1.5,
            selection_hash="selection-hash",
        )


def test_w2_checkpoint_policy_requires_reproducible_calibration_selection():
    import w2

    with pytest.raises(ValueError, match="provenance|dataset_names|selection_hash"):
        w2._policy_as_dict(
            {
                "source": "haller_gt_calibration",
                "prediction_threshold": 0.5,
                "variance_gate": 0.01,
                "dataset_gate_count": 1,
            },
            variance_gate=0.01,
        )


def test_w2_batch_requires_split_contained_window_provenance():
    import w2

    provenance = _anchor_provenance()
    provenance.pop("window")
    with pytest.raises(ValueError, match="window"):
        w2.build_w2_batch(
            torch.zeros(1, 3, 2, 7),
            torch.zeros(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            sampling_source="legacy_p85",
            split_name="train",
            anchor_hash="haller-artifact-hash-v1",
            provenance=provenance,
            anchor_metadata=_anchor_provenance()["anchor"],
        )


def test_w2_batch_requires_canonical_local_ivd_feature_schema():
    import w2

    provenance = _anchor_provenance()
    provenance["window"]["feature_schema"]["name"] = "standard_haller_ivd"
    with pytest.raises(ValueError, match="feature schema|local-IVD|local_ivd"):
        w2.build_w2_batch(
            torch.zeros(1, 3, 2, 7),
            torch.zeros(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            sampling_source="legacy_p85",
            split_name="train",
            anchor_hash="haller-artifact-hash-v1",
            provenance=provenance,
            anchor_metadata=_anchor_provenance()["anchor"],
        )

    crossing = _anchor_provenance()
    crossing["window"]["frame_end"] = 51
    with pytest.raises(ValueError, match="window|split"):
        w2.build_w2_batch(
            torch.zeros(1, 3, 2, 7),
            torch.zeros(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            sampling_source="legacy_p85",
            split_name="train",
            anchor_hash="haller-artifact-hash-v1",
            provenance=crossing,
            anchor_metadata=_anchor_provenance()["anchor"],
        )


def test_w2_training_batch_requires_train_anchor_metadata():
    import w2

    with pytest.raises(ValueError, match="anchor_metadata|Haller"):
        w2.build_w2_batch(
            torch.zeros(1, 3, 2, 7),
            torch.zeros(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            sampling_source="legacy_p85",
            split_name="train",
            anchor_hash="haller-artifact-hash-v1",
            provenance=_anchor_provenance(),
        )


def test_w2_training_batch_requires_pending_verified_haller_anchor_metadata():
    import w2

    cases = []
    missing_literature = _anchor_provenance()["anchor"].copy()
    missing_literature.pop("literature")
    cases.append(missing_literature)
    bad_literature = _anchor_provenance()["anchor"].copy()
    bad_literature["literature"] = {"status": "verified"}
    cases.append(bad_literature)
    legacy_fallback = _anchor_provenance()["anchor"].copy()
    legacy_fallback["legacy_p85_used"] = True
    cases.append(legacy_fallback)
    explicit_fallback = _anchor_provenance()["anchor"].copy()
    explicit_fallback["fallback_used"] = "legacy_p85"
    cases.append(explicit_fallback)
    missing_coverage = _anchor_provenance()["anchor"].copy()
    missing_coverage.pop("coverage")
    cases.append(missing_coverage)

    for metadata in cases:
        with pytest.raises(ValueError, match="literature|pending|legacy|fallback|coverage"):
            w2.build_w2_batch(
                torch.zeros(1, 3, 2, 7),
                torch.zeros(1, 2),
                torch.ones(1, 2, dtype=torch.bool),
                torch.zeros(1, 2, dtype=torch.bool),
                torch.zeros(1, 2, dtype=torch.bool),
                sampling_source="legacy_p85",
                split_name="train",
                anchor_hash="haller-artifact-hash-v1",
                provenance=_anchor_provenance(),
                anchor_metadata=metadata,
            )


def test_w2_accepts_canonical_haller_coverage_mapping():
    import w2

    metadata = _anchor_provenance()["anchor"].copy()
    metadata["coverage"] = {
        "fluid_cells": 10,
        "known_cells": 6,
        "known_fraction_fluid": 0.6,
        "negative_cells": 4,
        "negative_fraction_fluid": 0.4,
        "positive_cells": 2,
        "positive_fraction_fluid": 0.2,
        "solid_cells": 1,
        "total_unknown_cells_including_solid": 5,
        "unknown_cells": 4,
        "unknown_fraction_fluid": 0.4,
    }

    batch = w2.build_w2_batch(
        torch.zeros(1, 3, 2, 7),
        torch.zeros(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.bool),
        sampling_source="legacy_p85",
        split_name="train",
        anchor_hash="haller-artifact-hash-v1",
        provenance=_anchor_provenance(),
        anchor_metadata=metadata,
    )

    assert batch.anchor_metadata["coverage"]["known_fraction_fluid"] == pytest.approx(0.6)


def test_contract_test_key_guard_only_allows_structural_split_ranges_test_key():
    import weak_supervision_contract as contract

    with pytest.raises(ValueError, match="test"):
        contract._reject_test_only_keys(
            {"test": {"metric": 1.0}}, context="fixture"
        )

    # A split_ranges mapping legitimately uses train/calibration/test as
    # structural child keys; that key must not disable the broader guard.
    contract._reject_test_only_keys(
        {"split_ranges": {"fixture": {"test": [10, 20]}}},
        context="fixture",
    )


def test_cuda_checkpoint_contract_is_loaded_on_cpu_before_device_transfer(monkeypatch):
    """CUDA map_location must not move RNG contract tensors before validation."""
    import weak_supervision_contract as contract

    calls = {}

    def fake_torch_load(path, **kwargs):
        calls["path"] = path
        calls.update(kwargs)
        return {"sentinel": True}

    monkeypatch.setattr(torch, "load", fake_torch_load)

    assert contract._torch_load("fixture.pt", "cuda:0") == {"sentinel": True}
    assert calls == {
        "path": "fixture.pt",
        "map_location": "cpu",
        "weights_only": True,
    }


def test_cuda_rng_restore_rejects_visible_device_layout_mismatch(monkeypatch):
    import weak_supervision_contract as contract

    state = contract.capture_rng_state()
    state["torch_cuda"] = [torch.zeros(4, dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert contract.restore_rng_state(state, strict_cuda=False) is False
    with pytest.raises(ValueError, match="CUDA|device|数量|length|长度"):
        contract.restore_rng_state(state, strict_cuda=True)

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device=None: torch.zeros(8, dtype=torch.uint8),
    )
    with pytest.raises(ValueError, match="CUDA|length|长度"):
        contract.restore_rng_state(state, strict_cuda=True)


def test_checkpoint_loader_never_retries_unsafe_pickle_after_type_error(monkeypatch):
    """A loader failure must fail closed instead of silently enabling pickle."""
    import weak_supervision_contract as contract

    calls = []

    def fake_torch_load(path, **kwargs):
        calls.append((path, kwargs))
        raise TypeError("malformed checkpoint")

    monkeypatch.setattr(torch, "load", fake_torch_load)

    with pytest.raises(ValueError, match="safe|unsafe|weights_only"):
        contract._torch_load("fixture.pt", "cuda:0")
    assert len(calls) == 1
    assert calls[0][1] == {"map_location": "cpu", "weights_only": True}


def test_w2_loss_only_accepts_unknown_pseudo_labels_passing_both_gates():
    import w2

    batch = _batch()
    student = torch.tensor(
        [[0.90, 0.10, 0.95, 0.05, 0.95, 0.95]],
        requires_grad=True,
    )
    views = torch.tensor([
        [[0.95, 0.05, 0.96, 0.50, 0.95, 0.95]],
        [[0.95, 0.05, 0.94, 0.60, 0.95, 0.95]],
        [[0.95, 0.05, 0.95, 0.40, 0.95, 0.95]],
    ])

    loss, stats = w2.compute_w2_loss(
        student,
        views,
        batch,
        config=w2.W2Config(variance_gate=0.01),
        epoch=12,
    )

    assert torch.isfinite(loss)
    assert stats["view_count"] == 3
    assert stats["pseudo_eligible_count"] == 2
    assert stats["pseudo_accepted_count"] == 1
    assert stats["pseudo_positive_count"] == 1
    assert stats["pseudo_negative_count"] == 0
    assert stats["variance_gate"] == pytest.approx(0.01)
    loss.backward()
    assert student.grad is not None


class _StochasticTinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(3.0))
        self.calls = 0

    def forward(self, data):
        _dummy, pathlines = data
        offset = (self.calls % 3 - 1) * 0.02
        self.calls += 1
        probability = torch.sigmoid(self.bias + offset)
        return probability.expand(pathlines.shape[0], pathlines.shape[2])


class _RngTinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        _dummy, pathlines = data
        noise = torch.rand(
            pathlines.shape[0], pathlines.shape[2], device=pathlines.device)
        return torch.sigmoid(self.bias + 0.1 * (noise - 0.5))


def test_w2_allows_frame_specific_anchor_metadata_within_one_manifest():
    import w2

    student = _StochasticTinyModel()
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    trainer = w2.W2Trainer(
        student,
        optimizer,
        config=w2.W2Config(variance_gate=0.01),
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=23,
    )

    summary = trainer.run_epoch(
        [_batch(), _batch(anchor_updates={"input_hash": "input-hash-v2"})],
        epoch=0,
        max_steps=2,
    )

    assert summary["steps"] == 2
    assert trainer.global_step == 2


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


def _checkpoint_metrics():
    return {
        "view_count": 3,
        "variance_gate": 0.01,
        "pseudo_accepted_count": 1,
        "pseudo_acceptance": 0.5,
        "pseudo_positive_count": 1,
        "pseudo_negative_count": 0,
        "pseudo_positive_ratio": 1.0,
        "pseudo_negative_ratio": 0.0,
        "mean_probability_mean": 0.95,
        "mean_probability_std": 0.01,
        "mean_probability_min": 0.90,
        "mean_probability_max": 0.99,
        "predictive_variance_mean": 0.001,
        "predictive_variance_std": 0.0001,
        "predictive_variance_min": 0.0,
        "predictive_variance_max": 0.01,
        "entropy_mean": 0.1,
        "entropy_std": 0.01,
        "entropy_min": 0.0,
        "entropy_max": 0.2,
        "teacher_student_disagreement": 0.01,
        "accepted_teacher_student_disagreement": 0.01,
    }


def test_w2_checkpoint_metrics_reject_invalid_counts_and_ratios():
    import w2

    for field, value in (
        ("pseudo_accepted_count", -1),
        ("pseudo_acceptance", 1.01),
        ("pseudo_positive_ratio", -0.01),
        ("pseudo_negative_ratio", 2.0),
        ("mean_probability_mean", 1.01),
        ("predictive_variance_mean", -0.01),
        ("entropy_mean", -0.01),
        ("teacher_student_disagreement", 1.01),
    ):
        metrics = _checkpoint_metrics()
        metrics[field] = value
        with pytest.raises(ValueError, match="metrics|ratio|acceptance|count"):
            w2._validate_w2_checkpoint_metrics(metrics, variance_gate=0.01)


@pytest.mark.parametrize(
    "field",
    [
        "mean_probability_std",
        "mean_probability_min",
        "mean_probability_max",
        "predictive_variance_std",
        "predictive_variance_min",
        "predictive_variance_max",
        "entropy_std",
        "entropy_min",
        "entropy_max",
    ],
)
def test_w2_checkpoint_metrics_require_full_distributions(field):
    import w2

    metrics = _checkpoint_metrics()
    metrics.pop(field)
    with pytest.raises(ValueError, match="metrics|诊断|distribution"):
        w2._validate_w2_checkpoint_metrics(metrics, variance_gate=0.01)


def test_w2_gate_rejects_variance_outside_bernoulli_population_range():
    import w2

    with pytest.raises(ValueError, match="variance|0.25"):
        w2.apply_w2_uncertainty_gate(
            torch.tensor([[0.95]]),
            torch.tensor([[0.50]]),
            torch.ones(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool),
            variance_gate=0.05,
        )


def test_w2_resume_defaults_to_strict_cuda_rng_layout_guard():
    import w2

    default = inspect.signature(w2.W2Trainer.load_checkpoint).parameters[
        "strict_cuda_rng"
    ].default
    assert default is True


def test_w2_teacher_view_seam_uses_reproducible_distinct_rng_per_view():
    import w2

    student = _RngTinyModel()
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    trainer = w2.W2Trainer(
        student,
        optimizer,
        config=w2.W2Config(variance_gate=0.01),
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=23,
    )

    first = trainer.predict_teacher_views(_batch())
    repeated = trainer.predict_teacher_views(_batch())

    assert len(first) == 3
    assert not torch.equal(first[0], first[1])
    assert not torch.equal(first[1], first[2])
    assert all(torch.equal(left, right) for left, right in zip(first, repeated))


def test_w2_cpu_smoke_runs_five_epochs_and_records_view_gate_diagnostics():
    import w2

    student = _StochasticTinyModel()
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    trainer = w2.W2Trainer(
        student,
        optimizer,
        config=w2.W2Config(variance_gate=0.01),
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=23,
    )

    logs = [trainer.run_epoch([_batch()], epoch=epoch) for epoch in range(5)]

    assert trainer.global_step == 5
    assert all(log["view_count"] == 3 for log in logs)
    assert all(log["variance_gate"] == pytest.approx(0.01) for log in logs)
    assert all(log["pseudo_accepted_count"] >= 1 for log in logs)
    assert all(np.isfinite(log["loss"]) for log in logs)
    assert student.calls == 5
    assert trainer.teacher.calls == 15


def test_w2_checkpoint_round_trip_records_gate_views_and_calibration_policy(tmp_path):
    import w2
    import weak_supervision_contract as contract

    config = w2.W2Config(variance_gate=0.01)
    student = _StochasticTinyModel()
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    trainer = w2.W2Trainer(
        student,
        optimizer,
        scheduler=scheduler,
        config=config,
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=31,
        anchor_metadata=_anchor_provenance()["anchor"],
    )
    trainer.train_step(_batch(), epoch=1)
    dataset_config = {"dataset_name": "fixture", "normalization": "train_only"}
    split_config = {"split_name": "train", "frame_range": [0, 20]}
    sampling_config = {"t_win": 3, "window_step": 1}
    policy = _calibration_selection()

    no_anchor_student = _StochasticTinyModel()
    no_anchor_optimizer = torch.optim.AdamW(no_anchor_student.parameters(), lr=0.01)
    no_anchor_scheduler = torch.optim.lr_scheduler.StepLR(
        no_anchor_optimizer, step_size=2)
    no_anchor_trainer = w2.W2Trainer(
        no_anchor_student,
        no_anchor_optimizer,
        scheduler=no_anchor_scheduler,
        config=config,
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=31,
    )
    with pytest.raises(ValueError, match="anchor_metadata|Haller.*metadata"):
        no_anchor_trainer.save_checkpoint(
            tmp_path / "w2-no-anchor-metadata.pt",
            epoch=1,
            metrics=_checkpoint_metrics(),
            dataset_config=dataset_config,
            split_config=split_config,
            sampling_config=sampling_config,
            calibration_policy=policy,
        )

    with pytest.raises(ValueError, match="calibration"):
        trainer.save_checkpoint(
            tmp_path / "w2-no-calibration-policy.pt",
            epoch=1,
            metrics=_checkpoint_metrics(),
            dataset_config=dataset_config,
            split_config=split_config,
            sampling_config=sampling_config,
        )

    with pytest.raises(ValueError, match="acceptance|metrics"):
        trainer.save_checkpoint(
            tmp_path / "w2-incomplete-metrics.pt",
            epoch=1,
            metrics={"view_count": 3, "variance_gate": 0.01},
            dataset_config=dataset_config,
            split_config=split_config,
            sampling_config=sampling_config,
            calibration_policy=policy,
        )

    checkpoint = trainer.save_checkpoint(
        tmp_path / "w2.pt",
        epoch=1,
        metrics=_checkpoint_metrics(),
        dataset_config=dataset_config,
        split_config=split_config,
        sampling_config=sampling_config,
        calibration_policy=policy,
    )
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert blob["mode"] == contract.MODE_W2
    assert blob["label_source"] == contract.LABEL_SOURCE_HALLER_TRAIN
    assert blob["extra_metadata"]["w2_config"]["view_count"] == 3
    assert blob["extra_metadata"]["uncertainty_gate"]["variance_gate"] == pytest.approx(0.01)
    assert blob["calibration_policy"]["source"] == "haller_gt_calibration"
    assert blob["calibration_policy"]["variance_gate"] == pytest.approx(0.01)
    assert blob["extra_metadata"]["haller_anchor"]["input_hash"] == "input-hash-v1"

    restored_student = _StochasticTinyModel()
    restored_optimizer = torch.optim.AdamW(restored_student.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=2)
    restored = w2.W2Trainer(
        restored_student,
        restored_optimizer,
        scheduler=restored_scheduler,
        config=config,
        sampling_source="legacy_p85",
        anchor_hash="haller-artifact-hash-v1",
        seed=999,
    )
    result = restored.load_checkpoint(
        checkpoint,
        device="cpu",
        expected_dataset_config=dataset_config,
        expected_split_config=split_config,
        expected_sampling_config=sampling_config,
    )

    assert result["mode"] == contract.MODE_W2
    assert result["calibration_policy"]["variance_gate"] == pytest.approx(0.01)
    assert restored.anchor_metadata == trainer.anchor_metadata
    assert restored.global_step == 1
    for name, value in trainer.student.state_dict().items():
        assert torch.equal(value, restored.student.state_dict()[name])
    for name, value in trainer.teacher.state_dict().items():
        assert torch.equal(value, restored.teacher.state_dict()[name])

    resumed_checkpoint = restored.save_checkpoint(
        tmp_path / "w2-resumed.pt",
        epoch=2,
        dataset_config=dataset_config,
        split_config=split_config,
        sampling_config=sampling_config,
    )
    resumed_blob = torch.load(
        resumed_checkpoint, map_location="cpu", weights_only=True)
    assert resumed_blob["calibration_policy"]["source"] == "haller_gt_calibration"
    assert resumed_blob["calibration_policy"]["selection_hash"] == "selection-hash"

    tampered_blob = copy.deepcopy(blob)
    tampered_blob["extra_metadata"]["uncertainty_gate"]["view_count"] = 2
    tampered_checkpoint = tmp_path / "w2-tampered-metadata.pt"
    torch.save(tampered_blob, tampered_checkpoint)
    with pytest.raises(ValueError, match="uncertainty_gate|W2 metadata|view"):
        restored.load_checkpoint(
            tampered_checkpoint,
            device="cpu",
            expected_dataset_config=dataset_config,
            expected_split_config=split_config,
            expected_sampling_config=sampling_config,
        )


def test_w2_training_batch_rejects_test_haller_provenance():
    import w2

    with pytest.raises(ValueError, match="haller_gt_test|test|Haller"):
        w2.build_w2_batch(
            torch.zeros(1, 3, 2, 7),
            torch.zeros(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            sampling_source="legacy_p85",
            split_name="train",
            anchor_hash="haller-artifact-hash-v1",
            provenance={
                **_anchor_provenance(),
                "leaked_label": "haller_gt_test",
            },
            anchor_metadata=_anchor_provenance()["anchor"],
        )


def test_w2_training_batch_rejects_test_only_anchor_metadata():
    import w2

    with pytest.raises(ValueError, match="test.*metric|metric.*test|test"):
        w2.build_w2_batch(
            torch.zeros(1, 3, 2, 7),
            torch.zeros(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            sampling_source="legacy_p85",
            split_name="train",
            anchor_hash="haller-artifact-hash-v1",
            anchor_metadata={
                **_anchor_provenance()["anchor"],
                "test_metrics": {"f1": 1.0},
            },
            provenance=_anchor_provenance(),
        )


def test_w2_training_batch_rejects_sampling_provenance_drift():
    import w2

    provenance = _anchor_provenance()
    provenance["sampling"]["source"] = "local_p90_p60"
    with pytest.raises(ValueError, match="sampling.*source|source.*sampling"):
        w2.build_w2_batch(
            torch.zeros(1, 3, 2, 7),
            torch.zeros(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            torch.zeros(1, 2, dtype=torch.bool),
            sampling_source="legacy_p85",
            split_name="train",
            anchor_hash="haller-artifact-hash-v1",
            provenance=provenance,
            anchor_metadata=_anchor_provenance()["anchor"],
        )
