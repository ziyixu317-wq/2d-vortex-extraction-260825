"""09 票：六方法 pilot orchestration contract。"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


DATASETS = (
    "boussinesq",
    "cylinder2d",
    "doublegyre2d",
    "fourcenters2d",
    "jungtelziemniak2d",
    "pipedcylinder2d",
)
MODES = ("B0", "B1", "W1-P", "W1-H", "W2", "W3")


def _window_provenance(mode, source):
    anchor = {
        "source": "haller_anchor_train",
        "algorithm_version": "haller-anchor-v1.0",
        "parameter_hash": "parameter-hash-v1",
        "input_hash": "input-hash-v1",
        "mask_hash": "mask-hash-v1",
        "parameters": {"fixture_parameter_set": "v1"},
        "failure_count": 0,
        "coverage": {
            "fluid_cells": 100,
            "known_cells": 100,
            "positive_cells": 50,
            "negative_cells": 50,
            "unknown_cells": 0,
            "known_fraction_fluid": 1.0,
        },
        "literature": {"status": "pending_verification", "zotero_key": "L2PX3NQX"},
        "legacy_p85_used": False,
        "fallback_used": None,
    }
    provenance = {
        "window": {
            "dataset_name": "fixture",
            "split_name": "train",
            "frame_start": 0,
            "frame_end": 8,
            "split_start": 0,
            "split_end": 50,
            "t_win": 8,
            "window_step": 1,
            "generation_version": "fixture-v1",
            "generation_hash": "fixture-generation-hash",
            "contract_hash": "fixture-contract-hash",
        },
        "sampling": {"source": "legacy_p85"},
    }
    if source == "haller_anchor_train":
        provenance["anchor"] = anchor
    return provenance


def _calibration_records(mode):
    from evaluation_report import CalibrationPredictionRecord
    import w2

    if mode == "W2":
        return [
            w2.W2CalibrationRecord(
                dataset_name=name,
                mean_probability=np.asarray([0.95, 0.05]),
                predictive_variance=np.asarray([0.01, 0.01]),
                labels=np.asarray([1.0, 0.0]),
                known_mask=np.asarray([True, True]),
                provenance={"source": "haller_gt_calibration"},
            )
            for name in DATASETS
        ]
    return [
        CalibrationPredictionRecord(
            dataset_name=name,
            prediction=np.asarray([0.95, 0.05]),
            labels=np.asarray([1.0, 0.0]),
            known_mask=np.asarray([True, True]),
            provenance={"source": "haller_gt_calibration"},
        )
        for name in DATASETS
    ]


def _test_records(*, with_variance=False):
    from evaluation_report import TestEvaluationRecord

    return [
        TestEvaluationRecord(
            dataset_name=name,
            prediction=np.asarray([0.95, 0.05]),
            labels=np.asarray([1.0, 0.0]),
            known_mask=np.asarray([True, True]),
            unknown_mask=np.asarray([False, False]),
            predictive_variance=(np.asarray([0.01, 0.01]) if with_variance else None),
            sample_count=2,
            provenance={
                "source": "haller_gt_test",
                "algorithm_version": "haller-anchor-v1.0",
         "parameter_hash": "parameter-hash-v1",
         "input_hash": "input-hash-v1",
         "mask_hash": "mask-hash-v1",
         "parameters": {"fixture_parameter_set": "v1"},
         "failure_count": 0,
                "literature": {"status": "pending_verification", "zotero_key": "L2PX3NQX"},
            },
        )
        for name in DATASETS
    ]


def _make_method(
    mode,
    events,
    *,
    selection=None,
    train_source_override=None,
    train_schema_override=None,
):
    from e2e_weak_supervision import PilotMethod
    import weak_supervision_contract as contract

    source = train_source_override or {
        "B0": "legacy_p85",
        "B1": "legacy_p85",
        "W1-P": "local_p90_p60",
        "W1-H": "haller_anchor_train",
        "W2": "haller_anchor_train",
        "W3": "haller_anchor_train",
    }[mode]
    checkpoint_anchor_hash = None if source != "haller_anchor_train" else "anchor-hash-v1"
    checkpoint_anchor = None if checkpoint_anchor_hash is None else {
        "source": "haller_anchor_train",
        "algorithm_version": "haller-anchor-v1.0",
         "parameter_hash": "parameter-hash-v1",
         "input_hash": "input-hash-v1",
         "mask_hash": "mask-hash-v1",
         "parameters": {"fixture_parameter_set": "v1"},
         "failure_count": 0,
        "coverage": {
            "fluid_cells": 100,
            "known_cells": 100,
            "positive_cells": 50,
            "negative_cells": 50,
            "unknown_cells": 0,
            "known_fraction_fluid": 1.0,
        },
        "literature": {"status": "pending_verification", "zotero_key": "L2PX3NQX"},
        "legacy_p85_used": False,
        "fallback_used": None,
    }
    if mode == "W3":
        assert selection is not None
        assert selection.w2_selection.variance_gate >= 0.0

    def train_batches(epoch):
        return [{
            "mode": mode,
            "split_name": "train",
            "label_source": source,
            "sampling_source": "legacy_p85",
            "feature_schema": (
                contract.feature_schema_for_mode(mode).as_dict()
                if train_schema_override is None else train_schema_override
            ),
            "input_schema": contract.mode_spec(mode).adapter_input_schema.as_dict(),
            "provenance": _window_provenance(mode, source),
        }]

    def train_epoch(batches, epoch):
        list(batches)
        events.append(("train", mode, epoch))
        return {
            "loss": 1.0 / epoch,
            "anchor_coverage": 1.0,
            "pseudo_acceptance": 0.5,
            "teacher_student_disagreement": 0.0,
            "epoch": epoch,
        }

    def save_checkpoint(path, *, epoch, metrics, calibration_policy):
        blob = {
            "mode": mode,
            "epoch": epoch,
            "split_config": {
                "split_name": "train", "split_start": 0, "split_end": 50,
                "t_win": 8, "window_step": 1,
            },
            "anchor_hash": checkpoint_anchor_hash,
            "calibration_policy": calibration_policy,
            "metrics": dict(metrics),
            "seed": 0,
            "rng_state": {"fixture": True},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, sort_keys=True), encoding="utf-8")
        return path

    def load_checkpoint(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def calibration_records():
        events.append(("calibration", mode))
        return _calibration_records(mode)

    def evaluate_test(prediction_threshold, variance_gate):
        events.append(("test", mode, prediction_threshold, variance_gate))
        return _test_records(with_variance=mode in {"W2", "W3"})

    return PilotMethod(
        mode=mode,
        train_batches=train_batches,
        train_epoch=train_epoch,
        calibration_records=calibration_records,
        evaluate_test=evaluate_test,
        save_checkpoint=save_checkpoint,
        load_checkpoint=load_checkpoint,
        role="diagnostic" if mode == "B1" else "headline_candidate",
        anchor_hash=checkpoint_anchor_hash,
        anchor_metadata=checkpoint_anchor,
        variance_gate=(
            0.01 if mode == "W2" else (
                None if mode != "W3" or selection is None
                else selection.w2_selection.variance_gate
            )
        ),
    )


def _config():
    from e2e_weak_supervision import PilotConfig

    return PilotConfig(
        epochs=2,
        seed=0,
        dataset_names=DATASETS,
        dataset_config={"datasets": list(DATASETS), "split_mode": "weak_supervision"},
        split_config={
            "split_name": "train",
            "split_start": 0,
            "split_end": 50,
            "t_win": 8,
            "window_step": 1,
        },
        sampling_config={"source": "legacy_p85", "t_win": 8, "window_step": 1},
        threshold_candidates=(0.5, 0.9),
        variance_candidates=(0.01,),
    )


def test_pilot_runs_all_methods_calibrates_before_test_and_roundtrips_checkpoint(tmp_path):
    from e2e_weak_supervision import run_pilot

    events = []
    methods = {
        mode: (
            (lambda selection, mode=mode: _make_method(mode, events, selection=selection))
            if mode == "W3"
            else _make_method(mode, events)
        )
        for mode in MODES
    }

    report = run_pilot(methods, config=_config(), output_dir=tmp_path)

    assert tuple(report["methods"]) == MODES
    assert report["pilot"]["epochs"] == 2
    assert report["selection"]["best_baseline"] in {"W1-H", "W2"}
    assert report["selection"]["proposed_method"] == "W3"
    assert report["methods"]["B1"]["headline_eligible"] is False
    assert all(report["methods"][mode]["checkpoint_roundtrip"]["mode"] == mode for mode in MODES)
    assert all(report["methods"][mode]["test"]["macro"]["f1"] == pytest.approx(1.0) for mode in MODES)
    assert report["methods"]["W3"]["checkpoint_roundtrip"]["calibration_policy"]["variance_gate"] == pytest.approx(
        report["selection"]["w2"]["variance_gate"]
    )
    assert (tmp_path / "pilot_report.json").exists()

    first_test = next(index for index, event in enumerate(events) if event[0] == "test")
    last_calibration = max(index for index, event in enumerate(events) if event[0] == "calibration")
    assert last_calibration < first_test
    assert sum(event[0] == "test" for event in events) == len(MODES)


def test_pilot_default_is_fifty_epochs_one_seed_and_w3_receives_frozen_gate():
    from e2e_weak_supervision import PilotConfig

    config = PilotConfig(
        dataset_config={"datasets": list(DATASETS), "split_mode": "weak_supervision"},
        split_config={"split_name": "train", "split_start": 0, "split_end": 50, "t_win": 8, "window_step": 1},
        sampling_config={"source": "legacy_p85", "t_win": 8, "window_step": 1},
    )
    assert config.epochs == 50
    assert config.seeds == (0,)
    assert config.ramp_up_epochs == 12


def test_pilot_config_requires_explicit_split_bounds_and_sampling_source():
    from e2e_weak_supervision import PilotConfig

    base = _config()
    incomplete_split = dict(base.split_config)
    incomplete_split.pop("split_end")
    with pytest.raises(ValueError, match="split_start|split_end|显式"):
        PilotConfig(
            dataset_config=base.dataset_config,
            split_config=incomplete_split,
            sampling_config=base.sampling_config,
        )

    incomplete_sampling = dict(base.sampling_config)
    incomplete_sampling.pop("source")
    with pytest.raises(ValueError, match="sampling_source|source|显式"):
        PilotConfig(
            dataset_config=base.dataset_config,
            split_config=base.split_config,
            sampling_config=incomplete_sampling,
        )


def test_pilot_config_accepts_explicit_per_dataset_train_split_ranges():
    from e2e_weak_supervision import PilotConfig

    base = _config()
    split = {
        "split_name": "train",
        "split_ranges": {
            name: [0, 40 + index]
            for index, name in enumerate(DATASETS)
        },
        "t_win": 8,
        "window_step": 1,
    }
    config = PilotConfig(
        dataset_config=base.dataset_config,
        split_config=split,
        sampling_config=base.sampling_config,
    )
    assert config.split_config["split_ranges"]["pipedcylinder2d"] == [0, 45]
    assert "split_start" not in config.split_config
    assert "split_end" not in config.split_config


def test_pilot_train_guard_validates_explicit_collated_windows():
    from e2e_weak_supervision import _validate_train_batch
    import weak_supervision_contract as contract

    window = {
        "dataset_name": "fixture",
        "split_name": "train",
        "frame_start": 0,
        "frame_end": 8,
        "split_start": 0,
        "split_end": 50,
        "t_win": 8,
        "window_step": 1,
    }
    batch = {
        "mode": "B0",
        "split_name": "train",
        "label_source": "legacy_p85",
        "sampling_source": "legacy_p85",
        "feature_schema": contract.FEATURE_SCHEMA_7.as_dict(),
        "input_schema": contract.FEATURE_SCHEMA_7.as_dict(),
        "provenance": {"windows": [window, {**window, "frame_start": 8, "frame_end": 16}]},
    }
    assert _validate_train_batch(batch, mode="B0") is batch


def test_pilot_train_guard_allows_failure_policy_parameter_only():
    from e2e_weak_supervision import _reject_test_data

    assert _reject_test_data(
        {
            "parameters": {
                "failure_fallback_calibration_test": "invalid_frame",
            }
        },
        context="fixture",
        forbidden_sources=(),
    ) is None

    with pytest.raises(ValueError, match="test-only|test"):
        _reject_test_data(
            {"parameters": {"test_metric": "f1"}},
            context="fixture",
            forbidden_sources=(),
        )


def test_pilot_haller_metadata_requires_full_parameters():
    from dataclasses import replace

    method = _make_method("W1-H", [])
    metadata = dict(method.anchor_metadata)
    metadata.pop("parameters")
    with pytest.raises(ValueError, match="parameters"):
        replace(method, anchor_metadata=metadata)


def test_checkpoint_roundtrip_rejects_test_only_metadata(tmp_path):
    from e2e_weak_supervision import _validate_roundtrip

    method = _make_method("B0", [])
    config = _config()
    checkpoint = method.save_checkpoint(
        tmp_path / "b0.json",
        epoch=config.epochs,
        metrics={"loss": 1.0},
        calibration_policy={
            "source": "haller_gt_calibration",
            "prediction_threshold": 0.5,
            "dataset_names": list(DATASETS),
            "record_hashes": ["record-hash"],
            "candidate_count": 1,
        },
    )
    loaded = method.load_checkpoint(checkpoint)
    loaded["test_metrics"] = {"f1": 1.0}

    with pytest.raises(ValueError, match="test-only|test_metrics|test"):
        _validate_roundtrip(
            loaded,
            method=method,
            config=config,
            threshold=0.5,
            gate=None,
        )


def test_pilot_train_source_guard_rejects_test_haller_loudly(tmp_path):
    from e2e_weak_supervision import run_pilot

    events = []
    methods = {
        mode: (
            (lambda selection, mode=mode: _make_method(mode, events, selection=selection))
            if mode == "W3"
            else _make_method(mode, events)
        )
        for mode in MODES
    }
    methods["B0"] = _make_method(
        "B0", events, train_source_override="haller_gt_test"
    )

    with pytest.raises(ValueError, match="haller_gt_test|train|source"):
        run_pilot(methods, config=_config(), output_dir=tmp_path)


def test_pilot_train_guard_rejects_w1_feature_schema_drift(tmp_path):
    import weak_supervision_contract as contract
    from e2e_weak_supervision import run_pilot

    events = []
    methods = {
        mode: (
            (lambda selection, mode=mode: _make_method(mode, events, selection=selection))
            if mode == "W3"
            else _make_method(mode, events)
        )
        for mode in MODES
    }
    methods["W1-P"] = _make_method(
        "W1-P", events, train_schema_override=contract.FEATURE_SCHEMA_6.as_dict()
    )

    with pytest.raises(ValueError, match="feature schema|channel|local-IVD"):
        run_pilot(methods, config=_config(), output_dir=tmp_path)


def test_contract_trainer_runs_b1_step_and_contract_checkpoint_roundtrip(tmp_path):
    import weak_supervision_contract as contract
    from e2e_weak_supervision import ContractTrainer

    class TinySixChannelModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 1)

        def forward(self, data):
            _dummy, pathlines = data
            return torch.sigmoid(self.fc(pathlines).mean(dim=1).squeeze(-1))

    provenance = {
        "window": {
            "dataset_name": "fixture",
            "split_name": "train",
            "frame_start": 0,
            "frame_end": 3,
            "split_start": 0,
            "split_end": 10,
            "t_win": 3,
            "window_step": 1,
        },
        "sampling": {"source": "legacy_p85"},
    }
    batch = contract.WeakSupervisionBatch(
        pathlines=torch.zeros(1, 2, 3, 6),
        labels=torch.tensor([[1.0, 0.0, 1.0]]),
        label_source="legacy_p85",
        sampling_source="legacy_p85",
        split_name="train",
        feature_schema=contract.FEATURE_SCHEMA_6,
        input_schema=contract.FEATURE_SCHEMA_7,
        mode=contract.MODE_B1,
        provenance=provenance,
    )
    student = contract.ChannelSelectingAdapter(TinySixChannelModel(), contract.MODE_B1)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    criterion = contract.ModeAwareLoss(contract.MODE_B1, nn.BCELoss())
    trainer = ContractTrainer(
        student,
        optimizer,
        criterion,
        mode=contract.MODE_B1,
        sampling_source="legacy_p85",
        scheduler=scheduler,
        seed=0,
    )

    metrics = trainer.run_epoch([batch], epoch=1, device="cpu")
    assert metrics["steps"] == 1
    assert trainer.global_step == 1

    checkpoint = trainer.save_checkpoint(
        tmp_path / "b1.pt",
        epoch=1,
        dataset_config={"datasets": ["fixture"], "split_mode": "weak_supervision"},
        split_config={"split_name": "train", "t_win": 3},
        sampling_config={"source": "legacy_p85", "t_win": 3},
        metrics=metrics,
        calibration_policy={
            "source": "haller_gt_calibration",
            "prediction_threshold": 0.5,
            "dataset_names": ["fixture"],
            "record_hashes": ["record-hash"],
            "candidate_count": 1,
            "selection_hash": "selection-hash",
        },
    )
    restored_student = contract.ChannelSelectingAdapter(TinySixChannelModel(), contract.MODE_B1)
    restored_optimizer = torch.optim.AdamW(restored_student.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=2)
    restored = ContractTrainer(
        restored_student,
        restored_optimizer,
        contract.ModeAwareLoss(contract.MODE_B1, nn.BCELoss()),
        mode=contract.MODE_B1,
        sampling_source="legacy_p85",
        scheduler=restored_scheduler,
        seed=99,
    )
    loaded = restored.load_checkpoint(
        checkpoint,
        expected_dataset_config={"datasets": ["fixture"], "split_mode": "weak_supervision"},
        expected_split_config={"split_name": "train", "t_win": 3},
        expected_sampling_config={"source": "legacy_p85", "t_win": 3},
        device="cpu",
        load_mode="inference",
        restore_rng=False,
    )
    assert loaded["mode"] == contract.MODE_B1
    assert loaded["split_config"]["split_name"] == "train"
    assert loaded["adapter_input_schema"] == contract.FEATURE_SCHEMA_7.as_dict()
    assert loaded["seed"] == 0
