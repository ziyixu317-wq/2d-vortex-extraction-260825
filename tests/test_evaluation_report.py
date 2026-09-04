"""09 票：explicit Haller test evaluation 与 calibration report seam。"""

import numpy as np
import pytest


DATASETS = (
    "boussinesq",
    "cylinder2d",
    "doublegyre2d",
    "fourcenters2d",
    "jungtelziemniak2d",
    "pipedcylinder2d",
)


def _test_record(dataset_name="boussinesq"):
    from evaluation_report import TestEvaluationRecord

    return TestEvaluationRecord(
        dataset_name=dataset_name,
        prediction=np.asarray([0.9, 0.1, 0.9, 0.8, 0.2]),
        labels=np.asarray([1.0, 0.0, 1.0, 0.0, 0.0]),
        known_mask=np.asarray([True, True, False, True, False]),
        unknown_mask=np.asarray([False, False, True, False, True]),
        solid_mask=np.asarray([False, False, False, False, True]),
        sample_count=5,
        frame_count=1,
        provenance={
            "source": "haller_gt_test",
            "algorithm_version": "haller-anchor-v1.0",
            "parameter_hash": "parameter-hash-v1",
            "input_hash": "input-hash-v1",
            "mask_hash": "mask-hash-v1",
            "failure_count": 0,
            "literature": {"status": "pending_verification", "zotero_key": "L2PX3NQX"},
        },
    )


def _calibration_record(dataset_name, prediction, labels):
    from evaluation_report import CalibrationPredictionRecord

    return CalibrationPredictionRecord(
        dataset_name=dataset_name,
        prediction=np.asarray(prediction, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        known_mask=np.ones(len(prediction), dtype=bool),
        provenance={"source": "haller_gt_calibration"},
    )


def test_test_report_excludes_unknown_solid_and_reports_confusion_and_coverage():
    from evaluation_report import build_evaluation_report

    report = build_evaluation_report(
        [_test_record()],
        prediction_threshold=0.5,
        method="W1-H",
    )

    row = report["per_dataset"]["boussinesq"]
    assert row["true_positive"] == 1
    assert row["false_positive"] == 1
    assert row["false_negative"] == 0
    assert row["true_negative"] == 1
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(2.0 / 3.0)
    assert row["iou"] == pytest.approx(0.5)
    assert row["effective_cell_count"] == 3
    assert row["effective_sample_count"] == 3
    assert row["unknown_count"] == 1
    assert row["solid_count"] == 1
    assert row["haller_known_coverage"] == pytest.approx(0.75)
    assert row["haller_unknown_coverage"] == pytest.approx(0.25)
    assert row["predicted_area_ratio"] == pytest.approx(2.0 / 3.0)
    assert row["ground_truth_area_ratio"] == pytest.approx(1.0 / 3.0)


def test_report_has_six_independent_rows_and_equal_macro():
    from evaluation_report import build_evaluation_report

    records = []
    for dataset_name in DATASETS:
        record = _test_record(dataset_name)
        records.append(
            type(record)(
                **{
                    **record.__dict__,
                    "prediction": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
                    "labels": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
                }
            )
        )

    report = build_evaluation_report(
        records,
        prediction_threshold=0.5,
        method="W3",
        dataset_names=DATASETS,
    )

    assert tuple(report["per_dataset"]) == DATASETS
    assert all(report["per_dataset"][name]["f1"] == pytest.approx(1.0) for name in DATASETS)
    assert report["macro"]["precision"] == pytest.approx(1.0)
    assert report["macro"]["recall"] == pytest.approx(1.0)
    assert report["macro"]["f1"] == pytest.approx(1.0)
    assert report["macro"]["iou"] == pytest.approx(1.0)
    assert report["boussinesq_stress"]["dataset_name"] == "boussinesq"


def test_global_calibration_selection_rejects_test_source_and_per_dataset_tuning():
    from evaluation_report import select_global_threshold

    selected = select_global_threshold(
        [
            _calibration_record("dataset-a", [0.9, 0.1], [1.0, 0.0]),
            _calibration_record("dataset-b", [0.8, 0.2], [1.0, 0.0]),
        ],
        thresholds=(0.5, 0.95),
    )
    assert selected.source == "haller_gt_calibration"
    assert selected.threshold == pytest.approx(0.5)
    assert selected.dataset_names == ("dataset-a", "dataset-b")
    assert selected.as_dict()["dataset_threshold_count"] == 0

    with pytest.raises(ValueError, match="haller_gt_test|test|calibration"):
        select_global_threshold([
            {
                "dataset_name": "leak",
                "prediction": [0.9],
                "labels": [1.0],
                "known_mask": [True],
                "split_name": "test",
                "label_source": "haller_gt_test",
            }
        ])

    with pytest.raises(ValueError, match="dataset|global|per-dataset"):
        select_global_threshold([
            {
                "dataset_name": "leak",
                "prediction": [0.9],
                "labels": [1.0],
                "known_mask": [True],
                "split_name": "calibration",
                "label_source": "haller_gt_calibration",
                "dataset_threshold": 0.5,
            }
        ])


def test_calibration_allows_failure_policy_parameter_but_rejects_test_results():
    from evaluation_report import CalibrationPredictionRecord

    record = CalibrationPredictionRecord(
        dataset_name="boussinesq",
        prediction=np.asarray([0.9], dtype=np.float32),
        labels=np.asarray([1.0], dtype=np.float32),
        known_mask=np.asarray([True]),
        provenance={
            "source": "haller_gt_calibration",
            "parameters": {
                "failure_fallback_calibration_test": "invalid_frame",
            },
        },
    )
    assert (
        record.provenance["parameters"]["failure_fallback_calibration_test"]
        == "invalid_frame"
    )

    with pytest.raises(ValueError, match="test-only|test"):
        CalibrationPredictionRecord(
            dataset_name="boussinesq",
            prediction=np.asarray([0.9], dtype=np.float32),
            labels=np.asarray([1.0], dtype=np.float32),
            known_mask=np.asarray([True]),
            provenance={
                "source": "haller_gt_calibration",
                "parameters": {"test_metric": "f1"},
            },
        )

def test_invalid_frame_and_cell_are_recorded_but_never_enter_confusion():
    from evaluation_report import TestEvaluationRecord, build_evaluation_report

    record = TestEvaluationRecord(
        dataset_name="boussinesq",
        prediction=np.asarray([0.9]),
        labels=np.asarray([1.0]),
        known_mask=np.asarray([False]),
        unknown_mask=np.asarray([False]),
        invalid_mask=np.asarray([True]),
        split_name="test",
        label_source="haller_gt_test",
        frame_count=1,
        invalid_frame_count=1,
        failure_count=2,
        frame_valid=False,
        provenance={"source": "haller_gt_test", "literature": {"status": "pending_verification"}},
    )

    row = build_evaluation_report(
        [record], prediction_threshold=0.5, method="W3"
    )["per_dataset"]["boussinesq"]
    assert row["effective_frame_count"] == 0
    assert row["effective_cell_count"] == 0
    assert row["invalid_cell_count"] == 1
    assert row["invalid_count"] == 2
    assert row["failure_count"] == 2
    assert row["true_positive"] == row["false_positive"] == 0
