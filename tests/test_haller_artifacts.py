"""02 票：Haller artifact、来源隔离和可复现 metadata 测试。"""

import json

import numpy as np
import pytest

import haller_anchors


def fixture_result(source=haller_anchors.SOURCE_TRAIN):
    coords = np.linspace(-5.0, 5.0, 101, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radial = np.exp(-(xx * xx + yy * yy) / (2.0 * 1.2**2))
    u = -yy * radial
    v = xx * radial
    return haller_anchors.extract_haller_anchors(
        u, v, coords, coords, np.zeros_like(u, dtype=bool),
        source=source, frame_index=11,
    )


def test_artifact_round_trip_contains_arrays_hashes_and_coverage(tmp_path):
    result = fixture_result()
    paths = haller_anchors.save_haller_artifact(result, tmp_path)

    for name in ("haller_gt.npy", "anchor_state.npy", "anchor_confidence.npy",
                 "standard_ivd.npy", "omega.npy", "anchor_meta.json"):
        assert (tmp_path / name).exists()
    assert set(paths) >= {"haller_gt", "anchor_state", "anchor_confidence", "metadata"}

    metadata = json.loads((tmp_path / "anchor_meta.json").read_text(encoding="utf-8"))
    assert metadata["source"] == haller_anchors.SOURCE_TRAIN
    assert metadata["artifact_id"] == "haller_v1_haller_anchor_train"
    assert metadata["algorithm_version"].startswith("haller-anchor-v1")
    assert metadata["literature"]["status"] == "pending_verification"
    assert metadata["literature"]["zotero_key"] == "L2PX3NQX"
    assert metadata["parameter_hash"] == result["metadata"]["parameter_hash"]
    assert metadata["input_hash"] == result["metadata"]["input_hash"]
    assert metadata["mask_hash"] == result["metadata"]["mask_hash"]
    assert metadata["failure_count"] == 0
    assert metadata["coverage"]["fluid_cells"] == 101 * 101
    assert metadata["coverage"]["positive_cells"] > 0
    assert metadata["coverage"]["negative_cells"] > 0

    loaded = haller_anchors.load_haller_artifact(tmp_path)
    assert loaded["metadata"] == metadata
    for key in ("haller_gt", "anchor_state", "anchor_confidence",
                "standard_ivd", "omega"):
        assert np.allclose(loaded[key], result[key], atol=1e-6)


def test_artifact_source_cannot_overwrite_different_source(tmp_path):
    train = fixture_result(haller_anchors.SOURCE_TRAIN)
    test = fixture_result(haller_anchors.SOURCE_TEST)
    haller_anchors.save_haller_artifact(train, tmp_path)

    with pytest.raises(ValueError, match="source|来源|覆盖"):
        haller_anchors.save_haller_artifact(test, tmp_path)

    metadata = json.loads((tmp_path / "anchor_meta.json").read_text(encoding="utf-8"))
    assert metadata["source"] == haller_anchors.SOURCE_TRAIN


def test_test_gt_requires_explicit_source_and_uses_separate_directory(tmp_path):
    test_dir = tmp_path / "haller_gt_test"
    result = fixture_result(haller_anchors.SOURCE_TEST)
    haller_anchors.save_haller_artifact(result, test_dir)
    loaded = haller_anchors.load_haller_artifact(test_dir, expected_source=haller_anchors.SOURCE_TEST)
    assert loaded["metadata"]["source"] == haller_anchors.SOURCE_TEST

    with pytest.raises(ValueError, match="expected_source|source|来源"):
        haller_anchors.load_haller_artifact(test_dir)

    with pytest.raises(ValueError, match="expected_source|source|来源"):
        haller_anchors.load_haller_artifact(test_dir, expected_source=haller_anchors.SOURCE_TRAIN)


def test_failure_artifact_keeps_unknown_state_and_records_invalid_frame(tmp_path):
    u = np.zeros((20, 20), dtype=np.float64)
    v = np.zeros_like(u)
    coords = np.linspace(-1.0, 1.0, 20)
    result = haller_anchors.extract_haller_anchors(
        u, v, coords, coords, None, source=haller_anchors.SOURCE_TEST,
        frame_index=99,
    )
    haller_anchors.save_haller_artifact(result, tmp_path)
    metadata = json.loads((tmp_path / "anchor_meta.json").read_text(encoding="utf-8"))

    assert metadata["frame_valid"] is False
    assert metadata["failure_count"] == 1
    assert metadata["coverage"]["unknown_cells"] == 400
    assert np.all(np.load(tmp_path / "haller_gt.npy") == haller_anchors.UNKNOWN)


def test_artifact_loader_rejects_persisted_array_tampering(tmp_path):
    result = fixture_result()
    haller_anchors.save_haller_artifact(result, tmp_path)
    tampered = np.load(tmp_path / "anchor_confidence.npy", allow_pickle=False)
    tampered[0, 0] = 0.25 if tampered[0, 0] != 0.25 else 0.75
    np.save(tmp_path / "anchor_confidence.npy", tampered)

    with pytest.raises(ValueError, match="hash"):
        haller_anchors.load_haller_artifact(tmp_path)


def test_artifact_loader_rejects_parameter_metadata_tampering(tmp_path):
    result = fixture_result()
    haller_anchors.save_haller_artifact(result, tmp_path)
    metadata_path = tmp_path / "anchor_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["parameters"]["unknown_band_factor"] = 999.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="parameter_hash"):
        haller_anchors.load_haller_artifact(tmp_path)
