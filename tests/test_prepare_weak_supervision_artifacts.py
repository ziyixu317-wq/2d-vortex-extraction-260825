"""WS-9 artifact preparer 的输入 contract 回归测试。"""

import json

import numpy as np
import pytest

import haller_anchors
import prepare_weak_supervision_artifacts as preparer
import weak_supervision_contract as contract


def test_prepare_weak_dataset_passes_formal_legacy_source_constants(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_prepare_dataset(**kwargs):
        captured.update(kwargs)
        out = kwargs["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        shape = [4, 8, 8]
        fake_arrays = {
            "u.npy": np.zeros(shape, dtype=np.float32),
            "v.npy": np.zeros(shape, dtype=np.float32),
            "ivd.npy": np.zeros(shape, dtype=np.float32),
            "label_field.npy": np.zeros(shape, dtype=np.uint8),
            "mask.npy": np.zeros(shape[1:], dtype=np.uint8),
        }
        metadata = {
            "dataset_name": "fixture",
            "shape": shape,
            "split_mode": "weak_supervision",
            "label_source": contract.LABEL_SOURCE_LEGACY_P85,
            "sampling_source": contract.LABEL_SOURCE_LEGACY_P85,
            "loss_label_source": contract.LABEL_SOURCE_LEGACY_P85,
            "normalization_source": "train",
            "normalization_frozen": True,
            "window": {"t_win": 2, "window_step": 1},
            "contract_hash": "fixture-contract",
        }
        for filename, array in fake_arrays.items():
            np.save(out / filename, array)
        (out / "meta.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return metadata

    monkeypatch.setattr(preparer.dataset, "prepare_dataset", fake_prepare_dataset)

    source = {
        "name": "fixture",
        "shape": (4, 8, 8),
        "u": np.zeros((4, 8, 8), dtype=np.float32),
        "v": np.zeros((4, 8, 8), dtype=np.float32),
        "xdim": np.arange(8, dtype=np.float64),
        "ydim": np.arange(8, dtype=np.float64),
        "tdim": np.arange(4, dtype=np.float64),
        "mask": np.zeros((8, 8), dtype=np.uint8),
        "ivd": np.zeros((4, 8, 8), dtype=np.float32),
        "labels": np.zeros((4, 8, 8), dtype=np.uint8),
    }
    metadata = preparer.prepare_weak_dataset(
        source, tmp_path, t_win=2, window_step=1
    )

    assert captured["label_source"] == contract.LABEL_SOURCE_LEGACY_P85
    assert captured["sampling_source"] == contract.LABEL_SOURCE_LEGACY_P85
    assert captured["loss_label_source"] == contract.LABEL_SOURCE_LEGACY_P85
    assert metadata["contract_hash"] == "fixture-contract"


def test_prepare_haller_source_parallel_workers_preserve_frame_order(tmp_path):
    coords = np.linspace(-3.0, 3.0, 31, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radial = np.exp(-(xx * xx + yy * yy) / (2.0 * 0.9**2))
    u = np.stack([-yy * radial] * 3)
    v = np.stack([xx * radial] * 3)
    mask = np.zeros((31, 31), dtype=bool)
    weak_root = tmp_path / "weak"
    weak_root.mkdir()
    (weak_root / "meta.json").write_text(
        json.dumps({
            "split_ranges": {"train": [0, 3]},
            "window": {
                "t_win": preparer.DEFAULT_T_WIN,
                "window_step": preparer.DEFAULT_WINDOW_STEP,
            },
        }), encoding="utf-8"
    )

    manifest = preparer.prepare_haller_source(
        {
            "name": "fixture",
            "weak_root": weak_root,
            "u": u,
            "v": v,
            "xdim": coords,
            "ydim": coords,
            "mask": mask,
        },
        tmp_path / "haller",
        source=haller_anchors.SOURCE_TRAIN,
        workers=3,
    )

    assert manifest["frame_count"] == 3
    assert [item["frame_index"] for item in manifest["frame_artifacts"]] == [0, 1, 2]
    assert manifest["literature"]["status"] == "pending_verification"
    assert manifest["backend"] == haller_anchors.BACKEND_NUMPY
    assert manifest["cuda_used"] is False
    assert manifest["contour_mode"] == haller_anchors.CONTOUR_MODE_OPTIMIZED
    frame_meta = json.loads(
        (tmp_path / "haller" / haller_anchors.SOURCE_TRAIN / "fixture" /
         "frame0" / "anchor_meta.json").read_text(encoding="utf-8")
    )
    assert frame_meta["contour_mode"] == haller_anchors.CONTOUR_MODE_OPTIMIZED
    assert frame_meta["split_name"] == "train"
    assert frame_meta["window"]["t_win"] == preparer.DEFAULT_T_WIN
    assert frame_meta["window"]["window_step"] == preparer.DEFAULT_WINDOW_STEP
    assert manifest["split_name"] == "train"
    assert manifest["window"]["t_win"] == preparer.DEFAULT_T_WIN
    assert manifest["window"]["window_step"] == preparer.DEFAULT_WINDOW_STEP


def test_prepare_haller_source_accepts_cpu_fast_haller_backend(tmp_path):
    coords = np.linspace(-3.0, 3.0, 25, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radial = np.exp(-(xx * xx + yy * yy) / (2.0 * 0.9**2))
    weak_root = tmp_path / "weak-fast"
    weak_root.mkdir()
    (weak_root / "meta.json").write_text(
        json.dumps({
            "split_ranges": {"train": [0, 1]},
            "window": {"t_win": 24, "window_step": 4},
        }),
        encoding="utf-8",
    )
    manifest = preparer.prepare_haller_source(
        {
            "name": "fast-fixture",
            "weak_root": weak_root,
            "u": np.stack([-yy * radial]),
            "v": np.stack([xx * radial]),
            "xdim": coords,
            "ydim": coords,
            "mask": np.zeros((25, 25), dtype=bool),
        },
        tmp_path / "haller-fast",
        source=haller_anchors.SOURCE_TRAIN,
        backend="fast_haller",
        workers=1,
    )

    assert manifest["backend"] == "fast_haller"
    assert manifest["resolved"] == "fast_haller"
    assert manifest["cuda_used"] is False
    frame_meta = json.loads(
        (tmp_path / "haller-fast" / haller_anchors.SOURCE_TRAIN /
         "fast-fixture" / "frame0" / "anchor_meta.json").read_text(encoding="utf-8")
    )
    assert frame_meta["parameters"]["fast_global_level_count"] == 64


def test_prepare_haller_source_resume_repairs_a_corrupt_frame_and_manifest(tmp_path):
    coords = np.linspace(-3.0, 3.0, 31, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radial = np.exp(-(xx * xx + yy * yy) / (2.0 * 0.9**2))
    u = np.stack([-yy * radial] * 3)
    v = np.stack([xx * radial] * 3)
    mask = np.zeros((31, 31), dtype=bool)
    weak_root = tmp_path / "weak"
    weak_root.mkdir()
    (weak_root / "meta.json").write_text(
        json.dumps({
            "split_ranges": {"train": [0, 3]},
            "window": {"t_win": 24, "window_step": 4},
        }),
        encoding="utf-8",
    )
    source_data = {
        "name": "fixture",
        "weak_root": weak_root,
        "u": u,
        "v": v,
        "xdim": coords,
        "ydim": coords,
        "mask": mask,
    }
    output_root = tmp_path / "haller"
    initial = preparer.prepare_haller_source(
        source_data, output_root, source=haller_anchors.SOURCE_TRAIN, workers=1
    )
    frame_dir = output_root / haller_anchors.SOURCE_TRAIN / "fixture" / "frame1"
    (frame_dir / "anchor_state.npy").unlink()

    repaired = preparer.prepare_haller_source(
        source_data, output_root, source=haller_anchors.SOURCE_TRAIN, workers=1
    )
    loaded = haller_anchors.load_haller_artifact(
        frame_dir, expected_source=haller_anchors.SOURCE_TRAIN
    )

    assert repaired["frame_count"] == initial["frame_count"] == 3
    assert len(repaired["frame_artifacts"]) == 3
    assert loaded["metadata"]["frame_index"] == 1
    assert repaired["frame_artifacts"][1]["artifact_array_hashes"] == (
        loaded["metadata"]["artifact_array_hashes"]
    )


def test_prepare_all_rejects_cuda_haller_backend_until_backend_is_implemented(tmp_path):
    with pytest.raises(ValueError, match="CPU-only|CUDA Haller backend|未实现"):
        preparer.prepare_all(
            input_root=tmp_path / "input",
            output_root=tmp_path / "weak",
            haller_root=tmp_path / "haller",
            haller_backend="cuda",
        )


def test_prepare_haller_source_rejects_unknown_contour_mode():
    try:
        preparer._normalize_contour_mode("not-a-contour-mode")
    except ValueError as exc:
        assert "contour_mode" in str(exc)
    else:
        raise AssertionError("unknown contour mode must fail loudly")
