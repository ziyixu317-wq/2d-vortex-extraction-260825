"""弱监督新 split/窗口契约的 TDD 测试。

这些测试只观察 split/label seam 的公开行为：按帧比例生成半开区间，
并且任何 pathline window 都必须完整落在单一 split 内。
"""

import numpy as np
import pytest

import dataset as ds


VALID_DATASETS = (
    "boussinesq",
    "cylinder2d",
    "doublegyre2d",
    "fourcenters2d",
    "jungtelziemniak2d",
    "pipedcylinder2d",
)


@pytest.mark.parametrize("dataset_name", VALID_DATASETS)
def test_weak_supervision_slices_use_floor_boundaries(dataset_name):
    """六个有效数据集共享 0/50/60/100 的半开 frame-index 语义。"""
    T = 101

    got = ds.weak_supervision_slices(T, dataset_name=dataset_name)

    assert got == {
        "train": (0, 50),
        "calibration": (50, 60),
        "test": (60, 101),
    }
    covered = np.zeros(T, dtype=bool)
    for start, end in got.values():
        covered[start:end] = True
    assert covered.all()


def test_split_windows_are_complete_and_crossing_start_fails_loudly():
    """边界前/边界处可枚举，跨界 start 明确失败。"""
    splits = ds.weak_supervision_slices(100, dataset_name="cylinder2d")
    starts = ds.window_starts(
        *splits["train"], t_win=10, step=4, dataset_name="cylinder2d",
        split_name="train", T=100,
    )

    assert starts[0] == 0
    assert starts[-1] + 10 <= 50
    assert np.all(starts + 10 <= 50)
    assert ds.validate_window_start(
        40, split_start=0, split_end=50, t_win=10,
        dataset_name="cylinder2d", split_name="train", T=100,
    ) == 40

    with pytest.raises(ValueError, match=r"cylinder2d.*T=100.*train.*\[0, 50\).*t_win=10"):
        ds.validate_window_start(
            45, split_start=0, split_end=50, t_win=10,
            dataset_name="cylinder2d", split_name="train", T=100,
        )


def test_short_split_reports_dataset_boundary_and_window():
    """split 长度不足以容纳窗口时不得静默返回空数组。"""
    with pytest.raises(
        ValueError,
        match=r"tiny-dataset.*T=100.*calibration.*\[50, 60\).*t_win=11",
    ):
        ds.window_starts(
            50, 60, t_win=11, step=1, dataset_name="tiny-dataset",
            split_name="calibration", T=100,
        )


def test_window_guard_rejects_non_integral_frame_values():
    """窗口 guard 不得通过 int 截断悄悄改变 frame 起点。"""
    with pytest.raises(ValueError, match=r"window.*integer"):
        ds.validate_window_start(
            10.5, split_start=0, split_end=50, t_win=10,
            dataset_name="non-integral", split_name="train", T=100,
        )


def test_weak_prepare_rejects_non_integral_supplied_split_boundaries(tmp_path):
    """弱监督 supplied split 的半帧边界不得被 int 截断后静默接受。"""
    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    with pytest.raises(ValueError, match=r"weak supervision split.*integer|边界.*整数"):
        ds.prepare_dataset(
            None,
            str(tmp_path / "non-integral-split"),
            u=u,
            v=v,
            ivd=ivd,
            labels=labels,
            xdim=xdim,
            ydim=ydim,
            tdim=tdim,
            dataset_name="non-integral-split",
            split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
            label_source="legacy_p85",
            slices={
                "train": (0, 50.5),
                "calibration": (50, 60),
                "test": (60, 101),
            },
            min_area=1,
            t_win=10,
            window_step=1,
        )


def _weak_fixture(T=101):
    """返回足够小但能覆盖三个 split 的 (u, v, IVD, labels, coords)。"""
    xdim = np.linspace(-1.0, 1.0, 16)
    ydim = np.linspace(-1.0, 1.0, 16)
    tdim = np.linspace(0.0, (T - 1) * 0.1, T)
    u = np.ones((T, len(ydim), len(xdim)), dtype=np.float32)
    v = np.zeros_like(u)
    ivd = np.ones_like(u)
    labels = np.zeros_like(u, dtype=np.uint8)
    return u, v, ivd, labels, xdim, ydim, tdim


def test_weak_prepare_records_split_window_feature_and_provenance(tmp_path):
    """新数据契约的 metadata 具有可审计的 split/window/source/hash 字段。"""
    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()

    meta = ds.prepare_dataset(
        None,
        str(tmp_path / "fixture"),
        u=u,
        v=v,
        ivd=ivd,
        labels=labels,
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        dataset_name="fixture",
        split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
        label_source="local_p90_p60",
        sampling_source="legacy_p85",
        loss_label_source="local_p90_p60",
        min_area=1,
        t_win=10,
        window_step=3,
    )

    assert meta["slices"] == {
        "train": [0, 50],
        "calibration": [50, 60],
        "test": [60, 101],
    }
    assert meta["split_ranges"] == meta["slices"]
    assert meta["window"]["t_win"] == 10
    assert meta["window"]["window_step"] == 3
    assert meta["window"]["complete_only"] is True
    assert meta["feature_schema"]["channels"] == [
        "px", "py", "t", "ivd", "distance", "u", "v",
    ]
    assert meta["label_source"] == "local_p90_p60"
    assert meta["label_provenance"] == {
        "field_source": "local_p90_p60",
        "sampling_source": "legacy_p85",
        "loss_source": "local_p90_p60",
    }
    assert meta["normalization_source"] == "train"
    assert meta["normalization_frozen"] is True
    assert meta["generation_version"]
    assert len(meta["generation_hash"]) == 64
    assert meta["generation_hash"] == meta["contract_hash"]


def test_weak_normalization_uses_train_only_and_rejects_other_slice(tmp_path):
    """calibration/test 的极端值不能影响新契约的冻结 normalization。"""
    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    ivd[:50] = 1.0
    ivd[50:60] = 100.0
    ivd[60:] = 1000.0
    u[:50] = 2.0
    u[50:60] = 20.0
    u[60:] = 200.0

    meta = ds.prepare_dataset(
        None,
        str(tmp_path / "train_stats"),
        u=u,
        v=v,
        ivd=ivd,
        labels=labels,
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        dataset_name="stats-fixture",
        split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
        label_source="legacy_p85",
        min_area=1,
        t_win=10,
        window_step=1,
    )
    assert meta["ivd_mu"] == pytest.approx(1.0)
    assert meta["ivd_sigma"] == pytest.approx(0.0)
    assert meta["speed_max"] == pytest.approx(2.0)

    with pytest.raises(ValueError, match=r"normalization.*train.*calibration"):
        ds.prepare_dataset(
            None,
            str(tmp_path / "bad_stats"),
            u=u,
            v=v,
            ivd=ivd,
            labels=labels,
            xdim=xdim,
            ydim=ydim,
            tdim=tdim,
            dataset_name="stats-fixture",
            split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
            label_source="legacy_p85",
            ivd_stats_slice="calibration",
            min_area=1,
            t_win=10,
            window_step=1,
        )


def test_haller_test_source_requires_evaluator_declaration(tmp_path):
    """训练/校准消费者拒绝 test GT，evaluation 必须显式声明 source。"""
    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    root = tmp_path / "test_gt"
    ds.prepare_dataset(
        None,
        str(root),
        u=u,
        v=v,
        ivd=ivd,
        labels=labels,
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        dataset_name="test-gt-fixture",
        split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
        label_source="haller_gt_test",
        min_area=1,
        patch_size=(8, 8),
        stride=(8, 8),
        t_win=10,
        window_step=2,
    )

    with pytest.raises(ValueError, match=r"train.*haller_gt_test"):
        ds.WeakLabelPathlineDataset(
            str(root), split="test", samples_per_epoch=2,
            patch_size=(8, 8), stride=(8, 8), t_win=10, window_step=2,
            groups=(2, 2), L=4,
        )
    with pytest.raises(ValueError, match=r"explicit.*haller_gt_test"):
        ds.WeakLabelPathlineDataset(
            str(root), split="test", consumer="evaluation", samples_per_epoch=2,
            patch_size=(8, 8), stride=(8, 8), t_win=10, window_step=2,
            groups=(2, 2), L=4,
        )

    evaluation_ds = ds.WeakLabelPathlineDataset(
        str(root), split="test", consumer="evaluation",
        label_source="haller_gt_test", samples_per_epoch=2,
        patch_size=(8, 8), stride=(8, 8), t_win=10, window_step=2,
        groups=(2, 2), L=4,
    )
    assert evaluation_ds.store.label_source == "haller_gt_test"


def test_weak_prepare_requires_explicit_source_instead_of_legacy_fallback(tmp_path):
    """新 split 缺少 label source 时必须失败，不能悄悄生成旧 p85 标签。"""
    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    with pytest.raises(ValueError, match=r"label_source.*legacy_p85"):
        ds.prepare_dataset(
            None,
            str(tmp_path / "missing_source"),
            u=u,
            v=v,
            ivd=ivd,
            labels=labels,
            xdim=xdim,
            ydim=ydim,
            tdim=tdim,
            dataset_name="missing-source-fixture",
            split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
            min_area=1,
            t_win=10,
            window_step=1,
        )


def test_weak_prepare_rejects_p85_as_nonlegacy_formal_loss(tmp_path):
    """p85 可以记录为 sampling source，但不能冒充 W1 formal loss source。"""
    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    with pytest.raises(ValueError, match=r"legacy_p85.*formal.*loss"):
        ds.prepare_dataset(
            None,
            str(tmp_path / "p85_loss"),
            u=u,
            v=v,
            ivd=ivd,
            labels=labels,
            xdim=xdim,
            ydim=ydim,
            tdim=tdim,
            dataset_name="p85-loss-fixture",
            split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
            label_source="local_p90_p60",
            sampling_source="legacy_p85",
            loss_label_source="legacy_p85",
            min_area=1,
            t_win=10,
            window_step=1,
        )


def test_each_split_consumes_frozen_train_normalization_and_reports_window_meta(tmp_path):
    """三个消费者都读取同一份 train 统计，window metadata 不丢 provenance。"""
    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    ivd[:50] = 1.0
    ivd[50:60] = 100.0
    ivd[60:] = 1000.0
    u[:50] = 2.0
    u[50:60] = 20.0
    u[60:] = 200.0
    root = tmp_path / "frozen"
    ds.prepare_dataset(
        None,
        str(root),
        u=u,
        v=v,
        ivd=ivd,
        labels=labels,
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        dataset_name="frozen-fixture",
        split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
        label_source="local_p90_p60",
        sampling_source="legacy_p85",
        loss_label_source="local_p90_p60",
        min_area=1,
        patch_size=(8, 8),
        stride=(8, 8),
        t_win=10,
        window_step=2,
    )

    stores = [
        ds.WeakLabelPathlineDataset(
            str(root), split=split, consumer=consumer, samples_per_epoch=2,
            patch_size=(8, 8), stride=(8, 8), t_win=10, window_step=2,
            L=4, groups=(2, 2),
        ).store
        for split, consumer in (
            ("train", "train"),
            ("calibration", "calibration"),
            ("test", "evaluation"),
        )
    ]
    assert {store.ivd_mu for store in stores} == {1.0}
    assert {store.ivd_sigma for store in stores} == {0.0}
    assert {store.speed_max for store in stores} == {2.0}
    for store, frame in zip(stores, (0, 50, 60)):
        info = store.window_metadata(frame)
        assert info["split_name"] in {"train", "calibration", "test"}
        assert info["frame_start"] == frame
        assert info["frame_end"] == frame + 10
        assert info["normalization_source"] == "train"
        assert info["label_source"] == "local_p90_p60"
        assert len(info["generation_hash"]) == 64


def test_evaluation_store_must_declare_test_haller_source(tmp_path):
    """evaluation adapter 只能从显式 config source 读取 test Haller GT。"""
    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    root = tmp_path / "eval_adapter"
    ds.prepare_dataset(
        None,
        str(root),
        u=u,
        v=v,
        ivd=ivd,
        labels=labels,
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        dataset_name="eval-adapter-fixture",
        split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
        label_source="legacy_p85",
        min_area=1,
        patch_size=(8, 8),
        stride=(8, 8),
        t_win=10,
        window_step=2,
    )
    from evaluate import _make_single_store

    common = {
        "patch_size": [8, 8],
        "stride": [8, 8],
        "t_win": 10,
        "window_step": 2,
        "groups": [2, 2],
        "L": 4,
    }
    with pytest.raises(ValueError, match=r"explicit.*haller_gt_test"):
        _make_single_store(str(root), "test", {**common})

    store = _make_single_store(
        str(root),
        "test",
        {
            **common,
            "evaluation_label_source": "haller_gt_test",
            "haller_test_root": str(tmp_path / "haller_artifacts"),
            "sampling_label_source": "legacy_p85",
        },
    )
    assert store.consumer == "evaluation"
    assert store.label_source == "legacy_p85"

    from evaluate import _extract_one_sample
    with pytest.raises(ValueError, match=r"eval-adapter-fixture.*T=101.*test.*t_win=10"):
        _extract_one_sample(
            store, 0, 0, 55, t_scale=0.25, rng_base=0)

    from evaluate import _dense_extract
    with pytest.raises(ValueError, match=r"eval-adapter-fixture.*T=101.*test.*t_win=10"):
        _dense_extract(
            store, 55, np.zeros((1, 2), dtype=np.float64), t_scale=0.25)


def test_weak_evaluation_cannot_bypass_explicit_test_haller_source(tmp_path):
    """weak test evaluation 不能以 legacy/p90-p60 标签替代显式 Haller GT。"""
    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    root = tmp_path / "eval_without_haller_gt"
    ds.prepare_dataset(
        None,
        str(root),
        u=u,
        v=v,
        ivd=ivd,
        labels=labels,
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        dataset_name="eval-without-haller-gt",
        split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
        label_source="local_p90_p60",
        min_area=1,
        patch_size=(8, 8),
        stride=(8, 8),
        t_win=10,
        window_step=2,
    )
    from evaluate import _make_single_store

    with pytest.raises(ValueError, match=r"evaluation.*haller_gt_test|test.*Haller"):
        _make_single_store(
            str(root),
            "test",
            {
                "patch_size": [8, 8],
                "stride": [8, 8],
                "t_win": 10,
                "window_step": 2,
                "groups": [2, 2],
                "L": 4,
            },
        )


def test_weak_loader_rejects_window_config_drift(tmp_path):
    """加载器不能用不同窗口参数静默重解释既有弱监督产物。"""
    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    root = tmp_path / "window_drift"
    ds.prepare_dataset(
        None,
        str(root),
        u=u,
        v=v,
        ivd=ivd,
        labels=labels,
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        dataset_name="window-drift-fixture",
        split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
        label_source="local_p90_p60",
        min_area=1,
        patch_size=(8, 8),
        stride=(8, 8),
        t_win=10,
        window_step=2,
    )
    with pytest.raises(ValueError, match=r"window.*metadata.*t_win"):
        ds.WeakLabelPathlineDataset(
            str(root), split="train", samples_per_epoch=2,
            patch_size=(8, 8), stride=(8, 8), t_win=8, window_step=2,
        )


def test_weak_loader_rejects_tampered_generation_hash(tmp_path):
    """generation/contract hash 不一致时加载必须失败。"""
    import json

    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    root = tmp_path / "tampered"
    ds.prepare_dataset(
        None,
        str(root),
        u=u,
        v=v,
        ivd=ivd,
        labels=labels,
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        dataset_name="tampered-fixture",
        split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
        label_source="legacy_p85",
        min_area=1,
        patch_size=(8, 8),
        stride=(8, 8),
        t_win=10,
        window_step=2,
    )
    meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["generation_hash"] = "0" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match=r"generation.*hash"):
        ds.WeakLabelPathlineDataset(
            str(root), split="train", samples_per_epoch=2,
            patch_size=(8, 8), stride=(8, 8), t_win=10, window_step=2,
        )


def test_weak_loader_rejects_non_integral_metadata_boundary(tmp_path):
    """metadata 边界不能通过 int 截断绕过 split/hash 合同。"""
    import json

    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    root = tmp_path / "tampered-boundary"
    ds.prepare_dataset(
        None,
        str(root),
        u=u,
        v=v,
        ivd=ivd,
        labels=labels,
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        dataset_name="tampered-boundary-fixture",
        split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
        label_source="legacy_p85",
        min_area=1,
        patch_size=(8, 8),
        stride=(8, 8),
        t_win=10,
        window_step=2,
    )
    meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["split_ranges"]["train"][1] = 50.5
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match=r"boundary|边界|integer|整数"):
        ds.WeakLabelPathlineDataset(
            str(root), split="train", samples_per_epoch=2,
            patch_size=(8, 8), stride=(8, 8), t_win=10, window_step=2,
        )


def test_weak_loader_rejects_tampered_frozen_normalization(tmp_path):
    """冻结统计量本身也属于不可变契约，不能只改顶层数值。"""
    import json

    u, v, ivd, labels, xdim, ydim, tdim = _weak_fixture()
    root = tmp_path / "tampered_norm"
    ds.prepare_dataset(
        None,
        str(root),
        u=u,
        v=v,
        ivd=ivd,
        labels=labels,
        xdim=xdim,
        ydim=ydim,
        tdim=tdim,
        dataset_name="tampered-norm-fixture",
        split_mode=ds.WEAK_SUPERVISION_SPLIT_MODE,
        label_source="legacy_p85",
        min_area=1,
        patch_size=(8, 8),
        stride=(8, 8),
        t_win=10,
        window_step=2,
    )
    meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["ivd_mu"] = 123.0
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match=r"normalization|generation.*hash"):
        ds.WeakLabelPathlineDataset(
            str(root), split="train", samples_per_epoch=2,
            patch_size=(8, 8), stride=(8, 8), t_win=10, window_step=2,
        )
