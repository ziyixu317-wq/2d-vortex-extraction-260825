"""evaluate.py 测试 —— 08 票：TTA 滑窗推理 → 网格投影 → 对比图/动画/弱定量表。"""

import json
import pathlib

import numpy as np
import pytest


# ============================================================================
# Slice 1: project_to_grid + sliding_window_patches（纯函数）
# ============================================================================

class TestProjectToGrid:
    """网格投影：累积 + 计数平均。"""

    def test_single_trace_one_cell(self):
        """单迹线 → 该迹线概率落在对应格。"""
        from evaluate import project_to_grid

        preds = np.array([0.7], dtype=np.float32)
        seeds = np.array([[0.5, 0.5]], dtype=np.float64)
        xdim = np.array([0.0, 1.0], dtype=np.float64)
        ydim = np.array([0.0, 1.0], dtype=np.float64)
        prob = project_to_grid(preds, seeds, xdim, ydim, (2, 2))

        assert prob.shape == (2, 2)
        # (0.5,0.5) in 2×2 [0,1]: gy=(0.5-0)/1=0.5, j=floor(1.0)=1
        assert prob[1, 1] == pytest.approx(0.7)
        assert prob[0, 0] == 0.0

    def test_multi_trace_same_cell_average(self):
        """多条迹线落同一格 → 均值。"""
        from evaluate import project_to_grid

        preds = np.array([0.3, 0.5, 0.7], dtype=np.float32)
        seeds = np.array([[0.1, 0.1], [0.1, 0.1], [0.9, 0.9]], dtype=np.float64)
        xdim = np.array([0.0, 1.0], dtype=np.float64)
        ydim = np.array([0.0, 1.0], dtype=np.float64)
        prob = project_to_grid(preds, seeds, xdim, ydim, (2, 2))

        # (0.1,0.1): gy=0.1, j=0 → cell (0,0); 两条 = 0.4
        assert prob[0, 0] == pytest.approx(0.4)
        # (0.9,0.9): gy=0.9, j=1 → cell (1,1); 一条 = 0.7
        assert prob[1, 1] == pytest.approx(0.7)

    def test_empty_cell_is_zero(self):
        """无迹线格 = 0。"""
        from evaluate import project_to_grid

        preds = np.array([0.8], dtype=np.float32)
        seeds = np.array([[0.0, 0.0]], dtype=np.float64)
        xdim = np.linspace(0, 2, 4, dtype=np.float64)
        ydim = np.linspace(0, 2, 4, dtype=np.float64)
        prob = project_to_grid(preds, seeds, xdim, ydim, (4, 4))

        # (0,0): gy=0, j=0
        assert prob[0, 0] == pytest.approx(0.8)
        assert prob[1, 1] == 0.0
        assert prob[3, 3] == 0.0

    def test_dtype_float32(self):
        """输出 dtype = float32。"""
        from evaluate import project_to_grid

        prob = project_to_grid(
            np.array([0.5], dtype=np.float32),
            np.array([[0.5, 0.5]], dtype=np.float64),
            np.array([0.0, 1.0]), np.array([0.0, 1.0]), (2, 2))
        assert prob.dtype == np.float32


class TestSlidingWindowPatches:
    """全场滑窗 patch 位置。"""

    def test_covers_field(self):
        from evaluate import sliding_window_patches

        patches = sliding_window_patches(100, 200, (32, 32), (16, 16))
        assert len(patches) > 0
        for y0, x0 in patches:
            assert 0 <= y0 <= 100 - 32
            assert 0 <= x0 <= 200 - 32

    def test_no_overlap_when_stride_equals_patch(self):
        from evaluate import sliding_window_patches

        patches = sliding_window_patches(64, 96, (32, 32), (32, 32))
        assert len(patches) == (64 // 32) * (96 // 32)  # 2×3=6

    def test_last_patch_touches_edge(self):
        from evaluate import sliding_window_patches

        patches = sliding_window_patches(50, 50, (32, 32), (16, 16))
        max_y0 = max(p[0] for p in patches)
        max_x0 = max(p[1] for p in patches)
        # 贴边：range(0, 50-32+1=19, 16)=[0,16]；补最后 y0=50-32=18、x0=18 → 全场覆盖
        assert max_y0 == 18
        assert max_x0 == 18


# ============================================================================
# 公共 fixtures
# ============================================================================

_MINIMAL_MODEL_YAML = """\
model:
  NAME: BaseSeg
  encoder_args:
    NAME: PathlineTransformerV0
    in_channels: 7
    PathlineGroups: 64
    KpathlinePerGroup: 4
    num_classes: 1
    num_encoder_layers: 1
    dmodel: 64
    k: 4
  criterion_args:
    NAME: BCELoss
data:
  root: __placeholder__
  patch_size: [32, 32]
  stride: [16, 16]
  t_win: 24
  window_step: 4
  groups: [8, 8]
  delta_frac: 0.05
  L: 16
  n_substeps: 4
  seed: 0
  t_scale: 0.25
  samples_per_epoch: 8
"""


def _make_tiny_model():
    import yaml
    from train_kaggle import build_model_from_config

    cfg = yaml.safe_load(_MINIMAL_MODEL_YAML)
    model = build_model_from_config(cfg)
    model.eval()
    return model


def _build_tiny_store():
    """合成 (30,40,50) 数据集 → _DatasetStore (test split)。"""
    import dataset as ds

    T, Y, X = 30, 40, 50
    rng = np.random.default_rng(123)
    u = rng.uniform(-1, 1, (T, Y, X)).astype(np.float32)
    v = rng.uniform(-1, 1, (T, Y, X)).astype(np.float32)
    xdim = np.linspace(0, 5, X, dtype=np.float64)
    ydim = np.linspace(-1, 2, Y, dtype=np.float64)
    tdim = np.linspace(0, 1, T, dtype=np.float64)
    mask2d = np.zeros((Y, X), dtype=bool)

    out_dir = pathlib.Path("outputs/eval_test_ds")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.prepare_dataset(
        u=u, v=v, xdim=xdim, ydim=ydim, tdim=tdim,
        mask=mask2d, out_dir=str(out_dir),
        split_mode="frac", train_frac=0.6, val_frac=0.0,
        t_win=24, window_step=4, patch_size=(32, 32), stride=(16, 16))
    store = ds._DatasetStore(
        str(out_dir), split="test",
        patch_size=(32, 32), stride=(16, 16),
        t_win=24, window_step=4, groups=(8, 8),
        delta_frac=0.05, L=16, n_substeps=4, seed=0)
    return store


@pytest.fixture(scope="module")
def tiny_model():
    return _make_tiny_model()


@pytest.fixture(scope="module")
def tiny_store():
    return _build_tiny_store()


# ============================================================================
# Slice 2: TTA 滑窗推理
# ============================================================================

class TestInferFrame:
    """滑窗 TTA 推理。"""

    def test_output_shape(self, tiny_model, tiny_store):
        from evaluate import infer_frame

        prob = infer_frame(tiny_store, tiny_model, frame=20, t_scale=0.25,
                           device="cpu", tta=1, seed=42)
        assert prob.shape == (tiny_store.Y, tiny_store.X)

    def test_probability_range(self, tiny_model, tiny_store):
        from evaluate import infer_frame

        prob = infer_frame(tiny_store, tiny_model, frame=20, t_scale=0.25,
                           device="cpu", tta=1, seed=42)
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0

    def test_different_seeds_differ(self, tiny_model, tiny_store):
        from evaluate import infer_frame

        p1 = infer_frame(tiny_store, tiny_model, frame=20, t_scale=0.25,
                         device="cpu", tta=1, seed=1)
        p2 = infer_frame(tiny_store, tiny_model, frame=20, t_scale=0.25,
                         device="cpu", tta=1, seed=2)
        assert not np.allclose(p1, p2, atol=1e-6)

    def test_tta_averaging(self, tiny_model, tiny_store):
        from evaluate import infer_frame

        p5 = infer_frame(tiny_store, tiny_model, frame=20, t_scale=0.25,
                         device="cpu", tta=5, seed=42)
        assert p5.min() >= 0.0
        assert p5.max() <= 1.0

    def test_fixed_seed_reproducible(self, tiny_model, tiny_store):
        from evaluate import infer_frame

        p1 = infer_frame(tiny_store, tiny_model, frame=20, t_scale=0.25,
                         device="cpu", tta=3, seed=99)
        p2 = infer_frame(tiny_store, tiny_model, frame=20, t_scale=0.25,
                         device="cpu", tta=3, seed=99)
        assert np.allclose(p1, p2, atol=1e-6)

    def test_all_solid_raises(self, tiny_model, tiny_store):
        """全场固体 → 无可用 patch → ValueError。"""
        from evaluate import infer_frame

        old = tiny_store._mask2d.copy()
        tiny_store._mask2d[:] = True
        try:
            with pytest.raises(ValueError, match="无可用 patch"):
                infer_frame(tiny_store, tiny_model, frame=20, t_scale=0.25,
                            device="cpu", tta=1, seed=42)
        finally:
            tiny_store._mask2d[:] = old


# ============================================================================
# Slice 3: 加密种子展示帧推理
# ============================================================================

class TestInferDense:
    """加密种子展示帧推理。"""

    def test_output_shape(self, tiny_model, tiny_store):
        from evaluate import infer_dense

        prob = infer_dense(tiny_store, tiny_model, frame=20, t_scale=0.25,
                           device="cpu", tta=1, seed=42, step=4)
        assert prob.shape == (tiny_store.Y, tiny_store.X)

    def test_probability_range(self, tiny_model, tiny_store):
        from evaluate import infer_dense

        prob = infer_dense(tiny_store, tiny_model, frame=20, t_scale=0.25,
                           device="cpu", tta=1, seed=42, step=4)
        assert prob.min() >= 0.0
        assert prob.max() <= 1.0

    def test_fixed_seed_reproducible(self, tiny_model, tiny_store):
        from evaluate import infer_dense

        p1 = infer_dense(tiny_store, tiny_model, frame=20, t_scale=0.25,
                         device="cpu", tta=2, seed=77, step=4)
        p2 = infer_dense(tiny_store, tiny_model, frame=20, t_scale=0.25,
                         device="cpu", tta=2, seed=77, step=4)
        assert np.allclose(p1, p2, atol=1e-6)

    def test_step_controls_spacing(self, tiny_model, tiny_store):
        from evaluate import infer_dense

        prob4 = infer_dense(tiny_store, tiny_model, frame=20, t_scale=0.25,
                            device="cpu", tta=1, seed=42, step=4)
        prob8 = infer_dense(tiny_store, tiny_model, frame=20, t_scale=0.25,
                            device="cpu", tta=1, seed=42, step=8)
        assert prob4.shape == prob8.shape

    def test_dense_seeds_coordinate_order(self):
        """dense 种子物理坐标须为 [x, y]（与 extractor 全场约定一致）。

        _integrate_batched 把 seeds[:, 0] 当 x、seeds[:, 1] 当 y（pos[:, :, 0] →
        CH_PX）。若把 [x, y] 误构为 [y, x]，则 x/y 轴交换：种子第一分量的均值
        会接近 y 轴中心而非 x 轴中心。用非重叠轴域（x∈[0,4]、y∈[10,14]）区分。
        """
        from evaluate import _dense_seeds

        Y, X = 12, 20
        xdim = np.linspace(0, 4, X, dtype=np.float64)    # x 轴 ∈ [0,4]
        ydim = np.linspace(10, 14, Y, dtype=np.float64)  # y 轴 ∈ [10,14]
        seeds = _dense_seeds(Y, X, xdim, ydim, step=2)
        assert seeds.shape[1] == 2
        x_center = 0.5 * (xdim[0] + xdim[-1])   # 2
        y_center = 0.5 * (ydim[0] + ydim[-1])   # 12
        # 第 0 分量均值应更接近 x 轴中心
        assert abs(float(np.mean(seeds[:, 0])) - x_center) <= 1.0
        # 第 1 分量均值应更接近 y 轴中心
        assert abs(float(np.mean(seeds[:, 1])) - y_center) <= 1.0


# ============================================================================
# Slice 4: 弱定量表（纯函数）
# ============================================================================

class TestComputeFrameMetrics:
    """逐帧定量指标。"""

    def test_perfect_match(self):
        from evaluate import compute_frame_metrics

        label = np.zeros((5, 5), dtype=np.uint8)
        label[1:4, 1:4] = 1
        prob = label.astype(np.float32)
        m = compute_frame_metrics(prob, label, threshold=0.5)

        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["iou"] == pytest.approx(1.0)

    def test_all_negative(self):
        from evaluate import compute_frame_metrics

        label = np.zeros((3, 3), dtype=np.uint8)
        prob = np.zeros((3, 3), dtype=np.float32)
        m = compute_frame_metrics(prob, label)

        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_all_positive(self):
        from evaluate import compute_frame_metrics

        label = np.ones((4, 4), dtype=np.uint8)
        prob = np.ones((4, 4), dtype=np.float32)
        m = compute_frame_metrics(prob, label)

        assert m["tp"] == 16
        assert m["recall"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)

    def test_half_correct(self):
        from evaluate import compute_frame_metrics

        label = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        prob = np.array([[1, 0], [0, 0]], dtype=np.float32)
        m = compute_frame_metrics(prob, label, threshold=0.5)

        assert m["tp"] == 1
        assert m["fn"] == 3
        assert m["f1"] == pytest.approx(0.4)

    def test_mask_excludes_solid(self):
        from evaluate import compute_frame_metrics

        label = np.ones((3, 3), dtype=np.uint8)
        prob = np.ones((3, 3), dtype=np.float32)
        mask2d = np.array([[True, False, False],
                           [False, False, False],
                           [False, False, False]], dtype=bool)
        m = compute_frame_metrics(prob, label, mask2d=mask2d)
        assert m["tp"] == 8
        assert m["n_fluid"] == 8

    def test_threshold(self):
        from evaluate import compute_frame_metrics

        label = np.ones((2, 2), dtype=np.uint8)
        prob = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=np.float32)
        m05 = compute_frame_metrics(prob, label, threshold=0.5)
        m08 = compute_frame_metrics(prob, label, threshold=0.8)

        assert m05["tp"] == 2
        assert m08["tp"] == 0
        assert m08["fn"] == 4


class TestFrameContinuity:
    """帧间连续性。"""

    def test_identical(self):
        from evaluate import frame_continuity

        p1 = np.array([[1, 0], [0, 1]], dtype=np.float32)
        p2 = np.array([[1, 0], [0, 1]], dtype=np.float32)
        assert frame_continuity(p1, p2) == pytest.approx(1.0)

    def test_disjoint(self):
        from evaluate import frame_continuity

        p1 = np.array([[1, 0], [0, 0]], dtype=np.float32)
        p2 = np.array([[0, 0], [0, 1]], dtype=np.float32)
        assert frame_continuity(p1, p2) == pytest.approx(0.0)

    def test_partial(self):
        from evaluate import frame_continuity

        p1 = np.array([[1, 1], [0, 0]], dtype=np.float32)
        p2 = np.array([[1, 0], [1, 0]], dtype=np.float32)
        # 交集 1, 并集 3 → 1/3
        assert frame_continuity(p1, p2) == pytest.approx(1.0 / 3.0)

    def test_all_negative_gives_one(self):
        from evaluate import frame_continuity

        p1 = np.array([[0.3, 0.3]], dtype=np.float32)
        p2 = np.array([[0.3, 0.3]], dtype=np.float32)
        assert frame_continuity(p1, p2, threshold=0.5) == pytest.approx(1.0)

    def test_sequence(self):
        from evaluate import frame_continuity_sequence

        frames = [np.array([[1, 0], [0, 0]], dtype=np.float32),
                  np.array([[1, 0], [0, 0]], dtype=np.float32),
                  np.array([[0, 0], [0, 1]], dtype=np.float32)]
        conts = frame_continuity_sequence(frames)
        assert len(conts) == 2
        assert conts[0] == pytest.approx(1.0)
        assert conts[1] == pytest.approx(0.0)


# ============================================================================
# Slice 5: 可视化
# ============================================================================

class TestMakeComparisonFigure:
    """四联对比图。"""

    def test_figure_created(self, tmp_path):
        from evaluate import make_comparison_figure

        Y, X = 40, 60
        rng = np.random.default_rng(42)
        prob = rng.uniform(0, 1, (Y, X)).astype(np.float32)
        ivd = rng.uniform(0, 3, (Y, X)).astype(np.float32)
        q_field = rng.uniform(-2, 2, (Y, X)).astype(np.float32)
        speed = rng.uniform(0, 2, (Y, X)).astype(np.float32)
        label = (ivd > 1.0).astype(np.uint8)
        xdim = np.linspace(0, 5, X, dtype=np.float64)
        ydim = np.linspace(-1, 2, Y, dtype=np.float64)

        out = tmp_path / "comparison_t100.png"
        result = make_comparison_figure(
            prob, ivd, q_field, speed, label, xdim, ydim,
            frame_idx=100, out_path=str(out))
        assert out.exists()
        assert result == str(out)

    def test_with_mask(self, tmp_path):
        from evaluate import make_comparison_figure

        Y, X = 20, 30
        rng = np.random.default_rng(42)
        prob = rng.uniform(0, 1, (Y, X)).astype(np.float32)
        ivd = rng.uniform(0, 2, (Y, X)).astype(np.float32)
        q_field = rng.uniform(-1, 1, (Y, X)).astype(np.float32)
        speed = rng.uniform(0, 2, (Y, X)).astype(np.float32)
        label = np.zeros((Y, X), dtype=np.uint8)
        mask2d = np.zeros((Y, X), dtype=bool)
        mask2d[0:5, 0:5] = True
        xdim = np.linspace(0, 3, X)
        ydim = np.linspace(0, 2, Y)

        out = tmp_path / "with_mask.png"
        make_comparison_figure(
            prob, ivd, q_field, speed, label, xdim, ydim,
            frame_idx=50, out_path=str(out), mask2d=mask2d)
        assert out.exists()


class TestMakeAnimation:
    """MP4 动画。"""

    def test_animation_created(self, tmp_path):
        from evaluate import make_animation

        Y, X = 30, 40
        rng = np.random.default_rng(42)
        frames_prob = [rng.uniform(0, 1, (Y, X)).astype(np.float32) for _ in range(5)]
        frames_ivd = [rng.uniform(0, 2, (Y, X)).astype(np.float32) for _ in range(5)]
        frames_spd = [rng.uniform(0, 2, (Y, X)).astype(np.float32) for _ in range(5)]
        xdim = np.linspace(0, 4, X, dtype=np.float64)
        ydim = np.linspace(0, 3, Y, dtype=np.float64)

        out = tmp_path / "vortex_anim.mp4"
        result = make_animation(frames_prob, frames_spd,
                                xdim, ydim, out_path=str(out), fps=5)
        # 接受 .mp4 或 .gif（ffmpeg 不可用时回退）
        result_path = pathlib.Path(result)
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_single_frame(self, tmp_path):
        from evaluate import make_animation

        Y, X = 20, 30
        prob = np.random.default_rng(42).uniform(0, 1, (Y, X)).astype(np.float32)
        ivd = np.zeros((Y, X), dtype=np.float32)
        spd = np.ones((Y, X), dtype=np.float32)
        xdim = np.linspace(0, 2, X)
        ydim = np.linspace(0, 1, Y)

        out = tmp_path / "single_frame.mp4"
        result = make_animation([prob], [spd], xdim, ydim,
                                out_path=str(out), fps=2)
        result_path = pathlib.Path(result)
        assert result_path.exists()
        assert result_path.stat().st_size > 0


# ============================================================================
# Slice 6: CLI + 端到端
# ============================================================================

class TestCLI:
    """CLI 入口。"""

    def test_help(self):
        from evaluate import main
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_missing_ckpt(self):
        from evaluate import main
        with pytest.raises(SystemExit) as exc:
            main(["--config", "nonexistent.yaml"])
        assert exc.value.code != 0


class TestEndToEndSmoke:
    """端到端冒烟（随机模型，结构验证）。"""

    def test_full_pipeline(self, tmp_path):
        """run_evaluation 全管线跑通。"""
        import yaml
        import dataset as ds
        from evaluate import run_evaluation
        from train_kaggle import build_model_from_config

        T, Y, X = 30, 40, 50
        rng = np.random.default_rng(123)
        u = rng.uniform(-1, 1, (T, Y, X)).astype(np.float32)
        v = rng.uniform(-1, 1, (T, Y, X)).astype(np.float32)
        xdim = np.linspace(0, 5, X, dtype=np.float64)
        ydim = np.linspace(-1, 2, Y, dtype=np.float64)
        tdim = np.linspace(0, 1, T, dtype=np.float64)
        mask2d = np.zeros((Y, X), dtype=bool)

        ds_dir = tmp_path / "dataset"
        ds.prepare_dataset(
            u=u, v=v, xdim=xdim, ydim=ydim, tdim=tdim,
            mask=mask2d, out_dir=str(ds_dir),
            split_mode="frac", train_frac=0.6, val_frac=0.0,
            t_win=8, window_step=4, patch_size=(32, 32), stride=(16, 16))

        cfg = yaml.safe_load(_MINIMAL_MODEL_YAML)
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["root"] = str(ds_dir)
        cfg["data"]["t_win"] = 8   # 窄窗口适配小数据集 (T=30, test=[18,30))

        model = build_model_from_config(cfg)
        model.eval()

        out_dir = tmp_path / "eval_out"
        summary = run_evaluation(
            model=model, config=cfg, data_root=str(ds_dir),
            out_dir=str(out_dir), device="cpu", tta=1,
            display_frames=[18, 22], anim_frames=range(18, 26, 2),
            t_scale=0.25, seed=42)

        assert summary is not None
        assert "frames" in summary
        assert "summary" in summary
        assert "f1_mean" in summary["summary"]
        assert (out_dir / "comparison_t0018.png").exists()
        assert (out_dir / "comparison_t0022.png").exists()
        # 动画接受 .mp4 或 .gif
        anim_files = list(out_dir.glob("vortex_animation.*"))
        assert len(anim_files) > 0
        assert (out_dir / "quantitative_table.json").exists()

        table = json.loads((out_dir / "quantitative_table.json").read_text(
            encoding="utf-8"))
        assert "frames" in table
        assert "per_dataset" in table

    def test_multi_dataset(self, tmp_path):
        """多数据集评估。"""
        import yaml
        import dataset as ds
        from evaluate import run_evaluation
        from train_kaggle import build_model_from_config

        roots = []
        for i in range(2):
            T, Y, X = 30, 40, 50
            rng = np.random.default_rng(100 + i)
            u = rng.uniform(-1, 1, (T, Y, X)).astype(np.float32)
            v = rng.uniform(-1, 1, (T, Y, X)).astype(np.float32)
            xdim = np.linspace(0, 5, X, dtype=np.float64)
            ydim = np.linspace(-1, 2, Y, dtype=np.float64)
            tdim = np.linspace(0, 1, T, dtype=np.float64)
            mask2d = np.zeros((Y, X), dtype=bool)
            d = tmp_path / f"ds{i}"
            ds.prepare_dataset(
                u=u, v=v, xdim=xdim, ydim=ydim, tdim=tdim,
                mask=mask2d, out_dir=str(d),
                split_mode="frac", train_frac=0.6, val_frac=0.0,
                t_win=8, window_step=4, patch_size=(32, 32), stride=(16, 16))
            roots.append(str(d))

        cfg = yaml.safe_load(_MINIMAL_MODEL_YAML)
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["root"] = roots
        cfg["data"]["t_win"] = 8

        model = build_model_from_config(cfg)
        model.eval()

        out_dir = tmp_path / "eval_multi"
        summary = run_evaluation(
            model=model, config=cfg, data_root=roots,
            out_dir=str(out_dir), device="cpu", tta=1,
            display_frames=[20], anim_frames=range(20, 24, 2),
            t_scale=0.25, seed=42)

        assert summary is not None
        assert len(summary.get("per_dataset", [])) == 2
        assert (out_dir / "quantitative_table.json").exists()


# ============================================================================
# Slice 7: τ 敏感性评估（09 票）
# ============================================================================

def _build_tau_tiny_store():
    """合成 (30,40,50) 数据集 → _DatasetStore (test split, t_win=8)（τ 敏感性专用）。

    窄窗口 t_win=8 使 test=[18,30) 有有效窗口（便于逐帧评估）；与 test_full_pipeline
    同用窄窗口的既有约定。IVD 用**高斯涡斑**（连续分布）——分位阈值越高 → 覆盖
    越少 → 标签涡面积占比单调下降（度量灵敏度可测），避免均匀随机场 IVD 恒 0。
    """
    import dataset as ds

    T, Y, X = 30, 40, 50
    rng = np.random.default_rng(123)
    u = rng.uniform(-1, 1, (T, Y, X)).astype(np.float32)
    v = rng.uniform(-1, 1, (T, Y, X)).astype(np.float32)
    xdim = np.linspace(0, 5, X, dtype=np.float64)
    ydim = np.linspace(-1, 2, Y, dtype=np.float64)
    tdim = np.linspace(0, 1, T, dtype=np.float64)
    mask2d = np.zeros((Y, X), dtype=bool)

    # 高斯涡斑 IVD（中心 (2.5,0.5)；宽谱 → 各分位阈值给出不同覆盖）
    xx, yy = np.meshgrid(xdim, ydim)
    bump = np.exp(-((xx - 2.5) ** 2 + (yy - 0.5) ** 2) / 3.0).astype(np.float32)
    ivd = np.repeat(bump[None], T, axis=0).astype(np.float32)

    out_dir = pathlib.Path("outputs/eval_tau_test_ds")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.prepare_dataset(
        u=u, v=v, xdim=xdim, ydim=ydim, tdim=tdim,
        mask=mask2d, ivd=ivd, out_dir=str(out_dir),
        split_mode="frac", train_frac=0.6, val_frac=0.0,
        t_win=8, window_step=4, patch_size=(32, 32), stride=(16, 16))
    return ds._DatasetStore(
        str(out_dir), split="test",
        patch_size=(32, 32), stride=(16, 16),
        t_win=8, window_step=4, groups=(8, 8),
        delta_frac=0.05, L=16, n_substeps=4, seed=0)


@pytest.fixture(scope="module")
def tau_store():
    return _build_tau_tiny_store()


class TestTauCandidatesForStore:
    """τ 候选（分位 + μ+3σ）。"""

    def test_candidate_names(self, tau_store):
        from evaluate import tau_candidates_for_store

        cfgs = tau_candidates_for_store(tau_store, percentiles=(95.0, 85.0))
        assert set(cfgs.keys()) == {"p95", "p85", "musigma"}
        for name, cfg in cfgs.items():
            # 逐时间片 dict（分位与 μ+3σ 都是 dict{片名:tau}）
            assert isinstance(cfg, dict)
            assert set(cfg.keys()) == set(
                tau_store._meta["slices"].keys())

    def test_p85_matches_compute_tau(self, tau_store):
        """p85 候选 τ 与 weak_labels.compute_tau 一致（同一公式单一源）。"""
        import weak_labels
        from evaluate import tau_candidates_for_store

        slices = {k: tuple(int(x) for x in v)
                  for k, v in tau_store._meta["slices"].items()}
        expected = weak_labels.compute_tau(
            np.asarray(tau_store._ivd_mm, dtype=np.float32),
            tau_store._mask2d, slices, percentile=85.0)
        got = tau_candidates_for_store(
            tau_store, percentiles=(85.0,), include_musigma=False)["p85"]
        assert got == expected

    def test_musigma_arithmetic(self, tau_store):
        """μ+3σ = 流体区 IVD 均值 + 3σ（各时间片）。"""
        from evaluate import tau_candidates_for_store

        slices = {k: tuple(int(x) for x in v)
                  for k, v in tau_store._meta["slices"].items()}
        mask = tau_store._mask2d
        ivd = np.asarray(tau_store._ivd_mm, dtype=np.float32)
        cfgs = tau_candidates_for_store(
            tau_store, percentiles=(), include_musigma=True)
        for name, (i0, i1) in slices.items():
            vals = np.asarray(ivd[i0:i1])[:, ~mask]
            assert cfgs["musigma"][name] == pytest.approx(
                float(vals.mean() + 3.0 * vals.std()))


class TestLabelAtCfg:
    """单帧重标。"""

    def test_matches_labeled_mask(self, tau_store):
        """_label_at_cfg 与弱标签单一口径（binary_label + filter_min_area）一致。"""
        import weak_labels
        from evaluate import _label_at_cfg

        slices = {k: tuple(int(x) for x in v)
                  for k, v in tau_store._meta["slices"].items()}
        frame = 20
        # 用 p85 候选
        import evaluate
        cfgs = evaluate.tau_candidates_for_store(
            tau_store, percentiles=(85.0,), include_musigma=False)["p85"]
        got = _label_at_cfg(tau_store, frame, cfgs, slices, 25)
        # 手工：该帧 τ → binary_label + filter_min_area
        ivd_t = np.asarray(tau_store._ivd_mm[frame], dtype=np.float64)
        tau_t = weak_labels._tau_for_frame(cfgs, frame, slices)
        expected = weak_labels.filter_min_area(
            weak_labels.binary_label(ivd_t, tau_t) & (~tau_store._mask2d),
            min_area=25)
        assert np.array_equal(got, expected)

    def test_out_of_slice_raises(self, tau_store):
        """帧不在任何时间片 → fail loud。"""
        import weak_labels
        from evaluate import _label_at_cfg

        # 直接触发 label_frame_at_cfg 的"不在任何时间片"守卫（用不覆盖帧 20 的 slices）
        ivd_t = np.asarray(tau_store._ivd_mm[20], dtype=np.float64)
        with pytest.raises(ValueError, match="不在任何时间片"):
            weak_labels.label_frame_at_cfg(
                ivd_t, tau_store._mask2d, {"test": (0, 18)}, {"test": 1.0},
                20, min_area=25)


class TestRunTauSensitivity:
    """端到端 τ 敏感性（合成 store + tiny model，结构验证）。"""

    def _run(self, tau_store, tmp_path, percentiles=(95.0, 85.0)):
        import yaml
        from evaluate import run_tau_sensitivity
        from train_kaggle import build_model_from_config

        cfg = yaml.safe_load(_MINIMAL_MODEL_YAML)
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["root"] = str(pathlib.Path(tau_store._root))
        cfg["data"]["t_win"] = 8
        cfg["data"]["f1_split"] = "test"
        model = build_model_from_config(cfg)
        model.eval()
        return run_tau_sensitivity(
            model=model, config=cfg, data_root=cfg["data"]["root"],
            out_dir=str(tmp_path / "tau"), device="cpu", tta=1,
            display_frames=[20, 22], t_scale=0.25, threshold=0.5,
            percentiles=percentiles, include_musigma=True)

    def test_rows_and_artifacts(self, tau_store, tmp_path):
        report = self._run(tau_store, tmp_path)
        assert len(report["per_dataset"]) == 1
        rows = report["per_dataset"][0]["rows"]
        assert [r["tau_cfg"] for r in rows] == ["p95", "p85", "musigma"]
        for r in rows:
            assert 0.0 <= r["f1"] <= 1.0
            assert 0.0 <= r["iou"] <= 1.0
        assert (tmp_path / "tau" / "tau_sensitivity_table.json").exists()
        assert (tmp_path / "tau" / "tau_sensitivity_report.md").exists()
        imported = json.loads(
            (tmp_path / "tau" / "tau_sensitivity_table.json").read_text(
                encoding="utf-8"))
        assert "per_dataset" in imported
        assert "global" in imported

    def test_area_ratio_monotonic(self, tau_store, tmp_path):
        """涡面积占比随 τ（分位）升高单调下降。"""
        report = self._run(tau_store, tmp_path, percentiles=(95.0, 85.0, 75.0))
        rows = report["per_dataset"][0]["rows"]
        by = {r["tau_cfg"]: r["vortex_area_ratio"] for r in rows}
        assert by["p95"] < by["p85"] < by["p75"]

    def test_continuity_constant_across_tau(self, tau_store, tmp_path):
        """帧间连续性 τ 无关（跨候选为常数）。"""
        report = self._run(tau_store, tmp_path, percentiles=(95.0, 85.0))
        rows = report["per_dataset"][0]["rows"]
        conts = {r["tau_cfg"]: report["per_dataset"][0]["continuity_mean"]
                 for r in rows}
        assert len(set(conts.values())) == 1
        assert report["per_dataset"][0]["continuity_is_tau_independent"] is True

    def test_global_summary(self, tau_store, tmp_path):
        """全局汇总行覆盖所有候选。"""
        report = self._run(tau_store, tmp_path, percentiles=(95.0, 85.0))
        gnames = [r["tau_cfg"] for r in report["global"]]
        assert gnames == ["p95", "p85", "musigma"]