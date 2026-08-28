"""05 票：数据集类（dataset.py）测试 —— 数据准备缝（主验收接缝）。

领域词汇（HANDOFF §4/§6，唯一权威）：
- 时间划分 train [0,10]s / val (10,12.5] / test (12.5,15]：帧 0-1000 / 1000-1250 /
  1250-1500（闭包口径 DEFAULT_SLICES (0,1001)/(1001,1251)/(1251,1501)，无时间泄漏）；
- patch 32×32 stride 16、窗口 T_win=24 帧、窗口起点步长 4 帧；
- 7 通道 = [px, py, t, ivd, distance(距种子), u, v]；归一化：px,py → patch 内 [-1,1]；
  t → [0,1]×t_scale（默认 0.25）；ivd 标准化（train 流体区 z-score）；distance 用
  归一化坐标；u,v ÷ 全局最大速度；
- 返回 ((dummy_field, pathlines), labels) 匹配模型输入（PathlineTransformerV0）；
- 标签 = 重播种后种子格（label_field，含 5×5 面积过滤与固体强制 0）；
- 每 epoch 40000 样本（下限 20000）、50% 正样本过采样（正样本 = patch 内存在
  ≥1 条涡迹线；判据与 weak_labels.patch_positive_map 单公式共用）。

期望值来源（独立于实现）：
- 归一化公式 = 规格字面量（已知常数注入，逐通道断言）；
- 合成 Rankine 涡场由本测试直接构造（解析涡量、τ 为已知字面量 2.0）；
- 真实数据断言对照 HANDOFF §2 已核实事实（仅在本机存在数据集时运行）。
"""

import pathlib
import time

import numpy as np
import pytest

import extractor
import geometry
import weak_labels

# ---------------------------------------------------------------- 合成数据工具

# 合成场：Rankine 涡（涡心偏置右上 → 只有一个 patch 位置为正区）
SYNTH_OMEGA = 6.0
SYNTH_RC = 0.22
SYNTH_TAU = 2.0              # 已知字面量：标签块稳定（probe 实测）


def synth_grid(Y=48, X=96, T=40, dt=0.05):
    """与合成流场配套的物理坐标（等距，域 [-1,1]²）。"""
    xdim = np.linspace(-1.0, 1.0, X)
    ydim = np.linspace(-1.0, 1.0, Y)
    tdim = np.linspace(0.0, (T - 1) * dt, T)
    return xdim, ydim, tdim


def synth_field(xdim, ydim, tdim, cx=0.75, cy=0.5):
    """Rankine 涡 + 轻微时间依赖（u 通道），为可解析 IVD 的合成场。

    Von Kármán 特征：涡心 → ω=2Ω 旋转主导；ω 突变壳 → IVD 高值壳区。
    """
    T = len(tdim)
    tt, yy, xx = np.meshgrid(tdim, ydim, xdim, indexing="ij")
    r = np.hypot(xx - cx, yy - cy) + 1e-12
    vtheta = SYNTH_OMEGA * r * (r <= SYNTH_RC) + SYNTH_OMEGA * SYNTH_RC ** 2 / r * (r > SYNTH_RC)
    u = -vtheta * (yy - cy) / r + 0.1 * np.sin(2 * np.pi * (tt / tdim[-1]))
    v = vtheta * (xx - cx) / r
    return np.asarray(u, dtype=np.float32), np.asarray(v, dtype=np.float32)


def synth_prepared(root, T=40, percentile=95.0):
    """构造合成数据集（prepare_dataset 的输入侧）：返回 (u, v, xdim, ydim, tdim)。"""
    xdim, ydim, tdim = synth_grid(T=T)
    u, v = synth_field(xdim, ydim, tdim)
    return u, v, xdim, ydim, tdim


# ================================================================ 切片 1：时间划分（验收 2）

class TestTimeSlices:
    """时间划分无泄漏：train/val/test 帧区间互斥且各自窗口完全在片内。"""

    def test_default_slices_cover_all_frames(self):
        """DEFAULT_SLICES 全覆盖 1501 帧（帧 0-1500 恰好被三片瓜分，无泄漏）。"""
        slices = weak_labels.DEFAULT_SLICES
        covered = np.zeros(1501, dtype=bool)
        for i0, i1 in slices.values():
            covered[i0:i1] = True
        assert covered.all()

    def test_window_starts_within_slice(self):
        """窗口起点 ∈ [i0, i1−t_win]（窗口 24 帧完全在片内，不跨片）。"""
        from dataset import window_starts
        for name, (i0, i1) in weak_labels.DEFAULT_SLICES.items():
            starts = window_starts(i0, i1, t_win=24, step=4)
            assert starts.min() >= i0
            assert starts.max() + 24 <= i1
            assert np.all(np.diff(starts) == 4)

    def test_window_frames_mutually_exclusive(self):
        """三片窗口帧集合互斥（train 窗口帧 ≤1000 < val 帧 1001+ ≤ 1250 < test 帧）。"""
        from dataset import window_starts
        spans = {}
        for name, (i0, i1) in weak_labels.DEFAULT_SLICES.items():
            starts = window_starts(i0, i1, t_win=24, step=4)
            frames = np.concatenate([s + np.arange(24) for s in starts])
            spans[name] = set(int(f) for f in frames)
        assert not (spans["train"] & spans["val"])
        assert not (spans["val"] & spans["test"])
        assert not (spans["train"] & spans["test"])


class TestFractionSlices:
    """按时间 60/40 划分（票 07 延伸：多数据集各数据集帧前 60% 训 / 后 40% 测）。

    与 DEFAULT_SLICES（绝对秒数）不同：多数据集帧数/时长各异（512~2001 帧、
    t∈[0,20] 且 jung telziemniak t 从 1.107 起）——绝对秒数不通用，按帧比例划分。
    """

    def test_frac_slices_60_40_contiguous_cover(self):
        """1501 帧：train [0,900)、test [900,1501)（floor(1501×0.6)=900）；全覆盖无泄漏。"""
        from dataset import fraction_slices
        s = fraction_slices(1501, train_frac=0.6)
        assert s == {"train": (0, 900), "test": (900, 1501)}
        covered = np.zeros(1501, dtype=bool)
        for i0, i1 in s.values():
            covered[i0:i1] = True
        assert covered.all()

    def test_frac_slices_windows_disjoint(self):
        """窗口（T_win=24）帧集合三片互斥且各自完全在片内（无时间泄漏）。"""
        from dataset import fraction_slices, window_starts
        s = fraction_slices(1501, train_frac=0.5, val_frac=0.1)
        spans = {}
        for name, (i0, i1) in s.items():
            starts = window_starts(i0, i1, t_win=24, step=4)
            frames = set()
            for st in starts:
                frames |= set(int(f) for f in np.arange(st, st + 24))
            spans[name] = frames
        assert not (spans["train"] & spans["val"])
        assert not (spans["val"] & spans["test"])
        assert not (spans["train"] & spans["test"])

    def test_frac_slices_with_val_literals(self):
        """带 val（50/10/40）：i1=floor(1501×0.5)=750、i2=floor(1501×0.6)=900。"""
        from dataset import fraction_slices
        s = fraction_slices(1501, train_frac=0.5, val_frac=0.1)
        assert s == {"train": (0, 750), "val": (750, 900), "test": (900, 1501)}

    def test_frac_slices_small_T_and_nyquist(self):
        """小数据集：512 帧 → train (0,307)、test (307,512)（floor 512×0.6=307.2→307）。"""
        from dataset import fraction_slices
        assert fraction_slices(512, train_frac=0.6) == {
            "train": (0, 307), "test": (307, 512)}

    def test_prepare_dataset_frac_mode(self, tmp_path):
        """prepare_dataset split_mode=frac：T=48 → train (0,28)、test (28,48)
        （floor(48×0.6)=28.8→28；τ 与标签沿用 train/test 逐片口径）。"""
        import dataset as ds
        u, v, xdim, ydim, tdim = synth_prepared(tmp_path / "p", T=48)
        meta = ds.prepare_dataset(None, str(tmp_path / "d"), u=u, v=v, xdim=xdim,
                                  ydim=ydim, tdim=tdim, split_mode="frac")
        assert meta["slices"] == {"train": [0, 28], "test": [28, 48]}
        assert meta["split_mode"] == "frac"
        assert set(meta["taus"]) == {"train", "test"}

    def test_frac_slices_invalid_params(self):
        """非法参数 fail loud：train_frac 越界、val_frac 越界、留出为空。"""
        from dataset import fraction_slices
        with pytest.raises(ValueError):
            fraction_slices(100, train_frac=0.0)
        with pytest.raises(ValueError):
            fraction_slices(100, train_frac=1.2)
        with pytest.raises(ValueError):
            fraction_slices(100, train_frac=0.6, val_frac=0.6)   # 留出为空
        with pytest.raises(ValueError):
            fraction_slices(100, train_frac=0.6, val_frac=-0.1)


class TestPatchLocations:
    """patch 位置网格：完全落在帧内、按 (y 外, x 内) 序。"""

    def test_locations_inside_frame(self):
        from dataset import patch_locations
        locs = patch_locations(48, 96, patch_size=(32, 32), stride=(16, 16))
        assert len(locs) == 2 * 5                     # (48-32)//16+1=2, (96-32)//16+1=5
        for y0, x0 in locs:
            assert 0 <= y0 <= 48 - 32 and 0 <= x0 <= 96 - 32
        assert locs[0] == (0, 0)
        assert locs[1] == (0, 16)


# ================================================================ 切片 2：7 通道归一化（验收 3）

class TestNormalizePathlines:
    """归一化口径（spec Implementation Decisions）：t→[0,1]×t_scale；ivd z-score；
    distance 归一化坐标；u,v ÷ 全局最大速度；px/py 保持 patch 归一化。"""

    def test_t_scale_and_uv_speed_max(self):
        from dataset import normalize_pathlines
        L, K = 4, 8
        raw = np.zeros((L, K, 7), dtype=np.float32)
        raw[:, :, extractor.CH_T] = np.linspace(3.0, 3.6, L)[:, None]
        raw[:, :, extractor.CH_U] = 8.0
        raw[:, :, extractor.CH_V] = -6.0
        seeds = np.zeros((K, 2))
        geo = {"cx": 1.0, "cy": 2.0, "hx": 3.0, "hy": 4.0}
        out = normalize_pathlines(raw, seeds, geo, t0=3.0, t_span=0.6,
                                  t_scale=0.25, ivd_mu=1.0, ivd_sigma=2.0,
                                  speed_max=10.0)
        # t: (t−t0)/t_span×t_scale ∈ [0, 0.25]
        assert out[:, 0, extractor.CH_T] == pytest.approx(np.linspace(0.0, 0.25, L))
        # u,v ÷ speed_max（u=8→0.8, v=−6→−0.6）
        assert out[:, :, extractor.CH_U] == pytest.approx(0.8)
        assert out[:, :, extractor.CH_V] == pytest.approx(-0.6)

    def test_ivd_zscore(self):
        from dataset import normalize_pathlines
        L, K = 4, 8
        raw = np.zeros((L, K, 7), dtype=np.float32)
        raw[:, :, extractor.CH_IVD] = 7.0
        seeds = np.zeros((K, 2))
        geo = {"cx": 0.0, "cy": 0.0, "hx": 1.0, "hy": 1.0}
        out = normalize_pathlines(raw, seeds, geo, t0=0.0, t_span=1.0,
                                  t_scale=0.25, ivd_mu=1.0, ivd_sigma=2.0,
                                  speed_max=1.0)
        assert out[:, :, extractor.CH_IVD] == pytest.approx(3.0)   # (7−1)/2

    def test_distance_normalized_coords(self):
        """distance = 归一化坐标下距（重播种后）种子的距离：hypot(px−sx_n, py−sy_n)。"""
        from dataset import normalize_pathlines
        L, K = 3, 2
        raw = np.zeros((L, K, 7), dtype=np.float32)
        # 点 A 在 (px,py)=(0.5, 0.0)；点 B 在 (0.0, 0.0)（即归一化原点）
        raw[:, 0, extractor.CH_PX] = 1.0
        raw[:, 1, extractor.CH_PX] = 0.0
        raw[:, 0, extractor.CH_PY] = 0.0
        raw[:, 1, extractor.CH_PY] = 1.0
        # 种子物理坐标：cx=0, cy=0, hx=1, hy=1 → 种子归一化 (0.3, −0.2) 与 (−0.3, 0.2)
        seeds = np.array([[0.3, -0.2], [-0.3, 0.2]])
        geo = {"cx": 0.0, "cy": 0.0, "hx": 1.0, "hy": 1.0}
        out = normalize_pathlines(raw, seeds, geo, t0=0.0, t_span=1.0,
                                  t_scale=0.25, ivd_mu=0.0, ivd_sigma=1.0,
                                  speed_max=1.0)
        # 点 0: (1.0−0.3, 0−(−0.2)) → hypot(0.7, 0.2)
        assert out[0, 0, extractor.CH_DIST] == pytest.approx(np.hypot(0.7, 0.2))
        assert out[0, 1, extractor.CH_DIST] == pytest.approx(np.hypot(0.3, -0.8))


# ================================================================ 切片 3：prepare_dataset（memmap 预计算）

class TestPrepareDataset:
    """prepare_dataset：u/v/ivd/label memmap + meta.json（含 τ、speed_max、IVD 统计）。"""

    def test_prepare_writes_memmap_and_meta(self, tmp_path):
        import dataset as ds
        u, v, xdim, ydim, tdim = synth_prepared(tmp_path)
        meta = ds.prepare_dataset(
            None, str(tmp_path / "ds"), u=u, v=v, xdim=xdim, ydim=ydim, tdim=tdim,
            taus={"train": SYNTH_TAU})
        root = pathlib.Path(tmp_path) / "ds"
        for f in ("u.npy", "v.npy", "ivd.npy", "label_field.npy", "mask.npy", "meta.json"):
            assert (root / f).exists(), f
        # memmap 数值与输入一致
        u_mm = np.load(root / "u.npy", mmap_mode="r")
        assert np.allclose(np.asarray(u_mm[0]), u[0], atol=1e-6)
        # meta 字段
        assert meta["shape"] == [len(tdim), len(ydim), len(xdim)]
        assert meta["taus"] == {"train": SYNTH_TAU}
        assert np.isfinite(meta["speed_max"]) and meta["speed_max"] > 0
        assert np.isfinite(meta["ivd_mu"]) and np.isfinite(meta["ivd_sigma"])

    def test_ivd_stats_train_fluid_only(self, tmp_path):
        """ivd 标准化统计 = train 片流体区 IVD 的 μ/σ（独立 numpy 表达式验证）。"""
        import dataset as ds
        import weak_labels as wl
        u, v, xdim, ydim, tdim = synth_prepared(tmp_path)
        meta = ds.prepare_dataset(
            None, str(tmp_path / "ds2"), u=u, v=v, xdim=xdim, ydim=ydim, tdim=tdim,
            taus={"train": SYNTH_TAU})
        ivd = wl.compute_ivd(u, v, xdim, ydim, mask=None).astype(np.float32)
        want_mu = float(ivd[0:40].mean())
        want_sigma = float(ivd[0:40].std())
        assert meta["ivd_mu"] == pytest.approx(want_mu, rel=1e-5)
        assert meta["ivd_sigma"] == pytest.approx(want_sigma, rel=1e-5)

    def test_label_field_matches_tau(self, tmp_path):
        """标签场 = IVD≥τ（面积过滤后）；合成场有正块（>25 格连通域）。"""
        import dataset as ds
        u, v, xdim, ydim, tdim = synth_prepared(tmp_path)
        ds.prepare_dataset(None, str(tmp_path / "ds3"), u=u, v=v, xdim=xdim, ydim=ydim,
                           tdim=tdim, taus={"train": SYNTH_TAU})
        lab = np.load(tmp_path / "ds3" / "label_field.npy")
        assert lab.dtype == np.uint8
        pos_frac = float(lab[0].mean())
        assert pos_frac > 0.005                          # 涡壳区成块
        assert pos_frac < 0.5
        # 正块面积 ≥ 25（5×5 面积过滤）：每个非零连通块 ≥25 格
        labels2d, n = geometry.label_components(lab[0].astype(bool))
        sizes = np.bincount(labels2d.ravel(), minlength=n + 1)[1:]
        if sizes.size:
            assert (sizes >= 25).all(), f"存在面积 <25 的标签块: {sizes.min()}"


# ================================================================ 切片 4：WeakLabelPathlineDataset（主缝）

class TestWeakLabelDataset:
    """数据集类：((dummy_field, pathlines), labels)；形状/标签一致性/过采样/性能。"""

    @staticmethod
    def make_ds(tmp_path, seed=0, samples_per_epoch=200, tag="ds4"):
        import dataset as ds
        u, v, xdim, ydim, tdim = synth_prepared(tmp_path)
        root = tmp_path / tag
        ds.prepare_dataset(None, str(root), u=u, v=v, xdim=xdim, ydim=ydim,
                           tdim=tdim, taus={"train": SYNTH_TAU})
        return ds.WeakLabelPathlineDataset(str(root), split="train",
                                           samples_per_epoch=samples_per_epoch,
                                           seed=seed)

    def test_sample_shape_matches_model_input(self, tmp_path):
        """样本形状：dummy_field (1,1,1,1)（参考口径）；pathlines (16,256,7) float32
        有限无 NaN；labels (256,) ∈{0,1}（模型输入接口连通）。"""
        d = self.make_ds(tmp_path)
        d.set_epoch(0)
        (dummy, pathlines), labels = d[0]
        assert np.asarray(dummy).shape == (1, 1, 1, 1)
        assert pathlines.shape == (16, 256, 7)
        assert pathlines.dtype == np.float32
        assert np.isfinite(pathlines).all()
        assert not (pathlines == -1000.0).any()
        assert labels.shape == (256,)
        assert set(np.unique(labels)) <= {0.0, 1.0}

    def test_labels_match_seed_ivd_threshold(self, tmp_path):
        """验收 3：标签与种子点 IVD 阈值判定一致——label==1 ⇒ 重播种后种子处
        IVD ≥ τ（τ=taus[train]，label_field 由 IVD≥τ 派生，方向性无假正）。"""
        import dataset as ds
        d = self.make_ds(tmp_path)
        d.set_epoch(0)
        (_, _), labels = d[0]
        combo = d.pool_positive[0] if len(d.pool_positive) else d.pool_negative[0]
        py, px, frame = combo
        seeds = d.seeds_for(py, px, frame)
        meta = ds.load_dataset_meta(d._root)
        ivd = np.asarray(np.load(pathlib.Path(d._root) / "ivd.npy", mmap_mode="r")[frame],
                         dtype=np.float64)
        xdim = np.asarray(meta["xdim"], dtype=np.float64)
        ydim = np.asarray(meta["ydim"], dtype=np.float64)
        dx, dy = xdim[1] - xdim[0], ydim[1] - ydim[0]
        j = np.clip(np.rint((seeds[:, 1] - ydim[0]) / dy).astype(int), 0, len(ydim) - 1)
        i = np.clip(np.rint((seeds[:, 0] - xdim[0]) / dx).astype(int), 0, len(xdim) - 1)
        seed_ivd = ivd[j, i]
        tau = float(meta["taus"]["train"])
        positive = labels > 0.0
        if positive.any():
            assert np.all(seed_ivd[positive] >= tau - 1e-9)

    def test_positive_fraction_oversampling(self, tmp_path):
        """验收 4：正样本占比≈50%（过采样生效；正样本 = 样本含 ≥1 条涡迹线，
        即标签 any>0）。合成场正池 100% 正、负池 100% 负（无固体无重播种）→
        实测正比例 = 50% ± 采样波动（120 样本 σ≈4.6%）；区间 [0.35, 0.65]
        足以捕获机制失效（如未过采样 → ≈20 组合比例或 0）。"""
        d = self.make_ds(tmp_path, samples_per_epoch=300)
        d.set_epoch(0)
        fracs = []
        for idx in range(120):
            (_, _), labels = d[idx]
            fracs.append(float((labels > 0.5).any()))
        mean_frac = float(np.mean(fracs))
        assert 0.35 <= mean_frac <= 0.65, f"正样本占比 {mean_frac:.3f} 偏离 50% 过采样"

    def test_pool_stays_within_split(self, tmp_path):
        """样本池只在 split 片内（窗口不跨片；train 片 = 帧 [0,1000] 闭包窗口）。"""
        import dataset as ds
        u, v, xdim, ydim, tdim = synth_prepared(tmp_path)
        root = tmp_path / "ds5"
        ds.prepare_dataset(None, str(root), u=u, v=v, xdim=xdim, ydim=ydim,
                           tdim=tdim, taus={"train": SYNTH_TAU})
        d = ds.WeakLabelPathlineDataset(str(root), split="train",
                                        samples_per_epoch=10, seed=1)
        i0, i1 = weak_labels.DEFAULT_SLICES["train"]
        t_win = d.t_win
        assert all(i0 <= f and f + t_win <= min(i1, len(tdim))
                   for (_py, _px, f) in d.pool_negative + d.pool_positive), \
            "样本池组合窗口越出 train 片"

    def test_sample_time_under_5ms(self, tmp_path):
        """验收 1：单样本生成 <5ms（规格目标）。

        用户已确认（2026-08-25）：时间性能不纠结，能跑即可——本测试以降级判据
        （<1s 冒烟上限 + 记录实测中位数）守护"不出现数量级回归"，实测值（~38ms，
        on-the-fly numpy 批量化）记入票完成记录与 HANDOFF §11。
        """
        d = self.make_ds(tmp_path, samples_per_epoch=50)
        d.set_epoch(0)
        for _ in range(5):                                # 预热（页缓存/首载）
            d[0]
        times = []
        for idx in range(15):
            t0 = time.perf_counter()
            d[idx]
            times.append((time.perf_counter() - t0) * 1000.0)
        median = float(np.median(times))
        assert median < 1000.0, f"单样本生成中位数 {median:.2f}ms（回归超 1s 冒烟上限）"

    def test_reproducible_same_seed(self, tmp_path):
        """同 seed 重建 dataset → 同样本序列（set_epoch 后顺序确定、样本确定）。"""
        d1 = self.make_ds(tmp_path, seed=42, tag="ds4_rep1")
        d2 = self.make_ds(tmp_path, seed=42, tag="ds4_rep2")
        d1.set_epoch(0)
        d2.set_epoch(0)
        for idx in range(5):
            (_, p1), l1 = d1[idx]
            (_, p2), l2 = d2[idx]
            assert np.array_equal(p1, p2) and np.array_equal(l1, l2)

    def test_time_varying_channels_not_frozen(self, tmp_path):
        """时变语义守护（Spec 轴审查核心缺陷的回归）：窗口切片场配窗口 tdim →
        样本 u 通道沿 t 通道演化（非冻结在窗口末帧）。传全场 tdim 的错配
        曾令时间映射 clamp 到窗口末帧，u/v/ivd 通道与迹线积分全部冻结。"""
        import dataset as ds
        xdim, ydim, tdim = synth_grid(T=40)
        tt, yy, xx = np.meshgrid(tdim, ydim, xdim, indexing="ij")
        u = (0.1 + 0.05 * tt).astype(np.float32)      # 纯时变平流（弱：15 步不越界）
        v = np.zeros_like(u)
        labels = np.zeros((len(tdim), len(ydim), len(xdim)), dtype=np.uint8)
        labels[0, 0:5, 0:5] = 1                       # 正块（≥25 格）→ 正池非空
        root = tmp_path / "ds_tv"
        ds.prepare_dataset(None, str(root), u=u, v=v, xdim=xdim, ydim=ydim,
                           tdim=tdim, taus={"train": SYNTH_TAU}, labels=labels)
        d = ds.WeakLabelPathlineDataset(str(root), split="train",
                                        samples_per_epoch=10, seed=0)
        d.set_epoch(0)
        checked = 0
        for idx in range(5):
            ((_, pl), _) = d[idx]
            for k in range(256):
                ts = pl[:, k, extractor.CH_T]
                us = pl[:, k, extractor.CH_U]
                # 纯平流时变场：末步 u 应高于首步（首末差>0 即时变生效；
                # 截断迹线末点重复 → 末点仍为窗口后段物理时刻）
                if ts[-1] - ts[0] > 0.1 * (ts[-1] + 1e-12) and us[-1] > us[0]:
                    checked += 1
                    break
        assert checked >= 3, "样本 u 通道未沿 t 演化（时变冻结回归）"

    def test_forward_compatible_with_model(self, tmp_path):
        """模型缝联测：dataset 输出喂 PathlineTransformerV0 前向 → (B, 256)。"""
        import torch
        from vendor.DeepUtils.models import build_model_from_cfg
        d = self.make_ds(tmp_path)
        d.set_epoch(0)
        (dummy, pathlines), labels = d[0]
        cfg = {
            "NAME": "PathlineTransformerV0",
            "in_channels": 7,
            "PathlineGroups": 64,
            "KpathlinePerGroup": 4,
        }
        model = build_model_from_cfg(cfg)
        model.eval()
        with torch.no_grad():
            out = model((torch.zeros(1, 1, 1, 1), torch.from_numpy(pathlines[None])))
        assert out.shape == (1, 256)
        assert out.dtype == torch.float32


# ================================================================ 切片 5：不可用 patch 过滤（回归）

class TestUnusablePatchFilter:
    """池构建排除不可用 patch（票 03 边界：全固体 → ValueError，上层避开）。

    回归背景（真实数据冒烟发现）：pipedcylinder2d 存在**非全固体但
    种子-中心线段全固体**的 patch（patch 中心在壁面内）→ 提取必然失败；
    精确判据 = 每个固体种子沿 seed→center 线段 201 点采样存在流体格。
    """

    @staticmethod
    def make_solid_ds(tmp_path, tag):
        import dataset as ds
        xdim, ydim, tdim = synth_grid()
        u, v = synth_field(xdim, ydim, tdim)
        Y, X = len(ydim), len(xdim)
        # 固体块：覆盖 y 格 0..31 × x 格 0..31（patch (0,0) 全固体）
        # 与 y 格 40..48 × x 格 24..56（垂直条带穿越 patch (16,32) 中心区域）
        mask = np.zeros((Y, X), dtype=bool)
        mask[0:32, 0:32] = True
        mask[40:, 24:56] = True
        root = tmp_path / tag
        ds.prepare_dataset(None, str(root), u=u, v=v, xdim=xdim, ydim=ydim,
                           tdim=tdim, taus={"train": SYNTH_TAU}, mask=mask)
        return ds.WeakLabelPathlineDataset(str(root), split="train",
                                           samples_per_epoch=20, seed=0)

    def test_fully_solid_patch_excluded(self, tmp_path):
        d = self.make_solid_ds(tmp_path, "ds_excl1")
        combos = set((p, q) for p, q, _f in d.pool_positive + d.pool_negative)
        assert (0, 0) not in combos                 # 全固体 patch 不入池
        assert not d._patch_usable(0, 0)
        assert d._patch_usable(0, 32)               # 不覆盖固体 → 可用
        assert (0, 32) in combos

    def test_usable_patches_extract_ok(self, tmp_path):
        """池内全部组合可提取（不抛 ValueError；回归"种子-中心线段全固体"失败）。"""
        d = self.make_solid_ds(tmp_path, "ds_excl2")
        d.set_epoch(0)
        assert len(d.pool_positive) + len(d.pool_negative) > 0
        for idx in range(20):
            ((_, pl), lab) = d[idx]
            assert pl.shape == (16, 256, 7) and np.isfinite(pl).all()


# ================================================================ 切片 6：真实数据（HANDOFF §2 已核实事实）

REAL_NC = pathlib.Path(r"C:\Users\徐子屹\Desktop\AI CFD\CFD数据集\pipedcylinder2d.nc")
REAL_NC = REAL_NC if REAL_NC.exists() else None


@pytest.mark.skipif(REAL_NC is None, reason="真实数据集不在本机")
class TestRealDatasetSmoke:
    """真实数据冒烟（验收支撑）：单窗口提取 + 归一化（不 prepare 全量 memmap）。"""

    def test_real_window_extract_and_normalize(self):
        """真实窗口（卷街成熟期 t=12s, patch 涡街区）：批量提取 + 归一化
        → 有限、t 通道 [0, t_scale]、u/v 归一化、种子无固体。"""
        import dataset as ds
        import h5py
        with h5py.File(str(REAL_NC), "r") as f:
            xdim = f["xdim"][:].astype(np.float64)
            ydim = f["ydim"][:].astype(np.float64)
            tdim = f["tdim"][:].astype(np.float64)
            t0f = 1200
            u_win = np.asarray(f["u"][t0f:t0f + 24], dtype=np.float32)
            v_win = np.asarray(f["v"][t0f:t0f + 24], dtype=np.float32)
        mask = np.load(r"C:\Users\徐子屹\Desktop\AI CFD\cylinder_vortex_pipeline\outputs\geometry\mask.npy")
        raw, seeds = extractor.extract_pathlines_batched(
            u_win, v_win, mask[0].astype(bool), None, xdim, ydim, tdim,
            patch_yx=(100, 280), t0=float(tdim[t0f]), L=16, rng=t0f, return_seeds=True)
        geo = extractor.patch_geometry((100, 280), (32, 32), xdim, ydim)
        t_span = 23.0 * (tdim[1] - tdim[0])
        out = ds.normalize_pathlines(raw, seeds, geo, t0=float(tdim[t0f]), t_span=t_span,
                                     t_scale=0.25, ivd_mu=0.0, ivd_sigma=1.0,
                                     speed_max=4.6)
        assert out.shape == (16, 256, 7) and np.isfinite(out).all()
        assert out[:, :, 2].min() >= -1e-6 and out[:, :, 2].max() <= 0.25 + 1e-6
        assert np.abs(out[:, :, 5]).max() <= 1.0 + 1e-3
        for k in range(256):
            assert not extractor.mask_at(mask[0].astype(bool), seeds[k, 0], seeds[k, 1],
                                         xdim, ydim)
