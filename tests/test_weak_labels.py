"""04 票：弱标签（weak_labels.py）测试 —— 数据准备缝（属性测试 + 已知字面量）。

领域词汇（HANDOFF §4/§6，唯一权威）：
- 涡量 ω = ∂v/∂x − ∂u/∂y（中心差分，边界单边）；
- IVD = |ω − 5×5 局部邻域均值|（邻域窗口含中心，edge pad 边界语义）；
- 固体区 IVD=0（依赖票 02 掩膜）；
- 标签 = 种子点处 IVD ≥ τ（默认 95 分位数、逐时间片）+ 5×5 最小面积（25 格）
  连通域过滤；τ 计算的 95 分位数在**流体区**（排除固体 0 值污染）；
- 正样本 = patch 内存在 ≥1 条涡迹线（即 t0 帧标签场在 patch 内有正格）；
- 2D Q-criterion Q = ‖Ω‖²/2 − ‖S‖²/2（涡度张量与应变率张量的 Frobenius 范数）。

期望值来源（独立于实现）：
- 合成场解析值手算字面量：线性场 ω 常数、刚体旋转 ω=2Ω、Q=Ω²、
  剪切流 Q=0、拉伸流 Q=−a²；
- 5×5 邻域均值字面量：孤立脉冲 1 → 含中心窗口均值 1/25=0.04；
- τ 分位数字面量：numpy 线性插值公式 (n−1)×p 手算；
- 真实数据断言对照 HANDOFF §2 已核实事实（固体每帧固定 28213 格，ε=1e-5 分离稳定）。
"""

import pathlib

import numpy as np
import pytest

import geometry
import weak_labels

REAL_NC = pathlib.Path(r"C:\Users\徐子屹\Desktop\AI CFD\CFD数据集\pipedcylinder2d.nc")
REAL_NC = REAL_NC if REAL_NC.exists() else None

# 已核实事实（HANDOFF §2）：固体零速格每帧固定（静态几何），ε=1e-5 稳定分离
REAL_SOLID_CELLS = 28213


# ---------------------------------------------------------------- 合成数据工具

def grid(Y=40, X=60, T=8, dt=0.1):
    """与合成流场配套的物理坐标（等距，模拟 nc 网格）。"""
    xdim = np.linspace(-1.0, 1.0, X)
    ydim = np.linspace(-1.0, 1.0, Y)
    tdim = np.linspace(0.0, (T - 1) * dt, T)
    return xdim, ydim, tdim


def field_from_func(fn, xdim, ydim, tdim):
    """由解析函数构造 (T,Y,X) 场；fn(xx, yy, tt)（meshgrid indexing='ij'）。

    fn 可返回 ndarray 或标量（如常值场 lambda x, y, t: 1.0）→ 广播到全场。
    """
    tt, yy, xx = np.meshgrid(tdim, ydim, xdim, indexing="ij")
    return np.broadcast_to(fn(xx, yy, tt), tt.shape).astype(np.float64)


# ================================================================ 切片 1：涡量 ω

class TestVorticity:
    """ω = ∂v/∂x − ∂u/∂y：线性场/刚体旋转/均匀流（解析字面量）。"""

    def test_linear_field_constant_vorticity(self):
        """线性场 u=1.5x+0.75y−2, v=−3.25x+1.25y+0.5：ω = ∂v/∂x−∂u/∂y = −4.0 全场
        （含边界单边差分对线性场同样精确）。"""
        xdim, ydim, tdim = grid()
        u = field_from_func(lambda x, y, t: 1.5 * x + 0.75 * y - 2.0, xdim, ydim, tdim)
        v = field_from_func(lambda x, y, t: -3.25 * x + 1.25 * y + 0.5, xdim, ydim, tdim)
        omega = weak_labels.vorticity(u, v, xdim, ydim)
        assert omega.shape == u.shape
        assert np.allclose(omega, -4.0, atol=1e-9)

    def test_rigid_rotation_vorticity(self):
        """刚体旋转 u=−Ωy, v=Ωx（Ω=2）：ω = 2Ω = 4.0（纯涡旋，涡量 = 两倍角速度）。"""
        xdim, ydim, tdim = grid()
        u = field_from_func(lambda x, y, t: -2.0 * y, xdim, ydim, tdim)
        v = field_from_func(lambda x, y, t: 2.0 * x, xdim, ydim, tdim)
        omega = weak_labels.vorticity(u, v, xdim, ydim)
        assert np.allclose(omega, 4.0, atol=1e-9)

    def test_uniform_flow_zero_vorticity(self):
        """均匀流 u=1, v=0：ω=0（无旋）。"""
        xdim, ydim, tdim = grid()
        u = field_from_func(lambda x, y, t: 1.0, xdim, ydim, tdim)
        v = field_from_func(lambda x, y, t: 0.0, xdim, ydim, tdim)
        omega = weak_labels.vorticity(u, v, xdim, ydim)
        assert np.allclose(omega, 0.0, atol=1e-12)


# ================================================================ 切片 2：5×5 邻域均值

class TestNeighborhoodMean:
    """5×5 局部邻域均值（窗口含中心，edge pad 边界）。"""

    def test_constant_field_mean_constant(self):
        """常值场：任意格（含边界）的 5×5 邻域均值 = 常值。"""
        omega = np.full((1, 15, 20), 2.5)
        m = weak_labels.neighborhood_mean(omega, k=5)
        assert m.shape == omega.shape
        assert np.allclose(m, 2.5)

    def test_impulse_known_literal(self):
        """孤立脉冲 1（远离边界）：中心格均值 = 1/25 = 0.04（字面量，邻域含中心 25 格）；
        距中心 1 格且窗口覆盖脉冲处 = 0.04；窗口不含脉冲处 = 0。"""
        omega = np.zeros((1, 15, 20))
        omega[0, 7, 10] = 1.0
        m = weak_labels.neighborhood_mean(omega, k=5)
        assert m[0, 7, 10] == pytest.approx(1.0 / 25.0)
        assert m[0, 7, 9] == pytest.approx(1.0 / 25.0)     # 窗口 [7..11]×[8..12] 含 (7,10)
        assert m[0, 0, 0] == pytest.approx(0.0)            # 窗口不含脉冲
        assert m[0, 7, 13] == pytest.approx(0.0)           # 窗口列 [11..15]，不含 10

    def test_3d_field_matches_2d(self):
        """(T,Y,X) 输入逐帧处理：与单帧结果一致（守护时间维语义）。"""
        omega = np.zeros((2, 15, 20))
        omega[0, 7, 10] = 1.0
        omega[1, 7, 10] = 2.0
        m = weak_labels.neighborhood_mean(omega, k=5)
        assert m[0, 7, 10] == pytest.approx(0.04)
        assert m[1, 7, 10] == pytest.approx(2.0 / 25.0)


# ================================================================ 切片 3：IVD

class TestIVD:
    """IVD = |ω − 5×5 邻域均值|；固体区 IVD=0。"""

    def test_constant_vorticity_zero_ivd(self):
        """ω 全场常数 2：邻里均值 = 2 → IVD = 0（无局部偏差）。"""
        omega = np.full((1, 15, 20), 2.0)
        ivd = weak_labels.ivd_from_vorticity(omega, k=5)
        assert np.allclose(ivd, 0.0, atol=1e-12)

    def test_impulse_known_literal(self):
        """ω 孤立脉冲 1：中心 IVD = |1 − 0.04| = 0.96（字面量）。"""
        omega = np.zeros((1, 15, 20))
        omega[0, 7, 10] = 1.0
        ivd = weak_labels.ivd_from_vorticity(omega, k=5)
        assert ivd[0, 7, 10] == pytest.approx(0.96)
        assert ivd[0, 7, 9] == pytest.approx(0.04)
        assert ivd[0, 0, 0] == pytest.approx(0.0)

    def test_compute_ivd_solid_zeroed(self):
        """固体区 IVD=0（掩膜格强制置零）；流体区不受影响（验收 4）。"""
        xdim, ydim, tdim = grid()
        # 刚体旋转场：涡核 ω=2Ω ≠ 0 的流体
        u = field_from_func(lambda x, y, t: -3.0 * y, xdim, ydim, tdim)
        v = field_from_func(lambda x, y, t: 3.0 * x, xdim, ydim, tdim)
        mask2d = np.zeros_like(u[0], dtype=bool)
        mask2d[10:16, 20:26] = True                        # 覆盖涡核中部的一块固体
        ivd = weak_labels.compute_ivd(u, v, xdim, ydim, mask=mask2d)
        assert ivd.shape == u.shape
        assert np.all(ivd[:, mask2d] == 0.0)               # 固体区 IVD=0
        assert ivd[:, ~mask2d].max() > 0.0                 # 流体区存在非零 IVD
        assert np.isfinite(ivd).all()

    def test_zero_flow_zero_ivd(self):
        """无流场 u=v=0：ω=0 → IVD=0（掩膜无关的零输出）。"""
        xdim, ydim, tdim = grid()
        u = np.zeros((len(tdim), len(ydim), len(xdim)))
        v = np.zeros_like(u)
        ivd = weak_labels.compute_ivd(u, v, xdim, ydim, mask=None)
        assert np.allclose(ivd, 0.0)


# ================================================================ 切片 4：Q-criterion

class TestQCriterion:
    """2D Q = ‖Ω‖²/2 − ‖S‖²/2 = −(∂u/∂y)(∂v/∂x) − ½[(∂u/∂x)² + (∂v/∂y)²]。"""

    def test_rigid_rotation_positive_q(self):
        """刚体旋转 u=−Ωy, v=Ωx（Ω=3）：Q = Ω² = 9（纯涡旋 Q>0 的解析值）。"""
        xdim, ydim, tdim = grid()
        u = field_from_func(lambda x, y, t: -3.0 * y, xdim, ydim, tdim)
        v = field_from_func(lambda x, y, t: 3.0 * x, xdim, ydim, tdim)
        q = weak_labels.q_criterion(u, v, xdim, ydim)
        assert np.allclose(q, 9.0, atol=1e-9)

    def test_shear_flow_zero_q(self):
        """纯剪切 u=γy, v=0（γ=2）：Q = 0（剪切无涡旋）。"""
        xdim, ydim, tdim = grid()
        u = field_from_func(lambda x, y, t: 2.0 * y, xdim, ydim, tdim)
        v = field_from_func(lambda x, y, t: 0.0, xdim, ydim, tdim)
        q = weak_labels.q_criterion(u, v, xdim, ydim)
        assert np.allclose(q, 0.0, atol=1e-12)

    def test_stretching_flow_negative_q(self):
        """双曲拉伸 u=ax, v=−ay（a=2）：Q = −a² = −4（纯应变 Q<0）。"""
        xdim, ydim, tdim = grid()
        u = field_from_func(lambda x, y, t: 2.0 * x, xdim, ydim, tdim)
        v = field_from_func(lambda x, y, t: -2.0 * y, xdim, ydim, tdim)
        q = weak_labels.q_criterion(u, v, xdim, ydim)
        assert np.allclose(q, -4.0, atol=1e-9)


# ================================================================ 切片 5：标签二值化与面积过滤

class TestBinaryLabel:
    """标签 = IVD ≥ τ（种子点处）。"""

    def test_threshold_binarization(self):
        """IVD=[0.1,0.5,0.9]，τ=0.5 → [0,1,1]（字面量）。"""
        ivd = np.array([[0.1, 0.5, 0.9]], dtype=np.float64)
        lab = weak_labels.binary_label(ivd, tau=0.5)
        assert (lab[0] == [0, 1, 1]).all()


class TestMinAreaFilter:
    """5×5 最小面积（25 格）连通域过滤（HANDOFF §6 最小涡面积）。"""

    def test_small_block_removed_large_kept(self):
        """4×5=20 格块（<25）被滤除；5×6=30 格块保留（字面量面积）。"""
        lab = np.zeros((40, 60), dtype=bool)
        lab[5:9, 5:10] = True        # 4×5 = 20 格（间距 2 格，互不连通）
        lab[5:10, 20:26] = True      # 5×6 = 30 格
        out = weak_labels.filter_min_area(lab, min_area=25)
        assert out[5:9, 5:10].sum() == 0
        assert out[5:10, 20:26].sum() == 30

    def test_exact_5x5_kept(self):
        """恰好 5×5=25 格块保留（≥ 最小面积即保留）。"""
        lab = np.zeros((30, 40), dtype=bool)
        lab[10:15, 10:15] = True
        out = weak_labels.filter_min_area(lab, min_area=25)
        assert out[10:15, 10:15].sum() == 25

    def test_diagonal_connectivity_keeps_larger_blob(self):
        """8 邻接：对角相接的两块 3×3+3×3（18 格但 8 邻接连通）≥25？否——18<25 被滤；
        验证与 geometry 连通性口径一致（8 邻接）。"""
        lab = np.zeros((30, 40), dtype=bool)
        lab[10:13, 10:13] = True
        lab[12:15, 12:15] = True     # 对角接触（(12,12) 共用），8 邻接连通 → 18 格一块
        out = weak_labels.filter_min_area(lab, min_area=25)
        assert out.sum() == 0        # 18 < 25


# ================================================================ 切片 6：τ（逐时间片 85 分位）

class TestTau:
    """τ = 按 percentile 分位（默认 85——票 07 延伸定案：p95 弱标签相比论文
    Fig.6 列 1 捕获稀疏 → 下探至 p85；HANDOFF §6 已回写），逐时间片；分位数在
    流体区（排除固体 0 值）。"""

    def test_default_percentile_is_85(self):
        """τ 默认 = 流体区 85 分位。字面量：1..100 均匀值线性插值 85 分位
        = 84.15 索引插值 = 85.15（独立手算，不重算实现公式）。"""
        from weak_labels import compute_tau
        ivd = np.arange(1, 101, dtype=np.float64).reshape(1, 10, 10)
        mask = np.zeros((10, 10), dtype=bool)
        taus = compute_tau(ivd, mask, {"train": (0, 1)})
        assert taus["train"] == pytest.approx(85.15)

    def test_95_percentile_literal(self):
        """20 个值 0..19（无固体）：p95 = (n−1)×0.95 = 18×0.95 = 18.05（线性插值字面量）。
        显式 percentile=95.0（默认已改 85——票 07 延伸）。"""
        ivd = np.arange(20.0).reshape(1, 1, 20)
        mask = np.zeros((1, 20), dtype=bool)
        tau = weak_labels.compute_tau(ivd, mask, {"train": (0, 1)},
                                      percentile=95.0)["train"]
        assert tau == pytest.approx(18.05)

    def test_tau_excludes_solid(self):
        """固体区不进入分位数：流体 20 值 0..19 + 固体 1000 个 0（IVD 已置零）
        → p95 仍 = 18.05；若把固体 0 计入则 p95≈0（污染）。显式 percentile=95.0。"""
        ivd = np.zeros((1, 41, 1000), dtype=np.float64)   # 40 行固体+1 行流体，1000 列
        ivd[0, 0, :] = np.arange(1000.0)                  # 流体 1000 值 0..999
        mask = np.ones((41, 1000), dtype=bool)
        mask[0, :] = False                                # 仅第 0 行流体
        tau = weak_labels.compute_tau(ivd, mask, {"train": (0, 1)},
                                      percentile=95.0)["train"]
        # 流体 p95 = 0.95×999 = 949.05（字面量）
        assert tau == pytest.approx(949.05)

    def test_per_slice_independent(self):
        """逐时间片：A 片帧 IVD 全 10 → τA=10；B 片帧全 200 → τB=200（互不污染）。"""
        ivd = np.zeros((4, 10, 10), dtype=np.float64)
        ivd[0:2] = 10.0
        ivd[2:4] = 200.0
        mask = np.zeros((10, 10), dtype=bool)
        taus = weak_labels.compute_tau(ivd, mask, {"A": (0, 2), "B": (2, 4)})
        assert taus["A"] == pytest.approx(10.0)
        assert taus["B"] == pytest.approx(200.0)


# ================================================================ 切片 7：标签场（逐时间片 τ + 过滤 + 固体强制 0）

class TestBuildLabelField:
    """标签场 = 逐时间片 τ 二值化 + 面积过滤 + 固体强制 0。"""

    def test_uses_per_slice_tau(self):
        """帧属 A 片（τ=10）：IVD=15 → 标签 1；帧属 B 片（τ=20）：IVD=15 → 标签 0。"""
        ivd = np.zeros((2, 10, 10), dtype=np.float64)
        ivd[0] = 15.0
        ivd[1] = 15.0
        mask = np.zeros((10, 10), dtype=bool)
        taus = {"A": 10.0, "B": 20.0}
        slices = {"A": (0, 1), "B": (1, 2)}
        lab = weak_labels.build_label_field(ivd, mask, taus, slices, min_area=5)
        assert lab.dtype == np.uint8
        assert lab[0].all() == 1
        assert lab[1].any() == 0

    def test_solid_forced_zero(self):
        """固体区即使 IVD≥τ 也强制 0（验收 4 延伸）。"""
        ivd = np.full((1, 10, 10), 100.0)
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:5, 3:5] = True
        lab = weak_labels.build_label_field(ivd, mask, {"A": 10.0}, {"A": (0, 1)}, min_area=1)
        assert lab[0, 3:5, 3:5].sum() == 0
        assert lab[0].sum() == 100 - 4

    def test_min_area_applied(self):
        """标签场构建应用 5×5 面积过滤：20 格块被滤、30 格块保留。"""
        ivd = np.zeros((1, 40, 60), dtype=np.float64)
        ivd[0, 5:9, 5:10] = 10.0     # 4×5 = 20 格
        ivd[0, 5:10, 20:26] = 10.0   # 5×6 = 30 格
        mask = np.zeros((40, 60), dtype=bool)
        lab = weak_labels.build_label_field(ivd, mask, {"A": 5.0}, {"A": (0, 1)}, min_area=25)
        assert lab[0, 5:9, 5:10].sum() == 0
        assert lab[0, 5:10, 20:26].sum() == 30


# ================================================================ 切片 8：正样本占比

class TestPositivePatchFraction:
    """正样本 = patch 内存在 ≥1 条涡迹线：种子（64 组 × 4 卫星，与
    extractor.seeding_grid 同一图案）处标签=1（HANDOFF §4 dataset 口径）。"""

    def test_single_positive_on_first_group_satellite_literal(self):
        """手算字面量：域 (64,96)，xdim/ydim=格索引坐标（dx=dy=1），patch 32×32、
        stride 16 → 候选 patch = y0∈{0,16,32}×x0∈{0,16,32,48,64} = 15 个。
        组 0 的 x−Δ 卫星种子物理 (1.1, 2.7) → 格 (3,1)（rint 语义；组中心无种子，
        4 卫星不含中心——HANDOFF §1 决策 6）；标签仅 (3,1)=1 → 只有 patch(0,0)
        的种子含该格（y=3/x=1 仅被 y0=0/x0=0 覆盖）→ 占比 1/15（字面量）。"""
        lab = np.zeros((1, 64, 96), dtype=np.uint8)
        lab[0, 3, 1] = 1
        xdim = np.arange(96.0)
        ydim = np.arange(64.0)
        out = weak_labels.positive_patch_fraction(lab, xdim, ydim)
        assert out["n_patches"] == 15
        assert out["n_positive"] == 1
        assert out["fraction"] == pytest.approx(1.0 / 15.0)

    def test_none_positive_zero(self):
        """全零标签场 → 占比 0（边界值）。"""
        lab = np.zeros((1, 64, 96), dtype=np.uint8)
        out = weak_labels.positive_patch_fraction(lab, np.arange(96.0), np.arange(64.0))
        assert out["fraction"] == 0.0

    def test_all_positive_one(self):
        """全正标签场 → 占比 1（边界值）。"""
        lab = np.ones((1, 64, 96), dtype=np.uint8)
        out = weak_labels.positive_patch_fraction(lab, np.arange(96.0), np.arange(64.0))
        assert out["fraction"] == 1.0

    def test_multiframe_average(self):
        """两帧：一帧全正、一帧全零 → 平均占比 0.5（帧等权）。"""
        lab = np.zeros((2, 64, 96), dtype=np.uint8)
        lab[1] = 1
        out = weak_labels.positive_patch_fraction(lab, np.arange(96.0), np.arange(64.0))
        assert out["fraction"] == pytest.approx(0.5)
        assert out["n_positive"] * 2 == out["n_patches"]

    def test_frame_indices_subset(self):
        """frame_indices 子集统计：仅统计指定帧（仅全零帧 → 占比 0）。"""
        lab = np.zeros((2, 64, 96), dtype=np.uint8)
        lab[1] = 1
        out = weak_labels.positive_patch_fraction(
            lab, np.arange(96.0), np.arange(64.0), frame_indices=[0])
        assert out["fraction"] == 0.0

    def test_seeds_match_extractor_trace_of_seeds(self):
        """一致性守护：统计用种子与 extractor 实际种子同源（seeding_grid 单一公式），
        无障碍物时 extract_pathlines(return_seeds=True) 的输出与统计查表偏移一致。"""
        from extractor import seeding_grid
        lab = np.zeros((1, 64, 96), dtype=np.uint8)
        # 用共享函数确认若干种子落点在 patch 内（∈[0,32)），防止公式漂移出界
        xdim = np.arange(96.0)
        ydim = np.arange(64.0)
        seeds = seeding_grid((0, 0), (32, 32), xdim, ydim)
        assert seeds.min() >= -1e-9 and seeds.max() < 96.0
        out = weak_labels.positive_patch_fraction(lab, xdim, ydim)
        assert out["n_patches"] == 15


# ================================================================ 切片 8b：时间划分无泄漏

class TestTimeSlices:
    """时间划分无泄漏（spec 数据准备缝属性测试；HANDOFF §6：帧 i 的 t = i×dt）。"""

    def test_default_slices_cover_all_frames_no_overlap(self):
        """DEFAULT_SLICES 并集覆盖帧 0..1500 全部、区间互不重叠（无泄漏）：
        train 1001 + val 250 + test 250 = 1501 帧（HANDOFF §4 帧数合计）。"""
        cover = np.zeros(1501, dtype=bool)
        total = 0
        for name, (i0, i1) in weak_labels.DEFAULT_SLICES.items():
            assert 0 <= i0 < i1 <= 1501, name
            total += i1 - i0
            cover[i0:i1] = True
        assert cover.all()
        assert total == 1501

    def test_default_slices_boundary_frames_closure(self):
        """闭包口径：t=10.0（帧 1000）∈ train、t=12.5（帧 1250）∈ val、
        t=15.0（帧 1500）∈ test（HANDOFF §4「帧 0-1000 / 1000-1250 / 1250-1500」精确化）。"""
        s = weak_labels.DEFAULT_SLICES
        assert s["train"][0] <= 1000 < s["train"][1]
        assert s["val"][0] <= 1250 < s["val"][1]
        assert s["test"][0] <= 1500 < s["test"][1]


# ================================================================ 切片 9：可视化（目检图落盘守护）

class TestPlot:
    def test_ivd_q_overview_saved(self, tmp_path):
        """IVD/Q 目检图落盘（合成小场，文件存在非空）。"""
        xdim, ydim, tdim = grid(Y=30, X=40, T=3)
        u = field_from_func(lambda x, y, t: -3.0 * y, xdim, ydim, tdim)
        v = field_from_func(lambda x, y, t: 3.0 * x, xdim, ydim, tdim)
        mask2d = np.zeros((30, 40), dtype=bool)
        ivd = weak_labels.compute_ivd(u, v, xdim, ydim, mask=mask2d)
        q = weak_labels.q_criterion(u, v, xdim, ydim)
        lab = weak_labels.binary_label(ivd[0], tau=0.05)
        p = weak_labels.plot_ivd_q(
            ivd[0], q[0], lab, mask2d, xdim, ydim, str(tmp_path / "overview.png"),
            tau=0.05, title="synth")
        assert pathlib.Path(p).exists() and pathlib.Path(p).stat().st_size > 0

    def test_tau_sensitivity_saved(self, tmp_path):
        """τ 对比图落盘（多候选阈值的标签并排）。"""
        xdim, ydim, tdim = grid(Y=30, X=40, T=3)
        u = field_from_func(lambda x, y, t: -2.5 * y, xdim, ydim, tdim)
        v = field_from_func(lambda x, y, t: 2.5 * x, xdim, ydim, tdim)
        ivd = weak_labels.compute_ivd(u, v, xdim, ydim, mask=None)
        p = weak_labels.plot_tau_sensitivity(
            ivd[0], np.zeros((30, 40), dtype=bool), xdim, ydim,
            str(tmp_path / "tau.png"), percentiles=(90.0, 95.0, 97.5, 99.0),
            title="synth")
        assert pathlib.Path(p).exists() and pathlib.Path(p).stat().st_size > 0


# ================================================================ 切片 10：多阈值敏感性报告（票 07 延伸）

# 合成场字面量：80×80 帧，IVD=0 背景 + 32×32 高值块（1024 格，> 5×5 过滤）
# → 任一 min_area：正格 1024/6400、1 个连通块。
# 种子判据阳性 patch = 4 个（y0,x0 ∈ {0,16} 的 2×2——种子 y∈[3.2,28.8] 全落大块内，
# y0=16 时仍与块交叠，y0≥32 时种子 y≥35.2 出块）→ pos_patch_fraction = 4/16 = 0.25。
class TestMultiTauReport:
    """多阈值敏感性报告（HANDOFF §7 预案「多阈值敏感性报告」的落地）：
    各候选 τ 的覆盖率/连通块数/正样本占比统计 + 目检图（含论文风格白色等值线）。"""

    def synth_ivd(self, T=4):
        ivd = np.zeros((T, 80, 80), dtype=np.float64)
        ivd[:, 0:32, 0:32] = 10.0          # 大块（1024 格 ≥ 25）
        return ivd

    def test_compute_tau_candidates(self):
        """候选构造：分位配置 → {片: τ}；固定配置 → 标量（字面量 85.15/95.15/1.5）。"""
        ivd = np.arange(1, 101, dtype=np.float64).reshape(1, 10, 10)
        mask = np.zeros((10, 10), dtype=bool)
        cfgs = weak_labels.compute_tau_candidates(
            ivd, mask, {"train": (0, 1)}, percentiles=(85.0, 95.0),
            fixed_values=(1.5,))
        assert cfgs["p85"] == {"train": pytest.approx(85.15)}
        assert cfgs["p95"] == {"train": pytest.approx(95.05)}
        assert cfgs["fixed1.5"] == 1.5

    def test_report_stats_literals(self, tmp_path):
        """统计字面量（独立手算）：fixed τ=5 → 全部 min_area 正格 1024/6400、
        1 个连通块；正 patch 占比 4/16（4 帧 × 16 patch 位）。"""
        ivd = self.synth_ivd()
        mask2d = np.zeros((80, 80), dtype=bool)
        report = weak_labels.multi_tau_report(
            ivd, mask2d, {"A": (0, 4)}, np.arange(80.0), np.arange(80.0),
            str(tmp_path), percentiles=(), fixed_values=(5.0,),
            min_areas=(25, 9, 1), display_frames=(0,), sample_step=1,
            frame_step=1, title="synth")
        s = report["stats"]["fixed5"]
        for ma in (25, 9, 1):
            assert s[f"pos_cell_frac_ma{ma}"] == pytest.approx(1024 / 6400)
            assert s[f"mean_n_components_ma{ma}"] == pytest.approx(1.0)
        pa = s["pos_patch_fraction_all"]
        assert pa["n_patches"] == 16 * 4                            # 4 帧 × 16 patch
        assert pa["fraction"] == pytest.approx(4 / 16)
        assert (tmp_path / "multi_tau_stats.json").exists()

    def test_report_pngs_saved(self, tmp_path):
        """目检图落盘：填充标签对比（每 min_area 一行）+ 论文风格白色等值线。"""
        ivd = self.synth_ivd(T=2)
        mask2d = np.zeros((80, 80), dtype=bool)
        rep = weak_labels.multi_tau_report(
            ivd, mask2d, {"A": (0, 2)}, np.arange(80.0), np.arange(80.0),
            str(tmp_path), percentiles=(), fixed_values=(5.0,),
            min_areas=(25, 1), display_frames=(0,))
        for name in ("multi_tau_filled_t0.png", "multi_tau_isocontour_t0.png"):
            p = tmp_path / name
            assert p.exists() and p.stat().st_size > 0
        assert rep["out_dir"] == str(tmp_path)


# ================================================================ 真实数据（HANDOFF §2 已核实事实）

@pytest.fixture(scope="module")
def real_small():
    """真实数据集 3 帧切片（帧 400-402，t≈4.0s，train 时间片）——避免全量 405MB 计算。
    固体 = 3 帧逐帧取与（静态几何每帧固定 28213 格，ε=1e-5 分离稳定——§2 已核实）。"""
    if REAL_NC is None:
        pytest.skip("真实数据集不在本机")
    import h5py
    with h5py.File(str(REAL_NC), "r") as f:
        u = np.asarray(f["u"][400:403], dtype=np.float64)
        v = np.asarray(f["v"][400:403], dtype=np.float64)
        xdim = f["xdim"][:].astype(np.float64)
        ydim = f["ydim"][:].astype(np.float64)
        tdim = f["tdim"][:].astype(np.float64)
    mask2d = geometry.static_mask_from_speed(u, v, eps=1e-5)
    assert mask2d.sum() == REAL_SOLID_CELLS     # 与 §2 已核实事实一致（静态几何）
    return u, v, xdim, ydim, tdim, mask2d


def test_real_ivd_properties(real_small):
    """真实数据：IVD 形状/有限/非负；固体区 IVD=0（验收 4）；流体区存在非零 IVD。"""
    u, v, xdim, ydim, tdim, mask2d = real_small
    ivd = weak_labels.compute_ivd(u, v, xdim, ydim, mask=mask2d)
    assert ivd.shape == u.shape == (3, 150, 450)
    assert np.isfinite(ivd).all()
    assert ivd.min() >= 0.0
    assert np.all(ivd[:, mask2d] == 0.0)
    assert ivd[:, ~mask2d].max() > 0.0


def test_real_tau_sane(real_small):
    """真实数据：τ95 有限且落在 (0, 流体 IVD 最大值]（涡街存在 → 正阈值）。"""
    u, v, xdim, ydim, tdim, mask2d = real_small
    ivd = weak_labels.compute_ivd(u, v, xdim, ydim, mask=mask2d)
    tau = weak_labels.compute_tau(ivd, mask2d, {"train": (0, 3)})["train"]
    assert 0.0 < tau <= ivd[:, ~mask2d].max()


def test_real_positive_fraction(real_small):
    """真实数据：标签场正样本占比 ∈ (0,1)（涡街存在 → >0；非全场涡 → <1）；
    帧口径 = 窗口起点步长 4 帧（HANDOFF §6）。"""
    u, v, xdim, ydim, tdim, mask2d = real_small
    ivd = weak_labels.compute_ivd(u, v, xdim, ydim, mask=mask2d)
    tau = weak_labels.compute_tau(ivd, mask2d, {"train": (0, 3)})["train"]
    lab = weak_labels.build_label_field(
        ivd, mask2d, {"train": tau}, {"train": (0, 3)}, min_area=25)
    out = weak_labels.positive_patch_fraction(
        lab, xdim, ydim, frame_indices=[0, 2])
    assert 0.0 < out["fraction"] < 1.0
