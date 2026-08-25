"""03 票：迹线提取（extractor.py）测试 —— 数据准备缝（属性测试 + 已知字面量）。

领域词汇（HANDOFF §4/§6，唯一权威）：
- 迹线 = 从种子点按 RK4 + 三线性时空插值积分（每输出步 4 子步）生成的轨迹；
- 每样本 256 条 = 64 组 × 4 轴向卫星点（不含中心），Δ = patch 边长×0.05，组主序编组；
- 7 通道 = [px, py, t, ivd, distance(距种子点), u, v]；
- 种子落固体 → 重播种（仿 C++ JittorReSeeding：朝 patch 中心随机移动）；
- 迹线入固体 → 截断并重复末点（不引入 -1000 毒值）；
- 位置按 patch 归一化到 [-1,1]（可超界）；全局场积分允许迹线离开 patch。

期望值来源（独立于实现）：
- 合成场由本测试直接构造（已知字面量：常值/线性场解析值）；
- 真实数据断言对照 HANDOFF §2 已核实事实（两个圆柱位置等）。

参考实现（只读）：PyflowVis-main CppProjects/src/VectorFieldCompute.cpp
（PathhlineIntegrationRK4v2 / PathlineIntegrationInfoCollect2D / GroupSeeding::JittorReSeeding）
与 FLowUtils/flowlineIntegral.py（出域 clamp 语义）。
"""

import pathlib

import numpy as np
import pytest

import extractor
from extractor import (CH_DIST, CH_IVD, CH_PX, CH_PY, CH_T, CH_U, CH_V)


# ---------------------------------------------------------------- 合成数据工具

def grid(Y=40, X=60, T=8, dt=0.1):
    """与合成流场配套的物理坐标（等距，模拟 nc 网格）。"""
    xdim = np.linspace(-1.0, 1.0, X)
    ydim = np.linspace(-1.0, 1.0, Y)
    tdim = np.linspace(0.0, (T - 1) * dt, T)
    return xdim, ydim, tdim


# ================================================================ 切片 1：三线性插值

class TestTrilinearInterp:
    """三线性时空插值：空间双线性 + 时间线性；越界 clamp 到边界格（有限值）。"""

    def test_constant_field_anywhere(self):
        """常值场：任意 (x,y,t) 插值 = 常值（含格点与非格点）。"""
        xdim, ydim, tdim = grid()
        field = np.full((len(tdim), len(ydim), len(xdim)), 2.5, dtype=np.float32)
        for x, y, t in [(-1.0, -1.0, 0.0), (0.13, -0.27, 0.35), (0.9, 0.9, 0.7)]:
            assert extractor.trilinear_interp(field, x, y, t, xdim, ydim, tdim) == pytest.approx(2.5)

    def test_linear_field_matches_analytic(self):
        """线性场 f = 2x − 3y + 4t + 1：非格点插值 = 解析值（独立来源）。"""
        Y, X, T = 20, 30, 6
        xdim = np.linspace(0.0, 1.0, X)
        ydim = np.linspace(0.0, 1.0, Y)
        tdim = np.linspace(0.0, 1.0, T)
        tt, yy, xx = np.meshgrid(tdim, ydim, xdim, indexing="ij")   # (T, Y, X)
        field = (2.0 * xx - 3.0 * yy + 4.0 * tt + 1.0).astype(np.float32)
        x, y, t = 0.37, 0.61, 0.42
        want = 2.0 * x - 3.0 * y + 4.0 * t + 1.0
        got = extractor.trilinear_interp(field, x, y, t, xdim, ydim, tdim)
        assert got == pytest.approx(want, abs=1e-5)

    def test_temporal_linearity(self):
        """时间线性：两层场，t 居中 → 帧间线性混合（解析值）。"""
        Y, X, T = 4, 6, 3
        xdim, ydim, tdim = grid(Y, X, T)
        field = np.zeros((T, Y, X), dtype=np.float32)
        field[0] = 0.0
        field[2] = 10.0
        # t=0.15 为帧 1 与帧 2 中点（tdim = [0, 0.1, 0.2]）：0.5×0 + 0.5×10 = 5
        assert extractor.trilinear_interp(field, 0.0, 0.0, 0.15, xdim, ydim, tdim) == pytest.approx(5.0)

    def test_out_of_bounds_clamps_to_edge(self):
        """越界 clamp：x 超右边界 → 边界列值；t 超界 → 边界帧值（有限无 NaN）。"""
        Y, X, T = 4, 6, 3
        xdim, ydim, tdim = grid(Y, X, T)
        field = np.zeros((T, Y, X), dtype=np.float32)
        field[..., :] = np.arange(X, dtype=np.float32)[None, None, :]   # f = x 格索引
        assert extractor.trilinear_interp(field, 5.0, -1.0, 0.0, xdim, ydim, tdim) == pytest.approx(X - 1)
        assert extractor.trilinear_interp(field, -5.0, -1.0, 0.0, xdim, ydim, tdim) == pytest.approx(0.0)
        # t 越上界 → clamp 到最后一帧；x/y 取格中心保证空间插值精确
        assert extractor.trilinear_interp(field, -1.0, -1.0, 99.0, xdim, ydim, tdim) == pytest.approx(0.0)

    def test_vectorized_interp_matches_scalar(self):
        """一致性守护：向量版 interp_path 与标量版 trilinear_interp 公式双份，
        随机点（含越界）结果必须一致（防双份实现漂移）。"""
        Y, X, T = 12, 18, 5
        xdim, ydim, tdim = grid(Y, X, T)
        rng = np.random.default_rng(7)
        field = rng.standard_normal((T, Y, X)).astype(np.float32)
        pts = rng.uniform(-1.5, 1.5, size=(40, 2))     # 含域内与越界点
        ts = rng.uniform(-0.2, 0.8, size=40)
        got_vec = extractor.interp_path(field, pts, ts, xdim, ydim, tdim)
        got_scalar = np.array([
            extractor.trilinear_interp(field, x, y, t, xdim, ydim, tdim)
            for (x, y), t in zip(pts, ts)])
        assert got_vec == pytest.approx(got_scalar, abs=1e-6)


# ================================================================ 切片 2：RK4 积分器

def _const_velocity_field(vx, vy, Y=40, X=60, T=8, dt=0.1):
    """常速度场（全时全空），与 grid() 配套。"""
    xdim, ydim, tdim = grid(Y, X, T, dt)
    u = np.full((T, Y, X), vx, dtype=np.float32)
    v = np.full((T, Y, X), vy, dtype=np.float32)
    return u, v, xdim, ydim, tdim


class TestIntegratePathline:
    """全局场 RK4 积分：常速度场直线（解析）、子步精度、出域停止、入固体停止。"""

    def test_constant_velocity_straight_line(self):
        """常速度场 (u=1, v=0)：位置/时间与解析一致；RK4 精确。"""
        u, v, xdim, ydim, tdim = _const_velocity_field(1.0, 0.0)
        pos, times, status = extractor.integrate_pathline(
            u, v, None, (0.0, 0.0), t0=0.0, dt_out=0.1, L=5,
            xdim=xdim, ydim=ydim, tdim=tdim, n_substeps=4)
        assert status == extractor.STATUS_COMPLETE
        assert len(pos) == 5
        assert times == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
        assert pos[:, 0] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
        assert pos[:, 1] == pytest.approx([0.0] * 5)

    def test_substeps_increase_accuracy(self):
        """子步生效：旋转场 (u=−y, v=x) 从 (1,0) 出发，解析圆 (cos t, sin t)；
        4 子步误差远小于 1 子步（每输出步 4 子步是 HANDOFF §6 口径）。"""
        Y, X, T = 40, 60, 30
        xdim, ydim, tdim = grid(Y, X, T, dt=0.1)
        yy, xx = np.mgrid[0:Y, 0:X]
        u = np.broadcast_to((-ydim[yy])[None], (T, Y, X)).astype(np.float32)
        v = np.broadcast_to((xdim[xx])[None], (T, Y, X)).astype(np.float32)
        dt_out = 0.5
        pos1, _, _ = extractor.integrate_pathline(
            u, v, None, (1.0, 0.0), t0=0.0, dt_out=dt_out, L=3,
            xdim=xdim, ydim=ydim, tdim=tdim, n_substeps=1)
        pos4, _, _ = extractor.integrate_pathline(
            u, v, None, (1.0, 0.0), t0=0.0, dt_out=dt_out, L=3,
            xdim=xdim, ydim=ydim, tdim=tdim, n_substeps=4)
        want = np.array([np.cos(1.0), np.sin(1.0)])
        err1 = np.abs(pos1[-1] - want).max()
        err4 = np.abs(pos4[-1] - want).max()
        assert err1 > 1e-4          # 大步长 RK4 有明显误差
        assert err4 < 1e-5          # 4 子步显著更准

    def test_stops_at_spatial_domain_edge(self):
        """空间出域停止：常速度场向 x 正方向，到达格边缘后停止（不越界插值）。"""
        Y, X, T = 8, 11, 30
        xdim, ydim, tdim = grid(Y, X, T, dt=0.1)      # xdim = [-1, -0.8, ..., 1]，dx=0.2，域上界 1.1
        u = np.full((T, Y, X), 1.0, dtype=np.float32)
        v = np.zeros((T, Y, X), dtype=np.float32)
        pos, times, status = extractor.integrate_pathline(
            u, v, None, (0.0, 0.0), t0=0.0, dt_out=0.2, L=10,
            xdim=xdim, ydim=ydim, tdim=tdim, n_substeps=4)
        assert status == extractor.STATUS_OUT_OF_DOMAIN
        assert pos[:, 0] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # 未越过格边缘（域上界 = xdim[-1] + dx/2 = 1.1）
        assert pos[-1, 0] <= 1.1

    def test_stops_at_temporal_domain_edge(self):
        """时间出域停止：t 超过场数据 tdim 上界后停止。"""
        Y, X, T = 8, 11, 11
        xdim, ydim, tdim = grid(Y, X, T, dt=0.1)      # tdim = [0, ..., 1.0]
        u = np.full((T, Y, X), 1.0, dtype=np.float32)
        v = np.zeros((T, Y, X), dtype=np.float32)
        pos, times, status = extractor.integrate_pathline(
            u, v, None, (0.0, 0.0), t0=0.85, dt_out=0.1, L=10,
            xdim=xdim, ydim=ydim, tdim=tdim, n_substeps=4)
        assert status == extractor.STATUS_OUT_OF_DOMAIN
        assert times == pytest.approx([0.85, 0.95])   # 下一步 1.05 > tdim[-1]

    def test_stops_entering_solid(self):
        """入固体停止：掩膜格 5（xdim=[0..1] 时中心 x=0.5）→ 截断于格 4 中心，
        不采纳越界输出步。"""
        Y, X, T = 8, 11, 30
        xdim = np.linspace(0.0, 1.0, X)               # 格 5 中心 = 0.5
        ydim, tdim = grid(Y, X, T, dt=0.1)[1:]
        u = np.full((T, Y, X), 1.0, dtype=np.float32)
        v = np.zeros((T, Y, X), dtype=np.float32)
        mask = np.zeros((Y, X), dtype=bool)
        mask[:, 5] = True                              # 固体条带（x=0.5）
        pos, _, status = extractor.integrate_pathline(
            u, v, mask, (0.0, 0.0), t0=0.0, dt_out=0.1, L=10,
            xdim=xdim, ydim=ydim, tdim=tdim, n_substeps=4)
        assert status == extractor.STATUS_HIT_SOLID
        assert pos[:, 0] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
        assert len(pos) == 5


# ================================================================ 切片 3：截断补齐（重复末点）

class TestPadRepeatLast:
    """截断轨迹补齐：末点重复到 L 行，不引入 -1000 毒值（HANDOFF §4/票 03）。"""

    def test_truncated_rows_repeat_last(self):
        """(3, 7) 截断行补齐到 L=5：第 4、5 行 = 第 3 行（末点重复）。"""
        feat = np.arange(21.0).reshape(3, 7)
        out = extractor.pad_repeat_last(feat, L=5)
        assert out.shape == (5, 7)
        assert np.array_equal(out[:3], feat)
        assert np.array_equal(out[3], feat[2])
        assert np.array_equal(out[4], feat[2])

    def test_complete_length_unchanged(self):
        """n == L：原样返回（完整轨迹不补齐）。"""
        feat = np.ones((6, 7))
        out = extractor.pad_repeat_last(feat, L=6)
        assert out.shape == (6, 7)
        assert np.array_equal(out, feat)

    def test_single_point_repeats(self):
        """n = 1（种子即停）：整行重复到 L。"""
        feat = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        out = extractor.pad_repeat_last(feat, L=4)
        assert out.shape == (4, 7)
        assert np.allclose(out, feat[0])

    def test_no_minus1000_poison(self):
        """补齐不含 -1000 毒值（C++ 参考用 -1000 填充，本项目决策不用）。"""
        feat = np.array([[0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0]])
        out = extractor.pad_repeat_last(feat, L=16)
        assert not (out == -1000.0).any()
        assert np.isfinite(out).all()


# ================================================================ 切片 4：种子落固体重播种

def _solid_disk_mask(Y, X, cy, cx, r):
    """格中心在圆内的固体掩膜（合成圆柱）。"""
    jj, ii = np.mgrid[0:Y, 0:X]
    return (jj - cy) ** 2 + (ii - cx) ** 2 < r ** 2


class TestReseed:
    """种子落固体 → 重播种（仿 C++ JittorReSeeding：seed + shift×(center−seed)）。"""

    def test_seed_in_solid_moves_toward_center(self):
        """种子在固体：重播种后不在固体，且位移沿 center−seed 方向（点积为正）。"""
        Y, X = 40, 60
        xdim, ydim, _ = grid(Y, X)
        mask = _solid_disk_mask(Y, X, cy=20, cx=30, r=8)   # 圆柱在格 (30,20)
        center = np.array([0.5, 0.0])                        # patch 中心（圆柱右侧）
        rng = np.random.default_rng(0)
        new = extractor.reseed((xdim[30], ydim[20]), mask, center, xdim, ydim, rng=rng)
        assert not extractor.mask_at(mask, new[0], new[1], xdim, ydim)
        direction = center - np.array([xdim[30], ydim[20]])
        assert np.dot(new - np.array([xdim[30], ydim[20]]), direction) > 0.0

    def test_seed_not_in_solid_unchanged(self):
        """种子不在固体：原样返回（无扰动）。"""
        Y, X = 40, 60
        xdim, ydim, _ = grid(Y, X)
        mask = _solid_disk_mask(Y, X, cy=20, cx=30, r=8)
        seed = np.array([-0.8, -0.8])
        new = extractor.reseed(seed, mask, np.array([0.0, 0.0]), xdim, ydim,
                               rng=np.random.default_rng(1))
        assert np.array_equal(new, seed)

    def test_all_solid_patch_raises(self):
        """patch 全固体：无法重播种 → ValueError（上层采样应避开全固体 patch）。"""
        Y, X = 10, 10
        xdim, ydim, _ = grid(Y, X)
        mask = np.ones((Y, X), dtype=bool)
        with pytest.raises(ValueError):
            extractor.reseed((0.0, 0.0), mask, np.array([0.3, 0.1]), xdim, ydim,
                             rng=np.random.default_rng(2))


# ================================================================ 切片 5：样本组装

class TestExtractPathlines:
    """extract_pathlines：原始场 + patch/窗口参数 → (L, 256, 7) 迹线张量
    （数据准备缝；验收：计数恒 256、特征有限无 NaN、组主序、归一化、截断、重播种）。"""

    def test_shape_and_group_major_order(self):
        """零速度场：迹线静止；形状 (L, 256, 7)；组主序编组（组 0 的 4 条在前）；
        组内卫星偏移归一化 ±0.1（Δ = patch 边长×0.05，半宽 = 边长/2 → 2×0.05）。"""
        Y, X, T = 40, 60, 8
        u, v, xdim, ydim, tdim = _const_velocity_field(0.0, 0.0, Y, X, T)
        out = extractor.extract_pathlines(
            u, v, None, None, xdim, ydim, tdim,
            patch_yx=(0, 0), t0=0.0, L=16, rng=np.random.default_rng(0))
        assert out.shape == (16, 256, 7)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        # 静止：每条迹线所有时间步位置相同
        for k in (0, 3, 63, 255):
            assert np.allclose(out[:, k, :2], out[0, k, :2])
        # 组主序：组 0 = k 0..3，组 1 = k 4..7
        g0_x = out[0, 0:4, CH_PX]
        g0_y = out[0, 0:4, CH_PY]
        assert g0_x.max() - g0_x.min() == pytest.approx(0.2)   # ±0.1 卫星
        assert g0_y.max() - g0_y.min() == pytest.approx(0.2)
        # 组间距离 > 组内卫星跨度（8×8 网格 0.1~0.9 区间）
        assert abs(np.median(g0_x) - np.median(out[0, 4:8, CH_PX])) > 0.2

    def test_channels_semantics_constant_flow(self):
        """常速度场 (u=1)：t 通道 = t0 + 步×dt_out；distance = 距种子（直线）；
        u/v 通道 = 1/0；px 归一化单调递增。"""
        Y, X, T = 40, 150, 40
        xdim = np.linspace(-2.0, 3.0, X)             # 宽场域：16 步位移不触域边界
        u, v = np.full((T, Y, X), 1.0, dtype=np.float32), np.zeros((T, Y, X), dtype=np.float32)
        _, ydim, tdim = grid(Y, X, T, 0.1)
        dt = tdim[1] - tdim[0]
        dt_out = 23.0 * dt / 15.0
        out = extractor.extract_pathlines(
            u, v, None, None, xdim, ydim, tdim,
            patch_yx=(4, 14), t0=0.5, L=16, rng=np.random.default_rng(0))
        k = 0
        assert out[:, k, CH_T] == pytest.approx(0.5 + np.arange(16) * dt_out)
        assert out[:, k, CH_U] == pytest.approx(1.0)
        assert out[:, k, CH_V] == pytest.approx(0.0)
        assert out[0, k, CH_DIST] == pytest.approx(0.0)          # 种子处距离 0
        assert out[5, k, CH_DIST] == pytest.approx(5 * dt_out)    # 直线推进
        assert np.all(np.diff(out[:, k, CH_PX]) > 0)              # 向右单调

    def test_ivd_channel_interpolated_or_zero(self):
        """ivd 通道：传 ivd 场 → 三线性插值；不传 → 0（票 04 接入前占位）。"""
        Y, X, T = 40, 60, 8
        u, v, xdim, ydim, tdim = _const_velocity_field(0.0, 0.0, Y, X, T)
        ivd = np.full((T, Y, X), 7.0, dtype=np.float32)
        out = extractor.extract_pathlines(
            u, v, None, ivd, xdim, ydim, tdim,
            patch_yx=(0, 0), t0=0.0, L=4, rng=np.random.default_rng(0))
        assert out[:, :, CH_IVD] == pytest.approx(7.0)
        out0 = extractor.extract_pathlines(
            u, v, None, None, xdim, ydim, tdim,
            patch_yx=(0, 0), t0=0.0, L=4, rng=np.random.default_rng(0))
        assert out0[:, :, CH_IVD] == pytest.approx(0.0)

    def test_solid_truncation_repeats_endpoint(self):
        """入固体截断（端到端）：固体条带在 patch 内 → 左侧种子列 3 步后撞入
        条带（n≥3 不触发重试）→ 截断并重复末点；右侧种子完整推进；
        全样本 (L, 256, 7) 无 NaN、无 -1000。"""
        Y, X, T = 40, 150, 40
        xdim = np.linspace(-2.0, 3.0, X)             # 宽场域：避免域边界截断干扰
        u = np.full((T, Y, X), 0.3, dtype=np.float32)   # 1.37 格/步：撞固体发生在 n≥3
        v = np.zeros((T, Y, X), dtype=np.float32)
        _, ydim, tdim = grid(Y, X, T, 0.1)
        mask = np.zeros((Y, X), dtype=bool)
        mask[0:32, 28:34] = True                     # 固体条带：x 格 28..33（patch 中心 35.5 外）
        out = extractor.extract_pathlines(
            u, v, mask, None, xdim, ydim, tdim,
            patch_yx=(0, 20), t0=0.0, L=16, rng=np.random.default_rng(0))
        assert out.shape == (16, 256, 7)
        assert np.isfinite(out).all()
        assert not (out == -1000.0).any()
        # 存在截断迹线：末两步 px 相等（重复末点）
        truncated = [(out[-1, k, CH_PX] == out[-2, k, CH_PX]).any() for k in range(256)]
        assert any(truncated)
        # 也存在完整迹线：末两步 px 严格推进
        complete = [(out[-1, k, CH_PX] > out[-2, k, CH_PX]).any() for k in range(256)]
        assert any(complete)

    def test_reseed_effective_no_seed_in_solid(self):
        """重播种生效（端到端）：圆柱覆盖部分组中心 → 所有 256 个种子
        （重播种后）均不在固体；且存在推进迹线。"""
        Y, X, T = 40, 150, 40
        xdim = np.linspace(-2.0, 3.0, X)
        u, v = np.full((T, Y, X), 1.0, dtype=np.float32), np.zeros((T, Y, X), dtype=np.float32)
        _, ydim, tdim = grid(Y, X, T, 0.1)
        mask = _solid_disk_mask(Y, X, cy=16, cx=28, r=6)
        out, seeds = extractor.extract_pathlines(
            u, v, mask, None, xdim, ydim, tdim,
            patch_yx=(0, 20), t0=0.0, L=16, rng=np.random.default_rng(0),
            return_seeds=True)
        assert seeds.shape == (256, 2)
        for k in range(256):
            assert not extractor.mask_at(mask, seeds[k, 0], seeds[k, 1], xdim, ydim), \
                f"迹线 {k} 种子仍在固体（重播种未生效）"
        # 至少存在一条推进迹线（积分正常工作）
        assert any(out[-1, k, CH_PX] > out[0, k, CH_PX] for k in range(256))

    def test_short_integration_retries_reseed(self):
        """积分太短重试（仿 C++ suc 判据）：种子非固体但一步撞固体（n<3）→
        朝 patch 中心移动种子重试 → 无静止退化迹线（位移全 > 0）。"""
        Y, X, T = 40, 150, 40
        xdim = np.linspace(-2.0, 3.0, X)
        u = np.full((T, Y, X), 1.0, dtype=np.float32)   # 4.57 格/步：近固体种子 n<3
        v = np.zeros((T, Y, X), dtype=np.float32)
        _, ydim, tdim = grid(Y, X, T, 0.1)
        mask = np.zeros((Y, X), dtype=bool)
        mask[0:32, 28:34] = True                        # 固体条带（patch 中心 35.5 外）
        out = extractor.extract_pathlines(
            u, v, mask, None, xdim, ydim, tdim,
            patch_yx=(0, 20), t0=0.0, L=16, rng=np.random.default_rng(0))
        for k in range(256):
            dx = out[-1, k, CH_PX] - out[0, k, CH_PX]
            assert dx > 0.0, f"迹线 {k} 静止退化（重试未兜底）"

    def test_allows_leaving_patch(self):
        """全局场积分允许离开 patch：位移 > patch 半宽时 px 归一化超界（>1），
        迹线仍完整推进 16 步（不因离开 patch 停止）。"""
        Y, X, T = 40, 200, 40
        xdim = np.linspace(-5.0, 5.0, X)
        ydim = np.linspace(-1.0, 1.0, Y)
        tdim = np.linspace(0.0, 3.9, T)
        u = np.full((T, Y, X), 1.0, dtype=np.float32)
        v = np.zeros((T, Y, X), dtype=np.float32)
        dx = xdim[1] - xdim[0]
        dt_out = 2.0 * (32.0 * dx) / 15.0          # 总位移 = 2 个 patch 宽
        out = extractor.extract_pathlines(
            u, v, None, None, xdim, ydim, tdim,
            patch_yx=(0, X // 2 - 16), t0=0.0, L=16, dt_out=dt_out,
            rng=np.random.default_rng(0))
        assert out[:, 0, CH_PX].max() > 1.0        # 离开 patch（归一化可超界）
        assert out.shape == (16, 256, 7)
        assert np.isfinite(out).all()


class TestPlotPathlines:
    """目检图落盘（验收 1 的人工复核物）：PNG 存在且尺寸合理。"""

    def test_plot_saves_png(self, tmp_path):
        Y, X, T = 40, 150, 40
        xdim = np.linspace(-2.0, 3.0, X)
        u, v = np.full((T, Y, X), 0.5, dtype=np.float32), np.zeros((T, Y, X), dtype=np.float32)
        _, ydim, tdim = grid(Y, X, T, 0.1)
        out, seeds = extractor.extract_pathlines(
            u, v, None, None, xdim, ydim, tdim,
            patch_yx=(0, 20), t0=0.0, L=4,
            rng=np.random.default_rng(0), return_seeds=True)
        geo = extractor.patch_geometry((0, 20), (32, 32), xdim, ydim)
        phys = np.empty_like(out[:, :, :2])
        phys[:, :, 0] = out[:, :, CH_PX] * geo["hx"] + geo["cx"]
        phys[:, :, 1] = out[:, :, CH_PY] * geo["hy"] + geo["cy"]
        png = pathlib.Path(tmp_path) / "plot.png"
        extractor.plot_pathlines((u[0], v[0]), None, phys, seeds,
                                 xdim, ydim, png, title="test")
        assert png.exists() and png.stat().st_size > 10_000


# ================================================================ 真实数据（HANDOFF §2 已核实事实）

REAL_NC = pathlib.Path(r"C:\Users\徐子屹\Desktop\AI CFD\CFD数据集\pipedcylinder2d.nc")
REAL_NC = REAL_NC if REAL_NC.exists() else None


@pytest.fixture(scope="module")
def real_data():
    """真实数据（逐帧读取算掩膜，不驻留全量 u/v；HANDOFF §2 掩膜 28213 格）。"""
    if REAL_NC is None:
        pytest.skip("真实数据集不在本机")
    import h5py
    with h5py.File(str(REAL_NC), "r") as f:
        xdim = f["xdim"][:].astype(np.float64)
        ydim = f["ydim"][:].astype(np.float64)
        tdim = f["tdim"][:].astype(np.float64)
        solid = np.ones((len(ydim), len(xdim)), dtype=bool)
        for t in range(len(tdim)):
            solid &= np.hypot(f["u"][t], f["v"][t]) < 1e-5
    return xdim, ydim, tdim, solid


class TestRealData:
    """真实数据冒烟（验收：3~5 个 patch×24 帧窗口）：
    计数恒 256、7 通道有限无 NaN、无 -1000、种子不在固体。"""

    WINDOWS = [
        (400, (100, 280)),    # 拐角圆柱 (≈3,1) 下游涡街区
        (800, (100, 280)),    # 涡街发展中
        (1200, (100, 280)),   # 涡街成熟
        (1200, (0, 60)),      # 入口圆柱 (≈0,0) 下游
    ]

    def test_real_windows_valid(self, real_data):
        xdim, ydim, tdim, solid = real_data
        import h5py
        with h5py.File(str(REAL_NC), "r") as f:
            for t0f, (py, px) in self.WINDOWS:
                u_win = np.asarray(f["u"][t0f:t0f + 24], dtype=np.float32)
                v_win = np.asarray(f["v"][t0f:t0f + 24], dtype=np.float32)
                out = extractor.extract_pathlines(
                    u_win, v_win, solid, None, xdim, ydim, tdim,
                    patch_yx=(py, px), t0=float(tdim[t0f]), L=16,
                    rng=np.random.default_rng(t0f))
                assert out.shape == (16, 256, 7), f"窗口 t0={t0f}"
                assert np.isfinite(out).all(), f"窗口 t0={t0f} 含 NaN"
                assert not (out == -1000.0).any(), f"窗口 t0={t0f} 含 -1000 毒值"

    def test_real_seeds_outside_solid(self, real_data):
        """重播种生效：全部 256 个真实种子（重播种后）不在固体掩膜中。"""
        xdim, ydim, tdim, solid = real_data
        import h5py
        t0f, (py, px) = self.WINDOWS[2]
        with h5py.File(str(REAL_NC), "r") as f:
            u_win = np.asarray(f["u"][t0f:t0f + 24], dtype=np.float32)
            v_win = np.asarray(f["v"][t0f:t0f + 24], dtype=np.float32)
        _, seeds = extractor.extract_pathlines(
            u_win, v_win, solid, None, xdim, ydim, tdim,
            patch_yx=(py, px), t0=float(tdim[t0f]), L=16,
            rng=np.random.default_rng(t0f), return_seeds=True)
        assert seeds.shape == (256, 2)
        for k in range(256):
            assert not extractor.mask_at(solid, seeds[k, 0], seeds[k, 1], xdim, ydim), \
                f"迹线 {k} 种子仍在固体"

    def test_real_pathlines_follow_flow_and_avoid_solid(self, real_data):
        """数值目检（替代无视觉会话的看图；PNG 图已由 CLI 落盘供人工复核）：
        迹线有效点（未截断重复段）不穿固体；每步位移方向与速度方向
        夹角余弦均值 > 0.9（跟随流场）。"""
        xdim, ydim, tdim, solid = real_data
        t0f, (py, px) = self.WINDOWS[2]              # 拐角圆柱下游涡街 t=12s
        import h5py
        with h5py.File(str(REAL_NC), "r") as f:
            u_win = np.asarray(f["u"][t0f:t0f + 24], dtype=np.float32)
            v_win = np.asarray(f["v"][t0f:t0f + 24], dtype=np.float32)
        out = extractor.extract_pathlines(
            u_win, v_win, solid, None, xdim, ydim, tdim,
            patch_yx=(py, px), t0=float(tdim[t0f]), L=16,
            rng=np.random.default_rng(t0f))
        geo = extractor.patch_geometry((py, px), (32, 32), xdim, ydim)
        px_phys = out[:, :, CH_PX] * geo["hx"] + geo["cx"]
        py_phys = out[:, :, CH_PY] * geo["hy"] + geo["cy"]
        t_phys = out[:, :, CH_T]
        n_bad_solid = 0
        cos_sum, cos_cnt = 0.0, 0
        n_valid_points = 0
        for k in range(256):
            valid = np.ones(16, dtype=bool)
            valid[1:] = np.diff(t_phys[:, k]) > 0    # 截断重复段（t 相等）不算
            idx = np.nonzero(valid)[0]
            n_valid_points += len(idx)
            for s in idx:
                if extractor.mask_at(solid, px_phys[s, k], py_phys[s, k], xdim, ydim):
                    n_bad_solid += 1
            for s in idx[:-1]:
                dx = px_phys[s + 1, k] - px_phys[s, k]
                dy = py_phys[s + 1, k] - py_phys[s, k]
                uu, vv = out[s, k, CH_U], out[s, k, CH_V]
                nrm = np.hypot(dx, dy) * np.hypot(uu, vv)
                if nrm > 0:
                    cos_sum += (dx * uu + dy * vv) / nrm
                    cos_cnt += 1
        assert n_valid_points > 0
        assert n_bad_solid == 0, f"{n_bad_solid} 个迹线点穿入固体"
        mean_cos = cos_sum / cos_cnt
        assert mean_cos > 0.9, f"切向一致性余弦 {mean_cos:.4f} 过低"
        # 有效长度分布：多数迹线完整推进（>50% 有效点占比）
        assert n_valid_points / (256 * 16) > 0.5
