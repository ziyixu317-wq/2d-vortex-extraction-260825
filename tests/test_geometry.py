"""02 票：固体几何掩膜（geometry.py）测试 —— 数据准备缝（属性测试）。

领域词汇（HANDOFF §4/§6，唯一权威）：
- 掩膜 = |v|<ε 逐帧取与 + 连通域标记，随数据集 (T,Y,X) 存储；
- 圆柱 = 不与壁相连的孤立连通块（无尺寸判据）；
- 无障碍物数据集输出空掩膜、代码路径不变。

期望值来源（独立于实现）：
- 合成数据由本测试直接构造（已知字面量）；
- 真实数据断言对照 HANDOFF §2 已核实事实：28213 个近零速细胞（41.8%）为
  原始事实；4 连通块 / 2 圆柱 / 圆心 (0,0)/(3,1) / radius=0.0625 为票 02
  实测后按 §11 协议回写 HANDOFF §2 的权威事实（测试不重算实现路径）。
"""

import json
import pathlib

import numpy as np
import pytest

import geometry

REAL_NC = pathlib.Path(r"C:\Users\徐子屹\Desktop\AI CFD\CFD数据集\pipedcylinder2d.nc")
REAL_NC = REAL_NC if REAL_NC.exists() else None

# 已知事实（HANDOFF §2 实测 + 元数据）
REAL_SOLID_CELLS = 28213
REAL_N_COMPONENTS = 4
REAL_N_CYLINDERS = 2
REAL_RADIUS = 0.0625          # nc 元数据 radius（Re=160=U·D/ν 自洽）
REAL_DX = 6.0 / 449           # x 格尺寸（物理单位/格）
REAL_DY = 2.0 / 149


# ---------------------------------------------------------------- 合成数据工具

def make_flow(T=5, Y=40, X=60, vel=1.0, zero_cells=()):
    """构造速度场：默认全 vel（x 向），zero_cells 处恒零速（u=v=0）。"""
    u = np.full((T, Y, X), vel, dtype=np.float32)
    v = np.zeros((T, Y, X), dtype=np.float32)
    for (j, i) in zero_cells:
        u[:, j, i] = 0.0
        v[:, j, i] = 0.0
    return u, v


def make_disk_cells(cx, cy, r, Y=40, X=60):
    """格中心在圆内的格集合（合成圆柱）。"""
    jj, ii = np.mgrid[0:Y, 0:X]
    return set(zip(jj[(jj - cy) ** 2 + (ii - cx) ** 2 < r**2].tolist(),
                   ii[(jj - cy) ** 2 + (ii - cx) ** 2 < r**2].tolist()))


def grid_of(Y=40, X=60):
    """与合成流场配套的物理坐标（等距，模拟 nc 网格）。"""
    xdim = np.linspace(-1.0, 1.0, X)
    ydim = np.linspace(-1.0, 1.0, Y)
    return xdim, ydim


# ---------------------------------------------------------------- 合成：无障碍物

def test_empty_mask_when_no_obstacle(tmp_path):
    """无障碍物数据集：输出空掩膜、不报错、圆柱列表为空（验收 3）。"""
    u, v = make_flow()
    meta = geometry.build_geometry_mask(
        u, v, *grid_of(), out_dir=str(tmp_path), eps=1e-5)
    assert meta["solid_cells"] == 0
    assert meta["n_components"] == 0
    assert meta["cylinders"] == []
    # 落盘产物存在且为空
    mask = np.load(tmp_path / "mask.npy")
    assert mask.shape == (u.shape[0], u.shape[1], u.shape[2])
    assert not mask.any()


# ---------------------------------------------------------------- 合成：静态掩膜

def test_static_mask_detects_fixed_zero_region():
    """恒零速区进入掩膜；仅部分帧零速的格不进掩膜（逐帧取与）。"""
    Y, X = 20, 30
    fixed = {(5, 5), (5, 6), (6, 5), (6, 6)}          # 恒零速
    u, v = make_flow(T=4, Y=Y, X=X, zero_cells=fixed)
    u[1, 10, 10] = 0.0                                 # 仅 t=1 零速（瞬态）
    v[1, 10, 10] = 0.0
    mask = geometry.static_mask_from_speed(u, v, eps=1e-5)
    assert mask.dtype == bool and mask.shape == (Y, X)
    for (j, i) in fixed:
        assert mask[j, i]
    assert not mask[10, 10]                            # 瞬态零速被取与排除
    assert mask.sum() == len(fixed)


# ---------------------------------------------------------------- 合成：连通性属性

def test_component_stats_connectivity_property():
    """连通性属性：掩膜细胞 = 各块大小之和；8 邻接对角相连合并为一块。"""
    Y, X = 20, 30
    # 块 A：2×2 实心；块 B：两个 1×1 对角相邻（8 邻接 → 一块）；块 C：分离单格
    cells = {(4, 4), (4, 5), (5, 4), (5, 5),
             (10, 10), (11, 11),
             (15, 20)}
    u, v = make_flow(Y=Y, X=X, zero_cells=cells)
    mask = geometry.static_mask_from_speed(u, v, eps=1e-5)
    stats = geometry.component_stats(mask, *grid_of(Y, X))
    # 块数：A + B(对角相连) + C = 3
    assert len(stats) == 3
    assert sum(c["cells"] for c in stats) == mask.sum() == len(cells)
    sizes = sorted(c["cells"] for c in stats)
    assert sizes == [1, 2, 4]


# ---------------------------------------------------------------- 合成：圆柱定位

def test_cylinder_located_as_isolated_component():
    """圆柱 = 不与壁相连的孤立连通块：合成圆盘被正确定位，圆心/半径与输入一致。"""
    Y, X = 40, 60
    cx, cy, r = 25.0, 15.0, 6.0                        # 格坐标
    disk = make_disk_cells(cx, cy, r, Y, X)
    wall = {(j, i) for j in range(0, 5) for i in range(X)}   # 接触边界的壁面
    u, v = make_flow(Y=Y, X=X, zero_cells=disk | wall)
    meta = geometry.build_geometry_mask(
        u, v, *grid_of(Y, X), out_dir=None, eps=1e-5)
    assert len(meta["cylinders"]) == 1
    cyl = meta["cylinders"][0]
    xdim, ydim = grid_of(Y, X)
    dx, dy = xdim[1] - xdim[0], ydim[1] - ydim[0]
    # 圆心：质心与输入差 < 1 格
    assert abs(cyl["center_x"] - xdim[int(cx)]) < dx
    assert abs(cyl["center_y"] - ydim[int(cy)]) < dy
    # 半径：面积等效半径（平均格尺寸口径）与输入差 < 1 格
    assert abs(cyl["r_eff"] - r * np.sqrt(dx * dy)) < dx
    # 壁面块不算圆柱
    assert cyl["cells"] == len(disk)
    assert cyl["touches_border"] is False


def test_cylinder_min_block_filter():
    """min_block_cells 是显式收紧选项：默认（1）时小噪声块算圆柱；显式 4 时被过滤。"""
    Y, X = 20, 30
    disk = make_disk_cells(15.0, 10.0, 4.0, Y, X)
    noise = {(3, 3), (3, 4)}                           # 2 格小块
    u, v = make_flow(Y=Y, X=X, zero_cells=disk | noise)
    # 默认：规格字面——任何孤立块都是圆柱
    meta = geometry.build_geometry_mask(u, v, *grid_of(Y, X), out_dir=None, eps=1e-5)
    assert len(meta["cylinders"]) == 2
    # 显式收紧：2 格噪声块被过滤
    meta = geometry.build_geometry_mask(
        u, v, *grid_of(Y, X), out_dir=None, eps=1e-5, min_block_cells=4)
    assert len(meta["cylinders"]) == 1
    assert meta["cylinders"][0]["cells"] == len(disk)


# ---------------------------------------------------------------- 合成：落盘与时间不变性

def test_mask_saved_time_invariant_tyx(tmp_path):
    """掩膜随数据集 (T,Y,X) 存储且每帧相同（时间不变）；meta 落盘可读。"""
    Y, X = 20, 30
    cells = make_disk_cells(15.0, 10.0, 5.0, Y, X)
    u, v = make_flow(T=7, Y=Y, X=X, zero_cells=cells)
    meta = geometry.build_geometry_mask(
        u, v, *grid_of(Y, X), out_dir=str(tmp_path), eps=1e-5)
    mask = np.load(tmp_path / "mask.npy")
    assert mask.shape == (7, Y, X)
    for t in range(1, 7):
        assert np.array_equal(mask[t], mask[0])        # 时间不变
    assert mask[0].sum() == len(cells)
    meta2 = json.loads((tmp_path / "geometry_meta.json").read_text(encoding="utf-8"))
    assert meta2["solid_cells"] == len(cells)
    assert len(meta2["cylinders"]) == 1


# ---------------------------------------------------------------- 真实数据（HANDOFF §2 已核实事实）

@pytest.fixture(scope="module")
def real_data():
    if REAL_NC is None:
        pytest.skip("真实数据集不在本机")
    import h5py
    with h5py.File(str(REAL_NC), "r") as f:
        u = f["u"][:]
        v = f["v"][:]
        xdim = f["xdim"][:].astype(np.float64)
        ydim = f["ydim"][:].astype(np.float64)
    return u, v, xdim, ydim


def test_real_solid_cells_match_known_fact(real_data):
    """真实数据：近零速全帧取与 = 28213 格（41.8%），与 HANDOFF §2 一致。"""
    u, v, xdim, ydim = real_data
    meta = geometry.build_geometry_mask(u, v, xdim, ydim, out_dir=None, eps=1e-5)
    assert meta["solid_cells"] == REAL_SOLID_CELLS
    assert meta["shape"] == [1501, 150, 450]


def test_real_components_structure(real_data):
    """真实数据：4 个连通块（两块壁面接触边界 + 两个孤立圆柱）。"""
    u, v, xdim, ydim = real_data
    meta = geometry.build_geometry_mask(u, v, xdim, ydim, out_dir=None, eps=1e-5)
    assert meta["n_components"] == REAL_N_COMPONENTS
    walls = [c for c in meta["components"] if c["touches_border"]]
    cyls = meta["cylinders"]
    assert len(walls) == 2
    assert len(cyls) == REAL_N_CYLINDERS
    assert all(not c["touches_border"] for c in cyls)


def test_real_cylinder_centers_and_radius(real_data):
    """真实数据：圆柱圆心位于入口 (0,0) 与拐角后管道 (3,1)（±1 格）；
    物理半径估计（含表面层修正）与元数据 radius=0.0625 差 < 1 格（验收 2）。"""
    u, v, xdim, ydim = real_data
    meta = geometry.build_geometry_mask(u, v, xdim, ydim, out_dir=None, eps=1e-5)
    assert len(meta["cylinders"]) == 2
    by_x = sorted(meta["cylinders"], key=lambda c: c["center_x"])
    c_inlet, c_down = by_x
    assert abs(c_inlet["center_x"] - 0.0) < 2 * REAL_DX
    assert abs(c_inlet["center_y"] - 0.0) < 2 * REAL_DY
    assert abs(c_down["center_x"] - 3.0) < 2 * REAL_DX
    assert abs(c_down["center_y"] - 1.0) < 2 * REAL_DY
    # 半径：r_phys（零速区半径 + 表面层 1 格）与元数据差 < 1 格
    for cyl in meta["cylinders"]:
        assert abs(cyl["r_phys"] - REAL_RADIUS) < REAL_DX
        # 零速区面积等效半径本身在 [0.040, 0.060]（表面层效应的内切区）
        assert 0.040 <= cyl["r_eff"] <= 0.060


def test_real_mask_time_invariant(real_data, tmp_path):
    """真实数据：掩膜 (T,Y,X) 落盘后每帧相同（静态几何时间不变）。"""
    u, v, xdim, ydim = real_data
    geometry.build_geometry_mask(
        u, v, xdim, ydim, out_dir=str(tmp_path), eps=1e-5)
    mask = np.load(tmp_path / "mask.npy")
    assert mask.shape == (1501, 150, 450)
    for t in (1, 750, 1500):
        assert np.array_equal(mask[t], mask[0])
    assert mask[0].sum() == REAL_SOLID_CELLS
