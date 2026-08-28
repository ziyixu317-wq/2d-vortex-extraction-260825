# -*- coding: utf-8 -*-
"""票 07 延伸运行反馈四期：域边缘批式积分掩膜索引回归测试。"""
import numpy as np

import extractor


def _jung_like_grid():
    """jung 类网格字面量（nc 实测 dy=0.02010047435760498，末格略小）：跨度/dy
    = 199.0000628 > 199——上缘半格内位置 (y-ydim[0])/dy+0.5 取整可达 Y（OOB 根源）。"""
    dy = 0.02010047435760498
    ydim = np.concatenate([-2.0 + np.arange(199) * dy, [2.0]])
    xdim = np.linspace(-3.0, 7.0, 450)
    return dy, ydim, xdim


def test_batched_edge_fringe_mask_lookup_no_oob():
    """域上缘半格内（out_ok）位置的掩膜索引须裁剪（回归守护）。

    Kaggle 多数据集训练实测崩溃：jung 网格非整跨度（span/dy = 199.0000628），
    上缘半格内位置映射索引 j == Y → 批式积分 m2[j, i] IndexError（size 200）；
    标量 mask_at 有越界前置（返回 False）不崩，批式缺该守护。
    """
    dy, ydim, xdim = _jung_like_grid()
    tdim = np.arange(6, dtype=np.float64) * 0.05      # t 0..0.25 ≥ dt_out×L=0.3? 用 0.2 采样
    T, Y, X = len(tdim), len(ydim), len(xdim)
    u = np.zeros((T, Y, X), dtype=np.float32)
    v = np.zeros_like(u)
    mask = np.zeros((Y, X), dtype=bool)
    assert (ydim[-1] - ydim[0]) / dy > Y - 1     # 前提：非整跨度（网格舍入）
    fracs = np.array([0.49994, 0.49996, 0.49998, 0.40])
    ys = ydim[-1] + fracs * dy
    gy = (ys - ydim[0]) / dy
    assert np.all(ys < ydim[-1] + dy / 2.0)      # out_ok 半格容差内
    for f in fracs[:3]:
        assert int(np.floor(gy[f == fracs][0] + 0.5)) >= Y   # 未裁剪必越界
    seeds = np.stack([np.full(len(fracs), 0.0), ys], axis=1)
    pos, times, n = extractor._integrate_batched(
        u, v, mask, seeds, 0.0, 0.1, 3, xdim, ydim, tdim, n_substeps=4)
    assert pos.shape == (len(fracs), 3, 2)
    assert np.all(n == 3)                # 静止场 + 流体边界格：fringe 视为域内，不冻结
    # 边界格为固体 → fringe 位置应冻结（n=1，截断语义比标量越界视为流体更物理）
    mask_solid = np.zeros((Y, X), dtype=bool)
    mask_solid[Y - 1, :] = True
    _, _, n_solid = extractor._integrate_batched(
        u, v, mask_solid, seeds[:1], 0.0, 0.1, 3, xdim, ydim, tdim, n_substeps=4)
    assert n_solid[0] == 1
