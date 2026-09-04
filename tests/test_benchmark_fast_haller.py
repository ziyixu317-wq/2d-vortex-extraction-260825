"""Small deterministic tests for the fast_haller benchmark report helpers."""

import numpy as np

from benchmark_fast_haller import compare_final_boundaries


def _item(row, col, area, radius):
    points = np.asarray([
        [row - radius, col - radius],
        [row - radius, col + radius],
        [row + radius, col + radius],
        [row + radius, col - radius],
        [row - radius, col - radius],
    ], dtype=np.float64)
    return {
        "peak": {"row": row, "col": col, "value": 1.0},
        "area": area,
        "points_grid": points.tolist(),
        "points_xy": points[:, ::-1].tolist(),
    }


def test_compare_final_boundaries_reports_only_common_final_peaks():
    strict = [_item(5, 5, 4.0, 1.0), _item(10, 10, 9.0, 1.5)]
    fast = [_item(5, 5, 5.0, 2.0), _item(12, 12, 4.0, 1.0)]

    report = compare_final_boundaries(strict, fast, (20, 20))

    assert report["strict_vortex_count"] == 2
    assert report["fast_vortex_count"] == 2
    assert report["matched_vortex_count"] == 1
    assert report["strict_only_count"] == 1
    assert report["fast_only_count"] == 1
    assert report["area_abs_error"]["mean"] == 1.0
    assert 0.0 < report["boundary_iou"]["mean"] < 1.0
