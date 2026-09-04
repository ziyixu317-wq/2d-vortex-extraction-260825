"""02 票：Haller-IVD 单帧 anchor 接缝测试。

测试以 ``haller_anchors`` 的公共输入/输出为主，并通过公开的 geometry
diagnostics seam 检查筛选原因。解析旋涡场、人工 mask 和多种退化 fixture
都在本文件内构造，避免使用 test split 或 legacy p85 标签。
"""

import numpy as np
import pytest

import haller_anchors


def swirling_field(size=161, extent=6.0, sigma=1.35):
    """构造平滑的 Rankine-style/涡旋样速度场（u/v 为 2D 单帧）。"""
    coords = np.linspace(-extent, extent, size, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radial = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return -yy * radial, xx * radial, coords, coords


def test_standard_ivd_uses_fluid_mean_and_never_labels_solid():
    """standard IVD 使用 fluid 涡量均值，solid 格只作排除而不参与均值。"""
    xdim = np.linspace(-1.0, 1.0, 21)
    ydim = np.linspace(-1.0, 1.0, 19)
    yy, xx = np.meshgrid(ydim, xdim, indexing="ij")
    u = np.zeros_like(xx)
    v = 0.5 * xx * xx                         # omega = dv/dx = x
    mask = ~((xx >= 0.0) & (xx <= 0.8))       # 非对称 solid，fluid mean 可观测

    out = haller_anchors.compute_standard_ivd(u, v, xdim, ydim, mask)
    assert out.shape == u.shape
    fluid_mean = float(xx[~mask].mean())
    assert np.all(out[mask] == 0.0)
    assert np.allclose(
        out[~mask], np.abs(xx[~mask] - fluid_mean), atol=1e-12
    )


def test_rankine_style_field_has_peak_closed_outermost_contour_and_three_states():
    """合成涡旋应产生峰、合法闭合轮廓、positive interior、unknown band 和 negative。"""
    u, v, xdim, ydim = swirling_field()
    result = haller_anchors.extract_haller_anchors(
        u, v, xdim, ydim, np.zeros_like(u, dtype=bool),
        source=haller_anchors.SOURCE_TRAIN,
        frame_index=7,
    )
    center = (len(ydim) // 2, len(xdim) // 2)

    assert result["valid"] is True
    assert result["failure_count"] == 0
    assert result["peaks"]
    assert min(abs(p["row"] - center[0]) + abs(p["col"] - center[1])
               for p in result["peaks"]) <= 1
    assert result["contours"]
    contour = result["contours"][0]
    assert contour["closed"] is True
    assert contour["level_fraction"] <= 0.1 + 1e-12
    selected_peak = contour["peak"]
    selected_peak_diagnostics = [
        item for item in result["metadata"]["contour_diagnostics"]
        if item["peak"] == selected_peak
    ]
    assert len(selected_peak_diagnostics) >= (
        result["metadata"]["parameters"]["contour_level_count"]
    )
    assert {item["level_fraction"] for item in selected_peak_diagnostics} >= {
        1.0, 0.1,
    }
    assert contour["perimeter"] >= result["metadata"]["parameters"][
        "minimum_perimeter_factor"] * result["metadata"]["spacing"]["max"]
    assert contour["convexity_defect"] <= 0.10 + 1e-12

    state = result["anchor_state"]
    assert state[center] == haller_anchors.POSITIVE
    assert np.any(state == haller_anchors.UNKNOWN)
    assert np.any(state == haller_anchors.NEGATIVE)
    assert np.all(result["anchor_confidence"][state == haller_anchors.UNKNOWN] == 0.0)
    assert np.all(result["anchor_confidence"][state != haller_anchors.UNKNOWN] == 1.0)


def test_contour_rejection_diagnostics_cover_open_small_and_nonconvex_cases():
    """几何筛选拒绝非闭合、过小周长和超凸度候选，且不静默 fallback。"""
    params = haller_anchors._resolve_parameters()

    coords = np.linspace(-1.0, 1.0, 81, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    solid = np.zeros_like(xx, dtype=bool)

    edge_field = np.exp(-(((xx + 0.8) ** 2 + yy**2) / (2.0 * 0.35**2)))
    edge_peaks = haller_anchors.find_local_maxima(edge_field, solid)
    _, edge_diagnostics = haller_anchors._select_contours(
        edge_field, solid, coords, coords, edge_peaks, params,
        coords[1] - coords[0], coords[1] - coords[0],
    )
    assert any(item.get("rejection_reason") == "open_contour"
               for item in edge_diagnostics)

    tiny_coords = np.linspace(-1.0, 1.0, 21, dtype=np.float64)
    tiny_yy, tiny_xx = np.meshgrid(tiny_coords, tiny_coords, indexing="ij")
    tiny_field = np.exp(-(tiny_xx**2 + tiny_yy**2) / (2.0 * 0.02**2))
    tiny_peaks = haller_anchors.find_local_maxima(tiny_field, np.zeros_like(tiny_field, dtype=bool))
    tiny_selected, tiny_diagnostics = haller_anchors._select_contours(
        tiny_field, np.zeros_like(tiny_field, dtype=bool), tiny_coords, tiny_coords,
        tiny_peaks, params, tiny_coords[1] - tiny_coords[0], tiny_coords[1] - tiny_coords[0],
    )
    assert tiny_selected == []
    assert any(item.get("rejection_reason") == "minimum_perimeter"
               for item in tiny_diagnostics)

    theta = np.arctan2(yy, xx)
    radius = np.sqrt(xx**2 + yy**2)
    nonconvex_field = np.where(
        radius == 0.0,
        1.0,
        np.exp(-radius / (0.55 * (1.0 + 0.82 * np.cos(3.0 * theta)))),
    )
    nonconvex_peaks = haller_anchors.find_local_maxima(nonconvex_field, solid)
    nonconvex_selected, nonconvex_diagnostics = haller_anchors._select_contours(
        nonconvex_field, solid, coords, coords, nonconvex_peaks, params,
        coords[1] - coords[0], coords[1] - coords[0],
    )
    assert nonconvex_selected == []
    assert any(item.get("rejection_reason") == "convexity_defect"
               for item in nonconvex_diagnostics)


def test_multiple_valid_contours_use_explicit_union():
    """多个候选的 interior union 不退化为交集或最后一个候选。"""
    shape = (20, 20)
    first = np.asarray([[4.0, 4.0], [4.0, 10.0], [10.0, 10.0], [10.0, 4.0]])
    second = np.asarray([[4.0, 9.0], [4.0, 15.0], [10.0, 15.0], [10.0, 9.0]])
    contours = [{"points_grid": first.tolist()}, {"points_grid": second.tolist()}]

    union, band = haller_anchors._rasterize_union(
        contours, shape, dx=1.0, dy=1.0, unknown_band_width=0.0
    )
    first_mask = haller_anchors.grid_points_in_poly(shape, first)
    second_mask = haller_anchors.grid_points_in_poly(shape, second)
    assert np.array_equal(union, first_mask | second_mask)
    assert union.sum() > max(first_mask.sum(), second_mask.sum())
    assert not band.any()


def test_solid_cells_are_unknown_even_when_inside_candidate_contour():
    """solid 不生成 positive/negative；贴近 solid 的边界保持 unknown。"""
    u, v, xdim, ydim = swirling_field(size=121, extent=5.0)
    yy, xx = np.meshgrid(ydim, xdim, indexing="ij")
    mask = (xx + 1.3) ** 2 + yy**2 <= 0.55**2
    result = haller_anchors.extract_haller_anchors(
        u, v, xdim, ydim, mask, source=haller_anchors.SOURCE_TRAIN
    )
    assert np.all(result["anchor_state"][mask] == haller_anchors.UNKNOWN)
    assert not np.any(result["anchor_state"][mask] == haller_anchors.POSITIVE)
    assert not np.any(result["anchor_state"][mask] == haller_anchors.NEGATIVE)
    assert result["metadata"]["coverage"]["solid_cells"] == int(mask.sum())


@pytest.mark.parametrize("source", [
    haller_anchors.SOURCE_TRAIN,
    haller_anchors.SOURCE_CALIBRATION,
    haller_anchors.SOURCE_TEST,
])
def test_no_legal_contour_uses_source_specific_failure_contract(source):
    """无合法 contour 不制造全负标签，train/test 的失败语义明确区分。"""
    u = np.zeros((32, 40), dtype=np.float64)
    v = np.zeros_like(u)
    xdim = np.linspace(-1.0, 1.0, u.shape[1])
    ydim = np.linspace(-1.0, 1.0, u.shape[0])
    result = haller_anchors.extract_haller_anchors(
        u, v, xdim, ydim, np.zeros_like(u, dtype=bool), source=source
    )

    assert result["valid"] is False
    assert result["failure_count"] == 1
    assert np.all(result["anchor_state"] == haller_anchors.UNKNOWN)
    assert not np.any(result["anchor_state"] == haller_anchors.NEGATIVE)
    assert result["metadata"]["failure_policy"] == (
        "train_unknown" if source == haller_anchors.SOURCE_TRAIN else "invalid_frame"
    )
    assert result["metadata"]["frame_valid"] is (source == haller_anchors.SOURCE_TRAIN)


def test_frozen_parameters_and_no_legacy_fallback_are_observable():
    """正式 artifact 使用冻结参数，并显式记录无 legacy fallback。"""
    u, v, xdim, ydim = swirling_field(size=101, extent=4.0)
    result = haller_anchors.extract_haller_anchors(
        u, v, xdim, ydim, None,
        source=haller_anchors.SOURCE_TRAIN,
    )
    params = result["metadata"]["parameters"]
    assert params["contour_level_count"] == haller_anchors.DEFAULT_CONTOUR_LEVEL_COUNT
    assert params["contour_level_start"] == haller_anchors.DEFAULT_CONTOUR_LEVEL_START
    assert params["contour_level_end"] == haller_anchors.DEFAULT_CONTOUR_LEVEL_END
    assert params["convexity_defect_max"] == haller_anchors.DEFAULT_CONVEXITY_DEFECT_MAX
    assert params["minimum_perimeter_factor"] == haller_anchors.DEFAULT_MINIMUM_PERIMETER_FACTOR
    assert params["unknown_band_factor"] == haller_anchors.DEFAULT_UNKNOWN_BAND_FACTOR
    assert params["negative_percentile"] == haller_anchors.DEFAULT_NEGATIVE_PERCENTILE
    assert result["metadata"]["parameter_hash"]
    assert result["metadata"]["fallback_used"] is None
    assert result["metadata"]["legacy_p85_used"] is False
    assert result["metadata"]["contour_mode"] == haller_anchors.CONTOUR_MODE_OPTIMIZED
    assert result["metadata"]["contour_backend"] == "skimage_peak_local_roi_exact"
    evaluation = result["metadata"]["contour_evaluation"]
    assert evaluation["level_schedule"] == "per_peak_relative_frozen"
    # The unique-level cache is the logical deduplication.  ROI boundary
    # retries are real additional calls and are part of the timing audit.
    assert evaluation["find_contours_call_count"] >= evaluation["unique_absolute_level_count"]
    assert evaluation["requested_peak_level_count"] == (
        len(result["peaks"]) * params["contour_level_count"]
    )
    assert evaluation["find_contours_total_seconds"] >= 0.0
    assert evaluation["find_contours_mean_seconds"] >= 0.0
    assert evaluation["find_contours_p95_seconds"] >= 0.0
    assert evaluation["find_contours_mean_seconds"] == pytest.approx(
        evaluation["find_contours_total_seconds"]
        / evaluation["find_contours_call_count"]
    )


def test_standard_ivd_metadata_declares_whole_fluid_domain_mean():
    """artifact metadata must make the global fluid-domain IVD definition auditable."""
    u, v, xdim, ydim = swirling_field(size=51, extent=3.0)
    result = haller_anchors.extract_haller_anchors(
        u, v, xdim, ydim, None,
        source=haller_anchors.SOURCE_TRAIN,
    )

    assert result["metadata"]["ivd_formula"] == "abs(omega - mean_fluid(omega))"
    assert result["metadata"]["mean_scope"] == "entire_fluid_spatial_domain"


def test_component_roi_expands_at_domain_boundary_without_dropping_contours():
    """ROI acceleration must expose boundary contact and expand conservatively."""
    expanded, touched = haller_anchors.expand_component_roi(
        (1, 4, 2, 5), (8, 9), margin_cells=1
    )
    assert expanded == (0, 5, 1, 6)
    assert touched is False

    expanded, touched = haller_anchors.expand_component_roi(
        (0, 3, 0, 3), (8, 9), margin_cells=1
    )
    assert expanded == (0, 4, 0, 4)
    assert touched is True


def test_roi_boundary_contact_retries_full_domain_instead_of_dropping_path():
    """局部 crop 的边界接触必须触发 full-domain contour retry。"""
    ivd = np.zeros((10, 10), dtype=np.float64)
    ivd[2:5, 2:5] = 2.0
    fluid = np.ones_like(ivd, dtype=bool)
    calls = []

    def fake_find_contours(field, *, level, mask):
        calls.append(tuple(field.shape))
        if field.shape == (5, 5):
            return [np.asarray([[0.0, 1.0], [0.0, 3.0], [3.0, 3.0],
                                [3.0, 1.0], [0.0, 1.0]])]
        return [np.asarray([[2.0, 2.0], [2.0, 4.0], [4.0, 4.0],
                            [4.0, 2.0], [2.0, 2.0]])]

    paths, bounds, fallback, call_count = haller_anchors._find_contours_with_exact_roi(
        ivd, fluid, 1.0, fake_find_contours
    )

    assert calls == [(5, 5), (10, 10)]
    assert bounds == (0, 10, 0, 10)
    assert fallback is True
    assert call_count == 2
    assert np.array_equal(paths[0], np.asarray([
        [2.0, 2.0], [2.0, 4.0], [4.0, 4.0], [4.0, 2.0], [2.0, 2.0]
    ]))


def test_peak_local_roi_paths_equal_full_domain_reference_after_expansion():
    """ROI must reproduce the same marching-squares path for the peak."""
    import skimage.measure

    coords = np.linspace(-5.0, 5.0, 101, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    field = np.exp(-(xx**2 + yy**2) / (2.0 * 0.9**2))
    fluid = np.ones_like(field, dtype=bool)
    peak = np.asarray([[50.0, 50.0]], dtype=np.float64)
    level = 0.4

    reference = tuple(skimage.measure.find_contours(
        field, level=level, mask=fluid))
    optimized, bounds, fallback, call_count = (
        haller_anchors._find_contours_with_exact_roi(
            field, fluid, level, skimage.measure.find_contours,
            peak_points=peak,
        )
    )

    assert len(reference) == len(optimized) == 1
    np.testing.assert_allclose(optimized[0], reference[0], rtol=0.0, atol=1e-12)
    assert bounds != (0, field.shape[0], 0, field.shape[1])
    assert fallback is False
    assert call_count >= 1


def test_optimized_contours_match_reference_for_unequal_nested_peak_levels():
    """Unequal peaks still use the frozen per-peak levels and exact geometry rules."""
    coords = np.linspace(-6.0, 6.0, 121, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    field = np.exp(-((xx + 2.1) ** 2 + yy**2) / (2.0 * 0.8**2))
    field += 0.63 * np.exp(-((xx - 2.0) ** 2 + yy**2) / (2.0 * 0.65**2))
    solid = np.zeros_like(field, dtype=bool)
    params = haller_anchors._resolve_parameters()
    peaks = [
        {"row": 60, "col": 39, "value": 1.0},
        {"row": 60, "col": 80, "value": 0.63},
    ]

    reference = haller_anchors._select_contours(
        field, solid, coords, coords, peaks, params,
        coords[1] - coords[0], coords[1] - coords[0], contour_mode="reference",
    )
    optimized = haller_anchors._select_contours(
        field, solid, coords, coords, peaks, params,
        coords[1] - coords[0], coords[1] - coords[0], contour_mode="optimized",
    )

    # The optimized path only materializes candidates in the exact peak-local
    # ROI; diagnostics for unrelated global contours are intentionally not
    # required for the decision.  Selected contours must remain identical.
    assert optimized[0] == reference[0]
    assert len(optimized[0]) == 2
    requested = [
        peak["value"] * fraction
        for peak in peaks
        for fraction in np.linspace(
            params["contour_level_start"],
            params["contour_level_end"],
            params["contour_level_count"],
        )
    ]
    assert len(set(requested)) == 64


def test_optimized_contour_cache_is_reference_equivalent_and_has_no_early_stop():
    """全局 exact-level cache 不改变逐涡结果或完整 32-level 审计轨迹。"""
    coords = np.linspace(-5.0, 5.0, 101, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    field = np.exp(-((xx + 2.0) ** 2 + yy**2) / (2.0 * 0.75**2))
    field += np.exp(-((xx - 2.0) ** 2 + yy**2) / (2.0 * 0.75**2))
    solid = np.zeros_like(field, dtype=bool)
    params = haller_anchors._resolve_parameters()
    peaks = [
        {"row": 50, "col": 30, "value": 1.0},
        {"row": 50, "col": 70, "value": 1.0},
    ]

    reference = haller_anchors._select_contours(
        field, solid, coords, coords, peaks, params, coords[1] - coords[0],
        coords[1] - coords[0], contour_mode="reference",
    )
    optimized = haller_anchors._select_contours(
        field, solid, coords, coords, peaks, params, coords[1] - coords[0],
        coords[1] - coords[0], contour_mode="optimized",
    )
    assert optimized[0] == reference[0]
    assert not any(item.get("status") == "skipped" for item in optimized[1])
    for peak in peaks:
        peak_diagnostics = [item for item in optimized[1] if item["peak"] == peak]
        assert len(peak_diagnostics) >= params["contour_level_count"]


def test_optimized_contour_cache_evaluates_each_repeated_absolute_level_once(monkeypatch):
    """相同 peak level 只触发一次全局 find_contours；不是减少 levels。"""
    import skimage.measure

    coords = np.linspace(-5.0, 5.0, 101, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    field = np.exp(-((xx + 2.0) ** 2 + yy**2) / (2.0 * 0.75**2))
    field += np.exp(-((xx - 2.0) ** 2 + yy**2) / (2.0 * 0.75**2))
    solid = np.zeros_like(field, dtype=bool)
    params = haller_anchors._resolve_parameters()
    peaks = [
        {"row": 50, "col": 30, "value": 1.0},
        {"row": 50, "col": 70, "value": 1.0},
    ]
    calls = []
    original = skimage.measure.find_contours

    def counted(*args, **kwargs):
        calls.append(float(kwargs["level"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(skimage.measure, "find_contours", counted)
    haller_anchors._haller_geometry_dependencies.cache_clear()
    try:
        haller_anchors._select_contours(
            field, solid, coords, coords, peaks, params,
            coords[1] - coords[0], coords[1] - coords[0], contour_mode="optimized",
        )
    finally:
        haller_anchors._haller_geometry_dependencies.cache_clear()
    assert len(calls) >= params["contour_level_count"]
    assert len(set(calls)) == params["contour_level_count"]
