"""Tests for the local ContourPy-based fast Haller backend."""

import numpy as np
import pytest

import fast_haller
import haller_anchors


def _params():
    return haller_anchors._resolve_parameters()


def _grid(size=15):
    coords = np.arange(size, dtype=np.float64)
    return coords, coords


def _concave_but_convex_enough_path(row_offset=0.0, col_offset=0.0):
    """A closed contour whose hull is intentionally not the returned path."""
    path = np.asarray([
        [3.0, 3.0], [3.0, 9.0], [5.0, 9.0], [5.0, 8.6],
        [6.0, 8.6], [6.0, 9.0], [9.0, 9.0], [9.0, 3.0],
        [3.0, 3.0],
    ])
    return path + np.asarray([row_offset, col_offset])


def test_fast_backend_is_cpu_only_and_has_a_distinct_runtime_identity():
    metadata = haller_anchors.resolve_haller_backend("fast_haller", "cpu")

    assert metadata["resolved"] == fast_haller.BACKEND_FAST_HALLER
    assert metadata["backend"] == fast_haller.BACKEND_FAST_HALLER
    assert metadata["cuda_used"] is False
    assert metadata["backend_version"].startswith("fast_haller-contourpy-")

    with pytest.raises(ValueError, match="cpu"):
        haller_anchors.resolve_haller_backend("fast_haller", "cuda:0")

    with pytest.raises(ValueError, match="strict oracle"):
        haller_anchors._select_contours(
            np.zeros((4, 4)),
            np.zeros((4, 4), dtype=bool),
            np.arange(4.0),
            np.arange(4.0),
            [],
            _params(),
            1.0,
            1.0,
            backend="fast_haller",
            contour_mode="reference",
        )


def test_pts_in_poly_mask_explicitly_preserves_zero_one_and_multiple_maxima():
    polygon = np.asarray([[1.0, 1.0], [1.0, 8.0], [8.0, 8.0], [8.0, 1.0]])

    assert fast_haller.pts_in_poly_mask(np.empty((0, 2)), polygon).shape == (0,)
    np.testing.assert_array_equal(
        fast_haller.pts_in_poly_mask(np.asarray([[4.0, 4.0]]), polygon),
        np.asarray([True]),
    )
    np.testing.assert_array_equal(
        fast_haller.pts_in_poly_mask(
            np.asarray([[4.0, 4.0], [12.0, 12.0], [4.0, 6.0]]), polygon
        ),
        np.asarray([True, False, True]),
    )


def test_fast_uses_one_global_generator_and_shared_levels(monkeypatch):
    coords = np.arange(15, dtype=np.float64)
    field = np.zeros((15, 15), dtype=np.float64)
    field[7, 7] = 10.0
    fluid = np.ones_like(field, dtype=bool)
    peaks = [{"row": 7, "col": 7, "value": 10.0}]
    path = np.asarray([
        [3.0, 3.0], [3.0, 11.0], [11.0, 11.0], [11.0, 3.0], [3.0, 3.0]
    ])
    created = []

    class Generator:
        def __init__(self):
            self.levels = []

        def lines(self, level):
            self.levels.append(float(level))
            return [path]

    def make_generator(*args, **kwargs):
        generator = Generator()
        created.append((generator, kwargs.get("roi")))
        return generator

    monkeypatch.setattr(fast_haller, "_make_contour_generator", make_generator)
    selected, _, evaluation = fast_haller.select_contours(
        field, ~fluid, coords, coords, peaks, _params(), 1.0, 1.0,
        global_level_count=5, refinement_iterations=0, return_evaluation=True,
    )

    assert len(selected) == 1
    assert len(created) == 1
    assert created[0][1] is None
    assert created[0][0].levels == sorted(created[0][0].levels)
    assert len(created[0][0].levels) == 5
    assert evaluation["global_contour_call_count"] == 5
    assert evaluation["refinement_call_count"] == 0
    assert evaluation["global_level_count"] == 5


def test_public_fast_extractor_records_default_64_level_contract():
    coords = np.linspace(-3.0, 3.0, 51, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radial = np.exp(-(xx * xx + yy * yy) / (2.0 * 0.8**2))
    result = haller_anchors.extract_haller_anchors(
        -yy * radial,
        xx * radial,
        coords,
        coords,
        np.zeros((51, 51), dtype=bool),
        backend="fast_haller",
        fast_refinement_iterations=0,
    )

    assert result["metadata"]["backend"] == "fast_haller"
    assert result["metadata"]["parameters"]["fast_global_level_count"] == 64
    assert result["metadata"]["contour_evaluation"]["global_level_count"] == 64
    assert result["metadata"]["legacy_p85_used"] is False


def test_multi_maximum_contour_is_distributed_to_all_contained_peaks():
    coords, _ = _grid()
    field = np.zeros((15, 15), dtype=np.float64)
    field[6, 5] = 10.0
    field[6, 9] = 9.0
    fluid = np.ones_like(field, dtype=bool)
    peaks = [
        {"row": 6, "col": 5, "value": 10.0},
        {"row": 6, "col": 9, "value": 9.0},
    ]
    path = np.asarray([
        [2.0, 2.0], [2.0, 12.0], [12.0, 12.0], [12.0, 2.0], [2.0, 2.0]
    ])

    class Generator:
        def lines(self, level):
            return [path]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fast_haller, "_make_contour_generator", lambda *a, **k: Generator())
    try:
        selected, diagnostics, _ = fast_haller.select_contours(
            field, ~fluid, coords, coords, peaks, _params(), 1.0, 1.0,
            global_level_count=3, refinement_iterations=0, return_evaluation=True,
        )
    finally:
        monkeypatch.undo()

    assert {item["peak"]["col"] for item in selected} == {5, 9}
    assert any(item.get("maximum_count") == 2 for item in diagnostics)


def test_fast_refines_between_nearest_outer_invalid_and_valid_and_keeps_raw_contour(
    monkeypatch,
):
    coords, _ = _grid(15)
    field = np.zeros((15, 15), dtype=np.float64)
    field[6, 6] = 10.0
    fluid = np.ones_like(field, dtype=bool)
    peaks = [{"row": 6, "col": 6, "value": 10.0}]
    raw_path = _concave_but_convex_enough_path()
    created = []

    class Generator:
        def __init__(self, roi):
            self.roi = roi

        def lines(self, level):
            # The first two coarse levels are outside the legal contour band.
            # The final valid coarse level brackets a transition at 3.2.
            if float(level) < 3.2:
                return [np.asarray([[3.0, 3.0], [3.0, 9.0], [9.0, 3.0]])]
            return [raw_path[:, ::-1]]

    def make_generator(*args, **kwargs):
        generator = Generator(kwargs.get("roi"))
        created.append(generator)
        return generator

    monkeypatch.setattr(fast_haller, "_make_contour_generator", make_generator)
    selected, diagnostics, evaluation = fast_haller.select_contours(
        field, ~fluid, coords, coords, peaks, _params(), 1.0, 1.0,
        global_level_count=3, refinement_iterations=7,
        return_evaluation=True,
    )

    assert len(selected) == 1
    assert evaluation["global_contour_call_count"] == 3
    assert evaluation["refinement_call_count"] == 7
    assert evaluation["refinement_iterations"] == 7
    assert selected[0]["level"] == pytest.approx(3.2, abs=0.02)
    # The accepted boundary remains the original IVD contour.  It is not the
    # ConvexHull polygon used internally to calculate deficiency.
    np.testing.assert_allclose(selected[0]["points_grid"], raw_path)
    assert selected[0]["points_grid"] != pytest.approx(
        np.asarray([[3.0, 3.0], [3.0, 9.0], [9.0, 9.0], [9.0, 3.0], [3.0, 3.0]])
    )
    assert any(item.get("rejection_reason") == "open_contour" for item in diagnostics)
    assert len(created) > 1


def test_refinement_roi_expands_when_local_path_touches_boundary(monkeypatch):
    coords = np.arange(20, dtype=np.float64)
    field = np.zeros((20, 20), dtype=np.float64)
    field[10, 10] = 10.0
    fluid = np.ones_like(field, dtype=bool)
    coarse = np.asarray([
        [8.0, 8.0], [8.0, 12.0], [12.0, 12.0], [12.0, 8.0], [8.0, 8.0]
    ])
    global_generator = object()
    calls = []

    class Generator:
        def __init__(self, roi):
            self.roi = roi

        def lines(self, level):
            calls.append(self.roi)
            row0, row1, col0, col1 = self.roi
            if len(calls) == 1:
                return [np.asarray([
                    [float(coords[col0]), float(coords[row0])],
                    [float(coords[col1 - 1]), float(coords[row0])],
                    [float(coords[col1 - 1]), float(coords[row1 - 1])],
                    [float(coords[col0]), float(coords[row1 - 1])],
                    [float(coords[col0]), float(coords[row0])],
                ])]
            return [coarse[:, ::-1]]

    def make_generator(*args, **kwargs):
        return Generator(kwargs["roi"])

    monkeypatch.setattr(fast_haller, "_make_contour_generator", make_generator)
    paths, bounds, fallback, call_count = fast_haller._find_refinement_contours(
        field, fluid, 5.0, coords, coords, coarse,
        global_generator=global_generator, halo_cells=1,
    )

    assert len(paths) == 1
    assert bounds[0] < bounds[1] and bounds[2] < bounds[3]
    assert bounds != (0, 20, 0, 20)
    assert fallback is False
    assert call_count == 2
    assert calls[1][0] < calls[0][0]
