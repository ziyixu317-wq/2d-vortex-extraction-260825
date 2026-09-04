"""Tests for the unmodified upstream NumbaCS rotcohvrt adapter."""

import numpy as np

import haller_anchors
import numbacs_haller


def test_numbacs_adapter_calls_upstream_with_transposed_ivd_and_contract_params(
    monkeypatch,
):
    rows, cols = 11, 9
    xdim = np.linspace(-1.0, 1.0, cols)
    ydim = np.linspace(-2.0, 2.0, rows)
    ivd = np.zeros((rows, cols), dtype=np.float64)
    ivd[5, 4] = 3.0
    calls = {}

    def fake_rotcohvrt(lavd, x, y, r, **kwargs):
        calls["lavd_shape"] = lavd.shape
        calls["x"] = np.asarray(x).copy()
        calls["y"] = np.asarray(y).copy()
        calls["r"] = r
        calls["kwargs"] = kwargs
        contour = np.asarray([
            [-0.5, -0.8], [0.5, -0.8], [0.5, 0.8], [-0.5, 0.8], [-0.5, -0.8]
        ])
        return [[contour, np.asarray([0.0, 0.0])]]

    monkeypatch.setattr(
        numbacs_haller,
        "_load_rotcohvrt",
        lambda: (fake_rotcohvrt, "0.2.0", "/upstream/elliptic.py"),
    )
    params = haller_anchors._resolve_parameters()
    params.update({
        "numbacs_radius_factor": 1.0,
    })

    selected, diagnostics, evaluation = numbacs_haller.select_contours(
        ivd,
        np.zeros_like(ivd, dtype=bool),
        xdim,
        ydim,
        params,
        xdim[1] - xdim[0],
        ydim[1] - ydim[0],
        return_evaluation=True,
    )

    assert len(selected) == len(diagnostics) == 1
    assert calls["lavd_shape"] == (cols, rows)
    np.testing.assert_array_equal(calls["x"], xdim)
    np.testing.assert_array_equal(calls["y"], ydim)
    assert calls["kwargs"] == {
        "convexity_deficiency": 0.1,
        "min_len": 8.0 * max(xdim[1] - xdim[0], ydim[1] - ydim[0]),
    }
    assert calls["r"] == max(xdim[1] - xdim[0], ydim[1] - ydim[0])
    assert evaluation["contour_algorithm"] == "numbacs.rotcohvrt"
    assert evaluation["numbacs_native_call"] is True
    assert evaluation["numbacs_default_min_val"] == -1.0
    assert evaluation["numbacs_default_nlevs"] == 20
    assert evaluation["numbacs_explicit_kwargs"] == [
        "convexity_deficiency",
        "min_len",
    ]
    assert evaluation["numbacs_omitted_kwargs"] == [
        "min_val",
        "nlevs",
        "start_level",
        "end_level",
    ]
    assert evaluation["numbacs_return_count"] == 1
    assert selected[0]["boundary_source"] == "numbacs_rotcohvrt_convex_hull"
    assert selected[0]["points_grid"][0][0] == np.min(selected[0]["points_grid"], axis=0)[0]


def test_numbacs_backend_normalizes_contour_mode_and_records_runtime_identity(
    monkeypatch,
):
    coords = np.linspace(-2.0, 2.0, 31)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radial = np.exp(-(xx * xx + yy * yy) / (2.0 * 0.8**2))

    monkeypatch.setattr(
        numbacs_haller,
        "_load_rotcohvrt",
        lambda: (
            lambda *args, **kwargs: [[
                np.asarray([
                    [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
                    [-0.5, 0.5], [-0.5, -0.5],
                ]),
                np.asarray([0.0, 0.0]),
            ]],
            "0.2.0",
            "/upstream/elliptic.py",
        ),
    )

    result = haller_anchors.extract_haller_anchors(
        -yy * radial,
        xx * radial,
        coords,
        coords,
        np.zeros_like(xx, dtype=bool),
        backend=haller_anchors.BACKEND_NUMBACS,
    )

    metadata = result["metadata"]
    assert metadata["backend"] == "numbacs"
    assert metadata["resolved"] == "numbacs"
    assert metadata["backend_version"].startswith("numbacs-0.2.0+")
    assert metadata["contour_mode"] == haller_anchors.CONTOUR_MODE_NUMBACS
    assert metadata["contour_backend"] == "numbacs.extraction.elliptic.rotcohvrt"
    assert metadata["parameters"]["numbacs_method"].endswith("rotcohvrt")
    assert metadata["parameters"]["numbacs_contour_level_count"] == 20
    assert metadata["parameters"]["numbacs_nlevs_passed_explicitly"] is False
    assert "numbacs_min_val" not in metadata["parameters"]
    assert metadata["parameters"]["numbacs_default_min_val"] == -1.0
    assert metadata["legacy_p85_used"] is False
