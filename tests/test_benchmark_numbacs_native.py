"""Tests for the isolated native NumbaCS baseline benchmark."""

import sys
import types

import numpy as np

import benchmark_numbacs_native


def test_rotcohvrt_baseline_omits_all_optional_search_arguments(monkeypatch):
    calls = []

    def fake_rotcohvrt(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    numbacs = types.ModuleType("numbacs")
    numbacs.__path__ = []
    extraction = types.ModuleType("numbacs.extraction")
    extraction.__path__ = []
    elliptic = types.ModuleType("numbacs.extraction.elliptic")
    elliptic.rotcohvrt = fake_rotcohvrt
    monkeypatch.setitem(sys.modules, "numbacs", numbacs)
    monkeypatch.setitem(sys.modules, "numbacs.extraction", extraction)
    monkeypatch.setitem(sys.modules, "numbacs.extraction.elliptic", elliptic)

    result, _ = benchmark_numbacs_native._run_rotcohvrt(
        np.zeros((3, 4), dtype=np.float64),
        np.arange(4, dtype=np.float64),
        np.arange(3, dtype=np.float64),
        r=1.0,
        convexity_deficiency=0.1,
        min_len=2.0,
    )

    assert result == []
    assert calls[0][1] == {
        "convexity_deficiency": 0.1,
        "min_len": 2.0,
    }


def test_rotcohvrt_sensitivity_only_adds_explicit_level_count(monkeypatch):
    calls = []

    def fake_rotcohvrt(*args, **kwargs):
        calls.append(kwargs)
        return []

    elliptic = types.ModuleType("numbacs.extraction.elliptic")
    elliptic.rotcohvrt = fake_rotcohvrt
    monkeypatch.setitem(sys.modules, "numbacs.extraction.elliptic", elliptic)

    benchmark_numbacs_native._run_rotcohvrt(
        np.zeros((3, 4), dtype=np.float64),
        np.arange(4, dtype=np.float64),
        np.arange(3, dtype=np.float64),
        r=1.0,
        convexity_deficiency=0.1,
        min_len=2.0,
        nlevs=64,
    )

    assert calls[0] == {
        "convexity_deficiency": 0.1,
        "min_len": 2.0,
        "nlevs": 64,
    }
