"""Direct prediction coverage, independent of reference class and rejection."""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np


def support_summary(count: Any, state: Any, valid_fluid: Any, solid: Any, invalid: Any) -> dict[str, Any]:
    count = np.asarray(count)
    state = np.asarray(state)
    valid = np.asarray(valid_fluid, dtype=bool)
    solid = np.asarray(solid, dtype=bool)
    invalid = np.asarray(invalid, dtype=bool)
    if any(a.shape != count.shape for a in (state, valid, solid, invalid)):
        raise ValueError("support/reference shapes differ")
    if not np.issubdtype(count.dtype, np.integer) or np.any(count < 0):
        raise ValueError("prediction count must be nonnegative integers")
    support = count > 0
    fluid = valid & ~solid & ~invalid
    masks = {"valid_fluid": fluid, "known_positive": fluid & (state == 1),
             "known_negative": fluid & (state == 0), "unknown": fluid & (state < 0),
             "solid": solid, "invalid": invalid}
    rows = {}
    for name, mask in masks.items():
        total = int(mask.sum())
        covered = int((mask & support).sum())
        rows[name] = {"total": total, "covered": covered, "missing": total - covered,
                      "coverage": covered / total if total else None}
    return {"denominators": rows, "whole_grid_count": int(count.size),
            "direct_prediction_count": int(count.sum()),
            "missing_probability_semantics": "unobserved; not a negative prediction",
            "positive_reference_omission_count": rows["known_positive"]["missing"]}


def common_support(counts: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    arrays = {name: np.asarray(value) > 0 for name, value in counts.items()}
    if not arrays or len({a.shape for a in arrays.values()}) != 1:
        raise ValueError("methods require matching prediction count shapes")
    common = np.logical_and.reduce(list(arrays.values()))
    return common, {"common_count": int(common.sum()), "methods": {
        name: {"covered_count": int(a.sum()), "outside_common_count": int((a & ~common).sum())}
        for name, a in arrays.items()}}


def merge_support_summaries(summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = {}
    for name in summaries[0]["denominators"]:
        total = sum(s["denominators"][name]["total"] for s in summaries)
        covered = sum(s["denominators"][name]["covered"] for s in summaries)
        rows[name] = {"total": total, "covered": covered, "missing": total - covered,
                      "coverage": covered / total if total else None}
    return {"denominators": rows,
            "whole_grid_count": sum(s["whole_grid_count"] for s in summaries),
            "direct_prediction_count": sum(s["direct_prediction_count"] for s in summaries),
            "positive_reference_omission_count": rows["known_positive"]["missing"]}


def reference_prediction_support(count: Any, reference_mask: Any) -> dict[str, Any]:
    count = np.asarray(count)
    mask = np.asarray(reference_mask, dtype=bool)
    if count.shape != mask.shape:
        raise ValueError("reviewed reference and prediction count shapes differ")
    total = int(mask.sum())
    covered = int((mask & (count > 0)).sum())
    return {"reference_cell_count": total, "covered": covered, "missing": total - covered,
            "coverage": covered / total if total else None,
            "direct_prediction_count": int(count[mask].sum()),
            "minimum_cell_prediction_count": int(count[mask].min()) if total else None}
