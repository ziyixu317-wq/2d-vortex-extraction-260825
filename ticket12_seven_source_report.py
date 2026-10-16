"""Metrics for the registered seven-source development prediction support."""
from __future__ import annotations

import numpy as np

import prediction_support

LEGACY_SIX = ("boussinesq", "cylinder2d", "doublegyre2d", "fourcenters2d",
              "jungtelziemniak2d", "pipedcylinder2d")
SOURCE_ORDER = LEGACY_SIX + ("mfc_airfoil",)
METRIC_KEYS = ("precision", "recall", "f1", "iou", "auprc")


def metrics_from_counts(counts):
    tp, fp, fn, tn = (int(counts[k]) for k in ("tp", "fp", "fn", "tn"))
    observed = tp + fp + fn + tn
    return {**counts, "observed_known": observed,
            "precision": tp / (tp + fp) if tp + fp else (0. if observed else None),
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else (0. if observed else None),
            "iou": tp / (tp + fp + fn) if tp + fp + fn else (0. if observed else None)}


def average_precision(probability, positive):
    """Step-integral PR area (average precision), grouping equal scores together."""
    scores = np.asarray(probability, dtype=np.float64).ravel()
    labels = np.asarray(positive, dtype=bool).ravel()
    if scores.size != labels.size or not np.isfinite(scores).all():
        raise ValueError("AP requires matching finite scores and labels")
    positives = int(labels.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked, labels = scores[order], labels[order]
    endpoints = np.r_[np.flatnonzero(ranked[:-1] != ranked[1:]), len(ranked) - 1]
    true_positive = np.cumsum(labels)[endpoints]
    precision = true_positive / (endpoints + 1)
    increments = np.diff(np.r_[0, true_positive]) / positives
    return float(np.sum(increments * precision))


def score_frame(probability, count, state, valid, solid, invalid, *, threshold):
    arrays = [np.asarray(a) for a in (probability, count, state, valid, solid, invalid)]
    probability, count, state, valid, solid, invalid = arrays
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("prediction/reference shapes differ")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite probability")
    if np.any((count > 0) != np.isfinite(probability)):
        raise ValueError("direct count and finite probability disagree")
    fluid = valid.astype(bool) & ~solid.astype(bool) & ~invalid.astype(bool)
    observed = (count > 0) & fluid & np.isin(state, [0, 1])
    scores, labels = probability[observed], state[observed] == 1
    predicted = scores >= threshold
    counts = {"tp": int((predicted & labels).sum()), "fp": int((predicted & ~labels).sum()),
              "fn": int((~predicted & labels).sum()), "tn": int((~predicted & ~labels).sum())}
    metrics = {**metrics_from_counts(counts), "auprc": average_precision(scores, labels)}
    return {"metrics": metrics,
            "support": prediction_support.support_summary(count, state, valid, solid, invalid)}, scores, labels


def summarize_frames(rows, scores, labels):
    counts = {k: sum(r["metrics"][k] for r in rows) for k in ("tp", "fp", "fn", "tn")}
    attempted = sum(r["attempted_pathlines"] for r in rows)
    failed = sum(r["failed_pathlines"] for r in rows)
    return {"metrics": {**metrics_from_counts(counts),
                        "auprc": average_precision(np.concatenate(scores), np.concatenate(labels))},
            "support": prediction_support.merge_support_summaries([r["support"] for r in rows]),
            "frame_count": len(rows), "attempted_pathlines": attempted, "failed_pathlines": failed,
            "failed_rate": failed / attempted if attempted else None}


def source_macros(summaries, split):
    result = {}
    for group, names in (("old_six", LEGACY_SIX), ("mfc", ("mfc_airfoil",)), ("seven", SOURCE_ORDER)):
        metrics = [summaries[f"{name}/{split}"]["metrics"] for name in names]
        result[group] = {"source_count": len(names), "weighting": "equal source after pooling frames",
                         "evaluable_source_counts": {key: sum(m[key] is not None for m in metrics) for key in METRIC_KEYS}}
        for key in METRIC_KEYS:
            available = [m[key] for m in metrics if m[key] is not None]
            result[group][key] = float(np.mean(available)) if len(available) == len(names) else None
    return result


def replication_summary(runs, required_seeds=(0, 1, 2)):
    """Keep missing runs explicit; no partial mean is described as three-seed evidence."""
    methods = sorted({run["method"] for run in runs})
    result = {}
    for method in methods:
        selected = [run for run in runs if run["method"] == method]
        seeds = [run["seed"] for run in selected]
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate method/seed evaluation")
        missing = sorted(set(required_seeds) - set(seeds))
        table = {}
        for split in ("train", "calibration"):
            for group in ("old_six", "mfc", "seven"):
                table[f"{group}/{split}"] = {}
                for key in METRIC_KEYS:
                    values = {str(run["seed"]): run["macro"][split][group][key] for run in selected}
                    valid = [v for v in values.values() if v is not None]
                    complete = not missing and len(valid) == len(required_seeds)
                    table[f"{group}/{split}"][key] = {"by_seed": values,
                        "mean": float(np.mean(valid)) if complete else None,
                        "sample_std": float(np.std(valid, ddof=1)) if complete and len(valid) > 1 else None}
        result[method] = {"status": "complete" if not missing else "awaiting_seed_evaluations",
                          "available_seeds": sorted(seeds), "missing_seeds": missing, "metrics": table}
    return result


def markdown_report(result):
    def number(value):
        return "NA" if value is None else f"{value:.6f}"

    lines = ["# Ticket12 seven-source development report", "",
        f"Status: `{result['status']}`. Global threshold: {result['probability_threshold']}. Old final-test: sealed.", "",
        "These scores measure P95/P80 operational weak-reference agreement on directly observed known cells. "
        "Unknown, missing, solid, invalid and failed paths are not negative examples. "
        "The spatial subset is ordered uniform patch sampling, not full-field coverage or a physical oracle.", ""]
    for run in result["runs"]:
        lines += [f"## {run['method']} seed {run['seed']}", "",
            f"Checkpoint: `{run['checkpoint']}`; completed epochs {run['completed_epochs']}, updates {run['global_step']}.", "",
            "| Source/split | Precision | Recall | F1 | IoU | AUPRC | Direct fluid coverage | Missing fluid | Failed/attempted paths |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for split in ("train", "calibration"):
            for name in SOURCE_ORDER:
                row = run["source_summaries"][f"{name}/{split}"]
                fluid = row["support"]["denominators"]["valid_fluid"]
                lines.append(f"| {name}/{split} | " + " | ".join(number(row["metrics"][key]) for key in METRIC_KEYS)
                    + f" | {number(fluid['coverage'])} | {fluid['missing']} | {row['failed_pathlines']}/{row['attempted_pathlines']} |")
        lines += ["", "| Macro/split | Precision | Recall | F1 | IoU | AUPRC |",
                  "|---|---:|---:|---:|---:|---:|"]
        for split in ("train", "calibration"):
            for group in ("old_six", "mfc", "seven"):
                lines.append(f"| {group}/{split} | " + " | ".join(number(run["macro"][split][group][key]) for key in METRIC_KEYS) + " |")
        lines += [""]
    lines += ["## Replication availability", ""]
    for method, replication in result["replication"].items():
        lines.append(f"{method}: available seeds {replication['available_seeds']}; missing seeds {replication['missing_seeds']}. "
                     "Means and sample standard deviations are reported only when all registered seeds are available.")
    lines += ["", "## Additional evidence", "",
        "Qrotstrain, lambda-ci and Gamma2 seven-source audits remain NA unless actual matched criterion artifacts are supplied. "
        "Gamma1 has no registered binary physical oracle. Historical six-source audits remain a separate scope. "
        "MFC Haller integration timed out after 600 seconds and supplies no usable anchor. "
        "W2 selective risk is NA until a seven-source gate and selected accepted-set predictions exist.", "",
        "The JSON includes every frame's support, unknown/solid/invalid denominators, failed path counts, registered coordinates, "
        "view seeds, per-source pooled metrics, macros and seed availability.", ""]
    return "\n".join(lines)
