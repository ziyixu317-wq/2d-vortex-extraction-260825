"""Re-score existing registered train/calibration predictions; never open final-test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import prediction_support

import evaluate_mfc_mixed_w1p as evaluation
import run_mfc_mixed_w1p_training as training
import ticket04_real_training as legacy
import ticket12_seven_source_report as report


def evaluation_complete(root):
    path = Path(root) / "evaluation_report.json"
    if not path.is_file():
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A process can disappear during the existing per-frame JSON write.
        return False
    if saved.get("status") != "complete":
        return False
    expected = {(p["name"], p["split"], frame) for p in saved["frame_plan"] for frame in p["frames"]}
    actual = {(r["name"], r["split"], r["frame"]) for r in saved["frames"]}
    return bool(expected) and actual == expected and len(actual) == len(saved["frames"])


def evaluation_attempt(root):
    """Reuse a completed attempt or retain incomplete attempts and select a new directory."""
    root = Path(root)
    if not root.exists() or evaluation_complete(root):
        return root
    attempt = 1
    while True:
        candidate = root.with_name(f"{root.name}-retry-{attempt:03d}")
        if not candidate.exists() or evaluation_complete(candidate):
            return candidate
        attempt += 1


def infer_missing_runs(args, spec, infer=evaluation.main):
    for entry in spec["runs"]:
        registered = Path(entry["evaluation_root"])
        root = evaluation_attempt(registered)
        if root != registered:
            entry["registered_evaluation_root"] = str(registered)
            entry["evaluation_root"] = str(root)
        if evaluation_complete(root):
            print(json.dumps({"event": "reuse_completed_inference", "method": entry["method"],
                              "seed": entry["seed"], "evaluation_root": str(root)}), flush=True)
            continue
        cli = []
        for name in ("input-root", "ticket03-root", "mfc-flow", "mfc-mask", "mfc-cache"):
            cli += ["--" + name, str(getattr(args, name.replace("-", "_")))]
        cli += ["--config", str(entry.get("config", args.config)), "--checkpoint", entry["checkpoint"],
            "--output", str(root), "--device", args.device, "--gpu-memory-limit-gib", "18",
            "--max-patches-per-frame", "64", "--inference-batch-size", "16", "--seed", str(entry["seed"])]
        print(json.dumps({"event": "registered_inference", "method": entry["method"], "seed": entry["seed"],
                          "registered_evaluation_root": str(registered), "argv": cli}), flush=True)
        infer(cli)


def rebuild_run(entry, reference, *, threshold, expected_plan=None):
    root = Path(entry["evaluation_root"])
    original = json.loads((root / "evaluation_report.json").read_text(encoding="utf-8"))
    if original.get("test_frame_indices_read") != []:
        raise ValueError("development report requires explicit empty test access")
    if original["seed"] != entry["seed"]:
        raise ValueError("evaluation seed differs from registered run")
    if original.get("status") != "complete":
        raise ValueError("prediction evaluation is incomplete")
    if original.get("training_status") != "complete" or original["completed_epochs"] != 100 or original["global_step"] != 39100:
        raise ValueError("development replication requires the completed registered 100-epoch model")
    if expected_plan is not None and original["frame_plan"] != expected_plan:
        raise ValueError("evaluation spatial/frame plans differ")
    frames, grouped, scores, labels = [], {}, {}, {}
    for old in original["frames"]:
        split, name, frame = old["split"], old["name"], old["frame"]
        if split not in ("train", "calibration"):
            raise ValueError("only development splits are permitted")
        state, valid, solid, invalid = reference(name, split, frame)
        scored, score, label = report.score_frame(np.load(root / old["probability_file"]),
            np.load(root / old["count_file"]), state, valid, solid, invalid, threshold=threshold)
        row = {**old, **scored}
        frames.append(row)
        key = f"{name}/{split}"
        grouped.setdefault(key, []).append(row)
        scores.setdefault(key, []).append(score)
        labels.setdefault(key, []).append(label)
    expected = {(p["name"], p["split"], frame) for p in original["frame_plan"] for frame in p["frames"]}
    actual = {(r["name"], r["split"], r["frame"]) for r in frames}
    if actual != expected or len(actual) != len(frames):
        raise ValueError("prediction rows must cover the registered plan exactly once")
    summaries = {key: report.summarize_frames(rows, scores[key], labels[key]) for key, rows in grouped.items()}
    return {"method": entry["method"], "seed": entry["seed"], "evaluation_root": str(root),
            "method_role": entry.get("method_role", "Ticket10 unconditional development method"),
            "auxiliary_branch": original.get("auxiliary_branch"),
            "checkpoint": original["checkpoint"], "global_step": original["global_step"],
            "completed_epochs": original["completed_epochs"], "sampling_stages": original["sampling_stages"],
            "frame_plan": original["frame_plan"], "inference_batch_size": original["inference_batch_size"],
            "model_views": original["model_views"], "threshold": threshold,
            "frames": frames, "source_summaries": summaries,
            "macro": {split: report.source_macros(summaries, split) for split in ("train", "calibration")}}


def paired_common_report(entries, runs, reference, threshold):
    """Compare classification probabilities only on per-frame common known support."""
    paired = []
    for seed in sorted({run["seed"] for run in runs}):
        members = [run for run in runs if run["seed"] == seed]
        if len(members) < 2:
            continue
        entry_by_method = {entry["method"]: entry for entry in entries if entry["seed"] == seed}
        frames_by_method = {run["method"]: {(r["name"], r["split"], r["frame"]): r for r in run["frames"]} for run in members}
        collected = {run["method"]: {} for run in members}
        common_rows = []
        for identity in next(iter(frames_by_method.values())):
            name, split, frame = identity
            counts, probabilities = {}, {}
            for method, frames in frames_by_method.items():
                root, row = Path(entry_by_method[method]["evaluation_root"]), frames[identity]
                counts[method] = np.load(root / row["count_file"])
                probabilities[method] = np.load(root / row["probability_file"])
            common, common_summary = prediction_support.common_support(counts)
            state, valid, solid, invalid = reference(name, split, frame)
            common_rows.append({"name": name, "split": split, "frame": frame, **common_summary})
            for method in counts:
                probability = np.where(common, probabilities[method], np.nan)
                count = np.where(common, counts[method], 0)
                scored, scores, labels = report.score_frame(probability, count, state, valid, solid, invalid, threshold=threshold)
                # Extraction failures are method-specific and retained from the original run.
                row = {**frames_by_method[method][identity], **scored}
                collected[method].setdefault(f"{name}/{split}", []).append((row, scores, labels))
        method_rows = {}
        for method, grouped in collected.items():
            summaries = {key: report.summarize_frames([v[0] for v in values], [v[1] for v in values], [v[2] for v in values])
                         for key, values in grouped.items()}
            method_rows[method] = {"source_summaries": summaries,
                "macro": {split: report.source_macros(summaries, split) for split in ("train", "calibration")}}
        paired.append({"seed": seed, "methods": method_rows, "common_frames": common_rows,
                       "scope": "classification probabilities on common direct support; independent projection is not a segmentation head"})
    return paired


def ticket09_decision_inputs(paired):
    rows = []
    for pair in paired:
        methods = pair["methods"]
        if not {"W1-P", "W3"}.issubset(methods):
            continue
        row = {"seed": pair["seed"], "support": "per-frame common direct observed-known"}
        for method, prefix in (("W1-P", "baseline"), ("W3", "w3")):
            item = methods[method]
            macro = item["macro"]["calibration"]["seven"]
            row[prefix + "_macro_f1"], row[prefix + "_macro_iou"] = macro["f1"], macro["iou"]
            values = [item["source_summaries"][f"{name}/calibration"]["metrics"]["f1"] for name in report.SOURCE_ORDER]
            row[prefix + "_worst_dataset_f1"] = min(values) if all(v is not None for v in values) else None
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "ticket03-root", "mfc-flow", "mfc-mask", "mfc-cache", "config", "runs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, help="Ticket10 method/threshold decision JSON")
    parser.add_argument("--infer-missing", action="store_true", help="run actual registered inference for missing evaluation directories")
    parser.add_argument("--device", default="cuda:0", help="single visible GPU for --infer-missing")
    args = parser.parse_args(argv)
    spec = json.loads(args.runs.read_text(encoding="utf-8"))
    threshold = float(spec["probability_threshold"])
    if args.infer_missing:
        infer_missing_runs(args, spec)
    sources, _, _, _ = training.load_sources(args, json.loads(args.config.read_text(encoding="utf-8")))
    manifest = legacy.protocol.read_p95p80_protocol_manifest(args.ticket03_root / "p95p80_protocol_manifest.json")
    by_name = {source.name: source for source in sources}
    calibrations = {s.name: evaluation.inference._load_relocated_calibration_artifact(legacy.w1_p,
        args.ticket03_root / "datasets" / s.name / "calibration_reference", manifest) for s in sources if s.source_id < 6}

    def reference(name, split, frame):
        source = by_name[name]
        if frame not in evaluation.source_frames(source, split):
            raise ValueError("prediction frame is outside registered development plan")
        return evaluation.frame_reference(source, split, frame, calibrations.get(name))

    runs, plan = [], None
    for entry in spec["runs"]:
        run = rebuild_run(entry, reference, threshold=threshold, expected_plan=plan)
        plan = run["frame_plan"]
        runs.append(run)
        print(json.dumps({"event": "rescored", "method": run["method"], "seed": run["seed"],
                          "calibration": run["macro"]["calibration"]}), flush=True)
    replication = report.replication_summary(runs, tuple(spec.get("required_seeds", [0, 1, 2])))
    result = {"scope": "seven-source bounded train/calibration development report",
        "status": "complete" if all(v["status"] == "complete" for v in replication.values()) else "awaiting_seed_evaluations",
        "probability_threshold": threshold, "threshold_rule": spec["threshold_rule"],
        "test_frame_indices_read": [], "old_final_test": "sealed", "runs": runs, "replication": replication,
        "paired_common_support": paired_common_report(spec["runs"], runs, reference, threshold),
        "auprc_definition": "average precision: step-integral PR area with equal-score groups pooled",
        "reference": "source train-derived P95/P80 operational known reference; unknown excluded from binary metrics",
        "aggregation": "pool frame counts/scores within each source; equal source macro; equal seed mean and sample std",
        "interpretation": "agreement on directly observed known support; no independent physical accuracy or full-field claim",
        "cross_criterion": {"standard_ivd": "reported through P95/P80 operational segmentation reference",
            "Qrotstrain": {"status": "NA", "reason": "no seven-source criterion audit supplied; historical six-source audit is a separate scope"},
            "lambda_ci": {"status": "NA", "reason": "no seven-source criterion audit supplied"},
            "Gamma1": {"status": "NA", "reason": "registered definition supplies no binary physical oracle"},
            "Gamma2": {"status": "NA", "reason": "no seven-source scale-sensitive audit supplied"}},
        "haller": {"status": "diagnostic_only", "mfc_airfoil": {"status": "NA",
            "reason": "600-second integration timed out; no usable anchor available; unknown is not negative"}},
        "selective": {"status": "NA", "reason": "W2 has no selected seven-source gate and accepted-set predictions in this run"}}
    if args.protocol:
        result["ticket10_decision"] = json.loads(args.protocol.read_text(encoding="utf-8"))
        result["ticket10_decision_file"] = str(args.protocol)
    result["ticket09_paired_decision_inputs"] = ticket09_decision_inputs(result["paired_common_support"])
    args.output.mkdir(parents=True, exist_ok=False)
    evaluation.write_json(args.output / "ticket12_seven_source_report.json", result)
    evaluation.write_json(args.output / "evaluation_runs.json", spec)
    (args.output / "report.md").write_text(report.markdown_report(result), encoding="utf-8")
    print(json.dumps({"event": "report", "status": result["status"], "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
