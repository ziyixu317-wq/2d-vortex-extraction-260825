"""Bounded train/calibration response diagnosis of the completed mixed W1-P.

Spatial sampling is independent of weak labels. Missing and failed predictions
remain unobserved. This entry does not select a threshold or access final-test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

import dataset
import extractor
import infer_ticket04_current as inference
import mfc_mixed_w1p as mixed
import prediction_support
import run_mfc_mixed_w1p_training as training
import ticket04_real_training as legacy
import train_kaggle
import weak_supervision_contract as contract


def uniform_subset(values, count):
    """Deterministic ordered spatial/window subset; no label-based sampling."""
    values = tuple(values)
    indices = np.rint(np.linspace(0, len(values) - 1, min(count, len(values)))).astype(int)
    return tuple(values[i] for i in indices)


def binary_metrics(probability, state, valid):
    observed = np.isfinite(probability) & valid & np.isin(state, [0, 1])
    predicted = probability >= 0.1
    positive = state == 1
    counts = {"tp": int((observed & predicted & positive).sum()),
              "fp": int((observed & predicted & ~positive).sum()),
              "fn": int((observed & ~predicted & positive).sum()),
              "tn": int((observed & ~predicted & ~positive).sum())}
    return metrics_from_counts(counts)


def metrics_from_counts(counts):
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    available = tp + fp + fn + tn > 0
    return {**counts, "observed_known": tp + fp + fn + tn,
            "f1": (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.) if available else None,
            "iou": (tp / (tp + fp + fn) if tp + fp + fn else 0.) if available else None}


def aggregate_frames(rows):
    counts = {key: sum(row["metrics"][key] for row in rows) for key in ("tp", "fp", "fn", "tn")}
    return {"metrics": metrics_from_counts(counts),
            "support": prediction_support.merge_support_summaries([r["support"] for r in rows]),
            "attempted_pathlines": sum(r["attempted_pathlines"] for r in rows),
            "failed_pathlines": sum(r["failed_pathlines"] for r in rows)}


def infer_patches(model, patches, extract_sample, *, shape, valid, device, batch_size, model_view_seed):
    accumulator = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)
    attempted = failed = 0
    torch.manual_seed(model_view_seed)
    model.eval()
    for offset in range(0, len(patches), batch_size):
        samples = [extract_sample(patch) for patch in patches[offset:offset + batch_size]]
        paths = np.stack([sample["pathlines"] for sample in samples])
        with torch.inference_mode():
            values = model((torch.zeros((len(samples), 1, 1, 1), device=device),
                            torch.from_numpy(paths).to(device))).cpu().numpy()
        for value, sample in zip(values, samples):
            cells, failures = sample["seed_cells"], sample["failed"]
            attempted += len(failures)
            failed += int(failures.sum())
            keep = ~failures & valid[cells[:, 0], cells[:, 1]]
            j, i = cells[keep].T
            np.add.at(accumulator, (j, i), value[keep])
            np.add.at(count, (j, i), 1)
    probability = np.full(shape, np.nan, dtype=np.float32)
    observed = count > 0
    probability[observed] = accumulator[observed] / count[observed]
    return probability, count, attempted, failed


def legacy_sample(source, item, *, seed):
    cfg = source.config
    paths, cells = inference._extract_normalized(
        legacy, extractor, dataset, source.view, py=item[0], px=item[1], frame=item[2],
        **{k: v for k, v in asdict(cfg).items() if k not in ("stride", "window_step")}, seed=seed)
    failed = ~np.isclose(paths[-1, :, extractor.CH_T], cfg.t_scale, rtol=0, atol=1e-5)
    return {"pathlines": paths, "seed_cells": cells, "failed": failed}


def source_frames(source, split):
    if source.source_id == 6:
        available = source.source.window_starts(split, t_win=source.config.t_win,
                                               step=source.config.window_step)
        return uniform_subset(available, 3)
    if split == "train":
        return inference._snap_train_frames(legacy, source.view,
            inference.HALLER_REFERENCE_FRAMES[source.name], source.config.t_win)
    frames = inference.CALIBRATION_FRAMES[source.name]
    for frame in frames:
        if not inference._split_contains_window(legacy, source.view.record, split, frame, source.config.t_win):
            raise ValueError(f"incomplete calibration window: {source.name}/{frame}")
    return frames


def frame_reference(source, split, frame, calibration):
    if source.source_id == 6:
        cache = source.cache
        state = np.asarray(cache.state[frame])
        solid = source.source.geometry
        invalid = np.asarray(cache.invalid_mask[frame], dtype=bool)
        valid = np.asarray(cache.valid_fluid_mask[frame], dtype=bool) & ~solid & ~invalid
    else:
        view = source.view
        target = view.target if split == "train" else calibration
        row = {int(f): i for i, f in enumerate(target.frame_indices)}[frame]
        state = np.asarray(target.state[row])
        solid = view.mask
        standard_row = view.standard_row_by_frame[frame]
        invalid = np.asarray(view.standard_invalid[standard_row], dtype=bool)
        valid = np.asarray(view.standard_valid[standard_row], dtype=bool) & ~solid & ~invalid
    return state, valid, solid, invalid


def history_response(history, sampling_stages):
    result = {}
    for source_id in history[0]["sources"]:
        rows = [entry["sources"][source_id] for entry in history]
        samples = sum(r["samples"] for r in rows)
        failures = sum(r["failed_pathlines"] for r in rows)
        stages = []
        for stage in sampling_stages:
            lo, hi = stage["start_epoch"], min(stage["end_epoch"], len(history))
            selected = rows[lo:hi]
            stages.append({"epochs": [lo, hi], "batch_size": stage["batch_size"],
                           "samples": sum(r["samples"] for r in selected),
                           "updates": sum(r["updates"] for r in selected)})
        result[source_id] = {"name": rows[0]["name"], "samples": samples,
            "updates": sum(r["updates"] for r in rows), "failed_pathlines": failures,
            "failed_rate": failures / (samples * 256), "stages": stages,
            "first5_batch_mean_loss": float(np.mean([r["loss"] for r in rows[:5]])),
            "last5_batch_mean_loss": float(np.mean([r["loss"] for r in rows[-5:]]))}
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "ticket03-root", "mfc-flow", "mfc-mask", "mfc-cache", "config", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=18.)
    parser.add_argument("--max-patches-per-frame", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reproduce-from", type=Path, help="completed evaluation directory for one real reload/reproduction measurement")
    parser.add_argument("--reproduce-frame", help="registered source/split/frame identity to reproduce once")
    args = parser.parse_args(argv)
    if args.max_patches_per_frame <= 0 or args.inference_batch_size <= 0:
        parser.error("patch and batch budgets must be positive")
    reproduction = None
    if (args.reproduce_from is None) != (args.reproduce_frame is None):
        parser.error("reproduction requires both --reproduce-from and --reproduce-frame")
    if args.reproduce_from is not None:
        original = json.loads((args.reproduce_from / "evaluation_report.json").read_text(encoding="utf-8"))
        name, split, frame = args.reproduce_frame.split("/")
        frame = int(frame)
        rows = [r for r in original["frames"] if (r["name"], r["split"], r["frame"]) == (name, split, frame)]
        if original.get("status") != "complete" or len(rows) != 1:
            raise ValueError("reproduction requires a frame from a completed evaluation")
        if (original["seed"] != args.seed or original["inference_batch_size"] != args.inference_batch_size
                or original["checkpoint"] != str(args.checkpoint)):
            raise ValueError("reproduction uses the original checkpoint, seed and batch size")
        reproduction = {"original": original, "row": rows[0], "identity": (name, split, frame)}
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    device = training.apply_cuda_limit(args.device, args.gpu_memory_limit_gib)
    sources, datasets, splits, sampling = training.load_sources(args, cfg)
    model = train_kaggle.build_model_from_config(cfg, mode=contract.MODE_W1_P).to(device)
    blob = contract.load_checkpoint(args.checkpoint, student=model, device=device,
        expected_mode=contract.MODE_W1_P, expected_feature_schema=contract.FEATURE_SCHEMA_7,
        expected_dataset_config=datasets, expected_split_config=splits, expected_sampling_config=sampling,
        expected_label_source=contract.LABEL_SOURCE_P95P80,
        expected_sampling_source=contract.LABEL_SOURCE_P95P80, expected_anchor_hash=None,
        load_mode="inference", restore_rng=False)
    for module in model.modules():
        if hasattr(module, "activation_checkpointing"):
            module.activation_checkpointing = False
    progress = blob["extra_metadata"]["mixed_training_progress"]
    manifest = legacy.protocol.read_p95p80_protocol_manifest(args.ticket03_root / "p95p80_protocol_manifest.json")
    plan = []
    for source in sources:
        if source.source_id < 6:
            patches = source.view.usable_patches
        else:
            physical, config = source.source, source.config
            patches = tuple(patch for patch in dataset.patch_locations(physical.shape[1], physical.shape[2],
                config.patch_size, config.stride) if legacy._patch_is_usable(physical.geometry,
                    physical.xdim, physical.ydim, patch, patch_size=config.patch_size,
                    groups=config.groups, delta_frac=config.delta_frac))
        chosen = uniform_subset(patches, args.max_patches_per_frame)
        for split in ("train", "calibration"):
            plan.append({"source_id": source.source_id, "name": source.name, "split": split,
                         "frames": list(source_frames(source, split)), "available_patches": len(patches),
                         "selected_patches": [list(p) for p in chosen]})
    if reproduction is not None:
        name, split, frame = reproduction["identity"]
        original_plan = next(p for p in reproduction["original"]["frame_plan"] if p["name"] == name and p["split"] == split)
        current_plan = next(p for p in plan if p["name"] == name and p["split"] == split)
        if current_plan != original_plan:
            raise ValueError("reproduction input frame/patch plan differs from the saved evaluation")
        plan = [{**original_plan, "frames": [frame]}]
    report = {"scope": "bounded seven-source train/calibration response diagnosis",
        "checkpoint": str(args.checkpoint), "global_step": blob["global_step"],
        "completed_epochs": len(progress["history"]), "sampling_stages": progress["sampling_stages"],
        "training_status": "complete" if len(progress["history"]) == progress["sampling_stages"][-1]["end_epoch"] else "in_progress",
        "model_role": "student", "seed": args.seed, "probability_threshold": 0.1,
        "inference_batch_size": args.inference_batch_size,
        "model_views": "one stochastic grouping view per frame; Torch seed from [seed, source_id, split_id, frame]",
        "threshold_selected_in_this_run": False, "test_frame_indices_read": [],
        "patch_selection": "uniform ordered subset of usable spatial patches, independent of labels",
        "max_patches_per_frame": args.max_patches_per_frame, "frame_plan": plan,
        "normalization": "source-specific saved train statistics; no calibration fit",
        "metric_scope": "observed known seed grid cells versus P95/P80 operational weak reference",
        "all_rk4_internal_support_verified": False,
        "history_response": history_response(progress["history"], progress["sampling_stages"]),
        "frames": [], "source_summaries": {}}
    if reproduction is not None:
        report["scope"] = "single-frame classification checkpoint reload/reproduction measurement"
    if "seven_source_w3" in blob["extra_metadata"]:
        branch = blob["extra_metadata"]["seven_source_w3"]
        report["auxiliary_branch"] = {"name": "seven_source_w3", "config": branch["config"],
            "trajectory_identity_source": branch["trajectory_identity_source"],
            "haller_anchor_identity_source": branch["haller_anchor_identity_source"],
            "supervision_base": branch["supervision_base"],
            "projection_saved_in_checkpoint": "projection_state_dict" in branch,
            "classification_output": "student W1-P head; independent projection is not used for segmentation"}
    write_json(args.output / "evaluation_plan.json", {k: v for k, v in report.items() if k != "history_response"})
    write_json(args.output / "training_response.json", report["history_response"])
    started = time.monotonic()
    for source in sources:
        if not any(p["source_id"] == source.source_id for p in plan):
            continue
        calibration = (inference._load_relocated_calibration_artifact(legacy.w1_p,
            args.ticket03_root / "datasets" / source.name / "calibration_reference", manifest)
            if source.source_id < 6 else None)
        for entry in [p for p in plan if p["source_id"] == source.source_id]:
            split = entry["split"]
            for frame in entry["frames"]:
                frame_started = time.monotonic()
                state, valid, solid, invalid = frame_reference(source, split, frame, calibration)
                items = tuple((py, px, frame) for py, px in entry["selected_patches"])
                extract = (lambda item: source.extract_sample(item, seed=args.seed, split=split)) if source.source_id == 6 else (
                    lambda item: legacy_sample(source, item, seed=args.seed))
                model_view_seed = int(np.random.SeedSequence([
                    args.seed, source.source_id, 0 if split == "train" else 1, frame]).generate_state(1)[0])
                probability, count, attempted, failures = infer_patches(model, items, extract,
                    shape=state.shape, valid=valid, device=device, batch_size=args.inference_batch_size,
                    model_view_seed=model_view_seed)
                stem = f"{source.name}-{split}-{frame}"
                np.save(args.output / f"{stem}-probability.npy", probability)
                np.save(args.output / f"{stem}-count.npy", count)
                row = {"name": source.name, "source_id": source.source_id, "split": split, "frame": frame,
                    "selected_patch_count": len(items), "available_patch_count": entry["available_patches"],
                    "model_view_seed": model_view_seed,
                    "attempted_pathlines": attempted, "failed_pathlines": failures,
                    "failed_rate": failures / attempted, "elapsed_seconds": time.monotonic() - frame_started,
                    "metrics": binary_metrics(probability, state, valid),
                    "support": prediction_support.support_summary(count, state, valid, solid, invalid),
                    "probability_file": f"{stem}-probability.npy", "count_file": f"{stem}-count.npy"}
                report["frames"].append(row)
                write_json(args.output / "evaluation_report.json", report)
                print(json.dumps({"event": "frame", **row}), flush=True)
            rows = [r for r in report["frames"] if r["name"] == source.name and r["split"] == split]
            report["source_summaries"][f"{source.name}/{split}"] = aggregate_frames(rows)
    report["macro"] = {}
    if reproduction is not None:
        old = reproduction["row"]
        new = report["frames"][0]
        measurement = reproduction_difference(
            np.load(args.reproduce_from / old["probability_file"]), np.load(args.reproduce_from / old["count_file"]),
            np.load(args.output / new["probability_file"]), np.load(args.output / new["count_file"]))
        measurement.update(original_evaluation=str(args.reproduce_from), checkpoint=str(args.checkpoint),
            identity=list(reproduction["identity"]), seed=args.seed, model_view_seed=new["model_view_seed"],
            original_model_view_seed=old["model_view_seed"], inference_batch_size=args.inference_batch_size,
            elapsed_seconds=time.monotonic() - started, scope="ordinary real single-view measurement; no selection rule")
        report.update(status="measurement_complete", reproduction_measurement=measurement,
                      elapsed_seconds=measurement["elapsed_seconds"])
        write_json(args.output / "evaluation_report.json", report)
        write_json(args.output / "reproduction_measurement.json", measurement)
        print(json.dumps({"event": "reproduction_measurement", **measurement}), flush=True)
        return 0
    for split in ("train", "calibration"):
        for group, names in (("legacy_six", legacy.DATASET_ORDER), ("seven", [s.name for s in sources])):
            metrics = [report["source_summaries"][f"{name}/{split}"]["metrics"] for name in names]
            available = sum(m["f1"] is not None for m in metrics)
            report["macro"][f"{group}/{split}"] = {
                "source_count": len(names), "evaluable_source_count": available,
                **{key: float(np.mean([m[key] for m in metrics])) if available == len(names) else None
                   for key in ("f1", "iou")}}
    report.update(status="complete", elapsed_seconds=time.monotonic() - started)
    write_json(args.output / "evaluation_report.json", report)
    print(json.dumps({"event": "complete", "frames": len(report["frames"]), "macro": report["macro"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0


def reproduction_difference(original_probability, original_count, repeated_probability, repeated_count):
    original_probability, repeated_probability = np.asarray(original_probability), np.asarray(repeated_probability)
    original_count, repeated_count = np.asarray(original_count), np.asarray(repeated_count)
    if len({a.shape for a in (original_probability, original_count, repeated_probability, repeated_count)}) != 1:
        raise ValueError("reproduction grid shapes differ")
    original_observed, repeated_observed = np.isfinite(original_probability), np.isfinite(repeated_probability)
    common = original_observed & repeated_observed
    difference = np.abs(original_probability[common].astype(np.float64) - repeated_probability[common].astype(np.float64))
    return {"common_observed_cells": int(common.sum()),
            "changed_prediction_count_cells": int((original_count != repeated_count).sum()),
            "changed_finite_probability_cells": int((original_observed != repeated_observed).sum()),
            "maximum_abs_probability_difference": float(difference.max()) if difference.size else None,
            "mean_abs_probability_difference": float(difference.mean()) if difference.size else None}


if __name__ == "__main__":
    raise SystemExit(main())
