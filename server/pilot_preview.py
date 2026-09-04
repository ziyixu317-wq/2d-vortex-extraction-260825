"""Preview frozen B0/B1 pilot checkpoints without changing pilot selection.

This is a diagnostic-only entry point.  It reads each method's calibration
threshold from its completed worker JSON, loads the frozen checkpoint, and
uses ``haller_gt_test`` only for post-freeze test metrics and figures.  It
does not train, recalibrate, or select a method.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import dataset as dataset_module
import e2e_weak_supervision as e2e
import evaluation_report
import extractor
import haller_anchors
import prepare_weak_supervision_artifacts as artifact_preparer
import run_ws9_pilot as runner
import weak_supervision_contract as contract


MODES = (contract.MODE_B0, contract.MODE_B1)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 必须是 object：{path}")
    return value


def _pilot_context(
    *,
    model_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    weak_root: pathlib.Path,
    eval_samples: int,
    eval_batch_size: int,
    device: str,
) -> tuple[e2e.PilotConfig, runner.RealPilotData]:
    names = tuple(runner.DATASETS)
    split_ranges = {
        name: list(manifest["weak_contracts"][name]["split_ranges"]["train"])
        for name in names
    }
    dataset_config = {
        "datasets": list(names),
        "split_mode": dataset_module.WEAK_SUPERVISION_SPLIT_MODE,
        "weak_root": str(weak_root.resolve()),
        "artifact_manifest_hash": manifest["manifest_hash"],
    }
    split_config = {
        "split_name": "train",
        "split_ranges": split_ranges,
        "t_win": int(manifest["t_win"]),
        "window_step": int(manifest["window_step"]),
    }
    sampling_config = {
        "source": contract.LABEL_SOURCE_LEGACY_P85,
        "t_win": int(manifest["t_win"]),
        "window_step": int(manifest["window_step"]),
        "samples_per_epoch": 20_000,
        "batch_size": 100,
    }
    pilot_config = e2e.PilotConfig(
        dataset_config=dataset_config,
        split_config=split_config,
        sampling_config=sampling_config,
        epochs=e2e.PILOT_EPOCHS,
        seed=e2e.PILOT_SEED,
        device=device,
        variance_candidates=runner.PILOT_VARIANCE_CANDIDATES,
    )
    data_cfg = model_config.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise ValueError("model config data 必须是 object")
    train_cfg = model_config.get("train", {})
    if not isinstance(train_cfg, Mapping):
        raise ValueError("model config train 必须是 object")
    data = runner.RealPilotData(
        weak_root=weak_root,
        haller_root=pathlib.Path(manifest["haller_root"]),
        model_config=model_config,
        manifest_summary=manifest,
        batch_size=100,
        samples_per_epoch=20_000,
        data_workers=0,
        eval_samples_per_dataset=eval_samples,
        eval_batch_size=eval_batch_size,
        seed=e2e.PILOT_SEED,
        device=device,
        ramp_up_epochs=e2e.PILOT_RAMP_UP_EPOCHS,
        parallel_devices=(device,),
    )
    return pilot_config, data


def _build_method(
    data: runner.RealPilotData,
    *,
    mode: str,
    pilot_config: e2e.PilotConfig,
    model_config: Mapping[str, Any],
    device: str,
) -> e2e.PilotMethod:
    train_cfg = model_config["train"]
    methods = data.build_methods(
        pilot_config=pilot_config,
        lr=float(train_cfg.get("lr", 1e-4)),
        second_lr=float(train_cfg.get("second_lr", 5e-6)),
        warmup_epochs=int(train_cfg.get("warmup_epochs", 60)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        modes=(mode,),
        method_devices={mode: device},
        method_parallel_devices={mode: (device,)},
    )
    method = methods[mode]
    # The formal multi-GPU pilot checkpoint was saved with the vendor model
    # under ``model.module``.  Keep the strict checkpoint contract intact and
    # use a one-device DataParallel shell for this inference-only preview so
    # the saved state has the same topology.  It does not change the model,
    # batch, or training semantics.
    if str(device).startswith("cuda"):
        student = method.trainer.student
        inner = student.model
        if not isinstance(inner, torch.nn.DataParallel):
            parsed = torch.device(str(device))
            index = torch.cuda.current_device() if parsed.index is None else int(parsed.index)
            student.model = torch.nn.DataParallel(
                inner, device_ids=[index], output_device=index
            )
    return method


def _project_probabilities(
    data: runner.RealPilotData,
    model: torch.nn.Module,
    *,
    mode: str,
    dataset_index: int,
    frame: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer every usable patch for one frame and project trajectory scores."""
    eval_dataset = data._eval_dataset("test", "evaluation")
    store = eval_dataset.stores[dataset_index]
    patches = list(store._usable_patches)
    if not patches:
        raise ValueError(f"test/{store.dataset_name} 没有可用 patch")
    pathlines = []
    seeds = []
    for py, px in patches:
        (_dummy, raw), _sampling_labels, sample_seeds = eval_dataset.sample_at(
            dataset_index, int(py), int(px), int(frame)
        )
        pathlines.append(np.asarray(raw, dtype=np.float32))
        seeds.append(np.asarray(sample_seeds, dtype=np.float64))
    raw_array = np.stack(pathlines, axis=0)
    seed_array = np.stack(seeds, axis=0)
    predictions = []
    for start in range(0, raw_array.shape[0], batch_size):
        predictions.append(data._predict_once(
            model,
            mode,
            raw_array[start:start + batch_size],
            seed=data.seed + 100_003 * (start + 1),
            consumer="evaluation",
        ))
    prediction = np.concatenate(predictions, axis=0)
    flat_seeds = seed_array.reshape(-1, seed_array.shape[-1])
    flat_prediction = prediction.reshape(-1)
    rows, cols = extractor.nearest_cell(
        flat_seeds[:, 0], flat_seeds[:, 1], store._xdim, store._ydim
    )
    rows = np.clip(rows, 0, store.Y - 1)
    cols = np.clip(cols, 0, store.X - 1)
    accumulated = np.zeros((store.Y, store.X), dtype=np.float32)
    counts = np.zeros((store.Y, store.X), dtype=np.float32)
    np.add.at(accumulated, (rows, cols), flat_prediction)
    np.add.at(counts, (rows, cols), 1.0)
    probability = np.divide(
        accumulated,
        np.maximum(counts, 1.0),
        out=np.zeros_like(accumulated),
        where=counts > 0,
    )
    probability[np.asarray(store._mask2d, dtype=bool)] = 0.0
    return probability.astype(np.float32), counts.astype(np.float32)


def _frame_starts(data: runner.RealPilotData, dataset_index: int, count: int) -> list[int]:
    eval_dataset = data._eval_dataset("test", "evaluation")
    store = eval_dataset.stores[dataset_index]
    starts = dataset_module.window_starts(
        store.split_i0,
        store.split_i1,
        data.t_win,
        data.window_step,
        dataset_name=store.dataset_name,
        split_name="test",
        T=store.T,
    )
    if len(starts) == 0:
        raise ValueError(f"test/{store.dataset_name} 没有完整窗口")
    indices = np.linspace(0, len(starts) - 1, min(count, len(starts)), dtype=np.int64)
    return [int(starts[int(index)]) for index in np.unique(indices)]


def _frame_reference(
    data: runner.RealPilotData,
    *,
    dataset_index: int,
    frame: int,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eval_dataset = data._eval_dataset("test", "evaluation")
    store = eval_dataset.stores[dataset_index]
    name = store.dataset_name
    artifact = haller_anchors.load_haller_artifact(
        data.haller_root / runner.TEST_SOURCE / name / f"frame{frame}",
        expected_source=runner.TEST_SOURCE,
    )
    state = np.asarray(artifact["anchor_state"], dtype=np.int8)
    solid = np.asarray(artifact["solid_mask"], dtype=bool)
    invalid = not bool(artifact["metadata"].get("valid", False))
    known = (state >= 0) & ~solid & ~invalid
    label = np.where(known, state.astype(np.float32), np.nan)
    ivd = np.asarray(artifact["standard_ivd"], dtype=np.float32)
    mask = np.asarray(store._mask2d, dtype=bool)
    if invalid:
        label[...] = np.nan
    return name, label, ivd, solid, mask, state.astype(np.float32), np.asarray(
        artifact["metadata"].get("failure_count", 0), dtype=np.int64
    )


def _write_figure(
    path: pathlib.Path,
    *,
    name: str,
    frame: int,
    b0_probability: np.ndarray,
    b1_probability: np.ndarray,
    haller_label: np.ndarray,
    standard_ivd: np.ndarray,
    solid: np.ndarray,
    mask: np.ndarray,
    thresholds: Mapping[str, float],
    xdim: np.ndarray,
    ydim: np.ndarray,
) -> None:
    extent = [float(xdim[0]), float(xdim[-1]), float(ydim[0]), float(ydim[-1])]
    b0 = np.where(solid | mask, np.nan, b0_probability)
    b1 = np.where(solid | mask, np.nan, b1_probability)
    ivd = np.where(solid | mask, np.nan, standard_ivd)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = (
        (axes[0, 0], b0, f"B0 probability (threshold={thresholds['B0']:.3f})", "viridis", 0.0, 1.0),
        (axes[0, 1], b1, f"B1 probability (threshold={thresholds['B1']:.3f})", "viridis", 0.0, 1.0),
        (axes[1, 0], haller_label, "Haller GT (test; evaluation only)", "coolwarm", 0.0, 1.0),
        (axes[1, 1], ivd, "Haller standard IVD (test reference)", "turbo", None, None),
    )
    for axis, values, title, cmap, vmin, vmax in panels:
        image = axis.imshow(values, origin="lower", extent=extent, cmap=cmap,
                            vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"{name} frame {frame} — frozen B0/B1 checkpoint preview")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B0/B1 frozen pilot checkpoint preview")
    parser.add_argument("--weak-root", default="outputs/weak_supervision_numbacs_native")
    parser.add_argument("--haller-root", default="outputs/haller_artifacts_numbacs_native")
    parser.add_argument("--model-config", default="config/pathline_transformer_b1.yaml")
    parser.add_argument("--b0-checkpoint", required=True)
    parser.add_argument("--b1-checkpoint", required=True)
    parser.add_argument("--b0-calibration", required=True)
    parser.add_argument("--b1-calibration", required=True)
    parser.add_argument("--output-dir", default="outputs/b0_b1_preview_numbacs_native")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-samples-per-dataset", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--figure-dataset", default="pipedcylinder2d")
    parser.add_argument("--figure-count", type=int, default=3)
    parser.add_argument("--figure-batch-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA preview 但 CUDA 不可用")
        torch.cuda.set_device(torch.device(args.device))
    weak_root = pathlib.Path(args.weak_root).resolve()
    haller_root = pathlib.Path(args.haller_root).resolve()
    model_config_path = pathlib.Path(args.model_config)
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    if not isinstance(model_config, Mapping):
        raise ValueError("model config 必须是 object")
    manifest = runner.validate_prepared_artifacts(
        weak_root,
        haller_root,
        t_win=model_config.get("data", {}).get("t_win"),
        window_step=model_config.get("data", {}).get("window_step"),
    )
    manifest = dict(manifest)
    manifest["haller_root"] = str(haller_root)
    pilot_config, data = _pilot_context(
        model_config=model_config,
        manifest=manifest,
        weak_root=weak_root,
        eval_samples=int(args.eval_samples_per_dataset),
        eval_batch_size=int(args.eval_batch_size),
        device=str(args.device),
    )
    calibration_paths = {
        contract.MODE_B0: pathlib.Path(args.b0_calibration),
        contract.MODE_B1: pathlib.Path(args.b1_calibration),
    }
    checkpoint_paths = {
        contract.MODE_B0: pathlib.Path(args.b0_checkpoint),
        contract.MODE_B1: pathlib.Path(args.b1_checkpoint),
    }
    thresholds: dict[str, float] = {}
    for mode, path in calibration_paths.items():
        payload = _read_json(path)
        if payload.get("status") != "complete" or payload.get("mode") != mode:
            raise ValueError(f"{mode} calibration JSON 不是 complete result：{path}")
        policy = payload.get("calibration_policy")
        if not isinstance(policy, Mapping) or policy.get("source") != evaluation_report.CALIBRATION_SOURCE:
            raise ValueError(f"{mode} 缺少 calibration-only policy：{path}")
        thresholds[mode] = float(payload["prediction_threshold"])

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities: dict[str, dict[tuple[str, int], np.ndarray]] = {mode: {} for mode in MODES}
    metrics: dict[str, Any] = {}
    for mode in MODES:
        method = _build_method(
            data,
            mode=mode,
            pilot_config=pilot_config,
            model_config=model_config,
            device=str(args.device),
        )
        loaded = method.load_checkpoint(checkpoint_paths[mode])
        if loaded.get("mode") != mode or loaded.get("epoch") != e2e.PILOT_EPOCHS:
            raise ValueError(f"{mode} checkpoint round-trip metadata 不匹配")
        records = data.test_records(mode, method.trainer, variance_gate=None)
        metrics[mode] = evaluation_report.build_evaluation_report(
            records,
            prediction_threshold=thresholds[mode],
            variance_gate=None,
            method=mode,
            dataset_names=runner.DATASETS,
            checkpoint_epoch=e2e.PILOT_EPOCHS,
        )
        figure_dataset_index = runner.DATASETS.index(args.figure_dataset)
        frames = _frame_starts(data, figure_dataset_index, int(args.figure_count))
        for frame in frames:
            eval_dataset = data._eval_dataset("test", "evaluation")
            probability, _counts = _project_probabilities(
                data,
                method.trainer.student,
                mode=mode,
                dataset_index=figure_dataset_index,
                frame=frame,
                batch_size=int(args.figure_batch_size),
            )
            probabilities[mode][(args.figure_dataset, frame)] = probability
        del method
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eval_dataset = data._eval_dataset("test", "evaluation")
    store = eval_dataset.stores[runner.DATASETS.index(args.figure_dataset)]
    figure_paths = []
    frames = sorted({frame for name, frame in probabilities["B0"] if name == args.figure_dataset})
    for frame in frames:
        name, label, standard_ivd, solid, mask, _state, _failure = _frame_reference(
            data,
            dataset_index=runner.DATASETS.index(args.figure_dataset),
            frame=frame,
        )
        figure_path = output_dir / "figures" / f"{name}_t{frame:04d}_b0_b1.png"
        _write_figure(
            figure_path,
            name=name,
            frame=frame,
            b0_probability=probabilities["B0"][(name, frame)],
            b1_probability=probabilities["B1"][(name, frame)],
            haller_label=label,
            standard_ivd=standard_ivd,
            solid=solid,
            mask=mask,
            thresholds=thresholds,
            xdim=np.asarray(store._xdim),
            ydim=np.asarray(store._ydim),
        )
        figure_paths.append(str(figure_path))

    report = {
        "schema_version": "b0-b1-frozen-preview-v1",
        "status": "diagnostic_preview_only",
        "label_source": evaluation_report.TEST_SOURCE,
        "threshold_source": {
            mode: str(calibration_paths[mode].resolve()) for mode in MODES
        },
        "checkpoint_paths": {
            mode: str(checkpoint_paths[mode].resolve()) for mode in MODES
        },
        "thresholds": thresholds,
        "eval_samples_per_dataset": int(args.eval_samples_per_dataset),
        "eval_batch_size": int(args.eval_batch_size),
        "figure_dataset": args.figure_dataset,
        "figure_paths": figure_paths,
        "metrics": metrics,
        "selection_or_model_tuning_performed": False,
        "artifact_manifest": manifest["manifest_path"],
        "artifact_manifest_hash": manifest["manifest_hash"],
    }
    report_path = output_dir / "b0_b1_preview_report.json"
    report_path.write_text(
        json.dumps(e2e._jsonable(report), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "report": str(report_path),
        "figures": figure_paths,
        "thresholds": thresholds,
        "macro": {
            mode: metrics[mode]["macro"] for mode in MODES
        },
    }
    print(json.dumps(e2e._jsonable(summary), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
