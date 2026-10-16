"""Prepare exact train-only MFC IVD statistics with a disk-backed value pool.

Shared standard-IVD computes each full frame before any values are selected.
Population moments and float32 speed maxima follow ticket04_real_training;
P80/P95 follow Ticket03's exact NumPy linear-percentile semantics. A partial
selection is a resource measurement, not a formal training-statistics artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from mfc_pathline_source import MFCPathlineSource
import standard_ivd


FULL_TRAIN_INDICES = tuple(range(400))
VALUE_DTYPE = np.dtype(np.float64)


def peak_rss_bytes():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def select_frames(source, mode, frame_indices):
    """Resolve explicit absolute train indices before any flow-field read."""
    train_start, train_stop = source.frame_ranges["train"]
    if mode not in ("full_train", "partial"):
        raise ValueError("mode must be full_train or partial")
    if mode == "full_train":
        if (train_start, train_stop) != (0, 400):
            raise ValueError("full_train requires the complete declared train [0,400)")
        frames = FULL_TRAIN_INDICES if frame_indices is None else tuple(frame_indices)
        if frames != FULL_TRAIN_INDICES:
            raise ValueError("full_train requires exactly the ordered indices 0..399")
    else:
        if frame_indices is None:
            raise ValueError("partial requires explicit frame_indices")
        frames = tuple(frame_indices)
    if (not frames or any(not isinstance(index, (int, np.integer)) for index in frames)
            or len(set(frames)) != len(frames)
            or any(not train_start <= index < train_stop for index in frames)):
        raise ValueError("frame_indices must be nonempty, unique indices entirely inside train")
    return tuple(int(index) for index in frames)


def prepare_statistics(source, output, *, mode, frame_indices=None, max_seconds=None):
    frames = select_frames(source, mode, frame_indices)
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive when supplied")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    value_path = output / "ivd_valid_values.float64.dat"
    report_path = output / "mfc_ivd_statistics.json"
    started = time.monotonic()
    count, ivd_sum, ivd_sum_sq, speed_max = 0, 0., 0., 0.
    report = {
        "status": "in_progress", "mode": mode, "statistics_split": "train",
        "scope": "full_train" if mode == "full_train" else "partial_resource_measurement",
        "formal_training_statistics": False,
        "source": source.metadata, "requested_frame_indices": list(frames),
        "actual_frame_indices": [], "frames": [],
        "support": "full-frame native derivative_valid & non-solid, intersect shared-IVD valid_fluid_mask & non-invalid & finite IVD",
        "ivd_source": {"entry_point": "standard_ivd.compute_standard_ivd_details",
                       "implementation_version": standard_ivd.IMPLEMENTATION_VERSION,
                       "source_identity": standard_ivd.SOURCE_IDENTITY,
                       "mean_scope": "full-frame declared valid-fluid support",
                       "mean_quantity": "signed omega"},
        "precision": {"ivd": "float64", "moment_accumulation": "float64",
                      "variance_ddof": 0, "variance_method": "max(sum_squares/count - mean**2,0)",
                      "speed": "numpy.hypot(float32 u,float32 v) on the same valid support",
                      "percentiles": "numpy.percentile([80,95],method='linear',overwrite_input=True)"},
        "disk": {"value_path": str(value_path), "report_path": str(report_path),
                 "value_dtype": VALUE_DTYPE.name, "value_dtype_descriptor": VALUE_DTYPE.str,
                 "value_count": 0, "value_bytes": 0,
                 "value_layout": "packed valid train IVD values; in-place percentile partitions and destroys frame order",
                 "value_role": "retained percentile working file, not a spatial IVD artifact",
                 "additional_full_field_artifacts": False,
                 "working_file_deleted": False},
        "normalization": None, "thresholds": None,
        "calibration_flow_read": False, "test_source_read": False,
        "training_executed": False, "max_seconds": max_seconds,
        "time_limit_scope": "checked between frames and before/after exact percentile; a NumPy percentile call cannot be interrupted internally",
        "timing": {"read_seconds": 0., "ivd_seconds": 0.,
                   "statistics_and_write_seconds": 0., "percentile_seconds": None},
    }

    def publish():
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_process_rss_bytes"] = peak_rss_bytes()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")

    def check_time():
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            raise TimeoutError("declared statistics max-seconds reached")

    publish()
    try:
        with value_path.open("xb") as packed:
            for index in frames:
                check_time()
                before = time.monotonic()
                frame = source.read_frame(index)
                read_seconds = time.monotonic() - before
                before = time.monotonic()
                details = standard_ivd.compute_standard_ivd_details(
                    frame.u, frame.v, frame.xdim, frame.ydim,
                    solid_mask=frame.geometry, valid_fluid_mask=frame.valid_fluid_support,
                    input_identity=f"MFC train statistics frame {index}: {source.flow_path}")
                ivd_seconds = time.monotonic() - before
                before = time.monotonic()
                support = details.valid_fluid_mask & ~details.solid_mask & ~details.invalid_mask
                support &= np.isfinite(details.ivd)
                values = np.asarray(details.ivd[support], dtype=VALUE_DTYPE)
                # Shared IVD already rejects any frame with empty support.
                frame_count = int(values.size)
                count += frame_count
                ivd_sum += float(values.sum(dtype=np.float64))
                ivd_sum_sq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
                speed = np.hypot(np.asarray(frame.u, dtype=np.float32),
                                 np.asarray(frame.v, dtype=np.float32))
                speed_values = speed[support]
                speed_values = speed_values[np.isfinite(speed_values)]
                frame_speed_max = float(speed_values.max()) if speed_values.size else None
                if frame_speed_max is not None:
                    speed_max = max(speed_max, frame_speed_max)
                values.tofile(packed)
                packed.flush()
                stats_seconds = time.monotonic() - before
                report["actual_frame_indices"].append(index)
                report["frames"].append({
                    "frame_index": index, "physical_time": float(frame.tdim[0]),
                    "solver_step": int(frame.solver_step[0]),
                    "omega_bar": float(details.omega_bar), "valid_fluid_count": frame_count,
                    "solid_count": int(details.solid_mask.sum()),
                    "invalid_fluid_count": int(details.invalid_mask.sum()),
                    "speed_max": frame_speed_max, "read_seconds": read_seconds,
                    "ivd_seconds": ivd_seconds, "statistics_and_write_seconds": stats_seconds})
                report["timing"]["read_seconds"] += read_seconds
                report["timing"]["ivd_seconds"] += ivd_seconds
                report["timing"]["statistics_and_write_seconds"] += stats_seconds
                report["disk"].update(value_count=count, value_bytes=count * VALUE_DTYPE.itemsize)
                del frame, details, support, values, speed, speed_values
                publish()
                print(json.dumps({"statistics_completed_train_frame": index,
                                  "valid_values_written": count,
                                  "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
        if count <= 0 or speed_max <= 0:
            raise ValueError("MFC has no valid train normalization support with positive speed_max")
        mean = ivd_sum / count
        sigma = math.sqrt(max(ivd_sum_sq / count - mean * mean, 0.))
        check_time()
        print(json.dumps({"exact_percentile_started": True, "value_count": count,
                          "working_file": str(value_path)}), flush=True)
        before = time.monotonic()
        values_mm = np.memmap(value_path, dtype=VALUE_DTYPE, mode="r+", shape=(count,))
        try:
            p80, p95 = np.percentile(values_mm, [80., 95.], method="linear", overwrite_input=True)
            values_mm.flush()
        finally:
            del values_mm
        report["timing"]["percentile_seconds"] = time.monotonic() - before
        check_time()
        report["normalization"] = {"ivd_mu": float(mean), "ivd_sigma": float(sigma),
                                   "speed_max": float(speed_max), "ivd_valid_cell_count": count,
                                   "statistics_split": "train"}
        report["thresholds"] = {"p80": float(p80), "p95": float(p95),
                                "negative": float(p80), "positive": float(p95),
                                "method": "linear", "valid_fluid_count": count,
                                "statistics_split": "train"}
        report.update(status="complete", formal_training_statistics=mode == "full_train",
                      complete_declared_train_coverage=report["actual_frame_indices"] == list(range(*source.frame_ranges["train"])))
    except TimeoutError as exc:
        report.update(status="time_limit", interruption=str(exc))
    except Exception as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        publish()
        raise
    publish()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; working values remain here")
    parser.add_argument("--mode", choices=("full_train", "partial"), required=True)
    parser.add_argument("--train-range", default="0,400")
    parser.add_argument("--calibration-range", default="400,501")
    parser.add_argument("--frame-indices", help="comma-separated absolute train frames; required for partial")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    source = MFCPathlineSource(args.flow, mask_path=args.mask, frame_ranges={
        "train": tuple(map(int, args.train_range.split(","))),
        "calibration": tuple(map(int, args.calibration_range.split(",")))})
    report = prepare_statistics(source, args.output, mode=args.mode,
        frame_indices=None if args.frame_indices is None else tuple(map(int, args.frame_indices.split(","))),
        max_seconds=args.max_seconds)
    print(json.dumps({"status": report["status"], "formal_training_statistics": report["formal_training_statistics"],
                      "actual_frame_count": len(report["actual_frame_indices"]),
                      "elapsed_seconds": report["elapsed_seconds"], "report": report["disk"]["report_path"]}), flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
