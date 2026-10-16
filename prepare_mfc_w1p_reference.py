"""Prepare full-frame MFC W1-P references from completed train statistics.

Train and calibration use the same source-specific train normalization and
P80/P95. Shared standard-IVD precedes full-frame positive-component filtering.
Arrays use absolute saved-frame indexing; consumers must require their window
to be present in prepared_frame_indices, including when this is a partial run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from mfc_pathline_source import MFCPathlineSource
import standard_ivd
import ticket03_standard_ivd_artifacts as labels


ARRAYS = {"standard_ivd": np.float64, "state": np.int8,
          "valid_fluid_mask": np.bool_, "invalid_mask": np.bool_}


def read_train_statistics(path, source):
    statistics = json.loads(Path(path).read_text(encoding="utf-8"))
    train_range = list(source.frame_ranges["train"])
    actual_train = list(range(*source.frame_ranges["train"]))
    if (statistics["status"] != "complete" or statistics["mode"] != "full_train"
            or statistics["formal_training_statistics"] is not True
            or statistics["actual_frame_indices"] != actual_train):
        raise ValueError("cache requires completed full_train statistics covering every declared train frame")
    recorded = statistics["source"]
    if (recorded["source_name"] != source.source_name
            or recorded["shape_tyx"] != list(source.shape)
            or recorded["frame_ranges"] != {name: list(bounds) for name, bounds in source.frame_ranges.items()}
            or Path(recorded["flow_path"]).resolve() != source.flow_path.resolve()):
        raise ValueError("train statistics and reference source/ranges differ")
    if statistics["ivd_source"]["source_identity"] != standard_ivd.SOURCE_IDENTITY:
        raise ValueError("train statistics use a different shared standard-IVD source")
    norm, thresholds = statistics["normalization"], statistics["thresholds"]
    if norm["statistics_split"] != "train" or thresholds["statistics_split"] != "train":
        raise ValueError("normalization and thresholds must come from train")
    if (not np.isfinite([norm["ivd_mu"], norm["ivd_sigma"], norm["speed_max"],
                         thresholds["p80"], thresholds["p95"]]).all()
            or norm["ivd_sigma"] < 0 or norm["speed_max"] <= 0
            or not 0 <= thresholds["p80"] < thresholds["p95"]):
        raise ValueError("train normalization/threshold values do not support P95/P80 classification")
    return {
        "source_split": "train", "train_frame_range": train_range,
        "ivd_mu": float(norm["ivd_mu"]), "ivd_sigma": float(norm["ivd_sigma"]),
        "speed_max": float(norm["speed_max"]),
    }, {
        "statistics_split": "train", "train_frame_range": train_range,
        "p80": float(thresholds["p80"]), "p95": float(thresholds["p95"]),
    }


def select_reference_frames(source, mode, frame_indices):
    allowed = sorted({frame for bounds in source.frame_ranges.values() for frame in range(*bounds)})
    if mode == "full":
        if frame_indices is not None:
            raise ValueError("full mode selects all train and calibration frames")
        return tuple(allowed)
    if mode != "partial" or frame_indices is None:
        raise ValueError("partial mode requires explicit frame indices or a frame range")
    frames = tuple(frame_indices)
    if (not frames or any(not isinstance(frame, (int, np.integer)) for frame in frames)
            or len(set(frames)) != len(frames) or any(frame not in allowed for frame in frames)):
        raise ValueError("reference frames must be unique and inside declared train/calibration")
    return tuple(int(frame) for frame in frames)


def peak_rss_bytes():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class ReferenceTimeLimit(TimeoutError):
    pass


def prepare_reference(source, statistics_path, output, *, mode,
                      frame_indices=None, max_seconds=None):
    normalization, thresholds = read_train_statistics(statistics_path, source)
    frames = select_reference_frames(source, mode, frame_indices)
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive when provided")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata_path = output / "cache_metadata.json"
    started = time.monotonic()
    all_frames = sorted({frame for bounds in source.frame_ranges.values() for frame in range(*bounds)})
    metadata = {
        "source_name": source.source_name, "flow_path": str(source.flow_path.resolve()),
        "frame_ranges": {name: list(bounds) for name, bounds in source.frame_ranges.items()},
        "shape_tyx": list(source.shape), "frame_indexing": "absolute saved global frame index",
        "standard_ivd_source_identity": standard_ivd.SOURCE_IDENTITY,
        "ivd_mean_scope": "full_frame_valid_fluid",
        "ivd_support": "native derivative_valid & non-solid, further restricted by shared finite velocity/vorticity support",
        "normalization": normalization, "thresholds": thresholds,
        "statistics_path": str(Path(statistics_path).resolve()),
        "statistics_refitted": False, "protocol_status": "development_candidate",
        "statistics_coverage_role": "full candidate train coverage, not approval of a final training protocol",
        "classification": {"entry_point": "ticket03_standard_ivd_artifacts.classify_p95p80_frame",
                           "positive_min_area_cells": labels.DEFAULT_MIN_AREA_CELLS,
                           "connectivity": labels.DEFAULT_CONNECTIVITY,
                           "component_scope": "full_frame_before_any_patch_crop",
                           "filtered_positive_state": int(labels.UNKNOWN)},
        "state_codes": {"positive": 1, "negative": 0, "unknown": -1, "solid": -2, "invalid": -3},
        "mode": mode, "requested_frame_indices": list(frames), "prepared_frame_indices": [],
        "status": "initializing", "complete": False, "actual_prepared_frame_count": 0,
        "arrays_initialized": False,
        "unprepared_frame_values": {"standard_ivd": 0., "state": -3,
                                    "valid_fluid_mask": False, "invalid_mask": True},
        "consumer_support": "only consume complete windows in prepared_frame_indices; unprepared zero IVD is not evidence",
        "calibration_flow_read": False, "calibration_frame_indices_read": [],
        "calibration_role": "reference construction using the unchanged train normalization and thresholds",
        "test_source_read": False, "training_executed": False,
        "disk": {"arrays": {name: {"path": str(output / f"{name}.npy"),
                                    "dtype": np.dtype(dtype).name} for name, dtype in ARRAYS.items()},
                 "array_payload_bytes": int(np.prod(source.shape)) * sum(np.dtype(dtype).itemsize for dtype in ARRAYS.values()),
                 "files_deleted": False},
        "max_seconds": max_seconds,
        "time_limit_scope": "between frames; an individual read/IVD/full-frame component call cannot be interrupted internally",
        "frames": [], "timing": {"initialize_seconds": None, "read_seconds": 0.,
                                   "ivd_seconds": 0., "classification_seconds": 0., "write_seconds": 0.},
    }

    def publish():
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["peak_process_rss_bytes"] = peak_rss_bytes()
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")

    def check_time():
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            raise ReferenceTimeLimit("declared reference max-seconds reached")

    publish()
    maps = {}
    try:
        before = time.monotonic()
        for name, dtype in ARRAYS.items():
            maps[name] = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+",
                                                  dtype=dtype, shape=source.shape)
        # Newly created IV D/valid files are zero-filled by file extension.
        # Explicit invalid states/masks cover every unprepared global frame.
        # Initialization completes before reading any frame, so a partial cache
        # never exposes an unprepared cell as negative/valid evidence.
        for frame_index in range(source.shape[0]):
            maps["state"][frame_index] = labels.INVALID
            maps["invalid_mask"][frame_index] = True
        for array in maps.values():
            array.flush()
        metadata["timing"]["initialize_seconds"] = time.monotonic() - before
        metadata.update(status="in_progress", arrays_initialized=True)
        publish()
        # The six-source P95P80Thresholds class registers historical datasets.
        # This public classifier only needs these two scalar values; preserve
        # MFC's own provenance instead of constructing a historical artifact.
        threshold_values = SimpleNamespace(negative=thresholds["p80"], positive=thresholds["p95"])
        for index in frames:
            check_time()
            before = time.monotonic()
            frame = source.read_frame(index)
            read_seconds = time.monotonic() - before
            before = time.monotonic()
            details = standard_ivd.compute_standard_ivd_details(
                frame.u, frame.v, frame.xdim, frame.ydim,
                solid_mask=frame.geometry, valid_fluid_mask=frame.valid_fluid_support,
                input_identity=f"MFC full-frame reference {index}: {source.flow_path}")
            ivd_seconds = time.monotonic() - before
            before = time.monotonic()
            result = labels.classify_p95p80_frame(
                details.ivd, solid_mask=details.solid_mask,
                valid_fluid_mask=details.valid_fluid_mask, invalid_mask=details.invalid_mask,
                thresholds=threshold_values,
                min_area=labels.DEFAULT_MIN_AREA_CELLS, connectivity=labels.DEFAULT_CONNECTIVITY)
            classification_seconds = time.monotonic() - before
            before = time.monotonic()
            maps["standard_ivd"][index] = details.ivd
            maps["state"][index] = result.state
            maps["valid_fluid_mask"][index] = result.valid_fluid_mask
            maps["invalid_mask"][index] = result.invalid_mask
            for array in maps.values():
                array.flush()
            write_seconds = time.monotonic() - before
            split = "train" if source.frame_ranges["train"][0] <= index < source.frame_ranges["train"][1] else "calibration"
            metadata["prepared_frame_indices"].append(index)
            metadata["actual_prepared_frame_count"] += 1
            if split == "calibration":
                metadata["calibration_flow_read"] = True
                metadata["calibration_frame_indices_read"].append(index)
            metadata["frames"].append({"frame_index": index, "split": split,
                "physical_time": float(frame.tdim[0]), "solver_step": int(frame.solver_step[0]),
                "omega_bar": float(details.omega_bar), "counts": dict(result.counts),
                "positive_component_filter": dict(result.filter_audit),
                "read_seconds": read_seconds, "ivd_seconds": ivd_seconds,
                "classification_seconds": classification_seconds, "write_seconds": write_seconds})
            for key, seconds in (("read_seconds", read_seconds), ("ivd_seconds", ivd_seconds),
                                 ("classification_seconds", classification_seconds), ("write_seconds", write_seconds)):
                metadata["timing"][key] += seconds
            del frame, details, result
            publish()
            print(json.dumps({"reference_completed_global_frame": index, "split": split,
                              "actual_prepared_frame_count": metadata["actual_prepared_frame_count"],
                              "elapsed_seconds": metadata["elapsed_seconds"]}), flush=True)
        metadata.update(status="complete", complete=sorted(metadata["prepared_frame_indices"]) == all_frames)
    except ReferenceTimeLimit as exc:
        metadata.update(status="time_limit", interruption=str(exc))
    except Exception as exc:
        metadata.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        publish()
        raise
    finally:
        for array in maps.values():
            array.flush()
        maps.clear()
    publish()
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "partial"), required=True)
    parser.add_argument("--train-range", default="0,400")
    parser.add_argument("--calibration-range", default="400,501")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--frame-indices", help="comma-separated absolute saved-frame indices")
    selection.add_argument("--frame-range", help="half-open start,stop for a partial preparation")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    source = MFCPathlineSource(args.flow, mask_path=args.mask, frame_ranges={
        "train": tuple(map(int, args.train_range.split(","))),
        "calibration": tuple(map(int, args.calibration_range.split(",")))})
    frames = None
    if args.frame_indices is not None:
        frames = tuple(map(int, args.frame_indices.split(",")))
    elif args.frame_range is not None:
        lower, upper = map(int, args.frame_range.split(","))
        frames = tuple(range(lower, upper))
    metadata = prepare_reference(source, args.statistics, args.output, mode=args.mode,
                                 frame_indices=frames, max_seconds=args.max_seconds)
    print(json.dumps({"status": metadata["status"], "complete": metadata["complete"],
                      "actual_prepared_frame_count": metadata["actual_prepared_frame_count"],
                      "metadata": str(Path(args.output) / "cache_metadata.json")}), flush=True)
    return 0 if metadata["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
