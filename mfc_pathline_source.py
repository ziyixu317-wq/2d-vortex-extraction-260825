"""Read native MFC fields for a separately configured pathline experiment.

This source exposes physical velocities, coordinates, and derivative support.
It does not construct IVD/P95-P80 labels, fit normalization, or register itself
as one of the historical six datasets. ``iter_train_frames`` is the streaming
seam for the shared standard-IVD implementation and train-only statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import h5py
import numpy as np


@dataclass
class MFCFlowSlice:
    """One physical frame (YX) or a consecutive/strided region window (TYX)."""

    frame_indices: np.ndarray
    u: np.ndarray
    v: np.ndarray
    derivative_valid: np.ndarray
    geometry: np.ndarray
    xdim: np.ndarray
    ydim: np.ndarray
    tdim: np.ndarray
    solver_step: np.ndarray

    @property
    def valid_fluid_support(self) -> np.ndarray:
        return self.derivative_valid & ~self.geometry


class MFCPathlineSource:
    """Read-only HDF5 source with explicit half-open development ranges.

    Example::

        source = MFCPathlineSource(
            "/data/xuziyi/mfc-airfoil-training-20261009/training-501/flow.h5",
            frame_ranges={"train": (0, 400), "calibration": (400, 501)},
        )
        window = source.read_window("train", 0, t_win=24)

    Coordinates must be uniform because the current pathline extractor maps
    physical positions and times using the first spacing. Fields are read only
    for the requested frame or continuous window; no field stays open between
    calls, so source instances may be passed to DataLoader worker processes.
    """

    source_name = "mfc_airfoil"

    def __init__(
        self,
        flow_path: str | Path,
        *,
        frame_ranges: Mapping[str, tuple[int, int]],
        mask_path: str | Path | None = None,
    ) -> None:
        self.flow_path = Path(flow_path)
        self.mask_path = None if mask_path is None else Path(mask_path)
        with h5py.File(self.flow_path, "r") as field:
            self.xdim = np.asarray(field["xdim"][:], dtype=np.float64)
            self.ydim = np.asarray(field["ydim"][:], dtype=np.float64)
            self.tdim = np.asarray(field["tdim"][:], dtype=np.float64)
            self.solver_step = np.asarray(field["solver_step"][:])
            self.geometry = np.asarray(field["geometry"][:], dtype=bool)
            self.shape = (len(self.tdim), len(self.ydim), len(self.xdim))
            for name in ("u", "v", "derivative_valid"):
                if field[name].shape != self.shape:
                    raise ValueError(f"MFC {name} shape must be {self.shape}")
        if self.geometry.shape != self.shape[1:]:
            raise ValueError("MFC geometry must be a static YX mask")
        if self.solver_step.shape != self.tdim.shape:
            raise ValueError("MFC solver_step must correspond to each saved time")
        if self.mask_path is not None:
            supplied_mask = np.asarray(np.load(self.mask_path, allow_pickle=False), dtype=bool)
            if not np.array_equal(supplied_mask, self.geometry):
                raise ValueError("MFC mask.npy differs from HDF5 geometry")
        for name in ("xdim", "ydim", "tdim"):
            coordinates = getattr(self, name)
            spacing = np.diff(coordinates)
            if (len(spacing) == 0 or not np.isfinite(coordinates).all()
                    or not np.all(spacing > 0)
                    or not np.allclose(spacing, spacing[0], rtol=1e-9, atol=1e-12)):
                raise ValueError(f"MFC {name} must increase uniformly for the pathline extractor")
        if set(frame_ranges) != {"train", "calibration"}:
            raise ValueError("Provide explicit train and calibration frame ranges")
        self.frame_ranges = {}
        for name, bounds in frame_ranges.items():
            start, end = bounds
            if (not isinstance(start, (int, np.integer))
                    or not isinstance(end, (int, np.integer))
                    or not 0 <= start < end <= self.shape[0]):
                raise ValueError(f"Invalid half-open {name} frame range: {bounds}")
            self.frame_ranges[name] = (int(start), int(end))
        train_start, train_end = self.frame_ranges["train"]
        cal_start, cal_end = self.frame_ranges["calibration"]
        if max(train_start, cal_start) < min(train_end, cal_end):
            raise ValueError("MFC train and calibration ranges overlap")

    @property
    def metadata(self) -> dict:
        """Source description; this is not historical frac/label metadata."""
        return {
            "source_name": self.source_name,
            "flow_path": str(self.flow_path),
            "mask_path": None if self.mask_path is None else str(self.mask_path),
            "shape_tyx": list(self.shape),
            "frame_ranges": {name: list(bounds) for name, bounds in self.frame_ranges.items()},
            "range_convention": "half-open frame indices",
            "saved_dt": float(self.tdim[1] - self.tdim[0]),
            "physical_fields_rescaled": False,
            "derivative_support": "native derivative_valid intersect non-solid",
            "normalization_source": "train",
            "labels_constructed": False,
        }

    def window_starts(self, split: str, *, t_win: int = 24, step: int = 4) -> tuple[int, ...]:
        """Enumerate windows entirely inside the requested development range."""
        if not isinstance(t_win, (int, np.integer)) or t_win <= 0:
            raise ValueError("t_win must be a positive integer")
        if not isinstance(step, (int, np.integer)) or step <= 0:
            raise ValueError("step must be a positive integer")
        start, end = self.frame_ranges[split]
        return tuple(range(start, end - int(t_win) + 1, int(step)))

    def _read(
        self,
        selection: int | slice,
        indices: np.ndarray,
        *,
        region: tuple[slice, slice] | None = None,
    ) -> MFCFlowSlice:
        field_selection = selection if region is None else (selection, *region)
        with h5py.File(self.flow_path, "r") as field:
            u = np.asarray(field["u"][field_selection])
            v = np.asarray(field["v"][field_selection])
            derivative_valid = np.asarray(field["derivative_valid"][field_selection], dtype=bool)
        y_slice, x_slice = (slice(None), slice(None)) if region is None else region
        return MFCFlowSlice(
            frame_indices=indices,
            u=u,
            v=v,
            derivative_valid=derivative_valid,
            geometry=self.geometry[y_slice, x_slice],
            xdim=self.xdim[x_slice],
            ydim=self.ydim[y_slice],
            tdim=self.tdim[indices],
            solver_step=self.solver_step[indices],
        )

    def read_frame(self, frame_index: int) -> MFCFlowSlice:
        if not isinstance(frame_index, (int, np.integer)) or not 0 <= frame_index < self.shape[0]:
            raise ValueError("frame_index must identify an existing saved frame")
        return self._read(int(frame_index), np.array([frame_index], dtype=np.int64))

    def read_window(self, split: str, frame_start: int, *, t_win: int = 24) -> MFCFlowSlice:
        start, end = self.frame_ranges[split]
        if (not isinstance(frame_start, (int, np.integer))
                or not isinstance(t_win, (int, np.integer)) or t_win <= 0
                or not start <= frame_start < frame_start + t_win <= end):
            raise ValueError(f"Window must lie entirely inside {split} [{start}, {end})")
        frame_stop = int(frame_start + t_win)
        return self._read(
            slice(int(frame_start), frame_stop),
            np.arange(frame_start, frame_stop, dtype=np.int64),
        )

    def read_region_window(
        self,
        split: str,
        frame_start: int,
        frame_stop: int,
        *,
        y_bounds: tuple[int, int],
        x_bounds: tuple[int, int],
        frame_step: int = 1,
    ) -> MFCFlowSlice:
        """Read a TYX region directly using global half-open index bounds.

        ``frame_step=1`` keeps every saved frame; 2 selects alternate frames
        with their actual times and solver steps. Spatial axes retain the
        selected native cell-center coordinates, without rebasing or scaling.
        Geometry and native derivative support are cropped to the same region;
        ROI edges do not redefine the original derivative stencil support.

        This method reads raw u/v and support. For standard IVD, compute each
        full frame's mean with the shared implementation before cropping IVD
        to this region; an ROI vorticity mean is a different quantity.
        """
        start, end = self.frame_ranges[split]
        if (not isinstance(frame_start, (int, np.integer))
                or not isinstance(frame_stop, (int, np.integer))
                or not start <= frame_start < frame_stop <= end):
            raise ValueError(f"Window must lie entirely inside {split} [{start}, {end})")
        if not isinstance(frame_step, (int, np.integer)) or frame_step not in (1, 2):
            raise ValueError("frame_step must be 1 or 2 saved frames")
        spatial_slices = []
        for name, bounds, size in (
            ("y_bounds", y_bounds, self.shape[1]),
            ("x_bounds", x_bounds, self.shape[2]),
        ):
            lower, upper = bounds
            if (not isinstance(lower, (int, np.integer))
                    or not isinstance(upper, (int, np.integer))
                    or not 0 <= lower < upper <= size):
                raise ValueError(f"{name} must be half-open indices inside [0, {size})")
            spatial_slices.append(slice(int(lower), int(upper)))
        return self._read(
            slice(int(frame_start), int(frame_stop), int(frame_step)),
            np.arange(frame_start, frame_stop, frame_step, dtype=np.int64),
            region=tuple(spatial_slices),
        )

    def iter_train_frames(self) -> Iterator[MFCFlowSlice]:
        """Stream training frames for IVD construction and normalization fitting.

        A caller can pass each frame's u/v, axes, geometry, and
        valid_fluid_support to standard_ivd.compute_standard_ivd_details, then
        accumulate train statistics without storing the complete time field.
        """
        start, end = self.frame_ranges["train"]
        for frame_index in range(start, end):
            yield self.read_frame(frame_index)
