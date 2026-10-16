"""Canonical Haller standard-IVD calculation shared by all consumers.

The public seams in this module deliberately separate the numerical source
from thresholding, connected-component filtering, and Haller contour logic.
For every frame the source computes

    IVD(x, t) = abs(omega(x, t) - omega_bar(t))

where ``omega_bar`` is the signed vorticity mean over the declared
valid-fluid support only.  Solid and invalid cells are represented by masks;
their stored numeric values are zero only so finite artifact arrays cannot be
mistaken for evidence without consulting those masks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

import experiment_protocol as protocol


IMPLEMENTATION_VERSION = "standard-ivd-v1"
FORMULA_VERSION = protocol.P95P80_IVD_DEFINITION_ID
FORMULA = protocol.P95P80_IVD_FORMULA
SOURCE_ID = protocol.P95P80_HALLER_SOURCE_ID
PROTOCOL_VERSION = protocol.P95P80_PROTOCOL_VERSION


def _canonical_source_identity() -> dict[str, Any]:
    """Return the immutable Ticket 01 identity for the shared IVD source."""
    artifact_identity = protocol.build_p95p80_artifact_identity(SOURCE_ID)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source_id": SOURCE_ID,
        "definition_id": FORMULA_VERSION,
        "formula": FORMULA,
        "artifact_identity": artifact_identity,
        "legacy_artifacts_formal_source": False,
    }


SOURCE_IDENTITY = _canonical_source_identity()


def _array_hash(array: Any) -> str:
    """Hash dtype, shape, and C-order bytes without changing the input."""
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _hash_named_arrays(named_arrays: Mapping[str, Any]) -> str:
    """Build a stable identity for the exact field/coordinate inputs."""
    digest = hashlib.sha256()
    for name in sorted(named_arrays):
        digest.update(str(name).encode("utf-8"))
        digest.update(_array_hash(named_arrays[name]).encode("ascii"))
    return digest.hexdigest()


def _validate_coordinates(
    xdim: Any,
    ydim: Any,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Validate uniform physical axes and return their actual spacing."""
    x = np.asarray(xdim, dtype=np.float64)
    y = np.asarray(ydim, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("xdim/ydim 必须是一维坐标轴")
    if len(shape) not in (2, 3) or len(x) != shape[-1] or len(y) != shape[-2]:
        raise ValueError(
            "速度/涡量 shape 必须是 (Y,X) 或 (T,Y,X)，且坐标长度匹配；"
            f"shape={shape}, len(ydim)={len(y)}, len(xdim)={len(x)}"
        )
    if len(x) < 2 or len(y) < 2:
        raise ValueError("xdim/ydim 每个方向至少需要两个坐标")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("xdim/ydim 必须全部有限")
    dxs = np.diff(x)
    dys = np.diff(y)
    if np.any(dxs <= 0.0) or np.any(dys <= 0.0):
        raise ValueError("xdim/ydim 必须严格递增")
    if not np.allclose(dxs, dxs[0], rtol=1e-3, atol=1e-12):
        raise ValueError("xdim 坐标非等距，standard IVD 不支持")
    if not np.allclose(dys, dys[0], rtol=1e-3, atol=1e-12):
        raise ValueError("ydim 坐标非等距，standard IVD 不支持")
    return x, y, float(dxs[0]), float(dys[0])


def _difference(field: Any, axis: int, spacing: float) -> np.ndarray:
    """Second-order centered interior and first-order one-sided boundary."""
    value = np.asarray(field, dtype=np.float64)
    if value.ndim not in (2, 3):
        raise ValueError(f"field 必须是二维或三维，实际 shape={value.shape}")
    axis = int(axis)
    length = value.shape[axis]
    if length < 2:
        raise ValueError("差分轴至少需要两个网格点")
    out = np.full(value.shape, np.nan, dtype=np.float64)
    interior = [slice(None)] * value.ndim
    interior[axis] = slice(1, -1)
    out[tuple(interior)] = (
        np.take(value, np.arange(2, length), axis=axis)
        - np.take(value, np.arange(0, length - 2), axis=axis)
    ) / (2.0 * float(spacing))
    lower = [slice(None)] * value.ndim
    lower[axis] = 0
    upper = [slice(None)] * value.ndim
    upper[axis] = -1
    out[tuple(lower)] = (
        np.take(value, 1, axis=axis) - np.take(value, 0, axis=axis)
    ) / float(spacing)
    out[tuple(upper)] = (
        np.take(value, -1, axis=axis) - np.take(value, -2, axis=axis)
    ) / float(spacing)
    return out


def vorticity(u: Any, v: Any, xdim: Any, ydim: Any) -> np.ndarray:
    """Compute ``omega = d(v)/d(x) - d(u)/d(y)`` on a rectilinear grid."""
    u_value = np.asarray(u, dtype=np.float64)
    v_value = np.asarray(v, dtype=np.float64)
    if u_value.shape != v_value.shape or u_value.ndim not in (2, 3):
        raise ValueError(
            "u/v 必须是同形状的二维单帧或三维时间场，"
            f"实际 {u_value.shape}/{v_value.shape}"
        )
    _, _, dx, dy = _validate_coordinates(
        xdim, ydim, tuple(int(size) for size in u_value.shape)
    )
    return _difference(v_value, -1, dx) - _difference(u_value, -2, dy)


def _coerce_mask(
    value: Any,
    shape: tuple[int, ...],
    *,
    name: str,
    default: bool,
) -> np.ndarray:
    """Coerce a 2-D mask or a matching 3-D per-frame mask."""
    if value is None:
        return np.full(shape, default, dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if len(shape) == 3 and mask.ndim == 2 and mask.shape == shape[-2:]:
        mask = np.broadcast_to(mask, shape).copy()
    if mask.shape != shape:
        raise ValueError(f"{name} shape={mask.shape} 与 field shape={shape} 不一致")
    return mask.copy()


def _normalise_input_identity(input_identity: Any) -> str | None:
    if input_identity is None:
        return None
    if not isinstance(input_identity, str) or not input_identity.strip():
        raise ValueError("input_identity 必须是非空字符串或 None")
    return input_identity.strip()


@dataclass(frozen=True)
class StandardIVDDetails:
    """Numerical result and masks emitted by the shared standard-IVD source."""

    omega: np.ndarray
    ivd: np.ndarray
    omega_bar: float | np.ndarray
    solid_mask: np.ndarray
    valid_fluid_mask: np.ndarray
    invalid_mask: np.ndarray
    dx: float | None
    dy: float | None
    metadata: Mapping[str, Any]

    @property
    def fluid_mask(self) -> np.ndarray:
        """Return geometry fluid support, including cells later marked invalid."""
        return ~np.asarray(self.solid_mask, dtype=bool)

    @property
    def fluid_mean(self) -> float:
        """Backward-compatible name for the signed ``omega_bar`` value."""
        value = np.asarray(self.omega_bar)
        if value.ndim != 0:
            raise ValueError(
                "fluid_mean 只适用于单帧结果；多帧结果请读取 omega_bar 的逐帧数组"
            )
        return float(value)


def _details_from_omega(
    omega: Any,
    *,
    solid_mask: Any = None,
    valid_fluid_mask: Any = None,
    invalid_mask: Any = None,
    dx: float | None = None,
    dy: float | None = None,
    input_identity: Any = None,
    entry_point: str,
) -> StandardIVDDetails:
    """Apply the global signed-mean definition to a vorticity field."""
    omega_raw = np.asarray(omega, dtype=np.float64)
    if omega_raw.ndim not in (2, 3):
        raise ValueError(
            f"omega 必须是二维单帧或三维时间场，实际 shape={omega_raw.shape}"
        )
    shape = tuple(int(size) for size in omega_raw.shape)
    solid = _coerce_mask(
        solid_mask, shape, name="solid_mask", default=False
    )
    declared_valid = _coerce_mask(
        valid_fluid_mask, shape, name="valid_fluid_mask", default=True
    )
    explicit_invalid = _coerce_mask(
        invalid_mask, shape, name="invalid_mask", default=False
    )
    fluid = ~solid
    finite = np.isfinite(omega_raw)
    # Cells outside the declared support are represented as invalid for the
    # output contract, while solid remains a separate geometry state.
    invalid = fluid & (~declared_valid | explicit_invalid | ~finite)
    valid = fluid & declared_valid & ~explicit_invalid & finite
    if not np.any(valid):
        raise ValueError(
            "standard IVD 没有可用 valid-fluid support，无法计算 omega_bar"
        )
    safe_omega = np.zeros(shape, dtype=np.float64)
    safe_ivd = np.zeros(shape, dtype=np.float64)
    if omega_raw.ndim == 2:
        frame_valid = valid
        omega_bars = np.asarray(
            [np.mean(omega_raw[frame_valid], dtype=np.float64)], dtype=np.float64
        )
        safe_omega[frame_valid] = omega_raw[frame_valid]
        safe_ivd[frame_valid] = np.abs(omega_raw[frame_valid] - omega_bars[0])
    else:
        omega_bars_list: list[float] = []
        for frame_index in range(omega_raw.shape[0]):
            frame_valid = valid[frame_index]
            if not np.any(frame_valid):
                raise ValueError(
                    "standard IVD 某一时间帧没有可用 valid-fluid support，"
                    f"frame={frame_index}"
                )
            omega_bars_list.append(float(np.mean(
                omega_raw[frame_index][frame_valid], dtype=np.float64
            )))
            safe_omega[frame_index][frame_valid] = omega_raw[frame_index][frame_valid]
            safe_ivd[frame_index][frame_valid] = np.abs(
                omega_raw[frame_index][frame_valid] - omega_bars_list[-1]
            )
        omega_bars = np.asarray(omega_bars_list, dtype=np.float64)
    omega_bar: float | np.ndarray = (
        float(omega_bars[0]) if omega_raw.ndim == 2 else omega_bars
    )

    identity = _normalise_input_identity(input_identity)
    if identity is None:
        identity = _hash_named_arrays({
            "omega": omega_raw,
            "solid_mask": solid,
            "valid_fluid_mask": declared_valid,
            "invalid_mask": explicit_invalid,
        })
    spacing = {
        "dx": None if dx is None else float(dx),
        "dy": None if dy is None else float(dy),
    }
    source_identity = _canonical_source_identity()
    metadata: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "source_id": SOURCE_ID,
        "source_identity": source_identity,
        "artifact_identity": source_identity["artifact_identity"],
        "definition_id": FORMULA_VERSION,
        "formula_version": FORMULA_VERSION,
        "formula": FORMULA,
        "ivd_formula": FORMULA,
        "mean_scope": "declared_valid_fluid_support",
        "mean_quantity": "signed omega",
        "omega_bar": (
            omega_bar if isinstance(omega_bar, float) else omega_bar.tolist()
        ),
        "omega_bar_by_frame": omega_bars.tolist(),
        "fluid_vorticity_mean": (
            omega_bar if isinstance(omega_bar, float) else omega_bar.tolist()
        ),
        "spacing": spacing,
        "grid_spacing": spacing,
        "shape": list(shape),
        "input_identity": identity,
        "input_field_identity": identity,
        "valid_fluid_mask_semantics": (
            "declared_valid_fluid_support intersect finite_omega and non_solid"
        ),
        "solid_mask_semantics": (
            "solid cells are excluded from omega_bar and label support; "
            "solid is not a valid negative"
        ),
        "invalid_mask_semantics": (
            "non-finite omega, explicit invalid, or outside declared valid-fluid "
            "support; excluded from omega_bar and label support"
        ),
        "storage_semantics": {
            "solid": "zero_for_artifact_storage_only",
            "invalid": "zero_for_artifact_storage_only",
            "consumer_must_use_masks": True,
        },
        "counts": {
            "solid_cells": int(np.count_nonzero(solid)),
            "fluid_cells": int(np.count_nonzero(fluid)),
            "valid_fluid_cells": int(np.count_nonzero(valid)),
            "invalid_cells": int(np.count_nonzero(invalid)),
        },
        "counts_by_frame": [
            {
                "solid_cells": int(np.count_nonzero(solid[index])),
                "fluid_cells": int(np.count_nonzero(fluid[index])),
                "valid_fluid_cells": int(np.count_nonzero(valid[index])),
                "invalid_cells": int(np.count_nonzero(invalid[index])),
            }
            for index in range(omega_raw.shape[0])
        ] if omega_raw.ndim == 3 else [{
            "solid_cells": int(np.count_nonzero(solid)),
            "fluid_cells": int(np.count_nonzero(fluid)),
            "valid_fluid_cells": int(np.count_nonzero(valid)),
            "invalid_cells": int(np.count_nonzero(invalid)),
        }],
        "provenance": {
            "module": "standard_ivd",
            "entry_point": entry_point,
            "implementation_version": IMPLEMENTATION_VERSION,
            "difference_scheme": {
                "vorticity": "d(v)/d(x) - d(u)/d(y)",
                "interior": "second_order_centered",
                "boundary": "first_order_one_sided",
            },
            "source_identity": source_identity,
        },
    }
    return StandardIVDDetails(
        omega=safe_omega,
        ivd=safe_ivd,
        omega_bar=omega_bar,
        solid_mask=solid,
        valid_fluid_mask=valid,
        invalid_mask=invalid,
        dx=None if dx is None else float(dx),
        dy=None if dy is None else float(dy),
        metadata=metadata,
    )


def compute_standard_ivd_from_vorticity(
    omega: Any,
    *,
    solid_mask: Any = None,
    valid_fluid_mask: Any = None,
    invalid_mask: Any = None,
    xdim: Any = None,
    ydim: Any = None,
    input_identity: Any = None,
) -> StandardIVDDetails:
    """Compute standard IVD when the vorticity field is already available."""
    omega_value = np.asarray(omega, dtype=np.float64)
    dx = dy = None
    if xdim is not None or ydim is not None:
        if xdim is None or ydim is None:
            raise ValueError("xdim 与 ydim 必须同时提供")
        _, _, dx, dy = _validate_coordinates(
            xdim, ydim, tuple(int(size) for size in omega_value.shape)
        )
    return _details_from_omega(
        omega_value,
        solid_mask=solid_mask,
        valid_fluid_mask=valid_fluid_mask,
        invalid_mask=invalid_mask,
        dx=dx,
        dy=dy,
        input_identity=input_identity,
        entry_point="standard_ivd.compute_standard_ivd_from_vorticity",
    )


def compute_standard_ivd_details(
    u: Any,
    v: Any,
    xdim: Any,
    ydim: Any,
    *,
    solid_mask: Any = None,
    valid_fluid_mask: Any = None,
    invalid_mask: Any = None,
    input_identity: Any = None,
) -> StandardIVDDetails:
    """Compute one or more frames from velocity fields using the shared source."""
    u_value = np.asarray(u, dtype=np.float64)
    v_value = np.asarray(v, dtype=np.float64)
    if u_value.shape != v_value.shape or u_value.ndim not in (2, 3):
        raise ValueError(
            "u/v 必须是同形状的二维单帧或三维时间场，"
            f"实际 {u_value.shape}/{v_value.shape}"
        )
    x, y, dx, dy = _validate_coordinates(
        xdim, ydim, tuple(int(size) for size in u_value.shape)
    )
    identity = _normalise_input_identity(input_identity)
    if identity is None:
        identity = _hash_named_arrays({
            "u": u_value,
            "v": v_value,
            "xdim": x,
            "ydim": y,
        })
    omega = vorticity(u_value, v_value, x, y)
    # A centered stencil can omit the value at its own cell.  Preserve the
    # input-field invalid state explicitly so a NaN/Inf velocity sample is
    # never admitted merely because its derived omega happens to be finite.
    input_nonfinite = ~np.isfinite(u_value) | ~np.isfinite(v_value)
    if invalid_mask is None:
        effective_invalid = input_nonfinite
    else:
        effective_invalid = (
            _coerce_mask(
                invalid_mask,
                tuple(int(size) for size in u_value.shape),
                name="invalid_mask",
                default=False,
            )
            | input_nonfinite
        )
    return _details_from_omega(
        omega,
        solid_mask=solid_mask,
        valid_fluid_mask=valid_fluid_mask,
        invalid_mask=effective_invalid,
        dx=dx,
        dy=dy,
        input_identity=identity,
        entry_point="standard_ivd.compute_standard_ivd_details",
    )


def compute_standard_ivd(
    u: Any,
    v: Any,
    xdim: Any,
    ydim: Any,
    solid_mask: Any = None,
    *,
    valid_fluid_mask: Any = None,
    invalid_mask: Any = None,
    input_identity: Any = None,
    return_details: bool = False,
) -> np.ndarray | StandardIVDDetails:
    """Return the standard IVD array, or details when explicitly requested."""
    details = compute_standard_ivd_details(
        u,
        v,
        xdim,
        ydim,
        solid_mask=solid_mask,
        valid_fluid_mask=valid_fluid_mask,
        invalid_mask=invalid_mask,
        input_identity=input_identity,
    )
    return details if return_details else details.ivd


standard_ivd = compute_standard_ivd


__all__ = [
    "IMPLEMENTATION_VERSION",
    "FORMULA_VERSION",
    "FORMULA",
    "SOURCE_ID",
    "PROTOCOL_VERSION",
    "SOURCE_IDENTITY",
    "StandardIVDDetails",
    "vorticity",
    "compute_standard_ivd_from_vorticity",
    "compute_standard_ivd_details",
    "compute_standard_ivd",
    "standard_ivd",
]
