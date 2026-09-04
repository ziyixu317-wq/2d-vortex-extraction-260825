"""Haller-IVD 单帧 physics-anchor 提取器（02 票）。

本模块只负责单帧 ``u/v + geometry mask`` 的物理候选和三态 artifact：

``速度场 → 涡量 → standard IVD → fluid 局部峰 → 闭合等值线 →
几何筛选 → positive/unknown/negative``。

Haller 原始文献在当前工程中只有 Zotero 候选条目 ``L2PX3NQX``，全文和
算法细节尚未核实。因此本文件中的参数是已确认的**工程参数**，而不是
canonical paper 参数；每个 artifact 都会记录这一证据状态、完整参数、
输入 hash 和失败计数。

三态编码固定为：

* ``POSITIVE = 1``：闭合轮廓内部且远离 unknown band；
* ``NEGATIVE = 0``：闭合轮廓/边界带外且 IVD 不高于本帧 fluid p60；
* ``UNKNOWN = -1``：边界带、未决区域、solid 或失败帧。

``haller_gt.npy`` 与 ``anchor_state.npy`` 都保留上述三态编码；前者是
下游读取时的明确 GT 名称，后者是训练 anchor 名称。test GT 的读取必须
通过 ``load_haller_artifact(..., expected_source=SOURCE_TEST)`` 显式声明，
防止默认 loader 把 test 标签带入训练。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

import weak_labels


# --------------------------------------------------------------------------- 公共编码和冻结工程参数

ALGORITHM_VERSION = "haller-anchor-v1.0"

SOURCE_TRAIN = "haller_anchor_train"
SOURCE_CALIBRATION = "haller_gt_calibration"
SOURCE_TEST = "haller_gt_test"
VALID_SOURCES = frozenset({SOURCE_TRAIN, SOURCE_CALIBRATION, SOURCE_TEST})

POSITIVE = np.int8(1)
NEGATIVE = np.int8(0)
UNKNOWN = np.int8(-1)

DEFAULT_CONTOUR_LEVEL_COUNT = 32
DEFAULT_CONTOUR_LEVEL_START = 1.0
DEFAULT_CONTOUR_LEVEL_END = 0.1
DEFAULT_CONVEXITY_DEFECT_MAX = 0.10
DEFAULT_MINIMUM_PERIMETER_FACTOR = 8.0
DEFAULT_UNKNOWN_BAND_FACTOR = 2.0
DEFAULT_NEGATIVE_PERCENTILE = 60.0
DEFAULT_CLOSURE_TOLERANCE_FACTOR = 1.5
# This is only the initial search-window size.  It is not a Haller criterion,
# threshold, prominence, peak filter, or contour rejection rule.  The window
# is expanded until neither the high-side support nor a returned contour
# touches its boundary; otherwise the exact full-domain call is used.
_ROI_INITIAL_RADIUS_CELLS = 8

BACKEND_NUMPY = "numpy"
BACKEND_FAST_HALLER = "fast_haller"
BACKEND_NUMBACS = "numbacs"
# Ticket 09 deliberately exposes only the CPU Haller implementation.  A CUDA
# implementation belongs to a separate ticket and must not be reachable from
# this artifact seam by an implicit fallback or a public backend selector.
VALID_BACKENDS = frozenset({BACKEND_NUMPY, BACKEND_FAST_HALLER, BACKEND_NUMBACS})

CONTOUR_MODE_OPTIMIZED = "optimized"
CONTOUR_MODE_REFERENCE = "reference"
CONTOUR_MODE_NUMBACS = "numbacs"
VALID_CONTOUR_MODES = frozenset({
    CONTOUR_MODE_OPTIMIZED,
    CONTOUR_MODE_REFERENCE,
    CONTOUR_MODE_NUMBACS,
})

_LITERATURE_METADATA = {
    "status": "pending_verification",
    "zotero_key": "L2PX3NQX",
    "note": "Zotero metadata matches the Haller 2016 candidate; full text and algorithm details remain unverified.",
}


def _normalize_contour_mode(contour_mode: str) -> str:
    """Normalize and validate the contour implementation before extraction."""
    normalized = str(contour_mode).lower()
    if normalized not in VALID_CONTOUR_MODES:
        allowed = ", ".join(sorted(VALID_CONTOUR_MODES))
        raise ValueError(
            f"未知 contour_mode={contour_mode!r}；必须是 {allowed}"
        )
    return normalized


def _contour_mode_for_backend(backend: str, contour_mode: str) -> str:
    """Resolve the explicit contour implementation for a backend."""
    normalized_backend = str(backend).lower()
    normalized_mode = _normalize_contour_mode(contour_mode)
    if normalized_backend == BACKEND_NUMBACS:
        if normalized_mode == CONTOUR_MODE_REFERENCE:
            raise ValueError(
                "numbacs backend 使用 upstream rotcohvrt；严格 oracle 请使用 "
                "backend='numpy', contour_mode='reference'"
            )
        return CONTOUR_MODE_NUMBACS
    if normalized_mode == CONTOUR_MODE_NUMBACS:
        raise ValueError(
            "contour_mode='numbacs' 必须与 backend='numbacs' 一起使用"
        )
    return normalized_mode


@lru_cache(maxsize=1)
def _haller_geometry_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Load contour-only dependencies when extraction, rather than loading, is used."""
    try:
        from scipy import ndimage
        from scipy.spatial import ConvexHull, QhullError
        from skimage.measure import find_contours, grid_points_in_poly, points_in_poly
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Haller contour extraction requires scipy and scikit-image; "
            "loading a precomputed Haller artifact does not"
        ) from exc
    return (
        ndimage,
        ConvexHull,
        QhullError,
        find_contours,
        grid_points_in_poly,
        points_in_poly,
    )


def grid_points_in_poly(shape: tuple[int, int], points: Any, *args: Any, **kwargs: Any) -> Any:
    """Backward-compatible lazy wrapper for the contour rasterizer helper."""
    dependencies = _haller_geometry_dependencies()
    return dependencies[4](shape, points, *args, **kwargs)


@dataclass(frozen=True)
class _StandardIVDDetails:
    """standard IVD 的内部计算结果，避免重复计算涡量和 fluid mean。"""

    omega: np.ndarray
    ivd: np.ndarray
    fluid_mean: float
    solid_mask: np.ndarray
    dx: float
    dy: float


@dataclass(frozen=True)
class _BackendDetails:
    """Haller 数值后端的请求、解析结果和运行时审计字段。"""

    requested: str
    resolved: str
    device: str
    cuda_used: bool
    fallback_reason: str | None
    backend_version: str
    compute_dtype: str


def _resolve_backend(backend: str, device: Any = None) -> _BackendDetails:
    """Resolve the CPU-only numerical backend and reject future backends."""
    requested = str(backend).lower()
    if requested not in VALID_BACKENDS:
        allowed = ", ".join(sorted(VALID_BACKENDS))
        raise ValueError(
            "当前 ticket 09 只实现 CPU NumPy、CPU fast_haller 和 CPU numbacs Haller；"
            f"未知或未实现 backend={backend!r}，允许值为 {allowed}"
        )

    device_text = None if device is None else str(device)
    if device_text is not None and device_text.lower() != "cpu":
        raise ValueError(
            f"Haller CPU backend 只接受 cpu device，实际收到 {device_text}"
        )
    resolved = requested
    return _BackendDetails(
        requested=requested,
        resolved=resolved,
        device="cpu",
        cuda_used=False,
        fallback_reason=None,
        backend_version=(
            "numpy-v1"
            if resolved == BACKEND_NUMPY
            else (
                "fast_haller-contourpy-v1"
                if resolved == BACKEND_FAST_HALLER
                else "numbacs-0.2.0+c067f542543f"
            )
        ),
        compute_dtype="float64",
    )


def resolve_haller_backend(backend: str = BACKEND_NUMPY, device: Any = None) -> dict[str, Any]:
    """Return JSON-friendly runtime metadata for a requested Haller backend."""
    details = _resolve_backend(backend, device)
    return {
        "backend_requested": details.requested,
        "resolved": details.resolved,
        "backend": details.resolved,
        "device": details.device,
        "cuda_used": details.cuda_used,
        "backend_fallback_reason": details.fallback_reason,
        "backend_version": details.backend_version,
        "compute_dtype": details.compute_dtype,
    }


# --------------------------------------------------------------------------- 输入、hash 和参数

def _validate_coords(xdim: Any, ydim: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """校验等距、递增的物理网格坐标并返回格距。"""
    x = np.asarray(xdim, dtype=np.float64)
    y = np.asarray(ydim, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("xdim/ydim 必须是一维坐标轴")
    if len(x) != shape[1] or len(y) != shape[0]:
        raise ValueError(
            f"坐标长度与单帧形状不符：shape={shape}, len(ydim)={len(y)}, len(xdim)={len(x)}"
        )
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Haller 单帧网格每个方向至少需要 2 个坐标")
    dxs = np.diff(x)
    dys = np.diff(y)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("xdim/ydim 必须有限")
    if np.any(dxs <= 0.0) or np.any(dys <= 0.0):
        raise ValueError("Haller 单帧实现要求 xdim/ydim 递增")
    if not np.allclose(dxs, dxs[0], rtol=1e-3, atol=1e-12):
        raise ValueError("xdim 坐标非等距，Haller 涡量不支持")
    if not np.allclose(dys, dys[0], rtol=1e-3, atol=1e-12):
        raise ValueError("ydim 坐标非等距，Haller 涡量不支持")
    return x, y, float(dxs[0]), float(dys[0])


def _coerce_frame(u: Any, v: Any) -> tuple[np.ndarray, np.ndarray]:
    """把输入限制为同形状、有限的二维速度单帧。"""
    u2 = np.asarray(u, dtype=np.float64)
    v2 = np.asarray(v, dtype=np.float64)
    if u2.ndim != 2 or v2.ndim != 2 or u2.shape != v2.shape:
        raise ValueError(f"u/v 必须是同形状二维单帧，实际 {u2.shape}/{v2.shape}")
    if not np.all(np.isfinite(u2)) or not np.all(np.isfinite(v2)):
        raise ValueError("u/v 必须全部有限")
    return u2, v2


def _coerce_solid_mask(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    """规格化 geometry mask；多帧 mask 只接受长度为 1 的单帧包装。"""
    if mask is None:
        return np.zeros(shape, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 3:
        if m.shape[0] != 1:
            raise ValueError(
                "单帧 Haller 输入不能静默选择多帧 geometry mask；请显式传入 mask[t]"
            )
        m = m[0]
    if m.ndim != 2 or m.shape != shape:
        raise ValueError(f"geometry mask 必须是 {shape}，实际 {m.shape}")
    return m.copy()


def _hash_array(array: Any) -> str:
    """对数组的 dtype、shape 和 C-order 内容计算稳定 SHA256。"""
    arr = np.ascontiguousarray(np.asarray(array))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _hash_named_arrays(named_arrays: Mapping[str, Any]) -> str:
    """按名字排序组合多个输入数组 hash，避免字段顺序造成歧义。"""
    h = hashlib.sha256()
    for name in sorted(named_arrays):
        h.update(name.encode("utf-8"))
        h.update(_hash_array(named_arrays[name]).encode("ascii"))
    return h.hexdigest()


def _artifact_array_hashes(
    haller_gt: Any,
    anchor_state: Any,
    anchor_confidence: Any,
    standard_ivd: Any,
    omega: Any,
    solid_mask: Any = None,
) -> dict[str, str]:
    """Hash the exact dtypes written to a Haller artifact.

    Input and mask hashes identify the source frame, while these hashes make
    the persisted arrays independently auditable after a save/load round trip.
    The casts mirror :func:`save_haller_artifact` exactly.
    """
    arrays: dict[str, np.ndarray] = {
        "haller_gt": np.asarray(haller_gt, dtype=np.int8),
        "anchor_state": np.asarray(anchor_state, dtype=np.int8),
        "anchor_confidence": np.asarray(anchor_confidence, dtype=np.float32),
        "standard_ivd": np.asarray(standard_ivd, dtype=np.float32),
        "omega": np.asarray(omega, dtype=np.float32),
    }
    if solid_mask is not None:
        arrays["solid_mask"] = np.asarray(solid_mask, dtype=np.uint8)
    return {name: _hash_array(array) for name, array in arrays.items()}


def _json_hash(value: Mapping[str, Any]) -> str:
    """对可 JSON 序列化的参数对象计算稳定 hash。"""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_source(source: str) -> str:
    if source not in VALID_SOURCES:
        allowed = ", ".join(sorted(VALID_SOURCES))
        raise ValueError(f"未知 Haller artifact source={source!r}；必须是 {allowed}")
    return source


def _resolve_parameters(
) -> dict[str, Any]:
    """返回本票冻结的完整工程参数表。

    sensitivity 变体不通过正式 artifact extractor 的参数覆盖入口实现；
    这样每个正式 source 的 Haller 语义只有一个可复现版本。
    """
    params: dict[str, Any] = {
        "domain_mean": "fluid_vorticity_mean",
        "local_maximum_neighborhood": 8,
        "contour_level_count": DEFAULT_CONTOUR_LEVEL_COUNT,
        "contour_level_start": DEFAULT_CONTOUR_LEVEL_START,
        "contour_level_end": DEFAULT_CONTOUR_LEVEL_END,
        "convexity_defect_max": DEFAULT_CONVEXITY_DEFECT_MAX,
        "minimum_perimeter_factor": DEFAULT_MINIMUM_PERIMETER_FACTOR,
        "unknown_band_factor": DEFAULT_UNKNOWN_BAND_FACTOR,
        "negative_percentile": DEFAULT_NEGATIVE_PERCENTILE,
        "closure_tolerance_factor": DEFAULT_CLOSURE_TOLERANCE_FACTOR,
        "contour_search": "all_frozen_levels_outermost_selection",
        "outermost": "maximum_physical_area_per_local_peak",
        "solid_policy": "solid_cells_are_unknown_and_never_known_anchor",
        "negative_rule": "outside_contour_and_band_and_ivd_le_frame_fluid_p60",
        "failure_fallback_train": "fluid_unknown",
        "failure_fallback_calibration_test": "invalid_frame",
    }
    if int(params["contour_level_count"]) < 2:
        raise ValueError("contour_level_count 必须至少为 2")
    if not (float(params["contour_level_start"]) > 0.0):
        raise ValueError("contour_level_start 必须为正")
    if not (0.0 < float(params["contour_level_end"]) < float(params["contour_level_start"])):
        raise ValueError("contour_level_end 必须位于 (0, contour_level_start) 内")
    if not (0.0 <= float(params["convexity_defect_max"]) <= 1.0):
        raise ValueError("convexity_defect_max 必须位于 [0,1]")
    for key in (
        "minimum_perimeter_factor",
        "unknown_band_factor",
        "closure_tolerance_factor",
    ):
        if float(params[key]) <= 0.0:
            raise ValueError(f"{key} 必须为正")
    if not (0.0 <= float(params["negative_percentile"]) <= 100.0):
        raise ValueError("negative_percentile 必须位于 [0,100]")
    params["contour_level_count"] = int(params["contour_level_count"])
    for key in (
        "contour_level_start",
        "contour_level_end",
        "convexity_defect_max",
        "minimum_perimeter_factor",
        "unknown_band_factor",
        "negative_percentile",
        "closure_tolerance_factor",
    ):
        params[key] = float(params[key])
    return params


# --------------------------------------------------------------------------- standard IVD

def _compute_standard_ivd_details(
    u: Any,
    v: Any,
    xdim: Any,
    ydim: Any,
    mask: Any = None,
    *,
    backend: str = BACKEND_NUMPY,
    device: Any = None,
) -> _StandardIVDDetails:
    """计算 standard IVD 的数组和元数据（内部共享给主入口）。"""
    u2, v2 = _coerce_frame(u, v)
    x, y, dx, dy = _validate_coords(xdim, ydim, u2.shape)
    solid = _coerce_solid_mask(mask, u2.shape)
    fluid = ~solid
    if not np.any(fluid):
        raise ValueError("geometry mask 将整帧标为 solid，无法计算 fluid vorticity mean")

    backend_details = _resolve_backend(backend, device)
    # 与既有弱标签保持同一个中心/边界差分口径；mean 只在 fluid 上统计。
    # 这是 ticket 09 唯一可用的 Haller 数值路径；CUDA backend 不在本票实现。
    omega = np.asarray(weak_labels.vorticity(u2, v2, x, y), dtype=np.float64)
    if not np.all(np.isfinite(omega)):
        raise ValueError("涡量计算结果包含非有限值")
    fluid_mean = float(omega[fluid].mean())
    ivd = np.abs(omega - fluid_mean)
    # solid 上的数值不参与任何候选、分位或 contour；显式置零避免污染 artifact。
    omega = omega.copy()
    omega[solid] = 0.0
    ivd = ivd.astype(np.float64, copy=False)
    ivd[solid] = 0.0
    return _StandardIVDDetails(omega, ivd, fluid_mean, solid, dx, dy)


def compute_standard_ivd(
    u: Any,
    v: Any,
    xdim: Any,
    ydim: Any,
    mask: Any = None,
    *,
    backend: str = BACKEND_NUMPY,
    device: Any = None,
) -> np.ndarray:
    """返回二维 standard IVD：``abs(omega - mean_fluid(omega))``。

    只接受单帧二维速度；solid 值被置零且不参加 fluid mean。函数返回数组
    以保持与现有 ``weak_labels.compute_ivd`` 的调用习惯一致；主提取入口
    还会在 metadata 中记录 fluid mean、格距和输入 hash。
    """
    return _compute_standard_ivd_details(
        u,
        v,
        xdim,
        ydim,
        mask,
        backend=backend,
        device=device,
    ).ivd


standard_ivd = compute_standard_ivd


# --------------------------------------------------------------------------- fluid 局部极大值

def find_local_maxima(
    ivd: Any,
    mask: Any = None,
    *,
    backend: str = BACKEND_NUMPY,
    device: Any = None,
) -> list[dict[str, Any]]:
    """寻找 fluid 8-neighborhood 局部极大值并压缩等值平台。

    返回按 ``(-value, row, col)`` 排序的字典列表；不增加未注册的
    prominence、最小峰间距或幅值阈值。完全平坦的场没有局部峰。
    """
    field = np.asarray(ivd, dtype=np.float64)
    if field.ndim != 2 or not np.all(np.isfinite(field)):
        raise ValueError("ivd 必须是有限的二维数组")
    solid = _coerce_solid_mask(mask, field.shape)
    fluid = ~solid
    if not np.any(fluid):
        return []

    backend_details = _resolve_backend(backend, device)
    padded_field = np.pad(field, 1, mode="constant", constant_values=-np.inf)
    padded_solid = np.pad(solid, 1, mode="constant", constant_values=True)
    is_max = fluid.copy()
    is_strict = np.zeros(field.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor = padded_field[1 + dr:1 + dr + field.shape[0],
                                    1 + dc:1 + dc + field.shape[1]]
            neighbor_solid = padded_solid[1 + dr:1 + dr + field.shape[0],
                                          1 + dc:1 + dc + field.shape[1]]
            valid_neighbor = ~neighbor_solid
            is_max &= (~valid_neighbor) | (field >= neighbor)
            is_strict |= valid_neighbor & (field > neighbor)
    candidate = is_max & is_strict & fluid & (field > 0.0)
    if not candidate.any():
        return []

    # 同一局部极大平台可能产生多个网格格；规格没有额外 peak distance，
    # 只在精确等值平台内压成一个确定代表点。
    ndimage, _, _, _, _, _ = _haller_geometry_dependencies()
    plateau, n_plateau = ndimage.label(candidate, structure=np.ones((3, 3), dtype=bool))
    peaks: list[dict[str, Any]] = []
    for label in range(1, n_plateau + 1):
        ys, xs = np.nonzero(plateau == label)
        order = np.lexsort((xs, ys, -field[ys, xs]))
        idx = int(order[0])
        peaks.append({
            "row": int(ys[idx]),
            "col": int(xs[idx]),
            "value": float(field[ys[idx], xs[idx]]),
        })
    peaks.sort(key=lambda p: (-p["value"], p["row"], p["col"]))
    return peaks


detect_local_maxima = find_local_maxima


# --------------------------------------------------------------------------- contour geometry 和筛选

def _grid_path_to_xy(path: np.ndarray, xdim: np.ndarray, ydim: np.ndarray) -> np.ndarray:
    """把 skimage 的 ``(row,col)`` contour 转为物理 ``(x,y)`` 点。"""
    path = np.asarray(path, dtype=np.float64)
    return np.column_stack((
        np.interp(path[:, 1], np.arange(len(xdim), dtype=np.float64), xdim),
        np.interp(path[:, 0], np.arange(len(ydim), dtype=np.float64), ydim),
    ))


def _closed_path(path: Any, dx: float, dy: float, tolerance_factor: float) -> tuple[bool, np.ndarray]:
    """判断 contour 是否闭合；不为开放路径人工补端点。"""
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 4 or not np.all(np.isfinite(p)):
        return False, p
    # Canonicalize only the representation of marching-squares coordinates.
    # This removes sub-ULP differences between a translated safe ROI and the
    # equivalent full-domain call; it does not alter any frozen level or
    # geometry threshold at the engineering precision recorded in metadata.
    p = np.round(p, decimals=12)
    physical_gap = float(np.hypot((p[0, 0] - p[-1, 0]) * dy,
                                  (p[0, 1] - p[-1, 1]) * dx))
    return physical_gap <= tolerance_factor * max(dx, dy), p


def _contour_crosses_solid(path: np.ndarray, solid: np.ndarray, dx: float, dy: float) -> bool:
    """以小于半格的物理步长采样 contour，保守拒绝穿过 solid 的路径。"""
    if not solid.any():
        return False
    Y, X = solid.shape
    spacing = max(min(dx, dy) * 0.5, np.finfo(float).eps)
    # 额外包含最后一点到第一点的闭合段；find_contours 通常重复首点，
    # 但这里不把该库的表示细节当作 solid policy 的前提。
    for p0, p1 in zip(path, np.roll(path, -1, axis=0)):
        segment_length = float(np.hypot((p1[0] - p0[0]) * dy,
                                        (p1[1] - p0[1]) * dx))
        count = max(1, int(np.ceil(segment_length / spacing)))
        rows = np.rint(np.linspace(p0[0], p1[0], count + 1)).astype(int)
        cols = np.rint(np.linspace(p0[1], p1[1], count + 1)).astype(int)
        valid = (rows >= 0) & (rows < Y) & (cols >= 0) & (cols < X)
        if np.any(solid[rows[valid], cols[valid]]):
            return True
    return False


def contour_metrics(contour: Any, xdim: Any, ydim: Any) -> dict[str, Any]:
    """计算 contour 的物理面积、周长、凸包面积和凸度缺陷。

    这是公开的 geometry seam，便于测试和 sensitivity diagnostics 复用同一
    公式；输入点仍使用 skimage 的 ``(row,col)`` 网格坐标。
    """
    path = np.asarray(contour, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 3:
        raise ValueError("contour 必须是至少 3 个 (row,col) 点")
    x, y, dx, dy = _validate_coords(xdim, ydim, (len(ydim), len(xdim)))
    xy = _grid_path_to_xy(path, x, y)
    if not np.allclose(xy[0], xy[-1]):
        closed_xy = np.vstack([xy, xy[0]])
    else:
        closed_xy = xy
    area = 0.5 * abs(float(np.dot(closed_xy[:-1, 0], closed_xy[1:, 1])
                              - np.dot(closed_xy[:-1, 1], closed_xy[1:, 0])))
    perimeter = float(np.linalg.norm(np.diff(closed_xy, axis=0), axis=1).sum())
    unique_xy = closed_xy[:-1] if np.allclose(closed_xy[0], closed_xy[-1]) else closed_xy
    hull_area = 0.0
    if len(unique_xy) >= 3:
        _, ConvexHull, QhullError, _, _, _ = _haller_geometry_dependencies()
        try:
            hull_area = float(ConvexHull(unique_xy).volume)
        except QhullError:
            hull_area = 0.0
    defect = float("inf") if hull_area <= 0.0 else max(0.0, (hull_area - area) / hull_area)
    return {
        "area": float(area),
        "perimeter": perimeter,
        "hull_area": hull_area,
        "convexity_defect": defect,
        "n_points": int(len(path)),
        "dx": dx,
        "dy": dy,
    }


def _candidate_record(
    path: np.ndarray,
    peak: Mapping[str, Any],
    level: float,
    level_fraction: float,
    xdim: np.ndarray,
    ydim: np.ndarray,
    closed: bool,
    reason: str | None = None,
    *,
    include_points: bool = True,
) -> dict[str, Any]:
    """构造可 JSON 序列化的 contour diagnostics 记录。"""
    record: dict[str, Any] = {
        "peak": {
            "row": int(peak["row"]),
            "col": int(peak["col"]),
            "value": float(peak["value"]),
        },
        "level": float(level),
        "level_fraction": float(level_fraction),
        "closed": bool(closed),
        "status": "valid" if reason is None else "rejected",
    }
    if reason is not None:
        record["rejection_reason"] = reason
        return record
    metrics = contour_metrics(path, xdim, ydim)
    for key in ("area", "perimeter", "hull_area", "convexity_defect"):
        record[key] = float(np.round(float(metrics[key]), decimals=10))
    record["n_points"] = int(metrics["n_points"])
    # 只有 selected contour 保留点列；32-level diagnostics 只保留 metrics，
    # 避免真实帧的 metadata 因重复路径膨胀到数百 MB。
    if include_points:
        record["points_grid"] = np.asarray(path, dtype=np.float64).round(12).tolist()
        record["points_xy"] = _grid_path_to_xy(path, xdim, ydim).round(12).tolist()
    return record


def _compress_contour_diagnostics(
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按 peak/level/status/reason 聚合重复候选，保留可审计计数。

    真实流场的一个 level 可能返回大量不相关的小 contour。逐条把它们
    写入 JSON 会让 metadata 从 KB 膨胀到数百 MB；算法仍逐条筛选，只有
    diagnostics 的持久化表示做无损计数聚合，selected contour 几何另存。
    """
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for record in diagnostics:
        peak = record["peak"]
        key = (
            int(peak["row"]),
            int(peak["col"]),
            float(peak["value"]),
            float(record["level"]),
            float(record["level_fraction"]),
            str(record["status"]),
            record.get("rejection_reason"),
        )
        if key not in grouped:
            summary = dict(record)
            summary["candidate_count"] = int(record.get("candidate_count", 1))
            grouped[key] = summary
            order.append(key)
        else:
            grouped[key]["candidate_count"] += int(record.get("candidate_count", 1))
    # The reference implementation discovers records in peak/level/path
    # order.  The optimized implementation aggregates the same categories by
    # level, so canonicalize only the compressed presentation order.  The
    # grouping key is the public diagnostic identity; sorting it cannot change
    # candidate counts, geometry, or the selected contour.
    return [
        grouped[key]
        for key in sorted(
            order,
            key=lambda key: (
                int(key[0]), int(key[1]), float(key[2]), float(key[3]),
                float(key[4]), str(key[5]), "" if key[6] is None else str(key[6]),
            ),
        )
    ]


@dataclass(frozen=True)
class _ContourPathEntry:
    """One globally evaluated contour path and its peak membership relation."""

    path: np.ndarray
    closed: bool
    contains_peak_indices: frozenset[int]
    rejection_reason: str | None
    metrics: Mapping[str, Any] | None


@dataclass(frozen=True)
class _ContourLevelEntry:
    """All paths returned by one global ``find_contours(level=...)`` call."""

    paths: tuple[_ContourPathEntry, ...] = ()
    error: str | None = None
    roi_bounds: tuple[int, int, int, int] | None = None
    roi_boundary_fallback: bool = False
    find_contours_call_count: int = 0
    find_contours_elapsed_seconds: tuple[float, ...] = ()
    roi_attempt_count: int = 0
    full_domain_call_count: int = 0


def _contour_timing_summary(durations: Any) -> dict[str, Any]:
    """Summarize actual ``find_contours`` wall times for artifact audit."""
    values = np.asarray(tuple(float(value) for value in durations), dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0.0)]
    count = int(values.size)
    total = float(values.sum()) if count else 0.0
    mean = float(total / count) if count else 0.0
    p95 = float(np.percentile(values, 95.0)) if count else 0.0
    return {
        "find_contours_call_count": count,
        "find_contours_total_seconds": total,
        "find_contours_mean_seconds": mean,
        "find_contours_p95_seconds": p95,
    }


def _superlevel_peak_index(
    ivd: np.ndarray,
    fluid: np.ndarray,
    level: float,
    peak_points: np.ndarray,
    peak_values: np.ndarray,
    ndimage: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, frozenset[int]]]:
    """Index eligible peaks by the exact high-side component of one level.

    ``find_contours`` remains the source of all candidate geometry.  This
    index only replaces repeated point-in-polygon calls with a containment
    lookup: under skimage's default ``fully_connected='low'`` convention,
    high-valued cells are 4-connected, and a closed level path cannot split a
    connected high-valued component without crossing that path.  Every
    eligible peak in one such component therefore has the same relation to a
    candidate contour.  The component representative is still checked with
    the original ``points_in_poly`` predicate below.
    """
    structure = np.asarray(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )
    labels, _ = ndimage.label(fluid & (ivd >= float(level)), structure=structure)
    if len(peak_points) == 0:
        return labels, np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.intp), {}
    rows = np.asarray(peak_points[:, 0], dtype=np.intp)
    cols = np.asarray(peak_points[:, 1], dtype=np.intp)
    peak_labels = labels[rows, cols]
    eligible = (peak_values >= float(level)) & (peak_labels > 0)
    component_peaks: dict[int, set[int]] = {}
    for index in np.flatnonzero(eligible):
        component = int(peak_labels[index])
        component_peaks.setdefault(component, set()).add(int(index))
    components = np.asarray(sorted(component_peaks), dtype=np.intp)
    representatives = np.asarray(
        [peak_points[min(component_peaks[int(component)])] for component in components],
        dtype=np.float64,
    ).reshape((-1, 2))
    frozen = {
        component: frozenset(indices)
        for component, indices in component_peaks.items()
    }
    return labels, representatives, components, frozen


def _peaks_contained_by_component_index(
    path: np.ndarray,
    peak_points: np.ndarray,
    representatives: np.ndarray,
    representative_components: np.ndarray,
    component_peaks: Mapping[int, frozenset[int]],
    points_in_poly: Any,
) -> frozenset[int]:
    """Return peak indices using representative polygon membership exactly."""
    if len(representatives) == 0:
        return frozenset()
    row_min, col_min = np.min(path, axis=0)
    row_max, col_max = np.max(path, axis=0)
    candidate_indices = np.flatnonzero(
        (representatives[:, 0] >= row_min)
        & (representatives[:, 0] <= row_max)
        & (representatives[:, 1] >= col_min)
        & (representatives[:, 1] <= col_max)
    )
    if len(candidate_indices) == 0:
        return frozenset()
    inside = points_in_poly(representatives[candidate_indices], path)
    contained_components = np.unique(
        representative_components[candidate_indices[np.flatnonzero(inside)]]
    )
    result: set[int] = set()
    for component in contained_components:
        result.update(component_peaks[int(component)])
    return frozenset(result)


def expand_component_roi(
    roi: tuple[int, int, int, int],
    shape: tuple[int, int],
    *,
    margin_cells: int = 1,
) -> tuple[tuple[int, int, int, int], bool]:
    """Expand a half-open grid ROI and report whether it touched the domain.

    This is a computational search-window helper only.  It never replaces the
    contour test with a threshold component: callers still run the unchanged
    ``find_contours``/closed/containment/solid/convexity/outermost checks.  A
    true boundary-contact flag tells the caller that the crop must be retried
    on a larger window (or on the full domain when no larger window exists).
    """
    if len(shape) != 2 or any(int(value) <= 0 for value in shape):
        raise ValueError(f"shape 必须是正二维 shape，实际 {shape!r}")
    if len(roi) != 4:
        raise ValueError(f"ROI 必须是 (row0,row1,col0,col1)，实际 {roi!r}")
    rows, cols = (int(shape[0]), int(shape[1]))
    row0, row1, col0, col1 = (int(value) for value in roi)
    if not (0 <= row0 < row1 <= rows and 0 <= col0 < col1 <= cols):
        raise ValueError(f"ROI 越界或为空：roi={roi!r}, shape={shape!r}")
    margin = int(margin_cells)
    if margin < 0:
        raise ValueError(f"margin_cells 不能为负，实际 {margin_cells!r}")
    touches_domain = row0 == 0 or row1 == rows or col0 == 0 or col1 == cols
    return (
        (
            max(0, row0 - margin),
            min(rows, row1 + margin),
            max(0, col0 - margin),
            min(cols, col1 + margin),
        ),
        touches_domain,
    )


def _contour_level_schedule(
    peaks: list[dict[str, Any]], params: Mapping[str, Any]
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Return the frozen per-peak fractions and exact absolute level requests."""
    fractions = np.linspace(
        float(params["contour_level_start"]),
        float(params["contour_level_end"]),
        int(params["contour_level_count"]),
    )
    requested_levels = tuple(
        float(peak["value"]) * float(fraction)
        for peak in peaks
        for fraction in fractions
    )
    return fractions, requested_levels


def _contour_search_roi(
    ivd: np.ndarray,
    fluid: np.ndarray,
    level: float,
    peak_points: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    """Return the initial conservative crop for exact peak-local search.

    With ``peak_points`` this starts around the requested peaks and is grown
    by :func:`_find_contours_with_exact_roi` until the local high-side support
    and all returned paths are interior to the crop.  Without peak points the
    old support-bounds helper remains available for direct seam tests.
    Neither form is a label substitute: the caller still invokes the same
    marching-squares implementation and all contour geometry predicates.
    """
    shape = tuple(int(value) for value in ivd.shape)
    if peak_points is not None:
        points = np.asarray(peak_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            raise ValueError("peak_points 必须是非空的 (N,2) 坐标数组")
        radius = int(_ROI_INITIAL_RADIUS_CELLS)
        row0 = int(np.floor(np.min(points[:, 0]))) - radius
        row1 = int(np.ceil(np.max(points[:, 0]))) + radius + 1
        col0 = int(np.floor(np.min(points[:, 1]))) - radius
        col1 = int(np.ceil(np.max(points[:, 1]))) + radius + 1
        return (
            max(0, row0), min(shape[0], row1),
            max(0, col0), min(shape[1], col1),
        )
    rows, cols = np.nonzero(fluid & (ivd >= float(level)))
    if len(rows) == 0:
        return (0, shape[0], 0, shape[1])
    bounds = (int(rows.min()), int(rows.max()) + 1,
              int(cols.min()), int(cols.max()) + 1)
    expanded, _ = expand_component_roi(bounds, shape, margin_cells=1)
    return expanded


def _path_touches_crop_boundary(path: np.ndarray, shape: tuple[int, int]) -> bool:
    """Detect a clipped marching-squares path at a local crop boundary."""
    array = np.asarray(path, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
        return True
    height, width = (int(shape[0]), int(shape[1]))
    return bool(
        np.any(array[:, 0] <= 1e-12)
        or np.any(array[:, 0] >= float(height - 1) - 1e-12)
        or np.any(array[:, 1] <= 1e-12)
        or np.any(array[:, 1] >= float(width - 1) - 1e-12)
    )


def _find_contours_with_exact_roi(
    ivd: np.ndarray,
    fluid: np.ndarray,
    level: float,
    find_contours: Any,
    peak_points: np.ndarray | None = None,
) -> tuple[tuple[np.ndarray, ...], tuple[int, int, int, int], bool, int]:
    """Find exact peak-local contours, expanding until the crop is safe.

    A local call is equivalent to the full-domain call for every contour that
    can contain a requested peak once (1) the high-side support containing the
    peak is strictly interior and (2) every returned path is strictly
    interior.  Condition (1) prevents an enclosing level-set boundary from
    being outside the crop; condition (2) prevents a marching-squares path
    from being clipped by the crop.  If either condition fails, the crop is
    expanded geometrically and retried.  A domain-sized crop, or a crop that
    cannot be expanded further, executes the exact full-domain call.  No path
    is discarded because it touches an ROI boundary.

    For direct seam tests without ``peak_points`` the initial ROI is derived
    from all level-supporting cells, preserving the historical helper API.
    """
    shape = tuple(int(value) for value in ivd.shape)
    domain_bounds = (0, shape[0], 0, shape[1])
    roi = _contour_search_roi(ivd, fluid, level, peak_points)
    if roi == domain_bounds:
        paths = tuple(
            np.asarray(path, dtype=np.float64).copy()
            for path in find_contours(ivd, level=level, mask=fluid)
        )
        return paths, domain_bounds, False, 1

    call_count = 0
    used_full_domain_fallback = False
    while True:
        row0, row1, col0, col1 = roi
        local_shape = (row1 - row0, col1 - col0)
        call_count += 1
        local_paths = tuple(
            np.asarray(path, dtype=np.float64).copy()
            for path in find_contours(
                ivd[row0:row1, col0:col1],
                level=level,
                mask=fluid[row0:row1, col0:col1],
            )
        )
        high_side = fluid[row0:row1, col0:col1] & (
            ivd[row0:row1, col0:col1] >= float(level)
        )
        support_touches = bool(
            np.any(high_side[0, :])
            or np.any(high_side[-1, :])
            or np.any(high_side[:, 0])
            or np.any(high_side[:, -1])
        )
        contour_touches = any(
            _path_touches_crop_boundary(path, local_shape)
            for path in local_paths
        )
        if not support_touches and not contour_touches:
            shifted = tuple(
                np.asarray(path, dtype=np.float64)
                + np.asarray([row0, col0], dtype=np.float64)
                for path in local_paths
            )
            return shifted, roi, bool(
                used_full_domain_fallback and roi == domain_bounds
            ), call_count

        if roi == domain_bounds:
            # The previous call was already the full-domain reference.  This
            # branch is only reached for a domain-edge contour/support; it is
            # still a valid result and no candidate was dropped.
            shifted = tuple(np.asarray(path, dtype=np.float64).copy()
                            for path in local_paths)
            return shifted, domain_bounds, used_full_domain_fallback, call_count

        height = row1 - row0
        width = col1 - col0
        expanded, _ = expand_component_roi(
            roi, shape, margin_cells=max(1, height, width)
        )
        if expanded == roi:
            # This can only happen when the crop already spans the domain in
            # every expandable direction.  Run the explicit full-domain
            # reference if the local call was not itself that call.
            if roi != domain_bounds:
                call_count += 1
                paths = tuple(
                    np.asarray(path, dtype=np.float64).copy()
                    for path in find_contours(ivd, level=level, mask=fluid)
                )
                return paths, domain_bounds, True, call_count
        if expanded == domain_bounds:
            used_full_domain_fallback = True
        roi = expanded


def _valid_contour_record(
    path: np.ndarray,
    peak: Mapping[str, Any],
    level: float,
    level_fraction: float,
    metrics: Mapping[str, Any],
    xdim: np.ndarray,
    ydim: np.ndarray,
) -> dict[str, Any]:
    """Build a valid diagnostic record from metrics computed by the hierarchy."""
    record: dict[str, Any] = {
        "peak": {
            "row": int(peak["row"]),
            "col": int(peak["col"]),
            "value": float(peak["value"]),
        },
        "level": float(level),
        "level_fraction": float(level_fraction),
        "closed": True,
        "status": "valid",
    }
    record.update({
        key: (
            int(metrics[key])
            if key == "n_points"
            else float(np.round(float(metrics[key]), decimals=10))
        )
        for key in ("area", "perimeter", "hull_area", "convexity_defect", "n_points")
    })
    return record


def _build_contour_hierarchy(
    ivd: np.ndarray,
    solid: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
    peaks: list[dict[str, Any]],
    params: Mapping[str, Any],
    dx: float,
    dy: float,
) -> tuple[dict[float, _ContourLevelEntry], np.ndarray]:
    """Evaluate frozen exact levels in peak-local ROIs and index paths.

    Each unique absolute level is evaluated once for the requested peak set,
    using the same ``skimage.measure.find_contours`` algorithm in a crop that
    is expanded until no high-side support or returned path touches its
    boundary.  If that proof obligation cannot be met locally, the helper
    executes the full-domain reference call.  The hierarchy then applies the
    unchanged closed/contains/solid/convexity/perimeter tests and lets the
    selector choose the outermost valid contour.
    """
    _, _, QhullError, find_contours, _, points_in_poly = _haller_geometry_dependencies()
    fluid = ~solid
    fractions, requested_levels = _contour_level_schedule(peaks, params)
    hierarchy: dict[float, _ContourLevelEntry] = {}
    min_perimeter = float(params["minimum_perimeter_factor"]) * max(dx, dy)
    peak_points = np.asarray(
        [[peak["row"], peak["col"]] for peak in peaks], dtype=np.float64
    )
    requested_peaks_by_level: dict[float, list[int]] = {}
    for peak_index, peak in enumerate(peaks):
        for fraction in fractions:
            requested_peaks_by_level.setdefault(
                float(peak["value"]) * float(fraction), []
            ).append(peak_index)

    for level in dict.fromkeys(requested_levels):
        requested_peak_indices = requested_peaks_by_level.get(float(level), [])
        if not requested_peak_indices:
            raise RuntimeError(
                f"contour hierarchy level={level!r} 缺少原始 peak request"
            )
        requested_peak_points = peak_points[requested_peak_indices]
        contour_durations: list[float] = []

        def timed_find_contours(*args: Any, **kwargs: Any) -> Any:
            contour_start = time.perf_counter()
            try:
                return find_contours(*args, **kwargs)
            finally:
                contour_durations.append(time.perf_counter() - contour_start)

        try:
            raw_paths, roi_bounds, roi_boundary_fallback, call_count = (
                _find_contours_with_exact_roi(
                    ivd,
                    fluid,
                    level,
                    timed_find_contours,
                    peak_points=requested_peak_points,
                )
            )
        except (ValueError, RuntimeError) as exc:
            hierarchy[level] = _ContourLevelEntry(
                error=f"contour_error:{type(exc).__name__}",
                find_contours_call_count=len(contour_durations),
                find_contours_elapsed_seconds=tuple(contour_durations),
            )
            continue
        entries: list[_ContourPathEntry] = []
        for raw_path in raw_paths:
            # The ROI helper has already translated a successful local path
            # to global row/column coordinates.  The same _closed_path and
            # all geometry predicates therefore see global coordinates in
            # either execution path.
            closed, path = _closed_path(
                raw_path, dx, dy, float(params["closure_tolerance_factor"])
            )
            if not closed:
                entries.append(_ContourPathEntry(
                    path=path,
                    closed=False,
                    contains_peak_indices=frozenset(),
                    rejection_reason="open_contour",
                    metrics=None,
                ))
                continue

            inside = points_in_poly(
                requested_peak_points,
                path,
            )
            contains = frozenset(
                int(requested_peak_indices[index])
                for index in np.flatnonzero(inside)
            )
            # A contour that contains no local maximum can never be selected
            # for any peak.  The reference path records the same
            # ``does_not_enclose_peak`` rejection before solid/geometry
            # screening, so skipping those expensive checks is an exact
            # reordering rather than a new rejection rule.  Keep the entry
            # because the per-peak diagnostic expansion below still reports
            # it for every peak.
            if not contains:
                entries.append(_ContourPathEntry(
                    path=path,
                    closed=True,
                    contains_peak_indices=contains,
                    rejection_reason="does_not_enclose_peak",
                    metrics=None,
                ))
                continue
            if _contour_crosses_solid(path, solid, dx, dy):
                entries.append(_ContourPathEntry(
                    path=path,
                    closed=True,
                    contains_peak_indices=contains,
                    rejection_reason="solid_crossing",
                    metrics=None,
                ))
                continue
            try:
                metrics = contour_metrics(path, xdim, ydim)
            except (ValueError, QhullError) as exc:
                entries.append(_ContourPathEntry(
                    path=path,
                    closed=True,
                    contains_peak_indices=contains,
                    rejection_reason=f"geometry_error:{type(exc).__name__}",
                    metrics=None,
                ))
                continue
            if metrics["area"] <= 0.0:
                reason = "zero_area"
            elif metrics["convexity_defect"] > float(params["convexity_defect_max"]) + 1e-12:
                reason = "convexity_defect"
            elif metrics["perimeter"] + 1e-12 < min_perimeter:
                reason = "minimum_perimeter"
            else:
                reason = None
            entries.append(_ContourPathEntry(
                path=path,
                closed=True,
                contains_peak_indices=contains,
                rejection_reason=reason,
                metrics=metrics,
            ))
        hierarchy[level] = _ContourLevelEntry(
            paths=tuple(entries),
            roi_bounds=roi_bounds,
            roi_boundary_fallback=bool(roi_boundary_fallback),
            find_contours_call_count=int(call_count),
            find_contours_elapsed_seconds=tuple(contour_durations),
            roi_attempt_count=(
                int(call_count - 1)
                if roi_boundary_fallback
                else int(
                    call_count
                    if roi_bounds != (0, ivd.shape[0], 0, ivd.shape[1])
                    else 0
                )
            ),
            full_domain_call_count=(
                1
                if roi_bounds == (0, ivd.shape[0], 0, ivd.shape[1])
                else 0
            ),
        )
    return hierarchy, fractions


def _select_contours_from_hierarchy(
    ivd: np.ndarray,
    solid: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
    peaks: list[dict[str, Any]],
    params: Mapping[str, Any],
    dx: float,
    dy: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply the unchanged per-peak outermost rule to a global hierarchy.

    The hierarchy has already paid for each exact absolute level once.  This
    second pass must not undo that saving by scanning every path once per peak.
    Instead, a level is traversed once and the path membership relation is
    distributed to the peaks it actually contains.  ``does_not_enclose_peak``
    is the exact complement among closed paths, and diagnostics are aggregated
    with the same identity used by ``_compress_contour_diagnostics``.  No
    candidate is removed: all paths, geometry checks, and per-peak selections
    remain represented by their exact counts and first representative.
    """
    hierarchy, fractions = _build_contour_hierarchy(
        ivd, solid, xdim, ydim, peaks, params, dx, dy
    )
    selected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    requests_by_level: dict[float, list[tuple[int, float]]] = {}
    fractions_by_peak_level: list[dict[float, list[float]]] = [
        {} for _ in peaks
    ]
    for peak_index, peak in enumerate(peaks):
        peak_value = float(peak["value"])
        for fraction in fractions:
            level_fraction = float(fraction)
            level = float(peak_value * level_fraction)
            requests_by_level.setdefault(level, []).append(
                (peak_index, level_fraction)
            )
            fractions_by_peak_level[peak_index].setdefault(level, []).append(
                level_fraction
            )

    valid_entries_by_peak: list[list[tuple[float, _ContourPathEntry]]] = [
        [] for _ in peaks
    ]
    peak_count = len(peaks)
    for level, level_entry in hierarchy.items():
        requests = requests_by_level.get(float(level), [])
        if not requests:
            raise RuntimeError(
                f"contour hierarchy level={level!r} 缺少原始 level request"
            )
        if level_entry.error is not None:
            for peak_index, level_fraction in requests:
                diagnostics.append(_candidate_record(
                    np.empty((0, 2)), peaks[peak_index], level,
                    level_fraction, xdim, ydim, False, level_entry.error,
                ))
            continue
        if not level_entry.paths:
            for peak_index, level_fraction in requests:
                diagnostics.append(_candidate_record(
                    np.empty((0, 2)), peaks[peak_index], level,
                    level_fraction, xdim, ydim, False, "no_contour",
                ))
            continue

        requested_peak_indices = {peak_index for peak_index, _ in requests}
        open_entries = [entry for entry in level_entry.paths if not entry.closed]
        closed_entries = [entry for entry in level_entry.paths if entry.closed]
        first_open = open_entries[0] if open_entries else None
        first_closed = closed_entries[0] if closed_entries else None
        contained_counts = np.zeros(peak_count, dtype=np.int64)
        valid_counts = np.zeros(peak_count, dtype=np.int64)
        valid_first: list[_ContourPathEntry | None] = [None] * peak_count
        rejected_by_peak: list[dict[str, tuple[int, _ContourPathEntry]]] = [
            {} for _ in peaks
        ]

        # Each global path is visited once.  The component-index membership
        # relation was established by the unchanged polygon predicate in the
        # hierarchy builder; distribute only to peaks requested at this level.
        for entry in closed_entries:
            for peak_index in entry.contains_peak_indices:
                if peak_index not in requested_peak_indices:
                    continue
                contained_counts[peak_index] += 1
                if entry.rejection_reason is None:
                    valid_counts[peak_index] += 1
                    if valid_first[peak_index] is None:
                        valid_first[peak_index] = entry
                    valid_entries_by_peak[peak_index].append((float(level), entry))
                else:
                    reason = str(entry.rejection_reason)
                    previous = rejected_by_peak[peak_index].get(reason)
                    rejected_by_peak[peak_index][reason] = (
                        1 if previous is None else previous[0] + 1,
                        entry if previous is None else previous[1],
                    )

        for peak_index, level_fraction in requests:
            peak = peaks[peak_index]
            if first_open is not None:
                record = _candidate_record(
                    first_open.path, peak, level, level_fraction,
                    xdim, ydim, False, "open_contour",
                )
                record["candidate_count"] = len(open_entries)
                diagnostics.append(record)

            non_containing_count = len(closed_entries) - int(
                contained_counts[peak_index]
            )
            if non_containing_count:
                if first_closed is None:
                    raise RuntimeError("closed contour count 与 hierarchy 不一致")
                record = _candidate_record(
                    first_closed.path, peak, level, level_fraction,
                    xdim, ydim, True, "does_not_enclose_peak",
                )
                record["candidate_count"] = non_containing_count
                diagnostics.append(record)

            for reason, (count, representative) in rejected_by_peak[peak_index].items():
                record = _candidate_record(
                    representative.path, peak, level, level_fraction,
                    xdim, ydim, True, reason,
                )
                record["candidate_count"] = int(count)
                diagnostics.append(record)

            representative = valid_first[peak_index]
            if representative is not None:
                if representative.metrics is None:
                    raise RuntimeError("contour hierarchy valid entry 缺少 metrics")
                record = _valid_contour_record(
                    representative.path, peak, level, level_fraction,
                    representative.metrics, xdim, ydim,
                )
                record["candidate_count"] = int(valid_counts[peak_index])
                diagnostics.append(record)

    for peak_index, peak in enumerate(peaks):
        valid_for_peak: list[tuple[dict[str, Any], np.ndarray]] = []
        for level, entry in valid_entries_by_peak[peak_index]:
            if entry.metrics is None:
                raise RuntimeError("contour hierarchy valid entry 缺少 metrics")
            for level_fraction in fractions_by_peak_level[peak_index][level]:
                valid_for_peak.append((
                    _valid_contour_record(
                        entry.path, peak, level, level_fraction,
                        entry.metrics, xdim, ydim,
                    ),
                    entry.path,
                ))
        if valid_for_peak:
            chosen_record, chosen_path = max(valid_for_peak, key=lambda item: (
                float(item[0]["area"]), -float(item[0]["level"]),
            ))
            chosen = dict(chosen_record)
            chosen["points_grid"] = np.asarray(chosen_path, dtype=np.float64).round(12).tolist()
            chosen["points_xy"] = _grid_path_to_xy(
                chosen_path, xdim, ydim
            ).round(12).tolist()
            selected.append(chosen)
    requested_levels = tuple(
        float(peak["value"]) * float(fraction)
        for peak in peaks
        for fraction in fractions
    )
    level_entries = list(hierarchy.values())
    contour_durations = tuple(
        duration
        for entry in level_entries
        for duration in entry.find_contours_elapsed_seconds
    )
    evaluation = {
        "requested_peak_level_count": len(requested_levels),
        "unique_absolute_level_count": len(hierarchy),
        **_contour_timing_summary(contour_durations),
        "level_key": "exact_float64_absolute_ivd_level",
        "level_schedule": "per_peak_relative_frozen",
        "roi_policy": (
            "peak_local_exact_expand_on_high_support_or_contour_boundary_then_full_domain"
        ),
        "roi_attempt_count": sum(
            int(entry.roi_attempt_count) for entry in level_entries
        ),
        "roi_boundary_fallback_count": sum(
            int(entry.roi_boundary_fallback) for entry in level_entries
        ),
        "full_domain_level_count": sum(
            int(entry.full_domain_call_count) for entry in level_entries
        ),
        "roi_level_count": sum(
            int(entry.roi_attempt_count > 0) for entry in level_entries
        ),
    }
    return selected, _compress_contour_diagnostics(diagnostics), evaluation


def _select_contours(
    ivd: np.ndarray,
    solid: np.ndarray,
    xdim: np.ndarray,
    ydim: np.ndarray,
    peaks: list[dict[str, Any]],
    params: Mapping[str, Any],
    dx: float,
    dy: float,
    *,
    backend: str = BACKEND_NUMPY,
    device: Any = None,
    contour_mode: str = CONTOUR_MODE_OPTIMIZED,
    fast_global_level_count: int = 64,
    fast_refinement_iterations: int = 7,
    fast_refinement_halo_cells: int = 2,
    return_evaluation: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Search every frozen level while sharing exact contour evaluations.

    ``reference`` calls ``find_contours`` at each peak/level request, which is
    the deliberately slow audit implementation.  ``optimized`` groups equal
    exact absolute levels, then invokes the same contour algorithm in a
    peak-local ROI.  The ROI is expanded until its local result is safe under
    the interior-support/interior-path equivalence conditions, otherwise the
    full-domain reference call is used.  The level schedule, path geometry,
    nesting checks, convexity checks, and outermost selection are unchanged.

    The frozen specification defines levels relative to each peak.  Therefore
    two different peaks usually produce different absolute levels; grouping
    only equal exact requests does not alter the level discretisation.
    """
    contour_mode = _normalize_contour_mode(contour_mode)
    # Geometry is intentionally CPU/skimage-owned for both numerical
    # backends.  Still resolve the requested backend here so this private seam
    # cannot accidentally turn an explicit CUDA request into a CPU run.
    backend_details = _resolve_backend(backend, device)
    if backend_details.resolved == BACKEND_FAST_HALLER:
        if contour_mode != CONTOUR_MODE_OPTIMIZED:
            raise ValueError(
                "fast_haller 使用自身的 ContourPy 搜索；严格 oracle 请使用 "
                "strict oracle: backend='numpy', contour_mode='reference'"
            )
        from fast_haller import select_contours as select_fast_contours

        return select_fast_contours(
            ivd,
            solid,
            xdim,
            ydim,
            peaks,
            params,
            dx,
            dy,
            global_level_count=fast_global_level_count,
            refinement_iterations=fast_refinement_iterations,
            refinement_halo_cells=fast_refinement_halo_cells,
            return_evaluation=return_evaluation,
        )
    if backend_details.resolved == BACKEND_NUMBACS:
        if contour_mode == CONTOUR_MODE_REFERENCE:
            raise ValueError(
                "numbacs backend 使用 upstream rotcohvrt；严格 oracle 请使用 "
                "backend='numpy', contour_mode='reference'"
            )
        from numbacs_haller import select_contours as select_numbacs_contours

        return select_numbacs_contours(
            ivd,
            solid,
            xdim,
            ydim,
            params,
            dx,
            dy,
            return_evaluation=return_evaluation,
        )
    if contour_mode == CONTOUR_MODE_NUMBACS:
        raise ValueError(
            "contour_mode='numbacs' 必须与 backend='numbacs' 一起使用"
        )
    if contour_mode == CONTOUR_MODE_OPTIMIZED:
        selected, diagnostics, evaluation = _select_contours_from_hierarchy(
            ivd, solid, xdim, ydim, peaks, params, dx, dy
        )
        if return_evaluation:
            return selected, diagnostics, evaluation
        return selected, diagnostics
    _, _, QhullError, find_contours, _, points_in_poly = _haller_geometry_dependencies()
    fluid = ~solid
    selected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    min_perimeter = float(params["minimum_perimeter_factor"]) * max(dx, dy)
    fractions, _ = _contour_level_schedule(peaks, params)
    contour_durations: list[float] = []

    for peak in peaks:
        peak_value = float(peak["value"])
        valid_for_peak: list[tuple[dict[str, Any], np.ndarray]] = []
        for fraction in fractions:
            level = peak_value * float(fraction)
            contour_start = time.perf_counter()
            try:
                paths = tuple(
                    np.asarray(path, dtype=np.float64)
                    for path in find_contours(ivd, level=level, mask=fluid)
                )
            except (ValueError, RuntimeError) as exc:
                contour_durations.append(time.perf_counter() - contour_start)
                diagnostics.append(_candidate_record(
                    np.empty((0, 2)), peak, level, float(fraction), xdim, ydim,
                    False, f"contour_error:{type(exc).__name__}",
                ))
                continue
            contour_durations.append(time.perf_counter() - contour_start)
            if not paths:
                diagnostics.append(_candidate_record(
                    np.empty((0, 2)), peak, level, float(fraction), xdim, ydim,
                    False, "no_contour",
                ))
                continue
            for path in paths:
                closed, path = _closed_path(
                    path, dx, dy, float(params["closure_tolerance_factor"])
                )
                if not closed:
                    diagnostics.append(_candidate_record(
                        path, peak, level, float(fraction), xdim, ydim,
                        False, "open_contour",
                    ))
                    continue
                if not bool(points_in_poly(
                    np.asarray([[peak["row"], peak["col"]]], dtype=np.float64), path
                )[0]):
                    diagnostics.append(_candidate_record(
                        path, peak, level, float(fraction), xdim, ydim,
                        True, "does_not_enclose_peak",
                    ))
                    continue
                if _contour_crosses_solid(path, solid, dx, dy):
                    diagnostics.append(_candidate_record(
                        path, peak, level, float(fraction), xdim, ydim,
                        True, "solid_crossing",
                    ))
                    continue
                try:
                    metrics = contour_metrics(path, xdim, ydim)
                except (ValueError, QhullError) as exc:
                    diagnostics.append(_candidate_record(
                        path, peak, level, float(fraction), xdim, ydim,
                        True, f"geometry_error:{type(exc).__name__}",
                    ))
                    continue
                if metrics["area"] <= 0.0:
                    diagnostics.append(_candidate_record(
                        path, peak, level, float(fraction), xdim, ydim,
                        True, "zero_area",
                    ))
                    continue
                if metrics["convexity_defect"] > float(params["convexity_defect_max"]) + 1e-12:
                    diagnostics.append(_candidate_record(
                        path, peak, level, float(fraction), xdim, ydim,
                        True, "convexity_defect",
                    ))
                    continue
                if metrics["perimeter"] + 1e-12 < min_perimeter:
                    diagnostics.append(_candidate_record(
                        path, peak, level, float(fraction), xdim, ydim,
                        True, "minimum_perimeter",
                    ))
                    continue
                record = _candidate_record(
                    path, peak, level, float(fraction), xdim, ydim, True, None,
                    include_points=False,
                )
                diagnostics.append(record)
                # 合法记录留给下面的 per-peak 候选集合；diagnostics 同时
                # 保留完整 32-level 的 rejected/valid 轨迹。
                valid_for_peak.append((record, path))
        if valid_for_peak:
            # 最外围定义为最大物理面积；level 作为确定性的 tie-break。
            chosen_record, chosen_path = max(valid_for_peak, key=lambda item: (
                float(item[0]["area"]), -float(item[0]["level"]),
            ))
            chosen = dict(chosen_record)
            chosen["points_grid"] = np.asarray(chosen_path, dtype=np.float64).round(12).tolist()
            chosen["points_xy"] = _grid_path_to_xy(
                chosen_path, xdim, ydim
            ).round(12).tolist()
            selected.append(chosen)
    compressed = _compress_contour_diagnostics(diagnostics)
    if return_evaluation:
        _, requested_levels = _contour_level_schedule(peaks, params)
        evaluation = {
            "requested_peak_level_count": len(requested_levels),
            "unique_absolute_level_count": len(set(requested_levels)),
            "level_key": "exact_float64_absolute_ivd_level",
            "level_schedule": "per_peak_relative_frozen",
            "roi_policy": "not_used_reference_mode",
            "roi_attempt_count": 0,
            "roi_boundary_fallback_count": 0,
            "full_domain_level_count": len(requested_levels),
        }
        evaluation.update(_contour_timing_summary(contour_durations))
        return selected, compressed, evaluation
    return selected, compressed




# --------------------------------------------------------------------------- 三态 anchor 分类

def _rasterize_union(
    contours: list[dict[str, Any]], shape: tuple[int, int], dx: float, dy: float,
    unknown_band_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """将每个合法 contour 栅格化，并按显式 union 规则合并 interior/band。"""
    ndimage, _, _, _, grid_points_in_poly, _ = _haller_geometry_dependencies()
    interior_union = np.zeros(shape, dtype=bool)
    band_union = np.zeros(shape, dtype=bool)
    for contour in contours:
        path = np.asarray(contour["points_grid"], dtype=np.float64)
        interior = grid_points_in_poly(shape, path)
        interior_union |= interior
        # 分别从 inside/outside 的最近网格格计算物理距离，等价于一个
        # 与 dx/dy 一致的 morphology band；不对不同数据集静默改宽度。
        inside_distance = ndimage.distance_transform_edt(interior, sampling=(dy, dx))
        outside_distance = ndimage.distance_transform_edt(~interior, sampling=(dy, dx))
        band_union |= np.where(interior, inside_distance, outside_distance) <= unknown_band_width
    return interior_union, band_union


def _coverage(state: np.ndarray, solid: np.ndarray) -> dict[str, Any]:
    """按 fluid 分母计算三态 coverage，同时单列 solid 数。"""
    fluid = ~solid
    fluid_cells = int(fluid.sum())
    pos = int(np.count_nonzero((state == POSITIVE) & fluid))
    neg = int(np.count_nonzero((state == NEGATIVE) & fluid))
    unk = int(np.count_nonzero((state == UNKNOWN) & fluid))
    total_unknown = int(np.count_nonzero(state == UNKNOWN))
    denom = float(fluid_cells) if fluid_cells else 1.0
    return {
        "solid_cells": int(solid.sum()),
        "fluid_cells": fluid_cells,
        "positive_cells": pos,
        "negative_cells": neg,
        "unknown_cells": unk,
        "total_unknown_cells_including_solid": total_unknown,
        "known_cells": pos + neg,
        "positive_fraction_fluid": float(pos / denom),
        "negative_fraction_fluid": float(neg / denom),
        "unknown_fraction_fluid": float(unk / denom),
        "known_fraction_fluid": float((pos + neg) / denom),
    }


def _failure_state(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """失败时所有格均 unknown，confidence 为 0。"""
    return (
        np.full(shape, UNKNOWN, dtype=np.int8),
        np.zeros(shape, dtype=np.float32),
    )


# --------------------------------------------------------------------------- 主公共 seam

def extract_haller_anchors(
    u: Any,
    v: Any,
    xdim: Any,
    ydim: Any,
    mask: Any = None,
    *,
    source: str = SOURCE_TRAIN,
    frame_index: int = 0,
    backend: str = BACKEND_NUMPY,
    device: Any = None,
    contour_mode: str = CONTOUR_MODE_OPTIMIZED,
    fast_global_level_count: int = 64,
    fast_refinement_iterations: int = 7,
    fast_refinement_halo_cells: int = 2,
) -> dict[str, Any]:
    """从一帧速度场生成 standard IVD、闭合 contour 和三态 Haller artifact。

    ``source`` 必须显式属于 ``haller_anchor_train``、``haller_gt_calibration``
    或 ``haller_gt_test``。训练失败采用整帧 fluid unknown；calibration/test
    失败采用 invalid frame。任何失败都不会调用 legacy p85 fallback。

    返回 dict 的公共字段包括 ``omega``、``standard_ivd``、``peaks``、
    ``contours``、``anchor_state``、``anchor_confidence``、``haller_gt``、
    ``valid``、``failure_count`` 和 ``metadata``。
    """
    source = _validate_source(source)
    contour_mode = _normalize_contour_mode(contour_mode)
    try:
        frame_index = int(frame_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame_index 必须是整数，实际 {frame_index!r}") from exc
    if frame_index < 0:
        raise ValueError("frame_index 不能为负")
    # Formal artifacts use one frozen engineering parameter set.  Sensitivity
    # variants, if needed later, must be a separately named diagnostic path.
    params = _resolve_parameters()
    backend_details = _resolve_backend(backend, device)
    contour_mode = _contour_mode_for_backend(
        backend_details.resolved, contour_mode
    )
    if backend_details.resolved == BACKEND_FAST_HALLER:
        try:
            fast_global_level_count = int(fast_global_level_count)
            fast_refinement_iterations = int(fast_refinement_iterations)
            fast_refinement_halo_cells = int(fast_refinement_halo_cells)
        except (TypeError, ValueError) as exc:
            raise ValueError("fast_haller 参数必须是整数") from exc
        if fast_global_level_count < 2:
            raise ValueError("fast_haller fast_global_level_count 必须至少为 2")
        if fast_refinement_iterations < 0:
            raise ValueError("fast_haller fast_refinement_iterations 不能为负")
        if fast_refinement_halo_cells < 0:
            raise ValueError("fast_haller fast_refinement_halo_cells 不能为负")
        params = dict(params)
        params.update({
            "fast_global_level_count": fast_global_level_count,
            "fast_refinement_iterations": fast_refinement_iterations,
            "fast_refinement_halo_cells": fast_refinement_halo_cells,
            "fast_level_schedule": (
                "geomspace(min_peak*contour_level_end,max_peak*contour_level_start)"
            ),
            "fast_refinement": "outer_invalid_to_valid_bisection",
        })
    elif backend_details.resolved == BACKEND_NUMBACS:
        from numbacs_haller import (
            UPSTREAM_DEFAULT_END_LEVEL,
            UPSTREAM_DEFAULT_MIN_VAL,
            UPSTREAM_DEFAULT_NLEVS,
            UPSTREAM_DEFAULT_START_LEVEL,
            runtime_mode as numbacs_runtime_mode,
        )

        params = dict(params)
        params.update({
            "numbacs_method": "numbacs.extraction.elliptic.rotcohvrt",
            "numbacs_upstream_version": "0.2.0",
            "numbacs_upstream_commit": "c067f542543f5dd4ae3dc45fc506213e8d98b845",
            "numbacs_runtime_mode": numbacs_runtime_mode(),
            # These are the official defaults used by the formal native call.
            # Keep them backend-scoped because the generic Haller reference
            # parameters above intentionally describe a different 32-level
            # engineering extractor.
            "numbacs_contour_level_count": UPSTREAM_DEFAULT_NLEVS,
            "numbacs_nlevs_passed_explicitly": False,
            "numbacs_default_min_val": UPSTREAM_DEFAULT_MIN_VAL,
            "numbacs_default_start_level": UPSTREAM_DEFAULT_START_LEVEL,
            "numbacs_default_end_level": UPSTREAM_DEFAULT_END_LEVEL,
            "numbacs_explicit_kwargs": [
                "convexity_deficiency",
                "min_len",
            ],
            "numbacs_omitted_kwargs": [
                "min_val",
                "nlevs",
                "start_level",
                "end_level",
            ],
            "numbacs_schedule_start": "upstream_default_percentile70",
            "numbacs_schedule_end": "upstream_default_maximum",
            "numbacs_radius_factor": 1.0,
            "numbacs_convexity_method": "upstream_default_convex_hull",
            "numbacs_boundary": "upstream_convex_hull",
        })

    details = _compute_standard_ivd_details(
        u,
        v,
        xdim,
        ydim,
        mask,
        backend=backend_details.resolved,
        device=backend_details.device,
    )
    solid = details.solid_mask
    fluid = ~solid
    x = np.asarray(xdim, dtype=np.float64)
    y = np.asarray(ydim, dtype=np.float64)
    peaks = find_local_maxima(
        details.ivd,
        solid,
        backend=backend_details.resolved,
        device=backend_details.device,
    )
    contours, contour_diagnostics, contour_evaluation = _select_contours(
        details.ivd,
        solid,
        x,
        y,
        peaks,
        params,
        details.dx,
        details.dy,
        backend=backend_details.resolved,
        device=backend_details.device,
        contour_mode=contour_mode,
        fast_global_level_count=fast_global_level_count,
        fast_refinement_iterations=fast_refinement_iterations,
        fast_refinement_halo_cells=fast_refinement_halo_cells,
        return_evaluation=True,
    )

    failure_reasons: list[str] = []
    if not peaks:
        failure_reasons.append("no_local_maximum")
    if not contours:
        failure_reasons.append("no_legal_contour")
    failed = not contours
    if failed:
        state, confidence = _failure_state(details.ivd.shape)
        frame_valid = source == SOURCE_TRAIN
        failure_count = 1
        frame_p60 = float(np.percentile(details.ivd[fluid], float(params["negative_percentile"])))
        interior_union = np.zeros(details.ivd.shape, dtype=bool)
        band_union = np.zeros(details.ivd.shape, dtype=bool)
    else:
        unknown_band_width = float(params["unknown_band_factor"]) * max(details.dx, details.dy)
        interior_union, band_union = _rasterize_union(
            contours, details.ivd.shape, details.dx, details.dy, unknown_band_width
        )
        frame_p60 = float(np.percentile(details.ivd[fluid], float(params["negative_percentile"])))
        state = np.full(details.ivd.shape, UNKNOWN, dtype=np.int8)
        outside_known = fluid & ~interior_union & ~band_union
        state[outside_known & (details.ivd <= frame_p60)] = NEGATIVE
        state[fluid & interior_union & ~band_union] = POSITIVE
        # Unknown band has precedence over positive/negative. Solid was unknown
        # from initialization and is never written by either known assignment.
        state[solid] = UNKNOWN
        confidence = np.where(state == UNKNOWN, 0.0, 1.0).astype(np.float32)
        frame_valid = True
        failure_count = 0

    metadata: dict[str, Any] = {
        "artifact_type": "haller_ivd_three_state",
        "algorithm_version": ALGORITHM_VERSION,
        "artifact_id": f"haller_v1_{source}",
        "source": source,
        "label_source": source,
        "frame_index": frame_index,
        "literature": dict(_LITERATURE_METADATA),
        "state_encoding": {
            "positive": int(POSITIVE),
            "negative": int(NEGATIVE),
            "unknown": int(UNKNOWN),
        },
        "parameters": params,
        "parameter_hash": _json_hash(params),
        "backend_requested": backend_details.requested,
        "resolved": backend_details.resolved,
        "backend": backend_details.resolved,
        "device": backend_details.device,
        "cuda_used": backend_details.cuda_used,
        "backend_version": backend_details.backend_version,
        "compute_dtype": backend_details.compute_dtype,
        "backend_fallback_reason": backend_details.fallback_reason,
        "contour_backend": (
            "contourpy_serial_global_levels_coarse_bisection_roi"
            if backend_details.resolved == BACKEND_FAST_HALLER
            else "numbacs.extraction.elliptic.rotcohvrt"
            if backend_details.resolved == BACKEND_NUMBACS
            else (
                "skimage_peak_local_roi_exact"
                if contour_mode == CONTOUR_MODE_OPTIMIZED
                else "skimage_reference_per_peak_level"
            )
        ),
        "contour_mode": contour_mode,
        "contour_hierarchy": (
            "fast_haller_global_shared_levels_outer_invalid_bisection"
            if backend_details.resolved == BACKEND_FAST_HALLER
            else "upstream_numbacs_rotcohvrt_global_levels"
            if backend_details.resolved == BACKEND_NUMBACS
            else (
                "peak_local_exact_absolute_level_paths"
                if contour_mode == CONTOUR_MODE_OPTIMIZED
                else "reference_nested_peak_level_search"
            )
        ),
        "contour_evaluation": {
            **contour_evaluation,
        },
        "spacing": {
            "dx": details.dx,
            "dy": details.dy,
            "max": max(details.dx, details.dy),
        },
        "shape": [int(vv) for vv in details.ivd.shape],
        "fluid_vorticity_mean": details.fluid_mean,
        "ivd_formula": "abs(omega - mean_fluid(omega))",
        "mean_scope": "entire_fluid_spatial_domain",
        "frame_fluid_p60": frame_p60,
        "input_hash": _hash_named_arrays({
            "u": np.asarray(u, dtype=np.float64),
            "v": np.asarray(v, dtype=np.float64),
            "xdim": x,
            "ydim": y,
        }),
        "mask_hash": _hash_array(solid),
        "solid_policy": "solid cells are always unknown; no positive or negative anchor",
        "frame_valid": bool(frame_valid),
        "valid": bool(not failed),
        "failure_policy": "train_unknown" if source == SOURCE_TRAIN else "invalid_frame",
        "failure_count": int(failure_count),
        "failure_reasons": failure_reasons,
        "fallback_used": None,
        "legacy_p85_used": False,
        "artifact_array_hashes": _artifact_array_hashes(
            state,
            state,
            confidence,
            details.ivd,
            details.omega,
            solid,
        ),
        "n_peaks": int(len(peaks)),
        "n_selected_contours": int(len(contours)),
        "n_contour_candidates": int(sum(
            int(item.get("candidate_count", 1)) for item in contour_diagnostics
        )),
        "n_contour_diagnostics": int(len(contour_diagnostics)),
        "n_rejected_contours": int(sum(
            int(item.get("candidate_count", 1))
            for item in contour_diagnostics
            if item.get("status") == "rejected"
        )),
        "selected_contours": contours,
        "contour_diagnostics": contour_diagnostics,
        "coverage": _coverage(state, solid),
    }

    result = {
        "omega": details.omega.astype(np.float64, copy=True),
        "standard_ivd": details.ivd.astype(np.float64, copy=True),
        "ivd": details.ivd.astype(np.float64, copy=True),
        "solid_mask": solid.astype(bool, copy=True),
        "fluid_mask": fluid.astype(bool, copy=True),
        "peaks": peaks,
        "contours": contours,
        "interior_mask": interior_union,
        "unknown_band_mask": band_union,
        "anchor_state": state,
        "haller_gt": state.copy(),
        "anchor_confidence": confidence,
        "valid": bool(not failed),
        "failure_count": int(failure_count),
        "metadata": metadata,
    }
    return result


generate_haller_anchors = extract_haller_anchors


# --------------------------------------------------------------------------- artifact 落盘和读取

def _fsync_directory(path: pathlib.Path) -> None:
    """Best-effort fsync for a directory after same-filesystem replacement."""
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_save_npy(path: pathlib.Path, array: np.ndarray) -> None:
    """Write one NumPy array through a same-directory temporary file."""
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _atomic_write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    """Publish metadata only after its complete UTF-8 JSON body is durable."""
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

def _result_array(result: Mapping[str, Any], key: str, dtype: Any, shape: tuple[int, int]) -> np.ndarray:
    if key not in result:
        raise ValueError(f"Haller result 缺少字段 {key!r}")
    arr = np.asarray(result[key], dtype=dtype)
    if arr.shape != shape:
        raise ValueError(f"Haller result {key} 形状应为 {shape}，实际 {arr.shape}")
    return arr


def save_haller_artifact(
    result: Mapping[str, Any], out_dir: str | pathlib.Path, *, overwrite: bool = True
) -> dict[str, str]:
    """保存 Haller 三态 artifact，并拒绝不同 source 的目录覆盖。

    ``out_dir`` 本身代表一个 source-specific artifact 目录；调用方应将
    train/calibration/test 分别放入不同目录。若目标已有 metadata，source
    不同会无条件 fail loud，即使 ``overwrite=True``。
    """
    if "metadata" not in result or not isinstance(result["metadata"], Mapping):
        raise ValueError("Haller result 缺少 metadata")
    metadata = dict(result["metadata"])
    source = _validate_source(str(metadata.get("source", "")))
    shape_value = metadata.get("shape")
    if not isinstance(shape_value, (list, tuple)) or len(shape_value) != 2:
        raise ValueError("Haller metadata 缺少二维 shape")
    shape = (int(shape_value[0]), int(shape_value[1]))
    state = _result_array(result, "anchor_state", np.int8, shape)
    gt = _result_array(result, "haller_gt", np.int8, shape)
    confidence = _result_array(result, "anchor_confidence", np.float32, shape)
    ivd = _result_array(result, "standard_ivd", np.float32, shape)
    omega = _result_array(result, "omega", np.float32, shape)
    if not np.array_equal(state, gt):
        raise ValueError("haller_gt 与 anchor_state 不一致，拒绝保存歧义 artifact")
    if np.any(~np.isfinite(confidence)) or np.any(~np.isfinite(ivd)) or np.any(~np.isfinite(omega)):
        raise ValueError("Haller artifact 数组包含非有限值")
    solid_array = None
    if "solid_mask" in result:
        solid_array = _result_array(result, "solid_mask", np.uint8, shape)
    artifact_array_hashes = _artifact_array_hashes(
        gt, state, confidence, ivd, omega, solid_array
    )
    declared_array_hashes = metadata.get("artifact_array_hashes")
    if declared_array_hashes is not None and dict(declared_array_hashes) != artifact_array_hashes:
        raise ValueError("Haller artifact 数组 hash 与 metadata 不一致，拒绝保存")
    metadata["artifact_array_hashes"] = artifact_array_hashes

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "anchor_meta.json"
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if not overwrite:
                raise ValueError(
                    f"目标 artifact metadata 无法读取，overwrite=False 拒绝覆盖：{meta_path}"
                ) from exc
            # A malformed metadata file is treated as an interrupted write.
            # The caller's source-specific directory and the complete result
            # supplied here provide the explicit repair boundary.
            existing = None
        if existing is not None and not isinstance(existing, Mapping):
            if not overwrite:
                raise ValueError(f"目标 artifact metadata 不是 object：{meta_path}")
            existing = None
        if existing is not None:
            existing_source = existing.get("source")
            if existing_source != source:
                raise ValueError(
                    f"artifact source 不允许覆盖：已有 {existing_source!r}，新建 {source!r}"
                )
            if not overwrite:
                raise FileExistsError(f"artifact 已存在且 overwrite=False：{out}")
    else:
        known_files = [out / name for name in (
            "haller_gt.npy", "anchor_state.npy", "anchor_confidence.npy",
            "standard_ivd.npy", "omega.npy",
        )]
        if any(path.exists() for path in known_files) and not overwrite:
            raise ValueError("目标目录已有 Haller 数组但缺少 metadata，拒绝盲目覆盖")

    _atomic_save_npy(out / "haller_gt.npy", gt)
    _atomic_save_npy(out / "anchor_state.npy", state)
    _atomic_save_npy(out / "anchor_confidence.npy", confidence)
    _atomic_save_npy(out / "standard_ivd.npy", ivd)
    _atomic_save_npy(out / "omega.npy", omega)
    # 作为 artifact 内部审计副本保存 geometry mask；不进入模型输入。
    if solid_array is not None:
        _atomic_save_npy(out / "solid_mask.npy", solid_array)
    elif (out / "solid_mask.npy").exists():
        (out / "solid_mask.npy").unlink()
        _fsync_directory(out)
    # Metadata is published last.  A crash leaves either the old complete
    # metadata or a recoverable partial directory; the resume loader validates
    # hashes and recomputes before accepting it.
    _atomic_write_json(meta_path, metadata)
    return {
        "haller_gt": str(out / "haller_gt.npy"),
        "anchor_state": str(out / "anchor_state.npy"),
        "anchor_confidence": str(out / "anchor_confidence.npy"),
        "standard_ivd": str(out / "standard_ivd.npy"),
        "omega": str(out / "omega.npy"),
        "metadata": str(meta_path),
    }


def load_haller_artifact(
    out_dir: str | pathlib.Path, *, expected_source: str | None = None
) -> dict[str, Any]:
    """读取并校验 artifact；test source 必须显式传 expected_source。"""
    out = pathlib.Path(out_dir)
    meta_path = out / "anchor_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"缺少 Haller artifact metadata：{meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("Haller artifact metadata 必须是 JSON object")
    source = _validate_source(str(metadata.get("source", "")))
    if expected_source is not None:
        _validate_source(expected_source)
        if expected_source != source:
            raise ValueError(
                f"expected_source={expected_source!r} 与 artifact source={source!r} 不一致"
            )
    elif source == SOURCE_TEST:
        raise ValueError(
            "haller_gt_test 只能通过显式 expected_source=SOURCE_TEST 读取，禁止默认 loader 混淆"
        )
    shape_value = metadata.get("shape")
    if not isinstance(shape_value, (list, tuple)) or len(shape_value) != 2:
        raise ValueError("Haller artifact metadata 缺少二维 shape")
    shape = (int(shape_value[0]), int(shape_value[1]))
    arrays = {
        "haller_gt": np.load(out / "haller_gt.npy", allow_pickle=False),
        "anchor_state": np.load(out / "anchor_state.npy", allow_pickle=False),
        "anchor_confidence": np.load(out / "anchor_confidence.npy", allow_pickle=False),
        "standard_ivd": np.load(out / "standard_ivd.npy", allow_pickle=False),
        "omega": np.load(out / "omega.npy", allow_pickle=False),
    }
    expected_dtypes = {
        "haller_gt": np.dtype(np.int8),
        "anchor_state": np.dtype(np.int8),
        "anchor_confidence": np.dtype(np.float32),
        "standard_ivd": np.dtype(np.float32),
        "omega": np.dtype(np.float32),
    }
    for key, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"artifact {key} 形状 {array.shape} 与 metadata {shape} 不符")
        if array.dtype != expected_dtypes[key]:
            raise ValueError(
                f"artifact {key} dtype {array.dtype} 与保存契约 {expected_dtypes[key]} 不符"
            )
    if not np.array_equal(arrays["haller_gt"], arrays["anchor_state"]):
        raise ValueError("artifact haller_gt 与 anchor_state 不一致")
    if not np.all(np.isin(arrays["anchor_state"], [UNKNOWN, NEGATIVE, POSITIVE])):
        raise ValueError("artifact anchor_state 含有未知三态编码")
    if (np.any(~np.isfinite(arrays["anchor_confidence"]))
            or np.any(~np.isfinite(arrays["standard_ivd"]))
            or np.any(~np.isfinite(arrays["omega"]))):
        raise ValueError("artifact 数组包含非有限值")
    parameters = metadata.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Haller metadata 缺少 parameters object")
    if metadata.get("parameter_hash") != _json_hash(parameters):
        raise ValueError("Haller metadata parameter_hash 校验失败")
    expected_array_hashes = metadata.get("artifact_array_hashes")
    if not isinstance(expected_array_hashes, Mapping):
        raise ValueError("Haller metadata 缺少 artifact_array_hashes")
    actual_array_hashes = _artifact_array_hashes(
        arrays["haller_gt"],
        arrays["anchor_state"],
        arrays["anchor_confidence"],
        arrays["standard_ivd"],
        arrays["omega"],
    )
    solid_path = out / "solid_mask.npy"
    solid = None
    if solid_path.exists():
        solid = np.load(solid_path, allow_pickle=False)
        if solid.shape != shape:
            raise ValueError(f"artifact solid_mask 形状 {solid.shape} 与 metadata {shape} 不符")
        if solid.dtype != np.dtype(np.uint8):
            raise ValueError(f"artifact solid_mask dtype {solid.dtype} 与保存契约 uint8 不符")
        if not np.all(np.isin(solid, [0, 1])):
            raise ValueError("artifact solid_mask 必须是 0/1")
        actual_array_hashes["solid_mask"] = _hash_array(solid)
    if dict(expected_array_hashes) != actual_array_hashes:
        raise ValueError("Haller artifact 数组 hash 校验失败")
    result: dict[str, Any] = {**arrays, "metadata": metadata}
    if solid is not None:
        result["solid_mask"] = solid.astype(bool)
    return result


def generate_haller_artifact(
    u: Any,
    v: Any,
    xdim: Any,
    ydim: Any,
    mask: Any = None,
    *,
    out_dir: str | pathlib.Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """提取并可选保存 source-specific Haller artifact 的便利入口。"""
    result = extract_haller_anchors(u, v, xdim, ydim, mask, **kwargs)
    if out_dir is not None:
        result["artifact_paths"] = save_haller_artifact(result, out_dir)
    return result


# --------------------------------------------------------------------------- 轻量 CLI（真实 train frame preview 用）

def main(argv: list[str] | None = None) -> int:
    """从 h5py 可读 nc 读取一帧，生成 source-specific artifact。"""
    parser = argparse.ArgumentParser(description="单帧 Haller-IVD 三态 anchor 生成器")
    parser.add_argument("nc_path", help="h5py 可读的 nc 数据集路径")
    parser.add_argument("--frame", type=int, default=0, help="单帧索引")
    parser.add_argument("--mask", default=None, help="可选 geometry mask.npy（2D 或单帧 3D）")
    parser.add_argument("--out-dir", required=True, help="source-specific artifact 输出目录")
    parser.add_argument("--source", choices=sorted(VALID_SOURCES), default=SOURCE_TRAIN)
    parser.add_argument("--backend", choices=sorted(VALID_BACKENDS), default=BACKEND_NUMPY)
    parser.add_argument("--device", default=None, help="CPU device；只接受 cpu")
    parser.add_argument(
        "--contour-mode",
        choices=sorted(VALID_CONTOUR_MODES),
        default=CONTOUR_MODE_OPTIMIZED,
        help="optimized 使用全局 exact-level contour cache；reference 用于逐涡对照",
    )
    args = parser.parse_args(argv)

    import geometry

    u, v, xdim, ydim, _ = geometry.load_field(args.nc_path)
    if not (0 <= args.frame < u.shape[0]):
        raise ValueError(f"frame={args.frame} 越界，数据集时间长度为 {u.shape[0]}")
    mask = None if args.mask is None else np.load(args.mask, allow_pickle=False)
    if mask is not None and mask.ndim == 3:
        if mask.shape[0] != u.shape[0]:
            raise ValueError("3D geometry mask 时间长度必须与 nc 一致")
        mask = mask[args.frame]
    result = generate_haller_artifact(
        u[args.frame], v[args.frame], xdim, ydim, mask,
        out_dir=args.out_dir, source=args.source, frame_index=args.frame,
        backend=args.backend, device=args.device, contour_mode=args.contour_mode,
    )
    print(json.dumps({
        "source": result["metadata"]["source"],
        "valid": result["valid"],
        "failure_count": result["failure_count"],
        "coverage": result["metadata"]["coverage"],
        "artifact_paths": result["artifact_paths"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
