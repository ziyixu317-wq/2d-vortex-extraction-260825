"""Mixed-source W1-P sampling and adapters, without a training CLI.

MFC references must already contain full-frame shared standard IVD and five-state
P95/P80 labels. This module reads them; it never recomputes an ROI vorticity mean.
The existing six-source runner and its artifact validators remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import dataset
import extractor
from mfc_pathline_source import MFCPathlineSource
import standard_ivd
import ticket04_real_training as legacy
import w1_p
import weak_labels
import weak_supervision_contract as contract


@dataclass(frozen=True)
class ExtractionConfig:
    patch_size: tuple[int, int] = (32, 32)
    stride: tuple[int, int] = (16, 16)
    t_win: int = 24
    window_step: int = 4
    groups: tuple[int, int] = (8, 8)
    delta_frac: float = 0.05
    L: int = 16
    n_substeps: int = 4
    t_scale: float = 0.25


class MFCReferenceCache:
    """Explicit new-source cache input; no historical Ticket03 impersonation.

    Required files: cache_metadata.json, standard_ivd.npy, state.npy,
    valid_fluid_mask.npy, invalid_mask.npy. All arrays are full-source TYX,
    indexed by global saved-frame number. State codes are 1/0/-1/-2/-3.
    """

    def __init__(self, root: str | Path, source: MFCPathlineSource):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "cache_metadata.json").read_text(encoding="utf-8"))
        meta = self.metadata
        if meta["source_name"] != source.source_name:
            raise ValueError("MFC cache source_name differs from physical source")
        if meta["frame_ranges"] != {k: list(v) for k, v in source.frame_ranges.items()}:
            raise ValueError("MFC cache frame ranges differ from physical source")
        if (meta["standard_ivd_source_identity"] != standard_ivd.SOURCE_IDENTITY
                or meta["ivd_mean_scope"] != "full_frame_valid_fluid"):
            raise ValueError("MFC cache must use full-frame shared standard IVD")
        self.normalization = dict(meta["normalization"])
        train_range = list(source.frame_ranges["train"])
        if (self.normalization["source_split"] != "train"
                or self.normalization["train_frame_range"] != train_range
                or meta["thresholds"]["statistics_split"] != "train"
                or meta["thresholds"]["train_frame_range"] != train_range):
            raise ValueError("MFC normalization and thresholds must use the declared train range")
        mu, sigma, speed = (float(self.normalization[k]) for k in ("ivd_mu", "ivd_sigma", "speed_max"))
        if not np.isfinite([mu, sigma, speed]).all() or sigma < 0 or speed <= 0:
            raise ValueError("MFC normalization requires finite statistics and positive speed")
        p80, p95 = (float(meta["thresholds"][k]) for k in ("p80", "p95"))
        if not np.isfinite([p80, p95]).all() or p80 >= p95:
            raise ValueError("MFC thresholds require finite P80 < P95")
        for name in ("standard_ivd", "state", "valid_fluid_mask", "invalid_mask"):
            value = np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if value.shape != source.shape:
                raise ValueError(f"MFC cache {name} must have full-source shape {source.shape}")
            setattr(self, name, value)
        if self.state.dtype != np.int8:
            raise ValueError("MFC cache state must be int8 five-state reference")
        self.prepared_frames = frozenset(int(i) for i in meta["prepared_frame_indices"])


def build_epoch_specs(
    sources: Sequence[Any], *, samples_per_epoch: int = 20_000,
    batch_size: int = 32, positive_fraction: float = 0.5,
    seed: int = 0, epoch: int = 0,
) -> list[dict[str, Any]]:
    """Seven equal batch sources, using the original seed/epoch RNG meaning.

    A sample is a patch containing 256 trajectories; positive_fraction concerns
    patch pools, not the proportion of positive trajectory labels. Sources are
    ordered with their existing six ids followed by MFC id 6. The epoch start
    advances by the batch-count remainder, distributing extra batches across
    sources over successive epochs (625 updates: start = epoch * 2 modulo 7).
    """
    if tuple(s.source_id for s in sources) != tuple(range(7)):
        raise ValueError("Mixed W1-P expects six ordered legacy sources and MFC id 6")
    if samples_per_epoch <= 0 or batch_size <= 0 or not 0 < positive_fraction < 1:
        raise ValueError("Use positive sample/batch budgets and 0 < positive_fraction < 1")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    specs, remaining, index = [], int(samples_per_epoch), 0
    source_offset = (int(epoch) * (math.ceil(samples_per_epoch / batch_size) % 7)) % 7
    while remaining:
        size = min(int(batch_size), remaining)
        source_id = (source_offset + index) % 7
        source = sources[source_id]
        frame = int(rng.choice(source.batch_frame_choices))
        n_positive = max(1, min(size - 1, int(round(size * positive_fraction))))
        pos = source.positive_pool_by_frame[frame]
        neg = source.negative_pool_by_frame[frame]
        items = [pos[int(i)] for i in rng.integers(0, len(pos), size=n_positive)]
        items.extend(neg[int(i)] for i in rng.integers(0, len(neg), size=size - n_positive))
        rng.shuffle(items)
        specs.append({"dataset_index": source_id, "frame_start": frame, "items": tuple(items)})
        remaining -= size
        index += 1
    return specs


class LegacySourceAdapter:
    """Wrap a validated old DatasetView and retain its actual artifact batch."""

    def __init__(self, view: legacy.DatasetView, *, config: ExtractionConfig = ExtractionConfig()):
        self.view, self.config = view, config
        self.source_id = view.ds_id
        self.name = view.name
        self.normalization = view.normalization
        self.batch_frame_choices = view.batch_frame_choices
        self.positive_pool_by_frame = view.positive_pool_by_frame
        self.negative_pool_by_frame = view.negative_pool_by_frame

    def make_batch(self, spec: dict[str, Any], *, seed: int = 0) -> w1_p.W1PBatch:
        if spec["dataset_index"] != self.source_id:
            raise ValueError("Legacy source/spec id mismatch")
        cfg = self.config
        pathlines, cells = zip(*legacy._extract_normalized_batch(
            self.view, spec["items"], patch_size=cfg.patch_size, t_win=cfg.t_win,
            groups=cfg.groups, delta_frac=cfg.delta_frac, L=cfg.L,
            n_substeps=cfg.n_substeps, t_scale=cfg.t_scale, seed=seed,
        ))
        payload = {
            "dataset_index": self.source_id, "frame_start": spec["frame_start"],
            "frame_indices": np.full(len(pathlines), spec["frame_start"], dtype=np.int64),
            "pathlines": np.stack(pathlines), "seed_cells": np.stack(cells),
        }
        return legacy._make_contract_batch(
            payload, self.view, mode=contract.MODE_W1_P,
            t_win=cfg.t_win, window_step=cfg.window_step,
            anchor_metadata=None, anchor_hash=None,
        )


class MFCSourceAdapter:
    """MFC id 6, core coordinates, train-derived halo, and precomputed references."""

    source_id = 6
    name = "mfc_airfoil"

    def __init__(self, source: MFCPathlineSource, cache: MFCReferenceCache,
                 *, config: ExtractionConfig = ExtractionConfig()):
        self.source, self.cache, self.config = source, cache, config
        self.normalization = cache.normalization
        self.positive_pool_by_frame = {}
        self.negative_pool_by_frame = {}
        self.batch_frame_choices = ()

    def prepare_sampling_pools(self) -> None:
        """Build original positive/negative patch pools; no source-area weighting."""
        cfg, src = self.config, self.source
        patches = dataset.patch_locations(src.shape[1], src.shape[2], cfg.patch_size, cfg.stride)
        usable = [i for i, patch in enumerate(patches) if legacy._patch_is_usable(
            src.geometry, src.xdim, src.ydim, patch, patch_size=cfg.patch_size,
            groups=cfg.groups, delta_frac=cfg.delta_frac,
        )]
        for frame in src.window_starts("train", t_win=cfg.t_win, step=cfg.window_step):
            if not set(range(frame, frame + cfg.t_win)).issubset(self.cache.prepared_frames):
                continue
            state = np.asarray(self.cache.state[frame])
            if not np.isin(state, [1, 0, -1, -2, -3]).all():
                raise ValueError("MFC reference contains an unrecognized state code")
            positive = weak_labels.patch_positive_map(
                state == 1, src.xdim, src.ydim, cfg.patch_size, cfg.stride,
                cfg.groups, cfg.delta_frac,
            ).ravel()
            self.positive_pool_by_frame[frame] = tuple((*patches[i], frame) for i in usable if positive[i])
            self.negative_pool_by_frame[frame] = tuple((*patches[i], frame) for i in usable if not positive[i])
        self.batch_frame_choices = tuple(frame for frame in self.positive_pool_by_frame
                                        if self.positive_pool_by_frame[frame]
                                        and self.negative_pool_by_frame[frame])
        if not self.batch_frame_choices:
            raise ValueError("MFC has no complete train window with both patch pools")

    def halo_bounds(self, item: tuple[int, int, int], *, split: str = "train") -> tuple[tuple[int, int], tuple[int, int]]:
        py, px, frame = map(int, item)
        cfg, src = self.config, self.source
        start, end = src.frame_ranges[split]
        if not start <= frame < frame + cfg.t_win <= end:
            raise ValueError(f"MFC extraction window must lie entirely inside {split}")
        ph, pw = cfg.patch_size
        if not 0 <= py < py + ph <= src.shape[1] or not 0 <= px < px + pw <= src.shape[2]:
            raise ValueError("MFC core must lie inside the native source domain")
        duration = float(src.tdim[frame + cfg.t_win - 1] - src.tdim[frame])
        displacement = float(self.normalization["speed_max"]) * duration
        hy = math.ceil(displacement / float(src.ydim[1] - src.ydim[0])) + 2
        hx = math.ceil(displacement / float(src.xdim[1] - src.xdim[0])) + 2
        return ((max(0, py - hy), min(src.shape[1], py + ph + hy)),
                (max(0, px - hx), min(src.shape[2], px + pw + hx)))

    def extract_sample(self, item: tuple[int, int, int], *, seed: int = 0,
                       split: str = "train") -> dict[str, Any]:
        py, px, frame = map(int, item)
        cfg, src = self.config, self.source
        by, bx = self.halo_bounds(item, split=split)
        if not set(range(frame, frame + cfg.t_win)).issubset(self.cache.prepared_frames):
            raise ValueError("MFC reference cache does not cover the complete extraction window")
        window = src.read_region_window(split, frame, frame + cfg.t_win,
                                        y_bounds=by, x_bounds=bx)
        selection = (slice(frame, frame + cfg.t_win), slice(*by), slice(*bx))
        ivd = np.asarray(self.cache.standard_ivd[selection])
        support = (np.asarray(self.cache.valid_fluid_mask[selection], dtype=bool)
                   & window.valid_fluid_support)
        t_span = float(window.tdim[-1] - window.tdim[0])
        raw, seeds = extractor.extract_pathlines_batched(
            window.u, window.v, window.geometry, ivd,
            window.xdim, window.ydim, window.tdim,
            patch_yx=(py - by[0], px - bx[0]), patch_size=cfg.patch_size,
            t0=float(window.tdim[0]), L=cfg.L, groups=cfg.groups,
            delta_frac=cfg.delta_frac, t_win_frames=cfg.t_win, n_substeps=cfg.n_substeps,
            dt_out=t_span / (cfg.L - 1),
            rng=dataset._comb_rng_base(seed, py, px, frame, ds_id=self.source_id),
            return_seeds=True, retain_initial_failures=True,
        )
        geo = extractor.patch_geometry((py, px), cfg.patch_size, src.xdim, src.ydim)
        pos = np.stack((raw[:, :, extractor.CH_PX] * geo["hx"] + geo["cx"],
                        raw[:, :, extractor.CH_PY] * geo["hy"] + geo["cy"]), axis=-1)
        sampled_support = extractor.interp_path(
            support.astype(np.float32), pos.reshape(-1, 2), raw[:, :, extractor.CH_T].ravel(),
            window.xdim, window.ydim, window.tdim,
        ).reshape(cfg.L, -1)
        failed = (~np.isfinite(raw).all(axis=(0, 2))
                  | ~np.isclose(raw[-1, :, extractor.CH_T], window.tdim[-1], rtol=0, atol=1e-6)
                  | np.any(sampled_support < 1 - 1e-6, axis=0))
        paths = dataset.normalize_pathlines(
            raw, seeds, geo, float(window.tdim[0]), t_span,
            cfg.t_scale, float(self.normalization["ivd_mu"]),
            float(self.normalization["ivd_sigma"]), float(self.normalization["speed_max"]),
        )
        if not np.isfinite(paths).all():
            raise ValueError("MFC extraction produced nonfinite model inputs")
        j, i = extractor.nearest_cell(seeds[:, 0], seeds[:, 1], src.xdim, src.ydim)
        cells = np.stack((j, i), axis=1).astype(np.int64)
        state = np.asarray(self.cache.state[frame, j, i])
        solid = src.geometry[j, i] | (state == -2)
        native_invalid = (state == -3) | np.asarray(self.cache.invalid_mask[frame, j, i], dtype=bool)
        known = np.isin(state, [0, 1]) & ~solid & ~native_invalid & ~failed
        return {"pathlines": paths, "seed_cells": cells, "labels": (state == 1).astype(np.float32),
                "known": known, "unknown": ~known, "solid": solid,
                "native_invalid": native_invalid, "failed": failed,
                "read_bounds_yx": [list(by), list(bx)]}

    def make_batch(self, spec: dict[str, Any], *, seed: int = 0) -> w1_p.W1PBatch:
        if spec["dataset_index"] != self.source_id:
            raise ValueError("MFC source/spec id mismatch")
        if any(int(item[2]) != spec["frame_start"] for item in spec["items"]):
            raise ValueError("MFC batch must have one exact frame window")
        samples = [self.extract_sample(item, seed=seed) for item in spec["items"]]
        fields = {key: np.stack([row[key] for row in samples]) for key in
                  ("pathlines", "labels", "known", "unknown", "solid", "native_invalid", "failed")}
        provenance = {
            "dataset_name": self.name, "source_id": self.source_id,
            "frame_start": spec["frame_start"], "t_win": self.config.t_win,
            "frame_ranges": {k: list(v) for k, v in self.source.frame_ranges.items()},
            "normalization": self.normalization, "thresholds": self.cache.metadata["thresholds"],
            "standard_ivd_source_identity": standard_ivd.SOURCE_IDENTITY,
            "ivd_mean_scope": "full_frame_valid_fluid",
            "reference_cache": str(self.cache.root),
            "failed_pathline_mask": fields["failed"].tolist(),
            "native_invalid_mask": fields["native_invalid"].tolist(),
            "read_bounds_yx": [row["read_bounds_yx"] for row in samples],
        }
        return w1_p.build_w1_p_batch(
            fields["pathlines"], fields["labels"], fields["known"], fields["unknown"], fields["solid"],
            # Solid is its own ignored loss category. Failed/native-invalid
            # observations remain in provenance, including observations on
            # solid seeds; invalid support is the non-solid remainder.
            invalid_mask=(fields["native_invalid"] | fields["failed"]) & ~fields["solid"],
            sampling_source=contract.LABEL_SOURCE_P95P80, split_name="train", provenance=provenance,
        )
