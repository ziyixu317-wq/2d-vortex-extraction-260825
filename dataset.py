"""弱标签迹线数据集（dataset.py）——05 票：数据集类。

领域词汇（HANDOFF §4/§6，唯一权威）：
- 时间划分 train [0,10]s / val (10,12.5] / test (12.5,15]：帧 0-1000 / 1000-1250 /
  1250-1500（闭包口径 weak_labels.DEFAULT_SLICES，无时间泄漏）；
- patch 32×32 stride 16、窗口 T_win=24 帧、窗口起点步长 4 帧；
- u,v 与预计算 IVD 存 memmap（IVD 一次算好；≈405MB @ float32）；
- 7 通道 = [px, py, t, ivd, distance(距种子), u, v]（extractor.N_CHANNELS 口径）；
  归一化：px,py → patch 内 [-1,1]（extractor 已做）；t → [0,1]×t_scale（默认 0.25）；
  ivd 标准化（z-score，μ/σ 取 train 片流体区，避免固体 0 值污染与时间泄漏）；
  distance 用归一化坐标（hypot 归一化重构）；u,v ÷ 全局最大速度；
- 返回 ((dummy_field, pathlines), labels) 匹配模型输入（PathlineTransformerV0 取
  data[1] = pathline_src (B, L, K, C)；dummy_field = zeros((1,1,1,1)) 参考口径）；
- 标签 = 重播种后种子格处 label_field 值（label_field 含 5×5 面积过滤与固体强制 0）；
- 每 epoch 40000 样本（下限 20000）、50% 正样本过采样（正样本 = patch 内存在
  ≥1 条涡迹线；池判据与 weak_labels.patch_positive_map 单公式共用）；
- 多数据集（票 07 延伸，HANDOFF §1 决策 8 落实）：MultiDatasetPathlineDataset
  合并多个 prepare_dataset 产物做采样（各数据集帧前 60% 训/后 40% 测的 frac
  划分；τ 与归一化逐数据集各自——跨数据集输入尺度一致化）。

性能说明（验收记录披露）：on-the-fly 提取 = extract_pathlines_batched（真实窗口
实测 ~35ms；服务器多进程 DataLoader 可隐藏大部分加载时间，不构成训练瓶颈；
本地 <5ms 预算未达成，用户已确认不纠结——仅要求能跑）。

实现约束：h5py 直读中文路径（prepare_dataset 的 nc_path 分支）；纯 python/numpy
（遵守 §2 依赖清单：torch、numpy、h5py、yaml、matplotlib、tqdm）。
"""

from __future__ import annotations

import json
import pathlib
import zlib

import numpy as np

import extractor
import weak_labels

# --------------------------------------------------------------------------- 默认参数（HANDOFF §6）

DEFAULT_PATCH_SIZE = (32, 32)
DEFAULT_STRIDE = (16, 16)
DEFAULT_T_WIN = 24
DEFAULT_WINDOW_STEP = 4
DEFAULT_SAMPLES_PER_EPOCH = 40000
# 每 epoch 样本数规格下限 20000（HANDOFF §6：默认 40000、下限 20000）——
# 训练配置的语义约束（训练脚本选用 ≥20000），不作为运行时钳制
#（合成/测试可用更小值快速迭代）。
DEFAULT_POSITIVE_FRACTION = 0.5
DEFAULT_T_SCALE = 0.25
DEFAULT_L = 16
DEFAULT_GROUPS = (8, 8)
DEFAULT_DELTA_FRAC = 0.05

# 存储文件名（prepare_dataset 与 WeakLabelPathlineDataset 共用，防路径漂移）
FN_U = "u.npy"
FN_V = "v.npy"
FN_IVD = "ivd.npy"
FN_LABEL = "label_field.npy"
FN_MASK = "mask.npy"
FN_META = "meta.json"


# --------------------------------------------------------------------------- 采样几何（时间划分 / patch 位置）

def window_starts(i0, i1, t_win=DEFAULT_T_WIN, step=DEFAULT_WINDOW_STEP):
    """时间片 [i0, i1) 内的窗口起点帧：s ∈ [i0, i1−t_win]（窗口完全在片内）。

    步长 step（HANDOFF §6：窗口起点步长 4 帧）。返回 np.ndarray 升序。
    """
    stop = i1 - t_win
    if stop < i0:
        return np.empty(0, dtype=np.intp)
    return np.arange(i0, stop + 1, step, dtype=np.intp)


def fraction_slices(T, train_frac=0.6, val_frac=0.0):
    """按帧比例的时间片划分（票 07 延伸：多数据集按时间 60/40，无 val 时仅 train/test）。

    DEFAULT_SLICES 为绝对秒数口径（仅适用 1501 帧/15s 的 pipedcylinder2d）；
    多数据集帧数/时长各异（512~2001 帧、t∈[0,20]，jung telziemniak 的 t
    从 1.107 起）→ 按帧比例划分才通用。返回 {name: (i0, i1)}：
    train [0, i1)、val [i1, i2)（val_frac>0 时）、test [i2, T)；累积取整
    （正数 floor = int()）→ 三片（或两片）全覆盖、无时间泄漏（与
    DEFAULT_SLICES 同闭包语义）。train_frac ∈ (0,1)（严格）、
    val_frac ∈ [0, 1−train_frac)（留出非空），违规 fail loud。
    """
    T, train_frac, val_frac = int(T), float(train_frac), float(val_frac)
    if T < 2:
        raise ValueError(f"T 过小无法划分: {T}")
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac 必须在 (0,1) 内，实际 {train_frac}")
    if val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError(f"val_frac 必须在 [0, 1−train_frac) 内，实际 {val_frac}")
    i1 = int(T * train_frac)
    i2 = int(T * (train_frac + val_frac))
    if i1 <= 0 or i2 < i1:
        raise ValueError(f"时间片划分过窄: train_end={i1} val_end={i2}")
    out = {"train": (0, i1)}
    if val_frac > 0:
        out["val"] = (i1, i2)
    out["test"] = (i2, T)
    return out


def patch_locations(H, W, patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE):
    """patch 起点格网格（y 外 x 内；与 weak_labels.patch_positive_map 的行列序一致）。

    返回 list[(y0, x0)]，全部满足 (y0 ≤ H−ph) 且 (x0 ≤ W−pw)。
    """
    ph, pw = patch_size
    sy, sx = stride
    return [(int(y0), int(x0))
            for y0 in range(0, H - ph + 1, sy)
            for x0 in range(0, W - pw + 1, sx)]


# --------------------------------------------------------------------------- 7 通道归一化

def normalize_pathlines(raw, seeds, geo, t0, t_span, t_scale, ivd_mu, ivd_sigma,
                        speed_max):
    """extractor 输出 → 全归一化样本 (L, K=256, 7) float32。

    口径（spec Implementation Decisions）：
    - CH_PX/CH_PY：保持 extractor 的 patch 内归一化 [-1,1]（可超界）；
    - CH_T：t → [0,1]×t_scale（t_span = 窗口物理时长 = (T_win−1)×dt）；
    - CH_IVD：(ivd − ivd_mu) / ivd_sigma（z-score；ivd_sigma≤0 时置 0 通道防御）；
    - CH_DIST：重算为归一化坐标下距（重播种后）种子的距离
      hypot(px − sx_n, py − sy_n)，sx_n/sy_n = (seed − center) / half；
    - CH_U/CH_V：÷ speed_max（全局最大速度）。
    raw: (L, K, 7) float32；seeds: (K, 2) 物理坐标（重播种后）；geo: patch_geometry。
    """
    raw = np.asarray(raw, dtype=np.float32)
    out = raw.copy()
    out[:, :, extractor.CH_T] = (
        (raw[:, :, extractor.CH_T] - float(t0)) / float(t_span) * float(t_scale))
    if ivd_sigma is not None and float(ivd_sigma) > 0:
        out[:, :, extractor.CH_IVD] = (
            (raw[:, :, extractor.CH_IVD] - float(ivd_mu)) / float(ivd_sigma))
    else:
        out[:, :, extractor.CH_IVD] = 0.0
    seed_px = (seeds[:, 0] - geo["cx"]) / geo["hx"]
    seed_py = (seeds[:, 1] - geo["cy"]) / geo["hy"]
    out[:, :, extractor.CH_DIST] = np.hypot(
        out[:, :, extractor.CH_PX] - seed_px[None, :],
        out[:, :, extractor.CH_PY] - seed_py[None, :])
    out[:, :, extractor.CH_U] = raw[:, :, extractor.CH_U] / float(speed_max)
    out[:, :, extractor.CH_V] = raw[:, :, extractor.CH_V] / float(speed_max)
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------- 元数据

def load_dataset_meta(data_root):
    """读取 data_root/meta.json → dict（含 shape/坐标/slices/taus/speed_max/IVD 统计）。"""
    root = pathlib.Path(data_root)
    meta_path = root / FN_META
    if not meta_path.exists():
        raise FileNotFoundError(f"数据集元数据缺失: {meta_path}（先运行 prepare_dataset）")
    return json.loads(meta_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 预计算（memmap 落盘）

def _mask_2d(mask, Y, X):
    """掩膜规格化为 (Y,X) bool：None → 空掩膜（无障碍物数据集路径不变）。"""
    if mask is None:
        return np.zeros((Y, X), dtype=bool)
    if isinstance(mask, (str, pathlib.Path)):
        m = np.load(str(mask))
    else:
        m = np.asarray(mask)
    m = m.astype(bool)
    if m.ndim == 3:
        m = m[0]
    if m.shape != (Y, X):
        raise ValueError(f"掩膜形状 {m.shape} ≠ (Y,X)={(Y, X)}")
    return m


def _fit_slices(slices, T):
    """时间片截断到数据集帧数（子集数据集时：i1 截断到 T；i0 ≥ T 的片剔除）。

    与 weak_labels CLI 的截断口径一致（覆盖全部帧的无泄漏划分）。
    """
    out = {}
    for name, (i0, i1) in slices.items():
        if i0 < T:
            out[name] = (int(i0), min(int(i1), T))
    if not out:
        raise ValueError("时间片全部超出数据集帧数")
    return out


def prepare_dataset(nc_path=None, out_dir="outputs/dataset", *,
                    u=None, v=None, xdim=None, ydim=None, tdim=None,
                    mask=None, ivd=None, labels=None, min_area=weak_labels.DEFAULT_MIN_AREA,
                    percentile=weak_labels.DEFAULT_PERCENTILE, taus=None, slices=None,
                    split_mode="abs", train_frac=0.6, val_frac=0.0,
                    speed_max=None, ivd_stats_slice="train", ivd_mu=None, ivd_sigma=None,
                    patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE,
                    t_win=DEFAULT_T_WIN, window_step=DEFAULT_WINDOW_STEP):
    """数据准备（memmap 预计算）：u/v/ivd/label/mask 落盘 + meta.json（返回 meta dict）。

    输入：nc_path（h5py 直读，中文路径可用；u/v 逐帧流式，不全量驻留）
    或内存数组 (u, v, xdim, ydim, tdim)（合成场/测试）。
    复用/覆盖：mask（None=无固体；(Y,X) 或 (T,Y,X) 数组/路径）、ivd（None=自算，
    数组/路径=复用票 04 产物）、labels（None=build_label_field）、taus（None=按
    percentile 在流体区逐时间片统计——弱标签口径）、speed_max（None=全场速度模最大）。

    归一化统计（写 meta）：ivd_mu/ivd_sigma = ivd_stats_slice 片内流体区 IVD 的
    均值/标准差（默认 train，避免 val/test 统计泄漏）；σ=0 时写 0（normalize 防除零）。
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 坐标与形状
    if nc_path is not None:
        import h5py
        with h5py.File(str(nc_path), "r") as f:
            xdim = f["xdim"][:].astype(np.float64)
            ydim = f["ydim"][:].astype(np.float64)
            tdim = f["tdim"][:].astype(np.float64)
            T = len(tdim)
    xdim = np.asarray(xdim, dtype=np.float64)
    ydim = np.asarray(ydim, dtype=np.float64)
    tdim = np.asarray(tdim, dtype=np.float64)
    T, Y, X = len(tdim), len(ydim), len(xdim)
    if u is not None:
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        if u.shape != (T, Y, X) or v.shape != (T, Y, X):
            raise ValueError(f"u/v 形状需为 (T,Y,X)={(T, Y, X)}，实际 {u.shape}")

    # ---- 掩膜（先规格化：流式 IVD 计算需要）
    mask2d = _mask_2d(mask, Y, X)
    np.save(out_dir / FN_MASK, mask2d.astype(np.uint8))

    # ---- u/v/IVD：nc 流式 或 内存数组
    if nc_path is not None:
        import h5py
        with h5py.File(str(nc_path), "r") as f:
            umm = np.lib.format.open_memmap(out_dir / FN_U, mode="w+",
                                            dtype=np.float32, shape=(T, Y, X))
            vmm = np.lib.format.open_memmap(out_dir / FN_V, mode="w+",
                                            dtype=np.float32, shape=(T, Y, X))
            reuse_ivd = ivd is not None and isinstance(ivd, (str, pathlib.Path))
            if not reuse_ivd:
                ivd_mm = np.lib.format.open_memmap(out_dir / FN_IVD, mode="w+",
                                                   dtype=np.float32, shape=(T, Y, X))
            sp = 0.0
            for t in range(T):
                ut = np.asarray(f["u"][t], dtype=np.float32)
                vt = np.asarray(f["v"][t], dtype=np.float32)
                umm[t] = ut
                vmm[t] = vt
                sp = max(sp, float(np.hypot(ut, vt).max()))
                if not reuse_ivd:
                    ivd_mm[t] = weak_labels.compute_ivd(
                        ut[None], vt[None], xdim, ydim, mask=mask2d).astype(np.float32)[0]
            umm.flush(); vmm.flush()
            del umm, vmm
            if not reuse_ivd:
                ivd_mm.flush()
                del ivd_mm
        speed_max = sp if speed_max is None else float(speed_max)
        ivd_source = "computed_streaming" if not reuse_ivd else "provided"
    else:
        umm = np.lib.format.open_memmap(out_dir / FN_U, mode="w+",
                                        dtype=np.float32, shape=(T, Y, X))
        vmm = np.lib.format.open_memmap(out_dir / FN_V, mode="w+",
                                        dtype=np.float32, shape=(T, Y, X))
        umm[:] = u
        vmm[:] = v
        umm.flush(); vmm.flush()
        del umm, vmm
        speed_max = float(np.hypot(u, v).max()) if speed_max is None else float(speed_max)
        ivd_source = "computed_inmemory"
        if ivd is not None:
            ivd_arr = np.asarray(ivd, dtype=np.float32)[:T]
            np.save(out_dir / FN_IVD, ivd_arr)
        else:
            np.save(out_dir / FN_IVD,
                    weak_labels.compute_ivd(u, v, xdim, ydim, mask=mask2d).astype(np.float32))

    # ---- IVD 复用路径（票 04 产物）
    if ivd is not None and isinstance(ivd, (str, pathlib.Path)):
        ivd_arr = np.asarray(np.load(str(ivd)), dtype=np.float32)[:T]
        np.save(out_dir / FN_IVD, ivd_arr)
        ivd_source = "provided"

    # ---- 时间片与 τ
    if split_mode == "frac":
        # 票 07 延伸：按帧比例划分（60/40，可带 val）——多数据集通用口径
        slices = fraction_slices(T, train_frac=train_frac, val_frac=val_frac)
    slices = _fit_slices(slices or weak_labels.DEFAULT_SLICES, T)
    ivd_mm = np.load(out_dir / FN_IVD, mmap_mode="r")
    if taus is None:
        # 流体区逐时间片分位数（弱标签口径：排除固体 0 值污染）
        taus = weak_labels.compute_tau(ivd_mm, mask2d, slices, percentile=percentile)
    taus = {k: float(val) for k, val in taus.items()}

    # ---- 标签场（面积过滤 + 固体强制 0；复用 weak_labels 单一口径）
    if labels is not None and isinstance(labels, (str, pathlib.Path)):
        lab = np.asarray(np.load(str(labels)), dtype=np.uint8)[:T]
        np.save(out_dir / FN_LABEL, lab)
    else:
        if labels is not None:
            np.save(out_dir / FN_LABEL, np.asarray(labels, dtype=np.uint8)[:T])
        else:
            lab = weak_labels.build_label_field(ivd_mm, mask2d, taus, slices,
                                                min_area=min_area)
            np.save(out_dir / FN_LABEL, lab)

    # ---- 归一化统计（IVD z-score：train 片流体区，σ=0 防护）
    if ivd_mu is None or ivd_sigma is None:
        i0, i1 = slices.get(ivd_stats_slice, slices[next(iter(slices))])
        vals = np.asarray(ivd_mm[i0:i1])[:, ~mask2d]
        if vals.size == 0:
            raise ValueError(f"统计片 {ivd_stats_slice} 无流体格")
        mu = float(vals.mean()) if ivd_mu is None else float(ivd_mu)
        sg = float(vals.std()) if ivd_sigma is None else float(ivd_sigma)
        if sg <= 0:
            sg = 0.0
    else:
        mu, sg = float(ivd_mu), float(ivd_sigma)
    del ivd_mm

    # ---- meta.json
    meta = {
        "source_nc": str(nc_path) if nc_path is not None else "in-memory field",
        "shape": [T, Y, X],
        "xdim": [float(x) for x in xdim],
        "ydim": [float(y) for y in ydim],
        "tdim": [float(t) for t in tdim],
        "dt": float(tdim[1] - tdim[0]),
        "slices": {k: [int(a), int(b)] for k, (a, b) in slices.items()},
        "split_mode": split_mode,
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "taus": taus,
        "percentile": float(percentile),
        "min_area": int(min_area),
        "speed_max": float(speed_max),
        "ivd_mu": mu,
        "ivd_sigma": sg,
        "ivd_stats_slice": ivd_stats_slice,
        "ivd_source": ivd_source,
        "mask_source": "none" if not mask2d.any() else "provided_or_computed",
        "mask_solid_cells": int(mask2d.sum()),
        "params": {
            "patch_size": [int(p) for p in patch_size],
            "stride": [int(s) for s in stride],
            "t_win": int(t_win),
            "window_step": int(window_step),
        },
    }
    (out_dir / FN_META).write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- 数据集类

def _comb_rng_base(seed, py, px, frame, ds_id=None):
    """组合级确定性随机基：同 (ds_id, seed, patch, 帧) → 同 base（跨会话稳定）。

    供批量提取的 per-k 重播种派生（SeedSequence([base, k])）与池判定共用，
    保证"池判定 / 标签判定 / 提取"三者一致（可复现）。ds_id 标识数据集
    归属（多数据集池；None = 单数据集——字节兼容旧口径 f"{seed}:{py}:{px}:{frame}"，
    避免升级后单数据集续训采样序漂移）。
    """
    if ds_id is None:
        key = f"{int(seed)}:{int(py)}:{int(px)}:{int(frame)}"
    else:
        key = f"{int(seed)}:{int(ds_id)}:{int(py)}:{int(px)}:{int(frame)}"
    return zlib.crc32(key.encode("utf-8"))


class _DatasetStore:
    """单数据集准备产物的存储与提取（一个 prepare_dataset 输出目录）。

    弱标签口径（HANDOFF §1 决策 8 / 票 05）：池判定 = patch_positive_map
    （weak_labels 单一公式）；标签 = 重播种后种子格 label_field；归一化统计
    取本数据集 meta.json（IVD z-score 的 μ/σ 与 speed_max 逐数据集各自——
    票 07 延伸定案：跨数据集输入尺度一致化）；组合级确定性 rng 基 =
    _comb_rng_base(seed, py, px, frame, ds_id)。

    采样/过采样（epoch 序、50% 正样本）不属于 store——由
    WeakLabelPathlineDataset（单数据集）与 MultiDatasetPathlineDataset
    （多数据集池）各自实现；sample_at 为公开入口（任意组合直接取，预览/滑窗用）。
    """

    def __init__(self, data_root, split="train", *,
                 patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE,
                 t_win=DEFAULT_T_WIN, window_step=DEFAULT_WINDOW_STEP,
                 seed=0, groups=DEFAULT_GROUPS, delta_frac=DEFAULT_DELTA_FRAC,
                 L=DEFAULT_L, n_substeps=4, ds_id=None):
        root = pathlib.Path(data_root)
        self._root = root
        self._meta = load_dataset_meta(root)
        shape = self._meta["shape"]
        self.T, self.Y, self.X = shape
        self._xdim = np.asarray(self._meta["xdim"], dtype=np.float64)
        self._ydim = np.asarray(self._meta["ydim"], dtype=np.float64)
        self._tdim = np.asarray(self._meta["tdim"], dtype=np.float64)
        slices = {k: (int(a), int(b)) for k, (a, b) in self._meta["slices"].items()}
        if split not in slices:
            raise ValueError(f"split {split!r} 不在时间片 {sorted(slices)} 内")
        self.split = split
        self.split_i0, self.split_i1 = slices[split]
        self._i0, self._i1 = self.split_i0, self.split_i1
        self.patch_size = tuple(int(s) for s in patch_size)
        self.stride = tuple(int(s) for s in stride)
        self.t_win = int(t_win)
        self.window_step = int(window_step)
        self.seed = int(seed)
        self.groups = tuple(int(g) for g in groups)
        self.delta_frac = float(delta_frac)
        self.L = int(L)
        self.n_substeps = int(n_substeps)
        self.ds_id = ds_id
        self.speed_max = float(self._meta["speed_max"])
        self.ivd_mu = float(self._meta["ivd_mu"])
        self.ivd_sigma = float(self._meta["ivd_sigma"])
        self.t_span = (self.t_win - 1) * (self._tdim[1] - self._tdim[0])

        self._u_mm = np.load(root / FN_U, mmap_mode="r")
        self._v_mm = np.load(root / FN_V, mmap_mode="r")
        self._ivd_mm = np.load(root / FN_IVD, mmap_mode="r")
        self._label_mm = np.load(root / FN_LABEL, mmap_mode="r")
        self._mask2d = np.asarray(np.load(root / FN_MASK), dtype=bool)
        self._patches = patch_locations(self.Y, self.X, self.patch_size, self.stride)
        self.pool_positive, self.pool_negative = self._build_pools()

    # ---------------- 池构建（正样本判据：与 weak_labels 单公式共用）

    def _patch_usable(self, y0, x0):
        """patch 位置可否用于提取（种子重播种可行性，静态精确判定）。

        票 03 边界："patch 全固体 ValueError（上层采样应避开全固体 patch）"；
        实测几何（pipedcylinder2d）存在**非全固体但种子-中心线段全固体**的
        patch（patch 中心在壁面/圆柱内）→ 重播种必然失败。
        判据：对每个落固体的种子，沿 seed→patch 中心线段采样 201 点
        （与 reseed 细扫同密度），存在流体格 → 可用；否则不可用。
        种子全流体 → 无需重播种 → 恒可用。
        """
        seeds = extractor.seeding_grid(
            (y0, x0), self.patch_size, self._xdim, self._ydim,
            self.groups, self.delta_frac)
        geo = extractor.patch_geometry((y0, x0), self.patch_size, self._xdim, self._ydim)
        center = np.array([geo["cx"], geo["cy"]])
        solid = self._solid_seeds(np.asarray(seeds, dtype=np.float64))
        if len(solid) == 0:
            return True
        s = np.linspace(0.0, 1.0, 201)[:, None]
        for k in solid:
            pts = seeds[k] + s * (center - seeds[k][None, :])
            j, i = extractor.nearest_cell(pts[:, 0], pts[:, 1], self._xdim, self._ydim)
            i = np.clip(i, 0, self.X - 1)
            j = np.clip(j, 0, self.Y - 1)
            if not self._mask2d[j, i].all():
                return True
        return False

    def _build_pools(self):
        """正/负样本池 = (patch 位置, 窗口起点帧) 组合；判定用 patch_positive_map。

        不可用 patch 排除（_patch_usable：种子全固体或种子-中心线段全固体）——
        该类 patch 提取必然失败（票 03 ValueError 语义），不入池。
        """
        usable_idx = [i for i, (y0, x0) in enumerate(self._patches)
                      if self._patch_usable(y0, x0)]
        self._usable_patches = [self._patches[i] for i in usable_idx]
        pos, neg = [], []
        for frame in window_starts(self._i0, self._i1, self.t_win, self.window_step):
            pm = weak_labels.patch_positive_map(
                self._label_mm[frame], self._xdim, self._ydim,
                self.patch_size, self.stride, self.groups, self.delta_frac)
            pm_flat = pm.reshape(-1)
            for i in usable_idx:
                (y0, x0) = self._patches[i]
                (pos if pm_flat[i] else neg).append((y0, x0, int(frame)))
        return pos, neg

    # ---------------- 确定性种子（重播种后；__getitem__/标签判定的判据）

    def _solid_seeds(self, seeds):
        """种子格在固体掩膜中的 k 索引（向量化检查；extractor.nearest_cell 单一公式）。"""
        j, i = extractor.nearest_cell(seeds[:, 0], seeds[:, 1], self._xdim, self._ydim)
        i = np.clip(i, 0, self.X - 1)
        j = np.clip(j, 0, self.Y - 1)
        return np.nonzero(self._mask2d[j, i])[0]

    def seeds_for(self, py, px, frame):
        """(py, px, frame) → 重播种后 256 个种子物理坐标 (K,2)（确定性）。

        与 __getitem__ 使用同一组合级 rng 派生（_comb_rng_base）与同一
        _extract 路径（含短迹线重试）→ 池判定/标签判定/提取三者严格一致。
        仅用于测试与诊断（提取完整样本以取种子，成本与 __getitem__ 同）。
        """
        _raw, seeds, _geo = self._extract(py, px, frame)
        return seeds

    # ---------------- 样本生成（on-the-fly 提取 + 归一化 + 标签）

    def _extract(self, py, px, frame):
        """组合 → (raw (L,K,7), seeds (K,2), geo)：批量提取（组合级确定性 rng）。

        时变语义：u/v/ivd 窗口切片（T_win 帧）必须配**窗口 tdim**（时间索引
        相对窗口起点；传全场 tdim 会把时间映射 clamp 到窗口末帧——Spec 审查
        实测复现的时变冻结 bug，此处为单一修复点）。
        """
        geo = extractor.patch_geometry((py, px), self.patch_size, self._xdim, self._ydim)
        base = _comb_rng_base(self.seed, py, px, frame, ds_id=self.ds_id)
        u_win = np.asarray(self._u_mm[frame:frame + self.t_win], dtype=np.float32)
        v_win = np.asarray(self._v_mm[frame:frame + self.t_win], dtype=np.float32)
        ivd_win = np.asarray(self._ivd_mm[frame:frame + self.t_win], dtype=np.float32)
        tdim_win = self._tdim[frame:frame + self.t_win]
        raw, seeds = extractor.extract_pathlines_batched(
            u_win, v_win, self._mask2d, ivd_win, self._xdim, self._ydim, tdim_win,
            patch_yx=(py, px), patch_size=self.patch_size,
            t0=float(self._tdim[frame]), L=self.L,
            groups=self.groups, delta_frac=self.delta_frac,
            t_win_frames=self.t_win, n_substeps=self.n_substeps,
            rng=base, return_seeds=True)
        return raw, seeds, geo

    def _labels_for(self, seeds, frame):
        """重播种后种子最近格（单一公式 nearest_cell）→ label_field 值 (K,)。"""
        j, i = extractor.nearest_cell(seeds[:, 0], seeds[:, 1], self._xdim, self._ydim)
        i = np.clip(i, 0, self.X - 1)
        j = np.clip(j, 0, self.Y - 1)
        return self._label_mm[frame][j, i].astype(np.float32)

    def sample_at(self, py, px, frame, t_scale=DEFAULT_T_SCALE):
        """指定 (patch 位置 y0,x0, 窗口起点帧) 的完整样本——预览/诊断公开入口。

        与 __getitem__ 同路径（_extract + normalize_pathlines + 标签判定），
        返回 ((dummy_field, pathlines), labels, seeds)；不依赖 set_epoch/采样序
        （任意 (patch, 帧) 组合可直接取——票 07 预览、票 08 滑窗评估的基础）。
        """
        raw, seeds, geo = self._extract(py, px, frame)
        pathlines = normalize_pathlines(raw, seeds, geo, float(self._tdim[frame]),
                                        self.t_span, t_scale, self.ivd_mu,
                                        self.ivd_sigma, self.speed_max)
        labels = self._labels_for(seeds, frame)
        return (np.zeros((1, 1, 1, 1), dtype=np.float32), pathlines), labels, seeds


# --------------------------------------------------------------------------- 单数据集包装（票 05 公开面）

def _mixed_order(pool_positive, pool_negative, samples_per_epoch, positive_fraction,
                 seed, epoch):
    """50% 正池（放回）+ 50% 负池（放回）后打乱的采样序——单一公式（确定性 (seed, epoch)）。

    单数据集（WeakLabelPathlineDataset，池编码 (y0,x0,frame)）与多数据集
    （MultiDatasetPathlineDataset，编码 (si,y0,x0,frame)）共用：行数/行宽不同，
    rng 语义一致（同 seed+epoch → 同序）。
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    n_pos = int(round(int(samples_per_epoch) * float(positive_fraction)))
    n_neg = int(samples_per_epoch) - n_pos
    if not pool_positive:
        raise ValueError(
            "正样本池为空：无正 patch（检查 τ/标签场/时间片；多数据集时为联合池）")
    if n_neg > 0 and not pool_negative:
        raise ValueError("负样本池为空（样本池不完整）")
    pick_p = np.asarray(pool_positive, dtype=np.int64)
    pidx = pick_p[rng.integers(0, len(pool_positive), size=n_pos)]
    if n_neg:
        pick_n = np.asarray(pool_negative, dtype=np.int64)
        nidx = pick_n[rng.integers(0, len(pool_negative), size=n_neg)]
        order = np.concatenate([pidx, nidx])
    else:
        order = pidx
    rng.shuffle(order)
    return [tuple(int(x) for x in c) for c in order]


class WeakLabelPathlineDataset:
    """弱标签迹线数据集（on-the-fly；h5py+memmap）——单数据集包装（票 05 口径）。

    构造：data_root 为 prepare_dataset 的输出目录（meta.json + memmap）。
    set_epoch(epoch) 重建 50% 正样本过采样的采样序（每 epoch 调用一次；
    首次使用前必须调用；同 (seed, epoch) → 同序，字节兼容票 05 实现）。
    __getitem__(idx) 返回 ((dummy_field, pathlines), labels)。

    样本池：正 = patch 内存在 ≥1 条涡迹线（weak_labels.patch_positive_map 判据，
    与票 04 正样本统计单公式共用）；负 = 其余。标签 = 重播种后种子格处
    label_field 值（与输入迹线的实际出发位置自洽；正池零误差、负池掺正 ≤2%
    为票 04 已披露近似，不影响过采样设计）。存储/提取委托 _DatasetStore
    （ds_id=None → 组合级 rng 基与旧实现逐字节一致；跨数据集预览传
    dataset_idx 作 ds_id——与多数据集池同构）。
    """

    def __init__(self, data_root, split="train", *,
                 patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE,
                 t_win=DEFAULT_T_WIN, window_step=DEFAULT_WINDOW_STEP,
                 samples_per_epoch=DEFAULT_SAMPLES_PER_EPOCH,
                 positive_fraction=DEFAULT_POSITIVE_FRACTION,
                 t_scale=DEFAULT_T_SCALE, seed=0,
                 groups=DEFAULT_GROUPS, delta_frac=DEFAULT_DELTA_FRAC,
                 L=DEFAULT_L, n_substeps=4, ds_id=None):
        self._store = _DatasetStore(data_root, split, patch_size=patch_size,
                                    stride=stride, t_win=t_win,
                                    window_step=window_step, seed=seed,
                                    groups=groups, delta_frac=delta_frac,
                                    L=L, n_substeps=n_substeps, ds_id=ds_id)
        # 公开别名（票 05 池名/测试/预览引用不变；委托存储）
        self.pool_positive = self._store.pool_positive
        self.pool_negative = self._store.pool_negative
        self._patch_usable = self._store._patch_usable
        self.seeds_for = self._store.seeds_for
        self.T, self.Y, self.X = self._store.T, self._store.Y, self._store.X
        self.patch_size = self._store.patch_size
        self.stride = self._store.stride
        self.split = split
        self._root = self._store._root
        self.t_win = self._store.t_win
        self.window_step = self._store.window_step
        self.groups = self._store.groups
        self.delta_frac = self._store.delta_frac
        self.L = self._store.L
        self.n_substeps = self._store.n_substeps
        self.samples_per_epoch = int(samples_per_epoch)
        self.positive_fraction = float(positive_fraction)
        self.t_scale = float(t_scale)
        self.seed = int(seed)
        self.speed_max = self._store.speed_max
        self.ivd_mu = self._store.ivd_mu
        self.ivd_sigma = self._store.ivd_sigma
        self.t_span = self._store.t_span
        self._xdim = self._store._xdim
        self._ydim = self._store._ydim
        self._tdim = self._store._tdim
        self._label_mm = self._store._label_mm
        self._ivd_mm = self._store._ivd_mm
        self._u_mm = self._store._u_mm
        self._v_mm = self._store._v_mm
        self._mask2d = self._store._mask2d
        self._order = None
        self._epoch = None

    # ---------------- epoch 采样（50% 正样本过采样；与票 05 同序口径）

    def set_epoch(self, epoch):
        """重建采样序：50% 正池（放回）+ 50% 负池（放回）后打乱。

        每 epoch 调用一次；同 (seed, epoch) → 同序（确定性可复现，_mixed_order
        单一公式与多数据集共用）。
        """
        self._epoch = int(epoch)
        self._order = _mixed_order(self.pool_positive, self.pool_negative,
                                   self.samples_per_epoch, self.positive_fraction,
                                   self.seed, self._epoch)
        return self._order

    def set_epoch_natural(self, epoch=0):
        """按池自然比例（正/负池大小比）重建采样序——自然分布评估口径。

        训练监控用 50% 平衡（set_epoch）；自然分布（真实正负占比）用于训练
        收尾的 val F1 记录（票 07 验收 4；正式弱定量表属票 08）。
        """
        n_pos = len(self.pool_positive)
        n_neg = len(self.pool_negative)
        total = n_pos + n_neg
        self.positive_fraction = n_pos / total if total > 0 else 0.5
        return self.set_epoch(int(epoch))

    def sample_at(self, py, px, frame):
        """指定 (patch 位置 y0,x0, 窗口起点帧) 的完整样本——预览/诊断公开入口。

        委托 store（与 __getitem__ 同路径）；返回 ((dummy_field, pathlines),
        labels, seeds)。
        """
        return self._store.sample_at(py, px, frame, self.t_scale)

    @property
    def store(self):
        """底层 _DatasetStore（预览/滑窗按数据集取样本的委托接缝）。"""
        return self._store

    # ---------------- __getitem__

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if self._order is None:
            raise RuntimeError("先调用 set_epoch(epoch) 再采样（每 epoch 一次）")
        py, px, frame = self._order[idx]
        return self._store.sample_at(py, px, frame, self.t_scale)[:2]


# --------------------------------------------------------------------------- 多数据集联合池（票 07 延伸）

class MultiDatasetPathlineDataset:
    """多数据集联合采样池（票 07 延伸：各数据集帧前 60% 一起训练/后 40% 留出评估）。

    池 = 各数据集 store（_DatasetStore）的组合并集，组合编码 =
    (store_idx, y0, x0, frame)；set_epoch(epoch) 重建 50% 正样本过采样序
    （同 (seed, epoch) 确定性）；set_epoch_natural 按联合池自然比例；
    sample_at(si, y0, x0, frame) 公开入口（预览/票 08 滑窗按数据集取样本）。
    τ 与归一化逐数据集（各 store 自身 meta 统计：ivd z-score、u/v÷speed_max、
    px/py 为 patch 内归一化——跨数据集输入尺度一致）；组合级 rng 基含
    ds_id 派生（同语义、与单数据集不同构）。
    """

    def __init__(self, roots, split="train", *,
                 patch_size=DEFAULT_PATCH_SIZE, stride=DEFAULT_STRIDE,
                 t_win=DEFAULT_T_WIN, window_step=DEFAULT_WINDOW_STEP,
                 samples_per_epoch=DEFAULT_SAMPLES_PER_EPOCH,
                 positive_fraction=DEFAULT_POSITIVE_FRACTION,
                 t_scale=DEFAULT_T_SCALE, seed=0,
                 groups=DEFAULT_GROUPS, delta_frac=DEFAULT_DELTA_FRAC,
                 L=DEFAULT_L, n_substeps=4):
        roots = [pathlib.Path(r) for r in roots]
        if not roots:
            raise ValueError("roots 为空：至少一个数据集目录")
        self._stores = [_DatasetStore(
            r, split, patch_size=patch_size, stride=stride, t_win=t_win,
            window_step=window_step, seed=seed, groups=groups,
            delta_frac=delta_frac, L=L, n_substeps=n_substeps, ds_id=i)
            for i, r in enumerate(roots)]
        self.pool_positive = [(i, *combo) for i, s in enumerate(self._stores)
                              for combo in s.pool_positive]
        self.pool_negative = [(i, *combo) for i, s in enumerate(self._stores)
                              for combo in s.pool_negative]
        self.samples_per_epoch = int(samples_per_epoch)
        self.positive_fraction = float(positive_fraction)
        self.t_scale = float(t_scale)
        self.seed = int(seed)
        self._order = None
        self._epoch = None

    @property
    def stores(self):
        """各数据集 store（预览/滑窗按数据集取样本与 patch 位置）。"""
        return self._stores

    def set_epoch(self, epoch):
        """重建采样序：联合池 50% 正样本过采样（同 (seed, epoch) → 同序）。"""
        self._epoch = int(epoch)
        self._order = _mixed_order(self.pool_positive, self.pool_negative,
                                   self.samples_per_epoch, self.positive_fraction,
                                   self.seed, self._epoch)
        return self._order

    def set_epoch_natural(self, epoch=0):
        """按联合池自然比例（正/负池大小比）重建采样序（留出评估口径）。"""
        n_pos = len(self.pool_positive)
        n_neg = len(self.pool_negative)
        total = n_pos + n_neg
        self.positive_fraction = n_pos / total if total > 0 else 0.5
        return self.set_epoch(int(epoch))

    def sample_at(self, si, y0, x0, frame):
        """指定 (数据集索引, patch 位置, 窗口起点帧) 的完整样本——公开入口。

        返回 ((dummy_field, pathlines), labels, seeds)；归一化取该数据集
        store 自己的统计。
        """
        return self._stores[int(si)].sample_at(y0, x0, frame, self.t_scale)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if self._order is None:
            raise RuntimeError("先调用 set_epoch(epoch) 再采样（每 epoch 一次）")
        si, py, px, frame = self._order[idx]
        return self._stores[si].sample_at(py, px, frame, self.t_scale)[:2]


# --------------------------------------------------------------------------- CLI（prepare_dataset 入口）

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="数据集准备：u/v/ivd/label/mask memmap + meta.json "
                    "（弱标签迹线数据集；时间划分/τ 与 weak_labels 口径一致）")
    ap.add_argument("nc_path", nargs="?", default=None,
                    help="nc 数据集路径（h5py 直读，支持中文路径；缺省用内存数组参数）")
    ap.add_argument("--out-dir", default="outputs/dataset",
                    help="数据集输出目录（meta.json + memmap）")
    ap.add_argument("--mask", default=None,
                    help="固体掩膜 mask.npy 路径（缺省从速度场计算或空掩膜）")
    ap.add_argument("--ivd", default=None, help="复用票 04 产物 ivd.npy 路径")
    ap.add_argument("--labels", default=None, help="复用票 04 产物 label_field.npy 路径")
    ap.add_argument("--percentile", type=float, default=weak_labels.DEFAULT_PERCENTILE,
                    help="τ 分位数（默认 85——票 07 延伸；HANDOFF §6）")
    ap.add_argument("--split-mode", choices=("abs", "frac"), default="abs",
                    help="时间片划分口径：abs=绝对秒数 DEFAULT_SLICES（单数据集默认）；"
                         "frac=按帧比例（多数据集 60/40，票 07 延伸）")
    ap.add_argument("--train-frac", type=float, default=0.6,
                    help="frac 口径的训练帧比例（默认 0.6）")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="frac 口径的 val 帧比例（默认 0=无 val 片）")
    args = ap.parse_args(argv)

    meta = prepare_dataset(args.nc_path, args.out_dir, mask=args.mask,
                           ivd=args.ivd, labels=args.labels,
                           percentile=args.percentile,
                           split_mode=args.split_mode,
                           train_frac=args.train_frac, val_frac=args.val_frac)
    print(f"数据集已准备: {args.out_dir}")
    print(f"  shape={meta['shape']} slices={meta['slices']}")
    print(f"  taus={meta['taus']}")
    print(f"  speed_max={meta['speed_max']:.6g} ivd_mu={meta['ivd_mu']:.6g} "
          f"ivd_sigma={meta['ivd_sigma']:.6g}")
    return 0


if __name__ == "__main__":
    main()
