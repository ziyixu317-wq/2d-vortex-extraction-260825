"""票 07 延伸：多数据集逐数据集预处理驱动（prepare_multi.py）测试。

领域词汇（HANDOFF §1 决策 8 / 票 07 延伸，唯一权威）：
- 逐数据集预处理 = geometry 掩膜 → IVD/label → τ 逐时间片 → memmap+meta.json
  （决策 8：掩膜逐数据集、不进模型输入）；
- frac 60/40 按帧比例划分（各数据集自身帧数），无 val 默认；
- τ = 各数据集逐时间片 85 分位（需求 A 定案；逐数据集各自）。
"""

import pathlib

import h5py
import numpy as np
import pytest


def _write_tiny_nc(path, T=16, Y=48, X=64):
    """旋转场（无固体零速区……中心点为 0 速 → 掩膜 1 格，属预期）小 nc 写入。"""
    xdim = np.linspace(-2.0, 2.0, X)
    ydim = np.linspace(-2.0, 2.0, Y)
    tdim = np.linspace(0.0, (T - 1) * 0.05, T)
    tt, yy, xx = np.meshgrid(tdim, ydim, xdim, indexing="ij")
    u = -3.0 * yy + 0.0 * tt
    v = 3.0 * xx + 0.0 * tt
    with h5py.File(str(path), "w") as f:
        f.create_dataset("tdim", data=tdim)
        f.create_dataset("xdim", data=xdim)
        f.create_dataset("ydim", data=ydim)
        f.create_dataset("u", data=np.asarray(u, dtype=np.float32))
        f.create_dataset("v", data=np.asarray(v, dtype=np.float32))
    return tdim


class TestPrepareMulti:
    """多数据集预处理驱动：一条命令 → 每数据集 geometry + dataset + 目检图 + 汇总。"""

    def test_prepare_one_end_to_end(self, tmp_path):
        """单数据集端到端：geometry 掩膜 + frac 60/40 slices + p85 τ + 目检图。"""
        import json
        from prepare_multi import prepare_one
        nc = tmp_path / "synthetic_a.nc"
        _write_tiny_nc(nc, T=16)
        summary = prepare_one(nc, str(tmp_path / "out"),
                              display_frames=(0, 10), percentile=85.0)
        geo_dir = tmp_path / "out" / "synthetic_a" / "geometry"
        ds_dir = tmp_path / "out" / "synthetic_a" / "dataset"
        assert (geo_dir / "mask.npy").exists()
        assert (geo_dir / "geometry_meta.json").exists()
        for f in ("meta.json", "u.npy", "v.npy", "ivd.npy", "label_field.npy",
                  "mask.npy"):
            assert (ds_dir / f).exists(), f
        meta = json.loads((ds_dir / "meta.json").read_text(encoding="utf-8"))
        # frac：16×0.6 = 9.6 → floor 9 → train (0,9)、test (9,16)
        assert meta["slices"] == {"train": [0, 9], "test": [9, 16]}
        assert meta["split_mode"] == "frac"
        assert summary["taus"] == meta["taus"]
        assert set(summary["taus"]) == {"train", "test"}
        assert (tmp_path / "out" / "synthetic_a" / "previews" / "ivd_q_t0.png"
                ).exists()
        assert (tmp_path / "out" / "synthetic_a" / "previews" / "ivd_q_t10.png"
                ).exists()

    def test_prepare_one_explicit_weak_supervision_mode(self, tmp_path):
        """多数据集驱动显式启用新 0/50/60/100 split，默认旧 frac 不漂移。"""
        from prepare_multi import prepare_one
        nc = tmp_path / "cylinder2d.nc"
        _write_tiny_nc(nc, T=50)
        out_root = tmp_path / "out_weak"
        summary = prepare_one(
            nc,
            out_root,
            split_mode="weak_supervision",
            label_source="legacy_p85",
            t_win=2,
            window_step=2,
            display_frames=(0, 49),
        )

        assert summary["split_mode"] == "weak_supervision"
        assert summary["slices"] == {
            "train": [0, 25],
            "calibration": [25, 30],
            "test": [30, 50],
        }
        assert summary["label_source"] == "legacy_p85"
        assert summary["window"]["complete_only"] is True

    def test_prepare_one_rejects_excluded_dataset_in_weak_mode(self, tmp_path):
        """新实验池拒绝 Duffing 等非六个有效数据集，避免产物污染。"""
        from prepare_multi import prepare_one
        nc = tmp_path / "forceddampedduffing2d.nc"
        nc.touch()
        with pytest.raises(ValueError, match=r"six valid|六个有效|forceddampedduffing2d"):
            prepare_one(
                nc,
                tmp_path / "out_excluded",
                split_mode="weak_supervision",
                label_source="legacy_p85",
                t_win=2,
                window_step=1,
            )

    def test_main_writes_multi_meta(self, tmp_path):
        """CLI 主入口：--nc-dir/--names/--out-root → multi_meta.json 汇总。"""
        import json
        from prepare_multi import main
        nc = tmp_path / "synthetic_b.nc"
        _write_tiny_nc(nc, T=12)
        out_root = tmp_path / "outb"
        assert main(["--nc-dir", str(tmp_path), "--out-root", str(out_root),
                     "--names", "synthetic_b.nc", "--frames", "0",
                     "--percentile", "85"]) == 0
        blob = json.loads((out_root / "multi_meta.json")
                          .read_text(encoding="utf-8"))
        assert len(blob["datasets"]) == 1
        d = blob["datasets"][0]
        assert d["name"] == "synthetic_b"
        assert d["shape"] == [12, 48, 64]
        assert set(d["taus"]) == {"train", "test"}
        assert (out_root / "synthetic_b" / "dataset" / "meta.json").exists()

    def test_missing_names_fails(self, tmp_path):
        """--names 指定文件不存在 → FileNotFoundError（fail loud）。"""
        from prepare_multi import main
        with pytest.raises(FileNotFoundError):
            main(["--nc-dir", str(tmp_path), "--out-root", str(tmp_path / "o"),
                  "--names", "not_there.nc"])
