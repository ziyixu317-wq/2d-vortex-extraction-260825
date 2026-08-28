"""票 07 延伸：多数据集联合采样池（dataset.MultiDatasetPathlineDataset）测试。

领域词汇（HANDOFF §1 决策 8 / 票 07 延伸要求，唯一权威）：
- 多数据集 = 多个 prepare_dataset 产物目录；池 = 各数据集 (patch, 窗口帧) 组合并集；
- 前 60% 训 / 后 40% 测（dataset.fraction_slices，按帧比例、逐数据集各自）；
- 池判定/标签/提取一致（组合级确定性 rng；ds_id 派生——多数据集与单数据集
  的 rng 基不同构：单数据集字节兼容口径不变，多数据集为同语义不同随机实现）；
- 归一化/τ 逐数据集各自（各 store 自己 meta 的 ivd μ/σ、speed_max——票 07
  延伸定案：输入尺度跨数据集一致化，ivd≈N(0,1)、u/v∈[−1,1]）；
- 正样本 = patch 内存在 ≥1 条涡迹线（weak_labels.patch_positive_map 单一公式）。

期望值来源：合成 Rankine 涡场（test_dataset 同款，已知 τ 字面量 SYNTH_TAU）；
池比例/划分边界为数据自描述公开属性 + 手工字面量。
"""

import numpy as np
import pytest

import dataset as ds
import test_dataset as tds


def prepare_synth_pair(tmp_path, tag_a="A", tag_b="B", T=40, speed_a=None,
                       speed_b=None, name_a="synth_a", name_b="synth_b"):
    """两个合成数据集目录（Rankine 涡场，同构场不同统计；返回 (root_a, root_b, u, v...)。"""
    root_a = tmp_path / name_a
    root_b = tmp_path / name_b
    u, v, xdim, ydim, tdim = tds.synth_prepared(root_a, T=T)
    meta_a = ds.prepare_dataset(None, str(root_a), u=u, v=v, xdim=xdim, ydim=ydim,
                                tdim=tdim, taus={"train": tds.SYNTH_TAU,
                                                 "test": tds.SYNTH_TAU},
                                slices={"train": (0, 24), "test": (24, T)},
                                speed_max=speed_a)
    meta_b = ds.prepare_dataset(None, str(root_b), u=u, v=v, xdim=xdim, ydim=ydim,
                                tdim=tdim, taus={"train": tds.SYNTH_TAU,
                                                 "test": tds.SYNTH_TAU},
                                slices={"train": (0, 24), "test": (24, T)},
                                speed_max=speed_b)
    return root_a, root_b, meta_a, meta_b


class TestMultiDatasetPool:
    """多数据集池：合并、逐数据集采样/归一化、确定性、划分无泄漏。"""

    def test_pool_union_counts(self, tmp_path):
        """池 = 各数据集组合并集（正/负池大小分别为两 store 之和——数据自描述属性）。"""
        root_a, root_b, ma, mb = prepare_synth_pair(tmp_path)
        m = ds.MultiDatasetPathlineDataset([str(root_a), str(root_b)], split="train",
                                           samples_per_epoch=8)
        sa = ds.WeakLabelPathlineDataset(str(root_a), split="train", samples_per_epoch=8)
        sb = ds.WeakLabelPathlineDataset(str(root_b), split="train", samples_per_epoch=8)
        assert len(m.pool_positive) == len(sa.pool_positive) + len(sb.pool_positive)
        assert len(m.pool_negative) == len(sa.pool_negative) + len(sb.pool_negative)
        # 组合编码 = (store_idx, y0, x0, frame)；单数据集组合 = (y0, x0, frame)
        assert len(m.pool_positive[0]) == 4

    def test_per_dataset_normalization(self, tmp_path):
        """归一化逐数据集：同一场不同 speed_max（1.0 vs 10.0）→ u/v 通道相差 10×。"""
        root_a, root_b, ma, mb = prepare_synth_pair(tmp_path, speed_a=1.0,
                                                    speed_b=10.0)
        m = ds.MultiDatasetPathlineDataset([str(root_a), str(root_b)], split="train",
                                           samples_per_epoch=8)
        combo = m.pool_positive[0] if m.pool_positive else m.pool_negative[0]
        si, other = combo[0], 1 - combo[0]
        y0, x0, frame = combo[1], combo[2], combo[3]
        (_, pa0), _, _ = m.sample_at(si, y0, x0, frame)
        (_, pa1), _, _ = m.sample_at(other, y0, x0, frame)
        from extractor import CH_U, CH_V
        assert np.allclose(pa1[:, :, CH_U], pa0[:, :, CH_U] / 10.0, atol=1e-6)
        assert np.allclose(pa1[:, :, CH_V], pa0[:, :, CH_V] / 10.0, atol=1e-6)
        # z-score 也逐数据集：两个 store 同一场同一 μ/σ（同统计覆盖）→ ivd 通道一致
        from extractor import CH_IVD
        assert np.allclose(pa1[:, :, CH_IVD], pa0[:, :, CH_IVD], atol=1e-6)

    def test_set_epoch_deterministic_and_natural(self, tmp_path):
        """同 (seed, epoch) → 同采样序；set_epoch_natural 正负比例 = 联合池比例。"""
        root_a, root_b, ma, mb = prepare_synth_pair(tmp_path)
        m = ds.MultiDatasetPathlineDataset([str(root_a), str(root_b)], split="train",
                                           samples_per_epoch=16)
        o1 = m.set_epoch(3)
        o2 = m.set_epoch(3)
        assert o1 == o2
        m.set_epoch_natural(0)
        n_pos = sum(len(s.pool_positive) for s in m.stores)
        n_neg = sum(len(s.pool_negative) for s in m.stores)
        assert m.positive_fraction == pytest.approx(n_pos / (n_pos + n_neg))

    def test_sample_and_getitem_shapes(self, tmp_path):
        """sample_at/__getitem__：pathlines (16,256,7) 有限、labels ∈ {0,1}。"""
        root_a, root_b, ma, mb = prepare_synth_pair(tmp_path)
        m = ds.MultiDatasetPathlineDataset([str(root_a), str(root_b)], split="train",
                                           samples_per_epoch=4)
        m.set_epoch(0)
        ((dummy, path), labels) = m[0]
        assert dummy.shape == (1, 1, 1, 1)
        assert path.shape == (16, 256, 7) and np.isfinite(path).all()
        assert set(np.unique(labels)) <= {0.0, 1.0}
        combo = m.pool_positive[0] if m.pool_positive else m.pool_negative[0]
        (_, p2), l2, seeds = m.sample_at(combo[0], combo[1], combo[2], combo[3])
        assert seeds.shape == (256, 2) and np.isfinite(seeds).all()

    def test_no_time_leak_train_test(self, tmp_path):
        """划分无泄漏：train 池帧 < test 片左界（各数据集各自边界），窗口在片内。"""
        root_a, root_b, ma, mb = prepare_synth_pair(tmp_path, T=48)
        mtr = ds.MultiDatasetPathlineDataset([str(root_a), str(root_b)],
                                             split="train", samples_per_epoch=4)
        mte = ds.MultiDatasetPathlineDataset([str(root_a), str(root_b)],
                                             split="test", samples_per_epoch=4)
        for si, store in enumerate(mtr.stores):
            i1 = store.split_i1          # train 片右界
            for (_y0, _x0, f) in store.pool_positive + store.pool_negative:
                assert f + 24 <= i1      # 窗口完全在片内
                assert f < i1
            for (_y0, _x0, f) in mte.stores[si].pool_positive + mte.stores[si].pool_negative:
                assert f >= i1
        assert len(mtr.stores) == 2 and len(mte.stores) == 2

    def test_roots_missing_fails(self, tmp_path):
        """缺根目录 fail loud（FileNotFoundError，不静默跳过）。"""
        with pytest.raises(FileNotFoundError):
            ds.MultiDatasetPathlineDataset([str(tmp_path / "nope")], split="train",
                                           samples_per_epoch=4)


class TestMultiTrainIntegration:
    """train_kaggle 多数据集接入（票 07 延伸）：data.root 列表 → 多数据集池；
    --report-f1 --f1-split test 对留出 40% 出自然分布 F1/IoU 记录。"""

    @staticmethod
    def make_multi_cfg(tmp_path, epochs=1):
        """两个合成数据集（train/test 时间片）+ 多 root 训练 YAML。"""
        import yaml
        from train_kaggle import load_config
        import test_train as tt
        root_a, root_b = tmp_path / "ma", tmp_path / "mb"
        u, v, xdim, ydim, tdim = tds.synth_prepared(root_a, T=48)
        for r in (root_a, root_b):
            ds.prepare_dataset(None, str(r), u=u, v=v, xdim=xdim, ydim=ydim,
                               tdim=tdim,
                               taus={"train": tds.SYNTH_TAU, "test": tds.SYNTH_TAU},
                               slices={"train": (0, 24), "test": (24, 48)})
        cfg_path = tmp_path / "multi.yaml"
        cfg = load_config()
        cfg["data"]["root"] = [str(root_a), str(root_b)]
        cfg["data"]["num_workers"] = 0
        cfg["data"]["samples_per_epoch"] = 8
        cfg["data"]["batch_size"] = 4
        cfg["train"]["epochs"] = epochs
        cfg["train"]["val_freq"] = 1
        cfg["train"]["seed"] = 0
        cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpts_multi")
        cfg["train"]["run_name"] = "pathline_transformer_multi"
        cfg["model"]["encoder_args"].update(tt.make_small_model_cfg())
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        return cfg_path, tmp_path / "ckpts_multi"

    def test_make_dataset_list_root(self, tmp_path):
        """data.root 为列表 → MultiDatasetPathlineDataset（stores 数 = 列表长）。"""
        from train_kaggle import _make_dataset, load_config
        cfg_path, _ = self.make_multi_cfg(tmp_path)
        cfg = load_config(str(cfg_path))
        d = _make_dataset(cfg["data"], "train")
        assert isinstance(d, ds.MultiDatasetPathlineDataset)
        assert len(d.stores) == 2

    def test_report_f1_on_test_split(self, tmp_path):
        """--f1-split test：留出 40%（跨数据集 test 片）自然分布 F1/IoU 记录落盘。"""
        import json
        from train_kaggle import main
        cfg_path, ckpts = self.make_multi_cfg(tmp_path)
        assert main(["--config", str(cfg_path), "--max-steps", "2",
                     "--report-f1", "--f1-split", "test"]) == 0
        f1_path = ckpts / "pathline_transformer_multi_test_f1.json"
        assert f1_path.exists()
        data = json.loads(f1_path.read_text(encoding="utf-8"))
        assert data["split"] == "test"
        assert data["n"] > 0
        assert 0.0 <= data["f1"] <= 1.0 and 0.0 <= data["iou"] <= 1.0
        assert data["tp"] + data["fp"] + data["fn"] + data["tn"] == data["n"]

    def test_report_f1_missing_split_fails_loud(self, tmp_path):
        """--f1-split 指定的片不存在 → ValueError（不静默跳过）。"""
        from train_kaggle import main
        cfg_path, _ = self.make_multi_cfg(tmp_path)
        with pytest.raises(ValueError):
            main(["--config", str(cfg_path), "--max-steps", "2",
                  "--report-f1", "--f1-split", "nope"])

    def test_multi_config_yaml_shape(self, tmp_path):
        """生产多数据集配置（config/pathline_transformer_multi.yaml）：7 roots、
        frac 口径（val_split none）、run_name 独立（与单数据集训练不混写）。"""
        import yaml
        cfg = yaml.safe_load(open("config/pathline_transformer_multi.yaml",
                                  encoding="utf-8"))
        assert len(cfg["data"]["root"]) == 7
        assert all("outputs/datasets/" in r for r in cfg["data"]["root"])
        assert cfg["data"]["val_split"] == "none"
        assert cfg["train"]["run_name"] == "pathline_transformer_multi"
        assert cfg["train"]["ckpt_dir"] == "outputs/train_multi"

    def test_preview_multi_root(self, tmp_path):
        """跨数据集预览（票 07 延伸）：多 root 配置 + --dataset 选数据集 →
        单帧 模型/IVD/弱标签 对比图落盘（留出 test 片推理）。"""
        from train_kaggle import main as train_main
        from kaggle.preview_eval import main as preview_main
        cfg_path, ckpts = self.make_multi_cfg(tmp_path)
        assert train_main(["--config", str(cfg_path), "--max-steps", "2"]) == 0
        out_png = tmp_path / "preview_multi_t_24.png"
        assert preview_main(["--config", str(cfg_path), "--ckpt",
                             str(ckpts
                                 / "pathline_transformer_multi_ckpt_latest.pth"),
                             "--frame", "24", "--dataset", "1",
                             "--out", str(out_png)]) == 0
        assert out_png.exists() and out_png.stat().st_size > 1000

    def test_preview_dataset_index_out_of_range(self, tmp_path):
        """--dataset 越界 → ValueError（fail loud）。"""
        from kaggle.preview_eval import main as preview_main
        cfg_path, ckpts = self.make_multi_cfg(tmp_path)
        with pytest.raises(ValueError):
            preview_main(["--config", str(cfg_path), "--ckpt",
                          str(ckpts / "does_not_matter.pth"),
                          "--frame", "24", "--dataset", "7"])
