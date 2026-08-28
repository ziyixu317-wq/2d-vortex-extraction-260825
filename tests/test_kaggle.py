"""07 票：Kaggle 训练支撑（val F1 记录 / 分块规划 / Dataset A 打包）测试。

领域词汇（HANDOFF §2/§5/§6/§7 与规格 Implementation Decisions，唯一权威）：
- 票 07 验收 4：最终 checkpoint 归档（含 optimizer 状态）、val F1 记录；
- 自然分布 val 评估口径：正负比例 = val 池比例（非训练同款 50% 平衡；平衡采样
  是训练监控口径，自然分布才是模型真实精度的观察口径——票 06 已注释明示
  「自然分布精度评估归票 08 弱定量表」，此处仅为训练收尾的 val F1 记录）；
- Kaggle 分块：12h 会话硬上限 → 每块 ≤8h（预留自检/打包），每 epoch checkpoint
  + --resume auto 跨会话断点续训（票 07 What to build）；
- Dataset A = nc 数据文件 + prepare_dataset 产物（meta.json + memmap)，
  随 Kaggle Dataset 上传（票 05 产物，gitignore 不走 GitHub）；
- mock 模型 forward 返回常量概率 → 混淆矩阵手算字面量（独立来源，不重算实现路径）。

合成数据工具复用 test_dataset.py（同目录同包；Rankine 涡场 + prepare_dataset）。
"""

import pathlib

import numpy as np
import pytest
import torch

import test_dataset as tds  # 复用合成场构造（Rankine 涡 + 已知 τ 字面量）


# ================================================================ 常量预测模型与数据集

class ConstPredModel(torch.nn.Module):
    """mock：forward 返回常量概率张量 (B, K)（独立于训练路径；手算混淆矩阵来源）。"""

    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, batch):
        dummy, pathlines = batch
        return torch.full((pathlines.shape[0], pathlines.shape[2]),
                          self.value, dtype=torch.float32)


class VariedPredModel(torch.nn.Module):
    """mock：按样本索引交替 0.9/0.1 输出（部分判正、部分判负的混合场景）。"""

    def forward(self, batch):
        dummy, pathlines = batch
        n = pathlines.shape[0]
        vals = torch.tensor([0.9 if i % 2 == 0 else 0.1 for i in range(n)],
                            dtype=torch.float32)
        return torch.zeros(n, pathlines.shape[2]) + vals[:, None]


class _FlatBatchDS(torch.utils.data.Dataset):
    """((dummy(1,1,1,1), pathlines(L,K,C)), labels(K,)) 批结构的合成样本。"""

    def __init__(self, n_samples, K=8, L=16, C=7):
        self.n = n_samples
        self.K, self.L, self.C = K, L, C

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        path = torch.rand(self.L, self.K, self.C, dtype=torch.float32)
        labels = torch.zeros(self.K, dtype=torch.float32)
        labels[: self.K // 2] = 1.0          # 每样本一半正（标签结构确定）
        return (torch.zeros(1, 1, 1, 1), path), labels


def make_loader(batch_size=2):
    from torch.utils.data import DataLoader
    return DataLoader(_FlatBatchDS(4), batch_size=batch_size, num_workers=0)


# ================================================================ 切片 A：evaluate_f1

class TestEvaluateF1:
    """验收 4「val F1 记录」的统计核心：混淆矩阵 → precision/recall/F1。

    期望值来源：手算混淆矩阵字面量（如 tp=6/fp=2 → precision=0.75），
    不通过实现公式重算（防同构空洞通过）。
    """

    def test_all_positive_prediction(self):
        """全部判正（概率恒 0.9 > 0.5）：precision=正命中/判正数、recall=1、F1 字面量。

        每样本 4 正 4 负 × 4 样本（batch 2×2）→ tp=16, fp=16 → P=0.5, R=1, F1=2/3。
        """
        from train_kaggle import evaluate_f1
        model = ConstPredModel(0.9)
        r = evaluate_f1(model, make_loader(), device="cpu")
        assert r["tp"] == 16 and r["fp"] == 16 and r["fn"] == 0
        assert r["precision"] == pytest.approx(0.5)
        assert r["recall"] == pytest.approx(1.0)
        assert r["f1"] == pytest.approx(2.0 / 3.0)
        assert r["n"] == 32

    def test_all_negative_prediction(self):
        """全部判负（概率恒 0.1 ≤ 0.5）：tp=0 → precision=recall=F1=0（无除零崩溃）。"""
        from train_kaggle import evaluate_f1
        r = evaluate_f1(ConstPredModel(0.1), make_loader(), device="cpu")
        assert r["tp"] == 0 and r["fp"] == 0 and r["fn"] == 16
        assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0

    def test_threshold_boundary_semantics(self):
        """阈值语义：概率 0.5 是判负边界（>0.5 判正）；阈值 0.49 时判正。

        字面量：2 样本 × 4 正 4 负（1 批，共 8 正 8 负），阈值 0.49 时全部判正
        → tp=8/fp=8；阈值 0.5 时全部判负 → tp=0/fp=0/fn=8（8 个正被漏检）。
        """
        from torch.utils.data import DataLoader
        from train_kaggle import evaluate_f1
        two = DataLoader(_FlatBatchDS(2), batch_size=2, num_workers=0)
        r_neg = evaluate_f1(ConstPredModel(0.5), two, device="cpu",
                            threshold=0.5)
        assert r_neg["tp"] == 0 and r_neg["fp"] == 0 and r_neg["fn"] == 8
        r_pos = evaluate_f1(ConstPredModel(0.5), two, device="cpu",
                            threshold=0.49)
        assert r_pos["tp"] == 8 and r_pos["fp"] == 8 and r_pos["fn"] == 0

    def test_partial_correctness(self):
        """混判样本：一半样本判正（正全中+负全误报）、一半判负（漏检正）。

        4 样本 = 2 正判 + 2 负判：tp=8、fp=8、fn=8 → P=0.5、R=0.5、F1=0.5。
        """
        from train_kaggle import evaluate_f1
        loader = make_loader()   # 4 样本 = 2 判正（0.9）+ 2 判负（0.1）的混合场景
        r = evaluate_f1(VariedPredModel(), loader, device="cpu")
        assert r["tp"] == 8 and r["fp"] == 8 and r["fn"] == 8
        assert r["precision"] == pytest.approx(0.5)
        assert r["recall"] == pytest.approx(0.5)
        assert r["f1"] == pytest.approx(0.5)

    def test_iou_literal(self):
        """IoU（票 07 延伸：留出评估指标） = tp/(tp+fp+fn)：全判正（tp=16,
        fp=16, fn=0）→ IoU = 16/32 = 0.5（手算字面量）。"""
        from train_kaggle import evaluate_f1
        r = evaluate_f1(ConstPredModel(0.9), make_loader(), device="cpu")
        assert r["iou"] == pytest.approx(0.5)
        assert r["tp"] == 16 and r["fp"] == 16 and r["fn"] == 0

    def test_empty_loader_raises(self):
        """空 loader：报错而非静默返回 0（无样本时 F1 无意义）。"""
        from torch.utils.data import DataLoader
        from train_kaggle import evaluate_f1
        with pytest.raises(ValueError):
            evaluate_f1(ConstPredModel(0.9), DataLoader(_FlatBatchDS(0)),
                        device="cpu")


# ================================================================ 切片 B：--report-f1（val F1 记录）

class TestReportF1CLI:
    """验收 4「val F1 记录」的 CLI 层：训练完成后 --report-f1 写 val_f1.json。

    口径：自然分布（正负比例 = val 池比例，非 50% 平衡——平衡是训练监控口径）；
    json 含混淆计数与指标（文本可审计、可回填票文件）。
    """

    @staticmethod
    def make_val_cfg(tmp_path, epochs=1, with_val=True):
        """合成 train(+val) 片数据 + 训练 YAML（复用 test_train 合成场工具）。

        with_val=False：只有 train 片（无 val 时间片场景——main 应跳过验证/F1）。
        """
        import yaml
        import dataset as ds
        from train_kaggle import load_config
        import test_train as tt
        root = tmp_path / "ds_val"
        T = 48 if with_val else 40
        u, v, xdim, ydim, tdim = tds.synth_prepared(root, T=T)
        slices = {"train": (0, 24), "val": (24, 48)} if with_val else {"train": (0, 24)}
        taus = {"train": tds.SYNTH_TAU, "val": tds.SYNTH_TAU} if with_val \
            else {"train": tds.SYNTH_TAU}
        ds.prepare_dataset(None, str(root), u=u, v=v, xdim=xdim, ydim=ydim,
                           tdim=tdim, taus=taus, slices=slices)
        cfg_path = tmp_path / "train_val.yaml"
        cfg = load_config()
        cfg["data"]["root"] = str(root)
        cfg["data"]["num_workers"] = 0
        cfg["data"]["samples_per_epoch"] = 8
        cfg["data"]["batch_size"] = 4
        cfg["train"]["epochs"] = epochs
        cfg["train"]["val_freq"] = 1
        cfg["train"]["seed"] = 0
        cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpts_f1")
        cfg["model"]["encoder_args"].update(tt.make_small_model_cfg())
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        return cfg_path, tmp_path / "ckpts_f1"

    def test_report_f1_writes_json(self, tmp_path):
        """训练完成后 --report-f1：val_f1.json 落盘且字段自洽（tp+fp+fn=n、F1∈[0,1]）。"""
        import json
        from train_kaggle import main
        cfg_path, ckpts = self.make_val_cfg(tmp_path)
        assert main(["--config", str(cfg_path), "--max-steps", "2",
                     "--report-f1"]) == 0
        f1_path = ckpts / "pathline_transformer_cylinder_val_f1.json"
        assert f1_path.exists()
        data = json.loads(f1_path.read_text(encoding="utf-8"))
        for k in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou", "n",
                  "split", "epoch", "threshold"):
            assert k in data, f"缺失字段 {k}"
        assert data["split"] == "val"
        assert data["epoch"] == 0
        assert 0.0 <= data["f1"] <= 1.0
        # 混淆矩阵完备：tp+fp+fn+tn == n（全部迹线被四格覆盖）
        assert data["tp"] + data["fp"] + data["fn"] + data["tn"] == data["n"]
        assert data["n"] > 0
        assert data["threshold"] == pytest.approx(0.5)

    def test_no_report_f1_no_json(self, tmp_path):
        """未传 --report-f1：不写 F1 json（开关语义；训练行为不变）。"""
        from train_kaggle import main
        cfg_path, ckpts = self.make_val_cfg(tmp_path)
        assert main(["--config", str(cfg_path), "--max-steps", "2"]) == 0
        assert not (ckpts / "pathline_transformer_cylinder_val_f1.json").exists()

    def test_report_f1_without_val_split_skips(self, tmp_path):
        """无 val 时间片（单片数据集）：--report-f1 安全跳过（不崩溃、无 json）。"""
        from train_kaggle import main
        cfg_path, ckpts = self.make_val_cfg(tmp_path, with_val=False)
        assert main(["--config", str(cfg_path), "--max-steps", "2",
                     "--report-f1"]) == 0
        assert not (ckpts / "pathline_transformer_cylinder_val_f1.json").exists()


# ================================================================ 切片 C：set_epoch_natural（自然分布序）

class TestSetEpochNatural:
    """自然分布采样序（val F1 口径）：正负比例 = 池比例（非 50% 平衡）。"""

    def test_natural_fraction_matches_pool_ratio(self, tmp_path):
        """set_epoch_natural 后：正池占比与池比例一致（构造已知比例合成数据集验证）。"""
        import dataset as ds
        root = tmp_path / "ds_nat"
        u, v, xdim, ydim, tdim = tds.synth_prepared(root, T=48)
        ds.prepare_dataset(None, str(root), u=u, v=v, xdim=xdim, ydim=ydim,
                           tdim=tdim, taus={"train": tds.SYNTH_TAU})
        d = ds.WeakLabelPathlineDataset(str(root), split="train",
                                        samples_per_epoch=200, seed=0)
        n_pos, n_neg = len(d.pool_positive), len(d.pool_negative)
        assert n_pos + n_neg > 0
        d.set_epoch_natural(0)
        order = d._order
        # 抽样比例 = 池比例（放回抽样；样本量足够时比例收敛）
        frac = d.positive_fraction
        assert frac == pytest.approx(n_pos / (n_pos + n_neg))
        assert len(order) == d.samples_per_epoch


# ================================================================ 切片 D：分块规划（kaggle/chunking.py）

class TestPlanChunks:
    """Kaggle 分块规划：12h 会话硬上限 → 每块 ≤8h（预留自检/打包），跨会话续训。

    期望值来源：手算字面量（200 epoch ÷ 每块上限 → 块序列；不通过实现公式重算）。
    """

    BUDGET = 8 * 3600            # 8h 会话预算（秒）
    HOUR = 3600

    def test_two_hundred_epochs_plan(self):
        """200 epoch @8min/epoch、8h 预算 → 60 epoch/块 → [60, 60, 60, 20]。

        手算：8h=28800s ÷ 480s = 60 epoch/块；200 = 3×60 + 20（尾块 20）。
        """
        from kaggle.chunking import plan_chunks
        plan = plan_chunks(200, 480, self.BUDGET)
        assert plan == [60, 60, 60, 20]
        assert sum(plan) == 200
        for c in plan:
            assert c * 480 <= self.BUDGET

    def test_single_chunk_when_budget_covers(self):
        """预算覆盖全部 → 单块（块数最少化）：40 epoch @5min/epoch、8h 预算 → [40]。"""
        from kaggle.chunking import plan_chunks
        plan = plan_chunks(40, 5 * 60, self.BUDGET)
        assert plan == [40]

    def test_exact_division_yields_uniform_chunks(self):
        """整除预算：60 epoch @8min → 每块恰好 60 → [60]（无 0 长度尾块）。"""
        from kaggle.chunking import plan_chunks
        assert plan_chunks(60, 480, self.BUDGET) == [60]
        assert plan_chunks(120, 480, self.BUDGET) == [60, 60]

    def test_single_epoch_chunk(self):
        """total=1（冷启动冒烟）→ [1]（退化路径）。"""
        from kaggle.chunking import plan_chunks
        assert plan_chunks(1, 480, self.BUDGET) == [1]

    def test_minimum_chunk_size_guard(self):
        """每 epoch 耗时超预算（无完整 epoch 可放）→ 每块至少 1 epoch（不退化为 0）。"""
        from kaggle.chunking import plan_chunks
        plan = plan_chunks(3, 10 * self.HOUR, self.BUDGET)   # 1 epoch = 10h > 8h 预算
        assert plan == [1, 1, 1]

    def test_invalid_arguments_raise(self):
        """非法参数（total ≤ 0 / 耗时或预算 ≤ 0）→ 报错而非错误规划。"""
        from kaggle.chunking import plan_chunks
        cases = [
            dict(total_epochs=0, seconds_per_epoch=480, budget_seconds=self.BUDGET),
            dict(total_epochs=-1, seconds_per_epoch=480, budget_seconds=self.BUDGET),
            dict(total_epochs=10, seconds_per_epoch=0, budget_seconds=self.BUDGET),
            dict(total_epochs=10, seconds_per_epoch=-1, budget_seconds=self.BUDGET),
            dict(total_epochs=10, seconds_per_epoch=480, budget_seconds=0),
            dict(total_epochs=10, seconds_per_epoch=480, budget_seconds=-1),
        ]
        for kwargs in cases:
            with pytest.raises(ValueError):
                plan_chunks(**kwargs)

    def test_pick_bench_source_prefers_new(self, tmp_path):
        """步速基准来源：本会话实测优先于还原（新测值最新）。"""
        import json
        from kaggle.chunking import pick_bench_source
        new = tmp_path / "bench_info.json"
        old = tmp_path / "restored.json"
        new.write_text(json.dumps({"seconds_per_epoch": 100.0}), encoding="utf-8")
        old.write_text(json.dumps({"seconds_per_epoch": 200.0}), encoding="utf-8")
        p, bench = pick_bench_source(str(new), str(old))
        assert p == str(new) and bench["seconds_per_epoch"] == 100.0

    def test_pick_bench_source_falls_back_to_restored(self, tmp_path):
        """本会话无实测时回退到还原的基准（跨会话复用，省 ~18min 校准）。"""
        import json
        from kaggle.chunking import pick_bench_source
        old = tmp_path / "restored.json"
        old.write_text(json.dumps({"seconds_per_epoch": 1057.0}), encoding="utf-8")
        p, bench = pick_bench_source(str(tmp_path / "missing.json"), str(old))
        assert p == str(old) and bench["seconds_per_epoch"] == 1057.0

    def test_pick_bench_source_none_when_missing(self, tmp_path):
        """两个来源都不存在 → (None, None)（调用方跑校准）。"""
        from kaggle.chunking import pick_bench_source
        p, bench = pick_bench_source(str(tmp_path / "a.json"), str(tmp_path / "b.json"))
        assert p is None and bench is None

    def test_pick_bench_source_corrupt_json_raises(self, tmp_path):
        """基准文件损坏 → 报错而非静默假值（fail loud）。"""
        from kaggle.chunking import pick_bench_source
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            pick_bench_source(str(bad), str(tmp_path / "none.json"))


# ================================================================ 切片 E：Dataset A 打包（kaggle/prepare_dataset_a.py）

def make_fake_dataset_dir(root):
    """人造 prepare_dataset 产物目录：meta.json + u/v/ivd/label/mask（随机字节内容）。"""
    import json
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps({"shape": [24, 48, 96]}),
                                    encoding="utf-8")
    for name in ("u.npy", "v.npy", "ivd.npy", "label_field.npy", "mask.npy"):
        (root / name).write_bytes(np.random.default_rng(0).bytes(1024))
    return root


class TestBuildDatasetA:
    """Kaggle Dataset A 组装：nc + prepare_dataset 产物 + manifest（审计/自检）。"""

    def test_layout_and_content_integrity(self, tmp_path):
        """输出结构 = nc + dataset/（逐字节一致）+ manifest（sha256 引用同一来源）。"""
        import hashlib
        import json
        from kaggle.prepare_dataset_a import build_dataset_a
        nc = tmp_path / "pipedcylinder2d.nc"
        nc.write_bytes(b"NETCDF-PLACEHOLDER")
        ds_dir = make_fake_dataset_dir(tmp_path / "src_dataset")
        out = tmp_path / "out_ds_a"

        manifest = build_dataset_a(str(nc), str(ds_dir), str(out))

        assert (out / "pipedcylinder2d.nc").read_bytes() == b"NETCDF-PLACEHOLDER"
        for name in ("meta.json", "u.npy", "v.npy", "ivd.npy", "label_field.npy",
                     "mask.npy"):
            src = ds_dir / name
            dst = out / "dataset" / name
            assert dst.exists()
            assert hashlib.sha256(dst.read_bytes()).hexdigest() == \
                hashlib.sha256(src.read_bytes()).hexdigest()
        # manifest 自洽：每条记录的文件存在且哈希匹配（可作 Kaggle 端自检清单）
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert m["files"]
        for f in m["files"]:
            p = out / f["path"]
            assert p.exists()
            assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"]
        assert m["total_bytes"] > 0

    def test_aux_pngs_copied_only(self, tmp_path):
        """aux 目录（weak_labels 目检图）：仅复制 .png（大数组不入 A，避免冗余）。"""
        from kaggle.prepare_dataset_a import build_dataset_a
        nc = tmp_path / "nc.bin"
        nc.write_bytes(b"x")
        ds_dir = make_fake_dataset_dir(tmp_path / "ds")
        aux = tmp_path / "aux"
        aux.mkdir()
        (aux / "ivd_q_t400.png").write_bytes(b"PNG1")
        (aux / "tau_sensitivity_t400.png").write_bytes(b"PNG2")
        (aux / "ivd.npy").write_bytes(b"NUMPY-AUX-BIG")     # 非 png 不复制
        out = tmp_path / "out_aux"

        manifest = build_dataset_a(str(nc), str(ds_dir), str(out), aux_dirs=[str(aux)])

        assert (out / "aux" / "ivd_q_t400.png").exists()
        assert (out / "aux" / "tau_sensitivity_t400.png").exists()
        assert not (out / "aux" / "ivd.npy").exists()       # 大数组仅 dataset/ 一份
        assert any(f["path"] == "aux/ivd_q_t400.png" for f in manifest["files"])

    def test_zip_roundtrip(self, tmp_path):
        """zip 模式：成员路径与内容与目录版逐字节一致（Kaggle 网页可直接上传 zip）。"""
        import hashlib
        import zipfile
        from kaggle.prepare_dataset_a import build_dataset_a, make_zip
        nc = tmp_path / "nc.bin"
        nc.write_bytes(b"DATA")
        ds_dir = make_fake_dataset_dir(tmp_path / "ds")
        out = tmp_path / "out_zip"
        build_dataset_a(str(nc), str(ds_dir), str(out))
        zip_path = tmp_path / "kaggle_dataset_a.zip"
        make_zip(str(out), str(zip_path))

        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            assert "dataset/meta.json" in names
            assert "pipedcylinder2d.nc" in names or "nc.bin" in names
            got = zf.read("dataset/u.npy")
            assert hashlib.sha256(got).hexdigest() == \
                hashlib.sha256((ds_dir / "u.npy").read_bytes()).hexdigest()

    def test_multi_layout_and_manifest(self, tmp_path):
        """多数据集打包（票 07 延伸）：data/<nc> + datasets/<name>/ 布局 + manifest。"""
        import json
        from kaggle.prepare_dataset_a import build_dataset_a_multi
        nc1 = tmp_path / "boussinesq.nc"; nc1.write_bytes(b"NC1")
        nc2 = tmp_path / "cylinder2d.nc"; nc2.write_bytes(b"NC2")
        ds1 = make_fake_dataset_dir(tmp_path / "ds1")
        ds2 = make_fake_dataset_dir(tmp_path / "ds2")
        out = tmp_path / "out_multi"
        manifest = build_dataset_a_multi([(str(nc1), str(ds1)), (str(nc2), str(ds2))],
                                         str(out))
        for rel in ("data/boussinesq.nc", "data/cylinder2d.nc",
                    "datasets/ds1/dataset/meta.json", "datasets/ds2/dataset/meta.json",
                    "datasets/ds2/dataset/u.npy"):
            assert (out / rel).exists(), rel
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert m["total_bytes"] > 0
        for f in m["files"]:
            assert (out / f["path"]).exists()
        assert not (out / "dataset").exists()          # 单数据集布局不混用

    def test_multi_standard_prepare_layout(self, tmp_path):
        """标准 prepare_dataset 布局（<数据集名>/dataset/）：打包名取父目录名，
        zip 内路径 = datasets/<名>/dataset/...（与 config root 布局一致）。"""
        from kaggle.prepare_dataset_a import build_dataset_a_multi
        nc = tmp_path / "fourcenters2d.nc"; nc.write_bytes(b"NC")
        ds = make_fake_dataset_dir(tmp_path / "fourcenters2d" / "dataset")
        out = tmp_path / "out_std"
        manifest = build_dataset_a_multi([(str(nc), str(ds))], str(out))
        assert (out / "datasets" / "fourcenters2d" / "dataset" / "meta.json").exists()
        assert any(f["path"] == "datasets/fourcenters2d/dataset/u.npy"
                   for f in manifest["files"])

    def test_multi_skip_nc_layout(self, tmp_path):
        """--skip-nc 多数据集打包：无 data/（省空间），datasets/ 布局与 manifest 正常。"""
        import json
        from kaggle.prepare_dataset_a import build_dataset_a_multi
        nc = tmp_path / "boussinesq.nc"; nc.write_bytes(b"NC1")
        ds1 = make_fake_dataset_dir(tmp_path / "ds1")
        out = tmp_path / "out_skip"
        manifest = build_dataset_a_multi([(str(nc), str(ds1))], str(out),
                                         include_nc=False)
        assert not (out / "data").exists()
        assert (out / "datasets" / "ds1" / "dataset" / "meta.json").exists()
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert m["include_nc"] is False
        assert all("data/" not in f["path"] for f in m["files"])

    def test_multi_pairs_mismatch_raises(self, tmp_path):
        """--nc 与 --dataset-dir 个数不匹配 → ValueError（防错配静默）。"""
        from kaggle.prepare_dataset_a import main as pa_main
        with pytest.raises(ValueError):
            pa_main(["--nc", "a.nc", "b.nc", "--dataset-dir", "ds1", "--out", "x"])


# ================================================================ 切片 I：挂载布局探测（kaggle/mount_probe.py，票 07 延伸运行反馈）

class TestMountProbe:
    """Dataset A 挂载布局探测（Kaggle 多级嵌套 /kaggle/input/datasets/<owner>/<slug>/）。

    票 07 三期实测：挂载 = {root}/datasets/<owner>/<slug>/...（多级嵌套）；本切片
    用 tmp 树复现嵌套与浅层两种布局，守护 probe_layout 深度优先命中。
    """

    @staticmethod
    def _write_fake_meta(dataset_dir):
        import json
        (dataset_dir / "meta.json").write_text(
            json.dumps({"slices": {"train": [0, 1], "test": [1, 2]}}),
            encoding="utf-8")

    def test_nested_multi_layout(self, tmp_path):
        """Kaggle 嵌套挂载：root/datasets/ziyixu317/dataset-a-multi/{data,datasets/<名>/dataset}。"""
        from kaggle.mount_probe import probe_layout
        mount = tmp_path / "datasets" / "ziyixu317" / "dataset-a-multi"
        (mount / "data").mkdir(parents=True)
        (mount / "data" / "boussinesq.nc").write_bytes(b"x")
        for name in ("boussinesq", "pipedcylinder2d"):
            d = mount / "datasets" / name / "dataset"
            d.mkdir(parents=True)
            self._write_fake_meta(d)
        single, multi = probe_layout(str(tmp_path))
        assert multi == mount
        assert single is None

    def test_nested_single_layout(self, tmp_path):
        """Kaggle 嵌套单数据集：root/datasets/owner/2d-.../{<nc>,dataset/meta.json}。"""
        from kaggle.mount_probe import probe_layout
        mount = tmp_path / "datasets" / "ziyixu317" / "2d-unsteady-cylinder"
        (mount / "dataset").mkdir(parents=True)
        (mount / "pipedcylinder2d.nc").write_bytes(b"x")
        self._write_fake_meta(mount / "dataset")
        single, multi = probe_layout(str(tmp_path))
        assert single == mount
        assert multi is None

    def test_shallow_single_layout(self, tmp_path):
        """浅层单数据集（root/dataset/meta.json）同样命中。"""
        from kaggle.mount_probe import probe_layout
        (tmp_path / "dataset").mkdir()
        self._write_fake_meta(tmp_path / "dataset")
        single, multi = probe_layout(str(tmp_path))
        assert single == tmp_path and multi is None

    def test_no_layout_returns_none(self, tmp_path):
        """无命中 → (None, None)（调用方走 zip 解压回退/fail loud）。"""
        from kaggle.mount_probe import probe_layout
        (tmp_path / "anything").mkdir()
        assert probe_layout(str(tmp_path)) == (None, None)


# ================================================================ 切片 F：Notebook 环境自检（kaggle/self_check.py）

class TestSelfCheck:
    """验收 1（Notebook 环境 import vendor + 数据加载通过）的本地可验证实现。

    Kaggle 端真实运行 = 用户执行（README §运行）；此处验证模块在任意
    数据目录上返回正确的检查结论（合成数据，CPU）。
    """

    def test_self_check_all_passes(self, tmp_path):
        """完整自检：vendor 构建 + 前向 (B,256) 域(0,1) + on-the-fly 样本有限 + 标签 0/1。"""
        import dataset as ds
        import test_train as tt
        from kaggle.self_check import self_check
        root = tmp_path / "ds_self"
        u, v, xdim, ydim, tdim = tds.synth_prepared(root, T=40)
        ds.prepare_dataset(None, str(root), u=u, v=v, xdim=xdim, ydim=ydim,
                           tdim=tdim, taus={"train": tds.SYNTH_TAU})
        r = self_check(str(root), model_cfg=tt.make_small_model_cfg(),
                       n_samples=4)
        assert r["config_ok"] is True
        assert r["model_forward_ok"] is True
        assert r["dataset_ok"] is True
        assert r["n_samples"] == 4
        assert 0 < r["label_sum"] <= r["n_samples"] * 256   # 标签非全 0（有正样本）

    def test_self_check_missing_data_root_raises(self, tmp_path):
        """数据根缺失 → FileNotFoundError（自检必须失败闭，不静默通过）。"""
        import test_train as tt
        from kaggle.self_check import self_check
        with pytest.raises(FileNotFoundError):
            self_check(str(tmp_path / "nope"),
                       model_cfg=tt.make_small_model_cfg(), n_samples=1)


# ================================================================ 切片 G：TF32 与中途评估入口

class TestTf32AndMidwayEval:
    """步速校准的工程参数落地（票 07 验收 2）：TF32 加速 + 中途 F1 评估入口。"""

    def test_enable_tf32_sets_flags(self):
        """enable_tf32：matmul/cudnn allow_tf32 置 True（T4 张量核；数值仍 fp32 语义）。"""
        from train_kaggle import enable_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        enable_tf32()
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True

    def test_eval_only_when_epochs_equals_progress(self, tmp_path):
        """中途评估入口：--epochs == 续训进度时训练循环为空 → 仅执行 --report-f1。

        行为证据：先训 1 epoch（progress=1），再以 --epochs 1 --report-f1 运行 →
        不训练（不报 loader 空），写 val_f1.json（epoch 字段 = 已完成进度-1）。
        """
        import json
        from train_kaggle import main
        cfg_path, ckpts = TestReportF1CLI.make_val_cfg(tmp_path, epochs=1)
        assert main(["--config", str(cfg_path), "--max-steps", "2"]) == 0
        assert main(["--config", str(cfg_path), "--max-steps", "2",
                     "--epochs", "1", "--report-f1"]) == 0
        f1_path = ckpts / "pathline_transformer_cylinder_val_f1.json"
        assert f1_path.exists()
        data = json.loads(f1_path.read_text(encoding="utf-8"))
        assert data["epoch"] == 0
        assert data["n"] > 0


# ================================================================ 切片 H：单帧预览（kaggle/preview_eval.py）

class TestProjectToGrid:
    """逐迹线概率 → 网格投影（预览版；正式滑窗/TTA/定量表属票 08）。"""

    def test_projection_accumulate_and_average(self):
        """投影 = 累积 + 计数平均（重叠格取均值）；字面量手算（独立来源）。

        seeds 行 = [x, y]（nearest_cell 口径）：三粒种子 → 格 (j,i) =
        (2,1)、(2,1)（重叠）、(0,3)。
        """
        from kaggle.preview_eval import project_to_grid
        xdim = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ydim = np.array([0.0, 1.0, 2.0, 3.0])
        shape = (4, 5)
        seeds = np.array([[0.7, 2.4], [0.8, 2.2], [3.3, 0.1]])
        preds = np.array([0.9, 0.5, 0.2])
        prob = project_to_grid(preds, seeds, xdim, ydim, shape)
        assert prob[2, 1] == pytest.approx((0.9 + 0.5) / 2)   # 重叠格平均
        assert prob[0, 3] == pytest.approx(0.2)
        assert prob[0, 0] == 0.0                              # 无迹线格为 0
        assert float(np.nansum(prob)) == pytest.approx(0.7 + 0.2)

    def test_preview_main_writes_png(self, tmp_path):
        """端到端预览：合成数据集 + 小模型 ckpt → 单帧对比图 png 落盘（4 联布局）。"""
        import dataset as ds
        import test_train as tt
        from train_kaggle import load_config, main as train_main
        from kaggle.preview_eval import main as preview_main
        root = tmp_path / "ds_prev"
        u, v, xdim, ydim, tdim = tds.synth_prepared(root, T=48)
        slices = {"train": (0, 24), "val": (24, 48)}
        ds.prepare_dataset(None, str(root), u=u, v=v, xdim=xdim, ydim=ydim,
                           tdim=tdim,
                           taus={"train": tds.SYNTH_TAU, "val": tds.SYNTH_TAU},
                           slices=slices)
        cfg_path = tmp_path / "prev.yaml"
        cfg = load_config()
        cfg["data"]["root"] = str(root)
        cfg["data"]["num_workers"] = 0
        cfg["data"]["samples_per_epoch"] = 8
        cfg["data"]["batch_size"] = 4
        cfg["train"]["epochs"] = 1
        cfg["train"]["val_freq"] = 1
        cfg["train"]["seed"] = 0
        cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpts_prev")
        cfg["model"]["encoder_args"].update(tt.make_small_model_cfg())
        with open(cfg_path, "w", encoding="utf-8") as f:
            import yaml
            yaml.safe_dump(cfg, f)
        train_main(["--config", str(cfg_path), "--max-steps", "2"])

        out_png = tmp_path / "preview_t_24.png"
        assert preview_main(["--config", str(cfg_path), "--ckpt",
                             str(tmp_path / "ckpts_prev"
                                 / "pathline_transformer_cylinder_ckpt_latest.pth"),
                             "--frame", "24", "--out", str(out_png)]) == 0
        assert out_png.exists() and out_png.stat().st_size > 1000
