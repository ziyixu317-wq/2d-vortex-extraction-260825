"""06 票：训练脚本（train_kaggle.py）测试 —— 模型缝（训练循环/调度/checkpoint/配置）。

领域词汇（HANDOFF §2/§6 与规格 Implementation Decisions，唯一权威）：
- TwoStep 调度：warmup 60 epoch @1e-4 → 5e-6（两段常数阶梯，epoch 粒度；
  语义与原仓库 TwoStepLRScheduler 一致：epoch<60 用 lr，epoch≥60 用 second_lr）；
- AdamW(wd 1e-6)、lr 1e-4、batch 100、200 epoch、梯度裁剪 1.0、BCE 损失；
- 每 epoch 存 checkpoint（含 optimizer 状态）；断点续训从 checkpoint['epoch']+1 恢复；
- 全部超参走 YAML 配置（config/pathline_transformer_cylinder.yaml）；
- CPU 冒烟：1~2 步训练 loss 数值有限且形状正确（下降趋势为 Kaggle 全量观察项）。

期望值来源（独立于实现）：HANDOFF §2 训练超参字面量（论文附录 C）、
HANDOFF §6 参数表、规格 Implementation Decisions。

合成数据工具复用 test_dataset.py（同目录同包；Rankine 涡场 + prepare_dataset）。
"""

import pathlib

import numpy as np
import pytest
import torch

import test_dataset as tds  # 复用合成场构造（Rankine 涡 + 已知 τ 字面量）


# ================================================================ 通用 fixture

@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    """合成数据集（prepare_dataset 产物）——测试训练循环/checkpoint 用。"""
    import dataset as ds
    root = tmp_path_factory.mktemp("train_ds") / "ds"
    u, v, xdim, ydim, tdim = tds.synth_prepared(root)
    ds.prepare_dataset(None, str(root), u=u, v=v, xdim=xdim, ydim=ydim,
                       tdim=tdim, taus={"train": tds.SYNTH_TAU})
    return root


def make_small_model_cfg():
    """测试用小模型配置（dmodel/组数/层数缩小，CPU 冒烟快；语义与生产 cfg 同构）。"""
    return {
        "NAME": "PathlineTransformerV0",
        "in_channels": 7,
        "PathlineGroups": 16,
        "KpathlinePerGroup": 4,
        "num_classes": 1,
        "num_encoder_layers": 1,
        "dmodel": 32,
        "k": 8,
    }


def fresh_small_model():
    """测试用小模型实例（每次新初始化，测试间互不污染）。"""
    from vendor.DeepUtils.models import build_model_from_cfg
    return build_model_from_cfg(make_small_model_cfg())


def fresh_adamw(model, lr=1e-4, weight_decay=1e-6):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def assert_forward_shape_and_range(model, B=1, L=16, K=256, C=7):
    """模型缝前向断言（两张量输入）：输出 (B, K) 概率、float32、数值域 (0,1)。"""
    x = torch.rand(B, L, K, C)
    model.eval()
    with torch.no_grad():
        out = model((torch.zeros(B, 1, 1, 1), x))
    assert out.shape == (B, K)
    assert out.dtype == torch.float32
    assert (out > 0).all() and (out < 1).all()


# ================================================================ 切片 A：TwoStep 调度

class TestTwoStepLR:
    """TwoStep：epoch<warmup 用 lr、epoch≥warmup 用 second_lr（两段常数阶梯）。"""

    LR = 1e-4
    SECOND_LR = 5e-6
    WARMUP = 60

    def test_schedule_values(self):
        """调度值 = HANDOFF §2 字面量：epoch 0/59 → 1e-4；60/199 → 5e-6。"""
        from train_kaggle import two_step_lr
        assert two_step_lr(0, self.LR, self.SECOND_LR, self.WARMUP) == pytest.approx(1e-4)
        assert two_step_lr(59, self.LR, self.SECOND_LR, self.WARMUP) == pytest.approx(1e-4)
        assert two_step_lr(60, self.LR, self.SECOND_LR, self.WARMUP) == pytest.approx(5e-6)
        assert two_step_lr(199, self.LR, self.SECOND_LR, self.WARMUP) == pytest.approx(5e-6)

    def test_wrapper_sets_optimizer_lr(self):
        """TwoStepLR.step(epoch) 驱动 optimizer.param_groups lr（行为：优化器策略生效）。"""
        from train_kaggle import TwoStepLR
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=self.LR)
        sched = TwoStepLR(opt, lr=self.LR, second_lr=self.SECOND_LR,
                          warmup_epochs=self.WARMUP)
        sched.step(0)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)
        sched.step(59)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)
        sched.step(60)
        assert opt.param_groups[0]["lr"] == pytest.approx(5e-6)

    def test_state_roundtrip(self):
        """调度器状态往返（checkpoint 支持：state 含 epoch，恢复后阶梯位置一致）。"""
        from train_kaggle import TwoStepLR
        sched = TwoStepLR(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))],
                                            lr=self.LR),
                          lr=self.LR, second_lr=self.SECOND_LR, warmup_epochs=self.WARMUP)
        sched.step(70)                                    # 已过 60 → 第二段
        state = sched.state_dict()
        fresh = TwoStepLR(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))],
                                            lr=self.LR),
                          lr=self.LR, second_lr=self.SECOND_LR, warmup_epochs=self.WARMUP)
        fresh.load_state_dict(state)
        fresh.step(fresh.state_dict()["epoch"] + 1)       # 续训起点 = 恢复 epoch+1
        assert fresh.state_dict()["epoch"] == 71
        assert fresh.get_lr(71) == pytest.approx(5e-6)    # 第二段（5e-6）

    def test_warmup_boundary_inclusive(self):
        """边界语义：epoch=60 恰好进入第二段（warmup 60 epoch = epoch 0..59 用 lr）。"""
        from train_kaggle import two_step_lr
        assert two_step_lr(59, self.LR, self.SECOND_LR, self.WARMUP) == pytest.approx(1e-4)
        assert two_step_lr(60, self.LR, self.SECOND_LR, self.WARMUP) == pytest.approx(self.SECOND_LR)


# ================================================================ 切片 B：YAML 配置驱动 + 模型构建

class TestConfigDriven:
    """验收 4：全部超参走 YAML 配置（config/pathline_transformer_cylinder.yaml）。

    期望值 = HANDOFF §2/§6 与规格字面量（论文附录 C 超参、决策 6 口径）。
    """

    def test_default_config_path_exists(self):
        from train_kaggle import DEFAULT_CONFIG_PATH
        assert pathlib.Path(DEFAULT_CONFIG_PATH).exists()

    def test_config_hyperparams_match_spec(self):
        """关键超参字段与规格字面量一致（模型/训练/数据三段）。"""
        from train_kaggle import load_config
        cfg = load_config()
        m = cfg["model"]["encoder_args"]
        assert m["NAME"] == "PathlineTransformerV0"
        assert m["in_channels"] == 7
        assert m["PathlineGroups"] == 64                 # 决策 6：64 组
        assert m["KpathlinePerGroup"] == 4               # 决策 6：4 卫星（不含中心）
        assert m["num_encoder_layers"] == 3
        assert m["k"] == 16
        assert cfg["model"]["criterion_args"]["NAME"] == "BCELoss"
        t = cfg["train"]
        assert t["lr"] == pytest.approx(1e-4)
        assert t["weight_decay"] == pytest.approx(1e-6)
        assert t["warmup_epochs"] == 60
        assert t["second_lr"] == pytest.approx(5e-6)
        assert t["grad_clip"] == pytest.approx(1.0)
        assert t["epochs"] == 200
        d = cfg["data"]
        assert d["batch_size"] == 100
        assert d["samples_per_epoch"] >= 20000           # HANDOFF §6 下限
        assert d["t_win"] == 24
        assert d["t_scale"] == pytest.approx(0.25)
        assert d["patch_size"] == [32, 32]
        assert d["stride"] == [16, 16]
        assert d["groups"] == [8, 8]                     # 十字采样组网格
        assert d["delta_frac"] == pytest.approx(0.05)
        assert d["L"] == 16
        assert d["n_substeps"] == 4

    def test_make_dataset_consumes_yaml_params(self, synth_root):
        """验收 4 集成面：data 段 patch/窗口/十字采样/时间采样字段全部进入数据集构造。

        回归背景（Spec 轴审查）：data.patch_size/stride/t_win/window_step 曾为
        死配置（YAML 改值不生效）；groups/delta_frac/L/n_substeps 曾不进 YAML。
        """
        from train_kaggle import _make_dataset, load_config
        cfg = load_config()
        cfg["data"]["root"] = str(synth_root)
        d = _make_dataset(cfg["data"], "train")
        assert d.patch_size == tuple(cfg["data"]["patch_size"])
        assert d.stride == tuple(cfg["data"]["stride"])
        assert d.t_win == cfg["data"]["t_win"]
        assert d.window_step == cfg["data"]["window_step"]
        assert d.t_scale == pytest.approx(cfg["data"]["t_scale"])
        assert d.groups == tuple(cfg["data"]["groups"])
        assert d.delta_frac == pytest.approx(cfg["data"]["delta_frac"])
        assert d.L == cfg["data"]["L"]
        assert d.n_substeps == cfg["data"]["n_substeps"]

    def test_build_model_from_config(self):
        """从配置构建模型：YAML model 段（BaseSeg+PathlineTransformerV0 生产口径，
        dmodel=144/3 层/k=16）前向 → (B, 256) 概率域 (0,1)（模型缝行为）。"""
        from train_kaggle import build_model_from_config, load_config
        model = build_model_from_config(load_config())
        assert isinstance(model, torch.nn.Module)
        assert_forward_shape_and_range(model)


# ================================================================ 切片 C：训练循环（CPU 冒烟）

def make_train_loader(synth_root, samples_per_epoch=12, batch_size=4, seed=0):
    """合成数据集 DataLoader（num_workers=0：Windows 多进程 spawn 不可靠）。"""
    from torch.utils.data import DataLoader
    import dataset as ds
    d = ds.WeakLabelPathlineDataset(str(synth_root), split="train",
                                    samples_per_epoch=samples_per_epoch, seed=seed)
    d.set_epoch(0)
    return DataLoader(d, batch_size=batch_size, num_workers=0)


class TestTrainLoop:
    """验收 1：CPU 冒烟 1~2 步训练 loss 数值有限且形状正确（下降趋势为观察项）。

    行为断言：loss 为非负有限标量；参数经梯度更新（训练确实发生）。
    """

    def test_run_epoch_two_steps(self, synth_root):
        from train_kaggle import (build_criterion_from_config, load_config, run_epoch)
        model = fresh_small_model()
        cfg = load_config()
        criterion = build_criterion_from_config(cfg)
        opt = fresh_adamw(model)
        loader = make_train_loader(synth_root)
        before = [p.detach().clone() for p in model.parameters()]
        loss = run_epoch(model, loader, criterion, opt, device="cpu",
                         grad_clip=1.0, max_steps=2)
        assert np.isfinite(loss) and loss > 0
        assert any(not torch.equal(a, b) for a, b in
                   zip(before, [p.detach() for p in model.parameters()]))
        # 形状正确：BCE 损失为标量（run_epoch 返回均值浮点）；模型输出 (B,256) 已由前向测试守护

    def test_run_epoch_grad_clip_applied(self, synth_root):
        """梯度裁剪 1.0 生效：训练一步后参数**全局梯度范数** ≤ 裁剪值 + 容差。

        裁剪语义（torch.nn.utils.clip_grad_norm_）：全体参数梯度拼接的全局范数
        受限（单参数个体范数可 >1.0，断言全局范数才是梯度裁剪的独立来源语义）。
        """
        from train_kaggle import (build_criterion_from_config, load_config, run_epoch)
        model = fresh_small_model()
        criterion = build_criterion_from_config(load_config())
        opt = fresh_adamw(model)
        loader = make_train_loader(synth_root)
        run_epoch(model, loader, criterion, opt, device="cpu", grad_clip=1.0, max_steps=1)
        grads = [g for g in model.parameters() if g.grad is not None]
        total_norm = float(sum(float(g.grad.norm()) ** 2 for g in grads)) ** 0.5
        assert total_norm <= 1.0 + 1e-3, f"全局梯度范数 {total_norm:.3f} > 1.0"

    def test_evaluate_loss_finite(self, synth_root):
        """evaluate：val 损失非负有限（无梯度更新；模型缝观察指标）。"""
        from train_kaggle import (build_criterion_from_config, evaluate, load_config)
        model = fresh_small_model()
        criterion = build_criterion_from_config(load_config())
        loader = make_train_loader(synth_root)
        loss = evaluate(model, loader, criterion, device="cpu", max_steps=2)
        assert np.isfinite(loss) and loss >= 0


# ================================================================ 切片 D：checkpoint 往返 + 断点续训

class TestCheckpoint:
    """验收 2/3：checkpoint 保存/加载往返一致（含 optimizer 状态）；从指定 epoch 恢复。"""

    def test_roundtrip_model_and_optimizer(self, synth_root, tmp_path):
        """往返一致：model/optimizer state_dict 逐键相等（含 AdamW 动量状态）。

        期望值 = 保存前状态本身（往返一致性；不是重算实现路径）。
        """
        from train_kaggle import (load_ckpt, run_epoch, save_ckpt,
                                  build_criterion_from_config, load_config,
                                  TwoStepLR)
        model = fresh_small_model()
        criterion = build_criterion_from_config(load_config())
        opt = fresh_adamw(model)
        sched = TwoStepLR(opt, lr=1e-4, second_lr=5e-6, warmup_epochs=60)
        loader = make_train_loader(synth_root)
        run_epoch(model, loader, criterion, opt, device="cpu", grad_clip=1.0, max_steps=3)
        sched.step(70)
        ckpt = tmp_path / "run_ckpt_latest.pth"
        save_ckpt(ckpt, model, opt, sched, epoch=70, metrics={"train_loss": 0.5})

        model2 = fresh_small_model()
        opt2 = fresh_adamw(model2)
        sched2 = TwoStepLR(opt2, lr=1e-4, second_lr=5e-6, warmup_epochs=60)
        start_epoch, metrics = load_ckpt(ckpt, model2, opt2, sched2)
        assert start_epoch == 71                        # 从 checkpoint['epoch']+1 恢复
        assert metrics["train_loss"] == pytest.approx(0.5)
        for (k1, v1), (k2, v2) in zip(model.state_dict().items(),
                                      model2.state_dict().items()):
            assert k1 == k2 and torch.equal(v1, v2)
        s1, s2 = opt.state_dict(), opt2.state_dict()
        assert s1["param_groups"][0]["weight_decay"] == s2["param_groups"][0]["weight_decay"]
        assert set(s1["state"].keys()) == set(s2["state"].keys())
        for k in s1["state"]:
            d1, d2 = s1["state"][k], s2["state"][k]
            assert set(d1.keys()) == set(d2.keys())
            for name in d1:
                v1, v2 = d1[name], d2[name]
                # 逐张量比较（含 AdamW 动量 exp_avg/exp_avg_sq 与 step 计数）
                if isinstance(v1, torch.Tensor):
                    assert torch.equal(v1, v2)
                else:
                    assert v1 == v2

    def test_resume_continues_training(self, synth_root, tmp_path):
        """断点续训从指定 epoch 恢复后继续训练：loss 有限、lr 处于恢复位置。

        行为证据：恢复 epoch=70（第二段）→ 续训步后 lr=5e-6（TwoStep 位置正确）。
        """
        from train_kaggle import (TwoStepLR, build_criterion_from_config, evaluate,
                                  load_ckpt, load_config, run_epoch, save_ckpt)
        model = fresh_small_model()
        criterion = build_criterion_from_config(load_config())
        opt = fresh_adamw(model)
        sched = TwoStepLR(opt, lr=1e-4, second_lr=5e-6, warmup_epochs=60)
        sched.step(70)
        ckpt = tmp_path / "resume_ckpt.pth"
        save_ckpt(ckpt, model, opt, sched, epoch=70)

        model2 = fresh_small_model()
        opt2 = fresh_adamw(model2)
        sched2 = TwoStepLR(opt2, lr=1e-4, second_lr=5e-6, warmup_epochs=60)
        start_epoch, _ = load_ckpt(ckpt, model2, opt2, sched2)
        assert start_epoch == 71
        sched2.step(71)
        assert sched2.get_lr(71) == pytest.approx(5e-6)  # 恢复后仍在第二段
        loader = make_train_loader(synth_root)
        loss = run_epoch(model2, loader, criterion, opt2, device="cpu",
                         grad_clip=1.0, max_steps=2)
        assert np.isfinite(loss) and loss >= 0
        val_loss = evaluate(model2, loader, criterion, device="cpu", max_steps=2)
        assert np.isfinite(val_loss)

    def test_ckpt_contains_epoch_and_metric_metadata(self, tmp_path):
        """checkpoint 自描述：epoch/metrics/config 元数据落盘（审计/归档可追溯）。"""
        from train_kaggle import load_ckpt, save_ckpt
        model = torch.nn.Linear(1, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ckpt = tmp_path / "meta_ckpt.pth"
        save_ckpt(ckpt, model, opt, epoch=12, metrics={"train_loss": 0.3},
                  config={"run_name": "smoke"})
        blob = torch.load(ckpt, map_location="cpu")
        assert blob["epoch"] == 12
        assert blob["metrics"]["train_loss"] == pytest.approx(0.3)
        assert blob["config"]["run_name"] == "smoke"


# ================================================================ 切片 E：main CLI 集成冒烟（验收 1 + 3 端到端）

class TestMainCLI:
    """验收 1/3 的 CLI 层证据：真实主循环 1 epoch 冒烟 + 断点续训从恢复 epoch 继续。

    数据/超参全部来自（临时）YAML（验收 4 的集成面）；合成场无 val 片 →
    main 跳过 val（防御路径，Kaggle 数据完整）。
    """

    @staticmethod
    def write_cfg(cfg_path, cfg):
        import yaml
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)

    def test_smoke_epoch_then_resume(self, synth_root, tmp_path):
        from train_kaggle import load_config, main
        cfg_path = tmp_path / "train.yaml"
        cfg = load_config()
        cfg["data"]["root"] = str(synth_root)
        cfg["data"]["num_workers"] = 0
        cfg["data"]["samples_per_epoch"] = 12
        cfg["data"]["batch_size"] = 4
        cfg["train"]["epochs"] = 1
        cfg["train"]["val_freq"] = 1
        cfg["train"]["save_freq"] = 1
        cfg["train"]["seed"] = 0
        cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpts")
        cfg["model"]["encoder_args"].update(make_small_model_cfg())
        self.write_cfg(cfg_path, cfg)

        assert main(["--config", str(cfg_path), "--max-steps", "2"]) == 0
        root = tmp_path / "ckpts"
        latest = root / "pathline_transformer_cylinder_ckpt_latest.pth"
        assert latest.exists()
        blob = torch.load(latest, map_location="cpu")
        assert blob["epoch"] == 0
        assert (root / "pathline_transformer_cylinder_E1.pth").exists()  # milestone

        # 断点续训：同配置 epochs=2 → resume(auto) 从 epoch 1 继续，latest 更新到 epoch 1
        cfg["train"]["epochs"] = 2
        self.write_cfg(cfg_path, cfg)
        assert main(["--config", str(cfg_path), "--max-steps", "2"]) == 0
        blob2 = torch.load(latest, map_location="cpu")
        assert blob2["epoch"] == 1
        assert blob2["metrics"].get("train_loss", 0.0) >= 0
        assert np.isfinite(blob2["metrics"]["train_loss"])

    def test_deterministic_seed_reproducibility(self, synth_root, tmp_path):
        """同 seed + 同配置两次运行 → 首 epoch 损失一致（可复现冒烟；Kaggle 观察项）。"""
        from train_kaggle import load_config, main
        losses = []
        for tag in ("a", "b"):
            cfg_path = tmp_path / f"train_{tag}.yaml"
            cfg = load_config()
            cfg["data"]["root"] = str(synth_root)
            cfg["data"]["num_workers"] = 0
            cfg["data"]["samples_per_epoch"] = 8
            cfg["data"]["batch_size"] = 4
            cfg["train"]["epochs"] = 1
            cfg["train"]["val_freq"] = 1
            cfg["train"]["seed"] = 7
            cfg["train"]["ckpt_dir"] = str(tmp_path / f"ckpts_{tag}")
            cfg["model"]["encoder_args"].update(make_small_model_cfg())
            self.write_cfg(cfg_path, cfg)
            main(["--config", str(cfg_path), "--max-steps", "2"])
            blob = torch.load(tmp_path / f"ckpts_{tag}"
                              / "pathline_transformer_cylinder_ckpt_latest.pth",
                              map_location="cpu")
            losses.append(blob["metrics"]["train_loss"])
        assert losses[0] == pytest.approx(losses[1], rel=1e-6)

    def test_val_path_when_split_present(self, tmp_path):
        """val 时间片存在时：main 构建 val 数据集并评估（val_loss 入 metrics）。

        回归背景：真实数据冒烟发现 val_ds 未 set_epoch → RuntimeError
        （合成场只有 train 片时走跳过分支，此路径先前无覆盖）。
        """
        import dataset as ds
        from train_kaggle import load_config, main
        root = tmp_path / "ds_val"
        u, v, xdim, ydim, tdim = tds.synth_prepared(root, T=48)
        # 两片各 24 帧 ≥ 窗口 T_win=24（窗口起点 [片首]，各片 9 个 patch 组合）
        slices = {"train": (0, 24), "val": (24, 48)}
        ds.prepare_dataset(None, str(root), u=u, v=v, xdim=xdim, ydim=ydim,
                           tdim=tdim,
                           taus={"train": tds.SYNTH_TAU, "val": tds.SYNTH_TAU},
                           slices=slices)
        cfg_path = tmp_path / "train_val.yaml"
        cfg = load_config()
        cfg["data"]["root"] = str(root)
        cfg["data"]["num_workers"] = 0
        cfg["data"]["samples_per_epoch"] = 8
        cfg["data"]["batch_size"] = 4
        cfg["train"]["epochs"] = 1
        cfg["train"]["val_freq"] = 1
        cfg["train"]["seed"] = 0
        cfg["train"]["ckpt_dir"] = str(tmp_path / "ckpts_val")
        cfg["model"]["encoder_args"].update(make_small_model_cfg())
        self.write_cfg(cfg_path, cfg)
        assert main(["--config", str(cfg_path), "--max-steps", "2"]) == 0
        blob = torch.load(tmp_path / "ckpts_val"
                          / "pathline_transformer_cylinder_ckpt_latest.pth",
                          map_location="cpu")
        assert "val_loss" in blob["metrics"]
        assert np.isfinite(blob["metrics"]["val_loss"])

    def test_small_model_forward_shape_and_range(self):
        """模型缝前向（小模型快速冒烟）：输入 (B,L,K,C) → 输出 (B,256) 概率域 (0,1)。"""
        assert_forward_shape_and_range(fresh_small_model(), B=2)
