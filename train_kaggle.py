"""训练脚本（train_kaggle.py）——06 票：自写训练循环（不依赖参考仓库任何训练代码）。

领域词汇（HANDOFF §2/§6，唯一权威）：
- 训练超参（论文附录 C）：AdamW(wd 1e-6)、lr 1e-4、TwoStep 调度（warmup 60 epoch
  维持 lr → 5e-6，两段常数阶梯）、batch 100、200 epoch、梯度裁剪 1.0、BCE 损失；
- 每 epoch 存 checkpoint（含 optimizer 状态）支持断点续训（Kaggle 12h 会话硬上限）；
- 可选 DataParallel/AMP（Kaggle T4×2 单机；本地 CPU 无 CUDA 不启用）；
- 全部超参走 YAML 配置（config/pathline_transformer_cylinder.yaml）；
- 数据集 = dataset.WeakLabelPathlineDataset：set_epoch(epoch) 重建 50% 正样本
  过采样序（同 (seed, epoch) 确定性 → 断点续训的采样序与中断前逐样本一致）；
- 返回 ((dummy_field, pathlines), labels) 匹配 PathlineTransformerV0（sigmoid 输出
  每迹线涡概率 (B, K=256)）；损失 BCE（torch.nn.BCELoss）。

实现约束：纯 torch/numpy/yaml（§2 依赖清单）；不 import 原仓库 train.py。
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn
import yaml

import dataset as ds
from vendor.DeepUtils.models import build_model_from_cfg
from vendor.DeepUtils.loss import build_criterion_from_cfg
from vendor.DeepUtils.utils.random import set_random_seed

DEFAULT_CONFIG_PATH = "config/pathline_transformer_cylinder.yaml"


def enable_tf32():
    """启用 TF32 matmul/卷积（T4 tensor core 加速，数值仍为 fp32 语义）。

    2026-08-25 票 07 步速校准：T4×2 全精度实测 ~5s/步（上游 KNN 暴力 O(N²) +
    DP 同步 + T4 fp32 无张量核加速；torch 2.10 默认 allow_tf32=False），
    启用 TF32 预期 1.5~2×。训练界标准做法（matmul 尾数 10 位，收敛行为
    与 fp32 一致）；论文口径的可复现性不受影响（同环境确定性一致）。
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------- TwoStep 调度

def two_step_lr(epoch, lr, second_lr, warmup_epochs):
    """TwoStep 阶梯（原仓库 TwoStepLRScheduler 语义）：epoch < warmup_epochs 用 lr，
    epoch ≥ warmup_epochs 用 second_lr。epoch 从 0 计（epoch 粒度，sched_on_epoch）。
    """
    return lr if epoch < warmup_epochs else second_lr


class TwoStepLR:
    """epoch 粒度的 TwoStep 调度器包装：step(epoch) 设置 optimizer 各参数组 lr。

    状态 = 当前 epoch（checkpoint 往返；恢复后从 epoch+1 续调）。
    """

    def __init__(self, optimizer, lr, second_lr, warmup_epochs):
        self.optimizer = optimizer
        self.lr = float(lr)
        self.second_lr = float(second_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.epoch = 0

    def get_lr(self, epoch):
        return two_step_lr(epoch, self.lr, self.second_lr, self.warmup_epochs)

    def step(self, epoch):
        self.epoch = int(epoch)
        lr = self.get_lr(self.epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self):
        return {"epoch": self.epoch}

    def load_state_dict(self, state):
        self.epoch = int(state["epoch"])


# --------------------------------------------------------------------------- 配置与构建

def load_config(path=DEFAULT_CONFIG_PATH):
    """YAML 配置 → dict（训练/数据/模型全部超参单一来源）。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "model" not in cfg or "train" not in cfg:
        raise ValueError(f"配置缺少 model/train 段: {path}")
    return cfg


def build_model_from_config(config):
    """配置 model 段 → 模型（vendor build_model_from_cfg；BaseSeg 包装可复用官方形态）。"""
    return build_model_from_cfg(config["model"])


def build_criterion_from_config(config):
    """配置 model.criterion_args → 损失（BCELoss：模型 sigmoid 输出 + 0/1 标签）。"""
    return build_criterion_from_cfg(config["model"]["criterion_args"])


def _make_dataset(data_cfg, split):
    """YAML data 段 → WeakLabelPathlineDataset/MultiDatasetPathlineDataset。

    data.root 为单个目录 → 单数据集（票 05 口径）；为目录列表 → 多数据集联合池
    （票 07 延伸：各数据集前 60% 一起训练/后 40% 留出评估）。patch/窗口/十字
    采样/时间采样参数全部从 YAML 传入（HANDOFF §6 参数表；与 prepare_dataset
    口径一致）。值监控口径：val 损失按训练同款 50% 平衡采样计算；自然分布
    精度评估属 --report-f1/票 08 弱定量表。
    """
    common = dict(
        patch_size=tuple(int(v) for v in data_cfg.get("patch_size", ds.DEFAULT_PATCH_SIZE)),
        stride=tuple(int(v) for v in data_cfg.get("stride", ds.DEFAULT_STRIDE)),
        t_win=int(data_cfg.get("t_win", ds.DEFAULT_T_WIN)),
        window_step=int(data_cfg.get("window_step", ds.DEFAULT_WINDOW_STEP)),
        samples_per_epoch=int(data_cfg["samples_per_epoch"]),
        positive_fraction=float(data_cfg.get("positive_fraction", 0.5)),
        t_scale=float(data_cfg.get("t_scale", 0.25)),
        seed=int(data_cfg.get("seed", 0)),
        groups=tuple(int(v) for v in data_cfg.get("groups", ds.DEFAULT_GROUPS)),
        delta_frac=float(data_cfg.get("delta_frac", ds.DEFAULT_DELTA_FRAC)),
        L=int(data_cfg.get("L", ds.DEFAULT_L)),
        n_substeps=int(data_cfg.get("n_substeps", 4)))
    root = data_cfg.get("root", "outputs/dataset")
    if isinstance(root, (list, tuple)):
        return ds.MultiDatasetPathlineDataset(list(root), split=split, **common)
    return ds.WeakLabelPathlineDataset(root, split=split, **common)


# --------------------------------------------------------------------------- 训练/评估循环

def _to_device(batch, device):
    """((dummy_field, pathlines), labels) → 张量搬到 device（labels 转 float32 供 BCE）。"""
    ((dummy, pathlines), labels) = batch
    return ((dummy.to(device), pathlines.to(device)), labels.to(device).float())


def _predict_loss(model, batch, criterion, device, use_amp=False):
    """一批 → BCE 损失（autocast 只包前向；AMP 与否单一路径，无分支漂移）。"""
    ((dummy, pathlines), labels) = _to_device(batch, device)
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        pred = model((dummy, pathlines))
    return criterion(pred, labels)


def _iter_batches(loader, max_steps):
    """loader 的前 max_steps 批（None=全部）；批数为 0 由调用方 raise（各自语义）。"""
    n = 0
    for batch in loader:
        if max_steps is not None and n >= max_steps:
            return
        n += 1
        yield batch


def run_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0,
              max_steps=None, scaler=None):
    """一个 epoch（或 max_steps 步）的训练：前向 → BCE → 反向 → 梯度裁剪 → step。

    返回平均 loss（标量 float）。梯度裁剪按 HANDOFF（grad_clip 默认 1.0）。
    AMP（scaler 非 None 且 cuda）时走 autocast + scaler.scale/step（Kaggle 可选）。
    """
    model.train()
    use_amp = scaler is not None and device.type == "cuda"
    total, n = 0.0, 0
    for batch in _iter_batches(loader, max_steps):
        optimizer.zero_grad()
        loss = _predict_loss(model, batch, criterion, device, use_amp)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += float(loss.detach().cpu())
        n += 1
    if n == 0:
        raise ValueError("loader 为空：训练循环无样本可跑")
    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_steps=None):
    """验证：evaluate 损失（模型缝观察指标；无梯度更新，确定性评估留 TTA/票 08）。"""
    model.eval()
    total, n = 0.0, 0
    for batch in _iter_batches(loader, max_steps):
        loss = _predict_loss(model, batch, criterion, device, use_amp=False)
        total += float(loss.detach().cpu())
        n += 1
    if n == 0:
        raise ValueError("loader 为空：评估循环无样本可跑")
    return total / n


@torch.no_grad()
def evaluate_f1(model, loader, device, threshold=0.5, max_steps=None):
    """自然分布 val/test 片上的 F1/IoU 评估（训练收尾记录，票 07 验收 4 + 票 07 延伸）。

    口径：模型输出 sigmoid 概率 > threshold 判正（默认 0.5）；与弱标签逐迹线
    0/1 求混淆矩阵 → precision/recall/F1/IoU（IoU = tp/(tp+fp+fn)，票 07 延伸
    留出评估指标）。自然分布指采样序的正负比例 = 池比例（非训练同款 50% 平衡；
    平衡采样是训练监控口径，自然分布为真实精度观察口径，正式弱定量表属票 08）。
    返回 dict：tp/fp/fn/tn/precision/recall/f1/iou/n（n = 评估迹线总数）。
    """
    model.eval()
    tp = fp = fn = tn = n = 0
    for batch in _iter_batches(loader, max_steps):
        ((dummy, pathlines), labels) = _to_device(batch, device)
        pred = model((dummy, pathlines))
        pred_b = pred > float(threshold)
        tp += int(((pred_b) & (labels == 1)).sum().item())
        fp += int(((pred_b) & (labels == 0)).sum().item())
        fn += int(((~pred_b) & (labels == 1)).sum().item())
        tn += int(((~pred_b) & (labels == 0)).sum().item())
        n += int(labels.numel())
    if n == 0:
        raise ValueError("loader 为空：F1 评估无样本可跑")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "iou": float(iou), "n": n}


# --------------------------------------------------------------------------- checkpoint（断点续训）

def _strip_module_prefix(state_dict):
    """DataParallel 保存的 'module.' 前缀归一化（Kaggle T4×2 启用 DP 后仍可续训）。"""
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def save_ckpt(path, model, optimizer=None, scheduler=None, epoch=0, metrics=None,
              config=None):
    """保存 checkpoint：model + optimizer（含动量状态）+ scheduler + epoch 元数据。

    path 为文件路径（Kaggle 每 epoch 落盘；run_name_ckpt_latest.pth 命名由调用方）。
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": _strip_module_prefix(
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "optimizer": optimizer.state_dict() if optimizer is not None else {},
        "scheduler": scheduler.state_dict() if scheduler is not None else {},
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
        "config": config if config is not None else {},
    }
    torch.save(blob, str(path))
    return path


def load_ckpt(path, model, optimizer=None, scheduler=None, device="cpu"):
    """加载 checkpoint → (start_epoch, metrics)：start_epoch = epoch+1（续训起点）。

    model 权重 strict 加载；optimizer/scheduler 状态可选恢复（断点续训无损）。
    """
    blob = torch.load(str(path), map_location=device)
    state = _strip_module_prefix(blob["model"])
    model.load_state_dict(state, strict=True)
    if optimizer is not None and blob.get("optimizer"):
        optimizer.load_state_dict(blob["optimizer"])
    if scheduler is not None and blob.get("scheduler"):
        scheduler.load_state_dict(blob["scheduler"])
    return int(blob["epoch"]) + 1, dict(blob.get("metrics", {}))


# --------------------------------------------------------------------------- 主流程（CLI）

try:
    from tqdm import tqdm
except ImportError:                       # tqdm 在依赖清单；缺失时退化为无进度条
    tqdm = lambda it, **kwargs: it        # type: ignore[assignment]


def _resolve_device(device_cfg):
    device_cfg = str(device_cfg).lower()
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _make_loader(dataset, batch_size, num_workers, device, shuffle=False):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle,
                      num_workers=int(num_workers), pin_memory=device.type == "cuda")


def main(argv=None):
    """训练入口：YAML 配置驱动全部超参；--resume 默认自动续训（latest checkpoint）。

    Kaggle 分块流程（HANDOFF §5）：每 epoch 更新 latest（含 optimizer 状态）+
    save_freq 里程碑快照；块尾打包 Dataset 新版本 → 下次会话 --resume auto 无损伤恢复。
    CPU 冒烟：--max-steps 1~2 + 小样本数（本地验证训练循环）。
    """
    ap = argparse.ArgumentParser(
        description="迹线 Transformer 涡提取训练（TwoStep 调度/每 epoch checkpoint/"
                    "可选 DataParallel/AMP；超参走 YAML）")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML 配置路径")
    ap.add_argument("--resume", default="auto",
                    help="续训 checkpoint 路径；'auto'=ckpt_dir/run_name_ckpt_latest.pth"
                         "（存在则续）；'none'=从零开始")
    ap.add_argument("--epochs", type=int, default=None,
                    help="覆盖 train.epochs（Kaggle 分块 ≤8h 用）")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="每 epoch 最多训练步数（CPU 冒烟 1~2 步用）")
    ap.add_argument("--report-f1", action="store_true",
                    help="训练完成后记录自然分布 F1/IoU（val_f1.json，票 07 验收 4）")
    ap.add_argument("--f1-split", default=None,
                    help="F1/IoU 记录的时间片名（默认 val_split；多数据集 60/40 "
                         "无 val 时传 test——对留出 40% 跨数据集 test 片推理，票 07 延伸）")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    train_cfg = config["train"]
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    set_random_seed(int(train_cfg.get("seed", 0)))
    enable_tf32()                 # T4 tensor core 加速（num 语义仍 fp32；见 enable_tf32 docstring）
    device = _resolve_device(train_cfg.get("device", "auto"))

    # ---- 模型/损失/优化器/调度
    model = build_model_from_config(config).to(device)
    criterion = build_criterion_from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]),
                                  weight_decay=float(train_cfg["weight_decay"]))
    scheduler = TwoStepLR(optimizer, lr=float(train_cfg["lr"]),
                          second_lr=float(train_cfg["second_lr"]),
                          warmup_epochs=int(train_cfg["warmup_epochs"]))

    # ---- 数据集（WeakLabelPathlineDataset/MultiDatasetPathlineDataset；set_epoch 每 epoch 重建过采样序）
    data_cfg = config["data"]
    train_ds = _make_dataset(data_cfg, data_cfg.get("split", "train"))
    train_loader = _make_loader(train_ds, data_cfg["batch_size"],
                                data_cfg.get("num_workers", 0), device)
    val_loader = None
    val_split = data_cfg.get("val_split", "val")
    raw_roots = data_cfg["root"] if isinstance(data_cfg["root"], (list, tuple)) \
        else [data_cfg["root"]]
    root_slice_sets = [set(ds.load_dataset_meta(r)["slices"]) for r in raw_roots]
    common_slices = set.intersection(*root_slice_sets) if root_slice_sets else set()
    if val_split in common_slices:
        val_ds = _make_dataset(data_cfg, val_split)
        val_ds.set_epoch(0)      # 固定确定性 val 采样序（跨 epoch 可比；训练序每 epoch 重建）
        val_loader = _make_loader(val_ds, data_cfg["batch_size"],
                                  data_cfg.get("num_workers", 0), device)
    else:
        print(f"[train] 警告: 数据集无 {val_split!r} 时间片（小数据集/合成场/多数据集 60/40）"
              f"→ 跳过验证；留出评估用 --report-f1 --f1-split test")

    # ---- 续训（auto/none/路径）
    ckpt_dir = pathlib.Path(train_cfg.get("ckpt_dir", "outputs/train"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(train_cfg.get("run_name", "run"))
    latest_path = ckpt_dir / f"{run_name}_ckpt_latest.pth"
    resume = args.resume
    start_epoch, _metrics = 0, {}
    if str(resume).lower() == "none":
        pass
    else:
        path = latest_path if str(resume).lower() == "auto" else pathlib.Path(resume)
        if path.exists():
            start_epoch, _metrics = load_ckpt(path, model, optimizer, scheduler,
                                              device=str(device))
            print(f"[train] 断点续训: {path} → 从 epoch {start_epoch} 继续")
        elif str(resume).lower() != "auto":
            raise FileNotFoundError(f"续训 checkpoint 不存在: {path}")

    # ---- 可选 DataParallel（Kaggle T4×2）
    if bool(train_cfg.get("data_parallel")) and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"[train] DataParallel 已启用 ({torch.cuda.device_count()} GPU)")

    # ---- 主训练循环
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    val_freq = int(train_cfg.get("val_freq", 5))
    save_freq = int(train_cfg.get("save_freq", 30))
    use_amp = bool(train_cfg.get("amp")) and device.type == "cuda"
    # torch.amp.GradScaler('cuda')：2.10 起 cuda.amp.GradScaler 为 deprecated API（警告）；
    # 本地 CPU 路径 enabled=False 构造无害
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        train_ds.set_epoch(epoch)
        lr = scheduler.step(epoch)
        train_loss = run_epoch(model, tqdm(train_loader, desc=f"epoch {epoch + 1}"),
                               criterion, optimizer, device, grad_clip=grad_clip,
                               max_steps=args.max_steps, scaler=scaler)
        metrics = {"train_loss": float(train_loss), "epoch": int(epoch), "lr": float(lr)}
        if val_loader is not None and ((epoch + 1) % val_freq == 0 or epoch == start_epoch):
            val_loss = evaluate(model, val_loader, criterion, device)
            metrics["val_loss"] = float(val_loss)
            print(f"[train] epoch {epoch + 1}/{train_cfg['epochs']} "
                  f"loss={train_loss:.4f} val={val_loss:.4f} lr={lr:g}")
        else:
            print(f"[train] epoch {epoch + 1}/{train_cfg['epochs']} "
                  f"loss={train_loss:.4f} lr={lr:g}")
        # 每 epoch 更新 latest（含 optimizer 状态）；save_freq 里程碑快照
        save_ckpt(latest_path, model, optimizer, scheduler, epoch=epoch,
                  metrics=metrics, config=config)
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            save_ckpt(ckpt_dir / f"{run_name}_E{epoch + 1}.pth", model, optimizer,
                      scheduler, epoch=epoch, metrics=metrics, config=config)

    # ---- 训练完成 → 自然分布 F1/IoU 记录（票 07 验收 4 + 票 07 延伸留出评估；
    #      --report-f1 显式开关；--f1-split 指定片，默认 val_split，
    #      多数据集 60/40 无 val 时传 test——对留出 40% 跨数据集 test 片推理）
    if args.report_f1:
        f1_split = args.f1_split if args.f1_split else val_split
        if f1_split not in common_slices:
            if args.f1_split:
                raise ValueError(
                    f"--f1-split {f1_split!r} 不在各数据集共有的时间片 "
                    f"{sorted(common_slices)} 内")
            print(f"[train] 无 {f1_split!r} 时间片 → 跳过 F1 记录"
                  f"（如需留出评估传 --f1-split test）")
        else:
            f1_ds = _make_dataset(data_cfg, f1_split)
            f1_ds.set_epoch_natural(0)   # 自然分布序（正负比例 = 池比例；
                                         # 池空时 set_epoch fail loud——数据异常不静默）
            f1_loader = _make_loader(f1_ds, data_cfg["batch_size"],
                                     data_cfg.get("num_workers", 0), device)
            f1_metrics = evaluate_f1(model, f1_loader, device)
            f1_blob = {"epoch": int(train_cfg["epochs"]) - 1, "split": f1_split,
                       "threshold": 0.5, **f1_metrics}
            f1_name = (f"{run_name}_val_f1.json" if f1_split == val_split
                       else f"{run_name}_{f1_split}_f1.json")
            f1_path = ckpt_dir / f1_name
            f1_path.write_text(json.dumps(f1_blob, indent=2, ensure_ascii=False),
                               encoding="utf-8")
            print(f"[train] F1/IoU（自然分布，{f1_split}）："
                  f"F1={f1_metrics['f1']:.4f} IoU={f1_metrics['iou']:.4f} "
                  f"P={f1_metrics['precision']:.4f} R={f1_metrics['recall']:.4f} "
                  f"(tp={f1_metrics['tp']} fp={f1_metrics['fp']} "
                  f"fn={f1_metrics['fn']} n={f1_metrics['n']}) → {f1_path}")
    return 0


if __name__ == "__main__":
    main()
