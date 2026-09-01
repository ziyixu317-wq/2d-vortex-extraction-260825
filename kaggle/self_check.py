"""历史自检实现（deprecated Kaggle 入口）。

服务器请使用 ``python server/self_check.py``；此模块保留同一检查实现以兼容旧测试。

领域词汇（HANDOFF §4/§5 与规格，唯一权威）：
- 验收 1：服务器环境 import vendor + 数据加载通过（本模块保留历史实现，
  当前入口为 ``server/self_check.py``）；
- 模型缝：迹线样本 (B, L=16, K=256, C=7) → 每迹线涡概率 (B, 256)、域 (0,1)（sigmoid）；
- 数据加载：WeakLabelPathlineDataset on-the-fly（set_epoch 后取 n_samples）；
  样本有限无 NaN、标签 0/1（弱标签口径）。

兼容用法：python kaggle/self_check.py --data-root outputs/datasets/pipedcylinder2d/dataset
      [--config config/pathline_transformer_cylinder.yaml] [--device cpu]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# CLI 入口（python kaggle/self_check.py）从任意 cwd 运行时也能 import 项目模块
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def self_check(data_root, config_path=None, device="cpu", n_samples=4,
               model_cfg=None):
    """环境自检 → dict（各检查项结果；任何失败 raise，不静默通过）。

    - config_ok：YAML 可读且含 model/train/data 段（config_path 为 None 时只走
      model_cfg 直传——两路均缺则报错，无不可用默认）；
    - model_forward_ok：模型构建 + 前向输出 (B, 256) 且域 (0,1)（model_cfg 缺省
      从 config 的 model 段构建——生产口径；测试可传小模型 cfg 加速）；
    - dataset_ok：数据集加载 + n_samples 个 on-the-fly 样本有限无 NaN、
      标签 ∈ {0,1}（label_sum 返回供打印）。
    """
    import numpy as np
    import torch
    import yaml

    # ---- 配置可读（self_check 的 config 参数与 train_kaggle 同源）
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for seg in ("model", "train", "data"):
            if seg not in cfg:
                raise ValueError(f"配置缺少 {seg} 段: {config_path}")
    else:
        cfg = {"model": model_cfg}
    if cfg["model"] is None:
        raise ValueError("请提供模型来源：--config（生产 YAML）或 model_cfg（测试直传）")

    # ---- vendor import + 模型前向（模型缝）
    from vendor.DeepUtils.models import build_model_from_cfg
    device = torch.device(device)
    model = build_model_from_cfg(cfg["model"] if model_cfg is None else model_cfg)
    model = model.to(device)
    model.eval()
    B, L, K, C = 1, 16, 256, 7
    with torch.no_grad():
        x = torch.rand(B, L, K, C, device=device)
        out = model((torch.zeros(B, 1, 1, 1, device=device), x))
    if tuple(out.shape) != (B, K):
        raise AssertionError(f"前向输出形状 {tuple(out.shape)} ≠ (B, K)={(B, K)}")
    if not bool((out > 0).all() and (out < 1).all()):
        raise AssertionError("前向输出不在 (0,1)（sigmoid 概率域）")

    # ---- 数据加载（on-the-fly 样本有限性 + 标签 0/1）
    import dataset as ds
    d = ds.WeakLabelPathlineDataset(
        data_root, split="train",
        samples_per_epoch=int(n_samples) if n_samples else 4, seed=0)
    d.set_epoch(0)
    label_sum = 0
    for i in range(min(int(n_samples), len(d))):
        (dummy, pathlines), labels = d[i]
        if pathlines.shape != (L, K, C):
            raise AssertionError(f"样本形状 {tuple(pathlines.shape)} ≠ {(L, K, C)}")
        if not np.isfinite(pathlines).all():
            raise AssertionError("迹线样本含 NaN/Inf")
        if not np.all((labels == 0) | (labels == 1)):
            raise AssertionError("标签不在 {0,1}（弱标签口径）")
        label_sum += int(labels.sum())

    print(f"[自检] vendor import OK（models/loss/utils）")
    print(f"[自检] 模型前向 OK：输出 {tuple(out.shape)}，概率域 (0,1)")
    print(f"[自检] 数据加载 OK：{int(n_samples)} 样本 (16,256,7) 有限无 NaN，"
          f"标签 ∈{{0,1}}（正标签 {label_sum} 条）")
    return {"config_ok": True, "model_forward_ok": True, "dataset_ok": True,
            "n_samples": int(n_samples), "label_sum": label_sum}


def main(argv=None):
    ap = argparse.ArgumentParser(description="历史环境自检（服务器请使用 server/self_check.py）")
    ap.add_argument("--data-root", default="outputs/datasets/pipedcylinder2d/dataset",
                    help="服务器上的逐数据集 memmap 目录")
    ap.add_argument("--config", default=None, help="生产 YAML 配置路径（默认不读）")
    ap.add_argument("--device", default="cpu", help="设备（服务器可用 cuda）")
    ap.add_argument("--n-samples", type=int, default=4, help="抽查样本数")
    args = ap.parse_args(argv)
    r = self_check(args.data_root, config_path=args.config,
                   device=args.device, n_samples=args.n_samples)
    print(f"[自检] 全部通过: {r}")
    return 0


if __name__ == "__main__":
    main()
