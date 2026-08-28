"""Kaggle Dataset A 挂载布局探测（mount_probe.py）——票 07 延伸运行反馈修复。

领域词汇（HANDOFF §2 / 票 07 三期与延伸，唯一权威）：
- Kaggle Add Input 的挂载结构 = `{root}/datasets/<owner>/<slug>/...`（多级嵌套，
  含 slug 多级目录——三期实测 `datasets/ziyixu317/2d-...`）；Dataset A zip 上传后
  Kaggle 自动解压，布局原样保留：
    * 单数据集：`<挂载根>/<nc>` + `<挂载根>/dataset/meta.json`
    * 多数据集：`<挂载根>/data/<nc>` + `<挂载根>/datasets/<名>/dataset/meta.json`
- probe_layout 返回挂载根（**含布局标志的最浅目录**）：单数据集根 或 多数据集根
  （多优先——多布局内部的 `datasets/<名>` 子目录含 dataset/meta.json，若先判单
  会误命中）；未命中返回 (None, None)（调用方走 zip 解压回退 / fail loud）。
- 纯 pathlib 实现（不依赖挂载实际路径深度——本地测试用 tmp 树模拟 Kaggle 结构）。
"""

from __future__ import annotations

import pathlib


def probe_layout(root="/kaggle/input"):
    """按布局标志定位挂载根 → (single_root, multi_root)（含多级嵌套挂载）。

    扫描范围 = root 自身 + 全部子目录（深度优先，浅层先命中）；判据：
    - 多数据集：目录下有 datasets/<名>/dataset/meta.json（至少一个数据集）；
    - 单数据集：目录下有 dataset/meta.json（仅在无多数据集命中的情况下判定）。
    """
    root = pathlib.Path(root)
    dirs = [root] + sorted((d for d in root.rglob("*") if d.is_dir()),
                           key=lambda p: (len(p.parts), str(p)))
    for d in dirs:
        ds_dir = d / "datasets"
        if ds_dir.is_dir() and any(
                (sub / "dataset" / "meta.json").exists()
                for sub in ds_dir.iterdir() if sub.is_dir()):
            return None, d           # 多优先，避免多布局内子目录误判
    for d in dirs:
        if (d / "dataset" / "meta.json").exists():
            return d, None
    return None, None
