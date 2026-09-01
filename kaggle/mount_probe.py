"""输入目录布局探测的兼容接口。

``probe_layout`` 识别单数据集和多数据集两种 ``dataset/meta.json`` 目录形态，
供既有测试和快照读取嵌套归档；服务器正式工作区使用固定路径。
"""

from __future__ import annotations

import pathlib


def probe_layout(root="/kaggle/input"):
    """按布局标志定位输入根 → ``(single_root, multi_root)``。

    扫描范围为 root 自身及全部子目录（浅层先命中）；多数据集布局优先于单数据集布局。
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
