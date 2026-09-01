"""断点续训的分块规划兼容接口。

``plan_chunks`` 根据总 epoch、单 epoch 步速和预算生成分块序列，
``pick_bench_source`` 选择当前或已恢复的步速基准；两者供既有测试和快照格式使用。
服务器常规训练直接依靠逐 epoch checkpoint 与 ``--resume auto``。
"""

from __future__ import annotations

import json
import os


def pick_bench_source(new_path, restored_path):
    """选择当前或已恢复的步速基准，返回 ``(来源路径, bench dict)``。

    两个候选路径都不存在时返回 ``(None, None)``，由调用方执行首次校准。
    """
    for p in (str(new_path), str(restored_path)):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return p, json.load(f)
    return None, None


def plan_chunks(total_epochs, seconds_per_epoch, budget_seconds):
    """总 epoch → 每块 epoch 序列（每块耗时 ≤ budget_seconds、块数最少）。

    规则：每块最多 floor(budget/seconds_per_epoch) 个 epoch（至少 1 个——
    单 epoch 超预算时不退化）；尾块为剩余 epoch（可为更小块）。
    返回 list[int]：sum == total_epochs。
    """
    total_epochs = int(total_epochs)
    seconds_per_epoch = float(seconds_per_epoch)
    budget_seconds = float(budget_seconds)
    if total_epochs <= 0:
        raise ValueError(f"total_epochs 必须为正，实际 {total_epochs}")
    if seconds_per_epoch <= 0:
        raise ValueError(f"seconds_per_epoch 必须为正，实际 {seconds_per_epoch}")
    if budget_seconds <= 0:
        raise ValueError(f"budget_seconds 必须为正，实际 {budget_seconds}")

    per_chunk = max(1, int(budget_seconds // seconds_per_epoch))
    chunks = []
    remaining = total_epochs
    while remaining > 0:
        n = min(per_chunk, remaining)
        chunks.append(n)
        remaining -= n
    return chunks
