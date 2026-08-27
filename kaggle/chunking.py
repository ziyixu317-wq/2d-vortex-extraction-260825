"""Kaggle 分块训练规划（07 票）：12h 会话硬上限 → 每块 ≤ 会话预算。

领域词汇（HANDOFF §5/§7 与规格，唯一权威）：
- Kaggle 会话 12h 硬上限 → 训练分块 ≤8h（HANDOFF §5：块尾打包为 Kaggle Dataset
  新版本，下次会话 `--resume auto` 从 latest checkpoint 无损伤恢复）；
- train_kaggle.py 的 `--epochs` 为块目标（覆盖 config 的 total；resume 从
  checkpoint['epoch']+1 续到目标）→ 本模块只需规划「每块跑多少 epoch」。
"""

from __future__ import annotations


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
