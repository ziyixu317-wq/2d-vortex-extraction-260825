"""kaggle 训练支撑包（07 票）：分块规划（chunking）。

Kaggle 会话 12h 硬上限 → 训练按 ≤8h 分块（预留自检/打包时间），
每 epoch checkpoint + `--resume auto` 跨会话断点续训（HANDOFF §5/§7）。
"""
