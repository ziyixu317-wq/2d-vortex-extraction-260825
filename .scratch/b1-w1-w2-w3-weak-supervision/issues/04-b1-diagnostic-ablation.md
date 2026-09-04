# 04 — B1 diagnostic local-IVD input ablation

**Title:** B1 diagnostic local-IVD input ablation
**What to build:** 在新 split 上实现仅移除 local-IVD 输入通道的 B1 诊断运行，继续使用 B0 对齐的 legacy p85 supervision，并独立保存结果。
**Blocked by:** 01-split-label-provenance.md, 03-mode-checkpoint-contract.md
**Status:** ready-for-agent
**Primary seam:** training/evaluation seam

## What to build

- 使用 B1 显式 6-channel schema `[px, py, t, distance, u, v]`，由外部 adapter 选择 channel；不得修改 vendor。
- 在新 0–50/50–60/60–100 split 上运行 B1，监督和 train patch sampling 继续使用 `legacy_p85`，保持与新 split B0 的可比性。
- B1 从头训练，独立保存 mode、seed、split、feature schema、checkpoint 和 diagnostic report；不加载旧 B0 checkpoint。
- B1 不生成 Haller anchors，不使用 Haller GT，也不作为 W1-P/W1-H 的 prerequisite。
- 结果必须与 B0、W 方法和 historical old-split 结果分开归档；B1 默认不进入 final headline 的 best-baseline slot。

## Acceptance criteria

- B1 单批前向和 loss 成功运行，模型收到的 channel 数为 6，且 local-IVD channel 不在输入中；错误 schema fail loudly。
- 除输入 channel 外，B1 的 label source、split、window contract 和 train-only normalization 与新 split B0 对齐。
- B1 smoke 在 5–10 epochs 内完成，并生成可恢复 checkpoint 和独立 diagnostic metadata。
- B1 训练日志明确写出 `legacy_p85` source，并证明没有读取 Haller train/test artifact。
- B1 的结果不覆盖 B0 或 W 方法产物，也不改变 vendor 文件。

## Verification commands

```powershell
python -m pytest tests/test_b1_ablation.py -q
python -m pytest tests/test_weak_supervision_contract.py -q
```

```bash
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_b1_ablation.py -q
```
