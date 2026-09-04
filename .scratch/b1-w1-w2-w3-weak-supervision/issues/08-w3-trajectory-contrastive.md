# 08 — W3 trajectory contrastive extension

**Title:** W3 trajectory contrastive extension
**What to build:** 在 W2 之上通过 vendor 外部 local adapter 暴露 trajectory embedding，加入两 stochastic views 的 in-batch contrastive objective，并严格执行第一版资源上限。
**Blocked by:** 07-w2-uncertainty-gate.md
**Status:** ready-for-agent
**Primary seam:** training/evaluation seam

## What to build

- 在 local adapter 中取得每条 trajectory 的 pre-classifier embedding，再接 projection head；不得修改 `vendor/DeepUtils`。
- 每条 trajectory 生成 2 个 stochastic views，同一 trajectory 的两视图构成 positive pair，in-batch 其他样本构成 negatives。
- Haller anchor 或 W2 pseudo-label 只有在 known/reliable 状态下才能形成语义 pair；unknown 不参与语义正负 pair。
- projection dimension 固定为 64，temperature 固定为 0.1；contrastive loss 与 W2 unsupervised terms 使用约 12 epoch ramp-up。
- 第一版固定 single GPU、两视图合计最多 512 个送入 contrastive loss 的 embeddings、in-batch contrastive、无 memory bank、无跨 GPU gathering。
- 保存 projection head、pair statistics、effective embedding count、unknown exclusion count 和 contrastive loss 到 checkpoint/log。

## Acceptance criteria

- local adapter 能在不改 vendor 的前提下同时产生 classification probability 和 trajectory embedding；embedding 与 trajectory identity 对齐。
- 每个有效 identity 正好产生 2 个 stochastic views；同 identity pair 是 positive，in-batch negatives 不访问 memory bank。
- 两视图合计送入 contrastive loss 的 embedding 数不超过 512；超限时有确定性行为并记录有效 pair/embedding count。
- unknown、solid、invalid 和不可靠 pseudo-label 不形成语义 pair；known Haller anchor 或 gated pseudo-label 可形成 pair。
- projection dimension、temperature、view count、single-GPU mode 和 cap 写入 checkpoint；5–10 epoch single-GPU smoke 不 OOM。
- vendor 文件、旧 B0 产物和 W1/W2 输入 schema 保持不变。

## Verification commands

```powershell
python -m pytest tests/test_w3_trajectory_contrastive.py -q
python -m pytest tests/test_w2_uncertainty_gate.py -q
```

```bash
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_w3_trajectory_contrastive.py -q
```
