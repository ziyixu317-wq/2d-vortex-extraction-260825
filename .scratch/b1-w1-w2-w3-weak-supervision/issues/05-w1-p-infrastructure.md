# 05 — W1-P p90/p60/unknown teacher infrastructure

**Title:** W1-P p90/p60/unknown teacher infrastructure
**What to build:** 先以 train-only local-IVD p90 positive、p60 negative、中间 unknown 跑通 W1 的 masked weak loss、EMA teacher、pseudo-label、consistency 和 ramp-up 基础设施。
**Blocked by:** 01-split-label-provenance.md, 03-mode-checkpoint-contract.md
**Status:** done
**Primary seam:** training/evaluation seam

## What to build

- 在 train frames 上按当前 local-IVD 语义生成 p90 positive、p60 negative 和中间 unknown；solid 从监督 mask 中排除。
- positive 使用既有有效区域/最小面积语义；`p60 < local-IVD < p90` 保持 unknown，不强行二值化。
- student 和 EMA teacher 初始权重相同；student optimizer step 后更新 teacher，EMA decay 使用预注册值 `0.99`。
- 已知 p90/p60 区域使用 masked BCE；unknown 区域只接受 teacher probability `>= 0.90` 或 `<= 0.10` 的 pseudo-label。
- 实现 student/teacher consistency loss，并将 pseudo/consistency 权重从 0 ramp 到目标值，ramp-up 固定为约 12 epochs。
- `legacy_p85` 可以继续用于 train patch sampling，但不得进入 W1-P formal `Lweak`；batch 和日志必须区分 sampling source 与 loss source。
- 保留现有 5×5 local-IVD 输入和 7-channel schema；W1-P 不依赖 B1。

## Acceptance criteria

- train fixture 能生成三态 p90/p60/unknown label 和 solid/unknown mask；中间区间不会被静默转成 negative。
- masked BCE 只在已知 positive/negative anchor 上计算；solid、unknown 和无效 anchor 不贡献 anchor loss。
- EMA teacher 在 student 更新后更新，且 synthetic step 能验证 teacher 权重不是直接 alias student 权重。
- pseudo-label 只接受 `>= 0.90` 或 `<= 0.10`，被拒绝区域保留 unknown；pseudo acceptance、anchor coverage 和 disagreement 有统计。
- ramp-up 在第 0 epoch 为 0，在约第 12 epoch 达到目标，并可在 checkpoint/config 中复现。
- 5–10 epoch W1-P smoke 成功；训练路径没有读取 test Haller GT，formal `Lweak` metadata 不含 `legacy_p85`。

## Verification commands

```powershell
python -m pytest tests/test_w1_p_infrastructure.py -q
python -m pytest tests/test_weak_supervision_contract.py -q
```

```bash
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_w1_p_infrastructure.py -q
```

## Implementation result

- 新增 `w1_p.py`：train-only local-IVD p90/p60/unknown 三态 target、solid/unknown mask、masked BCE、confident pseudo-label、student/EMA teacher、consistency、12 epoch ramp-up，以及显式 source/split/schema guard。
- 新增 `tests/test_w1_p_infrastructure.py`：覆盖 target 生成、mask/loss、EMA、ramp、5 epoch CPU smoke、checkpoint round-trip、test Haller source 拒绝。
- W1-P 保持现有 7-channel local-IVD 输入；`legacy_p85` 仅作为显式 sampling source，formal loss/checkpoint label source 固定为 `local_p90_p60`。
- 验证：targeted 6 passed；shared contract 63 passed；受影响模块回归 142 passed。最终全量回归 337 passed，另有 19 个既有 evaluate fixture 权限 errors；本票未引入其他失败。
- 后续：票 06 W1-H 复用本票训练/teacher seam；本票不实现 Haller anchor、W2/W3 或端到端实验。
