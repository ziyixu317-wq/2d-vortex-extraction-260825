# 09 — Primary end-to-end training, evaluation and pilot selection

**Title:** Primary end-to-end training, evaluation and pilot selection
**What to build:** 打通本 feature 的主端到端训练/评价缝，并运行所有预注册方法的 50 epoch、1 seed pilot；使用 calibration 做冻结范围内的全局选择，不访问 test 做决策。
**Blocked by:** 03-mode-checkpoint-contract.md, 04-b1-diagnostic-ablation.md, 05-w1-p-infrastructure.md, 06-w1-h-haller-integration.md, 07-w2-uncertainty-gate.md, 08-w3-trajectory-contrastive.md
**Status:** ready-for-agent
**Primary seam:** training/evaluation seam（主验收）

## What to build

- 从 split-contained pathline batch 开始，完成 B0、B1、W1-P、W1-H、W2、W3 的训练、checkpoint 保存/恢复和 test evaluation 闭环。
- evaluation 在模型、global threshold、W2 gate 和 method selection 冻结后显式读取 `haller_gt_test`；训练、pseudo-label 和 calibration selection 不得读取 test source。
- 对六个有效数据集分别累计 test confusion 并报告 Precision、Recall、F1、IoU、有效 frame/cell/sample count、Haller known/unknown coverage、invalid/failure count、预测/GT positive area ratio，再计算等权 macro average。
- Boussinesq 使用与其他数据集相同的 global threshold/gate，单独展示为 threshold/domain-drift stress test，不做 per-dataset test tuning。
- pilot 所有预注册方法各运行 50 epochs、1 seed，unsupervised ramp-up 约 12 epochs；B1 结果保持 diagnostic。
- calibration Haller GT 只用于 global prediction threshold、global W2 variance gate 和 best-baseline method selection；best baseline 候选固定为 W1-H/W2，proposed 固定为 W3，B1 不进入 headline。
- 记录 anchor coverage、pseudo acceptance、teacher/student disagreement、W2 uncertainty、W3 pair count、threshold/gate、来源 hash 和 selection decision。

## Acceptance criteria

- synthetic end-to-end fixture 能完成 pathline batch → training step → checkpoint → explicit test-Haller evaluation，并生成 per-dataset metrics 与 macro。
- 六个有效数据集的 pilot report 均有独立 Precision、Recall、F1、IoU 和 macro；Boussinesq 有单独 stress-test 记录。
- evaluator 只能通过显式 `haller_gt_test` source 读取 test GT；训练或 calibration selection 访问 test label/metric 时 fail loudly。
- calibration 选择记录 global threshold、global variance gate 和 W1-H/W2 baseline winner；选择过程没有 test metric 输入。
- B1 结果与其他方法分开归档，不被作为 W1 prerequisite 或 final headline method。
- checkpoint round-trip 后 pilot 的 method mode、split、anchor hash、threshold/gate、metrics schema 和 RNG metadata 保持一致。

## Verification commands

```powershell
python -m pytest tests/test_e2e_weak_supervision.py -q
python -m pytest tests/test_evaluation_report.py -q
```

```bash
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_e2e_weak_supervision.py tests/test_evaluation_report.py -q
```
