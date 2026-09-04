# 10 — Final headline experiments and reporting

**Title:** Final headline experiments and reporting
**What to build:** 在 pilot/calibration 规则冻结后，从头训练并运行 B0、best baseline 和 proposed method 的最终 headline 实验，固定 130 epochs、3 seeds，并生成可审计报告。
**Blocked by:** 09-e2e-pilot-evaluation.md
**Status:** ready-for-agent
**Primary seam:** training/evaluation seam 的 final contract

## What to build

- final 只运行 B0、calibration 选出的 best baseline（W1-H 或 W2）和 proposed W3；B1 保持单独 diagnostic，不进入 headline。
- 三个 headline method 均从头训练，epoch 固定为 130，seeds 固定为 `[0, 1, 2]`；旧 B0 checkpoint 只能另行作为 `warm_start_aux=true` auxiliary。
- 使用 pilot 中冻结的 global prediction threshold、W2 global variance gate、Haller parameter hash、feature schema 和 split config；不能在 test 结果上重新选择。
- 六个有效数据集分别报告每个 seed 的 Precision、Recall、F1、IoU、coverage、有效 count、invalid/failure count 和区域比例，并报告三 seeds 的 mean/std 与六数据集 macro。
- Boussinesq 的结果在最终报告中单独展示；B1、historical old-split 和 warm-start auxiliary 单独分节。
- 保存 final run manifest、command/config、checkpoint index、seed/RNG metadata、calibration decision 和结果文件 hash。

## Acceptance criteria

- B0、best baseline 和 W3 各完成 130 epoch × 3 seeds，所有主运行均标记 `warm_start_aux=false` 且从头初始化。
- 最终报告只包含在 calibration 阶段冻结的 threshold/gate/method；没有 test metric 驱动的模型选择或参数回写。
- 六个有效数据集都存在独立 P/R/F1/IoU 和 macro；每项均可追溯到 dataset、seed、epoch、GT source 和 checkpoint。
- 3 seeds 的 mean/std、Haller known/unknown coverage、invalid/failure count 和 Boussinesq stress-test 行完整。
- B1 diagnostic、历史 old-split B0 和 warm-start auxiliary 没有混入 headline 表格。
- final manifest 能在 checkpoint metadata 中恢复 split、feature schema、Haller hash、calibration policy、epoch、seed 和 method。

## Verification commands

```powershell
python -m pytest tests/test_final_experiment_manifest.py -q
python -m pytest tests/test_evaluation_report.py -q
```

```bash
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_final_experiment_manifest.py tests/test_evaluation_report.py -q
```
