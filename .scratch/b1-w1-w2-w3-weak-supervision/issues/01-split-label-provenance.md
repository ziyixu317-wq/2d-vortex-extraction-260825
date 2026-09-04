# 01 — Split-contained windows and label provenance contract

**Title:** Split-contained windows and label provenance contract
**What to build:** 为六个有效数据集建立弱监督新 split、完整 pathline window、train-only normalization 和 label provenance 的统一数据契约，并加入 fail-loud 泄漏保护。
**Blocked by:** None
**Status:** ready-for-agent
**Primary seam:** split/label seam

## What to build

- 将每个数据集按 frame index 划分为 train `[0, floor(0.50*T))`、calibration `[floor(0.50*T), floor(0.60*T))`、test `[floor(0.60*T), T)` 半开区间。
- 复用现有 pathline extractor 的 `t_win` 和 `window_step` 语义，但只枚举完全落在单一 split 内的窗口；不得截断窗口或借用相邻 split。
- 让 dataset/window metadata 携带 `split_name`、frame start/end、`t_win`、window step、生成版本、feature schema、label source 和 hash。
- 明确区分 `legacy_p85`、`local_p90_p60`、`haller_anchor_train`、`haller_gt_calibration` 和 `haller_gt_test`；p85 sampling membership 不能伪装成 W1 formal loss label。
- normalization statistics 只从 train 数据生成并冻结；calibration/test 只消费冻结统计量。
- 训练数据加载或 loss 路径显式拒绝 `haller_gt_test`；test Haller GT 只能由 evaluator 显式传入。
- 保持旧 `.scratch/vortex-extraction-pipeline/`、旧数据产物和 `vendor/DeepUtils` 不变。

## Acceptance criteria

- 六个有效数据集都能得到精确的 0/50/60/100 frame boundaries，边界使用半开区间并记录到 metadata。
- 边界前、边界处和边界后的 window fixture 均满足 `start >= split_start` 且 `start + t_win <= split_end`；跨界 start 必须 fail loudly。
- split 长度不足以容纳一个完整 window 时，错误信息包含 dataset、T、split boundary 和 t_win，不静默缩短窗口。
- metadata 能区分 split、window、feature schema、label source、normalization source 和 generation version/hash。
- 任何训练入口尝试读取 `haller_gt_test` 都被拒绝；evaluation 入口必须显式声明 test GT source。
- 既有 dataset/multidataset 回归测试保持通过。

## Verification commands

```powershell
python -m pytest tests/test_weak_supervision_split.py -q
python -m pytest tests/test_dataset.py tests/test_multidataset.py -q
```

```bash
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_weak_supervision_split.py -q
```
