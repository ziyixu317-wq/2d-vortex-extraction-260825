# 07 — W2 uncertainty-gated pseudo labels

**Title:** W2 uncertainty-gated pseudo labels
**What to build:** 在 W1-H 之上加入 3-view stochastic teacher prediction、mean/variance gate 和 calibration-controlled global uncertainty threshold。
**Blocked by:** 06-w1-h-haller-integration.md
**Status:** done
**Primary seam:** training/evaluation seam

## What to build

- 对同一 unknown candidate 生成 3 次 stochastic teacher views，计算 mean probability、predictive variance 和 Bernoulli entropy。
- 冻结 positive/negative confidence 为 mean `>=0.90` 和 `<=0.10`；只有 predictive variance 不超过单一 global uncertainty threshold 时才接受 pseudo-label。
- variance 是主 gate，entropy 只作诊断；不按数据集分别调 gate。
- calibration Haller GT 可以在模型训练完成后用于选择 global prediction threshold 和 global variance gate；不能更新模型权重、normalization、Haller 参数或 pseudo-label 规则。
- test Haller GT 只能在最终评价阶段读取；任何 test metric 或 test label 访问都必须被检测并拒绝。
- 记录 view count、mean/variance/entropy 分布、gate acceptance、正负 pseudo 比例和 disagreement，并将 gate 配置写入 checkpoint。

## Acceptance criteria

- synthetic probability fixture 能验证 3-view mean、variance、entropy 的数值和 shape。
- mean 在 `[0.10,0.90]` 或 variance 超过 global gate 时样本保持 unknown；仅同时满足 confidence 和 uncertainty 条件时接受 pseudo-label。
- global gate 能通过 calibration fixture 选择并复现；不同数据集不能各自得到独立 gate。
- calibration 读取不会产生 optimizer step；test Haller GT 被传入训练或 gate-selection 路径时 fail loudly。
- W2 5–10 epoch smoke 成功，并在 checkpoint/日志记录 gate、view count 和 acceptance statistics。

## Verification commands

```powershell
python -m pytest tests/test_w2_uncertainty_gate.py -q
python -m pytest tests/test_w1_h_integration.py -q
```

```bash
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_w2_uncertainty_gate.py -q
```

## Implementation result

- 新增 `w2.py`，复用 W1-H 的 7-channel、5×5 local-IVD batch；固定同一 EMA teacher 的 3 个显式可复现 stochastic views，输出 probability mean、population predictive variance 和 Bernoulli entropy。
- pseudo-label 仅在 mean `>=0.90` 或 `<=0.10` 且 variance `<=` 单一 global gate 时接受；variance 是主 gate，entropy 仅记录诊断。calibration records 跨数据集合并进行一个全局 threshold/gate 选择，禁止 per-dataset gate、optimizer 更新和 test Haller/test-only 数据。
- W2 batch 现在强制携带显式 split-contained window provenance，并校验 `pathline_7ch` 的 canonical 7-channel、5×5 local-IVD schema；W1-H collated batch 的每个 window 也逐一校验，split/window/source/schema 任何漂移均 fail loudly。
- W2 checkpoint 强制保存并校验 gate/view/acceptance/distribution/disagreement metrics、global calibration policy reproducibility provenance，以及 Haller train anchor 的 algorithm/parameter/input/mask/failure metadata；load 后恢复 anchor metadata 和 calibration policy，支持无显式 metrics 的续训保存。共享 checkpoint loader 先在 CPU 以安全 weights-only 方式读取，拒绝不安全 pickle 回退，并对 CUDA RNG device layout 做 strict 校验。
- 收口补强 Haller train metadata 的 `pending_verification`、no-fallback、coverage contract（兼容 artifact 的 canonical coverage mapping），允许同一 manifest 内 frame-specific input hash；同时重校验 typed calibration record、拒绝 test-only alias 与非结构化 `test` key，并强制完整 uncertainty distribution metrics、population variance 范围和 W2 strict CUDA RNG 默认值。

## Verification result

- `python -m pytest tests/test_w2_uncertainty_gate.py -q`：23 passed，2 个 pytest cache 权限 warning。
- `python -m pytest tests/test_w1_h_integration.py -q`：7 passed。
- 相关回归（W2/W1-H/Haller artifact/weak-supervision contract+split/W1-P/B1，使用 `-p no:cacheprovider`）：146 passed，3 个 Paramiko deprecation warning。
- SHU-server targeted：在 `/data/xuziyi/cylinder_vortex_pipeline` 使用 `/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_w2_uncertainty_gate.py -q`：23 passed；GPU0 运行前确认 PyTorch `2.7.1+cu118`、CUDA `11.8`、CUDA 可用，且仅暴露 1 张 GPU。
- SHU-server 真实 W2 smoke：真实 `cylinder2d` 240 帧空间裁剪、正式 `t_win=24` train/calibration/test split、真实 W1-H `haller_anchor_train` artifact、vendor `PathlineTransformerV0`，GPU0 运行 5 epoch；pathlines `(1,4,16,7)`、固定 3 views、global variance gate `0.05`，5 个 loss 均 finite（`0.62019 → 0.01297`），`global_step=5`。checkpoint `/data/xuziyi/tmp/w2-real-smoke-server-final.pt`，日志 `/data/xuziyi/tmp/w2-real-smoke-server-final.log`；随后以当前 strict CUDA RNG loader 在 GPU0 完成 load→resume 校验，恢复 `mode=W2`、`epoch=4`、`global_step=5`、`label_source=haller_anchor_train`、calibration source `haller_gt_calibration`、anchor metadata、`rng_restored=true` 和 `cuda_rng_restored=true`。
- `python -m pytest tests -q -p no:cacheprovider`：369 passed、3 warnings、19 errors；19 项仍全部是既有 evaluator fixture 写入受保护 `outputs/eval_test_ds`/`outputs/eval_tau_test_ds` 的 PermissionError，W2 与相关回归无失败。
- 最终代码收口验证（2026-09-02）：`python -m pytest tests/test_w2_uncertainty_gate.py -q -p no:cacheprovider`：46 passed；相关 W2/W1-H/Haller artifact/contract/split/W1-P/B1 回归：169 passed、3 个 Paramiko deprecation warning；SHU-server `/data/xuziyi/cylinder_vortex_pipeline` 使用 `/data/xuziyi/envs/xuziyi/bin/python` 的 W2 targeted：46 passed。服务器预检确认 Python 3.12.14、PyTorch 2.7.1+cu118、CUDA 11.8、CUDA 可用、GPU0 单卡可见。
- 最终真实 smoke 使用真实 `cylinder2d-real-smoke-crop` 数据和 `haller_anchor_train`/`haller_gt_calibration` artifact，在 SHU-server GPU0 完成 5 epoch：batch `(1,4,16,7)`、3 views、gate `0.05`、5 个 loss finite、`global_step=5`；保存并 strict resume `/data/xuziyi/tmp/w2-real-smoke-server-final.pt`，恢复 `mode=W2`、epoch `4`、global step `5`、Haller/calibration source、selection hash、CPU/CUDA RNG。由于该小裁剪负采样池为空，smoke 使用公开 `sample_at` seam；正式 sampler 的 fail-loud 约束保持不变。
- 最终完整回归：`python -m pytest tests -q -p no:cacheprovider` 为 `392 passed`、3 warnings、19 errors；19 项全部是既有 evaluator fixture 写入受保护 `outputs/eval_test_ds`/`outputs/eval_tau_test_ds` 的 PermissionError，W2 与相关回归无失败。Haller 文献及工程参数 canonical 对应仍为 pending-verification；未修改 vendor、旧 baseline、旧 spec/issues。下一张依赖票为 08/WS-8 W3。

小裁剪 smoke 的完整随机 sampler 预检会因负样本池为空而 fail loudly，因此服务器验收使用真实 train window 的显式 `sample_at` seam；这不改变正式 sampler 的 fail-loud contract，后续正式多数据集 artifact 生成时需补充非空采样池覆盖。Haller 文献全文及工程参数的 canonical paper 对应仍按 HANDOFF 标记为 pending-verification；本票未修改 vendor、旧 baseline 或旧 spec/issues。下一张依赖票为 08/WS-8 W3。
