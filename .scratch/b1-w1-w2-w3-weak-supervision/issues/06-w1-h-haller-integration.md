# 06 — W1-H Haller physics-anchor integration

**Title:** W1-H Haller physics-anchor integration
**What to build:** 在 W1-P teacher/pseudo/consistency 基础设施上接入 train-only Haller-IVD closed-contour physics anchors，形成正式 W1-H。
**Blocked by:** 02-haller-anchor-extractor.md, 03-mode-checkpoint-contract.md, 05-w1-p-infrastructure.md
**Status:** done
**Primary seam:** Haller anchor seam → training/evaluation seam

## What to build

- W1-H 保留 7-channel 5×5 local-IVD model input；standard Haller-IVD 只作为 train physics anchor source。
- 使用已确认的 Haller engineering parameters：每帧 fluid vorticity mean、8-neighborhood maxima、32 个 `1.0*peak` 到 `0.1*peak` 的线性 levels、convexity defect `<=0.10`、minimum perimeter `>=8*max(dx,dy)`、outermost contour、`2*max(dx,dy)` unknown band、frame p60 low-IVD negative。
- Haller positive/negative known cells 进入 anchor masked BCE；boundary band、failed train frame 和 solid 进入 unknown/ignored。
- 继续使用 W1-P 的 EMA teacher、high-confidence pseudo-label 和 consistency ramp-up；formal `Lweak` 不包含 legacy p85。
- train Haller artifact 必须显式带 source、parameter hash、input/mask hash、coverage 和 failure count；test Haller GT 不能被 W1-H loader 读取。
- Haller 原始文献依据仍标记为“待核实”；不要把工程参数写成 canonical paper parameters。

## Acceptance criteria

- W1-H 能在 synthetic fixture 和真实 train frame 上读取独立 Haller anchor artifact，并完成 5–10 epoch smoke。
- batch 能区分 Haller known positive/negative、unknown、solid 和 failed-frame mask；unknown 不贡献 anchor BCE。
- W1-H checkpoint 和日志中的 loss source 明确为 `haller_anchor_train`，不含 `legacy_p85` formal loss source。
- 训练过程尝试读取 `haller_gt_test` 时 fail loudly；calibration/test artifact 不进入训练、pseudo-label 或 threshold tuning。
- anchor coverage、positive/negative/unknown count、failure count、pseudo acceptance 和 teacher/student disagreement 均可报告。
- 不覆盖既有 local-IVD dataset label、B0/B1 checkpoint 或 vendor 文件。

## Verification commands

```powershell
python -m pytest tests/test_w1_h_integration.py -q
python -m pytest tests/test_haller_anchor.py -q
```

```bash
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_w1_h_integration.py tests/test_haller_anchor.py -q
```

## Implementation result

- 新增 `w1_h.py` 与 `tests/test_w1_h_integration.py`：通过显式
  `haller_anchor_train` artifact resolver 将独立 Haller train anchor 接入现有
  7-channel、5×5 local-IVD pathline window；保留 base dataset 的公开采样 seam、
  split/window metadata 和 sampling source。
- W1-H batch 携带 known positive/negative、unknown、solid、failed-frame mask，
  使用 masked anchor BCE，并复用 W1-P 的 EMA teacher、confident pseudo-label、
  consistency 与 12-epoch ramp；formal loss/checkpoint source 固定为
  `haller_anchor_train`，artifact、parameter/input/mask hash、coverage、failure
  count 和 pseudo/teacher diagnostics 可审计。
- resolver、dataset adapter、collate、checkpoint load/save 均对 source、split、
  frame/window、sampling source、anchor hash 和 test/calibration provenance
  fail loudly；Haller Zotero `L2PX3NQX` 与工程参数状态保留为
  `pending_verification`。
- 验证：`python -m pytest tests/test_w1_h_integration.py -q`（7 passed）；
  `python -m pytest tests/test_haller_anchor.py -q`（9 passed）；相关回归命令
  （W1-H/W1-P/Haller artifact/shared contract/weak split）104 passed；完整
  `python -m pytest -q` 为 346 passed、6 warnings、19 个既有 evaluate fixture
  PermissionError（受保护的 `outputs/eval_test_ds` 与 `outputs/eval_tau_test_ds`）。
  真实 train frame 700 artifact 只读 loader 与 5 epoch CPU smoke 均成功：
  `global_step=5`、loss finite、`loss_source=haller_anchor_train`、
  `failure_count=0`，且 source、参数/input/mask hash 和文献待核实状态齐全。
- 后续仍需在新 `weak_supervision` 六数据集产物上生成/挂载 Haller train artifacts
  并进入 WS-7/W2；当前不改既有 `frac` 数据、vendor、旧 baseline 或下游实现。
