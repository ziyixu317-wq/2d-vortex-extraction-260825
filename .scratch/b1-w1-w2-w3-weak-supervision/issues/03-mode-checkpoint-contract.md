# 03 — Shared method-mode and checkpoint contract

**Title:** Shared method-mode and checkpoint contract
**What to build:** 建立 B0/B1/W1/W2/W3 的显式 mode、feature schema、输入适配、batch source/mask 和可恢复 checkpoint 契约，为后续方法票提供统一训练边界。
**Blocked by:** 01-split-label-provenance.md
**Status:** done
**Primary seam:** training/evaluation seam 的公共 contract

## What to build

- 固定 B0/W1/W2/W3 的 7 通道顺序 `[px, py, t, ivd, distance, u, v]`；固定 B1 的 6 通道顺序 `[px, py, t, distance, u, v]`。
- 在 vendor 外部实现 channel-selecting adapter 和显式 schema 校验；不得修改 `vendor/DeepUtils`。
- 让 batch/loss 接口携带 label mask、unknown mask、label source、split name 和 anchor/pseudo provenance，避免把 sampling membership 当监督。
- 建立 mode-aware 的 model/loss/checkpoint dispatch；不兼容的 channel schema、split 或 mode 必须拒绝加载。
- checkpoint 至少保存 `format_version`、mode、feature schema/channel order、dataset/split config、t_win/sampling config、student、可用时的 EMA teacher/projection head、optimizer/scheduler、epoch/global step、metrics、seed/RNG、anchor hash、calibration policy 和 warm-start 标志。
- 主实验默认 `warm_start_aux=false` 且从头初始化；旧 B0 checkpoint 只能在显式 auxiliary mode 下加载。

## Acceptance criteria

- B0/W1/W2/W3 的 7-channel synthetic batch 和 B1 的 6-channel synthetic batch 均能通过 schema 校验；错误顺序或错误通道数 fail loudly。
- 训练 batch 能区分 `legacy_p85`、W1 label、Haller train anchor、calibration GT 和 test GT source；训练路径拒绝 test source。
- student-only、student+teacher、student+teacher+projection head checkpoint 都能保存并恢复，恢复后输出、optimizer/scheduler state 和 RNG contract 可验证。
- 不兼容 checkpoint（wrong mode、wrong feature schema、wrong split 或 wrong anchor hash）被拒绝且错误信息说明原因。
- vendor 文件内容和旧 baseline behavior 没有被修改。

## Verification commands

```powershell
python -m pytest tests/test_weak_supervision_contract.py -q
python -m pytest tests/test_vendor_migration.py -q
```

```bash
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_weak_supervision_contract.py -q
```

## Implementation result

- Added `weak_supervision_contract.py` and `tests/test_weak_supervision_contract.py`; updated
  `train_kaggle.py` with explicit mode-aware factory and contract-checkpoint aliases.
- B1 uses an external raw 7-channel → model-facing 6-channel adapter; B0/W1/W2/W3 remain on the
  fixed 7-channel schema.
- Batch provenance, source/split/leakage guards, mask-aware loss seam, and strict checkpoint
  save/load/resume/inference/legacy contracts are fail-loud and covered by synthetic tests.
- Verification: contract 63 passed; vendor migration 8 passed; related regression 171 passed;
  full regression 323 passed with 19 known PermissionErrors from the protected evaluate fixtures.
- Method-specific training and end-to-end experiment wiring remain in WS-4–WS-10/downstream tickets.
