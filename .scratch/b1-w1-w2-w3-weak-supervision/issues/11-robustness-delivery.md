# 11 — Robustness protocol and delivery documentation

**Title:** Robustness protocol and delivery documentation
**What to build:** 在 final headline 结果之后执行固定 clean-Haller GT 的 noise/downsampling robustness protocol，并完成结果索引、README 和 HANDOFF 交付同步。
**Blocked by:** 10-final-experiments.md
**Status:** ready-for-agent
**Primary seam:** training/evaluation seam 的 robustness extension

## What to build

- clean CFD 的 Haller-IVD GT 只生成一次，保存输入、geometry、Haller 参数和 artifact hash；该 clean GT 在所有 robustness case 中保持不变。
- 对 `u/v` 注入零均值 Gaussian noise，solid/geometry mask 不加噪，不做 clipping；噪声标准差为每数据集 train fluid speed RMS 的 `alpha ∈ {0.01, 0.05, 0.10}`。
- 对输入执行 anti-aliased、mask-aware factor 2 和 factor 4 downsampling，再插值回模型网格；记录聚合、抗混叠、mask 和插值配置。
- 对扰动/重建后的 `u/v` 使用相同固定参数重新计算 local-IVD 和 Haller-IVD；不因 robustness 结果修改 clean GT 或 Haller 参数。
- 分别报告 Model vs clean Haller GT、Model vs recomputed Haller-IVD、recomputed Haller-IVD vs clean Haller GT，区分模型误差与 physics label drift。
- 更新结果索引、README 和 HANDOFF §12/§11，明确 clean GT hash、扰动强度、recompute 参数和报告路径。

## Acceptance criteria

- clean Haller GT hash 在所有 robustness case 中一致，且扰动运行不能覆盖 clean artifact。
- noise case 的字段、alpha、train speed RMS source、solid mask policy 和 no-clipping policy 可从 manifest 复现。
- factor 2/4 downsample case 记录 anti-alias、mask-aware aggregation、reconstruction/interpolation 和 recompute 配置。
- 三类 Model/Haller comparison 均有逐数据集 P/R/F1/IoU 或等价 confusion report，并保留 known/unknown/invalid coverage。
- robustness 脚本不使用 test 结果调节 clean GT、模型、threshold、gate 或 Haller 参数。
- README、结果索引和 HANDOFF §12/§11 能链接到 final 与 robustness 产物；旧 baseline 记录仍可追溯。

## Verification commands

```powershell
python -m pytest tests/test_robustness_protocol.py -q
python -m pytest tests/test_evaluation_report.py -q
```

```bash
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_robustness_protocol.py tests/test_evaluation_report.py -q
```
