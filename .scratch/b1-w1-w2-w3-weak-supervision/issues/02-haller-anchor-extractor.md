# 02 — Haller-IVD closed-contour anchor generator

**Title:** Haller-IVD closed-contour anchor generator
**What to build:** 实现单帧 `u/v + geometry mask` 到 standard IVD、闭合 contour、三态 Haller anchor/GT artifact 的独立 seam；Haller 原始文献算法细节仍标记为“待核实”。
**Blocked by:** None
**Status:** done
**Primary seam:** Haller anchor seam

## What to build

- 只在 fluid、非 solid 单元上计算二维瞬时 vorticity，并使用每帧 fluid vorticity mean 构造 `abs(omega - mean_fluid(omega))` standard IVD 候选场。
- 使用 fluid 8-neighborhood 寻找 local maxima；每个峰搜索从 `1.0 * peak` 到 `0.1 * peak` 的 32 个线性 contour levels。
- 只保留闭合、无 solid crossing、convexity defect `(A_hull-A_contour)/A_hull <= 0.10` 且 perimeter `>= 8 * max(dx,dy)` 的 contour。
- 对每个局部峰选择最外层合法 nested contour；多个合法候选的 positive/unknown 区域按显式 union 规则合并。
- contour 内部且不在 `2 * max(dx,dy)` unknown band 内为 positive；boundary band、未决区域和失败帧为 unknown；contour/band 外且 standard IVD 不高于该帧 fluid p60 的区域为 negative。
- train anchor 失败时整帧 fluid 为 unknown 并记录 failure；calibration/test GT 失败时 frame 标记 invalid，metric denominator 排除但 failure count 必须报告；禁止 p85 fallback。
- train anchor、calibration GT、test GT 使用分离的 source/artifact 标识，记录算法版本、完整参数、输入/mask hash、参数 hash、coverage 和 failure count。
- 允许使用 `scipy`、`scikit-image` 等成熟 contour、convex-hull 和 morphology 工具；不修改 vendor。

## Acceptance criteria

- synthetic Rankine-style fixture 能产出符合预期的局部峰、闭合 contour、positive interior、unknown band 和 negative region。
- contour level、convexity、minimum perimeter、outermost 和 union 规则在 artifact metadata 中可复现。
- solid 区域永远不生成 positive 或 negative anchor；solid-adjacent ambiguous cells 可保持 unknown。
- 非闭合 contour、过小 perimeter、convexity 超限和无合法 contour 的 fixture 返回规定的 unknown/invalid fallback，并记录 failure。
- train artifact 与 `haller_gt_test` artifact 不能互相覆盖；artifact source 不可由默认 loader 混淆。
- 测试只使用 synthetic fixture 和真实 train frame preview，不读取 test frame 来选择参数。

## Verification commands

```powershell
python -m pytest tests/test_haller_anchor.py -q
python -m pytest tests/test_haller_artifacts.py -q
```

```bash
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests/test_haller_anchor.py tests/test_haller_artifacts.py -q
```

## Implementation result

- `haller_anchors.py` now provides the independent single-frame seam and source-specific artifact I/O.
- Formal extraction uses the frozen engineering parameter set only; metadata retains the Haller Zotero
  candidate `L2PX3NQX` with `pending_verification` literature status.
- Targeted verification passed: 15 tests; related `weak_labels`/`geometry` regression passed: 50 tests.
- A real train-only `pipedcylinder2d` frame 700 preview was generated at
  `E:\codex\AI CFD\haller_preview_b1_ticket\pipedcylinder2d\frame700\`.
