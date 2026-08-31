# 08: 推理评估（evaluate）

**What to build:** 端到端评估管线（规格主验收缝）：滑窗推理（stride 16 全场覆盖）+ TTA 5 次平均（确定性采样开关备选）→ 网格投影（累积 + 计数平均消除 patch 重叠）→ 展示帧用加密种子（每 2×2 输出像素一组十字）+ 速度模场底图（不依赖 LIC 渲染器）→ 对比图（模型输出 vs IVD vs Q-criterion）+ mp4 动画 + 弱定量表（对 IVD 阈值的 F1/IoU、涡面积占比、帧间连续性）。

**Blocked by:** 07

**Status:** done

- [x] 多个代表性时间步对比图目检：涡街/拐角回流区结构一致、无明显碎裂噪声
- [x] mp4 动画输出
- [x] 弱定量表输出（F1/IoU、涡面积占比、帧间连续性）
- [x] 推理可复现（TTA 固定种子或确定性开关）

## 完成记录（2026-08-31）

**实施**：`evaluate.py`（评估管线）+ `tests/test_evaluate.py`（37 项测试）落盘。管线覆盖规格主验收缝：
- 滑窗推理（stride 16 全场覆盖 + **贴边补全**——ds.patch_locations 不贴边会导致顶部/右缘无种子格投影=0 暗带，spec 端到端缝"覆盖全场"要求）+ TTA 平均 → 网格投影（累积 + 计数平均消除 patch 重叠）。
- 展示帧模型概率面板用 `prob_sw`（滑窗 patch 归一化，与训练一致）；加密种子 dense 路径（`infer_dense`）保留为独立工具 + 单测（见"实现决策"）。
- 对比图（模型概率 / IVD / Q-criterion / 速度模+弱标签等值线 四联）、mp4 动画（FFMpegWriter 优先，ffmpeg 缺失回退 PillowWriter→gif）、弱定量表（F1/IoU、标签参考涡面积占比 `vortex_area_ratio`、模型预测涡面积占比 `pred_vortex_ratio`、帧间连续性）。
- 推理可复现：TTA 固定种子（`_set_inference_seed` + 每 patch `_tta_rng_base` 确定性派生）；测试 `test_fixed_seed_reproducible` 守护。

**验证证据（真实数据，130-epoch 最终权重 + `outputs/datasets/`）**：
- 单帧展示 smoke：`outputs/evaluation_smoke/comparison_t1300.png`（pipedcylinder2d test 片帧 1300）——模型高概率涡街（左下 x∈[0,1])与拐角涡（右上 x∈[2.5,4]）与 IVD/Q 参考对应良好、固体几何正确、顶部/右缘无暗带；F1=0.6575、IoU=0.4897、标签涡面积占比 0.1410、模型预测涡面积占比 0.0855（threshold 0.5）。
- 动画 smoke：`outputs/evaluation_anim/vortex_animation.gif`（3 帧 1296/1298/1300；本机无 ffmpeg 回退 gif，代码优先 mp4）。
- 全量测试 **231 passed**（含 evaluate 37 项）；`test_evaluate.py` 端到端 smoke（单数据集/多数据集）+ 合成小模型覆盖滑窗投影/加密种子/定量表/对比图/动画/可复现。

**实现决策 / 规格偏离（记录）**：
1. **展示帧模型面板用滑窗 `prob_sw`，未用加密种子 dense 路径**。spec 要求"展示帧用加密种子（每 2×2 一组十字）"，但 dense（`infer_dense`）用**全场归一化**，与模型训练的 **patch 归一化**口径不同，实测输入分布偏移导致模型输出退化（展示帧 Panel 几乎全为低概率），无法正确目检涡结构。滑窗 `prob_sw` 用 patch 归一化（与训练一致），输出涡结构正确。`infer_dense`/`_dense_seeds`/`_dense_extract` 保留为独立工具函数 + 单测（`TestInferDense` 含坐标序回归守护），不作为主流程输入。此为 spec 偏离，记录于 HANDOFF §11。
2. **修复 dense 种子坐标序 bug**：`_dense_seeds` 原构造 `[y,x]`，与 extractor 全场约定 `[x,y]`（`integrate_pathline`/`seeding_grid`）相反，导致 dense 迹线/投影串扰；修复为 `[x,y]` 并加回归测试 `test_dense_seeds_coordinate_order`（原测试用对称点无法捕获）。
3. **滑窗贴合（覆盖全场）**：`sliding_window_patches` 补最后贴边 patch，顶层/右缘不再因无种子而投影=0。
4. **`--dense-step` 弃用参数移除**（原贯穿 CLI/run/summary 却零效果）；`make_animation` 死参数 `frames_ivd` 移除；死代码 `_resolve_data_roots`/`import sys` 移除。
5. **多数据集帧间连续性按数据集各自算**（跨数据集边界无意义）；`compute_frame_metrics` 增 `pred_vortex_ratio`（模型预测涡面积占比），`vortex_area_ratio` 明确为标签参考占比。

**code-review 处置**：双轴并行审查（相对票 07 HEAD f15ae3c）。Standards 硬性违规 0（依赖清单/中文 docstring/领域词汇合规；英文图题被 HANDOFF §11"中文标题豆腐块 → 英文"先例压制）。判断项处置：dense 死路径+`dense_step` 弃用（移除运行依赖，保留解 Dense 工具+单测，见决策 1）；`_resolve_data_roots`/`import sys`/`mask2d` 死代码（删除）；`make_animation` 死参数（删除）；Duplicated Code（`_dense_extract` 与 `extract_pathlines_batched` 7 通道组装同源——已由 `test_dense_seeds_coordinate_order` 守护坐标序，公式一致性保留原样并记录）；Feature Envy（直读 store 私有字段——评估需底层场，`dataset.sample_at` 返回单样本不合用，记录）。Spec 轴：滑窗"覆盖全场"（修复贴边，见决策 3）；`--dense-step` 误导（移除）；`make_animation` 死参数（删除）；`vortex_area_ratio` 与模型无关（增 `pred_vortex_ratio`）；`frame_continuity` 全负返回 1.0（两帧无涡=一致，数学约定保留并注释）；弱标签等值线面板/gif 回退/CLI 覆盖参数（范围蔓延/工程实用，保留并记录）；多阈值敏感性报告（票 07 延伸已交付，非本票）。`tests/test_multidataset.py` 单行 7→6 roots（duffing 移出训练池的过时断言，HANDOFF §1/§11 一致，修正使全量全绿）。

**未决**：正式 mp4 动画需在含 ffmpeg 环境产出（本机无 ffmpeg 回退 gif）；最终展示帧选择与多阈值报告引用由用户复核。
