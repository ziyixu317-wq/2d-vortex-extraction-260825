# 10: 整理交付（阶段 7）

**What to build:** 阶段 7 收尾：结果目录归档、复现 README（数据→服务器训练→评估全流程）、参数表（τ / t_scale / epoch 样本数等定值回写 HANDOFF §6）、checkpoint 归档清单；HANDOFF §11 记录交付状态。

**Blocked by:** 09

**Status:** done

**当前执行说明**：复现入口以 `server/README.md` 为准；下方完成记录保留交付时的审计信息。

- [x] README 按步骤可复现全流程
- [x] 参数表定值回写 HANDOFF §6
- [x] 结果与 checkpoint 归档清单完整
- [x] HANDOFF §11 记录交付状态

## 完成记录（2026-08-31）

**实施（阶段 7 收尾）**：
- **复现 README**：新增根 `README.md` ——（一）目录结构简略（指向 HANDOFF §4）；（二）从零复现四步（prepare_multi 数据预处理 → Remote-SSH 服务器运行 `train_kaggle.py` → `evaluate.py` 推理评估 → `--tau-sensitivity` τ 敏感性）；（三）结果表（130-epoch 权重、多数据集结算指标 P=0.4967/R=0.9549/F1=0.6535/IoU=0.4853、对比图/动画/弱定量表、τ 敏感性表、多阈值报告）；（四）checkpoint 归档清单；（五）测试运行命令。链接 HANDOFF/spec/票目录。
- **参数表复核 §6**：复核确认现行——τ=p85（单数据集 0.6539/0.6512/0.6113；多数据集逐数据集 p85）、t_scale 0.25、epoch 样本数 20000、TTA 5、最小涡面积 5×5、结算口径 130 epoch——**均已在 §6**（票 04/05/06/07 已回写），本票无新增定值，未改动数值。
- **结果与 checkpoint 归档清单**：整理入 README「结果」「checkpoint 归档清单」两张表，当前交付目录为 `outputs/_ckpt130/train_multi/`（130-epoch 最终权重/test_f1/bench_info）以及 evaluation、τ 敏感性和弱标签分析产物。
- **HANDOFF §11**：追加变更日志条目（票 09+10 完成、全量 240 passed、130-epoch 权重恢复至 `outputs/_ckpt130/train_multi/`）；§3 验收 4（评估指标级 τ 敏感性）与 §4 evaluate.py 职责行同步更新。

**验证证据**：全量 `python -m pytest tests -q` → **240 passed**（231 基线 + 票 09 的 9 项 τ 用例）。（注：§11 与票 09 记录中的「240 passed」与此一致。）

**未决**：仅剩用户侧执行项——最终展示帧目检复核、多数据集联合定格表（票 08 记录同）；本地 130-epoch 权重已从用户 Kaggle 快照 `results (2).zip` 恢复。
