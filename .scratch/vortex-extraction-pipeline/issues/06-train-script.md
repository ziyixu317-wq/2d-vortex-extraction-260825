# 06: 训练脚本（train）

**What to build:** 自写训练循环（不依赖参考仓库任何训练代码）：BCE 损失、AdamW(wd 1e-6)、lr 1e-4、TwoStep 调度（warmup 60 epoch → 5e-6）、梯度裁剪 1.0、batch 100；每 epoch 存 checkpoint（含 optimizer 状态）支持断点续训；DataParallel/AMP 可选；全部超参走 YAML 配置；本地 CPU 冒烟 1~2 步。

**Blocked by:** 01、05

**Status:** done

- [x] CPU 冒烟：1~2 步训练 loss 数值有限且形状正确（下降趋势作为 Kaggle 全量的观察项）
- [x] checkpoint 保存/加载往返一致（含 optimizer 状态）
- [x] 断点续训从指定 epoch 恢复
- [x] YAML 配置驱动全部超参

## 完成记录

**做了什么**：新增 `train_kaggle.py`（自写训练循环，不 import 参考仓库任何训练代码）+ `config/pathline_transformer_cylinder.yaml` + `tests/test_train.py`（18 项，全量 131 passed）。

- 建构建面：`load_config`/`build_model_from_config`/`build_criterion_from_config`/`_make_dataset`（YAML data 段 → 数据集构造唯一入口，patch/窗口/十字采样/时间采样参数全驱动）；模型段沿用官方形态 `BaseSeg` 包装 `PathlineTransformerV0`（in_channels=7、PathlineGroups=64、KpathlinePerGroup=4、dmodel=144、3 层、k=16）+ `BCELoss`。
- 调度：`two_step_lr`/`TwoStepLR`（zepoch 粒度两段常数阶梯，语义与原仓库 `TwoStepLRScheduler` 逐行核实一致；state_dict 往返）。
- 训练核心：`run_epoch`（前向 → BCE → 反向 → 梯度裁剪 1.0 → step；AMP 单路径 autocast + GradScaler）/`evaluate`（val 损失监控；`_predict_loss`/`_iter_batches` 共享骨架）；选项：`data_parallel`、`amp`（YAML 开关，Kaggle T4×2 用）。
- checkpoint：`save_ckpt`/`load_ckpt`（model strict + optimizer/scheduler 状态 + epoch/metrics/config 元数据；DataParallel `module.` 前缀归一化）；latest 每 epoch 更新 + `save_freq` 里程碑快照。
- 主流程 `main`：`--config`/`--resume auto|none|路径`/`--epochs`/`--max-steps`；训练序每 epoch `set_epoch(epoch)`（同 (seed,epoch) 确定性 → 断点续训采样序与中断前一致）；val 缺片防御跳过；确定性 seed 复现测试守护。

**验证证据**：
- `tests/test_train.py` 18 项全绿：调度值（0/59→1e-4、60/199→5e-6 及边界）与状态往返；YAML 字段字面量断言（模型/训练/数据三段）+ 生产配置构建前向 (B,256) 域 (0,1) + `_make_dataset` 死配置回归；run_epoch 2 步冒烟（loss 有限、参数更新）+ 全局梯度范数 ≤1.0 裁剪守护 + evaluate 有限；checkpoint 往返（model 逐键、optimizer 动量逐张量一致，start_epoch=71）+ 续训后继续训练且 lr 处于恢复位置 + 元数据落盘；CLI 集成（1 epoch 冒烟 → resume auto 续到 epoch 1；同 seed 首 epoch loss 复现；val 片存在路径 → val_loss 入 metrics）。
- 真实数据 CPU 冒烟（生产默认模型 dmodel=144，`outputs/dataset` 产物，2 epoch × 2 步，batch 2）：epoch1 loss=0.9147 val=0.7877 → epoch2 loss=0.7153 val=0.5841（**早期下降趋势符合观察**）；checkpoint 5.6MB 含 model/optimizer/scheduler/epoch/metrics/config。
- 审查期间发现并修复：① val_ds 未 `set_epoch` → RuntimeError（真实冒烟复现；补 `test_val_path_when_split_present` 回归，修复后冒烟通过）；② YAML `data.patch_size/stride/t_win/window_step` 与 `groups/delta_frac/L/n_substeps` 未进入训练路径（Spec 轴死配置）→ `_make_dataset` 全量传入 + 回归测试。

**/code-review 双轴处置**：Standards 硬性 0；判断项 7 → 修 5（train/val 数据集构造重复 → `_make_dataset`；run_epoch/evaluate 骨架重复 → `_iter_batches`；AMP 分支重复前向 → `autocast(enabled=)` 单路径；测试样板/冗余 import → `fresh_small_model`/`fresh_adamw`/`assert_forward_shape_and_range` + 删函数内冗余 import；前向断言重复 → helper），保留 2（`build_model_from_config`/`build_criterion_from_config` 为 vendor 薄封装——边界情形成立；`small_model` fixture 删后由 helper 替代——属合并类）。Spec 3 条：缺失 1（YAML 死配置，已修 + 回归）；蔓延 1（checkpoint 双写——latest 每 epoch + save_freq=30 里程碑（对齐原仓库，200 epoch 约 ~230 文件），保留并记录）；口径 1（val 损失按训练同款 50% 平衡采样监控——注释明示口径；自然分布精度评估归票 08 弱定量表）。

**产物**：`outputs/train_smoke/`（gitignore，冒烟 checkpoint/config）。未决：无阻塞。下一步：按 frontier 票 07（Kaggle 训练：1 epoch 冒烟实测每步耗时 → 校准 epoch 样本数）可启动。
