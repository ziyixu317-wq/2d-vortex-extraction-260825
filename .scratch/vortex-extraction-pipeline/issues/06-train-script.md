# 06: 训练脚本（train）

**What to build:** 自写训练循环（不依赖参考仓库任何训练代码）：BCE 损失、AdamW(wd 1e-6)、lr 1e-4、TwoStep 调度（warmup 60 epoch → 5e-6）、梯度裁剪 1.0、batch 100；每 epoch 存 checkpoint（含 optimizer 状态）支持断点续训；DataParallel/AMP 可选；全部超参走 YAML 配置；本地 CPU 冒烟 1~2 步。

**Blocked by:** 01、05

**Status:** ready-for-agent

- [ ] CPU 冒烟：1~2 步训练 loss 数值有限且形状正确（下降趋势作为 Kaggle 全量的观察项）
- [ ] checkpoint 保存/加载往返一致（含 optimizer 状态）
- [ ] 断点续训从指定 epoch 恢复
- [ ] YAML 配置驱动全部超参
