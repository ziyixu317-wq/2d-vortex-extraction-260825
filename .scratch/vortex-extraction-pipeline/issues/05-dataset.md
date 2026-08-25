# 05: 数据集类（dataset）

**What to build:** 弱标签迹线数据集：train/val/test 按时间划分（10 / 12.5 / 15 s，帧 0-1000 / 1000-1250 / 1250-1500，无时间泄漏）；patch 32×32 stride 16、窗口 T_win=24 帧、窗口起点步长 4 帧；u,v 与预计算 IVD 存 memmap（IVD 一次算好）；7 通道归一化（px,py → patch 内 [-1,1]；t → [0,1]×t_scale 默认 0.25；ivd 标准化；distance 用归一化坐标；u,v ÷ 全局最大速度）；每 epoch 40000 样本（下限 20000）、50% 正样本过采样；返回 `((dummy_field, pathlines), labels)` 匹配模型输入。

**Blocked by:** 03、04

**Status:** ready-for-agent

- [ ] 单样本生成 <5ms
- [ ] 时间划分无泄漏（train/val/test 帧区间互斥）
- [ ] 每样本 256 条×16 步×7 通道；标签与种子点 IVD 阈值判定一致
- [ ] 正样本占比≈50%（过采样生效）
