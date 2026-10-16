# 二维非定常流场的迹线 Transformer 涡结构学习

当前方法为 **standard IVD + P95/P80 + EMA teacher 的七源 W1-P**。原六源加 MFC 翼型连续流场的三 seed 开发实验已完成；每 seed 100 epoch、39,100 次更新、2,000,000 个 sampled patches。W3 对照两 seed 按登记条件判为 no-go，核心方法保留 W1-P。

## 当前结果

固定全局概率阈值 `p=0.1`；每源三个 calibration 帧先合并计数，再进行等源宏平均。下表为三 seed 均值 ± 样本标准差。

| Calibration 范围 | F1 | IoU | AUPRC |
|---|---:|---:|---:|
| 原六源 | 0.959088 ± 0.000670 | 0.926053 ± 0.001389 | 0.988858 ± 0.001063 |
| MFC | 0.983234 ± 0.003058 | 0.967033 ± 0.005927 | 0.999100 ± 0.000281 |
| 七源 | **0.962538 ± 0.001011** | **0.931907 ± 0.001953** | **0.990321 ± 0.000934** |

这里评价的是直接 observed-known 支持上与 P95/P80 操作性弱参考的一致性。MFC 的直接有效流体覆盖约 **0.334%**；IVD 同时参与输入和弱参考，不能将这些结果解释为全场恢复或独立物理准确率。旧 final-test 保持封存。

- [三 seed 权重、逐源结果及选择说明](results/seven-source-w1p/README.md)
- [完整三 seed 统计 JSON](results/seven-source-w1p/three-seed-summary.json)
- [训练与评价复现说明](docs/seven_source_reproduction.md)
- [方法、监督与支持语义](docs/seven_source_method.md)

默认展示权重为 [seed0](results/seven-source-w1p/seed0/epoch_0100.pt)，其七源 calibration macro F1、IoU 在三个 seed 中最高；seed1 的 AUPRC 和最差源 F1 更好。三份完整 checkpoint 均保留，方法判断以三 seed 证据为依据。

## 运行入口

| 工作 | 入口 |
|---|---|
| 七源 W1-P seed1/2 完整训练与恢复 | `run_ticket11_seven_source_core.py` |
| 七源共享训练实现与单批 CPU 预取 | `run_mfc_mixed_w1p_training.py` |
| 完整 epoch 边界的 batch32→64 迁移 | `migrate_mfc_batch_size_checkpoint.py` |
| 完整模型的 42 帧开发预测与单帧复载测量 | `evaluate_mfc_mixed_w1p.py` |
| 逐源、三 seed 与共同支持报告 | `run_ticket12_seven_source_report.py` |
| W3 配对诊断实现 | `run_ticket09_seven_source_w3.py` |
| 六源 standard IVD / P95/P80 参考 | `ticket03_standard_ivd_artifacts.py` |
| MFC 连续流场读取与参考生成 | `mfc_pathline_source.py`、`prepare_mfc_ivd_statistics.py`、`prepare_mfc_w1p_reference.py` |

运行依赖见 [requirements.txt](requirements.txt)。这些实验实际使用 Python 3.12.14、PyTorch 2.7.1+cu118、CUDA 运行时 11.8；训练和推理入口使用 CUDA。最终 batch64 段峰值 allocated 约 15.24 GiB，reserved 约 16.63 GiB，正式设置的 allocator 上限为 18 GiB。

原始速度、几何、standard-IVD/P95P80 参考及完整预测数组需要另行提供。发布 checkpoint 保留原训练状态与来源身份；原 loader 的来源、split 和 sampling 校验保持，任意移动数据目录不能自动保证旧权重加载。具体要求见复现说明。

## 归属与许可

模型实现来自 PyFlowVis / DeepUtils 并保留现有 [Apache 2.0 LICENSE](LICENSE) 和 [NOTICE](NOTICE)。相关论文：Xingdi Zhang、Peter Rautek、Markus Hadwiger，*VortexTransformer: End-to-End Objective Vortex Detection in 2D Unsteady Flow Using Transformers*，Computer Graphics Forum，2025，e70042。

MFC 工况资产保留其目录内的 [MIT 许可](config/mfc_airfoil_training/LICENSE)。
