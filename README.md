# 迹线 Transformer 涡提取项目（cylinder_vortex_pipeline）

用 VortexTransformer 论文（CGF 2025，DOI 10.1111/cgf.70042）的**迹线 Transformer**
（`PathlineTransformerV0`）在 2D 非定常仿真流场上做**涡提取**：从迹线（pathline）直接
输出每条迹线的涡概率，再投影回网格。

> **唯一权威上下文是 [`HANDOFF.md`](HANDOFF.md)**：已拍板决策（§1）、已核实事实（§2）、
> 阶段计划（§5）、默认参数表（§6）、风险预案（§7）、工作流（§9）与更新协议（§11）
> 全部在其中。遇到任何冲突，以 HANDOFF 为准，并在其 §11 记一条修正。本 README 是
> **从零复现路径**的入口说明，不替代 HANDOFF。

---

## 目录结构（简略，完整见 HANDOFF §4）

```
cylinder_vortex_pipeline/
├── vendor/DeepUtils/      # 从 PyflowVis-main 迁移的最小纯 torch 子集（Apache 2.0 署名）
├── geometry.py            # 固体掩膜（逐数据集预处理；不进模型输入）
├── extractor.py           # 迹线生成（256 条/样本，7 通道，RK4+三线性插值，掩膜处理）
├── weak_labels.py         # IVD（5×5 邻域偏差）+ Q-criterion + τ 标签 + 多阈值报告
├── dataset.py             # WeakLabelPathlineDataset + MultiDatasetPathlineDataset
├── prepare_multi.py       # 多数据集逐数据集预处理驱动（geometry→IVD/τ→memmap+meta）
├── train_kaggle.py        # 自写训练脚本（TwoStep、断点续训、DataParallel；可选 --report-f1）
├── evaluate.py            # 推理评估（滑窗 TTA→网格投影→对比图/动画/弱定量表 + τ 敏感性）
├── config/                # 训练/数据配置（单数据集 cylinder / 多数据集 multi）
├── kaggle/                # Kaggle Notebook/打包/分块/自检/操作手册（见 kaggle/README.md）
├── tests/                 # pytest 测试（含验收接缝测试）
├── HANDOFF.md             # 唯一权威上下文
└── .scratch/vortex-extraction-pipeline/   # 规格书 spec.md + 一票一文件 issues/
```

## 依赖

`torch`、`numpy`、`h5py`、`PyYAML`（import 名 `yaml`）、`matplotlib`、`tqdm`。
本地 CPU（torch 2.10.0+cpu，无 CUDA）跑冒烟与单测；全量训练/评估在 Kaggle GPU。

---

## 从零复现（数据 → 训练 → 评估）

### 1. 数据预处理（多数据集逐数据集）

原始 nc 数据集放在 `../CFD数据集/`（中文路径，h5py 直读）。对每个数据集跑：

```powershell
python prepare_multi.py            # 默认扫 ../CFD数据集/，产出 outputs/datasets/<名>/
python prepare_multi.py --names boussinesq,cylinder2d,doublegyre2d,fourcenters2d,jungtelziemniak2d,pipedcylinder2d
```

产物：`outputs/datasets/<名>/geometry`（掩膜）、`.../dataset`（u/v/ivd/label/mask memmap +
meta.json，含逐时间片 τ、speed_max、ivd μ/σ）、`.../previews`（目检图）+ 顶层
`multi_meta.json`（逐数据集 shape/slices/taus/统计汇总）。约 3.8GB，走 Kaggle Dataset 打包。

### 2. 训练（Kaggle T4×2，130 epoch 结算）

详见 Kaggle 操作手册 `kaggle/README.md`（打包 Dataset A → 上传 Notebook → Run All 分块
续训 → 收尾 `--report-f1 --f1-split test`）。关键命令：

```powershell
# 本地打包 Dataset A（多数据集布局 data/<nc> + datasets/<名>/）
python kaggle\prepare_dataset_a.py --nc <各 nc 路径> --dataset-dir <各 outputs\datasets\*\dataset> `
    --out kaggle_dataset_a_multi --zip

# （Kaggle Notebook cell 6 训练命令，分块 ≤7.5h/会话，--resume auto 续训）
python train_kaggle.py --config config/pathline_transformer_multi.yaml --resume auto --epochs <块目标>
# 收尾块自动带 --report-f1 --f1-split test → outputs/train_multi/*_test_f1.json
```

本地 CPU 冒烟：`python train_kaggle.py --config config/pathline_transformer_cylinder.yaml
--epochs 2 --max-steps 2 --device cpu`（1~2 步，验证前向与 loss 下降）。

### 3. 推理评估（本地 CPU 可跑）

用 130-epoch 权重对 pipedcylinder2d test 片做评估（TTA 滑窗 → 投影 → 对比图/动画/定量表）：

```powershell
python evaluate.py --config config/pathline_transformer_multi.yaml `
    --ckpt outputs/_ckpt130/train_multi/pathline_transformer_multi_ckpt_latest.pth `
    --data-root outputs/datasets/pipedcylinder2d/dataset `
    --out-dir outputs/evaluation --tta 5 --device cpu `
    --display-frames 1260,1300,1400 --split test
```

### 4. τ 敏感性评估（弱标签阈值敏感，票 09）

```powershell
python evaluate.py --config config/pathline_transformer_multi.yaml `
    --ckpt outputs/_ckpt130/train_multi/pathline_transformer_multi_ckpt_latest.pth `
    --data-root outputs/datasets/pipedcylinder2d/dataset `
    --tau-sensitivity --tau-out-dir outputs/tau_sensitivity `
    --display-frames 1260,1300,1400 --tta 1 --split test
```

产物：`outputs/tau_sensitivity/tau_sensitivity_table.json`（τ 候选 × F1/IoU/涡面积占比/
预测涡面积占比，逐数据集 + 全局）+ `tau_sensitivity_report.md`（敏感性表 + 稳健性结论）。

---

## 结果

| 交付物 | 位置 |
|---|---|
| 130-epoch 最终权重 | `outputs/_ckpt130/train_multi/pathline_transformer_multi_ckpt_latest.pth`（5.6MB；epoch 129/130，train_loss 0.0797） |
| 多数据集结算指标（test 片自然分布，阈值 0.5） | `outputs/_ckpt130/train_multi/pathline_transformer_multi_test_f1.json`：**P=0.4967 / R=0.9549 / F1=0.6535**，IoU=0.4853，n=5,120,000 迹线 |
| 对比图 + 动画 + 弱定量表 | `outputs/evaluation_smoke/`、`outputs/evaluation_anim/`（`comparison_t*.png`、`vortex_animation.gif/mp4`、`quantitative_table.json`） |
| τ 敏感性表 + 报告 | `outputs/tau_sensitivity/`（票 09） |
| 多阈值(label 级)报告 | `outputs/weak_labels/multi_tau/`（多阈值敏感性统计 + 目检图，票 07 延伸） |

> **结算指标解读（HANDOFF §11 票 07 延伸）**：P<0.5 + R≈0.95 为联合过分割，主因
> boussinesq τ 跨时间片漂移（train τ=0.0555 vs test τ=0.5955）；其余数据集 F1 0.88-0.96
> 健康。正式网格投影级弱定量表见票 08（evaluate.py）。

## checkpoint 归档清单

| 目录 | 内容 |
|---|---|
| `outputs/_ckpt130/train_multi/` | 130-epoch 最终权重 `pathline_transformer_multi_ckpt_latest.pth` + `_test_f1.json` + `bench_info.json`（627.8 s/epoch、20000 样本） |
| `outputs/baseline_25ep/outputs/train/` | 单数据集 25-epoch 基线权重（p95 旧标签时代的中间模型，历史参考） |
| `outputs/train_smoke/` | 本地 CPU 冒烟模型（2-epoch，管线验证用，非交付权重） |

> 完整/历史 checkpoint 需用户从各自 Kaggle 会话快照下载归档（本地仅存最终权重与
> 里程碑缺失段，见 HANDOFF §11）。

## 运行测试

```powershell
python -m pytest tests -q          # 全量测试（约 240 项）
```

---

## 相关文档

- `HANDOFF.md` — 唯一权威上下文（决策/事实/参数/风险/工作流/变更日志）
- `kaggle/README.md` — Kaggle 训练操作手册（上传、分块、续训、故障排查）
- `.scratch/vortex-extraction-pipeline/spec.md` — 规格书（测试接缝与验收标准）
- `.scratch/vortex-extraction-pipeline/issues/` — 一票一文件（01..10）
- `CLAUDE.md` — agent 入口说明（唯一权威仍是 HANDOFF）
- `vendor/DeepUtils/LICENSE`、`vendor/DeepUtils/NOTICE` — 迁移代码的 Apache 2.0 署名
