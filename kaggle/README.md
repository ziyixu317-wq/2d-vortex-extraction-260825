# Kaggle 训练操作手册（票 07：上传与 200 epoch 训练）

本手册是从本仓库到 Kaggle 完成 200 epoch 全量训练的**用户操作路径**。代码侧配套：

| 文件 | 用途 |
|---|---|
| `kaggle/train_kaggle.ipynb` | Kaggle Notebook（导入后 Run All；自检/步速校准/分块训练/版本发布/收尾归档） |
| `kaggle/prepare_dataset_a.py` | 本地打包 Dataset A（nc + prepare_dataset 产物 + manifest 审计） |
| `kaggle/chunking.py` | 分块规划（12h 会话 → ≤8h/块，纯函数，有测试） |
| `kaggle/self_check.py` | Notebook 环境自检（验收 1；本地可用 `--device cpu` 先验） |
| `train_kaggle.py --report-f1` | 训练完成后记录 val 自然分布 F1（验收 4） |

**谁能做哪步**：步骤 0（打包）与步骤 1（推送）在你本机普通终端做；步骤 2–5 在 Kaggle 网页做；步骤 6–7 在 Kaggle Notebook 中执行（每会话一块 ≤8h，总计约 3–4 个会话）。

---

## 0. 本地打包 Dataset A

Dataset A = 原始 nc 数据 + 训练预处理产物（`outputs/dataset/` 的 meta.json + u/v/ivd/label/mask memmap ≈1.3GB + `outputs/weak_labels/` 目检图）。本机已跑过票 02–05 的预处理，直接打包：

```powershell
cd "C:\Users\徐子屹\Desktop\AI CFD\cylinder_vortex_pipeline"
python kaggle\prepare_dataset_a.py --nc "..\CFD数据集\pipedcylinder2d.nc" `
    --dataset-dir outputs\dataset --aux-dirs outputs\weak_labels `
    --out kaggle_dataset_a --zip
```

产物：`kaggle_dataset_a/`（含 `manifest.json`：逐文件 sha256，供 Kaggle 端自检核对）与 `kaggle_dataset_a.zip`（≈2GB，Kaggle 网页可直接上传；`kaggle_dataset_a/` 与 zip 均不在 git 内，请勿提交）。

## 1. 推送代码到 GitHub

```
git push -c http.sslBackend=openssl origin main
```

Kaggle Notebook 通过 `git clone https://github.com/ziyixu317-wq/2d-vortex-extraction-260825.git`
获取代码（notebook 的 `REPO_URL` 变量）。仓库须为公开（训练不依赖私有凭据）。

## 2. 创建 Kaggle Dataset（Dataset A）

- 网页：Kaggle → Datasets → New Dataset → Upload 上传 `kaggle_dataset_a.zip`（或 API：`kaggle datasets create -p kaggle_dataset_a -t "2d vortex pathline transformer" -v`）。
- 名称不限（notebook 自动探测含 `dataset/meta.json` 的 input）。公开/私有均可（训练非竞赛，私有即可）。

## 3. 创建 Notebook 并配置运行环境

1. Kaggle → Code → New Notebook → 上传 `kaggle/train_kaggle.ipynb`（File → Upload Notebook）。
2. Notebook Settings：**Accelerator = GPU T4 x2**、**Internet = ON**。
3. Add Input → 挂载第 2 步的 Dataset A。

## 4.（可选）checkpoint 数据集与 secrets（跨会话自动续训，模式 A）

1. Kaggle → Datasets → New Dataset：先建一个**空**数据集（占位 slug，如 `yourname/vortex-train-ckpt`），notebook 的 `CKPT_DATASET_SLUG` 填该 slug；
2. Notebook → Add-ons → Secrets：添加 `KAGGLE_USERNAME` / `KAGGLE_KEY`（Kaggle 个人设置页生成 API key）→ Settings 里把两个 secret 挂到本 notebook；
3. 之后笔记本内 `kaggle datasets version` 会在块尾自动发布新版本，下次会话 `kaggle datasets download --unzip` 自动还原 `outputs/train/`。

不配置（模式 B，手动）则跳过：每次会话把上一会话的 `ckpt_snapshot.zip`（在 `/kaggle/working`，会话结束前下载）上传为 Dataset 并 Add Input；或挂载上一会话输出的 checkpoint 目录。

## 5. 执行（Run All）

每个 cell 的作用与产物：

| Cell | 作用 | 产物/输出（/kaggle/working/repo 下） |
|---|---|---|
| 1 | 安装 h5py/PyYAML/matplotlib/tqdm/kaggle | 版本打印 |
| 2 | git clone 代码 | `repo/` |
| 3 | 挂载 Dataset A + 还原 checkpoint | `outputs/dataset/`（symlink） |
| 4 | **验收 1**：环境自检（vendor import + 数据加载 + 模型前向） | 自检打印 |
| 5 | **验收 2**：1 epoch 实测步速（全量样本，**跳过 val**）+ 预算检查 | `outputs/bench_info.json` + 分块计划（校准 ≈17.6 min） |
| 6 | **验收 3**：分块训练（本块 1 块；`--resume auto` 续训；每 10 epoch 一次 val ≈17.5min） | `outputs/train/*ckpt_latest.pth` 每 epoch + E 里程碑 |
| 7 | 块尾打包（模式 A 发布 Dataset 新版本 / 模式 B 手动下载） | `ckpt_snapshot.zip` |
| 8 | **验收 4**：训练完成后打印 val F1 + 最终归档 | `final_ckpt.zip` + `val_f1.json` |
| 9 | （可选）**中途预览**：单帧 模型 vs IVD vs 弱标签三联图 | `outputs/preview/prob_vs_ivd_t1300.png` |

**日志节奏提示**：校准后若无输出，是在跑 val 评估（无进度条、约 17.5 分钟；`PYTHONUNBUFFERED=1` 后 print 实时）。正常节奏 = 校准（~18 min）→ [train] epoch 1 训练 tqdm（200 步 × ~5.3s ≈ 17.6 min/epoch）→ 每 10 epoch 停顿 ~17.5 min 跑 val。

**跨会话免重复校准**：`bench_info.json` 会在 cell 7 块尾打包时随 checkpoint 数据集一起保存（`outputs/train/bench_info.json`），下个会话 cell 3 还原后 cell 5 自动复用（优先本会话实测 → 还原值），打印来源——**首个会话校准一次，后续会话省 ~18 min/会话**。

**每会话一块**（`CHUNK_BUDGET_H=7.5h`，来自 12h 硬上限留自检/打包余量；`kaggle/chunking.py` 的 `plan_chunks` 是参数化纯函数——其测试用 8h 预算仅为算例，notebook 实际传 7.5h）；块尾打包 checkpoint 后本会话结束——**重启会话再 Run All** 即从 latest 无损续训（`--resume auto`：checkpoint 含 optimizer/scheduler 状态，采样序按 (seed, epoch) 确定性重建）。**步速实测基准（2026-08-25 Kaggle T4×2 DP + TF32 + 20000 样本）**：~5.3 s/步 → ~17.6 min/epoch → 200 epoch ≈ **59h ≈ 8 个会话**（TF32 实测收益 <10%：KNN 距离计算与 softmax 为逐元素/归约操作，TF32 仅加速 matmul；该基准已写入 `bench_info.json`，回填 HANDOFF §6/§11）。

## 6. 收尾与回填

### 6.1 中途如何看模型效果（12h 会话内，可选）

训练分块进行中（前几个会话、几十 epoch）想看模型学到什么，两个入口：

1. **数字（推荐）**：进度基线跑一次纯评估（不训练）——`--epochs` 取当前进度（=latest 的 epoch+1），此时训练循环为空、只执行 `--report-f1`，写 `val_f1.json`（自然分布 F1/P/R + 混淆计数）：
   ```bash
   python train_kaggle.py --config config/pathline_transformer_cylinder.yaml \
       --epochs <当前进度> --resume auto --report-f1
   ```
2. **图像（notebook cell 9）**：单帧「模型概率 vs IVD vs 弱标签」三联图（`kaggle/preview_eval.py`，TTA 次数可选；帧默认 1300 ∈ test 片）。

**预期**（工程经验，以实测为准）：几十 epoch（lr=1e-4 段）高概率区域应大致落在涡街/拐角回流区，但边界毛糙、背景有噪声，F1 显著低于 200 epoch 终点——中途模型只用于**管线正确性检查**（标签流入、投影几何正确），不作为交付结果；正式评估（多帧对比/定量表/动画）在票 08。

### 6.2 回填

200 epoch 完成后（cell 6 输出 `progress()==200`，cell 8 打出 val F1 json）：

1. 下载 `final_ckpt.zip` → 本机 `outputs/archive/`（gitignore，勿提交）；
2. 从 `val_f1.json` 与 `bench_info.json` 回填：
   - 票文件 `07-kaggle-training.md`：四个验收项勾选 + 完成记录（val F1、步速、checkpoint 位置）；
   - HANDOFF §6 参数表 / §5 阶段 5 判据：epoch 样本数校准结论、val F1 记录；
   - HANDOFF §11：追加变更日志条目（训练完成事实、分块数与耗时）；
3. 之后按 frontier 启动票 08（推理评估，依赖 07）。

## 7. 多数据集联合训练（票 07 延伸）

本地已交付：7 数据集逐数据集预处理（`outputs/datasets/<名>/{geometry,dataset,previews}` + `multi_meta.json`，≈3.8GB）、`config/pathline_transformer_multi.yaml`（data.root 列表、frac 60/40、val_split none）。Kaggle 侧执行：

```powershell
# 0. 本地打包 Dataset A（多对：nc 与数据集目录一一对应，布局 data/<nc> + datasets/<名>/）
python kaggle\prepare_dataset_a.py --nc <7 个 nc 路径> --dataset-dir <7 个 outputs\datasets\*\dataset> `
    --out kaggle_dataset_a_multi --zip
```

Notebook 适配（在 `train_kaggle.ipynb` cell 3 挂载探测处多一层）：Dataset A 内布局为
`data/<nc>` + `datasets/<名>/dataset/meta.json`——把挂载根下的 `datasets/*` 逐目录
symlink 到 `repo/outputs/datasets/<名>`（**数据集名 slug 与本地目录名一致化**），
再执行 cell 4-9 其余流程；cell 9 预览加 `--dataset <索引>`（多 root 配置下选数据集）。

训练与留出评估（cell 6 训练命令）：

```bash
python train_kaggle.py --config config/pathline_transformer_multi.yaml \
    --resume auto --epochs <块目标>          # 与单数据集同分块/续训机制
# 训练完成后（收尾 cell 8）留出 40% 自然分布 F1/IoU：
python train_kaggle.py --config config/pathline_transformer_multi.yaml \
    --epochs <当前进度> --report-f1 --f1-split test   # → outputs/train_multi/*_test_f1.json
```

无 val 片（60/40）→ 训练期 val_loss 监控自动跳过（"数据集无 'val' 时间片"警告）；
留出评估指标 = **F1 + IoU**（`evaluate_f1`，自然分布序）；`--f1-split` 指定的片不存在时
fail loud（不静默跳过）。τ 与归一化逐数据集各自（各 meta 的 p85 分位 τ、ivd μ/σ、
speed_max——跨数据集输入尺度一致化）。

**注意**：多数据集联合训练与单数据集（p85 标签）重训是两条独立训练线——
run_name/ckpt_dir 各不相同（`pathline_transformer_multi` vs
`pathline_transformer_cylinder`），互不覆盖。

## 8. 故障排查

| 现象 | 处置 |
|---|---|
| Cell 3 断言失败：未找到 Dataset A | 检查 Add Input 是否挂载、zip 内是否含 `dataset/meta.json`（用 manifest.json 核对） |
| 报 `No module named 'dataset'`（路径含 `/kaggle/src/script.py`） | 运行的不是 Notebook 全流程：请用 Notebook 页面（File → Upload Notebook 上传 `train_kaggle.ipynb` → Run All）。`train_kaggle.py` 不是独立入口——它依赖同目录 `dataset.py`/`vendor/`，notebook 的 git clone + `sys.path` 自举才提供这些；Script 单文件环境无此结构 |
| `git clone` 失败 | Internet 未开；仓库 URL/可见性；clone 后手动 `os.chdir` 再 Run All |
| 训练 OOM（CUDA out of memory，`softmax`/KNN 分配失败） | 默认 `data_parallel: true`（T4×2：batch 100 → 每卡 50 ≈6.8GB 峰值 <16GB）。**AMP 默认关闭**——上游模型 `propagate_features` 原地 index-put 在 Half 下 dtype 冲突（实测 RuntimeError；迁移忠实性不改 vendor）。仍不足则降 `data.batch_size`（100→64，论文口径工程放宽）或 `samples_per_epoch`（下限 20000） |
| 带 `amp: true` 时报 `Index put requires the source and destination dtypes match` | 上游模型不兼容 AMP（Half/Float 原地赋值）；按上面默认关 AMP，显存交给 DP 拆分 |
| 步速校准远超 7.5h/块 | 属预期（单 epoch > 预算时每块至少 1 epoch）；可降 `samples_per_epoch`（下限 20000，HANDOFF §6）或启用 YAML `amp/data_parallel` |
| 会话断在训练中途 | `ckpt_latest.pth` 每 epoch 更新；重启会话 Run All 自动续（loss 从该 epoch 位置继续） |
| `kaggle datasets version` 报错 | API key 缺 scope（须 Create+Read+Write）/ slug 拼写 / 数据集不存在（先建占位） |
| 磁盘不足（/kaggle/working 12GB） | `outputs/bench` 与 `outputs/train` 各占几 MB~数百 MB；memmap 走 symlink（不复制）；必要时删 `outputs/bench` |
| 想要确定步速上限做预分配 | 参考 HANDOFF §5：T4 单卡 0.8~1.3s/步 → 40000 样本/100 batch = 400 步/epoch → 5.3~8.7 min/epoch |

## 9. 相关约束（HANDOFF 摘录，实现已对齐）

- 训练口径：AdamW(wd 1e-6)、lr 1e-4、TwoStep（warmup 60 → 5e-6）、batch 100、200 epoch、梯度裁剪 1.0、BCE（HANDOFF §2 论文附录 C / §6）；
- 每 epoch checkpoint（含 optimizer 状态）是跨会话无损续训的硬性要求（HANDOFF §7 风险预案）；
- 数据一律 h5py/memmap 直读（中文路径：本地打包即走 h5py；Kaggle 路径 ASCII 无此问题）；
- val 损失口径 = 训练同款 50% 平衡采样（监控）；val F1 = 自然分布（池比例），属票 07 验收记录，正式弱定量表在票 08。
