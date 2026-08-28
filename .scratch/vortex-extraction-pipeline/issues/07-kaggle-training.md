# 07: Kaggle 上传与 200 epoch 训练

**What to build:** 在 Kaggle 完成全量训练：Dataset A = nc 数据文件、Dataset B = pipeline 代码目录；Notebook 安装依赖（h5py/yaml/matplotlib/tqdm）后可 import vendor 并加载数据；先 1 epoch 实测步速校准每 epoch 样本数；按 Kaggle 12h 会话上限分块（≤8h）训练 200 epoch，每 epoch checkpoint、跨会话断点续训（块尾打包为 Kaggle Dataset 新版本）；最终 checkpoint 归档、val F1 记录。超预算时启用 DataParallel/AMP/降样本数。

**Blocked by:** 06

**Status:** ready-for-human

**交付物 vs 执行边界**（agent 交付完成 → 用户 Kaggle 执行 → 回填后勾选并改 done）：
- 交付：`kaggle/` 全部（Notebook/打包/分块/自检/手册）、`train_kaggle.py --report-f1`、`dataset.set_epoch_natural`、`tests/test_kaggle.py`（20 项，全量 151 passed）
- 用户执行：Kaggle 端 `Run All`（首会话自检 → 步速实测 → 分块训练）；`kaggle/README.md` §0-6 逐步指引与回填清单

- [ ] Notebook 环境 import vendor + 数据加载通过（**交付**：self_check.py 已本地真实验证——生产模型前向 (1,256) 域 (0,1)、4 样本 (16,256,7) 有限无 NaN；**执行**：用户首会话 cell 4，失败 raise）
- [ ] 1 epoch 实测步速，epoch 样本数校准后回写参数表（**交付**：notebook 块 5 自动基准+总时长预算检查+超预算自动生成 train_opt.yaml（AMP/DataParallel/降样本至下限 20000）+回填提示；**执行**：用户 Kaggle 实测，回写 HANDOFF §6）
- [ ] 200 epoch 训练完成（分块 + 断点续训达成，12h 会话内无丢失）（**交付**：chunking.plan_chunks ≤7.5h/块 + `--resume auto` + 块尾打包/发布 + 基准复用；**执行**：用户 3–4 个会话）
- [ ] 最终 checkpoint 归档（含 optimizer 状态），val F1 记录（**交付**：收尾 cell 打包 final_ckpt.zip + `--report-f1` 写 val_f1.json（自然分布，含 tp/fp/fn/tn）；**执行**：用户下载归档与回填值）

## 完成记录（2026-08-25，agent 侧交付完成）

**做了什么**：为票 07 的 Kaggle 全量训练交付全部可自动化的部分，并遵循 /tdd（20 项测试红→绿，切片 A–F）与 /code-review 双轴审查处置。

1. **`train_kaggle.py`**：`evaluate_f1()`（自然分布 val F1：sigmoid > 0.5 判正 → tp/fp/fn/tn/precision/recall/F1/n，手算混淆矩阵字面量测试）+ `--report-f1`（显式开关；训练完成后 val 片重设自然分布序 → 写 `{run}_val_f1.json`（epoch/split/threshold/全混淆计数）；无 val 片安全跳过）。
2. **`dataset.py`**：`set_epoch_natural(epoch=0)`（自然分布采样序：正负比例 = 池比例；训练监控 50% 平衡口径不变）——`WeakLabelPathlineDataset` 池名统一为公开 `pool_positive/pool_negative`（消除双套命名，无外部行为变化）。
3. **`kaggle/`** 新目录：
   - `chunking.py`：`plan_chunks(total, sps, budget)` 纯函数分块规划（≤预算/块、区块最少、单 epoch 超预算不退化、非法参数报错）；
   - `prepare_dataset_a.py`：Dataset A 打包（nc + dataset/ memmap 产物 + aux 目检 PNG + manifest.json sha256 审计；zip 模式；`.gitignore` 防呆加入）；
   - `self_check.py`：验收 1 自检（vendor import + 模型前向 (B,256) 域 (0,1) + on-the-fly 样本有限/标签 {0,1}；CLI 可本地 CPU 先验）；
   - `train_kaggle.ipynb`：9 cell Notebook（安装/克隆/挂载+checkpoint 还原/自检/步速校准+预算检查/分块训练/块尾打包发布/收尾归档）；TOTAL_EPOCHS 从 config 读、基准存在则复用不重跑、超预算自动生成 train_opt.yaml；
   - `README.md`：用户操作手册（0–6 步 + 故障排查 + 回填清单）。
4. **验证证据**：全量 151 passed（基线 131 + 新增 20）；真实数据实证 = `self_check.py` 生产模型（dmodel 144/3 层/k16）+ `outputs/dataset` 真产物 CPU 通过（前向 (1,256) 域 (0,1)、4 样本 (16,256,7) 有限无 NaN、正标签 192 条）；Dataset A 规模预估 ≈2.03GB（nc 773MB + memmap 1.26GB + 目检图 1MB）；notebook JSON 合法性校验通过。

**/code-review 双轴处置**：
- Standards 硬性 0、判断项 8 → 修 6：① 池名双套统一（`_pool_pos/_pool_neg` 删除，公开口径）；② main report-f1 块重复守卫删除（归 set_epoch_natural）；③ 测试样板 make_val_cfg 加 `with_val` 参数复用；④ `_mixed_loader` 与 make_loader 同构删除；⑤ self_check 不可用缺省模型路径改 fail loud；⑥ README 预算口径说明（7.5h notebook 默认 vs 8h 测试算例）。保留 2 并记录：⑦ TestSetEpochNatural 期望 frac 与实现同源（池比例为数据自描述公开属性，守护「natural 口径正确接池」语义，独立性局限披露）；⑧ evaluate_f1 8 字段 dict（Data Clumps，按票 04/05 先例保留）。
- Spec 3 → 修 2：① TOTAL_EPOCHS 硬编码改从 config 读（注释与实现一致）；② 验收 2 校准自动化补全（基准复用 + 总时长预算检查 + 超预算自动启用 AMP/DataParallel/降样本至下限 + HANDOFF §6 回填提示）。记录 2：③ 自然分布 val F1 提前实现为票 07 验收 4 明示需求（与票 06 注释「归票 08」的计划冲突以票为准——票 07 验收 4 需要 val F1 记录；正式网格投影级弱定量表仍归票 08）；④ `--aux-dirs` 打包 weak_labels 目检 PNG（HANDOFF 票 04 记录「目检图走 Kaggle Dataset」的既有表述；1MB 级）。

**用户执行（收尾后）**：① 本机普通终端 `git push -c http.sslBackend=openssl origin main`（沙箱 git 无法 GCM）；② `python kaggle\prepare_dataset_a.py --nc ..\CFD数据集\pipedcylinder2d.nc --dataset-dir outputs\dataset --aux-dirs outputs\weak_labels --out kaggle_dataset_a --zip`；③ Kaggle 按 `kaggle/README.md` §2–6 建 Dataset/Notebook → Run All（约 3–4 个 12h 会话）→ 下载 final_ckpt.zip → 回填本验收项勾选、val F1、步速与 HANDOFF §6/§11。

## 完成记录-延伸段（2026-08-25，票 07 延伸：多数据集联合训练 + τ 对齐论文 Fig.6）

**范围**（用户两个问题）：① 需求 A——弱标签 IVD 阈值对齐论文 Fig.6 列 1（IVD 在该数据集的白色等值线呈现）；② 需求 B——按论文 §4.2 评估范式（多数据集训练 → 真实数据测试，未接触测试数据）做多数据集联合训练 + 跨数据集留出评估。**口径说明（诚实性）**：本实现为**时间留出近似**（各数据集自身前 60% 参与训练、后 40% 时间上未见、覆盖 7 类不同流场）；**非**论文严格零样本（严格零样本须按数据集留出，用户拍板按时间 60/40 未采用）。**拍板（用户确认）**：τ=p85 逐时间片分位、5×5 面积过滤保留、按时间各数据集帧前 60% 训/后 40% 测（无 val）、τ/归一化逐数据集各自、训练池=全部 7 数据集（6 新 + pipedcylinder2d）。

**需求 A（τ 对齐）——已交付**：
- 多阈值敏感性报告：`weak_labels.compute_tau_candidates`/`multi_tau_report` + CLI `--multi-tau-dir`（候选 95/90/85/80 分位 × 固定 2.5/2.0/1.5/1.0 × min_area 25/9/1；统计：正格占比/连通块数/正样本占比；目检图含论文风格 IVD 白色等值线——**连续 IVD 参考 vs 二值训练目标两语义分开标注**）；产物 `outputs/weak_labels/multi_tau/`（stats json + 图，gitignore）。
- **实测结论（pipedcylinder2d，流体区）**：p95=4.81% 正格（稀疏，用户观察成立）→ p90=9.90%（τ≈1.08-1.18、涡块 6.8→7.0/帧——覆盖率翻倍但结构块数不变：主要"填补"涡街/拐角回流，最接近论文呈现）→ p85=14.80%（τ≈0.61-0.65、块 9.8）→ p80=19.73%（块 12.2、碎片 38.8/帧不过滤）；固定 τ 跨数据集不可移植（IVD 量纲 300× 差）；**5×5 过滤在 p85 只删 0.15% 正格但消 ~15 碎片块/帧 → 保留**。用户拍板 p85。
- 实现：τ 默认 95→85（`weak_labels.DEFAULT_PERCENTILE`，compute_tau/prepare_dataset/双 CLI 同步；HANDOFF §6 回写）；单数据集产物 `outputs/dataset` 本地重生成 p85 标签（taus=0.6539/0.6512/0.6113）；**影响标注**：已训练 25-epoch 模型为旧 τ（p95）标签 → 新 τ 标签需重训（用户 Kaggle 执行）；preview 标题语义标注生效。

**需求 B（多数据集）——已交付**：
- `dataset.fraction_slices`（按帧比例 60/40/可选 val；floor 取整、全覆盖无泄漏，守护测试 6 项）；`prepare_dataset` 增 `split_mode=abs|frac` + CLI；
- dataset.py 重构：`_DatasetStore`（单数据集存储/提取/池）→ `WeakLabelPathlineDataset` 薄包装（公开面与票 05 逐字节兼容：**单数据集续训采样序不变**，既有测试全绿守护）+ **`MultiDatasetPathlineDataset`**（7 roots 联合池 (ds_idx,y0,x0,frame)；50% 过采样/set_epoch_natural/sample_at(si,…)/stores 公开；组合级 rng 基含 ds_id 派生——同语义、多数据集与单数据集不同构）；归一化逐数据集（各 store meta）；
- `train_kaggle.py`：`_make_dataset` 支持 `data.root` 列表；`evaluate_f1` 增 **IoU**；`--f1-split`（默认 val_split；指定片不存在 fail loud；多数据集 `--report-f1 --f1-split test` 对留出 40% 出自然分布 F1/IoU → `{run}_test_f1.json`）；
- `kaggle/preview_eval.py`：`--dataset` 索引跨数据集推理（单 ckpt → 各数据集 test 片，时间留出泛化观察简版——非严格零样本，见本段口径说明）；`kaggle/prepare_dataset_a.py`：`--nc/--dataset-dir` 多对打包（data/<nc>+datasets/<名>/ 布局 + manifest，单数据集布局不变）；
- `prepare_multi.py`（7 数据集逐数据集预处理驱动：geometry→IVD/label/τ→memmap+meta+目检图，复用票 02/05）；`config/pathline_transformer_multi.yaml`（7 roots、frac、val_split none、run_name 独立）；`kaggle/README.md` §7 多数据集执行说明。

**验证证据**：全量 **186 passed**（159 基线 + 27 新增，含：多阈值报告 3、fraction_slices 6、多数据集池 6、train 集成 4、IoU 1、preview 2、打包 2、prepare_multi 3）；真实数据 = 7 数据集预处理产物（`outputs/datasets/`，§2 盘点表——boussinesq 1 障碍物 242 格、cylinder2d 1 圆柱 60 格、doublegyre/duffing/fourcenters/jung 无障碍（jung 12 格、doublegyre 4 格单格噪声点）、pipedcylinder2d 2 圆柱；IVD 量纲 300× 差 → 逐数据集分位 τ 必要；jung t 从 1.107 起 → 帧比例划分必要）+ 多数据集池实测（train 正 78,351/负 82,757、构建 2.0s；跨数据集样本逐数据集归一化/有限性/确定性序 ✓）+ test 片自然分布 F1/IoU 跑通（随机小模型 n=16384, F1=0.281, IoU=0.163）+ 跨数据集预览（25-epoch 基线 → 未见过的 duffing 帧 350 → `outputs/preview/multi_duffing_t350.png`）。

**/code-review 双轴处置**：见会话记录（Standards/Spec 各发现均已处置；本段不重复）。

**用户执行（延伸部分）**：① git push（本延伸 commit）；② 本地重新打包 Dataset A（单数据集 p85：`--dataset-dir outputs\dataset`；多数据集：§7 命令）；③ Kaggle 两条训练线任选或都跑：单数据集重训（p85 标签，`--config config/pathline_transformer_cylinder.yaml`）与多数据集联合训练（`--config config/pathline_transformer_multi.yaml --report-f1 --f1-split test`，notebook 按 README §7 适配挂载）；④ 回填 HANDOFF §6 实测值与 §11。

**执行边界**：新 τ 标签重训与多数据集 200 epoch 训练/留出评估级记录 = 用户 Kaggle 执行（本会话交付代码/产物/文档与本地 CPU 验证）；正式弱定量表（网格投影级、多帧/动画）属票 08。

