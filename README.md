# 迹线 Transformer 涡提取项目（cylinder_vortex_pipeline）

本项目用 VortexTransformer 论文（CGF 2025，DOI 10.1111/cgf.70042）的迹线 Transformer（`PathlineTransformerV0`）从 2D 非定常流场提取涡概率，再投影回网格。唯一权威上下文是 [`HANDOFF.md`](HANDOFF.md)；本 README 只提供最短复现入口。

## 当前执行方式

代码在本地 E 盘维护，通过 VS Code Remote-SSH 在 `SHU-server` 上训练、评估和生成预览；服务器执行细节见 [`server/README.md`](server/README.md)。

本地工作区：`E:\codex\AI CFD\cylinder_vortex_pipeline`
服务器工作区：`/data/xuziyi/cylinder_vortex_pipeline`
服务器预处理数据：`outputs/datasets/<dataset>/dataset/`（训练/评估实际读取 memmap）；原始 nc 只在需要重算时放入服务器数据分区。

## 目录结构

```text
cylinder_vortex_pipeline/
├── geometry.py / extractor.py / weak_labels.py / dataset.py  # 预处理、迹线与弱标签
├── prepare_multi.py                                          # 多数据集 memmap 预处理
├── train_kaggle.py                                            # 服务器训练入口（保留既有文件名以兼容配置/断点）
├── evaluate.py                                                # 推理、投影、τ 敏感性
├── config/                                                    # multi、cylinder 与 cavity 评估配置
├── server/                                                    # Remote-SSH 自检、预览与执行手册
├── kaggle/                                                    # 兼容接口（仅供既有测试与快照）
├── outputs/datasets/                                          # 训练/评估数据（不入 git）
├── outputs/_ckpt130/train_multi/                              # 130 epoch 最终权重与指标
├── tests/                                                     # pytest 验收测试
├── HANDOFF.md                                                # 唯一权威上下文
└── .scratch/vortex-extraction-pipeline/                     # spec.md 与 issues/
```

## 依赖与服务器环境

基线依赖为 `torch`、`numpy`、`h5py`、`PyYAML`（import 名 `yaml`）、`matplotlib`、`tqdm`；弱监督 Haller-IVD 扩展允许新增 `scipy` 与 `scikit-image`。本地可用 CPU 环境运行测试；服务器已核实 Python 3.12.14、PyTorch 2.7.1+cu118、CUDA 11.8、4×RTX 3090 24 GiB。服务器为共享环境，运行前用 `nvidia-smi` 选择空闲卡；默认 `CUDA_VISIBLE_DEVICES=0`、`data_parallel: false`、`amp: false`、`num_workers: 8`，资源竞争时降低 worker 数。根分区已满，缓存统一放 `/data/xuziyi/`。弱监督扩展的交接与实施入口见 [`HANDOFF.md` §12](HANDOFF.md#12-b1w1w2w3-弱监督持续交接阶段-0-已完成)。

## 最短复现路径（Remote-SSH）

### 1. 克隆代码并同步数据

用户将本地改动推送到 GitHub 后，在服务器执行：

```bash
cd /data/xuziyi
git clone https://github.com/ziyixu317-wq/2d-vortex-extraction-260825.git cylinder_vortex_pipeline
cd /data/xuziyi/cylinder_vortex_pipeline
git fetch origin && git checkout main
```

从本地 PowerShell 7 上传未纳入 git 的 memmap：

```powershell
ssh SHU-server "mkdir -p /data/xuziyi/cylinder_vortex_pipeline/outputs"
scp -r "E:\codex\AI CFD\cylinder_vortex_pipeline\outputs\datasets" SHU-server:/data/xuziyi/cylinder_vortex_pipeline/outputs/
scp -r "E:\codex\AI CFD\cylinder_vortex_pipeline\outputs\datasets_new" SHU-server:/data/xuziyi/cylinder_vortex_pipeline/outputs/
```

原始数据按需上传到 `/data/xuziyi/cfd_raw/`；完整同步、缓存设置和元数据检查见 [`server/README.md`](server/README.md)。

### 2. 自检与测试

```bash
cd /data/xuziyi/cylinder_vortex_pipeline
export TMPDIR=/data/xuziyi/tmp TMP=/data/xuziyi/tmp TEMP=/data/xuziyi/tmp
export PIP_CACHE_DIR=/data/xuziyi/pip_cache MPLCONFIGDIR=/data/xuziyi/mplconfig
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"
/data/xuziyi/envs/xuziyi/bin/python server/self_check.py \
  --config config/pathline_transformer_multi.yaml \
  --data-root outputs/datasets/pipedcylinder2d/dataset --device cuda
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests -q
```

### 3. 训练与断点续训

```bash
export CUDA_VISIBLE_DEVICES=0
/data/xuziyi/envs/xuziyi/bin/python train_kaggle.py \
  --config config/pathline_transformer_multi.yaml \
  --resume auto --report-f1 --f1-split test
```

脚本每个 epoch 保存 checkpoint；SSH 会话中断后重复同一命令即可由 `--resume auto` 从 `ckpt_dir` 续训，无需分块。服务器显存/并行设置按实测调整，AMP 默认关闭。

### 4. 评估与 τ 敏感性

```bash
/data/xuziyi/envs/xuziyi/bin/python evaluate.py \
  --config config/pathline_transformer_multi.yaml \
  --ckpt outputs/_ckpt130/train_multi/pathline_transformer_multi_ckpt_latest.pth \
  --split test --out-dir outputs/evaluation_server

/data/xuziyi/envs/xuziyi/bin/python evaluate.py \
  --config config/pathline_transformer_multi.yaml \
  --ckpt outputs/_ckpt130/train_multi/pathline_transformer_multi_ckpt_latest.pth \
  --tau-sensitivity --out-dir outputs/tau_sensitivity_server
```

严格零样本 cavity 使用 `config/pathline_transformer_cavity_eval.yaml` 和 `server/preview_eval.py`。服务器缺少 ffmpeg 时，动画保留 PNG 序列或 GIF 回退，不影响数值评估。

## 当前交付产物

- `outputs/_ckpt130/train_multi/`：130 epoch 最终权重、`*_test_f1.json` 与 `bench_info.json`。
- `outputs/evaluation_smoke/`、`outputs/evaluation_anim/`：对比图、动画和弱定量表。
- `outputs/tau_sensitivity/`：τ 候选敏感性表和报告。
- `outputs/weak_labels/multi_tau/`：多阈值弱标签统计与目检图。
- `outputs/datasets/`、`outputs/datasets_new/`：训练及严格零样本评估所需 memmap（由 `.gitignore` 忽略）。

现有结算指标为 P=0.4967、R=0.9549、F1=0.6535、IoU=0.4853（test 迹线，阈值 0.5）；解释和限制以 `HANDOFF.md` §11 为准。

## 测试

```powershell
python -m pytest tests -q
```

迁移后的验收基线为 **240 passed**。测试会写入临时评估目录，均已被 `.gitignore` 忽略。

## 相关文档

- [`HANDOFF.md`](HANDOFF.md)：决策、已核实事实、参数、风险、工作流和变更日志。
- [`server/README.md`](server/README.md)：Remote-SSH 服务器准备、同步、训练和评估手册。
- `kaggle/`：兼容接口；日常执行入口见 [`server/README.md`](server/README.md)。
- [`.scratch/vortex-extraction-pipeline/spec.md`](.scratch/vortex-extraction-pipeline/spec.md) 与 [`issues/`](.scratch/vortex-extraction-pipeline/issues/)：规格和票据上下文。
- [`CLAUDE.md`](CLAUDE.md)：agent 入口和文档导航。
