# VS Code Remote-SSH 服务器执行手册

本项目的训练、评估与预览均在 VS Code 的 Remote-SSH 终端中执行；代码入口、数据同步和资源参数统一按本手册操作。

## 服务器与目录

默认服务器为 `SHU-server`（`58.199.164.190:50011`，用户 `xuziyi`），固定工作区为 `/data/xuziyi`。建议仓库放在 `/data/xuziyi/cylinder_vortex_pipeline`，原始 NetCDF 可选放在 `/data/xuziyi/cfd_raw`，训练和评估实际读取预处理 memmap：

```text
/data/xuziyi/cylinder_vortex_pipeline/outputs/datasets/<dataset>/dataset/
/data/xuziyi/cylinder_vortex_pipeline/outputs/datasets_new/cavity2d_re1000/dataset/
```

已核实的环境为 Python 3.12.14、PyTorch 2.7.1+cu118、CUDA 11.8、4 张 NVIDIA RTX 3090（每张 24 GiB）、40 个 CPU 核和约 251 GiB 内存。服务器为多人共享环境，每次运行前仍应执行 `nvidia-smi`；默认只使用明确选择且已空闲的 GPU（示例为 GPU 0），不要假定其余 GPU 可用。服务器根分区已满，缓存和临时文件必须放在 `/data/xuziyi/`。

## 首次准备

在用户将本地 E 盘提交推送后，服务器上执行：

```bash
cd /data/xuziyi
git clone https://github.com/ziyixu317-wq/2d-vortex-extraction-260825.git cylinder_vortex_pipeline
cd /data/xuziyi/cylinder_vortex_pipeline
git fetch origin
git checkout main
```

每次登录后建议设置工作目录和缓存：

```bash
cd /data/xuziyi/cylinder_vortex_pipeline
export TMPDIR=/data/xuziyi/tmp
export TMP=/data/xuziyi/tmp
export TEMP=/data/xuziyi/tmp
export PIP_CACHE_DIR=/data/xuziyi/pip_cache
export MPLCONFIGDIR=/data/xuziyi/mplconfig
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"
nvidia-smi
```

目标解释器是 `/data/xuziyi/envs/xuziyi/bin/python`。依赖安装只使用该环境的 pip，并将缓存放在数据分区：

```bash
/data/xuziyi/envs/xuziyi/bin/pip install --cache-dir /data/xuziyi/pip_cache h5py PyYAML matplotlib tqdm
```

## 从本地 E 盘同步数据

在本地 PowerShell 7 终端执行。若仓库已通过 git 同步，只需上传未纳入 git 的 `outputs/datasets`（以及严格零样本 cavity 数据）：

```powershell
ssh SHU-server "mkdir -p /data/xuziyi/cylinder_vortex_pipeline/outputs /data/xuziyi/cfd_raw"
scp -r "E:\codex\AI CFD\cylinder_vortex_pipeline\outputs\datasets" SHU-server:/data/xuziyi/cylinder_vortex_pipeline/outputs/
scp -r "E:\codex\AI CFD\cylinder_vortex_pipeline\outputs\datasets_new" SHU-server:/data/xuziyi/cylinder_vortex_pipeline/outputs/
```

原始 nc 仅用于需要重算预处理时上传：

```powershell
scp -r "E:\codex\AI CFD\CFD数据集\*" SHU-server:/data/xuziyi/cfd_raw/
```

服务器上检查 memmap 和元数据：

```bash
find outputs/datasets -name multi_meta.json -o -name u.npy | sort
/data/xuziyi/envs/xuziyi/bin/python - <<'PY'
import json
from pathlib import Path
for p in sorted(Path("outputs/datasets").glob("*/multi_meta.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    print(p, d.get("dataset_names", d.get("name", "(未命名)")))
PY
```

如需从 nc 重新生成六数据集 memmap，可运行（名称按实际文件调整）：

```bash
/data/xuziyi/envs/xuziyi/bin/python prepare_multi.py \
  --nc-dir /data/xuziyi/cfd_raw \
  --out-root outputs/datasets \
  --names boussinesq.nc,cylinder2d.nc,doublegyre2d.nc,fourcenters2d.nc,jungtelziemniak2d.nc,pipedcylinder2d.nc \
  --percentile 85 --train-frac 0.6 --val-frac 0
```

## 自检、训练与断点续训

先做数据和模型自检，再开始长任务：

```bash
/data/xuziyi/envs/xuziyi/bin/python server/self_check.py \
  --config config/pathline_transformer_multi.yaml \
  --data-root outputs/datasets/pipedcylinder2d/dataset --device cuda
/data/xuziyi/envs/xuziyi/bin/python -m pytest tests -q
```

默认配置针对单卡运行：`CUDA_VISIBLE_DEVICES=0`、`data_parallel: false`、`amp: false`、`num_workers: 8`。若 GPU 0 被占用，应先根据 `nvidia-smi` 选择空闲卡并同步修改可见设备；只有明确预留多张卡时才启用 `data_parallel: true`。AMP 保持关闭，除非在目标服务器上完成数值回归验证。CPU 负载竞争时可将 `num_workers` 降到 4 或更低。

训练命令（入口文件名沿用既有配置与 checkpoint 约定）：

```bash
export CUDA_VISIBLE_DEVICES=0
/data/xuziyi/envs/xuziyi/bin/python train_kaggle.py \
  --config config/pathline_transformer_multi.yaml \
  --resume auto --report-f1 --f1-split test
```

脚本每个 epoch 保存 checkpoint，`--resume auto` 会在 SSH 会话中断后从 `ckpt_dir` 自动续训；不需要手工分块，也没有会话时限。最终权重和日志写入 `outputs/_ckpt130/train_multi/`（或配置指定的 checkpoint 目录）。

## 评估与预览

多数据集评估示例：

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

严格零样本 cavity 使用 `config/pathline_transformer_cavity_eval.yaml` 和 `server/preview_eval.py`：

```bash
/data/xuziyi/envs/xuziyi/bin/python evaluate.py \
  --config config/pathline_transformer_cavity_eval.yaml \
  --ckpt outputs/_ckpt130/train_multi/pathline_transformer_multi_ckpt_latest.pth \
  --split train --display-frames 3 --out-dir outputs/evaluation_cavity_server
/data/xuziyi/envs/xuziyi/bin/python server/preview_eval.py \
  --config config/pathline_transformer_cavity_eval.yaml \
  --ckpt outputs/_ckpt130/train_multi/pathline_transformer_multi_ckpt_latest.pth \
  --dataset 0 --frame 3 --tta 3 --device cuda \
  --out outputs/evaluation_cavity_server/prob_vs_ivd_t3.png
```

服务器未安装 ffmpeg 时，动画评估会自动保留 PNG 序列或使用已有 GIF 回退；这不影响数值评估。交付产物按 `HANDOFF.md` 约定保存，临时诊断、冒烟模型和测试缓存不作为交付内容。

## 相关文档

- `HANDOFF.md`：唯一权威上下文、参数、风险和变更日志。
- `README.md`：项目概览和最短复现路径。
- `kaggle/`：兼容接口；当前执行入口以本手册为准。
