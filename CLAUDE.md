# CLAUDE.md

本仓库为**迹线 Transformer 涡提取项目**（cylinder_vortex_pipeline）。

**唯一权威上下文是 `HANDOFF.md`**：新会话先完整阅读它（已拍板决策、已核实事实、阶段计划、参数、风险、工作流与更新协议全部在其中），按 §9 工作流推进，按 §11 协议维护。

- 工作语言：中文；领域词汇以 HANDOFF.md 为准。
- 参考仓库 `PyflowVis-main`（本机只读参考，不在本仓库内；不直接依赖）。
- 代码托管：https://github.com/ziyixu317-wq/2d-vortex-extraction-260825。
- 当前执行方式：本地 `E:\codex\AI CFD\cylinder_vortex_pipeline` 维护代码，通过 VS Code Remote-SSH 连接 `SHU-server`，在 `/data/xuziyi/cylinder_vortex_pipeline` 执行训练、评估和预览。
- 服务器入口：先阅读 `server/README.md`，确认 `nvidia-smi`、`CUDA_VISIBLE_DEVICES`、`data_parallel`、`amp` 和 `num_workers` 与当前共享服务器资源一致。
- 预处理数据：训练/评估读取 `outputs/datasets/<dataset>/dataset/` memmap；原始 nc 仅在服务器上重算预处理时使用。断点续训保留 `--resume auto` 和逐 epoch checkpoint。

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles: `needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
