# CLAUDE.md

本仓库为**迹线 Transformer 涡提取项目**（cylinder_vortex_pipeline）。

**唯一权威上下文是 `HANDOFF.md`**：新会话先完整阅读它（已拍板决策、已核实事实、阶段计划、参数、风险、工作流与更新协议全部在其中），按 §9 工作流推进，按 §11 协议维护。

- 工作语言：中文；领域词汇以 HANDOFF.md 为准。
- 参考仓库 `PyflowVis-main`（本机只读参考，不在本仓库内；不直接依赖）。
- 代码托管：https://github.com/etl5736/2d-vortex-extraction-260825 （Kaggle 训练从该仓库克隆）。

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles: `needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
