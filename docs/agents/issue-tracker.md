# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`, never a single combined tickets file
- Triage state is recorded as a `Status:` line near the top of each issue file (see `triage-labels.md` for the role strings)
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/` (creating the directory if needed).

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly.

## 代码托管与迁移备注

代码托管于 GitHub：https://github.com/ziyixu317-wq/2d-vortex-extraction-260825；服务器工作区从该仓库 clone/fetch，预处理 memmap 通过本地 E 盘同步到服务器，不进 GitHub。

**Issue 追踪保持本地 markdown**（本文件），与代码托管解耦。若日后改用 GitHub Issues 作为追踪器：重跑 `/setup-matt-pocock-skills` 切换（需要 `gh` CLI 与认证），切换前本配置保持有效。
