# Kaggle 工件处置说明

本目录的当前执行入口已迁移到 VS Code Remote-SSH 服务器；日常训练、评估和数据准备请阅读 [`../server/README.md`](../server/README.md)。项目不再要求 Kaggle Dataset、Notebook Run All、12 小时会话分块或 Kaggle API 密钥。

## 目录状态

以下文件是早期 Kaggle 运行留下的兼容性/历史实现，已不属于当前执行路径：

- `train_kaggle.ipynb` 已移除；Notebook 不再维护。
- `prepare_dataset_a.py` 仅保留旧 Dataset 打包脚本，标记为 deprecated；服务器直接同步 `outputs/datasets/`，不打包上传。
- `chunking.py` 仅保留旧会话分块纯函数，标记为 deprecated；服务器不受 12 小时会话上限约束，但 `--resume auto` 仍保留。
- `mount_probe.py` 仅保留旧 Kaggle 输入布局探测，标记为 deprecated；服务器使用固定 `/data/xuziyi/cylinder_vortex_pipeline` 工作区。
- `self_check.py`、`preview_eval.py` 的服务器入口位于 `../server/`；本目录版本保留向后兼容，新的命令请使用 `python server/self_check.py` 和 `python server/preview_eval.py`。

旧脚本不再参与 README 所描述的训练流程，也不应作为新服务器部署的依赖。它们保留的原因是让历史测试和旧快照仍可读取；若未来不再需要兼容旧快照，可单独删除整个 `kaggle/` 目录并同步移除对应历史测试。
