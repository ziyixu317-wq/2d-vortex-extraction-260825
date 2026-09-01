# 兼容模块说明

训练、评估、预览和数据同步的现行命令集中在 [`../server/README.md`](../server/README.md)。
服务器入口是 `python server/self_check.py`、`python train_kaggle.py`、
`python evaluate.py` 和 `python server/preview_eval.py`。

本目录保存与既有测试和快照格式相容的纯 Python 接口：

- `chunking.py`：确定性分块规划函数；
- `prepare_dataset_a.py`：memmap 目录的 manifest/zip 组装工具；
- `mount_probe.py`：输入目录布局探测函数；
- `self_check.py`、`preview_eval.py`：与服务器入口共享行为的兼容实现。

这些接口的回归覆盖位于 `tests/test_kaggle.py`；新服务器部署按 `../server/README.md` 的
路径、缓存和 GPU 参数执行。
