# 07: Kaggle 上传与 200 epoch 训练

**What to build:** 在 Kaggle 完成全量训练：Dataset A = nc 数据文件、Dataset B = pipeline 代码目录；Notebook 安装依赖（h5py/yaml/matplotlib/tqdm）后可 import vendor 并加载数据；先 1 epoch 实测步速校准每 epoch 样本数；按 Kaggle 12h 会话上限分块（≤8h）训练 200 epoch，每 epoch checkpoint、跨会话断点续训（块尾打包为 Kaggle Dataset 新版本）；最终 checkpoint 归档、val F1 记录。超预算时启用 DataParallel/AMP/降样本数。

**Blocked by:** 06

**Status:** ready-for-agent

- [ ] Notebook 环境 import vendor + 数据加载通过
- [ ] 1 epoch 实测步速，epoch 样本数校准后回写参数表
- [ ] 200 epoch 训练完成（分块 + 断点续训达成，12h 会话内无丢失）
- [ ] 最终 checkpoint 归档（含 optimizer 状态），val F1 记录
