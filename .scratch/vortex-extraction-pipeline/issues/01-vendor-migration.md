# 01: vendor 迁移：最小纯 torch 子集

**What to build:** 工程自包含的第一步：把参考仓库（PyflowVis-main，只读参考）的模型/损失/工具代码中最小纯 torch 子集迁移进本项目 vendor 目录，重写工具包入口剔除依赖 multimethod 的配置模块，保留 Apache 2.0 署名（LICENSE/NOTICE），使模型可从配置构建并在本地 CPU 完成一次前向。迁移后全项目不再引用参考仓库。

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] 本地 `python -c "from vendor.DeepUtils.models import build_model_from_cfg"` 导入通过
- [ ] 全项目不再 import 参考仓库（PyflowVis-main）
- [ ] LICENSE/NOTICE 随迁移代码保留
- [ ] 随机输入跑通一次前向：输出形状 (B, 256)、数值域 (0,1)
- [ ] 不迁移数据集/流场工具/训练测试脚本/Misc 模块（拖入 numba/pybind/GUI/wandb 者一律剔除）
