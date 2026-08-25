# 01: vendor 迁移：最小纯 torch 子集

**What to build:** 工程自包含的第一步：把参考仓库（PyflowVis-main，只读参考）的模型/损失/工具代码中最小纯 torch 子集迁移进本项目 vendor 目录，重写工具包入口剔除依赖 multimethod 的配置模块，保留 Apache 2.0 署名（LICENSE/NOTICE），使模型可从配置构建并在本地 CPU 完成一次前向。迁移后全项目不再引用参考仓库。

**Blocked by:** None (can start immediately)

**Status:** done

- [x] 本地 `python -c "from vendor.DeepUtils.models import build_model_from_cfg"` 导入通过
- [x] 全项目不再 import 参考仓库（PyflowVis-main）
- [x] LICENSE/NOTICE 随迁移代码保留
- [x] 随机输入跑通一次前向：输出形状 (B, 256)、数值域 (0,1)
- [x] 不迁移数据集/流场工具/训练测试脚本/Misc 模块（拖入 numba/pybind/GUI/wandb 者一律剔除）

## 完成记录

- **2026-08-25 完成**（commit `1af7b7c`；收尾文档见后续 docs commit）。
- **做了什么**：
  - 从 `PyflowVis-main/DeepUtils` 复制 `models/`（classification/reconstruction/segmentation/layers/build.py/__init__.py 全树）、`loss/`（build/cross_entropy/distill_loss）与 `utils/{registry,ckpt_util,random}` 进 `vendor/DeepUtils/`，共 38 个文件，SHA256 与源逐字一致（复制忠实）。
  - 重写 `vendor/DeepUtils/utils/__init__.py`：剔除 `from .config import EasyConfig`（config.py 依赖 multimethod），其余导出符号原样保留；`config.py` 未迁移。
  - 复制参考仓库根 `LICENSE`（Apache 2.0 全文）与 `NOTICE`（PyFlowVis 版权 + VortexTransformer 引用）到项目根。
  - 新增 `tests/test_vendor_migration.py`（8 项测试）+ `tests/conftest.py`，覆盖：导入缝、前向缝（形状 (B,256)、数值域 (0,1)、有限、种子可复现）、迁移边界（无参考仓库 import——AST 扫描 import 语句、LICENSE/NOTICE 存在、排除模块未迁移、EasyConfig 不可导入、BCELoss 已注册）。
  - 未迁移：`DeepUtils/dataset`、`DeepUtils/optim`、`DeepUtils/scheduler`、`MiscFunctions.py`、`TestClass.py`、`stable_hash.py`、`FLowUtils`、`config.py` 及一切 numba/pybind/GUI/wandb 依赖。
- **验证证据**：
  - `python -c "from vendor.DeepUtils.models import build_model_from_cfg"` 通过；
  - `python -m pytest tests/ -q` → **8 passed**（本地 CPU，torch 2.10.0+cpu，Python 3.12.3）；
  - 随机输入前向实测：`out.shape == (2, 256)`、`0 < out < 1`、无 NaN；
  - 全项目 grep 无 `PyflowVis` import；迁移文件与源 SHA256 一致（仅 utils/__init__.py 有意重写）。
- **/code-review 结果与处置**（双轴并行评审，commit 前修复）：
  - Standards 轴：硬违规 1 处——验收 2 守护测试原用 `enumerate(read_text(...))` 逐字符迭代致正则永不命中（空洞通过）。**已修复**：改 AST 解析仅扫描 import/from 语句（同时避免误报署名注释中的 "PyflowVis" 字样）。其余核查通过（忠实性/依赖纯净性/署名）。
  - Spec 轴：验收 1/3/4/5 满足，验收 2 随守护测试修复后满足。两条意见处置：
    1. "models/loss 全树超最小子集"：**保留全树**——HANDOFF §4 目录树（权威正文）明确列出 classification/、reconstruction/ 等目录，且全树均纯 torch、不违反验收 5；仅 `utils/__init__.py` 一处重写。
    2. 上游 `loss/build.py`（SmoothCrossEntropy ignore_index/weight 分支）与 `point_transformers.py`（PosE_Initial）含硬编码 `.cuda()`：CPU-only 下启用会崩。**未改动**（保持复制忠实，改上游属后续票范围）；已记 HANDOFF §7 风险表。当前 PathlineTransformerV0 + BCELoss 路径不触碰该代码。
  - 工作区无关未跟踪文件 `.scratch/vortex-extraction-pipeline/probe_geometry_overview.png`（非本票产物）：未提交、未改动。
- **遗留**：无阻塞性问题。下一步按 frontier：票 02（geometry 掩膜，无阻塞可并行启动）。
