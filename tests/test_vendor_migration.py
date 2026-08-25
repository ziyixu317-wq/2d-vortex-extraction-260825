"""票 01 vendor 迁移验收测试。

测试缝（规格 §Testing Decisions 模型缝）：
- 导入缝：`vendor.DeepUtils.models.build_model_from_cfg` 可导入；
- 前向缝：随机迹线输入跑通一次前向，输出形状 (B, 256)、数值域 (0, 1)；
- 迁移边界：全项目不引用参考仓库、LICENSE/NOTICE 保留、排除模块一律不迁移。
期望值来源：HANDOFF §2/§4 与票 01 验收标准（独立于实现）。
"""
import re

import pytest
import torch


@pytest.fixture(scope="module")
def pathline_model():
    from vendor.DeepUtils.models import build_model_from_cfg

    cfg = {
        "NAME": "PathlineTransformerV0",
        "in_channels": 7,
        "PathlineGroups": 64,
        "KpathlinePerGroup": 4,
    }
    return build_model_from_cfg(cfg)


# ---------- 验收 1：build_model_from_cfg 导入通过 ----------

def test_build_model_from_cfg_importable():
    from vendor.DeepUtils.models import build_model_from_cfg

    assert callable(build_model_from_cfg)


# ---------- 验收 2：全项目不再 import 参考仓库 ----------

def test_no_reference_repo_import_anywhere():
    """全项目（含 vendor 与 tests）不得 import 参考仓库 PyflowVis-main。

    用 AST 只检查 import/from 语句，避免误报 vendor 署名注释与本文档字符串
    中出现的 "PyflowVis" 字样（那些不是对参考仓库的代码引用）。
    """
    import ast

    from conftest import PROJECT_ROOT

    offenders = []
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        if ".git" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                if "PyflowVis" in name:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}: import {name}")
    assert not offenders, "发现 import 参考仓库的代码:\n" + "\n".join(offenders)


# ---------- 验收 3：LICENSE/NOTICE 随迁移代码保留 ----------

def test_license_and_notice_kept():
    from conftest import PROJECT_ROOT

    for name in ("LICENSE", "NOTICE"):
        f = PROJECT_ROOT / name
        assert f.is_file(), f"{name} 缺失"
        text = f.read_text(encoding="utf-8")
        assert "Apache" in text, f"{name} 中未找到 Apache 署名"


# ---------- 验收 4：随机输入跑通一次前向 ----------

def test_forward_output_shape_and_range(pathline_model):
    torch.manual_seed(0)
    B, L, K, C = 2, 16, 256, 7
    pathline_src = torch.randn(B, L, K, C)
    dummy_field = torch.zeros(B, 1)
    out = pathline_model((dummy_field, pathline_src))

    assert out.shape == (B, K), f"输出形状应为 (B, 256)，实际 {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "输出包含 NaN/Inf"
    assert (out > 0).all() and (out < 1).all(), "sigmoid 输出应位于 (0, 1)"


def test_forward_reproducible_under_seed(pathline_model):
    """固定随机种子后，两次前向输出应完全一致（随机性只来自 PSL，与模型无关）。"""
    pathline_src = torch.randn(1, 16, 256, 7)
    dummy_field = torch.zeros(1, 1)

    torch.manual_seed(42)
    out1 = pathline_model((dummy_field, pathline_src))
    torch.manual_seed(42)
    out2 = pathline_model((dummy_field, pathline_src))
    assert torch.equal(out1, out2)


# ---------- 验收 5：排除模块一律不迁移 ----------

def test_excluded_modules_not_migrated():
    from conftest import PROJECT_ROOT

    vendor = PROJECT_ROOT / "vendor"
    forbidden = [
        "DeepUtils/dataset",
        "DeepUtils/MiscFunctions.py",
        "DeepUtils/optim",
        "DeepUtils/scheduler",
        "DeepUtils/utils/config.py",
    ]
    for rel in forbidden:
        assert not (vendor / rel).exists(), f"不应迁移排除模块: {rel}"


def test_easyconfig_not_importable():
    """重写后的 utils 入口不得再导出依赖 multimethod 的 EasyConfig。"""
    from vendor.DeepUtils.utils import registry, ckpt_util, random  # noqa: F401

    with pytest.raises(ImportError):
        from vendor.DeepUtils.utils import EasyConfig  # noqa: F401


# ---------- 附：模型可从配置构建、BCELoss 已注册（HANDOFF §2 事实） ----------

def test_bceloss_registered():
    from vendor.DeepUtils.loss import build_criterion_from_cfg

    crit = build_criterion_from_cfg({"NAME": "BCELoss"})
    assert isinstance(crit, torch.nn.BCELoss)
