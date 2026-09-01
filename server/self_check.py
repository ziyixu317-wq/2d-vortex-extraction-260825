"""服务器环境自检入口。

自检验证 vendor 模型前向、数据集 memmap 可读性以及 7 通道迹线样本的有限性。
实现暂复用历史模块以保持行为与 checkpoint 兼容；服务器执行请使用本入口。

用法：
    python server/self_check.py --config config/pathline_transformer_multi.yaml \
        --data-root outputs/datasets/pipedcylinder2d/dataset --device cuda
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from kaggle.self_check import main, self_check

__all__ = ["main", "self_check"]


if __name__ == "__main__":
    main()
