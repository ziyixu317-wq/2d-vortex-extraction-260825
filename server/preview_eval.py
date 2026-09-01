"""服务器单帧预览入口。

实现复用兼容层以保持既有评估行为；正式多帧评估和 τ 敏感性请直接运行
``evaluate.py``。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from kaggle.preview_eval import main, project_to_grid, run_preview

__all__ = ["main", "project_to_grid", "run_preview"]


if __name__ == "__main__":
    main()
