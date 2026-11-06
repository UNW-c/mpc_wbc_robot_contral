"""项目内路径引导。

GBC486 项目自带 ``Algorithm/``，运行时不依赖上层 circleg 工作区。
导入本模块会：

1. 关闭字节码写入，避免在源码目录留下 ``__pycache__``；
2. 把本项目根目录放到 ``sys.path`` 最前面，使 ``import Algorithm``
   指向 ``gbc486/Algorithm``。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "Algorithm" / "__init__.py").is_file():
    raise ImportError(f"未在 {PROJECT_ROOT} 找到本项目的 Algorithm/ 包")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
