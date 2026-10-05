"""只读复用上层 circleg 的 ``Algorithm/``（部署方案决策 2 / 7）。

导入本模块会：
1. 关闭字节码写入，避免在上层目录生成 ``__pycache__``；
2. 把 circleg 仓库根目录加入 ``sys.path``，使 ``import Algorithm`` 可用。

本项目不向上层写任何文件。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

CIRCLEG_ROOT = Path(__file__).resolve().parents[2]
if not (CIRCLEG_ROOT / "Algorithm" / "__init__.py").is_file():
    raise ImportError(f"未在 {CIRCLEG_ROOT} 找到 circleg 的 Algorithm/ 包")
if str(CIRCLEG_ROOT) not in sys.path:
    sys.path.append(str(CIRCLEG_ROOT))
