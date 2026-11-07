"""把仓库根目录放进 sys.path，这样 tests/ 里可以直接 ``import Algorithm``。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
