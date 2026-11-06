"""本工作区的日志配置唯一入口。

职责（只做一件事）：决定日志**往哪写、写成什么格式、什么级别**。

为什么单独成模块（用户 2026-10-07 要求，叠加全局 AGENTS.md 第 3 节）：
- 之前 ``logging.basicConfig`` 散落在 6 个工具里、格式各不相同、级别写死，
  业务模块无法统一控制；日志一多就分不清哪个模块在说话。
- 现在**只有本模块允许碰 handler / formatter / level**。其余模块（``simulation`` /
  ``wbc`` / ``dynamics`` / 各 ``tools``）只做两件事：``get_logger(__name__)`` 取 logger、
  然后 ``_LOGGER.info(...)`` 写日志。业务模块里禁止再出现 ``basicConfig``。

级别怎么调（禁止写死在业务代码里）：
1. 环境变量 ``GBC_LOG_LEVEL``（DEBUG / INFO / WARNING / ERROR，大小写不敏感）优先级最高；
2. 未设置时用调用方传入的 ``default_level``（各工具按自己需要给，例如离线核算给 WARNING）；
3. 都没有则回落 ``INFO``。

输出去向：默认 stderr（与工具报告的 stdout 分开——报告走 print/stdout，日志走 stderr），
需要落盘时传 ``log_file``，固定 UTF-8，避免 Windows GBK 终端把中文日志打成乱码。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# 含时间、级别、模块名（模块名是 ``gbc_mpc_wbc.simulation`` 这种全名，便于按模块过滤）
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL_ENV = "GBC_LOG_LEVEL"


def resolve_level(default_level: str | int = "INFO") -> int:
    """定级别：环境变量 > 调用方默认值 > INFO。非法取值回落并告警。"""
    raw = os.environ.get(LOG_LEVEL_ENV)
    if raw is None or not str(raw).strip():
        raw = default_level
    if isinstance(raw, int):
        return raw
    name = str(raw).strip().upper()
    level = logging.getLevelName(name)
    if not isinstance(level, int):
        logging.getLogger(__name__).warning(
            "无法识别的日志级别 %r（来源：%s 或调用默认值），回落 INFO",
            raw, LOG_LEVEL_ENV if os.environ.get(LOG_LEVEL_ENV) else "default_level")
        return logging.INFO
    return level


def setup_logging(default_level: str | int = "INFO", stream=None,
                  log_file: str | Path | None = None, force: bool = True) -> int:
    """装配根 logger 的 handler 与级别，返回最终生效的级别。

    ``stream`` 给 None 时用 stderr；``log_file`` 非空时额外挂一个 UTF-8 文件 handler。
    ``force=True`` 会先清掉既有 handler，保证重复调用（测试里常见）不会叠加输出。
    """
    level = resolve_level(default_level)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr if stream is None else stream)]
    if log_file is not None:
        handlers.append(logging.FileHandler(Path(log_file), encoding="utf-8"))
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=handlers, force=force)
    return level


def get_logger(name: str) -> logging.Logger:
    """模块级取 logger 的唯一方式；配置由 ``setup_logging`` 负责（这里不碰 handler）。"""
    return logging.getLogger(name)
