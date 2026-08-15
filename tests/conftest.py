"""测试共享路径工具。"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """返回仓库根目录，不依赖测试进程的当前工作目录。"""
    return Path(__file__).resolve().parents[1]
