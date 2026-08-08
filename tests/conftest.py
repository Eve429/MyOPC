"""版图与几何测试共享路径和确定性工具。"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """返回仓库根目录，不依赖测试进程的当前工作目录。"""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def reticle_dir(project_root: Path) -> Path:
    """返回已经纳入仓库的 GDS 回归数据目录。"""
    return project_root / "TestReticle"
