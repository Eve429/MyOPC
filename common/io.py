"""JSON/NPZ 载荷的同目录临时文件原子写出。"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_write_json(path: Path, payload: dict) -> Path:
    """把 JSON 载荷经同目录临时文件原子写出，避免留下半截 plan。"""
    # 临时文件必须与目标同目录同卷，os.replace 才有原子性
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".json", dir=path.parent)
    os.close(handle)  # 只借用文件名，内容用文本模式重写
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)  # 中文可读输出
        os.replace(temporary, path)
    finally:  # 无论成败清理临时文件
        if temporary.exists():
            temporary.unlink()
    return path


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> Path:
    """把 NPZ 载荷经同目录临时文件原子写出。"""
    # 临时文件必须与目标同目录同卷，os.replace 才有原子性
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".npz", dir=path.parent)
    os.close(handle)  # 只借用文件名，内容用二进制模式重写
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)  # 不压缩 NPZ
        os.replace(temporary, path)
    finally:  # 无论成败清理临时文件
        if temporary.exists():
            temporary.unlink()
    return path
