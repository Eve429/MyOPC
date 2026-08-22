"""JSON/NPZ 载荷的同目录临时文件原子写出。"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_write_json(path: Path, payload: dict) -> Path:
    """把 JSON 载荷经同目录临时文件原子写出，避免留下半截 plan。"""
    # 与目标同目录同卷的临时文件（os.replace 的原子性要求）
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".json", dir=path.parent)
    os.close(handle)  # 只借用文件名，内容用文本模式重写
    temporary = Path(temporary_name)  # Path 化
    try:  # 写入并原子替换
        with temporary.open("w", encoding="utf-8") as stream:  # 文本写
            json.dump(payload, stream, ensure_ascii=False, indent=2)  # 中文可读输出
        os.replace(temporary, path)  # 原子替换目标
    finally:  # 无论成败清理临时文件
        if temporary.exists():  # 尚存即删除
            temporary.unlink()  # 删除
    return path  # 返回最终路径


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> Path:
    """把 NPZ 载荷经同目录临时文件原子写出。"""
    # 与目标同目录同卷的临时文件（os.replace 的原子性要求）
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".npz", dir=path.parent)
    os.close(handle)  # 关闭句柄
    temporary = Path(temporary_name)  # Path 化
    try:  # 写出并原子替换
        with temporary.open("wb") as stream:  # 二进制写
            np.savez(stream, **arrays)  # 不压缩 NPZ
        os.replace(temporary, path)  # 原子替换
    finally:  # 清理
        if temporary.exists():  # 尚存
            temporary.unlink()  # 删除
    return path  # 返回路径
