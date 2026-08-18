"""JSON/NPZ 载荷的同目录临时文件原子写出。"""

import json  # 序列化 JSON 载荷
import os  # 原子替换与句柄管理
import tempfile  # 创建与目标同目录的临时文件
from pathlib import Path  # 全部路径统一使用 Path 对象

import numpy as np  # NPZ 数组载体


def atomic_write_json(path: Path, payload: dict) -> Path:
    """把 JSON 载荷经同目录临时文件原子写出，避免留下半截 plan。"""
    handle, temporary_name = tempfile.mkstemp(  # 与目标同目录同卷
        prefix=f".{path.stem}-", suffix=".json", dir=path.parent)  # 临时文件名
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
    handle, temporary_name = tempfile.mkstemp(  # 同目录临时文件
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent)  # 命名
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
