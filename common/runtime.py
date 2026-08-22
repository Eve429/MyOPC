"""运行环境解析（common 内唯一依赖 torch 的模块）。"""

import torch


def resolve_device(device: str) -> str:
    """把配置设备字符串解析为实际设备（auto 时有 CUDA 用 CUDA）。"""
    if device == "auto":  # 自动选择
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device  # 显式设备原样透传
