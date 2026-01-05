"""
向量处理工具函数
"""

import numpy as np
from typing import List, Union
import torch


def to_numpy(vec: Union[List[float], np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    将向量转换为numpy数组

    Args:
        vec: 输入向量，可以是list、numpy数组或torch张量

    Returns:
        numpy数组
    """
    if isinstance(vec, torch.Tensor):
        return vec.cpu().numpy()
    elif isinstance(vec, list):
        return np.array(vec)
    return vec


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    归一化向量数组

    Args:
        vectors: 向量数组，shape为(n, d)

    Returns:
        归一化后的向量数组
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    return vectors / norms
