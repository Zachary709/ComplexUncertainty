"""
相似度计算工具函数
"""

import numpy as np
from typing import List, Union
import torch

from .vector import to_numpy, normalize_vectors


def cosine_similarity(
    vec1: Union[List[float], np.ndarray, torch.Tensor],
    vec2: Union[List[float], np.ndarray, torch.Tensor],
) -> float:
    """
    计算两个向量的余弦相似度

    Args:
        vec1: 第一个向量
        vec2: 第二个向量

    Returns:
        余弦相似度值，范围[-1, 1]
    """
    vec1 = to_numpy(vec1)
    vec2 = to_numpy(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def cosine_similarity_matrix(
    hidden_states: List[Union[List[float], np.ndarray, torch.Tensor]]
) -> np.ndarray:
    """
    计算一组hidden_states之间的余弦相似度矩阵

    Args:
        hidden_states: hidden_state向量列表

    Returns:
        余弦相似度矩阵，shape为(n, n)
    """
    vectors = np.array([to_numpy(hs) for hs in hidden_states])
    normalized = normalize_vectors(vectors)
    similarity_matrix = np.dot(normalized, normalized.T)
    return similarity_matrix


def pairwise_cosine_similarity(
    hidden_states1: List[Union[List[float], np.ndarray, torch.Tensor]],
    hidden_states2: List[Union[List[float], np.ndarray, torch.Tensor]],
) -> np.ndarray:
    """
    计算两组hidden_states之间的成对余弦相似度

    Args:
        hidden_states1: 第一组hidden_state向量列表
        hidden_states2: 第二组hidden_state向量列表

    Returns:
        余弦相似度矩阵，shape为(len(hidden_states1), len(hidden_states2))
    """
    vectors1 = np.array([to_numpy(hs) for hs in hidden_states1])
    vectors2 = np.array([to_numpy(hs) for hs in hidden_states2])

    normalized1 = normalize_vectors(vectors1)
    normalized2 = normalize_vectors(vectors2)

    similarity_matrix = np.dot(normalized1, normalized2.T)
    return similarity_matrix


def average_cosine_similarity(
    hidden_states: List[Union[List[float], np.ndarray, torch.Tensor]]
) -> float:
    """
    计算一组hidden_states的平均余弦相似度（不包括对角线）

    Args:
        hidden_states: hidden_state向量列表

    Returns:
        平均余弦相似度
    """
    sim_matrix = cosine_similarity_matrix(hidden_states)
    n = len(hidden_states)

    if n <= 1:
        return 1.0

    # 排除对角线元素
    mask = ~np.eye(n, dtype=bool)
    avg_sim = sim_matrix[mask].mean()

    return float(avg_sim)


def consecutive_cosine_similarity(
    hidden_states: List[Union[List[float], np.ndarray, torch.Tensor]]
) -> List[float]:
    """
    计算相邻hidden_states之间的余弦相似度

    Args:
        hidden_states: hidden_state向量列表

    Returns:
        相邻token之间的余弦相似度列表
    """
    similarities = []
    for i in range(len(hidden_states) - 1):
        sim = cosine_similarity(hidden_states[i], hidden_states[i + 1])
        similarities.append(sim)
    return similarities
