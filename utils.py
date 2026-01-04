"""
工具函数模块
"""

import numpy as np
from typing import List, Union
import torch


def cosine_similarity(vec1: Union[List[float], np.ndarray, torch.Tensor], 
                      vec2: Union[List[float], np.ndarray, torch.Tensor]) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1: 第一个向量
        vec2: 第二个向量
    
    Returns:
        余弦相似度值，范围[-1, 1]
    """
    # 转换为numpy数组
    if isinstance(vec1, torch.Tensor):
        vec1 = vec1.cpu().numpy()
    elif isinstance(vec1, list):
        vec1 = np.array(vec1)
    
    if isinstance(vec2, torch.Tensor):
        vec2 = vec2.cpu().numpy()
    elif isinstance(vec2, list):
        vec2 = np.array(vec2)
    
    # 计算余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def cosine_similarity_matrix(hidden_states: List[Union[List[float], np.ndarray]]) -> np.ndarray:
    """
    计算一组hidden_states之间的余弦相似度矩阵
    
    Args:
        hidden_states: hidden_state向量列表
    
    Returns:
        余弦相似度矩阵，shape为(n, n)
    """
    n = len(hidden_states)
    
    # 转换为numpy数组
    vectors = []
    for hs in hidden_states:
        if isinstance(hs, torch.Tensor):
            vectors.append(hs.cpu().numpy())
        elif isinstance(hs, list):
            vectors.append(np.array(hs))
        else:
            vectors.append(hs)
    
    vectors = np.array(vectors)
    
    # 归一化
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    normalized = vectors / norms
    
    # 计算相似度矩阵
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix


def pairwise_cosine_similarity(hidden_states1: List[Union[List[float], np.ndarray]], 
                                hidden_states2: List[Union[List[float], np.ndarray]]) -> np.ndarray:
    """
    计算两组hidden_states之间的成对余弦相似度
    
    Args:
        hidden_states1: 第一组hidden_state向量列表
        hidden_states2: 第二组hidden_state向量列表
    
    Returns:
        余弦相似度矩阵，shape为(len(hidden_states1), len(hidden_states2))
    """
    # 转换为numpy数组
    def to_numpy(hs_list):
        vectors = []
        for hs in hs_list:
            if isinstance(hs, torch.Tensor):
                vectors.append(hs.cpu().numpy())
            elif isinstance(hs, list):
                vectors.append(np.array(hs))
            else:
                vectors.append(hs)
        return np.array(vectors)
    
    vectors1 = to_numpy(hidden_states1)
    vectors2 = to_numpy(hidden_states2)
    
    # 归一化
    norms1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(vectors2, axis=1, keepdims=True)
    norms1[norms1 == 0] = 1
    norms2[norms2 == 0] = 1
    
    normalized1 = vectors1 / norms1
    normalized2 = vectors2 / norms2
    
    # 计算相似度矩阵
    similarity_matrix = np.dot(normalized1, normalized2.T)
    
    return similarity_matrix


def average_cosine_similarity(hidden_states: List[Union[List[float], np.ndarray]]) -> float:
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


def consecutive_cosine_similarity(hidden_states: List[Union[List[float], np.ndarray]]) -> List[float]:
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

def logprobs2probs(logprobs: List[float]) -> List[float]:
    """
    将log概率转换为概率值
    
    Args:
        logprobs: log概率列表
    
    Returns:
        概率值列表
    """
    return [float(np.exp(lp)) for lp in logprobs]


def entropy_from_logprobs(logprobs: List[float]) -> float:
    """
    计算概率分布的熵值，输入为log概率列表
    
    Args:
        logprobs: log概率列表
    
    Returns:
        熵值
    """
    probs = np.exp(logprobs)
    probs /= probs.sum()  # 归一化
    entropy = -np.sum(probs * np.log(probs + 1e-12))  # 避免log(0)
    return float(entropy)