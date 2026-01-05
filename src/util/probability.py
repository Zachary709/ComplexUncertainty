"""
概率计算工具函数
"""

import numpy as np
from typing import List, Union
import torch

from .vector import to_numpy


def softmax(x: Union[List[float], np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    计算softmax

    Args:
        x: 输入向量

    Returns:
        softmax后的概率分布
    """
    x = to_numpy(x)
    e_x = np.exp(x - np.max(x))  # 数值稳定性
    return e_x / e_x.sum()


def logprobs_to_probs(logprobs: List[float]) -> List[float]:
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


def kl_divergence_from_hidden(
    h1: Union[List[float], np.ndarray, torch.Tensor],
    h2: Union[List[float], np.ndarray, torch.Tensor],
) -> float:
    """
    计算两个hidden state的KL散度（近似）
    将hidden state通过softmax转换为概率分布后计算KL散度

    Args:
        h1: 第一个hidden state
        h2: 第二个hidden state

    Returns:
        KL散度值
    """
    h1 = to_numpy(h1)
    h2 = to_numpy(h2)

    p = softmax(h1)
    q = softmax(h2)

    # 添加小常数避免log(0)
    eps = 1e-10
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)

    return float(np.sum(p * np.log(p / q)))
