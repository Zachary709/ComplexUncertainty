"""
工具函数模块
"""

from .similarity import (
    cosine_similarity,
    cosine_similarity_matrix,
    pairwise_cosine_similarity,
    average_cosine_similarity,
    consecutive_cosine_similarity,
)
from .probability import (
    logprobs_to_probs,
    entropy_from_logprobs,
    kl_divergence_from_hidden,
    softmax,
)
from .vector import (
    to_numpy,
    normalize_vectors,
)

__all__ = [
    # similarity
    "cosine_similarity",
    "cosine_similarity_matrix",
    "pairwise_cosine_similarity",
    "average_cosine_similarity",
    "consecutive_cosine_similarity",
    # probability
    "logprobs_to_probs",
    "entropy_from_logprobs",
    "kl_divergence_from_hidden",
    "softmax",
    # vector
    "to_numpy",
    "normalize_vectors",
]
