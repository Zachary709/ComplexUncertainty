"""
工具函数单元测试
"""

import os
import sys
import unittest
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.util import (
    cosine_similarity,
    cosine_similarity_matrix,
    average_cosine_similarity,
    consecutive_cosine_similarity,
    logprobs_to_probs,
    entropy_from_logprobs,
    softmax,
    to_numpy,
    normalize_vectors,
)


class TestSimilarityFunctions(unittest.TestCase):
    """相似度函数测试"""

    def test_cosine_similarity_identical(self):
        """测试相同向量的余弦相似度"""
        vec = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0, places=6)

    def test_cosine_similarity_orthogonal(self):
        """测试正交向量的余弦相似度"""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), 0.0, places=6)

    def test_cosine_similarity_opposite(self):
        """测试相反向量的余弦相似度"""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        self.assertAlmostEqual(cosine_similarity(vec1, vec2), -1.0, places=6)

    def test_cosine_similarity_matrix(self):
        """测试余弦相似度矩阵"""
        vectors = [[1, 0], [0, 1], [1, 1]]
        matrix = cosine_similarity_matrix(vectors)
        self.assertEqual(matrix.shape, (3, 3))
        # 对角线应该是1
        np.testing.assert_array_almost_equal(np.diag(matrix), [1, 1, 1])

    def test_average_cosine_similarity(self):
        """测试平均余弦相似度"""
        vectors = [[1, 0], [1, 0], [1, 0]]  # 完全相同的向量
        avg_sim = average_cosine_similarity(vectors)
        self.assertAlmostEqual(avg_sim, 1.0, places=6)

    def test_consecutive_cosine_similarity(self):
        """测试相邻余弦相似度"""
        vectors = [[1, 0], [1, 1], [0, 1]]
        sims = consecutive_cosine_similarity(vectors)
        self.assertEqual(len(sims), 2)


class TestProbabilityFunctions(unittest.TestCase):
    """概率函数测试"""

    def test_softmax_sum_to_one(self):
        """测试softmax和为1"""
        x = [1.0, 2.0, 3.0]
        result = softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0, places=6)

    def test_logprobs_to_probs(self):
        """测试log概率转换"""
        logprobs = [np.log(0.5), np.log(0.3), np.log(0.2)]
        probs = logprobs_to_probs(logprobs)
        self.assertAlmostEqual(probs[0], 0.5, places=6)
        self.assertAlmostEqual(probs[1], 0.3, places=6)
        self.assertAlmostEqual(probs[2], 0.2, places=6)

    def test_entropy_uniform(self):
        """测试均匀分布的熵"""
        # 均匀分布的熵应该最大
        logprobs = [np.log(0.25)] * 4
        entropy = entropy_from_logprobs(logprobs)
        expected = -np.sum([0.25 * np.log(0.25) for _ in range(4)])
        self.assertAlmostEqual(entropy, expected, places=5)


class TestVectorFunctions(unittest.TestCase):
    """向量函数测试"""

    def test_to_numpy_from_list(self):
        """测试从列表转换"""
        result = to_numpy([1, 2, 3])
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_normalize_vectors(self):
        """测试向量归一化"""
        vectors = np.array([[3, 4], [1, 0]])
        normalized = normalize_vectors(vectors)
        # 检查模长为1
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1, 1])


if __name__ == "__main__":
    unittest.main()
