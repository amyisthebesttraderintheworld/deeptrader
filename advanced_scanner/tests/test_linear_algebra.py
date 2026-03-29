"""
Unit tests for linear_algebra.py
"""

import unittest
import numpy as np
from advanced_scanner.linear_algebra import eigen_decomposition, singular_value_decomposition, principal_component_analysis, matrix_rank, matrix_inverse

class TestLinearAlgebra(unittest.TestCase):
    def test_eigen_decomposition(self):
        matrix = np.array([[2, 0], [0, 3]])
        vals, vecs = eigen_decomposition(matrix)
        self.assertTrue(np.allclose(sorted(vals), [2, 3]))

    def test_svd(self):
        matrix = np.array([[1, 2], [3, 4], [5, 6]])
        u, s, vh = singular_value_decomposition(matrix)
        self.assertEqual(u.shape, (3, 3))
        self.assertEqual(len(s), 2)
        self.assertEqual(vh.shape, (2, 2))

    def test_pca(self):
        data = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])
        transformed, components, vals = principal_component_analysis(data, 1)
        self.assertEqual(transformed.shape, (10, 1))
        self.assertEqual(components.shape, (2, 1))
        self.assertEqual(len(vals), 1)

    def test_matrix_rank(self):
        matrix = np.eye(4)
        self.assertEqual(matrix_rank(matrix), 4)

    def test_matrix_inverse(self):
        matrix = np.array([[1, 2], [3, 4]])
        inv = matrix_inverse(matrix)
        identity = np.dot(matrix, inv)
        self.assertTrue(np.allclose(identity, np.eye(2)))

if __name__ == '__main__':
    unittest.main()
