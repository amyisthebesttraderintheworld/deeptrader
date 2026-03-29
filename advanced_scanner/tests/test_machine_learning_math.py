"""
Unit tests for machine_learning_math.py
"""

import unittest
import numpy as np
from advanced_scanner.machine_learning_math import soft_max, sigmoid, relu, cross_entropy, kernel_matrix, distance_metric, covariance_matrix, correlation_matrix

class TestMLMath(unittest.TestCase):
    def test_activation_functions(self):
        x = np.array([1.0, 2.0, 3.0])
        sm = soft_max(x)
        self.assertAlmostEqual(np.sum(sm), 1.0, places=5)
        
        self.assertAlmostEqual(sigmoid(0), 0.5)
        
        self.assertTrue(np.allclose(relu(np.array([-1, 0, 1])), [0, 0, 1]))

    def test_cross_entropy(self):
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1]])
        loss = cross_entropy(y_true, y_pred)
        # -ln(0.7) approx 0.356
        self.assertAlmostEqual(loss, 0.356, delta=0.1)

    def test_kernel_matrix(self):
        X = np.array([[1, 2], [3, 4]])
        K_linear = kernel_matrix(X, kernel='linear')
        self.assertTrue(np.allclose(K_linear, [[5, 11], [11, 25]]))
        
        K_rbf = kernel_matrix(X, kernel='rbf', gamma=0.5)
        self.assertEqual(K_rbf.shape, (2, 2))
        self.assertTrue(np.allclose(np.diag(K_rbf), [1.0, 1.0]))

    def test_distance_metrics(self):
        x = np.array([0, 0])
        y = np.array([3, 4])
        self.assertEqual(distance_metric(x, y, 'euclidean'), 5.0)
        self.assertEqual(distance_metric(x, y, 'manhattan'), 7.0)
        
        # Cosine distance
        x2 = np.array([1, 0])
        y2 = np.array([0, 1])
        # orthogonal vectors, cosine similarity 0, distance 1
        self.assertEqual(distance_metric(x2, y2, 'cosine'), 1.0)

    def test_cov_corr(self):
        X = np.array([[1, 2], [2, 4], [3, 6]]) # perfectly correlated
        cov = covariance_matrix(X)
        self.assertEqual(cov.shape, (2, 2))
        
        corr = correlation_matrix(X)
        self.assertAlmostEqual(corr[0, 1], 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
