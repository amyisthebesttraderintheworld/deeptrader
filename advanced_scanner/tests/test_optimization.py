"""
Unit tests for optimization.py
"""

import unittest
import numpy as np
from advanced_scanner.optimization import gradient_descent, constrained_optimization, mean_variance_portfolio, sharpe_ratio_portfolio

class TestOptimization(unittest.TestCase):
    def test_gradient_descent(self):
        # f(x) = (x-3)^2 => f'(x) = 2(x-3)
        def f(x): return (x[0] - 3.0)**2
        def grad_f(x): return np.array([2.0 * (x[0] - 3.0)])
        x0 = np.array([0.0])
        x_min = gradient_descent(f, grad_f, x0)
        self.assertAlmostEqual(x_min[0], 3.0, places=5)

    def test_constrained_optimization(self):
        # Minimize: x^2 + y^2 subject to x + y = 2
        def f(v): return v[0]**2 + v[1]**2
        def constraint(v): return v[0] + v[1] - 2.0
        x0 = np.array([0.0, 0.0])
        constraints = [{'type': 'eq', 'fun': constraint}]
        x_opt, f_opt = constrained_optimization(f, x0, constraints=constraints)
        # Optima is at (1, 1) where x + y = 2 and f = 2
        self.assertTrue(np.allclose(x_opt, [1.0, 1.0]))
        self.assertAlmostEqual(f_opt, 2.0, places=5)

    def test_portfolio_optimization(self):
        returns = np.array([0.1, 0.15, 0.12])
        cov_matrix = np.array([
            [0.05, 0.01, 0.02],
            [0.01, 0.06, 0.01],
            [0.02, 0.01, 0.04]
        ])
        weights, risk = mean_variance_portfolio(returns, cov_matrix, target_return=0.12)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertAlmostEqual(weights @ returns, 0.12, places=5)

    def test_sharpe_ratio_portfolio(self):
        returns = np.array([0.1, 0.15, 0.12])
        cov_matrix = np.array([
            [0.05, 0.01, 0.02],
            [0.01, 0.06, 0.01],
            [0.02, 0.01, 0.04]
        ])
        weights, sharpe = sharpe_ratio_portfolio(returns, cov_matrix, risk_free_rate=0.02)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertTrue(sharpe > 0)

if __name__ == '__main__':
    unittest.main()
