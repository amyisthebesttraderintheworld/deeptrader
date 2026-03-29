"""
Unit tests for stochastic_processes.py
"""

import unittest
import numpy as np
from advanced_scanner.stochastic_processes import brownian_motion, geometric_brownian_motion, monte_carlo_paths, jump_diffusion

class TestStochasticProcesses(unittest.TestCase):
    def test_brownian_motion(self):
        T, N = 1.0, 100
        W = brownian_motion(T, N)
        self.assertEqual(len(W), N + 1)
        self.assertEqual(W[0], 0.0)

    def test_gbm(self):
        S0, mu, sigma, T, N = 100.0, 0.05, 0.2, 1.0, 100
        S = geometric_brownian_motion(S0, mu, sigma, T, N)
        self.assertEqual(len(S), N + 1)
        self.assertEqual(S[0], S0)
        self.assertTrue(np.all(S > 0)) # Prices should be positive

    def test_monte_carlo_paths(self):
        S0, mu, sigma, T, N, n_paths = 100.0, 0.05, 0.2, 1.0, 100, 10
        paths = monte_carlo_paths(S0, mu, sigma, T, N, n_paths)
        self.assertEqual(paths.shape, (10, 101))

    def test_jump_diffusion(self):
        S0, mu, sigma, T, N, lambda_j, mu_j, sigma_j = 100.0, 0.05, 0.2, 1.0, 100, 5, 0.01, 0.05
        S = jump_diffusion(S0, mu, sigma, T, N, lambda_j, mu_j, sigma_j)
        self.assertEqual(len(S), N + 1)
        self.assertEqual(S[0], S0)
        self.assertTrue(np.all(S > 0))

if __name__ == '__main__':
    unittest.main()
