"""
Unit tests for chaos_and_complexity.py
"""

import unittest
import numpy as np
from advanced_scanner.chaos_and_complexity import lyapunov_exponent, hurst_exponent, fractal_dimension, logistic_map, entropy

class TestChaosComplexity(unittest.TestCase):
    def test_lyapunov_exponent(self):
        # Logistic map in chaotic regime (r=4.0)
        x = logistic_map(4.0, 0.5, 1000)
        lyap = lyapunov_exponent(x)
        # Should be positive for chaotic systems
        self.assertTrue(lyap > 0)

    def test_hurst_exponent(self):
        # Random walk Hurst should be 0.5
        n = 1000
        x = np.cumsum(np.random.normal(0, 1, n))
        h = hurst_exponent(x)
        self.assertAlmostEqual(h, 0.5, delta=0.1)
        
        # Mean reverting series Hurst < 0.5
        x_mr = np.random.normal(0, 1, n)
        h_mr = hurst_exponent(x_mr)
        self.assertTrue(h_mr < 0.5)

    def test_fractal_dimension(self):
        n = 1000
        x = np.cumsum(np.random.normal(0, 1, n))
        fd = fractal_dimension(x)
        # Brownian motion fractal dimension is approx 1.5
        self.assertTrue(1.0 < fd < 2.0)

    def test_logistic_map(self):
        x = logistic_map(3.5, 0.5, 100)
        self.assertEqual(len(x), 100)
        self.assertTrue(np.all(x >= 0) and np.all(x <= 1))

    def test_entropy(self):
        # Uniform distribution should have higher entropy than constant
        x_unif = np.random.uniform(0, 1, 1000)
        x_const = np.ones(1000)
        e_unif = entropy(x_unif)
        e_const = entropy(x_const)
        self.assertTrue(e_unif > e_const)

if __name__ == '__main__':
    unittest.main()
