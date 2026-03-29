"""
Unit tests for stats_models.py
"""

import unittest
import numpy as np
from advanced_scanner.stats_models import probability_density, cumulative_density, t_test, monte_carlo_simulation, z_score_stats, identify_outliers

class TestStatsModels(unittest.TestCase):
    def test_pdf_cdf(self):
        # Normal distribution PDF at 0 (mean 0, std 1)
        pdf_val = probability_density('norm', 0, loc=0, scale=1)
        self.assertAlmostEqual(pdf_val, 0.39894, places=5)
        
        # Normal distribution CDF at 0
        cdf_val = cumulative_density('norm', 0, loc=0, scale=1)
        self.assertEqual(cdf_val, 0.5)

    def test_t_test(self):
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [10, 11, 12, 13, 14]
        t_stat, p_val = t_test(sample1, sample2)
        self.assertTrue(p_val < 0.05)

    def test_z_score_outliers(self):
        data = np.array([1, 1.1, 0.9, 1.2, 10.0]) # 10 is an outlier
        z = z_score_stats(data)
        self.assertEqual(len(z), 5)
        outliers = identify_outliers(data, threshold=1.5) # lower threshold for testing
        self.assertIn(4, outliers)

    def test_monte_carlo(self):
        def f(a, b): return a + b
        # Expected value of sum of two uniforms [0, 1] is 1.0
        import random
        def func(): return random.random() + random.random()
        mean, std = monte_carlo_simulation(func, n_trials=1000)
        self.assertAlmostEqual(mean, 1.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()
