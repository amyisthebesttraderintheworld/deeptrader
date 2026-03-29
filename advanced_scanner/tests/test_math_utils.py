"""
Unit tests for math_utils.py
"""

import unittest
import math
from advanced_scanner.math_utils import factorial, gamma_function, combinations, set_precision, to_mpf

class TestMathUtils(unittest.TestCase):
    def test_factorial(self):
        self.assertEqual(factorial(5), 120)
        self.assertEqual(factorial(0), 1)
        with self.assertRaises(ValueError):
            factorial(-1)

    def test_gamma_function(self):
        # Gamma(n) = (n-1)!
        self.assertAlmostEqual(float(gamma_function(5)), 24.0, places=5)
        self.assertAlmostEqual(float(gamma_function(6)), 120.0, places=5)

    def test_combinations(self):
        self.assertEqual(combinations(5, 2), 10)
        self.assertEqual(combinations(10, 3), 120)
        self.assertEqual(combinations(5, 0), 1)
        self.assertEqual(combinations(5, 5), 1)
        self.assertEqual(combinations(5, 6), 0)

    def test_mpmath_precision(self):
        set_precision(100)
        import mpmath
        self.assertEqual(mpmath.mp.dps, 100)
        v = to_mpf("1.234567890123456789")
        self.assertTrue(isinstance(v, mpmath.mpf))

if __name__ == '__main__':
    unittest.main()
