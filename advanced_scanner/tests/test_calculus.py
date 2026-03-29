"""
Unit tests for calculus.py
"""

import unittest
import numpy as np
from advanced_scanner.calculus import numerical_derivative, gradient, hessian, partial_derivative

class TestCalculus(unittest.TestCase):
    def test_numerical_derivative(self):
        def f(x): return x**2
        self.assertAlmostEqual(numerical_derivative(f, 3), 6.0, places=5)
        
        def g(x): return np.sin(x)
        self.assertAlmostEqual(numerical_derivative(g, 0), 1.0, places=5)

    def test_gradient(self):
        # f(x, y) = x^2 + y^2
        def f(v): return v[0]**2 + v[1]**2
        x = np.array([3.0, 4.0])
        grad = gradient(f, x)
        self.assertTrue(np.allclose(grad, [6.0, 8.0]))

    def test_hessian(self):
        # f(x, y) = x^2 + y^2
        def f(v): return v[0]**2 + v[1]**2
        x = np.array([3.0, 4.0])
        h = hessian(f, x)
        expected = np.array([[2.0, 0.0], [0.0, 2.0]])
        # print(f"Hessian: {h}")
        self.assertTrue(np.allclose(h, expected, atol=1e-3))

    def test_partial_derivative(self):
        # f(x, y) = x * y
        def f(v): return v[0] * v[1]
        x = np.array([3.0, 4.0])
        self.assertAlmostEqual(partial_derivative(f, x, 0), 4.0, places=5)
        self.assertAlmostEqual(partial_derivative(f, x, 1), 3.0, places=5)

if __name__ == '__main__':
    unittest.main()
