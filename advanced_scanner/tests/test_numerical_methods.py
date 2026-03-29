"""
Unit tests for numerical_methods.py
"""

import unittest
import numpy as np
from advanced_scanner.numerical_methods import bisection_method, newton_raphson, secant_method, cubic_spline_interpolation, simpson_integration, runge_kutta_4

class TestNumericalMethods(unittest.TestCase):
    def test_root_finding(self):
        # f(x) = x^2 - 4, root at x=2
        def f(x): return x**2 - 4.0
        def df(x): return 2.0 * x
        
        root1 = bisection_method(f, 0, 4)
        self.assertAlmostEqual(root1, 2.0, places=5)
        
        root2 = newton_raphson(f, df, 3.0)
        self.assertAlmostEqual(root2, 2.0, places=5)
        
        root3 = secant_method(f, 1.0, 3.0)
        self.assertAlmostEqual(root3, 2.0, places=5)

    def test_interpolation(self):
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 4]) # y = x^2
        x_new = np.array([0.5, 1.5])
        y_interp = cubic_spline_interpolation(x, y, x_new)
        self.assertAlmostEqual(y_interp[0], 0.25, delta=0.5)
        self.assertAlmostEqual(y_interp[1], 2.25, delta=0.5)

    def test_integration(self):
        # Integral of x^2 from 0 to 1 is 1/3
        def f(x): return x**2
        integral = simpson_integration(f, 0, 1, 100)
        self.assertAlmostEqual(integral, 1/3, places=5)

    def test_runge_kutta(self):
        # dy/dt = y, y(0) = 1 => y(t) = exp(t)
        def f(t, y): return y
        t = np.linspace(0, 1, 101)
        y = runge_kutta_4(f, 1.0, t)
        self.assertAlmostEqual(y[-1], np.exp(1.0), places=3)

if __name__ == '__main__':
    unittest.main()
