"""
numerical_methods.py  ─  Root-finding, Interpolation, Numerical Integration
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Callable, List, Tuple

def bisection_method(f: Callable, a: float, b: float, tol: float = 1e-6, max_iter: int = 100):
    """Finds a root of f in [a, b] using bisection."""
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    
    for _ in range(max_iter):
        mid = (a + b) / 2
        if abs(f(mid)) < tol or (b - a) / 2 < tol:
            return mid
        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2

def newton_raphson(f: Callable, df: Callable, x0: float, tol: float = 1e-6, max_iter: int = 100):
    """Finds a root of f using Newton-Raphson iteration."""
    x = x0
    for _ in range(max_iter):
        fx = f(x); dfx = df(x)
        if abs(dfx) < 1e-12:
            break
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def secant_method(f: Callable, x0: float, x1: float, tol: float = 1e-6, max_iter: int = 100):
    """Finds a root of f using the Secant method."""
    for _ in range(max_iter):
        fx0 = f(x0); fx1 = f(x1)
        if abs(fx1 - fx0) < 1e-12:
            break
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(x_new - x1) < tol:
            return x_new
        x0, x1 = x1, x_new
    return x1

def cubic_spline_interpolation(x_points: np.ndarray, y_points: np.ndarray, x_new: np.ndarray):
    """Interpolates values using cubic splines."""
    cs = CubicSpline(x_points, y_points)
    return cs(x_new)

def simpson_integration(f: Callable, a: float, b: float, n: int = 100):
    """Computes integral using Simpson's rule (n must be even)."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    s = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2])
    return s * h / 3

def runge_kutta_4(f: Callable[[float, float], float], y0: float, t: np.ndarray):
    """Solves dy/dt = f(t, y) using 4th order Runge-Kutta."""
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        dt = t[i+1] - t[i]
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt/2, y[i] + dt*k1/2)
        k3 = f(t[i] + dt/2, y[i] + dt*k2/2)
        k4 = f(t[i] + dt, y[i] + dt*k3)
        y[i+1] = y[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y
