"""
calculus.py  ─  Differentiation, Integration, Optimization Helpers
"""

import numpy as np
from typing import Callable, Union, List

def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-7) -> float:
    """Computes the numerical derivative of f at x using central difference."""
    return (f(x + h) - f(x - h)) / (2 * h)

def gradient(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """Computes the gradient vector of a scalar function f at vector x."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def hessian(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Computes the Hessian matrix of a scalar function f at vector x."""
    n = len(x)
    hess = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # Second partial derivative f_ii
                x_plus = x.copy(); x_plus[i] += h
                x_minus = x.copy(); x_minus[i] -= h
                hess[i, j] = (f(x_plus) - 2 * f(x) + f(x_minus)) / (h**2)
            else:
                # Mixed partial derivative f_ij
                x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
                x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
                x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
                x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
                hess[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
    return hess

def partial_derivative(f: Callable[[np.ndarray], float], x: np.ndarray, index: int, h: float = 1e-7) -> float:
    """Computes the partial derivative of f with respect to x[index] at vector x."""
    x_plus = x.copy(); x_plus[index] += h
    x_minus = x.copy(); x_minus[index] -= h
    return (f(x_plus) - f(x_minus)) / (2 * h)
