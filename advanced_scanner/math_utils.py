"""
math_utils.py  ─  Core Math Functions
"""

import math
from typing import Union
import mpmath

# Set default precision for mpmath
mpmath.mp.dps = 50

def factorial(n: int) -> int:
    """Returns the factorial of n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    return math.factorial(n)

def gamma_function(x: Union[float, int, complex]) -> Union[float, complex]:
    """Returns the Gamma function of x using mpmath for high precision."""
    return mpmath.gamma(x)

def combinations(n: int, k: int) -> int:
    """Returns the number of combinations of n items taken k at a time."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k
    
    numerator = 1
    for i in range(k):
        numerator = numerator * (n - i) // (i + 1)
    return numerator

def set_precision(dps: int):
    """Sets the decimal precision for mpmath operations."""
    mpmath.mp.dps = dps

def to_mpf(val):
    """Converts a value to an mpmath float."""
    return mpmath.mpf(val)
