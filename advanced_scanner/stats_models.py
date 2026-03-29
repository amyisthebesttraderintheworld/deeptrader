"""
stats_models.py  ─  Probabilities, Distributions, Hypothesis Testing
"""

import numpy as np
from scipy import stats
from typing import List, Callable, Dict, Any, Union

def probability_density(dist_name: str, x: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
    """Returns the probability density (pdf/pmf) for a distribution at x."""
    dist = getattr(stats, dist_name, None)
    if dist is None:
        raise ValueError(f"Unknown distribution: {dist_name}")
    
    if hasattr(dist, 'pdf'):
        return dist.pdf(x, **kwargs)
    elif hasattr(dist, 'pmf'):
        return dist.pmf(x, **kwargs)
    else:
        raise ValueError(f"Distribution {dist_name} does not have a pdf or pmf.")

def cumulative_density(dist_name: str, x: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
    """Returns the cumulative density (cdf) for a distribution at x."""
    dist = getattr(stats, dist_name, None)
    if dist is None or not hasattr(dist, 'cdf'):
        raise ValueError(f"Unknown or non-cdf distribution: {dist_name}")
    return dist.cdf(x, **kwargs)

def t_test(sample1: List[float], sample2: List[float]):
    """Performs a two-sample t-test."""
    t_stat, p_val = stats.ttest_ind(sample1, sample2)
    return t_stat, p_val

def chi_square_test(observed: List[float], expected: List[float] = None):
    """Performs a Chi-square test of goodness-of-fit."""
    return stats.chisquare(observed, f_exp=expected)

def monte_carlo_simulation(func: Callable, n_trials: int = 1000, **kwargs):
    """Basic Monte Carlo simulation for a function with given parameters."""
    results = [func(**kwargs) for _ in range(n_trials)]
    return np.mean(results), np.std(results)

def z_score_stats(data: np.ndarray):
    """Returns z-scores of a dataset."""
    if np.std(data) < 1e-12: return np.zeros_like(data)
    return stats.zscore(data)

def identify_outliers(data: np.ndarray, threshold: float = 3.0):
    """Identifies outliers using z-score threshold."""
    z = z_score_stats(data)
    return np.where(np.abs(z) > threshold)[0]
