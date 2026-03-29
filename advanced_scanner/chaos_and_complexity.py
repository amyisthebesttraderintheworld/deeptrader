"""
chaos_and_complexity.py  ─  Fractals, Lyapunov Exponents, Non-linear Dynamics
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def fast_rolling_sum(arr: np.ndarray, window: int):
    """Blazing fast rolling sum using cumsum."""
    n = len(arr)
    if n < window: return np.zeros(n)
    cs = np.cumsum(arr)
    res = np.zeros(n)
    res[window-1] = cs[window-1]
    res[window:] = cs[window:] - cs[:-window]
    return res

def rolling_std(series: np.ndarray, window: int):
    """Fast rolling standard deviation using fast_rolling_sum."""
    if len(series) < window: return np.zeros_like(series)
    sum_s = fast_rolling_sum(series, window)
    sum_s2 = fast_rolling_sum(series**2, window)
    
    mean = sum_s / window
    var = (sum_s2 / window) - (mean**2)
    # Use index slicing to only apply sqrt where needed
    std = np.sqrt(np.maximum(var[window-1:], 0))
    
    res = np.zeros_like(series)
    res[window-1:] = std
    return res

def rolling_hurst(series: np.ndarray, window: int = 50):
    """Vectorized rolling Hurst exponent."""
    n = len(series)
    if n < window: return np.full_like(series, 0.5)
    
    lags = np.arange(2, 20)
    log_lags = np.log(lags)
    var_log_lags = np.var(log_lags)
    
    log_taus = []
    for lag in lags:
        diffs = np.zeros(n)
        diffs[lag:] = series[lag:] - series[:-lag]
        std_v = rolling_std(diffs, window)
        log_taus.append(np.log(np.maximum(std_v, 1e-12)))
    
    log_taus = np.array(log_taus)
    E_y = np.mean(log_taus, axis=0)
    E_x = np.mean(log_lags)
    E_xy = np.mean(log_taus * log_lags[:, np.newaxis], axis=0)
    
    slopes = (E_xy - E_x * E_y) / var_log_lags
    slopes[:window-1] = 0.5
    return slopes

def rolling_entropy(series: np.ndarray, window: int = 50, bins: int = 10):
    """Fully vectorized rolling entropy using fast_rolling_sum."""
    n = len(series)
    if n < window: return np.zeros(n)
    
    s_min, s_max = np.min(series), np.max(series)
    if s_max == s_min: return np.zeros(n)
    
    quantized = np.clip(((series - s_min) / (s_max - s_min) * (bins - 1)).astype(int), 0, bins - 1)
    
    # Probabilities p: shape (bins, n)
    res_entropy = np.zeros(n)
    
    # We'll calculate p for each bin rolling-ly
    ps = []
    for b in range(bins):
        bin_indicator = (quantized == b).astype(float)
        counts = fast_rolling_sum(bin_indicator, window)
        ps.append(counts / window)
        
    ps = np.array(ps) # (bins, n)
    
    log_p = np.zeros_like(ps)
    mask = ps > 0
    log_p[mask] = np.log2(ps[mask])
    
    entropy_vals = -np.sum(ps * log_p, axis=0)
    return entropy_vals

def hurst_exponent(series: np.ndarray):
    lags = np.arange(2, 20)
    tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
    tau = np.maximum(tau, 1e-12)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

def entropy(series: np.ndarray, bins: int = 10):
    hist, _ = np.histogram(series, bins=bins, density=True)
    p = hist[hist > 0]
    return -np.sum(p * np.log2(p))

def lyapunov_exponent(series: np.ndarray):
    """
    Estimates the Lyapunov exponent of a 1D time series.
    For a map x_{n+1} = f(x_n), lambda = mean(log|f'(x_i)|).
    We approximate |f'(x_i)| as |(x_{i+1} - x_i) / (x_i - x_{i-1})|.
    """
    n = len(series)
    if n < 3: return 0.0
    diffs = np.diff(series)
    # Avoid division by zero and log of zero
    ratios = np.abs(diffs[1:] / (diffs[:-1] + 1e-12))
    ratios = ratios[ratios > 1e-10]
    if len(ratios) == 0: return 0.0
    return np.mean(np.log(ratios))

def fractal_dimension(series: np.ndarray, method: str = 'hurst', threshold: float = None):
    if method == 'hurst': return 2.0 - hurst_exponent(series)
    if threshold is None: threshold = np.std(series) * 0.1
    binary = (np.abs(np.diff(series)) > threshold).astype(int)
    scales = 2**np.arange(1, int(np.log2(len(binary)//4)))
    counts = []
    for s in scales:
        n_points = (len(binary) // s) * s
        if n_points == 0: continue
        reshaped = binary[:n_points].reshape(-1, s)
        counts.append(np.sum(np.any(reshaped, axis=1)))
    if len(counts) < 2: return 1.0
    poly = np.polyfit(np.log(1/scales), np.log(counts), 1)
    return poly[0]

def logistic_map(r: float, x0: float, n: int):
    x = np.zeros(n); x[0] = x0
    for i in range(n - 1): x[i+1] = r * x[i] * (1 - x[i])
    return x
