"""
time_series_analysis.py  ─  ARIMA, GARCH, Spectral Analysis, Fourier
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
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

def autocorrelation(series: np.ndarray, lag: int) -> float:
    if lag == 0: return 1.0
    mean = np.mean(series)
    denom = np.sum((series - mean)**2)
    if denom == 0: return 0.0
    num = np.sum((series[:-lag] - mean) * (series[lag:] - mean))
    return num / denom

def partial_autocorrelation(series: np.ndarray, lag: int) -> float:
    """
    Calculates the partial autocorrelation for a given lag.
    For lag k, it's the last coefficient of the AR(k) process.
    """
    if lag == 0: return 1.0
    n = len(series)
    if n <= lag: return 0.0
    
    # Solve Y = Xw where Y is series[lag:] and X is lagged versions
    # We use a simple OLS approach for PACF
    Y = series[lag:]
    X = np.zeros((len(Y), lag))
    for i in range(lag):
        X[:, i] = series[lag - 1 - i: -1 - i]
    
    # Center Y and X
    Y_m = Y - np.mean(Y)
    X_m = X - np.mean(X, axis=0)
    
    try:
        # PACF(lag) is the last coefficient in the AR(lag) model
        w, _, _, _ = np.linalg.lstsq(X_m, Y_m, rcond=None)
        return w[-1]
    except:
        return 0.0

def rolling_autocorr(series: np.ndarray, window: int = 20, lag: int = 1):
    """Fully vectorized rolling autocorrelation using fast_rolling_sum."""
    n = len(series)
    if n < window + lag: return np.zeros(n)
    
    x = series
    x_lag = np.zeros(n)
    x_lag[lag:] = series[:-lag]
    
    # We only care about windows where both x and x_lag are present
    # The original logic used v[lag:] and v[:-lag] where v has length window.
    # So we use a "window effective" sum.
    # Actually, let's just use the full window sum of (x * x_lag) where x_lag is valid.
    
    # Define a helper for rolling stats on a window of size 'window' 
    # but only considering the last 'window - lag' elements.
    
    # Create a masked version of series
    x_m = x.copy(); x_m[:lag] = 0 # Not useful for any window ending before 'lag'
    
    # E[X], E[Y], E[X^2], E[Y^2], E[XY]
    # For each window [i-window+1 : i+1], we only use j in [i-window+1+lag : i+1]
    # This is equivalent to rolling sum of size (window-lag)
    w_eff = window - lag
    
    # Correct approach:
    # sum_xy[i] = sum_{j=i-w_eff+1}^{i} (x[j] * x[j-lag])
    s_x = fast_rolling_sum(x, w_eff)
    s_y = fast_rolling_sum(x_lag, w_eff)
    s_x2 = fast_rolling_sum(x**2, w_eff)
    s_y2 = fast_rolling_sum(x_lag**2, w_eff)
    s_xy = fast_rolling_sum(x * x_lag, w_eff)
    
    mean_x = s_x / w_eff
    mean_y = s_y / w_eff
    
    cov_xy = (s_xy / w_eff) - (mean_x * mean_y)
    var_x = (s_x2 / w_eff) - (mean_x**2)
    var_y = (s_y2 / w_eff) - (mean_y**2)
    
    corr = cov_xy / np.sqrt(np.maximum(var_x * var_y, 1e-12))
    
    # Since we used window size w_eff, the results are shifted.
    # We want them to align with the end of the original 'window'.
    res = np.zeros(n)
    res[window-1:] = corr[window-1:] # Aligning
    return res

def spectral_density(series: np.ndarray, fs: float = 1.0):
    n = len(series)
    yf = fft(series)
    xf = fftfreq(n, 1.0/fs)[:n//2]
    psd = 2.0/n * np.abs(yf[0:n//2])**2
    return xf, psd

def rolling_spectral_density_peak(series: np.ndarray, window: int = 50, fs: float = 1.0):
    n = len(series)
    if n < window: return np.zeros(n)
    f, t, Sxx = spectrogram(series, fs=fs, nperseg=window, noverlap=window-1)
    peaks = np.max(Sxx[1:, :], axis=0) if Sxx.shape[0] > 1 else np.zeros(Sxx.shape[1])
    res = np.zeros(n)
    res[window-1:] = peaks[:n - window + 1]
    return res

def fft_analysis(series: np.ndarray, fs: float = 1.0):
    xf, psd = spectral_density(series, fs)
    peaks = np.argsort(psd)[-5:][::-1]
    return xf[peaks], psd[peaks]

def stationary_test(series: np.ndarray):
    split = len(series) // 2
    m1, m2 = np.mean(series[:split]), np.mean(series[split:])
    v1, v2 = np.var(series[:split]), np.var(series[split:])
    return (m1, m2), (v1, v2)

def moving_average_volatility(series: np.ndarray, window: int = 20):

    returns = np.diff(np.log(series))

    nr = len(returns)

    if nr < window: return np.zeros(nr)

    sq_returns = returns**2

    s_sq = fast_rolling_sum(sq_returns, window)

    s_r = fast_rolling_sum(returns, window)

    var = (s_sq / window) - (s_r / window)**2

    vol = np.sqrt(np.maximum(var, 0))

    return vol


