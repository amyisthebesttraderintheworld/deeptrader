"""
indicators.py  ─  Technical Analysis Functions (Vectorized with NumPy)
"""

import numpy as np
from scipy.signal import lfilter

def rsi(prices, n=14):
    """
    Vectorized RSI calculation for the entire array using scipy.signal.lfilter for smoothing.
    """
    prices = np.asarray(prices)
    if len(prices) <= n:
        return np.full_like(prices, 50)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(len(prices), dtype=float)
    avg_loss = np.zeros(len(prices), dtype=float)
    
    avg_gain[n] = np.mean(gains[:n])
    avg_loss[n] = np.mean(losses[:n])
    
    # Wilder's smoothing using lfilter: y[i] = y[i-1] * (1 - alpha) + x[i] * alpha
    # Where alpha = 1/n. Phemex/TradingView use this formula.
    alpha = 1 / n
    b = [alpha]
    a = [1, -(1 - alpha)]
    
    avg_gain[n:] = lfilter(b, a, gains[n-1:], zi=[avg_gain[n] * (1-alpha)])[0]
    avg_loss[n:] = lfilter(b, a, losses[n-1:], zi=[avg_loss[n] * (1-alpha)])[0]
        
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_vals = 100 - (100 / (1 + rs))
    rsi_vals[:n] = 50 # padding
    return rsi_vals

def bb(prices, n=20):
    """
    Vectorized Bollinger Bands calculation using rolling window views.
    """
    prices = np.asarray(prices)
    if len(prices) < n:
        p = np.full_like(prices, np.nan)
        return p, p, p
    
    # Efficient rolling mean
    sma = np.full_like(prices, np.nan)
    weights = np.ones(n) / n
    sma[n-1:] = np.convolve(prices, weights, mode='valid')
    
    # Efficient rolling std using E[X^2] - (E[X])^2
    sq_prices = prices**2
    sum_sq = np.convolve(sq_prices, np.ones(n), mode='valid')
    sum_p = np.convolve(prices, np.ones(n), mode='valid')
    
    var = (sum_sq / n) - (sum_p / n)**2
    std = np.sqrt(np.maximum(var, 0))
    
    std_padded = np.full_like(prices, np.nan)
    std_padded[n-1:] = std
        
    return sma - 2 * std_padded, sma, sma + 2 * std_padded

def ema(prices, n):
    """
    Vectorized EMA calculation using lfilter.
    """
    prices = np.asarray(prices)
    if len(prices) < n:
        return np.full_like(prices, np.nan)
    
    alpha = 2 / (n + 1)
    ema_vals = np.zeros(len(prices), dtype=float)
    initial_mean = np.mean(prices[:n])
    
    b = [alpha]
    a = [1, -(1 - alpha)]
    
    ema_vals[n-1:] = lfilter(b, a, prices[n-1:], zi=[initial_mean * (1-alpha)])[0]
    ema_vals[:n-1] = np.nan
    return ema_vals

def macd(prices, fast=12, slow=26, sig=9):
    """
    Vectorized MACD calculation.
    """
    prices = np.asarray(prices)
    if len(prices) < slow + sig:
        z = np.zeros_like(prices)
        return z, z, z
    
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = np.full_like(macd_line, np.nan)
    
    valid_macd = macd_line[slow-1:]
    signal_line[slow+sig-2:] = ema(valid_macd, sig)[sig-1:]
        
    macd_hist = macd_line - signal_line
    return np.nan_to_num(macd_line), np.nan_to_num(signal_line), np.nan_to_num(macd_hist)

def atr(highs, lows, closes, n=14):
    """
    Vectorized ATR calculation using lfilter.
    """
    highs, lows, closes = np.asarray(highs), np.asarray(lows), np.asarray(closes)
    if len(closes) <= n:
        return np.zeros_like(closes)
    
    tr = np.zeros_like(closes)
    tr[1:] = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(np.abs(highs[1:] - closes[:-1]), 
                                   np.abs(lows[1:] - closes[:-1])))
    
    atr_vals = np.zeros_like(closes)
    initial_atr = np.mean(tr[1:n+1])
    
    alpha = 1 / n
    b = [alpha]
    a = [1, -(1 - alpha)]
    
    atr_vals[n:] = lfilter(b, a, tr[n:], zi=[initial_atr * (1-alpha)])[0]
    return atr_vals

def adx(highs, lows, closes, n=14):
    """
    Vectorized ADX calculation using lfilter and optimized smoothing.
    """
    highs, lows, closes = np.asarray(highs), np.asarray(lows), np.asarray(closes)
    if len(closes) <= n * 2:
        return np.zeros_like(closes)
    
    tr = np.zeros_like(closes)
    tr[1:] = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(np.abs(highs[1:] - closes[:-1]), 
                                   np.abs(lows[1:] - closes[:-1])))
    
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    
    pdm = np.zeros_like(closes)
    ndm = np.zeros_like(closes)
    
    pdm[1:] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    ndm[1:] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    def wilders_sum(data, n):
        res = np.zeros(len(data), dtype=float)
        init_sum = np.sum(data[1:n+1])
        res[n] = init_sum
        
        decay = (n - 1) / n
        b = [1.0]
        a = [1.0, -decay]
        
        res[n:] = lfilter(b, a, data[n:], zi=[init_sum * decay])[0]
        return res
    
    atr_s = wilders_sum(tr, n)
    pdm_s = wilders_sum(pdm, n)
    ndm_s = wilders_sum(ndm, n)
    
    pdi = 100 * np.divide(pdm_s, atr_s, out=np.zeros_like(pdm_s), where=atr_s != 0)
    ndi = 100 * np.divide(ndm_s, atr_s, out=np.zeros_like(ndm_s), where=atr_s != 0)
    
    dx = 100 * np.divide(np.abs(pdi - ndi), (pdi + ndi), out=np.zeros_like(pdi), where=(pdi + ndi) != 0)
    
    adx_vals = np.zeros_like(dx)
    init_adx = np.mean(dx[n:2*n])
    
    alpha = 1 / n
    b = [alpha]
    a = [1, -(1 - alpha)]
    
    adx_vals[2*n-1:] = lfilter(b, a, dx[2*n-1:], zi=[init_adx * (1-alpha)])[0]
    return adx_vals

def stochastic(highs, lows, closes, n=14):
    """
    Vectorized Stochastic Oscillator using sliding_window_view.
    """
    highs, lows, closes = np.asarray(highs), np.asarray(lows), np.asarray(closes)
    if len(closes) < n:
        return np.full_like(closes, 50)
    
    from numpy.lib.stride_tricks import sliding_window_view
    
    h_win = sliding_window_view(highs, window_shape=n)
    l_win = sliding_window_view(lows, window_shape=n)
    
    rh = np.max(h_win, axis=1)
    rl = np.min(l_win, axis=1)
    
    stoch = np.zeros(len(closes), dtype=float)
    denom = rh - rl
    stoch[n-1:] = np.where(denom != 0, 100 * (closes[n-1:] - rl) / denom, 50.0)
    stoch[:n-1] = 50.0
            
    return stoch

def z_score(prices, n=20):
    """
    Vectorized Z-score calculation.
    """
    prices = np.asarray(prices)
    if len(prices) < n:
        return np.zeros(len(prices), dtype=float)
    
    sma = np.full(len(prices), np.nan, dtype=float)
    weights = np.ones(n) / n
    sma[n-1:] = np.convolve(prices, weights, mode='valid')
    
    sq_prices = prices**2
    sum_sq = np.convolve(sq_prices, np.ones(n), mode='valid')
    sum_p = np.convolve(prices, np.ones(n), mode='valid')
    
    var = (sum_sq / n) - (sum_p / n)**2
    std = np.sqrt(np.maximum(var, 0))
    
    std_padded = np.zeros(len(prices), dtype=float)
    std_padded[n-1:] = std
    
    z = np.divide(prices - sma, std_padded, out=np.zeros(len(prices), dtype=float), where=std_padded != 0)
    return np.nan_to_num(z)
