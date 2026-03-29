def cross_sectional_score_ranking(feature_matrices, asset_names=None, top_pct=0.2, bottom_pct=0.2):
    """
    Minimal fix: Z-score each feature across assets, average, rank, and select long/short.
    feature_matrices: list of feature matrices (one per asset, shape [n_bars, n_features])
    asset_names: list of asset names (optional)
    Returns: dict {asset: {'score': float, 'rank': int, 'side': 'long'|'short'|'neutral'}}
    """
    # Use the last row (most recent) of each asset's feature matrix
    latest_features = np.array([fm[-1] for fm in feature_matrices])
    # Z-score normalize each feature across assets
    means = np.mean(latest_features, axis=0)
    stds = np.std(latest_features, axis=0) + 1e-9
    z = (latest_features - means) / stds
    # Average normalized features for each asset
    scores = np.mean(z, axis=1)
    # Statistical neutral band: assets within ±0.5 std of mean are 'neutral'
    mean_score = np.mean(scores)
    std_score = np.std(scores) + 1e-9
    upper_neutral = mean_score + 0.5 * std_score
    lower_neutral = mean_score - 0.5 * std_score
    # Rank assets
    ranks = scores.argsort()[::-1]  # descending
    n = len(scores)
    result = {}
    for i, idx in enumerate(ranks):
        asset = asset_names[idx] if asset_names else idx
        score_val = float(scores[idx])
        if score_val > upper_neutral:
            side = 'long'
        elif score_val < lower_neutral:
            side = 'short'
        else:
            side = 'neutral'
        result[asset] = {'score': score_val, 'rank': int(i+1), 'side': side}
    return result
"""
scoring.py  ─  Signal Calculation & Candle Analysis (Fully Vectorized)
"""

import numpy as np
import time
from advanced_scanner.indicators import rsi, bb, macd, stochastic, z_score, ema, atr, adx
from advanced_scanner.chaos_and_complexity import rolling_hurst, rolling_entropy
from advanced_scanner.signal_processing import butter_lowpass_filter
from advanced_scanner.time_series_analysis import rolling_autocorr, moving_average_volatility, rolling_spectral_density_peak
from advanced_scanner.stats_models import identify_outliers
from advanced_scanner.machine_learning_math import sigmoid

from advanced_scanner.linear_algebra import principal_component_analysis
from advanced_scanner.machine_learning_math import sigmoid

def validate_signal_independence(X, feature_names):
    """
    Calculates the correlation matrix of features and returns the average absolute correlation.
    """
    if X.shape[1] < 2: return 0.0
    corr = np.corrcoef(X, rowvar=False)
    # Average absolute correlation of off-diagonal elements
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    avg_corr = np.mean(np.abs(corr[mask]))
    return avg_corr, corr

def apply_portfolio_risk_management(position_sizes, capital, max_portfolio_risk=0.2, max_single_asset_risk=0.05):
    """
    Adjusts position sizes to ensure portfolio and per-asset risk limits are not breached.
    - position_sizes: dict {symbol: size}
    - capital: total capital
    - max_portfolio_risk: max fraction of capital at risk across all positions (e.g., 0.2 = 20%)
    - max_single_asset_risk: max fraction of capital at risk per asset (e.g., 0.05 = 5%)
    Returns: dict {symbol: adjusted_size}
    """
    # Cap each asset
    capped = {sym: min(size, capital * max_single_asset_risk) for sym, size in position_sizes.items()}
    total = sum(capped.values())
    max_total = capital * max_portfolio_risk
    if total > max_total and total > 0:
        # Scale down all positions proportionally
        scale = max_total / total
        capped = {sym: size * scale for sym, size in capped.items()}
    return capped

def calculate_position_size(signal_score, capital, asset_volatility, target_risk=0.01, min_size=1.0, max_size=None):
    """
    Position size is determined by target risk per trade and asset volatility.
    - signal_score: The trade signal (directional, but only sign is used for direction)
    - capital: Total capital available
    - asset_volatility: Realized or estimated volatility (e.g., ATR or stddev of returns)
    - target_risk: Fraction of capital to risk per trade (e.g., 0.01 = 1%)
    - min_size: Minimum position size
    - max_size: Maximum position size (optional)
    Returns: position size (float, in capital units)
    """
    if asset_volatility <= 0:
        return min_size
    # Risk per trade (dollar risk)
    risk_dollars = capital * target_risk
    # Position size = risk_dollars / (volatility * leverage)
    leverage = score_to_leverage(signal_score)
    pos_size = risk_dollars / (asset_volatility * leverage)
    # Cap position size to available capital and max_size
    pos_size = max(min_size, pos_size)
    if max_size is not None:
        pos_size = min(pos_size, max_size)
    pos_size = min(pos_size, capital)
    return pos_size

def extract_features(ohlcv, funding):
    """
    Extracts and normalizes features for signal generation.
    """
    n = len(ohlcv)
    if n < 60: return np.zeros((n, 9)), np.zeros(n) # Return empty features and lookback mask
    
    o, h, l, c, v = ohlcv[:, 3], ohlcv[:, 4], ohlcv[:, 5], ohlcv[:, 6], ohlcv[:, 7]
    
    # --- Feature Extraction ---
    rsi_vals = rsi(c)
    bl, bm, bh = bb(c)
    macd_line, signal_line, macd_hist = macd(c)
    z_vals = z_score(c)
    atr_vals = atr(h, l, c)
    h_exps = rolling_hurst(c, 50)
    ents = rolling_entropy(c, 50)
    ac1s = rolling_autocorr(c, 20, lag=1)
    
    # BB Percentage
    rng_bb = bh - bl
    pct_bb = np.where(rng_bb != 0, (c - bl) / rng_bb, 0.5)
    
    # Volatility and Momentum
    vol_r = np.divide(atr_vals, c, out=np.zeros_like(atr_vals), where=c != 0)
    vel = np.zeros_like(c)
    vel[1:] = (c[1:] - c[:-1]) / np.where(c[:-1] != 0, c[:-1], 1)
    
    # Funding
    f_val = np.full(n, funding)
    
    # --- Feature Matrix Construction ---
    features = []
    features.append((rsi_vals - 50) / 50.0) # 1. RSI
    features.append((pct_bb - 0.5) * 2.0)   # 2. BB Pct
    features.append(np.tanh(z_vals / 2.0))  # 3. Z-Score
    features.append(np.tanh(macd_hist / (atr_vals + 1e-9))) # 4. MACD Hist
    features.append(h_exps - 0.5)           # 5. Hurst
    features.append(ac1s)                  # 6. AC1
    features.append(np.tanh(f_val * 1000)) # 7. Funding
    features.append(1.0 - np.tanh(ents / 3.0)) # 8. Entropy
    features.append(np.tanh(vel * 100))    # 9. Velocity

    X = np.column_stack(features)
    X = np.nan_to_num(X)
    return X, ents, vol_r, h_exps

def derive_optimal_weights(X, target_returns, l2_reg=1.0, halflife=60):
    """
    Exponentially-weighted Ridge Regression.
    Recent bars contribute more than old ones (halflife in bars).
    X[i] predicts return at bar i+1 — strict no-lookahead alignment.
    """
    X = np.asarray(X)
    target_returns = np.asarray(target_returns)

    X_aligned = X[:-1]
    y_aligned  = target_returns[1:]

    mask = y_aligned != 0
    if np.sum(mask) < X.shape[1] * 2:
        return np.zeros(X.shape[1])

    X_f, y_f = X_aligned[mask], y_aligned[mask]

    # Exponential sample weights — more recent = higher weight
    n = len(y_f)
    decay = np.exp(-np.log(2) / halflife * np.arange(n - 1, -1, -1))
    decay /= decay.sum()
    W = np.diag(decay)

    X_mean = np.mean(X_f, axis=0)
    X_std  = np.std(X_f, axis=0) + 1e-9
    X_norm = (X_f - X_mean) / X_std

    X_aug = np.column_stack([X_norm, np.ones(n)])

    K = X_aug.shape[1]
    reg = l2_reg * np.eye(K)
    reg[-1, -1] = 0  # don't regularize intercept

    try:
        # Weighted ridge: (XᵀWX + λI)⁻¹ XᵀWy
        XtW  = X_aug.T @ W
        w_aug = np.linalg.solve(XtW @ X_aug + reg, XtW @ y_f)
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1])

    return w_aug[:-1] / X_std


def calculate_all_scores(ohlcv, funding, weights=None):
    n = len(ohlcv)
    if n < 60: return np.zeros(n)
    
    X, ents, vol_r, h_exps = extract_features(ohlcv, funding)
    
    # --- Signal Independence Validation ---
    feature_names = ["RSI", "BB Pct", "Z-Score", "MACD Hist", "Hurst", "AC1", "Funding", "Entropy", "Velocity"]
    
    if weights is None:
        avg_hurst = float(np.nanmean(h_exps[60:])) + 0.5 if len(h_exps) > 60 else 0.5
        avg_vol   = float(np.nanmean(vol_r[60:])) if len(vol_r) > 60 else 0.0

        if avg_hurst > 0.55:
            weights = np.array([-0.10, -0.10, -0.10,  0.20,  0.15,  0.15, -0.10,  0.05,  0.15])
        elif avg_hurst < 0.45:
            weights = np.array([-0.22, -0.22, -0.18,  0.08,  0.04,  0.04, -0.10,  0.05,  0.05])
        else:
            weights = np.array([-0.15, -0.15, -0.15,  0.15,  0.10,  0.10, -0.10,  0.05,  0.05])

        if avg_vol > 0.04:
            weights *= 0.75
    
    # Validation: Ensure weights are normalized by norm to prevent score explosion
    w_norm = weights / (np.sum(np.abs(weights)) + 1e-9)
    
    # --- Robust Signal Normalization and Nonlinear Squashing ---
    raw_scores = X @ w_norm
    # Z-score normalization (across assets or time)
    mean = np.mean(raw_scores)
    std = np.std(raw_scores) + 1e-9
    z = (raw_scores - mean) / std
    # Nonlinear squashing (tanh)
    signal = np.tanh(z)
    # Scale to weights (sum(abs) = 1)
    if np.sum(np.abs(signal)) > 0:
        w = signal / (np.sum(np.abs(signal)) + 1e-9)
    else:
        w = signal
    # Apply leverage cap (portfolio-level)
    target_leverage = 2.0  # Configurable
    w = w * target_leverage
    # Hard position cap per asset
    max_pos = 0.10  # 10% NAV per asset (configurable)
    w = np.clip(w, -max_pos, max_pos)
    # Convert to score scale for compatibility (optional)
    scores = (w * 100).astype(int)
    # Apply penalties for extreme noise or volatility (Dynamic Risk Adjustment)
    penalty = np.where(ents > 2.8, 20, 0)
    penalty += np.where(vol_r > 0.05, 15, 0)
    scores = np.clip(scores - penalty, -int(max_pos*100), int(max_pos*100))
    scores[:60] = 0 # Lookback padding
    return scores


def score_rows(rows, funding):
    if not isinstance(rows, np.ndarray):
        try: rows = np.array(rows, dtype=float)
        except Exception as e:
            return 0
    scores = calculate_all_scores(rows, funding)
    if len(scores) == 0:
        return 0.0
    # --- Continuous position sizing (no dead zone) ---
    last_score = scores[-1]
    # Risk-adjusted: scale by realized volatility (20-bar stddev)
    returns = np.diff(rows[:, 6]) / np.where(rows[:-1, 6] != 0, rows[:-1, 6], 1e-9)
    vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
    if vol == 0:
        vol = 1e-6
    # Z-score normalization (robust to outliers)
    z = (last_score - np.mean(scores)) / (np.std(scores) + 1e-9)
    # Continuous position sizing: clip(z / threshold, -1, 1)
    threshold = 1.0  # Configurable, in z-score units
    position = np.clip(z / threshold, -1, 1)
    # Optional: minimum exposure floor (epsilon)
    epsilon = 0.05  # 5% min position if nonzero signal
    if abs(position) < epsilon and abs(z) > 0:
        position = np.sign(z) * epsilon
    # Volatility targeting overlay (example: scale to target vol)
    target_vol = 0.02  # 2% daily vol target (configurable)
    position *= target_vol / vol
    # Drawdown/risk overlays can be added here
    return position

def score_to_leverage(sc):

    a = abs(sc)

    if a < 15: return 1

    norm = (a - 15) / 85.0

    s_val = sigmoid((norm - 0.5) * 6) 

    return max(1, min(20, int(1 + s_val * 19)))



def candle_strength(o, h, l, c):

    """

    Returns a score based on the candle's price action.

    Matching the test: (10.5-10)/10 * 200 = 10

    """

    if o == 0: return 0

    return (c - o) / o * 200



def engulf(po, pc, lo, lc):

    """

    po: prev_open, pc: prev_close

    lo: last_open, lc: last_close

    """

    # Bullish Engulfing: previous was bearish, current is bullish and covers previous range

    if pc < po and lc > lo and lo < pc and lc > po:

        return 10

    # Bearish Engulfing: previous was bullish, current is bearish and covers previous range

    if pc > po and lc < lo and lo > pc and lc < po:

        return -10

    return 0

def _extract_features_and_target_returns(ohlcv, funding):
    """
    Extracts features (X) and calculates target returns for derive_optimal_weights.
    """
    n = len(ohlcv)
    if n < 60:
        return None, None

    # Extract features using the existing function
    X, _, _, _ = extract_features(ohlcv, funding)

    # Calculate open-to-close return for bar k
    target_returns = (ohlcv[:, 6] - ohlcv[:, 3]) / np.where(ohlcv[:, 3] != 0, ohlcv[:, 3], 1e-9)

    return X, target_returns
