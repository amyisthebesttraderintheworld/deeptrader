"""
optimization.py  ─  Constrained/Unconstrained, Gradient Methods, and Hyperparameter Optimization
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Callable, List, Tuple, Union, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from advanced_scanner.utils import c
from advanced_scanner.config import MAX_WORKERS, BOLD, BRIGHT_WHITE, DIM, BRIGHT_GREEN

def gradient_descent(f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, lr: float = 0.01, tol: float = 1e-6, max_iter: int = 1000):
    """Simple gradient descent optimizer."""
    x = x0.copy()
    for _ in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - lr * g
    return x

def constrained_optimization(f: Callable, x0: np.ndarray, bounds: List[Tuple] = None, constraints: List[Dict] = None):
    """Wraps scipy's minimize for constrained optimization."""
    # constraints is a list of dicts like: {'type': 'eq', 'fun': lambda x: ...}
    res = minimize(f, x0, bounds=bounds, constraints=constraints)
    return res.x, res.fun

def mean_variance_portfolio(returns: np.ndarray, cov_matrix: np.ndarray, target_return: float = 0.05):
    """
    Optimizes portfolio for a given target return while minimizing risk (variance).
    returns: (N_assets,)
    cov_matrix: (N_assets, N_assets)
    """
    n_assets = len(returns)
    x0 = np.ones(n_assets) / n_assets
    
    # Minimize: x^T * Cov * x
    def objective(x):
        return x.T @ cov_matrix @ x
    
    # Constraint 1: Sum of weights = 1
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: x.T @ returns - target_return}
    ]
    
    # Weight bounds [0, 1]
    bounds = [(0, 1) for _ in range(n_assets)]
    
    res = minimize(objective, x0, bounds=bounds, constraints=constraints)
    return res.x, res.fun

def sharpe_ratio_portfolio(returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float = 0.02):
    """
    Optimizes portfolio to maximize Sharpe Ratio.
    """
    n_assets = len(returns)
    x0 = np.ones(n_assets) / n_assets
    
    # Maximize: (R_p - R_f) / sigma_p  => Minimize: -(R_p - R_f) / sigma_p
    def objective(x):
        p_ret = x.T @ returns
        p_vol = np.sqrt(x.T @ cov_matrix @ x)
        return -(p_ret - risk_free_rate) / p_vol
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in range(n_assets)]
    
    res = minimize(objective, x0, bounds=bounds, constraints=constraints)
    return res.x, -res.fun # Return weights and max Sharpe

def _opt_worker(sym_rows_map, funding_map, vol_map, capital, risk, t, h):
    # Local import to avoid circular dependencies and ensure fresh state
    from advanced_scanner.backtest import backtest_portfolio
    from advanced_scanner.stats import portfolio_stats
    
    all_trades, equity_curve, final_cap, concurrent_log = backtest_portfolio(
        sym_rows_map, funding_map, vol_map, threshold=t, hold_bars=h, 
        starting_capital=capital, risk_per_trade=risk
    )
    st = portfolio_stats(all_trades, equity_curve, final_cap, concurrent_log, capital)
    fit = st.get("fitness", 0) # Use the new fitness score
    return (t, h, fit, st.get("total_ret_pct", 0), st.get("n_trades", 0))

def optimize_parameters(sym_rows_map, funding_map, vol_map, capital, risk):
    print("\n" + c("  ◆ RUNNING HYPER-PARAMETER OPTIMIZATION...", BOLD + BRIGHT_WHITE))
    print(c("  Optimizing Threshold (15-45) and Hold Bars (2-12) for Fitness Score (min 30 trades)", DIM))
    
    best_fitness = -float("inf"); best_params = (0, 0); results = []
    
    thresholds = [15, 20, 25, 30, 35, 40, 45]
    holds = [2, 4, 6, 8, 10, 12]
    
    total = len(thresholds) * len(holds); count = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_opt_worker, sym_rows_map, funding_map, vol_map, capital, risk, t, h) for t in thresholds for h in holds]
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            t, h, fit, ret, nt = res
            if fit > best_fitness:
                best_fitness = fit; best_params = (t, h)
            count += 1
            import sys
            sys.stdout.write(f"\r  Progress: {count}/{total} simulations...")
            sys.stdout.flush()
    
    print("\n\n  " + c("OPTIMIZATION RESULTS", BOLD + BRIGHT_WHITE))
    print(c(f"  {'THRESH':<8} {'HOLD':<6} {'FITNESS':<10} {'RETURN':<10} {'TRADES':<7}", BOLD))
    for t, h, fit, ret, nt in sorted(results, key=lambda x: -x[2])[:10]:
        print(f"  {t:<8} {h:<6} {fit: <10.2f} {ret: >+8.2f}% {nt:<7}")
        
    print(f"\n  {c('BEST PARAMS', BRIGHT_GREEN)}: Threshold={best_params[0]}, Hold Bars={best_params[1]} (Fitness: {best_fitness:.2f})")
    return best_params

def _fuzz_worker(sym_rows_map, funding_map, vol_map, capital):
    import random, os, time
    from advanced_scanner.backtest import backtest_portfolio
    from advanced_scanner.stats import portfolio_stats
    random.seed(os.getpid() + int(time.time() * 1000) % 1234567)
    t = random.randint(15, 50); h = random.randint(1, 12); c_down = random.randint(1, 5)
    r = random.uniform(0.005, 0.025); af = random.choice([True, False])
    sl_m = random.uniform(1.0, 3.5); tp_m = random.uniform(2.0, 6.0)
    all_trades, equity_curve, final_cap, concurrent_log = backtest_portfolio(
        sym_rows_map, funding_map, vol_map, 
        threshold=t, hold_bars=h, cooldown=c_down,
        starting_capital=capital, risk_per_trade=r, adx_filter=af,
        sl_mult=sl_m, tp_mult=tp_m
    )
    st = portfolio_stats(all_trades, equity_curve, final_cap, concurrent_log, capital)
    fit = st.get("fitness", 0)
    params = {"threshold": t, "hold_bars": h, "cooldown": c_down, "risk_per_trade": r, "adx_filter": af, "sl_mult": sl_m, "tp_mult": tp_m}
    return (params, fit, st.get("total_ret_pct", 0), st.get("max_dd_pct", 0), st.get("n_trades", 0))

def _mc_worker(sym_rows_map, funding_map, vol_map, capital, params):
    import random, os, time
    from advanced_scanner.backtest import backtest_portfolio
    from advanced_scanner.stats import portfolio_stats
    random.seed(os.getpid() + int(time.time() * 1000) % 1234567)
    all_trades, equity_curve, final_cap, concurrent_log = backtest_portfolio(
        sym_rows_map, funding_map, vol_map, **params, starting_capital=capital
    )
    st = portfolio_stats(all_trades, equity_curve, final_cap, concurrent_log, capital)
    return st.get("fitness", 0)

def fuzz_sweep(sym_rows_map, funding_map, vol_map, capital, iterations=40, initial_threshold=None):
    from advanced_scanner.config import MAX_WORKERS, BRIGHT_CYAN, BRIGHT_RED, BRIGHT_GREEN, BOLD, BRIGHT_WHITE, DIM
    import sys
    print("\n" + c("  ◆ RUNNING STRATEGY PARAMETER SWEEP (FUZZING)...", BOLD + BRIGHT_WHITE))
    print(c(f"  Randomly sampling {iterations} parameter combinations to find robust outcomes", DIM))
    
    best_fitness = -float("inf"); best_params = {}; results = []
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_fuzz_worker, sym_rows_map, funding_map, vol_map, capital) for _ in range(iterations)]
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            results.append(res)
            fit = res[1]
            if fit > best_fitness:
                best_fitness = fit; best_params = res[0]
            sys.stdout.write(f"\r  Progress: {i+1}/{iterations} iterations...")
            sys.stdout.flush()
    
    print("\n\n  " + c("SWEEP RESULTS (TOP 10 BY FITNESS)", BOLD + BRIGHT_WHITE))
    print(c(f"  {'THRESH':<7} {'HOLD':<5} {'RISK%':<6} {'SL/TP':<10} {'ADX':<5} {'FITNESS':<8} {'RET%':<9} {'TRADES':<7}", BOLD))
    for p, fit, ret, dd, nt in sorted(results, key=lambda x: -x[1])[:10]:
        sltp = f"{p['sl_mult']:.1f}/{p['tp_mult']:.1f}"
        print(f"  {p['threshold']:<7} {p['hold_bars']:<5} {p['risk_per_trade']*100:<6.1f} {sltp:<10} {str(p['adx_filter']):<5} {fit: <8.2f} {ret: >+8.2f}% {nt:<7}")
        
    print(f"\n  {c('BEST PARAMS FOUND', BRIGHT_GREEN)}: {best_params} (Fitness: {best_fitness:.2f})")
    
    # Monte Carlo Robustness Check for Best Params
    print("\n" + c("  ◆ RUNNING MONTE CARLO ROBUSTNESS CHECK (10 runs)...", BOLD + BRIGHT_WHITE))
    mc_fits = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_mc_worker, sym_rows_map, funding_map, vol_map, capital, best_params) for _ in range(10)]
        for i, future in enumerate(as_completed(futures)):
            mc_fits.append(future.result())
            sys.stdout.write(f"\r  MC Progress: {i+1}/10...")
            sys.stdout.flush()
    
    avg_mc = sum(mc_fits)/len(mc_fits)
    print(f"\n  MC Average Fitness: {avg_mc:.2f} (Original: {best_fitness:.2f})")
    if avg_mc < best_fitness * 0.7:
        print(f"  {c('WARNING', BRIGHT_RED)}: High variance detected in Monte Carlo runs. Results might be lucky.")
    else:
        print(f"  {c('CONFIRMED', BRIGHT_GREEN)}: Strategy performance is stable across random fills.")

    return best_params