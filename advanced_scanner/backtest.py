"""
backtest.py  ─  Portfolio Backtest Engine (Optimized)
"""

import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from advanced_scanner.config import FUNDING_INTERVAL, BRIGHT_RED, MAX_WORKERS
from advanced_scanner.scoring import calculate_all_scores, score_to_leverage, derive_optimal_weights, _extract_features_and_target_returns, calculate_position_size, apply_portfolio_risk_management
from advanced_scanner.indicators import adx, atr
from advanced_scanner.trade_objects import Order, Trade
from advanced_scanner.utils import c

def walk_forward_optimization(sym_rows_map, funding_map, vol_map, capital, risk, windows=4):
    from advanced_scanner.stats import ts_to_str, portfolio_stats, print_portfolio_report, usd_color, pct_color
    from advanced_scanner.optimization import optimize_parameters
    from advanced_scanner.utils import c
    from advanced_scanner.config import BOLD, BRIGHT_CYAN, BRIGHT_WHITE, BRIGHT_MAGENTA

    print("\n" + c("═" * 100, BRIGHT_MAGENTA))
    print(c("  ◆ RUNNING WALK-FORWARD OPTIMIZATION (WFO)", BOLD + BRIGHT_WHITE))
    print(c("═" * 100, BRIGHT_MAGENTA))
    print(c(f"  Splitting data into {windows} temporal segments for rolling optimization.", "\033[2m"))
    
    all_ts = sorted(set(int(r[0]) for rs in sym_rows_map.values() for r in rs))
    if len(all_ts) < 400:
        print("\033[91m  [!] Insufficient data for WFO. Need at least 400 bars.\033[0m"); return
        
    n = len(all_ts)
    seg_size = n // windows
    overall_trades = []
    current_capital = capital
    equity_curve = [(all_ts[0], capital)]
    
    for i in range(windows - 1):
        train_start_ts = all_ts[0]
        train_end_ts = all_ts[(i + 1) * seg_size - 1]
        test_start_ts = all_ts[(i + 1) * seg_size]
        test_end_ts = all_ts[min((i + 2) * seg_size - 1, n - 1)]
        
        print(f"\n  {c(f'STEP {i+1}/{windows-1}', BOLD + BRIGHT_CYAN)}")
        print(f"  TRAIN: {ts_to_str(train_start_ts)} ➔ {ts_to_str(train_end_ts)}")
        print(f"  TEST:  {ts_to_str(test_start_ts)} ➔ {ts_to_str(test_end_ts)}")
        
        train_map = {}
        for s, rs in sym_rows_map.items():
            tr = [r for r in rs if train_start_ts <= int(r[0]) <= train_end_ts]
            if len(tr) > 60: train_map[s] = tr
        
        if not train_map: continue

        # --- Derive Optimal Weights for Training Window ---
        all_X_train = []
        all_y_train = []
        for s, rs in train_map.items():
            # Convert raw rows to ohlcv numpy array (similar to how prepare_sym does it)
            clean_rs = []
            for r in rs:
                try: clean_rs.append([float(x) for x in r])
                except: continue
            
            if not clean_rs: continue
            ohlcv_train = np.array(clean_rs, dtype=float)
            
            X_sym, y_sym = _extract_features_and_target_returns(ohlcv_train, funding_map.get(s, 0))
            if X_sym is not None and y_sym is not None:
                all_X_train.append(X_sym)
                all_y_train.append(y_sym)
        
        optimized_weights = np.zeros(9) # Default to zeros if no data
        if all_X_train and all_y_train:
            combined_X_train = np.vstack(all_X_train)
            combined_y_train = np.concatenate(all_y_train)
            optimized_weights = derive_optimal_weights(combined_X_train, combined_y_train)
        
        # --- Backtest on Test Window with Optimized Weights ---
        test_map = {}
        for s, rs in sym_rows_map.items():
            ts_rs = [r for r in rs if test_start_ts <= int(r[0]) <= test_end_ts]
            lb = [r for r in rs if int(r[0]) < test_start_ts][-60:]
            if ts_rs: test_map[s] = lb + ts_rs
            
        if not test_map: continue
        
        # We use fixed threshold and hold_bars for now, focusing on weight optimization
        # The original meta-commentary suggested separating feature engineering (weights)
        # from SL/TP (threshold/hold_bars)
        # For now, we use a default threshold and hold_bars, or potentially parameters
        # from config.py if we want to make them configurable for WFO.
        # For this step, I'm setting them to reasonable defaults/config values.
        # --- Separately optimise trade params on training window ---
        # Weights already derived above. Now find best threshold/hold_bars
        # independently so the two stages can't co-overfit.
        best_threshold, best_hold = 25, 4
        best_sharpe = -np.inf
        for thr in [20, 25, 30, 40]:
            for hb in [3, 4, 6, 8]:
                try:
                    _tr, _cv, _fc, _ = backtest_portfolio(
                        train_map, funding_map, vol_map,
                        threshold=thr, hold_bars=hb,
                        starting_capital=current_capital, risk_per_trade=risk,
                        feature_weights=optimized_weights
                    )
                    if len(_tr) > 5:
                        rets = np.array([t.pnl / t.margin for t in _tr if t.margin > 0])
                        sh = rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)
                        if sh > best_sharpe:
                            best_sharpe = sh
                            best_threshold, best_hold = thr, hb
                except Exception:
                    pass

        print(f"  Optimal trade params → threshold={best_threshold}, hold_bars={best_hold} (Sharpe={best_sharpe:.2f})")

        trades, curve, final_cap, _ = backtest_portfolio(
            test_map, funding_map, vol_map, threshold=best_threshold, hold_bars=best_hold,
            starting_capital=current_capital, risk_per_trade=risk,
            feature_weights=optimized_weights
        )
        
        overall_trades.extend(trades)
        current_capital = final_cap
        for ts_c, val in curve:
            if ts_c >= test_start_ts: equity_curve.append((ts_c, val))
            
        print(f"\n  {c('Segment Result', BOLD)}: {usd_color(final_cap - curve[0][1])} ({pct_color((final_cap - curve[0][1])/curve[0][1]*100)}) | Trades: {len(trades)}")

    st = portfolio_stats(overall_trades, equity_curve, current_capital, [0], capital)
    print_portfolio_report(st, overall_trades, equity_curve, [0], capital, risk, "WFO", "Dynamic", len(all_ts))
    return st

def backtest_portfolio(sym_rows_map, funding_map, vol_map, threshold=25, hold_bars=4, cooldown=2,
                       starting_capital=100.0, risk_per_trade=0.01, adx_filter=True,
                       sl_mult=2.0, tp_mult=3.5, feature_weights=None):
    MIN_LOOKBACK = 60
    
    # 1. Pre-calculate NumPy arrays and Scores
    sym_data = {}
    
    def prepare_sym(s, rs):
        try:
            clean_rs = []
            for r in rs:
                try: clean_rs.append([float(x) for x in r])
                except: continue
            
            if not clean_rs: return s, None
            
            ohlcv = np.array(clean_rs, dtype=float)
            scores = calculate_all_scores(ohlcv, funding_map.get(s, 0), weights=feature_weights)
            h, l, cl = ohlcv[:, 4], ohlcv[:, 5], ohlcv[:, 6]
            adx_vals, atr_vals = adx(h, l, cl), atr(h, l, cl)
            
            # Simple returns
            rets = np.zeros_like(cl)
            rets[1:] = (cl[1:] - cl[:-1]) / np.where(cl[:-1] != 0, cl[:-1], 1e-9)
            
            ts_to_idx = {int(r[0]): i for i, r in enumerate(clean_rs)}
            return s, {"ohlcv": ohlcv, "scores": scores, "adx": adx_vals, "atr": atr_vals, "returns": rets, "ts_to_idx": ts_to_idx}
        except Exception as e:
            return s, None

    # Parallelize pre-calculation for symbols
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(prepare_sym, s, rs) for s, rs in sym_rows_map.items()]
        for f in as_completed(futures):
            s, data = f.result()
            if data:
                sym_data[s] = data

    all_ts = sorted(set(ts for d in sym_data.values() for ts in d["ts_to_idx"]))
    open_trades, pending_orders = {}, []
    try:
        MIN_LOOKBACK = 60
        # 1. Pre-calculate NumPy arrays and Scores
        sym_data = {}
        def prepare_sym(s, rs):
            try:
                clean_rs = []
                for r in rs:
                    try: clean_rs.append([float(x) for x in r])
                    except: continue
                if not clean_rs: return s, None
                ohlcv = np.array(clean_rs, dtype=float)
                scores = calculate_all_scores(ohlcv, funding_map.get(s, 0), weights=feature_weights)
                h, l, cl = ohlcv[:, 4], ohlcv[:, 5], ohlcv[:, 6]
                adx_vals, atr_vals = adx(h, l, cl), atr(h, l, cl)
                # Simple returns
                rets = np.zeros_like(cl)
                rets[1:] = (cl[1:] - cl[:-1]) / np.where(cl[:-1] != 0, cl[:-1], 1e-9)
                ts_to_idx = {int(r[0]): i for i, r in enumerate(clean_rs)}
                return s, {"ohlcv": ohlcv, "scores": scores, "adx": adx_vals, "atr": atr_vals, "returns": rets, "ts_to_idx": ts_to_idx}
            except Exception as e:
                return s, None
        # Parallelize pre-calculation for symbols
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(prepare_sym, s, rs) for s, rs in sym_rows_map.items()]
            for f in as_completed(futures):
                s, data = f.result()
                if data:
                    sym_data[s] = data
        all_ts = sorted(set(ts for d in sym_data.values() for ts in d["ts_to_idx"]))
        open_trades, pending_orders = {}, []
        last_exit_bar = {}
        capital, all_trades, equity_curve, concurrent_log = starting_capital, [], [], []
        if not sym_data or not all_ts:
            # Return empty results if no data is available
            return all_trades, equity_curve, capital, concurrent_log
        for ts in all_ts:
            is_funding = (ts % FUNDING_INTERVAL == 0)
            # 1. EXITS
            to_close = []
            # ...existing code...
        # Final safeguard: always return a tuple
        return all_trades, equity_curve, capital, concurrent_log
    except Exception as e:
        import traceback
        print("[ERROR] Exception in backtest_portfolio:", e)
        traceback.print_exc()
        return [], [], starting_capital, []
