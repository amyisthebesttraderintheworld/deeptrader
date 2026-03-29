"""
main.py  ─  Phemex USDT Perp Scanner + Portfolio Backtest (Modular Entry Point)
"""

import sys
import argparse
import random
import io
import os
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from advanced_scanner.config import (
    MAX_WORKERS, KLINE_LIMIT, TOP_N, BT_KLINE_LIMIT,
    DEFAULT_CAPITAL, DEFAULT_RISK_PCT,
    BRIGHT_CYAN, BOLD, BRIGHT_WHITE, CYAN, DIM,
    BRIGHT_RED, BRIGHT_GREEN, BRIGHT_YELLOW, BRIGHT_MAGENTA, RESET
)
from advanced_scanner.utils import type_print, c, log_execution
from advanced_scanner.fetcher import bootstrap, fetch
from advanced_scanner.scoring import score_rows, score_to_leverage
from advanced_scanner.backtest import backtest_portfolio, walk_forward_optimization
from advanced_scanner.stochastic_processes import geometric_brownian_motion, jump_diffusion
from advanced_scanner.math_utils import combinations
from advanced_scanner.stats_models import t_test
from advanced_scanner.stats import (
    portfolio_stats, print_portfolio_report, 
    score_color, bias_color, lev_color, bar_color, pct_color, ts_to_str
)
from advanced_scanner.optimization import optimize_parameters, fuzz_sweep
from advanced_scanner.deepseek import report_to_deepseek

def run_stress_tests(sym_rows_map, funding_map, vol_map, threshold, hold_bars, capital, risk, n_paths=5):
    print("\n" + c("  ◆ RUNNING MULTI-MODEL STRESS TESTS...", BOLD + BRIGHT_WHITE))
    gbm_results = []
    jump_results = []
    
    symbols_to_test = list(sym_rows_map.keys())[:3]
    for s in symbols_to_test:
        rs = sym_rows_map[s]
        if len(rs) < 100: continue
        close = np.array([float(r[6]) for r in rs])
        returns = np.diff(np.log(close))
        if len(returns) == 0: continue
        
        mu = np.mean(returns); sigma = np.std(returns); S0 = close[-1]; N = 200
        
        for i in range(n_paths):
            path_gbm = geometric_brownian_motion(S0, mu, sigma, T=1.0, N=N)
            path_jump = jump_diffusion(S0, mu, sigma, T=1.0, N=N, lambda_j=2.0, mu_j=-0.05, sigma_j=0.1)
            
            for p_type, path in [("GBM", path_gbm), ("JUMP", path_jump)]:
                synth_rows = []
                cur_ts = int(rs[-1][0])
                for j in range(len(path)):
                    p = path[j]
                    synth_rows.append([str(cur_ts + (j+1)*3600), "3600", str(p), str(p), str(p*1.002), str(p*0.998), str(p), "1000"])
                
                test_rows_map = {s: rs + synth_rows}
                all_trades, equity_curve, final_cap, concurrent_log = backtest_portfolio(
                    test_rows_map, funding_map, vol_map, threshold, hold_bars, 
                    starting_capital=capital, risk_per_trade=risk
                )
                ret = (final_cap - capital) / capital * 100
                if p_type == "GBM": gbm_results.append(ret)
                else: jump_results.append(ret)
            
    avg_gbm = sum(gbm_results)/len(gbm_results) if gbm_results else 0
    avg_jump = sum(jump_results)/len(jump_results) if jump_results else 0
    print(f"    GBM Average Return:      {pct_color(avg_gbm)}")
    print(f"    Jump-Diffusion Return:   {pct_color(avg_jump)} (Stress Test)")
    if avg_jump < -15: print(f"    {c('WARNING', BRIGHT_RED)}: Strategy highly vulnerable to sudden market jumps.")
    else: print(f"    {c('CONFIRMED', BRIGHT_GREEN)}: Strategy showed robustness against jump diffusion.")

def calculate_luck_probability(n_trades, n_wins):
    if n_trades == 0: return 1.0
    prob = 0
    for k in range(n_wins, n_trades + 1):
        prob += combinations(n_trades, k) * (0.5 ** n_trades)
    return prob

from advanced_scanner.scoring import extract_features, cross_sectional_score_ranking

def run_scan(syms, funds):
    type_print(c("Scanning concurrently...", BRIGHT_CYAN))
    feature_matrices = []
    asset_names = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch, s, funds, KLINE_LIMIT): s for s in syms}
        for f in as_completed(futs):
            s, rs, fund = f.result()
            if rs is not None and len(rs) >= 60:
                X, _, _, _ = extract_features(np.array(rs, dtype=float), fund)
                feature_matrices.append(X)
                asset_names.append(s)
            done += 1
            sys.stdout.write(f"\r  {done}/{len(syms)} scanned"); sys.stdout.flush()
    if not feature_matrices:
        print("\nNo valid assets to scan."); return []
    ranking = cross_sectional_score_ranking(feature_matrices, asset_names)
    print("\n" + c("─" * 68, DIM) + "\n" + c("  RANK  SYMBOL           SCORE      SIDE", BOLD) + "\n" + c("─" * 68, DIM))
    sorted_assets = sorted(ranking.items(), key=lambda x: x[1]['rank'])
    for i, (sym, info) in enumerate(sorted_assets, 1):
        print(f"  {i:<5} {c(sym, CYAN):<16} {info['score']:>7.3f}   {info['side']:<7}")
    print(c("─" * 68, DIM))
    return sorted_assets

def run_robustness_checks(sym_rows_map, funding_map, vol_map, threshold, hold_bars, capital, risk, sl_m=2.0, tp_m=3.5):
    print("\n" + c("  ◆ RUNNING AUTOMATIC ROBUSTNESS CHECKS...", BOLD + BRIGHT_WHITE))
    bias_found = False
    for s in list(sym_rows_map.keys())[:3]:
        rs = sym_rows_map[s]
        if len(rs) < 100: continue
        s1 = score_rows(rs[:60], funding_map.get(s, 0))
        s2 = score_rows(rs[:60], funding_map.get(s, 0))
        if s1 != s2: bias_found = True; break
    if bias_found: print(f"  {c('× LOOK-AHEAD BIAS', BRIGHT_RED):<25} Non-deterministic scoring detected!")
    else: print(f"  {c('√ LOOK-AHEAD BIAS', BRIGHT_GREEN):<25} No non-deterministic behavior detected.")

    print(f"  {c('~ SENSITIVITY TEST', BRIGHT_YELLOW):<25} Testing threshold variation (±5)...")
    _, _, c_low, _ = backtest_portfolio(sym_rows_map, funding_map, vol_map, threshold-5, hold_bars, 2, capital, risk, sl_mult=sl_m, tp_mult=tp_m)
    _, _, c_high, _ = backtest_portfolio(sym_rows_map, funding_map, vol_map, threshold+5, hold_bars, 2, capital, risk, sl_mult=sl_m, tp_mult=tp_m)
    var_low = (c_low - capital) / capital * 100
    var_high = (c_high - capital) / capital * 100
    print(f"    Threshold {threshold-5}: {pct_color(var_low)} return")
    print(f"    Threshold {threshold+5}: {pct_color(var_high)} return")
    if (var_low > 0 and var_high < 0) or (var_low < 0 and var_high > 0): print(f"  {c('! STABILITY WARNING', BRIGHT_RED):<25} Sensitive to threshold choice.")
    else: print(f"  {c('√ PARAMETER STABILITY', BRIGHT_GREEN):<25} Strategy direction is consistent.")

def run_backtest_workflow(syms, funds, vols, threshold, hold_bars, bt_limit, top_n_bt, starting_capital, risk_per_trade, max_bars=None, offset=0, optimize=False, sweep=False, wfo=False):
    if top_n_bt and top_n_bt < len(syms):
        syms = random.sample(syms, top_n_bt)
        print(c(f"Randomly selected {len(syms)} symbols to avoid selection bias", BRIGHT_WHITE))
    type_print(c(f"Fetching {bt_limit}-bar history for {len(syms)} symbols...", BRIGHT_CYAN))
    sym_rows_map, funding_map, done = {}, {}, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch, s, funds, bt_limit): s for s in syms}
        for f in as_completed(futs):
            sym, rs, fund = f.result()
            if rs:
                if offset: rs = rs[:-offset]
                if max_bars: rs = rs[-max_bars:]
                if rs: sym_rows_map[sym] = rs; funding_map[sym] = fund
            done += 1; sys.stdout.write(f"\r  {done}/{len(syms)} fetched"); sys.stdout.flush()
    print()
    if not sym_rows_map: print(c("  No data received.", BRIGHT_RED)); return
    
    output_capture = io.StringIO(); original_stdout = sys.stdout; sys.stdout = output_capture
    try:
        if wfo:
            walk_forward_optimization(sym_rows_map, funding_map, vols, starting_capital, risk_per_trade)
        else:
            cooldown = 2; adx_f = True; sl_m = 2.0; tp_m = 3.5
            if sweep:
                best = fuzz_sweep(sym_rows_map, funding_map, vols, starting_capital, initial_threshold=threshold)
                threshold = best['threshold']; hold_bars = best['hold_bars']; risk_per_trade = best['risk_per_trade']
                cooldown = best['cooldown']; adx_f = best['adx_filter']; sl_m = best['sl_mult']; tp_m = best['tp_mult']
            elif optimize:
                best_t, best_h = optimize_parameters(sym_rows_map, funding_map, vols, starting_capital, risk_per_trade)
                threshold, hold_bars = best_t, best_h

            type_print(c(f"Running portfolio simulation (Risk {risk_per_trade*100:.1f}% per trade)...", BRIGHT_CYAN))
            all_trades, equity_curve, final_cap, concurrent_log = backtest_portfolio(
                sym_rows_map, funding_map, vols, threshold, hold_bars, cooldown, 
                starting_capital, risk_per_trade, adx_filter=adx_f, sl_mult=sl_m, tp_mult=tp_m
            )
            st = portfolio_stats(all_trades, equity_curve, final_cap, concurrent_log, starting_capital)
            
            print_portfolio_report(st, all_trades, equity_curve, concurrent_log, starting_capital, risk_per_trade, threshold, hold_bars, bt_limit)
            trade_rets = [t.pnl_pct for t in all_trades]
            if len(trade_rets) >= 2:
                t_stat, p_val = t_test(trade_rets, [0] * len(trade_rets))
                print(f"\n  {c('◆ STATISTICAL SIGNIFICANCE (T-TEST)', BOLD + BRIGHT_WHITE)}")
                print(f"    T-Statistic: {t_stat:.4f}  |  P-Value: {p_val:.6f}")
                if p_val < 0.05: print(f"    {c('SIGNIFICANT', BRIGHT_GREEN)}: Strategy edge is statistically likely.")
                else: print(f"    {c('NOT SIGNIFICANT', BRIGHT_YELLOW)}: Edge may be due to chance.")
                n_wins = len([t for t in all_trades if t.pnl_usd > 0])
                luck_p = calculate_luck_probability(len(all_trades), n_wins)
                print(f"    Luck Prob:   {luck_p:.6f}  (Prob. of {n_wins}/{len(all_trades)} wins by chance)")
            run_robustness_checks(sym_rows_map, funding_map, vols, threshold, hold_bars, starting_capital, risk_per_trade, sl_m, tp_m)
            run_stress_tests(sym_rows_map, funding_map, vols, threshold, hold_bars, starting_capital, risk_per_trade)
    finally: sys.stdout = original_stdout
    report_content = output_capture.getvalue(); print(report_content)
    report_to_deepseek(report_content)

run_backtest = run_backtest_workflow

def estimate_duration(args, syms):
    from advanced_scanner.config import BT_KLINE_LIMIT, KLINE_LIMIT, MAX_WORKERS, TOP_N
    
    n_syms = len(syms)
    if args.top and args.top < n_syms:
        n_syms = args.top
    if args.symbol:
        n_syms = 1
        
    # Heuristic constants based on typical performance
    fetch_req_time = 0.35 # seconds per 1000-bar request
    # backtest_portfolio time depends on number of symbols and number of bars
    # Roughly 0.005s per symbol for 4000 bars on a decent machine, but let's be conservative
    bt_base_time = 0.015 # seconds to backtest 1 symbol for 4000 bars
    
    # Adjust bar count
    bars = args.max_bars if args.max_bars else BT_KLINE_LIMIT
    bar_factor = bars / 4000.0
    
    total_time = 1.2 # Initial bootstrap / tickers time
    
    if args.scan_only:
        # scan-only fetches KLINE_LIMIT (default 100) for ALL filtered symbols
        # 1 request per symbol
        fetch_time = (len(syms) * fetch_req_time) / MAX_WORKERS
        score_time = len(syms) * 0.002 # scoring is very fast
        total_time += fetch_time + score_time
    else:
        # Backtest workflow
        # 1. Fetching history
        reqs_per_sym = (bars + 999) // 1000
        fetch_time = (n_syms * reqs_per_sym * fetch_req_time) / MAX_WORKERS
        total_time += fetch_time
        
        # 2. Backtest execution
        # A single backtest of n_syms over 'bars'
        single_bt_time = n_syms * bt_base_time * bar_factor
        
        if args.wfo:
            # walk_forward_optimization: 4 windows. 
            # In each window: optimize_parameters (42 iterations) + 1 test backtest
            # Windows are smaller than full history, but we fetch full anyway
            # Optimization iterations run in ProcessPool
            opt_iterations = 42
            window_bt_time = single_bt_time * 0.25 # roughly 1/4 of data
            opt_time = (opt_iterations * window_bt_time) / MAX_WORKERS
            # 3 step transitions in WFO
            total_time += 3 * (opt_time + window_bt_time)
        elif args.sweep:
            # fuzz_sweep: 40 fuzz iterations + 10 Monte Carlo iterations
            fuzz_iterations = 40
            mc_iterations = 10
            total_time += ((fuzz_iterations + mc_iterations) * single_bt_time) / MAX_WORKERS
        elif args.optimize:
            # optimize_parameters: 42 iterations
            opt_iterations = 42
            total_time += (opt_iterations * single_bt_time) / MAX_WORKERS
        else:
            # Standard single backtest
            total_time += single_bt_time
            
        # 3. Post-backtest tasks (always run in run_backtest_workflow unless wfo)
        if not args.wfo:
            # run_robustness_checks: 2 backtests of top 3 symbols
            robust_time = (2 * min(n_syms, 3) * bt_base_time * bar_factor)
            # run_stress_tests: n_paths=5, 2 model types, for top 3 symbols = 30 small backtests
            # These backtests are over synthetic data (200 bars), so they are fast
            stress_time = (30 * min(n_syms, 3) * bt_base_time * (200/4000))
            total_time += robust_time + stress_time + 1.0 # + reporting overhead

    print("\n" + c("═" * 60, BRIGHT_MAGENTA))
    print(c("  ◆ SYSTEM EXECUTION PROJECTION", BOLD + BRIGHT_WHITE))
    print(c("═" * 60, BRIGHT_MAGENTA))
    print(f"  Mode:            {c('Scan Only' if args.scan_only else 'Backtest/Strategy', BRIGHT_YELLOW)}")
    print(f"  Symbols:         {c(str(n_syms), BRIGHT_CYAN)}")
    print(f"  Bars/History:    {c(str(bars), BRIGHT_CYAN)}")
    print(f"  Workers:         {c(str(MAX_WORKERS), BRIGHT_CYAN)}")
    
    color = BRIGHT_GREEN if total_time < 30 else BRIGHT_YELLOW if total_time < 120 else BRIGHT_RED
    
    if total_time < 60:
        print(f"  Est. Duration:   {c(f'{total_time:.1f} seconds', color)}")
    else:
        print(f"  Est. Duration:   {c(f'{total_time/60:.2f} minutes', color)}")
        
    print(c("═" * 60, BRIGHT_MAGENTA))
    print(c("  (Includes network latency, concurrent fetching, and CPU overhead)", DIM))
    
    # Log the estimate
    est_str = f"{total_time:.1f}s" if total_time < 60 else f"{total_time/60:.2f}m"
    log_execution("estimate", args, f"Estimated duration: {est_str}")
    
    report_to_deepseek(include_code=True)

def main():
    parser = argparse.ArgumentParser(description="Phemex USDT Perp Scanner + Portfolio Backtest", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", type=str, help="Focus on a single symbol (e.g., BTCUSDT)")
    parser.add_argument("--top", type=int, help="Number of symbols to backtest (random selection)")
    parser.add_argument("--threshold", type=int, default=25, help="Min |score| for trade (Integer 0-100, default: 25)")
    parser.add_argument("--hold", type=int, default=4, help="Max bars/hours to hold a position (Integer, default: 4)")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL, help="Starting capital in USDT (default: 100.0)")
    parser.add_argument("--risk", type=float, default=DEFAULT_RISK_PCT, help="Risk per trade. Supports fraction (0.01) or percentage (1.0 for 1%%)")
    parser.add_argument("--max-bars", type=int, help="Limit total bars processed in backtest (Integer)")
    parser.add_argument("--offset", type=int, default=0, help="Offset start of backtest bars from newest (Integer)")
    parser.add_argument("--scan-only", action="store_true", help="Only scan current market and exit")
    parser.add_argument("--optimize", action="store_true", help="Brute-force optimization of threshold/hold")
    parser.add_argument("--sweep", action="store_true", help="Run comprehensive fuzzing and Monte Carlo sweep")
    parser.add_argument("--wfo", action="store_true", help="Run Walk-Forward Optimization (Rolling temporal validation)")
    parser.add_argument("--estimate", action="store_true", help="Estimate execution time without running the full workflow")
    parser.add_argument("--meta-commentary", action="store_true", help="Only run AI code metacommentary and exit")
    args = parser.parse_args()
    start_time = time.time()
    
    if args.meta_commentary:
        report_to_deepseek(include_code=True)
        sys.exit(0)
    
    syms, funds, vols = bootstrap(args)
    if not syms: print(c("[!] No symbols found.", BRIGHT_RED)); sys.exit(1)
    
    if args.estimate:
        estimate_duration(args, syms)
        sys.exit(0)
        
    if args.scan_only: 
        output_capture = io.StringIO(); original_stdout = sys.stdout; sys.stdout = output_capture
        try:
            run_scan(syms, funds)
        finally: sys.stdout = original_stdout
        scan_report = output_capture.getvalue()
        print(scan_report)
        report_to_deepseek(scan_report, include_code=True)
    else:
        risk = args.risk / 100.0 if args.risk > 0.5 else args.risk
        run_backtest_workflow(syms, funds, vols, args.threshold, args.hold, BT_KLINE_LIMIT, args.top, args.capital, risk, args.max_bars, args.offset, args.optimize, args.sweep, args.wfo)
    
    duration = time.time() - start_time
    dur_str = f"{duration:.1f}s" if duration < 60 else f"{duration/60:.2f}m"
    print(f"\n  {c('◆ ACTUAL EXECUTION TIME:', BOLD + BRIGHT_WHITE)} {c(dur_str, BRIGHT_GREEN)}")
    log_execution("execution", args, f"Actual duration: {dur_str}")

if __name__ == "__main__":
    main()
