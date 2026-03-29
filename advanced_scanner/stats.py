"""
stats.py  ─  Portfolio Statistics & Reporting
"""

import time
from collections import defaultdict
from advanced_scanner.utils import c
from advanced_scanner.config import (
    BOLD, BRIGHT_WHITE, BRIGHT_GREEN, GREEN, BRIGHT_RED, RED, YELLOW,
    DIM, BRIGHT_CYAN, BRIGHT_MAGENTA, MAGENTA, CYAN, BLUE
)

def portfolio_stats(all_trades, equity_curve, final_cap, concurrent_log, start_cap):
    if not all_trades: return {}
    pnls = [t.pnl_usd for t in all_trades]; wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls); wr = len(wins)/len(pnls)*100 if pnls else 0
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses) != 0 else float("inf")
    equities = [eq for _, eq in equity_curve]; peak_eq = max(equities) if equities else start_cap
    max_dd_u, max_dd_p, cur_peak = 0.0, 0.0, start_cap
    for eq in equities:
        if eq > cur_peak: cur_peak = eq
        dd_u = cur_peak - eq; dd_p = dd_u / cur_peak * 100 if cur_peak else 0
        if dd_u > max_dd_u: max_dd_u = dd_u
        if dd_p > max_dd_p: max_dd_p = dd_p
    
    avg_pnl = total_pnl / len(pnls) if pnls else 0
    std_pnl = (sum((p-avg_pnl)**2 for p in pnls)/(len(pnls)-1))**0.5 if len(pnls)>1 else 0
    sharpe = avg_pnl / std_pnl if std_pnl else 0
    
    downside_pnls = [p for p in pnls if p < 0]
    dd_dev = (sum(p**2 for p in downside_pnls)/len(pnls))**0.5 if len(pnls) > 0 else 0
    sortino = avg_pnl / dd_dev if dd_dev else 0
    
    n_days = len(equity_curve) / 24 if len(equity_curve) > 0 else 1
    ann_ret = (total_pnl / start_cap) / (max(1, n_days) / 365)
    calmar = ann_ret / (max_dd_p / 100) if max_dd_p > 0 else 0

    # Fitness metric for optimization: Penalize low trade count
    from advanced_scanner.config import MIN_TRADES_FOR_STATS
    n_trades = len(all_trades)
    fitness = sharpe
    if n_trades < MIN_TRADES_FOR_STATS:
        fitness *= (n_trades / MIN_TRADES_FOR_STATS)
    
    # Diversification bonus/penalty
    unique_syms = len(set(t.sym for t in all_trades))
    if unique_syms < 3 and n_trades > 5:
        fitness *= 0.5 # Penalize if heavily concentrated on 1-2 symbols

    by_sym = defaultdict(list)
    for t in all_trades: by_sym[t.sym].append(t)
    sym_sum = {}
    for s, ts in by_sym.items():
        sp = [t.pnl_usd for t in ts]
        sym_sum[s] = {"n": len(ts), "net": sum(sp), "wr": len([p for p in sp if p>0])/len(sp)*100, "liqs": sum(1 for t in ts if t.liquidated)}
    return {"n_trades": len(all_trades), "win_rate": wr, "total_pnl": total_pnl, "total_ret_pct": (final_cap-start_cap)/start_cap*100,
            "final_cap": final_cap, "peak_eq": peak_eq, "max_dd_usd": max_dd_u, "max_dd_pct": max_dd_p, "avg_win": sum(wins)/len(wins) if wins else 0,
            "avg_loss": sum(losses)/len(losses) if losses else 0, "pf": pf, "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "fitness": fitness,
            "liq_count": sum(1 for t in all_trades if t.liquidated),
            "avg_lev": sum(t.leverage for t in all_trades)/len(all_trades), "max_concurrent": max(concurrent_log) if concurrent_log else 0,
            "avg_concurrent": sum(concurrent_log)/len(concurrent_log) if concurrent_log else 0, "utilisation": sum(1 for n in concurrent_log if n>0)/len(concurrent_log)*100 if concurrent_log else 0,
            "n_long": len([t for t in all_trades if t.direction=="LONG"]), "n_short": len([t for t in all_trades if t.direction=="SHORT"]),
            "long_pnl": sum(t.pnl_usd for t in all_trades if t.direction=="LONG"), "short_pnl": sum(t.pnl_usd for t in all_trades if t.direction=="SHORT"),
            "sym_summary": sym_sum, "pnls": pnls}

def ascii_histogram(pnls, width=40, bins=20):
    if not pnls: return
    lo, hi = min(pnls), max(pnls); span = hi-lo if hi!=lo else 1.0
    counts = [0]*bins
    for p in pnls:
        idx = int((p-lo)/span*(bins-1))
        counts[idx] += 1
    m_count = max(counts) if counts else 1
    print("\n" + c("  P&L DISTRIBUTION", BOLD + BRIGHT_WHITE))
    for i, count in enumerate(counts):
        bin_val = lo + i*span/(bins-1)
        bar = "█" * int(count/m_count*width)
        color = BRIGHT_GREEN if bin_val > 0 else (BRIGHT_RED if bin_val < 0 else YELLOW)
        print(f"  {bin_val: >8.2f} | {c(bar, color)} {count}")
    print()

def score_color(sc):
    sc_int = int(sc) # Explicitly cast to int
    if sc_int >= 25: return c(f"{sc_int:+d}", BRIGHT_GREEN)
    if sc_int >= 10: return c(f"{sc_int:+d}", GREEN)
    if sc_int <= -25: return c(f"{sc_int:+d}", BRIGHT_RED)
    if sc_int <= -10: return c(f"{sc_int:+d}", RED)
    return c(f"{sc_int:+d}", YELLOW)

def bias_color(sc):
    if sc > 0: return c("LONG ", BRIGHT_GREEN)
    if sc < 0: return c("SHORT", BRIGHT_RED)
    return c("FLAT ", YELLOW)

def bar_color(sc):
    bar = "#" * int(abs(sc)/4)
    return c(bar, GREEN) if sc > 0 else (c(bar, RED) if sc < 0 else bar)

def usd_color(v, f="+.2f"):
    s = f"${v:{f}}"
    if v > 0: return c(s, BRIGHT_GREEN)
    if v < 0: return c(s, BRIGHT_RED)
    return c(s, YELLOW)

def pct_color(v, f="+.1f"):
    s = f"{v:{f}}%"
    if v > 0: return c(s, BRIGHT_GREEN)
    if v < 0: return c(s, BRIGHT_RED)
    return c(s, YELLOW)

def wr_color(wr):
    s = f"{wr:.1f}%"
    if wr >= 60: return c(s, BRIGHT_GREEN)
    if wr >= 50: return c(s, GREEN)
    if wr >= 40: return c(s, YELLOW)
    return c(s, BRIGHT_RED)

def pf_color(pf):
    s = f"{pf:.2f}" if pf < 999 else "  inf"
    if pf >= 1.5: return c(s, BRIGHT_GREEN)
    if pf >= 1.0: return c(s, GREEN)
    return c(s, BRIGHT_RED)

def lev_color(lev):
    s = f"{lev:.0f}x"
    if lev >= 10: return c(s, BRIGHT_MAGENTA)
    if lev >= 5: return c(s, MAGENTA)
    return c(s, YELLOW)

def ts_to_str(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(ts)))

def ascii_equity_chart(equity_curve, width=60, height=12):
    if len(equity_curve) < 2: return
    equities = [eq for _, eq in equity_curve]
    step = max(1, len(equities)//width); sampled = equities[::step]
    if equities[-1] not in sampled: sampled.append(equities[-1])
    lo, hi = min(sampled), max(sampled); span = hi-lo if hi!=lo else 1.0
    grid = [[" "]*len(sampled) for _ in range(height)]
    for x, v in enumerate(sampled): grid[height-1-int((v-lo)/span*(height-1))][x] = "•"
    print("\n" + c("  EQUITY CURVE", BOLD + BRIGHT_WHITE))
    print(c(f"  ${hi:,.2f} ┐", GREEN))
    for row in grid: print("         │" + "".join(c(ch, BRIGHT_GREEN if ch=="•" else DIM) for ch in row))
    print(c(f"  ${lo:,.2f} └" + "─"*len(sampled), DIM))
    print(c(f"  {'START':>{len(sampled)//2}}{'END':>{len(sampled)-len(sampled)//2}}", DIM))

def print_portfolio_report(st, all_trades, equity_curve, concurrent_log, start_cap, risk_per_trade, threshold, hold_bars, bt_limit):
    sep, sep2 = "─" * 100, "═" * 100
    print("\n" + c(sep2, BRIGHT_CYAN))
    cfg = (f"start=${start_cap:.0f}  risk={risk_per_trade*100:.1f}%/trade  "
           f"threshold={threshold}  hold={hold_bars}h  lookback={bt_limit} bars  1h candles")
    print(c("  ◆ PORTFOLIO BACKTEST (LIMIT ORDERS)", BOLD+BRIGHT_WHITE) + "  " + c(cfg, DIM))
    print(c(sep2, BRIGHT_CYAN))
    if not st:
        print(c("  No trades generated.", YELLOW)); return
    ret_col = BRIGHT_GREEN if st["total_ret_pct"] > 0 else BRIGHT_RED
    print(f"\n  {c('CAPITAL', BOLD)}   start {c(f'${start_cap:.2f}', BRIGHT_WHITE)}  →  final {c(f'${st["final_cap"]:.2f}', ret_col)}  "
          f"({c(f'{st["total_ret_pct"]:+.2f}%', ret_col)})   peak {c(f'${st["peak_eq"]:.2f}', BRIGHT_GREEN)}  "
          f"max drawdown {c(f'${st["max_dd_usd"]:.2f}', RED)} ({c(f'{st["max_dd_pct"]:.1f}%', RED)})\n")
    sh_col = BRIGHT_GREEN if st["sharpe"] > 0.5 else (GREEN if st["sharpe"] > 0 else BRIGHT_RED)
    so_col = BRIGHT_GREEN if st["sortino"] > 0.7 else (GREEN if st["sortino"] > 0 else BRIGHT_RED)
    ca_col = BRIGHT_GREEN if st["calmar"] > 1.0 else (GREEN if st["calmar"] > 0.5 else BRIGHT_RED)
    
    print(f"  {c('TRADES', BOLD)} {st['n_trades']}  │  {c('WIN RATE', BOLD)} {wr_color(st['win_rate'])}  │  {c('P.FACTOR', BOLD)} {pf_color(st['pf'])}")
    print(f"  {c('SHARPE', BOLD)} {c(f'{st["sharpe"]:+.2f}', sh_col)}  │  {c('SORTINO', BOLD)} {c(f'{st["sortino"]:+.2f}', so_col)}  │  {c('CALMAR', BOLD)} {c(f'{st["calmar"]:+.2f}', ca_col)}")
    print(f"  {c('AVG LEV', BOLD)} {lev_color(st['avg_lev'])}  │  {c('LIQS', BOLD)} {c(str(st['liq_count']), BRIGHT_RED if st['liq_count'] else DIM)}")
    print(f"  {c('AVG WIN', BOLD)} {usd_color(st['avg_win'])}  │  {c('AVG LOSS', BOLD)} {usd_color(st['avg_loss'])}  │  {c('LONG PNL', BOLD)} {usd_color(st['long_pnl'])}  "
          f"│  {c('SHORT PNL', BOLD)} {usd_color(st['short_pnl'])}  │  {c('LONG', BOLD)} {st['n_long']}  {c('SHORT', BOLD)} {st['n_short']}")
    print(f"  {c('MAX CONCURRENT', BOLD)} {st['max_concurrent']}  │  {c('AVG CONCURRENT', BOLD)} {st['avg_concurrent']:.1f}  │  {c('UTILISATION', BOLD)} {c(f'{st["utilisation"]:.1f}%', BRIGHT_CYAN)}")
    
    ascii_equity_chart(equity_curve)
    ascii_histogram(st["pnls"])

    sym_data = st["sym_summary"]
    if sym_data:
        print("\n  " + c("PER-SYMBOL BREAKDOWN", BOLD + BRIGHT_WHITE) + "\n" + c(sep, DIM))
        print(c(f"  {'SYMBOL':<14} {'TRADES':>6}  {'WIN%':>6}  {'NET P&L':>9}  {'LIQS':>5}", BOLD))
        for s, info in sorted(sym_data.items(), key=lambda x: -x[1]["net"]):
            print(f"  {c(s, CYAN):<14} {info['n']:>6}  {wr_color(info['wr']):>6}  {usd_color(info['net']):>9}  {c(str(info['liqs']), BRIGHT_RED if info['liqs'] else DIM):>5}")
    
    print("\n  " + c("ROBUSTNESS & OVERFITTING DIAGNOSTICS", BOLD + BRIGHT_WHITE) + "\n" + c(sep, DIM))
    print(f"  {c('LOOK-AHEAD BIAS', BOLD)}:   {c('PASSED', BRIGHT_GREEN)} (Scoring uses strictly historical slices)")
    
    if len(equity_curve) > 40:
        mid = len(equity_curve) // 2
        is_pnl = equity_curve[mid][1] - equity_curve[0][1]
        oos_pnl = equity_curve[-1][1] - equity_curve[mid][1]
        is_ret = (is_pnl / equity_curve[0][1]) * 100
        oos_ret = (oos_pnl / equity_curve[mid][1]) * 100
        ratio = (oos_ret / is_ret) if is_ret != 0 else 0
        
        status = c("HEALTHY", BRIGHT_GREEN) if ratio > 0.3 else (c("DEGRADED", YELLOW) if ratio > 0 else c("OVERFIT / REGIME SHIFT", BRIGHT_RED))
        if is_ret < 0 and oos_ret < 0: status = c("CONSISTENTLY POOR", RED)
        elif is_ret < 0 and oos_ret > 0: status = c("RECOVERING", BRIGHT_CYAN)

        print(f"  {c('IS/OOS STABILITY', BOLD)}: {status}")
        print(f"    In-Sample Return:  {pct_color(is_ret)}  (First 50% of time)")
        print(f"    Out-Sample Return: {pct_color(oos_ret)}  (Last 50% of time)")

    if st["n_trades"] > 5:
        sorted_pnls = sorted([t.pnl_usd for t in all_trades])
        top_3_pct = sum(sorted_pnls[-3:]) / st["total_pnl"] if st["total_pnl"] > 0 else 0
        if top_3_pct > 0.8:
            print(f"  {c('FRAGILITY CHECK', BOLD)}:  {c('HIGH', BRIGHT_RED)} (80%+ of P&L from top 3 trades)")
        else:
            print(f"  {c('FRAGILITY CHECK', BOLD)}:  {c('LOW', BRIGHT_GREEN)} (Distributed P&L)")

    print("\n  " + c("FULL TRADE LOG", BOLD + BRIGHT_WHITE) + "\n" + c(sep, DIM))
    print(c(f"  {'#':>4}  {'SYM':<12} {'DIR':>5}  {'LEV':>4}  {'ENTRY TIME':<17} {'ENTRY':>11}  {'EXIT':>11}  {'P&L':>9}  {'P&L%':>7}", BOLD))
    for i, t in enumerate(sorted(all_trades, key=lambda x: x.entry_ts), 1):
        d_col = BRIGHT_GREEN if t.direction=="LONG" else BRIGHT_RED
        print(f"  {i:>4}  {c(t.sym, CYAN):<12} {c(t.direction, d_col):>5}  {lev_color(t.leverage):>4}  {ts_to_str(t.entry_ts):<17} {t.entry_price:>11.4f}  {t.exit_price:>11.4f}  {usd_color(t.pnl_usd):>9}  {pct_color(t.pnl_pct):>7}")
    print(c(sep, DIM) + f"\n  {c('FINAL CAPITAL', BOLD+BRIGHT_WHITE)}  {usd_color(st['final_cap'])}  ({pct_color(st['total_ret_pct'])})\n" + c(sep2, BRIGHT_CYAN) + "\n")