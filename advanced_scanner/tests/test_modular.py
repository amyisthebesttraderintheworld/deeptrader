import pytest
import os
import re
import json
import io
import sys
import numpy as np
from unittest.mock import MagicMock

# Modular Imports
from advanced_scanner.config import *
from advanced_scanner.utils import strip_ansi, get_env_key, get_json
from advanced_scanner.indicators import rsi, bb, ema, macd, atr, adx, stochastic, z_score
from advanced_scanner.scoring import score_rows, candle_strength, engulf, score_to_leverage
from advanced_scanner.trade_objects import Order, Trade
from advanced_scanner.backtest import backtest_portfolio
from advanced_scanner.stats import portfolio_stats, print_portfolio_report
from advanced_scanner.fetcher import fetch, bootstrap
from advanced_scanner.main import run_scan, optimize_parameters, run_backtest

# --- Helpers ---
def create_synthetic_rows(length=60, price_func=lambda i: 100 + i):
    rows = []
    for i in range(length):
        p = price_func(i)
        rows.append([
            i * 3600, # ts
            0, 0, # ignored
            p - 0.5, # open
            p + 1.0, # high
            p - 1.0, # low
            p, # close
            1000 # volume
        ])
    return rows

# --- 1. Unit tests for pure functions ---

def test_strip_ansi():
    assert strip_ansi("\x1b[31mRed\x1b[0m") == "Red"
    assert strip_ansi("Normal") == "Normal"

def test_get_env_key(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_KEY", "env_val")
    assert get_env_key("TEST_KEY") == "env_val"
    
    d = tmp_path / "env_test"
    d.mkdir()
    old_cwd = os.getcwd()
    os.chdir(d)
    try:
        with open(".env", "w") as f:
            f.write('DOTENV_KEY="dotenv_val"\n')
            f.write("PLAIN_KEY=plain_val\n")
        assert get_env_key("DOTENV_KEY") == "dotenv_val"
        assert get_env_key("PLAIN_KEY") == "plain_val"
    finally:
        os.chdir(old_cwd)

def test_indicators():
    prices = [10 + i for i in range(50)]
    r = rsi(prices)
    assert (r >= 0).all() and (r <= 100).all()
    l, m, h = bb(prices)
    assert (l[20:] < m[20:]).all() and (m[20:] < h[20:]).all()
    assert len(ema(prices, 10)) == 50
    m_line, s_line, h_hist = macd(prices)
    assert isinstance(h_hist[-1], (float, int, np.float64))
    
    h_l = [p+1 for p in prices]; l_l = [p-1 for p in prices]; c_l = prices
    assert (atr(h_l, l_l, c_l, 10) >= 0).all()
    assert (adx(h_l, l_l, c_l, 10) >= 0).all()
    st = stochastic(h_l, l_l, c_l, 10)
    assert (st >= 0).all() and (st <= 100).all()
    assert isinstance(z_score(prices), np.ndarray)

def test_candle_patterns():
    assert candle_strength(10, 10.9, 7, 10.5) == 10 
    assert engulf(11, 10, 9.5, 11.5) == 10 # bullish engulf

# --- 2. Tests for score_rows() ---

def test_score_rows_basic():
    rows = create_synthetic_rows(100)
    s = score_rows(rows, 0)
    assert -100 <= s <= 100
    assert score_rows([], 0) == 0

# --- 3. Order and Trade mechanics ---

def test_order_and_trade():
    o = Order("BTC", "LONG", 10, 1000, 50000, 50, 100, 500, 1500, 0.01, 1e6)
    assert o.expires_bar == 14
    t = Trade(o, 50000, 11, 1060)
    assert t.sl_price == 49500
    t.update_funding(0.0001)
    assert t.funding_accumulated < 0
    t.close(12, 1120, 51000)
    assert t.pnl_usd > 0

# --- 4. Backtest Portfolio ---

def test_backtest_simple(monkeypatch):
    import advanced_scanner.backtest as backtest
    import random
    def mock_calc(ohlcv, f, weights=None):
        res = np.zeros(len(ohlcv))
        if len(ohlcv) > 61:
            res[61] = 50
        return res
    monkeypatch.setattr(backtest, "calculate_all_scores", mock_calc)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    
    rows = create_synthetic_rows(100, lambda i: 100)
    rows[61][5] = 99 # Low price to ensure fill if order placed at 100
    
    all_trades, equity_curve, final_cap, concurrent_log = backtest_portfolio(
        {"BTC": rows}, {"BTC": 0}, {"BTC": 1e9}, threshold=40, adx_filter=False
    )
    assert len(all_trades) >= 1

# --- 5. Portfolio Stats ---

def test_stats():
    o = Order("BTC", "LONG", 10, 1000, 50000, 50, 100, 500, 1500, 0.01, 1e6)
    t = Trade(o, 50000, 11, 1060)
    t.close(12, 1120, 51000)
    st = portfolio_stats([t], [(1000, 100), (1120, 110)], 110, [1], 100)
    assert st["win_rate"] == 100.0

# --- 6. Plumbing ---

def test_plumbing(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": {"rows": [[1000, 1, 2, 3, 4, 5, 6, 7]], "perpProductsV2": []}}
    import advanced_scanner.utils as utils
    monkeypatch.setattr(utils.SESSION, "get", lambda *a, **kw: mock_resp)
    assert get_json("http://test") == mock_resp.json()
    
    import advanced_scanner.fetcher as fetcher
    monkeypatch.setattr(fetcher, "get_json", lambda *a, **kw: mock_resp.json())
    s, r, f = fetch("BTC", {})
    assert len(r) == 1

def test_run_scan(monkeypatch):
    import main
    monkeypatch.setattr(main, "fetch", lambda s, f, l: (s, create_synthetic_rows(100), 0))
    ranked = run_scan(["BTC"], {})
    assert len(ranked) == 1

def test_optimize_and_backtest(monkeypatch):
    import main
    monkeypatch.setattr(main, "backtest_portfolio", lambda *a, **kw: ([], [], 100, []))
    monkeypatch.setattr(main, "report_to_deepseek", lambda r: None)
    
    rows = create_synthetic_rows(100)
    best = optimize_parameters({"BTC": rows}, {}, {}, 100, 0.01)
    assert len(best) == 2
    
    monkeypatch.setattr(main, "fetch", lambda s, f, l: (s, rows, 0))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    run_backtest(["BTC"], {}, {}, 25, 4, 100, None, 100, 0.01)