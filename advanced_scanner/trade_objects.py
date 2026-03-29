"""
trade_objects.py  ─  Trading Data Structures
"""

from advanced_scanner.config import MAKER_FEE, TAKER_FEE, BASE_SLIPPAGE, ORDER_EXPIRE_BARS
from advanced_scanner.scoring import score_to_leverage, calculate_position_size

class Order:
    __slots__ = ("sym", "direction", "placed_ts", "placed_bar", "price",
                 "signal_score", "margin", "sl_dist", "tp_dist", "expires_bar", "vol_ratio", "daily_vol", "asset_volatility", "capital")
    def __init__(self, sym, direction, placed_bar, placed_ts, price,
                 signal_score, asset_volatility, capital, sl_dist, tp_dist, vol_ratio, daily_vol):
        self.sym, self.direction, self.placed_bar, self.placed_ts = sym, direction, placed_bar, placed_ts
        self.price, self.signal_score = price, signal_score
        self.asset_volatility = asset_volatility
        self.capital = capital
        # Use volatility-normalized position sizing with portfolio-level risk control
        # (Assume signal_score is already thresholded and smoothed)
        self.margin = calculate_position_size(signal_score, capital, asset_volatility)
        # Optionally, add further portfolio-level checks in order creation pipeline
        self.sl_dist, self.tp_dist, self.vol_ratio = sl_dist, tp_dist, vol_ratio
        self.daily_vol = daily_vol
        self.expires_bar = placed_bar + ORDER_EXPIRE_BARS

class Trade:
    __slots__ = ("sym", "direction", "entry_bar", "entry_ts", "entry_price", "exit_bar", "exit_ts", "exit_price",
                 "signal_score", "leverage", "margin", "pnl_pct", "pnl_usd", "liquidated", "is_open",
                 "entry_fee", "exit_fee", "funding_accumulated", "sl_price", "tp_price", "is_maker", "daily_vol")
    def __init__(self, order, entry_price, entry_bar, entry_ts, is_maker=True):
        self.sym, self.direction, self.entry_bar, self.entry_ts = order.sym, order.direction, entry_bar, entry_ts
        self.entry_price, self.signal_score = entry_price, order.signal_score
        self.margin = order.margin
        self.leverage = score_to_leverage(order.signal_score)
        self.is_maker, self.daily_vol = is_maker, order.daily_vol
        if self.direction == "LONG":
            self.sl_price = self.entry_price - order.sl_dist
            self.tp_price = self.entry_price + order.tp_dist
        else:
            self.sl_price = self.entry_price + order.sl_dist
            self.tp_price = self.entry_price - order.tp_dist
        ef = MAKER_FEE if is_maker else TAKER_FEE
        self.entry_fee = (self.margin * self.leverage) * ef
        self.exit_fee = 0.0; self.funding_accumulated = 0.0
        self.is_open = True; self.liquidated = False

    def update_funding(self, rate):
        p = (self.margin * self.leverage) * rate
        if self.direction == "LONG": self.funding_accumulated -= p
        else: self.funding_accumulated += p

    def close(self, bar_idx, ts, px, force_liq=False):
        self.exit_bar, self.exit_ts, self.exit_price = bar_idx, ts, px
        self.is_open = False; self.liquidated = force_liq
        
        notional = self.margin * self.leverage
        market_impact = 0.0
        if self.daily_vol > 0:
            market_impact = 0.1 * (notional / self.daily_vol)**0.5
        
        slip = 0.0 if force_liq else (BASE_SLIPPAGE + (abs(px-self.entry_price)/self.entry_price * 0.1) + market_impact)
        if not force_liq:
            self.exit_price = px * (1-slip) if self.direction=="LONG" else px * (1+slip)
        self.exit_fee = (self.margin * self.leverage) * TAKER_FEE
        raw = (self.exit_price - self.entry_price)/self.entry_price if self.direction=="LONG" else (self.entry_price - self.exit_price)/self.entry_price
        self.pnl_usd = (raw * self.margin * self.leverage) - self.entry_fee - self.exit_fee + self.funding_accumulated
        if self.pnl_usd <= -self.margin or force_liq:
            self.pnl_usd = -self.margin; self.liquidated = True
        self.pnl_pct = (self.pnl_usd / self.margin) * 100

    @property
    def notional(self): return self.margin * self.leverage
    @property
    def capital_returned(self): return self.margin + self.pnl_usd
