"""
config.py  ─  Configuration and Constants
"""

BASE_URL          = "https://api.phemex.com"
MIN_24H_TURNOVER  = 2_000_000
MAX_WORKERS       = 20
KLINE_LIMIT       = 100
BT_KLINE_LIMIT    = 4000
TOP_N             = 30
TIMEOUT           = 12
TYPE_DELAY        = 0.015

MIN_TRADES_FOR_STATS = 30

DEFAULT_CAPITAL   = 100.0
DEFAULT_RISK_PCT  = 0.01  # Risk 1% of equity per trade

MAKER_FEE     = 0.0002  # 0.02% (Limit Order)
TAKER_FEE     = 0.0006  # 0.06% (Market/SL/TP/Liq)
BASE_SLIPPAGE = 0.0005  # 0.05% Base Slippage (Exits)
FUNDING_INTERVAL = 8 * 3600  # 8 hours in seconds
ORDER_EXPIRE_BARS = 4   # Hours until a limit order is cancelled

# ── ANSI COLORS ────────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; CYAN = "\033[36m"
BRIGHT_RED = "\033[91m"; BRIGHT_GREEN = "\033[92m"; BRIGHT_YELLOW = "\033[93m"
BRIGHT_CYAN = "\033[96m"; BRIGHT_WHITE = "\033[97m"
MAGENTA = "\033[35m"; BRIGHT_MAGENTA = "\033[95m"; BLUE = "\033[34m"
BRIGHT_BLUE = "\033[94m"
