"""
fetcher.py  ─  Data Retrieval Logic
"""

from advanced_scanner.utils import get_json, type_print, c
from advanced_scanner.config import BASE_URL, KLINE_LIMIT, MIN_24H_TURNOVER, BRIGHT_CYAN, BRIGHT_WHITE

def fetch(sym, funds, limit=KLINE_LIMIT):
    try:
        from advanced_scanner.config import BASE_URL
        from advanced_scanner.utils import get_json
        url = BASE_URL + "/exchange/public/md/v2/kline/last"
        all_rows = []
        to_ts = None
        remaining = limit
        
        while remaining > 0:
            params = {"symbol": sym, "resolution": 3600, "limit": min(remaining, 1000)}
            if to_ts: params["to"] = to_ts
            
            k = get_json(url, params=params)
            rows = k.get("data", {}).get("rows", [])
            if not rows: break
            
            # Remove symbol string if present (usually index 8)
            clean_rows = [r[:8] for r in rows]
            
            all_rows = clean_rows + all_rows # Phemex returns newest-to-oldest, we prepend
            if len(rows) < min(remaining, 1000): break # No more data
            
            # Find oldest ts in this chunk
            oldest_ts = min(int(r[0]) for r in rows)
            to_ts = oldest_ts - 1
            remaining -= len(rows)
        
        # Final sort to ensure chronological order (Oldest -> Newest)
        all_rows.sort(key=lambda x: x[0])
        return sym, all_rows, funds.get(sym, 0.0)
    except Exception:
        return sym, [], 0.0

def bootstrap(args):
    type_print(c("Fetching symbols...", BRIGHT_CYAN))
    pd = get_json(BASE_URL+"/public/products")
    if not pd:
        type_print(c("[!] Failed to fetch products. Check your network or Phemex API status.", "\033[91m"))
        return [], {}, {}
        
    all_syms = [p["symbol"] for p in pd.get("data", {}).get("perpProductsV2", []) if p.get("quoteCurrency") == "USDT" and p.get("status") == "Listed"]
    if not all_syms:
        type_print(c("[!] No USDT perp symbols found. Phemex might be having issues or the API has changed.", "\033[91m"))
        return [], {}, {}

    print(c(f"{len(all_syms)} USDT perp symbols found", BRIGHT_WHITE))
    
    type_print(c("Fetching 24h tickers...", BRIGHT_CYAN))
    tr_raw = get_json(BASE_URL+"/md/v3/ticker/24hr/all")
    tr = tr_raw.get("result", [])
    if not tr:
        type_print(c("[!] Failed to fetch 24h tickers. Liquidity filtering might be inaccurate.", "\033[93m"))

    vols, funds = {}, {}
    for t in tr:
        s = t.get("symbol")
        if s: vols[s], funds[s] = float(t.get("turnoverRv", 0)), float(t.get("fundingRateRr", 0))
    
    if args.symbol:
        syms = [args.symbol.upper()]
        print(c(f"Single-symbol mode: {syms[0]}", BRIGHT_WHITE))
    else:
        syms = [s for s in all_syms if vols.get(s, 0) >= MIN_24H_TURNOVER]
        print(c(f"{len(syms)} symbols pass liquidity filter (>= ${MIN_24H_TURNOVER:,})", BRIGHT_WHITE))
        if not syms and all_syms:
            print(c("[!] Warning: 0 symbols passed the liquidity filter. Consider lowering MIN_24H_TURNOVER in config.py", "\033[93m"))
            
    return syms, funds, vols
