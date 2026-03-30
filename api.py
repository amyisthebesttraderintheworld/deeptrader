from fastapi import Body, FastAPI, Header, HTTPException
import argparse
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from advanced_scanner.fetcher import bootstrap, fetch
from advanced_scanner.scoring import extract_features, cross_sectional_score_ranking
from advanced_scanner.config import KLINE_LIMIT, DEFAULT_CAPITAL, DEFAULT_RISK_PCT
from simulation_state import (
    get_simulation_snapshot,
    merge_simulation_state,
    reset_simulation_state,
)

# Optional: speed up bootstrap logs by disabling typewriter effect
import advanced_scanner.utils
advanced_scanner.utils.type_print = lambda text, delay=0: print(text)

app = FastAPI()


def _require_write_token(authorization: str | None) -> None:
    expected_token = os.getenv("SIM_STATE_WRITE_TOKEN")
    if not expected_token:
        return

    if authorization != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Invalid or missing simulation state bearer token")

@app.get("/md/v2/ticker/24hr/all")
def proxy_tickers():
    """Proxy endpoint to fix n8n's buggy r.result.data lookup"""
    from advanced_scanner.utils import get_json
    from advanced_scanner.config import BASE_URL
    tr_raw = get_json(BASE_URL+"/md/v3/ticker/24hr/all")
    tr = tr_raw.get("result", [])
    # This structure satisfies: tickerData = r.result?.data || r.data || [];
    return {"result": {"data": tr}}

@app.get("/scan")
def scan():
    args = argparse.Namespace(
        symbol=None, top=None, threshold=25, hold=4,
        capital=DEFAULT_CAPITAL, risk=DEFAULT_RISK_PCT, max_bars=None, offset=0,
        scan_only=True, optimize=False, sweep=False, wfo=False, estimate=False,
        meta_commentary=False
    )
    # bootstrap fetches all symbols and filters by liquidity
    syms, funds, vols = bootstrap(args)
    feature_matrices, asset_names = [], []
    
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch, s, funds, KLINE_LIMIT): s for s in syms}
        for f in as_completed(futs):
            try:
                s, rs, fund = f.result()
                if rs is not None and len(rs) >= 60:
                    # ohlcv expects [ts, interval, last_close, open, high, low, close, volume]
                    ohlcv = np.array(rs, dtype=float)
                    X, _, _, _ = extract_features(ohlcv, fund)
                    feature_matrices.append(X)
                    asset_names.append(s)
            except Exception as e:
                print(f"Error fetching/processing {futs[f]}: {e}")
                
    if not feature_matrices:
        return {"error": "No data"}
        
    ranking = cross_sectional_score_ranking(feature_matrices, asset_names)
    # Sort by rank ascending (rank 1 first)
    sorted_results = sorted(ranking.items(), key=lambda x: x[1]['rank'])
    
    # Return both scanner results AND ticker data for n8n
    from advanced_scanner.utils import get_json
    from advanced_scanner.config import BASE_URL
    tr_raw = get_json(BASE_URL+"/md/v3/ticker/24hr/all")
    tr = tr_raw.get("result", [])
    
    mapped_tickers = []
    for t in tr:
        mapped_tickers.append({
            "symbol": t.get("symbol"),
            "lastPrice": t.get("lastRp", t.get("indexRp", 0)),
            "turnover24h": t.get("turnoverRv", 0),
            "highPrice": t.get("highRp"),
            "lowRp": t.get("lowRp")
        })

    return {
        "results": [
            {
                "symbol": s, 
                "score": round(i['score'], 4), 
                "side": i['side'], 
                "rank": i['rank']
            } for s, i in sorted_results[:20]
        ],
        "tickers": mapped_tickers
    }


@app.get("/simulation/state")
def simulation_state():
    return get_simulation_snapshot()


@app.get("/simulation/positions")
def simulation_positions():
    snapshot = get_simulation_snapshot()
    return {
        "updated_at": snapshot.get("updated_at"),
        "source": snapshot.get("source"),
        "session": snapshot.get("session"),
        "positions": snapshot.get("positions", []),
        "summary": snapshot.get("summary", {}),
    }


@app.put("/simulation/state")
def upsert_simulation_state(
    payload: dict = Body(...),
    authorization: str | None = Header(default=None),
):
    _require_write_token(authorization)
    return merge_simulation_state(payload)


@app.delete("/simulation/state")
def clear_simulation_state(authorization: str | None = Header(default=None)):
    _require_write_token(authorization)
    return reset_simulation_state()

@app.get("/")
def health():
    snapshot = get_simulation_snapshot()
    return {
        "status": "ok",
        "simulation": {
            "updated_at": snapshot.get("updated_at"),
            "open_positions_count": snapshot.get("summary", {}).get("open_positions_count", 0),
            "equity": snapshot.get("summary", {}).get("equity", DEFAULT_CAPITAL),
        },
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
