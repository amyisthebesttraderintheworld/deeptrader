from fastapi import FastAPI
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from advanced_scanner.fetcher import bootstrap, fetch
from advanced_scanner.scoring import extract_features, cross_sectional_score_ranking
from advanced_scanner.config import KLINE_LIMIT

# Optional: speed up bootstrap logs by disabling typewriter effect
import advanced_scanner.utils
advanced_scanner.utils.type_print = lambda text, delay=0: print(text)

app = FastAPI()

@app.get("/scan")
def scan():
    args = argparse.Namespace(
        symbol=None, top=None, threshold=25, hold=4,
        capital=100.0, risk=0.01, max_bars=None, offset=0,
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
    
    return {
        "results": [
            {
                "symbol": s, 
                "score": round(i['score'], 4), 
                "side": i['side'], 
                "rank": i['rank']
            } for s, i in sorted_results[:20]
        ]
    }

@app.get("/")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
