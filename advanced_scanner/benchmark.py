
import time
import numpy as np
from main import run_backtest
from advanced_scanner.config import BT_KLINE_LIMIT, DEFAULT_CAPITAL, DEFAULT_RISK_PCT

def benchmark():
    # Mock some data to avoid fetching from network
    # (Though fetching is also part of the process, it's IO-bound)
    # For now, let's just run the actual thing but for a limited set to see
    
    start_time = time.time()
    
    # We'll use --top 5 to limit the symbols
    # We'll use --max-bars 500 to limit the history
    import subprocess
    cmd = ["python3", "main.py", "--top", "5", "--max-bars", "500", "--threshold", "25", "--hold", "4"]
    print(f"Running benchmark: {' '.join(cmd)}")
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")
    # print(res.stdout)
    if res.stderr:
        print("Errors during benchmark:")
        print(res.stderr)

if __name__ == "__main__":
    benchmark()
