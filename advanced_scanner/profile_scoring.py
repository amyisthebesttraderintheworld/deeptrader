
import time
import numpy as np
from advanced_scanner.scoring import calculate_all_scores

def profile():
    n = 500
    ohlcv = np.random.rand(n, 8)
    # Ensure some realistic values
    ohlcv[:, 3:7] = 100 + np.random.rand(n, 4)
    
    start = time.time()
    scores = calculate_all_scores(ohlcv, 0.0001)
    end = time.time()
    print(f"calculate_all_scores for {n} bars took {end - start:.4f} seconds")

if __name__ == "__main__":
    profile()
