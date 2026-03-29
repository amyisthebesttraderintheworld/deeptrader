"""
Unit tests for time_series_analysis.py
"""

import unittest
import numpy as np
from advanced_scanner.time_series_analysis import autocorrelation, partial_autocorrelation, spectral_density, fft_analysis, moving_average_volatility

class TestTimeSeriesAnalysis(unittest.TestCase):
    def test_autocorrelation(self):
        # Random walk
        n = 1000
        x = np.cumsum(np.random.normal(0, 1, n))
        self.assertAlmostEqual(autocorrelation(x, 1), 1.0, delta=0.1)

    def test_partial_autocorrelation(self):
        # AR(1) process: x_t = phi * x_{t-1} + e_t
        phi = 0.5
        n = 1000
        e = np.random.normal(0, 1, n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t-1] + e[t]
        
        # PACF at lag 1 should be approx phi
        pacf1 = partial_autocorrelation(x, 1)
        self.assertAlmostEqual(pacf1, phi, delta=0.2)
        
        # PACF at lag 2 should be approx 0
        pacf2 = partial_autocorrelation(x, 2)
        self.assertAlmostEqual(pacf2, 0.0, delta=0.2)

    def test_fft_analysis(self):
        # Sum of two sines
        n = 1000
        fs = 1000 # Use 1000 points over 1 second, fs=1000
        t = np.linspace(0, 1, n, endpoint=False) # Exactly 1 second
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        freqs, psd = fft_analysis(x, fs=fs)
        # Check if 50Hz and 120Hz are in freqs
        found_50 = any(np.isclose(freqs, 50, atol=2))
        found_120 = any(np.isclose(freqs, 120, atol=2))
        self.assertTrue(found_50)
        self.assertTrue(found_120)

    def test_moving_average_volatility(self):
        prices = np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
        vol = moving_average_volatility(prices, window=10)
        self.assertEqual(len(vol), 99)
        self.assertTrue(np.all(vol[10:] >= 0))

if __name__ == '__main__':
    unittest.main()
