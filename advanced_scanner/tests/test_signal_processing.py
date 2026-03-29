"""
Unit tests for signal_processing.py
"""

import unittest
import numpy as np
from advanced_scanner.signal_processing import butter_lowpass_filter, butter_highpass_filter, savitzky_golay_filter, wiener_filter, cross_correlation, convolution, detrend

class TestSignalProcessing(unittest.TestCase):
    def test_filters(self):
        t = np.linspace(0, 1, 1000)
        # Sum of low and high freq sines
        x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
        
        # Low-pass filter (cutoff at 20Hz) should remove high freq
        y_low = butter_lowpass_filter(x, cutoff=20, fs=1000)
        self.assertAlmostEqual(np.std(y_low - np.sin(2 * np.pi * 5 * t)), 0.0, delta=0.2)
        
        # High-pass filter (cutoff at 50Hz) should remove low freq
        y_high = butter_highpass_filter(x, cutoff=50, fs=1000)
        self.assertAlmostEqual(np.std(y_high - 0.5 * np.sin(2 * np.pi * 100 * t)), 0.0, delta=0.2)

    def test_smoothing(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)
        y_smooth = savitzky_golay_filter(y, window_length=11, polyorder=2)
        self.assertTrue(np.std(y_smooth - np.sin(x)) < np.std(y - np.sin(x)))

    def test_detrend(self):
        x = np.linspace(0, 10, 100)
        y = 2 * x + 5 + np.random.normal(0, 0.1, 100)
        y_detrended = detrend(y)
        self.assertAlmostEqual(np.mean(y_detrended), 0.0, places=5)

    def test_correlation_convolution(self):
        x = np.array([1, 2, 3])
        y = np.array([0, 1, 0])
        conv = convolution(x, y)
        self.assertTrue(np.allclose(conv, [0, 1, 2, 3, 0]))

if __name__ == '__main__':
    unittest.main()
