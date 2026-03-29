"""
signal_processing.py  ─  Filters, Wavelets, FFTs
"""

import numpy as np
from scipy import signal
from typing import List, Union

def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5):
    """Applies a zero-phase Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y

def butter_highpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5):
    """Applies a zero-phase Butterworth high-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y

def savitzky_golay_filter(data: np.ndarray, window_length: int = 5, polyorder: int = 2):
    """Applies a Savitzky-Golay filter for smoothing."""
    return signal.savgol_filter(data, window_length, polyorder)

def wiener_filter(data: np.ndarray, mysize: int = None, noise: float = None):
    """Applies a Wiener filter for noise reduction."""
    return signal.wiener(data, mysize=mysize, noise=noise)

def cross_correlation(x: np.ndarray, y: np.ndarray):
    """Computes cross-correlation between two signals."""
    return signal.correlate(x, y, mode='full')

def convolution(x: np.ndarray, y: np.ndarray):
    """Computes discrete convolution of two signals."""
    return signal.convolve(x, y, mode='full')

def detrend(data: np.ndarray):
    """Removes linear trend from data."""
    return signal.detrend(data)
