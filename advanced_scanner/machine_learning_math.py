"""
machine_learning_math.py  ─  PCA, SVD, Dimensionality Reduction, Kernel Math
"""

import numpy as np
from typing import List, Union

def soft_max(x: np.ndarray):
    """Computes the softmax activation for a vector."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x: Union[float, np.ndarray]):
    """Computes the sigmoid activation."""
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray):
    """Computes the ReLU activation."""
    return np.maximum(0, x)

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray):
    """Computes cross-entropy loss."""
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def kernel_matrix(X: np.ndarray, Y: np.ndarray = None, kernel: str = 'rbf', gamma: float = 0.1):
    """
    Computes a kernel matrix (Gram matrix) for the input data.
    """
    if Y is None:
        Y = X
    
    if kernel == 'rbf':
        # dist(x,y)^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
        norm_x = np.sum(X**2, axis=1).reshape(-1, 1)
        norm_y = np.sum(Y**2, axis=1).reshape(1, -1)
        dist_sq = norm_x + norm_y - 2 * np.dot(X, Y.T)
        return np.exp(-gamma * dist_sq)
    
    elif kernel == 'linear':
        return np.dot(X, Y.T)
    
    elif kernel == 'poly':
        # (gamma * x^T * y + r)^d (using default degree 3, r=1)
        return (gamma * np.dot(X, Y.T) + 1)**3
    
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

def distance_metric(x: np.ndarray, y: np.ndarray, metric: str = 'euclidean'):
    """Computes distance between two vectors."""
    if metric == 'euclidean':
        return np.linalg.norm(x - y)
    elif metric == 'manhattan':
        return np.sum(np.abs(x - y))
    elif metric == 'cosine':
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def covariance_matrix(X: np.ndarray):
    """Computes the covariance matrix of features in X."""
    return np.cov(X, rowvar=False)

def correlation_matrix(X: np.ndarray):
    """Computes the correlation matrix of features in X."""
    return np.corrcoef(X, rowvar=False)
