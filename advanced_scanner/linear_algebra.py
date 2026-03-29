"""
linear_algebra.py  ─  Vectors, Matrices, Eigenvalues, Decompositions
"""

import numpy as np
from scipy.linalg import svd, eig

def eigen_decomposition(matrix: np.ndarray):
    """Computes eigenvalues and eigenvectors of a square matrix."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for eigen decomposition.")
    vals, vecs = eig(matrix)
    return vals, vecs

def singular_value_decomposition(matrix: np.ndarray):
    """Computes SVD of a matrix (U, S, Vh)."""
    u, s, vh = svd(matrix)
    return u, s, vh

def principal_component_analysis(data: np.ndarray, n_components: int):
    """
    Performs PCA on the data.
    data: (N_samples, N_features)
    """
    # Mean centering
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Covariance matrix
    cov_matrix = np.cov(centered_data, rowvar=False)
    
    # Eigen decomposition (using eigh for symmetric matrices)
    vals, vecs = np.linalg.eigh(cov_matrix)
    
    # Sort by descending eigenvalues
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # Return first n_components
    components = vecs[:, :n_components]
    transformed = np.dot(centered_data, components)
    
    return transformed, components, vals[:n_components]

def matrix_rank(matrix: np.ndarray):
    """Returns the rank of a matrix."""
    return np.linalg.matrix_rank(matrix)

def matrix_inverse(matrix: np.ndarray):
    """Returns the inverse of a square matrix."""
    return np.linalg.inv(matrix)

def tensor_dot(a: np.ndarray, b: np.ndarray, axes=2):
    """Performs tensor dot product over specified axes."""
    return np.tensordot(a, b, axes=axes)
