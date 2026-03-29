"""
stochastic_processes.py  ─  Brownian Motion, Ito Calculus, Monte Carlo
"""

import numpy as np

def brownian_motion(T: float, N: int, delta: float = 1.0):
    """
    Simulates a standard Brownian Motion (Wiener process).
    T: Total time
    N: Number of steps
    delta: Diffusion coefficient
    """
    dt = T / N
    # dW = sqrt(dt) * normal(0, 1)
    # W(t) = sum(dW_i)
    dW = np.random.normal(0, np.sqrt(dt), N)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0.0) # Start at W(0) = 0
    return W

def geometric_brownian_motion(S0: float, mu: float, sigma: float, T: float, N: int):
    """
    Simulates Geometric Brownian Motion (GBM) for price modeling.
    S0: Initial price
    mu: Drift
    sigma: Volatility
    T: Total time
    N: Number of steps
    
    S(t) = S0 * exp((mu - 0.5 * sigma^2) * t + sigma * W(t))
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    W = brownian_motion(T, N, delta=1.0)
    
    # Calculate S(t)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return S

def monte_carlo_paths(S0: float, mu: float, sigma: float, T: float, N: int, n_paths: int):
    """
    Generates multiple GBM paths for Monte Carlo analysis.
    """
    paths = np.zeros((n_paths, N + 1))
    for i in range(n_paths):
        paths[i] = geometric_brownian_motion(S0, mu, sigma, T, N)
    return paths

def jump_diffusion(S0: float, mu: float, sigma: float, T: float, N: int, lambda_j: float, mu_j: float, sigma_j: float):
    """
    Simulates Merton Jump Diffusion model.
    lambda_j: Jump intensity (Poisson process)
    mu_j: Mean log-jump size
    sigma_j: Standard deviation of log-jump size
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    W = brownian_motion(T, N, delta=1.0)
    
    # Poisson process for jumps
    n_jumps = np.random.poisson(lambda_j * dt, N)
    jumps = np.zeros(N + 1)
    
    for i in range(N):
        if n_jumps[i] > 0:
            # Sum of normal log-jumps
            jump_size = np.random.normal(mu_j, sigma_j, n_jumps[i]).sum()
            jumps[i+1] = jump_size
    
    cumulative_jumps = np.cumsum(jumps)
    
    # Adjust for mean jump compensation
    # k = E[exp(J)-1] = exp(mu_j + 0.5 * sigma_j^2) - 1
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    S = S0 * np.exp((mu - 0.5 * sigma**2 - lambda_j * k) * t + sigma * W + cumulative_jumps)
    return S
