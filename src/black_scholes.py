import numpy as np
from scipy.special import ndtr

def _norm_cdf(x):
    return ndtr(x)

def _norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def _compute_d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes_call(S, K, T, r, sigma):
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    return S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def check_pcp(S, K, T, r, call_px, put_px):
    left = call_px - put_px
    right = S - K * np.exp(-r * T)
    return abs(left - right)

def vega(S, K, T, r, sigma):
    d1, _ = _compute_d1_d2(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * np.sqrt(T)

def implied_volatility(market_px, S, K, T, r, sigma_guess=0.2, tol=1e-6, max_iter=100):
    sigma = sigma_guess
    for i in range(max_iter):
        px = black_scholes_call(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)
        diff = px - market_px
        if abs(diff) < tol:
            return sigma, i + 1
        sigma = sigma - diff / v
    raise ValueError(f'Implied volatility failed to converge after {max_iter} iterations.')