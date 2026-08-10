import numpy as np

def transform_to_heat(r, sigma):
    k = 2 * r / sigma**2
    alpha = (1 - k) / 2
    beta = -(1 + k)**2 / 4
    return k, alpha, beta

def setup_heat(S, K, T, r, sigma, num_pts=1000, option_type='call'):
    k, alpha, beta = transform_to_heat(r, sigma)

    x_min = np.log(S / K) - 3 * sigma * np.sqrt(T)
    x_max = np.log(S / K) + 3 * sigma * np.sqrt(T)
    x = np.linspace(x_min, x_max, num_pts)

    tau_max = 0.5 * sigma**2 * T

    if option_type == 'call':
        u0 = np.maximum(np.exp(0.5 * (k + 1) * x) - np.exp(0.5 * (k - 1) * x), 0)
    else:
        u0 = np.maximum(np.exp(0.5 * (k - 1) * x) - np.exp(0.5 * (k + 1) * x), 0)

    return x, tau_max, u0, k, alpha, beta

def transform_to_black_scholes(u, x, tau, K, alpha, beta):
    return K * np.exp(alpha * x + beta * tau) * u

def get_option_price(u_final, x, tau_max, S, K, alpha, beta):
    x_curr = np.log(S / K)
    idx = np.argmin(np.abs(x - x_curr))
    return K * np.exp(alpha * x_curr + beta * tau_max) * u_final[idx]