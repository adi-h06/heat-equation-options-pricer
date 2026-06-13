import numpy as np
from scipy.stats import norm

def _compute_d1_d2(S, K, T, r, sigma):
    """
    Computes d1 and d2 for the Black-Scholes formula.
    
    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry in years
    r: risk free interest rate
    sigma: volatility
    
    Returns:
    d1, d2: Black-Scholes parameters
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes_call(S, K, T, r, sigma):
    """
    Evaluates the closed form Black-Scholes formula for the call price.

    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry in years
    r: risk free interest rate
    sigma: volatility
    
    Returns:
    call_px: call option price 

    """
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    call_px = S * norm.cdf(d1) - K * np.exp(-r * T)  * norm.cdf(d2)

    return call_px

def black_scholes_put(S, K, T, r, sigma):
    """
    Evaluates the closed form Black-Scholes formula for the put price.

    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry in years
    r: risk free interest rate
    sigma: volatility
    
    Returns:
    put_px: put option price 

    """
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    put_px = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_px

def check_pcp(S, K, T, r, call_px, put_px):
    """
    Checks put-call parity.

    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry in years
    r: risk free interest rate
    call_px: call option price
    put_px: put option price

    Returns:
    error: returns the difference between the two sides of the put-call parity equation
     
    """
    left = call_px - put_px
    right = S - K * np.exp(-r * T)
    error = abs(left - right)
    return error

def vega(S, K, T, r, sigma):
    """
    Calculates the vega of the call option.
    
    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry in years
    r: risk free interest rate
    sigma: volatility
    
    Returns:
    vega: rate of change of option price w.r.t. volatility
    
    """
    d1, _ = _compute_d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility(market_px, S, K, T, r, sigma_guess=0.2, tol=1e-6, max_iter=100):
    """
    Calculates the implied volatility using Newton's method.
    Raises ValueError if it did not converge after the maximum number
    of iterations. 
    
    Parameters:
    market_px: observed option price
    S: current stock price
    K: strike price
    T: time to expiry in years
    r: risk free interest rate
    sigma_guess: initial volatility guess
    tol: tolerance
    max_iter: maximum number of iterations
    
    Returns:
    sigma: implied volatility
    iter: number of iterations to converge
    
    """
    sigma = sigma_guess

    for i in range(max_iter):
        px = black_scholes_call(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma)

        diff = px - market_px

        if abs(diff) < tol:
            return sigma, i + 1
        
        sigma = sigma - diff / v
    
    raise ValueError(f'Implied volatility failed to converge after {max_iter} iterations.')
