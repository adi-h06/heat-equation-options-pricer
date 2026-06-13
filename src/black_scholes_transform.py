import numpy as np

def transform_to_heat(r, sigma):
    """
    Computes transformation constants to be used in the conversion
    between the Black-Scholes equation and the heat equation.

    Parameters:
    r: risk free interest rate
    sigma: volatility

    Returns:
    k: ratio of interest rate and variance
    alpha: constant chosen to eliminate first order term
    beta: constant chosen to eliminate zeroth order term
    """
    k = 2 * r / sigma**2
    alpha = (1 - k) / 2
    beta = -(1 + k)**2 / 4
    return k, alpha, beta

def setup_heat(S, K, T, r, sigma, num_pts=1000):
    """
    Constructs a spatial grid using log stock prices, computes
    transformed time variable, and evaluates the initial condition
    to find the call option payoff at expiry expressed in transformed variables.

    The spatial grid is centered at the current log stock price
    and extends 3 standard deviations in both directions to capture 99.7%
    of possible stock prices. 

    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry in years
    r: risk free interest rate
    sigma: volatility
    num_pts: number of spatial grid points

    Returns: 
    x: spatial grid in log stock price space
    tau_max: transformed time (corresponds to present time)
    u0: call payoff in transformed variables
    k: interest rate ratio
    alpha: transformation constant
    beta: transformation constant
    """
    k, alpha, beta = transform_to_heat(r, sigma)

    x_min = np.log(S / K) - 3 * sigma * np.sqrt(T)
    x_max = np.log(S / K) + 3 * sigma * np.sqrt(T)
    x = np.linspace(x_min, x_max, num_pts)

    tau_max = 0.5 * sigma**2 * T

    u0 = np.maximum(np.exp(0.5 * (k + 1) * x) - np.exp(0.5 * (k - 1) * x), 0)

    return x, tau_max, u0, k, alpha, beta

def transform_to_black_scholes(u, x, tau, K, alpha, beta):
    """
    Applies an inverse transformation to convert the heat equation
    solution back into option prices.

    Parameters:
    u: heat equation solution array
    x: spatial grid in log stock price space
    tau: current time
    K: strike price
    alpha: transformation constant
    beta: transformation constant

    Returns:
    option prices corresponding to each grid point
    """
    return K * np.exp(alpha * x + beta * tau) * u

def get_option_price(u_final, x, tau_max, S, K, alpha, beta):
    """
    Extracts the option price at the current stock price S
    from the solution array.

    Finds the grid point in x closest to ln(S/K) and applies
    the inverse transformation to return the option price.
    
    Parameters:
    u_final: heat equation solution at tau_max
    x: spatial grid in log stock price space
    tau_max: current time
    S: current stock price
    K: strike price
    alpha: transformation constant
    beta: transformation constant

    Returns:
    price: option price at current stock price 
    """
    x_curr = np.log(S / K)

    idx = np.argmin(np.abs(x - x_curr))

    price = K * np.exp(alpha * x_curr + beta * tau_max) * u_final[idx]

    return price