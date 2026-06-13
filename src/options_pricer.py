import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from heat_equation import setup, solve, plot_solution
from black_scholes import black_scholes_call, black_scholes_put, check_pcp
from black_scholes_transform import setup_heat, transform_to_heat, get_option_price

def _solve_heat(u0, x, tau_max, S, K, alpha, beta, num_pts, dt):
    """
    Helper function that runs the heat equation solver and
    returns option price at current stock price S.

    Parameters:
    u0: initial condition in transformed variables
    x: spatial grid in log stock price space
    tau_max: transformed time corresponding to present time
    S: current stock price
    K: strike price
    alpha: transformation constant
    beta: transformation constant
    num_pts: number of spatial grid points
    dt: time step size, computed automatically if None

    Returns:
    price: option price at current stock price S
    """
    dx = (x[-1] - x[0]) / num_pts
    if dt is None:
        dt = 0.4 * dx**2
    num_steps = int(tau_max / dt)
    all_u = solve(u0, alpha=1.0, dt=dt, dx=dx, num_steps=num_steps)
    return get_option_price(all_u[-1], x, tau_max, S, K, alpha, beta)

def price_european_call(S, K, T, r, sigma, num_pts=1000, dt=None):
    """
    Prices a European call option numerically by solving the Black-Scholes
    PDE via its equivalence to the heat equation.

    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry
    r: risk free interest rate
    sigma: volatility
    num_pts: number of spatial grid points
    dt: time step size, computed automatically if None

    Returns:
    price: numerical call option price
    """
    x, tau_max, u0, _, alpha, beta = setup_heat(S, K, T, r, sigma, num_pts)
        
    return _solve_heat(u0, x, tau_max, S, K, alpha, beta, num_pts, dt)

def verify_call_against_analytical(S, K, T, r, sigma,num_pts=1000):
    """
    Compares the numerical call price against the analytical 
    Black-Scholes formula and prints the results.

    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry
    r: risk free interest rate
    sigma: volatility
    num_pts: number of spatial grid points

    Returns:
    numerical_price: numerical call option price
    analytical_price: analytical call option price
    error: absolute difference between numerical and analytical prices
    """
    numerical_price = price_european_call(S, K, T, r, sigma, num_pts=num_pts)
    analytical_price = black_scholes_call(S, K, T, r, sigma)
    
    error = abs(numerical_price - analytical_price)
    error_pct = error / analytical_price * 100
    
    print(f'Analytical call price:  ${analytical_price:.4f}')
    print(f'Numerical call price:   ${numerical_price:.4f}')
    print(f'Absolute error:    ${error:.4f}')
    print(f'Percentage error:  {error_pct:.4f}%')
    
    return numerical_price, analytical_price, error

def price_european_put(S, K, T, r, sigma, num_pts=1000, dt=None):
    """
    Prices a European put option numerically by solving the Black-Scholes
    PDE via its equivalence to the heat equation.

    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry
    r: risk free interest rate
    sigma: volatility
    num_pts: number of spatial grid points
    dt: time step size, computed automatically if None

    Returns:
    price: numerical put option price
    """
    x, tau_max, u0, k, alpha, beta = setup_heat(S, K, T, r, sigma, num_pts)
    u0 = np.maximum(np.exp(0.5 * (k - 1) * x) - np.exp(0.5 * (k + 1) * x), 0)
    return _solve_heat(u0, x, tau_max, S, K, alpha, beta, num_pts, dt)

def verify_put_against_analytical(S, K, T, r, sigma,num_pts=1000):
    """
    Compares the numerical put price against the analytical 
    Black-Scholes formula and prints the results.

    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry
    r: risk free interest rate
    sigma: volatility
    num_pts: number of spatial grid points

    Returns:
    numerical_price: numerical put option price
    analytical_price: analytical put option price
    error: absolute difference between numerical and analytical prices
    """
    numerical_price = price_european_put(S, K, T, r, sigma, num_pts=num_pts)
    analytical_price = black_scholes_put(S, K, T, r, sigma)
    
    error = abs(numerical_price - analytical_price)
    error_pct = error / analytical_price * 100
    
    print(f'Analytical put price:  ${analytical_price:.4f}')
    print(f'Numerical put price:   ${numerical_price:.4f}')
    print(f'Absolute error:    ${error:.4f}')
    print(f'Percentage error:  {error_pct:.4f}%')
    
    return numerical_price, analytical_price, error

def verify_pcp(S, K, T, r, sigma):
    """
    Verifies put-call parity.

    Parameters:
    S: current stock price
    K: strike price
    T: time to expiry
    r: risk free interest rate
    sigma: volatility

    Returns:
    error: put-call parity error in dollars
    """
    call = price_european_call(S, K, T, r, sigma)
    put = price_european_put(S, K, T, r, sigma)
    
    error = check_pcp(S, K, T, r, call, put)
    
    print(f'Put call parity error: ${error:.6f}')
    
    return error

def plot_price_vs_stock(K, T, r, sigma, S_min=60, S_max=140, num_pts=1000):
    """
    Plots numerical and analytical call option prices across
    a range of stock prices

    Parameters:
    K: strike price
    T: time to expiry
    r: risk free interest rate
    sigma: volatility
    S_min: minimum stock price
    S_max: maximum stock price
    num_pts: number of spatial grid points
    """
    stock_prices = np.linspace(S_min, S_max, 20)
    numerical_prices = []
    analytical_prices = []
    
    for S in stock_prices:
        numerical_prices.append(price_european_call(S, K, T, r, sigma, num_pts=num_pts))
        analytical_prices.append(black_scholes_call(S, K, T, r, sigma))
    
    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, analytical_prices, label='Analytical', linewidth=2)
    plt.plot(stock_prices, numerical_prices, '--', label='Numerical', linewidth=2)
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Call Option Price ($)')
    plt.title('European Call Option Price vs Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/plots/price_vs_stock.png')
    plt.show()

def plot_error_vs_stock(K, T, r, sigma, S_min=60, S_max=140, num_pts=1000):
    """
    Plots absolute error between numerical and analytical call option prices across
    a range of stock prices

    Parameters:
    K: strike price
    T: time to expiry
    r: risk free interest rate
    sigma: volatility
    S_min: minimum stock price
    S_max: maximum stock price
    num_pts: number of spatial grid points
    """
    stock_prices = np.linspace(S_min, S_max, 20)
    errors = []
    
    for S in stock_prices:
        numerical = price_european_call(S, K, T, r, sigma, num_pts=num_pts)
        analytical = black_scholes_call(S, K, T, r, sigma)
        errors.append(abs(numerical - analytical))
    
    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, errors, linewidth=2)
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Absolute Error ($)')
    plt.title('Numerical Error vs Stock Price')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/plots/error_vs_stock.png')
    plt.show()
