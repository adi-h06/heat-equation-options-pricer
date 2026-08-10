import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from heat_equation import setup, solve, plot_solution
from black_scholes import black_scholes_call, black_scholes_put, check_pcp
from black_scholes_transform import setup_heat, transform_to_heat, get_option_price

def _solve_heat(u0, x, tau_max, S, K, alpha, beta, num_pts, dt):
    dx = (x[-1] - x[0]) / num_pts
    if dt is None:
        dt = 0.4 * dx**2
    num_steps = int(tau_max / dt)
    final_u = solve(u0, alpha=1.0, dt=dt, dx=dx, num_steps=num_steps, keep_history=False)[-1]
    return get_option_price(final_u, x, tau_max, S, K, alpha, beta)

def price_european_call(S, K, T, r, sigma, num_pts=1000, dt=None):
    x, tau_max, u0, _, alpha, beta = setup_heat(S, K, T, r, sigma, num_pts, option_type='call')
    return _solve_heat(u0, x, tau_max, S, K, alpha, beta, num_pts, dt)

def verify_call_against_analytical(S, K, T, r, sigma, num_pts=1000):
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
    x, tau_max, u0, k, alpha, beta = setup_heat(S, K, T, r, sigma, num_pts, option_type='put')
    return _solve_heat(u0, x, tau_max, S, K, alpha, beta, num_pts, dt)

def verify_put_against_analytical(S, K, T, r, sigma, num_pts=1000):
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
    call = price_european_call(S, K, T, r, sigma)
    put = price_european_put(S, K, T, r, sigma)

    error = check_pcp(S, K, T, r, call, put)

    print(f'Put call parity error: ${error:.6f}')

    return error

def plot_price_vs_stock(K, T, r, sigma, S_min=60, S_max=140, num_pts=1000):
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