import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from options_pricer import price_european_call, verify_call_against_analytical, verify_put_against_analytical, verify_pcp, plot_price_vs_stock, plot_error_vs_stock
from black_scholes import implied_volatility

def main():
    S = 105.0
    K = 110.0
    T = 0.5
    r = 0.05
    sigma = np.random.uniform(0.1, 0.5)

    print('=' * 50)
    print('European Call Option Pricing')
    print('=' * 50)
    print(f'Stock Price:    ${S}')
    print(f'Strike Price:   ${K}')
    print(f'Time to Expiry: {T} year')
    print(f'Interest Rate:  {r*100}%')
    print(f'Volatility:     {sigma*100:.2f}%')
    print('=' * 50)

    print('\nCall Option:')
    verify_call_against_analytical(S, K, T, r, sigma)

    print('\nPut Option:')
    verify_put_against_analytical(S, K, T, r, sigma)

    print('\nPut-Call Parity Check:')
    verify_pcp(S, K, T, r, sigma)

    print('\nImplied Volatility:')
    market_px = price_european_call(S, K, T, r, sigma)
    sigma_implied, iterations = implied_volatility(market_px, S, K, T, r)
    print(f'True volatility:     {sigma:.4f}')
    print(f'Implied volatility:  {sigma_implied:.4f}')
    print(f'Error:               {abs(sigma - sigma_implied):.6f}')
    print(f'Iterations:          {iterations}')

    plot_price_vs_stock(K, T, r, sigma, S_min=70, S_max=150)
    plot_error_vs_stock(K, T, r, sigma, S_min=70, S_max=150)

if __name__ == '__main__':
    main()