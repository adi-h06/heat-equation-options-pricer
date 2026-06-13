# Options Pricing with Heat Equation
## Introduction

This project numerically prices European options and computes implied volatility.

The Black-Scholes PDE is transformed into the heat equation through a change of variables and solved using the finite difference method. Option prices are verified against the closed-form Black-Scholes solution. Implied volatility is computed by inverting Black-Scholes using Newton's method.

For the full mathematical derivation see [math.pdf](math.pdf)
## Results
Volatility is drawn randomly from U(0.1, 0.5) each run. Using S=$105, K=$110, T=0.5 years, r=5%, σ=37.61%:

- Call price error: 0.54%
- Put price error: 0.26%
- Put-call parity error: $0.09
- Implied volatility recovered in 3 iterations with 0.49% error

![Price vs Stock Price](results/plots/price_vs_stock.png)
![Error vs Stock Price](results/plots/error_vs_stock.png)

Note: The jagged portion on the error plot is a result of using a small number of sample points. Increasing the number of points would produce a smoother curve at the cost of computation time.

## Assumptions

Original Black-Scholes assumptions:
- Constant volatility
- Constant risk free interest rate
- Continuous trading
- No transaction fees
- No dividends
- Stock prices follow log-normally distributed returns
- European options

Numerical method assumptions:
- Finite grid truncated at ±3σ√T from the current log stock price
- Discrete approximation of the second partial derivative of the transformed option value with respect to log stock price
- Explicit finite differences, fixed boundary conditions

## How to Run

```
git clone https://github.com/adi-h06/heat-equation-options-pricer.git
pip install numpy scipy matplotlib
python main.py
```
