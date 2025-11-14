# Option Pricing with the Black–Scholes Model

This project implements the Black–Scholes model in Python to price European call and put options, 
compute option Greeks, estimate historical volatility from market data, and recover implied volatility 
from market option prices. It also includes a Jupyter notebook with visualizations.

---

## Overview

The project shows how to:

- Use the **Black–Scholes formula** for European options  
- Fetch **real market data** (e.g. AAPL) with `yfinance`  
- Compute **historical volatility** from price returns  
- Calculate the main **Greeks**: Delta, Gamma, Vega, Theta, Rho  
- Solve for **implied volatility** using a Newton–Raphson method  
- Visualize option prices and Greeks in a **Jupyter notebook**

This is designed as a quant-style portfolio project.

---

## Project Structure

```text
option-pricing-black-scholes/
│
├─ black_scholes.py             # core Black–Scholes implementation + Greeks + IV solver
├─ notebooks/
│   └─ option_pricing_demo.ipynb  # notebook with plots and examples
├─ requirements.txt             # Python dependencies
└─ README.md                    # project documentation
