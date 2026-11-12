"""Black-Scholes-Merton pricing, Greeks and implied volatility for European options.

All functions accept floats or NumPy arrays (they broadcast), so a whole grid of
spot prices or strikes can be priced in one call.

Notation
--------
S      spot price of the underlying
K      strike price
T      time to expiry in years
r      continuously compounded risk-free rate (0.04 = 4%)
sigma  annualised volatility (0.20 = 20%)
q      continuous dividend yield (0 for a non-dividend stock)
kind   "call" or "put"
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

KINDS = ("call", "put")


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def d1(S, K, T, r, sigma, q=0.0):
    """How many standard deviations the (forward) spot sits above the strike, adjusted for drift."""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma, q=0.0):
    return d1(S, K, T, r, sigma, q) - np.asarray(sigma) * np.sqrt(np.asarray(T))


def price(S, K, T, r, sigma, kind="call", q=0.0):
    """Black-Scholes-Merton price of a European call or put."""
    _check_kind(kind)
    D1, D2 = d1(S, K, T, r, sigma, q), d2(S, K, T, r, sigma, q)
    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)
    if kind == "call":
        return discounted_spot * norm.cdf(D1) - discounted_strike * norm.cdf(D2)
    return discounted_strike * norm.cdf(-D2) - discounted_spot * norm.cdf(-D1)


# ---------------------------------------------------------------------------
# Greeks (sensitivities of the price)
# ---------------------------------------------------------------------------

def delta(S, K, T, r, sigma, kind="call", q=0.0):
    """Change in option price for a 1-unit change in the spot price."""
    _check_kind(kind)
    D1 = d1(S, K, T, r, sigma, q)
    if kind == "call":
        return np.exp(-q * T) * norm.cdf(D1)
    return np.exp(-q * T) * (norm.cdf(D1) - 1)


def gamma(S, K, T, r, sigma, q=0.0):
    """Change in delta for a 1-unit change in spot. Same for calls and puts."""
    D1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(D1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """Change in price for a change of 1.00 (i.e. 100 percentage points) in volatility.

    Divide by 100 for the more common "per 1 vol point" quote. Same for calls and puts.
    """
    D1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(D1) * np.sqrt(T)


def theta(S, K, T, r, sigma, kind="call", q=0.0):
    """Change in price per year of passing time (usually negative). Divide by 365 for per day."""
    _check_kind(kind)
    D1, D2 = d1(S, K, T, r, sigma, q), d2(S, K, T, r, sigma, q)
    decay = -S * np.exp(-q * T) * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
    if kind == "call":
        return decay - r * K * np.exp(-r * T) * norm.cdf(D2) + q * S * np.exp(-q * T) * norm.cdf(D1)
    return decay + r * K * np.exp(-r * T) * norm.cdf(-D2) - q * S * np.exp(-q * T) * norm.cdf(-D1)


def rho(S, K, T, r, sigma, kind="call", q=0.0):
    """Change in price for a change of 1.00 in the interest rate. Divide by 100 for per 1%."""
    _check_kind(kind)
    D2 = d2(S, K, T, r, sigma, q)
    if kind == "call":
        return K * T * np.exp(-r * T) * norm.cdf(D2)
    return -K * T * np.exp(-r * T) * norm.cdf(-D2)


def greeks(S, K, T, r, sigma, kind="call", q=0.0) -> dict[str, float]:
    """All Greeks at once, with theta per day and vega/rho per 1 point, as traders usually quote them."""
    return {
        "price": float(price(S, K, T, r, sigma, kind, q)),
        "delta": float(delta(S, K, T, r, sigma, kind, q)),
        "gamma": float(gamma(S, K, T, r, sigma, q)),
        "vega_per_vol_point": float(vega(S, K, T, r, sigma, q)) / 100,
        "theta_per_day": float(theta(S, K, T, r, sigma, kind, q)) / 365,
        "rho_per_rate_point": float(rho(S, K, T, r, sigma, kind, q)) / 100,
    }


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def no_arbitrage_bounds(S, K, T, r, kind="call", q=0.0) -> tuple[float, float]:
    """Range any European option price must lie in, whatever the volatility."""
    _check_kind(kind)
    forward_spot = S * np.exp(-q * T)
    pv_strike = K * np.exp(-r * T)
    if kind == "call":
        return max(forward_spot - pv_strike, 0.0), forward_spot
    return max(pv_strike - forward_spot, 0.0), pv_strike


def implied_volatility(market_price, S, K, T, r, kind="call", q=0.0, low=1e-6, high=5.0) -> float:
    """Volatility that makes the Black-Scholes price equal the market price.

    Uses Brent's method on [low, high], which always converges when a solution
    exists (unlike plain Newton-Raphson, which can overshoot). Raises ValueError
    if the price breaks the no-arbitrage bounds, because then no volatility fits.
    """
    lower, upper = no_arbitrage_bounds(S, K, T, r, kind, q)
    if not lower < market_price < upper:
        raise ValueError(
            f"Price {market_price:.4f} is outside the no-arbitrage range "
            f"({lower:.4f}, {upper:.4f}); no implied volatility exists."
        )

    def gap(sigma: float) -> float:
        return float(price(S, K, T, r, sigma, kind, q)) - market_price

    if gap(high) < 0:
        raise ValueError(f"Implied volatility is above {high:.0%}; widen the search range.")
    if gap(low) > 0:
        raise ValueError(f"Implied volatility is below {low:.0e}.")
    return brentq(gap, low, high, xtol=1e-10)


def historical_volatility(prices, periods_per_year: int = 252) -> float:
    """Annualised standard deviation of daily log returns."""
    values = np.asarray(prices, dtype=float)
    if values.size < 3:
        raise ValueError("Need at least three prices to estimate volatility.")
    log_returns = np.diff(np.log(values))
    return float(np.std(log_returns, ddof=1) * np.sqrt(periods_per_year))
