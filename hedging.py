"""Simulate delta hedging a short option, to see where the P&L comes from.

A dealer sells an option at a price based on the implied volatility, then
hedges by holding delta shares, rebalanced at fixed intervals, while the stock
actually moves with the realised volatility. Two lessons fall out:

1. Even when implied equals realised volatility, discrete rebalancing leaves a
   hedging error, which shrinks roughly with the square root of the number of
   rebalances.
2. If the option was sold at a higher volatility than the stock then realises,
   the hedged position makes money on average (and loses if it was sold too cheap).
   This is why options desks trade volatility rather than direction.
"""

from __future__ import annotations

import numpy as np

import black_scholes as bs


def simulate_paths(S, T, r, sigma, steps, paths, q=0.0, seed=0) -> np.ndarray:
    """Real-world stock paths (drift r - q for simplicity), shape (paths, steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    shocks = rng.standard_normal((paths, steps))
    log_steps = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.concatenate([np.zeros((paths, 1)), np.cumsum(log_steps, axis=1)], axis=1)
    return S * np.exp(log_paths)


def delta_hedge_pnl(
    S, K, T, r, implied_vol, realised_vol, kind="call", q=0.0, steps=52, paths=10_000, seed=0,
) -> np.ndarray:
    """Final P&L per path of selling one option at `implied_vol` and delta hedging it.

    The hedge uses the Black-Scholes delta at the implied volatility, rebalanced
    `steps` times, and cash earns or pays the risk-free rate. Dividends received
    on the shares are credited. Returns P&L in currency units at expiry.
    """
    stock = simulate_paths(S, T, r, realised_vol, steps, paths, q, seed)
    dt = T / steps
    premium = float(bs.price(S, K, T, r, implied_vol, kind, q))

    shares = bs.delta(S, K, T, r, implied_vol, kind, q) * np.ones(paths)
    cash = premium - shares * S
    for i in range(1, steps):
        time_left = T - i * dt
        spot = stock[:, i]
        cash = cash * np.exp(r * dt) + shares * stock[:, i - 1] * (np.exp(q * dt) - 1)
        new_shares = bs.delta(spot, K, time_left, r, implied_vol, kind, q)
        cash -= (new_shares - shares) * spot
        shares = new_shares
    final = stock[:, -1]
    cash = cash * np.exp(r * dt) + shares * stock[:, -2] * (np.exp(q * dt) - 1)
    payoff = np.maximum(final - K, 0.0) if kind == "call" else np.maximum(K - final, 0.0)
    return cash + shares * final - payoff
