"""Cox-Ross-Rubinstein binomial tree for European and American options.

Black-Scholes only prices European options (exercise at expiry). Most US
single-stock options are American: they can be exercised at any time. A
binomial tree handles that by checking, at every node, whether exercising now
is worth more than holding on.
"""

from __future__ import annotations

import numpy as np

from black_scholes import _check_kind


def crr_price(S, K, T, r, sigma, kind="call", q=0.0, steps=500, american=False) -> float:
    """Price an option on a CRR tree with `steps` time steps.

    Each step the price moves up by u = exp(sigma * sqrt(dt)) or down by d = 1/u.
    The risk-neutral probability p of an up move makes the expected growth equal
    to r - q. Working backwards from expiry, each node is the discounted
    expected value of the two nodes after it; for an American option it is the
    larger of that and the value of exercising immediately.
    """
    _check_kind(kind)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    growth = np.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("Time step too large for these inputs: risk-neutral probability outside (0, 1).")
    discount = np.exp(-r * dt)

    def payoff(spots: np.ndarray) -> np.ndarray:
        return np.maximum(spots - K, 0.0) if kind == "call" else np.maximum(K - spots, 0.0)

    # Spot prices at expiry: S * u^j * d^(steps - j), for j = 0..steps up moves.
    j = np.arange(steps + 1)
    values = payoff(S * u**j * d ** (steps - j))
    for step in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            j = np.arange(step + 1)
            values = np.maximum(values, payoff(S * u**j * d ** (step - j)))
    return float(values[0])


def early_exercise_premium(S, K, T, r, sigma, kind="put", q=0.0, steps=500) -> float:
    """How much more the American option is worth than the European one on the same tree."""
    american = crr_price(S, K, T, r, sigma, kind, q, steps, american=True)
    european = crr_price(S, K, T, r, sigma, kind, q, steps, american=False)
    return american - european
