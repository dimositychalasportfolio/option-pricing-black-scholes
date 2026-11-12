"""Monte Carlo pricing of European options, with two variance-reduction methods.

Monte Carlo is slow for plain European options (Black-Scholes has a formula),
but it is the method of choice for path-dependent or multi-asset products. Here
it is checked against the formula, which makes the effect of variance reduction
easy to measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from black_scholes import _check_kind


@dataclass
class MCResult:
    price: float
    std_error: float
    paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def terminal_prices(S, T, r, sigma, q, normals: np.ndarray) -> np.ndarray:
    """Stock prices at T under the risk-neutral measure (geometric Brownian motion)."""
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * normals)


def mc_price(
    S, K, T, r, sigma, kind="call", q=0.0, paths=100_000, seed=0,
    antithetic=False, control_variate=False,
) -> MCResult:
    """Discounted average payoff over simulated terminal prices.

    antithetic: pair every draw z with -z. The two payoffs are negatively
        correlated, so their average varies less than two independent draws.
    control_variate: use the terminal stock price, whose discounted expectation
        is known exactly (S * exp(-qT)), to correct the payoff estimate:
        payoff_adj = payoff - b * (stock - E[stock]), with b chosen to minimise variance.
    """
    _check_kind(kind)
    rng = np.random.default_rng(seed)
    if antithetic:
        half = rng.standard_normal(paths // 2)
        normals = np.concatenate([half, -half])
    else:
        normals = rng.standard_normal(paths)
    stock = terminal_prices(S, T, r, sigma, q, normals)
    payoff = np.maximum(stock - K, 0.0) if kind == "call" else np.maximum(K - stock, 0.0)
    discounted = np.exp(-r * T) * payoff
    control = np.exp(-r * T) * stock

    if antithetic:
        # Average each pair first: the pairs, not the single draws, are independent.
        n = len(discounted) // 2
        samples = 0.5 * (discounted[:n] + discounted[n:])
        control = 0.5 * (control[:n] + control[n:])
    else:
        samples = discounted

    if control_variate:
        # Regress on the same samples that are averaged, so b is optimal for them.
        expected = S * np.exp(-q * T)
        b = np.cov(samples, control, ddof=1)[0, 1] / np.var(control, ddof=1)
        samples = samples - b * (control - expected)

    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples))), len(normals))
