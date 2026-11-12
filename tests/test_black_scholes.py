"""Checks against textbook values, put-call parity and numerical derivatives.

Run with: python -m pytest
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import black_scholes as bs  # noqa: E402

# Standard example from Hull, "Options, Futures, and Other Derivatives" (Black-Scholes-Merton chapter):
# S=42, K=40, r=10%, sigma=20%, T=0.5 gives call 4.76 and put 0.81.
HULL = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)


def test_matches_hull_textbook_example():
    assert bs.price(**HULL, kind="call") == pytest.approx(4.76, abs=0.005)
    assert bs.price(**HULL, kind="put") == pytest.approx(0.81, abs=0.005)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    S, K, T, r, sigma = 100.0, 95.0, 0.75, 0.04, 0.3
    call = bs.price(S, K, T, r, sigma, "call", q)
    put = bs.price(S, K, T, r, sigma, "put", q)
    assert call - put == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_greeks_match_finite_differences(kind, q):
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.25
    h = 1e-4

    def p(**changes):
        args = dict(S=S, K=K, T=T, r=r, sigma=sigma, kind=kind, q=q)
        args.update(changes)
        return float(bs.price(**args))

    assert bs.delta(S, K, T, r, sigma, kind, q) == pytest.approx((p(S=S + h) - p(S=S - h)) / (2 * h), rel=1e-5)
    assert bs.gamma(S, K, T, r, sigma, q) == pytest.approx(
        (p(S=S + h) - 2 * p() + p(S=S - h)) / h**2, rel=1e-3)
    assert bs.vega(S, K, T, r, sigma, q) == pytest.approx(
        (p(sigma=sigma + h) - p(sigma=sigma - h)) / (2 * h), rel=1e-5)
    # Theta is the change as time passes, i.e. as T falls.
    assert bs.theta(S, K, T, r, sigma, kind, q) == pytest.approx(
        -(p(T=T + h) - p(T=T - h)) / (2 * h), rel=1e-5)
    assert bs.rho(S, K, T, r, sigma, kind, q) == pytest.approx((p(r=r + h) - p(r=r - h)) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("true_sigma", [0.05, 0.2, 0.8, 2.0])
def test_implied_volatility_recovers_the_input(kind, true_sigma):
    S, K, T, r = 100.0, 110.0, 0.4, 0.02
    market = float(bs.price(S, K, T, r, true_sigma, kind))
    assert bs.implied_volatility(market, S, K, T, r, kind) == pytest.approx(true_sigma, abs=1e-6)


def test_implied_volatility_rejects_impossible_prices():
    with pytest.raises(ValueError, match="no-arbitrage"):
        bs.implied_volatility(150.0, S=100, K=100, T=1, r=0.0, kind="call")  # above the spot
    with pytest.raises(ValueError, match="no-arbitrage"):
        bs.implied_volatility(1.0, S=120, K=100, T=1, r=0.0, kind="call")  # below intrinsic value


def test_functions_work_on_arrays():
    spots = np.array([80.0, 100.0, 120.0])
    prices = bs.price(spots, 100.0, 0.5, 0.03, 0.2, "call")
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) > 0)  # calls are worth more as the spot rises


def test_historical_volatility_of_a_known_series():
    rng = np.random.default_rng(0)
    daily = 0.01
    prices = 100 * np.exp(np.cumsum(rng.normal(0, daily, 20_000)))
    assert bs.historical_volatility(prices) == pytest.approx(daily * np.sqrt(252), rel=0.02)


def test_unknown_option_kind_is_rejected():
    with pytest.raises(ValueError):
        bs.price(**HULL, kind="straddle")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_dividend_yield_equals_pricing_on_a_reduced_spot(kind):
    """With yield q, the option is worth the same as one on a spot of S * exp(-qT) with no yield."""
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.25, 0.04
    with_yield = bs.price(S, K, T, r, sigma, kind, q)
    reduced_spot = bs.price(S * np.exp(-q * T), K, T, r, sigma, kind, 0.0)
    assert with_yield == pytest.approx(reduced_spot, abs=1e-10)
