"""Builds a fake option chain from a known smile and checks it is recovered."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import black_scholes as bs  # noqa: E402
import vol_smile  # noqa: E402

S, T, r = 100.0, 0.25, 0.03


def true_smile(K):
    """A typical equity skew: higher volatility for low strikes."""
    return 0.20 + 0.25 * (K / S - 1) ** 2 - 0.10 * (K / S - 1)


def fake_chain():
    strikes = np.arange(60.0, 141.0, 5.0)
    calls = pd.DataFrame({"strike": strikes, "mid": bs.price(S, strikes, T, r, true_smile(strikes), "call")})
    puts = pd.DataFrame({"strike": strikes, "mid": bs.price(S, strikes, T, r, true_smile(strikes), "put")})
    return calls, puts


def test_recovers_the_smile_used_to_price_the_chain():
    calls, puts = fake_chain()
    smile = vol_smile.smile_from_chain(calls, puts, S, T, r)
    assert np.allclose(smile["implied_vol"], true_smile(smile["strike"]), atol=1e-6)


def test_uses_puts_below_spot_and_calls_above():
    smile = vol_smile.smile_from_chain(*fake_chain(), S, T, r)
    assert (smile.loc[smile["kind"] == "put", "strike"] < S).all()
    assert (smile.loc[smile["kind"] == "call", "strike"] >= S).all()


def test_drops_far_strikes_bad_quotes_and_impossible_prices():
    calls, puts = fake_chain()
    puts.loc[puts["strike"] == 90.0, "mid"] = 0.0          # no quote
    puts.loc[puts["strike"] == 95.0, "mid"] = 120.0        # above the strike: impossible for a put
    smile = vol_smile.smile_from_chain(calls, puts, S, T, r)
    assert 90.0 not in set(smile["strike"]) and 95.0 not in set(smile["strike"])
    assert smile["moneyness"].between(0.7, 1.3).all()
    assert len(smile) == 11  # 70..130 in steps of 5 = 13 strikes, minus the two bad ones
