"""Runs the command-line script with fake market data, so no internet is needed."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import black_scholes as bs  # noqa: E402
import market_data  # noqa: E402
import price_option  # noqa: E402


def test_script_reports_implied_against_historical_volatility(monkeypatch, capsys):
    rng = np.random.default_rng(3)
    closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, 252))))
    spot = float(closes.iloc[-1])
    T = 30 / 365
    # Quotes priced at 35% volatility, so the implied volatility should come back as 35%.
    call = float(bs.price(spot, round(spot), T, 0.04, 0.35, "call"))
    put = float(bs.price(spot, round(spot), T, 0.04, 0.35, "put"))
    quote = market_data.OptionQuote(date(2027, 1, 15), float(round(spot)), call, put, 30)

    monkeypatch.setattr(market_data, "close_prices", lambda ticker, period: closes)
    monkeypatch.setattr(market_data, "atm_option_quote", lambda ticker, spot: quote)
    monkeypatch.setattr(sys, "argv", ["price_option.py", "test"])
    price_option.main()

    out = capsys.readouterr().out
    assert "TEST" in out
    assert out.count("35.0%") == 4  # implied vol, twice per option (table and sentence)
    assert "the market prices more volatility" in out
    assert "cheap" not in out and "expensive" not in out
