"""Download a spot price history and an option quote from Yahoo Finance (via yfinance).

Kept apart from black_scholes.py so the maths can be used and tested without
an internet connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class OptionQuote:
    expiry: date
    strike: float
    call_mid: float
    put_mid: float
    days_to_expiry: int


def _yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Market data needs yfinance: pip install yfinance") from exc
    return yf


def close_prices(ticker: str, period: str = "1y") -> pd.Series:
    """Daily closing prices, adjusted for splits and dividends."""
    data = _yfinance().download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No price data for {ticker}.")
    closes = data["Close"]
    if isinstance(closes, pd.DataFrame):  # newer yfinance versions return one column per ticker
        closes = closes.iloc[:, 0]
    return closes.dropna()


def _mid(row: pd.Series) -> float:
    """Mid of bid and ask when both exist; otherwise the last traded price."""
    bid, ask = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return float(row["lastPrice"])


def atm_option_quote(ticker: str, spot: float, min_days: int = 7) -> OptionQuote:
    """Call and put quotes at the strike nearest the spot, for the first expiry at least `min_days` away.

    Very short expiries are skipped because their prices are dominated by noise and bid-ask spreads.
    """
    tkr = _yfinance().Ticker(ticker)
    expiries = list(tkr.options or [])
    today = pd.Timestamp.today().date()
    usable = [e for e in expiries if (pd.to_datetime(e).date() - today).days >= min_days]
    if not usable:
        raise ValueError(f"No option expiries at least {min_days} days away for {ticker}.")
    expiry = usable[0]
    chain = tkr.option_chain(expiry)
    common = set(chain.calls["strike"]).intersection(chain.puts["strike"])
    if not common:
        raise ValueError(f"No strike with both a call and a put for {ticker} on {expiry}.")
    strike = min(common, key=lambda k: abs(k - spot))
    call = chain.calls.loc[chain.calls["strike"] == strike].iloc[0]
    put = chain.puts.loc[chain.puts["strike"] == strike].iloc[0]
    expiry_date = pd.to_datetime(expiry).date()
    return OptionQuote(expiry_date, float(strike), _mid(call), _mid(put), (expiry_date - today).days)
