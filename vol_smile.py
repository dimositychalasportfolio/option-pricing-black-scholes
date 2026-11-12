"""Implied volatility across strikes (the "smile" or "skew") from an option chain.

Black-Scholes assumes one volatility for every strike. Real markets price
out-of-the-money puts at higher implied volatility than at-the-money options,
because crashes are more likely than a normal distribution allows. Plotting
implied volatility against strike makes that visible.

Usage:
    python vol_smile.py AAPL            # nearest expiry at least 14 days away
    python vol_smile.py SPY --rate 0.04
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import black_scholes as bs


def smile_from_chain(calls: pd.DataFrame, puts: pd.DataFrame, S, T, r, q=0.0, max_moneyness=0.3) -> pd.DataFrame:
    """Implied volatility per strike, using out-of-the-money options only.

    Below the spot, puts are out of the money; above it, calls. OTM options are
    more liquid and carry less intrinsic value, so their prices say more about
    volatility. Strikes more than `max_moneyness` (30%) from the spot, quotes
    without a positive price, and prices outside the no-arbitrage bounds are dropped.
    """
    rows = []
    for kind, chain in (("put", puts), ("call", calls)):
        side = chain[chain["strike"] < S] if kind == "put" else chain[chain["strike"] >= S]
        for _, quote in side.iterrows():
            K = float(quote["strike"])
            if abs(K / S - 1) > max_moneyness + 1e-9:  # tolerance for floating-point error
                continue
            price = float(quote["mid"])
            if not np.isfinite(price) or price <= 0:
                continue
            try:
                iv = bs.implied_volatility(price, S, K, T, r, kind, q)
            except ValueError:
                continue
            rows.append({"strike": K, "moneyness": K / S, "kind": kind, "price": price, "implied_vol": iv})
    return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import market_data

    parser = argparse.ArgumentParser(description="Plot the implied volatility smile for a ticker.")
    parser.add_argument("ticker")
    parser.add_argument("--rate", type=float, default=0.04)
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument("--min-days", type=int, default=14)
    args = parser.parse_args()

    ticker = args.ticker.upper()
    closes = market_data.close_prices(ticker, "1y")
    spot = float(closes.iloc[-1])
    hist_vol = bs.historical_volatility(closes)
    expiry, days, calls, puts = market_data.option_chain(ticker, args.min_days)
    smile = smile_from_chain(calls, puts, spot, days / 365, args.rate, args.dividend_yield)
    if smile.empty:
        raise SystemExit("No usable quotes for this expiry.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind, colour in (("put", "#eb6834"), ("call", "#2a78d6")):
        side = smile[smile["kind"] == kind]
        ax.plot(side["strike"], side["implied_vol"] * 100, "o-", color=colour, ms=4, label=f"OTM {kind}s")
    ax.axhline(hist_vol * 100, color="#52514e", ls="--", lw=1, label=f"historical vol (1y) {hist_vol:.1%}")
    ax.axvline(spot, color="#52514e", ls=":", lw=1)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"{ticker} implied volatility smile, expiry {expiry} ({days} days), spot {spot:,.2f}")
    ax.legend()
    fig.tight_layout()
    out = f"smile_{ticker}.png"
    fig.savefig(out, dpi=160)
    print(smile.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
