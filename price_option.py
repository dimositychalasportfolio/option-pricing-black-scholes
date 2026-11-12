"""Price an at-the-money option on a real stock and compare implied with historical volatility.

Usage:
    python price_option.py AAPL
    python price_option.py MSFT --rate 0.04 --dividend-yield 0.007

The script does NOT say whether an option is "cheap" or "expensive". The gap
between the Black-Scholes price (using historical volatility) and the market
price only shows that the market expects a different volatility from the past
one. The useful comparison is implied versus historical volatility.
"""

from __future__ import annotations

import argparse

import black_scholes as bs
import market_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ticker", help="stock ticker, e.g. AAPL")
    parser.add_argument("--rate", type=float, default=0.04, help="risk-free rate, e.g. 0.04 for 4%%")
    parser.add_argument("--dividend-yield", type=float, default=0.0, help="continuous dividend yield")
    parser.add_argument("--period", default="1y", help="history used for historical volatility")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    closes = market_data.close_prices(ticker, args.period)
    spot = float(closes.iloc[-1])
    hist_vol = bs.historical_volatility(closes)
    quote = market_data.atm_option_quote(ticker, spot)
    T = quote.days_to_expiry / 365
    r, q = args.rate, args.dividend_yield

    print(f"\n{ticker}  spot {spot:,.2f}   historical volatility ({args.period}) {hist_vol:.1%}")
    print(f"Option: strike {quote.strike:,.2f}, expiry {quote.expiry} ({quote.days_to_expiry} days)\n")
    print(f"{'':6}{'market':>10}{'model (hist. vol)':>20}{'implied vol':>14}")
    for kind, market in (("call", quote.call_mid), ("put", quote.put_mid)):
        model = float(bs.price(spot, quote.strike, T, r, hist_vol, kind, q))
        try:
            iv = bs.implied_volatility(market, spot, quote.strike, T, r, kind, q)
            iv_text = f"{iv:.1%}"
        except ValueError as exc:
            iv, iv_text = None, "n/a"
            print(f"  ({kind}: {exc})")
        print(f"{kind:6}{market:>10.2f}{model:>20.2f}{iv_text:>14}")
        if iv is not None:
            direction = "more" if iv > hist_vol else "less"
            print(f"{'':6}the market prices {direction} volatility than the past {args.period} "
                  f"({iv:.1%} vs {hist_vol:.1%})")

    g = bs.greeks(spot, quote.strike, T, r, hist_vol, "call", q)
    print("\nCall Greeks at historical volatility:")
    for name, value in g.items():
        print(f"  {name:<20}{value:>10.4f}")
    print("\nAssumes European exercise; US single-stock options are American, so treat this as an approximation.")


if __name__ == "__main__":
    main()
