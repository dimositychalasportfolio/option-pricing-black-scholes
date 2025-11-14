import math
from scipy.stats import norm
import numpy as np
import pandas as pd
import yfinance as yf

# Black-Scholes formula for European call option price
#“Πόσα standard deviations πάνω ή κάτω από ένα κρίσιμο σημείο βρισκόμαστε;”

def d1(S, K, r, sigma, T):
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma * math.sqrt(T)

def call_price(S, K, r, sigma, T):
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    return S * norm.cdf(D1) - K * math.exp(-r * T) * norm.cdf(D2)

def put_price(S, K, r, sigma, T):
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    return K * math.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)

def call_delta(S, K, r, sigma, T):
    return norm.cdf(d1(S, K, r, sigma, T))

def put_delta(S, K, r, sigma, T):
    return call_delta(S, K, r, sigma, T) - 1

def gamma(S, K, r, sigma, T):
    D1 = d1(S, K, r, sigma, T)
    return norm.pdf(D1) / (S * sigma * math.sqrt(T))

def vega(S, K, r, sigma, T):
    D1 = d1(S, K, r, sigma, T)
    return S * norm.pdf(D1) * math.sqrt(T)

def theta(S, K, r, sigma, T):
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    term1 = -(S * norm.pdf(D1) * sigma) / (2 * math.sqrt(T))
    term2 = -r * K * math.exp(-r * T) * norm.cdf(D2)
    return term1 + term2  # call theta

def rho(S, K, r, sigma, T):
    D2 = d2(S, K, r, sigma, T)
    return K * T * math.exp(-r * T) * norm.cdf(D2)

def implied_vol_call(market_price, S, K, r, T, initial_sigma=0.2, tol=1e-6, max_iter=100):
    """
    Υπολογίζει την implied volatility για ένα call option
    με μέθοδο Newton-Raphson, χρησιμοποιώντας τη vega.
    """
    sigma = initial_sigma

    for _ in range(max_iter):
        # Τιμή call από Black–Scholes με το τωρινό sigma
        price = call_price(S, K, r, sigma, T)

        # Πόσο απέχει από την market price
        diff = price - market_price

        # Αν είμαστε αρκετά κοντά, σταμάτα
        if abs(diff) < tol:
            return sigma

        # Παράγωγος ως προς sigma = vega
        v = vega(S, K, r, sigma, T)
        if v == 0:
            break

        # Newton–Raphson update
        sigma -= diff / v

    return sigma  # αν δεν συγκλίνει τέλεια, επιστρέφει την τελευταία τιμή

def get_stock_price_and_vol(ticker, period="1y"):
    """
    Κατεβάζει την τελευταία τιμή S και υπολογίζει annualized volatility.
    Πλήρως ασφαλές για όλες τις περιπτώσεις που yfinance μπορεί να επιστρέψει Series ή DataFrame.
    """
    data = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError(f"No data downloaded for ticker {ticker}")

    # Πάρε την "Close"
    prices = data["Close"]

    # Αν "prices" είναι DataFrame (κάποιες εκδόσεις του yfinance το κάνουν αυτό)
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]  # πάρε την πρώτη στήλη

    # Returns
    returns = prices.pct_change().dropna()

    # daily_vol μπορεί να γίνει Series --> φροντίζουμε να γίνει scalar
    daily_vol = returns.std()

    if isinstance(daily_vol, pd.Series):
        daily_vol = daily_vol.iloc[0]

    daily_vol = float(daily_vol)

    # annualized volatility
    annual_vol = float(daily_vol * math.sqrt(252))

    # τελευταία τιμή μπορεί να είναι Series επίσης
    last_price = prices.iloc[-1]

    if isinstance(last_price, pd.Series):
        last_price = last_price.iloc[0]

    S = float(last_price)

    return S, annual_vol

if __name__ == "__main__":
    # Παράδειγμα με πραγματική μετοχή (π.χ. Apple)
    ticker = "AAPL"

    S, sigma = get_stock_price_and_vol(ticker, period="1y")

    K = S            # strike ~ at-the-money
    r = 0.04         # 4% risk-free rate (ενδεικτικά)
    T = 30 / 365     # 30 μέρες μέχρι λήξη

    print(f"Ticker: {ticker}")
    print(f"Spot price S: {S:.2f}")
    print(f"Annualized volatility sigma: {sigma:.4f}")

    print("Call price:", call_price(S, K, r, sigma, T))
    print("Put price:", put_price(S, K, r, sigma, T))
    print("Call delta:", call_delta(S, K, r, sigma, T))
    print("Put delta:", put_delta(S, K, r, sigma, T))

    # Παράδειγμα implied vol:
    market_call = call_price(S, K, r, sigma, T)  # υποθέτουμε ότι αυτή είναι η τιμή αγοράς
    iv = implied_vol_call(market_call, S, K, r, T)

    print(f"Market call price: {market_call:.4f}")
    print(f"Recovered implied vol: {iv:.4f}")

