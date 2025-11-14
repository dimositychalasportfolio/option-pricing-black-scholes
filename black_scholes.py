import math
from scipy.stats import norm
import numpy as np
import pandas as pd
import yfinance as yf
import os
import requests
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

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

def get_atm_option_market_data(ticker, S):
    """
    Φέρνει από το yfinance:
    - την πιο κοντινή λήξη options
    - το strike πιο κοντά στο S (ATM)
    - την market price για call και put σε αυτό το strike

    Επιστρέφει: K, T, market_call, market_put, expiry_date

    Αν δεν υπάρχουν καθόλου options, σηκώνει ValueError.
    """
    tkr = yf.Ticker(ticker)

    try:
        options_dates = tkr.options
    except Exception as e:
        raise ValueError(f"No option metadata available for {ticker}: {e}")

    if not options_dates:
        raise ValueError(f"No option expiries available for {ticker}")

    expiry_str = options_dates[0]  # παίρνουμε την πιο κοντινή λήξη
    opt_chain = tkr.option_chain(expiry_str)

    calls = opt_chain.calls
    puts = opt_chain.puts

    if calls.empty or puts.empty:
        raise ValueError(f"No option quotes (calls/puts) for {ticker} at expiry {expiry_str}")

    # βρίσκουμε το strike πιο κοντά στο S (ATM)
    call_row = calls.iloc[(calls["strike"] - S).abs().argmin()]
    put_row = puts.iloc[(puts["strike"] - S).abs().argmin()]

    K = float(call_row["strike"])

    market_call = float(call_row["lastPrice"])
    market_put = float(put_row["lastPrice"])

    # υπολογίζουμε χρόνο μέχρι λήξη (T σε χρόνια)
    expiry = pd.to_datetime(expiry_str).date()
    today = pd.Timestamp.today().date()
    days_to_expiry = (expiry - today).days
    if days_to_expiry <= 0:
        days_to_expiry = 1  # ασφαλιστικά
    T = days_to_expiry / 365.0

    return K, T, market_call, market_put, expiry

def get_stock_price_and_vol_alphavantage(ticker, outputsize="compact"):
    """
    Παίρνει daily prices από Alpha Vantage και υπολογίζει:
    - S: τελευταία τιμή (close)
    - annualized volatility από log returns

    Θέλει ALPHAVANTAGE_API_KEY.
    """
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("No ALPHAVANTAGE_API_KEY set.")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": outputsize,
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise ValueError(f"Alpha Vantage HTTP error: {resp.status_code}")

    data = resp.json()
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Alpha Vantage: no daily data for {ticker}. Response: {data}")

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index", dtype=float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    prices = df["4. close"]
    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError(f"Alpha Vantage: not enough data for {ticker}")

    daily_vol = float(returns.std())
    annual_vol = float(daily_vol * math.sqrt(252))

    S = float(prices.iloc[-1])
    return S, annual_vol

def get_stock_price_and_vol_finnhub(ticker):
    """
    Παράδειγμα function για Finnhub.
    Παίρνει ιστορικά prices και υπολογίζει S & annualized volatility.
    
    Θέλει FINNHUB_API_KEY.
    """
    if not FINNHUB_API_KEY:
        raise ValueError("No FINNHUB_API_KEY set.")

    # Παράδειγμα: daily candles (resolution=D, τελευταίο 1y)
    import time
    now = int(time.time())
    year_ago = now - 365 * 24 * 60 * 60

    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": ticker,
        "resolution": "D",
        "from": year_ago,
        "to": now,
        "token": FINNHUB_API_KEY,
    }
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise ValueError(f"Finnhub HTTP error: {resp.status_code}")

    data = resp.json()
    if data.get("s") != "ok":
        raise ValueError(f"Finnhub: no data for {ticker}. Response: {data}")

    closes = pd.Series(data["c"])
    returns = closes.pct_change().dropna()

    if returns.empty:
        raise ValueError(f"Finnhub: not enough data for {ticker}")

    daily_vol = float(returns.std())
    annual_vol = float(daily_vol * math.sqrt(252))
    S = float(closes.iloc[-1])
    return S, annual_vol

def get_spot_and_vol_multi(ticker):
    """
    Προσπαθεί διαδοχικά:
    1) yfinance (Yahoo)
    2) Alpha Vantage
    3) Finnhub

    και επιστρέφει (S, sigma).

    Αν αποτύχουν όλες οι πηγές, σηκώνει ValueError με συνοπτικά errors.
    """
    errors = []

    # 1) yfinance
    try:
        return get_stock_price_and_vol(ticker, period="1y")
    except Exception as e:
        errors.append(f"yfinance failed: {e}")

    # 2) Alpha Vantage
    try:
        return get_stock_price_and_vol_alphavantage(ticker)
    except Exception as e:
        errors.append(f"Alpha Vantage failed: {e}")

    # 3) Finnhub
    try:
        return get_stock_price_and_vol_finnhub(ticker)
    except Exception as e:
        errors.append(f"Finnhub failed: {e}")

    # Αν φτάσαμε εδώ, όλα απέτυχαν
    msg = "All data providers failed for ticker {}:\n".format(ticker)
    msg += "\n".join(errors)
    raise ValueError(msg)

if __name__ == "__main__":
    # Ζήτα από τον χρήστη το ticker
    ticker = input("Δώσε ticker μετοχής (π.χ. AAPL): ").upper().strip()

    # 1) Παίρνουμε spot price και ιστορική μεταβλητότητα από ΠΟΛΛΑ providers
    try:
        S, sigma = get_spot_and_vol_multi(ticker)
    except ValueError as e:
        print(f"\n[Σφάλμα] {e}")
        print("Δεν υπάρχουν επαρκή δεδομένα τιμών για αυτό το ticker από τους διαθέσιμους providers.")
        print("Δοκίμασε κάποιο άλλο, π.χ. AAPL, MSFT, TSLA, AMZN...\n")
        raise SystemExit

    # Default τιμές σε περίπτωση που ΔΕΝ βρούμε options
    r = 0.04
    K_default = S
    T_default = 30 / 365   # 30 μέρες

    options_available = True

    # 2) Προσπαθούμε να πάρουμε options (ATM, κοντινή λήξη) από yfinance
    try:
        K_opt, T_opt, market_call, market_put, expiry = get_atm_option_market_data(ticker, S)
    except ValueError as e:
        print(f"\n[Προειδοποίηση] Δεν βρέθηκαν options για {ticker}:")
        print(f"  {e}")
        print("Θα υπολογίσουμε ΜΟΝΟ θεωρητικές τιμές για:")
        print(f"  K = S (at-the-money), T = 30 μέρες.\n")
        options_available = False

    # 3) Ορίζουμε τα K, T που θα χρησιμοποιήσουμε στο μοντέλο
    if options_available:
        K = K_opt
        T = T_opt
    else:
        K = K_default
        T = T_default
        expiry = None  # δεν έχουμε πραγματική ημερομηνία λήξης

    # 4) Θεωρητικές τιμές από Black–Scholes
    model_call = call_price(S, K, r, sigma, T)
    model_put = put_price(S, K, r, sigma, T)

    print("\n=== Underlying Setup ===")
    print(f"Ticker: {ticker}")
    print(f"Spot price S: {S:.2f}")
    print(f"Historical volatility sigma: {sigma:.4f}")
    if expiry is not None:
        print(f"Expiry date (from options): {expiry} (T ≈ {T:.3f} years)")
    else:
        print(f"Assumed maturity T: {T:.3f} years (≈ 30 μέρες)")
    print(f"Strike K: {K:.2f}\n")

    print("=== Model Prices (Black–Scholes) ===")
    print(f"Call (model): {model_call:.4f}")
    print(f"Put  (model): {model_put:.4f}")

    # 5) Αν έχουμε options από αγορά, κάνουμε σύγκριση
    if options_available:
        print("\n=== Market vs Model Comparison ===")
        print(f"Call (market): {market_call:.4f}")
        diff_call = model_call - market_call
        print(f"Call diff (model - market): {diff_call:.4f}")
        if diff_call > 0:
            print("👉 Το μοντέλο λέει ότι το call είναι ΦΘΗΝΟ (underpriced / cheap).")
        elif diff_call < 0:
            print("👉 Το μοντέλο λέει ότι το call είναι ΑΚΡΙΒΟ (overpriced / expensive).")
        else:
            print("👉 Το μοντέλο δίνει τιμή σχεδόν ίση με της αγοράς.")

        print(f"\nPut (market): {market_put:.4f}")
        diff_put = model_put - market_put
        print(f"Put diff (model - market):  {diff_put:.4f}")
        if diff_put > 0:
            print("👉 Το μοντέλο λέει ότι το put είναι ΦΘΗΝΟ (underpriced / cheap).")
        elif diff_put < 0:
            print("👉 Το μοντέλο λέει ότι το put είναι ΑΚΡΙΒΟ (overpriced / expensive).")
        else:
            print("👉 Το μοντέλο δίνει τιμή σχεδόν ίση με της αγοράς.\n")
    else:
        print("\n⚠️ Δεν υπάρχουν δεδομένα options για αυτό το ticker στο Yahoo Finance.")
        print("Έχεις μόνο τις θεωρητικές τιμές του Black–Scholes για K = S και T = 30 μέρες.\n")

