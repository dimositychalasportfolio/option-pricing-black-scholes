import math
from scipy.stats import norm

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

if __name__ == "__main__":
    S = 100      # τιμή μετοχής
    K = 100      # strike
    r = 0.05     # 5% επιτόκιο χωρίς κίνδυνο
    sigma = 0.2  # 20% μεταβλητότητα
    T = 1        # 1 έτος μέχρι τη λήξη

    print("Call price:", call_price(S, K, r, sigma, T))
    print("Put price:", put_price(S, K, r, sigma, T))
    print("Call delta:", call_delta(S, K, r, sigma, T))
    print("Put delta:", put_delta(S, K, r, sigma, T))

