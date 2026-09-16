# strategy.py

def generate_signal(df):
    # مثال بسيط: تقاطع متوسطين
    short_ma = df["Close"].rolling(20).mean()
    long_ma = df["Close"].rolling(50).mean()

    if short_ma.iloc[-1] > long_ma.iloc[-1]:
        return "BUY"
    elif short_ma.iloc[-1] < long_ma.iloc[-1]:
        return "SELL"
    else:
        return None
