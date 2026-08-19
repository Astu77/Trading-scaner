import requests
import pandas as pd
import numpy as np


BASE_URL = "https://api.binance.com"


def get_symbols(limit=100):
    url = f"{BASE_URL}/api/v3/exchangeInfo"
    data = requests.get(url, timeout=10).json()

    symbols = []

    for item in data["symbols"]:
        if (
            item["status"] == "TRADING"
            and item["quoteAsset"] == "USDT"
            and item["isSpotTradingAllowed"]
        ):
            symbols.append(item["symbol"])

    return symbols[:limit]


def get_klines(symbol, interval="1h", limit=200):
    url = f"{BASE_URL}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    data = requests.get(url, params=params, timeout=10).json()

    df = pd.DataFrame(data, columns=[
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "buy_base",
        "buy_quote",
        "ignore"
    ])

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column])

    return df


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


def analyze(symbol):

    try:
        df = get_klines(symbol)

        close = df["close"]

        df["ema20"] = ema(close, 20)
        df["ema50"] = ema(close, 50)
        df["rsi"] = rsi(close)
        df["atr"] = atr(df)

        last = df.iloc[-1]

        price = float(last["close"])
        ema20 = float(last["ema20"])
        ema50 = float(last["ema50"])
        rsi_value = float(last["rsi"])
        atr_value = float(last["atr"])

        score = 0
        reasons = []

        # TREND
        if ema20 > ema50:
            score += 20
            reasons.append("EMA20 > EMA50")
        elif ema20 < ema50:
            score += 20
            reasons.append("EMA20 < EMA50")

        # PRICE POSITION
        if price > ema20:
            score += 15
            reasons.append("Price above EMA20")
        else:
            score += 15
            reasons.append("Price below EMA20")

        # RSI
        if 50 <= rsi_value <= 68:
            score += 20
            reasons.append("RSI bullish zone")
        elif 32 <= rsi_value < 50:
            score += 20
            reasons.append("RSI bearish zone")

        # MOMENTUM
        if price > ema50:
            score += 15
            reasons.append("Above EMA50")
        else:
            score += 15
            reasons.append("Below EMA50")

        # VOLATILITY
        if atr_value > 0:
            score += 10
            reasons.append("Valid volatility")

        # DIRECTION
        if ema20 > ema50 and price > ema20 and rsi_value > 50:
            direction = "BUY"

            entry = price
            sl = price - (atr_value * 1.5)
            tp = price + (atr_value * 3)

        elif ema20 < ema50 and price < ema20 and rsi_value < 50:
            direction = "SELL"

            entry = price
            sl = price + (atr_value * 1.5)
            tp = price - (atr_value * 3)

        else:
            direction = "WAIT"

            entry = price
            sl = None
            tp = None

        return {
            "symbol": symbol,
            "direction": direction,
            "score": score,
            "price": round(price, 8),
            "entry": round(entry, 8),
            "sl": round(sl, 8) if sl else None,
            "tp": round(tp, 8) if tp else None,
            "rsi": round(rsi_value, 2),
            "reasons": reasons
        }

    except Exception as e:
        return None


def scan():

    symbols = get_symbols(100)

    results = []

    for symbol in symbols:

        result = analyze(symbol)

        if result:
            results.append(result)

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results
