import requests
import pandas as pd
import numpy as np

BASE_URL = "https://api.binance.com"

# =========================
# MARKET DATA
# =========================

def get_symbols(limit=100):
    url = f"{BASE_URL}/api/v3/exchangeInfo"

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()

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

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(
        data,
        columns=[
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
        ]
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume"
    ]:
        df[column] = pd.to_numeric(df[column])

    return df


# =========================
# INDICATORS
# =========================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


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

    high_close = (
        df["high"] -
        df["close"].shift()
    ).abs()

    low_close = (
        df["low"] -
        df["close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    return true_range.rolling(period).mean()


# =========================
# ANALYSIS
# =========================

def analyze(symbol):

    try:

        df = get_klines(symbol)

        # Gunakan candle yang sudah selesai
        df = df.iloc[:-1].copy()

        if len(df) < 60:
            return None

        close = df["close"]

        df["ema20"] = ema(close, 20)
        df["ema50"] = ema(close, 50)

        df["rsi"] = rsi(close)

        df["atr"] = atr(df)

        # Volume average
        df["volume_ma20"] = (
            df["volume"]
            .rolling(20)
            .mean()
        )

        last = df.iloc[-1]
        previous = df.iloc[-2]

        price = float(last["close"])

        ema20 = float(last["ema20"])
        ema50 = float(last["ema50"])

        previous_ema20 = float(
            previous["ema20"]
        )

        previous_ema50 = float(
            previous["ema50"]
        )

        rsi_value = float(last["rsi"])

        atr_value = float(last["atr"])

        volume = float(last["volume"])

        volume_ma20 = float(
            last["volume_ma20"]
        )

        if np.isnan(atr_value):
            return None

        if np.isnan(rsi_value):
            return None

        # =========================
        # TREND
        # =========================

        bullish_trend = (
            ema20 > ema50
            and price > ema50
        )

        bearish_trend = (
            ema20 < ema50
            and price < ema50
        )

        # =========================
        # EMA MOMENTUM
        # =========================

        bullish_cross = (
            ema20 > ema50
            and previous_ema20 <= previous_ema50
        )

        bearish_cross = (
            ema20 < ema50
            and previous_ema20 >= previous_ema50
        )

        # =========================
        # RSI
        # =========================

        bullish_rsi = (
            50 <= rsi_value <= 68
        )

        bearish_rsi = (
            32 <= rsi_value <= 50
        )

        # Hindari kondisi terlalu overbought
        overbought = rsi_value > 70

        # Hindari kondisi terlalu oversold
        oversold = rsi_value < 30

        # =========================
        # VOLUME
        # =========================

        volume_strong = (
            volume > volume_ma20 * 1.2
        )

        # =========================
        # SCORE
        # =========================

        buy_score = 0
        sell_score = 0

        buy_reasons = []
        sell_reasons = []

        # TREND
        if bullish_trend:
            buy_score += 25
            buy_reasons.append(
                "Bullish trend"
            )

        if bearish_trend:
            sell_score += 25
            sell_reasons.append(
                "Bearish trend"
            )

        # PRICE VS EMA20
        if price > ema20:
            buy_score += 15
            buy_reasons.append(
                "Price above EMA20"
            )

        if price < ema20:
            sell_score += 15
            sell_reasons.append(
                "Price below EMA20"
            )

        # EMA CROSS
        if bullish_cross:
            buy_score += 15
            buy_reasons.append(
                "Bullish EMA cross"
            )

        if bearish_cross:
            sell_score += 15
            sell_reasons.append(
                "Bearish EMA cross"
            )

        # RSI
        if bullish_rsi and not overbought:
            buy_score += 20
            buy_reasons.append(
                "RSI bullish zone"
            )

        if bearish_rsi and not oversold:
            sell_score += 20
            sell_reasons.append(
                "RSI bearish zone"
            )

        # VOLUME
        if volume_strong:
            if bullish_trend:
                buy_score += 15
                buy_reasons.append(
                    "Strong volume"
                )

            if bearish_trend:
                sell_score += 15
                sell_reasons.append(
                    "Strong volume"
                )

        # =========================
        # FINAL SIGNAL
        # =========================

        score = max(
            buy_score,
            sell_score
        )

        direction = "WAIT"

        entry = price
        sl = None
        tp1 = None
        tp2 = None

        reasons = []

        if (
            buy_score >= 70
            and buy_score > sell_score
            and not overbought
        ):

            direction = "BUY"

            entry = price

            sl = price - (
                atr_value * 1.5
            )

            risk = entry - sl

            tp1 = entry + (
                risk * 1.5
            )

            tp2 = entry + (
                risk * 2.5
            )

            reasons = buy_reasons

        elif (
            sell_score >= 70
            and sell_score > buy_score
            and not oversold
        ):

            direction = "SELL"

            entry = price

            sl = price + (
                atr_value * 1.5
            )

            risk = sl - entry

            tp1 = entry - (
                risk * 1.5
            )

            tp2 = entry - (
                risk * 2.5
            )

            reasons = sell_reasons

        return {
            "symbol": symbol,
            "direction": direction,
            "score": int(score),

            "price": round(
                price,
                8
            ),

            "entry": round(
                entry,
                8
            ),

            "sl": (
                round(sl, 8)
                if sl is not None
                else None
            ),

            "tp1": (
                round(tp1, 8)
                if tp1 is not None
                else None
            ),

            "tp2": (
                round(tp2, 8)
                if tp2 is not None
                else None
            ),

            "rsi": round(
                rsi_value,
                2
            ),

            "atr": round(
                atr_value,
                8
            ),

            "volume_strong": volume_strong,

            "reasons": reasons
        }

    except Exception as e:

        print(
            f"Error analyzing {symbol}: {e}"
        )

        return None


# =========================
# MARKET SCANNER
# =========================

def scan():

    symbols = get_symbols(
        limit=100
    )

    results = []

    for symbol in symbols:

        result = analyze(symbol)

        if result is not None:
            results.append(result)

    # Setup terbaik berada paling atas
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results
