import os
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://api.twelvedata.com/time_series"

API_KEY = os.getenv("TWELVE_DATA_API_KEY")

TIMEFRAME = "1h"
CANDLE_COUNT = 200

MIN_SCORE = 70

FOREX_PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD",
    "EUR/GBP",
    "EUR/JPY",
    "GBP/JPY",
]


# =========================
# GET MARKET DATA
# =========================

def get_candles(symbol):

    if not API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY belum diset."
        )

    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "outputsize": CANDLE_COUNT,
        "apikey": API_KEY,
        "timezone": "UTC"
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") == "error":
        raise RuntimeError(
            data.get(
                "message",
                "Twelve Data API error"
            )
        )

    values = data.get("values")

    if not values:
        return None

    df = pd.DataFrame(values)

    for column in [
        "open",
        "high",
        "low",
        "close"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    return df


# =========================
# INDICATORS
# =========================

def calculate_indicators(df):

    close = df["close"]

    # EMA
    df["ema20"] = (
        close
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["ema50"] = (
        close
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # RSI
    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # ATR
    high_low = (
        df["high"] -
        df["low"]
    )

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

    df["atr"] = (
        true_range
        .rolling(14)
        .mean()
    )

    # Support / Resistance
    df["support"] = (
        df["low"]
        .rolling(20)
        .min()
    )

    df["resistance"] = (
        df["high"]
        .rolling(20)
        .max()
    )

    return df


# =========================
# ANALYZE ONE PAIR
# =========================

def analyze(symbol):

    try:

        df = get_candles(symbol)

        if df is None:
            return None

        if len(df) < 60:
            return None

        # Buang candle terakhir karena
        # bisa masih berjalan.
        df = df.iloc[:-1].copy()

        df = calculate_indicators(df)

        last = df.iloc[-1]
        previous = df.iloc[-2]

        price = float(last["close"])

        ema20 = float(
            last["ema20"]
        )

        ema50 = float(
            last["ema50"]
        )

        previous_ema20 = float(
            previous["ema20"]
        )

        previous_ema50 = float(
            previous["ema50"]
        )

        rsi = float(
            last["rsi"]
        )

        atr = float(
            last["atr"]
        )

        support = float(
            last["support"]
        )

        resistance = float(
            last["resistance"]
        )

        if (
            np.isnan(rsi)
            or np.isnan(atr)
        ):
            return None

        # =====================
        # SCORE
        # =====================

        buy_score = 0
        sell_score = 0

        buy_reasons = []
        sell_reasons = []

        # ---------------------
        # TREND
        # ---------------------

        bullish = (
            ema20 > ema50
            and price > ema50
        )

        bearish = (
            ema20 < ema50
            and price < ema50
        )

        if bullish:

            buy_score += 25

            buy_reasons.append(
                "Bullish trend"
            )

        if bearish:

            sell_score += 25

            sell_reasons.append(
                "Bearish trend"
            )

        # ---------------------
        # EMA20
        # ---------------------

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

        # ---------------------
        # EMA CROSS
        # ---------------------

        bullish_cross = (
            ema20 > ema50
            and
            previous_ema20
            <= previous_ema50
        )

        bearish_cross = (
            ema20 < ema50
            and
            previous_ema20
            >= previous_ema50
        )

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

        # ---------------------
        # RSI
        # ---------------------

        if 50 <= rsi <= 68:

            buy_score += 20

            buy_reasons.append(
                "RSI bullish zone"
            )

        if 32 <= rsi <= 50:

            sell_score += 20

            sell_reasons.append(
                "RSI bearish zone"
            )

        # Jangan entry kondisi ekstrem
        if rsi > 70:

            buy_score -= 20

        if rsi < 30:

            sell_score -= 20

        # ---------------------
        # SUPPORT
        # ---------------------

        support_distance = (
            price - support
        )

        if (
            support_distance > 0
            and
            support_distance
            <= atr * 1.5
        ):

            buy_score += 10

            buy_reasons.append(
                "Near support"
            )

        # ---------------------
        # RESISTANCE
        # ---------------------

        resistance_distance = (
            resistance - price
        )

        if (
            resistance_distance > 0
            and
            resistance_distance
            <= atr * 1.5
        ):

            sell_score += 10

            sell_reasons.append(
                "Near resistance"
            )

        # =====================
        # FINAL SIGNAL
        # =====================

        signal = "WAIT"

        entry = price

        sl = None
        tp1 = None
        tp2 = None

        reasons = []

        score = max(
            buy_score,
            sell_score
        )

        # BUY
        if (
            buy_score >= MIN_SCORE
            and
            buy_score > sell_score
        ):

            signal = "BUY"

            entry = price

            sl = (
                entry -
                (atr * 1.5)
            )

            risk = (
                entry - sl
            )

            tp1 = (
                entry +
                (risk * 1.5)
            )

            tp2 = (
                entry +
                (risk * 2.5)
            )

            reasons = buy_reasons

        # SELL
        elif (
            sell_score >= MIN_SCORE
            and
            sell_score > buy_score
        ):

            signal = "SELL"

            entry = price

            sl = (
                entry +
                (atr * 1.5)
            )

            risk = (
                sl - entry
            )

            tp1 = (
                entry -
                (risk * 1.5)
            )

            tp2 = (
                entry -
                (risk * 2.5)
            )

            reasons = sell_reasons

        return {
            "symbol": symbol,
            "signal": signal,
            "score": int(score),

            "price": round(
                price,
                6
            ),

            "entry": round(
                entry,
                6
            ),

            "sl": (
                round(sl, 6)
                if sl is not None
                else None
            ),

            "tp1": (
                round(tp1, 6)
                if tp1 is not None
                else None
            ),

            "tp2": (
                round(tp2, 6)
                if tp2 is not None
                else None
            ),

            "rsi": round(
                rsi,
                2
            ),

            "atr": round(
                atr,
                6
            ),

            "support": round(
                support,
                6
            ),

            "resistance": round(
                resistance,
                6
            ),

            "reasons": reasons
        }

    except Exception as e:

        return {
            "symbol": symbol,
            "signal": "ERROR",
            "score": 0,
            "error": str(e)
        }


# =========================
# SCAN ALL FOREX
# =========================

def scan():

    results = []

    for symbol in FOREX_PAIRS:

        result = analyze(
            symbol
        )

        if result is not None:

            results.append(
                result
            )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results
