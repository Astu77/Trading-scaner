import MetaTrader5 as mt5
import pandas as pd
import numpy as np


# =========================
# SETTINGS
# =========================

TIMEFRAME = mt5.TIMEFRAME_H1
CANDLE_COUNT = 200

MIN_SCORE = 70

# Pair utama yang ingin kita cari.
# Scanner akan mencocokkan dengan simbol
# yang benar-benar tersedia di MT5.
FOREX_PAIRS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
]


# =========================
# MT5 CONNECTION
# =========================

def connect_mt5():

    if not mt5.initialize():

        raise RuntimeError(
            f"MT5 initialization failed: {mt5.last_error()}"
        )

    return True


# =========================
# FIND BROKER SYMBOL
# =========================

def find_symbol(base_symbol):

    symbols = mt5.symbols_get()

    if symbols is None:
        return None

    base_symbol = base_symbol.upper()

    # Exact match terlebih dahulu
    for symbol in symbols:

        if symbol.name.upper() == base_symbol:
            return symbol.name

    # Kalau broker memakai suffix/prefix
    for symbol in symbols:

        name = symbol.name.upper()

        if base_symbol in name:

            # Hindari simbol aneh yang kebetulan
            # mengandung nama pair
            if (
                len(name) <= len(base_symbol) + 8
            ):
                return symbol.name

    return None


# =========================
# GET CANDLES
# =========================

def get_candles(symbol):

    rates = mt5.copy_rates_from_pos(
        symbol,
        TIMEFRAME,
        0,
        CANDLE_COUNT
    )

    if rates is None:
        return None

    if len(rates) < 100:
        return None

    df = pd.DataFrame(rates)

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s"
    )

    # Candle 0 adalah candle berjalan.
    # Kita tidak gunakan candle tersebut.
    df = df.iloc[:-1].copy()

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

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
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
# ANALYZE
# =========================

def analyze(symbol):

    df = get_candles(symbol)

    if df is None:
        return None

    df = calculate_indicators(df)

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

    rsi = float(last["rsi"])
    atr = float(last["atr"])

    support = float(
        last["support"]
    )

    resistance = float(
        last["resistance"]
    )

    if np.isnan(rsi) or np.isnan(atr):
        return None

    # =========================
    # SCORE
    # =========================

    buy_score = 0
    sell_score = 0

    buy_reasons = []
    sell_reasons = []

    # -------------------------
    # TREND
    # -------------------------

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

    # -------------------------
    # PRICE / EMA20
    # -------------------------

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

    # -------------------------
    # EMA CROSS
    # -------------------------

    bullish_cross = (
        ema20 > ema50
        and previous_ema20 <= previous_ema50
    )

    bearish_cross = (
        ema20 < ema50
        and previous_ema20 >= previous_ema50
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

    # -------------------------
    # RSI
    # -------------------------

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

    # Hindari entry saat terlalu ekstrem
    if rsi > 70:

        buy_score -= 20

    if rsi < 30:

        sell_score -= 20

    # -------------------------
    # SUPPORT / RESISTANCE
    # -------------------------

    support_distance = (
        price - support
    )

    resistance_distance = (
        resistance - price
    )

    # BUY lebih menarik jika harga
    # relatif dekat support.
    if (
        support_distance > 0
        and support_distance <= atr * 1.5
    ):

        buy_score += 10

        buy_reasons.append(
            "Near support"
        )

    # SELL lebih menarik jika harga
    # relatif dekat resistance.
    if (
        resistance_distance > 0
        and resistance_distance <= atr * 1.5
    ):

        sell_score += 10

        sell_reasons.append(
            "Near resistance"
        )

    # =========================
    # FINAL SIGNAL
    # =========================

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

    if (
        buy_score >= MIN_SCORE
        and buy_score > sell_score
    ):

        signal = "BUY"

        entry = price

        sl = price - (
            atr * 1.5
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
        sell_score >= MIN_SCORE
        and sell_score > buy_score
    ):

        signal = "SELL"

        entry = price

        sl = price + (
            atr * 1.5
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


# =========================
# SCAN MARKET
# =========================

def scan():

    connect_mt5()

    results = []

    try:

        for base_pair in FOREX_PAIRS:

            symbol = find_symbol(
                base_pair
            )

            if symbol is None:
                continue

            # Pastikan simbol tersedia
            # untuk digunakan.
            mt5.symbol_select(
                symbol,
                True
            )

            result = analyze(
                symbol
            )

            if result is not None:
                results.append(result)

        # Ranking terbaik
        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results

    finally:

        mt5.shutdown()
