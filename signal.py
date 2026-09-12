import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

TWELVE_API_KEY = os.environ["TWELVE_API_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

SYMBOL = "USD/JPY"

STATE_FILE = "state.json"


# -----------------------------
# Load / Save state
# -----------------------------

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_signal": ""}

    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {"last_signal": ""}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# -----------------------------
# Twelve Data
# -----------------------------

def get_data(interval, outputsize=300):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "timezone": "UTC",
        "apikey": TWELVE_API_KEY
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()

    if "values" not in data:
        raise Exception(f"Twelve Data error: {data}")

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)

    return df


# -----------------------------
# Indicators
# -----------------------------

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df, period=14):

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def add_indicators(df):

    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)

    df["rsi"] = rsi(df["close"], 14)

    macd_fast = ema(df["close"], 12)
    macd_slow = ema(df["close"], 26)

    df["macd"] = macd_fast - macd_slow
    df["macd_signal"] = ema(df["macd"], 9)

    df["atr"] = atr(df, 14)

    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()

    return df


# -----------------------------
# Get last CLOSED candle
# -----------------------------

def last_closed(df, minutes):

    now = datetime.now(timezone.utc)

    if minutes == 15:
        current_open = now.replace(
            minute=(now.minute // 15) * 15,
            second=0,
            microsecond=0
        )

    elif minutes == 60:
        current_open = now.replace(
            minute=0,
            second=0,
            microsecond=0
        )

    else:
        raise ValueError("Unsupported interval")

    closed = df[df["datetime"] < current_open]

    if len(closed) == 0:
        raise Exception("No closed candle available")

    return closed.iloc[-1]


# -----------------------------
# Telegram
# -----------------------------

def send_telegram(message):

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    r = requests.post(
        url,
        json=payload,
        timeout=30
    )

    r.raise_for_status()


# -----------------------------
# Main Strategy
# -----------------------------

def main():

    print("Downloading USDJPY data...")

    h1 = get_data("1h", 300)
    m15 = get_data("15min", 300)

    h1 = add_indicators(h1)
    m15 = add_indicators(m15)

    h1_bar = last_closed(h1, 60)
    m15_bar = last_closed(m15, 15)

    # -------------------------
    # H1 conditions
    # -------------------------

    h1_buy_trend = (
        h1_bar["close"] > h1_bar["ema200"]
    )

    h1_buy_structure = (
        h1_bar["ema50"] > h1_bar["ema200"]
    )

    h1_sell_trend = (
        h1_bar["close"] < h1_bar["ema200"]
    )

    h1_sell_structure = (
        h1_bar["ema50"] < h1_bar["ema200"]
    )

    # -------------------------
    # M15 conditions
    # -------------------------

    price = m15_bar["close"]
    atr_value = m15_bar["atr"]
    ema50 = m15_bar["ema50"]

    pullback = (
        abs(price - ema50)
        <= atr_value * 0.60
    )

    bullish_candle = (
        m15_bar["close"] > m15_bar["open"]
    )

    bearish_candle = (
        m15_bar["close"] < m15_bar["open"]
    )

    macd_bullish = (
        m15_bar["macd"]
        > m15_bar["macd_signal"]
    )

    macd_bearish = (
        m15_bar["macd"]
        < m15_bar["macd_signal"]
    )

    rsi_buy = (
        m15_bar["rsi"] > 50
        and m15_bar["rsi"] < 70
    )

    rsi_sell = (
        m15_bar["rsi"] < 50
        and m15_bar["rsi"] > 30
    )

    # -------------------------
    # BUY score
    # -------------------------

    buy_score = 0

    buy_score += int(h1_buy_trend)
    buy_score += int(h1_buy_structure)
    buy_score += int(pullback)
    buy_score += int(rsi_buy)
    buy_score += int(macd_bullish)
    buy_score += int(bullish_candle)

    # -------------------------
    # SELL score
    # -------------------------

    sell_score = 0

    sell_score += int(h1_sell_trend)
    sell_score += int(h1_sell_structure)
    sell_score += int(pullback)
    sell_score += int(rsi_sell)
    sell_score += int(macd_bearish)
    sell_score += int(bearish_candle)

    # -------------------------
    # S/R filter
    # -------------------------

    resistance = m15_bar["resistance"]
    support = m15_bar["support"]

    buy_sr = (
        resistance - price
        > atr_value * 0.30
    )

    sell_sr = (
        price - support
        > atr_value * 0.30
    )

    # -------------------------
    # Signal
    # -------------------------

    signal = None
    score = 0

    if buy_score >= 5 and buy_sr:
        signal = "BUY"
        score = buy_score

    elif sell_score >= 5 and sell_sr:
        signal = "SELL"
        score = sell_score

    print(
        f"BUY score={buy_score}, "
        f"SELL score={sell_score}"
    )

    print(
        f"Price={price:.3f}, "
        f"RSI={m15_bar['rsi']:.2f}, "
        f"ATR={atr_value:.3f}"
    )

    if signal is None:

        print("No valid signal.")

        return

    # -------------------------
    # Prevent duplicate alerts
    # -------------------------

    candle_time = m15_bar["datetime"].isoformat()

    signal_id = (
        f"{signal}_{candle_time}"
    )

    state = load_state()

    if state.get("last_signal") == signal_id:

        print("Signal already sent.")

        return

    # -------------------------
    # Entry / SL / TP
    # -------------------------

    entry = price

    risk = atr_value * 1.5

    if signal == "BUY":

        sl = entry - risk
        tp1 = entry + risk
        tp2 = entry + risk * 2

    else:

        sl = entry + risk
        tp1 = entry - risk
        tp2 = entry - risk * 2

    # -------------------------
    # Telegram message
    # -------------------------

    message = f"""
{'🟢 BUY' if signal == 'BUY' else '🔴 SELL'} USDJPY

Confirmation: {score}/6
Timeframe: M15

Entry: {entry:.3f}
SL: {sl:.3f}
TP1: {tp1:.3f}
TP2: {tp2:.3f}

Risk: 1.5 ATR
RR TP1: 1:1
RR TP2: 1:2

Candle:
{candle_time}
"""

    send_telegram(message)

    state["last_signal"] = signal_id
    save_state(state)

    print("Telegram signal sent.")


if __name__ == "__main__":
    main()
