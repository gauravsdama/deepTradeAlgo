"""Technical indicator calculations and signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _close_series(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.to_numeric(close, errors="coerce")


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute moving averages, RSI, and MACD without dropping most short demo ranges.
    """

    output = df.copy()
    close = _close_series(output)

    output["MA50"] = close.rolling(window=50, min_periods=5).mean()
    output["MA200"] = close.rolling(window=200, min_periods=20).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    output["RSI"] = 100 - (100 / (1 + rs))
    output["RSI"] = output["RSI"].replace([np.inf, -np.inf], np.nan).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    output["MACD"] = ema12 - ema26
    output["MACD_signal"] = output["MACD"].ewm(span=9, adjust=False).mean()

    output[["MA50", "MA200", "MACD", "MACD_signal"]] = output[
        ["MA50", "MA200", "MACD", "MACD_signal"]
    ].ffill()
    return output.dropna(subset=["Close", "MA50", "MA200"])


def generate_technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate simple buy/sell/hold signals.

    Signal values:
    1 = buy, -1 = sell, 0 = hold.
    """

    output = df.copy()
    output["Signal_TA"] = 0

    ma_buy = (output["MA50"] > output["MA200"]) & (
        output["MA50"].shift(1) <= output["MA200"].shift(1)
    )
    ma_sell = (output["MA50"] < output["MA200"]) & (
        output["MA50"].shift(1) >= output["MA200"].shift(1)
    )
    rsi_buy = (output["RSI"] < 30) & (output["RSI"].shift(1) >= 30)
    rsi_sell = (output["RSI"] > 70) & (output["RSI"].shift(1) <= 70)
    macd_buy = (output["MACD"] > output["MACD_signal"]) & (
        output["MACD"].shift(1) <= output["MACD_signal"].shift(1)
    )
    macd_sell = (output["MACD"] < output["MACD_signal"]) & (
        output["MACD"].shift(1) >= output["MACD_signal"].shift(1)
    )

    output.loc[ma_buy | rsi_buy | macd_buy, "Signal_TA"] = 1
    output.loc[ma_sell | rsi_sell | macd_sell, "Signal_TA"] = -1
    return output
