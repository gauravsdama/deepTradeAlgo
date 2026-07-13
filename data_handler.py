"""Market data access and deterministic demo-data fallback."""

from __future__ import annotations

import hashlib
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _stable_seed(parts: Iterable[str]) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    flattened = df.copy()
    for level in range(df.columns.nlevels):
        values = [str(value) for value in df.columns.get_level_values(level)]
        if any(value in PRICE_COLUMNS for value in values):
            flattened.columns = values
            return flattened

    flattened.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in df.columns.to_flat_index()
    ]
    return flattened


def _normalize_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_yfinance_columns(df).copy()
    if df.empty:
        return df

    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    for column in ["Open", "High", "Low", "Close", "Adj Close"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["Open", "High", "Low", "Adj Close"]:
        if column not in df.columns and "Close" in df.columns:
            df[column] = df["Close"]

    if "Volume" not in df.columns:
        df["Volume"] = 0
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    df = df.dropna(subset=["Close"]).sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df[PRICE_COLUMNS]


def generate_demo_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Create realistic, deterministic OHLCV data for offline demos and tests."""

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Dates must be parseable by pandas.")
    if end <= start:
        end = start + pd.Timedelta(days=365)

    dates = pd.bdate_range(start=start, end=end)
    if len(dates) < 2:
        dates = pd.bdate_range(end=end, periods=2)

    seed = _stable_seed([symbol.upper(), dates[0].isoformat(), dates[-1].isoformat()])
    rng = np.random.default_rng(seed)

    base_price = 75 + (seed % 9000) / 100
    drift = rng.normal(0.00045, 0.00012)
    volatility = rng.uniform(0.012, 0.026)
    seasonal = np.sin(np.linspace(0, np.pi * 5, len(dates))) * rng.uniform(0.001, 0.003)
    returns = rng.normal(drift, volatility, len(dates)) + seasonal
    close = base_price * np.cumprod(1 + returns)
    close = np.maximum(close, 2.0)

    open_price = np.concatenate(([base_price], close[:-1])) * (1 + rng.normal(0, 0.004, len(dates)))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.018, len(dates)))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.018, len(dates)))
    volume = rng.integers(1_000_000, 9_000_000, len(dates))

    df = pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    df.attrs["source"] = "demo"
    df.attrs["source_note"] = "Deterministic demo data generated because live market data was unavailable."
    return df


def fetch_historical_data(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch historical market data with a disclosed demo-data fallback."""

    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        raise ValueError("Ticker symbol is required.")

    allow_demo_fallback = os.getenv("DEEPTRADE_DEMO_FALLBACK", "1") != "0"
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            df = yf.download(
                clean_symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        df = _normalize_market_frame(df)
        if not df.empty:
            df.attrs["source"] = "yfinance"
            df.attrs["source_note"] = "Live/delayed Yahoo Finance market data."
            return df
        if not allow_demo_fallback:
            raise RuntimeError(f"No rows returned for {clean_symbol}.")
    except Exception as exc:  # yfinance can fail for rate limits or network issues.
        if not allow_demo_fallback:
            raise RuntimeError(f"Could not fetch data for {clean_symbol}: {exc}") from exc

    if allow_demo_fallback:
        return generate_demo_data(clean_symbol, start_date, end_date)

    return pd.DataFrame(columns=PRICE_COLUMNS)


def backdate_one_day(df: pd.DataFrame):
    """
    Return the previous and latest rows to simulate a one-day paper trade.
    """

    if len(df) < 2:
        raise ValueError("Not enough data to backdate.")

    yesterday = df.iloc[-2].copy()
    today = df.iloc[-1].copy()
    return yesterday, today
