"""Market data access and deterministic demo-data fallback."""

from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi.requests.exceptions import RequestException
from yfinance.exceptions import YFException

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
DOWNLOAD_TIMEOUT_SECONDS = 10
MAX_HISTORY_DAYS = 365 * 5 + 2


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
        "_".join(str(part) for part in column if str(part)) for column in df.columns.to_flat_index()
    ]
    return flattened


def _normalize_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_yfinance_columns(df).copy()
    if df.empty:
        return df
    if "Close" not in df.columns:
        raise ValueError("Market data is missing the Close column.")

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
    else:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Market data must use a DatetimeIndex.")
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    if not df.index.is_unique:
        raise ValueError("Market data contains duplicate timestamps.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Market data timestamps must be increasing.")
    price_values = df[["Open", "High", "Low", "Close", "Adj Close"]].to_numpy(dtype=float)
    if not np.isfinite(price_values).all() or (price_values <= 0).any():
        raise ValueError("Market prices must be finite and positive.")
    if (df["High"] < df[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("Market data contains an invalid high price.")
    if (df["Low"] > df[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("Market data contains an invalid low price.")
    volume_values = df["Volume"].to_numpy(dtype=float)
    if not np.isfinite(volume_values).all() or (volume_values < 0).any():
        raise ValueError("Market volume must be finite and non-negative.")

    return df[PRICE_COLUMNS]


def _download_market_data(symbol: str, start_date: str, end_date: str, interval: str):
    """Keep upstream availability errors separate from local normalization errors."""

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
    except (RequestException, YFException) as exc:
        raise RuntimeError(f"Market data provider request failed for {symbol}.") from exc


def generate_demo_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Create realistic, deterministic OHLCV data for offline demos and tests."""

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Dates must be parseable by pandas.")
    if end <= start:
        raise ValueError("End date must be after start date.")

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
    df.attrs["source_note"] = (
        "Deterministic demo data generated because live market data was unavailable."
    )
    return df


def fetch_historical_data(
    symbol: str, start_date: str, end_date: str, interval: str = "1d"
) -> pd.DataFrame:
    """Fetch historical market data with a disclosed demo-data fallback."""

    clean_symbol = (symbol or "").strip().upper()
    if not clean_symbol:
        raise ValueError("Ticker symbol is required.")

    if interval != "1d":
        raise ValueError("Only daily market data is supported.")
    start = pd.to_datetime(start_date, errors="raise")
    end = pd.to_datetime(end_date, errors="raise")
    if end <= start:
        raise ValueError("End date must be after start date.")
    if (end - start).days > MAX_HISTORY_DAYS:
        raise ValueError("Choose a date range of five years or less.")

    allow_demo_fallback = os.getenv("DEEPTRADE_DEMO_FALLBACK", "1") != "0"
    try:
        raw = _download_market_data(clean_symbol, start_date, end_date, interval)
    except RuntimeError:
        if not allow_demo_fallback:
            raise
        return generate_demo_data(clean_symbol, start_date, end_date)

    # Normalization intentionally sits outside the provider exception handler:
    # schema/data/programming errors must be visible, never replaced by fiction.
    df = _normalize_market_frame(raw)
    if not df.empty:
        df.attrs["source"] = "yfinance"
        df.attrs["source_note"] = "Live/delayed split-adjusted Yahoo Finance market data."
        return df

    if allow_demo_fallback:
        return generate_demo_data(clean_symbol, start_date, end_date)

    raise RuntimeError(f"No rows returned for {clean_symbol}.")


def backdate_one_day(df: pd.DataFrame):
    """
    Return the previous and latest rows to simulate a one-day paper trade.
    """

    if len(df) < 2:
        raise ValueError("Not enough data to backdate.")

    yesterday = df.iloc[-2].copy()
    today = df.iloc[-1].copy()
    return yesterday, today
