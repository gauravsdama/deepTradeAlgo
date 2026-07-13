"""Quote helpers for the Flask realtime endpoint."""

from __future__ import annotations

import datetime as dt
import io
from contextlib import redirect_stderr, redirect_stdout

import pandas as pd
import yfinance as yf

from data_handler import generate_demo_data


def _as_float(value):
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _quote_from_row(ticker: str, row, source: str):
    return {
        "ticker": ticker.upper(),
        "price": round(_as_float(row["Close"]), 2),
        "open": round(_as_float(row["Open"]), 2),
        "high": round(_as_float(row["High"]), 2),
        "low": round(_as_float(row["Low"]), 2),
        "volume": int(_as_float(row.get("Volume", 0))),
        "timestamp": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source": source,
    }


def fetch_realtime_data(ticker: str):
    """
    Fetch the latest available quote, falling back to deterministic demo data.
    """

    symbol = (ticker or "").strip().upper()
    if not symbol:
        return None

    try:
        stock = yf.Ticker(symbol)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            data = stock.history(period="5d", auto_adjust=False)
        if not data.empty:
            return _quote_from_row(symbol, data.iloc[-1], "yfinance")
    except Exception:
        pass

    end = dt.date.today()
    start = end - dt.timedelta(days=30)
    demo = generate_demo_data(symbol, start.isoformat(), end.isoformat())
    quote = _quote_from_row(symbol, demo.iloc[-1], "demo")
    quote["note"] = "Deterministic demo quote generated because live market data was unavailable."
    return quote
