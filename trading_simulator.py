"""Backtesting helpers for generated trading signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float(value) -> float:
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _execution_price(price: float, bps: float, side: str) -> float:
    multiplier = 1 + (bps / 10_000) if side == "buy" else 1 - (bps / 10_000)
    return price * multiplier


def _validated_inputs(signals: pd.Series, prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    if not isinstance(signals, pd.Series) or not isinstance(prices, pd.Series):
        raise TypeError("Signals and prices must be pandas Series.")
    if signals.empty:
        raise ValueError("At least one signal is required.")
    if not signals.index.equals(prices.index):
        raise ValueError("Signals and prices must have exactly the same index and order.")
    if not signals.index.is_unique:
        raise ValueError("Signals and prices must have a unique index.")
    if not signals.index.is_monotonic_increasing:
        raise ValueError("Signals and prices must be ordered by increasing index.")

    numeric_signals = pd.to_numeric(signals, errors="coerce")
    numeric_prices = pd.to_numeric(prices, errors="coerce")
    if not np.isfinite(numeric_signals.to_numpy(dtype=float)).all():
        raise ValueError("Signals must be finite values.")
    if not numeric_signals.isin([-1, 0, 1]).all():
        raise ValueError("Signals must contain only -1, 0, or 1.")
    if not np.isfinite(numeric_prices.to_numpy(dtype=float)).all() or (numeric_prices <= 0).any():
        raise ValueError("Prices must be finite and positive.")
    return numeric_signals.astype(int), numeric_prices.astype(float)


def backtest(
    signals: pd.Series,
    prices: pd.Series,
    initial_capital: float = 10000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
):
    """
    Run a long-only backtest using the next bar's supplied execution price.

    Each signal is shifted by one row before execution. A signal calculated from
    today's close can therefore only trade at the next row's price. Fees and
    slippage are charged on both entries and exits.

    Returns:
        final_value: float
        portfolio_df: DataFrame with one PortfolioValue row per input date.
    """

    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("Fees and slippage cannot be negative.")

    signals, prices = _validated_inputs(signals, prices)

    cash = float(initial_capital)
    shares = 0
    portfolio_values = []

    executable_signals = signals.shift(1).fillna(0)
    for date in executable_signals.index:
        signal = int(_as_float(executable_signals.loc[date]))
        price = _as_float(prices.loc[date])
        if signal == 1 and shares == 0:
            buy_price = _execution_price(price, slippage_bps, "buy")
            per_share_cost = buy_price * (1 + fee_bps / 10_000)
            shares = int(cash // per_share_cost)
            cash -= shares * per_share_cost
        elif signal == -1 and shares > 0:
            sell_price = _execution_price(price, slippage_bps, "sell")
            cash += shares * sell_price * (1 - fee_bps / 10_000)
            shares = 0

        portfolio_values.append(cash + shares * price)

    if shares > 0:
        sell_price = _execution_price(_as_float(prices.iloc[-1]), slippage_bps, "sell")
        cash += shares * sell_price * (1 - fee_bps / 10_000)
        shares = 0
        portfolio_values[-1] = cash

    portfolio_df = pd.DataFrame({"PortfolioValue": portfolio_values}, index=signals.index)
    return float(cash), portfolio_df


def buy_and_hold_value(
    prices: pd.Series,
    initial_capital: float = 10000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> float:
    """Return a cost-adjusted buy-and-hold benchmark over the same prices."""

    if not isinstance(prices, pd.Series) or prices.empty:
        raise ValueError("At least one benchmark price is required.")
    clean_prices = pd.to_numeric(prices, errors="coerce")
    if not prices.index.is_unique:
        raise ValueError("Benchmark prices must have a unique index.")
    if not np.isfinite(clean_prices.to_numpy(dtype=float)).all() or (clean_prices <= 0).any():
        raise ValueError("Benchmark prices must be finite and positive.")
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("Fees and slippage cannot be negative.")

    buy_price = _execution_price(float(clean_prices.iloc[0]), slippage_bps, "buy")
    per_share_cost = buy_price * (1 + fee_bps / 10_000)
    shares = int(initial_capital // per_share_cost)
    cash = initial_capital - shares * per_share_cost
    sell_price = _execution_price(float(clean_prices.iloc[-1]), slippage_bps, "sell")
    return float(cash + shares * sell_price * (1 - fee_bps / 10_000))


def simulate_paper_trade(
    strategy_signal, execution_open, valuation_close, initial_capital: float = 10000.0
):
    """
    Simulate a one-day trade decided at yesterday's close and entered today.

    A buy signal executes at today's open and is valued at today's close. This
    avoids using the signal-producing close as an attainable execution price.
    """

    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")
    cash = float(initial_capital)
    shares = 0
    today_open = _as_float(execution_open)
    today_close = _as_float(valuation_close)
    signal = int(_as_float(strategy_signal))
    if signal not in {-1, 0, 1}:
        raise ValueError("Signal must be -1, 0, or 1.")
    if not np.isfinite([today_open, today_close]).all() or today_open <= 0 or today_close <= 0:
        raise ValueError("Paper-trade prices must be finite and positive.")

    if signal == 1:
        shares = int(cash // today_open)
        cash -= shares * today_open

    if shares > 0:
        cash += shares * today_close

    profit = cash - initial_capital
    return float(cash), float(profit)
