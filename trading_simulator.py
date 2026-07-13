"""Backtesting helpers for generated trading signals."""

from __future__ import annotations

import pandas as pd


def _as_float(value) -> float:
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def backtest(signals: pd.Series, prices: pd.Series, initial_capital: float = 10000.0):
    """
    Run a simple long-only backtest using buy/sell/hold signals.

    Returns:
        final_value: float
        portfolio_df: DataFrame with one PortfolioValue row per input date.
    """

    signals, prices = signals.align(prices, join="inner")
    if len(signals) != len(prices):
        raise ValueError("Signals and prices must have the same length/index.")
    if signals.empty:
        raise ValueError("At least one signal is required.")

    cash = float(initial_capital)
    shares = 0
    portfolio_values = []

    for date in signals.index:
        signal = int(_as_float(signals.loc[date]))
        price = _as_float(prices.loc[date])
        if price <= 0:
            portfolio_values.append(cash)
            continue

        if signal == 1 and shares == 0:
            shares = int(cash // price)
            cash -= shares * price
        elif signal == -1 and shares > 0:
            cash += shares * price
            shares = 0

        portfolio_values.append(cash + shares * price)

    if shares > 0:
        cash += shares * _as_float(prices.iloc[-1])
        shares = 0
        portfolio_values[-1] = cash

    portfolio_df = pd.DataFrame({"PortfolioValue": portfolio_values}, index=signals.index)
    return float(cash), portfolio_df


def simulate_paper_trade(strategy_signal, yesterday_price, today_price, initial_capital: float = 10000.0):
    """
    Simulate a one-day paper trade based on yesterday's signal and today's close.
    """

    cash = float(initial_capital)
    shares = 0
    yesterday = _as_float(yesterday_price)
    today = _as_float(today_price)
    signal = int(_as_float(strategy_signal))

    if signal == 1 and yesterday > 0:
        shares = int(cash // yesterday)
        cash -= shares * yesterday

    if shares > 0:
        cash += shares * today

    profit = cash - initial_capital
    return float(cash), float(profit)
