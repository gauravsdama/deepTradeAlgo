"""Backtesting helpers for generated trading signals."""

from __future__ import annotations

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

    signals, prices = signals.align(prices, join="inner")
    if len(signals) != len(prices):
        raise ValueError("Signals and prices must have the same length/index.")
    if signals.empty:
        raise ValueError("At least one signal is required.")

    cash = float(initial_capital)
    shares = 0
    portfolio_values = []

    executable_signals = signals.shift(1).fillna(0)
    for date in executable_signals.index:
        signal = int(_as_float(executable_signals.loc[date]))
        price = _as_float(prices.loc[date])
        if price <= 0:
            portfolio_values.append(cash)
            continue

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

    clean_prices = pd.to_numeric(prices, errors="coerce").dropna()
    if clean_prices.empty:
        raise ValueError("At least one benchmark price is required.")

    buy_price = _execution_price(float(clean_prices.iloc[0]), slippage_bps, "buy")
    per_share_cost = buy_price * (1 + fee_bps / 10_000)
    shares = int(initial_capital // per_share_cost)
    cash = initial_capital - shares * per_share_cost
    sell_price = _execution_price(float(clean_prices.iloc[-1]), slippage_bps, "sell")
    return float(cash + shares * sell_price * (1 - fee_bps / 10_000))


def simulate_paper_trade(
    strategy_signal, yesterday_price, today_price, initial_capital: float = 10000.0
):
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
