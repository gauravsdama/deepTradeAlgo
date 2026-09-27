"""Shared, bounded strategy preparation for the web and CLI entry points."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from technical_strategy import compute_indicators, generate_technical_signals

MAX_MARKET_ROWS = 1_500
MAX_FORECAST_DAYS = 10
MAX_TRAINING_SECONDS = 15.0


@dataclass(frozen=True)
class StrategyAnalysis:
    observed: pd.DataFrame
    forecast: pd.DataFrame
    signal_column: str


def build_strategy_analysis(
    frame: pd.DataFrame,
    strategy: str,
    *,
    forecast_days: int = 0,
) -> StrategyAnalysis:
    """Build observed signals and a separate forecast without unbounded work."""

    if frame.empty:
        raise ValueError("No market rows were provided.")
    if len(frame) > MAX_MARKET_ROWS:
        raise ValueError(f"Analysis is limited to {MAX_MARKET_ROWS:,} market rows.")
    if not 0 <= forecast_days <= MAX_FORECAST_DAYS:
        raise ValueError(f"Forecast days must be between 0 and {MAX_FORECAST_DAYS}.")

    if strategy == "technical":
        observed = generate_technical_signals(compute_indicators(frame))
        return StrategyAnalysis(observed, pd.DataFrame(), "Signal_TA")
    if strategy != "deep_learning":
        raise ValueError("Choose a supported strategy.")

    # Importing this module loads PyTorch, so keep it out of technical-only
    # process startup and requests.
    from deep_learning import (
        forecast_future_prices,
        generate_deep_learning_signals,
        prepare_sequences,
        train_lstm_model,
    )

    sequence_length = 60
    if len(frame) < 140:
        raise ValueError("The LSTM demo needs at least 140 market rows.")
    train_end = int(len(frame) * 0.7)
    x_sequences, y_sequences, mean_price, std_price = prepare_sequences(
        frame,
        sequence_length=sequence_length,
        fit_rows=train_end,
    )
    training_sequence_count = train_end - sequence_length
    deadline = time.monotonic() + MAX_TRAINING_SECONDS
    model = train_lstm_model(
        x_sequences[:training_sequence_count],
        y_sequences[:training_sequence_count],
        epochs=3,
        hidden_size=32,
        deadline_monotonic=deadline,
    )
    full_history = generate_deep_learning_signals(
        frame,
        model,
        sequence_length=sequence_length,
        mean_p=mean_price,
        std_p=std_price,
    )
    observed = full_history.iloc[train_end - 1 :].copy()
    forecast = forecast_future_prices(
        full_history,
        model,
        mean_price,
        std_price,
        forecast_days=forecast_days,
        sequence_length=sequence_length,
        threshold=0.01,
    )
    return StrategyAnalysis(observed, forecast, "Signal_DL")
