#!/usr/bin/env python3
"""Export five deterministic technical-analysis snapshots for GitHub Pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "demo" / "analyses.json"
START_DATE = "2023-01-01"
END_DATE = "2025-01-01"
INITIAL_CAPITAL = 10_000.0
FEE_BPS = 5.0
SLIPPAGE_BPS = 5.0
SIGNAL_LABELS = {1: "Buy", 0: "Hold", -1: "Sell"}
SCENARIOS = [
    {
        "ticker": "AAPL",
        "title": "Trend check",
        "prompt": "Is the rule set participating in this price path?",
    },
    {
        "ticker": "MSFT",
        "title": "Crossover scan",
        "prompt": "Where do moving-average changes create entries or exits?",
    },
    {
        "ticker": "NVDA",
        "title": "Momentum stress",
        "prompt": "How does the strategy behave on a more volatile path?",
    },
    {
        "ticker": "GOOGL",
        "title": "Benchmark review",
        "prompt": "Does the simulated strategy beat buy-and-hold after costs?",
    },
    {
        "ticker": "TSLA",
        "title": "Signal density",
        "prompt": "How often does the rule set change its position?",
    },
]


def _return_pct(value: float) -> float:
    return round(((value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100, 2)


def _sample_series(history, count: int = 64) -> list[dict]:
    sample_indices = np.linspace(0, len(history) - 1, min(count, len(history)), dtype=int)
    return [
        {
            "date": history.index[index].date().isoformat(),
            "close": round(float(history["Close"].iloc[index]), 2),
        }
        for index in sample_indices
    ]


def _signal_events(history) -> list[dict]:
    events = history.loc[history["Signal_TA"] != 0].tail(5)
    return [
        {
            "date": index.date().isoformat(),
            "signal": SIGNAL_LABELS[int(row["Signal_TA"])],
            "close": round(float(row["Close"]), 2),
        }
        for index, row in events.iterrows()
    ]


def build_snapshot(scenario: dict) -> dict:
    from data_handler import generate_demo_data
    from technical_strategy import compute_indicators, generate_technical_signals
    from trading_simulator import backtest, buy_and_hold_value

    frame = generate_demo_data(scenario["ticker"], START_DATE, END_DATE)
    history = generate_technical_signals(compute_indicators(frame))
    final_value, _ = backtest(
        history["Signal_TA"],
        history["Open"],
        initial_capital=INITIAL_CAPITAL,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )
    benchmark_value = buy_and_hold_value(
        history["Open"],
        initial_capital=INITIAL_CAPITAL,
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
    )
    first_close = float(history["Close"].iloc[0])
    last_close = float(history["Close"].iloc[-1])
    latest_signal = int(history["Signal_TA"].iloc[-1])

    return {
        **scenario,
        "source": "Deterministic generated demo data",
        "period": {
            "start": history.index[0].date().isoformat(),
            "end": history.index[-1].date().isoformat(),
            "rows": len(history),
        },
        "latest_signal": SIGNAL_LABELS[latest_signal],
        "last_close": round(last_close, 2),
        "price_change_pct": round(((last_close - first_close) / first_close) * 100, 2),
        "strategy_return_pct": _return_pct(final_value),
        "benchmark_return_pct": _return_pct(benchmark_value),
        "buy_count": int((history["Signal_TA"] == 1).sum()),
        "sell_count": int((history["Signal_TA"] == -1).sum()),
        "series": _sample_series(history),
        "events": _signal_events(history),
    }


def export_payload() -> dict:
    return {
        "schema_version": 1,
        "generated_from": {
            "data": "Deterministic generated demo data",
            "strategy": "MA50/MA200, RSI, and MACD technical signals",
            "execution": "Next-row open",
            "fee_bps_per_side": FEE_BPS,
            "slippage_bps_per_side": SLIPPAGE_BPS,
            "initial_capital_usd": INITIAL_CAPITAL,
            "requested_period": [START_DATE, END_DATE],
        },
        "analyses": [build_snapshot(scenario) for scenario in SCENARIOS],
    }


def serialized_payload() -> str:
    return json.dumps(export_payload(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = serialized_payload()

    if args.check:
        if not args.output.exists() or args.output.read_text() != expected:
            print(f"Static analyses are stale: {args.output}")
            return 1
        print(f"Static analyses are current: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
