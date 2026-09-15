import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app import app
from data_handler import fetch_historical_data, generate_demo_data
from deep_learning import prepare_sequences
from technical_strategy import compute_indicators, generate_technical_signals
from trading_simulator import backtest


class DeepTradeAlgoTests(unittest.TestCase):
    def test_demo_data_is_deterministic(self):
        first = generate_demo_data("AAPL", "2024-01-01", "2024-06-01")
        second = generate_demo_data("AAPL", "2024-01-01", "2024-06-01")

        self.assertEqual(first.attrs["source"], "demo")
        pd.testing.assert_series_equal(first["Close"], second["Close"])

    def test_fetch_historical_data_falls_back_to_demo(self):
        with patch("data_handler.yf.download", side_effect=RuntimeError("rate limited")):
            df = fetch_historical_data("AAPL", "2024-01-01", "2024-06-01")

        self.assertFalse(df.empty)
        self.assertEqual(df.attrs["source"], "demo")

    def test_technical_backtest_returns_float(self):
        df = generate_demo_data("AAPL", "2023-01-01", "2025-01-01")
        df = generate_technical_signals(compute_indicators(df))

        final_value, portfolio_df = backtest(df["Signal_TA"], df["Close"])

        self.assertIsInstance(final_value, float)
        self.assertFalse(portfolio_df.empty)
        self.assertIn("PortfolioValue", portfolio_df.columns)

    def test_backtest_executes_signal_on_next_row(self):
        index = pd.date_range("2024-01-01", periods=3, freq="B")
        signals = pd.Series([1, 0, -1], index=index)
        opens = pd.Series([10.0, 20.0, 30.0], index=index)

        final_value, _ = backtest(
            signals,
            opens,
            initial_capital=10000.0,
            fee_bps=0,
            slippage_bps=0,
        )

        self.assertEqual(final_value, 15000.0)

    def test_indicators_do_not_backfill_future_values(self):
        demo = generate_demo_data("AAPL", "2024-01-01", "2024-06-01")
        indicators = compute_indicators(demo)

        self.assertGreater(indicators.index[0], demo.index[0])

    def test_lstm_normalization_uses_training_window_only(self):
        index = pd.date_range("2024-01-01", periods=100, freq="B")
        prices = np.concatenate([np.linspace(10, 20, 80), np.linspace(100, 120, 20)])
        frame = pd.DataFrame({"Close": prices}, index=index)

        _, _, mean_price, _ = prepare_sequences(frame, sequence_length=20, fit_rows=80)

        self.assertAlmostEqual(mean_price, float(prices[:80].mean()))

    def test_dashboard_post_renders_result(self):
        demo = generate_demo_data("AAPL", "2023-01-01", "2025-01-01")
        with patch("app.fetch_historical_data", return_value=demo):
            response = app.test_client().post(
                "/dashboard",
                data={
                    "ticker": "AAPL",
                    "start_date": "2023-01-01",
                    "end_date": "2025-01-01",
                    "strategy": "technical",
                    "forecast_days": "5",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AAPL analysis", response.data)
        self.assertIn(b"Backtest value", response.data)

    def test_predict_endpoint_returns_json(self):
        demo = generate_demo_data("MSFT", "2023-01-01", "2025-01-01")
        with patch("app.fetch_historical_data", return_value=demo):
            response = app.test_client().get(
                "/predict?ticker=MSFT&start_date=2023-01-01&end_date=2025-01-01&strategy=technical"
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["ticker"], "MSFT")
        self.assertIn(payload["recommendation"], {"Buy", "Hold", "Sell"})
        self.assertIn("benchmark_return", payload)
        self.assertIn("next market row", payload["execution_note"])

    def test_dashboard_rejects_unbounded_history(self):
        response = app.test_client().post(
            "/dashboard",
            data={
                "ticker": "AAPL",
                "start_date": "2010-01-01",
                "end_date": "2020-01-01",
                "strategy": "technical",
                "forecast_days": "5",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"five years or less", response.data)

    def test_unexpected_api_error_is_not_exposed(self):
        with patch(
            "app.fetch_historical_data", side_effect=RuntimeError("private upstream detail")
        ):
            response = app.test_client().get(
                "/predict?ticker=AAPL&start_date=2024-01-01&end_date=2025-01-01&strategy=technical"
            )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn(b"private upstream detail", response.data)

    def test_realtime_rejects_invalid_ticker(self):
        response = app.test_client().get("/realtime?ticker=bad%20ticker&format=json")

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
