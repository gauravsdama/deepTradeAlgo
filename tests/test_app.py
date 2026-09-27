import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from yfinance.exceptions import YFException

from analysis_pipeline import MAX_MARKET_ROWS, StrategyAnalysis, build_strategy_analysis
from app import app, run_analysis
from data_handler import PRICE_COLUMNS, fetch_historical_data, generate_demo_data
from deep_learning import prepare_sequences, train_lstm_model
from technical_strategy import compute_indicators, generate_technical_signals
from trading_simulator import backtest, simulate_paper_trade


class DeepTradeAlgoTests(unittest.TestCase):
    def test_demo_data_is_deterministic(self):
        first = generate_demo_data("AAPL", "2024-01-01", "2024-06-01")
        second = generate_demo_data("AAPL", "2024-01-01", "2024-06-01")

        self.assertEqual(first.attrs["source"], "demo")
        pd.testing.assert_series_equal(first["Close"], second["Close"])

    def test_fetch_historical_data_falls_back_to_demo(self):
        with patch("data_handler.yf.download", side_effect=YFException("rate limited")):
            df = fetch_historical_data("AAPL", "2024-01-01", "2024-06-01")

        self.assertFalse(df.empty)
        self.assertEqual(df.attrs["source"], "demo")

    def test_unexpected_download_error_does_not_fall_back(self):
        with patch("data_handler.yf.download", side_effect=TypeError("programming error")):
            with self.assertRaisesRegex(TypeError, "programming error"):
                fetch_historical_data("AAPL", "2024-01-01", "2024-06-01")

    def test_market_data_programming_or_schema_error_does_not_fall_back(self):
        malformed = pd.DataFrame(
            {
                "Open": [10.0],
                "High": [11.0],
                "Low": [9.0],
                "Close": [-1.0],
                "Volume": [100],
            },
            index=pd.date_range("2024-01-02", periods=1),
        )
        with patch("data_handler._download_market_data", return_value=malformed):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                fetch_historical_data("AAPL", "2024-01-01", "2024-06-01")

    def test_market_download_requests_adjusted_daily_prices(self):
        empty = pd.DataFrame(columns=PRICE_COLUMNS)
        with patch("data_handler.yf.download", return_value=empty) as download:
            df = fetch_historical_data("AAPL", "2024-01-01", "2024-06-01")

        self.assertEqual(df.attrs["source"], "demo")
        self.assertTrue(download.call_args.kwargs["auto_adjust"])
        self.assertEqual(download.call_args.kwargs["interval"], "1d")
        self.assertGreater(download.call_args.kwargs["timeout"], 0)

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

    def test_backtest_rejects_misaligned_or_invalid_inputs(self):
        first_index = pd.date_range("2024-01-01", periods=2, freq="B")
        shifted_index = pd.date_range("2024-01-02", periods=2, freq="B")

        with self.assertRaisesRegex(ValueError, "exactly the same index"):
            backtest(
                pd.Series([1, 0], index=first_index),
                pd.Series([10.0, 11.0], index=shifted_index),
            )
        with self.assertRaisesRegex(ValueError, "only -1, 0, or 1"):
            backtest(
                pd.Series([2, 0], index=first_index),
                pd.Series([10.0, 11.0], index=first_index),
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            backtest(
                pd.Series([1, 0], index=first_index),
                pd.Series([10.0, np.inf], index=first_index),
            )

    def test_paper_trade_enters_at_next_open(self):
        final_value, profit = simulate_paper_trade(1, 20.0, 30.0)

        self.assertEqual(final_value, 15000.0)
        self.assertEqual(profit, 5000.0)

    def test_indicators_do_not_backfill_future_values(self):
        demo = generate_demo_data("AAPL", "2023-01-01", "2024-06-01")
        indicators = compute_indicators(demo)

        self.assertEqual(indicators.index[0], demo.index[199])

    def test_indicator_golden_windows_and_rsi_edges(self):
        index = pd.date_range("2024-01-01", periods=220, freq="B")

        rising = compute_indicators(pd.DataFrame({"Close": np.arange(1.0, 221.0)}, index=index))
        flat = compute_indicators(pd.DataFrame({"Close": np.full(220, 25.0)}, index=index))
        falling = compute_indicators(
            pd.DataFrame({"Close": np.arange(220.0, 0.0, -1.0)}, index=index)
        )

        self.assertEqual(rising.index[0], index[199])
        self.assertAlmostEqual(rising["MA50"].iloc[0], np.arange(151.0, 201.0).mean())
        self.assertAlmostEqual(rising["MA200"].iloc[0], np.arange(1.0, 201.0).mean())
        self.assertEqual(rising["RSI"].iloc[-1], 100.0)
        self.assertEqual(flat["RSI"].iloc[-1], 50.0)
        self.assertEqual(falling["RSI"].iloc[-1], 0.0)

    def test_lstm_normalization_uses_training_window_only(self):
        index = pd.date_range("2024-01-01", periods=100, freq="B")
        prices = np.concatenate([np.linspace(10, 20, 80), np.linspace(100, 120, 20)])
        frame = pd.DataFrame({"Close": prices}, index=index)

        _, _, mean_price, _ = prepare_sequences(frame, sequence_length=20, fit_rows=80)

        self.assertAlmostEqual(mean_price, float(prices[:80].mean()))

    def test_analysis_row_limit_fails_before_strategy_work(self):
        oversized = pd.DataFrame(
            {"Close": np.ones(MAX_MARKET_ROWS + 1)},
            index=pd.date_range("2020-01-01", periods=MAX_MARKET_ROWS + 1, freq="B"),
        )

        with self.assertRaisesRegex(ValueError, f"{MAX_MARKET_ROWS:,}"):
            build_strategy_analysis(oversized, "technical")

    def test_lstm_training_honors_expired_deadline(self):
        sequences = np.ones((1, 4), dtype=float)
        targets = np.ones(1, dtype=float)

        with self.assertRaisesRegex(TimeoutError, "time budget"):
            train_lstm_model(
                sequences,
                targets,
                epochs=1,
                hidden_size=2,
                deadline_monotonic=time.monotonic() - 1,
            )

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

    def test_observed_and_forecast_signals_remain_separate(self):
        index = pd.date_range("2024-01-01", periods=3, freq="B")
        observed = pd.DataFrame(
            {
                "Open": [10.0, 11.0, 12.0],
                "Close": [10.0, 11.0, 12.0],
                "Signal_DL": [0, 0, 1],
            },
            index=index,
        )
        observed.attrs.update(source="demo", source_note="fixture")
        forecast = pd.DataFrame(
            {
                "Open": [9.0],
                "Close": [9.0],
                "Signal_Future": [-1],
            },
            index=pd.date_range("2024-01-04", periods=1, freq="B"),
        )
        analysis = StrategyAnalysis(observed, forecast, "Signal_DL")

        with (
            patch("app.fetch_historical_data", return_value=observed),
            patch("app.build_strategy_analysis", return_value=analysis),
        ):
            result = run_analysis(
                "AAPL", "2024-01-01", "2024-02-01", "deep_learning", forecast_days=1
            )

        self.assertEqual(result["latest_signal"], 1)
        self.assertEqual(result["forecast_signal"], -1)
        self.assertEqual(result["signal_label"], "Buy")
        self.assertEqual(result["forecast_signal_label"], "Sell")

    def test_app_import_does_not_eagerly_load_torch(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys, app; print('torch' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip(), "False")

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
