import unittest
from unittest.mock import patch

import pandas as pd

from app import app
from data_handler import fetch_historical_data, generate_demo_data
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


if __name__ == "__main__":
    unittest.main()
