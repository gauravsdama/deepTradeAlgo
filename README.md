# DeepTradeAlgo

DeepTradeAlgo is a Flask dashboard and CLI for studying stock-signal pipelines. It supports a technical-indicator strategy and a compact PyTorch LSTM, then visualizes simulated signals, recent activity, and long-only backtest results.

This is an educational simulator, not financial advice or evidence of a profitable strategy. It never connects to a brokerage or places trades.

## Highlights

- Flask dashboard with responsive Plotly charts
- Technical strategy using MA50/MA200, RSI, and MACD signals
- LSTM strategy trained on an earlier chronological window and evaluated on later rows
- Next-row-open execution with configurable fees and slippage
- Cost-adjusted buy-and-hold benchmark
- JSON prediction endpoint and quote endpoint
- Deterministic demo-data fallback when Yahoo Finance is unavailable or rate-limited
- Vendored, reproducible Plotly.js runtime with no chart CDN dependency
- Unit tests that run without live market access

## Quick Start

```bash
uv sync --locked
uv run python app.py
```

Open `http://127.0.0.1:5000`.

## CLI Demo

```bash
python main.py --symbol AAPL --start_date 2023-01-01 --end_date 2025-01-01 --strategy technical --mode backtest
python main.py --symbol MSFT --start_date 2023-01-01 --end_date 2025-01-01 --strategy deep_learning --mode backtest
```

## API Examples

```bash
curl "http://127.0.0.1:5000/predict?ticker=AAPL&start_date=2023-01-01&end_date=2025-01-01&strategy=technical"
curl "http://127.0.0.1:5000/realtime?ticker=AAPL&format=json"
```

## Demo Data Fallback

Yahoo Finance can rate-limit local and CI environments. By default, `fetch_historical_data` falls back to deterministic generated OHLCV data and labels the source as `demo` in the UI/API.

Disable fallback when you want strict live data behavior:

```bash
DEEPTRADE_DEMO_FALLBACK=0 python app.py
```

## Backtest assumptions

- Technical signals use information available through a row's close and can execute only at the next row's open.
- LSTM normalization and training use the first 70% of rows; displayed LSTM backtest results use only later rows.
- Entries and exits each include 5 basis points of fees and 5 basis points of slippage.
- Results include a cost-adjusted buy-and-hold comparison over the same evaluation rows.
- Date ranges are capped at five years and forecasts at ten business days to bound local request work.

These assumptions reduce obvious look-ahead and in-sample bias, but they do not model liquidity, taxes, corporate actions, order-book behavior, or changing market regimes. Historical simulations do not predict future results.

## Tests

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run pip-audit
```

## Project Structure

```text
app.py                  Flask dashboard and API routes
main.py                 CLI entry point
data_handler.py         yfinance access and deterministic demo data
technical_strategy.py   indicator and technical signal logic
deep_learning.py        PyTorch LSTM strategy helpers
trading_simulator.py    simple long-only backtester
templates/              Jinja templates
static/styles.css       dashboard styling
tests/                  offline-safe unit tests
```

## License

Licensed under Apache-2.0. See `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`.
