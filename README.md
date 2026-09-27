# DeepTradeAlgo

DeepTradeAlgo is a Flask dashboard and CLI for studying stock-signal pipelines. It supports a technical-indicator strategy and a compact PyTorch LSTM, then visualizes simulated signals, recent activity, and long-only backtest results.

This is an educational simulator, not financial advice or evidence of a profitable strategy. It never connects to a brokerage or places trades.

[Open the five-analysis static demo](https://gauravsdama.github.io/deepTradeAlgo/). It uses fixed generated data and does not run Python or fetch live quotes.

## Highlights

- Flask dashboard with responsive Plotly charts
- Technical strategy using full-window MA50/MA200, RSI, and MACD signals
- LSTM strategy trained on an earlier chronological window and evaluated on later rows
- Next-row-open execution with configurable fees and slippage
- Cost-adjusted buy-and-hold benchmark
- JSON prediction endpoint and quote endpoint
- Disclosed deterministic demo-data fallback for provider availability failures
- Vendored, reproducible Plotly.js runtime with no chart CDN dependency
- GitHub Pages deck with five deterministic technical-analysis snapshots
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

Yahoo Finance can rate-limit local and CI environments. By default, `fetch_historical_data` falls back to deterministic generated OHLCV data only when the provider request fails or returns no rows, and labels the source as `demo` in the UI/API. Malformed provider data and local processing errors are surfaced instead of being replaced with plausible-looking demo data.

Disable fallback when you want strict live data behavior:

```bash
DEEPTRADE_DEMO_FALLBACK=0 python app.py
```

## Backtest assumptions

- Technical signals use information available through a row's close and can execute only at the next row's open.
- The one-day paper mode likewise decides from the prior row and enters at the next row's open.
- MA50 and MA200 values require all 50 or 200 prior closes; partial windows are not relabeled as long-window averages.
- LSTM normalization and training use the first 70% of rows; displayed LSTM backtest results use only later rows.
- Observed-row LSTM signals remain separate from recursively generated future-row signals.
- Entries and exits each include 5 basis points of fees and 5 basis points of slippage.
- Results include a cost-adjusted buy-and-hold comparison over the same evaluation rows.
- Yahoo OHLC prices are split/dividend-adjusted consistently via `auto_adjust=True`.
- Date ranges are capped at five years, analysis inputs at 1,500 market rows, forecasts at ten business days, and LSTM training at 15 seconds. PyTorch loads only when the LSTM strategy is selected and uses at most two CPU threads.

These assumptions reduce obvious look-ahead and in-sample bias, but they do not model liquidity, taxes, corporate-action cash flows, order-book behavior, or changing market regimes. Historical simulations do not predict future results.

## Tests

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run pip-audit
```

Regenerate the Pages snapshots with `uv run python scripts/export_static_analyses.py`.

This repository is configured as a source application (`tool.uv.package = false`), not as a distributable Python library. Run it from the checkout with `uv run`.

## Project Structure

```text
app.py                  Flask dashboard and API routes
analysis_pipeline.py    shared bounded strategy preparation
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
