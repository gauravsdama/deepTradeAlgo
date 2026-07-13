# DeepTradeAlgo

DeepTradeAlgo is a Flask dashboard and CLI for experimenting with stock signals. It supports a technical-indicator strategy and a compact PyTorch LSTM strategy, then visualizes recommendations, recent signal activity, and simple long-only backtest results.

This project is an educational demo. It is not financial advice.

## Highlights

- Flask dashboard with responsive Plotly charts
- Technical strategy using MA50/MA200, RSI, and MACD signals
- LSTM strategy for short-horizon demo forecasts
- JSON prediction endpoint and quote endpoint
- Deterministic demo-data fallback when Yahoo Finance is unavailable or rate-limited
- Unit tests that run without live market access

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
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

## Tests

```bash
python -m unittest discover -s tests
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
