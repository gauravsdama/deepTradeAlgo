# Release readiness

Checkpoint: 2026-09-15

Status: **open-source release; static demo pending deployment verification**.

## Release boundary

DeepTradeAlgo is an educational signal and backtesting workbench. It is not a trading system, financial advice, or evidence of profitable future performance. It does not connect to a brokerage or place orders.

## Correctness changes

- Signals derived from a row's closing information execute only at the next row's open.
- Entries and exits apply 5 basis points of fees and 5 basis points of slippage per side.
- Results include a cost-adjusted buy-and-hold benchmark over the same rows.
- LSTM normalization and training use only the first 70% of the selected history. Its displayed backtest uses later rows.
- Moving-average inputs are no longer backfilled with future values.
- Web requests are limited to five years of history and ten forecast business days.
- User input errors remain specific, while unexpected internal exceptions return stable messages.
- Plotly.js is generated from the locked dependency and served locally instead of relying on a runtime CDN.
- Five deterministic technical-analysis snapshots are exported for a browser-only GitHub Pages demo. The page states that it does not run Flask, train PyTorch, or show market history.

## Reproducibility and stewardship

- `uv.lock` is the rebuild lock and CI installs it with `uv sync --locked`.
- CI checks formatting, lint, tests, the generated Plotly asset, and known dependency vulnerabilities.
- Apache-2.0 terms are in `LICENSE`; `NOTICE`, `SECURITY.md`, and `THIRD_PARTY_NOTICES.md` document stewardship and third-party boundaries.

## Limits that remain

Yahoo Finance access through `yfinance` is unofficial and can be delayed or unavailable. Generated fallback data is labeled as demo data. The simulator still omits liquidity, taxes, corporate actions, partial fills, order-book behavior, and market-regime changes. The compact LSTM is a teaching example rather than a validated forecasting model.

## Verification

Completed successfully in this checkout on 2026-09-15:

- `uv run ruff format --check .` — 14 files already formatted
- `uv run ruff check .` — all checks passed
- `uv run pytest` — 15 tests passed
- `uv run python scripts/export_plotly_asset.py --check` — generated asset is current
- `uv run python scripts/export_static_analyses.py --check` — five saved analyses are current
- `uv run pip-audit` — no known vulnerabilities
- Browser checks — technical and chronological LSTM flows rendered complete results with the local Plotly asset and no console warnings or errors
