# Release readiness

Checkpoint: 2026-10-14

Status: **published static demo verified; maintenance update verified for publication**.

## Release boundary

DeepTradeAlgo is an educational signal and backtesting workbench. It is not a trading system, financial advice, or evidence of profitable future performance. It does not connect to a brokerage or place orders.

## Correctness changes

- Signals derived from a row's closing information execute only at the next row's open.
- One-day paper simulation enters at the next row's open rather than the signal-producing close.
- Entries and exits apply 5 basis points of fees and 5 basis points of slippage per side.
- Results include a cost-adjusted buy-and-hold benchmark over the same rows.
- LSTM normalization and training use only the first 70% of the selected history. Its displayed backtest uses later rows.
- Moving-average inputs are no longer backfilled with future values.
- MA50/MA200 require complete 50/200-row windows, and RSI treats zero-loss gains as 100, zero-gain losses as 0, and flat periods as 50.
- Observed-row signals and recursively generated forecast-row signals are represented separately.
- Market data uses consistently adjusted OHLC values; malformed/non-finite data, misaligned indexes, and out-of-domain signals fail closed.
- Web requests are limited to five years of history and ten forecast business days.
- Shared web/CLI analysis is capped at 1,500 rows; LSTM training has a 15-second cooperative deadline, bounded parameters, and a two-thread CPU ceiling.
- PyTorch is lazy-loaded only for LSTM analyses, keeping technical-only startup smaller.
- User input errors remain specific, while unexpected internal exceptions return stable messages.
- Demo fallback is limited to provider request failure or empty results; normalization and programming errors are not converted into fictional data.
- Plotly.js is generated from the locked dependency and served locally instead of relying on a runtime CDN.
- Five deterministic technical-analysis snapshots are exported for a browser-only GitHub Pages demo. The page states that it does not run Flask, train PyTorch, or show market history.

## Reproducibility and stewardship

- `uv.lock` is the rebuild lock and CI installs it with `uv sync --locked`.
- CI checks formatting, lint, tests, the generated Plotly asset, and known dependency vulnerabilities.
- Apache-2.0 terms are in `LICENSE`; `NOTICE`, `SECURITY.md`, and `THIRD_PARTY_NOTICES.md` document stewardship and third-party boundaries.

## Limits that remain

Yahoo Finance access through `yfinance` is unofficial and can be delayed or unavailable. Generated fallback data is labeled as demo data. Adjusted OHLC values account for upstream corporate-action adjustments, but the simulator still omits cash-flow treatment, liquidity, taxes, partial fills, order-book behavior, and market-regime changes. The compact LSTM is a teaching example rather than a validated forecasting model.

## Verification

Completed successfully in this checkout on 2026-10-14:

- `uv run ruff format --check .` — 22 files already formatted
- `uv run ruff check .` — all checks passed
- `uv run pytest` — 26 tests passed
- `uv run python scripts/export_plotly_asset.py --check` — generated asset is current
- `uv run python scripts/export_static_analyses.py --check` — five saved analyses are current
- `uv run pip-audit` — no known vulnerabilities
- Local strategy checks — deterministic generated fixtures produced repeatable technical and chronological LSTM output; observed and forecast rows remained separate; PyTorch stayed within its two-thread CPU limit
- Static browser checks — all five analyses, keyboard focus, 320-pixel reflow, reduced-motion mode, refresh, failure handling, and same-origin-only asset loading passed without console or network errors
- Published Pages baseline — HTTPS is enforced at `https://gauravsdama.github.io/deepTradeAlgo/`, and every deployed asset matched the corresponding `main` asset before this maintenance update
