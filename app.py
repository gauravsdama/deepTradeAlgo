from __future__ import annotations

import datetime as dt
import re

import pandas as pd
import plotly.graph_objs as go
from flask import Flask, jsonify, render_template, request
from plotly.offline import plot

from data_handler import fetch_historical_data
from deep_learning import (
    forecast_future_prices,
    generate_deep_learning_signals,
    prepare_sequences,
    train_lstm_model,
)
from realtime_data import fetch_realtime_data
from technical_strategy import compute_indicators, generate_technical_signals
from trading_simulator import backtest


app = Flask(__name__)

STRATEGIES = {
    "technical": "Technical indicators",
    "deep_learning": "LSTM forecast",
}
SIGNAL_LABELS = {
    1: ("Buy", "positive"),
    0: ("Hold", "neutral"),
    -1: ("Sell", "negative"),
}


def _default_dates():
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * 3)
    return start.isoformat(), end.isoformat()


def _form_values(source=None):
    start, end = _default_dates()
    values = {
        "ticker": "AAPL",
        "start_date": start,
        "end_date": end,
        "strategy": "technical",
        "forecast_days": "5",
    }
    if source:
        for key in values:
            if source.get(key):
                values[key] = source.get(key)
    values["ticker"] = values["ticker"].upper()
    return values


def _parse_inputs(source):
    values = _form_values(source)
    ticker = values["ticker"].strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
        raise ValueError("Use a valid ticker symbol such as AAPL, MSFT, or BRK-B.")

    start_date = dt.datetime.strptime(values["start_date"], "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(values["end_date"], "%Y-%m-%d").date()
    if end_date <= start_date:
        raise ValueError("End date must be after start date.")

    strategy = values["strategy"]
    if strategy not in STRATEGIES:
        raise ValueError("Choose a supported strategy.")

    forecast_days = max(1, min(int(values.get("forecast_days") or 5), 20))
    return ticker, start_date.isoformat(), end_date.isoformat(), strategy, forecast_days


def _as_float(value) -> float:
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _signal(signal):
    return SIGNAL_LABELS.get(int(signal), SIGNAL_LABELS[0])


def _fmt_money(value) -> str:
    return f"${value:,.2f}"


def _fmt_pct(value) -> str:
    return f"{value:+.2f}%"


def _activity_rows(df: pd.DataFrame, signal_col: str):
    rows = []
    for index, row in df.tail(6).iterrows():
        label, tone = _signal(row[signal_col])
        rows.append(
            {
                "date": pd.to_datetime(index).strftime("%b %d, %Y"),
                "close": _fmt_money(_as_float(row["Close"])),
                "signal": label,
                "tone": tone,
            }
        )
    return rows


def _build_chart(ticker: str, strategy: str, history: pd.DataFrame, combined: pd.DataFrame, signal_col: str):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Close"],
            mode="lines",
            name="Close",
            line={"color": "#1f6feb", "width": 2.4},
            hovertemplate="%{x|%b %d, %Y}<br>Close: $%{y:.2f}<extra></extra>",
        )
    )

    if strategy == "technical":
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history["MA50"],
                mode="lines",
                name="MA50",
                line={"color": "#0f766e", "width": 1.3},
                hovertemplate="%{x|%b %d, %Y}<br>MA50: $%{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history["MA200"],
                mode="lines",
                name="MA200",
                line={"color": "#f59e0b", "width": 1.3},
                hovertemplate="%{x|%b %d, %Y}<br>MA200: $%{y:.2f}<extra></extra>",
            )
        )

    future = combined.loc[combined.index > history.index[-1]]
    if strategy == "deep_learning" and not future.empty:
        fig.add_trace(
            go.Scatter(
                x=future.index,
                y=future["Close"],
                mode="lines+markers",
                name="Forecast",
                line={"color": "#f97316", "width": 2.2, "dash": "dot"},
                marker={"size": 6},
                hovertemplate="%{x|%b %d, %Y}<br>Forecast: $%{y:.2f}<extra></extra>",
            )
        )

    buys = history[history[signal_col] == 1]
    sells = history[history[signal_col] == -1]
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Close"],
                mode="markers",
                name="Buy",
                marker={"symbol": "triangle-up", "size": 10, "color": "#15803d"},
                hovertemplate="%{x|%b %d, %Y}<br>Buy: $%{y:.2f}<extra></extra>",
            )
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["Close"],
                mode="markers",
                name="Sell",
                marker={"symbol": "triangle-down", "size": 10, "color": "#dc2626"},
                hovertemplate="%{x|%b %d, %Y}<br>Sell: $%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title={"text": f"{ticker} price action", "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        height=520,
        margin={"l": 52, "r": 24, "t": 58, "b": 44},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
        font={"family": "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False, tickprefix="$")

    return plot(
        fig,
        output_type="div",
        include_plotlyjs="cdn",
        config={"displayModeBar": False, "responsive": True},
    )


def run_analysis(ticker: str, start_date: str, end_date: str, strategy: str, forecast_days: int = 5):
    df = fetch_historical_data(ticker, start_date, end_date)
    if df.empty:
        raise ValueError("No data was available for that ticker and date range.")

    source = df.attrs.get("source", "unknown")
    source_note = df.attrs.get("source_note", "Market data source was not provided.")

    if strategy == "deep_learning":
        X_seq, y_seq, mean_p, std_p = prepare_sequences(df)
        model = train_lstm_model(X_seq, y_seq, epochs=3, hidden_size=32)
        history = generate_deep_learning_signals(df, model, sequence_length=60, mean_p=mean_p, std_p=std_p)
        future = forecast_future_prices(
            history,
            model,
            mean_p,
            std_p,
            forecast_days=forecast_days,
            sequence_length=60,
            threshold=0.01,
        )
        if not future.empty:
            future = future.rename(columns={"Signal_Future": "Signal_DL"})
            future["Signal_DL"] = future["Signal_DL"].astype(int)
        combined = pd.concat([history, future], axis=0, sort=False)
        signal_col = "Signal_DL"
    else:
        history = generate_technical_signals(compute_indicators(df))
        combined = history
        signal_col = "Signal_TA"

    if history.empty:
        raise ValueError("Not enough clean rows were available after indicator calculation.")

    final_value, portfolio_df = backtest(history[signal_col], history["Close"], initial_capital=10000.0)
    total_return = ((final_value - 10000.0) / 10000.0) * 100

    latest_signal = int(combined[signal_col].dropna().iloc[-1])
    signal_label, signal_tone = _signal(latest_signal)
    last_close = _as_float(history["Close"].iloc[-1])
    first_close = _as_float(history["Close"].iloc[0])
    price_change = ((last_close - first_close) / first_close) * 100 if first_close else 0

    return {
        "ticker": ticker,
        "strategy": strategy,
        "strategy_label": STRATEGIES[strategy],
        "source": source,
        "source_label": "Yahoo Finance" if source == "yfinance" else "Demo data",
        "source_note": source_note,
        "rows": len(history),
        "date_range": f"{history.index[0]:%b %d, %Y} to {history.index[-1]:%b %d, %Y}",
        "latest_signal": latest_signal,
        "signal_label": signal_label,
        "signal_tone": signal_tone,
        "last_close": _fmt_money(last_close),
        "price_change": _fmt_pct(price_change),
        "final_value": _fmt_money(final_value),
        "total_return": _fmt_pct(total_return),
        "buy_count": int((history[signal_col] == 1).sum()),
        "sell_count": int((history[signal_col] == -1).sum()),
        "activity": _activity_rows(history, signal_col),
        "plot_div": _build_chart(ticker, strategy, history, combined, signal_col),
        "portfolio_rows": len(portfolio_df),
    }


@app.route("/")
def home():
    return render_template("dashboard_form.html", values=_form_values(), strategies=STRATEGIES)


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if request.method == "GET":
        values = _form_values(request.args)
        if request.args.get("run"):
            try:
                ticker, start_date, end_date, strategy, forecast_days = _parse_inputs(request.args)
                result = run_analysis(ticker, start_date, end_date, strategy, forecast_days)
            except Exception as exc:
                return (
                    render_template(
                        "dashboard_form.html",
                        values=values,
                        strategies=STRATEGIES,
                        error=str(exc),
                    ),
                    400,
                )
            return render_template("dashboard.html", result=result, values=values, strategies=STRATEGIES)
        return render_template("dashboard_form.html", values=values, strategies=STRATEGIES)

    values = _form_values(request.form)
    try:
        ticker, start_date, end_date, strategy, forecast_days = _parse_inputs(request.form)
        result = run_analysis(ticker, start_date, end_date, strategy, forecast_days)
    except Exception as exc:
        return (
            render_template(
                "dashboard_form.html",
                values=values,
                strategies=STRATEGIES,
                error=str(exc),
            ),
            400,
        )

    return render_template("dashboard.html", result=result, values=values, strategies=STRATEGIES)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    payload = request.get_json(silent=True) or request.form or request.args
    try:
        ticker, start_date, end_date, strategy, forecast_days = _parse_inputs(payload)
        result = run_analysis(ticker, start_date, end_date, strategy, forecast_days)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "ticker": result["ticker"],
            "strategy": strategy,
            "signal": result["latest_signal"],
            "recommendation": result["signal_label"],
            "last_close": result["last_close"],
            "total_return": result["total_return"],
            "data_source": result["source"],
            "message": "Educational demo only. Not investment advice.",
        }
    )


@app.route("/realtime", methods=["GET"])
def realtime():
    ticker = (request.args.get("ticker") or "AAPL").upper()
    quote = fetch_realtime_data(ticker)
    if request.args.get("format") == "json":
        if quote is None:
            return jsonify({"error": "Ticker parameter is required"}), 400
        return jsonify(quote)
    return render_template("realtime.html", quote=quote, ticker=ticker)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
