import argparse

from analysis_pipeline import build_strategy_analysis
from data_handler import backdate_one_day, fetch_historical_data
from trading_simulator import backtest, buy_and_hold_value, simulate_paper_trade


def main():
    parser = argparse.ArgumentParser(description="Educational stock-signal simulator")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument(
        "--start_date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end_date", type=str, default="2021-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--strategy",
        type=str,
        default="technical",
        choices=["technical", "deep_learning"],
        help="Choose a strategy: technical or deep_learning",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="backtest",
        choices=["backtest", "paper"],
        help="Choose a historical backtest or a simulated one-day paper exercise.",
    )
    args = parser.parse_args()

    symbol = args.symbol
    start_date = args.start_date
    end_date = args.end_date
    strategy = args.strategy
    mode = args.mode

    print(
        "Educational simulation only. Generated or delayed data may be used. Not investment advice."
    )
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    df = fetch_historical_data(symbol, start_date, end_date)
    print(f"Data source: {df.attrs.get('source', 'unknown')} - {df.attrs.get('source_note', '')}")

    if df.empty:
        print("No data fetched. Please check symbol or date range.")
        return

    print("Preparing strategy signals...")
    analysis = build_strategy_analysis(df, strategy)
    signals = analysis.observed[analysis.signal_column]
    prices = analysis.observed["Open"]

    if mode == "backtest":
        print("Running backtest...")
        final_val, portfolio_df = backtest(
            signals,
            prices,
            initial_capital=10000.0,
            fee_bps=5.0,
            slippage_bps=5.0,
        )
        benchmark_val = buy_and_hold_value(
            prices,
            initial_capital=10000.0,
            fee_bps=5.0,
            slippage_bps=5.0,
        )
        print(f"Backtest completed over {len(portfolio_df)} rows.")
        print(f"Final Portfolio Value = ${final_val:,.2f}")
        total_return = (final_val - 10000) / 10000 * 100
        print(f"Total Return: {total_return:.2f}%")
        print(f"Buy-and-hold benchmark: ${benchmark_val:,.2f}")
        print("Execution: next-row open with 5 bps fees and 5 bps slippage per side.")

    else:
        print("Simulating 'paper trade' using yesterday's signal and today's outcome...")

        try:
            yesterday, today = backdate_one_day(df)
        except ValueError as e:
            print("Error in backdate data:", e)
            return

        decision_index = yesterday.name
        if decision_index not in signals.index:
            raise ValueError("No strategy signal exists for the prior market row.")
        strategy_signal = signals.loc[decision_index]

        today_open = today["Open"]
        today_close = today["Close"]

        final_cash, profit = simulate_paper_trade(strategy_signal, today_open, today_close)
        today_open = float(today_open.iloc[0]) if hasattr(today_open, "iloc") else float(today_open)
        today_close = (
            float(today_close.iloc[0]) if hasattr(today_close, "iloc") else float(today_close)
        )

        print(
            f"Yesterday's Signal = {strategy_signal}, "
            f"Today's Open = {today_open:.2f}, "
            f"Today's Close = {today_close:.2f}"
        )
        print(f"Paper Trade result: Final Cash = ${final_cash:.2f}, Profit = ${profit:.2f}")


if __name__ == "__main__":
    main()
