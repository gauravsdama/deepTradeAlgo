import argparse

from data_handler import backdate_one_day, fetch_historical_data
from deep_learning import generate_deep_learning_signals, prepare_sequences, train_lstm_model
from technical_strategy import compute_indicators, generate_technical_signals
from trading_simulator import backtest, buy_and_hold_value, simulate_paper_trade


def main():
    parser = argparse.ArgumentParser(description="Stock Trading Bot")
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
        help="Choose to backtest or do paper trading (simulated).",
    )
    args = parser.parse_args()

    symbol = args.symbol
    start_date = args.start_date
    end_date = args.end_date
    strategy = args.strategy
    mode = args.mode

    # Fetch data
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    df = fetch_historical_data(symbol, start_date, end_date)
    print(f"Data source: {df.attrs.get('source', 'unknown')} - {df.attrs.get('source_note', '')}")

    if df.empty:
        print("No data fetched. Please check symbol or date range.")
        return

    # If we do deep learning, we need to train the model
    if strategy == "deep_learning":
        print("Preparing data for LSTM training...")
        sequence_length = 60
        if len(df) < 140:
            raise ValueError("The LSTM demo needs at least 140 market rows.")
        train_end = int(0.7 * len(df))
        X_seq, y_seq, mean_p, std_p = prepare_sequences(
            df, sequence_length=sequence_length, fit_rows=train_end
        )
        training_sequence_count = train_end - sequence_length
        X_train = X_seq[:training_sequence_count]
        y_train = y_seq[:training_sequence_count]

        print("Training LSTM model...")
        model = train_lstm_model(X_train, y_train, epochs=3, hidden_size=32)

        print("Generating deep learning signals...")
        df_dl = generate_deep_learning_signals(
            df, model, sequence_length=sequence_length, mean_p=mean_p, std_p=std_p
        ).iloc[train_end - 1 :]
        signals = df_dl["Signal_DL"]
        prices = df_dl["Open"]

    else:
        # Use technical strategy
        print("Computing technical indicators and signals...")
        df_ta = compute_indicators(df)
        df_ta = generate_technical_signals(df_ta)
        signals = df_ta["Signal_TA"]
        prices = df_ta["Open"]

    if mode == "backtest":
        # Perform backtest on historical data
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
        # Simulate paper trade
        print("Simulating 'paper trade' using yesterday's signal and today's outcome...")

        # Let's assume the last 2 days of data represent "yesterday" and "today"
        # Alternatively, you could fetch data up to 'today' in real-time and then do the same logic
        try:
            yesterday, today = backdate_one_day(df)
        except ValueError as e:
            print("Error in backdate data:", e)
            return

        # We need the signal for 'yesterday' if available
        # If deep learning:
        if strategy == "deep_learning":
            # df_dl might not be the exact same index shape because of the 60-day lead
            # We'll try to get the signal for 'yesterday's date from signals
            yest_idx = signals.index[-2]
            strategy_signal = signals.loc[yest_idx]
        else:
            # technical
            yest_idx = signals.index[-2]
            strategy_signal = signals.loc[yest_idx]

        # Yesterday close price
        yesterday_price = yesterday["Close"]
        # Today close price
        today_price = today["Close"]

        final_cash, profit = simulate_paper_trade(strategy_signal, yesterday_price, today_price)
        if hasattr(yesterday_price, "iloc"):
            yesterday_price = float(yesterday_price.iloc[0])
        else:
            yesterday_price = float(yesterday_price)
        if hasattr(today_price, "iloc"):
            today_price = float(today_price.iloc[0])
        else:
            today_price = float(today_price)

        print(
            f"Yesterday's Signal = {strategy_signal}, "
            f"Yesterday Price = {yesterday_price:.2f}, "
            f"Today Price = {today_price:.2f}"
        )
        print(f"Paper Trade result: Final Cash = ${final_cash:.2f}, Profit = ${profit:.2f}")


if __name__ == "__main__":
    main()
