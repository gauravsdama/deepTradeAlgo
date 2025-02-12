import argparse
import pandas as pd
from data_handler import fetch_historical_data, backdate_one_day
from technical_strategy import compute_indicators, generate_technical_signals
from deep_learning import (
    prepare_sequences, train_lstm_model, generate_deep_learning_signals
)
from trading_simulator import backtest, simulate_paper_trade

def main():
    parser = argparse.ArgumentParser(description="Stock Trading Bot")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start_date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2021-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--strategy", type=str, default="technical", choices=["technical", "deep_learning"],
                        help="Choose a strategy: technical or deep_learning")
    parser.add_argument("--mode", type=str, default="backtest", choices=["backtest", "paper"],
                        help="Choose to backtest or do paper trading (simulated).")
    args = parser.parse_args()

    symbol = args.symbol
    start_date = args.start_date
    end_date = args.end_date
    strategy = args.strategy
    mode = args.mode

    # Fetch data
    print(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    df = fetch_historical_data(symbol, start_date, end_date)

    if df.empty:
        print("No data fetched. Please check symbol or date range.")
        return

    # If we do deep learning, we need to train the model
    if strategy == "deep_learning":
        print("Preparing data for LSTM training...")
        X_seq, y_seq, mean_p, std_p = prepare_sequences(df)
        train_size = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]

#        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        print("Training LSTM model...")
        model = train_lstm_model(X_train, y_train, epochs=2)  # short epochs for demo

        print("Generating deep learning signals...")
        df_dl = generate_deep_learning_signals(df, model, sequence_length=60, mean_p=mean_p, std_p=std_p)
        signals = df_dl['Signal_DL']
        prices = df_dl['Close']

    else:
        # Use technical strategy
        print("Computing technical indicators and signals...")
        df_ta = compute_indicators(df)
        df_ta = generate_technical_signals(df_ta)
        signals = df_ta['Signal_TA']
        prices = df_ta['Close']

    if mode == "backtest":
        # Perform backtest on historical data
        print("Running backtest...")
        final_val, portfolio_df = backtest(signals, prices, initial_capital=10000.0)
        print(f"Backtest completed. Final Portfolio Value = ${float(final_val.iloc[0]):.2f}")
        # Could do more analysis here: e.g. show total return, etc.
        total_return = ((final_val - 10000)/10000 * 100).iloc[0]
        print(f"Total Return: {total_return:.2f}%")

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
        yesterday_price = yesterday['Close']
        # Today close price
        today_price = today['Close']

        final_cash, profit = simulate_paper_trade(strategy_signal, yesterday_price, today_price)
        yesterday_price = float(yesterday_price.iloc[0])  # Extract float from Series
        today_price = float(today_price.iloc[0])  # Extract float from Series
        #final_cash = float(final_cash.iloc[0])  # Extract float from Series
        #profit = float(profit.iloc[0])  # Extract float from Series

        print(f"Yesterday's Signal = {strategy_signal}, Yesterday Price = {yesterday_price:.2f}, Today Price = {today_price:.2f}")
        print(f"Paper Trade result: Final Cash = ${final_cash:.2f}, Profit = ${profit:.2f}")

if __name__ == "__main__":
    main()


