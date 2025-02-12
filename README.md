How It Works

    Command-Line Arguments:
        --symbol: Stock ticker (default: AAPL).
        --start_date/--end_date: Date range for historical data.
        --strategy: Either "technical" or "deep_learning".
        --mode: "backtest" or "paper".

    Fetching Data: Uses fetch_historical_data from data_handler.py.

    Strategy Selection:
        If strategy == "deep_learning", we:
            Prepare sequences (prepare_sequences).
            Split into train/test sets.
            Train an LSTM (train_lstm_model).
            Generate signals for each day using the trained model (generate_deep_learning_signals).
        Otherwise (technical), we:
            Compute indicators (compute_indicators).
            Generate signals (generate_technical_signals).

    Mode Selection:
        If mode == "backtest", we run backtest on the generated signals for the entire historical DataFrame. It then prints out final portfolio value and total return.
        If mode == "paper", we call backdate_one_day to split the last two rows as “yesterday” and “today,” and we:
            Extract the “yesterday’s signal.”
            Simulate a single-day trade using simulate_paper_trade.
            Print out how much profit/loss we would have made had we followed the strategy from last close to today’s close.

command line arguments :
Argument Description Example
--symbol Stock ticker symbol --symbol AAPL
--start_date Start date for fetching data --start_date 2024-01-01
--end_date End date for fetching data --end_date 2025-01-01
--strategy Choose strategy: "technical" or "deep_learning" --strategy technical
--mode Choose mode: "backtest" or "paper" --mode backtest

Example : python main.py --symbol AAPL --start_date 2024-01-01 --end_date 2025-01-01 --strategy technical --mode backtest
This will:

    Fetch historical data for AAPL from 2024-01-01 to 2025-01-01.
    Apply technical indicators (RSI, MACD, Moving Averages).
    Backtest the strategy on past data.
    Print the final portfolio value and total return
