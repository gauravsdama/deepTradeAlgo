import pandas as pd

def backtest(signals, prices, initial_capital=10000.0):
    """
    Simple backtest on a single stock with buy/sell signals.
    signals: a pandas Series of {1,0,-1} (buy, hold, sell)
    prices: a pandas Series with the same index as signals
    Returns final portfolio value and a DataFrame of daily portfolio values.
    """

    if not (len(signals) == len(prices)):
        raise ValueError("Signals and prices must have the same length/index.")

    cash = initial_capital
    shares = 0
    portfolio_values = []
    
    for i in range(len(signals)):
        signal = signals.iloc[i]
        price = prices.iloc[i]

        # Buy signal
        if signal == 1 and shares == 0:
            shares_to_buy = int(cash // float(price.iloc[0]))
            if shares_to_buy > 0:
                shares = shares_to_buy
                cash -= shares * price

        # Sell signal
        elif signal == -1 and shares > 0:
            cash += shares * price
            shares = 0
        
        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)

    # If still holding at the end, let's assume we exit
    if shares > 0:
        cash += shares * prices.iloc[-1]
        shares = 0

    final_value = cash
    portfolio_df = pd.DataFrame({
        'PortfolioValue': portfolio_values
    }, index=signals.index)

    return final_value, portfolio_df


def simulate_paper_trade(strategy_signal, yesterday_price, today_price, initial_capital=10000.0):
    """
    Simulate a one-day 'paper trade' based on yesterday's signal and today's price outcome.
    """
    cash = initial_capital
    shares = 0

    # If yesterday's signal was buy, buy at yesterday_price
    if strategy_signal == 1:
        shares_to_buy = int(cash // yesterday_price)
        shares = shares_to_buy
        cash -= shares * yesterday_price
    
    # If the signal was sell, we do nothing if we have no shares.
    # If we had shares from before (not tracked here), that would be scenario-based, but let's keep simple.

    # Evaluate what happened by the end of 'today'
    # Sell at today's price to see result
    if shares > 0:
        cash += shares * today_price
        shares = 0

    profit = cash - initial_capital
    return cash, profit
