import pandas as pd

def compute_indicators(df):
    """
    Compute technical indicators (Moving Average, RSI, MACD).
    Returns df with new columns: MA50, MA200, RSI, MACD, MACD_signal
    """
    df = df.copy()

    # Moving Averages (example: 50-day and 200-day)
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # RSI (14-day)
    window_length = 14
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    roll_up = up.rolling(window_length).mean()
    roll_down = down.rolling(window_length).mean()
    rs = roll_up / roll_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12-day and 26-day EMAs, 9-day signal)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df.dropna(inplace=True)  # drop initial NaN rows
    return df


def generate_technical_signals(df):
    """
    Generate buy/sell signals based on technical indicators.
    1 = BUY, -1 = SELL, 0 = HOLD
    This is a simple example. Customize rules as you like.
    """
    df = df.copy()
    df['Signal_TA'] = 0

    # Moving Average Crossover
    buy_mask = (df['MA50'] > df['MA200']) & (df['MA50'].shift(1) <= df['MA200'].shift(1))
    sell_mask = (df['MA50'] < df['MA200']) & (df['MA50'].shift(1) >= df['MA200'].shift(1))
    df.loc[buy_mask, 'Signal_TA'] = 1
    df.loc[sell_mask, 'Signal_TA'] = -1

    # RSI-based signals
    buy_mask_rsi = (df['RSI'] < 30) & (df['RSI'].shift(1) >= 30)
    sell_mask_rsi = (df['RSI'] > 70) & (df['RSI'].shift(1) <= 70)
    df.loc[buy_mask_rsi, 'Signal_TA'] = 1
    df.loc[sell_mask_rsi, 'Signal_TA'] = -1

    # MACD-based signals
    buy_mask_macd = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
    sell_mask_macd = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))
    df.loc[buy_mask_macd, 'Signal_TA'] = 1
    df.loc[sell_mask_macd, 'Signal_TA'] = -1

    return df
