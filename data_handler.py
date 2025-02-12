import yfinance as yf
import pandas as pd
import numpy as np

def fetch_historical_data(symbol, start_date, end_date, interval='1d'):
    """
    Fetch historical price data from Yahoo Finance.
    """
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
    # Ensure we have a standard set of columns
    # df typically has: [Open, High, Low, Close, Adj Close, Volume]
    df.dropna(inplace=True)
    return df


def backdate_one_day(df):
    """
    Returns data for 'yesterday' and 'today' to simulate paper trading.
    For demonstration: 
    - 'yesterday' is df.iloc[-2]
    - 'today' is df.iloc[-1]
    """
    if len(df) < 2:
        raise ValueError("Not enough data to backdate.")
    
    # For demonstration, we treat second-to-last row as 'yesterday', last row as 'today'.
    yesterday = df.iloc[-2].copy()
    today = df.iloc[-1].copy()
    return yesterday, today
