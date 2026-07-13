"""Small LSTM model used for demo stock-price signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def _close_values(df: pd.DataFrame) -> np.ndarray:
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.to_numeric(close, errors="coerce").dropna().to_numpy(dtype=float)


def prepare_sequences(df: pd.DataFrame, sequence_length: int = 60):
    """
    Create normalized close-price sequences for LSTM training.
    """

    prices = _close_values(df)
    if len(prices) <= sequence_length:
        raise ValueError(f"Need more than {sequence_length} price rows for deep learning.")

    mean_p = float(prices.mean())
    std_p = float(prices.std())
    if std_p == 0:
        raise ValueError("Price series has no variance.")

    prices_norm = (prices - mean_p) / std_p
    x_sequences = []
    y_sequences = []
    for i in range(len(prices_norm) - sequence_length):
        x_sequences.append(prices_norm[i : i + sequence_length])
        y_sequences.append(prices_norm[i + sequence_length])

    return np.array(x_sequences), np.array(y_sequences), mean_p, std_p


def train_lstm_model(
    X,
    y,
    input_size: int = 1,
    hidden_size: int = 50,
    num_layers: int = 1,
    epochs: int = 5,
    lr: float = 0.001,
    batch_size: int = 32,
):
    """
    Train a compact LSTM. Defaults are intentionally small for demo latency.
    """

    if len(X) == 0:
        raise ValueError("No training sequences were generated.")

    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    model = StockLSTM(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.from_numpy(X).float().unsqueeze(-1)
    y_t = torch.from_numpy(y).float().unsqueeze(-1)
    indices = np.arange(X_t.shape[0])

    for _ in range(epochs):
        rng.shuffle(indices)
        model.train()
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_x = X_t[batch_indices]
            batch_y = y_t[batch_indices]

            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return model


def generate_deep_learning_signals(
    df: pd.DataFrame,
    model,
    sequence_length: int,
    mean_p: float,
    std_p: float,
    threshold: float = 0.01,
):
    """
    Compare next-day model estimates with current close to create signals.
    """

    output = df.copy()
    output["Signal_DL"] = 0
    prices = _close_values(output)

    model.eval()
    for i in range(sequence_length, len(output)):
        seq = prices[i - sequence_length : i]
        seq_norm = (seq - mean_p) / std_p
        X_seq = torch.from_numpy(seq_norm).float().unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            pred_norm = model(X_seq).item()
        pred_price = (pred_norm * std_p) + mean_p
        current_price = float(prices[i])

        if pred_price > current_price * (1 + threshold):
            output.at[output.index[i], "Signal_DL"] = 1
        elif pred_price < current_price * (1 - threshold):
            output.at[output.index[i], "Signal_DL"] = -1

    return output


def forecast_future_prices(
    df: pd.DataFrame,
    model,
    mean_p: float,
    std_p: float,
    forecast_days: int = 5,
    sequence_length: int = 60,
    threshold: float = 0.01,
):
    """
    Step forward one business day at a time using the latest actual/predicted closes.
    """

    columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df_future = pd.DataFrame(columns=columns)
    full_prices = _close_values(df)
    last_date = pd.to_datetime(df.index[-1])
    previous_price = float(full_prices[-1])

    model.eval()
    for _ in range(forecast_days):
        if len(full_prices) < sequence_length:
            break

        recent_seq = full_prices[-sequence_length:]
        seq_norm = (recent_seq - mean_p) / std_p
        X_seq = torch.from_numpy(seq_norm).float().unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            pred_norm = model(X_seq).item()
        pred_price = float((pred_norm * std_p) + mean_p)
        last_date = last_date + pd.offsets.BDay(1)

        if pred_price > previous_price * (1 + threshold):
            signal = 1
        elif pred_price < previous_price * (1 - threshold):
            signal = -1
        else:
            signal = 0

        df_future.loc[last_date, columns] = [
            pred_price,
            pred_price,
            pred_price,
            pred_price,
            pred_price,
            0,
        ]
        df_future.loc[last_date, "Signal_Future"] = signal
        full_prices = np.append(full_prices, pred_price)
        previous_price = pred_price

    return df_future
