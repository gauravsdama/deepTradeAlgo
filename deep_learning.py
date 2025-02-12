import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        print(f"Input shape to LSTM: {x.shape}")  # Debugging step

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

def prepare_sequences(df, sequence_length=60):
    """
    Create sequences for LSTM. df is expected to contain 'Close' price column.
    Returns X, y as numpy arrays.
    """
    prices = df['Close'].values
    # Normalize
    mean_p = prices.mean()
    std_p = prices.std()
    prices_norm = (prices - mean_p) / std_p

    X_sequences = []
    y_sequences = []

    for i in range(len(prices_norm) - sequence_length):
        X_seq = prices_norm[i:i+sequence_length]
        y_val = prices_norm[i+sequence_length]
        X_sequences.append(X_seq)
        y_sequences.append(y_val)

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    # Return data plus the params for un-normalizing
    return X_sequences, y_sequences, mean_p, std_p

def train_lstm_model(X, y, input_size=1, hidden_size=50, num_layers=1, epochs=10, lr=0.001, batch_size=32):
    """
    Train LSTM model with given hyperparameters.
    X, y are numpy arrays. X shape: (n_samples, seq_len), y shape: (n_samples,)
    """
    model = StockLSTM(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert to Torch tensors
    X_t = torch.from_numpy(X).float()
    X_t = X_t.unsqueeze(-1)  # shape: (n_samples, seq_len, 1)
    y_t = torch.from_numpy(y).float().unsqueeze(-1)

    dataset_size = X_t.shape[0]
    indices = np.arange(dataset_size)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        model.train()

        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = X_t[batch_indices]
            batch_y = y_t[batch_indices]
            batch_x = batch_x.squeeze(-1)  # remove redundant dimension
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (dataset_size / batch_size)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return model

def generate_deep_learning_signals(df, model, sequence_length, mean_p, std_p, threshold=0.01):
    """
    Generate signals using the trained LSTM model.
    We'll compare the model's predicted next-day price vs today's price.
    If predicted is > today's by threshold, buy; else if < today's by threshold, sell.
    """
    df = df.copy()
    df['Signal_DL'] = 0

    # We'll need the normalized sequences
    prices = df['Close'].values
    # We'll keep original length for signals
    for i in range(sequence_length, len(df)):
        # Prepare the last 'sequence_length' days as input
        seq = prices[i-sequence_length:i]
        seq_norm = (seq - mean_p) / std_p
        X_seq = torch.from_numpy(seq_norm).float().unsqueeze(0)  # shape: (1, seq_len, 1)

        with torch.no_grad():
            pred_norm = model(X_seq).item()
        pred_price = (pred_norm * std_p) + mean_p

        current_price = df['Close'].iloc[i]
        current_price = current_price.iloc[0]  # Convert Series to float

        print(f"pred_price: {pred_price}, current_price: {current_price}, type: {type(current_price)}") # Debugging step

        # If predicted price is 1% above current -> buy
        if pred_price > current_price * (1 + threshold):
            df.at[df.index[i], 'Signal_DL'] = 1
        elif pred_price < current_price * (1 - threshold):
            df.at[df.index[i], 'Signal_DL'] = -1
        else:
            df.at[df.index[i], 'Signal_DL'] = 0

    return df
