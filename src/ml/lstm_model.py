import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class LSTMModule(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class PyTorchLSTMForecaster:
    """
    Second Layer Forecaster: PyTorch LSTM
    Captures long-term time dependencies with a 30-day sliding window.
    """
    def __init__(self, sequence_length=30, hidden_dim=32, epochs=15, lr=0.01):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def prepare_sequences(self, df: pd.DataFrame):
        df = df.sort_values(["route_id", "date"])
        sequences = []
        targets = []
        
        for route_id, group in df.groupby("route_id"):
            values = group["actual_desi"].values
            if len(values) <= self.sequence_length:
                continue
            for i in range(len(values) - self.sequence_length):
                seq = values[i : i + self.sequence_length]
                target = values[i + self.sequence_length]
                sequences.append(seq)
                targets.append(target)
                
        if len(sequences) == 0:
            return np.empty((0, self.sequence_length, 1)), np.empty((0, 1))
            
        X_seq = np.array(sequences)[:, :, np.newaxis]
        y_target = np.array(targets)[:, np.newaxis]
        return X_seq, y_target

    def fit(self, df: pd.DataFrame):
        X_seq, y_target = self.prepare_sequences(df)
        if len(X_seq) == 0:
            print("[Warning] Dataset too short for 30-day LSTM window. Using 7-day window.")
            self.sequence_length = 7
            X_seq, y_target = self.prepare_sequences(df)

        self.model = LSTMModule(input_dim=1, hidden_dim=self.hidden_dim, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_target, dtype=torch.float32)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            # Simple fallback if model failed to train due to window constraints
            return df["rolling_mean_7"].values

        self.model.eval()
        preds = []
        df = df.sort_values(["route_id", "date"])
        
        with torch.no_grad():
            for route_id, group in df.groupby("route_id", sort=False):
                vals = group["actual_desi"].values
                if len(vals) >= self.sequence_length:
                    seq = vals[-self.sequence_length:][np.newaxis, :, np.newaxis]
                else:
                    pad = np.pad(vals, (self.sequence_length - len(vals), 0), 'edge')
                    seq = pad[np.newaxis, :, np.newaxis]
                
                tensor_seq = torch.tensor(seq, dtype=torch.float32)
                pred_val = self.model(tensor_seq).item()
                preds.extend([pred_val] * len(group))

        return np.maximum(10.0, np.array(preds))
