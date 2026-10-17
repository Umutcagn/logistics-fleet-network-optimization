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
    Day-ahead forecaster over a sliding window of past daily volumes per route.
    Each route is scaled by its training-period mean so one network serves all routes.
    """
    def __init__(self, sequence_length=30, hidden_dim=32, epochs=200, lr=0.01, seed=42):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.model = None
        self.route_scale = {}
        self.history = None

    def prepare_sequences(self, df: pd.DataFrame):
        df = df.sort_values(["route_id", "date"])
        sequences = []
        targets = []

        for route_id, group in df.groupby("route_id"):
            values = group["actual_desi"].values / self.route_scale[route_id]
            if len(values) <= self.sequence_length:
                continue
            for i in range(len(values) - self.sequence_length):
                sequences.append(values[i : i + self.sequence_length])
                targets.append(values[i + self.sequence_length])

        if len(sequences) == 0:
            return np.empty((0, self.sequence_length, 1)), np.empty((0, 1))

        X_seq = np.array(sequences)[:, :, np.newaxis]
        y_target = np.array(targets)[:, np.newaxis]
        return X_seq, y_target

    def fit(self, df: pd.DataFrame):
        torch.manual_seed(self.seed)
        self.route_scale = df.groupby("route_id")["actual_desi"].mean().clip(lower=1.0).to_dict()
        self.history = df[["route_id", "date", "actual_desi"]].copy()

        X_seq, y_target = self.prepare_sequences(df)
        if len(X_seq) == 0:
            print(f"[Warning] Dataset too short for {self.sequence_length}-day LSTM window. Using 7-day window.")
            self.sequence_length = 7
            X_seq, y_target = self.prepare_sequences(df)
        if len(X_seq) == 0:
            self.model = None
            return self

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
        """
        Day-ahead forecast for every row of `df`, returned in the same row order.
        Each row only sees actual volumes from dates strictly before its own date
        (training history plus earlier rows of `df`), never its own value.
        """
        if self.model is None:
            # Simple fallback if model failed to train due to window constraints
            return np.maximum(10.0, df["rolling_mean_7"].fillna(df["rolling_mean_7"].mean()).values)

        known = pd.concat([self.history, df[["route_id", "date", "actual_desi"]]])
        known = known.drop_duplicates(["route_id", "date"]).sort_values(["route_id", "date"])
        series = {route_id: g for route_id, g in known.groupby("route_id")}

        sequences, scales = [], []
        for route_id, date in zip(df["route_id"], df["date"]):
            scale = self.route_scale.get(route_id, 1.0)
            past = series[route_id]
            past = past.loc[past["date"] < date, "actual_desi"].values[-self.sequence_length:] / scale
            if len(past) == 0:
                past = np.ones(1)
            if len(past) < self.sequence_length:
                past = np.pad(past, (self.sequence_length - len(past), 0), "edge")
            sequences.append(past)
            scales.append(scale)

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(np.array(sequences)[:, :, np.newaxis], dtype=torch.float32)
            preds = self.model(X).squeeze(-1).numpy() * np.array(scales)

        return np.maximum(10.0, preds)
