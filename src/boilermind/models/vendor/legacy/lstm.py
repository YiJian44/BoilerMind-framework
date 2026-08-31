"""LSTM — steam flow prediction (5min -> 10min, R2=0.810)"""
import torch, torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=30, hidden=48, n_layers=1, dropout=0.3, n_past=20):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers>1 else 0)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
