"""TCN — steam flow prediction (30min -> 10min, R2=0.837)"""
import torch, torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x): return x[:, :, :-self.c]

class TCNBlock(nn.Module):
    def __init__(self, ni, no, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.c1 = nn.Conv1d(ni, no, kernel_size, 1, pad, dilation)
        self.ch1 = Chomp1d(pad)
        self.c2 = nn.Conv1d(no, no, kernel_size, 1, pad, dilation)
        self.ch2 = Chomp1d(pad)
        self.relu = nn.ReLU(); self.drop = nn.Dropout(dropout)
        self.ds = nn.Conv1d(ni, no, 1) if ni != no else None
        nn.init.normal_(self.c1.weight, 0, 0.01); nn.init.normal_(self.c2.weight, 0, 0.01)
    def forward(self, x):
        o = self.relu(self.ch1(self.c1(x))); o = self.drop(self.relu(self.ch2(self.c2(o))))
        r = x if self.ds is None else self.ds(x)
        return self.relu(o + r)

class TCNRegressor(nn.Module):
    def __init__(self, input_dim=30, channels=(64, 96, 128, 128), kernel_size=5, dropout=0.2, n_past=120):
        super().__init__()
        layers = []
        for i in range(len(channels)):
            inp = input_dim if i == 0 else channels[i-1]
            layers.append(TCNBlock(inp, channels[i], kernel_size, 2**i, dropout))
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], 1)
    def forward(self, x):
        o = self.net(x.permute(0, 2, 1))
        return self.fc(o[:, :, -1])
