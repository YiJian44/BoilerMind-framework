"""DLinear - decomposition linear baseline (AAAI 2023).

Architecture source migrated from the legacy
boiler_soft_sensor_models/dlinear/train.py (class only; the old
training main with absolute data paths was not migrated).
"""

import torch
import torch.nn as nn


class DLinear(nn.Module):
    def __init__(self, seq_len, n_feat, kernel=5):
        super().__init__()
        self.kernel = kernel
        in_dim = seq_len * n_feat
        self.trend_linear = nn.Linear(in_dim, 1)
        self.resid_linear = nn.Linear(in_dim, 1)

    def forward(self, x):  # x: (B, L, F)
        B, L, F = x.shape
        # Moving average for trend
        x_pad = torch.cat(
            [
                x[:, :1].expand(-1, self.kernel // 2, -1),
                x,
                x[:, -1:].expand(
                    -1,
                    self.kernel // 2,
                    -1,
                ),
            ],
            dim=1,
        )
        trend = (
            x_pad.unfold(1, self.kernel, 1)
            .mean(-1)
            .mean(-1)
            .unsqueeze(-1)
            .expand(-1, -1, F)
        )
        # Align trend to same length
        if trend.shape[1] > L:
            trend = trend[:, :L]
        residual = x - trend
        # Flatten and predict
        t_out = self.trend_linear(trend.reshape(B, -1))
        r_out = self.resid_linear(residual.reshape(B, -1))
        return t_out + r_out
