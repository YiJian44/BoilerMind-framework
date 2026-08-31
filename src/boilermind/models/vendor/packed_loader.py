"""Self-contained deep-model loader (vendor, BoilerMind-Trusted).

Resolves vendored architecture source relative to this file, so the
loader works standalone after cloning BoilerMind-Trusted. Checkpoints
and scalers are NOT vendored; they must be supplied externally.
"""
import json
import sys
from pathlib import Path

import numpy as np
import joblib
import torch

VENDOR = Path(__file__).resolve().parent


class PackagedModel:
    """Load a packaged deep model (config + scaler + weights) and predict."""

    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir)
        with open(self.package_dir / "config.json") as f:
            self.config = json.load(f)
        with open(self.package_dir / "feature_schema.json") as f:
            self.schema = json.load(f)
        self.scaler = joblib.load(self.package_dir / "scaler.joblib")
        self.model = self._build_model()
        self.model.load_state_dict(
            torch.load(self.package_dir / "model.pth", map_location="cpu"))
        self.model.eval()
        self.n_past = self.config.get("n_past", 20)
        self.offset = self.config.get("offset", 40)

    def _build_model(self):
        from torch import nn
        arch = self.config["architecture"]

        if arch == "transformer":
            sys.path.insert(0, str(VENDOR / "transformer"))
            from transformer import transformer as Transformer
            return Transformer(
                n_past=self.config["n_past"], n_future=1,
                d_model=self.config["d_model"], d_ff=self.config["d_ff"],
                num_heads=self.config["num_heads"], num_layers=self.config["num_layers"],
                dropout=self.config["dropout"], top_k=self.config["top_k"],
            )

        if arch == "dlinear":
            npast = self.config["n_past"]
            nfeat = self.config["n_feat"]

            class DLinear(nn.Module):
                def __init__(self):
                    super().__init__()
                    s = npast * nfeat
                    self.trend_linear = nn.Linear(s, 1)
                    self.resid_linear = nn.Linear(s, 1)

                def forward(self, x):
                    B, L, F = x.shape
                    hw = 2
                    pad = torch.cat([x[:, :1].expand(-1, hw, -1), x, x[:, -1:].expand(-1, hw, -1)], dim=1)
                    t = pad.unfold(1, 5, 1).mean(-1).mean(-1).unsqueeze(-1).expand(-1, -1, F)
                    if t.shape[1] > L:
                        t = t[:, :L]
                    r = x - t
                    return self.trend_linear(t.reshape(B, -1)) + self.resid_linear(r.reshape(B, -1))
            return DLinear()

        if arch in ("lstm", "gru"):
            idim = self.config["input_dim"]
            hdim = self.config["hidden"]
            nlay = self.config["n_layers"]
            dp = self.config["dropout"]

            class RecurrentWrapper(nn.Module):
                def __init__(self):
                    super().__init__()
                    # attribute name must match the trained checkpoint keys (lstm.* / gru.*)
                    if arch == "lstm":
                        self.lstm = nn.LSTM(idim, hdim, nlay, dropout=dp if nlay > 1 else 0, batch_first=True)
                    else:
                        self.gru = nn.GRU(idim, hdim, nlay, dropout=dp if nlay > 1 else 0, batch_first=True)
                    self.fc = nn.Sequential(nn.Linear(hdim, 32), nn.ReLU(), nn.Linear(32, 1))

                def forward(self, x):
                    out, _ = self.lstm(x) if arch == "lstm" else self.gru(x)
                    return self.fc(out[:, -1, :])
            return RecurrentWrapper()

        if arch in ("patchtst", "itransformer", "timesnet"):
            from types import SimpleNamespace
            sys.path.insert(0, str(VENDOR / "tsl"))
            sys.path.insert(0, str(VENDOR / "offline_shims"))
            if arch == "patchtst":
                from models.PatchTST import Model as TSLModel
                tcfg = dict(d_model=self.config["d_model"], n_heads=self.config["n_heads"],
                            e_layers=self.config["e_layers"], d_ff=self.config["d_ff"],
                            dropout=self.config["dropout"], activation=self.config["activation"],
                            factor=self.config["factor"], patch_len=self.config["patch_len"],
                            stride=self.config["stride"])
            elif arch == "itransformer":
                from models.iTransformer import Model as TSLModel
                tcfg = dict(d_model=self.config["d_model"], n_heads=self.config["n_heads"],
                            e_layers=self.config["e_layers"], d_ff=self.config["d_ff"],
                            dropout=self.config["dropout"], activation=self.config["activation"],
                            factor=self.config["factor"], embed=self.config["embed"],
                            freq=self.config["freq"])
            else:  # timesnet
                from models.TimesNet import Model as TSLModel
                tcfg = dict(d_model=self.config["d_model"], d_ff=self.config["d_ff"],
                            e_layers=self.config["e_layers"], dropout=self.config["dropout"],
                            activation=self.config["activation"], factor=self.config["factor"],
                            top_k=self.config["top_k"], num_kernels=self.config["num_kernels"],
                            embed=self.config["embed"], freq=self.config["freq"],
                            label_len=self.config["label_len"], c_out=self.config["c_out"])
            cfg = SimpleNamespace(
                task_name='long_term_forecast', seq_len=self.config["n_past"], pred_len=1,
                enc_in=self.config["enc_in"], **tcfg)

            class TSLSoftSensor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = TSLModel(cfg)
                    self.head = nn.Linear(cfg.enc_in, 1)

                def forward(self, x):
                    out = self.backbone(x, None, None, None)
                    return self.head(out[:, -1, :])
            return TSLSoftSensor()

        raise ValueError(f"Unknown architecture: {arch}")

    def predict(self, data: np.ndarray) -> np.ndarray:
        dummy_y = np.zeros((len(data), 1), dtype=np.float32)
        stacked = np.hstack([data, dummy_y])
        scaled = self.scaler.transform(stacked)
        X_scaled = scaled[:, :30]

        if len(data) < self.n_past:
            raise ValueError(f"Need at least {self.n_past} rows, got {len(data)}")

        windows = np.stack([
            X_scaled[i - self.n_past + 1: i + 1]
            for i in range(self.n_past - 1, len(X_scaled))
        ])
        x_enc = torch.FloatTensor(windows)

        with torch.no_grad():
            if self.config["architecture"] == "transformer":
                x_dec = torch.zeros(len(windows), 1, 1)
                pred_scaled = self.model(x_enc, x_dec).detach().numpy().squeeze()
            else:
                pred_scaled = self.model(x_enc).detach().numpy().squeeze()

        ymin = self.scaler.data_min_[30]
        ymax = self.scaler.data_max_[30]
        predictions = pred_scaled * (ymax - ymin) + ymin

        full_preds = np.full(len(data), np.nan)
        full_preds[self.n_past - 1:] = predictions
        return full_preds
