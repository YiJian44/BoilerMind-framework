from __future__ import annotations

from typing import Any, Callable


ArchitectureFactory = Callable[[Any, dict[str, Any]], Any]
_FACTORIES: dict[str, ArchitectureFactory] = {}


def register_torch_architecture(name: str, factory: ArchitectureFactory) -> None:
    key = name.strip().lower()
    if not key or key in _FACTORIES:
        raise ValueError(f"duplicate_or_empty_torch_architecture:{key}")
    _FACTORIES[key] = factory


def get_torch_architecture_factory(name: str) -> ArchitectureFactory:
    try:
        return _FACTORIES[name.strip().lower()]
    except KeyError as exc:
        raise KeyError(f"unknown_torch_architecture:{name}") from exc


def has_torch_architecture(name: str) -> bool:
    return name.strip().lower() in _FACTORIES


def build_torch_architecture(name: str, torch: Any, config: dict[str, Any]) -> Any:
    return get_torch_architecture_factory(name)(torch, config)


def _recurrent(kind: str) -> ArchitectureFactory:
    def factory(torch: Any, config: dict[str, Any]) -> Any:
        nn = torch.nn
        input_size = int(config["input_size"])
        hidden_size = int(config.get("hidden_size", 32))

        class RecurrentRegressor(nn.Module):
            def __init__(self):
                super().__init__()
                recurrent = nn.LSTM if kind == "lstm" else nn.GRU
                self.recurrent = recurrent(input_size, hidden_size, batch_first=True)
                self.head = nn.Linear(hidden_size, int(config.get("output_size", 1)))

            def forward(self, x):
                values, _state = self.recurrent(x)
                return self.head(values[:, -1, :])

        return RecurrentRegressor()
    return factory


def _dlinear(torch: Any, config: dict[str, Any]) -> Any:
    nn = torch.nn
    width = int(config["window_steps"]) * int(config["input_size"])
    return nn.Sequential(nn.Flatten(), nn.Linear(width, int(config.get("output_size", 1))))


def _transformer(torch: Any, config: dict[str, Any]) -> Any:
    nn = torch.nn
    input_size = int(config["input_size"])
    d_model = int(config.get("d_model", 32))
    nhead = int(config.get("nhead", 4))

    class TransformerRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(input_size, d_model)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(config.get("num_layers", 1)))
            self.head = nn.Linear(d_model, int(config.get("output_size", 1)))

        def forward(self, x):
            return self.head(self.encoder(self.input(x))[:, -1, :])

    return TransformerRegressor()


register_torch_architecture("lstm", _recurrent("lstm"))
register_torch_architecture("gru", _recurrent("gru"))
register_torch_architecture("transformer", _transformer)
register_torch_architecture("dlinear", _dlinear)
