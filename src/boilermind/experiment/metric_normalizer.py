from __future__ import annotations

from numbers import Real
from typing import Mapping


MetricValue = float | str


_ALIASES: dict[str, tuple[str, ...]] = {
    "MAE": ("MAE", "mae_m3_s", "mae_t_h", "mae"),
    "RMSE": ("RMSE", "rmse_m3_s", "rmse_t_h", "rmse"),
    "R2": ("R2", "r2_m3_s", "r2_t_h", "r2"),
    "MBE": ("MBE", "mbe_m3_s", "mbe_t_h", "mbe"),
}


def canonical_metric_name(metric_identifier: str) -> str:
    """Normalize a metric identifier without inventing a measurement.

    Planning contracts may include the validation scope in the identifier
    (for example ``validation_mae_t_h``), while result dictionaries store the
    metric itself (``MAE`` or ``mae_t_h``). Unknown identifiers remain unknown
    so downstream selection continues to fail closed.
    """

    identifier = str(metric_identifier).strip()
    comparable = identifier.casefold()
    if comparable.startswith("validation_"):
        comparable = comparable.removeprefix("validation_")
    for canonical, aliases in _ALIASES.items():
        if comparable in {alias.casefold() for alias in aliases}:
            return canonical
    return identifier.upper()


def _unit_from_keys(metrics: Mapping[str, object]) -> str | None:
    keys = set(metrics)
    has_volume = any(key.endswith("_m3_s") for key in keys)
    has_mass = any(key.endswith("_t_h") for key in keys)
    if has_volume == has_mass:
        # Neither unit is declared, or conflicting unit families are mixed.
        # In both cases the unit must remain unspecified.
        return None
    return "m3/s" if has_volume else "t/h"


def normalize_metrics(metrics: Mapping[str, object]) -> dict[str, MetricValue]:
    """Return canonical metric names without deriving missing measurements.

    Existing canonical keys always win. Unit-specific aliases are consulted in
    a deterministic order only when the canonical key is absent. Unknown keys
    are intentionally not copied into the normalized view; callers retain the
    original mapping separately as ``raw_metrics``.
    """

    normalized: dict[str, MetricValue] = {}
    for canonical, candidates in _ALIASES.items():
        for candidate in candidates:
            value = metrics.get(candidate)
            if isinstance(value, Real) and not isinstance(value, bool):
                normalized[canonical] = float(value)
                break
    unit = _unit_from_keys(metrics)
    if unit is not None:
        normalized["metric_unit"] = unit
    return normalized


def numeric_normalized_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """Canonical numeric view suitable for metric contract dictionaries."""

    return {
        key: float(value)
        for key, value in normalize_metrics(metrics).items()
        if key != "metric_unit"
    }
