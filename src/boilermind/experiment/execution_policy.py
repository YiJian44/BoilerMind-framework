from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPolicy:
    max_runtime_per_model: float | None = None
    max_epochs: int | None = None
    allowed_devices: tuple[str, ...] = ("cpu", "cuda")
    allow_partial_failure: bool = False

    @classmethod
    def from_contract(cls, contract):
        return cls(
            max_runtime_per_model=contract.max_runtime_per_model,
            max_epochs=contract.max_epochs,
            allowed_devices=tuple(contract.allowed_devices),
            allow_partial_failure=contract.allow_partial_failure,
        )

    def validate_device(self, device: str) -> None:
        if device not in self.allowed_devices:
            raise RuntimeError(f"device_not_allowed_by_execution_policy:{device}")
