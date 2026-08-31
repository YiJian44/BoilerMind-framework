from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any


def _module_available(module_name: str) -> bool:
    return (
        importlib.util.find_spec(module_name)
        is not None
    )


@dataclass(frozen=True)
class ExecutionEnvironment:
    """
    Lightweight, deterministic description of what the CURRENT
    machine can run. It performs NO performance benchmarking;
    it only answers capability questions (dependency / device).

    Deploying the same BoilerMind code on:
      - local Windows CPU
      - CPU server
      - GPU server
    must not require any research-business code change; only
    this environment changes.
    """

    os: str

    python_version: str

    sklearn_available: bool

    torch_available: bool

    cuda_available: bool

    gpu_available: bool

    gpu_device: str | None = None

    optional_dependencies: dict[str, bool] = field(
        default_factory=dict,
    )

    # Hardware metadata is informational only - never used as
    # a performance benchmark or as a registry gate.
    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    @classmethod
    def detect(cls) -> "ExecutionEnvironment":
        """
        Detect the current runtime capabilities.
        """

        torch_available = _module_available("torch")

        cuda_available = False
        gpu_available = False
        gpu_device: str | None = None

        if torch_available:
            try:
                import torch

                cuda_available = bool(
                    torch.cuda.is_available()
                )

                if cuda_available:
                    gpu_available = True
                    gpu_device = (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.device_count() > 0
                        else None
                    )
            except Exception:
                cuda_available = False

        optional_dependencies = {
            "dashscope": _module_available(
                "dashscope"
            ),
            "xgboost": _module_available("xgboost"),
            "pandas": _module_available("pandas"),
            "numpy": _module_available("numpy"),
            "joblib": _module_available("joblib"),
            "scipy": _module_available("scipy"),
            "openai": _module_available("openai"),
        }

        try:
            import psutil

            memory_gb = round(
                psutil.virtual_memory().total
                / (1024**3),
                1,
            )
        except Exception:
            memory_gb = None

        return cls(
            os=platform.system(),
            python_version=(
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            sklearn_available=_module_available(
                "sklearn"
            ),
            torch_available=torch_available,
            cuda_available=cuda_available,
            gpu_available=gpu_available,
            gpu_device=gpu_device,
            optional_dependencies=(
                optional_dependencies
            ),
            metadata={
                "cpu_count": os.cpu_count(),
                "memory_gb": memory_gb,
                "platform": platform.platform(),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "os": self.os,
            "python_version": self.python_version,
            "sklearn_available": (
                self.sklearn_available
            ),
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
            "gpu_available": self.gpu_available,
            "gpu_device": self.gpu_device,
            "optional_dependencies": dict(
                self.optional_dependencies
            ),
            "metadata": dict(self.metadata),
        }
