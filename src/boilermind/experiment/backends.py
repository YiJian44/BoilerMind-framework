from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from boilermind.models.execution_environment import (
    ExecutionEnvironment,
)


class ExecutionBackend(ABC):
    """
    Execution abstraction for future CPU / GPU / remote
    servers. The research business layer only calls
    runner.execute(contract); it never binds to a machine
    path.
    """

    name: str = "abstract"

    def __init__(
        self,
        *,
        environment: ExecutionEnvironment | None = None,
    ):
        self.environment = (
            environment
            or ExecutionEnvironment.detect()
        )

    @abstractmethod
    def execute(
        self,
        contract: Any,
        **kwargs: Any,
    ):
        ...

    def capability_summary(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "environment": self.environment.to_dict(),
        }


class LocalCPUBackend(ExecutionBackend):
    """
    Current local unified execution path. Target/framework routing remains
    inside UnifiedExperimentRunner so this machine backend cannot create a
    second scientific workflow.
    """

    name = "local_cpu"

    def __init__(
        self,
        *,
        environment: ExecutionEnvironment | None = None,
        runner: Any | None = None,
    ):
        super().__init__(environment=environment)

        from boilermind.experiment.unified_runner import (
            UnifiedExperimentRunner,
        )

        self.runner = (
            runner or UnifiedExperimentRunner()
        )

    def execute(
        self,
        contract: Any,
        **kwargs: Any,
    ):
        return self.runner.run(contract)


class RemoteCPUBackend(ExecutionBackend):
    """
    Interface placeholder for a future remote CPU server.
    """

    name = "remote_cpu"

    def execute(
        self,
        contract: Any,
        **kwargs: Any,
    ):
        raise NotImplementedError(
            "remote_cpu_backend_placeholder"
        )


class CUDABackend(ExecutionBackend):
    """
    Capability-aware CUDA backend placeholder.

    Fails closed when CUDA is not available.
    """

    name = "cuda"

    def execute(
        self,
        contract: Any,
        **kwargs: Any,
    ):
        if not self.environment.cuda_available:
            raise RuntimeError(
                "cuda_required_but_unavailable"
            )

        raise NotImplementedError(
            "cuda_backend_executor_not_wired"
        )


def get_execution_backend(
    name: str,
    *,
    environment: ExecutionEnvironment | None = None,
) -> ExecutionBackend:
    backends: dict[str, type[ExecutionBackend]] = {
        "local_cpu": LocalCPUBackend,
        "remote_cpu": RemoteCPUBackend,
        "cuda": CUDABackend,
    }

    if name not in backends:
        raise ValueError(
            "unknown_execution_backend:"
            f"{name}"
        )

    return backends[name](
        environment=environment,
    )
