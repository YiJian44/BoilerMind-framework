from typing import Any

__all__ = ["ResearchOrchestrator"]


def __getattr__(name: str) -> Any:
    if name == "ResearchOrchestrator":
        from .research_orchestrator import ResearchOrchestrator

        return ResearchOrchestrator
    raise AttributeError(name)
