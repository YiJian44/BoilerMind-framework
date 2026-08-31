from __future__ import annotations

from pathlib import Path


def test_production_has_only_canonical_research_entrypoint() -> None:
    root = Path(__file__).resolve().parents[2]
    banned = {
        "ResearchPipeline",
        "ResearchSupervisorAgent",
        "ResearchEngine",
        "execute_research_cycle",
        "execute_closed_loop",
        "TestOnlyExperimentRunner(",
        "SkillRuntime",
    }
    offenders: list[str] = []
    for directory in ("src", "server", "scripts"):
        for path in (root / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in banned:
                if token in text:
                    offenders.append(f"{path.relative_to(root)}:{token}")
    assert offenders == []


def test_no_tracked_backup_or_patch_sources_remain() -> None:
    root = Path(__file__).resolve().parents[2]
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix == ".bak"
            or path.name.startswith("patch_")
            or ".before_" in path.name
            or path.name.startswith("before_")
        )
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and not any(part in {".venv", ".venv311", "venv", "env"} for part in path.parts)
    ]
    assert offenders == []
