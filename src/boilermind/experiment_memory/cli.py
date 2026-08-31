from __future__ import annotations

import argparse
import json
from pathlib import Path

from .importer import import_experiment_history, validate_historical_experiment
from .observations import derive_experiment_observations
from .persistence import build_empirical_capability_profile
from .store import ExperimentMemoryStore


def migrate(source: str | Path, destination: str | Path) -> dict:
    records, issues = import_experiment_history(source)
    for record in records:
        for issue in validate_historical_experiment(record):
            issues.append({"experiment_id": record.experiment_id, "issue": issue, "source_locator": record.source_locator})
    observations = [observation for record in records for observation in derive_experiment_observations(record)]
    store = ExperimentMemoryStore(destination)
    store.replace_all(records, observations, issues)
    profile = build_empirical_capability_profile(records)
    profile_path = store.root / "empirical_capability_profile.json"
    profile_path.write_text(json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "source": str(Path(source).resolve()),
        "destination": str(store.root),
        "record_count": len(records),
        "observation_count": len(observations),
        "issue_count": len(issues),
        "planned_count": sum(record.evidence_tier.value == "PLANNED_NOT_EXECUTED" for record in records),
        "scientific_synthesis_eligible_count": sum(record.scientific_synthesis_eligible for record in records),
    }
    (store.root / "migration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Import historical experiment logs into BoilerMind experiment memory")
    parser.add_argument("source")
    parser.add_argument("--destination", default="runtime/experiment_memory")
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.destination), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
