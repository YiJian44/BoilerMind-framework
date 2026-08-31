from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from boilermind.core.contracts import ExperimentObservation, HistoricalExperimentRecord


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class ExperimentMemoryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.raw_path = self.root / "raw_experiments.jsonl"
        self.observations_path = self.root / "experiment_observations.jsonl"
        self.series_path = self.root / "experiment_series.jsonl"
        self.issues_path = self.root / "ingestion_issues.jsonl"
        self.index_path = self.root / "experiment_memory.sqlite3"

    def load_records(self) -> list[HistoricalExperimentRecord]:
        if not self.raw_path.is_file():
            return []
        return [HistoricalExperimentRecord.model_validate_json(line) for line in self.raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def load_observations(self) -> list[ExperimentObservation]:
        if not self.observations_path.is_file():
            return []
        return [ExperimentObservation.model_validate_json(line) for line in self.observations_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def replace_all(self, records: list[HistoricalExperimentRecord], observations: list[ExperimentObservation], issues: list[dict]) -> None:
        ids = [record.experiment_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate_experiment_id")
        if self.raw_path.is_file():
            existing = {item.experiment_id: item for item in self.load_records()}
            for record in records:
                previous = existing.get(record.experiment_id)
                if previous is not None:
                    old_payload = previous.model_dump(mode="json", exclude={"imported_at"})
                    new_payload = record.model_dump(mode="json", exclude={"imported_at"})
                    if old_payload != new_payload:
                        raise ValueError(f"immutable_experiment_conflict:{record.experiment_id}")
        _write_jsonl(self.raw_path, [item.model_dump(mode="json") for item in records])
        _write_jsonl(self.observations_path, [item.model_dump(mode="json") for item in observations])
        grouped: dict[str, list[HistoricalExperimentRecord]] = {}
        for record in records:
            grouped.setdefault(record.series_id, []).append(record)
        series_rows = [{
            "series_id": series_id,
            "experiment_ids": [item.experiment_id for item in items],
            "hypothesis_ids": sorted({item.hypothesis_id for item in items if item.hypothesis_id}),
            "summary": f"{len(items)} experiment(s)",
            "evidence_tier": max((item.evidence_tier for item in items), key=lambda tier: {
                "AUDITED_CONFIRMATORY": 5, "AUDITED_EXPLORATORY": 4, "LEGACY_INFORMATIVE": 3,
                "ENGINEERING_FAILURE": 2, "PLANNED_NOT_EXECUTED": 1,
            }[tier.value]).value,
        } for series_id, items in sorted(grouped.items())]
        _write_jsonl(self.series_path, series_rows)
        _write_jsonl(self.issues_path, issues)
        self.rebuild_index(records, observations)

    def append_record(self, record: HistoricalExperimentRecord, observations: list[ExperimentObservation]) -> None:
        records = self.load_records()
        existing = {item.experiment_id: item for item in records}
        if record.experiment_id in existing:
            if existing[record.experiment_id].model_dump(mode="json") != record.model_dump(mode="json"):
                raise ValueError(f"immutable_experiment_conflict:{record.experiment_id}")
            return
        records.append(record)
        all_observations = self.load_observations() + observations
        issues = [json.loads(line) for line in self.issues_path.read_text(encoding="utf-8").splitlines() if line.strip()] if self.issues_path.is_file() else []
        self.replace_all(records, all_observations, issues)

    def rebuild_index(self, records: list[HistoricalExperimentRecord] | None = None, observations: list[ExperimentObservation] | None = None) -> None:
        records = records if records is not None else self.load_records()
        observations = observations if observations is not None else self.load_observations()
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.index_path)
        try:
            connection.executescript("DROP TABLE IF EXISTS experiments; DROP TABLE IF EXISTS observations; DROP TABLE IF EXISTS memory_fts;")
            connection.execute("CREATE TABLE experiments (experiment_id TEXT PRIMARY KEY, series_id TEXT, evidence_tier TEXT, target_variable TEXT, prediction_mode TEXT, thermodynamic_standard TEXT, horizon INTEGER, window_steps INTEGER, dataset_sha256 TEXT, payload TEXT NOT NULL)")
            connection.execute("CREATE TABLE observations (observation_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, observation_type TEXT, invalid INTEGER, claim TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute("CREATE VIRTUAL TABLE memory_fts USING fts5(record_type, record_id UNINDEXED, content, tokenize='unicode61')")
            for record in records:
                payload = record.model_dump_json()
                connection.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    record.experiment_id, record.series_id, record.evidence_tier.value, record.scope.target_variable,
                    record.scope.prediction_mode, record.scope.thermodynamic_standard,
                    record.scope.prediction_horizon_steps, record.scope.window_steps, record.scope.dataset_sha256, payload,
                ))
                content = " ".join((record.raw_context, record.raw_hypothesis, record.raw_protocol, record.raw_result, record.raw_limitations))
                connection.execute("INSERT INTO memory_fts VALUES ('experiment', ?, ?)", (record.experiment_id, content))
            for observation in observations:
                experiment_id = observation.source_experiment_ids[0]
                connection.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)", (
                    observation.observation_id, experiment_id, observation.observation_type.value,
                    int(observation.invalid_for_scientific_synthesis), observation.claim, observation.model_dump_json(),
                ))
                connection.execute("INSERT INTO memory_fts VALUES ('observation', ?, ?)", (observation.observation_id, observation.claim))
            connection.commit()
        finally:
            connection.close()


def index_experiment_memory(records: list[HistoricalExperimentRecord], observations: list[ExperimentObservation], root: str | Path) -> ExperimentMemoryStore:
    store = ExperimentMemoryStore(root)
    store.replace_all(records, observations, [])
    return store
