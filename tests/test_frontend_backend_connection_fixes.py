# -*- coding: utf-8 -*-
"""前端-后端连接修复后的端点与投影字段对齐测试。"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from boilermind.core.contracts import ResearchRunState
from boilermind.orchestration.research_orchestrator import ResearchOrchestrator
from server.research_api.projector import project_run


app_module = importlib.import_module("server.research_api.app")


def _write_run(run_root: Path, run_id: str, question: str, status: str, mtime: int) -> None:
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "boilermind.research_run.v2",
        "run_id": run_id,
        "question": question,
        "status": status,
        "research_problem": {"target_variable": "steam_volumetric_flow"},
        "stage_traces": [],
        "batches": [],
        "report": {},
        "errors": [],
    }
    target = run_dir / "run.json"
    target.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.utime(target, (mtime, mtime))


def test_persist_retries_transient_windows_file_lock(tmp_path, monkeypatch) -> None:
    orchestrator = ResearchOrchestrator(run_root=tmp_path)
    state = ResearchRunState(run_id="RUN-LOCK", question="测试", status="RUNNING")
    original_replace = os.replace
    attempts = 0

    def locked_then_replace(source: str | Path, target: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "拒绝访问", str(target))
        original_replace(source, target)

    monkeypatch.setattr("boilermind.orchestration.research_orchestrator.os.replace", locked_then_replace)
    monkeypatch.setattr("boilermind.orchestration.research_orchestrator.sleep", lambda _: None)

    orchestrator._persist(state)

    assert attempts == 3
    assert json.loads((tmp_path / "RUN-LOCK" / "run.json").read_text(encoding="utf-8"))["run_id"] == "RUN-LOCK"


def test_state_prefers_terminal_memory_state_over_stale_run_file(tmp_path, monkeypatch) -> None:
    run_id = "RUN-STALE"
    _write_run(tmp_path, run_id, "旧状态", "RUNNING", 1000)
    monkeypatch.setattr(app_module.pipeline, "run_root", tmp_path)
    app_module._states[run_id] = {"run_id": run_id, "status": "FAILED", "errors": ["PermissionError"]}

    try:
        state = app_module._state(run_id)
    finally:
        app_module._states.pop(run_id, None)

    assert state["status"] == "FAILED"


def test_state_recovers_terminal_temporary_file_after_replace_failure(tmp_path, monkeypatch) -> None:
    run_id = "RUN-TEMP"
    _write_run(tmp_path, run_id, "旧状态", "RUNNING", 1000)
    temporary = tmp_path / run_id / "run.json.tmp"
    temporary.write_text(json.dumps({"run_id": run_id, "status": "FAILED", "errors": ["PermissionError"]}), encoding="utf-8")
    os.utime(temporary, (2000, 2000))
    monkeypatch.setattr(app_module.pipeline, "run_root", tmp_path)

    state = app_module._state(run_id)

    assert state["status"] == "FAILED"


def test_upload_endpoint_persists_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "UPLOAD_ROOT", tmp_path)
    client = TestClient(app_module.app)
    response = client.post(
        "/api/v1/uploads",
        files={"files": ("note.txt", "hello boilermind".encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    attachment = payload["data"]["attachments"][0]
    assert attachment["id"].startswith("ATT-")
    assert attachment["filename"] == "note.txt"
    stored = next(tmp_path.rglob("note.txt"))
    assert stored.read_text(encoding="utf-8") == "hello boilermind"


def test_assistant_acknowledges_uploaded_attachments(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "UPLOAD_ROOT", tmp_path)
    upload_dir = tmp_path / "ATT-TEST1"
    upload_dir.mkdir()
    (upload_dir / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    client = TestClient(app_module.app)
    response = client.post(
        "/api/v1/assistant",
        json={"question": "测试问题", "attachmentIds": ["ATT-TEST1"]},
    )
    assert response.status_code == 200
    assert "data.csv" in response.json()["answer"]


def test_history_list_filters_and_paginates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(app_module.pipeline, "run_root", tmp_path)
    _write_run(tmp_path, "RUN-AAA", "汽包压力约束下的蒸汽体积流量提升", "COMPLETED", 1000)
    _write_run(tmp_path, "RUN-BBB", "给煤量变化对锅炉效率的影响", "FAILED", 2000)
    _write_run(tmp_path, "RUN-CCC", "送风量优化与燃烧稳定性", "COMPLETED", 3000)
    client = TestClient(app_module.app)

    data = client.get("/api/v1/research-runs").json()["data"]
    assert data["total"] == 3 and len(data["items"]) == 3

    data = client.get("/api/v1/research-runs", params={"status": "completed"}).json()["data"]
    assert data["total"] == 2 and all(item["status"] == "completed" for item in data["items"])

    data = client.get("/api/v1/research-runs", params={"query": "蒸汽"}).json()["data"]
    assert data["total"] == 1 and "蒸汽" in data["items"][0]["question"]

    data = client.get("/api/v1/research-runs", params={"page": 2, "pageSize": 2}).json()["data"]
    assert data["total"] == 3 and len(data["items"]) == 1 and data["page"] == 2

    data = client.get("/api/v1/research-runs", params={"query": "不存在"}).json()["data"]
    assert data["total"] == 0 and data["items"] == []


def _completed_state() -> dict:
    return {
        "run_id": "RUN-UI-001",
        "question": "compare models",
        "status": "COMPLETED",
        "research_problem": {"target_variable": "steam_volumetric_flow"},
        "evidence_bundle": {"evidence": [{"evidence_id": "E1"}]},
        "hypotheses": [{
            "hypothesis_id": "H1", "title": "comparison",
            "hypothesis": "models differ", "confirmation_criteria": ["better"],
        }],
        "hypothesis_states": {"H1": {"latest_verdict": "supported"}},
        "ranking_snapshots": [{"entries": [{"hypothesis_id": "H1", "dynamic_score": 0.9}]}],
        "stage_traces": [
            {"stage": "problem_understanding", "status": "COMPLETED"},
            {"stage": "literature_retrieval", "status": "COMPLETED"},
            {"stage": "execution", "status": "COMPLETED"},
        ],
        "batches": [{
            "members": [{
                "hypothesis_id": "H1", "status": "COMPLETED",
                "plan": {
                    "primary_metric": "MAE", "random_seed": 42,
                    "execution_backend": "real_sklearn", "objective": "objective",
                    "hypothesis_statement": "models differ",
                },
                "contract": {
                    "primary_metric": "MAE", "secondary_metrics": ["RMSE"],
                    "locked_test_used_for_selection": False,
                },
                "outcome": {
                    "experiment_result": {
                        "experiment_id": "EXP-1",
                        "model_records": {
                            "bayesianridge": {
                                "fit_success": True,
                                "validation_metrics": {"MAE": 0.10, "RMSE": 0.20},
                                "locked_test_metrics": {"MAE": 0.15, "RMSE": 0.25},
                                "train_samples": 1200,
                                "validation_samples": 300,
                                "test_samples": 400,
                                "random_seed": 42,
                            },
                        },
                        "baseline_metrics": {"MAE": 0.20, "RMSE": 0.40},
                    },
                    "scientific_result": {"verdict": "supported"},
                    "audit": {"execution_valid": True, "leakage_check_passed": True},
                },
            }],
        }],
        "report": {},
        "errors": [],
    }


def test_frontend_schema_alignment_fields(tmp_path) -> None:
    (tmp_path / "structured_report.json").write_text("{}", encoding="utf-8")
    projected = project_run(_completed_state(), tmp_path)
    execution = projected["execution"]
    assert execution["rows"][0]["sample_counts"]["train"] == 1200
    assert execution["executed_step_ids"] == ["problem_understanding", "literature_retrieval", "execution"]
    assert execution["environment"]["adapter"]
    assert all(stage["summary"] for stage in projected["stages"])
    top = next(item for item in projected["hypotheses"] if item["rank"] == 1)
    assert top["selection_reason"]


def test_degraded_evidence_exposes_local_library_stats(tmp_path) -> None:
    state = {
        "run_id": "RUN-DEG-1",
        "question": "锅炉汽包压力对蒸汽体积流量的影响",
        "status": "COMPLETED",
        "research_problem": {
            "problem_id": "RP-DEG-1",
            "original_question": "锅炉汽包压力对蒸汽体积流量的影响",
            "research_object": "锅炉汽水系统",
            "target_variable": "steam_volumetric_flow",
            "operating_condition": "全工况",
            "research_goal": "验证汽包压力与蒸汽体积流量的关系",
            "manipulated_variables": ["汽包压力"],
            "observed_variables": ["蒸汽体积流量"],
            "context_variables": [],
        },
        "evidence_bundle": None,
        "hypotheses": [{"hypothesis_id": "H1", "hypothesis": "汽包压力升高会提升蒸汽体积流量"}],
        "stage_traces": [{"stage": "literature_retrieval", "status": "FAILED"}],
        "batches": [],
        "report": {},
        "errors": [],
    }
    projected = project_run(state, tmp_path)
    summary = projected["evidence_summary"]
    assert summary["degraded"] is True
    assert summary["local_stats"]["paper_count"] == 110
    assert summary["local_stats"]["chunk_count"] == 10640
    assert "core" in summary["local_stats"]["by_corpus_level"]
    assert "本地文献库" in (summary["degraded_note"] or "")
    assert summary["degraded_candidates"]
