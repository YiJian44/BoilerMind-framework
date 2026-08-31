"""两类知识图谱（文献图谱 + 演化图谱）的构建与 API 测试。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.research_api.knowledge_graph_api import (
    build_literature_graph,
    ensure_evolution_graph_synced,
    load_evolution_graph,
)


app_module = importlib.import_module("server.research_api.app")


def test_literature_graph_built_from_local_corpus() -> None:
    graph = build_literature_graph()
    assert graph["schema_version"] == "boilermind.literature_graph.v1"
    assert graph["summary"]["paper_count"] >= 100
    assert graph["summary"]["chunk_count"] >= 10000
    types = {node["type"] for node in graph["nodes"]}
    assert {"Paper", "Author", "Topic", "CorpusLevel"} <= types
    edge_types = {edge["type"] for edge in graph["edges"]}
    assert {"belongs_to", "authored", "about"} <= edge_types
    paper = next(node for node in graph["nodes"] if node["type"] == "Paper")
    assert paper["title"]
    assert paper["chunk_count"] > 0


def _make_completed_run(run_root: Path, run_id: str) -> None:
    problem_id = f"RP-{run_id}"
    state = {
        "schema_version": "boilermind.research_run.v2",
        "run_id": run_id,
        "question": "测试问题",
        "status": "COMPLETED",
        "research_problem": {
            "problem_id": problem_id,
            "original_question": "测试问题",
            "research_object": "锅炉",
            "target_variable": "steam_volumetric_flow",
            "operating_condition": "稳态",
            "research_goal": "验证假设",
        },
        "hypotheses": [{
            "hypothesis_id": "H_TEST",
            "id": "H_TEST",
            "title": "测试假设",
            "hypothesis": "增加输入可提升预测精度",
            "hypothesis_statement": "增加输入可提升预测精度",
            "mechanism_chain": "输入增加 -> 信息增强 -> 误差降低",
            "problem_id": problem_id,
        }],
        "batches": [{
            "batch_id": f"BATCH-{run_id}-01",
            "round_index": 1,
            "ranking_snapshot_id": f"RANK-{run_id}-000",
            "status": "COMPLETED",
            "members": [{
                "hypothesis_id": "H_TEST",
                "plan": {"plan_id": "PLAN-1", "hypothesis_id": "H_TEST"},
                "contract": {"experiment_id": f"EXP-{run_id}", "hypothesis_id": "H_TEST", "plan_id": "PLAN-1"},
                "status": "COMPLETED",
                "outcome": {
                    "experiment_result": {
                        "experiment_id": f"EXP-{run_id}",
                        "problem_id": problem_id,
                        "hypothesis_id": "H_TEST",
                        "plan_id": "PLAN-1",
                        "status": "completed",
                        "metrics": {"MAE": 0.04, "R2": 0.96},
                        "model_records": {"ridge": {"model_name": "ridge", "fit_success": True, "fit_converged": True}},
                        "experiment_valid": True,
                        "experiment_validity_issues": [],
                        "started_at": "2026-08-26T00:00:00Z",
                        "completed_at": "2026-08-26T00:00:01Z",
                    },
                    "scientific_result": {
                        "hypothesis_id": "H_TEST",
                        "experiment_id": f"EXP-{run_id}",
                        "verdict": "supported",
                        "rationale": "测试",
                    },
                    "audit": {
                        "experiment_id": f"EXP-{run_id}",
                        "execution_valid": True,
                        "dataset_frozen": True,
                        "leakage_check_passed": True,
                        "baseline_valid": True,
                        "metric_check_passed": True,
                        "issues": [],
                    },
                },
            }],
        }],
    }
    (run_root / run_id).mkdir(parents=True, exist_ok=True)
    (run_root / run_id / "run.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def test_evolution_graph_sync_replays_completed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs"
    graph_path = tmp_path / "evolution_graph.json"
    _make_completed_run(run_root, "RUN-KG-TEST")

    changed = ensure_evolution_graph_synced(run_root, graph_path)
    assert changed is True

    graph = load_evolution_graph(graph_path)
    assert any(node["id"] == "H_TEST" and node["type"] == "Hypothesis" for node in graph["nodes"])
    assert any(node["type"] == "ValidatedHypothesis" for node in graph["nodes"])
    edge_types = {edge["type"] for edge in graph["edges"]}
    assert {"validated_by", "generates"} <= edge_types

    # 幂等：再次同步不新增节点
    before = graph["summary"]["node_count"]
    ensure_evolution_graph_synced(run_root, graph_path)
    after = load_evolution_graph(graph_path)["summary"]["node_count"]
    assert before == after


def test_knowledge_graph_api_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "runs"
    graph_path = tmp_path / "evolution_graph.json"
    _make_completed_run(run_root, "RUN-KG-API")
    monkeypatch.setattr(app_module.pipeline, "run_root", run_root)
    monkeypatch.setattr(
        "server.research_api.knowledge_graph_api.EVOLUTION_GRAPH", graph_path
    )
    client = TestClient(app_module.app)

    literature = client.get("/api/v1/knowledge-graph/literature")
    assert literature.status_code == 200
    payload = literature.json()["data"]
    assert payload["summary"]["paper_count"] >= 100

    evolution = client.get("/api/v1/knowledge-graph/evolution")
    assert evolution.status_code == 200
    payload = evolution.json()["data"]
    assert any(node["id"] == "H_TEST" for node in payload["nodes"])
    assert payload["summary"]["node_count"] >= 3
