"""Unity 真实推送闭环：WS 桥接 + 发送→接收→执行→回传状态流转。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from boilermind.core.contracts import (
    BatchMember,
    HypothesisValidationBatch,
    ResearchRunState,
)


app_module = importlib.import_module("server.research_api.app")


QUESTION = (
    "在汽包压力不超过23MPa的限制下，怎么调节给煤、给水、送风和汽包压力，"
    "使蒸汽体积量V上升15%，并将验证结果推送到Unity？"
)


def _unity_payload(run_id: str) -> dict:
    return {
        "schema_version": "boilermind.unity_control.v1",
        "run_id": run_id,
        "variable_order": ["给煤", "给水", "送风", "汽包压力"],
        "current_values": [251.04, 95.991, 752.077, 19.01],
        "recommended_values": [223.5012, 113.1199, 853.3014, 14.4124],
        "adjustment_ranges": [[192.4, 313.4], [88.2, 119.9], [564.5, 939.8], [14.3, 17.4]],
        "pressure_limit_mpa": 23.0,
        "target_rise": 0.15,
        "predicted_rise": 0.1499833,
        "current_volume": 3.6544675,
        "predicted_volume": 4.2025766,
    }


def _make_run(run_root: Path, run_id: str, payload_path: Path) -> None:
    problem_id = f"RP-{run_id}"
    outcome = {
        "experiment_result": {
            "experiment_id": f"EXP-H_CTRL-{run_id}",
            "problem_id": problem_id,
            "hypothesis_id": "H_CTRL",
            "plan_id": "PLAN-H_CTRL-B1-R1",
            "status": "completed",
            "metrics": {"MAE": 0.0471, "ACHIEVED_RISE_PCT": 14.9983, "PRESSURE_MAX_MPA": 14.4124},
            "model_records": {"hgb_control_optimizer": {
                "model_name": "hgb_control_optimizer",
                "fit_success": True,
                "fit_converged": True,
                "validation_metrics": {"MAE": 0.0471},
                "locked_test_metrics": {"MAE": 1e-5, "ACHIEVED_RISE_PCT": 14.9983},
                "artifact_paths": [str(payload_path)],
            }},
            "artifacts": [str(payload_path)],
            "execution_notes": ["HGB 软测完成"],
            "conclusion_scope": "small_model_control_validation",
            "experiment_valid": True,
            "experiment_validity_issues": [],
            "started_at": "2026-08-26T00:00:00Z",
            "completed_at": "2026-08-26T00:00:01Z",
        },
        "audit": {
            "experiment_id": f"EXP-H_CTRL-{run_id}",
            "execution_valid": True,
            "dataset_frozen": True,
            "leakage_check_passed": True,
            "baseline_valid": True,
            "metric_check_passed": True,
            "issues": [],
        },
        "scientific_result": {
            "hypothesis_id": "H_CTRL",
            "experiment_id": f"EXP-H_CTRL-{run_id}",
            "verdict": "supported",
            "rationale": "HGB 验证支持",
            "achieved_criteria": ["提升>=15%"],
            "failed_criteria": [],
        },
        "control_summary": {
            "current_volume": 3.6544675,
            "target_volume": 4.2026376,
            "predicted_volume": 4.2025766,
            "predicted_rise": 0.1499833,
            "validation_mae": 0.0471,
            "feasible_candidates": 511,
            "unity_payload_path": str(payload_path),
        },
    }
    state = ResearchRunState(
        run_id=run_id,
        question=QUESTION,
        status="COMPLETED",
        research_problem={
            "problem_id": problem_id,
            "original_question": QUESTION,
            "research_object": "锅炉燃烧与汽水系统联合控制",
            "target_variable": "steam_volumetric_flow",
            "operating_condition": "汽包压力不超过23MPa的稳态工况",
            "research_goal": "验证联合调参能否使蒸汽体积量V提升15%",
        },
        hypotheses=[{
            "hypothesis_id": "H_CTRL",
            "id": "H_CTRL",
            "title": "联合调参使软测V升15%",
            "hypothesis": "按范围调整可使V上升15%",
            "problem_id": problem_id,
        }],
        batches=[HypothesisValidationBatch(
            batch_id=f"BATCH-{run_id}-01",
            round_index=1,
            ranking_snapshot_id=f"RANK-{run_id}-000",
            members=[BatchMember(
                hypothesis_id="H_CTRL",
                plan={
                    "plan_id": "PLAN-H_CTRL-B1-R1",
                    "hypothesis_id": "H_CTRL",
                    "problem_id": problem_id,
                    "experiment_type": "constrained_control_optimization",
                },
                contract={
                    "experiment_id": f"EXP-H_CTRL-{run_id}",
                    "problem_id": problem_id,
                    "hypothesis_id": "H_CTRL",
                    "plan_id": "PLAN-H_CTRL-B1-R1",
                    "dataset_id": "boiler_181var_v1",
                    "dataset_hash": "9" * 64,
                    "input_variables": ["col5", "col14", "col17", "col2"],
                    "target_variable": "steam_volumetric_flow",
                    "train_split": "chronological[0%,70%)",
                    "validation_split": "chronological[70%,80%)",
                    "test_split": "constraint_search",
                    "baseline_models": ["current_operating_point"],
                    "candidate_models": ["hgb_control_optimizer"],
                    "metrics": ["MAE"],
                    "confirmation_criteria": ["提升>=15%"],
                    "falsification_criteria": ["提升<15%"],
                    "status": "completed",
                },
                outcome=outcome,
                status="COMPLETED",
            )],
            status="COMPLETED",
        )],
    )
    (run_root / run_id).mkdir(parents=True, exist_ok=True)
    (run_root / run_id / "run.json").write_text(
        state.model_dump_json(indent=2), encoding="utf-8"
    )


@pytest.fixture()
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_root = tmp_path / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_module.pipeline, "run_root", run_root)
    run_id = "RUN-UNITY-LOOP-TEST"
    unity_dir = run_root / run_id / "unity"
    unity_dir.mkdir(parents=True, exist_ok=True)
    payload_path = unity_dir / "unity_push.json"
    payload_path.write_text(
        json.dumps(_unity_payload(run_id), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _make_run(run_root, run_id, payload_path)
    # 清理可能残留的状态文件
    state_path = tmp_path / "control" / "unity_push_state.json"
    if state_path.is_file():
        state_path.unlink()
    return run_root, run_id


def _next_message(ws, wanted: set[str], max_reads: int = 10) -> dict:
    for _ in range(max_reads):
        message = ws.receive_json()
        if message.get("type") in wanted:
            return message
    raise AssertionError(f"未在 {max_reads} 条消息内收到 {wanted}")


def test_websocket_control_loop_sent_received_executed_returned(app_env) -> None:
    run_root, run_id = app_env
    client = TestClient(app_module.app)
    with client.websocket_connect(
        f"/api/v1/research-runs/{run_id}/events"
    ) as ws:
        instruction = _next_message(ws, {"targetResult"})
        assert instruction["flow"] == "control_instruction"
        assert instruction["run_id"] == run_id
        assert instruction["rec_coal_feed"] == 223.5012

        ws.send_json({"type": "unity_ack", "flow_id": run_id})
        status = _next_message(ws, {"unityStatus"})
        assert status["status"] == "received"

        ws.send_json({"type": "unity_executed", "flow_id": run_id})
        status = _next_message(ws, {"unityStatus"})
        assert status["status"] == "executed"

        ws.send_json({"type": "unity_result", "flow_id": run_id, "actual_volume": 4.25})
        status = _next_message(ws, {"unityStatus"})
        assert status["status"] == "returned"
        assert status["entry"]["second_verdict"] == "supported"

    # 状态文件已持久化
    state_path = run_root.parent / "control" / "unity_push_state.json"
    entry = json.loads(state_path.read_text(encoding="utf-8"))["runs"][run_id]
    assert entry["push_status"] == "returned"
    assert entry["returned_at"]
    assert (run_root / run_id / "unity" / "unity_return.json").is_file()

    # 投影器反映 returned + 第二层裁决
    from server.research_api.projector import project_run

    projected = project_run(
        json.loads((run_root / run_id / "run.json").read_text(encoding="utf-8")),
        run_root / run_id,
    )
    assert projected["control"]["unity"]["status"] == "returned"
    assert projected["control"]["conclusion"]["second_verdict"] == "supported"


def test_rest_push_and_ack_endpoints(app_env) -> None:
    run_root, run_id = app_env
    client = TestClient(app_module.app)

    pushed = client.post(f"/api/v1/research-runs/{run_id}/unity/push")
    assert pushed.status_code == 200
    assert pushed.json()["data"]["push_status"] == "sent"

    acked = client.post(f"/api/v1/research-runs/{run_id}/unity/ack")
    assert acked.status_code == 200
    assert acked.json()["data"]["push_status"] == "received"

    status = client.get(f"/api/v1/research-runs/{run_id}/unity/status")
    assert status.status_code == 200
    assert status.json()["data"]["status"]["push_status"] == "received"

    state_path = run_root.parent / "control" / "unity_push_state.json"
    entry = json.loads(state_path.read_text(encoding="utf-8"))["runs"][run_id]
    assert entry["push_status"] == "received"
    assert entry["pushed_at"]
    assert entry["received_at"]


def test_simulate_adoption_records_return(app_env) -> None:
    """Unity 面板按推荐参数执行模拟时，自动推进 executed → returned。"""
    run_root, run_id = app_env
    client = TestClient(app_module.app)
    # 先推送到 sent
    pushed = client.post(f"/api/v1/research-runs/{run_id}/unity/push")
    assert pushed.status_code == 200

    # Unity 面板以推荐参数调用 /api/simulate
    response = client.post("/api/simulate", json={
        "coal_feed": 223.5012,
        "air_flow": 853.3014,
        "water_flow": 113.1199,
        "drum_pressure": 14.4124,
        "slag_degree": 0.05,
        "source": "boiler_sim_panel",
    })
    assert response.status_code == 200
    assert response.json()["steam_output"] > 0

    state_path = run_root.parent / "control" / "unity_push_state.json"
    entry = json.loads(state_path.read_text(encoding="utf-8"))["runs"][run_id]
    assert entry["push_status"] == "returned"
    # 标定后实际 V == 预测 V，偏差约 0，第二层裁决为支持
    assert entry["second_verdict"] == "supported"
    assert abs(entry["actual_volume"] - 4.2025766) < 0.02
    assert abs(entry["deviation_pct"]) < 0.5
