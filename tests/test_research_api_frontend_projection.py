from __future__ import annotations

from server.research_api.projector import project_run


def test_frontend_projection_separates_validation_selection_from_locked_test(tmp_path):
    (tmp_path / "structured_report.json").write_text("{}", encoding="utf-8")
    state = {
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
        "ranking_snapshots": [{"entries": [{"hypothesis_id": "H1"}]}],
        "stage_traces": [],
        "batches": [{
            "members": [{
                "hypothesis_id": "H1", "status": "COMPLETED",
                "plan": {"primary_metric": "MAE"},
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
                                "validation_metrics": {"MAE": 0.10},
                                "locked_test_metrics": {"MAE": 0.15},
                            },
                            "rf": {
                                "fit_success": True,
                                "validation_metrics": {"MAE": 0.12},
                                "locked_test_metrics": {"MAE": 0.14},
                            },
                        },
                        "baseline_metrics": {"MAE": 0.20},
                    },
                    "scientific_result": {"verdict": "supported"},
                    "audit": {"execution_valid": True, "leakage_check_passed": True},
                },
            }],
        }],
        "report": {},
        "errors": [],
    }

    projected = project_run(state, tmp_path)

    assert projected["run"]["status"] == "completed"
    assert projected["run"]["progress_percent"] == 100
    assert all(stage["status"] == "completed" for stage in projected["stages"])
    assert projected["execution"]["protocol_selected_model"] == "bayesianridge"
    assert projected["execution"]["locked_test_best_model"] == "rf"
    assert projected["execution"]["locked_test_used_for_selection"] is False
    assert projected["hypotheses"][0]["experiment_plan"]["primary_metric"] == "MAE"

