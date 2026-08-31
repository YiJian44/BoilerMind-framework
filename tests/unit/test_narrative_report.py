from boilermind.reporting.narrative_report import (
    bind_frozen_identifiers,
    build_narrative_prompt_record,
    validate_narrative_report,
    write_narrative_report,
)


def test_missing_frozen_identifiers_are_bound_before_validation():
    record = {
        "run_id": "RUN-1",
        "batches": [{
            "members": [{
                "hypothesis_id": "H001",
                "contract": {"experiment_id": "EXP-1"},
            }],
        }],
    }
    narrative = bind_frozen_identifiers(record, "这是千问生成的可读报告正文。")
    assert "RUN-1" in narrative
    assert "H001" in narrative
    assert "EXP-1" in narrative
    validate_narrative_report(record, narrative)


def test_already_bound_narrative_is_not_modified():
    record = {"run_id": "RUN-1", "batches": []}
    narrative = "RUN-1 已完成。"
    assert bind_frozen_identifiers(record, narrative) == narrative


def test_narrative_prompt_compacts_runtime_record_but_keeps_audit_facts():
    record = {
        "run_id": "RUN-1",
        "question": "比较模型",
        "research_problem": {"required_models": ["ridge", "rf"]},
        "hypotheses": [{
            "hypothesis_id": "H001",
            "hypothesis_statement": "比较指定模型。",
            "raw_hypothesis": {"large": "x" * 100_000},
        }],
        "batches": [{
            "batch_id": "B-1",
            "status": "COMPLETED",
            "members": [{
                "hypothesis_id": "H001",
                "status": "COMPLETED",
                "contract": {
                    "experiment_id": "EXP-1",
                    "candidate_models": ["ridge", "rf"],
                    "reference_models": ["persistence"],
                },
                "outcome": {
                    "closure_ok": True,
                    "audit": {"execution_valid": True},
                    "scientific_result": {"verdict": "supported"},
                    "experiment_result": {
                        "baseline_metrics": {"MAE": 2.0},
                        "model_records": {
                            "ridge": {
                                "fit_success": True,
                                "runtime_seconds": 1.0,
                                "validation_metrics": {
                                    "MAE": 1.0, "mae_t_h": 1.0,
                                },
                                "locked_test_metrics": {
                                    "mae_t_h": 1.2, "rmse_t_h": 1.5,
                                },
                                "artifact_paths": ["x" * 100_000],
                            },
                        },
                    },
                },
            }],
        }],
    }
    compact = build_narrative_prompt_record(record)
    encoded = __import__("json").dumps(compact, ensure_ascii=False)
    assert len(encoded) < 10_000
    assert compact["batches"][0]["members"][0]["closure_ok"] is True
    assert compact["batches"][0]["members"][0]["model_metrics"]["ridge"][
        "locked_test_metrics"
    ]["MAE"] == 1.2
    assert compact["batches"][0]["members"][0]["model_metrics"]["ridge"][
        "locked_test_metrics"
    ] == {"MAE": 1.2, "RMSE": 1.5}

    prompts = []
    write_narrative_report(record, generate=lambda prompt: prompts.append(prompt) or "报告")
    assert "x" * 100_000 not in prompts[0]
    assert "mae_t_h" not in prompts[0]
