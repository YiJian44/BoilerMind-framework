from boilermind.reporting import FinalResearchPlanSelector


def _member(hypothesis_id: str, experiment_id: str, verdict: str, *, valid=True):
    return {
        "hypothesis_id": hypothesis_id,
        "status": "COMPLETED",
        "plan": {"plan_id": f"PLAN-{experiment_id}"},
        "contract": {"plan_id": f"PLAN-{experiment_id}", "experiment_id": experiment_id},
        "outcome": {
            "experiment_result": {"status": "completed", "metrics": {"MAE": 1.0}},
            "audit": {"execution_valid": valid},
            "scientific_result": {"verdict": verdict, "rationale": "recorded"},
        },
    }


def test_selects_first_round_when_no_iteration_exists():
    state = {
        "hypotheses": [{"hypothesis_id": "H1"}],
        "ranking_snapshots": [],
        "batches": [{"round_index": 1, "members": [_member("H1", "E1", "SUPPORTED")]}],
    }
    result = FinalResearchPlanSelector().select(state)
    assert result.selection.round_index == 1
    assert result.selection.fallback_applied is False
    assert result.selection.selection_reason == "FIRST_ROUND_NO_ITERATION"


def test_selects_latest_valid_revision():
    state = {
        "hypotheses": [{"hypothesis_id": "H1"}],
        "ranking_snapshots": [],
        "batches": [
            {"round_index": 1, "members": [_member("H1", "E1", "INSUFFICIENT_EVIDENCE")]},
            {"round_index": 2, "members": [_member("H1", "E2", "SUPPORTED")]},
        ],
    }
    result = FinalResearchPlanSelector().select(state)
    assert result.selection.round_index == 2
    assert result.selection.revision_index == 1
    assert result.selection.selection_reason == "LATEST_VALID_REVISION"


def test_invalid_second_round_falls_back_to_first():
    state = {
        "hypotheses": [{"hypothesis_id": "H1"}],
        "ranking_snapshots": [],
        "batches": [
            {"round_index": 1, "members": [_member("H1", "E1", "INSUFFICIENT_EVIDENCE")]},
            {"round_index": 2, "members": [_member("H1", "E2", "SUPPORTED", valid=False)]},
        ],
    }
    result = FinalResearchPlanSelector().select(state)
    assert result.selection.round_index == 1
    assert result.selection.fallback_applied is True
    assert result.selection.selection_reason == "FALLBACK_TO_LAST_VALID_ROUND"
