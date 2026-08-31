import pytest

from boilermind.core.contracts import ExperimentContract
from boilermind.experiment.unified_runner import UnifiedExperimentRunner


def base_contract(**updates):
    values = dict(experiment_id="EXP-FI-FAIL", problem_id="P", hypothesis_id="H", plan_id="PL",
        experiment_type="feature_intervention", dataset_id="D", dataset_hash="hash",
        input_variables=["feature_01", "feature_02"], target_variable="main_steam_mass_flow",
        train_split="train", validation_split="validation", test_split="locked_test",
        baseline_models=["persistence"], reference_models=["persistence"], candidate_models=["ridge"],
        control={"model": "ridge", "features": ["feature_01"]},
        treatment={"model": "ridge", "features": ["feature_01", "feature_02"]},
        metrics=["MAE"], confirmation_criteria=["c"], falsification_criteria=["f"])
    values.update(updates)
    return ExperimentContract(**values)


@pytest.mark.parametrize("updates,reason", [
    ({"treatment": {"model": "lstm", "features": ["feature_01", "feature_02"]}}, "same_model"),
    ({"treatment": {"model": "ridge", "features": ["feature_01"]}}, "features_must_differ"),
    ({"candidate_models": ["lstm"]}, "candidate_model_mismatch"),
    ({"treatment": {"model": "ridge", "features": ["feature_99"]}}, "unknown_intervention_feature"),
    ({"locked_test_used_for_selection": True}, "locked_test_used_for_selection"),
])
def test_runner_rejects_mutated_or_invalid_design(updates, reason):
    runner = object.__new__(UnifiedExperimentRunner)
    with pytest.raises(ValueError, match=reason):
        runner._validate_intervention(base_contract(**updates))
