from boilermind.core.contracts import ExperimentPlan
from boilermind.planning.plan_contracts import ExperimentCapabilitySnapshot
from boilermind.planning.plan_gate import compile_plan_to_contract


def make_plan(**updates):
    values = dict(plan_id="PL-FI", problem_id="P-FI", hypothesis_id="H-FI",
        experiment_type="feature_intervention", objective="compare feature sets",
        experimental_design="frozen control treatment", baseline_description="baseline features",
        intervention_description="add pressure and temperature",
        required_variables=["feature_01", "feature_02", "feature_03"],
        target="main_steam_mass_flow", metrics=["MAE", "RMSE", "R2"],
        expected_observation="treatment improves MAE", confirmation_criteria=["delta_MAE_lt_0"],
        falsification_criteria=["delta_MAE_ge_0"], candidate_models=["ridge"],
        reference_models=["persistence"], control={"model": "ridge", "features": ["feature_01"]},
        treatment={"model": "ridge", "features": ["feature_01", "feature_02", "feature_03"]},
        current_executable=True, prediction_horizon_steps=40, sampling_interval_seconds=15,
        random_seed=17)
    values.update(updates)
    return ExperimentPlan(**values)


def snapshot():
    return ExperimentCapabilitySnapshot(snapshot_id="S", dataset_id="D", dataset_hash="hash",
        available_variables=["feature_01", "feature_02", "feature_03"],
        available_target_variables=["main_steam_mass_flow"], available_baseline_models=["persistence"],
        available_candidate_models=["ridge"], available_metrics=["MAE", "RMSE", "R2"],
        train_split="train", validation_split="validation", test_split="locked_test",
        data_frozen=True, leakage_policy_verified=True, prediction_horizon_steps=40,
        sampling_interval_seconds=15)


def test_plan_contract_freezes_control_treatment_and_ids():
    plan = make_plan()
    contract, report = compile_plan_to_contract(plan, snapshot(), target_variable=plan.target,
        baseline_models=["persistence"], candidate_models=["ridge"])
    assert report.passed
    assert (contract.problem_id, contract.hypothesis_id, contract.plan_id) == ("P-FI", "H-FI", "PL-FI")
    assert contract.control == plan.control and contract.treatment == plan.treatment
    plan.control["features"].append("feature_02")
    assert contract.control["features"] == ["feature_01"]


def test_plan_gate_rejects_invalid_intervention_design():
    plan = make_plan(treatment={"model": "lstm", "features": ["feature_01"]})
    contract, report = compile_plan_to_contract(plan, snapshot(), target_variable=plan.target,
        baseline_models=["persistence"], candidate_models=["ridge"])
    assert contract is None
    assert "intervention_requires_same_model" in report.issues
