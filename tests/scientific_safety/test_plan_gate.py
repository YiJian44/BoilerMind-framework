from boilermind.core.contracts import (
    ExperimentPlan,
    MechanismStep,
    ScientificHypothesis,
)

from boilermind.core.enums import (
    MechanismSupportType,
)

from boilermind.planning.plan_contracts import (
    ExperimentCapabilitySnapshot,
    PlanCritiqueDecision,
)

from boilermind.planning.plan_gate import (
    approve_and_compile_plan,
)


HASH = "e" * 64


def make_hypothesis():
    return ScientificHypothesis(
        hypothesis_id="H001",
        problem_id="P001",
        title="Dynamic lag hypothesis",
        research_significance=(
            "Explain prediction error during "
            "deep peak regulation."
        ),
        hypothesis=(
            "Dynamic lag increases steam-flow "
            "soft-sensor error."
        ),
        mechanism_chain=(
            "load change -> lag -> error"
        ),
        mechanism_steps=[
            MechanismStep(
                step=1,
                statement=(
                    "Dynamic lag may increase error."
                ),
                support_type=(
                    MechanismSupportType
                    .HYPOTHESIS_INFERENCE
                ),
                evidence_ids=[],
            )
        ],
        related_variables=[
            "load_mw",
            "fuel_flow",
            "feedwater_flow",
        ],
        applicability_conditions=[
            "deep peak regulation"
        ],
        verification_intent=(
            "Compare lag-aware model against "
            "non-lag baseline."
        ),
        expected_observation=(
            "Lag-aware model improves prediction."
        ),
        confirmation_criteria=[
            "MAE improvement reaches "
            "predeclared threshold."
        ],
        falsification_criteria=[
            "MAE improvement does not reach "
            "predeclared threshold."
        ],
        novelty_axis=(
            "Dynamic lag compensation"
        ),
        evidence_bundle_sha256=HASH,
    )


def make_plan():
    return ExperimentPlan(
        plan_id="PLAN-H001",
        hypothesis_id="H001",
        objective=(
            "Test whether lag compensation "
            "reduces prediction error."
        ),
        experimental_design=(
            "Compare lag-aware candidate against "
            "persistence baseline on frozen test data."
        ),
        baseline_description=(
            "Persistence baseline."
        ),
        intervention_description=(
            "Enable lag compensation."
        ),
        required_variables=[
            "load_mw",
            "fuel_flow",
            "feedwater_flow",
        ],
        metrics=[
            "MAE",
            "R2",
        ],
        expected_observation=(
            "Candidate MAE is lower."
        ),
        confirmation_criteria=[
            "MAE improvement reaches "
            "predeclared threshold."
        ],
        falsification_criteria=[
            "MAE improvement does not reach "
            "predeclared threshold."
        ],
    )


def make_capability():
    return ExperimentCapabilitySnapshot(
        snapshot_id="CAP-001",
        dataset_id="BOILER-DATA-001",
        dataset_hash="dataset-hash-001",
        available_variables=[
            "load_mw",
            "fuel_flow",
            "feedwater_flow",
            "pressure_mpa",
            "temperature_c",
        ],
        available_target_variables=[
            "steam_volume_flow"
        ],
        available_baseline_models=[
            "persistence"
        ],
        available_candidate_models=[
            "ridge_if97",
            "transformer"
        ],
        available_metrics=[
            "MAE",
            "R2",
        ],
        train_split="train",
        validation_split="validation",
        test_split="test_frozen",
        data_frozen=True,
        leakage_policy_verified=True,
    )


def make_good_critique():
    return PlanCritiqueDecision(
        plan_id="PLAN-H001",
        hypothesis_id="H001",
        hypothesis_experiment_alignment=True,
        intervention_valid=True,
        baseline_valid=True,
        metric_alignment=True,
        confirmation_falsification_valid=True,
        executable=True,
        issues=[],
        rationale=(
            "Plan directly tests the hypothesis "
            "with available resources."
        ),
    )


def test_valid_plan_compiles_to_experiment_contract():
    contract, report = (
        approve_and_compile_plan(
            make_hypothesis(),
            make_plan(),
            make_capability(),
            make_good_critique(),
            target_variable="steam_volume_flow",
            baseline_models=[
                "persistence"
            ],
            candidate_models=[
                "ridge_if97"
            ],
        )
    )

    assert report.passed is True
    assert contract is not None

    assert (
        contract.hypothesis_id
        == "H001"
    )

    assert (
        contract.dataset_id
        == "BOILER-DATA-001"
    )


def test_missing_dcs_variable_rejects_plan():
    plan = make_plan().model_copy(
        update={
            "required_variables": [
                "load_mw",
                "flame_temperature_field",
            ]
        }
    )

    contract, report = (
        approve_and_compile_plan(
            make_hypothesis(),
            plan,
            make_capability(),
            make_good_critique(),
            target_variable="steam_volume_flow",
            baseline_models=[
                "persistence"
            ],
            candidate_models=[
                "ridge_if97"
            ],
        )
    )

    assert contract is None
    assert report.passed is False

    assert (
        "missing_required_variable:"
        "flame_temperature_field"
        in report.issues
    )


def test_leakage_failure_rejects_plan():
    capability = (
        make_capability().model_copy(
            update={
                "leakage_policy_verified": False
            }
        )
    )

    contract, report = (
        approve_and_compile_plan(
            make_hypothesis(),
            make_plan(),
            capability,
            make_good_critique(),
            target_variable="steam_volume_flow",
            baseline_models=[
                "persistence"
            ],
            candidate_models=[
                "ridge_if97"
            ],
        )
    )

    assert contract is None
    assert report.passed is False

    assert (
        "leakage_policy_not_verified"
        in report.issues
    )


def test_bad_scientific_alignment_rejects_plan():
    critique = (
        make_good_critique().model_copy(
            update={
                "hypothesis_experiment_alignment": False,
                "rationale": (
                    "Experiment does not directly "
                    "test the proposed mechanism."
                ),
            }
        )
    )

    contract, report = (
        approve_and_compile_plan(
            make_hypothesis(),
            make_plan(),
            make_capability(),
            critique,
            target_variable="steam_volume_flow",
            baseline_models=[
                "persistence"
            ],
            candidate_models=[
                "ridge_if97"
            ],
        )
    )

    assert contract is None
    assert report.passed is False

    assert (
        "hypothesis_experiment_misalignment"
        in report.issues
    )