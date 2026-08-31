from boilermind.audit.criterion_assessment import (
    CriterionAssessment,
)

from boilermind.audit.experiment_auditor import (
    audit_experiment,
)

from boilermind.audit.verdict_engine import (
    derive_scientific_result,
)

from boilermind.core.contracts import (
    ExperimentContract,
)

from boilermind.core.enums import (
    ScientificVerdict,
)

from boilermind.experiment.test_runner import (
    TestOnlyExperimentRunner,
    TestOnlyOutcome,
)


def make_contract():
    return ExperimentContract(
        experiment_id="EXP-001",
        hypothesis_id="H001",
        plan_id="PLAN-001",
        dataset_id="DATA-001",
        dataset_hash="HASH-001",
        input_variables=[
            "load_mw",
            "fuel_flow",
        ],
        target_variable="steam_volume_flow",
        train_split="train",
        validation_split="validation",
        test_split="test_frozen",
        baseline_models=[
            "persistence"
        ],
        candidate_models=[
            "ridge_if97"
        ],
        metrics=[
            "MAE",
            "R2",
        ],
        confirmation_criteria=[
            "Candidate reaches target."
        ],
        falsification_criteria=[
            "Candidate fails target."
        ],
        random_seed=42,
    )


def make_runner(
    *,
    leakage=True,
):
    return TestOnlyExperimentRunner(
        {
            "H001": TestOnlyOutcome(
                hypothesis_id="H001",
                metrics={
                    "MAE": 0.020,
                    "R2": 0.98,
                },
                baseline_metrics={
                    "MAE": 0.030,
                    "R2": 0.97,
                },
                dataset_frozen=True,
                leakage_check_passed=leakage,
                baseline_valid=True,
                metric_check_passed=True,
            )
        }
    )


def test_test_only_runner_produces_result():
    contract = make_contract()

    result, trace = (
        make_runner().run(contract)
    )

    assert (
        result.hypothesis_id
        == "H001"
    )

    assert (
        "TEST_ONLY_EXECUTION"
        in result.execution_notes
    )

    assert trace.dataset_frozen is True


def test_supported_result_requires_valid_audit():
    contract = make_contract()

    result, trace = (
        make_runner().run(contract)
    )

    audit = audit_experiment(
        contract,
        result,
        trace,
    )

    assessment = CriterionAssessment(
        experiment_id="EXP-001",
        confirmation_met=True,
        falsification_met=False,
        achieved_criteria=[
            "Candidate reaches target."
        ],
        failed_criteria=[],
        rationale=(
            "Predeclared target was achieved."
        ),
    )

    scientific_result = (
        derive_scientific_result(
            hypothesis_id="H001",
            experiment_id="EXP-001",
            audit=audit,
            assessment=assessment,
        )
    )

    assert (
        scientific_result.verdict
        == ScientificVerdict.SUPPORTED
    )


def test_falsified_result_is_possible():
    contract = make_contract()

    result, trace = (
        make_runner().run(contract)
    )

    audit = audit_experiment(
        contract,
        result,
        trace,
    )

    assessment = CriterionAssessment(
        experiment_id="EXP-001",
        confirmation_met=False,
        falsification_met=True,
        achieved_criteria=[],
        failed_criteria=[
            "Candidate fails target."
        ],
        rationale=(
            "Predeclared target was not achieved."
        ),
    )

    scientific_result = (
        derive_scientific_result(
            hypothesis_id="H001",
            experiment_id="EXP-001",
            audit=audit,
            assessment=assessment,
        )
    )

    assert (
        scientific_result.verdict
        == ScientificVerdict.FALSIFIED
    )


def test_failed_audit_cannot_produce_supported_result():
    contract = make_contract()

    result, trace = (
        make_runner(
            leakage=False
        ).run(contract)
    )

    audit = audit_experiment(
        contract,
        result,
        trace,
    )

    assessment = CriterionAssessment(
        experiment_id="EXP-001",
        confirmation_met=True,
        falsification_met=False,
        achieved_criteria=[
            "Candidate reaches target."
        ],
        failed_criteria=[],
        rationale=(
            "Metric target appears achieved."
        ),
    )

    scientific_result = (
        derive_scientific_result(
            hypothesis_id="H001",
            experiment_id="EXP-001",
            audit=audit,
            assessment=assessment,
        )
    )

    assert (
        scientific_result.verdict
        == ScientificVerdict
        .INSUFFICIENT_EVIDENCE
    )