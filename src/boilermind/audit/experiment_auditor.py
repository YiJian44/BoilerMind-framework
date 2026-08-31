from boilermind.audit.execution_trace import (
    ExperimentExecutionTrace,
)

from boilermind.core.contracts import (
    ExperimentAudit,
    ExperimentContract,
    ExperimentResult,
)

from boilermind.core.enums import ExperimentStatus


def audit_experiment(
    contract: ExperimentContract,
    result: ExperimentResult,
    trace: ExperimentExecutionTrace,
) -> ExperimentAudit:
    issues: list[str] = []

    execution_valid = True

    if (
        result.experiment_id
        != contract.experiment_id
    ):
        issues.append(
            "experiment_id_mismatch"
        )
        execution_valid = False

    if (
        result.hypothesis_id
        != contract.hypothesis_id
    ):
        issues.append(
            "hypothesis_id_mismatch"
        )
        execution_valid = False

    if (
        trace.experiment_id
        != contract.experiment_id
    ):
        issues.append(
            "execution_trace_id_mismatch"
        )
        execution_valid = False

    if (
        result.status
        != ExperimentStatus.COMPLETED
    ):
        issues.append(
            "experiment_not_completed"
        )
        execution_valid = False

    # Audit consumes only the runner-published canonical view. Unit-specific
    # storage keys are normalized before ExperimentResult is constructed.
    missing_metrics = [
        str(metric).upper()
        for metric in contract.metrics
        if str(metric).upper() not in result.normalized_metrics
    ]

    for metric in missing_metrics:
        issues.append(
            f"missing_metric:{metric}"
        )

    metric_check_passed = (
        trace.metric_check_passed
        and not missing_metrics
    )

    if not trace.dataset_frozen:
        issues.append(
            "dataset_not_frozen"
        )

    if not trace.leakage_check_passed:
        issues.append(
            "data_leakage_check_failed"
        )

    if not trace.baseline_valid:
        issues.append(
            "baseline_invalid"
        )

    if not metric_check_passed:
        issues.append(
            "metric_check_failed"
        )

    if contract.experiment_type == "regime_stratified_evaluation":
        regime_issue_count = len(issues)
        required_regimes = {"ramp_up", "ramp_down"}
        required_metrics = {"MAE", "RMSE", "MBE", "sample_count"}
        if "regime_stratified_evaluation" not in contract.required_operations:
            issues.append("regime_operation_not_frozen_in_contract")
        if not result.regime_metrics:
            issues.append("regime_metrics_missing")
        for model_id, regimes in result.regime_metrics.items():
            missing_regimes = sorted(required_regimes - set(regimes))
            for regime in missing_regimes:
                issues.append(f"regime_missing:{model_id}:{regime}")
            for regime in required_regimes & set(regimes):
                missing = sorted(required_metrics - set(regimes[regime]))
                for metric in missing:
                    issues.append(
                        f"regime_metric_missing:{model_id}:{regime}:{metric}"
                    )
        if len(issues) > regime_issue_count:
            metric_check_passed = False
            issues.append("metric_check_failed")

    issues = list(
        dict.fromkeys(issues)
    )

    if issues:
        execution_valid = False

    return ExperimentAudit(
        experiment_id=contract.experiment_id,
        execution_valid=execution_valid,
        dataset_frozen=trace.dataset_frozen,
        leakage_check_passed=(
            trace.leakage_check_passed
        ),
        baseline_valid=trace.baseline_valid,
        metric_check_passed=(
            metric_check_passed
        ),
        issues=issues,
    )
