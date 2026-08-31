from __future__ import annotations

from typing import Any

from boilermind.audit.candidate_criteria import (
    assess_candidate_locked_test_criteria,
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
from boilermind.core.enums import ScientificVerdict

from boilermind.experiment.unified_runner import (
    UnifiedExperimentRunner,
)



def execute_real_experiment(
    contract: ExperimentContract | dict[str, Any],
    *,
    runner: UnifiedExperimentRunner | None = None,
) -> dict[str, Any]:
    """
    P0-5 real experiment closure:

    ExperimentContract
      -> UnifiedExperimentRunner (target-profile routing)
      -> ExperimentResult
      -> audit_experiment
      -> candidate criterion assessment (contract criteria)
      -> derive_scientific_result
      -> Verdict

    Audit failure is fail-closed: derive_scientific_result
    returns INSUFFICIENT_EVIDENCE.
    """

    if isinstance(contract, dict):
        contract = ExperimentContract.model_validate(
            contract
        )

    if not isinstance(contract, ExperimentContract):
        raise TypeError(
            "experiment_contract_required"
        )

    runner = runner or UnifiedExperimentRunner()

    result, trace = runner.run(contract)

    audit = audit_experiment(
        contract,
        result,
        trace,
    )

    assessment = assess_candidate_locked_test_criteria(
        contract,
        result,
    )

    scientific_result = derive_scientific_result(
        hypothesis_id=contract.hypothesis_id,
        experiment_id=contract.experiment_id,
        audit=audit,
        assessment=assessment,
    )

    # A staged regime experiment only tests the observable premise shared by
    # several mechanisms.  Never turn that narrower result into support for
    # the selected mechanism hypothesis.
    observable_premise_result = None
    if result.conclusion_scope == "problem_observable_premise_only":
        observable_premise_result = scientific_result
        scientific_result = scientific_result.model_copy(update={
            "verdict": ScientificVerdict.INSUFFICIENT_EVIDENCE,
            "rationale": (
                "工况实验仅验证问题中的可观察前提；尚未操纵或测量该候选"
                "机理所需变量，因此不得判定完整机理假设成立或被证伪。"
            ),
        })

    return {
        "experiment_contract": contract,
        "experiment_result": result,
        "execution_trace": trace,
        "audit": audit,
        "criterion_assessment": assessment,
        "scientific_result": scientific_result,
        "observable_premise_result": observable_premise_result,
        "closure_ok": audit.execution_valid,
        "status": "completed",
    }
