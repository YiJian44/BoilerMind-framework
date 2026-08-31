from __future__ import annotations

from typing import Any

from boilermind.core.contracts import ExperimentPlan

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)

from boilermind.planning.plan_gate import (
    compile_plan_to_contract,
)

from .base import BaseSkill


class ExperimentContractSkill(BaseSkill):
    """
    Compile a validated ExperimentPlan into a real
    ExperimentContract through plan_gate.

    The contract is NEVER reconstructed from scratch:
    plan_gate is the only path, so the ID chain and
    H002 experiment semantics cannot be lost.

    A non-executable plan fails closed here even if
    contract_skill is called directly.
    """

    name = "experiment_contract"
    description = "将实验规划通过 plan_gate 编译为机器可执行实验合同"

    def __init__(
        self,
        *,
        capability_registry: (
            ExperimentCapabilityRegistry | None
        ) = None,
    ):
        self.capability = (
            capability_registry
            or ExperimentCapabilityRegistry()
        )

    def execute(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:

        plan_data = context.get(
            "experiment_plan"
        )

        if not isinstance(plan_data, dict):
            raise ValueError(
                "experiment_plan_required"
            )

        plan = ExperimentPlan.model_validate(
            plan_data
        )

        snapshot = self.capability.to_snapshot()

        baseline_models = list(
            plan.reference_models
        ) or (
            [plan.reference_model]
            if plan.reference_model
            else []
        )

        candidate_models = list(
            plan.candidate_models
        ) or list(
            plan.model_candidates
        )

        target_variable = str(plan.target or "").strip()
        if target_variable.casefold() in {
            "", "unspecified", "unknown", "none", "null",
        }:
            return {
                "experiment_contract": None,
                "contract_report": None,
                "contract_compiled": False,
                "issues": ["target_variable_resolution_failed"],
                "status": "contract_rejected",
            }

        contract, report = (
            compile_plan_to_contract(
                plan,
                snapshot,
                target_variable=target_variable,
                baseline_models=baseline_models,
                candidate_models=candidate_models,
            )
        )

        if contract is None:
            return {
                "experiment_contract": None,
                "contract_report": report.model_dump(),
                "contract_compiled": False,
                "issues": list(report.issues),
                "status": "contract_rejected",
            }

        return {
            "experiment_contract": (
                contract.model_dump()
            ),
            "contract_report": report.model_dump(),
            "contract_compiled": True,
            "issues": [],
            "status": "contract_ready",
        }
