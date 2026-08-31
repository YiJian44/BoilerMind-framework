"""
Model-agnostic P0-5 chain:

    hypothesis without explicit models
      -> Planner selects from executable pool
      -> Contract preserves the selection
      -> Runner executes exactly the planned candidates
"""

from boilermind.core.enums import ScientificVerdict

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)

from boilermind.experiment.real_runner_adapter import (
    RealSklearnExperimentRunner,
)

from boilermind.orchestration.real_experiment_loop import (
    execute_real_experiment,
)
from boilermind.models.execution_environment import ExecutionEnvironment

from boilermind.skills.contract_skill import (
    ExperimentContractSkill,
)

from boilermind.skills.planning_skill import (
    PlanningSkill,
)


def make_model_agnostic_hypothesis():
    """
    Soft-sensing hypothesis that does NOT name any
    specific model. Persistence is mentioned only as
    the reference baseline.
    """

    return {
        "id": "H010",
        "hypothesis_id": "H010",
        "hypothesis": (
            "当前可执行模型对10分钟后主蒸汽流量的预测"
            "MAE与RMSE高于persistence模型。"
        ),
        "verification_intent": (
            "执行reference_model_comparison操作，在锁定"
            "测试集上计算当前可执行模型与persistence模型"
            "的MAE与RMSE。"
        ),
        "falsification_condition": (
            "至少一个当前可执行模型在MAE与RMSE上同时"
            "优于persistence模型。"
        ),
    }


def metric_set(mae, rmse):
    return {
        "mae_t_h": mae,
        "rmse_t_h": rmse,
        "r2": 0.5,
        "mbe_t_h": 0.0,
    }


def make_payload(model_ids):
    models = {
        model_id: {
            "fit_success": True,
            "fit_converged": True,
            "warnings": [],
            "failure_reason": None,
            "model_config": {},
            "validation_metrics": {},
            "locked_test_metrics": metric_set(
                0.30,
                0.40,
            ),
            "train_samples": 100,
            "validation_samples": 20,
            "test_samples": 20,
            "random_seed": 42,
            "dataset_sha256": "a" * 64,
            "model_artifact": (
                f"{model_id}.joblib"
            ),
            "prediction_artifact": (
                f"{model_id}_pred.csv"
            ),
            "artifact_paths": [
                f"{model_id}.joblib",
                f"{model_id}_pred.csv",
            ],
        }
        for model_id in model_ids
    }

    return {
        "experiment_id": "EXP-H010",
        "status": "completed",
        "dataset": {
            "path": "dataset.csv",
            "sha256": "a" * 64,
        },
        "reference_model": {
            "model_id": "persistence",
            "locked_test_metrics": metric_set(
                0.25,
                0.35,
            ),
        },
        "selected_model_by_validation": model_ids[0],
        "models": models,
        "split": {
            "locked_test_used_for_selection": False,
        },
        "result_artifact": (
            "experiment_result.json"
        ),
        "completed_at": (
            "2026-08-19T12:00:00+08:00"
        ),
    }


class RecordingStubBackend:
    """
    TEST-ONLY backend stub that records the exact
    model_candidates the runner sent to it.
    """

    __test__ = False

    def __init__(self, payload):
        self.payload = payload
        self.received_model_candidates = None

    def run(self, backend_contract):
        self.received_model_candidates = list(
            backend_contract["model_candidates"]
        )

        payload = dict(self.payload)
        payload["experiment_id"] = (
            backend_contract["experiment_id"]
        )

        return payload


def test_model_agnostic_planning_to_contract_to_runner(
    tmp_path,
):
    capability = ExperimentCapabilityRegistry(
        environment=ExecutionEnvironment(
            os="test", python_version="3.11.9",
            sklearn_available=True, torch_available=False,
            cuda_available=False, gpu_available=False,
        )
    )
    skill = PlanningSkill(capability_registry=capability)

    planning_result = skill.execute(
        {
            "problem_id": "RP-AGNOSTIC",
            "selected_hypothesis_id": "H010",
            "qualified_hypotheses": [
                make_model_agnostic_hypothesis(),
            ],
        }
    )

    assert planning_result["current_executable"] is True

    plan = planning_result["experiment_plan"]

    executable_pool = set(
        planning_result["executable_model_pool"]
    )

    assert executable_pool == {
        "bayesianridge",
        "elasticnet",
            "hgb",
            "gpr",
        "knn",
        "mlp",
        "persistence",
        "pls",
        "rf",
        "ridge",
        "svr",
    }

    assert (
        planning_result["selected_models"]
        == [
            "bayesianridge",
            "knn",
            "persistence",
            "pls",
        ]
    )

    # Planner's autonomous selection: up to 3 representative
    # executable models (cheapest first, one per family).
    expected_candidates = [
        "bayesianridge",
        "knn",
        "pls",
    ]

    assert plan["candidate_models"] == (
        expected_candidates
    )
    assert len(plan["candidate_models"]) == 3
    assert plan["reference_models"] == [
        "persistence",
    ]
    assert "ModelRegistry" in (
        plan["model_selection_rationale"]
    )
    assert "svr" not in plan["candidate_models"]

    # Contract preserves the planner's selection.
    contract_result = ExperimentContractSkill().execute(
        {
            "experiment_plan": plan,
        }
    )

    assert contract_result["contract_compiled"] is True

    contract = contract_result["experiment_contract"]

    assert contract["candidate_models"] == (
        plan["candidate_models"]
    )
    assert contract["reference_models"] == (
        plan["reference_models"]
    )

    # Runner executes exactly the planned candidates.
    registry = capability
    backend = RecordingStubBackend(
        make_payload(plan["candidate_models"])
    )

    runner = RealSklearnExperimentRunner(
        registry=registry,
        backend=backend,
        output_dir=tmp_path,
    )

    outcome = execute_real_experiment(
        contract,
        runner=runner,
    )

    assert backend.received_model_candidates == (
        plan["candidate_models"]
    )

    result = outcome["experiment_result"]

    assert set(
        result.candidate_locked_test_metrics
    ) == (
        set(plan["candidate_models"])
        | {"persistence"}
    )

    assert result.problem_id == "RP-AGNOSTIC"
    assert result.hypothesis_id == "H010"

    assert outcome["audit"].execution_valid is True
    assert outcome["scientific_result"].verdict in {
        ScientificVerdict.SUPPORTED,
        ScientificVerdict.FALSIFIED,
        ScientificVerdict.INSUFFICIENT_EVIDENCE,
    }
