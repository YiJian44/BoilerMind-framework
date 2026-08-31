import py_compile
from pathlib import Path

import pytest

from boilermind.experiment.capability_registry import (
    ExperimentCapabilityRegistry,
)

from boilermind.core.contracts import ExperimentContract

from boilermind.experiment.real_runner_adapter import (
    RealSklearnExperimentRunner,
)

from boilermind.models import (
    ExecutionEnvironment,
    build_default_registry,
    model_status_matrix,
)

from boilermind.skills.contract_skill import (
    ExperimentContractSkill,
)

from boilermind.skills.planning_skill import (
    PlanningSkill,
)


MODELS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "boilermind"
    / "models"
)


def make_ridge_hypothesis():
    return {
        "id": "H020",
        "hypothesis_id": "H020",
        "hypothesis": (
            "ridge模型对10分钟后主蒸汽流量的预测MAE与"
            "RMSE高于persistence模型。"
        ),
        "verification_intent": (
            "执行reference_model_comparison操作，在锁定"
            "测试集上计算ridge与persistence模型的MAE与"
            "RMSE。"
        ),
        "falsification_condition": (
            "ridge在MAE与RMSE上同时优于persistence模型。"
        ),
    }


def test_plan_contract_runner_executes_registry_model(tmp_path):
    """
    ExperimentPlan -> Contract -> Runner -> ACTUAL model code
    (real sklearn Ridge trained on the project dataset).
    """

    planning = PlanningSkill()

    planning_result = planning.execute(
        {
            "problem_id": "RP-CALL-TEST",
            "selected_hypothesis_id": "H020",
            "qualified_hypotheses": [
                make_ridge_hypothesis(),
            ],
        }
    )

    assert planning_result["current_executable"] is True

    plan = planning_result["experiment_plan"]

    assert plan["candidate_models"] == ["ridge"]

    contract_result = ExperimentContractSkill().execute(
        {
            "experiment_plan": plan,
        }
    )

    assert contract_result["contract_compiled"] is True

    contract = contract_result["experiment_contract"]

    assert contract["candidate_models"] == ["ridge"]

    runner = RealSklearnExperimentRunner(output_dir=tmp_path)

    result, trace = runner.run(
        ExperimentContract.model_validate(
            contract
        )
    )

    assert set(
        result.candidate_locked_test_metrics
    ) == {
        "ridge",
        "persistence",
    }

    assert trace.dataset_frozen is True
    assert trace.leakage_check_passed is True


def test_migrated_vendor_source_is_project_relative():
    registry = build_default_registry()

    for model_name in [
        "transformer",
        "lstm",
        "dlinear",
        "gru",
        "patchtst",
        "itransformer",
        "timesnet",
        "mtgnn",
        "csdi",
        "tcn",
        "legacy_tcn_1min",
        "legacy_transformer_10min",
    ]:
        spec = registry.get(model_name)

        assert spec.source_code_path is not None

        source_file = MODELS_DIR / (
            spec.source_code_path
        )

        assert source_file.is_file(), (
            f"{model_name} source missing: "
            f"{source_file}"
        )


def test_legacy_adapter_uses_trusted_vendor_loader():
    registry = build_default_registry()

    adapter = registry.build_adapter("transformer")

    assert adapter.spec.architecture_factory == "transformer"

    with pytest.raises(
        RuntimeError,
        match="model_not_fitted",
    ):
        adapter.predict(
            [[0.0] * 30] * 30,
        )


def test_vendor_sources_are_syntactically_valid():
    vendor_dir = MODELS_DIR / "vendor"

    py_files = sorted(
        vendor_dir.rglob("*.py")
    )

    assert len(py_files) >= 30

    for path in py_files:
        py_compile.compile(
            str(path),
            doraise=True,
        )


def test_status_matrix_reports_all_models():
    capability = ExperimentCapabilityRegistry(
        environment=ExecutionEnvironment(
            os="test", python_version="3.11.9",
            sklearn_available=True, torch_available=False,
            cuda_available=False, gpu_available=False,
        )
    )

    rows = model_status_matrix(
        capability=capability,
    )

    assert len(rows) == 30

    by_name = {
        row["model_name"]: row
        for row in rows
    }

    assert by_name["persistence"][
        "RUNNER_CALLABLE"
    ] is True

    for model_name in [
        "ridge",
        "bayesianridge",
        "hgb",
        "svr",
        "rf",
        "mlp",
        "elasticnet",
        "pls",
        "knn",
    ]:
        assert by_name[model_name][
            "RUNNER_CALLABLE"
        ] is True
        assert by_name[model_name][
            "EXECUTABLE"
        ] is True

    assert by_name["transformer"][
        "RUNNER_CALLABLE"
    ] is False
    assert by_name["transformer"][
        "source_code_exists"
    ] is True

    assert by_name["gpr"]["EXECUTABLE"] is True

    assert by_name["psfa_v0"][
        "source_code_path"
    ] is None
