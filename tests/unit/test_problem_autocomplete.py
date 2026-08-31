from __future__ import annotations

from boilermind.experiment.capability_registry import (
    DirectVolume31VCapabilityRegistry,
)
from boilermind.models.execution_environment import ExecutionEnvironment
from boilermind.skills.problem_skill import ProblemParsingSkill


def _skill() -> ProblemParsingSkill:
    environment = ExecutionEnvironment(
        os="test",
        python_version="3.11",
        sklearn_available=True,
        torch_available=False,
        cuda_available=False,
        gpu_available=False,
    )
    capability = DirectVolume31VCapabilityRegistry(environment=environment)
    return ProblemParsingSkill(capability_registry=capability)


def test_missing_metrics_are_completed_without_replacing_qwen_hypothesis_generation() -> None:
    result = _skill().execute({
        "research_question": (
            "在时间顺序划分下，比较 Ridge、BayesianRidge、"
            "RandomForest 与 Persistence 对蒸汽体积流量未来"
            "10分钟（h40）的预测"
        )
    })

    problem = result["research_problem"]
    assert result["problem_parser_type"] == (
        "deterministic_supported_question_v2_autocomplete"
    )
    assert problem["metrics"] == ["MAE", "RMSE", "R2", "MBE"]
    assert result["field_sources"]["metrics"] == "CAPABILITY_REGISTRY"
    assert result["field_sources"]["protocol_constraints"] == "SYSTEM_DEFAULT"
    assert "metrics_from_capability_registry" in result[
        "automatic_completions"
    ]
    assert "deterministic_hypotheses" not in result


def test_models_reference_horizon_and_protocol_can_be_completed() -> None:
    result = _skill().execute({
        "research_question": "比较蒸汽体积流量预测效果和误差"
    })

    problem = result["research_problem"]
    assert problem["required_models"]
    assert problem["reference_models"] == ["persistence"]
    assert problem["required_horizon_steps"] == 40
    assert set(problem["required_operations"]) == {
        "model_comparison",
        "reference_model_comparison",
        "chronological_validation",
        "locked_test_evaluation",
    }
    assert set(problem["protocol_constraints"]) == {
        "validation_only_model_selection",
        "locked_test_not_used_for_selection",
    }


def test_user_declared_fields_are_not_replaced() -> None:
    result = _skill().execute({
        "research_question": (
            "比较 Ridge 与 Persistence 对蒸汽体积流量h80预测的RMSE"
        )
    })

    problem = result["research_problem"]
    assert problem["required_models"] == ["ridge"]
    assert problem["reference_models"] == ["persistence"]
    assert problem["required_horizon_steps"] == 80
    assert problem["metrics"] == ["RMSE"]
    assert result["automatic_completions"] == [
        "locked_test_protocol_from_system_default",
        "chronological_validation_from_system_default",
    ]


def test_causal_question_does_not_use_safe_model_defaults(monkeypatch) -> None:
    called = {"qwen": False}

    class FakeQwenParser:
        def parse(self, question):
            called["qwen"] = True
            raise RuntimeError("qwen_called_for_semantic_ambiguity")

        def close(self):
            return None

    monkeypatch.setattr(
        "boilermind.skills.problem_skill.QwenProblemParser",
        FakeQwenParser,
    )
    try:
        _skill().execute({
            "research_question": "为什么煤量变化会导致蒸汽体积流量变化？"
        })
    except RuntimeError as exc:
        assert str(exc) == "qwen_called_for_semantic_ambiguity"
    else:
        raise AssertionError("semantic ambiguity must not be silently defaulted")
    assert called["qwen"] is True
