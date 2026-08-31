from datetime import datetime, timezone

from boilermind.core.contracts.experiment import (
    ExperimentResult,
    ModelExperimentRecord,
    ScientificResult,
)
from boilermind.core.enums import (
    ExperimentStatus,
    ScientificVerdict,
)
from knowledge_graph.evolution.experiment_adapter import (
    update_evolution_from_results,
)
from knowledge_graph.evolution.kg_update import update_evolution_graph


def experiment_result(**overrides):
    result = {
        "experiment_id": "EXP001",
        "hypothesis_id": "H001",
        "hypothesis": "科研假设文本",
        "mechanism_chain": "历史窗口增加 -> 时序信息增强 -> 预测误差降低",
        "verdict": "supported",
        "metrics": {"MAE": 0.038, "R2": 0.969},
        "experiment_valid": True,
        "applicable_scope": "160-320MW调峰工况",
    }
    result.update(overrides)
    return result


def test_valid_supported_experiment_generates_validated_hypothesis(tmp_path):
    graph_path = tmp_path / "evolution_graph.json"
    result = experiment_result()

    graph = update_evolution_graph(result, graph_path)

    hypothesis = next(node for node in graph["nodes"] if node["type"] == "Hypothesis")
    experiment = next(node for node in graph["nodes"] if node["type"] == "ExperimentResult")
    knowledge = next(node for node in graph["nodes"] if node["type"] == "ValidatedHypothesis")

    assert hypothesis == {
        "id": "H001",
        "type": "Hypothesis",
        "content": "科研假设文本",
        "status": "validated",
    }
    assert experiment["id"] == "EXP001"
    assert experiment["verdict"] == "supported"
    assert experiment["metrics"] == {"MAE": 0.038, "R2": 0.969}
    assert experiment["experiment_valid"] is True
    assert knowledge == {
        "id": "VH-H001-EXP001",
        "type": "ValidatedHypothesis",
        "hypothesis_id": "H001",
        "content": "科研假设文本",
        "mechanism_chain": "历史窗口增加 -> 时序信息增强 -> 预测误差降低",
        "validation_status": "supported",
        "experiment_id": "EXP001",
        "metrics": {"MAE": 0.038, "R2": 0.969},
        "applicable_scope": "160-320MW调峰工况",
    }

    assert {"source": "H001", "type": "validated_by", "target": "EXP001"} in graph["edges"]
    assert {"source": "EXP001", "type": "generates", "target": "VH-H001-EXP001"} in graph["edges"]


def test_invalid_experiment_does_not_generate_validated_hypothesis(tmp_path):
    graph = update_evolution_graph(
        experiment_result(experiment_valid=False),
        tmp_path / "evolution_graph.json",
    )

    assert not any(node["type"] == "ValidatedHypothesis" for node in graph["nodes"])
    assert not any(edge["type"] == "generates" for edge in graph["edges"])


def test_falsified_experiment_does_not_generate_validated_hypothesis(tmp_path):
    graph = update_evolution_graph(
        experiment_result(verdict="falsified"),
        tmp_path / "evolution_graph.json",
    )

    assert not any(node["type"] == "ValidatedHypothesis" for node in graph["nodes"])
    assert not any(edge["type"] == "generates" for edge in graph["edges"])


def test_inconclusive_experiment_does_not_generate_validated_hypothesis(tmp_path):
    graph = update_evolution_graph(
        experiment_result(verdict="inconclusive"),
        tmp_path / "evolution_graph.json",
    )

    assert not any(node["type"] == "ValidatedHypothesis" for node in graph["nodes"])
    assert not any(edge["type"] == "generates" for edge in graph["edges"])


def test_repeated_result_is_idempotent(tmp_path):
    graph_path = tmp_path / "evolution_graph.json"
    result = experiment_result()

    update_evolution_graph(result, graph_path)
    graph = update_evolution_graph(result, graph_path)

    assert len(graph["nodes"]) == 3
    assert len(graph["edges"]) == 2


def test_same_hypothesis_keeps_validations_from_multiple_experiments(tmp_path):
    graph_path = tmp_path / "evolution_graph.json"
    update_evolution_graph(experiment_result(), graph_path)
    graph = update_evolution_graph(
        experiment_result(experiment_id="EXP002", metrics={"MAE": 0.034, "R2": 0.974}),
        graph_path,
    )

    validated = [node for node in graph["nodes"] if node["type"] == "ValidatedHypothesis"]
    assert {node["experiment_id"] for node in validated} == {"EXP001", "EXP002"}
    assert len(validated) == 2


def real_results(*, experiment_valid=True):
    experiment = ExperimentResult(
        experiment_id="EXP-REAL-001",
        problem_id="P001",
        hypothesis_id="H-REAL-001",
        plan_id="PLAN001",
        status=ExperimentStatus.COMPLETED,
        metrics={"MAE": 0.038, "R2": 0.969},
        experiment_valid=experiment_valid,
        experiment_validity_issues=([] if experiment_valid else ["locked_test_failed"]),
        model_records={
            "dlinear": ModelExperimentRecord(
                model_name="dlinear",
                fit_success=True,
                fit_converged=True,
                runtime_seconds=1.25,
                locked_test_metrics={"MAE": 0.038, "R2": 0.969},
            )
        },
        started_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        completed_at=datetime(2026, 8, 22, 0, 1, tzinfo=timezone.utc),
    )
    scientific = ScientificResult(
        hypothesis_id="H-REAL-001",
        experiment_id="EXP-REAL-001",
        verdict=ScientificVerdict.SUPPORTED,
        rationale="有效锁定测试支持该假设",
    )
    hypothesis = {
        "hypothesis_id": "H-REAL-001",
        "hypothesis": "增加历史窗口能够改善调峰工况预测精度",
        "mechanism_chain": "历史窗口增加 -> 捕获调峰动态 -> 预测误差降低",
        "applicability_conditions": ["160-320MW调峰工况"],
    }
    return experiment, scientific, hypothesis


def test_real_contract_results_generate_validated_hypothesis(tmp_path):
    graph = update_evolution_from_results(
        *real_results(),
        graph_path=tmp_path / "evolution_graph.json",
    )

    validated = next(
        node for node in graph["nodes"]
        if node["type"] == "ValidatedHypothesis"
    )
    experiment = next(
        node for node in graph["nodes"]
        if node["type"] == "ExperimentResult"
    )
    assert validated["experiment_id"] == "EXP-REAL-001"
    assert validated["hypothesis_id"] == "H-REAL-001"
    assert validated["metrics"] == {"MAE": 0.038, "R2": 0.969}
    assert validated["applicable_scope"] == "160-320MW调峰工况"
    assert validated["mechanism_chain"] == (
        "历史窗口增加 -> 捕获调峰动态 -> 预测误差降低"
    )
    assert experiment["status"] == "completed"
    assert experiment["model_records"]["dlinear"]["fit_success"] is True


def test_invalid_real_contract_result_does_not_generate_validated_hypothesis(tmp_path):
    graph = update_evolution_from_results(
        *real_results(experiment_valid=False),
        graph_path=tmp_path / "evolution_graph.json",
    )

    assert not any(
        node["type"] == "ValidatedHypothesis"
        for node in graph["nodes"]
    )
    assert not any(edge["type"] == "generates" for edge in graph["edges"])
