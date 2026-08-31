"""Qwen 不可用时的确定性降级测试：任何问题都能跑完整六阶段流程。"""

from __future__ import annotations

from pathlib import Path

from boilermind.core.contracts import ResearchRequest
from boilermind.orchestration.research_orchestrator import ResearchOrchestrator


class _NoopEvolutionSink:
    def record(self, *args, **kwargs):
        return None


def _orchestrator(
    tmp_path: Path,
    *,
    problem_parser=None,
    hypothesis_generator=None,
):
    return ResearchOrchestrator(
        run_root=tmp_path / "runs",
        memory_root=tmp_path / "memory",
        evolution_sink=_NoopEvolutionSink(),
        problem_parser=problem_parser,
        hypothesis_generator=hypothesis_generator,
        evidence_retriever=lambda context: (_ for _ in ()).throw(
            RuntimeError("QwenSemanticJudgeError: connection failed")
        ),
        memory_retriever=lambda problem, *_: {
            "problem_id": str((problem or {}).get("problem_id") or "RP-MEMORY"),
            "supported_observations": [],
            "completed_experiment_ids": [],
        },
    )


def _failing_parser(context):
    raise RuntimeError("QwenProblemParserError: connection failed")


def _failing_generator(context):
    raise RuntimeError("APIConnectionError: connection failed")


def test_data_profile_question_falls_back_to_deterministic_problem(tmp_path: Path) -> None:
    question = (
        "分析最新锅炉数据，识别数据属性（非线性、时序、稀疏化、降维、非高斯），"
        "模型库里面哪个模型软测蒸汽体积量V的误差最小"
    )
    orchestrator = _orchestrator(tmp_path, problem_parser=_failing_parser)
    state = orchestrator.run(ResearchRequest(question=question, run_id="RUN-FB-PROFILE"))

    assert state.status in {"COMPLETED", "COMPLETED_WITH_REPORT_WARNING"}
    assert state.research_problem
    assert state.research_problem["target_variable"] == "steam_volumetric_flow"
    assert state.hypotheses
    assert any(
        member.status == "COMPLETED"
        for batch in state.batches
        for member in batch.members
    )
    assert any(
        provenance.source == "DETERMINISTIC"
        for provenance in state.field_provenance
    )


def test_model_comparison_question_falls_back_to_deterministic_hypotheses(
    tmp_path: Path,
) -> None:
    question = (
        "比较 Ridge、BayesianRidge、RandomForest 与 Persistence 对"
        "蒸汽体积流量未来10分钟（h40）的预测"
    )
    orchestrator = _orchestrator(
        tmp_path,
        hypothesis_generator=_failing_generator,
    )
    state = orchestrator.run(ResearchRequest(question=question, run_id="RUN-FB-H40"))

    assert state.status in {"COMPLETED", "COMPLETED_WITH_REPORT_WARNING"}
    assert state.hypotheses
    assert any(
        member.status == "COMPLETED"
        for batch in state.batches
        for member in batch.members
    )
    assert any(
        str(trace.stage) == "hypothesis_generation_and_gate"
        and trace.status == "COMPLETED"
        for trace in state.stage_traces
    )
