from pathlib import Path

from boilermind.core.contracts import ResearchRequest
from boilermind.orchestration.research_orchestrator import ResearchOrchestrator


class _NoopEvolutionSink:
    def record(self, *args, **kwargs):
        return None


QUESTION = (
    "在汽包压力不超过23MPa的限制下，怎么调节给煤、给水、送风和汽包压力，"
    "使蒸汽体积量V上升15%，并将验证结果推送到Unity？"
)


def test_control_question_completes_canonical_pipeline(tmp_path: Path) -> None:
    orchestrator = ResearchOrchestrator(
        run_root=tmp_path / "runs",
        memory_root=tmp_path / "memory",
        evolution_sink=_NoopEvolutionSink(),
        problem_parser=lambda context: {
            "research_problem": {
                "problem_id": "RP-CTRL-TEST",
                "original_question": context["research_question"],
                "research_object": "锅炉燃烧与汽水系统联合控制",
                "target_variable": "steam_volumetric_flow",
                "objective": "在压力约束下提升蒸汽体积量并生成Unity控制指令",
                "operating_condition": "汽包压力不超过23MPa的稳态工况",
                "manipulated_variables": ["给煤", "给水", "送风", "汽包压力"],
                "research_goal": "验证联合调参能否使蒸汽体积量提升15%",
                "success_criteria": ["预测提升不少于15%", "汽包压力不超过23MPa"],
                "required_horizon_steps": 1,
                "required_operations": [],
                "constraints": ["drum_pressure <= 23 MPa"],
                "research_task_type": "parameter_optimization",
            },
            "problem_parser_type": "deterministic_control",
        },
        evidence_retriever=lambda context: {"evidence_bundle": None},
        memory_retriever=lambda *_: {"supported_observations": [], "completed_experiment_ids": []},
    )
    state = orchestrator.run(ResearchRequest(question=QUESTION, run_id="RUN-CTRL-TEST"))

    assert state.status in {"COMPLETED", "COMPLETED_WITH_REPORT_WARNING"}
    assert state.hypotheses[0]["hypothesis_id"] == "H_CTRL"
    assert state.batches[0].members[0].status == "COMPLETED"
    outcome = state.batches[0].members[0].outcome
    assert outcome["scientific_result"]["verdict"] == "supported"
    assert Path(outcome["control_summary"]["unity_payload_path"]).exists()
    assert state.report["scientific_research_plan"]["word_path"]
