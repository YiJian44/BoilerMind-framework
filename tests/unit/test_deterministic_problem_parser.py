import pytest

from boilermind.adapters.deterministic_problem_parser import (
    DeterministicProblemParser,
    DeterministicProblemParserError,
)


def test_explicit_supported_question_parses_without_llm():
    problem = DeterministicProblemParser().parse(
        "在31变量数据上按时间顺序和锁定测试比较Ridge、"
        "BayesianRidge、HGB与Persistence对蒸汽体积流量h80的MAE和RMSE。"
    )
    assert problem.target_variable == "steam_volumetric_flow"
    assert problem.required_horizon_steps == 80
    assert problem.required_models == ["bayesianridge", "ridge", "hgb"]
    assert problem.reference_models == ["persistence"]
    assert problem.metrics == ["MAE", "RMSE"]
    assert "locked_test_evaluation" in problem.required_operations


def test_ambiguous_question_fails_closed_for_qwen_fallback():
    with pytest.raises(
        DeterministicProblemParserError,
        match="explicit_supported_target_required",
    ):
        DeterministicProblemParser().parse("综合下来推荐哪套办法？")
