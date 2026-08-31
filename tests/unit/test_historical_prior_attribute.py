"""历史先验的"数据属性先验"维度测试（增量：不影响无属性分假设）。"""
from __future__ import annotations

from boilermind.orchestration import ResearchOrchestrator  # noqa: F401  # 先加载编排，绕开 skills↔orchestration 循环导入
from boilermind.ranking.historical_prior import score_hypothesis


def _weak_hypothesis() -> dict:
    """无历史观测的弱假设（base prior 低，能看出属性分加成）。"""
    return {
        "hypothesis_id": "H_LSTM",
        "hypothesis": "lstm 软测 V 误差最小",
        "historical_assessment": {"reproducibility": 1.0},
        "source_observation_ids": ["CUR-001"],
        "verification_mapping": {"executable_now": True},
        "confirmation_criteria": ["c"],
        "falsification_criteria": ["f"],
    }


def test_attribute_prior_boosts_only_when_provided():
    base = score_hypothesis(_weak_hypothesis())
    boosted = score_hypothesis({**_weak_hypothesis(), "data_attribute_prior": 1.0})
    assert boosted.prior_score > base.prior_score
    assert abs((boosted.prior_score - base.prior_score) - 0.20) < 1e-6


def test_attribute_prior_scales_with_value():
    low = score_hypothesis({**_weak_hypothesis(), "data_attribute_prior": 0.5})
    high = score_hypothesis({**_weak_hypothesis(), "data_attribute_prior": 1.0})
    assert high.prior_score > low.prior_score


def test_hypothesis_without_attribute_prior_unaffected():
    # 未提供 data_attribute_prior 的假设，评分与不传该参数一致
    h = _weak_hypothesis()
    s_default = score_hypothesis(dict(h))
    s_from_dict = score_hypothesis(dict(h), data_attribute_prior=None)
    assert s_default.prior_score == s_from_dict.prior_score
    # 且等于"属性分 0"（无加成）
    assert score_hypothesis(dict(h), data_attribute_prior=0.0).prior_score == s_default.prior_score
