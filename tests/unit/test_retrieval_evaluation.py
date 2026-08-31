import pytest

from boilermind.evidence.retrieval_evaluation import (
    evaluate_ranked_results,
    summarize_query_evaluations,
)


def test_metrics_remain_closed_until_every_expected_result_is_judged():
    result = evaluate_ranked_results([
        {"relevance_judgment": "RELEVANT"},
        {"relevance_judgment": None},
    ], expected_count=2)
    assert result["complete"] is False
    assert result["strict_precision_at_k"] is None
    assert result["broad_precision_at_k"] is None
    assert result["graded_ndcg_at_k"] is None


def test_complete_graded_labels_produce_deterministic_metrics():
    result = evaluate_ranked_results([
        {"relevance_judgment": "RELEVANT"},
        {"relevance_judgment": "IRRELEVANT"},
        {"relevance_judgment": "PARTIAL"},
    ], expected_count=3)
    assert result["complete"] is True
    assert result["strict_precision_at_k"] == pytest.approx(1 / 3)
    assert result["broad_precision_at_k"] == pytest.approx(2 / 3)
    assert 0 < result["graded_ndcg_at_k"] <= 1


def test_unjudgeable_does_not_silently_count_as_irrelevant():
    result = evaluate_ranked_results([
        {"relevance_judgment": "RELEVANT"},
        {"relevance_judgment": "UNJUDGEABLE"},
    ], expected_count=2)
    assert result["complete"] is False


def test_overall_metrics_are_macro_averages_only_after_all_queries_complete():
    queries = [
        {"evaluation": evaluate_ranked_results([
            {"relevance_judgment": "RELEVANT"},
            {"relevance_judgment": "IRRELEVANT"},
        ], expected_count=2), "results": [
            {"relevance_judgment": "RELEVANT"},
            {"relevance_judgment": "IRRELEVANT"},
        ]},
        {"evaluation": evaluate_ranked_results([
            {"relevance_judgment": "PARTIAL"},
            {"relevance_judgment": "PARTIAL"},
        ], expected_count=2), "results": [
            {"relevance_judgment": "PARTIAL"},
            {"relevance_judgment": "PARTIAL"},
        ]},
    ]
    result = summarize_query_evaluations(queries)
    assert result["complete"] is True
    assert result["judgment_count"] == 4
    assert result["label_counts"] == {"IRRELEVANT": 1, "PARTIAL": 2, "RELEVANT": 1}
    assert result["macro_strict_precision_at_k"] == pytest.approx(0.25)
    assert result["macro_broad_precision_at_k"] == pytest.approx(0.75)
