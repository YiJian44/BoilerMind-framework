from boilermind.core.research_policy import (
    DEFAULT_RESEARCH_POLICY,
)


def test_hypothesis_generation_policy():
    policy = DEFAULT_RESEARCH_POLICY

    assert policy.min_generated_hypotheses == 5
    assert policy.max_generated_hypotheses == 10
    assert policy.min_qualified_hypotheses == 3


def test_primary_candidate_policy():
    policy = DEFAULT_RESEARCH_POLICY

    assert policy.primary_candidate_count == 3
    assert policy.execute_primary_sequentially is True


def test_stop_on_supported_hypothesis():
    policy = DEFAULT_RESEARCH_POLICY

    assert policy.stop_on_supported_hypothesis is True


def test_extended_validation_policy():
    policy = DEFAULT_RESEARCH_POLICY

    assert (
        policy.allow_parallel_extended_validation
        is True
    )


def test_knowledge_growth_policy():
    policy = DEFAULT_RESEARCH_POLICY

    assert (
        policy.update_knowledge_for_executed_hypotheses
        is True
    )


def test_best_effort_is_not_scientific_solution():
    policy = DEFAULT_RESEARCH_POLICY

    assert (
        policy.best_effort_is_scientific_solution
        is False
    )