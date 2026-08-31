import pytest

from boilermind.core.enums import ResearchRunStatus
from boilermind.core.state_machine import (
    InvalidStateTransition,
    transition,
)


def test_normal_transition():
    result = transition(
        ResearchRunStatus.CREATED,
        ResearchRunStatus.PROBLEM_PARSED,
    )

    assert result == ResearchRunStatus.PROBLEM_PARSED


def test_cannot_skip_directly_to_resolved():
    with pytest.raises(InvalidStateTransition):
        transition(
            ResearchRunStatus.CREATED,
            ResearchRunStatus.RESOLVED,
        )


def test_falsified_path_can_rerank():
    result = transition(
        ResearchRunStatus.RESULT_AUDITED,
        ResearchRunStatus.DYNAMIC_RERANKED,
    )

    assert result == ResearchRunStatus.DYNAMIC_RERANKED


def test_supported_path_can_resolve():
    result = transition(
        ResearchRunStatus.RESULT_AUDITED,
        ResearchRunStatus.RESOLVED,
    )

    assert result == ResearchRunStatus.RESOLVED