from boilermind.core.enums import ResearchRunStatus


class InvalidStateTransition(RuntimeError):
    pass


_ALLOWED_TRANSITIONS: dict[
    ResearchRunStatus,
    set[ResearchRunStatus],
] = {
    ResearchRunStatus.CREATED: {
        ResearchRunStatus.PROBLEM_PARSED,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.PROBLEM_PARSED: {
        ResearchRunStatus.EVIDENCE_RETRIEVED,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.EVIDENCE_RETRIEVED: {
        ResearchRunStatus.EVIDENCE_VERIFIED,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.EVIDENCE_VERIFIED: {
        ResearchRunStatus.EVIDENCE_FROZEN,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.EVIDENCE_FROZEN: {
        ResearchRunStatus.HYPOTHESES_GENERATED,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.HYPOTHESES_GENERATED: {
        ResearchRunStatus.HYPOTHESES_QUALIFIED,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.HYPOTHESES_QUALIFIED: {
        ResearchRunStatus.PRIOR_RANKED,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.PRIOR_RANKED: {
        ResearchRunStatus.TOP3_PLANNED,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.TOP3_PLANNED: {
        ResearchRunStatus.CHAMPION_TESTING,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.CHAMPION_TESTING: {
        ResearchRunStatus.RESULT_AUDITED,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.RESULT_AUDITED: {
        ResearchRunStatus.RESOLVED,
        ResearchRunStatus.DYNAMIC_RERANKED,
        ResearchRunStatus.EXTENDED_VALIDATION,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.DYNAMIC_RERANKED: {
        ResearchRunStatus.CHAMPION_TESTING,
        ResearchRunStatus.EXTENDED_VALIDATION,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.EXTENDED_VALIDATION: {
        ResearchRunStatus.RESOLVED,
        ResearchRunStatus.UNRESOLVED_BEST_EFFORT,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.RESOLVED: {
        ResearchRunStatus.KNOWLEDGE_UPDATED,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.UNRESOLVED_BEST_EFFORT: {
        ResearchRunStatus.KNOWLEDGE_UPDATED,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.KNOWLEDGE_UPDATED: {
        ResearchRunStatus.REPORT_READY,
        ResearchRunStatus.FAILED,
    },

    ResearchRunStatus.REPORT_READY: set(),

    ResearchRunStatus.FAILED: set(),
}


def can_transition(
    current: ResearchRunStatus,
    target: ResearchRunStatus,
) -> bool:
    return target in _ALLOWED_TRANSITIONS[current]


def transition(
    current: ResearchRunStatus,
    target: ResearchRunStatus,
) -> ResearchRunStatus:
    if not can_transition(current, target):
        raise InvalidStateTransition(
            f"Invalid research state transition: "
            f"{current.value} -> {target.value}"
        )

    return target