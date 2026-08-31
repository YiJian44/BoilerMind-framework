from enum import StrEnum


class ResearchRunStatus(StrEnum):
    CREATED = "created"
    PROBLEM_PARSED = "problem_parsed"

    EVIDENCE_RETRIEVED = "evidence_retrieved"
    EVIDENCE_VERIFIED = "evidence_verified"
    EVIDENCE_FROZEN = "evidence_frozen"

    HYPOTHESES_GENERATED = "hypotheses_generated"
    HYPOTHESES_QUALIFIED = "hypotheses_qualified"

    PRIOR_RANKED = "prior_ranked"
    TOP3_PLANNED = "top3_planned"

    CHAMPION_TESTING = "champion_testing"
    RESULT_AUDITED = "result_audited"
    DYNAMIC_RERANKED = "dynamic_reranked"

    EXTENDED_VALIDATION = "extended_validation"

    RESOLVED = "resolved"
    UNRESOLVED_BEST_EFFORT = "unresolved_best_effort"

    KNOWLEDGE_UPDATED = "knowledge_updated"
    REPORT_READY = "report_ready"

    FAILED = "failed"


class EvidenceStage(StrEnum):
    RETRIEVED = "retrieved"
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    REJECTED = "rejected"


class ClaimSupport(StrEnum):
    DIRECT = "direct"
    PARTIAL = "partial"
    CONTRADICTING = "contradicting"
    IRRELEVANT = "irrelevant"
    UNKNOWN = "unknown"


class ApplicabilityLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class MechanismSupportType(StrEnum):
    VERIFIED_EVIDENCE = "verified_evidence"
    DATA_OBSERVATION = "data_observation"
    DOMAIN_PRIOR = "domain_prior"
    HYPOTHESIS_INFERENCE = "hypothesis_inference"


class HypothesisStatus(StrEnum):
    GENERATED = "generated"
    QUALIFIED = "qualified"
    REJECTED = "rejected"

    PLANNED = "planned"
    TESTING = "testing"

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    FALSIFIED = "falsified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class RankingMethod(StrEnum):
    PRIOR_CONFIDENCE = "prior_confidence"
    EXPERIMENT_FEEDBACK = "experiment_feedback"
    PAIRWISE_ELO = "pairwise_elo"


class ExperimentStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    INVALID = "invalid"
    FAILED = "failed"


class ScientificVerdict(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    FALSIFIED = "falsified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ResearchStopReason(StrEnum):
    TARGET_METRIC_REACHED = "target_metric_reached"
    HYPOTHESIS_RESOLVED = "hypothesis_resolved"
    ALL_CANDIDATES_EXHAUSTED = "all_candidates_exhausted"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    TIME_BUDGET_EXHAUSTED = "time_budget_exhausted"
    NO_MEANINGFUL_IMPROVEMENT = "no_meaningful_improvement"
    CAPABILITY_BLOCKED = "capability_blocked"
    EXECUTION_FAILED = "execution_failed"


class ProblemResolutionStatus(StrEnum):
    SOLVED = "solved"
    PARTIALLY_SOLVED = "partially_solved"
    NOT_SOLVED = "not_solved"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EXECUTION_BLOCKED = "execution_blocked"


class ScoreSource(StrEnum):
    EVIDENCE_VERIFIER = "evidence_verifier"
    MECHANISM_CRITIC = "mechanism_critic"
    DATA_ANALYSIS = "data_analysis"
    EXPERIMENT_CAPABILITY = "experiment_capability"
    HYPOTHESIS_STRUCTURE = "hypothesis_structure"
    NOVELTY_ANALYZER = "novelty_analyzer"
    KNOWLEDGE_GRAPH = "knowledge_graph"
