from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchPolicy:
    """
    Authoritative scientific workflow policy for BoilerMind.
    """

    # Hypothesis generation
    min_generated_hypotheses: int = 5
    max_generated_hypotheses: int = 10

    # After scientific admission / quality gate
    min_qualified_hypotheses: int = 3

    # Primary candidate pool
    primary_candidate_count: int = 3

    # Champion validation is sequential
    execute_primary_sequentially: bool = True

    # Remaining hypotheses may be executed together
    # only after all Top-3 candidates are falsified.
    allow_parallel_extended_validation: bool = True

    # Stop immediately once a supported hypothesis appears.
    stop_on_supported_hypothesis: bool = True

    # Every actually executed hypothesis must update KG.
    update_knowledge_for_executed_hypotheses: bool = True

    # Never call best-effort result "supported".
    best_effort_is_scientific_solution: bool = False


DEFAULT_RESEARCH_POLICY = ResearchPolicy()