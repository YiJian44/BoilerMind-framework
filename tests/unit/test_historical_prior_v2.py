from boilermind.ranking.historical_prior import rank_hypotheses, score_hypothesis


def hypothesis(hid: str, direct: int, *, executable: bool = True, conflict: bool = False):
    return {
        "hypothesis_id": hid,
        "source_experiment_ids": [f"EXP-{hid}"] if direct else [],
        "source_observation_ids": [f"OBS-{hid}-{i}" for i in range(direct)],
        "historical_assessment": {
            "directly_supporting_observations": [f"OBS-{hid}-{i}" for i in range(direct)],
            "conflicting_observations": ["OBS-CONFLICT"] if conflict else [],
            "conditionally_related_observations": [],
            "scope_mismatches": [],
        },
        "verification_mapping": {"executable_now": executable},
        "confirmation_criteria": ["MAE < baseline"],
        "falsification_criteria": ["MAE >= baseline"],
        "problem_relevance": 1.0,
    }


def test_history_supported_hypotheses_rank_first_and_conflict_drops():
    ranking = rank_hypotheses([
        hypothesis("H2", 1), hypothesis("H1", 2),
        hypothesis("H3", 0), hypothesis("H4", 1, conflict=True),
    ])
    assert [item.hypothesis_id for item in ranking[:2]] == ["H1", "H2"]
    by_id = {item.hypothesis_id: item for item in ranking}
    assert by_id["H3"].eligible is False
    assert by_id["H4"].eligible is False
    assert "historical_direct_conflict" in by_id["H4"].dropped_reasons


def test_prior_formula_is_deterministic_and_has_no_novelty_component():
    first = score_hypothesis(hypothesis("H1", 1))
    second = score_hypothesis(hypothesis("H1", 1))
    assert first == second
    assert first.prior_score == 0.94
    assert "novelty" not in first.model_dump()


def test_executable_user_proposal_can_run_without_prior_history():
    candidate = hypothesis("H-NEW", 0)
    candidate["trigger_types"] = ["HUMAN_PROPOSAL"]
    score = score_hypothesis(candidate)
    assert score.eligible is True
    assert score.historical_support == 0.0
    assert "no_empirical_grounding" not in score.dropped_reasons
