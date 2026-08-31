from datetime import datetime, timezone

import pytest

from boilermind.core.contracts import (
    ExperimentResult,
    MechanismStep,
    ScientificHypothesis,
    ScientificResult,
)

from boilermind.core.enums import (
    ExperimentStatus,
    MechanismSupportType,
    ScientificVerdict,
)

from boilermind.knowledge import (
    KnowledgeEntityDraft,
    KnowledgeExtraction,
    KnowledgeRelationDraft,
    build_knowledge_update,
)


HASH = "f" * 64


def make_hypothesis():
    return ScientificHypothesis(
        hypothesis_id="H001",
        problem_id="P001",
        title="Dynamic lag hypothesis",
        research_significance=(
            "Explain soft-sensor error."
        ),
        hypothesis=(
            "Dynamic lag affects prediction error."
        ),
        mechanism_chain=(
            "load -> lag -> prediction error"
        ),
        mechanism_steps=[
            MechanismStep(
                step=1,
                statement=(
                    "Dynamic lag may affect error."
                ),
                support_type=(
                    MechanismSupportType
                    .HYPOTHESIS_INFERENCE
                ),
                evidence_ids=[],
            )
        ],
        related_variables=[
            "load_mw",
            "steam_volume_flow",
        ],
        applicability_conditions=[
            "deep peak regulation"
        ],
        verification_intent=(
            "Run controlled experiment."
        ),
        expected_observation=(
            "Prediction error changes."
        ),
        confirmation_criteria=[
            "Target achieved."
        ],
        falsification_criteria=[
            "Target not achieved."
        ],
        novelty_axis="Dynamic lag",
        evidence_bundle_sha256=HASH,
    )


def make_experiment_result():
    now = datetime.now(
        timezone.utc
    )

    return ExperimentResult(
        experiment_id="EXP-001",
        hypothesis_id="H001",
        status=ExperimentStatus.COMPLETED,
        metrics={
            "MAE": 0.03,
        },
        baseline_metrics={
            "MAE": 0.02,
        },
        artifacts=[],
        execution_notes=[],
        started_at=now,
        completed_at=now,
    )


def make_scientific_result(
    verdict,
):
    return ScientificResult(
        hypothesis_id="H001",
        experiment_id="EXP-001",
        verdict=verdict,
        rationale="Scientific result.",
        achieved_criteria=[],
        failed_criteria=[],
    )


def make_extraction():
    return KnowledgeExtraction(
        hypothesis_id="H001",
        entities=[
            KnowledgeEntityDraft(
                temp_key="load",
                name="锅炉负荷",
                entity_type="process_variable",
            ),
            KnowledgeEntityDraft(
                temp_key="lag",
                name="动态时滞",
                entity_type="mechanism",
            ),
            KnowledgeEntityDraft(
                temp_key="error",
                name="软测量预测误差",
                entity_type="performance_metric",
            ),
        ],
        relations=[
            KnowledgeRelationDraft(
                subject_key="load",
                predicate="influences",
                object_key="lag",
            ),
            KnowledgeRelationDraft(
                subject_key="lag",
                predicate="affects",
                object_key="error",
            ),
        ],
    )


def test_supported_hypothesis_updates_network():
    update = build_knowledge_update(
        make_hypothesis(),
        make_experiment_result(),
        make_scientific_result(
            ScientificVerdict.SUPPORTED
        ),
        make_extraction(),
    )

    assert update.network_updated is True

    assert update.message == (
        "科研网络已更新"
    )

    assert len(update.entities) == 3
    assert len(update.relations) == 2

    assert (
        update.verdict
        == ScientificVerdict.SUPPORTED
    )

    assert (
        len(update.highlighted_entity_ids)
        == 3
    )


def test_falsified_hypothesis_is_still_recorded():
    update = build_knowledge_update(
        make_hypothesis(),
        make_experiment_result(),
        make_scientific_result(
            ScientificVerdict.FALSIFIED
        ),
        make_extraction(),
    )

    assert update.network_updated is True

    assert (
        update.verdict
        == ScientificVerdict.FALSIFIED
    )

    assert all(
        entity.scientific_status
        == ScientificVerdict.FALSIFIED
        for entity in update.entities
    )


def test_relation_cannot_reference_unknown_entity():
    bad_extraction = KnowledgeExtraction(
        hypothesis_id="H001",
        entities=[
            KnowledgeEntityDraft(
                temp_key="load",
                name="锅炉负荷",
                entity_type="process_variable",
            )
        ],
        relations=[
            KnowledgeRelationDraft(
                subject_key="load",
                predicate="affects",
                object_key="UNKNOWN",
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="Unknown relation object",
    ):
        build_knowledge_update(
            make_hypothesis(),
            make_experiment_result(),
            make_scientific_result(
                ScientificVerdict.FALSIFIED
            ),
            bad_extraction,
        )


def test_unmatched_experiment_cannot_update_network():
    scientific_result = (
        make_scientific_result(
            ScientificVerdict.SUPPORTED
        ).model_copy(
            update={
                "experiment_id": "EXP-WRONG"
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="experiment ID mismatch",
    ):
        build_knowledge_update(
            make_hypothesis(),
            make_experiment_result(),
            scientific_result,
            make_extraction(),
        )