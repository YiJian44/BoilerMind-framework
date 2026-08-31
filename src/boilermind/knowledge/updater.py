from hashlib import sha256

from boilermind.core.contracts import (
    ExperimentResult,
    ScientificHypothesis,
    ScientificResult,
)

from .contracts import (
    KnowledgeEntityRecord,
    KnowledgeExtraction,
    KnowledgeRelationRecord,
    KnowledgeUpdate,
)


def _stable_id(
    prefix: str,
    *parts: str,
) -> str:
    raw = "|".join(parts)

    digest = sha256(
        raw.encode("utf-8")
    ).hexdigest()[:12]

    return f"{prefix}-{digest}"


def build_knowledge_update(
    hypothesis: ScientificHypothesis,
    experiment_result: ExperimentResult,
    scientific_result: ScientificResult,
    extraction: KnowledgeExtraction,
) -> KnowledgeUpdate:
    """
    Build a provenance-rich knowledge graph update
    from an actually executed hypothesis.

    This function does not decide whether the
    hypothesis is scientifically correct.

    SUPPORTED, FALSIFIED, PARTIALLY_SUPPORTED and
    INSUFFICIENT_EVIDENCE are all retained explicitly.
    """

    if (
        hypothesis.hypothesis_id
        != experiment_result.hypothesis_id
    ):
        raise ValueError(
            "Hypothesis/ExperimentResult ID mismatch."
        )

    if (
        hypothesis.hypothesis_id
        != scientific_result.hypothesis_id
    ):
        raise ValueError(
            "Hypothesis/ScientificResult ID mismatch."
        )

    if (
        experiment_result.experiment_id
        != scientific_result.experiment_id
    ):
        raise ValueError(
            "ExperimentResult/ScientificResult "
            "experiment ID mismatch."
        )

    if (
        extraction.hypothesis_id
        != hypothesis.hypothesis_id
    ):
        raise ValueError(
            "KnowledgeExtraction hypothesis "
            "ID mismatch."
        )

    temp_keys = [
        entity.temp_key
        for entity in extraction.entities
    ]

    if len(temp_keys) != len(set(temp_keys)):
        raise ValueError(
            "Knowledge extraction contains "
            "duplicate entity temp keys."
        )

    entity_map: dict[str, str] = {}

    entity_records: list[
        KnowledgeEntityRecord
    ] = []

    for entity in extraction.entities:
        entity_id = _stable_id(
            "ENT",
            hypothesis.hypothesis_id,
            experiment_result.experiment_id,
            entity.temp_key,
            entity.name,
            entity.entity_type,
        )

        entity_map[
            entity.temp_key
        ] = entity_id

        entity_records.append(
            KnowledgeEntityRecord(
                entity_id=entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                source_hypothesis_id=(
                    hypothesis.hypothesis_id
                ),
                source_experiment_id=(
                    experiment_result.experiment_id
                ),
                scientific_status=(
                    scientific_result.verdict
                ),
            )
        )

    relation_records: list[
        KnowledgeRelationRecord
    ] = []

    for relation in extraction.relations:

        if (
            relation.subject_key
            not in entity_map
        ):
            raise ValueError(
                "Unknown relation subject entity: "
                f"{relation.subject_key}"
            )

        if (
            relation.object_key
            not in entity_map
        ):
            raise ValueError(
                "Unknown relation object entity: "
                f"{relation.object_key}"
            )

        subject_id = entity_map[
            relation.subject_key
        ]

        object_id = entity_map[
            relation.object_key
        ]

        relation_id = _stable_id(
            "REL",
            hypothesis.hypothesis_id,
            experiment_result.experiment_id,
            subject_id,
            relation.predicate,
            object_id,
        )

        relation_records.append(
            KnowledgeRelationRecord(
                relation_id=relation_id,
                subject_entity_id=subject_id,
                predicate=relation.predicate,
                object_entity_id=object_id,
                source_hypothesis_id=(
                    hypothesis.hypothesis_id
                ),
                source_experiment_id=(
                    experiment_result.experiment_id
                ),
                scientific_status=(
                    scientific_result.verdict
                ),
            )
        )

    update_id = _stable_id(
        "KGU",
        hypothesis.hypothesis_id,
        experiment_result.experiment_id,
        scientific_result.verdict.value,
    )

    return KnowledgeUpdate(
        update_id=update_id,
        hypothesis_id=(
            hypothesis.hypothesis_id
        ),
        experiment_id=(
            experiment_result.experiment_id
        ),
        verdict=scientific_result.verdict,
        entities=entity_records,
        relations=relation_records,
        highlighted_entity_ids=[
            entity.entity_id
            for entity in entity_records
        ],
        highlighted_relation_ids=[
            relation.relation_id
            for relation in relation_records
        ],
        network_updated=True,
        message="科研网络已更新",
    )