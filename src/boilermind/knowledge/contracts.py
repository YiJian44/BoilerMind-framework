from pydantic import Field

from boilermind.core.contracts.base import ContractModel
from boilermind.core.enums import ScientificVerdict


class KnowledgeEntityDraft(ContractModel):
    temp_key: str = Field(min_length=1)

    name: str = Field(min_length=1)

    entity_type: str = Field(min_length=1)


class KnowledgeRelationDraft(ContractModel):
    subject_key: str = Field(min_length=1)

    predicate: str = Field(min_length=1)

    object_key: str = Field(min_length=1)


class KnowledgeExtraction(ContractModel):
    hypothesis_id: str = Field(min_length=1)

    entities: list[KnowledgeEntityDraft] = Field(
        min_length=1
    )

    relations: list[KnowledgeRelationDraft] = Field(
        default_factory=list
    )


class KnowledgeEntityRecord(ContractModel):
    entity_id: str = Field(min_length=1)

    name: str = Field(min_length=1)

    entity_type: str = Field(min_length=1)

    source_hypothesis_id: str = Field(
        min_length=1
    )

    source_experiment_id: str = Field(
        min_length=1
    )

    scientific_status: ScientificVerdict


class KnowledgeRelationRecord(ContractModel):
    relation_id: str = Field(min_length=1)

    subject_entity_id: str = Field(
        min_length=1
    )

    predicate: str = Field(min_length=1)

    object_entity_id: str = Field(
        min_length=1
    )

    source_hypothesis_id: str = Field(
        min_length=1
    )

    source_experiment_id: str = Field(
        min_length=1
    )

    scientific_status: ScientificVerdict


class KnowledgeUpdate(ContractModel):
    update_id: str = Field(min_length=1)

    hypothesis_id: str = Field(min_length=1)

    experiment_id: str = Field(min_length=1)

    verdict: ScientificVerdict

    entities: list[KnowledgeEntityRecord] = Field(
        default_factory=list
    )

    relations: list[KnowledgeRelationRecord] = Field(
        default_factory=list
    )

    highlighted_entity_ids: list[str] = Field(
        default_factory=list
    )

    highlighted_relation_ids: list[str] = Field(
        default_factory=list
    )

    network_updated: bool = True

    message: str = "科研网络已更新"