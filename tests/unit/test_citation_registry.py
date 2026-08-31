import json
from pathlib import Path

import pytest

from boilermind.evidence.citation_registry import (
    CitationRegistry,
    CitationRegistryError,
    FORMALLY_CITABLE,
    RETRIEVAL_ONLY,
    format_gbt7714_2015,
)


RAG_ROOT = Path(__file__).resolve().parents[2] / "resources" / "local_rag"


@pytest.fixture(scope="module")
def registry():
    return CitationRegistry(RAG_ROOT)


def test_registry_counts_are_consistent_with_current_approval_ledger(registry):
    candidates = [
        item["citation_eligibility"] for item in registry.records.values()
    ]
    effective = [registry.citation_state(document_id)[3] for document_id in registry.records]
    assert len(candidates) == 110
    assert candidates.count(FORMALLY_CITABLE) + candidates.count(RETRIEVAL_ONLY) == 110
    assert effective.count(FORMALLY_CITABLE) == len(registry.approvals)
    assert effective.count(FORMALLY_CITABLE) + effective.count(RETRIEVAL_ONLY) == 110


def test_candidate_records_render_deterministically_and_bind_to_pdf_hash(registry):
    document_id = next(
        item["document_id"]
        for item in registry.records.values()
        if item["citation_eligibility"] == FORMALLY_CITABLE
    )
    first = registry.candidate_citation(document_id, verify_pdf_hash=True)
    second = registry.candidate_citation(document_id, verify_pdf_hash=True)
    assert first == second
    assert first.endswith(".")
    assert "[J" in first


def test_unapproved_and_retrieval_only_records_cannot_be_formally_rendered(registry):
    candidate_id = next(
        item["document_id"]
        for item in registry.records.values()
        if item["citation_eligibility"] == FORMALLY_CITABLE
    )
    original_approval = registry.approvals.pop(candidate_id, None)
    try:
        with pytest.raises(CitationRegistryError, match="Human approval"):
            registry.formal_citation(candidate_id)
    finally:
        if original_approval is not None:
            registry.approvals[candidate_id] = original_approval

    record = registry.records[candidate_id]
    original_eligibility = record["citation_eligibility"]
    try:
        record["citation_eligibility"] = RETRIEVAL_ONLY
        with pytest.raises(CitationRegistryError, match="RETRIEVAL_ONLY"):
            registry.candidate_citation(candidate_id)
    finally:
        record["citation_eligibility"] = original_eligibility


def test_matching_human_approval_unlocks_only_its_bound_snapshot(registry):
    document_id = next(
        item["document_id"]
        for item in registry.records.values()
        if item["citation_eligibility"] == FORMALLY_CITABLE
    )
    original_approval = registry.approvals.get(document_id)
    try:
        snapshot = registry.approval_snapshot(document_id)
        registry.approvals[document_id] = {
            "document_id": document_id,
            "decision": "APPROVED",
            **snapshot,
        }
        assert registry.formal_citation(document_id)
        registry.approvals[document_id]["formatted_citation_sha256"] = "0" * 64
        with pytest.raises(CitationRegistryError, match="Human approval"):
            registry.formal_citation(document_id)
    finally:
        if original_approval is None:
            registry.approvals.pop(document_id, None)
        else:
            registry.approvals[document_id] = original_approval


def test_claim_binding_requires_exact_document_page_chunk_and_excerpt(registry):
    chunk = next(iter(registry.chunks.values()))
    excerpt = " ".join(chunk["text"].split())[:80]
    valid = registry.verify_claim_binding(
        document_id=chunk["document_id"],
        chunk_ids=[chunk["chunk_id"]],
        page_number=chunk["page_number"],
        supporting_excerpt=excerpt,
    )
    assert valid.valid

    invalid = registry.verify_claim_binding(
        document_id="DOC_WRONG",
        chunk_ids=[chunk["chunk_id"]],
        page_number=chunk["page_number"] + 1,
        supporting_excerpt="不存在的伪造摘录",
    )
    assert not invalid.valid
    assert any(error.startswith("document_mismatch") for error in invalid.errors)
    assert any(error.startswith("page_mismatch") for error in invalid.errors)
    assert any(error.startswith("excerpt_mismatch") for error in invalid.errors)


def test_identity_dataset_document_ids_match_retrieval_catalog(registry):
    assert set(registry.records) == set(registry.papers)


def test_conference_candidate_is_not_rendered_as_journal(registry):
    citation = registry.candidate_citation("DOC_11AC5EC36E6E")
    assert "[C]//Proceedings of the 4th International Conference" in citation
    assert "[J]" not in citation


def test_compact_initials_and_author_order_are_preserved(registry):
    digital_twin = registry.candidate_citation("DOC_ACE0C417331D")
    assert digital_twin.startswith("DESAI AS, NAVANEETH N, ADHIKARI S, et al")
    labor = registry.candidate_citation("DOC_71C6740B18CC")
    assert labor.startswith("ABBOUCHI O, DAVILA S, AL HASANI M, et al")


def test_corrected_fan_jin_name_is_rendered_in_family_given_order(registry):
    citation = registry.candidate_citation("DOC_32343C3D95EA")
    assert citation.startswith("FAN J, ZHANG K, HUANG Y, et al")


def test_online_first_journal_uses_combined_resource_marker_without_inventing_pagination():
    citation = format_gbt7714_2015({
        "document_id": "DOC_ONLINE_FIRST",
        "publication_type": "journal_article",
        "medium": "OL",
        "title": "Online first article",
        "authors": [{"family": "DOE", "given": "Jane"}],
        "issued_year": 2025,
        "container_title": "Example Journal",
        "volume": "",
        "issue": "",
        "pages": "",
        "doi": "10.1000/example",
    })
    assert citation == (
        "DOE J. Online first article[J/OL]. Example Journal. 2025. "
        "DOI:10.1000/example."
    )
