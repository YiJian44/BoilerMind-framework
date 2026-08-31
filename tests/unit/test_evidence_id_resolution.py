from boilermind.hypothesis.evidence_id_resolver import (
    normalize_evidence_ids,
    resolve_evidence_id,
)


def test_exact_evidence_id_wins():
    available = ["LOCAL-DOC_alpha", "LOCAL-DOC_alpha_P005_C003"]
    result = resolve_evidence_id("LOCAL-DOC_alpha", available)
    assert result.status == "exact"
    assert result.resolved_id == "LOCAL-DOC_alpha"


def test_unique_document_prefix_resolves_to_chunk_id():
    available = ["LOCAL-DOC_alpha_P005_C003", "LOCAL-DOC_beta_P001_C001"]
    assert normalize_evidence_ids(["LOCAL-DOC_alpha"], available) == [
        "LOCAL-DOC_alpha_P005_C003"
    ]


def test_ambiguous_prefix_remains_fail_closed():
    available = ["LOCAL-DOC_alpha_P005_C003", "LOCAL-DOC_alpha_P006_C001"]
    result = resolve_evidence_id("LOCAL-DOC_alpha", available)
    assert result.status == "ambiguous"
    assert result.resolved_id is None
    assert normalize_evidence_ids(["LOCAL-DOC_alpha"], available) == [
        "LOCAL-DOC_alpha"
    ]


def test_partial_token_is_not_a_valid_prefix_reference():
    result = resolve_evidence_id("LOCAL-DOC_al", ["LOCAL-DOC_alpha_P005_C003"])
    assert result.status == "unknown"
    assert result.resolved_id is None
