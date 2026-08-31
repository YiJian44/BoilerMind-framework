from datetime import (
    datetime,
    timezone,
)
import hashlib
from pathlib import Path

from boilermind.core.contracts import (
    EvidenceCandidate,
)

from boilermind.evidence.traceability_verifier import (
    EvidenceTraceabilityVerifier,
)


def make_candidate(**kwargs):
    defaults = dict(
        evidence_id="E-1",
        problem_id="P-1",
        source_type="local_literature",
        title="Scientific Paper",
        text="Scientific evidence text.",
        retrieval_score=1.0,
        retrieved_at=datetime.now(
            timezone.utc
        ),
    )

    defaults.update(kwargs)

    return EvidenceCandidate(
        **defaults
    )


def test_local_evidence_requires_real_source_file(
    tmp_path: Path,
):
    rag_root = tmp_path / "local_rag"

    pdf = (
        rag_root
        / "input"
        / "core"
        / "paper.pdf"
    )

    pdf.parent.mkdir(
        parents=True
    )

    pdf.write_bytes(
        b"%PDF-test"
    )

    candidate = make_candidate(
        document_id="DOC-001",
        chunk_id="DOC-001-P001-C001",
        page_number=1,
        source_file=(
            "input/core/paper.pdf"
        ),
        document_sha256=hashlib.sha256(b"%PDF-test").hexdigest(),
    )

    verifier = (
        EvidenceTraceabilityVerifier(
            rag_root
        )
    )

    result = verifier.verify(
        candidate
    )

    assert result.verified is True


def test_missing_local_pdf_is_rejected(
    tmp_path: Path,
):
    candidate = make_candidate(
        document_id="DOC-001",
        chunk_id="DOC-001-P001-C001",
        page_number=1,
        source_file=(
            "input/core/missing.pdf"
        ),
    )

    verifier = (
        EvidenceTraceabilityVerifier(
            tmp_path
        )
    )

    result = verifier.verify(
        candidate
    )

    assert result.verified is False


def test_crossref_doi_is_traceable():
    candidate = make_candidate(
        source_type="web_literature",
        citation=(
            "provider=crossref; "
            "doi=10.1109/example.123"
        ),
        source_url=(
            "https://doi.org/"
            "10.1109/example.123"
        ),
    )

    verifier = (
        EvidenceTraceabilityVerifier()
    )

    result = verifier.verify(
        candidate
    )

    assert result.verified is True


def test_arxiv_identifier_is_traceable():
    candidate = make_candidate(
        source_type="web_literature",
        citation=(
            "provider=arxiv; "
            "id=http://arxiv.org/abs/"
            "2407.11180v1"
        ),
        source_url=(
            "http://arxiv.org/abs/"
            "2407.11180v1"
        ),
    )

    verifier = (
        EvidenceTraceabilityVerifier()
    )

    result = verifier.verify(
        candidate
    )

    assert result.verified is True


def test_unidentified_web_result_is_rejected():
    candidate = make_candidate(
        source_type="web_literature",
        citation=(
            "provider=unknown"
        ),
        source_url=(
            "https://example.com/paper"
        ),
    )

    verifier = (
        EvidenceTraceabilityVerifier()
    )

    result = verifier.verify(
        candidate
    )

    assert result.verified is False
