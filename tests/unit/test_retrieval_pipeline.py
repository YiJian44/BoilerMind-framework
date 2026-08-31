from datetime import (
    datetime,
    timezone,
)

from boilermind.core.contracts import (
    EvidenceCandidate,
    ResearchProblemSpec,
)

from boilermind.evidence.retrieval_pipeline import (
    ScientificRetrievalPipeline,
    deduplicate_candidates,
)


def make_problem():
    question = (
        "How do delayed relations among "
        "boiler variables affect prediction?"
    )

    return ResearchProblemSpec(
        problem_id="P-RETRIEVAL-001",
        original_question=question,
        research_object="boiler plant",
        target_variable="prediction",
        operating_condition=(
            "dynamic operation"
        ),
        manipulated_variables=[],
        observed_variables=[
            "boiler variables",
        ],
        context_variables=[
            "delayed relations",
        ],
        research_goal=question,
        success_criteria=[
            "retrieve evidence"
        ],
        constraints=[],
    )


def candidate(
    *,
    evidence_id,
    source_type,
    title,
    citation=None,
    source_url=None,
    source_file=None,
    document_id=None,
    chunk_id=None,
    page_number=None,
    text=None,
):
    return EvidenceCandidate(
        evidence_id=evidence_id,
        problem_id="P-RETRIEVAL-001",
        source_type=source_type,
        title=title,
        source_url=source_url,
        citation=citation,
        text=(
            text
            or title
        ),
        retrieval_score=1.0,
        retrieved_at=datetime.now(
            timezone.utc
        ),
        source_file=source_file,
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=page_number,
    )


def test_cross_source_duplicate_is_collapsed():
    local = candidate(
        evidence_id="LOCAL-1",
        source_type="local_literature",
        title="01 2407.11180",
        source_file=(
            "input/core/01_2407.11180.pdf"
        ),
        document_id="DOC-1",
        chunk_id="DOC-1-P1-C1",
        page_number=1,
        text=(
            "Transformer-based Drum-level Prediction "
            "in a Boiler\n"
            "Plant with Delayed Relations among "
            "Multivariates"
        ),
    )

    arxiv = candidate(
        evidence_id="WEB-ARXIV-1",
        source_type="web_literature",
        title=(
            "Transformer-based Drum-level Prediction "
            "in a Boiler Plant with Delayed Relations "
            "among Multivariates"
        ),
        citation=(
            "provider=arxiv; "
            "id=http://arxiv.org/abs/"
            "2407.11180v1"
        ),
    )

    crossref = candidate(
        evidence_id="WEB-CROSSREF-1",
        source_type="web_literature",
        title=(
            "Transformer-based Drum-level Prediction "
            "in a Boiler Plant with Delayed Relations "
            "among Multivariates"
        ),
        citation=(
            "provider=crossref; "
            "doi=10.1109/example"
        ),
    )

    results = deduplicate_candidates(
        [
            local,
            arxiv,
            crossref,
        ]
    )

    assert len(results) == 1

    # Prefer the local copy because it contains
    # exact PDF/page/chunk provenance.
    assert (
        results[0].evidence_id
        == "LOCAL-1"
    )


class FakeSource:
    def __init__(
        self,
        source_type,
        results,
    ):
        self.source_type = source_type
        self.results = results

    def retrieve(
        self,
        problem,
    ):
        return self.results


def test_pipeline_merges_multiple_sources():
    local = candidate(
        evidence_id="LOCAL-A",
        source_type="local_literature",
        title="Local Boiler Study",
    )

    web = candidate(
        evidence_id="WEB-B",
        source_type="web_literature",
        title="Web Boiler Study",
    )

    pipeline = ScientificRetrievalPipeline(
        sources=[
            FakeSource(
                "local_literature",
                [local],
            ),
            FakeSource(
                "web_literature",
                [web],
            ),
        ]
    )

    results = pipeline.retrieve(
        make_problem()
    )

    assert len(results) == 2

    assert {
        item.source_type
        for item in results
    } == {
        "local_literature",
        "web_literature",
    }