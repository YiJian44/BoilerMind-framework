import json

from boilermind.core.contracts import (
    ResearchProblemSpec,
)

from boilermind.evidence.sources.web_literature import (
    WebLiteratureSource,
    build_web_literature_query,
)


def make_problem():
    question = (
        "How do delayed relations among "
        "boiler variables affect prediction?"
    )

    return ResearchProblemSpec(
        problem_id="P-WEB-001",
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
            "retrieve literature"
        ],
        constraints=[],
    )


def test_web_query_comes_from_problem():
    problem = make_problem()

    query = build_web_literature_query(
        problem
    )

    assert (
        problem.original_question
        in query
    )

    assert "boiler plant" in query


def test_crossref_records_become_candidates():
    source = WebLiteratureSource(
        crossref_results=2,
        arxiv_results=0,
        top_k=2,
    )

    payload = {
        "message": {
            "items": [
                {
                    "DOI": "10.1234/test.1",
                    "title": [
                        "Boiler Prediction Study"
                    ],
                    "author": [
                        {
                            "given": "A",
                            "family": "Researcher",
                        }
                    ],
                    "container-title": [
                        "Journal of Testing"
                    ],
                    "type": (
                        "journal-article"
                    ),
                    "URL": (
                        "https://doi.org/"
                        "10.1234/test.1"
                    ),
                    "abstract": (
                        "<jats:p>"
                        "Prediction abstract."
                        "</jats:p>"
                    ),
                }
            ]
        }
    }

    source._get = lambda url: (
        json.dumps(payload)
        .encode("utf-8")
    )

    results = source.retrieve(
        make_problem()
    )

    assert len(results) == 1

    item = results[0]

    assert (
        item.source_type
        == "web_literature"
    )

    assert (
        "Boiler Prediction Study"
        in item.text
    )

    assert (
        "10.1234/test.1"
        in item.citation
    )


def test_web_source_deduplicates_titles():
    source = WebLiteratureSource(
        crossref_results=0,
        arxiv_results=0,
        top_k=5,
    )

    source._search_crossref = (
        lambda query: [
            {
                "provider": "crossref",
                "external_id": "A",
                "doi": None,
                "title": "Same Paper",
                "url": "https://example/a",
                "text": "Same Paper",
                "rank": 0,
            }
        ]
    )

    source._search_arxiv = (
        lambda query: [
            {
                "provider": "arxiv",
                "external_id": "B",
                "doi": None,
                "title": "Same Paper",
                "url": "https://example/b",
                "text": "Same Paper",
                "rank": 0,
            }
        ]
    )

    results = source.retrieve(
        make_problem()
    )

    assert len(results) == 1