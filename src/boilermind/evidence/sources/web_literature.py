import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from datetime import datetime, timezone

from boilermind.core.contracts import (
    EvidenceCandidate,
    ResearchProblemSpec,
)


class WebLiteratureError(RuntimeError):
    pass


def build_web_literature_query(
    problem: ResearchProblemSpec,
) -> str:
    """
    Build a query dynamically from the current
    ResearchProblemSpec.

    No scientific question or target is hard-coded.
    """

    parts = [
        problem.original_question,
        problem.research_object,
        problem.target_variable,
        problem.operating_condition,
        problem.research_goal,
    ]

    parts.extend(problem.manipulated_variables)
    parts.extend(problem.observed_variables)
    parts.extend(problem.context_variables)

    unique_parts = []
    seen = set()

    for part in parts:
        text = str(part or "").strip()

        if not text:
            continue

        key = text.lower()

        if key in seen:
            continue

        seen.add(key)
        unique_parts.append(text)

    return " ".join(unique_parts)


_ARXIV_WORD_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9-]{2,}"
)

_ARXIV_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "how",
    "does",
    "affect",
    "among",
    "into",
    "using",
    "under",
    "during",
    "performance",
}


def _extract_arxiv_terms(
    values: list[str],
    *,
    max_terms: int = 6,
) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value or "").lower()

        for term in _ARXIV_WORD_PATTERN.findall(text):
            if term in _ARXIV_STOPWORDS:
                continue

            if term in seen:
                continue

            seen.add(term)
            terms.append(term)

            if len(terms) >= max_terms:
                return terms

    return terms


def build_arxiv_search_query(
    problem: ResearchProblemSpec,
) -> str:
    """
    Build a Boolean arXiv query dynamically from the
    current ResearchProblemSpec.

    No specific boiler research question is hard-coded.
    """

    object_terms = _extract_arxiv_terms(
        [problem.research_object],
        max_terms=3,
    )

    target_terms = _extract_arxiv_terms(
        [problem.target_variable],
        max_terms=3,
    )

    context_terms = _extract_arxiv_terms(
        [
            problem.operating_condition,
            *problem.context_variables,
            *problem.observed_variables,
            *problem.manipulated_variables,
        ],
        max_terms=6,
    )

    groups = []

    for terms in (
        object_terms,
        target_terms,
        context_terms,
    ):
        if terms:
            groups.append(terms)

    if not groups:
        fallback_terms = _extract_arxiv_terms(
            [
                problem.original_question,
                problem.research_goal,
            ],
            max_terms=8,
        )

        if not fallback_terms:
            return ""

        groups.append(fallback_terms)

    expressions = []

    for terms in groups:
        expression = " OR ".join(
            f"all:{term}"
            for term in terms
        )

        if len(terms) > 1:
            expression = (
                "(" + expression + ")"
            )

        expressions.append(expression)

    return " AND ".join(expressions)

def _normalize_title(text: str) -> str:
    text = re.sub(
        r"\s+",
        " ",
        str(text or ""),
    ).strip()

    return text


def _dedup_key(title: str) -> str:
    normalized = re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+",
        "",
        title.lower(),
    )

    return normalized


class WebLiteratureSource:
    """
    Real web literature adapter.

    Providers:
        - Crossref
        - arXiv

    Retrieval results remain EvidenceCandidate objects.
    Retrieval itself never means scientific verification.
    """

    source_type = "web_literature"

    CROSSREF_URL = (
        "https://api.crossref.org/works"
    )

    ARXIV_URL = (
        "https://export.arxiv.org/api/query"
    )

    def __init__(
        self,
        *,
        crossref_results: int = 6,
        arxiv_results: int = 6,
        top_k: int = 8,
        timeout: float | None = None,
        crossref_mailto: str | None = None,
    ):
        if crossref_results < 0:
            raise ValueError(
                "crossref_results must be >= 0"
            )

        if arxiv_results < 0:
            raise ValueError(
                "arxiv_results must be >= 0"
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be >= 1"
            )

        self.crossref_results = (
            crossref_results
        )

        self.arxiv_results = (
            arxiv_results
        )

        self.top_k = top_k

        if timeout is None:
            timeout = float(
                os.getenv(
                    "BOILERMIND_WEB_TIMEOUT",
                    "15",
                )
            )

        self.timeout = timeout

        self.crossref_mailto = (
            crossref_mailto
            or os.getenv(
                "BOILERMIND_CROSSREF_MAILTO"
            )
        )

        self.user_agent = (
            "BoilerMind-Trusted/0.1 "
            "(Scientific Literature Retrieval)"
        )

    def _get(
        self,
        url: str,
    ) -> bytes:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "*/*",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                return response.read()

        except Exception as exc:
            raise WebLiteratureError(
                f"Web literature request failed: "
                f"{url}: {exc}"
            ) from exc

    def _search_crossref(
        self,
        query: str,
    ) -> list[dict]:
        if self.crossref_results == 0:
            return []

        params = {
            "query.bibliographic": query,
            "rows": str(
                self.crossref_results
            ),
            "select": (
                "DOI,title,author,published,"
                "published-print,"
                "published-online,"
                "container-title,"
                "type,URL,abstract"
            ),
        }

        if self.crossref_mailto:
            params["mailto"] = (
                self.crossref_mailto
            )

        url = (
            self.CROSSREF_URL
            + "?"
            + urllib.parse.urlencode(
                params
            )
        )

        payload = json.loads(
            self._get(url).decode(
                "utf-8"
            )
        )

        items = (
            payload.get(
                "message",
                {},
            ).get(
                "items",
                [],
            )
        )

        results = []

        for index, item in enumerate(
            items
        ):
            title_values = item.get(
                "title"
            ) or []

            title = (
                _normalize_title(
                    title_values[0]
                )
                if title_values
                else ""
            )

            if not title:
                continue

            doi = str(
                item.get(
                    "DOI",
                    "",
                )
            ).strip()

            url_value = (
                item.get("URL")
                or (
                    f"https://doi.org/{doi}"
                    if doi
                    else None
                )
            )

            abstract = str(
                item.get(
                    "abstract",
                    "",
                )
            )

            abstract = re.sub(
                r"<[^>]+>",
                " ",
                abstract,
            )

            abstract = re.sub(
                r"\s+",
                " ",
                abstract,
            ).strip()

            container = item.get(
                "container-title"
            ) or []

            journal = (
                container[0]
                if container
                else ""
            )

            authors = []

            for author in (
                item.get("author")
                or []
            ):
                name = " ".join(
                    part
                    for part in [
                        author.get(
                            "given",
                            "",
                        ),
                        author.get(
                            "family",
                            "",
                        ),
                    ]
                    if part
                ).strip()

                if name:
                    authors.append(name)

            text_parts = [
                title,
            ]

            if abstract:
                text_parts.append(
                    abstract
                )

            if journal:
                text_parts.append(
                    f"Journal: {journal}"
                )

            if authors:
                text_parts.append(
                    "Authors: "
                    + ", ".join(authors)
                )

            results.append(
                {
                    "provider": "crossref",
                    "external_id": (
                        doi
                        or f"crossref-{index}"
                    ),
                    "doi": doi or None,
                    "title": title,
                    "url": url_value,
                    "text": "\n".join(
                        text_parts
                    ),
                    "rank": index,
                }
            )

        return results

    def _search_arxiv(
        self,
        query: str,
    ) -> list[dict]:
        if self.arxiv_results == 0:
            return []

        # One query per retrieval run.
        # No paging is performed in V0.1.
        params = {
            "search_query": query,
            "start": "0",
            "max_results": str(
                self.arxiv_results
            ),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        url = (
            self.ARXIV_URL
            + "?"
            + urllib.parse.urlencode(
                params
            )
        )

        xml_bytes = self._get(url)

        root = ET.fromstring(
            xml_bytes
        )

        atom = {
            "a": (
                "http://www.w3.org/"
                "2005/Atom"
            )
        }

        results = []

        for index, entry in enumerate(
            root.findall(
                "a:entry",
                atom,
            )
        ):
            title = _normalize_title(
                entry.findtext(
                    "a:title",
                    default="",
                    namespaces=atom,
                )
            )

            summary = re.sub(
                r"\s+",
                " ",
                entry.findtext(
                    "a:summary",
                    default="",
                    namespaces=atom,
                ),
            ).strip()

            entry_id = (
                entry.findtext(
                    "a:id",
                    default="",
                    namespaces=atom,
                )
                .strip()
            )

            authors = []

            for author in entry.findall(
                "a:author",
                atom,
            ):
                name = author.findtext(
                    "a:name",
                    default="",
                    namespaces=atom,
                ).strip()

                if name:
                    authors.append(name)

            if not title:
                continue

            text_parts = [
                title,
            ]

            if summary:
                text_parts.append(
                    summary
                )

            if authors:
                text_parts.append(
                    "Authors: "
                    + ", ".join(authors)
                )

            results.append(
                {
                    "provider": "arxiv",
                    "external_id": entry_id,
                    "doi": None,
                    "title": title,
                    "url": (
                        entry_id
                        or None
                    ),
                    "text": "\n".join(
                        text_parts
                    ),
                    "rank": index,
                }
            )

        return results

    def retrieve(
        self,
        problem: ResearchProblemSpec,
    ) -> list[EvidenceCandidate]:
        query = build_web_literature_query(
            problem
        )

        arxiv_query = build_arxiv_search_query(
            problem
        )

        if not query.strip():
            return []

        raw_results = []

        provider_errors = []

        try:
            raw_results.extend(
                self._search_crossref(
                    query
                )
            )
        except WebLiteratureError as exc:
            provider_errors.append(
                str(exc)
            )

        # arXiv legacy API requests must be
        # deliberately conservative.
        try:
            if arxiv_query:
                raw_results.extend(
                    self._search_arxiv(
                        arxiv_query
                    )
                )
        except WebLiteratureError as exc:
            provider_errors.append(
                str(exc)
            )

        if (
            not raw_results
            and provider_errors
        ):
            raise WebLiteratureError(
                "All configured web literature "
                "providers failed. "
                + " | ".join(
                    provider_errors
                )
            )

        # Deduplicate across providers.
        unique = []
        seen = set()

        for item in raw_results:
            key = (
                item.get("doi")
                or _dedup_key(
                    item["title"]
                )
            )

            if not key:
                continue

            if key in seen:
                continue

            seen.add(key)
            unique.append(item)

        selected = unique[
            : self.top_k
        ]

        if not selected:
            return []

        retrieved_at = datetime.now(
            timezone.utc
        )

        candidates = []

        total = len(selected)

        for index, item in enumerate(
            selected
        ):
            provider = item[
                "provider"
            ]

            external_id = str(
                item["external_id"]
            )

            # V0.1 retrieval score is a normalized
            # provider-rank score, not a scientific
            # confidence or truth probability.
            retrieval_score = (
                1.0
                if total == 1
                else (
                    1.0
                    - index
                    / total
                )
            )

            citation_parts = [
                f"provider={provider}",
                (
                    f"id={external_id}"
                ),
            ]

            if item.get("doi"):
                citation_parts.append(
                    f"doi={item['doi']}"
                )

            candidates.append(
                EvidenceCandidate(
                    evidence_id=(
                        "WEB-"
                        + provider.upper()
                        + "-"
                        + str(index + 1)
                    ),
                    problem_id=(
                        problem.problem_id
                    ),
                    source_type=(
                        self.source_type
                    ),
                    title=item[
                        "title"
                    ],
                    source_url=item[
                        "url"
                    ],
                    citation="; ".join(
                        citation_parts
                    ),
                    text=item[
                        "text"
                    ],
                    retrieval_score=round(
                        retrieval_score,
                        6,
                    ),
                    retrieved_at=(
                        retrieved_at
                    ),
                )
            )

        return candidates