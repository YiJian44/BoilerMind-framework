from __future__ import annotations

import re
from typing import Protocol

from boilermind.core.contracts import (
    EvidenceCandidate,
    ResearchProblemSpec,
)


class EvidenceSource(Protocol):
    source_type: str

    def retrieve(
        self,
        problem: ResearchProblemSpec,
    ) -> list[EvidenceCandidate]:
        ...


class ScientificRetrievalError(RuntimeError):
    pass


_DOI_PATTERN = re.compile(
    r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    re.IGNORECASE,
)

_ARXIV_PATTERN = re.compile(
    r"(?:arxiv[:/\s]|abs/|[_\s])"
    r"(\d{4}\.\d{4,5})"
    r"(?:v\d+)?",
    re.IGNORECASE,
)


def _normalize_title(
    text: str,
) -> str:
    return re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+",
        "",
        str(text or "").lower(),
    )


def _candidate_identity_keys(
    candidate: EvidenceCandidate,
) -> set[str]:
    """
    Build identifiers for cross-source deduplication.

    Possible identities:
    - DOI
    - arXiv identifier
    - normalized scientific title

    This is bibliographic deduplication only.
    It is NOT scientific evidence verification.
    """

    keys: set[str] = set()

    provenance_text = " ".join(
        str(value or "")
        for value in [
            candidate.citation,
            candidate.source_url,
            candidate.source_file,
            candidate.title,
        ]
    )

    for doi in _DOI_PATTERN.findall(
        provenance_text
    ):
        keys.add(
            "doi:"
            + doi.lower().rstrip(".,;)")
        )

    for arxiv_id in _ARXIV_PATTERN.findall(
        provenance_text
    ):
        keys.add(
            "arxiv:"
            + arxiv_id.lower()
        )

    # Explicit title.
    normalized_title = _normalize_title(
        candidate.title
    )

    # Ignore generic file-like titles such as
    # "01 2407.11180".
    title_letters = sum(
        ch.isalpha()
        for ch in candidate.title
    )

    if (
        len(normalized_title) >= 20
        and title_letters >= 10
    ):
        keys.add(
            "title:"
            + normalized_title
        )

    # For page-one local chunks, the first lines
    # often contain the real paper title even when
    # papers.jsonl still has a file-like title.
    if (
        candidate.page_number in {None, 1}
        and candidate.text
    ):
        lines = [
            line.strip()
            for line in candidate.text.splitlines()
            if line.strip()
        ]

        if lines:
            first_lines = " ".join(
                lines[:2]
            )

            if len(first_lines) <= 300:
                normalized_first_lines = (
                    _normalize_title(
                        first_lines
                    )
                )

                first_line_letters = sum(
                    ch.isalpha()
                    for ch in first_lines
                )

                if (
                    len(normalized_first_lines) >= 20
                    and first_line_letters >= 10
                ):
                    keys.add(
                        "title:"
                        + normalized_first_lines
                    )

    return keys


def _representative_score(
    candidate: EvidenceCandidate,
) -> tuple:
    """
    Prefer the scientifically richer copy when the same
    publication appears through multiple sources.

    Local literature is preferred because it can carry
    exact document/page/chunk provenance and full local
    text.

    This preference is about traceability, not truth.
    """

    is_local = (
        candidate.source_type
        == "local_literature"
    )

    has_document = bool(
        candidate.document_id
    )

    has_chunk = bool(
        candidate.chunk_id
    )

    has_page = (
        candidate.page_number
        is not None
    )

    has_source_file = bool(
        candidate.source_file
    )

    return (
        int(is_local),
        int(has_document),
        int(has_chunk),
        int(has_page),
        int(has_source_file),
        len(candidate.text),
    )


def deduplicate_candidates(
    candidates: list[EvidenceCandidate],
) -> list[EvidenceCandidate]:
    """
    Cross-source bibliographic deduplication.

    Uses connected identity groups so that transitive
    matches are handled correctly, e.g.:

        Local PDF
           ↕ arXiv ID
        arXiv record
           ↕ normalized title
        Crossref DOI record

    can collapse into one publication group.
    """

    if not candidates:
        return []

    parent = list(
        range(len(candidates))
    )

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[
                parent[index]
            ]

            index = parent[index]

        return index

    def union(
        left: int,
        right: int,
    ) -> None:
        root_left = find(left)
        root_right = find(right)

        if root_left != root_right:
            parent[root_right] = root_left

    key_owner: dict[str, int] = {}

    for index, candidate in enumerate(
        candidates
    ):
        keys = _candidate_identity_keys(
            candidate
        )

        for key in keys:
            previous = key_owner.get(
                key
            )

            if previous is None:
                key_owner[key] = index
            else:
                union(
                    index,
                    previous,
                )

    groups: dict[
        int,
        list[int],
    ] = {}

    for index in range(
        len(candidates)
    ):
        root = find(index)

        groups.setdefault(
            root,
            [],
        ).append(index)

    representatives: list[
        tuple[int, EvidenceCandidate]
    ] = []

    for indexes in groups.values():
        representative_index = max(
            indexes,
            key=lambda idx: (
                _representative_score(
                    candidates[idx]
                )
            ),
        )

        first_seen_index = min(
            indexes
        )

        representatives.append(
            (
                first_seen_index,
                candidates[
                    representative_index
                ],
            )
        )

    representatives.sort(
        key=lambda item: item[0]
    )

    return [
        candidate
        for _, candidate
        in representatives
    ]


class ScientificRetrievalPipeline:
    """
    Unified scientific evidence retrieval.

    Retrieval order:
        ResearchProblemSpec
            -> configured evidence sources
            -> EvidenceCandidate pool
            -> cross-source deduplication

    Retrieved candidates are NOT verified evidence.
    """

    def __init__(
        self,
        sources: list[EvidenceSource],
    ):
        if not sources:
            raise ValueError(
                "At least one evidence source "
                "is required."
            )

        self.sources = sources

    def retrieve(
        self,
        problem: ResearchProblemSpec,
    ) -> list[EvidenceCandidate]:
        candidates: list[
            EvidenceCandidate
        ] = []

        for source in self.sources:
            try:
                source_results = (
                    source.retrieve(
                        problem
                    )
                )

            except Exception as exc:
                raise ScientificRetrievalError(
                    "Scientific evidence source "
                    f"'{getattr(source, 'source_type', type(source).__name__)}' "
                    f"failed: {exc}"
                ) from exc

            candidates.extend(
                source_results
            )

        return deduplicate_candidates(
            candidates
        )