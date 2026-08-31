import json
import math
import re

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from boilermind.core.contracts import (
    EvidenceCandidate,
    ResearchProblemSpec,
)
from boilermind.evidence.citation_registry import CitationRegistry


_TOKEN_PATTERN = re.compile(
    r"[a-zA-Z0-9]+|[\u4e00-\u9fff]"
)


def _tokenize(text: str) -> list[str]:
    """
    Generic lexical tokenizer.

    English words/numbers are preserved.
    Chinese text is decomposed into characters.

    No scientific question or boiler target is
    hard-coded here.
    """

    normalized = (
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    return _TOKEN_PATTERN.findall(
        normalized
    )


def build_local_literature_query(
    problem: ResearchProblemSpec,
) -> str:
    """
    Dynamically construct a literature query from the
    current user-defined ResearchProblemSpec.
    """

    parts: list[str] = [
        problem.original_question,
        problem.research_object,
        problem.target_variable,
        problem.operating_condition,
        problem.research_goal,
    ]

    parts.extend(
        problem.manipulated_variables
    )

    parts.extend(
        problem.observed_variables
    )

    parts.extend(
        problem.context_variables
    )

    return " ".join(
        part
        for part in parts
        if part
    )


class LocalRAGSource:
    """
    Real local literature retrieval source.

    Reads the BoilerMind project-bundled literature
    corpus and returns EvidenceCandidate objects.

    Current retrieval backend:
        BM25-style sparse lexical retrieval.

    The backend can later be replaced by embedding or
    hybrid retrieval without changing downstream
    scientific contracts.
    """

    source_type = "local_literature"

    def __init__(
        self,
        rag_root: str | Path | None = None,
        *,
        top_k: int = 8,
    ):
        if top_k < 1:
            raise ValueError(
                "top_k must be >= 1."
            )

        if rag_root is None:
            project_root = (
                Path(__file__)
                .resolve()
                .parents[4]
            )

            rag_root = (
                project_root
                / "resources"
                / "local_rag"
            )

        self.rag_root = Path(
            rag_root
        ).resolve()

        self.top_k = top_k

        self.chunks_path = (
            self.rag_root
            / "artifacts"
            / "chunks"
            / "chunks.jsonl"
        )

        self.papers_path = (
            self.rag_root
            / "metadata"
            / "papers.jsonl"
        )

        self.input_root = (
            self.rag_root
            / "input"
        )

        self._validate_files()

        self._papers = (
            self._load_papers()
        )

        self._citation_registry = CitationRegistry(self.rag_root)

        self._chunks = (
            self._load_chunks()
        )

        self._build_sparse_index()

    def _validate_files(self) -> None:
        if not self.chunks_path.is_file():
            raise FileNotFoundError(
                "Local RAG chunks file missing: "
                f"{self.chunks_path}"
            )

        if not self.papers_path.is_file():
            raise FileNotFoundError(
                "Local RAG metadata file missing: "
                f"{self.papers_path}"
            )

    def _load_papers(
        self,
    ) -> dict[str, dict]:
        papers: dict[str, dict] = {}

        with self.papers_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                item = json.loads(
                    line
                )

                document_id = item.get(
                    "document_id"
                )

                if document_id:
                    papers[
                        document_id
                    ] = item

        return papers

    def _load_chunks(
        self,
    ) -> list[dict]:
        chunks: list[dict] = []

        with self.chunks_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                item = json.loads(
                    line
                )

                text = str(
                    item.get(
                        "text",
                        "",
                    )
                ).strip()

                if not text:
                    continue

                chunks.append(
                    item
                )

        if not chunks:
            raise ValueError(
                "Local RAG corpus contains no chunks."
            )

        return chunks

    def _build_sparse_index(
        self,
    ) -> None:
        self._token_counts: list[
            Counter[str]
        ] = []

        self._doc_lengths: list[
            int
        ] = []

        document_frequency: Counter[
            str
        ] = Counter()

        for chunk in self._chunks:
            tokens = _tokenize(
                str(
                    chunk.get(
                        "text",
                        "",
                    )
                )
            )

            counts = Counter(
                tokens
            )

            self._token_counts.append(
                counts
            )

            self._doc_lengths.append(
                len(tokens)
            )

            document_frequency.update(
                counts.keys()
            )

        self._document_frequency = (
            document_frequency
        )

        self._average_length = (
            sum(self._doc_lengths)
            / max(
                len(self._doc_lengths),
                1,
            )
        )

    def _bm25_score(
        self,
        index: int,
        query_tokens: list[str],
    ) -> float:
        counts = self._token_counts[
            index
        ]

        document_length = (
            self._doc_lengths[index]
        )

        total_documents = len(
            self._chunks
        )

        k1 = 1.5
        b = 0.75

        score = 0.0

        for token in set(
            query_tokens
        ):
            tf = counts.get(
                token,
                0,
            )

            if tf == 0:
                continue

            df = (
                self._document_frequency
                .get(
                    token,
                    0,
                )
            )

            idf = math.log(
                1.0
                + (
                    total_documents
                    - df
                    + 0.5
                )
                / (
                    df
                    + 0.5
                )
            )

            denominator = (
                tf
                + k1
                * (
                    1.0
                    - b
                    + b
                    * document_length
                    / max(
                        self._average_length,
                        1.0,
                    )
                )
            )

            score += (
                idf
                * (
                    tf
                    * (
                        k1
                        + 1.0
                    )
                )
                / denominator
            )

        return score

    def retrieve(
        self,
        problem: ResearchProblemSpec,
    ) -> list[EvidenceCandidate]:
        query = (
            build_local_literature_query(
                problem
            )
        )

        query_tokens = _tokenize(
            query
        )

        if not query_tokens:
            return []

        raw_scores: list[
            tuple[float, int]
        ] = []

        for index in range(
            len(self._chunks)
        ):
            score = self._bm25_score(
                index,
                query_tokens,
            )

            if score > 0:
                raw_scores.append(
                    (
                        score,
                        index,
                    )
                )

        raw_scores.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        # Document-level diversity:
        # keep only the highest-ranked chunk from each
        # source document so that one paper cannot
        # dominate the complete Top-K evidence set.
        selected: list[
            tuple[float, int]
        ] = []

        selected_document_ids: set[
            str
        ] = set()

        for score, index in raw_scores:
            chunk = self._chunks[index]

            document_id = str(
                chunk.get(
                    "document_id",
                    "",
                )
            ).strip()

            # Local literature evidence must remain
            # traceable to a concrete document.
            if not document_id:
                continue

            if (
                document_id
                in selected_document_ids
            ):
                continue

            selected.append(
                (
                    score,
                    index,
                )
            )

            selected_document_ids.add(
                document_id
            )

            if len(selected) >= self.top_k:
                break

        if not selected:
            return []

        max_score = max(
            score
            for score, _
            in selected
        )

        candidates: list[
            EvidenceCandidate
        ] = []

        retrieved_at = datetime.now(
            timezone.utc
        )

        for raw_score, index in selected:
            chunk = self._chunks[
                index
            ]

            document_id = str(
                chunk.get(
                    "document_id",
                    "",
                )
            )

            paper = self._papers.get(
                document_id,
                {},
            )

            identity_record = (
                self._citation_registry.records.get(
                    document_id,
                    {},
                )
            )

            chunk_id = str(
                chunk.get(
                    "chunk_id",
                    "",
                )
            )

            page_number = chunk.get(
                "page_number"
            )

            chunk_index = chunk.get(
                "chunk_index"
            )

            corpus_level = (
                chunk.get(
                    "corpus_level"
                )
                or paper.get(
                    "corpus_level"
                )
            )

            source_file = (
                chunk.get(
                    "source_file"
                )
                or paper.get(
                    "source_file"
                )
            )

            metadata_status = (
                paper.get(
                    "metadata_status"
                )
                or chunk.get(
                    "metadata_status"
                )
            )

            (
                identity_status,
                citation_candidate_eligibility,
                human_citation_approved,
                citation_eligibility,
                formatted_citation,
            ) = (
                self._citation_registry.citation_state(document_id)
            )

            title = (
                identity_record.get(
                    "title"
                )
                or paper.get(
                    "title"
                )
                or chunk.get(
                    "title"
                )
                or document_id
                or chunk_id
            )

            document_sha256 = (
                paper.get(
                    "sha256"
                )
            )

            citation = (
                f"{document_id}; "
                f"page={page_number}; "
                f"chunk={chunk_id}"
            )

            normalized_score = (
                raw_score
                / max_score
                if max_score > 0
                else 0.0
            )

            candidates.append(
                EvidenceCandidate(
                    evidence_id=(
                        "LOCAL-"
                        f"{chunk_id}"
                    ),
                    problem_id=(
                        problem.problem_id
                    ),
                    source_type=(
                        self.source_type
                    ),
                    title=str(
                        title
                    ),
                    source_url=None,
                    citation=citation,
                    text=str(
                        chunk.get(
                            "text",
                            "",
                        )
                    ),
                    retrieval_score=round(
                        min(
                            1.0,
                            normalized_score,
                        ),
                        6,
                    ),
                    retrieved_at=(
                        retrieved_at
                    ),
                    document_id=(
                        document_id
                        or None
                    ),
                    chunk_id=(
                        chunk_id
                        or None
                    ),
                    page_number=(
                        int(page_number)
                        if page_number
                        else None
                    ),
                    chunk_index=(
                        int(chunk_index)
                        if chunk_index
                        is not None
                        else None
                    ),
                    corpus_level=(
                        str(corpus_level)
                        if corpus_level
                        else None
                    ),
                    source_file=(
                        str(source_file)
                        if source_file
                        else None
                    ),
                    metadata_status=(
                        str(metadata_status)
                        if metadata_status
                        else None
                    ),
                    identity_status=identity_status,
                    citation_candidate_eligibility=citation_candidate_eligibility,
                    human_citation_approved=human_citation_approved,
                    citation_eligibility=citation_eligibility,
                    formatted_citation=formatted_citation,
                    document_sha256=(
                        str(
                            document_sha256
                        ).lower()
                        if document_sha256
                        else None
                    ),
                )
            )

        return candidates
