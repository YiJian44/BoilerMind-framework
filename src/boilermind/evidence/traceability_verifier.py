import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from boilermind.core.contracts import (
    EvidenceCandidate,
)


_DOI_PATTERN = re.compile(
    r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    re.IGNORECASE,
)

_ARXIV_PATTERN = re.compile(
    r"(?:arxiv.org/abs/|arxiv:)"
    r"(\d{4}\.\d{4,5})"
    r"(?:v\d+)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TraceabilityResult:
    verified: bool
    rationale: str
    source_hash: str | None = None


class EvidenceTraceabilityVerifier:
    """
    Deterministic source/provenance verification.

    This verifier checks whether an EvidenceCandidate
    can be traced back to a concrete scientific source.

    It does NOT judge:
    - whether the paper is scientifically correct;
    - whether it supports the current hypothesis;
    - whether it is semantically relevant.

    Those responsibilities belong to later verification.
    """

    def __init__(
        self,
        local_rag_root: str | Path | None = None,
    ):
        if local_rag_root is None:
            project_root = (
                Path(__file__)
                .resolve()
                .parents[3]
            )

            local_rag_root = (
                project_root
                / "resources"
                / "local_rag"
            )

        self.local_rag_root = Path(
            local_rag_root
        ).resolve()

    def verify(
        self,
        candidate: EvidenceCandidate,
    ) -> TraceabilityResult:
        if (
            candidate.source_type
            == "local_literature"
        ):
            return self._verify_local(
                candidate
            )

        if (
            candidate.source_type
            == "web_literature"
        ):
            return self._verify_web(
                candidate
            )

        return TraceabilityResult(
            verified=False,
            rationale=(
                "Unsupported evidence source type: "
                f"{candidate.source_type}"
            ),
        )

    def _verify_local(
        self,
        candidate: EvidenceCandidate,
    ) -> TraceabilityResult:
        missing = []

        if not candidate.document_id:
            missing.append("document_id")

        if not candidate.chunk_id:
            missing.append("chunk_id")

        if candidate.page_number is None:
            missing.append("page_number")

        if not candidate.source_file:
            missing.append("source_file")

        if missing:
            return TraceabilityResult(
                verified=False,
                rationale=(
                    "Local literature provenance "
                    "is incomplete: "
                    + ", ".join(missing)
                ),
            )

        source_path = (
            self.local_rag_root
            / candidate.source_file
        ).resolve()

        try:
            source_path.relative_to(
                self.local_rag_root
            )
        except ValueError:
            return TraceabilityResult(
                verified=False,
                rationale=(
                    "Local source path escapes "
                    "the configured RAG root."
                ),
            )

        if not source_path.is_file():
            return TraceabilityResult(
                verified=False,
                rationale=(
                    "Referenced local source file "
                    f"does not exist: {source_path}"
                ),
            )

        if (
            candidate.document_sha256
            and not re.fullmatch(
                r"[0-9a-f]{64}",
                candidate.document_sha256,
            )
        ):
            return TraceabilityResult(
                verified=False,
                rationale=(
                    "Invalid local document SHA256."
                ),
            )

        if not candidate.document_sha256:
            return TraceabilityResult(
                verified=False,
                rationale=(
                    "Local literature is missing the expected PDF SHA256."
                ),
            )

        actual_sha256 = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()

        if actual_sha256 != candidate.document_sha256:
            return TraceabilityResult(
                verified=False,
                rationale=(
                    "Local PDF content does not match the catalogued SHA256."
                ),
            )

        return TraceabilityResult(
            verified=True,
            rationale=(
                "Local evidence is traceable to "
                "a project-bundled PDF with "
                "document/chunk/page provenance."
            ),
            source_hash=(
                actual_sha256
            ),
        )

    def _verify_web(
        self,
        candidate: EvidenceCandidate,
    ) -> TraceabilityResult:
        provenance = " ".join(
            [
                str(candidate.citation or ""),
                str(candidate.source_url or ""),
            ]
        )

        has_doi = bool(
            _DOI_PATTERN.search(
                provenance
            )
        )

        has_arxiv = bool(
            _ARXIV_PATTERN.search(
                provenance
            )
        )

        if not (
            has_doi
            or has_arxiv
        ):
            return TraceabilityResult(
                verified=False,
                rationale=(
                    "Web literature lacks a "
                    "traceable DOI or arXiv identifier."
                ),
            )

        if not candidate.title.strip():
            return TraceabilityResult(
                verified=False,
                rationale=(
                    "Web literature title is missing."
                ),
            )

        return TraceabilityResult(
            verified=True,
            rationale=(
                "Web literature is traceable through "
                + (
                    "DOI."
                    if has_doi
                    else "arXiv identifier."
                )
            ),
        )
