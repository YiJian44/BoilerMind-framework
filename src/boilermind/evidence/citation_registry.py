from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMALLY_CITABLE = "FORMALLY_CITABLE"
RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
VERIFIED = "VERIFIED"


class CitationRegistryError(ValueError):
    """A record cannot safely be used as a formal citation."""


@dataclass(frozen=True)
class BindingVerification:
    valid: bool
    errors: tuple[str, ...]


def _authors(record: dict[str, Any]) -> str:
    rendered: list[str] = []
    for author in record.get("authors") or []:
        family = str(author.get("family") or "").strip()
        given = str(author.get("given") or "").strip()
        literal = str(author.get("literal") or "").strip()
        if family:
            compact_given = re.sub(r"[^A-Za-z]", "", given)
            if (
                compact_given
                and compact_given.isupper()
                and len(compact_given) <= 4
            ):
                initials = compact_given
            else:
                initials = "".join(part[0].upper() for part in given.split() if part)
            rendered.append(f"{family} {initials}".strip())
        elif literal:
            rendered.append(literal)
    if len(rendered) > 3:
        return ", ".join(rendered[:3]) + ", et al"
    return ", ".join(rendered)


def format_gbt7714_2015(record: dict[str, Any]) -> str:
    """Deterministically render the currently admitted publication types."""
    document_id = str(record.get("document_id") or "<unknown>")
    authors = _authors(record)
    title = str(record.get("title") or "").strip()
    year = int(record.get("issued_year") or 0)
    publication_type = str(record.get("publication_type") or "")
    if not authors or not title or not year:
        raise CitationRegistryError(
            f"Cannot format {document_id}: authors, title and year are required"
        )

    if publication_type == "preprint":
        parts = [authors, f"{title}[J/OL]"]
        arxiv_id = str(record.get("arxiv_id") or "").strip()
        if arxiv_id:
            parts.append(f"arXiv:{arxiv_id}")
        parts.append(str(year))
    elif publication_type == "journal_article":
        medium = str(record.get("medium") or "").strip().upper()
        resource_marker = "[J/OL]" if medium == "OL" else "[J]"
        parts = [authors, f"{title}{resource_marker}"]
        container = str(record.get("container_title") or "").strip()
        if container:
            parts.append(container)
        date = str(year)
        volume = str(record.get("volume") or "").strip()
        issue = str(record.get("issue") or "").strip()
        pages = str(record.get("pages") or record.get("article_number") or "").strip()
        if volume:
            date += f", {volume}"
        if issue:
            date += f"({issue})"
        if pages:
            date += f": {pages}"
        parts.append(date)
    elif publication_type == "conference_paper":
        conference = str(record.get("conference_name") or "").strip()
        if not conference:
            raise CitationRegistryError(
                f"Cannot format {document_id}: conference_name is required"
            )
        parts = [authors, f"{title}[C]//{conference}"]
        container = str(record.get("container_title") or "").strip()
        if container and container != conference:
            parts.append(container)
        date = str(year)
        volume = str(record.get("volume") or "").strip()
        pages = str(record.get("pages") or "").strip()
        if volume:
            date += f", {volume}"
        if pages:
            date += f": {pages}"
        parts.append(date)
    else:
        raise CitationRegistryError(
            f"Cannot format {document_id}: unsupported publication type {publication_type!r}"
        )

    doi = str(record.get("doi") or "").strip()
    if doi:
        parts.append(f"DOI:{doi}")
    return ". ".join(parts) + "."


class CitationRegistry:
    """Fail-closed registry joining identity records to the bundled PDF corpus."""

    def __init__(self, rag_root: str | Path):
        self.rag_root = Path(rag_root).resolve()
        identity_path = self.rag_root / "metadata" / "literature_identity.jsonl"
        papers_path = self.rag_root / "metadata" / "papers.jsonl"
        chunks_path = self.rag_root / "artifacts" / "chunks" / "chunks.jsonl"
        self.approvals_path = (
            self.rag_root / "audit" / "human_citation_approvals.jsonl"
        )
        for path in (identity_path, papers_path, chunks_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.records = self._load_by_id(identity_path, "document_id")
        self.papers = self._load_by_id(papers_path, "document_id")
        self.chunks = self._load_by_id(chunks_path, "chunk_id")
        self.approvals = (
            self._load_by_id(self.approvals_path, "document_id")
            if self.approvals_path.is_file()
            else {}
        )

    @staticmethod
    def _canonical_sha256(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def approval_snapshot(self, document_id: str) -> dict[str, str]:
        record = self.records.get(document_id)
        if not record:
            raise CitationRegistryError(f"Unknown document: {document_id}")
        citation = format_gbt7714_2015(record)
        return {
            "identity_record_sha256": self._canonical_sha256(record),
            "source_pdf_sha256": str(record.get("source_pdf_sha256") or "").lower(),
            "formatted_citation_sha256": hashlib.sha256(
                citation.encode("utf-8")
            ).hexdigest(),
        }

    def is_human_approved(self, document_id: str) -> bool:
        approval = self.approvals.get(document_id)
        if not approval or approval.get("decision") != "APPROVED":
            return False
        try:
            expected = self.approval_snapshot(document_id)
        except CitationRegistryError:
            return False
        return all(approval.get(key) == value for key, value in expected.items())

    @staticmethod
    def _load_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    item = json.loads(raw)
                    identifier = str(item.get(key) or "").strip()
                    if identifier:
                        result[identifier] = item
        return result

    def citation_state(
        self, document_id: str
    ) -> tuple[str, str, bool, str, str | None]:
        record = self.records.get(document_id)
        if not record:
            return "UNKNOWN", RETRIEVAL_ONLY, False, RETRIEVAL_ONLY, None
        identity = str(record.get("identity_status") or "")
        candidate_eligibility = str(
            record.get("citation_eligibility") or RETRIEVAL_ONLY
        )
        approved = self.is_human_approved(document_id)
        if (
            identity != VERIFIED
            or candidate_eligibility != FORMALLY_CITABLE
            or not approved
        ):
            return identity, candidate_eligibility, approved, RETRIEVAL_ONLY, None
        try:
            citation = self.formal_citation(document_id)
        except CitationRegistryError:
            return identity, candidate_eligibility, approved, RETRIEVAL_ONLY, None
        return identity, candidate_eligibility, approved, FORMALLY_CITABLE, citation

    def candidate_citation(
        self,
        document_id: str,
        *,
        verify_pdf_hash: bool = False,
    ) -> str:
        """Render an automated candidate; this does not grant report eligibility."""
        record = self.records.get(document_id)
        paper = self.papers.get(document_id)
        if not record or not paper:
            raise CitationRegistryError(f"Unknown document: {document_id}")
        if record.get("identity_status") != VERIFIED:
            raise CitationRegistryError(f"Identity is not VERIFIED: {document_id}")
        if record.get("citation_eligibility") != FORMALLY_CITABLE:
            raise CitationRegistryError(f"Document is RETRIEVAL_ONLY: {document_id}")
        expected = str(record.get("source_pdf_sha256") or "").lower()
        catalog_hash = str(paper.get("sha256") or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or expected != catalog_hash:
            raise CitationRegistryError(f"PDF identity hash mismatch: {document_id}")
        if verify_pdf_hash:
            source_path = (self.rag_root / str(paper.get("source_file") or "")).resolve()
            try:
                source_path.relative_to(self.rag_root)
            except ValueError as exc:
                raise CitationRegistryError("PDF path escapes RAG root") from exc
            if not source_path.is_file():
                raise CitationRegistryError(f"PDF missing: {document_id}")
            actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual != expected:
                raise CitationRegistryError(f"Bundled PDF hash mismatch: {document_id}")
        return format_gbt7714_2015(record)

    def formal_citation(self, document_id: str, *, verify_pdf_hash: bool = False) -> str:
        if not self.is_human_approved(document_id):
            raise CitationRegistryError(f"Human approval is required: {document_id}")
        return self.candidate_citation(
            document_id,
            verify_pdf_hash=verify_pdf_hash,
        )

    def verify_claim_binding(
        self,
        *,
        document_id: str,
        chunk_ids: list[str],
        page_number: int,
        supporting_excerpt: str,
    ) -> BindingVerification:
        errors: list[str] = []
        if not document_id:
            errors.append("document_id_required")
        if not chunk_ids:
            errors.append("chunk_ids_required")
        if page_number < 1:
            errors.append("page_number_required")
        excerpt = " ".join(str(supporting_excerpt or "").split())
        if not excerpt:
            errors.append("supporting_excerpt_required")
        for chunk_id in chunk_ids:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                errors.append(f"chunk_missing:{chunk_id}")
                continue
            if str(chunk.get("document_id") or "") != document_id:
                errors.append(f"document_mismatch:{chunk_id}")
            if int(chunk.get("page_number") or 0) != page_number:
                errors.append(f"page_mismatch:{chunk_id}")
            chunk_text = " ".join(str(chunk.get("text") or "").split())
            if excerpt and excerpt not in chunk_text:
                errors.append(f"excerpt_mismatch:{chunk_id}")
        return BindingVerification(valid=not errors, errors=tuple(errors))
