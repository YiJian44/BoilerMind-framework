from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from boilermind.evidence.citation_registry import (  # noqa: E402
    CitationRegistry,
    CitationRegistryError,
    FORMALLY_CITABLE,
)


RAG_ROOT = PROJECT_ROOT / "resources" / "local_rag"
CONFIRMATION = "VERIFIED_METADATA_AGAINST_PDF"


def _configure_windows_utf8_output() -> None:
    if sys.platform != "win32":
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="strict")


def _registry() -> CitationRegistry:
    return CitationRegistry(RAG_ROOT)


def list_candidates(registry: CitationRegistry) -> int:
    rows = []
    for document_id, record in registry.records.items():
        if record.get("citation_eligibility") != FORMALLY_CITABLE:
            continue
        rows.append(
            {
                "document_id": document_id,
                "human_approved": registry.is_human_approved(document_id),
                "title": record.get("title"),
                "year": record.get("issued_year"),
                "doi": record.get("doi"),
                "arxiv_id": record.get("arxiv_id"),
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def show_candidate(registry: CitationRegistry, document_id: str) -> int:
    record = registry.records.get(document_id)
    paper = registry.papers.get(document_id)
    if not record or not paper:
        raise CitationRegistryError(f"Unknown document: {document_id}")
    payload = {
        "document_id": document_id,
        "title": record.get("title"),
        "authors": record.get("authors"),
        "issued_year": record.get("issued_year"),
        "publication_type": record.get("publication_type"),
        "container_title": record.get("container_title"),
        "volume": record.get("volume"),
        "issue": record.get("issue"),
        "pages": record.get("pages"),
        "article_number": record.get("article_number"),
        "doi": record.get("doi"),
        "arxiv_id": record.get("arxiv_id"),
        "pdf": str(RAG_ROOT / str(paper.get("source_file") or "")),
        "pdf_sha256": record.get("source_pdf_sha256"),
        "candidate_gbt7714_2015": registry.candidate_citation(
            document_id,
            verify_pdf_hash=True,
        ),
        "approval_snapshot": registry.approval_snapshot(document_id),
        "human_approved": registry.is_human_approved(document_id),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def approve_candidate(
    registry: CitationRegistry,
    document_id: str,
    reviewer: str,
    confirmation: str,
) -> int:
    if confirmation != CONFIRMATION:
        raise CitationRegistryError(
            f"Approval requires --confirm {CONFIRMATION}"
        )
    if not reviewer.strip():
        raise CitationRegistryError("A non-empty reviewer is required")
    citation = registry.candidate_citation(document_id, verify_pdf_hash=True)
    record = registry.records[document_id]
    if record.get("citation_eligibility") != FORMALLY_CITABLE:
        raise CitationRegistryError("Only automated formal candidates can be reviewed")
    approval = {
        "schema_version": "boilermind.citation-approval.v1",
        "document_id": document_id,
        "decision": "APPROVED",
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "confirmation": confirmation,
        "formatted_citation": citation,
        **registry.approval_snapshot(document_id),
    }
    registry.approvals_path.parent.mkdir(parents=True, exist_ok=True)
    with registry.approvals_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(approval, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(approval, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    _configure_windows_utf8_output()
    parser = argparse.ArgumentParser(description="Review one real citation candidate")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    show = sub.add_parser("show")
    show.add_argument("--document-id", required=True)
    approve = sub.add_parser("approve")
    approve.add_argument("--document-id", required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--confirm", required=True)
    args = parser.parse_args()
    registry = _registry()
    if args.command == "list":
        return list_candidates(registry)
    if args.command == "show":
        return show_candidate(registry, args.document_id)
    return approve_candidate(
        registry,
        args.document_id,
        args.reviewer,
        args.confirm,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CitationRegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
