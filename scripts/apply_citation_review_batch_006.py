from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-006"
REVIEWER = "wmy"


def _load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected array in {path}")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _append_revision(
    revisions: list[dict],
    *,
    document_id: str,
    field: str,
    old_value,
    new_value,
    reason: str,
    evidence: str,
    reviewed_at: str,
) -> None:
    key = (document_id, field, BATCH_ID)
    if any(
        (item.get("document_id"), item.get("field"), item.get("batch_id")) == key
        for item in revisions
    ):
        return
    revisions.append(
        {
            "schema_version": "boilermind.literature-revision.v1",
            "batch_id": BATCH_ID,
            "document_id": document_id,
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
            "reviewer": REVIEWER,
            "reviewed_at": reviewed_at,
            "evidence": evidence,
        }
    )


def _add_note(record: dict, message: str, reviewed_at: str) -> None:
    notes = record.setdefault("verification_notes", [])
    if not any(item.get("message") == message for item in notes):
        notes.append(
            {
                "message": message,
                "timestamp": reviewed_at,
                "source": f"human_review:{REVIEWER}",
            }
        )


def main() -> int:
    records = _load_jsonl(IDENTITY_PATH)
    revisions = _load_jsonl(REVISIONS_PATH)
    by_id = {item["document_id"]: item for item in records}
    reviewed_at = datetime.now(timezone.utc).isoformat()

    deepvarma_id = "DOC_A9A47C9420DB"
    deepvarma_note = (
        "The bundled PDF does not visibly embed its arXiv identifier; title and authors "
        "were independently matched to the official arXiv:2404.17615v1 record."
    )
    _add_note(by_id[deepvarma_id], deepvarma_note, reviewed_at)
    _append_revision(
        revisions,
        document_id=deepvarma_id,
        field="identity_verification_scope",
        old_value="bundled PDF metadata only",
        new_value="bundled PDF title/authors plus official arXiv:2404.17615v1 identity",
        reason="Record how the arXiv identity was established despite no visible identifier in the PDF.",
        evidence="Bundled PDF first page and official arXiv API record",
        reviewed_at=reviewed_at,
    )

    hierarchy_id = "DOC_F7109B42A3A4"
    hierarchy_note = (
        "Cite as arXiv:2212.13706 (initial publication year 2022). The bundled PDF is "
        "v3, updated in 2025; arXiv reports DOI 10.1109/ICDMW58026.2022.00141, but "
        "complete proceedings metadata was not independently resolved in this review."
    )
    _add_note(by_id[hierarchy_id], hierarchy_note, reviewed_at)
    _append_revision(
        revisions,
        document_id=hierarchy_id,
        field="publication_scope",
        old_value="preprint without explicit version or DOI-upgrade warning",
        new_value="arXiv preprint citation only; bundled PDF is v3; formal DOI metadata pending",
        reason="Prevent incomplete conference metadata from being emitted as a formal proceedings citation.",
        evidence="Bundled arXiv:2212.13706v3 PDF and official arXiv API record",
        reviewed_at=reviewed_at,
    )

    graph_id = "DOC_0330BA6178FD"
    graph_note = (
        "The PDF states acceptance at the ICML 2026 AI for Science Workshop, but no "
        "independently verified proceedings metadata is stored; cite only as arXiv:2607.23197."
    )
    _add_note(by_id[graph_id], graph_note, reviewed_at)
    _append_revision(
        revisions,
        document_id=graph_id,
        field="publication_scope",
        old_value="preprint without explicit workshop warning",
        new_value="arXiv preprint only; workshop acceptance must not be promoted to proceedings metadata",
        reason="Keep the citation within the metadata that was actually verified.",
        evidence="Bundled arXiv:2607.23197v1 PDF",
        reviewed_at=reviewed_at,
    )

    IDENTITY_PATH.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    REVISIONS_PATH.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in revisions),
        encoding="utf-8",
    )
    print(json.dumps({"batch_id": BATCH_ID, "reviewer": REVIEWER}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
