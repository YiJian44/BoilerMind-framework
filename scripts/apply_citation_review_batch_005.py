from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-005"
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

    template_id = "DOC_D482A87B76DF"
    template_record = by_id[template_id]
    template_message = (
        "Human review found unresolved PVLDB template placeholders in the PDF "
        "(2020, XXX-XXX, placeholder DOI and artifact URL); these are not publication "
        "metadata and the work may only be cited as arXiv:2411.04669v1."
    )
    _add_note(template_record, template_message, reviewed_at)
    _append_revision(
        revisions,
        document_id=template_id,
        field="publication_scope",
        old_value="preprint without explicit placeholder warning",
        new_value="arXiv preprint only; ignore unresolved PVLDB template fields",
        reason="Prevent placeholder year, volume, pages and DOI from entering citations.",
        evidence="Bundled PDF first page and official arXiv:2411.04669v1 record",
        reviewed_at=reviewed_at,
    )

    author_id = "DOC_32343C3D95EA"
    author_record = by_id[author_id]
    old_authors = author_record.get("authors", [])
    new_authors = [dict(item) for item in old_authors]
    new_authors[0] = {
        "family": "FAN",
        "given": "Jin",
        "literal": "Fan Jin",
        "orcid": "",
    }
    author_record["authors"] = new_authors
    _append_revision(
        revisions,
        document_id=author_id,
        field="authors[0]",
        old_value=old_authors[0],
        new_value=new_authors[0],
        reason="Correct the first author's family/given name segmentation.",
        evidence="PDF first page, fanjin email address and official arXiv:2203.00971v1 author list",
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
