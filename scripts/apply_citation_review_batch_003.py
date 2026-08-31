from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-003"
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
    existing = {
        (item.get("document_id"), item.get("field"), item.get("batch_id"))
        for item in revisions
    }
    if key in existing:
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


def main() -> int:
    records = _load_jsonl(IDENTITY_PATH)
    revisions = _load_jsonl(REVISIONS_PATH)
    by_id = {item["document_id"]: item for item in records}
    reviewed_at = datetime.now(timezone.utc).isoformat()

    external_id = "DOC_E3C86CE52E9E"
    external = by_id[external_id]
    locator = "https://arxiv.org/abs/2511.05594v1"
    provenance = external.setdefault("field_provenance", {})
    old_locators = {}
    for field in ("title", "authors", "year"):
        field_provenance = provenance.setdefault(field, {})
        old_locators[field] = field_provenance.get("source_locator", "")
        field_provenance["source_locator"] = locator
    message = (
        "Bundled PDF does not embed an arXiv identifier; human review matched its "
        "title, author and content to the official arXiv:2511.05594v1 record."
    )
    notes = external.setdefault("verification_notes", [])
    if not any(item.get("message") == message for item in notes):
        notes.append(
            {
                "message": message,
                "timestamp": reviewed_at,
                "source": f"human_review:{REVIEWER}",
            }
        )
    _append_revision(
        revisions,
        document_id=external_id,
        field="field_provenance.source_locator",
        old_value=old_locators,
        new_value={field: locator for field in old_locators},
        reason="Bind the PDF identity to the official arXiv record before approval.",
        evidence="Official arXiv API record 2511.05594v1 plus PDF title, author and abstract",
        reviewed_at=reviewed_at,
    )

    version_id = "DOC_DF38427FBCC7"
    version = by_id[version_id]
    version_message = (
        "Human review retained the initial arXiv publication year 2024; "
        "the bundled PDF is arXiv:2412.19950v5 dated 2026-01-22."
    )
    version_notes = version.setdefault("verification_notes", [])
    if not any(item.get("message") == version_message for item in version_notes):
        version_notes.append(
            {
                "message": version_message,
                "timestamp": reviewed_at,
                "source": f"human_review:{REVIEWER}",
            }
        )
    _append_revision(
        revisions,
        document_id=version_id,
        field="issued_year_version_scope",
        old_value="2024 without explicit bundled-PDF revision note",
        new_value="2024 initial publication; bundled PDF is v5 dated 2026-01-22",
        reason="Preserve standard arXiv first-publication year while disclosing PDF revision.",
        evidence="PDF arXiv header and PDF metadata for arXiv:2412.19950v5",
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
    print(
        json.dumps(
            {
                "batch_id": BATCH_ID,
                "updated_documents": [external_id, version_id],
                "reviewer": REVIEWER,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
