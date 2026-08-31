from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-004"
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
    *, document_id: str,
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

    version_changes = [
        (
            "DOC_75F99FE3DFD0",
            "2024 without explicit bundled-PDF revision note",
            "2024 initial publication; bundled PDF is v2 dated 2025-11-15",
            "Human review retained the initial arXiv publication year 2024; the bundled PDF is arXiv:2410.20166v2 dated 2025-11-15.",
        ),
        (
            "DOC_908E25C6EC69",
            "2024 without explicit bundled-PDF revision note",
            "2024 initial publication; bundled PDF is v3 dated 2025-06-15",
            "Human review retained the initial arXiv publication year 2024; the bundled PDF is arXiv:2407.16739v3 dated 2025-06-15.",
        ),
    ]
    for document_id, old_value, new_value, message in version_changes:
        _add_note(by_id[document_id], message, reviewed_at)
        _append_revision(
            revisions,
            document_id=document_id,
            field="issued_year_version_scope",
            old_value=old_value,
            new_value=new_value,
            reason="Preserve standard arXiv first-publication year while disclosing PDF revision.",
            evidence="Bundled PDF arXiv version header and PDF metadata",
            reviewed_at=reviewed_at,
        )

    external_id = "DOC_44FB6FD57CC4"
    external = by_id[external_id]
    locator = "https://arxiv.org/abs/2605.21903v1"
    provenance = external.setdefault("field_provenance", {})
    old_locators = {}
    for field in ("title", "authors", "year"):
        item = provenance.setdefault(field, {})
        old_locators[field] = item.get("source_locator", "")
        item["source_locator"] = locator
    _add_note(
        external,
        "Bundled PDF does not embed its own arXiv identifier; human review matched title, author and content to the official arXiv:2605.21903v1 record.",
        reviewed_at,
    )
    _append_revision(
        revisions,
        document_id=external_id,
        field="field_provenance.source_locator",
        old_value=old_locators,
        new_value={field: locator for field in old_locators},
        reason="Bind the PDF identity to the official arXiv record before approval.",
        evidence="Official arXiv API record 2605.21903v1 plus PDF title, author and abstract",
        reviewed_at=reviewed_at,
    )

    author_changes = {
        "DOC_ACE0C417331D": [
            {"family": "DESAI", "given": "AS", "literal": "AS Desai", "orcid": ""},
            {"family": "NAVANEETH", "given": "N", "literal": "Navaneeth N", "orcid": ""},
            {"family": "ADHIKARI", "given": "Sondipon", "literal": "Sondipon Adhikari", "orcid": ""},
            {"family": "CHAKRABORTY", "given": "Souvik", "literal": "Souvik Chakraborty", "orcid": ""},
        ],
        "DOC_71C6740B18CC": [
            {"family": "ABBOUCHI", "given": "Omar", "literal": "Omar Abbouchi", "orcid": ""},
            {"family": "DAVILA", "given": "Sofia", "literal": "Sofia Davila", "orcid": ""},
            {"family": "AL HASANI", "given": "Meena", "literal": "Meena Al Hasani", "orcid": ""},
            {"family": "LE", "given": "Jessica", "literal": "Jessica Le", "orcid": ""},
            {"family": "NELSON-ARCHER", "given": "Adam", "literal": "Adam Nelson-Archer", "orcid": ""},
            {"family": "SEN", "given": "Aleia", "literal": "Aleia Sen", "orcid": ""},
        ],
    }
    for document_id, new_authors in author_changes.items():
        record = by_id[document_id]
        old_authors = record.get("authors", [])
        record["authors"] = new_authors
        _append_revision(
            revisions,
            document_id=document_id,
            field="authors",
            old_value=old_authors,
            new_value=new_authors,
            reason="Correct author name segmentation and order before GB/T rendering.",
            evidence="Bundled PDF first page and official arXiv author list",
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
