from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = (
    PROJECT_ROOT / "resources" / "local_rag" / "metadata" / "literature_identity.jsonl"
)
REVISIONS_PATH = (
    PROJECT_ROOT / "resources" / "local_rag" / "metadata" / "literature_revisions.jsonl"
)
REVIEWER = "wmy"
BATCH_ID = "CITATION-REVIEW-BATCH-001"


def _load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # The teammate delivery named this file .jsonl but stored the
    # original revision history as one JSON array. Accept it once,
    # then rewrite the preserved records as actual JSONL below.
    if text.startswith("["):
        loaded = json.loads(text)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected revision array in {path}")
        return loaded
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    records = _load_jsonl(IDENTITY_PATH)
    by_id = {item["document_id"]: item for item in records}
    reviewed_at = datetime.now(timezone.utc).isoformat()
    revisions = _load_jsonl(REVISIONS_PATH)
    existing_keys = {
        (item.get("document_id"), item.get("field"), item.get("batch_id"))
        for item in revisions
    }

    changes = [
        {
            "document_id": "DOC_7DBB144800E8",
            "field": "article_number",
            "new_value": "12252",
            "reason": "PDF first page states Scientific Reports (2026) 16:12252.",
            "evidence": "PDF page 1",
        },
        {
            "document_id": "DOC_DEA87EF9FB81",
            "field": "issued_year",
            "new_value": 2010,
            "reason": "Use the formal journal volume year, not the 2009 online publication date.",
            "evidence": "PDF page 1: Neural Comput & Applic (2010) 19:725-740",
        },
    ]

    for change in changes:
        record = by_id[change["document_id"]]
        old_value = record.get(change["field"])
        record[change["field"]] = change["new_value"]
        key = (change["document_id"], change["field"], BATCH_ID)
        if key not in existing_keys:
            revisions.append(
                {
                    "schema_version": "boilermind.literature-revision.v1",
                    "batch_id": BATCH_ID,
                    "document_id": change["document_id"],
                    "field": change["field"],
                    "old_value": old_value,
                    "new_value": change["new_value"],
                    "reason": change["reason"],
                    "reviewer": REVIEWER,
                    "reviewed_at": reviewed_at,
                    "evidence": change["evidence"],
                }
            )

    version_record = by_id["DOC_23EEDF435C23"]
    version_message = (
        "Human review retained the initial arXiv publication year 2020; "
        "the bundled PDF is arXiv:2007.15433v2 dated 2022-10-26/27."
    )
    notes = version_record.setdefault("verification_notes", [])
    if not any(item.get("message") == version_message for item in notes):
        notes.append(
            {
                "message": version_message,
                "timestamp": reviewed_at,
                "source": f"human_review:{REVIEWER}",
            }
        )
    version_key = ("DOC_23EEDF435C23", "issued_year_version_scope", BATCH_ID)
    if version_key not in existing_keys:
        revisions.append(
            {
                "schema_version": "boilermind.literature-revision.v1",
                "batch_id": BATCH_ID,
                "document_id": "DOC_23EEDF435C23",
                "field": "issued_year_version_scope",
                "old_value": "2020 without explicit bundled-PDF revision note",
                "new_value": "2020 initial publication; bundled PDF is v2 dated 2022",
                "reason": "Preserve standard arXiv first-publication year while disclosing PDF revision.",
                "reviewer": REVIEWER,
                "reviewed_at": reviewed_at,
                "evidence": "PDF title page and arXiv:2007.15433v2 header",
            }
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
                "reviewer": REVIEWER,
                "updated_documents": [
                    "DOC_7DBB144800E8",
                    "DOC_23EEDF435C23",
                    "DOC_DEA87EF9FB81",
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
