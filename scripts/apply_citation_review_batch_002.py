from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-002"
REVIEWER = "wmy"
DOCUMENT_ID = "DOC_11AC5EC36E6E"


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


def main() -> int:
    records = _load_jsonl(IDENTITY_PATH)
    revisions = _load_jsonl(REVISIONS_PATH)
    record = next(item for item in records if item["document_id"] == DOCUMENT_ID)
    reviewed_at = datetime.now(timezone.utc).isoformat()
    changes = {
        "publication_type": "conference_paper",
        "conference_name": (
            "Proceedings of the 4th International Conference on "
            "Mechatronics and Smart Systems"
        ),
    }
    reasons = {
        "publication_type": "PDF identifies the work as a conference proceeding, not a journal article.",
        "conference_name": "Conference name transcribed from PDF page 1.",
    }
    evidence = {
        "publication_type": "PDF page 1 proceedings header",
        "conference_name": "PDF page 1: Proceedings of the 4th International Conference on Mechatronics and Smart Systems",
    }
    existing = {
        (item.get("document_id"), item.get("field"), item.get("batch_id"))
        for item in revisions
    }
    for field, new_value in changes.items():
        old_value = record.get(field, "")
        record[field] = new_value
        key = (DOCUMENT_ID, field, BATCH_ID)
        if key not in existing:
            revisions.append(
                {
                    "schema_version": "boilermind.literature-revision.v1",
                    "batch_id": BATCH_ID,
                    "document_id": DOCUMENT_ID,
                    "field": field,
                    "old_value": old_value,
                    "new_value": new_value,
                    "reason": reasons[field],
                    "reviewer": REVIEWER,
                    "reviewed_at": reviewed_at,
                    "evidence": evidence[field],
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
    print(json.dumps({"batch_id": BATCH_ID, "document_id": DOCUMENT_ID}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
