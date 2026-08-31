from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-009"
REVIEWER = "wmy"
DOCUMENT_ID = "DOC_3D7FC5B57857"


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
    field: str,
    old_value: Any,
    new_value: Any,
    reason: str,
    evidence: str,
    reviewed_at: str,
) -> None:
    key = (DOCUMENT_ID, field, BATCH_ID)
    if any(
        (item.get("document_id"), item.get("field"), item.get("batch_id")) == key
        for item in revisions
    ):
        return
    revisions.append(
        {
            "schema_version": "boilermind.literature-revision.v1",
            "batch_id": BATCH_ID,
            "document_id": DOCUMENT_ID,
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
    record = by_id[DOCUMENT_ID]
    reviewed_at = datetime.now(timezone.utc).isoformat()
    evidence = (
        "Bundled arXiv:1809.03561v1 PDF, Elsevier publication record and Crossref DOI "
        "10.1016/j.ijforecast.2018.07.004"
    )
    updates = {
        "publication_type": "journal_article",
        "title": "Quantile regression for the qualifying match of GEFCom2017 probabilistic load forecasting",
        "issued_year": 2019,
        "container_title": "International Journal of Forecasting",
        "volume": "35",
        "issue": "4",
        "pages": "1400-1408",
        "doi": "10.1016/j.ijforecast.2018.07.004",
    }
    for field, new_value in updates.items():
        old_value = record.get(field, "")
        record[field] = new_value
        _append_revision(
            revisions,
            field=field,
            old_value=old_value,
            new_value=new_value,
            reason="Upgrade the bundled preprint identity to the independently verified journal publication.",
            evidence=evidence,
            reviewed_at=reviewed_at,
        )

    _add_note(
        record,
        "The bundled PDF is the 2018 arXiv preprint; the preferred formal citation is the 2019 "
        "International Journal of Forecasting version. Crossref also links erratum DOI "
        "10.1016/j.ijforecast.2021.01.010; consult it when citing corrected details.",
        reviewed_at,
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
