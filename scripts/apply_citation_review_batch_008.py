from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-008"
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


def _append_revision(
    revisions: list[dict],
    *,
    document_id: str,
    field: str,
    old_value: Any,
    new_value: Any,
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


def main() -> int:
    records = _load_jsonl(IDENTITY_PATH)
    revisions = _load_jsonl(REVISIONS_PATH)
    by_id = {item["document_id"]: item for item in records}
    reviewed_at = datetime.now(timezone.utc).isoformat()

    uncertainty_id = "DOC_117091501969"
    uncertainty = by_id[uncertainty_id]
    old_article_number = uncertainty.get("article_number", "")
    uncertainty["article_number"] = "94"
    _append_revision(
        revisions,
        document_id=uncertainty_id,
        field="article_number",
        old_value=old_article_number,
        new_value="94",
        reason="Complete the verified journal article locator for GB/T 7714 rendering.",
        evidence=(
            "Bundled accepted manuscript citation statement, official arXiv:2209.08307v2 "
            "record and Crossref DOI 10.1007/s10462-023-10698-8"
        ),
        reviewed_at=reviewed_at,
    )

    restrictions = [
        (
            "DOC_01919EFA54B0",
            "The bundled PDF is marked 'Preprint. Under review'; cite only as "
            "arXiv:2505.00590 unless a later formal publication is independently verified.",
            "preprint without explicit under-review warning",
            "arXiv preprint only; under-review status is not a formal publication",
            "Prevent a submission status from being represented as publication evidence.",
            "Bundled arXiv:2505.00590v1 PDF and official arXiv API record",
        ),
        (
            "DOC_8047E76A3FD8",
            "The PDF contains unresolved proceedings placeholders and a conflicting ICML 2024 "
            "Subject field; these are not verified publication metadata and the work may only "
            "be cited as arXiv:2501.05809.",
            "preprint without explicit template-metadata warning",
            "arXiv preprint only; ignore proceedings placeholders and PDF Subject metadata",
            "Prevent template remnants from entering the formal citation.",
            "Bundled arXiv:2501.05809v3 PDF and official arXiv API record",
        ),
        (
            "DOC_0D91C2E3629E",
            "The official arXiv record states 'Under review'; cite only as arXiv:2601.20448 "
            "unless a later formal publication is independently verified.",
            "preprint without explicit under-review warning",
            "arXiv preprint only; under-review status is not a formal publication",
            "Prevent a submission status from being represented as publication evidence.",
            "Bundled arXiv:2601.20448v1 PDF and official arXiv API record",
        ),
    ]
    for document_id, note, old_value, new_value, reason, evidence in restrictions:
        _add_note(by_id[document_id], note, reviewed_at)
        _append_revision(
            revisions,
            document_id=document_id,
            field="publication_scope",
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            evidence=evidence,
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
