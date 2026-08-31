from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-013"
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
    revisions: list[dict], *, document_id: str, field: str,
    old_value: Any, new_value: Any, reason: str, evidence: str, reviewed_at: str,
) -> None:
    key = (document_id, field, BATCH_ID)
    if any((item.get("document_id"), item.get("field"), item.get("batch_id")) == key for item in revisions):
        return
    revisions.append({
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
    })


def _add_note(record: dict, message: str, reviewed_at: str) -> None:
    notes = record.setdefault("verification_notes", [])
    if not any(item.get("message") == message for item in notes):
        notes.append({"message": message, "timestamp": reviewed_at, "source": f"human_review:{REVIEWER}"})


def main() -> int:
    records = _load_jsonl(IDENTITY_PATH)
    revisions = _load_jsonl(REVISIONS_PATH)
    by_id = {item["document_id"]: item for item in records}
    reviewed_at = datetime.now(timezone.utc).isoformat()
    common = {
        "identity_status": "VERIFIED",
        "citation_eligibility": "FORMALLY_CITABLE",
        "verified_by": REVIEWER,
        "verified_at": reviewed_at,
    }
    cases = [
        {
            "document_id": "DOC_029897DB0EDA",
            "updates": {**common},
            "reason": "Approve the PDF-verified final IEEE journal identity.",
            "evidence": "Bundled IEEE Transactions on Industrial Informatics PDF and DOI 10.1109/TII.2019.2902129",
            "note": "The 2019 dates are manuscript acceptance and online-publication history; the final volume 16 issue 5 citation year is 2020. HTTP 202 did not invalidate the DOI.",
        },
        {
            "document_id": "DOC_61E570B4C55C",
            "updates": {
                "title": "The M4 Competition: 100,000 time series and 61 forecasting methods",
                **common,
            },
            "reason": "Replace the journal-name extraction artifact with the verified article title.",
            "evidence": "Bundled International Journal of Forecasting PDF and DOI 10.1016/j.ijforecast.2019.04.014",
            "note": "The 2019 date reflects copyright and online history; the final volume 36 issue 1 citation year is 2020.",
        },
        {
            "document_id": "DOC_CF487D800ADA",
            "updates": {
                "publication_type": "conference_paper",
                "conference_name": "2021 IEEE International Conference on Big Data (Big Data)",
                "container_title": "2021 IEEE International Conference on Big Data (Big Data)",
                "doi": "10.1109/BigData52589.2021.9671925",
                **common,
            },
            "reason": "Upgrade the bundled preprint identity to the verified IEEE Big Data 2021 conference publication.",
            "evidence": "Bundled accepted PDF, IEEE Big Data 2021 program and Crossref DOI 10.1109/BigData52589.2021.9671925",
            "note": "The preferred formal citation is the IEEE Big Data 2021 proceedings paper rather than the bundled arXiv label.",
        },
        {
            "document_id": "DOC_9F8584134FAE",
            "updates": {
                "publication_type": "conference_paper",
                "title": "Auto-Encoder Based Model for High-Dimensional Imbalanced Industrial Data",
                "conference_name": "28th International Conference on Neural Information Processing (ICONIP 2021)",
                "container_title": "Neural Information Processing",
                "volume": "1516",
                "pages": "265-273",
                "doi": "10.1007/978-3-030-92307-5_31",
                "publisher": "Springer",
                **common,
            },
            "reason": "Upgrade the bundled preprint to the verified ICONIP 2021 proceedings chapter.",
            "evidence": "Bundled paper, Springer ICONIP 2021 Part V record and Crossref DOI 10.1007/978-3-030-92307-5_31",
            "note": "Crossref classifies the item as a book chapter because it is a chapter in a conference proceedings volume; GB/T rendering uses conference-paper form [C]// with CCIS volume 1516.",
        },
        {
            "document_id": "DOC_C5962DEB1847",
            "updates": {
                "publication_type": "journal_article",
                "pages": "",
                "article_number": "107776",
                **common,
            },
            "reason": "Upgrade the bundled preprint to the verified journal publication and store its locator as an article number.",
            "evidence": "Bundled arXiv:2102.01391 PDF, ScienceDirect record and DOI 10.1016/j.asoc.2021.107776",
            "note": "107776 is an article number, not a page range; the preferred formal citation is the 2021 Applied Soft Computing publication.",
        },
    ]
    for case in cases:
        record = by_id[case["document_id"]]
        for field, new_value in case["updates"].items():
            old_value = record.get(field, "")
            record[field] = new_value
            _append_revision(
                revisions, document_id=case["document_id"], field=field,
                old_value=old_value, new_value=new_value, reason=case["reason"],
                evidence=case["evidence"], reviewed_at=reviewed_at,
            )
        _add_note(record, case["note"], reviewed_at)
    IDENTITY_PATH.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    REVISIONS_PATH.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in revisions), encoding="utf-8")
    print(json.dumps({"batch_id": BATCH_ID, "reviewer": REVIEWER}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
