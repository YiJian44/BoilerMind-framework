from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-012"
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
            "document_id": "DOC_45A8C453BB2D",
            "updates": {
                "publication_type": "journal_article",
                "title": "Physics-Enhanced Graph Neural Networks for Soft Sensing in Industrial Internet of Things",
                "container_title": "IEEE Internet of Things Journal",
                "volume": "11",
                "issue": "21",
                "pages": "34978-34990",
                "doi": "10.1109/JIOT.2024.3434732",
                **common,
            },
            "reason": "Replace the truncated preprint title with the verified journal publication.",
            "evidence": "Bundled arXiv:2404.08061 PDF, METAS publication list and DOI 10.1109/JIOT.2024.3434732",
            "note": "One METAS listing labels the venue as IEEE Transactions on Industrial Informatics, but the DOI code, volume/pages and independent citations identify IEEE Internet of Things Journal; the citation follows the DOI-consistent journal identity.",
        },
        {
            "document_id": "DOC_DE9786DFB04F",
            "updates": {
                "title": "LightGCNet: A Lightweight Geometric Constructive Neural Network for Data-Driven Soft Sensors",
                **common,
            },
            "reason": "Replace a journal-template running header with the verified arXiv title while preserving preprint status.",
            "evidence": "Bundled arXiv:2312.12022 PDF and official arXiv record",
            "note": "The IEEE Transactions on Automation Science and Engineering running header is submission-template text, not proof of a formal journal publication; cite only as arXiv:2312.12022 until a volume, issue, pages and DOI are independently verified.",
        },
        {
            "document_id": "DOC_7A9117860242",
            "updates": {
                "publication_type": "conference_paper",
                "conference_name": "2021 IEEE International Conference on Big Data (Big Data)",
                "container_title": "2021 IEEE International Conference on Big Data (Big Data)",
                "doi": "10.1109/BigData52589.2021.9671991",
                **common,
            },
            "reason": "Complete the verified IEEE Big Data 2021 conference identity.",
            "evidence": "Bundled accepted conference PDF, IEEE Big Data 2021 program and DOI 10.1109/BigData52589.2021.9671991",
            "note": "The prior DOI-resolution failure was not evidence that the DOI was invalid; the paper and conference program independently confirm the conference identity.",
        },
        {
            "document_id": "DOC_597D3CB19356",
            "updates": {
                "authors": [
                    {"family": "LAHAT", "given": "Dana", "literal": "Dana Lahat", "orcid": ""},
                    {"family": "ADALI", "given": "Tülay", "literal": "Tülay Adalı", "orcid": ""},
                    {"family": "JUTTEN", "given": "Christian", "literal": "Christian Jutten", "orcid": ""},
                ],
                **common,
            },
            "reason": "Approve the PDF-verified journal identity and restore author-name diacritics.",
            "evidence": "Bundled Proceedings of the IEEE PDF and DOI 10.1109/JPROC.2015.2460697",
            "note": "The earlier HTTP 202 response did not invalidate the DOI; title, authors, volume, issue, pages and DOI are printed in the bundled journal PDF.",
        },
        {
            "document_id": "DOC_B78DD0DCD1CB",
            "updates": {
                "title": "Multisensor data fusion: A review of the state-of-the-art",
                **common,
            },
            "reason": "Replace the extraction-truncated title with the verified final journal title.",
            "evidence": "Bundled Information Fusion PDF, ScienceDirect record and DOI 10.1016/j.inffus.2011.08.001",
            "note": "The PDF's 2010 and 2011 dates are manuscript and online-history dates; the final volume 14 issue 1 publication year used for citation is 2013.",
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
