from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-014"
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
            "document_id": "DOC_4B03F2900023",
            "updates": {
                "publication_type": "journal_article",
                "title": "Deep generative models in condition and structural health monitoring: Opportunities, limitations and future outlook",
                "issued_year": 2026,
                "container_title": "Mechanical Systems and Signal Processing",
                "volume": "253",
                "article_number": "114277",
                "doi": "10.1016/j.ymssp.2026.114277",
                **common,
            },
            "reason": "Upgrade the bundled arXiv version to the verified formal journal publication.",
            "evidence": "Bundled arXiv:2507.15026 PDF, author publication record, ScienceDirect PII S0888327026004346 and DOI 10.1016/j.ymssp.2026.114277",
            "note": "The preferred formal citation is the 2026 Mechanical Systems and Signal Processing publication; arXiv:2507.15026 is retained as version provenance.",
        },
        {
            "document_id": "DOC_BA98353B889D",
            "updates": {
                "publication_type": "journal_article",
                "title": "Time Series Analysis in Compressor-Based Machines: A Survey",
                "issued_year": 2025,
                "container_title": "Neural Computing and Applications",
                "volume": "37",
                "issue": "17",
                "pages": "11001-11038",
                "doi": "10.1007/s00521-025-11065-0",
                **common,
            },
            "reason": "Upgrade the bundled arXiv version to the verified Springer journal publication.",
            "evidence": "Bundled arXiv:2402.17802 PDF, Springer publication metadata and DBLP record for DOI 10.1007/s00521-025-11065-0",
            "note": "The preferred formal citation is the 2025 Neural Computing and Applications article; arXiv:2402.17802 is retained as version provenance.",
        },
        {
            "document_id": "DOC_DE3FEDDE0070",
            "updates": {
                "title": "FTT-GRU: A Hybrid Fast Temporal Transformer with GRU for Remaining Useful Life Prediction",
                **common,
            },
            "reason": "Correct the truncated title and approve the PDF-bound arXiv identity without inventing journal metadata.",
            "evidence": "Bundled PDF first page and arXiv:2511.00564 identity record",
            "note": "No safely bindable journal DOI, volume or pages were verified; this remains a formally traceable preprint citation, not a peer-reviewed journal claim.",
        },
        {
            "document_id": "DOC_791CDB8FBADB",
            "updates": {
                "title": "An empirical evaluation of attention-based multihead deep learning models for improved remaining useful life prediction",
                **common,
            },
            "reason": "Remove the author-list extraction artifact and approve the exact title printed on the bundled PDF.",
            "evidence": "Bundled PDF first page, source SHA-256 791cdb8fbadbc8984ca382604599296371a100609268492f381d7d3f2da4b571 and arXiv:2109.01761 version record",
            "note": "The bundled PDF title differs from the current arXiv landing-page wording; this record deliberately follows the immutable local PDF to preserve source binding.",
        },
        {
            "document_id": "DOC_3798FFD52AC4",
            "updates": {
                "publication_type": "conference_paper",
                "conference_name": "2021 IEEE International Conference on Big Data (Big Data)",
                "container_title": "2021 IEEE International Conference on Big Data (Big Data)",
                "doi": "10.1109/BigData52589.2021.9671624",
                **common,
            },
            "reason": "Upgrade the bundled preprint identity to the verified IEEE Big Data 2021 conference publication.",
            "evidence": "Bundled arXiv:2109.15239 PDF, IEEE proceedings metadata and Crossref DOI 10.1109/BigData52589.2021.9671624",
            "note": "The preferred formal citation is the IEEE Big Data 2021 proceedings paper rather than the bundled arXiv label.",
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
