from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-017"
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


def _author(family: str, given: str, literal: str, orcid: str = "") -> dict[str, str]:
    return {"family": family, "given": given, "literal": literal, "orcid": orcid}


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
        "publication_type": "journal_article",
        "identity_status": "VERIFIED",
        "citation_eligibility": "FORMALLY_CITABLE",
        "verified_by": REVIEWER,
        "verified_at": reviewed_at,
    }
    cases = [
        {
            "document_id": "DOC_C3E39E98B583",
            "updates": {
                "authors": [
                    _author("LU", "Jie", "Jie Lu"),
                    _author("LIU", "Anjin", "Anjin Liu"),
                    _author("DONG", "Fan", "Fan Dong"),
                    _author("GU", "Feng", "Feng Gu"),
                    _author("GAMA", "João", "João Gama"),
                    _author("ZHANG", "Guangquan", "Guangquan Zhang"),
                ],
                "issued_year": 2019,
                "container_title": "IEEE Transactions on Knowledge and Data Engineering",
                "volume": "31",
                "issue": "12",
                "pages": "2346-2363",
                "article_number": "",
                "publisher": "IEEE",
                "doi": "10.1109/TKDE.2018.2876857",
                **common,
            },
            "reason": "Remove IEEE-role and truncated-name artifacts and replace an address-derived year with the final journal year.",
            "evidence": "Bundled IEEE PDF, volume 31 issue 12 pages 2346-2363 and DOI 10.1109/TKDE.2018.2876857",
            "note": "The old year 2007 came from the Sydney NSW 2007 postal address; the final journal citation year is 2019. João Gama was restored from the PDF author line.",
        },
        {
            "document_id": "DOC_2D572537D4B0",
            "updates": {
                "title": "Recurrent Neural Networks for Time Series Forecasting: Current status and future directions",
                "container_title": "International Journal of Forecasting",
                **common,
            },
            "reason": "Replace the journal-name extraction artifact with the verified article title.",
            "evidence": "Bundled International Journal of Forecasting PDF, volume 37 issue 1 pages 388-427 and DOI 10.1016/j.ijforecast.2020.06.008",
            "note": "The 2020 date is copyright and DOI history; the final volume 37 issue 1 citation year remains 2021.",
        },
        {
            "document_id": "DOC_168B7B85DE87",
            "updates": {
                "title": "Review of adaptation mechanisms for data-driven soft sensors",
                "container_title": "Computers & Chemical Engineering",
                **common,
            },
            "reason": "Replace the journal-name extraction artifact and decode the HTML entity in the container title.",
            "evidence": "Bundled Computers & Chemical Engineering PDF, volume 35 issue 1 pages 1-24 and DOI 10.1016/j.compchemeng.2010.07.034",
            "note": "The preferred container title uses a literal ampersand rather than the stored HTML entity &amp;.",
        },
        {
            "document_id": "DOC_3BFE4BA68B67",
            "updates": {
                "title": "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting",
                "container_title": "International Journal of Forecasting",
                **common,
            },
            "reason": "Replace the journal-name extraction artifact with the verified article title.",
            "evidence": "Bundled International Journal of Forecasting PDF, volume 37 issue 4 pages 1748-1764 and DOI 10.1016/j.ijforecast.2021.03.012",
            "note": "Authors, year, volume, issue, pages and DOI were already consistent with the final publication.",
        },
        {
            "document_id": "DOC_E9FB1DAB8A6C",
            "updates": {
                "pages": "",
                "article_number": "20200209",
                **common,
            },
            "reason": "Move the Royal Society article locator from the page field to the article-number field.",
            "evidence": "Bundled Philosophical Transactions of the Royal Society A PDF and DOI 10.1098/rsta.2020.0209",
            "note": "20200209 is an article number, not a page range; the final volume 379 issue 2194 citation year is 2021.",
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
