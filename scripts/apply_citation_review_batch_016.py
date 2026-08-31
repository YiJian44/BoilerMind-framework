from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-016"
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


def _author(family: str, given: str, literal: str) -> dict[str, str]:
    return {"family": family, "given": given, "literal": literal, "orcid": ""}


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
            "document_id": "DOC_2A038B73A98E",
            "updates": {
                "title": "Data fusion and machine learning for industrial prognosis: Trends and perspectives towards Industry 4.0",
                "authors": [
                    _author("DIEZ-OLIVAN", "Alberto", "Alberto Diez-Olivan"),
                    _author("DEL SER", "Javier", "Javier Del Ser"),
                    _author("GALAR", "Diego", "Diego Galar"),
                    _author("SIERRA", "Basilio", "Basilio Sierra"),
                ],
                "issued_year": 2019,
                "container_title": "Information Fusion",
                "volume": "50",
                "issue": "",
                "pages": "92-111",
                "article_number": "",
                "publisher": "Elsevier",
                "doi": "10.1016/j.inffus.2018.10.005",
                **common,
            },
            "reason": "Replace the journal-header extraction artifact and normalize the ligature-corrupted DOI.",
            "evidence": "Bundled Information Fusion PDF metadata and first page, volume 50 pages 92-111, DOI 10.1016/j.inffus.2018.10.005",
            "note": "The DOI year reflects acceptance history; the final volume citation year is 2019. The non-ASCII ff ligature in the old DOI was replaced with ASCII 'ff'.",
        },
        {
            "document_id": "DOC_096D2BFF61A1",
            "updates": {
                "title": "Data-driven soft sensor development based on deep learning technique",
                "authors": [
                    _author("SHANG", "Chao", "Chao Shang"),
                    _author("YANG", "Fan", "Fan Yang"),
                    _author("HUANG", "Dexian", "Dexian Huang"),
                    _author("LYU", "Wenxiang", "Wenxiang Lyu"),
                ],
                "issued_year": 2014,
                "container_title": "Journal of Process Control",
                "volume": "24",
                "issue": "3",
                "pages": "223-233",
                "article_number": "",
                "publisher": "Elsevier",
                "doi": "10.1016/j.jprocont.2014.01.012",
                **common,
            },
            "reason": "Restore the complete title, authors and final journal bibliographic fields from the bundled PDF.",
            "evidence": "Bundled Journal of Process Control PDF first page, volume 24 issue 3 pages 223-233 and DOI 10.1016/j.jprocont.2014.01.012",
            "note": "The old 2013 value was the received year; the final volume citation year is 2014.",
        },
        {
            "document_id": "DOC_29D08692BCA6",
            "updates": {
                "title": "Data-driven Soft Sensors in the process industry",
                "authors": [
                    _author("KADLEC", "Petr", "Petr Kadlec"),
                    _author("GABRYS", "Bogdan", "Bogdan Gabrys"),
                    _author("STRANDT", "Sibylle", "Sibylle Strandt"),
                ],
                "issued_year": 2009,
                "container_title": "Computers & Chemical Engineering",
                "volume": "33",
                "issue": "4",
                "pages": "795-814",
                "article_number": "",
                "publisher": "Elsevier",
                "doi": "10.1016/j.compchemeng.2008.12.012",
                **common,
            },
            "reason": "Replace journal-layout fragments with the verified review article identity.",
            "evidence": "Bundled Computers & Chemical Engineering PDF and DOI 10.1016/j.compchemeng.2008.12.012",
            "note": "The old 2008 value was manuscript history; the final volume 33 issue 4 citation year is 2009.",
        },
        {
            "document_id": "DOC_396380000676",
            "updates": {
                "authors": [
                    _author("YAO", "Le", "Le Yao"),
                    _author("GE", "Zhiqiang", "Zhiqiang Ge"),
                ],
                "issued_year": 2018,
                "container_title": "IEEE Transactions on Industrial Electronics",
                "volume": "65",
                "issue": "2",
                "pages": "1490-1498",
                "article_number": "",
                "publisher": "IEEE",
                "doi": "10.1109/TIE.2017.2733448",
                **common,
            },
            "reason": "Remove title and IEEE-role author artifacts and complete the final journal fields.",
            "evidence": "Bundled IEEE Transactions on Industrial Electronics PDF, volume 65 issue 2 pages 1490-1498 and DOI 10.1109/TIE.2017.2733448",
            "note": "2017 is the online-publication and DOI-history year; the final volume 65 issue 2 citation year is 2018.",
        },
        {
            "document_id": "DOC_93D916C9FFD5",
            "updates": {
                "authors": [
                    _author("YUAN", "Xiaofeng", "Xiaofeng Yuan"),
                    _author("HUANG", "Biao", "Biao Huang"),
                    _author("WANG", "Yalin", "Yalin Wang"),
                    _author("YANG", "Chunhua", "Chunhua Yang"),
                    _author("GUI", "Weihua", "Weihua Gui"),
                ],
                "issued_year": 2018,
                "container_title": "IEEE Transactions on Industrial Informatics",
                "volume": "14",
                "issue": "7",
                "pages": "3235-3243",
                "article_number": "",
                "publisher": "IEEE",
                "doi": "10.1109/TII.2018.2809730",
                **common,
            },
            "reason": "Remove title and IEEE-role author artifacts and restore the complete five-author journal identity.",
            "evidence": "Bundled IEEE Transactions on Industrial Informatics PDF, volume 14 issue 7 pages 3235-3243 and DOI 10.1109/TII.2018.2809730",
            "note": "The title fragment and IEEE membership label were false authors; Chunhua Yang and Weihua Gui were restored from the PDF first page.",
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
