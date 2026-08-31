from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-015"
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
        "identity_status": "VERIFIED",
        "citation_eligibility": "FORMALLY_CITABLE",
        "verified_by": REVIEWER,
        "verified_at": reviewed_at,
    }
    cases = [
        {
            "document_id": "DOC_CF9238E1149A",
            "updates": {
                "publication_type": "preprint",
                "title": "A Graph Neural Networks based Framework for Topology-Aware Proactive SLA Management in a Latency Critical NFV Application Use-case",
                "authors": [
                    _author("JALODIA", "Nikita", "Nikita Jalodia"),
                    _author("TANEJA", "Mohit", "Mohit Taneja"),
                    _author("DAVY", "Alan", "Alan Davy"),
                ],
                "issued_year": 2022,
                "container_title": "",
                "volume": "",
                "issue": "",
                "pages": "",
                "article_number": "",
                "publisher": "",
                "place": "",
                "institution": "",
                "doi": "",
                "arxiv_id": "2212.00714",
                "url": "",
                **common,
            },
            "reason": "Replace a false Springer dictionary binding with the identity printed on the bundled arXiv PDF.",
            "evidence": "Bundled PDF first page, source PDF SHA-256 cf9238e1149a and arXiv:2212.00714",
            "note": "The Springer dictionary DOI and Joseph L. Cavinato identity were false matches; the IEEE ACCESS DOI text in the PDF is an unresolved template placeholder and was not retained.",
        },
        {
            "document_id": "DOC_3215DAE07BD8",
            "updates": {
                "publication_type": "conference_paper",
                "title": "Practical Skills Demand Forecasting via Representation Learning of Temporal Dynamics",
                "authors": [
                    _author("GARCIA DE MACEDO", "Maysa Malfiza", "Maysa Malfiza Garcia de Macedo"),
                    _author("CLARKE", "Wyatt", "Wyatt Clarke"),
                    _author("LUCHERINI", "Eli", "Eli Lucherini"),
                    _author("BALDWIN", "Tyler", "Tyler Baldwin"),
                    _author("QUEIROZ NETO", "Dilermando", "Dilermando Queiroz Neto"),
                    _author("DE PAULA", "Rogério Abreu", "Rogério Abreu de Paula"),
                    _author("DAS", "Subhro", "Subhro Das"),
                ],
                "issued_year": 2022,
                "conference_name": "Proceedings of the 2022 AAAI/ACM Conference on AI, Ethics, and Society",
                "container_title": "Proceedings of the 2022 AAAI/ACM Conference on AI, Ethics, and Society",
                "volume": "",
                "issue": "",
                "pages": "285-294",
                "article_number": "",
                "publisher": "ACM",
                "doi": "10.1145/3514094.3534183",
                **common,
            },
            "reason": "Replace layout-fragment metadata and upgrade the bundled preprint to the verified AIES 2022 publication.",
            "evidence": "Bundled arXiv:2205.09508 PDF, AIES 2022 program, DBLP and ACM DOI 10.1145/3514094.3534183",
            "note": "The preferred formal citation is the AIES 2022 proceedings paper; the arXiv identifier remains version provenance.",
        },
        {
            "document_id": "DOC_5B94636015EF",
            "updates": {
                "publication_type": "preprint",
                "authors": [
                    _author("YAN", "Mi", "Mi Yan"),
                    _author("MACDONALD", "Jonathan C.", "Jonathan C. MacDonald"),
                    _author("REAUME", "Chris T.", "Chris T. Reaume"),
                    _author("COBB", "Wesley", "Wesley Cobb"),
                    _author("TOTH", "Tamas", "Tamas Toth"),
                    _author("KARTHIGAN", "Sarah S.", "Sarah S. Karthigan"),
                ],
                "issued_year": 2019,
                "container_title": "",
                "volume": "",
                "issue": "",
                "pages": "",
                "article_number": "",
                "publisher": "",
                "place": "",
                "institution": "",
                "doi": "",
                "arxiv_id": "1908.11319",
                "url": "",
                **common,
            },
            "reason": "Remove organization-name author artifacts and bind the record to the verified arXiv identity.",
            "evidence": "Bundled PDF first page, official 2019 AIoT workshop program and arXiv:1908.11319",
            "note": "The workshop occurrence is real, but the PDF contains a placeholder DOI and no safely bindable proceedings locator; the approved output therefore uses the traceable arXiv form.",
        },
        {
            "document_id": "DOC_CCA9717AB54E",
            "updates": {
                "publication_type": "conference_paper",
                "authors": [
                    _author("XU", "Mengen", "Mengen Xu"),
                    _author("TANG", "Zhenhao", "Zhenhao Tang"),
                    _author("LI", "Guanglei", "Guanglei Li"),
                ],
                "issued_year": 2025,
                "conference_name": "2025 6th International Conference on Energy Power and Automation Engineering (ICEPAE)",
                "container_title": "2025 6th International Conference on Energy Power and Automation Engineering (ICEPAE)",
                "volume": "",
                "issue": "",
                "pages": "398-401",
                "article_number": "",
                "publisher": "IEEE",
                "doi": "10.1109/ICEPAE66132.2025.11275728",
                **common,
            },
            "reason": "Replace the title-fragment author artifact and complete the verified IEEE conference identity.",
            "evidence": "Bundled IEEE Xplore PDF metadata, printed pages 398-401 and DOI 10.1109/ICEPAE66132.2025.11275728",
            "note": "The PDF title fragment was incorrectly parsed as an author; the verified author list contains Mengen Xu, Zhenhao Tang and Guanglei Li only.",
        },
        {
            "document_id": "DOC_30D60A0A0449",
            "updates": {
                "publication_type": "journal_article",
                "title": "Applied machine learning: Forecasting heat load in district heating system",
                "authors": [
                    _author("IDOWU", "Samuel", "Samuel Idowu"),
                    _author("SAGUNA", "Saguna", "Saguna Saguna"),
                    _author("ÅHLUND", "Christer", "Christer Åhlund"),
                    _author("SCHELÉN", "Olov", "Olov Schelén"),
                ],
                "issued_year": 2016,
                "container_title": "Energy and Buildings",
                "volume": "133",
                "issue": "",
                "pages": "478-488",
                "article_number": "",
                "publisher": "Elsevier",
                "doi": "10.1016/j.enbuild.2016.09.068",
                **common,
            },
            "reason": "Replace the journal-header extraction artifact with the verified article identity.",
            "evidence": "Bundled Energy and Buildings PDF first page, volume 133 pages 478-488 and DOI 10.1016/j.enbuild.2016.09.068",
            "note": "The 2015 value was the manuscript received year; the final journal citation year is 2016.",
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
