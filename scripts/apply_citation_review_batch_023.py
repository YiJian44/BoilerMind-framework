from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS = ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-023"
REVIEWER = "wmy"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def add_revision(items: list[dict], document_id: str, field: str, old: Any, new: Any,
                 reason: str, evidence: str, reviewed_at: str) -> None:
    key = (document_id, field, BATCH_ID)
    for item in items:
        if (item.get("document_id"), item.get("field"), item.get("batch_id")) == key:
            item["new_value"] = new
            item["reason"] = reason
            item["reviewer"] = REVIEWER
            item["reviewed_at"] = reviewed_at
            item["evidence"] = evidence
            return
    items.append({"schema_version": "boilermind.literature-revision.v1",
                  "batch_id": BATCH_ID, "document_id": document_id, "field": field,
                  "old_value": old, "new_value": new, "reason": reason,
                  "reviewer": REVIEWER, "reviewed_at": reviewed_at, "evidence": evidence})


def add_note(record: dict, message: str, reviewed_at: str) -> None:
    notes = record.setdefault("verification_notes", [])
    if not any(x.get("message") == message for x in notes):
        notes.append({"message": message, "timestamp": reviewed_at,
                      "source": f"human_review:{REVIEWER}"})


def main() -> int:
    records, revisions = load(IDENTITY), load(REVISIONS)
    by_id = {x["document_id"]: x for x in records}
    reviewed_at = datetime.now(timezone.utc).isoformat()
    common = {"identity_status": "VERIFIED", "citation_eligibility": "FORMALLY_CITABLE",
              "verified_by": REVIEWER, "verified_at": reviewed_at, "medium": ""}
    cases = [
        ("DOC_5C553DE8575E", {
            "publication_type": "journal_article",
            "title": "A Review on Basic Data-Driven Approaches for Industrial Process Monitoring",
            "authors": [{"family": "Yin", "given": "Shen"},
                        {"family": "Ding", "given": "Steven X."},
                        {"family": "Xie", "given": "Xiaochen"},
                        {"family": "Luo", "given": "Hao"}],
            "issued_year": 2014, "container_title": "IEEE Transactions on Industrial Electronics",
            "volume": "61", "issue": "11", "pages": "6418-6428", "article_number": "",
            "publisher": "IEEE", "doi": "10.1109/TIE.2014.2301773", **common},
         "Replace the IEEE journal header mistakenly extracted as the title and confirm the final bibliographic identity.",
         "Bundled final IEEE PDF showing title, authors, volume 61 issue 11 pages 6418-6428 and DOI",
         "The formal citation uses the article title printed on the PDF, not the running journal header."),
        ("DOC_355EF0C11043", {
            "publication_type": "journal_article",
            "title": "A Survey on Deep Learning for Data-Driven Soft Sensors",
            "authors": [{"family": "Sun", "given": "Qingqiang"},
                        {"family": "Ge", "given": "Zhiqiang"}],
            "issued_year": 2021, "container_title": "IEEE Transactions on Industrial Informatics",
            "volume": "17", "issue": "9", "pages": "5853-5866", "article_number": "",
            "publisher": "IEEE", "doi": "10.1109/TII.2021.3053128", **common},
         "Confirm the final IEEE journal identity against the bundled publication PDF.",
         "Bundled final IEEE PDF showing title, authors, volume 17 issue 9 pages 5853-5866 and DOI",
         "The PDF and formal publication metadata agree; the transient DOI HTTP status is not used as evidence against the verified identity."),
        ("DOC_EA91B12E0D5B", {
            "publication_type": "journal_article",
            "title": "DeepAR: Probabilistic forecasting with autoregressive recurrent networks",
            "authors": [{"family": "Salinas", "given": "David"},
                        {"family": "Flunkert", "given": "Valentin"},
                        {"family": "Gasthaus", "given": "Jan"},
                        {"family": "Januschowski", "given": "Tim"}],
            "issued_year": 2020, "container_title": "International Journal of Forecasting",
            "volume": "36", "issue": "3", "pages": "1181-1191", "article_number": "",
            "publisher": "Elsevier", "doi": "10.1016/j.ijforecast.2019.07.001", **common},
         "Replace the journal-name header mistakenly extracted as the title and confirm the final volume record.",
         "Bundled final PDF and official International Journal of Forecasting volume 36 issue 3 record",
         "The DOI was registered in 2019, while the preferred final citation uses the 2020 volume assignment."),
        ("DOC_64ACB65E4087", {
            "publication_type": "journal_article",
            "title": "Learning in Nonstationary Environments: A Survey",
            "authors": [{"family": "Ditzler", "given": "Gregory"},
                        {"family": "Roveri", "given": "Manuel"},
                        {"family": "Alippi", "given": "Cesare"},
                        {"family": "Polikar", "given": "Robi"}],
            "issued_year": 2015, "container_title": "IEEE Computational Intelligence Magazine",
            "volume": "10", "issue": "4", "pages": "12-25", "article_number": "",
            "publisher": "IEEE", "doi": "10.1109/MCI.2015.2471196", **common},
         "Replace the first-body-paragraph extraction with the article title printed on the PDF.",
         "Bundled final IEEE PDF showing title, authors, volume 10 issue 4 pages 12-25 and DOI",
         "No article number is added because volume, issue and page range are sufficient and directly verified."),
    ]
    for document_id, updates, reason, evidence, message in cases:
        record = by_id[document_id]
        for field, new in updates.items():
            old = record.get(field, "")
            record[field] = new
            add_revision(revisions, document_id, field, old, new, reason, evidence, reviewed_at)
        add_note(record, message, reviewed_at)
    IDENTITY.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    REVISIONS.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in revisions), encoding="utf-8")
    print(json.dumps({"batch_id": BATCH_ID, "reviewer": REVIEWER}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
