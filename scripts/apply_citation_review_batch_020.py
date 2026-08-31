from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS = ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-020"
REVIEWER = "wmy"


def load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected array in {path}")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def author(family: str, given: str, literal: str) -> dict[str, str]:
    return {"family": family, "given": given, "literal": literal, "orcid": ""}


def append_revision(revisions: list[dict], *, document_id: str, field: str,
                    old_value: Any, new_value: Any, reason: str,
                    evidence: str, reviewed_at: str) -> None:
    key = (document_id, field, BATCH_ID)
    if any((x.get("document_id"), x.get("field"), x.get("batch_id")) == key for x in revisions):
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


def add_note(record: dict, message: str, reviewed_at: str) -> None:
    notes = record.setdefault("verification_notes", [])
    if not any(x.get("message") == message for x in notes):
        notes.append({"message": message, "timestamp": reviewed_at, "source": f"human_review:{REVIEWER}"})


def main() -> int:
    records = load_jsonl(IDENTITY)
    revisions = load_jsonl(REVISIONS)
    by_id = {x["document_id"]: x for x in records}
    reviewed_at = datetime.now(timezone.utc).isoformat()
    common = {
        "identity_status": "VERIFIED",
        "citation_eligibility": "FORMALLY_CITABLE",
        "verified_by": REVIEWER,
        "verified_at": reviewed_at,
    }
    cases = [
        {
            "document_id": "DOC_8366A2D8439B",
            "updates": {
                "publication_type": "conference_paper",
                "authors": [
                    author("MALACARNE", "Sara", "Sara Malacarne"),
                    author("HOEL-HØISETH", "Eirik", "Eirik Hoel-Høiseth"),
                    author("AUNE", "Erlend", "Erlend Aune"),
                    author("ZSOLT BIRO", "David", "David Zsolt Biro"),
                    author("RUOCCO", "Massimiliano", "Massimiliano Ruocco"),
                ],
                "conference_name": "European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2026)",
                "container_title": "ESANN 2026 Proceedings",
                "volume": "", "issue": "", "pages": "299-304", "article_number": "",
                "publisher": "Ciaco", "doi": "10.14428/esann/2026.es2026-124", **common,
            },
            "reason": "Use the author order printed on the bundled PDF and upgrade the record to the verified ESANN 2026 proceedings paper.",
            "evidence": "Bundled arXiv:2604.27172 PDF, official ESANN 2026 proceedings and DOI 10.14428/esann/2026.es2026-124",
            "note": "The bundled PDF orders Erlend Aune before David Zsolt Biro; the formal citation follows that first-page order.",
        },
        {
            "document_id": "DOC_1085145489B9",
            "updates": {
                "publication_type": "conference_paper",
                "conference_name": "Annual Conference of the Prognostics and Health Management Society 2024",
                "container_title": "Proceedings of the Annual Conference of the PHM Society 2024",
                "volume": "16", "issue": "1", "pages": "", "article_number": "",
                "publisher": "PHM Society", "doi": "10.36001/phmconf.2024.v16i1.4082", **common,
            },
            "reason": "Upgrade the record to the verified PHM Society 2024 conference publication.",
            "evidence": "Bundled PDF and official PHM Society volume 16 issue 1 record with DOI 10.36001/phmconf.2024.v16i1.4082",
            "note": "The formal PDF and proceedings list Sanchari Das as the third of four authors; no page range was invented.",
        },
        {
            "document_id": "DOC_7F000353D58D",
            "updates": {
                "publication_type": "journal_article", "issued_year": 2026,
                "container_title": "IEEE Transactions on Artificial Intelligence",
                "volume": "7", "issue": "1", "pages": "195-209", "article_number": "",
                "publisher": "IEEE", "doi": "10.1109/TAI.2025.3570676", **common,
            },
            "reason": "Use the final IEEE journal volume, issue and pagination while retaining the DOI assigned at online publication.",
            "evidence": "Bundled accepted IEEE PDF and final IEEE/Crossref volume 7 issue 1 metadata for DOI 10.1109/TAI.2025.3570676",
            "note": "The paper was accepted and published online in 2025, then assigned to the 2026 volume; the preferred final citation uses 2026, 7(1):195-209.",
        },
        {
            "document_id": "DOC_64A8C0D9761F",
            "updates": {
                "publication_type": "conference_paper",
                "conference_name": "The Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25)",
                "container_title": "Proceedings of the AAAI Conference on Artificial Intelligence",
                "volume": "39", "issue": "20", "pages": "21375-21383", "article_number": "",
                "publisher": "AAAI", "doi": "10.1609/aaai.v39i20.35438", **common,
            },
            "reason": "Upgrade the preprint record to the verified AAAI-25 proceedings publication.",
            "evidence": "Bundled PDF and official AAAI volume 39 issue 20 page with DOI 10.1609/aaai.v39i20.35438",
            "note": "The preferred formal citation is the AAAI-25 proceedings paper; arXiv remains version provenance.",
        },
        {
            "document_id": "DOC_77A695A98585",
            "updates": {
                "publication_type": "preprint",
                "title": "Self-Supervised Learning for Time Series: A Review & Critique of FITS",
                "authors": [
                    author("EEFSEN", "Andreas Løvendahl", "Andreas Løvendahl Eefsen"),
                    author("LARSEN", "Nicholas Erup", "Nicholas Erup Larsen"),
                    author("HANSEN", "Oliver Glozmann Bork", "Oliver Glozmann Bork Hansen"),
                    author("AVENSTRUP", "Thor Højhus", "Thor Højhus Avenstrup"),
                ],
                "issued_year": 2024, "container_title": "", "volume": "", "issue": "",
                "pages": "", "article_number": "", "publisher": "", "place": "",
                "institution": "", "doi": "", "arxiv_id": "2410.18318", **common,
            },
            "reason": "Remove an unrelated Springer LNCS paper binding and restore the identity printed on the bundled 45-page arXiv report.",
            "evidence": "Bundled arXiv:2410.18318 PDF, arXiv identity record and DBLP CoRR record",
            "note": "DOI 10.1007/978-3-030-47426-3_39, the Jawed/Grabocka/Schmidt-Thieme authors, LNCS container and pages 499-511 belonged to an unrelated 2020 paper and were removed.",
        },
    ]
    for case in cases:
        record = by_id[case["document_id"]]
        for field, new_value in case["updates"].items():
            old_value = record.get(field, "")
            record[field] = new_value
            append_revision(revisions, document_id=case["document_id"], field=field,
                            old_value=old_value, new_value=new_value, reason=case["reason"],
                            evidence=case["evidence"], reviewed_at=reviewed_at)
        add_note(record, case["note"], reviewed_at)
    IDENTITY.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    REVISIONS.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in revisions), encoding="utf-8")
    print(json.dumps({"batch_id": BATCH_ID, "reviewer": REVIEWER}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
