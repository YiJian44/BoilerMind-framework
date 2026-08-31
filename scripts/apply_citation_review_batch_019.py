from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-019"
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


def _append_revision(revisions: list[dict], *, document_id: str, field: str,
                     old_value: Any, new_value: Any, reason: str,
                     evidence: str, reviewed_at: str) -> None:
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
            "document_id": "DOC_823DEB7EA512",
            "updates": {
                "publication_type": "preprint",
                "title": "CALM: A Framework for Continuous, Adaptive, and LLM-Mediated Anomaly Detection in Time-Series Streams",
                "container_title": "", "volume": "", "issue": "", "pages": "",
                "article_number": "", "publisher": "", "doi": "",
                "arxiv_id": "2508.21273", **common,
            },
            "reason": "Restore the complete title printed on the bundled PDF and retain the work as an arXiv preprint.",
            "evidence": "Bundled arXiv:2508.21273 PDF and arXiv identity record",
            "note": "No verified formal journal or conference publication was found; the citable identity is the 2025 arXiv preprint.",
        },
        {
            "document_id": "DOC_264819ED4BD8",
            "updates": {
                "publication_type": "conference_paper",
                "conference_name": "The Eighteenth ACM International Conference on Web Search and Data Mining (WSDM 2025)",
                "container_title": "Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining",
                "volume": "", "issue": "", "pages": "98-106", "article_number": "",
                "publisher": "ACM", "doi": "10.1145/3701551.3703494", **common,
            },
            "reason": "Upgrade the record to the verified WSDM 2025 conference publication.",
            "evidence": "Bundled PDF, official WSDM 2025 accepted-paper list and DOI 10.1145/3701551.3703494",
            "note": "The preferred formal citation is the WSDM 2025 proceedings paper; arXiv remains version provenance.",
        },
        {
            "document_id": "DOC_3BFB0FAD4EC2",
            "updates": {
                "publication_type": "preprint",
                "title": "PLanTS: Periodicity-aware Latent-state Representation Learning for Multivariate Time Series",
                "authors": [_author("WANG", "Jia", "Jia Wang"), _author("WANG", "Xiao", "Xiao Wang"), _author("ZHANG", "Chi", "Chi Zhang")],
                "issued_year": 2025, "container_title": "", "volume": "", "issue": "",
                "pages": "", "article_number": "", "publisher": "", "doi": "",
                "arxiv_id": "2509.05478", **common,
            },
            "reason": "Remove an unrelated European Journal of Neurology binding and restore the identity printed on the bundled arXiv PDF.",
            "evidence": "Bundled arXiv:2509.05478 PDF, arXiv identity record and author repository citation",
            "note": "DOI 10.1111/ene.70191 and all European Journal of Neurology metadata belonged to an unrelated medical record and were removed; an under-review submission is not formal publication.",
        },
        {
            "document_id": "DOC_21DBA5461D85",
            "updates": {
                "publication_type": "journal_article", "container_title": "Information Fusion",
                "volume": "106", "issue": "", "pages": "", "article_number": "102255",
                "publisher": "Elsevier", "doi": "10.1016/j.inffus.2024.102255", **common,
            },
            "reason": "Upgrade the preprint record to the verified Information Fusion journal publication and classify 102255 as an article number.",
            "evidence": "Bundled PDF and official ScienceDirect volume 106 article 102255 record",
            "note": "102255 is an article number rather than a page range; the preferred citation is the June 2024 journal publication.",
        },
        {
            "document_id": "DOC_59C9BF529CFF",
            "updates": {
                "publication_type": "conference_paper",
                "conference_name": "2024 IEEE 40th International Conference on Data Engineering (ICDE)",
                "container_title": "2024 IEEE 40th International Conference on Data Engineering (ICDE)",
                "volume": "", "issue": "", "pages": "625-638", "article_number": "",
                "publisher": "IEEE", "doi": "10.1109/ICDE60146.2024.00054", **common,
            },
            "reason": "Upgrade the preprint record to the verified ICDE 2024 conference publication.",
            "evidence": "Bundled PDF, official ICDE 2024 accepted-paper list and DOI 10.1109/ICDE60146.2024.00054",
            "note": "The preferred formal citation is the ICDE 2024 proceedings paper; arXiv remains version provenance.",
        },
    ]
    for case in cases:
        record = by_id[case["document_id"]]
        for field, new_value in case["updates"].items():
            old_value = record.get(field, "")
            record[field] = new_value
            _append_revision(revisions, document_id=case["document_id"], field=field,
                             old_value=old_value, new_value=new_value, reason=case["reason"],
                             evidence=case["evidence"], reviewed_at=reviewed_at)
        _add_note(record, case["note"], reviewed_at)
    IDENTITY_PATH.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    REVISIONS_PATH.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in revisions), encoding="utf-8")
    print(json.dumps({"batch_id": BATCH_ID, "reviewer": REVIEWER}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
