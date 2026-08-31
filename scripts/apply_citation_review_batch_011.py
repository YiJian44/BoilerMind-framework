from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-011"
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
            "document_id": "DOC_0811D3B65060",
            "updates": {
                "conference_name": "2021 International Conference on Machine Learning and Cybernetics (ICMLC)",
                "container_title": "2021 International Conference on Machine Learning and Cybernetics (ICMLC)",
                "doi": "10.1109/ICMLC54886.2021.9737262",
                **common,
            },
            "reason": "Complete the verified IEEE conference identity and correct the prior HTTP-202 DOI check interpretation.",
            "evidence": "Bundled IEEE conference PDF publication footer and DOI 10.1109/ICMLC54886.2021.9737262",
            "note": "The earlier HTTP 202 result did not invalidate the DOI; the DOI, conference, title and author byline are printed in the bundled IEEE PDF.",
        },
        {
            "document_id": "DOC_36A640225FDF",
            "updates": {
                "publication_type": "conference_paper",
                "title": "Steam Flow Estimation with Artificial Neural Network Based on Power Plant Operational Data",
                "authors": [
                    {"family": "REICHERT", "given": "Helena Haas", "literal": "Helena Haas Reichert", "orcid": ""},
                    {"family": "FONSECA JUNIOR", "given": "João Gari da Silva", "literal": "João Gari da Silva Fonseca Junior", "orcid": ""},
                    {"family": "SCHNEIDER", "given": "Paulo Smith", "literal": "Paulo Smith Schneider", "orcid": ""},
                ],
                "conference_name": "17th Brazilian Congress of Thermal Sciences and Engineering",
                "container_title": "17th Brazilian Congress of Thermal Sciences and Engineering",
                **common,
            },
            "reason": "Correct the title, complete the three-author byline and identify the bundled work as a conference paper.",
            "evidence": "Bundled ENCIT-2018-0207 conference PDF title page",
            "note": "No reliable DOI or proceedings page range was found; the formal identity is approved from the bundled conference paper itself, without inventing either field.",
        },
        {
            "document_id": "DOC_D409DE3C807E",
            "updates": {
                "publication_type": "journal_article",
                "title": "Predicting Machine Failures from Multivariate Time Series: An Industrial Case Study",
                "container_title": "Machines",
                "volume": "12",
                "issue": "6",
                "article_number": "357",
                "doi": "10.3390/machines12060357",
                **common,
            },
            "reason": "Replace the Highlights extraction artifact with the independently verified journal publication.",
            "evidence": "Bundled arXiv:2402.17804 PDF, MDPI publication page and DOI 10.3390/machines12060357",
            "note": "The preferred formal citation is the 2024 Machines journal article rather than the bundled preprint label.",
        },
        {
            "document_id": "DOC_8C6A7C03AC37",
            "updates": {
                "publication_type": "conference_paper",
                "title": "KANS: Knowledge Discovery Graph Attention Network for Soft Sensing in Multivariate Industrial Processes",
                "authors": [
                    {"family": "TEW", "given": "Hwa Hui", "literal": "Hwa Hui Tew", "orcid": ""},
                    {"family": "LI", "given": "Gaoxuan", "literal": "Gaoxuan Li", "orcid": ""},
                    {"family": "DING", "given": "Fan", "literal": "Fan Ding", "orcid": ""},
                    {"family": "LUO", "given": "Xuewen", "literal": "Xuewen Luo", "orcid": ""},
                    {"family": "LOO", "given": "Junn Yong", "literal": "Junn Yong Loo", "orcid": ""},
                    {"family": "TING", "given": "Chee-Ming", "literal": "Chee-Ming Ting", "orcid": ""},
                    {"family": "DING", "given": "Ze Yang", "literal": "Ze Yang Ding", "orcid": ""},
                    {"family": "TAN", "given": "Chee Pin", "literal": "Chee Pin Tan", "orcid": ""},
                ],
                "issued_year": 2024,
                "conference_name": "2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC)",
                "container_title": "2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC)",
                "pages": "4377-4383",
                "doi": "10.1109/SMC54092.2024.10831311",
                "publisher": "IEEE",
                **common,
            },
            "reason": "Replace incomplete arXiv metadata with the independently verified IEEE conference publication.",
            "evidence": "Bundled arXiv:2501.02015 PDF, Monash publication record and IEEE SMC 2024 proceedings metadata",
            "note": "The preferred formal citation is the 2024 IEEE SMC conference paper; the bundled arXiv upload date is 2025.",
        },
        {
            "document_id": "DOC_E951E997C909",
            "updates": {
                "publication_type": "journal_article",
                "title": "A deep latent variable model for semi-supervised multi-unit soft sensing in industrial processes",
                "issued_year": 2026,
                "container_title": "Applied Soft Computing",
                "volume": "186",
                "article_number": "114198",
                "doi": "10.1016/j.asoc.2025.114198",
                **common,
            },
            "reason": "Replace the truncated preprint title with the independently verified journal publication.",
            "evidence": "Bundled arXiv:2407.13310 PDF, ScienceDirect publication record and DOI 10.1016/j.asoc.2025.114198",
            "note": "The journal volume is dated 2026 although the DOI was registered in 2025; the formal citation uses the journal issue year.",
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
