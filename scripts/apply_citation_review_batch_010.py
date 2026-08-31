from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-010"
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
    revisions: list[dict],
    *,
    document_id: str,
    field: str,
    old_value: Any,
    new_value: Any,
    reason: str,
    evidence: str,
    reviewed_at: str,
) -> None:
    key = (document_id, field, BATCH_ID)
    if any(
        (item.get("document_id"), item.get("field"), item.get("batch_id")) == key
        for item in revisions
    ):
        return
    revisions.append(
        {
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
        }
    )


def _add_note(record: dict, message: str, reviewed_at: str) -> None:
    notes = record.setdefault("verification_notes", [])
    if not any(item.get("message") == message for item in notes):
        notes.append(
            {
                "message": message,
                "timestamp": reviewed_at,
                "source": f"human_review:{REVIEWER}",
            }
        )


def _apply_updates(
    record: dict,
    revisions: list[dict],
    updates: dict[str, Any],
    *,
    reason: str,
    evidence: str,
    reviewed_at: str,
) -> None:
    document_id = record["document_id"]
    for field, new_value in updates.items():
        old_value = record.get(field, "")
        record[field] = new_value
        _append_revision(
            revisions,
            document_id=document_id,
            field=field,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            evidence=evidence,
            reviewed_at=reviewed_at,
        )


def main() -> int:
    records = _load_jsonl(IDENTITY_PATH)
    revisions = _load_jsonl(REVISIONS_PATH)
    by_id = {item["document_id"]: item for item in records}
    reviewed_at = datetime.now(timezone.utc).isoformat()

    cases = [
        {
            "document_id": "DOC_08171DF8B56D",
            "updates": {
                "title": "Transformer-based Drum-level Prediction in a Boiler Plant with Delayed Relations among Multivariates",
                "authors": [
                    {"family": "SU", "given": "Gang", "literal": "Gang Su", "orcid": ""},
                    {"family": "YANG", "given": "Sun", "literal": "Sun Yang", "orcid": ""},
                    {"family": "LI", "given": "Zhishuai", "literal": "Zhishuai Li", "orcid": ""},
                    {"family": "LI", "given": "Ziyue", "literal": "Ziyue Li", "orcid": ""},
                ],
                "identity_status": "VERIFIED",
                "citation_eligibility": "FORMALLY_CITABLE",
                "verified_by": REVIEWER,
                "verified_at": reviewed_at,
            },
            "reason": "Correct the truncated PDF title and preserve the complete bundled-PDF author byline.",
            "evidence": "Bundled arXiv:2407.11180 PDF title page and official arXiv record",
            "note": "The bundled PDF lists four authors, while the official arXiv metadata lists three; the formal preprint citation follows the PDF byline. The paper's under-review note is not publication evidence.",
        },
        {
            "document_id": "DOC_524A66B85829",
            "updates": {
                "publication_type": "journal_article",
                "title": "基于数据驱动的燃煤锅炉NOx排放浓度动态修正预测模型",
                "authors": [
                    {"family": "", "given": "", "literal": "唐振浩", "orcid": ""},
                    {"family": "", "given": "", "literal": "朱得宇", "orcid": ""},
                    {"family": "", "given": "", "literal": "李扬", "orcid": ""},
                ],
                "issued_year": 2022,
                "container_title": "中国电机工程学报",
                "volume": "42",
                "issue": "14",
                "pages": "5182-5193",
                "doi": "10.13334/j.0258-8013.pcsee.211426",
                "language": "zh",
                "identity_status": "VERIFIED",
                "citation_eligibility": "FORMALLY_CITABLE",
                "verified_by": REVIEWER,
                "verified_at": reviewed_at,
            },
            "reason": "Replace truncated preprint metadata with the independently verified Chinese journal publication.",
            "evidence": "Bundled arXiv:2110.15600 PDF, official arXiv journal reference and DOI 10.13334/j.0258-8013.pcsee.211426",
            "note": "The bundled file is an arXiv version; the preferred formal citation is the 2022 journal article in 中国电机工程学报.",
        },
        {
            "document_id": "DOC_79DC5F7E3E5B",
            "updates": {
                "publication_type": "conference_paper",
                "conference_name": "ECAI 2024",
                "container_title": "Frontiers in Artificial Intelligence and Applications",
                "volume": "392",
                "pages": "4579-4585",
                "doi": "10.3233/FAIA241051",
                "publisher": "IOS Press",
                "identity_status": "VERIFIED",
                "citation_eligibility": "FORMALLY_CITABLE",
                "verified_by": REVIEWER,
                "verified_at": reviewed_at,
            },
            "reason": "Complete the independently verified ECAI 2024 proceedings metadata.",
            "evidence": "Bundled PDF publication header, IOS Press proceedings record and DOI 10.3233/FAIA241051",
            "note": "The earlier DOI-resolution failure was a network/checking failure, not evidence that the DOI was invalid.",
        },
        {
            "document_id": "DOC_671C296AB9A5",
            "updates": {
                "title": "Artificial neural networking model for the prediction of high efficiency boiler steam generation and distribution",
                "identity_status": "VERIFIED",
                "citation_eligibility": "FORMALLY_CITABLE",
                "verified_by": REVIEWER,
                "verified_at": reviewed_at,
            },
            "reason": "Replace the extraction-damaged title with the verified journal title.",
            "evidence": "Bundled journal PDF and Crossref DOI 10.1016/j.simpat.2015.06.003",
            "note": "The Crossref publication year 2015 is used; the PDF's earlier date reflects manuscript history, not the final journal issue year.",
        },
        {
            "document_id": "DOC_A161E86C6F11",
            "updates": {
                "publication_type": "journal_article",
                "title": "Design and Control of Steam Flow in Cement Production Process using Neural Network Based Controllers",
                "container_title": "Researcher",
                "volume": "12",
                "issue": "5",
                "pages": "76-84",
                "doi": "10.7537/marsrsj120520.09",
                "identity_status": "VERIFIED",
                "citation_eligibility": "FORMALLY_CITABLE",
                "verified_by": REVIEWER,
                "verified_at": reviewed_at,
            },
            "reason": "Remove abstract contamination from the title and complete the source journal metadata.",
            "evidence": "Bundled journal PDF and publisher issue page for Researcher 2020, 12(5), 76-84",
            "note": "The identity is traceable to the bundled PDF and publisher issue page, but this source's bibliographic verification must not be interpreted as a source-quality endorsement or increase its scientific evidence weight.",
        },
    ]

    for case in cases:
        record = by_id[case["document_id"]]
        _apply_updates(
            record,
            revisions,
            case["updates"],
            reason=case["reason"],
            evidence=case["evidence"],
            reviewed_at=reviewed_at,
        )
        _add_note(record, case["note"], reviewed_at)

    IDENTITY_PATH.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    REVISIONS_PATH.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in revisions),
        encoding="utf-8",
    )
    print(json.dumps({"batch_id": BATCH_ID, "reviewer": REVIEWER}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
