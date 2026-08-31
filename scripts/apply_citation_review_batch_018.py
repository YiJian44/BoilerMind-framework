from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-018"
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
            "document_id": "DOC_A9850F04D347",
            "updates": {
                "publication_type": "preprint",
                "title": "Ada-MoGE: Adaptive Mixture of Gaussian Expert Model for Time Series Forecasting",
                "authors": [
                    _author("NI", "Zhenliang", "Zhenliang Ni"),
                    _author("MA", "Xiaowen", "Xiaowen Ma"),
                    _author("WU", "Zhenkai", "Zhenkai Wu"),
                    _author("XIAO", "Shuai", "Shuai Xiao"),
                    _author("SHU", "Han", "Han Shu"),
                    _author("CHEN", "Xinghao", "Xinghao Chen"),
                ],
                "issued_year": 2025,
                "container_title": "",
                "volume": "",
                "issue": "",
                "pages": "",
                "article_number": "",
                "publisher": "",
                "place": "",
                "institution": "",
                "doi": "",
                "arxiv_id": "2512.02061",
                "url": "",
                **common,
            },
            "reason": "Replace a false SSRN financial-paper binding with the identity printed on the bundled arXiv PDF.",
            "evidence": "Bundled PDF metadata and first page, source PDF SHA-256 a9850f04d347 and arXiv:2512.02061",
            "note": "SSRN DOI 10.2139/ssrn.2339273, the Mixed Tempered Stable title, authors and 2013 year belonged to an unrelated paper and were removed.",
        },
        {
            "document_id": "DOC_0D096E5BEBE0",
            "updates": {
                "publication_type": "conference_paper",
                "title": "Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework for Dependency, Asynchrony, and Missingness",
                "issued_year": 2026,
                "conference_name": "International Conference on Learning Representations 2026 (ICLR 2026)",
                "container_title": "International Conference on Learning Representations 2026 (ICLR 2026)",
                "volume": "",
                "issue": "",
                "pages": "",
                "article_number": "",
                "publisher": "",
                "doi": "",
                **common,
            },
            "reason": "Replace the conference-status header with the verified title and upgrade to the formal ICLR 2026 publication.",
            "evidence": "Bundled arXiv:2506.08660 v4 PDF and the official ICLR 2026 proceedings page",
            "note": "The conference record does not provide a DOI or page range, so neither field was invented; arXiv remains version provenance.",
        },
        {
            "document_id": "DOC_D15C86D09365",
            "updates": {
                "publication_type": "conference_paper",
                "conference_name": "Advances in Neural Information Processing Systems 36",
                "container_title": "Advances in Neural Information Processing Systems 36",
                "doi": "10.52202/075280-3050",
                **common,
            },
            "reason": "Upgrade the bundled preprint to the verified NeurIPS 2023 main-conference publication.",
            "evidence": "Bundled arXiv:2311.06190 PDF and official NeurIPS 2023 proceedings record with DOI 10.52202/075280-3050",
            "note": "The preferred formal citation is the NeurIPS 2023 proceedings paper; arXiv remains version provenance.",
        },
        {
            "document_id": "DOC_DAF2D2EB7830",
            "updates": {
                "publication_type": "journal_article",
                "authors": [
                    _author("KIM", "Juhyeon", "Juhyeon Kim"),
                    _author("LEE", "Hyungeun", "Hyungeun Lee"),
                    _author("YU", "Seungwon", "Seungwon Yu"),
                    _author("HWANG", "Ung", "Ung Hwang"),
                    _author("JUNG", "Wooyeol", "Wooyeol Jung"),
                    _author("YOON", "Kijung", "Kijung Yoon"),
                ],
                "container_title": "IEEE Access",
                "volume": "11",
                "issue": "",
                "pages": "118386-118394",
                "article_number": "",
                "publisher": "IEEE",
                "doi": "10.1109/ACCESS.2023.3325041",
                **common,
            },
            "reason": "Use the verified final IEEE Access identity rather than the seven-author workshop/preprint version.",
            "evidence": "Bundled arXiv/workshop PDF, final IEEE Access PDF and DOI 10.1109/ACCESS.2023.3325041",
            "note": "The bundled workshop/arXiv version includes Miseon Park and spells Wooyul Jung differently; the preferred formal journal citation deliberately follows the six-author IEEE Access version.",
        },
        {
            "document_id": "DOC_65975A149461",
            "updates": {
                "publication_type": "journal_article",
                "title": "Online evolutionary neural architecture search for multivariate non-stationary time series forecasting",
                "container_title": "Applied Soft Computing",
                "volume": "145",
                "issue": "",
                "pages": "",
                "article_number": "110522",
                "publisher": "Elsevier",
                "doi": "10.1016/j.asoc.2023.110522",
                **common,
            },
            "reason": "Restore the complete title and upgrade the bundled preprint to the verified journal publication.",
            "evidence": "Bundled arXiv:2302.10347 PDF and ScienceDirect Applied Soft Computing volume 145 article 110522",
            "note": "110522 is an article number, not a page range; the preferred citation is the 2023 Applied Soft Computing publication.",
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
