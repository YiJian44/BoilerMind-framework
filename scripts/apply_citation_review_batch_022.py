from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS = ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-022"
REVIEWER = "wmy"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def add_revision(items: list[dict], document_id: str, field: str, old: Any, new: Any,
                 reason: str, evidence: str, reviewed_at: str) -> None:
    key = (document_id, field, BATCH_ID)
    if any((x.get("document_id"), x.get("field"), x.get("batch_id")) == key for x in items):
        return
    items.append({"schema_version": "boilermind.literature-revision.v1",
                  "batch_id": BATCH_ID, "document_id": document_id, "field": field,
                  "old_value": old, "new_value": new, "reason": reason,
                  "reviewer": REVIEWER, "reviewed_at": reviewed_at, "evidence": evidence})


def add_note(record: dict, message: str, reviewed_at: str) -> None:
    notes = record.setdefault("verification_notes", [])
    if not any(x.get("message") == message for x in notes):
        notes.append({"message": message, "timestamp": reviewed_at, "source": f"human_review:{REVIEWER}"})


def main() -> int:
    records, revisions = load(IDENTITY), load(REVISIONS)
    by_id = {x["document_id"]: x for x in records}
    reviewed_at = datetime.now(timezone.utc).isoformat()
    common = {"identity_status": "VERIFIED", "citation_eligibility": "FORMALLY_CITABLE",
              "verified_by": REVIEWER, "verified_at": reviewed_at, "medium": ""}
    cases = [
        ("DOC_CDCC3ACCC2AB", {
            "publication_type": "journal_article", "issued_year": 2024,
            "container_title": "International Journal of Forecasting", "volume": "40",
            "issue": "2", "pages": "470-489", "article_number": "",
            "publisher": "Elsevier", "doi": "10.1016/j.ijforecast.2023.04.007", **common},
         "Upgrade the bundled preprint to the verified final journal volume and pagination.",
         "Bundled arXiv:2110.13179 PDF and official ScienceDirect volume 40 issue 2 record",
         "The DOI was registered in 2023, while the preferred final citation is the April-June 2024 volume 40 issue 2 publication."),
        ("DOC_3123590E1157", {
            "publication_type": "conference_paper",
            "conference_name": "Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
            "container_title": "", "volume": "", "issue": "", "pages": "5816-5828",
            "article_number": "", "publisher": "Neural Information Processing Systems Foundation",
            "doi": "10.52202/068431-0421", **common},
         "Upgrade the bundled preprint to the verified NeurIPS 2022 main-conference publication.",
         "Bundled arXiv:2106.09305 PDF and official NeurIPS 2022 proceedings record",
         "The preferred formal citation is the NeurIPS 2022 proceedings paper; arXiv remains version provenance."),
        ("DOC_3D25AF9C9167", {
            "publication_type": "journal_article",
            "title": "A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting",
            "issued_year": 2020, "container_title": "International Journal of Forecasting",
            "volume": "36", "issue": "1", "pages": "75-85", "article_number": "",
            "publisher": "Elsevier", "doi": "10.1016/j.ijforecast.2019.03.017", **common},
         "Replace the journal-name header mistakenly extracted as the title with the title printed on the PDF.",
         "Bundled final PDF and official ScienceDirect volume 36 issue 1 record",
         "The article was available and copyrighted in 2019, then assigned to the 2020 final volume; the preferred citation uses 2020."),
        ("DOC_D49F5C4A5C05", {
            "publication_type": "journal_article", "issued_year": 2021,
            "container_title": "IEEE Sensors Journal", "volume": "21", "issue": "6",
            "pages": "7833-7848", "article_number": "", "publisher": "IEEE",
            "doi": "10.1109/JSEN.2019.2923982", **common},
         "Confirm the final IEEE Sensors Journal volume, issue, pagination and citation year.",
         "Bundled final IEEE PDF showing volume 21 issue 6 pages 7833-7848 and DOI 10.1109/JSEN.2019.2923982",
         "The article was accepted and published online in 2019, then assigned to the 2021 final volume; the preferred citation uses 2021."),
        ("DOC_B0B06FD8B67C", {
            "publication_type": "journal_article",
            "title": "A review of unsupervised feature learning and deep learning for time-series modeling",
            "issued_year": 2014, "container_title": "Pattern Recognition Letters",
            "volume": "42", "issue": "", "pages": "11-24", "article_number": "",
            "publisher": "Elsevier", "doi": "10.1016/j.patrec.2014.01.008", **common},
         "Replace the severely truncated title with the complete title printed on the final PDF.",
         "Bundled final PDF and official ScienceDirect Pattern Recognition Letters volume 42 record",
         "The preferred citation uses the complete title and the June 2014 final journal publication."),
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
