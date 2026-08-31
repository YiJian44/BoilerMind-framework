from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS = ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-021"
REVIEWER = "wmy"


def load(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def revision(items: list[dict], document_id: str, field: str, old: Any, new: Any,
             reason: str, evidence: str, reviewed_at: str) -> None:
    key = (document_id, field, BATCH_ID)
    if any((x.get("document_id"), x.get("field"), x.get("batch_id")) == key for x in items):
        return
    items.append({"schema_version": "boilermind.literature-revision.v1",
                  "batch_id": BATCH_ID, "document_id": document_id, "field": field,
                  "old_value": old, "new_value": new, "reason": reason,
                  "reviewer": REVIEWER, "reviewed_at": reviewed_at, "evidence": evidence})


def note(record: dict, message: str, reviewed_at: str) -> None:
    notes = record.setdefault("verification_notes", [])
    if not any(x.get("message") == message for x in notes):
        notes.append({"message": message, "timestamp": reviewed_at, "source": f"human_review:{REVIEWER}"})


def main() -> int:
    records, revisions = load(IDENTITY), load(REVISIONS)
    by_id = {x["document_id"]: x for x in records}
    reviewed_at = datetime.now(timezone.utc).isoformat()
    common = {"identity_status": "VERIFIED", "citation_eligibility": "FORMALLY_CITABLE",
              "verified_by": REVIEWER, "verified_at": reviewed_at}
    cases = [
        ("DOC_B417ED6C7FF9", {
            "publication_type": "journal_article",
            "title": "The GNAR-edge model: A network autoregressive model for networks with time-varying edge weights",
            "issued_year": 2023, "container_title": "Journal of Complex Networks",
            "volume": "11", "issue": "6", "pages": "", "article_number": "cnad039",
            "publisher": "Oxford University Press", "doi": "10.1093/comnet/cnad039",
            "medium": "", **common},
         "Restore the complete title and upgrade the bundled preprint to the verified journal article.",
         "Bundled arXiv:2305.16097 PDF and Oxford Academic volume 11 issue 6 article cnad039",
         "cnad039 is an article number rather than a page range; the preferred citation is the 2023 Journal of Complex Networks publication."),
        ("DOC_21C0EB27B33E", {
            "publication_type": "conference_paper",
            "conference_name": "Machine Learning and Knowledge Discovery in Databases: ECML PKDD 2022",
            "container_title": "Lecture Notes in Computer Science",
            "issued_year": 2023, "volume": "13718", "issue": "", "pages": "36-52",
            "article_number": "", "publisher": "Springer", "place": "Cham",
            "doi": "10.1007/978-3-031-26422-1_3", "medium": "", **common},
         "Upgrade the bundled preprint to the verified Springer ECML PKDD proceedings chapter.",
         "Bundled arXiv:2110.08255 PDF and Springer LNCS 13718 chapter record",
         "ECML PKDD 2022 is the conference name; the Springer chapter was first published online in 2023, which is used as the citation year."),
        ("DOC_DF3CA4D01FF7", {
            "publication_type": "journal_article", "issued_year": 2025,
            "container_title": "International Journal of Forecasting", "volume": "", "issue": "",
            "pages": "", "article_number": "", "publisher": "Elsevier",
            "doi": "10.1016/j.ijforecast.2025.10.001", "medium": "OL", **common},
         "Upgrade the preprint to the verified online-first corrected proof without inventing unassigned volume, issue or pages.",
         "Bundled arXiv:2502.19086 v5 PDF and official ScienceDirect corrected-proof record",
         "Available online 11 November 2025 as an in-press corrected proof; volume, issue and pagination remain deliberately empty until assigned."),
        ("DOC_D54D6AD3E9CF", {
            "publication_type": "journal_article", "container_title": "International Journal of Forecasting",
            "volume": "32", "issue": "3", "pages": "1029-1037", "article_number": "",
            "publisher": "Elsevier", "doi": "10.1016/j.ijforecast.2016.01.001",
            "medium": "", **common},
         "Upgrade the bundled preprint to the verified International Journal of Forecasting article.",
         "Bundled arXiv:1603.01376 PDF and official ScienceDirect volume 32 issue 3 record",
         "The preferred formal citation is the July-September 2016 journal publication."),
        ("DOC_B358E7E75ADB", {
            "publication_type": "conference_paper",
            "conference_name": "Proceedings of the 4th Table Representation Learning Workshop",
            "container_title": "", "volume": "", "issue": "", "pages": "156-165",
            "article_number": "", "publisher": "Association for Computational Linguistics",
            "place": "Vienna", "doi": "10.18653/v1/2025.trl-1.12", "medium": "", **common},
         "Upgrade the bundled preprint to the verified ACL Anthology workshop publication.",
         "Bundled arXiv:2410.11674 PDF and ACL Anthology 2025.trl-1.12 record",
         "The formal record was published in Vienna in July 2025 on pages 156-165."),
    ]
    for document_id, updates, reason, evidence, message in cases:
        record = by_id[document_id]
        for field, new in updates.items():
            old = record.get(field, "")
            record[field] = new
            revision(revisions, document_id, field, old, new, reason, evidence, reviewed_at)
        note(record, message, reviewed_at)
    IDENTITY.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    REVISIONS.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in revisions), encoding="utf-8")
    print(json.dumps({"batch_id": BATCH_ID, "reviewer": REVIEWER}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
