from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_identity.jsonl"
REVISIONS_PATH = PROJECT_ROOT / "resources/local_rag/metadata/literature_revisions.jsonl"
BATCH_ID = "CITATION-REVIEW-BATCH-007"
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


def _append_revision(
    revisions: list[dict],
    *,
    document_id: str,
    old_value: str,
    new_value: str,
    reason: str,
    evidence: str,
    reviewed_at: str,
) -> None:
    key = (document_id, "publication_scope", BATCH_ID)
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
            "field": "publication_scope",
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
            "reviewer": REVIEWER,
            "reviewed_at": reviewed_at,
            "evidence": evidence,
        }
    )


def main() -> int:
    records = _load_jsonl(IDENTITY_PATH)
    revisions = _load_jsonl(REVISIONS_PATH)
    by_id = {item["document_id"]: item for item in records}
    reviewed_at = datetime.now(timezone.utc).isoformat()

    restrictions = [
        (
            "DOC_B3A537D552DA",
            "The official arXiv record contains an ICLR 2026 journal reference, but no complete "
            "proceedings metadata was independently verified; cite only as arXiv:2601.19022.",
            "preprint without explicit conference-upgrade warning",
            "arXiv preprint only; do not synthesize ICLR proceedings fields",
            "Prevent an arXiv journal-reference string from being treated as complete proceedings metadata.",
            "Bundled arXiv:2601.19022v2 PDF and official arXiv API record",
        ),
        (
            "DOC_D02AA75A87FF",
            "The PDF and official arXiv record state acceptance at the NeurIPS 2025 BERT2S "
            "Workshop, but no independently verified proceedings metadata is stored; cite only "
            "as arXiv:2509.04449.",
            "preprint without explicit workshop warning",
            "arXiv preprint only; workshop acceptance must not be promoted to proceedings metadata",
            "Keep the citation within the bibliographic fields that were actually verified.",
            "Bundled arXiv:2509.04449v3 PDF and official arXiv API record",
        ),
        (
            "DOC_56683AAF1DE3",
            "The bundled PDF is marked 'Preprint. Under review'; cite only as arXiv:2506.09174 "
            "unless a later formal publication is independently verified.",
            "preprint without explicit under-review warning",
            "arXiv preprint only; under-review status is not a formal publication",
            "Prevent a submission status from being represented as publication evidence.",
            "Bundled arXiv:2506.09174v2 PDF and official arXiv API record",
        ),
    ]
    for document_id, note, old_value, new_value, reason, evidence in restrictions:
        _add_note(by_id[document_id], note, reviewed_at)
        _append_revision(
            revisions,
            document_id=document_id,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            evidence=evidence,
            reviewed_at=reviewed_at,
        )

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
