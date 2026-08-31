from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAG_ROOT = ROOT / "resources" / "local_rag"
ACCEPTANCE = ROOT / "outputs" / "rag_acceptance" / "rag_acceptance_latest.json"
LEDGER = RAG_ROOT / "evaluation" / "human_relevance_judgments.jsonl"
VALID_LABELS = {"RELEVANT", "PARTIAL", "IRRELEVANT", "UNJUDGEABLE"}
CONFIRMATION = "VERIFIED_RELEVANCE_AGAINST_CHUNK"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_result(query_id: str, chunk_id: str) -> tuple[dict, dict]:
    payload = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    query = next((item for item in payload["queries"] if item["query_id"] == query_id), None)
    if query is None:
        raise SystemExit(f"unknown query_id: {query_id}")
    result = next((item for item in query["results"] if item["chunk_id"] == chunk_id), None)
    if result is None:
        raise SystemExit(f"chunk is not in the current ranked result: {query_id}/{chunk_id}")
    return query, result


def list_results(query_id: str) -> int:
    payload = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    query = next((item for item in payload["queries"] if item["query_id"] == query_id), None)
    if query is None:
        raise SystemExit(f"unknown query_id: {query_id}")
    judgments = {(x["query_id"], x["chunk_id"]): x for x in load_jsonl(LEDGER)}
    print(json.dumps({
        "query_id": query_id,
        "question": query["question"],
        "results": [{**row, "human_judgment": judgments.get((query_id, row["chunk_id"]))}
                    for row in query["results"]],
    }, ensure_ascii=False, indent=2))
    return 0


def label_result(query_id: str, chunk_id: str, label: str, reviewer: str,
                 confirmation: str, note: str) -> int:
    if label not in VALID_LABELS:
        raise SystemExit(f"invalid label: {label}")
    if confirmation != CONFIRMATION:
        raise SystemExit(f"confirmation must be exactly {CONFIRMATION}")
    query, result = find_result(query_id, chunk_id)
    rows = load_jsonl(LEDGER)
    record = {
        "schema_version": "boilermind.relevance-judgment.v1",
        "query_id": query_id,
        "question": query["question"],
        "document_id": result["document_id"],
        "chunk_id": chunk_id,
        "page_number": result["page_number"],
        "rank_at_review": result["rank"],
        "label": label,
        "reviewer": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "confirmation": confirmation,
        "note": note,
    }
    key = (query_id, chunk_id)
    matches = [index for index, row in enumerate(rows) if (row.get("query_id"), row.get("chunk_id")) == key]
    if len(matches) > 1:
        raise SystemExit(f"duplicate ledger key: {query_id}/{chunk_id}")
    if matches:
        rows[matches[0]] = record
    else:
        rows.append(record)
    LEDGER.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Human relevance review for current RAG rankings")
    subparsers = parser.add_subparsers(dest="command", required=True)
    listing = subparsers.add_parser("list")
    listing.add_argument("--query-id", required=True)
    labeling = subparsers.add_parser("label")
    labeling.add_argument("--query-id", required=True)
    labeling.add_argument("--chunk-id", required=True)
    labeling.add_argument("--label", choices=sorted(VALID_LABELS), required=True)
    labeling.add_argument("--reviewer", required=True)
    labeling.add_argument("--confirm", required=True)
    labeling.add_argument("--note", default="")
    args = parser.parse_args()
    if args.command == "list":
        return list_results(args.query_id)
    return label_result(args.query_id, args.chunk_id, args.label, args.reviewer, args.confirm, args.note)


if __name__ == "__main__":
    raise SystemExit(main())
