from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from boilermind.core.contracts import ResearchProblemSpec  # noqa: E402
from boilermind.evidence.citation_registry import (  # noqa: E402
    CitationRegistry,
    FORMALLY_CITABLE,
)
from boilermind.evidence.sources.local_rag import LocalRAGSource  # noqa: E402
from boilermind.evidence.retrieval_evaluation import (  # noqa: E402
    evaluate_ranked_results,
    summarize_query_evaluations,
)


RAG_ROOT = PROJECT_ROOT / "resources" / "local_rag"
BENCHMARK = RAG_ROOT / "evaluation" / "retrieval_benchmark_v1.json"
OUTPUT = PROJECT_ROOT / "outputs" / "rag_acceptance" / "rag_acceptance_latest.json"
AUDIT_ROOT = RAG_ROOT / "audit"
JUDGMENTS = RAG_ROOT / "evaluation" / "human_relevance_judgments.jsonl"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_judgments() -> dict[tuple[str, str], dict]:
    if not JUDGMENTS.exists():
        return {}
    rows = [json.loads(line) for line in JUDGMENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {}
    for row in rows:
        key = (str(row.get("query_id") or ""), str(row.get("chunk_id") or ""))
        if key in result:
            raise RuntimeError(f"Duplicate relevance judgment: {key[0]}/{key[1]}")
        result[key] = row
    return result


def _refresh_audit_snapshots(registry: CitationRegistry) -> None:
    retrieval_only = []
    review_queue = []
    non_formally_citable = []
    for document_id, record in registry.records.items():
        if registry.is_human_approved(document_id):
            continue
        is_candidate = record.get("citation_eligibility") == FORMALLY_CITABLE
        reasons = (
            ["awaiting_human_approval"]
            if is_candidate
            else [str(record.get("identity_status") or "not_formally_citable").lower()]
        )
        retrieval_only.append({
            "document_id": document_id,
            "source_file": record.get("source_file", ""),
            "title": record.get("title", ""),
            "identity_status": record.get("identity_status", ""),
            "citation_eligibility": record.get("citation_eligibility", ""),
            "reasons": reasons,
        })
        review_queue.append({
            "document_id": document_id,
            "title": record.get("title", ""),
            "current_status": record.get("identity_status", ""),
            "citation_eligibility": record.get("citation_eligibility", ""),
            "reasons": reasons,
            "notes": record.get("verification_notes", []),
        })
        if not is_candidate:
            non_formally_citable.append({
                "document_id": document_id,
                "source_file": record.get("source_file", ""),
                "title": record.get("title", ""),
                "identity_status": record.get("identity_status", ""),
                "reasons": reasons,
            })
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_jsonl(AUDIT_ROOT / "retrieval_only.jsonl", retrieval_only)
    _write_jsonl(AUDIT_ROOT / "review_queue.jsonl", review_queue)
    _write_jsonl(AUDIT_ROOT / "non_formally_citable_records.jsonl", non_formally_citable)


def main() -> int:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    registry = CitationRegistry(RAG_ROOT)
    source = LocalRAGSource(rag_root=RAG_ROOT, top_k=8)
    candidate_count = sum(
        item.get("citation_eligibility") == FORMALLY_CITABLE
        for item in registry.records.values()
    )
    approved_count = sum(
        registry.is_human_approved(document_id)
        for document_id in registry.records
    )
    judgments = _load_judgments()
    results = []
    for query in benchmark["queries"]:
        problem = ResearchProblemSpec(
            problem_id=query["query_id"],
            original_question=query["question"],
            research_object="燃煤锅炉软测量系统",
            target_variable=query["target_variable"],
            operating_condition=query["operating_condition"],
            manipulated_variables=[],
            observed_variables=[],
            context_variables=query["concepts"],
            research_goal=query["question"],
            success_criteria=["返回可追溯且可人工判定相关性的文献切片"],
            constraints=["文献相关性不得替代真实实验结论"],
        )
        retrieved = source.retrieve(problem)
        ranked_rows = [
                    {
                        "rank": rank,
                        "evidence_id": item.evidence_id,
                        "document_id": item.document_id,
                        "chunk_id": item.chunk_id,
                        "page_number": item.page_number,
                        "title": item.title,
                        "retrieval_score": item.retrieval_score,
                        "citation_candidate_eligibility": item.citation_candidate_eligibility,
                        "human_citation_approved": item.human_citation_approved,
                        "effective_citation_eligibility": item.citation_eligibility,
                        "relevance_judgment": (judgments.get((query["query_id"], item.chunk_id)) or {}).get("label"),
                        "relevance_reviewer": (judgments.get((query["query_id"], item.chunk_id)) or {}).get("reviewer"),
                    }
                    for rank, item in enumerate(retrieved, 1)
                ]
        results.append(
            {
                "query_id": query["query_id"],
                "question": query["question"],
                "retrieved_count": len(retrieved),
                "evaluation": evaluate_ranked_results(ranked_rows, expected_count=8),
                "results": ranked_rows,
            }
        )
    all_complete = all(item["evaluation"]["complete"] for item in results)
    overall_evaluation = summarize_query_evaluations(results)
    payload = {
        "schema_version": "boilermind.rag-acceptance.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(BENCHMARK.relative_to(PROJECT_ROOT)),
        "paper_count": len(registry.records),
        "chunk_count": len(registry.chunks),
        "automated_formal_candidate_count": candidate_count,
        "human_approved_formal_count": approved_count,
        "effective_formal_count": approved_count,
        "retrieval_only_count": len(registry.records) - approved_count,
        "judgment_status": "COMPLETE" if all_complete else "PENDING_HUMAN_RELEVANCE_LABELS",
        "metric_status": "COMPUTED" if all_complete else "NOT_COMPUTED_UNTIL_EACH_TOP8_RESULT_IS_JUDGED",
        "overall_evaluation": overall_evaluation,
        "queries": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _refresh_audit_snapshots(registry)
    print(json.dumps({key: payload[key] for key in (
        "paper_count",
        "chunk_count",
        "automated_formal_candidate_count",
        "human_approved_formal_count",
        "effective_formal_count",
        "retrieval_only_count",
        "judgment_status",
        "metric_status",
    )}, ensure_ascii=False, indent=2))
    print(f"output={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
