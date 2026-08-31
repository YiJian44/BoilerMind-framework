from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boilermind.core.contracts import ResearchProblemSpec  # noqa: E402
from boilermind.evidence.bundle_freezer import freeze_evidence_bundle  # noqa: E402
from boilermind.evidence.qwen_semantic_judge import QwenSemanticEvidenceJudge  # noqa: E402
from boilermind.evidence.sources.local_rag import LocalRAGSource  # noqa: E402
from boilermind.evidence.traceability_verifier import EvidenceTraceabilityVerifier  # noqa: E402
from boilermind.evidence.verification_pipeline import verify_from_assessments  # noqa: E402


RAG_ROOT = ROOT / "resources" / "local_rag"
BENCHMARK = RAG_ROOT / "evaluation" / "retrieval_benchmark_v1.json"
JUDGMENTS = RAG_ROOT / "evaluation" / "human_relevance_judgments.jsonl"
OUTPUT_ROOT = ROOT / "outputs" / "formal_evidence_chain_acceptance"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_problem(query: dict) -> ResearchProblemSpec:
    return ResearchProblemSpec(
        problem_id=query["query_id"],
        original_question=query["question"],
        research_object="燃煤锅炉软测量系统",
        target_variable=query["target_variable"],
        operating_condition=query["operating_condition"],
        manipulated_variables=[],
        observed_variables=[],
        context_variables=query["concepts"],
        research_goal=query["question"],
        success_criteria=["返回经过追溯和语义判断的科学证据"],
        constraints=["文献不得替代当前真实实验结论"],
    )


def run(query_id: str) -> dict:
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    query = next((item for item in benchmark["queries"] if item["query_id"] == query_id), None)
    if query is None:
        raise ValueError(f"unknown query_id:{query_id}")
    problem = build_problem(query)
    candidates = LocalRAGSource(rag_root=RAG_ROOT, top_k=8).retrieve(problem)
    verifier = EvidenceTraceabilityVerifier(RAG_ROOT)
    traceability = {item.evidence_id: verifier.verify(item) for item in candidates}
    judge = QwenSemanticEvidenceJudge()
    try:
        assessments = []
        for start in range(0, len(candidates), 4):
            assessments.extend(judge.judge(problem, candidates[start:start + 4]))
    finally:
        judge.close()
    semantic = {item.evidence_id: item for item in assessments}
    verification = verify_from_assessments(candidates, traceability, semantic)
    bundle = freeze_evidence_bundle(problem.problem_id, list(verification.verified)) if verification.verified else None
    human = {(row["query_id"], row["chunk_id"]): row for row in load_jsonl(JUDGMENTS)}
    decisions = {item.evidence_id: item for item in verification.decisions}
    rejected = {item.evidence_id: item.reason for item in verification.rejected}
    rows = []
    for rank, candidate in enumerate(candidates, 1):
        trace = traceability[candidate.evidence_id]
        assessment = semantic[candidate.evidence_id]
        decision = decisions[candidate.evidence_id]
        rows.append({
            "rank": rank,
            "evidence_id": candidate.evidence_id,
            "document_id": candidate.document_id,
            "chunk_id": candidate.chunk_id,
            "page_number": candidate.page_number,
            "title": candidate.title,
            "retrieval_score": candidate.retrieval_score,
            "formatted_citation": candidate.formatted_citation,
            "human_raw_relevance": (human.get((query_id, candidate.chunk_id or "")) or {}).get("label"),
            "traceability_verified": trace.verified,
            "traceability_rationale": trace.rationale,
            "semantic_verified": assessment.semantic_verified,
            "claim_support": assessment.claim_support.value,
            "applicability": assessment.applicability.value,
            "core_claim_eligible": decision.core_claim_eligible,
            "hypothesis_inspiration_eligible": decision.hypothesis_inspiration_eligible,
            "formal_claim_support_eligible": decision.formal_claim_support_eligible,
            "semantic_rationale": assessment.verification_rationale,
            "final_status": "VERIFIED" if candidate.evidence_id not in rejected else "REJECTED",
            "rejection_reason": rejected.get(candidate.evidence_id),
            "text": candidate.text,
        })
    return {
        "schema_version": "boilermind.formal-evidence-chain-acceptance.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_id": query_id,
        "question": query["question"],
        "source_scope": "LOCAL_APPROVED_RAG_ONLY",
        "web_literature_enabled": False,
        "retrieved_count": len(candidates),
        "traceable_count": sum(item.verified for item in traceability.values()),
        "semantic_verified_count": sum(item.semantic_verified for item in assessments),
        "verified_count": len(verification.verified),
        "rejected_count": len(verification.rejected),
        "core_claim_eligible_count": sum(item.core_claim_eligible for item in verification.decisions),
        "hypothesis_inspiration_eligible_count": sum(item.hypothesis_inspiration_eligible for item in verification.decisions),
        "formal_claim_support_eligible_count": sum(item.formal_claim_support_eligible for item in verification.decisions),
        "semantic_judge_audit": judge.audit_events,
        "bundle": bundle.model_dump(mode="json") if bundle else None,
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-id", required=True)
    args = parser.parse_args()
    payload = run(args.query_id)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / f"{args.query_id.lower()}_latest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "query_id", "retrieved_count", "traceable_count", "semantic_verified_count",
        "verified_count", "rejected_count", "core_claim_eligible_count",
    )}, ensure_ascii=False, indent=2))
    print(f"output={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
