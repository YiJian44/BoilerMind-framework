from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from boilermind.core.contracts import ResearchProblemSpec, VerifiedEvidence
from boilermind.evidence.citation_registry import CitationRegistry
from boilermind.evidence.qwen_semantic_judge import QwenSemanticEvidenceJudge
from boilermind.evidence.sources.local_rag import LocalRAGSource
from boilermind.evidence.traceability_verifier import EvidenceTraceabilityVerifier
from boilermind.evidence.verification_pipeline import verify_from_assessments


@dataclass(frozen=True)
class ReportClaim:
    claim_id: str
    text: str
    citation_purpose: str = "BACKGROUND"


@dataclass(frozen=True)
class AutomaticCitationBinding:
    claim_id: str
    claim_text: str
    citation_number: int
    document_id: str
    chunk_id: str
    page_number: int
    supporting_excerpt: str
    formatted_citation: str


@dataclass(frozen=True)
class AutomaticCitationFailure:
    claim_id: str
    reason: str


@dataclass(frozen=True)
class AutomaticCitationResult:
    bindings: tuple[AutomaticCitationBinding, ...]
    failures: tuple[AutomaticCitationFailure, ...]
    references: tuple[str, ...]

    @property
    def rendered_claims(self) -> tuple[str, ...]:
        by_claim = {item.claim_id: item for item in self.bindings}
        return tuple(
            f"{item.claim_text}[{item.citation_number}]"
            for item in self.bindings
            if item.claim_id in by_claim
        )


def bind_approved_evidence(
    claims: Iterable[ReportClaim],
    evidence_by_claim: dict[str, list[VerifiedEvidence]],
    registry: CitationRegistry,
) -> AutomaticCitationResult:
    """Bind already-verified evidence without any interactive approval."""

    document_numbers: dict[str, int] = {}
    reference_by_number: dict[int, str] = {}
    bindings: list[AutomaticCitationBinding] = []
    failures: list[AutomaticCitationFailure] = []

    for claim in claims:
        eligible = [
            item for item in evidence_by_claim.get(claim.claim_id, [])
            if item.formal_claim_support_eligible
        ]
        eligible.sort(key=lambda item: item.retrieval_score, reverse=True)
        selected = None
        for item in eligible:
            if not item.document_id or not item.chunk_id or not item.page_number:
                continue
            binding = registry.verify_claim_binding(
                document_id=item.document_id,
                chunk_ids=[item.chunk_id],
                page_number=item.page_number,
                supporting_excerpt=item.text,
            )
            if binding.valid:
                selected = item
                break

        if selected is None:
            failures.append(AutomaticCitationFailure(
                claim_id=claim.claim_id,
                reason="NO_VERIFIED_CITATION_FOUND",
            ))
            continue

        document_id = str(selected.document_id)
        number = document_numbers.get(document_id)
        if number is None:
            number = len(document_numbers) + 1
            document_numbers[document_id] = number
            reference_by_number[number] = registry.formal_citation(document_id)

        bindings.append(AutomaticCitationBinding(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            citation_number=number,
            document_id=document_id,
            chunk_id=str(selected.chunk_id),
            page_number=int(selected.page_number),
            supporting_excerpt=selected.text,
            formatted_citation=reference_by_number[number],
        ))

    references = tuple(
        f"[{number}] {reference_by_number[number]}"
        for number in sorted(reference_by_number)
    )
    return AutomaticCitationResult(
        bindings=tuple(bindings),
        failures=tuple(failures),
        references=references,
    )


class AutomaticCitationPipeline:
    """Local approved-RAG -> semantic gate -> deterministic citation binding."""

    def __init__(self, rag_root: str | Path, *, top_k: int = 8):
        self.rag_root = Path(rag_root).resolve()
        self.top_k = top_k
        self.source = LocalRAGSource(rag_root=self.rag_root, top_k=top_k)
        self.registry = CitationRegistry(self.rag_root)
        self.traceability = EvidenceTraceabilityVerifier(self.rag_root)

    @staticmethod
    def _problem(claim: ReportClaim) -> ResearchProblemSpec:
        return ResearchProblemSpec(
            problem_id=f"CITATION-{claim.claim_id}",
            original_question=claim.text,
            research_object="工业锅炉科研报告",
            target_variable="报告主张的外部文献依据",
            operating_condition="报告引用检索",
            manipulated_variables=[],
            observed_variables=[],
            context_variables=[claim.citation_purpose],
            research_goal=f"为该报告主张寻找直接支持文献：{claim.text}",
            success_criteria=["仅返回真实、已批准、直接且高度适用的引用"],
            constraints=["文献不得替代当前实验结论", "找不到时返回无引用"],
        )

    def run(self, claims: Iterable[ReportClaim]) -> AutomaticCitationResult:
        claim_list = list(claims)
        evidence_by_claim: dict[str, list[VerifiedEvidence]] = {}
        judge = QwenSemanticEvidenceJudge()
        try:
            for claim in claim_list:
                problem = self._problem(claim)
                candidates = self.source.retrieve(problem)
                traces = {
                    item.evidence_id: self.traceability.verify(item)
                    for item in candidates
                }
                assessments = []
                for start in range(0, len(candidates), 4):
                    assessments.extend(judge.judge(problem, candidates[start:start + 4]))
                verification = verify_from_assessments(
                    candidates,
                    traces,
                    {item.evidence_id: item for item in assessments},
                )
                evidence_by_claim[claim.claim_id] = list(verification.verified)
        finally:
            judge.close()
        return bind_approved_evidence(claim_list, evidence_by_claim, self.registry)
