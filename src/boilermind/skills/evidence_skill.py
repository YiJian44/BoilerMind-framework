from __future__ import annotations

import os
from typing import Any

from boilermind.core.contracts import (
    ResearchProblemSpec,
)

from boilermind.evidence.bundle_freezer import (
    freeze_evidence_bundle,
)

from boilermind.evidence.qwen_semantic_judge import (
    QwenSemanticEvidenceJudge,
)

from boilermind.evidence.retrieval_pipeline import (
    ScientificRetrievalPipeline,
)

from boilermind.evidence.sources.local_rag import (
    LocalRAGSource,
)

from boilermind.evidence.sources.web_literature import (
    WebLiteratureSource,
)

from boilermind.evidence.traceability_verifier import (
    EvidenceTraceabilityVerifier,
)

from boilermind.evidence.verification_pipeline import (
    verify_from_assessments,
)

from .base import BaseSkill


class EvidenceRetrievalSkill(BaseSkill):

    name = "evidence_retrieval"

    description = (
        "从真实本地锅炉文献库及可选联网文献中检索、"
        "验证并冻结科研证据"
    )


    @staticmethod
    def _env_flag(
        name: str,
        default: bool,
    ) -> bool:

        raw = os.getenv(
            name,
            "1" if default else "0",
        )

        return (
            str(raw)
            .strip()
            .lower()
            not in {
                "0",
                "false",
                "no",
                "off",
            }
        )


    def execute(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:

        problem_payload = context.get(
            "research_problem"
        )

        if not problem_payload:
            raise ValueError(
                "research_problem_required"
            )


        if isinstance(
            problem_payload,
            ResearchProblemSpec,
        ):
            problem = problem_payload

        elif isinstance(
            problem_payload,
            dict,
        ):
            problem = (
                ResearchProblemSpec
                .model_validate(
                    problem_payload
                )
            )

        else:
            raise TypeError(
                "research_problem_must_be_"
                "ResearchProblemSpec_or_dict"
            )


        local_top_k = int(
            os.getenv(
                "BOILERMIND_LOCAL_RAG_TOP_K",
                "8",
            )
        )

        web_top_k = int(
            os.getenv(
                "BOILERMIND_WEB_RAG_TOP_K",
                "6",
            )
        )


        sources = [
            LocalRAGSource(
                top_k=local_top_k
            )
        ]


        web_enabled = self._env_flag(
            "BOILERMIND_ENABLE_WEB_LITERATURE",
            True,
        )


        if web_enabled:

            sources.append(
                WebLiteratureSource(
                    top_k=web_top_k
                )
            )


        retrieval_pipeline = (
            ScientificRetrievalPipeline(
                sources=sources
            )
        )


        candidates = (
            retrieval_pipeline.retrieve(
                problem
            )
        )


        if not candidates:
            raise RuntimeError(
                "no_evidence_candidates_retrieved"
            )


        traceability_verifier = (
            EvidenceTraceabilityVerifier()
        )


        traceability_by_id = {

            candidate.evidence_id:
                traceability_verifier.verify(
                    candidate
                )

            for candidate in candidates
        }


        semantic_judge = (
            QwenSemanticEvidenceJudge()
        )


        try:

            semantic_assessments = (
                semantic_judge.judge(
                    problem,
                    candidates,
                )
            )

        finally:

            semantic_judge.close()


        semantic_by_id = {

            assessment.evidence_id:
                assessment

            for assessment
            in semantic_assessments
        }


        verification = (
            verify_from_assessments(
                candidates=candidates,
                traceability_by_id=(
                    traceability_by_id
                ),
                semantic_by_id=(
                    semantic_by_id
                ),
            )
        )


        verified = list(
            verification.verified
        )


        if not verified:
            raise RuntimeError(
                "no_verified_scientific_evidence"
            )


        bundle = (
            freeze_evidence_bundle(
                problem_id=(
                    problem.problem_id
                ),
                evidence=verified,
            )
        )


        bundle_payload = (
            bundle.model_dump(
                mode="json"
            )
        )


        summary = {

            "problem_id":
                problem.problem_id,

            "web_literature_enabled":
                web_enabled,

            "retrieved_count":
                len(candidates),

            "verified_count":
                len(
                    verification.verified
                ),

            "rejected_count":
                len(
                    verification.rejected
                ),

            "core_claim_eligible_count":
                sum(
                    1
                    for item
                    in verification.verified
                    if item.core_claim_eligible
                ),

            "verified_evidence_ids": [
                item.evidence_id
                for item
                in verification.verified
            ],

            "rejected": [
                {
                    "evidence_id":
                        item.evidence_id,

                    "reason":
                        item.reason,
                }
                for item
                in verification.rejected
            ],

            "bundle_id":
                bundle.bundle_id,

            "bundle_sha256":
                bundle.sha256,
        }


        return {

            "evidence_bundle":
                bundle_payload,

            "evidence_retrieval_summary":
                summary,

            "status":
                "completed",
        }
