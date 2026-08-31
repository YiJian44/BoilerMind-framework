from __future__ import annotations

import json
from pathlib import Path

from boilermind.core.contracts import ResearchProblemSpec
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


#
# TEST-ONLY integration problem.
#
# This is NOT the future production entry.
# Production will use:
#
# user question
# -> QwenProblemParser
# -> ResearchProblemSpec
#
problem = ResearchProblemSpec(
    problem_id="P-EVIDENCE-SMOKE-001",
    original_question=(
        "How do delayed relations among multivariate "
        "boiler variables affect prediction performance?"
    ),
    research_object="boiler plant",
    target_variable="prediction performance",
    operating_condition="dynamic operation",
    manipulated_variables=[],
    observed_variables=[
        "multivariate boiler variables"
    ],
    context_variables=[
        "delayed relations",
        "time delay",
    ],
    research_goal=(
        "Identify scientifically relevant evidence "
        "about delayed multivariate relations and "
        "boiler prediction performance."
    ),
    success_criteria=[
        "retrieve traceable scientific evidence",
        "reject semantically irrelevant evidence",
        "freeze verified scientific evidence",
    ],
    constraints=[
        "scientific sources must be traceable"
    ],
)


pipeline = ScientificRetrievalPipeline(
    [
        LocalRAGSource(
            top_k=8,
        ),
        WebLiteratureSource(
            crossref_results=5,
            arxiv_results=5,
            top_k=8,
        ),
    ]
)


print("=" * 72)
print("STEP 1 - RETRIEVAL")
print("=" * 72)

candidates = pipeline.retrieve(
    problem
)

print(
    "Retrieved candidates:",
    len(candidates),
)

for index, candidate in enumerate(
    candidates,
    start=1,
):
    print(
        f"{index:02d}. "
        f"{candidate.evidence_id} | "
        f"{candidate.source_type} | "
        f"{candidate.title}"
    )


print()
print("=" * 72)
print("STEP 2 - TRACEABILITY")
print("=" * 72)

traceability_verifier = (
    EvidenceTraceabilityVerifier()
)

traceability_by_id = {}

for candidate in candidates:
    result = (
        traceability_verifier.verify(
            candidate
        )
    )

    traceability_by_id[
        candidate.evidence_id
    ] = result

    print(
        f"{candidate.evidence_id} | "
        f"traceable={result.verified} | "
        f"{result.rationale}"
    )

traceable_count = sum(
    result.verified
    for result
    in traceability_by_id.values()
)

print(
    "Traceable:",
    traceable_count,
    "/",
    len(candidates),
)


print()
print("=" * 72)
print("STEP 3 - QWEN SEMANTIC VERIFICATION")
print("=" * 72)

semantic_by_id = {}

judge = QwenSemanticEvidenceJudge()

#
# Keep individual Qwen requests reasonably small.
# This avoids coupling the evidence verifier to one
# very large LLM context.
#
batch_size = 5

try:
    for start in range(
        0,
        len(candidates),
        batch_size,
    ):
        batch = candidates[
            start:
            start + batch_size
        ]

        assessments = judge.judge(
            problem,
            batch,
        )

        for assessment in assessments:
            semantic_by_id[
                assessment.evidence_id
            ] = assessment

            print(
                f"{assessment.evidence_id} | "
                f"semantic="
                f"{assessment.semantic_verified} | "
                f"support="
                f"{assessment.claim_support.value} | "
                f"applicability="
                f"{assessment.applicability.value} | "
                f"core="
                f"{assessment.core_claim_eligible}"
            )

            print(
                "    ",
                assessment.verification_rationale,
            )

finally:
    judge.close()


print()
print("=" * 72)
print("STEP 4 - VERIFIED EVIDENCE")
print("=" * 72)

verification = verify_from_assessments(
    candidates,
    traceability_by_id,
    semantic_by_id,
)

print(
    "Decisions:",
    len(verification.decisions),
)

print(
    "Verified:",
    len(verification.verified),
)

print(
    "Rejected:",
    len(verification.rejected),
)


if verification.rejected:
    print()
    print("Rejected evidence:")

    for rejection in verification.rejected:
        print(
            f"- {rejection.evidence_id}: "
            f"{rejection.reason}"
        )


if not verification.verified:
    raise RuntimeError(
        "No evidence survived verification. "
        "EvidenceBundle will not be frozen."
    )


print()
print("=" * 72)
print("STEP 5 - FREEZE EVIDENCE BUNDLE")
print("=" * 72)

bundle = freeze_evidence_bundle(
    problem.problem_id,
    list(
        verification.verified
    ),
)

output_dir = Path(
    "runtime/evidence_bundles"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = (
    output_dir
    / f"{bundle.bundle_id}.json"
)

output_path.write_text(
    json.dumps(
        bundle.model_dump(
            mode="json"
        ),
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "Bundle ID:",
    bundle.bundle_id,
)

print(
    "Problem ID:",
    bundle.problem_id,
)

print(
    "Verified evidence:",
    len(bundle.evidence),
)

print(
    "SHA256:",
    bundle.sha256,
)

print(
    "Saved:",
    output_path.resolve(),
)

print()
print("EVIDENCE_PIPELINE_SMOKE_OK")