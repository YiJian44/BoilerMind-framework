import json
from datetime import datetime, timezone
from types import SimpleNamespace

from boilermind.core.contracts import (
    EvidenceCandidate,
    ResearchProblemSpec,
)

from boilermind.core.enums import (
    ApplicabilityLevel,
    ClaimSupport,
)

from boilermind.evidence.qwen_semantic_judge import (
    QwenSemanticEvidenceJudge,
    SemanticEvidenceAssessment,
)


def test_semantic_assessment_contract():
    result = SemanticEvidenceAssessment(
        evidence_id="E-001",
        semantic_verified=True,
        claim_support=(
            ClaimSupport.DIRECT
        ),
        applicability=(
            ApplicabilityLevel.HIGH
        ),
        core_claim_eligible=True,
        verification_rationale=(
            "Directly studies the target "
            "industrial prediction problem."
        ),
    )

    assert result.semantic_verified is True
    assert (
        result.claim_support
        == ClaimSupport.DIRECT
    )


def test_irrelevant_assessment_is_not_core():
    result = SemanticEvidenceAssessment(
        evidence_id="E-002",
        semantic_verified=False,
        claim_support=(
            ClaimSupport.IRRELEVANT
        ),
        applicability=(
            ApplicabilityLevel.LOW
        ),
        core_claim_eligible=False,
        verification_rationale=(
            "Keyword overlap only; unrelated domain."
        ),
    )

    assert result.core_claim_eligible is False


def test_contract_error_gets_exactly_one_controlled_retry():
    valid = json.dumps({"assessments": [{
        "evidence_id": "E-RETRY",
        "semantic_verified": True,
        "claim_support": "direct",
        "applicability": "high",
        "core_claim_eligible": True,
        "verification_rationale": "Direct support.",
    }]})
    responses = iter([
        '{"assessments":[{"evidence_id":"E-RETRY","semantic_verified":true,'
        '"claim_support":"low","applicability":"high",'
        '"core_claim_eligible":false,"verification_rationale":"bad enum"}]}',
        valid,
    ])

    class FakeCompletions:
        calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses))
            )])

    judge = QwenSemanticEvidenceJudge.__new__(QwenSemanticEvidenceJudge)
    completions = FakeCompletions()
    judge._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    judge.model = "test-model"
    judge.max_chars_per_candidate = 5000
    judge.audit_events = []
    problem = ResearchProblemSpec(
        problem_id="P-RETRY",
        original_question="测试问题",
        research_object="锅炉",
        target_variable="蒸汽流量",
        operating_condition="深度调峰",
        research_goal="验证证据",
        success_criteria=["准确"],
    )
    candidate = EvidenceCandidate(
        evidence_id="E-RETRY",
        problem_id="P-RETRY",
        source_type="local_literature",
        title="Test",
        text="Substantive evidence text.",
        retrieval_score=0.5,
        retrieved_at=datetime.now(timezone.utc),
    )

    result = judge.judge(problem, [candidate])

    assert result[0].claim_support == ClaimSupport.DIRECT
    assert completions.calls == 2
    assert [item["status"] for item in judge.audit_events] == ["retrying", "accepted"]
