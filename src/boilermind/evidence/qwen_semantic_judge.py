from __future__ import annotations

import json
import os

import httpx2

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

from pydantic import Field, field_validator

from boilermind.core.contracts import (
    ContractModel,
    EvidenceCandidate,
    ResearchProblemSpec,
)

from boilermind.core.enums import (
    ApplicabilityLevel,
    ClaimSupport,
)


class SemanticEvidenceAssessment(
    ContractModel
):
    evidence_id: str = Field(
        min_length=1
    )

    semantic_verified: bool

    claim_support: ClaimSupport

    applicability: ApplicabilityLevel

    core_claim_eligible: bool

    verification_rationale: str = Field(
        min_length=1
    )


    @field_validator(
        "claim_support",
        mode="before",
    )
    @classmethod
    def normalize_claim_support(
        cls,
        value,
    ):
        if isinstance(value, str):
            return value.strip().lower()

        return value

    @field_validator(
        "applicability",
        mode="before",
    )
    @classmethod
    def normalize_applicability(
        cls,
        value,
    ):
        if isinstance(value, str):
            return value.strip().lower()

        return value


class QwenSemanticJudgeError(
    RuntimeError
):
    pass


class QwenSemanticEvidenceJudge:
    """
    Semantic scientific evidence judge.

    Responsibilities:
    - judge scientific semantic relevance;
    - judge claim support;
    - judge applicability;
    - judge whether evidence may support a core claim.

    Non-responsibilities:
    - source existence verification;
    - DOI/PDF authenticity verification;
    - source hash verification;
    - retrieval.

    Network transport:
    - OpenAI-compatible SDK;
    - explicit direct HTTP client;
    - trust_env=False to prevent unintended
      system/environment proxy routing.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 90.0,
        max_chars_per_candidate: int = 5000,
    ):
        self.api_key = (
            api_key
            or os.getenv(
                "DASHSCOPE_API_KEY"
            )
        )

        self.base_url = (
            base_url
            or os.getenv(
                "BOILERMIND_QWEN_BASE_URL"
            )
        )

        self.model = (
            model
            or os.getenv(
                "BOILERMIND_QWEN_MODEL",
                "qwen3.7-plus",
            )
        )

        self.timeout = timeout

        self.max_chars_per_candidate = (
            max_chars_per_candidate
        )

        if not self.api_key:
            raise QwenSemanticJudgeError(
                "DASHSCOPE_API_KEY "
                "is not configured."
            )

        if not self.base_url:
            raise QwenSemanticJudgeError(
                "BOILERMIND_QWEN_BASE_URL "
                "is not configured."
            )

        #
        # IMPORTANT:
        # Direct connection is intentional.
        #
        # The current machine has already verified:
        #
        #   OpenAI SDK
        #   + httpx2.Client(trust_env=False)
        #   -> Qwen response OK
        #
        # Therefore the scientific runtime must not
        # silently inherit proxy settings.
        #
        self._http_client = httpx2.Client(
            trust_env=False,
            timeout=self.timeout,
        )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self._http_client,
            timeout=self.timeout,
            max_retries=0,
        )
        self.audit_events: list[dict[str, object]] = []

    def _build_payload(
        self,
        problem: ResearchProblemSpec,
        candidates: list[
            EvidenceCandidate
        ],
    ) -> dict:
        evidence_payload = []

        for candidate in candidates:
            evidence_payload.append(
                {
                    "evidence_id": (
                        candidate.evidence_id
                    ),
                    "source_type": (
                        candidate.source_type
                    ),
                    "title": (
                        candidate.title
                    ),
                    "citation": (
                        candidate.citation
                    ),
                    "text": (
                        candidate.text[
                            :
                            self.max_chars_per_candidate
                        ]
                    ),
                }
            )

        problem_payload = {
            "problem_id": (
                problem.problem_id
            ),
            "original_question": (
                problem.original_question
            ),
            "research_object": (
                problem.research_object
            ),
            "target_variable": (
                problem.target_variable
            ),
            "operating_condition": (
                problem.operating_condition
            ),
            "manipulated_variables": (
                problem.manipulated_variables
            ),
            "observed_variables": (
                problem.observed_variables
            ),
            "context_variables": (
                problem.context_variables
            ),
            "research_goal": (
                problem.research_goal
            ),
        }

        system_prompt = """
You are the Semantic Evidence Judge in a trusted
scientific AI system for industrial process research.

You receive:
1. one CURRENT ResearchProblemSpec;
2. multiple retrieved scientific literature candidates.

Your ONLY responsibility is scientific semantic
assessment.

Do not verify whether a DOI, PDF, URL or source exists.
That has already been handled by a deterministic
traceability verifier.

Do not treat retrieval rank as scientific confidence.

Do not mark evidence relevant merely because several
keywords overlap.

For each candidate independently evaluate:

SEMANTIC_VERIFIED

True only when the scientific content has meaningful
relevance to the current research problem.

CLAIM_SUPPORT

DIRECT:
The evidence directly studies the target phenomenon,
mechanism, variable relationship, prediction task,
experimental relationship, or a very closely matched
scientific problem.

PARTIAL:
The evidence provides scientifically useful mechanism,
method, variable relationship, or transferable findings,
but does not directly study the complete target problem.

CONTRADICTING:
The evidence is scientifically relevant but provides
evidence against a relevant scientific claim.

IRRELEVANT:
Keyword overlap may exist, but the actual scientific
object, mechanism, task, or domain is unrelated.

UNKNOWN:
The supplied evidence text is insufficient to make a
scientific judgment.

APPLICABILITY

HIGH:
Strong match to research object, task, variables,
mechanism or operating conditions.

MEDIUM:
Scientifically transferable but with a meaningful
object, condition, variable or task mismatch.

LOW:
Large scientific applicability gap.

UNKNOWN:
Insufficient information.

CORE_CLAIM_ELIGIBLE

True only when ALL are satisfied:
- semantic_verified is true;
- claim_support is DIRECT or PARTIAL;
- applicability is HIGH or MEDIUM;
- the evidence is sufficiently substantive to support
  a later hypothesis mechanism.

CONTRADICTING, IRRELEVANT and UNKNOWN evidence must
never be core_claim_eligible.

Return one assessment for EVERY input evidence_id.

Do not invent evidence IDs.
Do not omit evidence IDs.
Do not duplicate evidence IDs.

Return JSON only.
No markdown.
""".strip()

        user_payload = {
            "research_problem": (
                problem_payload
            ),
            "evidence_candidates": (
                evidence_payload
            ),
            "required_output": {
                "assessments": [
                    {
                        "evidence_id": (
                            "exact input evidence_id"
                        ),
                        "semantic_verified": True,
                        "claim_support": (
                            "direct | partial | "
                            "contradicting | "
                            "irrelevant | unknown"
                        ),
                        "applicability": (
                            "high | medium | "
                            "low | unknown"
                        ),
                        "core_claim_eligible": True,
                        "verification_rationale": (
                            "concise scientific reason"
                        ),
                    }
                ]
            },
        }

        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "Evaluate every evidence "
                        "candidate against the CURRENT "
                        "research problem.\n"
                        + json.dumps(
                            user_payload,
                            ensure_ascii=False,
                        )
                    ),
                },
            ],
            "response_format": {
                "type": "json_object"
            },
            "temperature": 0.0,
        }

    @staticmethod
    def _parse_json_content(
        content: str,
    ) -> dict:
        text = content.strip()

        #
        # Defensive cleanup only.
        # response_format=json_object should normally
        # prevent markdown fences.
        #
        if text.startswith("```"):
            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip().startswith(
                    "```"
                )
            ):
                lines = lines[:-1]

            text = "\n".join(
                lines
            ).strip()

        return json.loads(
            text
        )

    @staticmethod
    def _validate_assessment_set(candidates, assessments) -> None:
        expected_ids = {item.evidence_id for item in candidates}
        returned_ids = [item.evidence_id for item in assessments]
        if len(returned_ids) != len(set(returned_ids)):
            raise ValueError("duplicate evidence IDs")
        if set(returned_ids) != expected_ids:
            raise ValueError("evidence IDs do not match input candidates")
        for item in assessments:
            if not item.semantic_verified and item.core_claim_eligible:
                raise ValueError("semantically rejected evidence marked core")
            if item.claim_support not in {ClaimSupport.DIRECT, ClaimSupport.PARTIAL} and item.core_claim_eligible:
                raise ValueError("non-supporting evidence marked core")
            if item.applicability not in {ApplicabilityLevel.HIGH, ApplicabilityLevel.MEDIUM} and item.core_claim_eligible:
                raise ValueError("low-applicability evidence marked core")

    def judge(
        self,
        problem: ResearchProblemSpec,
        candidates: list[
            EvidenceCandidate
        ],
    ) -> list[
        SemanticEvidenceAssessment
    ]:
        if not candidates:
            return []

        payload = self._build_payload(
            problem,
            candidates,
        )

        messages = list(payload["messages"])
        assessments = None
        for attempt in (1, 2):
            try:
                completion = self._client.chat.completions.create(
                    model=payload["model"], messages=messages,
                    response_format=payload["response_format"],
                    temperature=payload["temperature"],
                    extra_body={"enable_thinking": False},
                )
            except APITimeoutError as exc:
                raise QwenSemanticJudgeError(
                    "Qwen semantic evidence request timed out."
                ) from exc
            except APIConnectionError as exc:
                raise QwenSemanticJudgeError(
                    f"Qwen semantic evidence connection failed: {exc}"
                ) from exc
            except APIStatusError as exc:
                raise QwenSemanticJudgeError(
                    f"Qwen semantic evidence request returned HTTP {exc.status_code}."
                ) from exc
            except Exception as exc:
                raise QwenSemanticJudgeError(
                    f"Qwen semantic evidence request failed: {exc}"
                ) from exc

            try:
                content = completion.choices[0].message.content
                if not content:
                    raise ValueError("empty response content")
                parsed = self._parse_json_content(content)
                raw_assessments = parsed["assessments"]
                if not isinstance(raw_assessments, list):
                    raise ValueError("assessments is not a list")
                assessments = [SemanticEvidenceAssessment(**item) for item in raw_assessments]
                self._validate_assessment_set(candidates, assessments)
            except Exception as exc:
                self.audit_events.append({
                    "operation": "semantic_contract_validation",
                    "attempt": attempt,
                    "status": "retrying" if attempt == 1 else "failed",
                    "error_type": type(exc).__name__,
                })
                if attempt == 2:
                    raise QwenSemanticJudgeError(
                        "Qwen semantic assessment violates the scientific contract after one controlled retry."
                    ) from exc
                messages = messages + [{
                    "role": "system",
                    "content": (
                        "Your previous response violated the required JSON scientific contract. "
                        "Correct the structure and enum values, return exactly the same input evidence IDs, "
                        "and return JSON only."
                    ),
                }]
                continue
            else:
                self.audit_events.append({
                    "operation": "semantic_contract_validation",
                    "attempt": attempt,
                    "status": "accepted",
                    "error_type": None,
                })
                break

        assert assessments is not None

        expected_ids = {
            item.evidence_id
            for item in candidates
        }

        returned_ids = [
            item.evidence_id
            for item in assessments
        ]

        if (
            len(returned_ids)
            != len(
                set(returned_ids)
            )
        ):
            raise QwenSemanticJudgeError(
                "Qwen returned duplicate "
                "evidence IDs."
            )

        returned_id_set = set(
            returned_ids
        )

        if (
            returned_id_set
            != expected_ids
        ):
            missing = (
                expected_ids
                - returned_id_set
            )

            unexpected = (
                returned_id_set
                - expected_ids
            )

            raise QwenSemanticJudgeError(
                "Qwen evidence IDs do not "
                "match input candidates. "
                f"Missing={missing}; "
                f"Unexpected={unexpected}"
            )

        for item in assessments:
            if (
                not item.semantic_verified
                and item.core_claim_eligible
            ):
                raise QwenSemanticJudgeError(
                    "Semantically rejected evidence "
                    "cannot be core-claim eligible."
                )

            if (
                item.claim_support
                not in {
                    ClaimSupport.DIRECT,
                    ClaimSupport.PARTIAL,
                }
                and item.core_claim_eligible
            ):
                raise QwenSemanticJudgeError(
                    "Only DIRECT/PARTIAL evidence "
                    "may be core-claim eligible."
                )

            if (
                item.applicability
                not in {
                    ApplicabilityLevel.HIGH,
                    ApplicabilityLevel.MEDIUM,
                }
                and item.core_claim_eligible
            ):
                raise QwenSemanticJudgeError(
                    "Only HIGH/MEDIUM applicability "
                    "evidence may be core-claim "
                    "eligible."
                )

        return assessments

    def close(self) -> None:
        self._client.close()
