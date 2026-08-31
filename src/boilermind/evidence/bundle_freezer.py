from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from boilermind.core.contracts import (
    EvidenceBundle,
    VerifiedEvidence,
)


class EvidenceBundleFreezeError(
    RuntimeError
):
    pass


def _canonical_payload(
    problem_id: str,
    evidence: list[VerifiedEvidence],
) -> dict:
    ordered = sorted(
        evidence,
        key=lambda item: item.evidence_id,
    )

    return {
        "problem_id": problem_id,
        "evidence": [
            item.model_dump(
                mode="json",
                exclude_none=False,
            )
            for item in ordered
        ],
    }


def compute_evidence_bundle_sha256(
    problem_id: str,
    evidence: list[VerifiedEvidence],
) -> str:
    """
    Content-addressed digest of the verified
    scientific evidence set.

    created_at and bundle_id are intentionally excluded
    so identical scientific evidence produces the same
    digest.
    """

    payload = _canonical_payload(
        problem_id,
        evidence,
    )

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def freeze_evidence_bundle(
    problem_id: str,
    evidence: list[VerifiedEvidence],
) -> EvidenceBundle:
    if not problem_id.strip():
        raise EvidenceBundleFreezeError(
            "problem_id must not be empty."
        )

    if not evidence:
        raise EvidenceBundleFreezeError(
            "Cannot freeze an empty evidence bundle."
        )

    for item in evidence:
        if item.problem_id != problem_id:
            raise EvidenceBundleFreezeError(
                "Verified evidence belongs to a "
                "different research problem: "
                f"{item.evidence_id}."
            )

    ordered = sorted(
        evidence,
        key=lambda item: item.evidence_id,
    )

    digest = compute_evidence_bundle_sha256(
        problem_id,
        ordered,
    )

    return EvidenceBundle(
        bundle_id=f"EB-{digest[:16]}",
        problem_id=problem_id,
        evidence=ordered,
        created_at=datetime.now(
            timezone.utc
        ),
        sha256=digest,
    )