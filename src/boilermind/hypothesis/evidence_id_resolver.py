from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


_PREFIX_BOUNDARIES = frozenset({"_", "-", ":", "/", "#"})


@dataclass(frozen=True)
class EvidenceIdResolution:
    requested_id: str
    resolved_id: str | None
    status: str
    candidates: tuple[str, ...] = ()


def resolve_evidence_id(
    requested_id: str,
    available_ids: Iterable[str],
) -> EvidenceIdResolution:
    """Resolve a reference exactly or by one boundary-aligned prefix."""

    requested = str(requested_id).strip()
    available = tuple(dict.fromkeys(str(item).strip() for item in available_ids))
    if requested in available:
        return EvidenceIdResolution(requested, requested, "exact", (requested,))

    matches = tuple(
        item
        for item in available
        if item.startswith(requested)
        and len(item) > len(requested)
        and item[len(requested)] in _PREFIX_BOUNDARIES
    )
    if len(matches) == 1:
        return EvidenceIdResolution(requested, matches[0], "unique_prefix", matches)
    if len(matches) > 1:
        return EvidenceIdResolution(requested, None, "ambiguous", matches)
    return EvidenceIdResolution(requested, None, "unknown")


def normalize_evidence_ids(
    requested_ids: Iterable[str],
    available_ids: Iterable[str],
) -> list[str]:
    """Canonicalize only safe references; preserve invalid IDs for gates."""

    available = tuple(available_ids)
    normalized: list[str] = []
    for requested in requested_ids:
        resolution = resolve_evidence_id(requested, available)
        normalized.append(resolution.resolved_id or resolution.requested_id)
    return list(dict.fromkeys(normalized))
