from __future__ import annotations

from dataclasses import dataclass


STEAM_VOLUMETRIC_FLOW = "steam_volumetric_flow"
MAIN_STEAM_MASS_FLOW = "main_steam_mass_flow"
UNSPECIFIED = "unspecified"


_VOLUME_ALIASES = (
    "蒸汽体积流量",
    "蒸汽体积量",
    "主蒸汽体积流量",
    "steam volumetric flow",
)

_MASS_ALIASES = (
    "蒸汽质量流量",
    "主蒸汽流量",
    "质量流量",
    "mass flow",
)


@dataclass(frozen=True)
class TargetNormalizationResult:
    raw_target_variable: str
    normalized_target_variable: str
    normalization_reason: str


def _alias_matches(text: str) -> set[str]:
    lowered = text.casefold()
    matches: set[str] = set()
    if any(alias in lowered for alias in _VOLUME_ALIASES):
        matches.add(STEAM_VOLUMETRIC_FLOW)
    if any(alias in lowered for alias in _MASS_ALIASES):
        matches.add(MAIN_STEAM_MASS_FLOW)
    return matches


def normalize_target_variable(
    raw_target_variable: object,
    problem_text: str,
) -> TargetNormalizationResult:
    """Normalize only targets represented by existing execution profiles.

    The raw field is authoritative when it is already canonical or contains a
    unique known alias. Otherwise the original user question may resolve a
    unique physical target. Task, horizon and operating-condition wording is
    never copied into the normalized target. Ambiguity remains fail-closed as
    ``unspecified``.
    """

    raw = (
        str(raw_target_variable).strip()
        if raw_target_variable is not None
        else ""
    )
    raw_lower = raw.casefold()
    if raw_lower in {STEAM_VOLUMETRIC_FLOW, MAIN_STEAM_MASS_FLOW}:
        return TargetNormalizationResult(
            raw_target_variable=raw,
            normalized_target_variable=raw_lower,
            normalization_reason="exact_canonical_target",
        )

    raw_matches = _alias_matches(raw)
    if len(raw_matches) == 1:
        return TargetNormalizationResult(
            raw_target_variable=raw,
            normalized_target_variable=next(iter(raw_matches)),
            normalization_reason="unique_alias_in_raw_target",
        )
    if len(raw_matches) > 1:
        return TargetNormalizationResult(
            raw_target_variable=raw,
            normalized_target_variable=UNSPECIFIED,
            normalization_reason="ambiguous_target_aliases_in_raw_target",
        )

    question_matches = _alias_matches(str(problem_text))
    if len(question_matches) == 1:
        return TargetNormalizationResult(
            raw_target_variable=raw,
            normalized_target_variable=next(iter(question_matches)),
            normalization_reason="unique_alias_in_original_question",
        )
    reason = (
        "ambiguous_target_aliases_in_original_question"
        if len(question_matches) > 1
        else "target_variable_resolution_failed"
    )
    return TargetNormalizationResult(
        raw_target_variable=raw,
        normalized_target_variable=UNSPECIFIED,
        normalization_reason=reason,
    )
