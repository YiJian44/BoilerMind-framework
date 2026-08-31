from __future__ import annotations

import re


_REFERENCE_HEADING = re.compile(
    r"(?:^|\n)\s*(?:references|bibliography|参考文献)\s*(?:\n|$)",
    re.IGNORECASE,
)
_NUMBERED_REFERENCE = re.compile(r"(?:^|\n)\s*\[\d{1,3}\]\s+")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_DOI_OR_JOURNAL = re.compile(
    r"\b(?:doi\s*:|https?://doi\.org/|vol\.?|pp\.?|journal|proceedings)\b",
    re.IGNORECASE,
)


def is_reference_list_only(text: str) -> bool:
    """Conservatively identify bibliography/reference-list chunks."""

    normalized = text.replace("\r\n", "\n").strip()
    numbered = len(_NUMBERED_REFERENCE.findall(normalized))
    citation_signals = len(_YEAR.findall(normalized)) + len(
        _DOI_OR_JOURNAL.findall(normalized)
    )
    return bool(
        (_REFERENCE_HEADING.search(normalized) and numbered >= 2)
        or (numbered >= 3 and citation_signals >= 3)
    )
