"""Coverage check for the frontend i18n table.

BoilerMind backend exposes a number of StrEnum value families
(research_run status, scientific verdict, hypothesis status, ...). The
frontend i18n.js must carry a Chinese label for every value the backend
can emit, otherwise the user sees a raw English string in the UI.

This test parses the backend enums directly and asserts the i18n table
has every value, so a new enum value added on the Python side will fail
here until a matching translation lands in JS.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENUM_FILE = ROOT / "src" / "boilermind" / "core" / "enums.py"
I18N_FILE = ROOT / "frontend" / "js" / "i18n.js"


def _load_str_enums() -> dict[str, set[str]]:
    """Pull every `class Foo(StrEnum)` body out of enums.py and extract its values."""
    text = ENUM_FILE.read_text(encoding="utf-8")
    pattern = re.compile(
        r"class\s+(\w+)\s*\(\s*StrEnum\s*\)\s*:\s*(?P<body>.*?)(?=^\s*class\s+\w+\s*\(|\Z)",
        re.M | re.S,
    )
    out: dict[str, set[str]] = {}
    for match in pattern.finditer(text):
        cls_name = match.group(1)
        body = match.group("body")
        values: set[str] = set()
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assignment = re.match(r"^([A-Z][A-Z0-9_]*)\s*=\s*[\"']([^\"']+)[\"']", stripped)
            if assignment:
                values.add(assignment.group(2))
        out[cls_name] = values
    return out


def _load_i18n_module() -> dict[str, dict[str, str]]:
    """Parse i18n.js into Python dicts by evaluating the const declarations.

    Avoids Node dependency; we only need the literal string maps. Comments
    and non-map declarations are tolerated by extracting the first balanced
    brace block after each `const NAME = {`.
    """
    text = I18N_FILE.read_text(encoding="utf-8")
    pattern = re.compile(
        r"const\s+(\w+)\s*=\s*\{(?P<body>[^}]*)\}\s*;",
        re.S,
    )
    out: dict[str, dict[str, str]] = {}
    for match in pattern.finditer(text):
        name = match.group(1)
        body = match.group("body")
        pairs = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*['\"]([^'\"]*)['\"]", body)
        if pairs:
            out[name] = {key: value for key, value in pairs}
    return out


ENUM_TO_I18N = {
    "ResearchRunStatus": "RESEARCH_RUN_STATUS",
    "HypothesisStatus": "HYPOTHESIS_STATUS",
    "ScientificVerdict": "SCIENTIFIC_VERDICT",
    "EvidenceStage": "EVIDENCE_STAGE",
    "ClaimSupport": "CLAIM_SUPPORT",
    "ApplicabilityLevel": "APPLICABILITY_LEVEL",
    "ExperimentStatus": "EXPERIMENT_STATUS",
    "ResearchStopReason": "RESEARCH_STOP_REASON",
    "ProblemResolutionStatus": "PROBLEM_RESOLUTION_STATUS",
    "MechanismSupportType": "MECHANISM_SUPPORT_TYPE",
}


@pytest.fixture(scope="module")
def enums() -> dict[str, set[str]]:
    return _load_str_enums()


@pytest.fixture(scope="module")
def i18n() -> dict[str, dict[str, str]]:
    return _load_i18n_module()


@pytest.mark.parametrize(
    "enum_name, table_name",
    list(ENUM_TO_I18N.items()),
)
def test_i18n_covers_all_enum_values(enums, i18n, enum_name, table_name):
    backend_values = enums.get(enum_name, set())
    assert backend_values, f"{enum_name} not parsed from backend enums.py"
    table = i18n.get(table_name, {})
    assert table, f"{table_name} not parsed from frontend i18n.js"
    missing = sorted(backend_values - set(table.keys()))
    assert not missing, (
        f"i18n {table_name} is missing Chinese labels for: {missing}. "
        f"Add them in frontend/js/i18n.js so users don't see raw English."
    )


def test_i18n_table_parsable_and_nonempty(i18n):
    # Sanity: every table mentioned by ENUM_TO_I18N must be present and have entries.
    for table_name in ENUM_TO_I18N.values():
        assert table_name in i18n, f"{table_name} missing in i18n.js"
        assert i18n[table_name], f"{table_name} empty in i18n.js"


def test_run_stack_still_works():
    """Smoke check: ensure the launcher module file is syntactically valid Python."""
    import ast
    source = (ROOT / "scripts" / "run_stack.py").read_text(encoding="utf-8")
    ast.parse(source)  # raises SyntaxError if invalid
