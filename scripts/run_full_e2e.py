from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boilermind.core.contracts import ResearchRequest  # noqa: E402
from boilermind.orchestration import ResearchOrchestrator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="BoilerMind unified research pipeline")
    parser.add_argument("--question", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    artifact = ResearchOrchestrator().run(ResearchRequest(question=args.question, run_id=args.run_id))
    print(json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if artifact.status in {"COMPLETED", "COMPLETED_WITH_REPORT_WARNING", "NO_EXECUTABLE_HYPOTHESES"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
