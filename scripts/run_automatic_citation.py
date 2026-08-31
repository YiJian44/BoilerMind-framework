from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boilermind.evidence.automatic_citation import (  # noqa: E402
    AutomaticCitationPipeline,
    ReportClaim,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.claims_json).read_text(encoding="utf-8"))
    claims = [ReportClaim(**row) for row in rows]
    pipeline = AutomaticCitationPipeline(ROOT / "resources" / "local_rag")
    result = pipeline.run(claims)
    payload = {
        "schema_version": "boilermind.automatic-citations.v1",
        "bindings": [asdict(item) for item in result.bindings],
        "failures": [asdict(item) for item in result.failures],
        "references": list(result.references),
        "rendered_claims": list(result.rendered_claims),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "bound": len(result.bindings),
        "unresolved": len(result.failures),
        "references": len(result.references),
        "output": str(output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
