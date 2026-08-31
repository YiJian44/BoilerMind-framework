from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable


VALID_RELEVANCE_LABELS = {"RELEVANT", "PARTIAL", "IRRELEVANT", "UNJUDGEABLE"}
RELEVANCE_GAINS = {"RELEVANT": 2, "PARTIAL": 1, "IRRELEVANT": 0}


def evaluate_ranked_results(rows: Iterable[dict], *, expected_count: int) -> dict:
    items = list(rows)
    labels = [str(item.get("relevance_judgment") or "") for item in items]
    judged = [label for label in labels if label in RELEVANCE_GAINS]
    complete = len(items) == expected_count and len(judged) == expected_count
    payload = {
        "expected_count": expected_count,
        "retrieved_count": len(items),
        "judged_count": len(judged),
        "complete": complete,
        "strict_precision_at_k": None,
        "broad_precision_at_k": None,
        "graded_ndcg_at_k": None,
    }
    if not complete:
        return payload
    gains = [RELEVANCE_GAINS[label] for label in labels]
    payload["strict_precision_at_k"] = labels.count("RELEVANT") / expected_count
    payload["broad_precision_at_k"] = sum(gain > 0 for gain in gains) / expected_count
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sorted(gains, reverse=True)
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    payload["graded_ndcg_at_k"] = dcg / idcg if idcg else 0.0
    return payload


def summarize_query_evaluations(queries: Iterable[dict]) -> dict:
    items = list(queries)
    evaluations = [item["evaluation"] for item in items]
    labels = Counter(
        str(row.get("relevance_judgment") or "UNJUDGED")
        for item in items
        for row in item.get("results", [])
    )
    complete = bool(items) and all(item.get("complete") for item in evaluations)
    payload = {
        "query_count": len(items),
        "completed_query_count": sum(bool(item.get("complete")) for item in evaluations),
        "judgment_count": sum(int(item.get("judged_count") or 0) for item in evaluations),
        "label_counts": dict(sorted(labels.items())),
        "complete": complete,
        "macro_strict_precision_at_k": None,
        "macro_broad_precision_at_k": None,
        "macro_graded_ndcg_at_k": None,
    }
    if not complete:
        return payload
    count = len(evaluations)
    payload["macro_strict_precision_at_k"] = sum(
        item["strict_precision_at_k"] for item in evaluations
    ) / count
    payload["macro_broad_precision_at_k"] = sum(
        item["broad_precision_at_k"] for item in evaluations
    ) / count
    payload["macro_graded_ndcg_at_k"] = sum(
        item["graded_ndcg_at_k"] for item in evaluations
    ) / count
    return payload
