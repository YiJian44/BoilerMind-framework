"""两类知识图谱的确定性构建与读取：

1. literature —— 基于 resources/local_rag 本地文献库（论文/作者/主题/语料层级）。
2. evolution —— 基于已验证科研假设的可增长演化图谱
   （knowledge_graph/evolution/evolution_graph.json，含历史运行回放同步）。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_ROOT = PROJECT_ROOT / "resources" / "local_rag"
EVOLUTION_GRAPH = (
    PROJECT_ROOT / "knowledge_graph" / "evolution" / "evolution_graph.json"
)

TEAM_GRAPH = PROJECT_ROOT / "knowledge_graph" / "team" / "kg_snapshot.json"

_LEVEL_LABELS = {
    "core": "核心文献",
    "domain_support": "领域支撑",
    "method_support": "方法支撑",
}

_STOPWORDS = {
    "based", "using", "for", "the", "with", "and", "in", "of", "to", "a", "an",
    "via", "from", "on", "by", "model", "models", "prediction", "predicting",
    "predictive", "data", "driven", "boiler", "boilers", "steam", "power",
    "plant", "plants", "system", "systems", "method", "methods", "approach",
    "study", "analysis", "based", "toward", "towards", "improved", "improving",
    "efficient", "efficiency", "optimization", "optimizationbased", "using",
    "real", "time", "industrial", "numerical", "comparative", "performance",
}


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def _title_tokens(title: str) -> list[str]:
    tokens: list[str] = []
    for match in re.findall(r"[A-Za-z][A-Za-z0-9-]*", title):
        token = match.lower()
        if len(token) >= 4 and token not in _STOPWORDS:
            tokens.append(token)
    for segment in re.findall(r"[\u4e00-\u9fff]+", title):
        if len(segment) == 1:
            continue
        if len(segment) <= 4:
            tokens.append(segment)
        else:
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


class _Graph:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self._node_ids: set[str] = set()
        self._edge_keys: set[tuple[str, str, str]] = set()

    def add_node(self, node: dict[str, Any]) -> None:
        node_id = str(node["id"])
        if node_id in self._node_ids:
            return
        self._node_ids.add(node_id)
        self.nodes.append(node)

    def add_edge(self, source: str, edge_type: str, target: str, **fields: Any) -> None:
        key = (source, edge_type, target)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append({"source": source, "type": edge_type, "target": target, **fields})


def build_literature_graph(
    rag_root: str | Path = RAG_ROOT,
    *,
    max_topics: int = 40,
    min_topic_frequency: int = 2,
) -> dict[str, Any]:
    """从本地文献库构建论文-作者-主题-语料层级图谱。"""
    root = Path(rag_root)
    identities = {
        str(record.get("document_id")): record
        for record in _load_jsonl(root / "metadata" / "literature_identity.jsonl")
        if record.get("document_id")
    }
    papers = [
        record
        for record in _load_jsonl(root / "metadata" / "papers.jsonl")
        if record.get("document_id")
    ]
    chunk_counts: Counter[str] = Counter()
    for chunk in _load_jsonl(root / "artifacts" / "chunks" / "chunks.jsonl"):
        document_id = str(chunk.get("document_id") or "")
        if document_id:
            chunk_counts[document_id] += 1

    graph = _Graph()
    for level in _LEVEL_LABELS:
        graph.add_node({
            "id": f"LVL-{level}",
            "type": "CorpusLevel",
            "name": _LEVEL_LABELS[level],
        })

    topic_counter: Counter[str] = Counter()
    for paper in papers:
        identity = identities.get(str(paper["document_id"])) or {}
        title = str(identity.get("title") or paper.get("title") or "")
        topic_counter.update(_title_tokens(title))
    selected_topics = {
        token
        for token, count in topic_counter.most_common(max_topics)
        if count >= min_topic_frequency
    }
    for token in selected_topics:
        graph.add_node({
            "id": _stable_id("TOP", token),
            "type": "Topic",
            "name": token,
            "count": topic_counter[token],
        })

    author_frequency: Counter[str] = Counter()
    for paper in papers:
        identity = identities.get(str(paper["document_id"])) or {}
        for author in identity.get("authors") or []:
            name = str(author.get("literal") or "").strip()
            if name:
                author_frequency[name] += 1

    for paper in papers:
        document_id = str(paper["document_id"])
        identity = identities.get(document_id) or {}
        title = str(identity.get("title") or paper.get("title") or document_id)
        level = str(
            paper.get("corpus_level") or identity.get("corpus_level") or "domain_support"
        )
        graph.add_node({
            "id": document_id,
            "type": "Paper",
            "title": title,
            "year": identity.get("issued_year") or paper.get("year") or 0,
            "corpus_level": level,
            "language": identity.get("language"),
            "publication_type": identity.get("publication_type"),
            "citation_eligibility": identity.get("citation_eligibility"),
            "doi": identity.get("doi") or paper.get("doi"),
            "chunk_count": chunk_counts.get(document_id, 0),
        })
        graph.add_edge(document_id, "belongs_to", f"LVL-{level}")
        for token in set(_title_tokens(title)):
            if token in selected_topics:
                graph.add_edge(document_id, "about", _stable_id("TOP", token))
        for author in identity.get("authors") or []:
            name = str(author.get("literal") or "").strip()
            if name:
                graph.add_edge(_stable_id("AUT", name), "authored", document_id)

    for name, count in author_frequency.items():
        graph.add_node({
            "id": _stable_id("AUT", name),
            "type": "Author",
            "name": name,
            "paper_count": count,
        })

    return {
        "schema_version": "boilermind.literature_graph.v1",
        "summary": {
            "paper_count": len(papers),
            "author_count": len(author_frequency),
            "topic_count": len(selected_topics),
            "chunk_count": sum(chunk_counts.values()),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
        },
        "nodes": graph.nodes,
        "edges": graph.edges,
    }


def _read_graph(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        return {"nodes": [], "edges": []}
    graph = json.loads(path.read_text(encoding="utf-8"))
    return {
        "nodes": list(graph.get("nodes") or []),
        "edges": list(graph.get("edges") or []),
    }


def ensure_evolution_graph_synced(
    run_root: str | Path,
    graph_path: str | Path | None = None,
    *,
    force: bool = False,
) -> bool:
    """历史完成运行回放到演化图谱（幂等）；图谱比最新运行新则跳过。"""
    root = Path(run_root)
    path = Path(graph_path) if graph_path is not None else Path(EVOLUTION_GRAPH)
    newest_run_mtime = 0.0
    if root.is_dir():
        for run_dir in root.iterdir():
            run_json = run_dir / "run.json"
            if run_json.is_file():
                newest_run_mtime = max(newest_run_mtime, run_json.stat().st_mtime)
    graph_mtime = path.stat().st_mtime if path.is_file() else 0.0
    if not force and graph_mtime >= newest_run_mtime:
        return False

    from knowledge_graph.evolution.experiment_adapter import (
        update_evolution_from_results,
    )

    changed = False
    for run_dir in sorted(root.iterdir(), key=lambda item: item.name):
        run_json = run_dir / "run.json"
        if not run_json.is_file():
            continue
        try:
            state = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if state.get("status") not in {"COMPLETED", "COMPLETED_WITH_REPORT_WARNING"}:
            continue
        hypotheses_by_id = {
            str(item.get("hypothesis_id") or item.get("id")): item
            for item in (state.get("hypotheses") or [])
        }
        for batch in state.get("batches") or []:
            for member in batch.get("members") or []:
                if member.get("status") != "COMPLETED" or not member.get("outcome"):
                    continue
                outcome = member.get("outcome") or {}
                experiment_result = dict(outcome.get("experiment_result") or {})
                # 部分运行未在结果中显式写出 experiment_valid；以执行审计结论兜底，
                # 否则图谱写入会因字段类型不符被拒绝，导致演化图谱无法增长。
                if not isinstance(experiment_result.get("experiment_valid"), bool):
                    audit = outcome.get("audit") or {}
                    if isinstance(audit.get("execution_valid"), bool):
                        experiment_result["experiment_valid"] = audit["execution_valid"]
                hypothesis = hypotheses_by_id.get(str(member.get("hypothesis_id"))) or {}
                try:
                    update_evolution_from_results(
                        experiment_result,
                        outcome.get("scientific_result"),
                        hypothesis,
                        graph_path=path,
                    )
                    changed = True
                except Exception:
                    continue
    return changed


def load_evolution_graph(
    graph_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(graph_path) if graph_path is not None else Path(EVOLUTION_GRAPH)
    graph = _read_graph(path)
    node_types: Counter[str] = Counter()
    edge_types: Counter[str] = Counter()
    for node in graph["nodes"]:
        node_types[str(node.get("type") or "Unknown")] += 1
    for edge in graph["edges"]:
        edge_types[str(edge.get("type") or "Unknown")] += 1
    return {
        "schema_version": "boilermind.evolution_graph.v1",
        "summary": {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
        },
        "nodes": graph["nodes"],
        "edges": graph["edges"],
    }


_TEAM_NODE_GROUPS = {
    "BoilerEntity": "机理层",
    "Constraint": "机理层",
    "FaultGroup": "机理层",
    "FaultMode": "机理层",
    "Mechanism": "机理层",
    "ModelPrior": "机理层",
    "Principle": "机理层",
    "Regime": "机理层",
    "RootCause": "机理层",
    "SensorNoise": "机理层",
    "Symptom": "机理层",
    "TimeWindow": "机理层",
    "Variable": "变量层",
}

_TEAM_NODE_LABELS = {
    "Principle": "原理",
    "Mechanism": "机理",
    "Constraint": "约束",
    "FaultGroup": "故障组",
    "FaultMode": "故障模式",
    "Symptom": "现象",
    "RootCause": "根因",
    "BoilerEntity": "锅炉实体",
    "Regime": "工况",
    "TimeWindow": "时间窗口",
    "SensorNoise": "传感器扰动",
    "ModelPrior": "模型先验",
    "Variable": "变量",
}

_TEAM_EDGE_LABELS = {
    "BELONGS_TO": "归属",
    "GOVERNS": "支配",
    "CONSTRAINS": "约束",
    "RELATES": "关联",
    "CAUSES": "因果",
    "MANIFESTS_AS": "表现",
    "DRIVES": "因果",
    "TRIGGERS": "触发",
    "AFFECTS": "影响",
    "INTENSIFIES": "加剧",
    "INJECTS": "注入",
    "RAISES_RISK": "抬升风险",
    "HURTS": "抑制",
    "FEEDS": "供给",
    "DERIVED_FROM": "来源于",
    "RELEASES": "释放",
    "RELAXES": "缓解",
    "STORES": "存储",
    "PROVIDES": "提供",
    "REQUIRES": "依赖",
    "SENSITIVE_TO": "敏感于",
    "SUITED_FOR": "适用于",
    "APPLIES_TO": "作用于",
    "RESISTANT_TO": "抵抗",
    "BENEFITS": "促进",
    "CORRELATES_WITH": "相关",
}


def _normalize_team_node(node: dict[str, Any]) -> dict[str, Any]:
    props = dict(node.get("props") or {})
    label = str(node.get("label") or "Unknown")
    key = node.get("key") or {}
    key_value = key.get("value")
    node_id = (
        f"{label}::{key_value}"
        if key_value is not None
        else f"neo::{node.get('neo_id')}"
    )
    return {
        "id": node_id,
        "type": label,
        "type_label": _TEAM_NODE_LABELS.get(label, label),
        "group": _TEAM_NODE_GROUPS.get(label, "其他"),
        "name": str(props.get("name") or props.get("id") or node_id),
        "description": str(props.get("description") or ""),
        "props": props,
    }


def _normalize_team_edge(
    rel: dict[str, Any],
    nodes_by_neo: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    source = nodes_by_neo.get(str(rel.get("src")))
    target = nodes_by_neo.get(str(rel.get("tgt")))
    if not source or not target:
        return None
    rel_type = str(rel.get("type") or "RELATES")
    props = dict(rel.get("props") or {})
    return {
        "source": source["id"],
        "target": target["id"],
        "type": rel_type,
        "label": _TEAM_EDGE_LABELS.get(rel_type, "关联"),
        "description": str(props.get("description") or ""),
        "props": props,
    }


def build_team_graph(
    *,
    include_variables: bool = False,
    include_correlation: bool = False,
    corr_threshold: float = 0.95,
) -> dict[str, Any]:
    """从队友迁移快照构建机理/变量知识图谱。

    - 机理层：非 Variable 节点 + 其间的语义关系边。
    - 变量层：加入 Variable 节点与语义边；相关性边按 |pearson_r| >= corr_threshold 裁剪。
    """
    path = Path(TEAM_GRAPH)
    if not path.is_file():
        return {
            "schema_version": "boilermind.team_graph.v1",
            "summary": {
                "node_count": 0,
                "edge_count": 0,
                "mechanism_node_count": 0,
                "variable_node_count": 0,
                "semantic_edge_count": 0,
                "correlation_edge_count": 0,
                "correlation_shown": 0,
                "exported_at": None,
            },
            "nodes": [],
            "edges": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_nodes = list(payload.get("nodes") or [])
    raw_rels = list(payload.get("rels") or [])
    nodes_by_neo = {
        str(node.get("neo_id")): _normalize_team_node(node)
        for node in raw_nodes
    }

    variable_count = sum(
        1 for node in nodes_by_neo.values() if node["type"] == "Variable"
    )
    mechanism_count = len(nodes_by_neo) - variable_count

    selected_nodes = [
        node
        for node in nodes_by_neo.values()
        if include_variables or node["type"] != "Variable"
    ]

    semantic_total = 0
    correlation_total = 0
    correlation_shown = 0
    edges: list[dict[str, Any]] = []
    for rel in raw_rels:
        rel_type = str(rel.get("type") or "RELATES")
        if rel_type == "CORRELATES_WITH":
            correlation_total += 1
            if not include_correlation or not include_variables:
                continue
            try:
                pearson_r = abs(
                    float((rel.get("props") or {}).get("pearson_r") or 0.0)
                )
            except (TypeError, ValueError):
                pearson_r = 0.0
            if pearson_r < corr_threshold:
                continue
            correlation_shown += 1
        else:
            semantic_total += 1
            if not include_variables:
                source = nodes_by_neo.get(str(rel.get("src")))
                target = nodes_by_neo.get(str(rel.get("tgt")))
                if not source or not target:
                    continue
                if source["type"] == "Variable" or target["type"] == "Variable":
                    continue
        edge = _normalize_team_edge(rel, nodes_by_neo)
        if edge is not None:
            edges.append(edge)

    node_types: Counter[str] = Counter()
    for node in selected_nodes:
        node_types[node["type"]] += 1
    edge_types: Counter[str] = Counter()
    for edge in edges:
        edge_types[edge["type"]] += 1

    meta = payload.get("meta") or {}
    return {
        "schema_version": "boilermind.team_graph.v1",
        "summary": {
            "node_count": len(selected_nodes),
            "edge_count": len(edges),
            "mechanism_node_count": mechanism_count,
            "variable_node_count": variable_count,
            "semantic_edge_count": semantic_total,
            "correlation_edge_count": correlation_total,
            "correlation_shown": correlation_shown,
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "exported_at": meta.get("exported_at"),
        },
        "nodes": selected_nodes,
        "edges": edges,
    }
