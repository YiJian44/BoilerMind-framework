"""探测：Qwen 假设生成时，历史实验/文献是否真的进到 prompt（尝试 A）。"""
from __future__ import annotations

import json

from boilermind.orchestration import ResearchOrchestrator
from boilermind.skills.hypothesis_skill import HypothesisGenerationSkill

captured: list[str] = []
_orig_generate = HypothesisGenerationSkill._generate


def _spy(prompt: str) -> str:
    captured.append(prompt)
    return _orig_generate(prompt)


HypothesisGenerationSkill._generate = staticmethod(_spy)

orch = ResearchOrchestrator()
question = (
    "分析最新锅炉数据，识别数据属性（非线性、时序、稀疏化、降维、非高斯），"
    "模型库里面哪个模型软测蒸汽体积量V的误差最小"
)

parsed = orch.problem_parser({"research_question": question})
context = dict(parsed)
problem = context["research_problem"]
print("=== 问题解析 ===")
print("  target:", problem.get("target_variable"), "| objective:", problem.get("objective"))

capability = orch.capability.snapshot()
memory = orch.memory_retriever(problem, capability, orch.memory_store)
memory_payload = (
    json.loads(memory.model_dump_json())
    if hasattr(memory, "model_dump_json")
    else memory
)
context["experiment_memory_bundle"] = memory_payload
context["scientific_context"] = capability

n_obs = len(
    memory_payload.get("supported_observations", [])
    + memory_payload.get("falsified_observations", [])
    + memory_payload.get("contradictions", [])
)
print("=== 实验记忆检索 ===")
print("  hits:", len(memory_payload.get("completed_experiment_ids", [])))
print("  观测数:", n_obs)

evidence = orch.evidence_retriever(context)
context.update(evidence)
print("=== 文献检索 ===")
bundle = evidence.get("evidence_bundle")
print("  evidence_bundle:", "yes" if bundle else "None")

generated = orch.hypothesis_generator(context)
hypotheses = list(generated.get("qualified_hypotheses") or generated.get("hypotheses") or [])[:8]
print("\n=== Qwen 生成的假设 (%d 条) ===" % len(hypotheses))
for h in hypotheses:
    hid = h.get("hypothesis_id") or h.get("id")
    stmt = str(h.get("hypothesis") or h.get("hypothesis_statement") or "")[:80]
    print(f"  - {hid}: {stmt}")
    print(f"      trigger={h.get('trigger_types')} evidence={h.get('evidence_ids')} src_exp={h.get('source_experiment_ids')}")

print("\n=== 种子 prompt 是否含历史实验/文献 ===")
if captured:
    sp = captured[0]
    print("  Qwen 调用次数:", len(captured))
    print("  含 'Historical Experiment Memory':", "Historical Experiment Memory" in sp)
    print("  含 'Verified Literature Claims':", "Verified Literature Claims" in sp)
    print("  含 'LIB31V':", "LIB31V" in sp)
    print("  含 'BM-REGIME':", "BM-REGIME" in sp)
    print("  prompt 开头:", sp[:200].replace("\n", " "))
else:
    print("  未捕获到 Qwen 调用")
