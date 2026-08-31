from __future__ import annotations

import json
import hashlib
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from boilermind.core.contracts import (
    EvidenceBundle,
)

from boilermind.core.llm_client import (
    LLMClient,
)

from boilermind.hypothesis.hypothesis_compiler import compile_hypotheses

from .base import BaseSkill


class HypothesisGenerationSkill(BaseSkill):

    name = "hypothesis_generation"

    description = (
        "从锅炉工程问题自由生成科研假设种子，随后执行本地事实、"
        "结构、历史实验与可执行性评估"
    )

    @staticmethod
    def _restore_provenance(original: dict, revised: dict) -> dict:
        for provenance_key in (
            "evidence_ids",
            "source_observation_ids",
            "source_experiment_ids",
            "opportunity_id",
            "trigger_types",
        ):
            original_value = original.get(provenance_key)
            revised[provenance_key] = (
                list(original_value)
                if isinstance(original_value, list)
                else original_value
            )
        return revised


    # ============================================================
    # JSON parsing
    # ============================================================

    @staticmethod
    def _extract_object(
        response: str,
    ) -> dict:

        text = str(response).strip()

        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

        except Exception:
            pass

        match = re.search(
            r"\{[\s\S]*\}",
            text,
        )

        if not match:
            return {}

        try:
            parsed = json.loads(
                match.group()
            )

            return (
                parsed
                if isinstance(parsed, dict)
                else {}
            )

        except Exception:
            return {}


    # ============================================================
    # LLM
    # ============================================================

    @staticmethod
    def _grounding_subset(problem: dict, seeds: list[dict]) -> dict:
        """Keep only provenance records referenced by the current candidates."""
        observation_ids = {
            str(item) for seed in seeds
            for item in seed.get("source_observation_ids", [])
        }
        experiment_ids = {
            str(item) for seed in seeds
            for item in seed.get("source_experiment_ids", [])
        }
        opportunity_ids = {
            str(seed.get("opportunity_id")) for seed in seeds
            if str(seed.get("opportunity_id", "")).strip()
        }
        subset = dict(problem)
        memory = problem.get("_experiment_memory")
        if isinstance(memory, dict):
            compact = dict(memory)
            for key in (
                "supported_observations", "falsified_observations",
                "contradictions", "engineering_failures",
            ):
                values = memory.get(key, [])
                compact[key] = [
                    item for item in values
                    if isinstance(item, dict) and (
                        str(item.get("observation_id", "")) in observation_ids
                        or str(item.get("experiment_id", "")) in experiment_ids
                        or bool(set(map(str, item.get("source_experiment_ids", []))) & experiment_ids)
                    )
                ]
            compact["completed_experiment_ids"] = [
                item for item in memory.get("completed_experiment_ids", [])
                if str(item) in experiment_ids
            ]
            subset["_experiment_memory"] = compact
        opportunities = problem.get("_opportunity_map")
        if isinstance(opportunities, dict):
            compact = dict(opportunities)
            compact["opportunities"] = [
                item for item in opportunities.get("opportunities", [])
                if isinstance(item, dict) and (
                    str(item.get("opportunity_id", "")) in opportunity_ids
                    or bool(set(map(str, item.get("source_observation_ids", []))) & observation_ids)
                )
            ]
            subset["_opportunity_map"] = compact
        current = problem.get("_current_observations")
        if isinstance(current, dict):
            compact = dict(current)
            compact["observations"] = [
                item for item in current.get("observations", [])
                if isinstance(item, dict)
                and str(item.get("observation_id", "")) in observation_ids
            ]
            subset["_current_observations"] = compact
        return subset

    @staticmethod
    def _evidence_subset(evidence_claims: list[dict], seeds: list[dict]) -> list[dict]:
        evidence_ids = {
            str(item) for seed in seeds for item in seed.get("evidence_ids", [])
        }
        return [
            claim for claim in evidence_claims
            if str(claim.get("evidence_id", "")) in evidence_ids
        ]

    @staticmethod
    def _compact_memory_observation(item: dict) -> dict:
        metrics = item.get("supporting_metrics", {})
        compact_metrics = {}
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    compact_metrics[key] = value
                elif isinstance(value, list) and len(value) <= 8:
                    compact_metrics[key] = value
                elif isinstance(value, dict) and len(value) <= 8 and all(
                    isinstance(nested, (str, int, float, bool)) or nested is None
                    for nested in value.values()
                ):
                    compact_metrics[key] = value
        scope = item.get("scope_signature", {})
        compact_scope = {}
        if isinstance(scope, dict):
            for key in (
                "target_variable", "target_unit", "prediction_mode", "dataset_id",
                "feature_count", "window_steps", "prediction_horizon_steps",
                "sampling_interval_seconds", "split_policy", "regime_definition",
            ):
                if key in scope:
                    compact_scope[key] = scope[key]
        return {
            "observation_id": item.get("observation_id"),
            "source_experiment_ids": list(item.get("source_experiment_ids", [])),
            "observation_type": item.get("observation_type"),
            "claim": item.get("claim"),
            "scope_signature": compact_scope,
            "supporting_metrics": compact_metrics,
            "counter_evidence": item.get("counter_evidence"),
            "confidence_level": item.get("confidence_level"),
            "reuse_policy": item.get("reuse_policy"),
        }

    @staticmethod
    def _generate(
        prompt: str,
    ) -> str:

        if os.environ.get("BOILERMIND_PIPELINE_PROGRESS") == "1":
            print(
                f"[pipeline] hypothesis_llm_request_chars:{len(prompt)}",
                flush=True,
            )

        client = LLMClient()

        return str(
            client.generate(prompt)
        )


    # ============================================================
    # Deterministic numeric grounding
    # ============================================================

    @staticmethod
    def _numeric_tokens(
        text: str,
    ) -> set[str]:

        return set(
            re.findall(
                r"(?<![A-Za-z0-9_])"
                r"[-+]?\d+(?:\.\d+)?"
                r"(?:%|％)?",
                str(text),
            )
        )


    @classmethod
    def _collect_seed_numbers(
        cls,
        seed: dict,
    ) -> set[str]:
        scientific_payload = {
            key: seed.get(key)
            for key in (
                "title",
                "hypothesis",
                "mechanism",
                "inference",
                "variables",
                "verification_intent",
                "falsification_condition",
                "evidence_gap",
            )
        }
        text = json.dumps(
            scientific_payload,
            ensure_ascii=False,
        )

        return cls._numeric_tokens(
            text
        )


    # ============================================================
    # Evidence claim extraction
    # ============================================================

    def _extract_evidence_claims(
        self,
        problem: dict,
        bundle: EvidenceBundle,
    ) -> list[dict]:

        eligible = [
            item
            for item in bundle.evidence
            if item.core_claim_eligible
        ]

        payload = []

        for item in eligible:

            payload.append(
                {
                    "evidence_id":
                        item.evidence_id,

                    "title":
                        item.title,

                    "claim_support":
                        item.claim_support.value,

                    "applicability":
                        item.applicability.value,

                    "verification_rationale":
                        item.verification_rationale,

                    "text":
                        item.text[:4000],
                }
            )


        prompt = f"""
你是 BoilerMind 的 Evidence Claim Extractor。

你的任务不是生成科研假设，而是把已经通过验证的文献证据
压缩成“可安全用于假设生成的原子科学事实”。

CURRENT ResearchProblemSpec：

{json.dumps(
    problem,
    ensure_ascii=False,
)}

Verified Evidence：

{json.dumps(
    payload,
    ensure_ascii=False,
)}

必须严格遵守：

1. 只能提取证据原文实际支持的内容。
2. 不得新增任何数字、阈值、百分比、模型性能值。
3. 如果指标越小越好，例如 MAE、MAPE、RMSE，
   必须正确保留数值方向，不得把更大的误差解释成更优。
4. 不得把“10 steps”解释为“10 minutes”，
   除非证据明确给出 sampling interval。
5. 不得把 partial evidence 扩大成 direct evidence。
6. 不得根据文献自行推断锅炉故障、报警、控制风险等新任务。
7. 必须明确写出每条证据的 scope_limits。
8. 如果证据不足以判断某结论，写入 scope_limits，
   不得自行补全。

Return JSON only：

{{
  "evidence_claims": [
    {{
      "evidence_id": "原始ID",
      "verified_claims": [
        "证据真正支持的事实"
      ],
      "scope_limits": [
        "该证据不能证明什么"
      ]
    }}
  ]
}}

必须为每一个输入 evidence_id 返回且只返回一个结果。
""".strip()


        response = self._generate(
            prompt
        )

        parsed = self._extract_object(
            response
        )

        claims = parsed.get(
            "evidence_claims",
            []
        )

        if not isinstance(
            claims,
            list,
        ):
            raise RuntimeError(
                "invalid_evidence_claim_extraction"
            )


        expected_ids = {
            item.evidence_id
            for item in eligible
        }

        returned = {}


        for item in claims:

            if not isinstance(
                item,
                dict,
            ):
                continue

            evidence_id = str(
                item.get(
                    "evidence_id",
                    "",
                )
            ).strip()

            if (
                evidence_id
                not in expected_ids
            ):
                continue

            if evidence_id in returned:
                continue

            returned[evidence_id] = {
                "evidence_id":
                    evidence_id,

                "verified_claims":
                    list(
                        item.get(
                            "verified_claims",
                            [],
                        )
                        or []
                    ),

                "scope_limits":
                    list(
                        item.get(
                            "scope_limits",
                            [],
                        )
                        or []
                    ),
            }


        missing = (
            expected_ids
            - set(returned)
        )

        if missing:
            raise RuntimeError(
                "evidence_claim_extraction_missing:"
                + ",".join(
                    sorted(missing)
                )
            )


        return [
            returned[evidence_id]
            for evidence_id
            in sorted(returned)
        ]


    # ============================================================
    # Conservative hypothesis seed generation
    # ============================================================

    def _generate_seeds(
        self,
        problem: dict,
    ) -> list[dict]:

        compact_problem = {
            key: problem.get(key)
            for key in (
                "problem_id", "original_question", "research_object",
                "target_variable", "objective", "operating_condition",
                "manipulated_variables", "observed_variables",
                "context_variables", "research_goal", "constraints",
                "required_objective_dimensions", "required_models",
                "reference_models", "metrics", "required_horizon_steps",
                "required_operations", "protocol_constraints",
                "_neutral_capabilities",
            )
            if problem.get(key) not in (None, "", [], {})
        }
        memory = problem.get("_experiment_memory", {})
        historical = {
            key: [
                self._compact_memory_observation(item)
                for item in list(memory.get(key, []))[:3]
                if isinstance(item, dict)
            ]
            for key in (
                "supported_observations", "falsified_observations",
                "contradictions", "engineering_failures",
            )
            if isinstance(memory, dict)
        }
        candidate_models = list(problem.get("_profile_candidate_models") or [])
        per_model_prompt = ""
        if candidate_models:
            per_model_prompt = f"""

候选模型（数据属性画像选型计划，必须逐模型生成假设）：
{json.dumps(candidate_models, ensure_ascii=False)}

重要（逐模型假设规则）：必须为上述每个候选模型生成一条独立假设，共
{len(candidate_models)} 条，不得遗漏、不得合并。每条假设的 hypothesis_statement
明确声明"模型 <家族名> 在软测蒸汽体积量 V 上误差最小"；engineering_mechanism
给出该模型与数据属性（时序/非线性/非高斯/共线/维度）匹配的理由；不同假设的
expected_observation 指向各自模型的软测误差表现。"""
        prompt = f"""
你是 BoilerMind 的锅炉工程科研假设生成器。根据工程师的原始问题提出
可观察、可证伪的工程或机理假设。明确的模型比较问题只生成1条汇总比较假设；
只有机理探索问题才生成最多3条不同假设。不要写实验规划书，不要根据当前系统容易
执行什么来改变研究问题，不要把普通工程问题改写成模型排行榜。{per_model_prompt}

历史真实实验是生成假设的首要依据。优先从审计通过且与当前问题同作用域的
supported observations形成假设；不得用TEST_ONLY、Mock或审计失败记录支撑
科研假设。falsified observations用于排除直接冲突假设。文献不参与本阶段；
不得引用文献、文献证据ID或根据文献生成假设。

Historical Experiment Memory:
{json.dumps(historical, ensure_ascii=False)}

上面的历史实验结果就是本次允许使用的完整科研来源。只能引用其中真实存在的
observation_id 和 source_experiment_ids，不得伪造ID、数值
结果、排名或已验证结论。不同候选必须具有不同机制、边界条件或可观察结果，
不能只更换模型名、指标名或措辞。若原问题明确要求模型比较，可以提出对应的
比较假设，但候选模型必须来自中性能力边界，并标明它是性能比较而非因果证明。
ResearchProblemSpec 中已经明确的 required_models、reference_models、metrics、
required_horizon_steps 与 protocol_constraints 是不可扩张、不可替换的硬边界；
能力列表只用于校验可执行性，不能把未被用户指定的模型加入实验。

重要（数值接地规则）：假设正文（hypothesis_statement / engineering_mechanism /
expected_observation / falsification_condition）中不得出现任何具体数值、百分比
或性能指标，例如 "MAE=0.145"、"增益20%"、"误差36%" 之类。只用定性、方向性
表述（如 "线性模型在调峰段更优"、"共线特征适合降维模型"）。所有具体数值由
确定性实验程序计算并报告，不由假设承载；假设只表达可观察、可证伪的定性主张。

重要（实验操作接地规则）：本问题只做"模型软测V对比"。假设不得使用会触发
额外实验操作的词，包括：噪声/鲁棒/毛刺/尖峰/漂移/冻结/缺失注入/污染比例、
显著/统计显著/significant、工况分层/ramp_up/ramp_down/steady、滞后/时滞、
特征消融/变量消融、多种子/多随机种子、运行时/推理延迟/资源占用。只用
"模型X（如线性/树集成/循环网络）适合/不适合软测V"这类纯模型比较的定性表述。

工程问题与中性能力边界：
{json.dumps(compact_problem, ensure_ascii=False)}

只返回JSON：
{{"seeds":[{{"title":"","hypothesis_statement":"",
"engineering_mechanism":"","expected_observation":"","key_variables":[],
"applicability_conditions":[],"falsification_condition":"",
"assumptions":[],"evidence_needed":[],
"source_observation_ids":[],"source_experiment_ids":[],"trigger_types":[],
"generation_source":"llm_grounded_generation"}}]}}
""".strip()

        response = self._generate(
            prompt
        )

        parsed = self._extract_object(
            response
        )

        seeds = parsed.get(
            "seeds",
            []
        )

        if not isinstance(
            seeds,
            list,
        ):
            raise RuntimeError(
                "hypothesis_seed_output_invalid"
            )


        normalized = []

        for item in seeds[:6]:

            if not isinstance(
                item,
                dict,
            ):
                continue

            evidence_ids = item.get(
                "evidence_ids",
                [],
            )

            if isinstance(
                evidence_ids,
                str,
            ):
                evidence_ids = [
                    evidence_ids
                ]

            variables = item.get("key_variables", item.get("variables", []))

            if isinstance(
                variables,
                str,
            ):
                variables = [
                    variables
                ]

            source_observation_ids = item.get("source_observation_ids", [])
            if isinstance(source_observation_ids, str):
                source_observation_ids = [source_observation_ids]
            source_experiment_ids = item.get("source_experiment_ids", [])
            if isinstance(source_experiment_ids, str):
                source_experiment_ids = [source_experiment_ids]
            trigger_types = item.get("trigger_types", [])
            if isinstance(trigger_types, str):
                trigger_types = [trigger_types]

            normalized_triggers = [
                str(x).strip() for x in trigger_types if str(x).strip()
            ]
            normalized_observation_ids = [
                str(x).strip()
                for x in source_observation_ids
                if str(x).strip()
            ]
            normalized_experiment_ids = [
                str(x).strip()
                for x in source_experiment_ids
                if str(x).strip()
            ]
            if normalized_experiment_ids:
                normalized_triggers = [
                    value for value in normalized_triggers
                    if value != "CURRENT_DATA_OBSERVATION"
                ]
                if "HISTORICAL_EXPERIMENT" not in normalized_triggers:
                    normalized_triggers.append("HISTORICAL_EXPERIMENT")
            elif (
                normalized_observation_ids
                and all(value.startswith("CUR-") for value in normalized_observation_ids)
                and "CURRENT_DATA_OBSERVATION" not in normalized_triggers
            ):
                normalized_triggers.append("CURRENT_DATA_OBSERVATION")

            statement = str(
                item.get("hypothesis_statement", item.get("hypothesis", ""))
            ).strip()
            mechanism = str(
                item.get("engineering_mechanism", item.get("mechanism", ""))
            ).strip()
            expected_observation = str(
                item.get("expected_observation", item.get("inference", ""))
            ).strip()
            applicability = item.get("applicability_conditions", [])
            assumptions = item.get("assumptions", [])
            evidence_needed = item.get("evidence_needed", [])
            for name, value in (
                ("applicability", applicability),
                ("assumptions", assumptions),
                ("evidence_needed", evidence_needed),
            ):
                if isinstance(value, str):
                    if name == "applicability":
                        applicability = [value]
                    elif name == "assumptions":
                        assumptions = [value]
                    else:
                        evidence_needed = [value]

            normalized.append(
                {
                    "title":
                        str(
                            item.get(
                                "title",
                                "",
                            )
                        ).strip(),

                    "hypothesis": statement,
                    "hypothesis_statement": statement,

                    "mechanism": mechanism,
                    "engineering_mechanism": mechanism,
                    "expected_observation": expected_observation,

                    "evidence_ids":
                        [
                            str(x).strip()
                            for x
                            in evidence_ids
                            if str(x).strip()
                        ],

                    "source_observation_ids": normalized_observation_ids,
                    "source_experiment_ids": normalized_experiment_ids,
                    "opportunity_id": str(item.get("opportunity_id", "")).strip(),
                    "trigger_types": normalized_triggers,

                    "inference": expected_observation,

                    "variables":
                        [
                            str(x).strip()
                            for x in variables
                            if str(x).strip()
                        ],
                    "key_variables": [
                        str(x).strip() for x in variables if str(x).strip()
                    ],
                    "applicability_conditions": [
                        str(x).strip() for x in applicability if str(x).strip()
                    ],
                    "assumptions": [
                        str(x).strip() for x in assumptions if str(x).strip()
                    ],
                    "evidence_needed": [
                        str(x).strip() for x in evidence_needed if str(x).strip()
                    ],
                    "generation_source": "llm_free_generation",

                    "verification_intent":
                        str(
                            item.get(
                                "verification_intent",
                                "",
                            )
                        ).strip(),

                    "falsification_condition":
                        str(
                            item.get(
                                "falsification_condition",
                                "",
                            )
                        ).strip(),

                    "evidence_gap":
                        str(
                            item.get(
                                "evidence_gap",
                                "",
                            )
                        ).strip(),
                }
            )


        return normalized

    @staticmethod
    def _lock_execution_fields(
        raw_seeds: list[dict],
        problem: dict,
        memory_observations: list[dict],
        opportunity_map: dict,
    ) -> list[dict]:
        """Seal LLM semantics without deriving or overwriting research claims."""
        allowed_observation_ids = {
            str(item.get("observation_id", "")).strip()
            for item in memory_observations
            if str(item.get("observation_id", "")).strip()
        }
        allowed_experiment_ids = {
            str(value).strip()
            for item in memory_observations
            for value in (
                list(item.get("source_experiment_ids", []))
                + ([item.get("experiment_id")] if item.get("experiment_id") else [])
            )
            if str(value).strip()
        }
        allowed_opportunity_ids = {
            str(item.get("opportunity_id", "")).strip()
            for item in opportunity_map.get("opportunities", [])
            if isinstance(item, dict) and str(item.get("opportunity_id", "")).strip()
        } if isinstance(opportunity_map, dict) else set()
        question_tokens = set(re.findall(
            r"[a-z0-9_]+|[\u4e00-\u9fff]",
            " ".join(str(problem.get(key, "")) for key in (
                "original_question", "research_goal", "target_variable",
                "operating_condition",
            )).casefold(),
        ))
        neutral = problem.get("_neutral_capabilities", {})
        available_variables = {
            str(item).strip().casefold()
            for item in neutral.get("available_variables", [])
            if str(item).strip()
        } if isinstance(neutral, dict) else set()
        sealed: list[dict] = []
        for raw in raw_seeds:
            seed = dict(raw)
            statement = str(
                seed.get("hypothesis_statement", seed.get("hypothesis", ""))
            ).strip()
            mechanism = str(
                seed.get("engineering_mechanism", seed.get("mechanism", ""))
            ).strip()
            expected = str(
                seed.get("expected_observation", seed.get("inference", ""))
            ).strip()
            raw_payload = {
                "title": str(seed.get("title", "")).strip(),
                "hypothesis_statement": statement,
                "engineering_mechanism": mechanism,
                "expected_observation": expected,
                "key_variables": list(seed.get("key_variables", seed.get("variables", [])) or []),
                "applicability_conditions": list(seed.get("applicability_conditions", []) or []),
                "falsification_condition": str(seed.get("falsification_condition", "")).strip(),
                "assumptions": list(seed.get("assumptions", []) or []),
                "evidence_needed": list(seed.get("evidence_needed", []) or []),
                "generation_source": str(
                    seed.get("generation_source", "llm_grounded_generation")
                ).strip() or "llm_grounded_generation",
            }
            digest = hashlib.sha256(
                json.dumps(raw_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            hypothesis_tokens = set(re.findall(
                r"[a-z0-9_]+|[\u4e00-\u9fff]",
                " ".join((
                    raw_payload["title"], statement, mechanism, expected,
                )).casefold(),
            ))
            relevance = len(question_tokens & hypothesis_tokens) / max(
                len(question_tokens), 1
            )
            unknown_variables = [
                str(item)
                for item in raw_payload["key_variables"]
                if available_variables
                and str(item).strip().casefold() not in available_variables
            ]
            source_observation_ids = list(dict.fromkeys(
                str(item).strip() for item in seed.get("source_observation_ids", [])
                if str(item).strip() in allowed_observation_ids
            ))
            source_experiment_ids = list(dict.fromkeys(
                str(item).strip() for item in seed.get("source_experiment_ids", [])
                if str(item).strip() in allowed_experiment_ids
            ))
            trigger_types = list(dict.fromkeys(
                str(item).strip() for item in seed.get("trigger_types", [])
                if str(item).strip()
            ))
            if source_experiment_ids and "HISTORICAL_EXPERIMENT" not in trigger_types:
                trigger_types.append("HISTORICAL_EXPERIMENT")
            if not source_experiment_ids and not source_observation_ids and not trigger_types:
                trigger_types.append("HUMAN_PROPOSAL")
            seed.update({
                **raw_payload,
                "hypothesis": statement,
                "mechanism": mechanism,
                "inference": expected,
                "variables": list(raw_payload["key_variables"]),
                "verification_intent": expected or "观测该假设声明的工程响应是否出现",
                "evidence_gap": (
                    "; ".join(raw_payload["evidence_needed"])
                    or "需要与该可观察预测直接对应的真实实验数据"
                ),
                "evidence_ids": list(dict.fromkeys(
                    str(item).strip() for item in seed.get("evidence_ids", [])
                    if str(item).strip()
                )),
                "source_observation_ids": source_observation_ids,
                "source_experiment_ids": source_experiment_ids,
                "opportunity_id": (
                    str(seed.get("opportunity_id", "")).strip()
                    if str(seed.get("opportunity_id", "")).strip() in allowed_opportunity_ids
                    else ""
                ),
                "trigger_types": trigger_types,
                "raw_hypothesis": raw_payload,
                "raw_hypothesis_sha256": digest,
                "deterministic_validation": {
                    "structure_valid": bool(
                        raw_payload["title"] and statement and mechanism
                        and expected and raw_payload["falsification_condition"]
                    ),
                    "problem_relevance_score": round(relevance, 4),
                    "responds_to_problem": relevance >= 0.03,
                    "unknown_variables": unknown_variables,
                },
                "workflow_status": "GENERATED",
            })
            sealed.append(seed)
        return sealed

    def _revise_seeds_batch(
        self,
        problem: dict,
        evidence_claims: list[dict],
        revision_inputs: list[dict],
    ) -> dict[str, dict | None]:
        """Repair all critic-marked candidates in one LLM request."""
        if not revision_inputs:
            return {}
        prompt = f"""
你是 BoilerMind Scientific Hypothesis Batch Revision Agent。
只修复每条候选已经列出的 critic_issues 和 deterministic_issues；不得发明新数字、
证据ID、实验ID、模型结果或能力，不得改变研究问题。每条仍只保留一个主要可证伪主张。

ResearchProblemSpec:
{json.dumps(problem, ensure_ascii=False)}

Grounding Claims:
{json.dumps(evidence_claims, ensure_ascii=False)}

Revision Inputs:
{json.dumps(revision_inputs, ensure_ascii=False)}

Return JSON only:
{{"revisions":[{{"hypothesis_id":"H001","decision":"REVISED|UNREPAIRABLE","seed":{{
"title":"","hypothesis":"","mechanism":"","evidence_ids":[],"inference":"",
"variables":[],"verification_intent":"","falsification_condition":"","evidence_gap":""
}}}}]}}
必须为每个输入 hypothesis_id 返回一次。
""".strip()
        parsed = self._extract_object(self._generate(prompt))
        output: dict[str, dict | None] = {}
        allowed_ids = {str(item["hypothesis"]["hypothesis_id"]) for item in revision_inputs}
        for item in parsed.get("revisions", []):
            if not isinstance(item, dict):
                continue
            hid = str(item.get("hypothesis_id", "")).strip()
            if hid not in allowed_ids:
                continue
            if str(item.get("decision", "")).upper() != "REVISED":
                output[hid] = None
                continue
            seed = item.get("seed")
            if not isinstance(seed, dict):
                output[hid] = None
                continue
            normalized = {}
            for field in (
                "title", "hypothesis", "mechanism", "inference",
                "verification_intent", "falsification_condition", "evidence_gap",
            ):
                normalized[field] = str(seed.get(field, "")).strip()
            for field in ("evidence_ids", "variables"):
                value = seed.get(field, [])
                if isinstance(value, str):
                    value = [value]
                normalized[field] = [str(value_item).strip() for value_item in value if str(value_item).strip()]
            output[hid] = normalized
        for hid in allowed_ids:
            output.setdefault(hid, None)
        return output


    # ============================================================
    # Deterministic gate
    # ============================================================

    def _deterministic_gate(
        self,
        seed: dict,
        *,
        valid_evidence_ids: set[str],
        allowed_numbers: set[str],
        problem: dict | None = None,
        scientific_context: dict | None = None,
    ) -> list[str]:

        issues = []

        if seed.get("compilation_status") == "UNSUPPORTED":
            issues.append("hypothesis_compilation_no_supported_operation")


        required_text = [
            "title",
            "hypothesis",
            "mechanism",
            "verification_intent",
            "falsification_condition",
            "evidence_gap",
        ]


        for field in required_text:

            if not str(
                seed.get(
                    field,
                    "",
                )
            ).strip():

                issues.append(
                    f"missing_{field}"
                )

        local_validation = seed.get("deterministic_validation", {})
        if isinstance(local_validation, dict):
            if not local_validation.get("structure_valid", False):
                issues.append("invalid_free_hypothesis_structure")
            if not local_validation.get("responds_to_problem", False):
                issues.append("hypothesis_not_relevant_to_problem")


        evidence_ids = seed.get(
            "evidence_ids",
            [],
        )


        source_observation_ids = seed.get("source_observation_ids", [])

        trigger_types = {
            str(item).strip()
            for item in seed.get("trigger_types", [])
            if str(item).strip()
        }
        if (
            not evidence_ids
            and not source_observation_ids
            and "HUMAN_PROPOSAL" not in trigger_types
        ):

            issues.append(
                "missing_grounding_source_ids"
            )


        for evidence_id in evidence_ids:

            if (
                evidence_id
                not in valid_evidence_ids
            ):
                issues.append(
                    "unknown_evidence_id:"
                    + evidence_id
                )

        for observation_id in source_observation_ids:
            if observation_id not in valid_evidence_ids:
                issues.append("unknown_observation_id:" + observation_id)


        # ------------------------------------------
        # Numeric grounding
        # ------------------------------------------

        seed_numbers = (
            self._collect_seed_numbers(
                seed
            )
        )


        unsupported_numbers = sorted(
            number
            for number in seed_numbers
            if number not in allowed_numbers
        )


        for number in unsupported_numbers:

            issues.append(
                "unsupported_numeric_claim:"
                + number
            )


        # ------------------------------------------
        # Temporal-unit protection
        # ------------------------------------------

        text = json.dumps(
            seed,
            ensure_ascii=False,
        ).lower()


        has_step = bool(
            re.search(
                r"\d+\s*(步|steps?|step)",
                text,
            )
        )

        has_minute = bool(
            re.search(
                r"\d+\s*(分钟|mins?|minutes?)",
                text,
            )
        )


        # Only inspect an actual time/step mapping.  A comparison phrase such
        # as "MAE 大于或等于 baseline" must not trigger this protection.
        minute_step = re.search(
            r"(\d+)\s*(?:分钟|mins?|minutes?).{0,16}?"
            r"(?:等于|即为|对应于|换算为|equivalent|equals|[（(])?\s*"
            r"(\d+)\s*(?:步|steps?|step)",
            text,
        )
        step_minute = re.search(
            r"(\d+)\s*(?:步|steps?|step).{0,16}?"
            r"(?:等于|即为|对应于|换算为|equivalent|equals|[（(])?\s*"
            r"(\d+)\s*(?:分钟|mins?|minutes?)",
            text,
        )
        mapping = minute_step or step_minute
        if has_step and has_minute and mapping:
            if minute_step:
                minutes, steps = map(int, minute_step.groups())
            else:
                steps, minutes = map(int, step_minute.groups())
            interval = (scientific_context or {}).get(
                "sampling_interval_seconds"
            )
            mapping_is_verified = bool(
                isinstance(interval, (int, float))
                and interval > 0
                and minutes * 60 / interval == steps
            )
            if not mapping_is_verified:
                issues.append(
                    "unjustified_step_to_time_mapping"
                )


        # ------------------------------------------
        # Falsifiability basic check
        # ------------------------------------------

        falsification = str(
            seed.get(
                "falsification_condition",
                "",
            )
        ).strip()


        if len(falsification) < 8:

            issues.append(
                "falsification_condition_too_weak"
            )

        # Facts and runtime capability are program decisions.  They are
        # recorded here before any scientific LLM critique.
        if problem is not None and scientific_context is not None:
            from boilermind.hypothesis.deterministic_admission import evaluate_candidate
            admission = evaluate_candidate(seed, problem, scientific_context)
            seed["deterministic_admission"] = admission


        return list(
            dict.fromkeys(
                issues
            )
        )


    # ============================================================
    # Independent scientific critic
    # ============================================================

    def _critic(
        self,
        problem: dict,
        evidence_claims: list[dict],
        seeds: list[dict],
        deterministic_issues: dict[str, list[str]],
    ) -> dict[str, dict]:

        prompt = f"""
你是 BoilerMind 的 Scientific Hypothesis Critic。

你不是来证明假设正确。
你的任务是判断：
该假设是否具备进入后续 Ranking 和实验设计的资格。

ResearchProblemSpec：

{json.dumps(
    problem,
    ensure_ascii=False,
)}

Evidence Claims：

{json.dumps(
    evidence_claims,
    ensure_ascii=False,
)}

Hypothesis Seeds：

{json.dumps(
    seeds,
    ensure_ascii=False,
)}

Deterministic Gate Issues：

{json.dumps(
    deterministic_issues,
    ensure_ascii=False,
)}

对每条假设严格检查：

scope_consistent：
是否仍然回答当前科研问题，没有自行扩展新任务。

evidence_grounded：
假设核心动机是否能从引用证据合理提出。

evidence_direction_correct：
是否正确理解 MAE/MAPE/RMSE/R² 等指标方向，
是否错误解读文献结果。

mechanism_plausible：
作用机制/预测机制是否合理，
是否存在明显物理跳跃。

temporal_consistent：
是否偷换 step/sample/second/minute 等时间单位。

falsifiable：
是否存在明确可执行的证伪条件。

causal_language_safe：
是否把待验证推论错误包装成已经证明的因果事实。

variable_scope_safe：
是否为了构造假设引入明显脱离当前问题和证据的新变量。

single_testable_claim：
是否只有一个主要可证伪科学主张。
若假设同时要求多个独立结果共同成立，
例如精度下降、离散度下降且所有模型均改善，
应优先判为 REVISE，
要求收敛为一个 primary claim。

model_identity_grounded：
是否存在未经 Verified Evidence 明确支持的模型身份推断，
例如 Baseline=LSTM、Baseline=Transformer、
某模型族等同于某具体模型。
存在此类偷换时应 REJECT 或 REVISE。

cross_domain_transfer_safe：
如果 Evidence 来自不同数据集、不同工况或不同领域，
假设是否明确把迁移关系描述为待验证 inference，
而不是已经成立的事实。

实验任务约束与证据事实必须严格区分：

如果某个预测时域、目标变量、运行工况或研究对象
明确存在于 CURRENT ResearchProblemSpec 中，
则它可以直接作为实验条件。

不得因为 Evidence 未重复证明该实验条件，
就判 temporal_consistent=false 或 evidence_grounded=false。

例如：
用户规定 forecast horizon=T，
Hypothesis 可以在 horizon=T 下提出待验证关系。

这不等于：
Evidence 已经验证 horizon=T。

只有当 Hypothesis 声称：
Evidence 中的 prediction step、process lag、
history window、sampling interval
与用户给出的物理时间相互等价时，
才需要明确换算证据。

如果跨工况/跨领域迁移已经明确写成
“待验证 inference”，
不得仅因为目标场景不同就直接 REJECT。

模型注册、操作支持、指标支持、数值来源、历史重复性和多目标覆盖
全部由本地确定性程序判断。Critic 不得推翻、补写或重新判断这些事实，
也不得因为当前能力缺失而否定一个科学机制候选。

关于 ResearchProblemSpec 中用户明确给出的约束：

- 用户明确给出的研究对象、目标变量、工况、预测时域、
  评价目标等，可以直接作为 CURRENT 实验任务条件。
- 不得仅因为文献没有重复给出这些条件，就判定假设不一致。
- 但不得把不同科学概念无依据地建立等价关系。
  例如 forecast horizon、process lag、history window、
  sampling interval、prediction step 是不同概念。
- 只有 Evidence、Dataset Contract 或明确实验定义提供了换算关系，
  才允许声称二者等价。
- 用户给出的任务条件不得被错误描述为“文献已经证明”。

重要：

- 新颖但证据很弱，不代表 PASS。
- 朴素但证据充分、可以实验验证，可以 PASS。
- 不允许因为“看起来有创新”而放宽标准。
- deterministic_issues 非空时，不得给 PASS。

Return JSON only：

{{
  "decisions": [
    {{
      "hypothesis_id": "H001",
      "decision": "PASS | REVISE | REJECT",
      "scope_consistent": true,
      "evidence_grounded": true,
      "evidence_direction_correct": true,
      "mechanism_plausible": true,
      "temporal_consistent": true,
      "falsifiable": true,
      "causal_language_safe": true,
      "variable_scope_safe": true,
      "single_testable_claim": true,
      "model_identity_grounded": true,
      "cross_domain_transfer_safe": true,
      "issues": [],
      "rationale": "..."
    }}
  ]
}}

必须为每个 hypothesis_id 返回且只返回一个 decision。
""".strip()


        response = self._generate(
            prompt
        )

        parsed = self._extract_object(
            response
        )

        decisions = parsed.get(
            "decisions",
            []
        )

        if not isinstance(
            decisions,
            list,
        ):
            return {}


        result = {}

        for item in decisions:

            if not isinstance(
                item,
                dict,
            ):
                continue

            hypothesis_id = str(
                item.get(
                    "hypothesis_id",
                    "",
                )
            ).strip()

            if hypothesis_id:
                result[
                    hypothesis_id
                ] = item


        return result


    # ============================================================
    # One-shot scientific revision
    # ============================================================

    def _revise_seed(
        self,
        problem: dict,
        evidence_claims: list[dict],
        seed: dict,
        critic_decision: dict,
        deterministic_issues: list[str],
    ) -> dict | None:

        prompt = f"""
你是 BoilerMind 的 Scientific Hypothesis Revision Agent。

你的任务不是重新发明一个科研假设，
而是只修复当前候选假设已经被发现的问题。

CURRENT ResearchProblemSpec：

{json.dumps(
    problem,
    ensure_ascii=False,
)}

Verified Evidence Claims：

{json.dumps(
    evidence_claims,
    ensure_ascii=False,
)}

Original Hypothesis Seed：

{json.dumps(
    seed,
    ensure_ascii=False,
)}

Deterministic Gate Issues：

{json.dumps(
    deterministic_issues,
    ensure_ascii=False,
)}

Scientific Critic Decision：

{json.dumps(
    critic_decision,
    ensure_ascii=False,
)}

严格规则：

1. 只修 Critic 和 Gate 明确指出的问题。

2. 不得改变 CURRENT ResearchProblemSpec 的研究对象、
   目标变量、工况或研究目标。

3. 不得创造新的 evidence_id。

4. 不得创造 ResearchProblemSpec 或 Evidence Claims
   中不存在的数字、阈值、百分比或性能结果。

5. 用户明确给出的实验任务条件可以继续保留，
   例如用户定义的预测时域。
   但不得把它与 process lag、history window、
   sampling interval、prediction step 等其他概念
   无依据地建立等价关系。

6. 不得因为文献中存在某个预测 step，
   就自行解释成某个物理时间。

7. 不得把通用领域 Evidence 外推成当前工业场景中
   已经成立的事实。
   跨场景迁移只能作为待验证 inference。

8. 不得新增与当前科研问题无关的任务。

9. 不得新增当前证据无法支撑的模型优劣结论。

10. 不得为了通过审查，把原假设完全改成另一个问题。

10.1 修订后的假设必须只有一个 primary scientific claim。
如果原假设同时包含多个独立结论，
请选择最有证据支撑且最直接回答当前科研问题的一个，
其余内容降级为 secondary observation，
不得继续作为主假设成立条件。

10.2 不得进行模型身份偷换。
如果 Verified Evidence 未明确说明 Baseline 等同于某具体模型，
修订时必须删除该等价关系。

10.3 跨领域、跨数据集、跨工况的 Evidence
只能支撑“迁移后值得验证”的推论，
不能被改写成当前场景已经成立的结论。

10.4 如果 Evidence 中的某个模型、
变量或指标不是当前科研问题的必要组成部分，
不得为了复制文献实验而强制加入当前假设。


10.5 绝对禁止为了使假设显得“统计严谨”
自行加入任何统计显著性阈值、百分比、
负荷阈值或经验常数。

例如不得自行产生：
p < 0.05
p > 0.05
30%
95%
相关系数阈值
误差改善百分比

除非该数字明确存在于
ResearchProblemSpec 或 Verified Evidence Claims。

需要表达支持/证伪时，
优先使用方向性、对照性的实验判据，
例如：
“Treatment 的 primary metric 优于 Control”
而不是自行创造显著性阈值。

11. 如果现有 Evidence 无法支持合理修订，
    必须返回 UNREPAIRABLE，
    不允许为了通过 Gate 编造内容。

如果能够修订，返回：

{{
  "decision": "REVISED",
  "reason": "...",
  "seed": {{
    "title": "...",
    "hypothesis": "...",
    "mechanism": "...",
    "evidence_ids": ["..."],
    "inference": "...",
    "variables": ["..."],
    "verification_intent": "...",
    "falsification_condition": "...",
    "evidence_gap": "..."
  }}
}}

如果无法修订，返回：

{{
  "decision": "UNREPAIRABLE",
  "reason": "..."
}}

Return JSON only.
""".strip()

        response = self._generate(
            prompt
        )

        parsed = self._extract_object(
            response
        )

        decision = str(
            parsed.get(
                "decision",
                "",
            )
        ).strip().upper()

        if decision != "REVISED":
            return None

        revised = parsed.get(
            "seed"
        )

        if not isinstance(
            revised,
            dict,
        ):
            return None


        evidence_ids = revised.get(
            "evidence_ids",
            [],
        )

        if isinstance(
            evidence_ids,
            str,
        ):
            evidence_ids = [
                evidence_ids
            ]


        variables = revised.get(
            "variables",
            [],
        )

        if isinstance(
            variables,
            str,
        ):
            variables = [
                variables
            ]


        normalized = {

            "title":
                str(
                    revised.get(
                        "title",
                        "",
                    )
                ).strip(),

            "hypothesis":
                str(
                    revised.get(
                        "hypothesis",
                        "",
                    )
                ).strip(),

            "mechanism":
                str(
                    revised.get(
                        "mechanism",
                        "",
                    )
                ).strip(),

            "evidence_ids":
                [
                    str(item).strip()
                    for item
                    in evidence_ids
                    if str(item).strip()
                ],

            "inference":
                str(
                    revised.get(
                        "inference",
                        "",
                    )
                ).strip(),

            "variables":
                [
                    str(item).strip()
                    for item
                    in variables
                    if str(item).strip()
                ],

            "verification_intent":
                str(
                    revised.get(
                        "verification_intent",
                        "",
                    )
                ).strip(),

            "falsification_condition":
                str(
                    revised.get(
                        "falsification_condition",
                        "",
                    )
                ).strip(),

            "evidence_gap":
                str(
                    revised.get(
                        "evidence_gap",
                        "",
                    )
                ).strip(),
        }

        return normalized


    # ============================================================
    # Execute
    # ============================================================

    def execute(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any]:

        problem = context.get(
            "research_problem"
        )

        experiment_memory = context.get("experiment_memory_bundle", {})
        memory_observations = []
        if isinstance(experiment_memory, dict):
            for key in (
                "supported_observations",
                "falsified_observations",
                "contradictions",
                "engineering_failures",
            ):
                values = experiment_memory.get(key, [])
                if isinstance(values, list):
                    memory_observations.extend(
                        item for item in values if isinstance(item, dict)
                    )

        if not isinstance(
            problem,
            dict,
        ):
            raise ValueError(
                "research_problem_required"
            )

        # Literature is deliberately excluded from hypothesis generation.
        # Keep the compatibility field empty for existing downstream readers;
        # the original EvidenceBundle remains available to reporting.
        evidence_claims: list[dict] = []

        # Runtime capability context.
        # This is system capability, NOT hard-coded research content.
        scientific_context = context.get(
            "scientific_context",
            {},
        )

        if not isinstance(
            scientific_context,
            dict,
        ):
            scientific_context = {}

        generation_problem = dict(problem)

        opportunity_map = context.get("opportunity_map", {})
        current_observations = context.get("current_observation_bundle", {})
        generation_problem["_experiment_memory"] = experiment_memory
        generation_problem["_opportunity_map"] = opportunity_map
        generation_problem["_current_observations"] = current_observations
        generation_problem["_profile_candidate_models"] = list(
            context.get("profile_to_run_families") or []
        )

        generation_problem["_neutral_capabilities"] = {
            "target_variable": scientific_context.get("target_variable"),
            "enabled_experiment_models": list(
                scientific_context.get("enabled_experiment_models", [])
                or scientific_context.get("models", [])
            ),
            "reference_model": scientific_context.get("reference_model"),
            "available_metrics": list(
                scientific_context.get("available_metrics", [])
                or scientific_context.get("metrics", [])
            ),
            "supported_experiment_operations": list(
                scientific_context.get("supported_experiment_operations", [])
                or scientific_context.get("operations", [])
            ),
            "prediction_horizon_steps": scientific_context.get(
                "prediction_horizon_steps"
            ),
            "supported_window_steps": list(
                scientific_context.get("supported_window_steps", [])
            ),
            "sampling_interval_seconds": scientific_context.get("sampling_interval_seconds"),
            "window_steps": scientific_context.get("window_steps"),
            "observed_variables": list(problem.get("observed_variables", [])),
            "manipulated_variables": list(problem.get("manipulated_variables", [])),
            "available_variables": list(
                scientific_context.get("available_variables", [])
            ),
        }

        # Trusted history and the runtime capability snapshot go directly into
        # the single generation request. Literature and ranking scores remain
        # excluded from generation.


        # --------------------------------------------------------
        # B. Conservative Seed Generation
        # --------------------------------------------------------

        generation_mode = str(
            context.get("hypothesis_generation_mode", "deep")
        ).strip().casefold()
        generation_problem["_generation_mode"] = generation_mode

        raw_seeds = (
            self._generate_seeds(
                generation_problem,
            )
        )

        if not raw_seeds:
            raise RuntimeError(
                "no_hypothesis_seeds_generated"
            )

        raw_seeds = self._lock_execution_fields(
            raw_seeds,
            generation_problem,
            memory_observations,
            opportunity_map if isinstance(opportunity_map, dict) else {},
        )

        seeds = []

        for index, raw_seed in enumerate(
            raw_seeds,
            start=1,
        ):

            hypothesis_id = (
                f"H{index:03d}"
            )

            seed = dict(
                raw_seed
            )

            seed["id"] = (
                hypothesis_id
            )

            seed["hypothesis_id"] = (
                hypothesis_id
            )

            seed["status"] = (
                "generated_seed"
            )

            seeds.append(
                seed
            )

        # Compile capability-bounded variants before the Quality Gate. The
        # immutable LLM originals remain available in the generation audit.
        original_generated_seeds = deepcopy(seeds)
        seeds, compilation_models = compile_hypotheses(
            seeds,
            problem,
            scientific_context,
        )
        hypothesis_compilation = [
            item.model_dump(mode="json") for item in compilation_models
        ]

        # --------------------------------------------------------
        # C. Grounding context
        # --------------------------------------------------------

        valid_evidence_ids: set[str] = set()

        valid_evidence_ids.update(
            str(item.get("observation_id"))
            for item in memory_observations
            if item.get("observation_id")
        )
        current_observation_items = (
            current_observations.get("observations", [])
            if isinstance(current_observations, dict)
            else []
        )
        valid_evidence_ids.update(
            str(item.get("observation_id"))
            for item in current_observation_items
            if isinstance(item, dict) and item.get("observation_id")
        )

        # Literature identifiers are never accepted in this stage.
        for seed in seeds:
            seed["evidence_ids"] = []


        numeric_source = (
            str(
                problem.get(
                    "original_question",
                    "",
                )
            )
            + "\n"
            + json.dumps(
                {
                    "required_horizon_steps": problem.get(
                        "required_horizon_steps"
                    ),
                    "sampling_interval_seconds": scientific_context.get(
                        "sampling_interval_seconds"
                    ),
                },
                ensure_ascii=False,
                default=str,
            )
            + "\n"
            + "\n".join(str(item.get("claim", "")) for item in memory_observations)
            + "\n"
            + "\n".join(
                json.dumps(
                    {
                        "observation_id": item.get("observation_id"),
                        "source_experiment_ids": item.get(
                            "source_experiment_ids", []
                        ),
                        "scope_signature": item.get("scope_signature", {}),
                        "supporting_metrics": item.get(
                            "supporting_metrics", {}
                        ),
                    },
                    ensure_ascii=False,
                    default=str,
                )
                for item in memory_observations
            )
            + "\n"
            + "\n".join(
                str(item.get("fact", ""))
                for item in current_observation_items
                if isinstance(item, dict)
            )
        )


        allowed_numbers = (
            self._numeric_tokens(
                numeric_source
            )
        )


        # --------------------------------------------------------
        # D. First deterministic gate
        # --------------------------------------------------------

        initial_gate = {}

        for seed in seeds:

            hypothesis_id = (
                seed["hypothesis_id"]
            )

            initial_gate[
                hypothesis_id
            ] = (
                self._deterministic_gate(
                    seed,
                    valid_evidence_ids=(
                        valid_evidence_ids
                    ),
                    allowed_numbers=(
                        allowed_numbers
                    ),
                    problem=problem,
                    scientific_context=scientific_context,
                )
            )


        # --------------------------------------------------------
        # E. Single deterministic admission
        # --------------------------------------------------------

        # Hypothesis prose is produced by the one generation request above.
        # Critic/revision LLM loops are intentionally disconnected: they used
        # a second identifier mapping and could reject already-passed seeds as
        # missing_critic_decision. All admission facts are deterministic.
        initial_decisions = {
            seed["hypothesis_id"]: {
                "hypothesis_id": seed["hypothesis_id"],
                "decision": (
                    "PASS" if not initial_gate[seed["hypothesis_id"]]
                    else "REJECT"
                ),
                "issues": list(initial_gate[seed["hypothesis_id"]]),
                "rationale": "single_deterministic_admission",
                "validation_mode": "local_deterministic_only",
            }
            for seed in seeds
        }


        qualified = []
        rejected = []
        revisions = []

        final_gate = {}
        final_decisions = {}

        # No automatic LLM revision is permitted in the formal path.
        revision_results: dict[str, dict | None] = {}


        # --------------------------------------------------------
        # F. PASS / REVISE / REJECT
        # --------------------------------------------------------

        for seed in seeds:

            hypothesis_id = (
                seed["hypothesis_id"]
            )

            hard_issues = list(
                initial_gate.get(
                    hypothesis_id,
                    [],
                )
            )

            decision = (
                initial_decisions.get(
                    hypothesis_id
                )
            )


            if decision is None:

                rejected.append(
                    {
                        "hypothesis":
                            seed,

                        "decision":
                            "REJECT",

                        "issues":
                            hard_issues
                            + [
                                "missing_critic_decision"
                            ],

                        "rationale":
                            "Scientific critic returned no decision.",
                    }
                )

                continue


            critic_status = str(
                decision.get(
                    "decision",
                    "REJECT",
                )
            ).strip().upper()


            critic_issues = list(
                decision.get(
                    "issues",
                    [],
                )
                or []
            )


            all_issues = list(
                dict.fromkeys(
                    hard_issues
                    + critic_issues
                )
            )


            # ----------------------------------------------------
            # Direct PASS
            # ----------------------------------------------------

            if (
                critic_status == "PASS"
                and not all_issues
            ):

                passed = dict(
                    seed
                )

                passed["status"] = (
                    "qualified"
                )

                passed[
                    "critic_rationale"
                ] = decision.get(
                    "rationale"
                )

                passed[
                    "revision_count"
                ] = 0

                qualified.append(
                    passed
                )

                final_gate[
                    hypothesis_id
                ] = []

                final_decisions[
                    hypothesis_id
                ] = decision

                continue


            # ----------------------------------------------------
            # REJECT means no automatic rescue.
            # ----------------------------------------------------

            if critic_status == "REJECT":

                rejected.append(
                    {
                        "hypothesis":
                            seed,

                        "decision":
                            "REJECT",

                        "issues":
                            all_issues,

                        "rationale":
                            decision.get(
                                "rationale"
                            ),
                    }
                )

                final_gate[
                    hypothesis_id
                ] = hard_issues

                final_decisions[
                    hypothesis_id
                ] = decision

                continue


            # ----------------------------------------------------
            # REVISE:
            # maximum ONE scientific revision.
            # ----------------------------------------------------

            revised = revision_results.get(hypothesis_id)


            if revised is None:

                rejected.append(
                    {
                        "hypothesis":
                            seed,

                        "decision":
                            "REJECT",

                        "issues":
                            all_issues
                            + [
                                "revision_unrepairable"
                            ],

                        "rationale":
                            decision.get(
                                "rationale"
                            ),
                    }
                )

                continue

            # Provenance is immutable across LLM revision.  The
            # revision agent may repair wording, but it may not
            # relabel a literature hypothesis as experimental (or
            # vice versa), drop source observations, or attach new
            # experiment/opportunity identifiers.
            self._restore_provenance(seed, revised)


            revised["id"] = (
                hypothesis_id
            )

            revised[
                "hypothesis_id"
            ] = hypothesis_id

            revised["status"] = (
                "revised_seed"
            )


            revised_gate = (
                self._deterministic_gate(
                    revised,
                    valid_evidence_ids=(
                        valid_evidence_ids
                    ),
                    allowed_numbers=(
                        allowed_numbers
                    ),
                    problem=problem,
                    scientific_context=scientific_context,
                )
            )


            revised_decision = {
                "hypothesis_id": hypothesis_id,
                "decision": "PASS" if not revised_gate else "REJECT",
                "issues": list(revised_gate),
                "rationale": (
                    "scientific_revision_passed_local_deterministic_recheck"
                    if not revised_gate
                    else "scientific_revision_failed_local_deterministic_recheck"
                ),
                "validation_mode": (
                    "local_deterministic_recheck_after_parallel_revision"
                ),
            }


            revisions.append(
                {
                    "hypothesis_id":
                        hypothesis_id,

                    "original":
                        seed,

                    "original_decision":
                        decision,

                    "revised":
                        revised,

                    "revised_gate":
                        revised_gate,

                    "revised_decision":
                        revised_decision,
                }
            )


            if revised_decision is None:

                rejected.append(
                    {
                        "hypothesis":
                            revised,

                        "decision":
                            "REJECT",

                        "issues":
                            revised_gate
                            + [
                                "missing_second_critic_decision"
                            ],

                        "rationale":
                            "No critic decision after revision.",
                    }
                )

                continue


            revised_status = str(
                revised_decision.get(
                    "decision",
                    "REJECT",
                )
            ).strip().upper()


            revised_critic_issues = list(
                revised_decision.get(
                    "issues",
                    [],
                )
                or []
            )


            revised_all_issues = list(
                dict.fromkeys(
                    revised_gate
                    + revised_critic_issues
                )
            )


            final_gate[
                hypothesis_id
            ] = revised_gate

            final_decisions[
                hypothesis_id
            ] = revised_decision


            if (
                revised_status == "PASS"
                and not revised_all_issues
            ):

                revised[
                    "status"
                ] = "qualified"

                revised[
                    "revision_count"
                ] = 1

                revised[
                    "critic_rationale"
                ] = revised_decision.get(
                    "rationale"
                )

                qualified.append(
                    revised
                )


            else:

                rejected.append(
                    {
                        "hypothesis":
                            revised,

                        "decision":
                            revised_status,

                        "issues":
                            revised_all_issues,

                        "rationale":
                            revised_decision.get(
                                "rationale"
                            ),
                    }
                )


        # --------------------------------------------------------
        # G. Audit
        # --------------------------------------------------------

        # Deep critique may comment on a seed, but the original LLM hypothesis
        # is the immutable semantic identity used by ranking and planning.
        original_by_id = {item["hypothesis_id"]: item for item in seeds}
        immutable_keys = (
            "title", "hypothesis", "hypothesis_statement", "mechanism",
            "engineering_mechanism", "inference", "expected_observation",
            "variables", "key_variables", "applicability_conditions",
            "falsification_condition", "assumptions", "evidence_needed",
            "generation_source", "raw_hypothesis", "raw_hypothesis_sha256",
        )
        for item in qualified:
            original = original_by_id.get(item.get("hypothesis_id"), {})
            for key in immutable_keys:
                if key in original:
                    item[key] = original[key]

        if isinstance(experiment_memory, dict) and experiment_memory:
            from boilermind.core.contracts import ExperimentMemoryBundle
            from boilermind.experiment_memory.opportunity import check_hypothesis_duplication

            memory_model = ExperimentMemoryBundle.model_validate(experiment_memory)
            deduplicated = []
            for item in qualified:
                duplicate = check_hypothesis_duplication(item, memory_model)
                item["duplicate_check"] = duplicate
                if not item.get("trigger_types"):
                    item["trigger_types"] = ["HUMAN_PROPOSAL"]
                item["provenance"] = {
                    "trigger_types": item.get("trigger_types", []),
                    "source_observation_ids": item.get("source_observation_ids", []),
                    "source_experiment_ids": item.get("source_experiment_ids", []),
                    "source_literature_ids": item.get("evidence_ids", []),
                    "human_proposal_ids": [],
                    "opportunity_id": item.get("opportunity_id") or None,
                    "expected_information_gain": 0.0,
                    "capability_match_status": "PENDING_PLANNING_GATE",
                }
                from boilermind.hypothesis.deterministic_admission import evaluate_candidate
                item["deterministic_admission"] = evaluate_candidate(
                    item, problem, scientific_context
                )
                deduplicated.append(item)
            qualified = deduplicated

        for item in qualified:
            item.setdefault("historical_assessment", {
                "directly_supporting_observations": [],
                "conflicting_observations": [],
                "conditionally_related_observations": [],
                "duplicate_experiment_ids": [],
                "scope_mismatches": [],
                "historical_support_level": "NONE",
                "duplicate_status": "NEW",
                "evidence_gap": list(item.get("evidence_needed", [])),
            })

        # Freeze the deterministic scientific meaning before ranking and
        # planning. The LLM prose remains available for explanation, but the
        # executable design may no longer silently drift afterwards.
        from boilermind.planning.experiment_requirement_parser import (
            freeze_hypothesis_design,
            frozen_design_sha256,
        )
        for item in qualified:
            if not item.get("scientific_design"):
                frozen_design = freeze_hypothesis_design(item)
                item["scientific_design"] = frozen_design.model_dump(mode="json")
                item["scientific_design_sha256"] = frozen_design_sha256(
                    frozen_design
                )

        project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        debug_dir = (
            project_root
            / "outputs"
            / "debug"
        )

        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        debug_path = (
            debug_dir
            / "hypothesis_seed_latest.json"
        )


        audit = {

            "hypothesis_generation_mode": generation_mode,

            "problem":
                problem,

            "evidence_bundle_id": None,

            "evidence_bundle_sha256": None,

            "literature_grounding_disabled": True,

            "evidence_claims":
                evidence_claims,

            "generated_seeds":
                original_generated_seeds,

            "compiled_hypotheses":
                seeds,

            "hypothesis_compilation":
                hypothesis_compilation,

            "initial_deterministic_gate":
                initial_gate,

            "initial_critic_decisions":
                initial_decisions,

            "revisions":
                revisions,

            "final_deterministic_gate":
                final_gate,

            "final_critic_decisions":
                final_decisions,

            "qualified_hypotheses":
                qualified,

            "rejected_hypotheses":
                rejected,

            "generated_count":
                len(original_generated_seeds),

            "qualified_count":
                len(qualified),

            "rejected_count":
                len(rejected),
        }


        debug_path.write_text(
            json.dumps(
                audit,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


        # Fail closed.
        if not qualified:

            raise RuntimeError(
                "no_qualified_hypothesis_seeds;"
                f"debug_file={debug_path}"
            )


        return {

            "hypotheses":
                qualified,

            "qualified_hypotheses":
                qualified,

            "rejected_hypotheses":
                rejected,

            "evidence_claims":
                evidence_claims,

            "generated_hypothesis_count":
                len(original_generated_seeds),

            "qualified_hypothesis_count":
                len(qualified),

            "rejected_hypothesis_count":
                len(rejected),

            "hypothesis_count":
                len(qualified),

            "hypothesis_audit_path":
                str(debug_path),

            "generation_audit":
                audit,

            "hypothesis_compilation":
                hypothesis_compilation,

            "original_hypotheses":
                original_generated_seeds,

            "status":
                "qualified",
        }

