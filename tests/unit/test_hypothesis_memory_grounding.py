import pytest
import json

from boilermind.skills.hypothesis_skill import HypothesisGenerationSkill


def test_seed_prompt_compacts_and_limits_historical_memory(monkeypatch):
    skill = HypothesisGenerationSkill()
    prompts = []
    monkeypatch.setattr(
        skill,
        "_generate",
        lambda prompt: prompts.append(prompt) or '{"seeds": []}',
    )
    observations = [
        {
            "observation_id": f"OBS-{index}",
            "source_experiment_ids": [f"EXP-{index}"],
            "observation_type": "SUPPORTED",
            "claim": f"claim-{index}",
            "supporting_metrics": {"MAE": float(index)},
            "unused_blob": "x" * 20_000,
        }
        for index in range(8)
    ]
    skill._generate_seeds({
        "problem_id": "P-1",
        "original_question": "比较Ridge与RF",
        "required_models": ["ridge", "rf"],
        "_experiment_memory": {
            "supported_observations": observations,
            "falsified_observations": [],
            "contradictions": [],
            "engineering_failures": [],
        },
    })

    assert len(prompts) == 1
    assert "OBS-0" in prompts[0]
    assert "OBS-2" in prompts[0]
    assert "OBS-3" not in prompts[0]
    assert "unused_blob" not in prompts[0]
    assert len(prompts[0]) < 15_000
    assert "模型比较问题只生成1条汇总比较假设" in prompts[0]


def test_generation_prompt_contains_history_and_capability_but_no_literature(monkeypatch):
    captured = {}

    def generate(prompt):
        captured["prompt"] = prompt
        return '{"seeds": []}'

    monkeypatch.setattr(HypothesisGenerationSkill, "_generate", staticmethod(generate))
    HypothesisGenerationSkill()._generate_seeds(
        {
            "problem_id": "P-1",
            "original_question": "哪个模型更好",
            "_neutral_capabilities": {
                "enabled_experiment_models": ["ridge", "rf"],
                "reference_model": "persistence",
            },
            "_experiment_memory": {
                "supported_observations": [{
                    "observation_id": "OBS-REAL",
                    "source_experiment_ids": ["EXP-REAL"],
                }],
            },
        },
    )
    prompt = captured["prompt"]
    assert "OBS-REAL" in prompt and "EXP-REAL" in prompt
    assert "EVIDENCE-REAL" not in prompt
    assert "Verified Literature Claims" not in prompt
    assert "evidence_claims" not in prompt
    assert "evidence_id" not in prompt
    assert "ridge" in prompt and "rf" in prompt and "persistence" in prompt


def test_formal_generation_never_calls_critic_or_revision(monkeypatch):
    skill = HypothesisGenerationSkill()
    raw = {
        "title": "可执行比较假设",
        "hypothesis_statement": "当前候选模型的预测误差存在可比较差异",
        "engineering_mechanism": "不同模型表示能力形成预测误差差异",
        "expected_observation": "候选模型的MAE排序存在差异",
        "key_variables": [],
        "applicability_conditions": [],
        "falsification_condition": "所有候选模型的MAE完全相同",
        "assumptions": [],
        "evidence_needed": [],
    }
    generation_calls = []
    monkeypatch.setattr(
        skill,
        "_generate_seeds",
        lambda *_args: generation_calls.append(_args) or [raw],
    )
    monkeypatch.setattr(
        skill,
        "_extract_evidence_claims",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("literature_extraction_must_not_run")
        ),
    )
    monkeypatch.setattr(skill, "_deterministic_gate", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        skill,
        "_critic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy_critic_must_not_run")
        ),
    )
    monkeypatch.setattr(
        skill,
        "_revise_seeds_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy_revision_must_not_run")
        ),
    )
    result = skill.execute({
        "research_problem": {
            "problem_id": "P-1",
            "original_question": "哪个模型预测效果更好",
            "target_variable": "steam_volumetric_flow",
        },
        "scientific_context": {},
        "evidence_bundle": {
            "bundle_id": "B-IGNORED",
            "problem_id": "P-1",
            "evidence": [{"evidence_id": "E-IGNORED", "text": "literature"}],
            "sha256": "a" * 64,
        },
        "hypothesis_generation_mode": "deep",
    })
    assert len(generation_calls) == 1
    assert result["evidence_claims"] == []
    assert len(result["qualified_hypotheses"]) == 1
    assert result["qualified_hypotheses"][0]["revision_count"] == 0


def test_free_hypothesis_semantics_are_not_overwritten_by_problem_or_history():
    locked = HypothesisGenerationSkill._lock_execution_fields(
        [{
            "title": "错误自由输出",
            "hypothesis": "RF在h4的RMSE更好",
            "mechanism": "待验证机制",
            "inference": "待验证推论",
            "variables": [{"name": "fake_sensor"}],
            "source_observation_ids": ["FAKE"],
            "source_experiment_ids": ["EXP-REAL"],
            "evidence_gap": "缺少当前实验",
        }],
        {
            "required_models": ["bayesianridge"],
            "reference_models": ["persistence"],
            "metrics": ["MAE", "RMSE"],
            "required_horizon_steps": 40,
            "target_variable": "steam_volumetric_flow",
            "operating_condition": "深度调峰",
        },
        [{
            "observation_id": "OBS-REAL",
            "source_experiment_ids": ["EXP-REAL"],
            "claim": "bayesianridge历史观察",
            "invalid_for_scientific_synthesis": False,
        }, {
            "observation_id": "OBS-WRONG-RIDGE-SUBSTRING",
            "source_experiment_ids": ["EXP-WRONG"],
            "claim": "ridge历史观察",
            "invalid_for_scientific_synthesis": False,
        }],
        {"opportunities": [{
            "opportunity_id": "OPP-REAL",
            "source_observation_ids": ["OBS-REAL"],
        }]},
    )
    assert len(locked) == 1
    seed = locked[0]
    assert seed["hypothesis"] == "RF在h4的RMSE更好"
    assert seed["mechanism"] == "待验证机制"
    assert seed["source_observation_ids"] == []
    assert seed["source_experiment_ids"] == ["EXP-REAL"]
    assert seed["trigger_types"] == ["HISTORICAL_EXPERIMENT"]
    assert len(seed["raw_hypothesis_sha256"]) == 64


def test_free_candidates_keep_original_order_and_identity():
    locked = HypothesisGenerationSkill._lock_execution_fields(
        [
            {
                "id": "H001",
                "hypothesis_id": "H001",
                "hypothesis": "bayesianridge优于persistence",
                "mechanism": "bayesianridge专属机制",
                "inference": "待验证",
                "evidence_gap": "缺口",
            },
            {
                "id": "H002",
                "hypothesis_id": "H002",
                "hypothesis": "ridge优于persistence",
                "mechanism": "ridge专属机制",
                "inference": "待验证",
                "evidence_gap": "缺口",
            },
        ],
        {
            "required_models": ["ridge", "bayesianridge"],
            "reference_models": ["persistence"],
            "metrics": ["MAE"],
            "required_horizon_steps": 80,
            "target_variable": "steam_volumetric_flow",
            "operating_condition": "深度调峰",
        },
        [],
        {},
    )
    assert len(locked) == 2
    assert all(seed["trigger_types"] == ["HUMAN_PROPOSAL"] for seed in locked)
    assert locked[0]["hypothesis"] == "bayesianridge优于persistence"
    assert locked[1]["hypothesis"] == "ridge优于persistence"
    assert locked[0]["mechanism"] == "bayesianridge专属机制"
    assert locked[1]["mechanism"] == "ridge专属机制"
    assert [seed["hypothesis_id"] for seed in locked] == ["H001", "H002"]


def test_historical_observation_can_reach_generation_without_literature(monkeypatch):
    skill = HypothesisGenerationSkill()
    captured = {}

    def capture(problem):
        captured["problem"] = problem
        return []

    monkeypatch.setattr(skill, "_generate_seeds", capture)

    with pytest.raises(RuntimeError, match="no_hypothesis_seeds_generated"):
        skill.execute({
            "research_problem": {
                "problem_id": "P-1",
                "original_question": "验证跨时间块稳定性",
            },
            "experiment_memory_bundle": {
                "problem_id": "P-1",
                "supported_observations": [],
                "falsified_observations": [],
                "contradictions": [{
                    "observation_id": "OBS-1",
                    "claim": "h80在不同时间块发生模型排名翻转。",
                    "source_experiment_ids": ["BM-TIME-01"],
                }],
                "engineering_failures": [],
            },
            "opportunity_map": {},
            "scientific_context": {},
        })
    assert captured["problem"]["_experiment_memory"]["contradictions"][0][
        "source_experiment_ids"
    ] == ["BM-TIME-01"]


def test_human_problem_is_a_valid_generation_source_without_history():
    sealed = HypothesisGenerationSkill._lock_execution_fields(
        [{
            "title": "响应滞后",
            "hypothesis_statement": "升负荷时蒸汽流量响应滞后更明显",
            "engineering_mechanism": "蓄热变化造成动态响应差异",
            "expected_observation": "升负荷阶段的滞后量更大",
            "falsification_condition": "两方向滞后量无差异或方向相反",
        }],
        {"problem_id": "P-1"}, [], {},
    )
    assert sealed[0]["generation_source"] == "llm_grounded_generation"
    assert sealed[0]["trigger_types"] == ["HUMAN_PROPOSAL"]


def test_revision_cannot_replace_provenance(monkeypatch):
    skill = HypothesisGenerationSkill()
    original = {
        "title": "原假设",
        "hypothesis": "原假设内容",
        "mechanism": "原机制",
        "evidence_ids": ["E-1"],
        "source_observation_ids": ["OBS-1"],
        "source_experiment_ids": ["EXP-1"],
        "opportunity_id": "OPP-1",
        "trigger_types": ["HISTORICAL_EXPERIMENT"],
        "inference": "原推论",
        "variables": ["x"],
        "verification_intent": "验证",
        "falsification_condition": "不优于对照",
        "evidence_gap": "待复验",
    }
    response = {
        "decision": "REVISED",
        "reason": "修订措辞",
        "seed": {
            **original,
            "title": "修订假设",
            "evidence_ids": ["E-FAKE"],
        },
    }
    monkeypatch.setattr(skill, "_generate", lambda _prompt: json.dumps(response))

    revised = skill._revise_seed({}, [], original, {}, [])
    skill._restore_provenance(original, revised)

    assert revised["evidence_ids"] == ["E-1"]
    assert revised["source_observation_ids"] == ["OBS-1"]


def test_numeric_grounding_ignores_digits_inside_provenance_ids():
    skill = HypothesisGenerationSkill()
    seed = {
        "title": "稳定性假设",
        "hypothesis": "候选模型比对照更稳定",
        "mechanism": "正则化可能降低漂移敏感性",
        "verification_intent": "比较跨时间块误差",
        "falsification_condition": "候选模型不优于对照时证伪",
        "evidence_gap": "仍需独立复验",
        "evidence_ids": [],
        "source_observation_ids": ["CUR-5892ad7960c8"],
        "source_experiment_ids": ["BM-TIME-01-H80"],
    }

    issues = skill._deterministic_gate(
        seed,
        valid_evidence_ids={"CUR-5892ad7960c8"},
        allowed_numbers=set(),
    )

    assert not any(issue.startswith("unsupported_numeric_claim") for issue in issues)


def test_metric_equality_does_not_look_like_time_mapping():
    skill = HypothesisGenerationSkill()
    seed = {
        "title": "误差假设",
        "hypothesis": "10分钟后（40步）的 MAE 更低",
        "mechanism": "正则化降低误差",
        "verification_intent": "比较 MAE",
        "falsification_condition": "MAE 大于或等于 Persistence 时证伪",
        "evidence_gap": "待验证",
        "evidence_ids": [],
    }
    issues = skill._deterministic_gate(
        seed,
        valid_evidence_ids=set(),
        allowed_numbers={"10", "40"},
        scientific_context={"sampling_interval_seconds": 15},
    )
    assert "unjustified_step_to_time_mapping" not in issues


def test_unverified_time_mapping_is_rejected():
    skill = HypothesisGenerationSkill()
    seed = {
        "title": "时域假设",
        "hypothesis": "10分钟对应40步",
        "mechanism": "待验证",
        "verification_intent": "比较误差",
        "falsification_condition": "不优于对照时证伪",
        "evidence_gap": "采样周期未知",
        "evidence_ids": [],
    }
    issues = skill._deterministic_gate(
        seed,
        valid_evidence_ids=set(),
        allowed_numbers={"10", "40"},
        scientific_context={},
    )
    assert "unjustified_step_to_time_mapping" in issues


def test_explicit_human_proposal_does_not_require_observation_id():
    skill = HypothesisGenerationSkill()
    seed = {
        "title": "用户明确提出的假设",
        "hypothesis": "Ridge 的 MAE 低于 Persistence",
        "mechanism": "正则化可能降低误差",
        "verification_intent": "锁定测试集比较 MAE",
        "falsification_condition": "Ridge MAE 不低于 Persistence 时证伪",
        "evidence_gap": "尚未执行本次实验",
        "evidence_ids": [],
        "source_observation_ids": [],
        "trigger_types": ["HUMAN_PROPOSAL"],
    }
    issues = skill._deterministic_gate(
        seed,
        valid_evidence_ids=set(),
        allowed_numbers=set(),
    )
    assert "missing_grounding_source_ids" not in issues


def test_candidate_grounding_subset_drops_unreferenced_history():
    problem = {
        "_experiment_memory": {
            "supported_observations": [
                {"observation_id": "OBS-KEEP", "experiment_id": "EXP-KEEP"},
                {"observation_id": "OBS-DROP", "experiment_id": "EXP-DROP"},
            ],
            "falsified_observations": [], "contradictions": [],
            "engineering_failures": [],
            "completed_experiment_ids": ["EXP-KEEP", "EXP-DROP"],
        },
        "_opportunity_map": {"opportunities": [
            {"opportunity_id": "OPP-KEEP", "source_observation_ids": ["OBS-KEEP"]},
            {"opportunity_id": "OPP-DROP", "source_observation_ids": ["OBS-DROP"]},
        ]},
    }
    seed = {
        "source_observation_ids": ["OBS-KEEP"],
        "source_experiment_ids": ["EXP-KEEP"],
        "opportunity_id": "OPP-KEEP", "evidence_ids": [],
    }
    subset = HypothesisGenerationSkill._grounding_subset(problem, [seed])
    assert [item["observation_id"] for item in subset["_experiment_memory"]["supported_observations"]] == ["OBS-KEEP"]
    assert subset["_experiment_memory"]["completed_experiment_ids"] == ["EXP-KEEP"]
    assert [item["opportunity_id"] for item in subset["_opportunity_map"]["opportunities"]] == ["OPP-KEEP"]
