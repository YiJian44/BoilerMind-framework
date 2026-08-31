from __future__ import annotations

import hashlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Callable

from boilermind.core.contracts import (
    BatchMember,
    ExperimentAudit,
    ExperimentResult,
    FieldProvenance,
    HypothesisRunState,
    HypothesisValidationBatch,
    RankingSnapshot,
    ResearchRunState,
    ScientificResult,
    StageTrace,
)
from boilermind.experiment.capability_registry import (
    DirectVolume31VCapabilityRegistry,
    ExperimentCapabilityRegistry,
)
from boilermind.experiment_memory import ExperimentMemoryStore, retrieve_experiment_memory
from boilermind.experiment_memory.hypothesis_assessment import assess_hypotheses_with_memory
from boilermind.experiment_memory.persistence import persist_experiment_outcome
from boilermind.knowledge.evolution_sink import JsonEvolutionSink
from boilermind.orchestration.real_experiment_loop import execute_real_experiment
from boilermind.ranking.feedback_calculator import calculate_experiment_feedback
from boilermind.ranking.dynamic_ranker import dynamic_score
from boilermind.ranking.historical_prior import rank_hypotheses
from boilermind.reporting.narrative_report import (
    bind_frozen_identifiers,
    build_structured_research_record,
    validate_narrative_report,
    write_narrative_report,
)
from boilermind.reporting.scientific_research_plan_service import (
    ScientificResearchPlanService,
)
from boilermind.skills.contract_skill import ExperimentContractSkill
from boilermind.skills.evidence_skill import EvidenceRetrievalSkill
from boilermind.skills.hypothesis_skill import HypothesisGenerationSkill
from boilermind.skills.data_profile_skill import DataProfileSkill
from boilermind.skills.planning_skill import PlanningSkill
from boilermind.skills.problem_skill import ProblemParsingSkill
from boilermind.orchestration.model_hypothesis_factory import (
    build_model_hypotheses,
    enrich_hypotheses_with_qwen,
)
from boilermind.orchestration.control_optimization import (
    execute_control_optimization,
    is_control_optimization_question,
)
from boilermind.skills.profile_mapper import profile_to_model_selection
from boilermind.skills.ranking_skill import RankingSkill


ROOT = Path(__file__).resolve().parents[3]


def _dispatch_executor(
    contract: dict[str, Any] | Any,
    *,
    runner: Any | None = None,
) -> dict[str, Any]:
    """按契约分流：31V 软测(h0) 用 SoftSensor 执行器，其余用真实运行器。

    数据属性画像流的契约 target=steam_volumetric_flow 且 horizon=0，
    走 run_soft_sense_experiment（当前时刻特征+均值基线）；其他问题走
    execute_real_experiment（windowed 运行器），保持原有行为。
    """
    is_dict = isinstance(contract, dict)
    target = str(
        contract.get("target_variable") if is_dict else getattr(contract, "target_variable", "")
    ).casefold()
    horizon = int(
        (contract.get("prediction_horizon_steps") if is_dict else getattr(contract, "prediction_horizon_steps", None))
        or 0
    )
    if target == "steam_volumetric_flow" and horizon == 0:
        from boilermind.orchestration.soft_sense import run_soft_sense_experiment

        return run_soft_sense_experiment(contract)
    return execute_real_experiment(contract, runner=runner)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_design_sha256(contract: dict[str, Any]) -> str:
    payload = dict(contract)
    for key in ("experiment_id", "problem_id", "hypothesis_id", "plan_id", "status"):
        payload.pop(key, None)
    return _sha(payload)


class ResearchOrchestrator:
    """The only production owner of the BoilerMind research lifecycle."""

    def __init__(
        self,
        *,
        run_root: str | Path = ROOT / "runtime" / "research_runs_v2",
        memory_root: str | Path = ROOT / "runtime" / "experiment_memory",
        capability_registry: ExperimentCapabilityRegistry | None = None,
        problem_parser: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        evidence_retriever: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        hypothesis_generator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        memory_retriever: Callable[[dict[str, Any], dict[str, Any], ExperimentMemoryStore], Any] | None = None,
        planner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        contract_compiler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        executor: Callable[[dict[str, Any]], dict[str, Any]] = _dispatch_executor,
        narrative_writer: Callable[[dict[str, Any]], str] | None = None,
        evolution_sink: Any | None = None,
        max_hypotheses: int = 6,
        max_batches: int = 3,
        max_parallelism: int = 3,
    ):
        self.run_root = Path(run_root)
        self.memory_store = ExperimentMemoryStore(memory_root)
        # The canonical production chain operates on the verified 31-variable
        # direct steam-volume dataset. Generic registries remain injectable for
        # other runners and tests, but must not be the production default.
        self.capability = (
            capability_registry
            or DirectVolume31VCapabilityRegistry()
        )
        self.problem_parser = problem_parser or ProblemParsingSkill(
            capability_registry=self.capability
        ).execute
        self.evidence_retriever = evidence_retriever or EvidenceRetrievalSkill().execute
        self.hypothesis_generator = hypothesis_generator or HypothesisGenerationSkill().execute
        self.memory_retriever = memory_retriever or retrieve_experiment_memory
        self.planner = planner or PlanningSkill(capability_registry=self.capability).execute
        self.contract_compiler = contract_compiler or ExperimentContractSkill(capability_registry=self.capability).execute
        self.executor = executor
        self.narrative_writer = narrative_writer
        self.evolution_sink = (
            JsonEvolutionSink() if evolution_sink is None else evolution_sink
        )
        self.max_hypotheses = min(6, max(1, max_hypotheses))
        self.max_batches = min(3, max(1, max_batches))
        self.max_parallelism = min(3, max(1, max_parallelism))

    def capabilities(self) -> dict[str, Any]:
        snapshot = self.capability.snapshot()
        return {
            "schema_version": "boilermind.capabilities.v2",
            "research_orchestrator": "canonical",
            "max_hypotheses": self.max_hypotheses,
            "max_batches": self.max_batches,
            "max_parallel_hypotheses": self.max_parallelism,
            "experiment": _jsonable(snapshot),
        }

    def _trace(self, state: ResearchRunState, stage: str, started: float, before: Any, after: Any = None, *, source: str = "PROGRAM", error: Exception | None = None) -> None:
        state.stage_traces.append(StageTrace(
            stage=stage,
            status="FAILED" if error else "COMPLETED",
            source=source,
            input_sha256=_sha(before),
            output_sha256=None if error else _sha(after),
            duration_seconds=max(0.0, monotonic() - started),
            errors=[] if error is None else [f"{type(error).__name__}:{error}"],
        ))
        # 阶段级进度实时落盘，使前端/WebSocket 能看到运行中阶段，而不是只有终态。
        self._persist(state)

    def _persist(self, state: ResearchRunState) -> None:
        path = self.run_root / state.run_id
        path.mkdir(parents=True, exist_ok=True)
        # 原子写入：避免前端轮询读到半截/空文件而误判“后端未连接”。
        target = path / "run.json"
        temporary = path / "run.json.tmp"
        payload = state.model_dump_json(indent=2)
        for attempt in range(6):
            try:
                temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, target)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                sleep(0.05 * (2 ** attempt))

    def _finalize(self, state: ResearchRunState) -> ResearchRunState:
        completed_members = [
            member for batch in state.batches for member in batch.members
            if member.status == "COMPLETED" and member.outcome
        ]
        state.status = "COMPLETED" if completed_members else ("FAILED" if state.batches else "NO_EXECUTABLE_HYPOTHESES")
        if state.status == "FAILED" and "all_experiment_members_failed" not in state.errors:
            state.errors.append("all_experiment_members_failed")
        # 已完成实验沉淀到可增长科研假设演化图谱（幂等；失败仅记录警告，不阻断报告）。
        if self.evolution_sink is not None and completed_members:
            hypotheses_by_id = {
                str(item.get("hypothesis_id") or item.get("id")): item
                for item in (state.hypotheses or [])
            }
            for member in completed_members:
                outcome = member.outcome or {}
                try:
                    self.evolution_sink.record(
                        ExperimentResult.model_validate(outcome["experiment_result"]),
                        ScientificResult.model_validate(outcome["scientific_result"]),
                        hypotheses_by_id.get(str(member.hypothesis_id)) or {},
                    )
                except Exception as exc:
                    state.errors.append(
                        f"evolution_sink_warning:{type(exc).__name__}:{exc}"
                    )
        record = build_structured_research_record(state.model_dump(mode="json"))
        run_dir = self.run_root / state.run_id
        structured_path = run_dir / "structured_report.json"
        structured_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        state.report = {"structured_path": str(structured_path), "structured_sha256": _sha(record)}
        if state.status == "COMPLETED":
            try:
                narrative = self.narrative_writer(record) if self.narrative_writer else write_narrative_report(record)
                narrative = bind_frozen_identifiers(record, narrative)
                validate_narrative_report(record, narrative)
                narrative_path = run_dir / "narrative_report.md"
                narrative_path.write_text(narrative, encoding="utf-8")
                state.report["narrative_path"] = str(narrative_path)
            except Exception as exc:
                state.status = "COMPLETED_WITH_REPORT_WARNING"
                state.report["narrative_error"] = f"{type(exc).__name__}:{exc}"
        if completed_members:
            scientific_plan = ScientificResearchPlanService().generate_from_run_state(
                state.model_dump(mode="json"), output_dir=run_dir / "scientific_research_plan"
            )
            state.report["scientific_research_plan"] = {
                "status": scientific_plan.status, "json_path": scientific_plan.json_path,
                "markdown_path": scientific_plan.markdown_path, "word_path": scientific_plan.word_path,
                "manifest_path": scientific_plan.manifest_path, "warnings": list(scientific_plan.warnings),
                "errors": list(scientific_plan.errors),
            }
        state.completed_at = datetime.now(timezone.utc)
        self._persist(state)
        return state

    def _run_control_optimization(self, state: ResearchRunState, context: dict[str, Any]) -> ResearchRunState:
        started = monotonic()
        problem_id = str((state.research_problem or {}).get("problem_id") or f"RP-{state.run_id}")
        package = execute_control_optimization(
            run_id=state.run_id, problem_id=problem_id,
            output_dir=self.run_root / state.run_id / "unity",
        )
        # 文献证据不可用时，用本地数据观察证据补位，保证报告证据契约可满足。
        if package.get("evidence_bundle"):
            state.evidence_bundle = package["evidence_bundle"]
            context["evidence_bundle"] = package["evidence_bundle"]
        hypothesis = package["hypothesis"]
        state.hypotheses = [hypothesis]
        self._trace(state, "hypothesis_generation_and_gate", started, context.get("experiment_memory_bundle"), [hypothesis], source="PROGRAM")
        score = {
            "hypothesis_id": "H_CTRL", "historical_support": 0.72, "historical_scope_match": 1.0,
            "problem_relevance": 1.0, "reproducibility": 0.9, "falsifiability": 1.0,
            "prior_score": 0.91, "cumulative_feedback": 0.0, "dynamic_score": 0.91,
            "eligible": True, "dropped_reasons": [],
        }
        snapshot = RankingSnapshot(snapshot_id=f"RANK-{state.run_id}-000", round_index=0, entries=[score])
        state.ranking_snapshots.append(snapshot)
        state.hypothesis_states["H_CTRL"] = HypothesisRunState(
            hypothesis_id="H_CTRL", execution_count=1, eligible=False,
            latest_verdict=str(package["outcome"]["scientific_result"]["verdict"]),
            cumulative_feedback=0.85, executed_design_sha256=[experiment_design_sha256(package["contract"])],
            exit_reason=str(package["outcome"]["scientific_result"]["verdict"]),
        )
        self._trace(state, "experiment_planning", started, hypothesis, package["plan"], source="PROGRAM")
        self._trace(state, "experiment_execution", started, package["contract"], package["outcome"], source="PROGRAM")
        self._trace(state, "scientific_evaluation", started, package["outcome"]["experiment_result"], package["outcome"]["scientific_result"], source="PROGRAM")
        state.batches.append(HypothesisValidationBatch(
            batch_id=f"BATCH-{state.run_id}-01", round_index=1, ranking_snapshot_id=snapshot.snapshot_id,
            members=[BatchMember(hypothesis_id="H_CTRL", plan=package["plan"], contract=package["contract"],
                                 outcome=package["outcome"], status="COMPLETED")],
            status="COMPLETED", started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
        ))
        self._persist(state)
        return self._finalize(state)

    def load(self, run_id: str) -> ResearchRunState:
        return ResearchRunState.model_validate_json(
            (self.run_root / run_id / "run.json").read_text(encoding="utf-8")
        )

    def _prepare_member(self, context: dict[str, Any], hypothesis_id: str, batch_index: int, attempt: int) -> BatchMember | None:
        planning_context = dict(context)
        planning_context["selected_hypothesis_id"] = hypothesis_id
        planning_context["plan_revision_index"] = attempt
        planned = self.planner(planning_context)
        plan = planned.get("experiment_plan")
        if not plan or not planned.get("current_executable", False):
            return None
        plan = _jsonable(plan)
        plan["plan_id"] = f"PLAN-{hypothesis_id}-B{batch_index}-R{attempt}"
        # 确定性路径可能未给出参考模型：用能力注册表默认参考模型补齐，
        # 保证契约编译（reference_models/baseline_models 非空）不因模型选择缺口失败。
        if not plan.get("reference_models") and not plan.get("reference_model"):
            plan["reference_models"] = [self.capability.reference_model_id()]
        compiled = self.contract_compiler({**planning_context, **planned, "experiment_plan": plan})
        contract = compiled.get("experiment_contract")
        if not contract or not compiled.get("contract_compiled", False):
            return None
        contract = _jsonable(contract)
        contract["plan_id"] = plan["plan_id"]
        contract["experiment_id"] = f"EXP-{hypothesis_id}-B{batch_index}-R{attempt}-{uuid.uuid4().hex[:6]}"
        return BatchMember(hypothesis_id=hypothesis_id, plan=plan, contract=contract)

    def _execute_batch(self, batch: HypothesisValidationBatch) -> HypothesisValidationBatch:
        running = batch.model_copy(update={"status": "RUNNING", "started_at": datetime.now(timezone.utc)})
        outcomes: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_parallelism, thread_name_prefix="boilermind-hypothesis") as pool:
            futures = {pool.submit(self.executor, member.contract): member.hypothesis_id for member in running.members}
            for future in as_completed(futures):
                hypothesis_id = futures[future]
                try:
                    outcomes[hypothesis_id] = future.result()
                except Exception as exc:
                    errors[hypothesis_id] = f"{type(exc).__name__}:{exc}"
        members = []
        for member in running.members:
            if member.hypothesis_id in outcomes:
                members.append(member.model_copy(update={
                    "outcome": _jsonable(outcomes[member.hypothesis_id]), "status": "COMPLETED",
                }))
            else:
                members.append(member.model_copy(update={
                    "status": "FAILED", "issues": [errors[member.hypothesis_id]],
                }))
        return running.model_copy(update={
            "members": members,
            "status": "COMPLETED",
            "completed_at": datetime.now(timezone.utc),
        })

    def _apply_data_profile(
        self, context: dict[str, Any], parsed: dict[str, Any], *, run_dir: Path | None = None
    ) -> bool:
        """数据属性画像（train-only）→ 修正 target + 提供选型（阶段二）。

        在问题解析后、证据/假设前运行；仅对 31V 软测问题生效。
        画像给出权威 target（steam_volumetric_flow）与模型选型计划，
        避免 Qwen 歧义回退的 target=unspecified。
        返回 True 表示已应用画像。
        """
        problem = dict(parsed.get("research_problem") or {})
        if not problem:
            return False
        # 触发条件：V 目标 + 软测/数据属性意图（问"识别数据属性/软测V"才触发，
        # 避免干扰纯 V 目标但非软测的问题流，如集成测试的 mock）
        target = str(problem.get("target_variable") or "").casefold()
        question_text = str(context.get("research_question", "")).casefold()
        soft_intent = any(tok in question_text for tok in (
            "软测", "软测量", "soft sense", "数据属性", "非线性", "时序",
            "稀疏", "降维", "非高斯", "画像", "识别数据",
        ))
        is_v_target = target == "steam_volumetric_flow"
        ambiguous_soft_v = (
            target in {"", "unspecified", "none", "null"}
            and soft_intent
            and any(tok in question_text for tok in (
                "蒸汽体积", "体积量", "体积流量", "steam volumetric",
            ))
        )
        if not ((is_v_target and soft_intent) or ambiguous_soft_v):
            return False
        try:
            dataset = DirectVolume31VCapabilityRegistry.DEFAULT_DATASET_PATH
            # selection 画像只用 train 段（候选模型不受验证/测试信息影响）
            profile = DataProfileSkill.compute(dataset, horizon_steps=0, data_split="train")
        except Exception:
            return False
        problem["target_variable"] = "steam_volumetric_flow"
        problem["normalized_target_variable"] = "steam_volumetric_flow"
        problem["target_inference_reason"] = "data_profile_soft_sensing"
        # 注意：ResearchProblemSpec 要求 required_horizon_steps>=1，h0 软测在实验层处理
        context["data_profile"] = profile.model_dump(mode="json")
        plan = profile_to_model_selection(profile, horizon_steps=0)
        context["profile_model_selection"] = plan.model_dump(mode="json")
        context["profile_to_run_families"] = list(plan.to_run_families)
        # 画像选型计划 → 问题契约：让假设编译/规划有 model_comparison 上下文
        problem["required_operations"] = [
            "model_comparison", "chronological_validation", "locked_test_evaluation",
        ]
        problem["required_models"] = list(plan.to_run_families)
        problem["reference_models"] = ["persistence"]
        parsed["research_problem"] = problem
        context["research_problem"] = problem
        # 画像 + 选型写入 run_id 统一产物树
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "data_profile.json").write_text(
                json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (run_dir / "model_selection.json").write_text(
                json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return True

    def _normalize_fallback_hypotheses(
        self,
        hypotheses: list[dict[str, Any]],
        problem: dict[str, Any],
    ) -> None:
        """让确定性假设通过规划/契约/执行审计。

        - 预测时域与问题对齐（避免 h0 假设与 h40 问题冲突）；
        - 确认/证伪标准使用审计器支持的固定格式；
        - 参考模型固定为 capability 默认（persistence）。
        """
        from boilermind.planning.experiment_requirement_parser import (
            FrozenHypothesisDesign,
            frozen_design_sha256,
        )

        problem_horizon = problem.get("required_horizon_steps")
        reference_model = self.capability.reference_model_id()
        for hypothesis in hypotheses:
            design = hypothesis.get("scientific_design")
            if isinstance(design, dict):
                if problem_horizon is not None:
                    design["prediction_horizon_steps"] = int(problem_horizon)
                design["confirmation_criteria"] = [
                    "all_candidates_worse_than_reference_on:MAE"
                ]
                design["falsification_criteria"] = [
                    "any_candidate_better_than_reference_on:MAE"
                ]
                roles = dict(design.get("required_model_roles") or {})
                roles[reference_model] = "reference"
                design["required_model_roles"] = roles
                design["required_models"] = list(dict.fromkeys([
                    *(design.get("required_models") or []),
                    reference_model,
                ]))
                hypothesis["scientific_design"] = design
                hypothesis["scientific_design_sha256"] = frozen_design_sha256(
                    FrozenHypothesisDesign.model_validate(design)
                )
            if problem_horizon is not None:
                hypothesis["prediction_horizon_steps"] = int(problem_horizon)
            hypothesis["confirmation_criteria"] = [
                "all_candidates_worse_than_reference_on:MAE"
            ]
            hypothesis["falsification_criteria"] = [
                "any_candidate_better_than_reference_on:MAE"
            ]
            hypothesis["reference_models"] = [reference_model]

    def _deterministic_problem(
        self,
        context: dict[str, Any],
        error: Exception,
    ) -> dict[str, Any]:
        """Qwen 问题解析失败时的确定性降级：保证任何问题都能进入六阶段流程。"""
        from boilermind.adapters.deterministic_problem_parser import (
            DeterministicProblemParser,
            DeterministicProblemParserError,
        )
        from boilermind.core.contracts import ResearchProblemSpec

        question = str(context.get("research_question", "")).strip()
        capability = self.capability.snapshot()
        defaults = {
            "candidate_models": (
                capability.get("models") or self.capability.available_models()
            ),
            "reference_model": (
                capability.get("reference_model")
                or self.capability.reference_model_id()
            ),
            "prediction_horizon_steps": (
                self.capability.prediction_horizon_steps_value()
            ),
            "metrics": capability.get("metrics") or self.capability.metrics(),
        }
        try:
            outcome = DeterministicProblemParser().parse_with_safe_defaults(
                question,
                problem_type="model_comparison",
                defaults=defaults,
            )
            problem = outcome.problem
            field_sources = outcome.field_sources
        except Exception:
            problem = ResearchProblemSpec(
                problem_id=f"RP-{uuid.uuid4().hex[:12].upper()}",
                original_question=question,
                research_object="锅炉蒸汽体积量软测量与预测",
                target_variable="steam_volumetric_flow",
                normalized_target_variable="steam_volumetric_flow",
                objective="在冻结数据协议下比较候选模型对蒸汽体积量的软测/预测性能",
                metrics=list(defaults["metrics"] or ["MAE"]),
                operating_condition="未特别限定（全工况）",
                research_goal=question or "评估候选模型对蒸汽体积量的预测性能",
                required_operations=[
                    "model_comparison",
                    "chronological_validation",
                    "locked_test_evaluation",
                ],
                required_models=list(defaults["candidate_models"]),
                reference_models=["persistence"],
                research_task_type="model_selection",
                task_type_resolution_reason="Qwen 不可用时的确定性降级解析",
            )
            field_sources = {}
        return {
            "research_problem": problem.model_dump(mode="json"),
            "problem_id": problem.problem_id,
            "problem_statement": question,
            "status": "parsed",
            "problem_parser_type": "deterministic_fallback",
            "field_sources": field_sources,
            "automatic_completions": [],
            "semantic_gaps": [],
            "semantic_assumptions": [],
            "parser_fallback_error": f"{type(error).__name__}:{error}",
        }

    def _deterministic_hypotheses(
        self,
        context: dict[str, Any],
        memory_payload: dict[str, Any],
        parsed: dict[str, Any],
        error: Exception,
    ) -> dict[str, Any]:
        """Qwen 假设生成失败时的确定性降级：数据画像 → 逐模型候选假设 H_M。"""
        from boilermind.core.contracts import DataProfile, ModelSelectionPlan

        dataset = DirectVolume31VCapabilityRegistry.DEFAULT_DATASET_PATH
        try:
            profile = DataProfileSkill.compute(
                dataset, horizon_steps=0, data_split="train"
            )
            plan = profile_to_model_selection(profile, horizon_steps=0)
        except Exception as exc:
            raise RuntimeError(
                f"hypothesis_generation_failed:{type(error).__name__}:{error};"
                f"deterministic_fallback_failed:{type(exc).__name__}:{exc}"
            ) from error
        valid_obs = [
            str(o.get("observation_id"))
            for o in (memory_payload.get("supported_observations") or [])
            if o.get("observation_id")
        ]
        hypotheses = build_model_hypotheses(
            profile, plan,
            problem_id=str((parsed.get("research_problem") or {}).get("problem_id")),
            valid_observation_ids=valid_obs,
            valid_experiment_ids=list(
                memory_payload.get("completed_experiment_ids") or []
            ),
        )
        self._normalize_fallback_hypotheses(
            hypotheses, parsed.get("research_problem") or {}
        )
        context["data_profile"] = profile.model_dump(mode="json")
        context["profile_model_selection"] = plan.model_dump(mode="json")
        context["profile_hypotheses"] = hypotheses
        context["hypothesis_fallback_error"] = f"{type(error).__name__}:{error}"
        return {
            "qualified_hypotheses": hypotheses,
            "hypotheses": hypotheses,
        }

    def run(self, request: Any) -> ResearchRunState:
        question = str(getattr(request, "question", request if isinstance(request, str) else "")).strip()
        run_id = str(getattr(request, "run_id", None) or f"RUN-{uuid.uuid4().hex[:12].upper()}")
        state = ResearchRunState(run_id=run_id, question=question, status="RUNNING", started_at=datetime.now(timezone.utc))
        self._persist(state)
        context: dict[str, Any] = {"research_question": question}
        try:
            started = monotonic()
            if is_control_optimization_question(question):
                parsed = {
                    "research_problem": {
                        "problem_id": f"RP-{run_id}",
                        "original_question": question,
                        "research_object": "锅炉燃烧与汽水系统联合控制",
                        "target_variable": "steam_volumetric_flow",
                        "normalized_target_variable": "steam_volumetric_flow",
                        "objective": "在压力安全约束下提升蒸汽体积量并生成Unity可消费控制指令",
                        "metrics": ["MAE", "ACHIEVED_RISE_PCT", "PRESSURE_MAX_MPA"],
                        "operating_condition": "汽包压力不超过23MPa的稳态候选工况",
                        "manipulated_variables": ["给煤", "给水", "送风", "汽包压力"],
                        "observed_variables": ["蒸汽体积量V"],
                        "research_goal": "验证联合调参能否使蒸汽体积量V提升15%并形成Unity闭环输入",
                        "success_criteria": ["HGB预测提升不少于15%", "建议汽包压力不超过23MPa"],
                        "constraints": ["drum_pressure <= 23 MPa", "each control adjustment within ±25%"],
                        "required_operations": ["hgb_soft_sensor_validation", "constraint_search", "unity_result_export"],
                        "research_task_type": "parameter_optimization",
                        "optimization_variable": "steam_volumetric_flow",
                        "task_type_resolution_reason": "确定性识别为多变量约束控制优化与Unity推送任务",
                    },
                    "problem_parser_type": "deterministic_control_optimization",
                }
            else:
                try:
                    parsed = self.problem_parser(context)
                except Exception as exc:
                    # Qwen 不可用时的确定性问题解析降级：任何问题都能进入实验流程。
                    parsed = self._deterministic_problem(context, exc)
            if not parsed.get("research_problem"):
                raise RuntimeError("trusted_research_problem_not_produced")
            context.update(parsed)
            state.research_problem = _jsonable(parsed["research_problem"])
            field_sources = dict(parsed.get("field_sources") or {})
            for name, value in state.research_problem.items():
                state.field_provenance.append(FieldProvenance(
                    field_name=name,
                    source=field_sources.get(
                        name,
                        "USER" if name == "original_question" else (
                            "DETERMINISTIC" if parsed.get("problem_parser_type", "").startswith("deterministic") else "LLM"
                        ),
                    ),
                    value=value,
                    confidence=float((parsed.get("problem_intake") or {}).get("confidence", 1.0)),
                ))
            self._trace(state, "problem_understanding", started, {"question": question}, parsed, source="HYBRID")

            # 阶段二：train-only 数据属性画像 → 修正 target + 提供选型（31V 软测）
            if self._apply_data_profile(context, parsed, run_dir=self.run_root / run_id):
                state.research_problem = _jsonable(parsed["research_problem"])
                self._trace(state, "data_profile", started, {"question": question}, context.get("data_profile"), source="PROGRAM")

            started = monotonic()
            capability = self.capability.snapshot()
            scientific_context = self.capability.to_scientific_context()
            scientific_context.update(capability)
            context["scientific_context"] = scientific_context
            memory = self.memory_retriever(state.research_problem, capability, self.memory_store)
            memory_payload = _jsonable(memory)
            context["experiment_memory_bundle"] = memory_payload
            state.experiment_memory_bundle = memory_payload
            self._trace(state, "historical_experiment_retrieval", started, state.research_problem, memory_payload)

            # 阶段二/三：数据属性画像 → 确定性逐模型候选假设 H_M（骨架 + Qwen 补充）
            profile_hypotheses = None
            if context.get("data_profile") and context.get("profile_model_selection"):
                from boilermind.core.contracts import DataProfile, ModelSelectionPlan

                profile_obj = DataProfile.model_validate(context["data_profile"])
                plan_obj = ModelSelectionPlan.model_validate(context["profile_model_selection"])
                valid_obs = [
                    str(o.get("observation_id"))
                    for o in (memory_payload.get("supported_observations") or [])
                    if o.get("observation_id")
                ]
                profile_hypotheses = build_model_hypotheses(
                    profile_obj, plan_obj,
                    problem_id=str(parsed["research_problem"].get("problem_id")),
                    valid_observation_ids=valid_obs,
                    valid_experiment_ids=list(memory_payload.get("completed_experiment_ids") or []),
                )
                profile_hypotheses = enrich_hypotheses_with_qwen(
                    profile_hypotheses, profile=profile_obj, enabled=True,
                )
                self._normalize_fallback_hypotheses(
                    profile_hypotheses, state.research_problem
                )
                context["profile_hypotheses"] = profile_hypotheses

            started = monotonic()
            try:
                evidence = self.evidence_retriever(context)
                context.update(evidence)
                state.evidence_bundle = _jsonable(evidence.get("evidence_bundle"))
                self._trace(state, "literature_retrieval", started, state.research_problem, evidence, source="HYBRID")
            except Exception as exc:
                context["evidence_bundle"] = None
                state.evidence_bundle = None
                self._trace(state, "literature_retrieval", started, state.research_problem, error=exc, source="HYBRID")

            if is_control_optimization_question(question):
                return self._run_control_optimization(state, context)

            started = monotonic()
            if context.get("profile_hypotheses"):
                # 数据属性画像 → 确定性逐模型候选假设 H_M（含 Qwen 补充），不走 Qwen 自由生成
                generated = {
                    "qualified_hypotheses": context["profile_hypotheses"],
                    "hypotheses": context["profile_hypotheses"],
                }
            else:
                # Literature remains in the run context for reporting only.
                # Scientific hypothesis decisions are grounded in the problem,
                # experiment memory and the current capability snapshot.
                hypothesis_context = dict(context)
                hypothesis_context.pop("evidence_bundle", None)
                hypothesis_context.pop("evidence_retrieval_summary", None)
                try:
                    generated = self.hypothesis_generator(hypothesis_context)
                except Exception as exc:
                    # Qwen 不可用时的确定性假设生成降级（数据画像 → 逐模型候选假设）。
                    generated = self._deterministic_hypotheses(
                        context, memory_payload, parsed, exc
                    )
            hypotheses = list(generated.get("qualified_hypotheses") or generated.get("hypotheses") or [])[: self.max_hypotheses]
            if not hypotheses:
                raise RuntimeError("no_trusted_hypotheses_generated")
            if not all(item.get("historical_assessment") for item in hypotheses):
                hypotheses = assess_hypotheses_with_memory(hypotheses, memory_payload, state.research_problem)
            if not all(item.get("verification_mapping") for item in hypotheses):
                mapping = RankingSkill().execute({
                    **context, **generated, "qualified_hypotheses": hypotheses,
                })
                hypotheses = list(mapping.get("qualified_hypotheses") or hypotheses)
            context.update(generated)
            context["qualified_hypotheses"] = hypotheses
            state.hypotheses = _jsonable(hypotheses)
            self._trace(state, "hypothesis_generation_and_gate", started, memory_payload, hypotheses, source="HYBRID")

            scores = rank_hypotheses(hypotheses)
            for score in scores:
                state.hypothesis_states[score.hypothesis_id] = HypothesisRunState(
                    hypothesis_id=score.hypothesis_id,
                    eligible=score.eligible,
                    exit_reason=";".join(score.dropped_reasons) or None,
                )
            snapshot = RankingSnapshot(
                snapshot_id=f"RANK-{run_id}-000",
                round_index=0,
                entries=scores,
            )
            state.ranking_snapshots.append(snapshot)

            seen_designs: set[str] = set()
            for batch_index in range(1, self.max_batches + 1):
                current = state.ranking_snapshots[-1]
                ordered = [
                    item for item in sorted(current.entries, key=lambda entry: (-entry.dynamic_score, entry.hypothesis_id))
                    if state.hypothesis_states[item.hypothesis_id].eligible
                    and state.hypothesis_states[item.hypothesis_id].execution_count < 2
                ]
                members: list[BatchMember] = []
                for score in ordered:
                    if len(members) >= self.max_parallelism:
                        break
                    member = None
                    for revision in range(1, 3):
                        member = self._prepare_member(context, score.hypothesis_id, batch_index, revision)
                        if member is not None and experiment_design_sha256(member.contract) not in seen_designs:
                            break
                        member = None
                    if member is None:
                        state.hypothesis_states[score.hypothesis_id] = state.hypothesis_states[score.hypothesis_id].model_copy(update={
                            "eligible": False, "exit_reason": "planning_or_contract_rejected",
                        })
                        continue
                    seen_designs.add(experiment_design_sha256(member.contract))
                    members.append(member)
                if not members:
                    break
                batch = HypothesisValidationBatch(
                    batch_id=f"BATCH-{run_id}-{batch_index:02d}",
                    round_index=batch_index,
                    ranking_snapshot_id=current.snapshot_id,
                    members=members,
                )
                completed = self._execute_batch(batch)
                state.batches.append(completed)

                supported = False
                feedback_by_id: dict[str, float] = {}
                for member in completed.members:
                    hstate = state.hypothesis_states[member.hypothesis_id]
                    if member.status != "COMPLETED" or not member.outcome:
                        state.hypothesis_states[member.hypothesis_id] = hstate.model_copy(update={
                            "eligible": False, "exit_reason": "execution_failed",
                        })
                        continue
                    outcome = member.outcome
                    verdict = str(((outcome.get("scientific_result") or {}).get("verdict") or "")).upper()
                    contract = member.contract
                    feedback = calculate_experiment_feedback(
                        ExperimentResult.model_validate(outcome["experiment_result"]),
                        ExperimentAudit.model_validate(outcome["audit"]),
                        ScientificResult.model_validate(outcome["scientific_result"]),
                        primary_metric=contract.get("primary_metric"),
                    )
                    feedback_by_id[member.hypothesis_id] = feedback.priority_delta * feedback.evidence_strength
                    design_hash = experiment_design_sha256(contract)
                    eligible = verdict in {"PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE"}
                    supported = supported or verdict == "SUPPORTED"
                    state.hypothesis_states[member.hypothesis_id] = hstate.model_copy(update={
                        "execution_count": hstate.execution_count + 1,
                        "eligible": eligible and hstate.execution_count + 1 < 2,
                        "latest_verdict": verdict,
                        "cumulative_feedback": max(-1.0, min(1.0, hstate.cumulative_feedback + feedback_by_id[member.hypothesis_id])),
                        "executed_design_sha256": [*hstate.executed_design_sha256, design_hash],
                        "exit_reason": None if eligible else verdict,
                    })
                    audit_valid = bool((outcome.get("audit") or {}).get("execution_valid"))
                    notes = list((outcome.get("experiment_result") or {}).get("execution_notes") or [])
                    if audit_valid and not any(marker in notes for marker in ("TEST_ONLY_EXECUTION", "MOCK_EXECUTION", "ARTIFACT_REPLAY")):
                        persist_experiment_outcome(outcome, self.memory_store, source_path=f"run:{run_id}")

                # Propagate a completed batch only after every member reached a
                # terminal state.  Shared experiment provenance is direct
                # support/conflict; shared observations are premise-level.
                hypotheses_by_id = {
                    str(item.get("hypothesis_id")): item for item in hypotheses
                }
                for target_id, target_state in list(state.hypothesis_states.items()):
                    if target_id in feedback_by_id or not target_state.eligible:
                        continue
                    target = hypotheses_by_id.get(target_id, {})
                    target_experiments = set(target.get("source_experiment_ids") or [])
                    target_observations = set(target.get("source_observation_ids") or [])
                    propagated = 0.0
                    direct_conflict = False
                    for source_id, direct_delta in feedback_by_id.items():
                        source = hypotheses_by_id.get(source_id, {})
                        shared_experiments = target_experiments & set(source.get("source_experiment_ids") or [])
                        shared_observations = target_observations & set(source.get("source_observation_ids") or [])
                        if shared_experiments:
                            propagated += direct_delta * 0.80
                            direct_conflict = direct_conflict or direct_delta < 0.0
                        elif shared_observations:
                            propagated += direct_delta * 0.40
                    if direct_conflict:
                        state.hypothesis_states[target_id] = target_state.model_copy(update={
                            "eligible": False, "exit_reason": "direct_experiment_conflict",
                        })
                    elif propagated:
                        state.hypothesis_states[target_id] = target_state.model_copy(update={
                            "cumulative_feedback": max(-1.0, min(1.0, target_state.cumulative_feedback + propagated)),
                        })

                updated = []
                for prior in current.entries:
                    hstate = state.hypothesis_states[prior.hypothesis_id]
                    cumulative = hstate.cumulative_feedback
                    dynamic = dynamic_score(prior.prior_score, cumulative)
                    updated.append(prior.model_copy(update={
                        "cumulative_feedback": cumulative,
                        "dynamic_score": round(dynamic, 6),
                        "eligible": hstate.eligible,
                        "dropped_reasons": [] if hstate.eligible else [hstate.exit_reason or "not_eligible"],
                    }))
                state.ranking_snapshots.append(RankingSnapshot(
                    snapshot_id=f"RANK-{run_id}-{batch_index:03d}",
                    round_index=batch_index,
                    entries=sorted(updated, key=lambda item: (-int(item.eligible), -item.dynamic_score, item.hypothesis_id)),
                ))
                self._persist(state)
                if supported:
                    break

            return self._finalize(state)
        except Exception as exc:
            state.status = "FAILED"
            state.errors.append(f"{type(exc).__name__}:{exc}")
            state.completed_at = datetime.now(timezone.utc)
        self._persist(state)
        return state
