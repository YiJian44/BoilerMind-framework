# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

BoilerMind-Trusted：面向锅炉工艺工程师的「可信 AI Scientist」系统。目标是把一个自然语言科研问题走完整条闭环：

科研问题 → 科学证据 → 科研假设 → 验证优先级排序 → 实验规划 → 真实实验 → 实验结论 → 假设反馈 → 动态重排

贯穿整个代码库的核心原则：**确定性可审计的可信核心 + LLM 仅做辅助**。LLM 负责「结构化」和「语义判断」，但任何科学结论（证据验证、假设排序、实验裁决、知识更新）必须由确定性程序决定。禁止为跑通闭环而 reintroduce：mock evidence、固定科研问题、固定假设、固定评分、虚假置信度更新、与假设无关的实验结果。

## 环境与运行

Python 3.11（`python_version.txt` 锁版本；本地仓库现锁 3.11.14，uv 默认未提供 3.11.15 时回退到此版本）。包名 `boilermind-trusted`，src 布局（`pyproject.toml` 的 `package-dir = {"": "src"}`，`pytest` 已配置 `pythonpath = ["src"]`、`testpaths = ["tests"]`、`addopts = "-q"`）。

```powershell
# 一次性创建 .venv 并安装依赖（需本机有 uv）
uv venv --python 3.11.14 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-server.txt -r requirements_frozen.txt
uv pip install --python .venv\Scripts\python.exe -e .

# 加载交接环境（设置 Qwen base_url/model、数据集路径等）
.\handoff_env.ps1
$env:DASHSCOPE_API_KEY = "YOUR_KEY"    # 单独设置，禁止写入任何项目文件

# 测试（使用项目内 .venv）
.\.venv\Scripts\python.exe -m pytest                                    # 全量
.\.venv\Scripts\python.exe -m pytest tests/unit/test_state_machine.py    # 单个测试文件
.\.venv\Scripts\python.exe -m pytest tests/unit/test_state_machine.py -k supported  # 按名称过滤

# 一键起后端 + 前端 + Unity（推荐，跨平台）：
python scripts/run_stack.py up                # 后台启动并等待 Ctrl+C
python scripts/run_stack.py up --detach       # 启动后立即返回
python scripts/run_stack.py status            # 查看服务状态
python scripts/run_stack.py down              # 停止全部

# 仅后端 / 仅前端
python scripts/run_stack.py backend
python scripts/run_stack.py frontend           # 默认含 Unity
python scripts/run_stack.py frontend --no-unity

# 环境体检
python scripts/run_stack.py doctor

# 旧的 PowerShell 脚本（start-all.ps1 / server/start-backend.ps1 /
# frontend/start-frontend.ps1）仍可用，但已标记 DEPRECATED，新代码
# 一律用上面的 run_stack.py。

# 确定性演示（不依赖 LLM/API，纯 test-only 适配器，可离线跑通）
.\.venv\Scripts\python.exe scripts\run_research_demo.py
```

关键环境变量：

- `DASHSCOPE_API_KEY` — 必填，Qwen 调用
- `BOILERMIND_QWEN_BASE_URL` — 默认 `https://trial.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`
- `BOILERMIND_QWEN_MODEL` — 默认 `qwen3.7-plus`
- `BOILERMIND_ENABLE_WEB_LITERATURE` — `0` 关闭（默认关，保证本地稳定）
- `BOILERMIND_REAL_DATASET_PATH` — 真实锅炉数据集路径
- `BOILERMIND_CROSSREF_MAILTO` / `BOILERMIND_WEB_TIMEOUT` — web 文献检索用

## 架构总览

理解本仓库的关键是：存在**两个并行的编排层**，分别对应「LLM 驱动的完整流水线」和「确定性可信核心」。

### 1. 确定性可信核心 — `ResearchEngine`

[src/boilermind/orchestration/research_engine.py](src/boilermind/orchestration/research_engine.py)：纯 Python 验证循环，无 LLM 依赖，可完全单测。

- `run_primary_loop`：从 Top-3 主候选池动态排序并顺序执行 → 每条执行后计算实验反馈 → 重排整个假设库 → 遇到 `SUPPORTED` 立即终止（`RESOLVED`）；主池耗尽则转 `EXTENDED_VALIDATION`。
- `run_extended_validation`：主池耗尽后执行剩余所有合格假设。
- `run_full_cycle`：串联以上两者。
- 关键规则：每条假设只执行一次；`SUPPORTED` 是唯一终裁决；`FALSIFIED`/`PARTIALLY_SUPPORTED`/`INSUFFICIENT_EVIDENCE` 非终止，只触发重排。

引擎通过注入的 `runner`（实验执行器）与 `knowledge_extractor` 工作。测试/演示用 [experiment/test_runner.py](src/boilermind/experiment/test_runner.py) 的 `TestOnlyExperimentRunner` 与 [knowledge/test_extractor.py](src/boilermind/knowledge/test_extractor.py) 的 `TestOnlyKnowledgeExtractor`；生产用 [experiment/runner.py](src/boilermind/experiment/runner.py) 的 `ExperimentRunner`。

### 2. Skill 编排层 — `ResearchSupervisorAgent`

[src/boilermind/agents/supervisor.py](src/boilermind/agents/supervisor.py)：LLM 驱动的端到端流水线，按固定顺序调用一组 `BaseSkill`：

problem → evidence → hypothesis → ranking → planning → contract → experiment → analysis → feedback → trace →（feedback 触发 `plan_refinement` 时进入二次实验循环）

每个 skill 在 [src/boilermind/skills/](src/boilermind/skills/) 下，继承 [skills/base.py](src/boilermind/skills/base.py) 的 `BaseSkill`（`name` + `execute(context: dict) -> dict`），经 `SkillRegistry` 注册、`SkillRuntime.execute(name, context)` 调用，用 dict context 逐 skill 传递并累积状态。skill 间的契约通过 context 的 key 隐式约定，没有显式 schema。

### 契约层 — `core/contracts/`

所有跨模块传递的数据结构是 Pydantic 模型（基类 `ContractModel` 在 [core/contracts/base.py](src/boilermind/core/contracts/base.py)）。[core/contracts/__init__.py](src/boilermind/core/contracts/__init__.py) 统一 re-export 全部领域对象，如 `ResearchProblemSpec`、`EvidenceBundle`/`EvidenceCandidate`/`VerifiedEvidence`、`ScientificHypothesis`、`RankingEntry`、`ExperimentContract`/`ExperimentPlan`/`ExperimentResult`/`ExperimentAudit`/`ScientificResult`、`ResearchRun`。新增类型先在 contracts 层定义。

### 状态机与枚举 — `core/`

- [core/state_machine.py](src/boilermind/core/state_machine.py)：`_ALLOWED_TRANSITIONS` 硬编码 `ResearchRunStatus` 的合法迁移，`transition()` 对非法迁移抛 `InvalidStateTransition`。
- [core/enums.py](src/boilermind/core/enums.py)：全部状态/裁决/等级用 `StrEnum`（`ResearchRunStatus`、`ScientificVerdict`、`HypothesisStatus`、`EvidenceStage`、`ClaimSupport` 等），不要写裸字符串字面量。

### 各领域模块

- `evidence/` — RAG 证据管线：[sources/local_rag.py](src/boilermind/evidence/sources/local_rag.py)（本地文献）、[sources/web_literature.py](src/boilermind/evidence/sources/web_literature.py)、`retrieval_pipeline.py`（检索）、`traceability_verifier.py`（确定性来源追溯）、`qwen_semantic_judge.py`（LLM 语义裁决）、`verification_pipeline.py`（两者合成，**fail-closed：任一层失败即拒绝**）、`bundle_freezer.py`（冻结 EvidenceBundle）。
- `hypothesis/` — `quality_gate.py`（确定性 gate）、`mechanism_critic.py`、`admission.py`。
- `ranking/` — `prior_scorer.py`、`dynamic_ranker.py`（`build_dynamic_ranking`）、`feedback_calculator.py`（`calculate_experiment_feedback` / `calculate_metric_effect`）。
- `audit/` — `experiment_auditor.py`（`audit_experiment`）、`criterion_assessment.py`、`verdict_engine.py`（`derive_scientific_result`）、`execution_trace.py`。
- `experiment/` — `real_sklearn_backend.py`（真实数据后端）、`runner.py`、`test_runner.py`。
- `knowledge/` — `extractor.py`、`updater.py`（`build_knowledge_update`）、`contracts.py`。
- `planning/` — `plan_contracts.py`、`plan_gate.py`。
- `models/model_runner.py` — `BoilerModelRunner` 从硬编码 `D:\BoilerMindTeamTest\...` 加载 joblib（外部依赖，非本仓库资源）。

### 实验后端协议 — `experiment/real_sklearn_backend.py`

真实锅炉质量流量预测：30 特征、时间窗口化（`window_steps` 默认 20）、未来预测步长（`prediction_horizon_steps` 默认 40）、按时间顺序 train/validation/locked-test 切分。**超参只在 validation 上选，locked test 仅评估**。支持模型：ridge / bayesianridge / hgb / rf / svr / elasticnet / mlp / pls / knn / gpr（gpr 对全量数据 fail-closed 禁用）。指标：`mae_t_h` / `rmse_t_h` / `r2` / `mbe_t_h`。

## 约定与注意事项

- 数据集 `resources/data/shortperiod_new.csv` **无表头**，31 列（前 30 特征 + 末列目标变量）。
- LLM 路径有两套并存：[orchestration/qwen_problem_parser.py](src/boilermind/orchestration/qwen_problem_parser.py) 的 `QwenProblemParser`（OpenAI 兼容客户端 + `httpx2`，注意不是 `httpx`；`trust_env=False` 强制直连以绕过本机系统代理 127.0.0.1:7892 的间歇性抖动，默认超时 90s、`max_retries=0`），以及 [core/llm_client.py](src/boilermind/core/llm_client.py) 的 `LLMClient`（直接走 `dashscope`）。
- 真实实验产物落 `outputs/experiments/<experiment_id>/`（joblib 模型 + `*_locked_test_predictions.csv` + `experiment_result.json`）；审计/调试产物在 `outputs/debug/` 与 `outputs/handoff/`。
- 仓库中存在多个历史快照文件（如 `orchestration/research_engine.py.before_primary_loop_fix_*.bak`、`skills/hypothesis_skill_before_*.py`），**不要被它们误导**，以当前 `.py` 为准。
- [HANDOFF_README.md](HANDOFF_README.md) 是阶段性交接文档，描述的是上一阶段结构（其中提到的 `skills/` 等已重构为当前模块布局）。它是理解项目意图与「当前建议冻结模块」的重要背景，但**模块清单以当前 `src/` 为准**。文档第 6 节列出的上游可信模块（Problem Parser、Scientific-RAG、证据验证、假设生成/批判、sklearn 后端、locked test 等）当前建议冻结，后续重点放在中间编排层。
