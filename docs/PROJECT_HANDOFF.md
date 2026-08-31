# BoilerMind Project Handoff / BoilerMind 项目交接

## Project scope / 项目定位

BoilerMind is a trusted AI-Scientist framework for industrial boiler soft-sensing research. Its intended workflow is:

`Research question → verified evidence → testable hypothesis → experiment plan → real experiment → audit → report → feedback`

BoilerMind 是面向工业锅炉软测量研究的可信 AI Scientist 框架，目标科研闭环为：

`科研问题 → 已验证证据 → 可检验假设 → 实验规划 → 真实实验 → 审计 → 报告 → 反馈`

## Current capabilities / 当前能力

- Natural-language research-problem structuring, with a deterministic fallback.
- Local literature retrieval, traceability verification, and evidence freezing.
- Conservative hypothesis generation, quality gates, ranking, planning, execution, and reporting.
- A FastAPI backend, static frontend, knowledge-graph views, and Unity WebGL control-bridge workflow.
- Research reports and scientific-plan exports in Markdown, JSON, DOCX, and PDF when the required local dependencies are installed.

- 支持自然语言科研问题结构化，并提供确定性降级解析。
- 支持本地文献检索、来源追溯核验与证据冻结。
- 支持保守型假设生成、质量门控、排序、规划、执行与报告。
- 包含 FastAPI 后端、静态前端、知识图谱视图和 Unity WebGL 控制桥接流程。
- 安装相应本地依赖后，可导出 Markdown、JSON、DOCX 与 PDF 科研计划/报告。

## Run locally / 本地运行

Requires Windows 10/11 and Python 3.11. / 需要 Windows 10/11 与 Python 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_frozen.txt
Copy-Item .env.example .env.local
python scripts\run_stack.py up
```

Default endpoints / 默认地址：

| Service / 服务 | Address / 地址 |
|---|---|
| Backend health / 后端健康检查 | `http://127.0.0.1:8765/health/ready` |
| Frontend / 前端 | `http://127.0.0.1:8081/#/chat` |
| Unity view / Unity 页面 | `http://127.0.0.1:8090/index_unity_only.html` |

Use `python scripts\run_stack.py doctor`, `status`, and `down` for environment checks, service status, and shutdown. / 使用 `doctor`、`status` 和 `down` 进行环境检查、状态查看与停止服务。

## Qwen and fallback / Qwen 与降级路径

Qwen is optional. Configure `DASHSCOPE_API_KEY`, `BOILERMIND_QWEN_MODEL`, and `BOILERMIND_QWEN_BASE_URL` only in local `.env.local`; never commit a real key. The integration uses an OpenAI-compatible client.

Qwen 为可选能力。仅在本地 `.env.local` 中配置 `DASHSCOPE_API_KEY`、`BOILERMIND_QWEN_MODEL` 与 `BOILERMIND_QWEN_BASE_URL`，不得提交真实密钥；系统通过 OpenAI 兼容客户端调用。

If Qwen or literature retrieval is unavailable, the application uses deterministic problem parsing and hypothesis templates. It must state the downgrade and must not fabricate literature evidence.

若 Qwen 或文献检索不可用，系统会改用确定性问题解析与假设模板；报告必须说明降级，且不得编造文献证据。

## Architecture pointers / 架构入口

| Area / 模块 | Entry point / 入口 |
|---|---|
| Backend API / 后端接口 | `server/research_api/app.py` |
| Research orchestration / 科研编排 | `src/boilermind/orchestration/research_orchestrator.py` |
| Evidence pipeline / 证据链 | `src/boilermind/evidence/` |
| Experiment execution / 实验执行 | `src/boilermind/experiment/` |
| Planning and reporting / 规划与报告 | `src/boilermind/planning/`, `src/boilermind/reporting/` |
| Frontend / 前端 | `frontend/` |
| Unity bridge / Unity 桥接 | `UnityWebgl/unity-bridge.js` |

## Verification and evidence / 验证与证据

The repository includes unit, contract, integration, and scientific-safety tests. Historical experiment conclusions, their audit status, and cross-regime/time limitations are consolidated in [Experiment Evidence Summary](EXPERIMENT_EVIDENCE_SUMMARY.md).

仓库包含单元、合同、集成与科研安全测试。历史实验结论、审计状态和跨工况/时间的适用边界汇总于[实验与复验证据摘要](EXPERIMENT_EVIDENCE_SUMMARY.md)。

## Boundaries and next work / 边界与后续工作

- The public repository excludes credentials, raw industrial data, literature PDFs, extracted corpus text, weights, runtime logs, generated outputs, and Unity prebuilt binaries.
- The Unity WebGL build is precompiled; modifying Unity C# or `.jslib` requires the original Unity project.
- The demonstration simulator is not a plant-grade thermodynamic model; deployment requires authorized data, domain validation, and engineering review.

- 公开仓库不包含凭证、原始工业数据、论文 PDF、抽取语料、模型权重、运行日志、生成产物和 Unity 预构建二进制。
- Unity WebGL 构建为预编译版本；若需改动 Unity C# 或 `.jslib`，必须取得原始 Unity 工程。
- 演示模拟器不是工厂级热力学模型；部署前必须取得授权数据、完成领域验证并通过工程评审。
