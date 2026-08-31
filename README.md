# BoilerMind Framework / BoilerMind 科研框架

BoilerMind is a trustworthy AI-Scientist framework for industrial boiler soft-sensing research. It turns an engineering question into a traceable workflow: problem structuring, evidence retrieval, hypothesis generation, experiment planning, real experiment execution, audit, and reporting.

BoilerMind 是面向工业锅炉软测量研究的可信 AI Scientist 框架。它将工程问题转化为可追溯的科研流程：问题结构化、证据检索、假设生成、实验规划、真实实验执行、审计与报告。

## Included / 包含内容

- FastAPI backend, static web frontend, Unity WebGL bridge, and research orchestration code.
- Deterministic fallback path for running without a large-language-model credential.
- Test suite, experiment contracts, reproducibility scripts, and technical documentation.

- FastAPI 后端、静态 Web 前端、Unity WebGL 桥接与科研编排代码。
- 未配置大模型凭证时仍可运行的确定性降级链路。
- 测试套件、实验合同、复现脚本与技术文档。

## Quick start / 快速启动

Windows 10/11 and Python 3.11 are required. / 需要 Windows 10/11 与 Python 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_frozen.txt
Copy-Item .env.example .env.local
python scripts\run_stack.py up
```

After startup / 启动后：

- Frontend / 前端：`http://127.0.0.1:8081/#/chat`
- Backend health check / 后端健康检查：`http://127.0.0.1:8765/health/ready`
- Unity view / Unity 页面：`http://127.0.0.1:8090/index_unity_only.html`

Run `python scripts\run_stack.py doctor` to inspect the environment and `python scripts\run_stack.py down` to stop the stack. / 使用前者检查环境，使用后者停止服务。

## Optional Qwen integration / 可选 Qwen 集成

Set the following values only in your local `.env.local` file. Never commit a real credential. / 以下内容仅写入本地 `.env.local`，不得提交真实凭证。

```text
DASHSCOPE_API_KEY=<your-key>
BOILERMIND_QWEN_MODEL=qwen3.7-plus
BOILERMIND_QWEN_BASE_URL=<OpenAI-compatible DashScope endpoint>
```

The application uses an OpenAI-compatible client. If the credential or service is unavailable, the supported deterministic fallback remains available for the research workflow. / 系统通过 OpenAI 兼容客户端调用 Qwen；凭证或服务不可用时，科研流程仍可使用确定性降级路径。

## Public-repository boundary / 公开仓库边界

This public repository intentionally excludes API credentials, raw boiler datasets, local literature PDFs and extracted corpus text, trained model weights, generated experiment outputs, local runtime logs, and Unity prebuilt binaries. Obtain authorized research data and full delivery artifacts through a controlled channel.

本公开仓库刻意不包含 API 凭证、原始锅炉数据、本地论文 PDF 与抽取语料、训练模型权重、生成实验产物、本地运行日志和 Unity 预构建二进制。经授权的数据与完整交付材料应通过受控渠道获取。

## Documentation / 文档

- [Project handoff / 项目交接与运行指南](docs/PROJECT_HANDOFF.md)
- [Frontend API contract / 前端接口契约](frontend/BACKEND_RESEARCH_PROGRESS_API_SPEC.md)
- [Unity integration guide / Unity 对接指南](UnityWebgl/后端通讯手册.md)
- [Experiment evidence summary / 实验与复验摘要](docs/EXPERIMENT_EVIDENCE_SUMMARY.md)
