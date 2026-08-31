# BoilerMind 六阶段研究进度接口规范

> 版本：v1.0  
> 日期：2026-08-15  
> 用途：交付后端开发人员，补齐研究任务实时进度、实验过程、科学评价与报告生成状态接口。

## 1. 背景与已确认现状

BoilerMind 当前存在两条独立链路：

1. `POST /api/v1/assistant`：工程问答接口，非流式执行，通常需要约 90～100 秒，最后一次性返回结论、证据、覆盖率与研究问题摘要。
2. `POST /api/v1/research-runs` 与 `GET /api/v1/research-runs/{run_id}`：完整研究任务创建和状态查询接口。

现有研究状态主要通过任务目录中是否出现特定 JSON 产物推导：

| 产物 | 当前推导状态 |
|---|---|
| 尚无产物 | `queued` |
| `rag_result.json` | `rag_completed` |
| `hypothesis_candidates.json` | `hypotheses_completed` |
| `elo_progress.json` | `elo_running` |
| `elo_result.json` 或 `elo_tournament_result.json` | `elo_completed` |
| `research_plan.json` | `research_plan_completed` |
| `workflow_result.json` | `completed`、`needs_human_review` 或 `failed` |

这说明后端并非完全没有阶段接口，但存在以下缺口：

- `/assistant` 不返回中间过程事件。
- 后端没有直接返回统一的六阶段产品状态，前端仍需根据原始状态推断。
- 实验执行、科学评价和报告生成阶段缺少细粒度状态。
- 缺少当前任务、已完成节点、阶段耗时和预计剩余时间。
- 缺少训练、锁定测试、工况切片与指标计算的实时进度。
- 缺少实时数据质量检查与数据泄漏检查状态。
- 很多指标和产物只有 `workflow_result.json` 完成后才出现。
- 缺少 SSE/WebSocket 事件流与断线恢复机制。
- 存在空问题 queued 任务，例如 `_steam_experiments`，会污染进行中任务统计。

目标不是给六个阶段分别创建六套接口，而是建立统一的任务快照和事件协议。

## 2. 六阶段产品模型

后端和前端统一使用以下六个阶段：

| 顺序 | `stage_id` | 中文名称 | 主要内容 |
|---:|---|---|---|
| 1 | `problem` | 问题拆解 | 研究目标、目标变量、输入变量、时间范围、工况范围 |
| 2 | `evidence` | 证据与假设 | 文献检索、证据评级、假设生成、Elo 比赛 |
| 3 | `plan` | 实验方案 | 候选方案、基线、模型、数据切分、指标与研究门禁 |
| 4 | `execution` | 实验执行 | 数据处理、训练、验证、锁定测试、工况切片 |
| 5 | `evaluation` | 科学评价 | 假设结论、关键指标、可靠性边界、部署门禁 |
| 6 | `report` | 科研报告 | 报告章节、图表、研究产物与阅读入口 |

每个阶段统一使用：

```text
waiting
running
completed
blocked
failed
```

整个研究任务统一使用：

```text
queued
running
needs_human_review
completed
failed
cancelled
```

### 2.1 核心约束

- 状态接口始终返回完整六阶段数组，未开始阶段返回 `waiting`，不能省略。
- 后端直接返回 `current_stage`，前端不再根据文件名推断产品阶段。
- 指标尚未产生时返回 `null`，不能用虚假 `0` 占位。
- 阶段与任务状态只能向前推进，已完成状态不能退回运行中。
- 所有时间使用带时区的 ISO 8601 字符串。
- 所有进度变化必须可在页面刷新后恢复。

## 3. 接口清单

### 3.1 保留现有接口

```http
POST /api/v1/assistant
POST /api/v1/uploads
POST /api/v1/research-runs
GET  /api/v1/research-runs
GET  /api/v1/research-runs/{run_id}
GET  /api/v1/research-runs/{run_id}/report
```

`GET /api/v1/research-runs` 支持 `query`（问题文本包含匹配）、`status`（completed/completed_with_warning/failed/running/queued/needs_human_review）、`page` 与 `pageSize` 筛选分页，返回 `{items, total, page, page_size}`。`POST /api/v1/uploads` 以 multipart 接收 `files`，落盘 `runtime/uploads/` 并返回附件 id 列表。

### 3.2 建议新增接口

```http
GET /api/v1/research-runs/{run_id}/events
GET /api/v1/research-runs/{run_id}/timeline
GET /api/v1/research-runs/{run_id}/artifacts/{artifact_id}
GET /api/v1/research-runs/{run_id}/artifacts/{artifact_id}/download
```

第一版允许不实现 SSE，但必须先完成统一进度快照。

## 4. 创建研究任务

### 4.1 请求

```http
POST /api/v1/research-runs
Content-Type: application/json
```

```json
{
  "question": "低负荷和降负荷工况下的预测可靠性边界在哪里？",
  "session_id": "bm_session_001",
  "client_request_id": "request_20260815_001",
  "options": {
    "data_source": "real_boiler_historical_data",
    "auto_start": true
  }
}
```

### 4.2 响应

接口必须快速返回，不等待研究完成：

```http
HTTP/1.1 202 Accepted
```

```json
{
  "success": true,
  "data": {
    "run_id": "ui_20260815_001",
    "status": "queued",
    "current_stage": "problem",
    "created_at": "2026-08-15T10:00:00+08:00",
    "status_url": "/api/v1/research-runs/ui_20260815_001",
    "events_url": "/api/v1/research-runs/ui_20260815_001/events"
  },
  "error": null
}
```

### 4.3 要求

- 正常情况下在 1～2 秒内返回。
- 必须返回真实 `run_id`。
- `question` 不能为空或仅包含空白字符。
- 使用 `client_request_id` 实现幂等；重复提交时返回原任务。
- 禁止产生无研究问题的幽灵任务。
- 创建成功后立即写入第一条 `run_queued` 事件和初始快照。

## 5. 获取完整进度快照

```http
GET /api/v1/research-runs/{run_id}
```

建议响应结构：

```json
{
  "success": true,
  "data": {
    "run_id": "ui_20260815_001",
    "question": "低负荷和降负荷工况下的预测可靠性边界在哪里？",
    "status": "running",
    "progress_percent": 58,
    "current_stage": "execution",
    "current_stage_index": 4,
    "current_task": {
      "task_id": "locked_test_dlinear",
      "name": "运行 DLinear 锁定测试",
      "detail": "正在评估低负荷和降负荷工况",
      "status": "running",
      "started_at": "2026-08-15T10:08:24+08:00",
      "elapsed_seconds": 83,
      "progress_percent": 42
    },
    "timing": {
      "started_at": "2026-08-15T10:00:00+08:00",
      "updated_at": "2026-08-15T10:09:47+08:00",
      "elapsed_seconds": 587,
      "estimated_remaining_seconds": 260
    },
    "stages": [
      {
        "stage_id": "problem",
        "name": "问题拆解",
        "status": "completed",
        "progress_percent": 100,
        "started_at": "2026-08-15T10:00:00+08:00",
        "completed_at": "2026-08-15T10:00:08+08:00",
        "elapsed_seconds": 8,
        "summary": "已确定目标变量、预测时间和工况范围",
        "completed_task_count": 3,
        "total_task_count": 3
      },
      {
        "stage_id": "evidence",
        "name": "证据与假设",
        "status": "completed",
        "progress_percent": 100,
        "started_at": "2026-08-15T10:00:08+08:00",
        "completed_at": "2026-08-15T10:03:42+08:00",
        "elapsed_seconds": 214,
        "summary": "检索到60条证据，完成10场Elo比赛",
        "completed_task_count": 4,
        "total_task_count": 4
      },
      {
        "stage_id": "plan",
        "name": "实验方案",
        "status": "completed",
        "progress_percent": 100,
        "started_at": "2026-08-15T10:03:42+08:00",
        "completed_at": "2026-08-15T10:06:15+08:00",
        "elapsed_seconds": 153,
        "summary": "选择DLinear并以Persistence为基线",
        "completed_task_count": 4,
        "total_task_count": 4
      },
      {
        "stage_id": "execution",
        "name": "实验执行",
        "status": "running",
        "progress_percent": 52,
        "started_at": "2026-08-15T10:06:15+08:00",
        "completed_at": null,
        "elapsed_seconds": 212,
        "summary": "正在运行锁定测试",
        "completed_task_count": 3,
        "total_task_count": 6
      },
      {
        "stage_id": "evaluation",
        "name": "科学评价",
        "status": "waiting",
        "progress_percent": 0,
        "started_at": null,
        "completed_at": null,
        "elapsed_seconds": 0,
        "summary": "等待实验结果",
        "completed_task_count": 0,
        "total_task_count": 3
      },
      {
        "stage_id": "report",
        "name": "科研报告",
        "status": "waiting",
        "progress_percent": 0,
        "started_at": null,
        "completed_at": null,
        "elapsed_seconds": 0,
        "summary": "等待科学评价",
        "completed_task_count": 0,
        "total_task_count": 3
      }
    ],
    "completed_tasks": [
      {
        "task_id": "load_dataset",
        "stage_id": "execution",
        "name": "读取真实锅炉历史数据",
        "status": "completed",
        "started_at": "2026-08-15T10:06:15+08:00",
        "completed_at": "2026-08-15T10:06:32+08:00",
        "elapsed_seconds": 17,
        "result_summary": "读取25146条样本"
      }
    ],
    "data_status": {
      "source_type": "real_boiler_historical_data",
      "source_name": "真实锅炉历史数据",
      "dataset_id": "boiler_181var_realdata",
      "is_real_data": true,
      "sample_count": 25146,
      "train_sample_count": 17599,
      "validation_sample_count": 3775,
      "locked_test_sample_count": 3772,
      "time_range": {
        "start": "2025-01-01T00:00:00+08:00",
        "end": "2025-06-30T23:59:59+08:00"
      }
    },
    "quality_checks": {
      "time_order_check": "passed",
      "data_leakage_check": "passed",
      "train_test_overlap_check": "passed",
      "missing_value_check": "passed",
      "unit_consistency_check": "passed",
      "guard_approved": true
    },
    "live_metrics": {
      "model": "DLinear",
      "baseline": "Persistence",
      "overall": {
        "model_mae": null,
        "baseline_mae": 12.42,
        "improvement_percent": null
      },
      "low_load": null,
      "ramp_down": null
    },
    "runtime": {
      "host": "local",
      "python": "3.11.9",
      "numpy": "2.4.6",
      "pandas": "3.0.5",
      "scikit_learn": "1.9.0",
      "torch": "2.13.0+cpu",
      "device": "cpu",
      "random_seed": 42
    },
    "artifacts": [
      {
        "artifact_id": "artifact_001",
        "type": "dataset_profile",
        "name": "数据集概览",
        "status": "ready",
        "created_at": "2026-08-15T10:06:32+08:00",
        "url": "/api/v1/research-runs/ui_20260815_001/artifacts/artifact_001"
      }
    ],
    "last_event_id": 128,
    "revision": 37
  },
  "error": null
}
```

### 5.1 快照接口要求

- 始终返回完整六阶段数组。
- `progress_percent` 单调不下降。
- `revision` 每次状态变化递增。
- `updated_at` 只在真实状态变化时更新。
- 阶段和任务耗时由后端计算，不能依赖前端猜测。
- 任务完成后仍可读取全过程。
- 任务失败时保留已经完成的节点、指标和产物。
- 快照必须在子任务开始、更新、完成、失败时及时刷新。

## 6. 实时事件流接口

推荐新增 SSE：

```http
GET /api/v1/research-runs/{run_id}/events
Accept: text/event-stream
```

响应头：

```http
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

事件示例：

```text
id: 101
event: stage_started
data: {"run_id":"ui_20260815_001","stage_id":"execution","timestamp":"2026-08-15T10:06:15+08:00"}

id: 102
event: task_started
data: {"run_id":"ui_20260815_001","stage_id":"execution","task_id":"load_dataset","name":"读取真实锅炉历史数据","timestamp":"2026-08-15T10:06:15+08:00"}

id: 103
event: data_profile_updated
data: {"run_id":"ui_20260815_001","sample_count":25146,"locked_test_sample_count":3772,"source_type":"real_boiler_historical_data"}

id: 104
event: quality_check_updated
data: {"run_id":"ui_20260815_001","check_id":"data_leakage_check","status":"passed"}

id: 105
event: task_completed
data: {"run_id":"ui_20260815_001","stage_id":"execution","task_id":"load_dataset","elapsed_seconds":17}

id: 106
event: metric_updated
data: {"run_id":"ui_20260815_001","scope":"overall","model":"Persistence","metric":"mae","value":12.42}

id: 107
event: artifact_created
data: {"run_id":"ui_20260815_001","artifact_id":"artifact_001","type":"dataset_profile","name":"数据集概览"}

id: 108
event: heartbeat
data: {"run_id":"ui_20260815_001","timestamp":"2026-08-15T10:06:30+08:00"}
```

### 6.1 事件类型

```text
run_queued
run_started
stage_started
stage_progress
stage_completed
task_started
task_progress
task_completed
task_failed
evidence_found
hypothesis_created
elo_match_started
elo_match_completed
experiment_plan_selected
data_profile_updated
quality_check_updated
metric_updated
artifact_created
human_review_required
report_section_completed
run_completed
run_failed
heartbeat
```

### 6.2 断线恢复

浏览器重连时发送：

```http
Last-Event-ID: 108
```

后端从下一条事件继续推送。若暂时不实现标准 `Last-Event-ID`，至少支持：

```http
GET /api/v1/research-runs/{run_id}/events?after=108
```

建议每 15～30 秒发送一次 `heartbeat`。

## 7. 获取历史时间线

```http
GET /api/v1/research-runs/{run_id}/timeline?after_event_id=0&limit=200
```

```json
{
  "success": true,
  "data": {
    "run_id": "ui_20260815_001",
    "events": [
      {
        "event_id": 101,
        "event_type": "stage_started",
        "stage_id": "execution",
        "task_id": null,
        "timestamp": "2026-08-15T10:06:15+08:00",
        "message": "开始实验执行",
        "payload": {}
      }
    ],
    "next_event_id": 102,
    "has_more": false
  },
  "error": null
}
```

时间线用于：

- 页面刷新后恢复过程。
- SSE 断线后的事件补偿。
- 审计任务为什么卡住或失败。
- 科研报告保留可追溯过程。

## 8. 获取研究产物

```http
GET /api/v1/research-runs/{run_id}/artifacts/{artifact_id}
```

```json
{
  "success": true,
  "data": {
    "artifact_id": "artifact_001",
    "run_id": "ui_20260815_001",
    "type": "elo_result",
    "name": "Elo 假设比赛结果",
    "mime_type": "application/json",
    "size_bytes": 18422,
    "created_at": "2026-08-15T10:03:12+08:00",
    "content": {},
    "download_url": "/api/v1/research-runs/ui_20260815_001/artifacts/artifact_001/download"
  },
  "error": null
}
```

建议支持以下产物类型：

```text
problem_decomposition
evidence_collection
hypothesis_candidates
elo_progress
elo_result
experiment_plan
guard_report
dataset_profile
data_quality_report
leakage_report
baseline_metrics
model_metrics
regime_metrics
scientific_evaluation
deployment_gate_report
final_report
chart_data
runtime_manifest
```

## 9. 统一进度记录器

不要让每个模块自行拼接 JSON。建议增加统一记录器：

```python
class ResearchProgressRecorder:
    def __init__(self, run_id: str):
        self.run_id = run_id

    def start_stage(self, stage_id: str, summary: str = ""):
        ...

    def update_stage(self, stage_id: str, progress_percent: float, summary: str = ""):
        ...

    def complete_stage(self, stage_id: str, summary: str = ""):
        ...

    def start_task(self, stage_id: str, task_id: str, name: str, detail: str = ""):
        ...

    def update_task(self, task_id: str, progress_percent: float, detail: str = ""):
        ...

    def complete_task(self, task_id: str, result: dict | None = None):
        ...

    def fail_task(self, task_id: str, error_code: str, message: str, retryable: bool):
        ...

    def update_metric(self, scope: str, model: str, metric: str, value: float):
        ...

    def update_quality_check(self, check_id: str, status: str, detail: str = ""):
        ...

    def add_artifact(self, artifact_type: str, name: str, path: str):
        ...
```

调用示例：

```python
progress.start_stage("execution", "开始真实数据实验")

progress.start_task(
    "execution",
    "load_dataset",
    "读取真实锅炉历史数据",
)

dataset = load_dataset()

progress.update_data_status(
    source_type="real_boiler_historical_data",
    dataset_id="boiler_181var_realdata",
    sample_count=len(dataset),
)

progress.complete_task(
    "load_dataset",
    result={"sample_count": len(dataset)},
)

progress.start_task(
    "execution",
    "data_leakage_check",
    "检查训练测试数据泄漏",
)

leakage_result = check_data_leakage(dataset)

progress.update_quality_check(
    "data_leakage_check",
    status="passed" if leakage_result.passed else "failed",
    detail=leakage_result.message,
)

progress.complete_task("data_leakage_check")
```

## 10. 推荐任务节点

### 10.1 问题拆解

```text
normalize_question
identify_research_goal
identify_target_variable
identify_input_variables
define_time_range
define_operating_scenarios
build_problem_contract
```

### 10.2 证据与假设

```text
search_local_literature
search_web_literature
deduplicate_evidence
score_evidence_quality
generate_hypotheses
validate_hypothesis_structure
run_elo_matches
select_hypothesis
```

### 10.3 实验方案

```text
generate_candidate_plans
select_dataset
select_baseline
select_models
define_time_split
define_locked_test
define_metrics
run_research_guard
approve_experiment_plan
```

### 10.4 实验执行

```text
load_dataset
validate_schema
align_timestamps
handle_missing_values
detect_outliers
split_train_validation_test
check_data_leakage
train_persistence_baseline
train_candidate_model
run_locked_test
run_regime_slices
calculate_prediction_intervals
save_raw_metrics
```

### 10.5 科学评价

```text
compare_with_baseline
evaluate_hypothesis
evaluate_low_load
evaluate_ramp_up
evaluate_ramp_down
evaluate_uncertainty
evaluate_physical_consistency
make_deployment_decision
```

### 10.6 科研报告

```text
assemble_problem_section
assemble_evidence_section
assemble_experiment_section
generate_charts
assemble_evaluation_section
assemble_limitations
generate_final_report
publish_report_artifact
```

## 11. 错误返回规范

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "DATA_LEAKAGE_DETECTED",
    "message": "锁定测试集与训练集存在时间区间重叠",
    "stage_id": "execution",
    "task_id": "check_data_leakage",
    "retryable": false,
    "details": {
      "overlap_sample_count": 128
    }
  }
}
```

建议错误代码：

```text
INVALID_RESEARCH_QUESTION
RUN_NOT_FOUND
DATASET_NOT_FOUND
DATA_SCHEMA_INVALID
DATA_LEAKAGE_DETECTED
INSUFFICIENT_SAMPLES
MODEL_TRAINING_FAILED
ELO_MATCH_FAILED
RESEARCH_GUARD_REJECTED
HUMAN_REVIEW_REQUIRED
REPORT_GENERATION_FAILED
RUN_TIMEOUT
RUN_CANCELLED
```

失败时必须：

- 保留已经完成的进度事件和产物。
- 返回失败的 `stage_id` 与 `task_id`。
- 明确 `retryable`。
- 保留失败前已经得到的指标。
- 禁止把所有错误统一写成“实验失败”。

## 12. 持久化建议

建议每个研究任务目录增加：

```text
progress_snapshot.json
progress_events.jsonl
```

- `progress_snapshot.json`：保存最新完整快照。
- `progress_events.jsonl`：每行一个不可变事件。
- 快照写入采用“临时文件 + 原子替换”，避免前端读取半截 JSON。
- 事件追加后立即刷新，不等待整个阶段完成。
- 任务完成后，快照和事件文件纳入研究产物清单。
- 事件 ID 在单个任务内严格递增。

## 13. 第一版最小实现

如果开发时间有限，第一版可以不做 SSE，只完成以下内容：

1. `POST /research-runs` 在 2 秒内返回 `run_id`。
2. 扩展 `GET /research-runs/{run_id}`。
3. 固定返回完整六阶段数组。
4. 返回 `current_stage`。
5. 返回 `current_task`。
6. 返回 `completed_tasks`。
7. 返回任务和阶段 `elapsed_seconds`。
8. 返回真实数据来源与样本数。
9. 返回数据泄漏及质量检查状态。
10. 返回当前已经产生的指标和产物。
11. 执行器每完成一个真实节点就更新 `progress_snapshot.json`。
12. 前端每 2 秒轮询状态接口。
13. 增加 `client_request_id` 幂等保护。
14. 拒绝空问题任务。

完成最小版本后再增加：

- `progress_events.jsonl`
- `/timeline`
- SSE `/events`
- `Last-Event-ID` 断线恢复

## 14. 后端验收标准

- [ ] 创建任务后 2 秒内取得 `run_id`。
- [ ] 新任务立即返回完整六阶段数组。
- [ ] `current_stage` 随真实流程推进。
- [ ] `current_task` 随真实执行节点变化。
- [ ] 每个完成节点进入 `completed_tasks`。
- [ ] 阶段状态只能向前推进。
- [ ] `elapsed_seconds` 持续增加。
- [ ] `progress_percent` 单调不下降。
- [ ] 真实数据任务返回正确来源、数据集 ID 和样本数。
- [ ] 数据泄漏检查完成前不能提前返回 `passed`。
- [ ] 指标未产生时返回 `null`，不补造 `0`。
- [ ] 页面刷新后可以恢复已有进度。
- [ ] 任务失败后仍能读取失败前的完整过程。
- [ ] 同一 `client_request_id` 不会创建重复实验。
- [ ] 空问题返回 400，不创建任务目录。
- [ ] 任务完成后可以读取完整报告和所有产物。
- [ ] 至少有一个测试任务能观察到 10 次以上真实状态变化。
- [ ] OpenAPI 文档与实际响应一致。

## 15. 前端接入方式

### 15.1 第一版轮询

```text
POST /research-runs
  ↓ 获得 run_id
每2秒 GET /research-runs/{run_id}
  ↓
revision 变化时更新六阶段、当前任务、指标和产物
  ↓
status 为 completed / failed / cancelled 时停止轮询
```

### 15.2 SSE 版本

```text
POST /research-runs
  ↓ 获得 run_id
GET /research-runs/{run_id} 获取初始快照
  ↓
连接 EventSource(/research-runs/{run_id}/events)
  ↓
按事件增量更新
  ↓
断线时使用 Last-Event-ID 重连
  ↓
必要时调用 /timeline 补偿事件
```

前端只展示后端真实返回的数据，不根据动画或计时虚构实验结果。

## 16. 可直接发送给后端队友的提示词

```text
请为 BoilerMind 研究执行器增加“六阶段实时进度能力”。

现有接口：
POST /api/v1/assistant
POST /api/v1/research-runs
GET /api/v1/research-runs/{run_id}
GET /api/v1/research-runs/{run_id}/report

当前问题：
1. assistant 是非流式接口，只返回最终回答。
2. research run 状态主要根据最终 JSON 产物判断。
3. 实验执行、科学评价和报告生成期间缺少 current_task、
   completed_tasks、stage_elapsed_seconds、实时指标和质量检查状态。
4. 前端无法真实展示研究正在执行什么。

请保持现有接口兼容，并完成：

1. POST /research-runs 快速返回 202、run_id、status_url、events_url。
2. 扩展 GET /research-runs/{run_id}，固定返回完整六阶段：
   problem、evidence、plan、execution、evaluation、report。
3. 每个阶段返回 waiting/running/completed/blocked/failed、
   progress_percent、started_at、completed_at、elapsed_seconds、
   summary、completed_task_count、total_task_count。
4. 顶层返回 current_stage、current_task、completed_tasks、
   progress_percent、timing、data_status、quality_checks、
   live_metrics、runtime、artifacts、revision、last_event_id。
5. 实现统一 ResearchProgressRecorder。研究执行器中的每个真实任务节点
   开始、更新、完成、失败时都必须调用记录器。
6. 在任务目录持久化 progress_snapshot.json 和
   progress_events.jsonl，使用原子写入。
7. 新增 GET /research-runs/{run_id}/timeline，支持刷新页面后恢复事件。
8. 推荐新增 SSE：GET /research-runs/{run_id}/events，
   支持 Last-Event-ID、heartbeat 和断线重连。
9. 禁止前端根据文件名推断阶段；后端直接返回产品阶段。
10. 字段没有产生时返回 null，禁止补造指标。
11. 失败时保留已完成节点、指标、产物和明确的失败 task_id。
12. 增加 client_request_id 幂等保护，禁止生成空问题幽灵任务。

真实任务节点至少覆盖：
问题拆解、文献检索、假设生成、Elo 比赛、方案选择、研究门禁、
数据读取、时间对齐、数据清洗、时序划分、泄漏检查、
Persistence 基线、候选模型训练、锁定测试、工况切片、
指标计算、假设评价、可靠性边界、报告和图表生成。

请同时补充：
- OpenAPI 定义；
- 单元测试；
- 端到端示例响应；
- 一个运行过程中可观察到至少 10 次状态变化的测试任务；
- README 中的前端接入说明。

验收重点：
前端每 2 秒轮询时必须能看到真实 current_task 和阶段变化；
不能直到 workflow_result.json 完成后才一次性显示所有结果。
```

## 17. 推荐实施顺序

1. 定义统一阶段、任务与事件数据结构。
2. 实现 `ResearchProgressRecorder`。
3. 在研究主流程的关键节点埋点。
4. 写入 `progress_snapshot.json`。
5. 扩展 `GET /research-runs/{run_id}`。
6. 完成轮询版前后端联调。
7. 增加 `progress_events.jsonl` 与 `/timeline`。
8. 增加 SSE `/events` 和断线恢复。
9. 补齐 OpenAPI、测试和文档。

优先完成轮询快照版，可以最快解决“前端只能看到最终结论、看不到实验过程”的核心问题。
