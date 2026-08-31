from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from boilermind.core.contracts import ResearchRequest
from boilermind.orchestration import ResearchOrchestrator
from boilermind.orchestration.control_optimization import (
    is_control_optimization_question,
)
from .boiler_sim import run_boiler_simulation
from .knowledge_graph_api import (
    build_team_graph,
    build_literature_graph,
    ensure_evolution_graph_synced,
    load_evolution_graph,
)
from .projector import artifact_display_name, project_run

app = FastAPI(title="BoilerMind Research API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):\d+",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
pipeline = ResearchOrchestrator()
_states: dict[str, dict] = {}
_lock = threading.Lock()
# 前端上传附件的落盘根目录
UPLOAD_ROOT = Path(__file__).resolve().parents[2] / "runtime" / "uploads"
# run_id -> 已连接的 Unity WebSocket 客户端
_unity_clients: dict[str, set[WebSocket]] = defaultdict(set)


def _execute(request: ResearchRequest) -> None:
    try:
        state = pipeline.run(request).model_dump(mode="json")
    except Exception as exc:
        state = {"run_id": request.run_id, "status": "FAILED", "errors": [f"{type(exc).__name__}:{exc}"]}
    with _lock:
        _states[str(request.run_id)] = state


def _state(run_id: str) -> dict:
    path = pipeline.run_root / run_id / "run.json"
    temporary = path.with_name("run.json.tmp")
    with _lock:
        in_memory = _states.get(run_id)
    if in_memory and in_memory.get("status") in {"FAILED", "COMPLETED", "COMPLETED_WITH_REPORT_WARNING"}:
        return in_memory
    if temporary.is_file() and (not path.is_file() or temporary.stat().st_mtime >= path.stat().st_mtime):
        try:
            recovered = json.loads(temporary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            recovered = None
        if recovered and recovered.get("status") in {"FAILED", "COMPLETED", "COMPLETED_WITH_REPORT_WARNING"}:
            return recovered
    if path.is_file():
        return pipeline.load(run_id).model_dump(mode="json")
    if in_memory is not None:
        return in_memory
    raise HTTPException(status_code=404, detail="research_run_not_found")


@app.get("/health/ready")
def health_ready() -> dict:
    return {"status": "ready", "service": "boilermind-research-v2", "port": 8765}


@app.get("/api/v1/capabilities")
def capabilities() -> dict:
    return pipeline.capabilities()


@app.post("/api/v1/assistant")
def assistant(request: dict) -> dict:
    """聊天模式入口：形成工程回答，并把问题作为科研问题摘要交回前端。

    前端收到 research_question_summary 后会自动进入六阶段完整实验。
    该实现为确定性回答，不依赖外部 LLM。
    """
    question = str(request.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="question_required")
    attachment_ids = [str(item) for item in (request.get("attachmentIds") or []) if item]
    attachment_names = []
    for attachment_id in attachment_ids:
        upload_dir = UPLOAD_ROOT / attachment_id
        if not upload_dir.is_dir():
            continue
        for child in sorted(upload_dir.iterdir()):
            if child.is_file():
                attachment_names.append(child.name)
    if is_control_optimization_question(question):
        answer = (
            "已识别为锅炉控制优化问题：在汽包压力≤23MPa约束下，联合调节"
            "给煤、给水、送风和汽包压力，使蒸汽体积量V提升15%并推送到Unity。"
            "将进入完整六阶段实验验证并生成控制指令。"
        )
    else:
        answer = "已形成可执行科研问题，将进入完整六阶段实验验证。"
    if attachment_names:
        answer += " 已接收附件：" + "、".join(attachment_names) + "（附件已保存；当前确定性回答未读取附件内容）。"
    return {
        "answer": answer,
        "provider": "boilermind-research-v2",
        "sources": [],
        "data_needs": [],
        "hypothesis_ready": False,
        "research_question_summary": question,
    }


@app.post("/api/v1/research-runs", status_code=202)
def create_research_run(request: ResearchRequest) -> dict:
    run_id = request.run_id or ("RUN-" + uuid.uuid4().hex[:12].upper())
    queued = request.model_copy(update={"run_id": run_id})
    with _lock:
        if run_id in _states:
            raise HTTPException(status_code=409, detail="research_run_exists")
        _states[run_id] = {"schema_version": "boilermind.research_run.v2", "run_id": run_id, "status": "QUEUED", "question": request.question}
    threading.Thread(target=_execute, args=(queued,), daemon=True).start()
    return _states[run_id]


@app.get("/api/v1/research-runs/{run_id}")
def get_research_run(run_id: str) -> dict:
    return _state(run_id)


@app.get("/api/v1/research-runs/{run_id}/frontend")
def get_research_run_frontend(run_id: str) -> dict:
    """Return the stable, product-facing projection without exposing internals."""

    state = _state(run_id)
    return {"success": True, "data": project_run(state, pipeline.run_root / run_id), "error": None}


@app.get("/api/v1/research-runs")
def list_research_runs(
    query: str = "",
    status: str = "",
    page: int = Query(1, ge=1),
    page_size: int = Query(20, alias="pageSize", ge=1, le=100),
    sort: str = "-updatedAt",
) -> dict:
    items = []
    if pipeline.run_root.is_dir():
        for path in sorted(pipeline.run_root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            run_path = path / "run.json"
            if not run_path.is_file():
                continue
            try:
                state = json.loads(run_path.read_text(encoding="utf-8"))
                projected = project_run(state, path)["run"]
                question = str(projected.get("question") or "").strip()
                if not question:
                    continue
                if query and query.casefold() not in question.casefold():
                    continue
                if status and str(projected.get("status") or "") != status:
                    continue
                items.append(projected)
            except (OSError, ValueError, TypeError):
                continue
    total = len(items)
    start = (page - 1) * page_size
    paged = items[start:start + page_size]
    return {
        "success": True,
        "data": {
            "items": paged,
            "total": total,
            "page": page,
            "page_size": page_size,
        },
        "error": None,
    }


@app.post("/api/v1/uploads")
async def upload_files(files: list[UploadFile] = File(...)) -> dict:
    """前端“添加资料”上传：落盘到 runtime/uploads/，返回附件 id 供前端引用。"""
    saved = []
    for file in files:
        if not file.filename:
            continue
        content = await file.read()
        if not content:
            continue
        attachment_id = f"ATT-{uuid.uuid4().hex[:12].upper()}"
        upload_dir = UPLOAD_ROOT / attachment_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename).name
        target = upload_dir / safe_name
        target.write_bytes(content)
        saved.append({
            "id": attachment_id,
            "filename": safe_name,
            "size_bytes": len(content),
        })
    return {"success": True, "data": {"attachments": saved}, "error": None}


@app.get("/api/v1/research-runs/{run_id}/report")
def get_report(run_id: str) -> dict:
    state = _state(run_id)
    if state.get("status") not in {"COMPLETED", "COMPLETED_WITH_REPORT_WARNING"}:
        raise HTTPException(status_code=409, detail="research_run_not_completed")
    scientific = pipeline.run_root / run_id / "scientific_research_plan" / "scientific_research_plan.json"
    if scientific.is_file():
        return json.loads(scientific.read_text(encoding="utf-8"))
    return json.loads(Path(state["report"]["structured_path"]).read_text(encoding="utf-8"))


def _artifact_path(run_id: str, artifact_id: str) -> Path:
    run_dir = (pipeline.run_root / run_id).resolve()
    paths = {
        "structured_report": run_dir / "structured_report.json",
        "narrative_report": run_dir / "narrative_report.md",
        "scientific_plan_json": run_dir / "scientific_research_plan" / "scientific_research_plan.json",
        "scientific_plan_markdown": run_dir / "scientific_research_plan" / "scientific_research_plan.md",
        "scientific_plan_word": run_dir / "scientific_research_plan" / "scientific_research_plan.docx",
        "scientific_plan_pdf": run_dir / "scientific_research_plan" / "scientific_research_plan.pdf",
        "scientific_plan_manifest": run_dir / "scientific_research_plan" / "manifest.json",
    }
    path = paths.get(artifact_id)
    if path is None or not path.is_file() or run_dir not in path.resolve().parents:
        raise HTTPException(status_code=404, detail="artifact_not_found")
    return path


@app.get("/api/v1/research-runs/{run_id}/artifacts")
def list_artifacts(run_id: str) -> dict:
    projected = project_run(_state(run_id), pipeline.run_root / run_id)
    return {"success": True, "data": projected["artifacts"], "error": None}


@app.get("/api/v1/research-runs/{run_id}/artifacts/{artifact_id}/download")
def download_artifact(run_id: str, artifact_id: str, preview: bool = False) -> FileResponse:
    path = _artifact_path(run_id, artifact_id)
    if preview and path.suffix.casefold() == ".pdf":
        # 内嵌预览：不带 filename，浏览器直接展示 PDF 而不是触发下载
        return FileResponse(path, media_type="application/pdf")
    return FileResponse(path, filename=artifact_display_name(artifact_id, path.name))


@app.get("/api/v1/research-runs/{run_id}/graph")
def get_graph(run_id: str) -> dict:
    state = _state(run_id)
    if state.get("status") not in {"COMPLETED", "COMPLETED_WITH_REPORT_WARNING"}:
        raise HTTPException(status_code=409, detail="research_run_not_completed")
    return {"schema_version": "boilermind.research_trace_graph.v2", "run_id": run_id,
            "nodes": state.get("stage_traces", []), "batches": state.get("batches", [])}


@app.get("/api/v1/knowledge-graph/evolution")
def get_evolution_knowledge_graph() -> dict:
    """可增长科研假设演化图谱（含历史完成运行回放同步）。"""
    ensure_evolution_graph_synced(pipeline.run_root)
    return {
        "success": True,
        "data": load_evolution_graph(),
        "error": None,
    }


@app.get("/api/v1/knowledge-graph/literature")
def get_literature_knowledge_graph() -> dict:
    """基于本地文献知识库（resources/local_rag）的论文-作者-主题图谱。"""
    return {
        "success": True,
        "data": build_literature_graph(),
        "error": None,
    }


@app.get("/api/v1/knowledge-graph/team")
def get_team_knowledge_graph(
    include_variables: bool = False,
    include_correlation: bool = False,
    corr_threshold: float = 0.95,
) -> dict:
    """队友 Neo4j 快照（knowledge_graph/team/kg_snapshot.json）的机理/变量知识图谱。"""
    return {
        "success": True,
        "data": build_team_graph(
            include_variables=include_variables,
            include_correlation=include_correlation,
            corr_threshold=corr_threshold,
        ),
        "error": None,
    }


def _unity_state_path() -> Path:
    return pipeline.run_root.parent / "control" / "unity_push_state.json"


def _load_unity_state() -> dict:
    path = _unity_state_path()
    if not path.is_file():
        return {"schema_version": "boilermind.unity_push_state.v1", "runs": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema_version": "boilermind.unity_push_state.v1", "runs": {}}
    payload.setdefault("runs", {})
    return payload


def _save_unity_state(payload: dict) -> None:
    path = _unity_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _unity_entry(run_id: str) -> dict:
    payload = _load_unity_state()
    entry = payload["runs"].setdefault(run_id, {})
    entry.setdefault("push_status", "payload_generated")
    return entry


def _set_unity_status(run_id: str, status: str, **fields: Any) -> dict:
    payload = _load_unity_state()
    entry = payload["runs"].setdefault(run_id, {})
    now = datetime.now(timezone.utc).isoformat()
    if status == "sent":
        entry.setdefault("pushed_at", now)
    elif status == "received":
        entry.setdefault("received_at", now)
    elif status == "executed":
        entry.setdefault("executed_at", now)
    elif status == "returned":
        entry.setdefault("returned_at", now)
    entry["push_status"] = status
    entry.update(fields)
    _save_unity_state(payload)
    return entry


def _completed_member_outcome(state: dict) -> dict | None:
    for batch in reversed(state.get("batches") or []):
        for member in batch.get("members") or []:
            if member.get("status") == "COMPLETED" and member.get("outcome"):
                return member.get("outcome") or {}
    return None


def _control_instruction(run_id: str) -> dict | None:
    """把已完成的控制优化结果构造成 Unity targetResult 控制指令。"""
    state = _state(run_id)
    if state.get("status") not in {"COMPLETED", "COMPLETED_WITH_REPORT_WARNING"}:
        return None
    outcome = _completed_member_outcome(state)
    summary = (outcome or {}).get("control_summary") or {}
    payload_path = summary.get("unity_payload_path")
    if not payload_path or not Path(payload_path).is_file():
        return None
    try:
        unity_payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if _unity_entry(run_id).get("push_status") == "returned":
        return None
    recommended = unity_payload.get("recommended_values") or []
    variable_order = unity_payload.get("variable_order") or ["给煤", "给水", "送风", "汽包压力"]
    current_values = unity_payload.get("current_values") or []
    if len(recommended) < 4:
        return None
    rec_coal, rec_water, rec_air, rec_pressure = (
        float(recommended[0]), float(recommended[1]),
        float(recommended[2]), float(recommended[3]),
    )
    sim = run_boiler_simulation(
        coal_feed=rec_coal,
        water_flow=rec_water,
        air_flow=rec_air,
        drum_pressure=rec_pressure,
        slag_degree=0.05,
    )
    current_volume = unity_payload.get("current_volume")
    predicted_volume = unity_payload.get("predicted_volume")
    predicted_rise = unity_payload.get("predicted_rise")
    return {
        "type": "targetResult",
        "flow": "control_instruction",
        "flow_id": run_id,
        "run_id": run_id,
        "payload_path": payload_path,
        "target_steam": sim["steam_output"],
        "wall_temp": sim["wall_temp"],
        "state_code": sim["state_code"],
        "state_name": sim["state_name"],
        "rec_coal_feed": rec_coal,
        "rec_air_flow": rec_air,
        "rec_water_flow": rec_water,
        "rec_drum_pressure": rec_pressure,
        "rec_slag_degree": 0.05,
        "rec_notes": (
            "BoilerMind 控制优化推荐方案：HGB 软测验证 + 固定随机种子约束搜索；"
            f"预测 V 由 {current_volume:.4f} 升至 {predicted_volume:.4f}"
            f"（约 +{predicted_rise * 100:.2f}%），汽包压力不超过 23 MPa。"
            "请按推荐值或可行范围在模拟面板执行调整，并回传实际蒸汽体积量。"
        ),
        "current_values": (
            dict(zip(variable_order, current_values)) if current_values else {}
        ),
        "recommended_values": dict(zip(variable_order, recommended)),
        "adjustment_ranges": unity_payload.get("adjustment_ranges"),
        "pressure_limit_mpa": unity_payload.get("pressure_limit_mpa"),
        "target_rise": unity_payload.get("target_rise"),
        "predicted_rise": predicted_rise,
        "current_volume": current_volume,
        "predicted_volume": predicted_volume,
        "sim_steam_recommended": sim["steam_output"],
        "source": "boilermind_control_optimization",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _mark_sent(run_id: str, instruction: dict) -> dict:
    payload = _load_unity_state()
    entry = payload["runs"].setdefault(run_id, {})
    now = datetime.now(timezone.utc).isoformat()
    prior = entry.get("push_status")
    if prior in (None, "payload_generated", "none"):
        entry["push_status"] = "sent"
        entry["pushed_at"] = now
        entry.setdefault("notes", []).append(
            f"控制指令已通过 WebSocket 推送给 Unity（{now}）。"
        )
    entry["payload_path"] = str(instruction.get("payload_path") or "")
    entry["recommended_values"] = instruction.get("recommended_values") or {}
    entry["current_values"] = instruction.get("current_values") or {}
    entry["sim_steam_recommended"] = instruction.get("sim_steam_recommended")
    entry["predicted_volume"] = instruction.get("predicted_volume")
    entry["current_volume"] = instruction.get("current_volume")
    entry["pressure_limit_mpa"] = instruction.get("pressure_limit_mpa")
    entry["target_rise"] = instruction.get("target_rise")
    _save_unity_state(payload)
    return entry


def _finalize_unity_return(
    run_id: str,
    actual_volume: float,
    *,
    notes: str | list[str] | None = None,
) -> dict:
    """Unity 回传实际 V，计算偏差并生成第二层裁决。"""
    state = _state(run_id)
    outcome = _completed_member_outcome(state) or {}
    summary = outcome.get("control_summary") or {}
    current_volume = float(summary.get("current_volume") or 0.0)
    predicted_volume = float(summary.get("predicted_volume") or 0.0)
    target_volume = float(summary.get("target_volume") or 0.0)
    target_rise = (
        target_volume / current_volume - 1.0
        if target_volume > 0 and current_volume > 0
        else 0.15
    )
    if current_volume <= 0:
        raise HTTPException(status_code=409, detail="control_baseline_missing")
    actual_rise = actual_volume / current_volume - 1.0
    deviation = (
        (actual_volume - predicted_volume) / predicted_volume * 100.0
        if predicted_volume
        else None
    )
    if actual_rise >= target_rise * 0.98:
        second_verdict = "supported"
    elif actual_rise >= target_rise * 0.90:
        second_verdict = "partially_supported"
    else:
        second_verdict = "falsified"
    note_lines = [notes] if isinstance(notes, str) else list(notes or [])
    entry = _set_unity_status(
        run_id,
        "returned",
        actual_volume=actual_volume,
        actual_rise_pct=actual_rise,
        deviation_pct=deviation,
        second_verdict=second_verdict,
        notes=[
            *(_load_unity_state()["runs"].get(run_id, {}).get("notes") or []),
            *note_lines,
            (
                f"Unity 回传实际 V={actual_volume:.4f}，实际提升 {actual_rise * 100:.2f}%，"
                f"第二层裁决={second_verdict}。"
            ),
        ],
    )
    run_dir = pipeline.run_root / run_id / "unity"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "unity_return.json").write_text(
        json.dumps({
            "schema_version": "boilermind.unity_return.v1",
            "run_id": run_id,
            "returned_at": entry["returned_at"],
            "actual_volume": actual_volume,
            "current_volume": current_volume,
            "predicted_volume": predicted_volume,
            "actual_rise_pct": actual_rise,
            "deviation_pct": deviation,
            "second_verdict": second_verdict,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return entry


async def _broadcast_run(run_id: str, message: dict) -> None:
    dead: list[WebSocket] = []
    for ws in list(_unity_clients.get(run_id) or ()):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _unity_clients[run_id].discard(ws)


async def _broadcast_all(message: dict) -> None:
    for run_id in list(_unity_clients):
        await _broadcast_run(run_id, message)


async def _handle_unity_message(run_id: str, message: dict) -> bool:
    """处理 Unity 客户端上行消息，返回 True 表示已处理（状态已广播）。"""
    msg_type = str(message.get("type") or "")
    if msg_type == "unity_ack":
        entry = _set_unity_status(
            run_id, "received",
            ack_detail=str(message.get("detail") or "Unity 已接收控制指令"),
        )
    elif msg_type == "unity_executed":
        entry = _set_unity_status(
            run_id, "executed",
            execute_detail=str(message.get("detail") or "Unity 已执行控制调整"),
        )
    elif msg_type == "unity_result":
        actual = message.get("actual_volume")
        if actual is None:
            return False
        try:
            actual = float(actual)
        except (TypeError, ValueError):
            return False
        entry = _finalize_unity_return(
            run_id, actual,
            notes=str(message.get("notes") or "Unity WebSocket 回传"),
        )
    else:
        return False
    await _broadcast_run(run_id, {
        "type": "unityStatus",
        "run_id": run_id,
        "status": entry.get("push_status"),
        "entry": entry,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return True


def _parse_sim_params(payload: dict) -> dict:
    return {
        "coal_feed": float(payload.get("coal_feed", 130.0)),
        "air_flow": float(payload.get("air_flow", 1100.0)),
        "water_flow": float(payload.get("water_flow", 950.0)),
        "drum_pressure": float(payload.get("drum_pressure", 17.5)),
        "slag_degree": float(payload.get("slag_degree", 0.1)),
    }


def _active_control_recommendation() -> dict | None:
    """返回最近一个处于 sent/received/executed 的控制任务的推荐参数（模拟面板单位）。"""
    payload = _load_unity_state()
    for run_id, entry in payload["runs"].items():
        if entry.get("push_status") not in {"sent", "received", "executed"}:
            continue
        recommended = entry.get("recommended_values") or {}
        if not recommended:
            continue
        return {
            "coal_feed": float(recommended.get("给煤", 130.0)),
            "water_flow": float(recommended.get("给水", 950.0)),
            "air_flow": float(recommended.get("送风", 1100.0)),
            "drum_pressure": float(recommended.get("汽包压力", 17.5)),
            "slag_degree": 0.05,
        }
    return None


async def _maybe_record_control_adoption(params: dict, result: dict) -> None:
    """Unity 面板模拟参数与推荐值匹配时，自动推进 executed → returned。"""
    payload = _load_unity_state()
    mapping = {
        "coal_feed": "给煤",
        "water_flow": "给水",
        "air_flow": "送风",
        "drum_pressure": "汽包压力",
    }
    for run_id, entry in payload["runs"].items():
        status = entry.get("push_status")
        if status not in {"sent", "received", "executed"}:
            continue
        recommended = entry.get("recommended_values") or {}
        matched = True
        for sim_key, var_name in mapping.items():
            rec = recommended.get(var_name)
            current = params.get(sim_key)
            if rec is None or current is None:
                matched = False
                break
            if abs(float(current) - float(rec)) > max(abs(float(rec)) * 0.03, 1e-6):
                matched = False
                break
        if not matched:
            continue
        if status == "sent":
            _set_unity_status(run_id, "received")
        if entry.get("push_status") != "returned":
            _set_unity_status(
                run_id, "executed",
                execute_detail="Unity 面板已按推荐值执行调整",
            )
        predicted = entry.get("predicted_volume")
        sim_rec = entry.get("sim_steam_recommended")
        if predicted and sim_rec and float(sim_rec) > 0:
            actual_volume = float(result["steam_output"]) / float(sim_rec) * float(predicted)
            entry = _finalize_unity_return(
                run_id, actual_volume,
                notes="Unity 面板模拟参数匹配推荐值后自动回传",
            )
            await _broadcast_run(run_id, {
                "type": "unityStatus",
                "run_id": run_id,
                "status": "returned",
                "entry": entry,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        return


@app.get("/api/v1/research-runs/{run_id}/unity/status")
def get_unity_status(run_id: str) -> dict:
    _state(run_id)
    entry = (_load_unity_state()["runs"]).get(run_id) or {"push_status": "payload_generated"}
    instruction = _control_instruction(run_id)
    return {
        "success": True,
        "data": {
            "run_id": run_id,
            "status": entry,
            "instruction_ready": instruction is not None,
        },
        "error": None,
    }


@app.post("/api/v1/research-runs/{run_id}/unity/push")
async def push_unity_instruction(run_id: str) -> dict:
    instruction = _control_instruction(run_id)
    if instruction is None:
        raise HTTPException(status_code=409, detail="control_instruction_unavailable")
    entry = _mark_sent(run_id, instruction)
    await _broadcast_run(run_id, instruction)
    await _broadcast_run(run_id, {
        "type": "unityStatus", "run_id": run_id,
        "status": entry.get("push_status"), "entry": entry,
    })
    return {"success": True, "data": {"run_id": run_id, "push_status": "sent", "instruction": instruction}, "error": None}


@app.post("/api/v1/research-runs/{run_id}/unity/ack")
async def ack_unity_instruction(run_id: str) -> dict:
    _state(run_id)
    entry = _set_unity_status(run_id, "received")
    await _broadcast_run(run_id, {
        "type": "unityStatus", "run_id": run_id,
        "status": entry.get("push_status"), "entry": entry,
    })
    return {"success": True, "data": {"run_id": run_id, "push_status": "received"}, "error": None}


@app.post("/api/v1/research-runs/{run_id}/unity/execute")
async def execute_unity_instruction(run_id: str, payload: dict | None = None) -> dict:
    _state(run_id)
    entry = _set_unity_status(
        run_id, "executed",
        execute_detail=str((payload or {}).get("detail") or "Unity 已执行控制调整"),
    )
    await _broadcast_run(run_id, {
        "type": "unityStatus", "run_id": run_id,
        "status": entry.get("push_status"), "entry": entry,
    })
    return {"success": True, "data": {"run_id": run_id, "push_status": "executed"}, "error": None}


@app.post("/api/v1/research-runs/{run_id}/unity/report")
async def report_unity_result(run_id: str, payload: dict) -> dict:
    """Record a real Unity intervention return and derive the second verdict."""
    state = _state(run_id)
    if state.get("status") not in {"COMPLETED", "COMPLETED_WITH_REPORT_WARNING"}:
        raise HTTPException(status_code=409, detail="research_run_not_completed")
    actual_volume = payload.get("actual_volume")
    if actual_volume is None:
        raise HTTPException(status_code=422, detail="actual_volume_required")
    try:
        actual_volume = float(actual_volume)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="actual_volume_must_be_number") from exc
    entry = _finalize_unity_return(run_id, actual_volume)
    await _broadcast_run(run_id, {
        "type": "unityStatus", "run_id": run_id,
        "status": "returned", "entry": entry,
    })
    return {
        "success": True,
        "data": {
            "run_id": run_id,
            "actual_volume": actual_volume,
            "actual_rise_pct": entry.get("actual_rise_pct"),
            "deviation_pct": entry.get("deviation_pct"),
            "second_verdict": entry.get("second_verdict"),
        },
        "error": None,
    }


@app.post("/api/simulate")
async def simulate_boiler(payload: dict) -> dict:
    """Unity 面板实时模拟：按输入参数计算蒸汽量，并检测是否采纳控制推荐。"""
    params = _parse_sim_params(payload)
    result = run_boiler_simulation(**params)
    await _maybe_record_control_adoption(params, result)
    await _broadcast_all({
        "type": "simResult",
        **result,
        "source": payload.get("source", "boiler_sim_panel"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return result


@app.post("/api/calculate_target")
async def calculate_target(payload: dict) -> dict:
    """Unity 面板目标蒸汽量计算；有活跃控制任务时推荐参数来自 BoilerMind。"""
    params = _parse_sim_params(payload)
    recommendation = _active_control_recommendation()
    rec = recommendation or {
        **params,
        "slag_degree": min(float(params["slag_degree"]), 0.1),
    }
    sim = run_boiler_simulation(**rec)
    result = {
        "status": "ok",
        "target_steam": sim["steam_output"],
        "wall_temp": sim["wall_temp"],
        "state_code": sim["state_code"],
        "state_name": sim["state_name"],
        "rec_coal_feed": rec["coal_feed"],
        "rec_air_flow": rec["air_flow"],
        "rec_water_flow": rec["water_flow"],
        "rec_drum_pressure": rec["drum_pressure"],
        "rec_slag_degree": rec["slag_degree"],
    }
    await _broadcast_all({
        "type": "targetResult",
        "target_steam": sim["steam_output"],
        "wall_temp": sim["wall_temp"],
        "state_code": sim["state_code"],
        "state_name": sim["state_name"],
        "rec_coal_feed": rec["coal_feed"],
        "rec_air_flow": rec["air_flow"],
        "rec_water_flow": rec["water_flow"],
        "rec_drum_pressure": rec["drum_pressure"],
        "rec_slag_degree": rec["slag_degree"],
        "source": payload.get("source", "boiler_sim_panel"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return result


@app.websocket("/api/v1/research-runs/{run_id}/events")
async def research_run_events(websocket: WebSocket, run_id: str) -> None:
    """Unity 实时桥接：下行状态/控制指令，上行确认/执行/回传。"""
    await websocket.accept()
    _unity_clients[run_id].add(websocket)
    previous = None
    instruction_sent = False
    try:
        while True:
            inbound = None
            try:
                inbound = await asyncio.wait_for(
                    websocket.receive_json(), timeout=0.5
                )
            except asyncio.TimeoutError:
                pass
            except Exception:
                break
            if inbound and await _handle_unity_message(run_id, inbound):
                continue
            try:
                state = _state(run_id)
            except HTTPException:
                await websocket.send_json({"run_id": run_id, "status": "NOT_FOUND"})
                await websocket.close(code=4404)
                return
            # 控制指令：连接后自动下发一次（payload_generated → sent）
            if not instruction_sent:
                instruction = _control_instruction(run_id)
                if instruction is not None:
                    await websocket.send_json(instruction)
                    instruction_sent = True
                    _mark_sent(run_id, instruction)
            status = state.get("status")
            unity_status = (
                (_load_unity_state()["runs"]).get(run_id) or {}
            ).get("push_status")
            fingerprint = (
                status,
                len(state.get("stage_traces", [])),
                len(state.get("batches", [])),
                unity_status,
            )
            if fingerprint != previous:
                await websocket.send_json(jsonable_encoder(state))
                previous = fingerprint
            # Keep the channel open after a terminal research state. Unity uses
            # the socket as a live bridge while the completed run is being
            # inspected; closing it here made a successfully synchronized
            # digital twin appear as "disconnected" immediately.
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    finally:
        _unity_clients[run_id].discard(websocket)


FRONTEND_ROOT = Path(__file__).resolve().parents[2] / "frontend"


@app.get("/")
def frontend_index() -> FileResponse:
    return FileResponse(FRONTEND_ROOT / "index.html")


app.mount("/assets", StaticFiles(directory=FRONTEND_ROOT / "assets"), name="frontend-assets")
app.mount("/js", StaticFiles(directory=FRONTEND_ROOT / "js"), name="frontend-js")
