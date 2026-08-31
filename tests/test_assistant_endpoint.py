"""聊天模式 /api/v1/assistant 端点测试。"""

from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


app_module = importlib.import_module("server.research_api.app")


def test_assistant_control_question_returns_research_summary() -> None:
    client = TestClient(app_module.app)
    question = (
        "在汽包压力不超过23MPa的限制下，怎么调节给煤、给水、送风和汽包压力，"
        "使蒸汽体积量V上升15%，并将验证结果推送到Unity？"
    )
    response = client.post("/api/v1/assistant", json={"question": question})
    assert response.status_code == 200
    payload = response.json()
    assert "控制优化" in payload["answer"]
    assert payload["research_question_summary"] == question
    assert payload["provider"]
    assert payload["hypothesis_ready"] is False


def test_assistant_generic_question_returns_research_summary() -> None:
    client = TestClient(app_module.app)
    response = client.post("/api/v1/assistant", json={"question": "比较 Ridge 与 LSTM 的预测性能"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["research_question_summary"] == "比较 Ridge 与 LSTM 的预测性能"
    assert payload["answer"]


def test_assistant_requires_question() -> None:
    client = TestClient(app_module.app)
    response = client.post("/api/v1/assistant", json={})
    assert response.status_code == 422
