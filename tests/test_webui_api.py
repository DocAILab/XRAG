from __future__ import annotations

import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from xrag.config import Config
from xrag.webui import server


@pytest.fixture(autouse=True)
def reset_webui_state():
    original = server.STATE
    cfg = Config()
    original_config = deepcopy(cfg.__dict__)
    server.STATE = server.AppState()
    yield
    server.STATE = original
    cfg.__dict__.clear()
    cfg.__dict__.update(original_config)


def _module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def test_capabilities_describe_six_step_contract():
    response = TestClient(server.app).get("/api/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == 1
    assert set(payload["orchestrator_support"]) == {
        "default", "self", "adaptive", "sim", "open"
    }
    assert "similarity_top_k" in payload["fields"]["retrieval"]
    assert "chunk_sizes" in payload["fields"]["index"]
    assert "SePer Evaluation" in payload["options"]["metric_groups"]


@pytest.mark.parametrize("orchestrator", ["default", "self", "adaptive", "sim", "open"])
def test_builds_each_orchestrator(monkeypatch, orchestrator):
    retriever = object()
    engine = SimpleNamespace(kind=orchestrator)
    server.STATE.index = object()

    monkeypatch.setattr(server, "_selected_orchestrator", lambda cfg: orchestrator)
    monkeypatch.setitem(
        sys.modules,
        "xrag.retrievers.retriever",
        _module("xrag.retrievers.retriever", get_retriver=lambda *args, **kwargs: retriever),
    )
    monkeypatch.setitem(
        sys.modules,
        "xrag.launcher",
        _module("xrag.launcher", build_query_engine=lambda *args, **kwargs: engine),
    )
    monkeypatch.setitem(
        sys.modules,
        "xrag.self_rag",
        _module("xrag.self_rag", SelfRAGPipeline=lambda *args, **kwargs: engine),
    )
    monkeypatch.setitem(
        sys.modules,
        "xrag.open_rag",
        _module("xrag.open_rag", OpenRAGPipeline=lambda *args, **kwargs: engine),
    )
    monkeypatch.setitem(
        sys.modules,
        "xrag.adaptive_rag.engine",
        _module("xrag.adaptive_rag.engine", AdaptiveRAG=lambda *args, **kwargs: engine),
    )

    result = server.build_query_engine_endpoint()

    assert result == {"ok": True, "engine": orchestrator}
    assert server.STATE.orchestrator == orchestrator
    if orchestrator == "sim":
        assert server.STATE.orchestrator_engine == {"retriever": retriever}
    else:
        assert server.STATE.orchestrator_engine is engine


def test_query_normalizes_pipeline_response():
    server.STATE.orchestrator = "self"
    server.STATE.orchestrator_engine = SimpleNamespace(
        query=lambda question: {
            "response": f"answer: {question}",
            "retrieved_documents": [{"id": "doc-1", "text": "context", "score": 0.75}],
        }
    )

    response = TestClient(server.app).post("/api/query", json={"question": "why"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "answer: why",
        "retrieved_documents": [{"id": "doc-1", "text": "context", "score": 0.75}],
        "orchestrator": "self",
    }


def test_experiment_uses_experiment_1_sample_count(monkeypatch):
    cfg = Config()
    original_experiment_1 = getattr(cfg, "experiment_1", False)
    original_count = cfg.test_init_total_number_documents
    cfg.test_init_total_number_documents = 17
    server.STATE.qa_dataset = {"documents": [], "test_data": {}}
    server.STATE.dataset = server.DatasetSummary("test", "Test", "json", 0, 0)
    server.STATE.orchestrator_engine = object()

    started = {}

    class DeferredThread:
        def __init__(self, target, args, **kwargs):
            started["target"] = target
            started["args"] = args

        def start(self):
            started["called"] = True

    monkeypatch.setattr(server.threading, "Thread", DeferredThread)
    try:
        response = TestClient(server.app).post(
            "/api/evaluate/start",
            json={"metrics": ["NLG_chrf"], "num_samples": 3, "experiment_1": True},
        )
    finally:
        cfg.experiment_1 = original_experiment_1
        cfg.test_init_total_number_documents = original_count

    assert response.status_code == 200
    payload = response.json()
    experiment = server.get_experiment(payload["experiment_id"])
    assert experiment["num_samples"] == 17
    assert experiment["task_id"] == payload["task_id"]
    assert started["args"][2] == 17
    assert started["called"] is True
    assert server.list_experiments()["experiments"][0]["experiment_id"] == payload["experiment_id"]


def test_evaluation_rejects_metrics_outside_capabilities():
    server.STATE.qa_dataset = {"documents": [], "test_data": {}}
    server.STATE.orchestrator_engine = object()

    response = TestClient(server.app).post(
        "/api/evaluate/start",
        json={"metrics": ["not-a-real-metric"], "num_samples": 1},
    )

    assert response.status_code == 400
    assert "Unsupported metrics" in response.json()["detail"]


def test_repeated_index_build_keeps_base_persist_dir(monkeypatch):
    from xrag.launcher import launch

    cfg = Config()
    original_persist_dir = cfg.persist_dir
    seen_paths = []
    monkeypatch.setattr(launch, "Settings", SimpleNamespace())
    monkeypatch.setattr(launch, "get_llm", lambda *args, **kwargs: object())
    monkeypatch.setattr(launch, "get_embedding", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        launch,
        "get_index",
        lambda documents, persist_dir, **kwargs: (seen_paths.append(persist_dir) or object(), None),
    )

    launch.build_index([])
    launch.build_index([])

    assert cfg.persist_dir == original_persist_dir
    assert seen_paths[0] == seen_paths[1]


def test_configuration_changes_invalidate_downstream_state():
    server.STATE.index = object()
    server.STATE.orchestrator_engine = object()
    response = server.update_vector(server.VectorDBUpdate(chunk_size=321))

    assert response["config"]["chunk_size"] == 321
    assert server.STATE.index is None
    assert server.STATE.orchestrator_engine is None
