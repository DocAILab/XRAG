"""XRAG WebUI FastAPI server.

Exposes REST endpoints that drive the existing XRAG CLI/launcher modules,
plus a static file mount for the HTML/CSS/JS frontend that matches the
prototype screenshots in the README.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --- xrag imports (deferred until needed so that the static-only /health
# endpoint can answer before heavy deps are loaded).
from xrag.config import Config


logger = logging.getLogger("xrag.webui")

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Static option catalogues (also used by the frontend if it needs them).
# ---------------------------------------------------------------------------
LLM_OPTIONS = ["openai", "huggingface", "ollama"]
HF_MODEL_OPTIONS = [
    "llama",
    "chatglm",
    "qwen",
    "qwen14_int8",
    "qwen7_int8",
    "qwen1.8",
    "baichuan",
    "falcon",
    "mpt",
    "yi",
]
EMBEDDING_OPTIONS = [
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-base-zh-v1.5",
    "BAAI/bge-small-zh-v1.5",
]
SPLIT_TYPE_OPTIONS = ["sentence", "sentence_window", "character", "hierarchical"]
RESPONSE_SYNTHESIZER_OPTIONS = [
    "refine", "compact", "compact_accumulate", "accumulate",
    "tree_summarize", "simple_summarize", "no_text", "generation",
]
PRESET_DATASET_OPTIONS = ["hotpot_qa", "drop", "natural_questions"]
DATASET_DISPLAY_MAP = {
    "hotpot_qa": "HotpotQA",
    "drop": "DropQA",
    "natural_questions": "NaturalQA",
}
DATASETS_INFO = {
    "HotpotQA": {
        "size": {
            "train": "86,830",
            "validation": "8,680",
            "test": "968",
        },
        "corpus": {"documents": "508,826", "source": "Wikipedia"},
        "features": {
            "Multi-hop": True,
            "Constrained": False,
            "Numerical": True,
            "Set-logical": False,
        },
    },
    "DropQA": {
        "size": {
            "train": "78,241",
            "validation": "7,824",
            "test": "870",
        },
        "corpus": {"documents": "6,147", "source": "Wikipedia"},
        "features": {
            "Multi-hop": True,
            "Constrained": False,
            "Numerical": True,
            "Set-logical": True,
        },
    },
    "NaturalQA": {
        "size": {
            "train": "100,093",
            "validation": "10,010",
            "test": "1,112",
        },
        "corpus": {"documents": "49,815", "source": "Wikipedia"},
        "features": {
            "Multi-hop": True,
            "Constrained": True,
            "Numerical": True,
            "Set-logical": False,
        },
    },
}
RETRIEVER_OPTIONS = [
    "BM25",
    "Vector",
    "Summary",
    "Tree",
    "Keyword",
    "Custom",
    "QueryFusion",
    "AutoMerging",
    "Recursive",
    "SentenceWindow",
]
RETRIEVER_MODE_OPTIONS = [0, 1]
POSTPROCESS_RERANK_OPTIONS = [
    "none",
    "long_context_reorder",
    "colbertv2_rerank",
    "bge-reranker-base",
]
QUERY_TRANSFORM_OPTIONS = [
    "none",
    "hyde_zeroshot",
    "hyde_fewshot",
    "stepback_zeroshot",
    "stepback_fewshot",
]
ORCHESTRATOR_OPTIONS = ["default", "self", "adaptive", "sim", "open"]
ORCHESTRATOR_DISPLAY = {
    "default": "Default",
    "self": "Self-RAG",
    "adaptive": "Adaptive-RAG",
    "sim": "SIM-RAG",
    "open": "Open-RAG",
}

# Metrics grouped exactly as in the prototype (Step 5).
METRIC_GROUPS: Dict[str, List[Dict[str, str]]] = {
    "NLG Evaluation": [
        {"id": "NLG_chrf", "label": "ChrF"},
        {"id": "NLG_meteor", "label": "METEOR"},
        {"id": "NLG_wer", "label": "WER"},
        {"id": "NLG_cer", "label": "CER"},
        {"id": "NLG_chrf_pp", "label": "ChrF++"},
        {"id": "NLG_perplexity", "label": "PPL"},
        {"id": "NLG_rouge_rouge1", "label": "ROUGE1"},
        {"id": "NLG_rouge_rouge2", "label": "ROUGE2"},
        {"id": "NLG_rouge_rougeL", "label": "ROUGEL"},
        {"id": "NLG_rouge_rougeLsum", "label": "ROUGELSUM"},
        {"id": "nlg-em", "label": "EM"},
    ],
    "LLaMA Evaluation": [
        {"id": "Llama_retrieval_Faithfulness", "label": "Llama-Response-Faithfulness"},
        {"id": "Llama_retrieval_Relevancy", "label": "Llama-Response-Relevance"},
        {"id": "Llama_response_correctness", "label": "Llama-Response-Correctness"},
        {"id": "Llama_response_semanticSimilarity", "label": "Llama-Response-Similarity"},
        {"id": "Llama_response_answerRelevancy", "label": "Llama-Response-Relevance++"},
        {"id": "Llama_retrieval_FaithfulnessG", "label": "Llama-Response-Faithfulness+"},
        {"id": "Llama_retrieval_RelevancyG", "label": "Llama-Response-Relevance+"},
    ],
    "DeepEval Evaluation": [
        {"id": "DeepEval_retrieval_contextualPrecision", "label": "DeepEval-Context-Recall"},
        {"id": "DeepEval_retrieval_contextualRecall", "label": "DeepEval-Context-Relevance"},
        {"id": "DeepEval_retrieval_contextualRelevancy", "label": "Uptrain-Context-Consistency"},
        {"id": "DeepEval_retrieval_faithfulness", "label": "DeepEval-Context-Faithfulness"},
        {"id": "DeepEval_response_answerRelevancy", "label": "DeepEval-Response-Relevancy"},
        {"id": "DeepEval_response_hallucination", "label": "DeepEval-Context-Hallucination"},
    ],
    "UpTrain Evaluation": [
        {"id": "UpTrain_Response_Completeness", "label": "Uptrain-Response-Completeness"},
        {"id": "UpTrain_Response_Conciseness", "label": "Uptrain-Response-Conciseness"},
        {"id": "UpTrain_Response_Relevance", "label": "Uptrain-Response-Relevance"},
        {"id": "UpTrain_Response_Valid", "label": "Uptrain-Response-Valid"},
        {"id": "UpTrain_Response_Consistency", "label": "Uptrain-Response-Consistency"},
        {"id": "UpTrain_Response_Response_Matching", "label": "Uptrain-Response-Matching"},
        {"id": "UpTrain_Retrieval_Context_Relevance", "label": "Uptrain-Context-Relevance"},
        {"id": "UpTrain_Retrieval_Context_Utilization", "label": "Uptrain-Factual-Accuracy"},
        {"id": "UpTrain_Retrieval_Factual_Accuracy", "label": "Uptrain-Factual-Accuracy"},
        {"id": "UpTrain_Retrieval_Context_Conciseness", "label": "Uptrain-Context-Conciseness"},
        {"id": "UpTrain_Retrieval_Code_Hallucination", "label": "Uptrain-Retrieval-Code-Hallucination"},
    ],
    "SePer Evaluation": [
        {"id": "SePer_with_context", "label": "SePer with Context"},
        {"id": "SePer_without_context", "label": "SePer without Context"},
        {"id": "SePer_delta", "label": "SePer Delta"},
    ],
}
ALL_METRIC_IDS: List[str] = [
    metric["id"] for group in METRIC_GROUPS.values() for metric in group
]

DEFAULT_PROMPT_QA = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
DEFAULT_PROMPT_REFINE = (
    "We have the opportunity to refine the original answer(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context and the original answer, refine the original answer to better answer the query.\n"
    "If the new context isn't useful, return the original answer.\n"
    "Query: {query_str}\n"
    "Original Answer: {existing_answer}\n"
    "Refined Answer: "
)


# ---------------------------------------------------------------------------
# Server-side session state.
# ---------------------------------------------------------------------------
@dataclass
class DatasetSummary:
    name: str
    display: str
    source: str  # "preset" | "json" | "folder"
    num_documents: int
    num_test_questions: int
    persist_path: Optional[str] = None


@dataclass
class AppState:
    dataset: Optional[DatasetSummary] = None
    qa_dataset: Any = None  # raw dict returned by xrag.data.qa_loader.get_qa_dataset
    index: Any = None
    hierarchical_storage_context: Any = None
    query_engine: Any = None
    open_rag_engine: Any = None
    orchestrator_engine: Any = None
    orchestrator: str = "default"
    last_evaluation: Optional[Dict[str, Any]] = None
    tasks: Dict[str, "EvalTask"] = field(default_factory=dict)
    experiments: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class EvalTask:
    task_id: str
    experiment_id: Optional[str] = None
    status: str = "pending"  # pending | running | done | error | cancelled
    progress: float = 0.0
    total: int = 0
    completed: int = 0
    failed: int = 0
    samples: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    queue: "queue.Queue[Dict[str, Any]]" = field(default_factory=queue.Queue)
    thread: Optional[threading.Thread] = None
    cancel_flag: threading.Event = field(default_factory=threading.Event)


STATE = AppState()


def _selected_orchestrator(cfg: Config) -> str:
    for name, section in (
        ("self", "self_rag"), ("adaptive", "adaptive_rag"),
        ("sim", "sim_rag"), ("open", "open_rag"),
    ):
        if bool(cfg.config.get(section, {}).get("enabled", False)):
            return name
    return "default"


def _top_k_field(retriever: str) -> str:
    return {
        "BM25": "similarity_top_k_BM25", "Vector": "similarity_top_k_VECTOR",
        "Summary": "similarity_top_k_SUMMARY", "QueryFusion": "similarity_top_k_QUERYFUSION",
        "AutoMerging": "similarity_top_k_AUTOMERGING", "Recursive": "similarity_top_k_RECURSIVE",
    }.get(retriever, "similarity_top_k_VECTOR")


def _retriever_top_k(cfg: Config, retriever: str) -> int:
    return int(getattr(cfg, _top_k_field(retriever), 3))


# ---------------------------------------------------------------------------
# Pydantic models for the request bodies.
# ---------------------------------------------------------------------------
class PresetDatasetRequest(BaseModel):
    name: str  # backend id: hotpot_qa / drop / natural_questions


class FolderDatasetRequest(BaseModel):
    folder_path: str
    output_json: str
    num_questions: int = 3
    sentence_length: int = -1


class LLMUpdate(BaseModel):
    llm: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_name: Optional[str] = None
    auth_token: Optional[str] = None
    huggingface_model: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_request_timeout: Optional[int] = None
    temperature: Optional[float] = None


class VectorDBUpdate(BaseModel):
    embeddings: Optional[str] = None
    split_type: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    persist_dir: Optional[str] = None
    window_size: Optional[int] = Field(default=None, ge=1)
    chunk_sizes: Optional[List[int]] = None


class RetrievalUpdate(BaseModel):
    orchestrator: Optional[str] = None
    retriever: Optional[str] = None
    retriever_mode: Optional[int] = None
    query_transform: Optional[str] = None
    postprocess_rerank: Optional[str] = None
    text_qa_template: Optional[str] = None
    refine_template: Optional[str] = None
    response_synthesizer: Optional[str] = None
    similarity_top_k: Optional[int] = Field(default=None, ge=1)


class EvaluationRequest(BaseModel):
    metrics: List[str] = Field(default_factory=list)
    num_samples: int = Field(default=5, ge=1)
    experiment_1: Optional[bool] = None


class ConfigSummary(BaseModel):
    llm: Dict[str, Any]
    embedding: Dict[str, Any]
    chunk: Dict[str, Any]
    retrieval: Dict[str, Any]
    dataset: Dict[str, Any]
    templates: Dict[str, str]


# ---------------------------------------------------------------------------
# Helper utilities.
# ---------------------------------------------------------------------------
def _cfg_to_dict(cfg: Config) -> Dict[str, Any]:
    """Convert Config singleton into a plain dict for the frontend."""
    return {
        "llm": {
            "llm": getattr(cfg, "llm", None),
            "api_key": getattr(cfg, "api_key", None),
            "api_base": getattr(cfg, "api_base", None),
            "api_name": getattr(cfg, "api_name", None),
            "auth_token": getattr(cfg, "auth_token", None),
            "huggingface_model": getattr(cfg, "huggingface_model", None),
            "ollama_model": getattr(cfg, "ollama_model", None),
            "ollama_request_timeout": getattr(cfg, "ollama_request_timeout", None),
            "temperature": getattr(cfg, "temperature", None),
        },
        "embedding": {
            "embedding_type": getattr(cfg, "embedding_type", None),
            "embeddings": getattr(cfg, "embeddings", None),
            "embed_batch_size": getattr(cfg, "embed_batch_size", None),
        },
        "chunk": {
            "split_type": getattr(cfg, "split_type", None),
            "chunk_size": getattr(cfg, "chunk_size", None),
            "chunk_overlap": getattr(cfg, "chunk_overlap", None),
            "window_size": getattr(cfg, "window_size", None),
            "chunk_sizes": getattr(cfg, "chunk_sizes", None),
            "persist_dir": getattr(cfg, "persist_dir", None),
        },
        "retrieval": {
            "orchestrator": _selected_orchestrator(cfg),
            "retriever": getattr(cfg, "retriever", None),
            "retriever_mode": getattr(cfg, "retriever_mode", None),
            "query_transform": getattr(cfg, "query_transform", None),
            "postprocess_rerank": getattr(cfg, "postprocess_rerank", None),
            "response_synthesizer": getattr(cfg, "responce_synthsizer", None),
            "similarity_top_k": _retriever_top_k(cfg, getattr(cfg, "retriever", "BM25")),
            "metrics": getattr(cfg, "metrics", []),
        },
        "dataset": {
            "dataset_type": getattr(cfg, "dataset_type", None),
            "dataset": getattr(cfg, "dataset", None),
            "dataset_path": getattr(cfg, "dataset_path", None),
            "persist_dir": getattr(cfg, "persist_dir", None),
            "test_init_total_number_documents": getattr(cfg, "test_init_total_number_documents", None),
            "n": getattr(cfg, "n", None),
        },
        "templates": {
            "text_qa_template": getattr(cfg, "text_qa_template_str", "") or DEFAULT_PROMPT_QA,
            "refine_template": getattr(cfg, "refine_template_str", "") or DEFAULT_PROMPT_REFINE,
        },
    }


def _apply_llm_update(cfg: Config, body: LLMUpdate) -> None:
    for field in ("llm", "api_key", "api_base", "api_name", "auth_token",
                  "huggingface_model", "ollama_model", "ollama_request_timeout",
                  "temperature"):
        value = getattr(body, field)
        if value is not None:
            setattr(cfg, field, value)


def _apply_vector_update(cfg: Config, body: VectorDBUpdate) -> None:
    for field in ("embeddings", "split_type", "chunk_size", "chunk_overlap", "persist_dir", "window_size", "chunk_sizes"):
        value = getattr(body, field)
        if value is not None:
            setattr(cfg, field, value)


def _apply_retrieval_update(cfg: Config, body: RetrievalUpdate) -> None:
    if body.orchestrator is not None:
        # Mirror the streamlit webui's behavior: enable exactly one orchestrator.
        for section in ("self_rag", "adaptive_rag", "sim_rag", "open_rag"):
            cfg.config.setdefault(section, {})["enabled"] = False
        section_map = {
            "self": "self_rag",
            "adaptive": "adaptive_rag",
            "sim": "sim_rag",
            "open": "open_rag",
        }
        section = section_map.get(body.orchestrator)
        if section is not None:
            cfg.config.setdefault(section, {})["enabled"] = True
    for field in ("retriever", "retriever_mode", "query_transform", "postprocess_rerank"):
        value = getattr(body, field)
        if value is not None:
            setattr(cfg, field, value)
    if body.response_synthesizer is not None:
        cfg.responce_synthsizer = body.response_synthesizer
    if body.similarity_top_k is not None:
        setattr(cfg, _top_k_field(body.retriever or cfg.retriever), body.similarity_top_k)
    # Persist prompt templates via the on-disk files the config reads.
    if body.text_qa_template is not None:
        path = getattr(cfg, "text_qa_template_path", None)
        if path:
            _write_template(Path(path), body.text_qa_template)
    if body.refine_template is not None:
        path = getattr(cfg, "refine_template_path", None)
        if path:
            _write_template(Path(path), body.refine_template)


def _write_template(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write prompt template to %s: %s", path, exc)


def _summary_for_dataset(qa: Dict[str, Any], name: str, source: str) -> DatasetSummary:
    documents = qa.get("documents", []) or []
    test_q = qa.get("test_data", {}).get("question", []) or []
    return DatasetSummary(
        name=name,
        display=DATASET_DISPLAY_MAP.get(name, name),
        source=source,
        num_documents=len(documents),
        num_test_questions=len(test_q),
    )


def _reset_engines() -> None:
    STATE.query_engine = None
    STATE.open_rag_engine = None
    STATE.orchestrator_engine = None
    STATE.orchestrator = "default"


# ---------------------------------------------------------------------------
# FastAPI app and routes.
# ---------------------------------------------------------------------------
app = FastAPI(title="XRAG WebUI", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    cfg = Config()
    logger.info("XRAG WebUI server started with dataset=%s", getattr(cfg, "dataset", None))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "engine_status": "initialized"}


@app.get("/api/options")
def get_options() -> Dict[str, Any]:
    return {
        "llms": LLM_OPTIONS,
        "hf_models": HF_MODEL_OPTIONS,
        "embeddings": EMBEDDING_OPTIONS,
        "split_types": SPLIT_TYPE_OPTIONS,
        "preset_datasets": [
            {"id": k, "label": v, "info": DATASETS_INFO.get(v, {})}
            for k, v in DATASET_DISPLAY_MAP.items()
        ],
        "retrievers": RETRIEVER_OPTIONS,
        "retriever_modes": RETRIEVER_MODE_OPTIONS,
        "postprocess_rerank": POSTPROCESS_RERANK_OPTIONS,
        "query_transform": QUERY_TRANSFORM_OPTIONS,
        "orchestrators": [
            {"id": k, "label": ORCHESTRATOR_DISPLAY[k]} for k in ORCHESTRATOR_OPTIONS
        ],
        "response_synthesizers": RESPONSE_SYNTHESIZER_OPTIONS,
        "metric_groups": METRIC_GROUPS,
        "default_templates": {
            "text_qa_template": DEFAULT_PROMPT_QA,
            "refine_template": DEFAULT_PROMPT_REFINE,
        },
    }


@app.get("/api/capabilities")
def get_capabilities() -> Dict[str, Any]:
    """Describe supported components and the parameters that affect a run."""
    return {
        "version": 1,
        "options": get_options(),
        "fields": {
            "index": ["embeddings", "split_type", "chunk_size", "chunk_overlap", "window_size", "chunk_sizes", "persist_dir"],
            "retrieval": ["retriever", "similarity_top_k", "query_transform", "postprocess_rerank", "response_synthesizer"],
            "evaluation": ["metrics", "num_samples", "experiment_1"],
        },
        "orchestrator_support": {
            "default": "native", "self": "native", "adaptive": "native",
            "sim": "native", "open": "native",
        },
    }


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    cfg = Config()
    return _cfg_to_dict(cfg)


@app.post("/api/config/llm")
def update_llm(body: LLMUpdate) -> Dict[str, Any]:
    cfg = Config()
    _apply_llm_update(cfg, body)
    _reset_engines()
    return {"ok": True, "config": _cfg_to_dict(cfg)["llm"]}


@app.post("/api/config/vector")
def update_vector(body: VectorDBUpdate) -> Dict[str, Any]:
    cfg = Config()
    _apply_vector_update(cfg, body)
    STATE.index = None
    STATE.hierarchical_storage_context = None
    _reset_engines()
    config = _cfg_to_dict(cfg)
    return {"ok": True, "config": {**config["embedding"], **config["chunk"]}}


@app.post("/api/config/retrieval")
def update_retrieval(body: RetrievalUpdate) -> Dict[str, Any]:
    cfg = Config()
    _apply_retrieval_update(cfg, body)
    _reset_engines()
    return {"ok": True, "config": _cfg_to_dict(cfg)["retrieval"]}


# ---------------------------------------------------------------------------
# Dataset endpoints.
# ---------------------------------------------------------------------------
@app.post("/api/dataset/preset")
def load_preset_dataset(body: PresetDatasetRequest) -> Dict[str, Any]:
    name = body.name
    if name not in PRESET_DATASET_OPTIONS:
        raise HTTPException(400, f"Unknown preset dataset: {name}")
    cfg = Config()
    # The loader pulls from HuggingFace; this can take a while.
    from xrag.data.qa_loader import get_qa_dataset
    try:
        qa = get_qa_dataset(name)
    except Exception as exc:
        logger.exception("Failed to load preset dataset")
        raise HTTPException(500, f"Failed to load dataset: {exc}") from exc
    cfg.dataset = name
    STATE.qa_dataset = qa
    STATE.dataset = _summary_for_dataset(qa, name, "preset")
    # Reset downstream artifacts.
    STATE.index = None
    STATE.hierarchical_storage_context = None
    _reset_engines()
    return {"ok": True, "dataset": STATE.dataset.__dict__}


@app.post("/api/dataset/upload-json")
async def upload_json_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept a JSON file uploaded via multipart form."""
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(400, "Only .json files are accepted")
    cfg = Config()
    # Materialise the upload to a temp file because get_qa_dataset expects a path
    # (or a streamlit file-like; we use a path for portability).
    upload_dir = Path.cwd() / "webui_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"upload-{uuid.uuid4().hex}.json"
    with target.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    from xrag.data.qa_loader import get_qa_dataset
    try:
        qa = get_qa_dataset("custom", str(target))
    except Exception as exc:
        logger.exception("Failed to parse uploaded JSON dataset")
        raise HTTPException(400, f"Invalid dataset JSON: {exc}") from exc
    cfg.dataset = "custom"
    cfg.dataset_path = str(target)
    STATE.qa_dataset = qa
    STATE.dataset = _summary_for_dataset(qa, "custom", "json")
    STATE.index = None
    STATE.hierarchical_storage_context = None
    _reset_engines()
    return {"ok": True, "dataset": STATE.dataset.__dict__}


@app.post("/api/dataset/from-folder")
def generate_from_folder(body: FolderDatasetRequest) -> Dict[str, Any]:
    """Read documents from a folder and ask the LLM to produce QA pairs."""
    cfg = Config()
    from xrag.data.qa_loader import generate_qa_from_folder, get_qa_dataset
    output_path = Path(body.output_json).expanduser().resolve()
    try:
        qa_pairs = generate_qa_from_folder(
            body.folder_path,
            str(output_path),
            body.num_questions,
            sentence_length=body.sentence_length,
        )
    except Exception as exc:
        logger.exception("Failed to generate QA pairs from folder")
        raise HTTPException(500, f"Failed to generate QA pairs: {exc}") from exc
    qa = get_qa_dataset("custom", str(output_path))
    cfg.dataset = "custom"
    cfg.dataset_path = str(output_path)
    STATE.qa_dataset = qa
    STATE.dataset = _summary_for_dataset(qa, "custom", "folder")
    STATE.index = None
    STATE.hierarchical_storage_context = None
    _reset_engines()
    return {
        "ok": True,
        "dataset": STATE.dataset.__dict__,
        "num_generated_qa_pairs": len(qa_pairs),
    }


@app.get("/api/dataset/current")
def current_dataset() -> Dict[str, Any]:
    return {
        "dataset": STATE.dataset.__dict__ if STATE.dataset else None,
        "has_documents": STATE.qa_dataset is not None,
    }


# ---------------------------------------------------------------------------
# Index / query engine endpoints.
# ---------------------------------------------------------------------------
@app.post("/api/index/build")
def build_index_endpoint() -> Dict[str, Any]:
    if STATE.qa_dataset is None:
        raise HTTPException(400, "No dataset loaded. Pick or upload a dataset first.")
    cfg = Config()
    from xrag.launcher import build_index as _build_index
    try:
        index, hierarchical_storage_context = _build_index(STATE.qa_dataset["documents"])
    except Exception as exc:
        logger.exception("build_index failed")
        raise HTTPException(500, f"build_index failed: {exc}") from exc
    STATE.index = index
    STATE.hierarchical_storage_context = hierarchical_storage_context
    _reset_engines()
    persist = cfg.persist_dir + "-" + cfg.dataset + "-" + cfg.embeddings + "-" + cfg.split_type + "-" + str(cfg.chunk_size)
    return {"ok": True, "persist_dir": persist}


@app.post("/api/query-engine/build")
def build_query_engine_endpoint() -> Dict[str, Any]:
    if STATE.index is None:
        raise HTTPException(400, "No index built. Call /api/index/build first.")
    cfg = Config()
    orchestrator = _selected_orchestrator(cfg)
    try:
        from xrag.retrievers.retriever import get_retriver
        retriever = get_retriver(
            cfg.retriever, STATE.index,
            hierarchical_storage_context=STATE.hierarchical_storage_context, cfg=cfg,
        )
        if orchestrator == "default":
            from xrag.launcher import build_query_engine as _build_query_engine
            STATE.query_engine = _build_query_engine(
                STATE.index, STATE.hierarchical_storage_context
            )
            STATE.orchestrator_engine = STATE.query_engine
        elif orchestrator == "open":
            from xrag.open_rag import OpenRAGPipeline
            STATE.orchestrator_engine = OpenRAGPipeline(cfg, external_retriever=retriever)
        elif orchestrator == "self":
            from xrag.self_rag import SelfRAGPipeline
            STATE.orchestrator_engine = SelfRAGPipeline(cfg, external_retriever=retriever)
        elif orchestrator == "adaptive":
            from xrag.adaptive_rag.engine import AdaptiveRAG
            STATE.orchestrator_engine = AdaptiveRAG(index=STATE.index, config=cfg)
        elif orchestrator == "sim":
            # Sim-RAG owns its generation pipeline but uses the configured external retriever.
            STATE.orchestrator_engine = {"retriever": retriever}
        STATE.open_rag_engine = STATE.orchestrator_engine if orchestrator == "open" else None
        STATE.query_engine = STATE.orchestrator_engine if orchestrator == "default" else None
        STATE.orchestrator = orchestrator
    except Exception as exc:
        logger.exception("build_query_engine failed")
        raise HTTPException(500, f"build_query_engine failed: {exc}") from exc
    return {"ok": True, "engine": orchestrator}


# ---------------------------------------------------------------------------
# Single-question inference endpoint (used as a smoke test before evaluation).
# ---------------------------------------------------------------------------
def _nodes_to_documents(response_obj: Any) -> List[Dict[str, Any]]:
    documents = []
    for index, scored_node in enumerate(getattr(response_obj, "source_nodes", []) or []):
        node = getattr(scored_node, "node", scored_node)
        metadata = getattr(node, "metadata", {}) or {}
        documents.append({
            "id": str(metadata.get("id", getattr(node, "node_id", index))),
            "title": metadata.get("title", ""),
            "text": getattr(node, "get_content", lambda: str(node))(),
            "score": float(getattr(scored_node, "score", 0.0) or 0.0),
        })
    return documents


def _run_query(question: str) -> Dict[str, Any]:
    """Normalize all XRAG orchestrators to one response contract."""
    cfg = Config()
    if STATE.orchestrator == "default":
        from xrag.process.query_transform import transform_and_query
        response_obj = transform_and_query(question, cfg, STATE.query_engine)
        return {"answer": response_obj.response, "documents": _nodes_to_documents(response_obj), "response_obj": response_obj}
    if STATE.orchestrator == "sim":
        from xrag.sim_rag import run_simrag
        result = run_simrag(question, cfg=cfg, retriever=STATE.orchestrator_engine["retriever"])
        documents = [
            {"id": str(doc_id), "text": text, "score": 0.0}
            for doc_id, text in zip(result.get("retrieved_ids", []), result.get("retrieved_texts", []))
        ]
        return {"answer": result.get("response", ""), "documents": documents, "response_obj": None}

    result = STATE.orchestrator_engine.query(question)
    response_obj = result.get("response_obj") if isinstance(result, dict) else None
    documents = result.get("retrieved_documents", []) if isinstance(result, dict) else []
    if not documents and response_obj is not None:
        documents = _nodes_to_documents(response_obj)
    return {
        "answer": result.get("response", "") if isinstance(result, dict) else str(result),
        "documents": documents,
        "response_obj": response_obj,
    }


@app.post("/api/query")
def query(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = (payload or {}).get("question", "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    if STATE.orchestrator_engine is None:
        raise HTTPException(400, "No query engine built. Build the query engine first.")
    try:
        result = _run_query(question)
        return {"answer": result["answer"], "retrieved_documents": result["documents"], "orchestrator": STATE.orchestrator}
    except Exception as exc:
        logger.exception("query failed")
        raise HTTPException(500, f"query failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------
def _snapshot_evaluation(task: EvalTask) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "total": task.total,
        "completed": task.completed,
        "failed": task.failed,
        "samples": task.samples[-50:],  # last 50 to avoid huge payloads
        "summary": task.summary,
        "error": task.error,
    }


def _set_experiment_status(task: EvalTask, status: str, **updates: Any) -> None:
    experiment = STATE.experiments.get(task.experiment_id or "")
    if experiment is not None:
        experiment["status"] = status
        experiment.update(updates)


def _run_evaluation(task: EvalTask, metrics: List[str], num_samples: int) -> None:
    try:
        _run_evaluation_inner(task, metrics, num_samples)
    except Exception as exc:
        logger.exception("Evaluation task failed")
        task.status = "error"
        task.error = str(exc)
        task.finished_at = time.time()
        _set_experiment_status(task, "error", error=task.error)
        task.queue.put({"event": "error", "error": task.error})


def _run_evaluation_inner(task: EvalTask, metrics: List[str], num_samples: int) -> None:
    cfg = Config()
    from xrag.eval.EvalModelAgent import EvalModelAgent
    from xrag.eval.evaluate_rag import EvaluationResult, evaluating
    from llama_index.core.schema import NodeWithScore, TextNode

    if STATE.qa_dataset is None:
        task.status = "error"
        task.error = "No dataset loaded"
        _set_experiment_status(task, "error", error=task.error)
        task.queue.put({"event": "error", "error": task.error})
        task.finished_at = time.time()
        return

    eval_agent = EvalModelAgent(cfg)
    results = EvaluationResult(metrics=metrics)

    qa = STATE.qa_dataset
    questions = qa["test_data"]["question"][:num_samples]
    expected_answers = qa["test_data"]["expected_answer"][:num_samples]
    golden_contexts = qa["test_data"]["golden_context"][:num_samples]
    golden_ids = qa["test_data"]["golden_context_ids"][:num_samples]
    task.total = len(questions)
    task.status = "running"
    task.queue.put({"event": "started", "total": task.total})

    class _EvaluationResponseAdapter:
        def __init__(self, answer, docs):
            self.response = answer
            self.source_nodes = [
                NodeWithScore(
                    node=TextNode(text=d.get("text", ""),
                                  metadata={"id": d.get("id", str(i)),
                                            "title": d.get("title", "")}),
                    score=float(d.get("score", 0.0) or 0.0),
                )
                for i, d in enumerate(docs)
            ]

    for idx, (question, expected, gold_ctx, gold_ids_row) in enumerate(
        zip(questions, expected_answers, golden_contexts, golden_ids)
    ):
        if task.cancel_flag.is_set():
            task.status = "cancelled"
            _set_experiment_status(task, "cancelled")
            task.queue.put({"event": "cancelled"})
            task.finished_at = time.time()
            return
        try:
            query_result = _run_query(question)
            actual = query_result["answer"]
            retrieved = query_result["documents"]
            response_obj = query_result["response_obj"] or _EvaluationResponseAdapter(actual, retrieved)
            retrieval_ids = [document.get("id", str(i)) for i, document in enumerate(retrieved)]
            retrieval_context = [document.get("text", "") for document in retrieved]

            eval_result = evaluating(
                question, response_obj, actual, retrieval_context, retrieval_ids,
                expected, gold_ctx, gold_ids_row, results.metrics, eval_agent,
            )
            results.add(eval_result)
            sample = {
                "index": idx,
                "question": question,
                "expected_answer": expected,
                "actual_response": actual,
                "retrieval_context": retrieval_context,
                "metrics": {
                    k: results.metrics_results[k]["score"] / results.metrics_results[k]["count"]
                    if results.metrics_results[k]["count"] > 0 else 0.0
                    for k in results.metrics_results
                    if k in metrics and not k.endswith("_rev")
                },
            }
            task.samples.append(sample)
            task.completed = idx + 1
            task.progress = task.completed / max(1, task.total)
            task.summary = _summary_from_results(results)
            task.queue.put({
                "event": "progress",
                "completed": task.completed,
                "total": task.total,
                "progress": task.progress,
                "sample": sample,
                "summary": task.summary,
            })
        except Exception as exc:
            logger.exception("Sample %d failed", idx)
            task.failed += 1
            task.completed = idx + 1
            task.progress = task.completed / max(1, task.total)
            task.queue.put({"event": "sample_error", "index": idx, "error": str(exc)})

    task.status = "done"
    task.summary = _summary_from_results(results)
    STATE.last_evaluation = task.summary
    _set_experiment_status(task, "done", result=task.summary, failed=task.failed)
    task.queue.put({"event": "done", "summary": task.summary})
    task.finished_at = time.time()


def _summary_from_results(results) -> Dict[str, Any]:
    summary = {
        "n": results.results["n"],
        "global": {},
        "metrics": {},
    }
    for key, value in results.results.items():
        if key in results.metrics:
            summary["global"][key] = value / max(1, results.results["n"])
    for key, value in results.metrics_results.items():
        if key in results.metrics and not key.endswith("_rev"):
            if value["count"] == 0:
                summary["metrics"][key] = {"score": 0.0, "valid_count": 0}
            else:
                summary["metrics"][key] = {
                    "score": value["score"] / value["count"],
                    "valid_count": value["count"],
                }
    return summary


@app.post("/api/evaluate/start")
def start_evaluation(body: EvaluationRequest) -> Dict[str, Any]:
    if STATE.qa_dataset is None or STATE.orchestrator_engine is None:
        raise HTTPException(400, "Build the index and query engine first.")
    if not body.metrics:
        raise HTTPException(400, "Choose at least one metric.")
    unknown_metrics = sorted(set(body.metrics) - set(ALL_METRIC_IDS))
    if unknown_metrics:
        raise HTTPException(400, f"Unsupported metrics: {', '.join(unknown_metrics)}")
    cfg = Config()
    cfg.metrics = list(body.metrics)
    if body.experiment_1 is not None:
        cfg.experiment_1 = body.experiment_1

    effective_samples = (
        int(getattr(cfg, "test_init_total_number_documents", body.num_samples))
        if cfg.experiment_1 else int(body.num_samples)
    )
    experiment_id = uuid.uuid4().hex
    task = EvalTask(task_id=uuid.uuid4().hex, experiment_id=experiment_id)
    STATE.experiments[experiment_id] = {
        "experiment_id": experiment_id,
        "created_at": time.time(),
        "status": "running",
        "dataset": STATE.dataset.__dict__ if STATE.dataset else None,
        "config": json.loads(json.dumps(_cfg_to_dict(cfg))),
        "orchestrator": STATE.orchestrator,
        "metrics": list(body.metrics),
        "num_samples": effective_samples,
        "task_id": task.task_id,
        "result": None,
    }
    STATE.tasks[task.task_id] = task
    thread = threading.Thread(
        target=_run_evaluation,
        args=(task, list(body.metrics), effective_samples),
        daemon=True,
        name=f"xrag-eval-{task.task_id}",
    )
    task.thread = thread
    thread.start()
    return {"ok": True, "experiment_id": experiment_id, "task_id": task.task_id, "snapshot": _snapshot_evaluation(task)}


@app.get("/api/experiments")
def list_experiments() -> Dict[str, Any]:
    experiments = sorted(STATE.experiments.values(), key=lambda item: item["created_at"], reverse=True)
    return {"experiments": experiments}


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> Dict[str, Any]:
    experiment = STATE.experiments.get(experiment_id)
    if experiment is None:
        raise HTTPException(404, "Unknown experiment id")
    return experiment


@app.get("/api/evaluate/{task_id}")
def get_evaluation(task_id: str) -> Dict[str, Any]:
    task = STATE.tasks.get(task_id)
    if task is None:
        raise HTTPException(404, "Unknown task id")
    return _snapshot_evaluation(task)


@app.post("/api/evaluate/{task_id}/cancel")
def cancel_evaluation(task_id: str) -> Dict[str, Any]:
    task = STATE.tasks.get(task_id)
    if task is None:
        raise HTTPException(404, "Unknown task id")
    task.cancel_flag.set()
    return {"ok": True}


@app.get("/api/evaluate/{task_id}/stream")
def stream_evaluation(task_id: str):
    """Server-Sent Events stream of evaluation progress."""
    from fastapi.responses import StreamingResponse
    task = STATE.tasks.get(task_id)
    if task is None:
        raise HTTPException(404, "Unknown task id")

    def event_stream():
        # Drain the queue. Each event becomes a single SSE message.
        while True:
            try:
                event = task.queue.get(timeout=15)
            except queue.Empty:
                # heartbeat to keep the connection alive
                yield "event: ping\ndata: {}\n\n"
                if task.status in ("done", "error", "cancelled"):
                    return
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("event") in ("done", "error", "cancelled"):
                return

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Static frontend.
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index() -> Any:
    index_html = STATIC_DIR / "index.html"
    if not index_html.exists():
        return JSONResponse({"error": "Frontend not built"}, status_code=500)
    return FileResponse(index_html)


@app.get("/favicon.ico")
def favicon() -> Any:
    fav = STATIC_DIR / "favicon.ico"
    if fav.exists():
        return FileResponse(fav)
    return Response(status_code=204)
