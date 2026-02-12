from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from ..launcher.launch import build_index, build_query_engine
from ..config import Config
from ..process.query_transform import transform_and_query_async
from ..data.qa_loader import get_qa_dataset, get_dataset
from ..retrievers.retriever import get_retriver
from ..open_rag import OpenRAGPipeline

app = FastAPI(
    title="XRAG API",
    description="RAG (Retrieval-Augmented Generation) API Service",
    version="0.1.0"
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    
class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

query_engine = None
rag_engine = None
engine_type = "default"   # "default" | "open_rag"
config = None
json_path = ''
dataset_folder = ''

def init_app(_json_path: str = '', _dataset_folder: str = ''):
    global json_path, dataset_folder
    json_path = _json_path
    dataset_folder = _dataset_folder
    return app

@app.on_event("startup")
async def startup_event():
    global query_engine, rag_engine, engine_type, config
    config = Config()

    if json_path:
        config.dataset = 'custom'
        config.dataset_path = json_path
    elif dataset_folder:
        config.dataset = 'folder'
        config.dataset_path = dataset_folder

    if config.dataset == 'custom':
        documents = get_qa_dataset(config.dataset, config.dataset_path)['documents']
    elif config.dataset == 'folder':
        documents = get_dataset(config.dataset_path)
    else:
        documents = get_qa_dataset(config.dataset)['documents']

    index, hierarchical_storage_context = build_index(documents)

    if config.config.get("open_rag", {}).get("enabled", False):
        retriever = get_retriver(
            config.retriever,
            index,
            hierarchical_storage_context=hierarchical_storage_context,
            cfg=config
        )
        rag_engine = OpenRAGPipeline(config, external_retriever=retriever)
        query_engine = None
        engine_type = "open_rag"
    else:
        query_engine = build_query_engine(index, hierarchical_storage_context, use_async=True)
        rag_engine = None
        engine_type = "default"

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        if engine_type == "open_rag":
            if rag_engine is None:
                raise HTTPException(status_code=500, detail="Open-RAG engine not initialized")

            out = rag_engine.query(request.query)
            retrieved_docs = out.get("retrieved_documents", []) or []
            sources = []
            for d in retrieved_docs:
                sources.append({
                    "content": d.get("text", ""),
                    "id": d.get("id", ""),
                    "score": d.get("score", None)
                })

            return QueryResponse(
                answer=out.get("response", "") or out.get("raw_response", ""),
                sources=sources
            )

        # default path
        if not query_engine:
            raise HTTPException(status_code=500, detail="Query engine not initialized")

        response = await transform_and_query_async(request.query, config, query_engine)
        sources = []
        for source_node in response.source_nodes:
            sources.append({
                "content": source_node.get_content(),
                "id": source_node.metadata.get("id", ""),
                "score": source_node.score if hasattr(source_node, "score") else None
            })

        return QueryResponse(
            answer=response.response,
            sources=sources
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    initialized = (rag_engine is not None) if engine_type == "open_rag" else (query_engine is not None)
    return {
        "status": "healthy",
        "engine_type": engine_type,
        "engine_status": "initialized" if initialized else "not_initialized"
    }

def run_api_server(host: str = "0.0.0.0", port: int = 8000, json_path: str = '', dataset_folder: str = ''):
    app_instance = init_app(json_path, dataset_folder)
    uvicorn.run(app_instance, host=host, port=port) 