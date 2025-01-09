from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from ..launcher.launch import build_index, build_query_engine
from ..config import Config
from ..process.query_transform import transform_and_query
from ..data.qa_loader import get_qa_dataset, get_dataset

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
    global query_engine, config
    config = Config()
    
    # 如果提供了 json_path，设置为自定义数据集
    if json_path:
        config.dataset = 'custom'
        config.dataset_path = json_path
    # 如果提供了 dataset_folder，设置为自定义数据集文件夹
    elif dataset_folder:
        config.dataset = 'folder'
        config.dataset_path = dataset_folder
        
    # 获取数据集并构建索引
    if config.dataset == 'custom':
        documents = get_qa_dataset(config.dataset, config.dataset_path)['documents']
    elif config.dataset == 'folder':
        documents = get_dataset(config.dataset_path)
    else:
        documents = get_qa_dataset(config.dataset)['documents']
    index, hierarchical_storage_context = build_index(documents)
    # 构建查询引擎
    query_engine = build_query_engine(index, hierarchical_storage_context)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not query_engine:
        raise HTTPException(status_code=500, detail="Query engine not initialized")
    
    try:
        # 使用 transform_and_query 处理查询
        response = transform_and_query(request.query, config, query_engine)
        
        # 构造返回结果
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_status": "initialized" if query_engine else "not_initialized"}

def run_api_server(host: str = "0.0.0.0", port: int = 8000, json_path: str = '', dataset_folder: str = ''):
    app_instance = init_app(json_path, dataset_folder)
    uvicorn.run(app_instance, host=host, port=port) 