import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings, PromptTemplate
from ..llms import get_llm
from ..index import get_index
from ..eval.evaluate_rag import evaluating
from ..embs.embedding import get_embedding
from ..data.qa_loader import get_qa_dataset
from ..config import Config
from ..retrievers.retriever import get_retriver, query_expansion, response_synthesizer
import warnings
from ..eval.evaluate_rag import EvaluationResult
from ..eval.EvalModelAgent import EvalModelAgent
from ..process.postprocess_rerank import get_postprocessor
from ..process.query_transform import transform_and_query
import random
import numpy as np
import torch
from ..sim_rag import run_simrag
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def build_index(documents):
    cfg = Config()
    llm = get_llm(cfg.llm)
    # Create and dl embeddings instance
    embeddings = get_embedding(cfg.embeddings,cfg.embed_batch_size)

    Settings.chunk_size = cfg.chunk_size
    Settings.llm = llm
    Settings.embed_model = embeddings
    # pip install llama-index-embeddings-langchain

    cfg.persist_dir = cfg.persist_dir + '-' + cfg.dataset + '-' + cfg.embeddings + '-' + cfg.split_type + '-' + str(
        cfg.chunk_size)

    semantic_setting={
        "buffer_size":cfg.buffer_size,
        "include_metadata":cfg.include_metadata,
        "include_prev_next_rel":cfg.include_prev_next_rel,
        "breakpoint_percentile_threshold":cfg.breakpoint_percentile_threshold
    }
    index, hierarchical_storage_context = get_index(documents, cfg.persist_dir, split_type=cfg.split_type,
                                                    chunk_size=cfg.chunk_size,chunk_overlap=cfg.chunk_overlap,
                                                    chunk_sizes=cfg.chunk_sizes,semantic_setting=semantic_setting, window_size=cfg.window_size)


    return index, hierarchical_storage_context

def build_query_engine(index, hierarchical_storage_context, use_async=False):
    cfg = Config()
    query_engine = RetrieverQueryEngine(
        retriever=get_retriver(cfg.retriever, index, hierarchical_storage_context=hierarchical_storage_context,cfg=cfg),
        response_synthesizer=response_synthesizer(cfg.responce_synthsizer),
        node_postprocessors=[get_postprocessor(cfg)]
    )

    text_qa_template_str = cfg.text_qa_template_str
    text_qa_template = PromptTemplate(text_qa_template_str)

    refine_template_str = cfg.refine_template_str
    refine_template = PromptTemplate(refine_template_str)

    query_engine.update_prompts({"response_synthesizer:text_qa_template": text_qa_template,
                                "response_synthesizer:refine_template": refine_template})
    # query_engine = query_expansion([query_engine], query_number=4, similarity_top_k=10)
    query_engine = RetrieverQueryEngine.from_args(query_engine, use_async=use_async)

    return query_engine


def eval_cli(qa_dataset, query_engine):
    cfg = Config()
    true_num = 0
    all_num = 0
    evaluateResults = EvaluationResult(metrics=cfg.metrics)
    evalAgent = EvalModelAgent(cfg)
    if cfg.experiment_1:
        if len(qa_dataset) < cfg.test_init_total_number_documents:
            warnings.filterwarnings('default')
            warnings.warn("使用的数据集长度大于数据集本身的最大长度，请修改。 本轮代码无法运行", UserWarning)
    else:
        cfg.test_init_total_number_documents = cfg.n
    for question, expected_answer, golden_context, golden_context_ids in zip(
            qa_dataset['test_data']['question'][:cfg.test_init_total_number_documents],
            qa_dataset['test_data']['expected_answer'][:cfg.test_init_total_number_documents],
            qa_dataset['test_data']['golden_context'][:cfg.test_init_total_number_documents],
            qa_dataset['test_data']['golden_context_ids'][:cfg.test_init_total_number_documents]
    ):
        response = transform_and_query(question, cfg, query_engine)
        # 返回node节点
        retrieval_ids = []
        retrieval_context = []
        for source_node in response.source_nodes:
            retrieval_ids.append(source_node.metadata['id'])
            retrieval_context.append(source_node.get_content())
        actual_response = response.response
        eval_result = evaluating(question, response, actual_response, retrieval_context, retrieval_ids,
                                 expected_answer, golden_context, golden_context_ids, evaluateResults.metrics,
                                 evalAgent)
        evaluateResults.add(eval_result)
        all_num = all_num + 1
        evaluateResults.print_results()
        print("总数：" + str(all_num))
    return evaluateResults
def run(cli=True, custom_dataset=None):

    seed_everything(42)
    cfg = Config()
    if cfg.dataset_type == 'local':
        print('Using local dataset')
        qa_dataset = get_qa_dataset(cfg.dataset, cfg.dataset_path)
    else:
        print('Using huggingface dataset')
        qa_dataset = get_qa_dataset(cfg.dataset, cfg.dataset_path)
    # If Sim-RAG is enabled (nested section), use its inference pipeline instead of the standard RetrieverQueryEngine
    sim_rag_cfg = cfg.config.get('sim_rag', {}) if hasattr(cfg, 'config') else {}
    if bool(sim_rag_cfg.get('enabled', False)):
        evaluateResults = EvaluationResult(metrics=cfg.metrics)
        evalAgent = EvalModelAgent(cfg)
        # Build embeddings/index once to speed up retrieval within Sim-RAG
        index, hierarchical_storage_context = build_index(qa_dataset['documents'])
        retriever = get_retriver(cfg.retriever, index, hierarchical_storage_context=hierarchical_storage_context, cfg=cfg)
        # Iterate questions and call Sim-RAG
        limit = cfg.test_init_total_number_documents if cfg.experiment_1 else cfg.n
        for question, expected_answer, golden_context, golden_context_ids in zip(
                qa_dataset['test_data']['question'][:limit],
                qa_dataset['test_data']['expected_answer'][:limit],
                qa_dataset['test_data']['golden_context'][:limit],
                qa_dataset['test_data']['golden_context_ids'][:limit]
        ):
            sim_out = run_simrag(question, retriever=retriever, cfg=cfg)
            actual_response = sim_out["response"]
            retrieval_context = sim_out["retrieved_texts"]
            retrieval_ids = sim_out["retrieved_ids"]
            # Build a lightweight response-like object to satisfy evaluators that expect .response and .source_nodes
            class _Resp:
                def __init__(self, text, texts):
                    self.response = text
                    from llama_index.core.schema import TextNode, NodeWithScore
                    self.source_nodes = [NodeWithScore(node=TextNode(text=t)) for t in texts]
            response = _Resp(actual_response, retrieval_context)
            eval_result = evaluating(question, response, actual_response, retrieval_context, retrieval_ids,
                                     expected_answer, golden_context, golden_context_ids, evaluateResults.metrics,
                                     evalAgent)
            evaluateResults.add(eval_result)
            evaluateResults.print_results()
        return evaluateResults
    else:
        index, hierarchical_storage_context = build_index(qa_dataset['documents'])
        query_engine = build_query_engine(index, hierarchical_storage_context)
        if cli:
            evaluateResults = eval_cli(qa_dataset, query_engine)
            return evaluateResults
        else:
            return query_engine, qa_dataset





if __name__ == '__main__':
    run()
    print('Success')
