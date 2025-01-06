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
    embeddings = get_embedding(cfg.embeddings)

    Settings.chunk_size = cfg.chunk_size
    Settings.llm = llm
    Settings.embed_model = embeddings
    # pip install llama-index-embeddings-langchain

    cfg.persist_dir = cfg.persist_dir + '-' + cfg.dataset + '-' + cfg.embeddings + '-' + cfg.split_type + '-' + str(
        cfg.chunk_size)

    index, hierarchical_storage_context = get_index(documents, cfg.persist_dir, split_type=cfg.split_type,
                                                    chunk_size=cfg.chunk_size)


    return index, hierarchical_storage_context

def build_query_engine(index, hierarchical_storage_context):

    cfg = Config()
    query_engine = RetrieverQueryEngine(
        retriever=get_retriver(cfg.retriever, index, hierarchical_storage_context=hierarchical_storage_context),
        response_synthesizer=response_synthesizer(0),
        node_postprocessors=[get_postprocessor(cfg)]
    )

    text_qa_template_str = cfg.text_qa_template_str
    text_qa_template = PromptTemplate(text_qa_template_str)

    refine_template_str = cfg.refine_template_str
    refine_template = PromptTemplate(refine_template_str)

    query_engine.update_prompts({"response_synthesizer:text_qa_template": text_qa_template,
                                 "response_synthesizer:refine_template": refine_template})
    query_engine = query_expansion([query_engine], query_number=4, similarity_top_k=10)
    query_engine = RetrieverQueryEngine.from_args(query_engine)

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
def run(cli=True):

    seed_everything(42)
    cfg = Config()
    qa_dataset = get_qa_dataset(cfg.dataset)
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
