# pip install -U sentence-transformers

import logging
import os
import sys

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor, LongContextReorder, \
    SentenceEmbeddingOptimizer, SentenceTransformerRerank, LLMRerank
from llama_index.core.data_structs import Node
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.legacy.postprocessor import CohereRerank, RankGPTRerank
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex, Settings, SummaryIndex, TreeIndex, QueryBundle, StorageContext,
)

# from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def bm25_retriever(index):
    retriever_bm25 = BM25Retriever.from_defaults(index=index, similarity_top_k=3)
    return retriever_bm25


def similarity_postprocessor(similarity_cutoff=0.7):
    return SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)


def keyword_node_postprocessor(required_keywords=None, exclude_keywords=None):
    if exclude_keywords is None:
        exclude_keywords = ["word3", "word4"]
    if required_keywords is None:
        required_keywords = ["word1", "word2"]
    return KeywordNodePostprocessor(required_keywords=required_keywords, exclude_keywords=exclude_keywords)


def long_context_reorder():
    return LongContextReorder()


def sentence_embedding_optimizer(percentile_cutoff=0.5):
    return SentenceEmbeddingOptimizer(
        embed_model=Settings.embed_model,
        percentile_cutoff=percentile_cutoff,
        # threshold_cutoff=0.7
    )


def llm_rerank():
    postprocessor = LLMRerank(top_n=10, service_context=None)
    return postprocessor


def transformer_rerank(top_n=3):
    return SentenceTransformerRerank(
        model="cross-encoder/stsb-distilroberta-base", top_n=top_n
    )
    # pip install -U sentence-transformers


def gpt_rerank():
    return RankGPTRerank(top_n=3, llm=Settings.llm)


if __name__ == '__main__':
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    Settings.llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")
    # 需要一个直接放文件的本地目录
    documents = SimpleDirectoryReader("D:\RAG_benchmark\doc").load_data()
    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=20,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    index_ = VectorStoreIndex(nodes)
    retriever = bm25_retriever(index_)  # 可用

    nodes = retriever.retrieve("请用中文回答我的毕业设计题目是什么")
    print(nodes)

    processor = transformer_rerank()
    filtered_nodes = processor.postprocess_nodes(nodes, query_str="请用中文回答我的毕业设计题目是什么")
    print(filtered_nodes)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[processor]
    )
    response = query_engine.query("请用中文回答我的毕业设计题目是什么")
    print(response)
