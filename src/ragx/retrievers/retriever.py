# pip install llama-index-retrievers-bm25

import os
import warnings
from typing import List, Optional

import openai
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.document_summary import DocumentSummaryIndexEmbeddingRetriever, \
    DocumentSummaryIndexLLMRetriever
from llama_index.core.indices.keyword_table.retrievers import BaseKeywordTableRetriever, KeywordTableGPTRetriever
from llama_index.core.indices.list import SummaryIndexEmbeddingRetriever, SummaryIndexRetriever
from llama_index.core.indices.tree import TreeAllLeafRetriever, TreeSelectLeafRetriever, \
    TreeSelectLeafEmbeddingRetriever, TreeRootRetriever
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SimilarityPostprocessor, LongContextReorder
from llama_index.core.query_engine import RetrieverQueryEngine

import logging
import sys

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex, Settings, SummaryIndex, TreeIndex, QueryBundle, StorageContext, DocumentSummaryIndex,
)
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, IndexNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.legacy.indices.keyword_table import KeywordTableSimpleRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever, AutoMergingRetriever, \
    RecursiveRetriever
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.llms.openai import OpenAI

from llama_index.core import get_response_synthesizer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# todo 常规检索模式: bm25检索 向量检索
def bm25_retriever(index):
    retriever_bm25 = BM25Retriever.from_defaults(index=index, similarity_top_k=3)
    return retriever_bm25


def vector_retriever(index):
    # index = VectorStoreIndex(node)
    retriever_vector = VectorIndexRetriever(index=index, similarity_top_k=3, show_progress=True,
                                            store_nodes_override=True)
    return retriever_vector


# todo 特定于某种索引的检索器类:汇总索引检索 树索引检索 关键字表索引检索 文档摘要索引检索
# note index必须为汇总索引 https://docs.llamaindex.ai/en/stable/api_reference/indices/list.html#llama_index.core.indices.list.SummaryIndex
def summary_retriever(summary_index, mode:int = 0):
    if mode > 2 or mode < 0:
        raise ValueError("Invalid mode for summary retriever."+ str(mode))
    if mode == 0:
        # 最简单的文档汇总索引
        retriever_summary = SummaryIndexRetriever(index=summary_index)
    elif mode == 1:
        # 基于编码的文档汇总索引
        retriever_summary = SummaryIndexEmbeddingRetriever(index=summary_index, embed_model=Settings.embed_model,
                                                           similarity_top_k=3)
    elif mode == 2:
        # 基于大模型的文档汇总索引
        retriever_summary = SummaryIndexEmbeddingRetriever(index=summary_index, llm=Settings.llm, similarity_top_k=3)

    return retriever_summary


# note index必须为树索引 https://docs.llamaindex.ai/en/latest/api_reference/indices/tree.html#llama_index.core.indices.tree.TreeIndex
def tree_retriever(index, mode=0):
    if mode > 3 or mode < 0:
        mode = 0

    # 只使用叶节点的树索引
    if mode == 1:
        retriever_tree = TreeAllLeafRetriever(index=index)
    # 使用部分叶节点
    elif mode == 2:
        retriever_tree = TreeSelectLeafRetriever(index=index)
    # 使用编码结合叶节点的检索器
    elif mode == 3:
        retriever_tree = TreeSelectLeafEmbeddingRetriever(index=index, embed_model=Settings.embed_model)
    # 使用根节点的检索器
    elif mode == 0:
        retriever_tree = TreeRootRetriever(index=index)

    return retriever_tree


# node index必须为关键字表索引 https://docs.llamaindex.ai/en/stable/api_reference/query/retrievers/table.html
def keyword_retriever(index, mode=0):
    if mode > 1 or mode < 0:
        raise ValueError("Invalid mode for keyword retriever."+ str(mode))
    # 基本关键字表检索器

    # GPT关键字表检索器

    retriever_keyword = KeywordTableGPTRetriever(index)
    return retriever_keyword


# 关于该检索器可能的bug：https://github.com/run-llama/llama_index/issues/7633
def document_summary_retrievers(index):
    retriever_d = DocumentSummaryIndexLLMRetriever(
        index,
        choice_batch_size=10,
        choice_top_k=1,
    )
    retriever_d = DocumentSummaryIndexEmbeddingRetriever(index)
    return retriever_d


# todo 高级检索模式：自定义检索器 融合检索 自动合并检索（扩增上下文检索） 元数据替换+句子窗口检索 递归检索
# 自定义检索器，定义了向量检索与bm25检索混合，混合模式为取合
class CustomRetriever(BaseRetriever):
    def __init__(
            self,
            vector_retriever_c: VectorIndexRetriever,
            bm25_retriever_c: BM25Retriever,
            keyword_retriever_c: KeywordTableSimpleRetriever,
            mode: str = "AND",
    ) -> None:
        self._vector_retriever = vector_retriever_c
        self._bm25_retriever = bm25_retriever_c
        self._keyword_retriever = keyword_retriever_c
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        bm25_ids = {n.node.node_id for n in bm25_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in bm25_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(bm25_ids)
        else:
            retrieve_ids = vector_ids.union(bm25_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


# 融合检索器 其将来自多个文档的索引作为输入，并自动进行问题扩充，以获得多次查询结果用于合并
# mode: 代表融合模式. 0 代表简单合并. 1: 采用RRF倒数排名融合
def query_fusion_retriever(index, num_queries=4, similarity_top_k=2, mode=0, retriever_weight=None):
    query_fusion_r = None
    if not isinstance(index, list):
        index = [index]
    if retriever_weight is None:
        retriever_weight = [1 / len(index)] * len(index)
    if mode == 0:
        query_fusion_r = QueryFusionRetriever(
            [index_s.as_retriever() for index_s in index],
            llm=Settings.llm,
            similarity_top_k=similarity_top_k,
            num_queries=num_queries,
            use_async=True,
            verbose=True,
            # query_gen_prompt="...",
            # 默认用于多轮询问的问题模板：
            # QUERY_GEN_PROMPT=(
            #     "You are a helpful assistant that generates multiple search queries based on a "
            #     "single input query. Generate {num_queries} search queries, one on each line, "
            #     "related to the following input query:\n"
            #     "Query: {query}\n"
            #     "Queries:\n"
            # )
        )
    elif mode == 1:
        query_fusion_r = QueryFusionRetriever(
            [index_s.as_retriever() for index_s in index],
            llm=Settings.llm,
            similarity_top_k=similarity_top_k,
            num_queries=num_queries,
            use_async=True,
            verbose=True,
            mode=FUSION_MODES.RECIPROCAL_RANK
        )

    return query_fusion_r


# 自动合并检索
# 其将自动合并子节点为高级节点，并进行扩增上下文操作 故需要提供节点类
# 也可使用特殊节点类进行初始化 https://docs.llamaindex.ai/en/latest/examples/retrievers/auto_merging_retriever.html
def auto_merging_retriever(index, hierarchical_storage_context):

    auto_merging_r = AutoMergingRetriever(index.as_retriever(similarity_top_k=6), storage_context=hierarchical_storage_context,
                                          verbose=True)
    return auto_merging_r


# 递归检索 + 句子节点引用rong
def recursive_retriever(base_nodes):
    sub_chunk_sizes = [128, 256, 512]
    sub_node_parsers = [
        SentenceSplitter(chunk_size=c, chunk_overlap=20) for c in sub_chunk_sizes
    ]

    all_nodes = []
    for base_node in base_nodes:
        for n in sub_node_parsers:
            sub_nodes = n.get_nodes_from_documents([base_node])
            sub_inodes = [
                IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
            ]
            all_nodes.extend(sub_inodes)

        # also add original node to node
        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)

    all_nodes_dict = {n.node_id: n for n in all_nodes}
    vector_index_chunk = VectorStoreIndex(all_nodes, embed_model=Settings.embed_model)
    vector_retriever_chunk = vector_index_chunk.as_retriever(similarity_top_k=3)
    retriever_chunk = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever_chunk},
        node_dict=all_nodes_dict,
        verbose=True,
    )
    return vector_retriever_chunk


# 句子窗口类需要特殊的节点生成的索引
def sentence_window_retriever(index):
    return index.as_retriever(node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ], )


def custom_retriever(index, mode=0):
    vector_r = vector_retriever(index)
    bm25_r = bm25_retriever(index)
    keyword_r = keyword_retriever(index)
    if mode > 1 or mode < 0:
        raise ValueError("Invalid mode for custom retriever."+ str(mode))

    if mode == 0:
        custom_r = CustomRetriever(vector_retriever_c=vector_r, bm25_retriever_c=bm25_r, mode="AND")
    elif mode == 1:
        custom_r = CustomRetriever(vector_retriever_c=vector_r, keyword_retriever_c=keyword_r, mode="OR")
    return custom_r


# refine：通过按顺序遍历每个检索到的文本块来创建和优化答案。 这将为每个节点/检索到的块进行单独的 LLM 调用。
# compact（默认）：事先压缩（连接）块，从而减少 LLM 调用。填充尽可能多的文本（从检索到的块中串联/打包）可以放入上下文窗口
# tree_summarize：根据需要多次使用提示查询 LLM，以便所有串联的块 被查询，从而产生同样多的答案，这些答案本身在 LLM 调用中以递归方式用作块 依此类推，直到只剩下一个块，因此只有一个最终答案。
# simple_summarize：截断所有文本块以适应单个 LLM 提示。适合快速 摘要目的，但可能会因截断而丢失详细信息。
# accumulate：给定一组文本块和查询，将查询应用于每个文本 块，同时将响应累积到数组中。返回 all 的串联字符串。当您需要对每个文本分别运行相同查询时，非常有用
# compact_accumulate：与 accumulate 相同，但会“压缩”每个类似于 的 LLM 提示符，并对每个文本块运行相同查询。compact
def response_synthesizer(mode=0):
    if mode > 7:
        mode = 1
    choose = [ResponseMode.REFINE, ResponseMode.COMPACT, ResponseMode.COMPACT_ACCUMULATE, ResponseMode.ACCUMULATE,
              ResponseMode.TREE_SUMMARIZE, ResponseMode.SIMPLE_SUMMARIZE, ResponseMode.NO_TEXT, ResponseMode.GENERATION]
    response_s = get_response_synthesizer(response_mode=choose[mode])
    return response_s

# 检索器类型包括：
# BM25
# Vector
# Summary: 汇总检索器 mode = 0 1 2 对应 最简单的文档汇总索引 基于编码的文档汇总索引 基于大模型的文档汇总索引 必须为关键字表索引 *
# Keyword: 关键字表检索器 mode = 0 1 对应 基本关键字表检索器 GPT关键字表检索器 必须为关键字表索引 *
# Custom
# QueryFusion: 融合检索器 其将来自多个文档的索引作为输入，并自动进行问题扩充，以获得多次查询结果用于合并,index需为一个list
# mode = 0, 1 分别使用llama simple重排序方法，1采用RRF
# AutoMerging
# Recursive
# SentenceWindow *
# Tree: index必须为树索引 *
# mode: 1,2,3,0 对应 TreeAllLeafRetriever TreeSelectLeafRetriever TreeSelectLeafEmbeddingRetriever TreeRootRetriever
# mode确定检索器的模式
def get_retriver(type: str, index, mode: int = 0, node = None, hierarchical_storage_context = None):
    if type == "BM25":
        retriever = bm25_retriever(index)
    elif type == "Vector":
        retriever = vector_retriever(index)
    elif type == "Summary":
        retriever = summary_retriever(index, mode=mode)
    elif type == "Tree":
        retriever = tree_retriever(index, mode=mode)
    elif type == "Keyword":
        retriever = keyword_retriever(index, mode=mode)
    elif type == "Custom":
        retriever = custom_retriever(index, mode=mode)
    elif type == "QueryFusion":
        retriever = query_fusion_retriever(index, mode=mode)
    elif type == "AutoMerging":
        retriever = auto_merging_retriever(index, hierarchical_storage_context)
    elif type == "Recursive":
        retriever = recursive_retriever(base_nodes=node)
    elif type == "SentenceWindow":
        retriever = sentence_window_retriever(index)
    else:
        raise Exception("retriever not supported: %s" % mode)

    return retriever


def query_expansion(ret, query_number=4, similarity_top_k=10):
    if ret is None:
        ret = []
        warnings.warn("query_expansion未传入检索器")
    return QueryFusionRetriever(
        ret,
        similarity_top_k=similarity_top_k,
        num_queries=query_number,  # set this to 1 to disable query generation
        use_async=True,
        verbose=True,
        # query_gen_prompt="...",  # we could override the query generation prompt here
    )


class AllRetriever:
    _doc = None,
    _nodes = None,
    _index = None,
    _retriever = []
    _query_number = 4,
    _similar_k_top = 10,
    _syn = 0

    def __init__(self, nodes_, vector_index_, summary_index_, tree_index_, keyword_index_, sentence_index_, mode=0):
        self.bm25_retriever = bm25_retriever(vector_index_)
        self._retriever.append(self.bm25_retriever)
        self.vector_retriever = vector_retriever(vector_index_)
        self._retriever.append(self.vector_retriever)
        self.summary_retriever = summary_retriever(summary_index_)
        self._retriever.append(self.summary_retriever)
        self.tree_retriever = tree_retriever(tree_index_)
        self._retriever.append(self.tree_retriever)
        self.keyword_retriever = keyword_retriever(keyword_index_)
        self._retriever.append(self.keyword_retriever)
        self.doc_s_retriever = document_summary_retrievers(vector_index_)
        self._retriever.append(self.doc_s_retriever)

        self.custom_retriever = custom_retriever(vector_index_)
        self._retriever.append(self.custom_retriever)
        self.query_fusion_retriever = query_fusion_retriever(vector_index_, mode=1)
        self._retriever.append(self.query_fusion_retriever)
        self.auto_merging_retriever = auto_merging_retriever(vector_index_, nodes_)
        self._retriever.append(self.auto_merging_retriever)
        self.recursive_retriever = recursive_retriever(nodes_)
        self._retriever.append(self.recursive_retriever)
        self.router_retriever = get_query_engine_by_router(summary_index_, vector_index_, keyword_index_)
        self._retriever.append(self.router_retriever)
        self.sentence_window_retriever = sentence_window_retriever(sentence_index_)
        self._retriever.append(self.sentence_window_retriever)
        self.response_syn = response_synthesizer(mode)

    def query_expansion(self, retriever, query_number: Optional[int], similarity_number: Optional[int]):
        if query_number is not None:
            self._query_number = query_number
        if similarity_number is not None:
            self._similar_k_top = similarity_number
        return query_expansion([retriever], query_number=self._query_number, similarity_top_k=self._similar_k_top)

    def get_response_mode(self, retriever_, mode=0):
        if mode != 0:
            self._syn = mode
        query_e = RetrieverQueryEngine(
            retriever=retriever_,
            response_synthesizer=response_synthesizer(mode),
            node_postprocessors=[LongContextReorder()]
        )
        return query_e


from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex, SummaryIndex

# define query engines
...


def get_query_engine_by_router(summary_index=None, vector_index=None, keyword_index=None):
    summary_tool = None
    vector_tool = None
    keyword_tool = None
    if summary_index is None:
        warnings.warn("Summary_index is None")
    else:
        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_retriever(summary_index),
            description="Useful for summarization questions related to the data source",
        )
    if vector_index is None:
        warnings.warn("vector_index is None")
    else:
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=custom_retriever(vector_index),
            description="Useful for retrieving specific context related to the data source",
        )

    if keyword_index is None:
        warnings.warn("keyword_index is None")
    else:
        keyword_tool = QueryEngineTool.from_defaults(
            query_engine=keyword_retriever(keyword_index),
            description="Useful for retrieving keyword related to the data source",
        )
    query_engine_ = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
            keyword_tool
        ],
    )
    return query_engine_


if __name__ == '__main__':
    Settings.llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")
    # 需要一个直接放文件的本地目录
    documents = SimpleDirectoryReader(f"E:\\junior_second_\\benchmark_llm\\RAG-benchmark").load_data()
    index_ = VectorStoreIndex.from_documents(documents,
                                    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=20)],
                                    show_progress=True)
    # splitter = SentenceSplitter(
    #     chunk_size=1024,
    #     chunk_overlap=20,
    # )
    # nodes = splitter.get_nodes_from_documents(documents)
    # index_ = VectorStoreIndex(nodes)
    retriever = vector_retriever(index_)  # 可用
    # retriever = vector_retriever(index_)  # 可用

    # retriever = summary_retriever(SummaryIndex(nodes))  # 可用
    # retriever = summary_retriever(index_)

    # retriever = tree_retriever(TreeIndex(nodes))  # 可用，看起来效果很差

    # retriever = keyword_retriever(index_)  # 未测试

    # vector_retriever = vector_retriever(index=index_)
    # bm25_retriever = bm25_retriever(index=index_)
    # retriever = CustomRetriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)  # 可用

    # retriever = query_fusion_retriever([index_, index_], mode=1)  #可用

    # retriever = auto_merging_retriever(index_, nodes)  # 可用

    #     node_parser = SentenceWindowNodeParser.from_defaults(
    #         window_size=3,
    #         window_metadata_key="window",
    #         original_text_metadata_key="original_text",
    #     )
    # retriever = sentence_window_retriever(index_)  # 可用

    # retriever = recursive_retriever(nodes)  # 可用
    response_syn = response_synthesizer(0)
    nodes = retriever.retrieve("请用中文回答我的毕业设计题目是什么")
    print(nodes)
    query_engine = RetrieverQueryEngine(
        retriever=get_retriver("QueryFusion", index_, mode=0),
        response_synthesizer=response_syn,
        node_postprocessors=[LongContextReorder()]
    )
    query_e = query_expansion([query_engine],query_number=4,similarity_top_k=3)
    query_engine = RetrieverQueryEngine.from_args(query_e)
    response = query_engine.query("请用中文回答我的毕业设计题目是什么")
    print(response)
