from ..llms import get_llm
from ..config import Config
from .utils import load_prompt
from llama_index.core import QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from ..retrievers.retriever import get_retriver, response_synthesizer


class SimpleResponse:
    def __init__(self, text: str):
        self.response = text
        self.source_nodes = []


class AdaptiveStrategy:
    """
    Base class for adaptive RAG strategies.
    """

    def execute(self, query: str, **kwargs) -> str:
        """
        Execute the strategy.

        Params:
            query (str): Input query
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        raise NotImplementedError


class DirectGenerationStrategy(AdaptiveStrategy):
    """
    Direct LLM generation without retrieval.
    """

    def __init__(self, llm=None):
        """
        Initialize direct generation strategy.

        Params:
            llm (optional): LLM instance
        """
        if llm is not None:
            self.llm = llm
        else:
            config = Config()
            self.llm = get_llm(config.llm)

    def execute(self, query: str, **kwargs) -> SimpleResponse:
        """
        Generate response directly using LLM.

        Params:
            query (str): Input query
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        resp = self.llm.complete(query)
        return SimpleResponse(resp.text)


class SingleRetrievalStrategy(AdaptiveStrategy):
    """
    Single-turn retrieval followed by generation.
    """

    def __init__(self, index, llm=None, retriever_type="Vector"):
        """
        Initialize single retrieval strategy.

        Params:
            index: Document index for retrieval
            llm (optional): LLM instance
            retriever_type (str, optional): Type of retriever to use
        """
        self.index = index
        self.config = Config()
        self.adaptive_rag_config = self.config.config.get("adaptive_rag", {})

        self.llm = llm if llm is not None else get_llm(self.config.llm)

        self.retriever_type = retriever_type or self.adaptive_rag_config.get(
            "retriever_type", "Vector"
        )

    def execute(self, query: str, **kwargs):
        """
        Execute single retrieval and generation.

        Params:
            query (str): Input query
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        retriever = get_retriver(self.retriever_type, self.index, cfg=self.config)
        synthesizer = response_synthesizer(self.config.responce_synthsizer)

        query_engine = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=synthesizer
        )
        return query_engine.query(query)


class IterativeRetrievalStrategy(AdaptiveStrategy):
    """
    Multi-turn iterative retrieval strategy.
    """

    def __init__(self, index, llm=None, max_iterations=3, retriever_type="Vector"):
        """
        Initialize iterative retrieval strategy.

        Params:
            index: Document index for retrieval
            llm (optional): LLM instance
            max_iterations (int, optional): Maximum number of retrieval iterations
            retriever_type (str, optional): Type of retriever to use
        """
        self.index = index
        self.config = Config()
        self.adaptive_rag_config = self.config.config.get("adaptive_rag", {})

        self.llm = llm if llm is not None else get_llm(self.config.llm)

        self.max_iterations = max_iterations or self.adaptive_rag_config.get(
            "max_iterations", 3
        )
        self.retriever_type = retriever_type or self.adaptive_rag_config.get(
            "retriever_type", "Vector"
        )

    def _is_information_sufficient(self, query: str, context: str):
        sufficiency_prompt = load_prompt("information_sufficiency")

        prompt = sufficiency_prompt.format(query=query, context=context)
        resp = self.llm.complete(prompt)
        text = resp.text.strip()
        decision = "YES" in text.upper()
        additional_info = text.replace("YES", "").replace("NO", "").strip()
        return decision, additional_info

    def execute(self, query: str, **kwargs):
        retriever = get_retriver(self.retriever_type, self.index, cfg=self.config)
        all_context = []
        current_query = query
        for _ in range(self.max_iterations):
            nodes = retriever.retrieve(QueryBundle(query_str=current_query))
            ctx_texts = [n.node.get_content() for n in nodes]
            all_context.extend(ctx_texts)
            combined = "\n".join(all_context)
            sufficient, info = self._is_information_sufficient(query, combined)
            if sufficient:
                break
            current_query = f"{query}\nAdditional information needed: {info}"
        synthesizer = response_synthesizer(self.config.responce_synthsizer)
        combined = "\n".join(all_context)
        context_query = f"Query: {query}\n\nRelevant Context:\n{combined}"
        query_engine = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=synthesizer
        )
        return query_engine.query(context_query)
