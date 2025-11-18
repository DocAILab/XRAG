from ..llms import get_llm
from ..config import Config
from .utils import load_prompt
from llama_index.core import QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from ..retrievers.retriever import get_retriver, response_synthesizer


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

    def execute(self, query: str, **kwargs) -> str:
        """
        Generate response directly using LLM.

        Params:
            query (str): Input query
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        response = self.llm.complete(query)
        return response.text


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

    def execute(self, query: str, **kwargs) -> str:
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

        response = query_engine.query(query)
        return response.response


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

    def _is_information_sufficient(self, query: str, context: str) -> tuple[bool, str]:
        """
        Check if retrieved information is sufficient.

        Params:
            query (str): Original query
            context (str): Retrieved context

        Returns:
            Tuple of (is_sufficient, additional_info_needed)
        """
        sufficiency_prompt = load_prompt("information_sufficiency")

        prompt = sufficiency_prompt.format(query=query, context=context)
        response = self.llm.complete(prompt)

        response_text = response.text.strip()
        decision = "YES" in response_text.upper()
        additional_info = response_text.replace("YES", "").replace("NO", "").strip()

        return decision, additional_info

    def execute(self, query: str, **kwargs) -> str:
        """
        Execute iterative retrieval and generation.

        Params:
            query (str): Input query
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        retriever = get_retriver(self.retriever_type, self.index, cfg=self.config)
        all_context = []
        current_query = query

        for iteration in range(self.max_iterations):
            nodes = retriever.retrieve(QueryBundle(query_str=current_query))

            # Extract context from nodes
            context_texts = [node.node.get_content() for node in nodes]
            all_context.extend(context_texts)

            # Check if information is sufficient
            combined_context = "\n".join(all_context)
            is_sufficient, additional_info = self._is_information_sufficient(
                query, combined_context
            )

            if is_sufficient or iteration == self.max_iterations - 1:
                break

            # Generate refined query for next iteration
            current_query = f"{query}\nAdditional information needed: {additional_info}"

        # Generate final response
        synthesizer = response_synthesizer(self.config.responce_synthsizer)
        combined_context = "\n".join(all_context)
        context_aware_query = f"Query: {query}\n\nRelevant Context:\n{combined_context}"

        query_engine = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=synthesizer
        )
        final_response = query_engine.query(context_aware_query)
        return final_response.response
