from ..llms import get_llm
from ..config import Config
from typing import Dict, Optional
from .query_classifier import QueryComplexityClassifier
from .strategies import (
    AdaptiveStrategy,
    DirectGenerationStrategy,
    SingleRetrievalStrategy,
    IterativeRetrievalStrategy,
)


class AdaptiveRAG:
    """
    Adaptive RAG engine that selects appropriate strategy based on query complexity.
    """

    def __init__(self, index, llm=None, config: Optional[Config] = None):
        """
        Initialize Adaptive RAG engine.

        Params:
            index: Document index for retrieval
            llm (optional): LLM instance to use
            config (Config, optional): Configuration object
        """
        self.index = index
        self.config = config or Config()
        self.adaptive_rag_config = self.config.config.get("adaptive_rag", {})

        # Use provided LLM or get from config
        self.llm = llm if llm is not None else get_llm(self.config.llm)

        # Initialize components
        self.classifier = QueryComplexityClassifier(llm=self.llm)

        # Initialize strategies
        self.strategies: Dict[str, AdaptiveStrategy] = {
            "SIMPLE": DirectGenerationStrategy(llm=self.llm),
            "MODERATE": SingleRetrievalStrategy(
                index=self.index,
                llm=self.llm,
                retriever_type=self.adaptive_rag_config.get("retriever_type", "Vector"),
            ),
            "COMPLEX": IterativeRetrievalStrategy(
                index=self.index,
                llm=self.llm,
                max_iterations=self.adaptive_rag_config.get("max_iterations", 3),
                retriever_type=self.adaptive_rag_config.get("retriever_type", "Vector"),
            ),
        }

    def query(
        self, query_str: str, force_strategy: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Process query using adaptive strategy selection.

        Params:
            query_str (str): Input query string
            force_strategy (str, optional): Force specific strategy (SIMPLE, MODERATE, COMPLEX)

        Returns:
            Dictionary containing response and metadata
        """
        # Classify query complexity
        if force_strategy and force_strategy in self.strategies:
            complexity = force_strategy
        else:
            complexity = self.classifier.classify(query_str)

        # Select and execute strategy
        strategy = self.strategies[complexity]
        response_obj = strategy.execute(query_str)

        # Normalize to text
        if hasattr(response_obj, "response"):
            response_text = response_obj.response
        elif hasattr(response_obj, "text"):
            response_text = response_obj.text
        else:
            response_text = str(response_obj)

        return {
            "response": response_text,
            "strategy_used": complexity,
            "query": query_str,
            "response_obj": response_obj,
        }
