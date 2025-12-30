from typing import Literal
from ..llms import get_llm
from ..config import Config
from .utils import load_prompt
from ..utils import get_module_logger

logger = get_module_logger(__name__)


class QueryClassificationError(Exception):
    """Custom exception for query classification failures."""

    pass


class QueryComplexityClassifier:
    """
    Classifies query complexity to determine appropriate RAG strategy.
    """

    def __init__(self, llm=None):
        """
        Initialize the query classifier.

        Params:
            llm (optional): LLM instance for classification
        """
        self.config = Config()
        self.adaptive_rag_config = self.config.config.get("adaptive_rag", {})
        self.llm = llm if llm is not None else get_llm(self.config.llm)

    def classify(self, query: str) -> Literal["SIMPLE", "MODERATE", "COMPLEX"]:
        """
        Classify query complexity with retry mechanism.

        Params:
            query (str): Input query to classify

        Returns:
            Classification level as string

        Raises:
            QueryClassificationError: When classification fails after max retries
        """
        max_retries = self.adaptive_rag_config.get("max_classification_retries", 3)

        classification_prompt = load_prompt("query_classification")
        prompt = classification_prompt.format(query=query)

        valid_classes = ["SIMPLE", "MODERATE", "COMPLEX"]
        for attempt in range(max_retries):
            try:
                response = self.llm.complete(prompt)
                classification_raw = response.text.strip()
                classification = classification_raw.upper()

                logger.info(f"Classification attempt {attempt + 1}: {classification}")

                if classification in valid_classes:
                    return classification

                # Try to extract last valid word from response
                words = [w.upper() for w in classification_raw.split() if w.isalpha()]
                for word in reversed(words):
                    if word in valid_classes:
                        logger.info(
                            f"Recovered valid classification from response: {word}"
                        )
                        return word

                # Invalid classification - prepare retry prompt
                if attempt < max_retries - 1:
                    retry_prompt = load_prompt("query_classification_retry_invalid")
                    prompt = retry_prompt.format(
                        previous_response=classification_raw, query=query
                    )

            except Exception as e:
                logger.error(f"Error during classification attempt {attempt + 1}: {e}")

                if attempt < max_retries - 1:
                    # Prepare retry prompt with error context
                    retry_prompt = load_prompt("query_classification_retry_error")
                    prompt = retry_prompt.format(error_message=str(e), query=query)
                    continue

        logger.error(
            f"Failed to classify query after {max_retries} attempts. Fallback to 'MODERATE'."
        )
        return "MODERATE"
