import re
from enum import Enum
from typing import Dict, Any
from ..utils import get_module_logger

logger = get_module_logger(__name__)


class SelfRAGTokens(Enum):
    """All Self-RAG special tokens."""

    # Relevance tokens
    IRRELEVANT = "[Irrelevant]"
    RELEVANT = "[Relevant]"

    # Retrieval tokens
    NO_RETRIEVAL = "[No Retrieval]"
    RETRIEVAL = "[Retrieval]"
    CONTINUE_EVIDENCE = "[Continue to Use Evidence]"

    # Utility tokens
    UTILITY_1 = "[Utility:1]"
    UTILITY_2 = "[Utility:2]"
    UTILITY_3 = "[Utility:3]"
    UTILITY_4 = "[Utility:4]"
    UTILITY_5 = "[Utility:5]"

    # Grounding tokens
    FULLY_SUPPORTED = "[Fully supported]"
    PARTIALLY_SUPPORTED = "[Partially supported]"
    NO_SUPPORT = "[No support / Contradictory]"

    # Other special tokens
    START = "<s>"
    END = "</s>"
    PAD = "[PAD]"
    UNK = "<unk>"
    PARAGRAPH_START = "<paragraph>"
    PARAGRAPH_END = "</paragraph>"

    @classmethod
    def get_control_tokens(cls):
        """Get all control tokens used for response cleaning."""
        return [
            cls.FULLY_SUPPORTED.value,
            cls.PARTIALLY_SUPPORTED.value,
            cls.NO_SUPPORT.value,
            cls.NO_RETRIEVAL.value,
            cls.RETRIEVAL.value,
            cls.IRRELEVANT.value,
            cls.RELEVANT.value,
            cls.PARAGRAPH_START.value,
            cls.PARAGRAPH_END.value,
            cls.UTILITY_1.value,
            cls.UTILITY_2.value,
            cls.UTILITY_3.value,
            cls.UTILITY_4.value,
            cls.UTILITY_5.value,
        ]


class SelfRAGResponseParser:
    """Parser for Self-RAG model responses with special tokens."""

    # Self-RAG special tokens
    RETRIEVAL_TOKEN = SelfRAGTokens.RETRIEVAL.value
    NO_RETRIEVAL_TOKEN = SelfRAGTokens.NO_RETRIEVAL.value
    RELEVANT_TOKEN = SelfRAGTokens.RELEVANT.value
    IRRELEVANT_TOKEN = SelfRAGTokens.IRRELEVANT.value
    FULLY_SUPPORTED_TOKEN = SelfRAGTokens.FULLY_SUPPORTED.value
    PARTIALLY_SUPPORTED_TOKEN = SelfRAGTokens.PARTIALLY_SUPPORTED.value
    NO_SUPPORT_TOKEN = SelfRAGTokens.NO_SUPPORT.value
    UTILITY_TOKEN_PATTERN = r"\[Utility:(\d+)\]"

    @classmethod
    def parse_response(cls, response: str) -> Dict[str, Any]:
        """
        Parse a Self-RAG response to extract components and metadata.

        Params:
            response (str): Raw response from Self-RAG model

        Returns:
            Dictionary containing parsed components
        """
        parsed = {
            "raw_response": response,
            "content": response,
            "needs_retrieval": False,
            "retrieved": False,
            "relevance": None,
            "support": None,
            "utility_score": None,
            "special_tokens": [],
        }

        # Parse retrieval decision
        if cls.RETRIEVAL_TOKEN in response:
            parsed["needs_retrieval"] = True
            parsed["retrieved"] = True
            parsed["special_tokens"].append(cls.RETRIEVAL_TOKEN)

        if cls.NO_RETRIEVAL_TOKEN in response:
            parsed["needs_retrieval"] = False
            parsed["special_tokens"].append(cls.NO_RETRIEVAL_TOKEN)

        # Parse relevance assessment
        if cls.RELEVANT_TOKEN in response:
            parsed["relevance"] = "relevant"
            parsed["special_tokens"].append(cls.RELEVANT_TOKEN)
        elif cls.IRRELEVANT_TOKEN in response:
            parsed["relevance"] = "irrelevant"
            parsed["special_tokens"].append(cls.IRRELEVANT_TOKEN)

        # Parse support assessment
        if cls.FULLY_SUPPORTED_TOKEN in response:
            parsed["support"] = "fully_supported"
            parsed["special_tokens"].append(cls.FULLY_SUPPORTED_TOKEN)
        elif cls.PARTIALLY_SUPPORTED_TOKEN in response:
            parsed["support"] = "partially_supported"
            parsed["special_tokens"].append(cls.PARTIALLY_SUPPORTED_TOKEN)
        elif cls.NO_SUPPORT_TOKEN in response:
            parsed["support"] = "no_support"
            parsed["special_tokens"].append(cls.NO_SUPPORT_TOKEN)

        # Parse utility score
        utility_match = re.search(cls.UTILITY_TOKEN_PATTERN, response)
        if utility_match:
            parsed["utility_score"] = int(utility_match.group(1))
            parsed["special_tokens"].append(utility_match.group(0))

        # Clean content by removing special tokens and markup
        content = cls._clean_content(response)
        parsed["content"] = content

        return parsed

    @classmethod
    def _clean_content(cls, response: str) -> str:
        """
        Clean content by removing special tokens and markup.
        Following original postprocess_answer_option_conditioned function.

        Params:
            response (str): Raw response

        Returns:
            Clean content string
        """
        content = response
        control_tokens = SelfRAGTokens.get_control_tokens()

        for token in control_tokens:
            content = content.replace(token, "")

        # Remove end-of-sequence tokens
        if SelfRAGTokens.END.value in content:
            content = content.replace(SelfRAGTokens.END.value, "")
        if "\n" in content:
            content = content.replace("\n", "")
        if "<|endoftext|>" in content:
            content = content.replace("<|endoftext|>", "")

        return content

    @classmethod
    def extract_content_only(cls, response: str) -> str:
        """
        Extract only the content from a Self-RAG response, removing all special tokens.

        Params:
            response (str): Raw response from Self-RAG model

        Returns:
            Clean content string
        """
        parsed = cls.parse_response(response)
        return parsed["content"]

    @classmethod
    def get_scoring_components(cls, response: str) -> Dict[str, float]:
        """
        Extract scoring components from a Self-RAG response for use in pipeline scoring.

        Params:
            response (str): Raw response from Self-RAG model

        Returns:
            Dictionary with relevance_score, ground_score, utility_score as floats
        """
        parsed = cls.parse_response(response)

        # Convert relevance to score (1.0 for relevant, 0.0 for irrelevant)
        relevance_score = 0.0
        if parsed["relevance"] == "relevant":
            relevance_score = 1.0
        elif parsed["relevance"] == "irrelevant":
            relevance_score = 0.0

        # Convert support to score
        ground_score = 0.0
        if parsed["support"] == "fully_supported":
            ground_score = 1.0
        elif parsed["support"] == "partially_supported":
            ground_score = 0.5
        elif parsed["support"] == "no_support":
            ground_score = 0.0

        # Convert utility score
        utility_score = 0.0
        if parsed["utility_score"] is not None:
            # 1->-1, 2->-0.5, 3->0, 4->0.5, 5->1
            utility_score = (parsed["utility_score"] - 3) * 0.5

        return {
            "relevance_score": relevance_score,
            "ground_score": ground_score,
            "utility_score": utility_score,
        }
