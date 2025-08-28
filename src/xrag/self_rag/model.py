from ..config import Config
from .utils import SelfRAGTokens
from typing import List, Optional
from vllm import LLM, SamplingParams
from ..utils import get_module_logger

logger = get_module_logger(__name__)


class SelfRAGModel:
    """Self-RAG model for retrieval-augmented generation with self-reflection."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Self-RAG model.

        Params:
            config (Config, optional): Configuration object. If None, uses global config.
        """
        self.config = config or Config()
        self.self_rag_config = self.config.config.get("self_rag", {})
        self.model = None
        self.sampling_params = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the VLLM model and sampling parameters."""
        try:
            model_name = self.self_rag_config.get(
                "model_name", "selfrag/selfrag_llama2_7b"
            )
            download_dir = self.self_rag_config.get("download_dir", None)
            dtype = self.self_rag_config.get("dtype", "half")

            # Initialize model
            model_kwargs = {"dtype": dtype}
            if download_dir:
                model_kwargs["download_dir"] = download_dir

            logger.info(f"Initializing Self-RAG model: {model_name}")
            self.model = LLM(model_name, **model_kwargs)

            # Initialize sampling parameters
            temperature = self.self_rag_config.get("temperature", 0.0)
            top_p = self.self_rag_config.get("top_p", 1.0)
            max_tokens = self.self_rag_config.get("max_tokens", 100)
            skip_special_tokens = self.self_rag_config.get("skip_special_tokens", False)

            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                skip_special_tokens=skip_special_tokens,
            )

            logger.info("Self-RAG model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Self-RAG model: {e}")
            raise

    def format_prompt(self, input_text: str, paragraph: Optional[str] = None) -> str:
        """
        Format the input prompt for Self-RAG model.

        Params:
            input_text (str): The user query or instruction
            paragraph (str, optional): Optional retrieved paragraph to include in the prompt

        Returns:
            Formatted prompt string
        """
        prompt = f"### Instruction:\n{input_text}\n\n### Response:\n"
        if paragraph is not None:
            prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
        return prompt

    def generate(
        self, queries: List[str], paragraphs: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate responses for a list of queries.

        Params:
            queries (List[str]): List of input queries
            paragraphs (List[str], optional): Optional list of retrieved paragraphs (one per query)

        Returns:
            List of generated responses
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        # Format prompts
        if paragraphs is None:
            prompts = [self.format_prompt(query) for query in queries]
        else:
            if len(paragraphs) != len(queries):
                raise ValueError("Number of paragraphs must match number of queries")
            prompts = [
                self.format_prompt(query, para)
                for query, para in zip(queries, paragraphs)
            ]

        # Generate responses
        try:
            logger.info(f"Generating responses for {len(queries)} queries")
            predictions = self.model.generate(prompts, self.sampling_params)
            responses = [pred.outputs[0].text for pred in predictions]
            logger.info("Response generation completed")
            return responses
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    def generate_single(self, query: str, paragraph: Optional[str] = None) -> str:
        """
        Generate a response for a single query.

        Params:
            query (str): Input query
            paragraph (str, optional): Optional retrieved paragraph

        Returns:
            Generated response
        """
        responses = self.generate([query], [paragraph] if paragraph else None)
        return responses[0]

    def generate_with_logprobs(
        self, query: str, paragraph: Optional[str] = None, max_new_tokens: int = 15
    ) -> dict:
        """
        Generate a response with log probabilities for special tokens.

        Params:
            query (str): Input query
            paragraph (str, optional): Optional retrieved paragraph
            max_new_tokens (int): Maximum tokens to generate

        Returns:
            Dictionary with 'text' and 'logprobs' keys
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        prompt = self.format_prompt(query, paragraph)

        # Create sampling params with logprobs enabled
        sampling_params = SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            max_tokens=max_new_tokens,
            skip_special_tokens=self.sampling_params.skip_special_tokens,
            logprobs=5000,
        )

        try:
            logger.debug(
                f"Generating response with logprobs for query: {query[:50]}..."
            )
            predictions = self.model.generate([prompt], sampling_params)
            pred = predictions[0]

            response = {
                "text": pred.outputs[0].text,
                "logprobs": [],
                "token_ids": pred.outputs[0].token_ids,
                "cumulative_logprob": pred.outputs[0].cumulative_logprob,
            }

            # Extract logprobs if available
            if pred.outputs[0].logprobs:
                for token_logprobs in pred.outputs[0].logprobs:
                    if token_logprobs:
                        # Convert to dict mapping token_id -> logprob
                        logprob_dict = {
                            token_id: logprob.logprob
                            for token_id, logprob in token_logprobs.items()
                        }
                        response["logprobs"].append(logprob_dict)

            return response

        except Exception as e:
            logger.error(f"Error during generation with logprobs: {e}")
            raise

    def get_special_tokens(self) -> tuple:
        """
        Get token IDs for Self-RAG special tokens.

        Returns:
            Tuple of (ret_tokens, rel_tokens, grd_tokens, ut_tokens) dictionaries
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")

        tokenizer = self.model.get_tokenizer()

        # Retrieval tokens
        ret_tokens = {
            SelfRAGTokens.RETRIEVAL.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.RETRIEVAL.value
            ),
            SelfRAGTokens.NO_RETRIEVAL.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.NO_RETRIEVAL.value
            ),
            SelfRAGTokens.CONTINUE_EVIDENCE.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.CONTINUE_EVIDENCE.value
            ),
        }

        # Relevance tokens
        rel_tokens = {
            SelfRAGTokens.RELEVANT.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.RELEVANT.value
            ),
            SelfRAGTokens.IRRELEVANT.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.IRRELEVANT.value
            ),
        }

        # Grounding/support tokens
        grd_tokens = {
            SelfRAGTokens.FULLY_SUPPORTED.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.FULLY_SUPPORTED.value
            ),
            SelfRAGTokens.PARTIALLY_SUPPORTED.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.PARTIALLY_SUPPORTED.value
            ),
            SelfRAGTokens.NO_SUPPORT.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.NO_SUPPORT.value
            ),
        }

        # Utility tokens
        ut_tokens = {
            SelfRAGTokens.UTILITY_1.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.UTILITY_1.value
            ),
            SelfRAGTokens.UTILITY_2.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.UTILITY_2.value
            ),
            SelfRAGTokens.UTILITY_3.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.UTILITY_3.value
            ),
            SelfRAGTokens.UTILITY_4.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.UTILITY_4.value
            ),
            SelfRAGTokens.UTILITY_5.value: tokenizer.convert_tokens_to_ids(
                SelfRAGTokens.UTILITY_5.value
            ),
        }

        return ret_tokens, rel_tokens, grd_tokens, ut_tokens
