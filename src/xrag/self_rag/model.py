from ..config import Config
from .utils import SelfRAGTokens
from typing import List, Optional
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..utils import get_module_logger

logger = get_module_logger(__name__)


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 100
    skip_special_tokens: bool = False


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
        """Initialize the transformers model, tokenizer, and sampling parameters."""
        try:
            model_name = self.self_rag_config.get(
                "model_name", "selfrag/selfrag_llama2_7b"
            )
            cache_dir = self.self_rag_config.get("download_dir", None)
            dtype = self.self_rag_config.get("dtype", "half")

            dtype_map = {
                "half": torch.float16,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            torch_dtype = dtype_map.get(str(dtype).lower(), torch.float16)

            device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Initializing Self-RAG model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir, use_fast=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            self.model.eval()

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
            device = next(self.model.parameters()).device
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=False
            ).to(device)
            with torch.no_grad():
                gen_outputs = self.model.generate(
                    **enc,
                    do_sample=(self.sampling_params.temperature > 0),
                    temperature=self.sampling_params.temperature
                    if self.sampling_params.temperature > 0
                    else None,
                    top_p=self.sampling_params.top_p,
                    max_new_tokens=self.sampling_params.max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            responses = []
            for i, prompt in enumerate(prompts):
                prompt_len = enc.input_ids[i].shape[0]
                generated_ids = gen_outputs[i][prompt_len:]
                text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=self.sampling_params.skip_special_tokens,
                )
                responses.append(text)
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
        device = next(self.model.parameters()).device

        try:
            logger.debug(
                f"Generating response with logprobs for query: {query[:50]}..."
            )
            enc = self.tokenizer([prompt], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **enc,
                    do_sample=(self.sampling_params.temperature > 0),
                    temperature=self.sampling_params.temperature
                    if self.sampling_params.temperature > 0
                    else None,
                    top_p=self.sampling_params.top_p,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            full_sequence = outputs.sequences[0]
            prompt_len = enc.input_ids.shape[1]
            generated_ids = full_sequence[prompt_len:]
            text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=self.sampling_params.skip_special_tokens,
            )

            logprobs_list = []
            cumulative_logprob = 0.0
            top_k_store = 5000
            for step, (scores_step, token_id) in enumerate(
                zip(outputs.scores, generated_ids)
            ):
                probs_log = torch.log_softmax(scores_step[0], dim=-1)
                chosen_logprob = probs_log[token_id].item()
                cumulative_logprob += chosen_logprob
                vocab_size = probs_log.size(0)
                k = min(top_k_store, vocab_size)
                top_logprobs_vals, top_indices = torch.topk(probs_log, k)
                step_dict = {
                    int(idx): float(val)
                    for idx, val in zip(top_indices.tolist(), top_logprobs_vals.tolist())
                }
                logprobs_list.append(step_dict)

            response = {
                "text": text,
                "logprobs": logprobs_list,
                "token_ids": generated_ids.tolist(),
                "cumulative_logprob": cumulative_logprob,
            }
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

        tokenizer = self.tokenizer

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
