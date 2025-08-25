"""SePer model implementations for generation and entailment checking"""

import torch
import torch.nn.functional as F
from ...utils import get_module_logger
from typing import List, Tuple, Optional, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

logger = get_module_logger(__name__)


class SePerGenerationModel:
    """Generation model wrapper for SePer evaluation."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 128,
        **kwargs,
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens

        logger.info(f"Loading generation model: {model_path}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "auto" else None,
            **kwargs,
        )

        if device != "auto":
            self.model = self.model.to(device)

        self.model.eval()

    def generate_responses(
        self, prompt: str, num_generations: int = 10, temperature: float = 1.0, **kwargs
    ) -> List[Tuple[str, List[float]]]:
        """
        Generate multiple responses for a given prompt.

        Params:
            prompt (str): Input prompt
            num_generations (int): Number of responses to generate
            temperature (float): Sampling temperature

        Returns:
            List of tuples containing (response_text, token_log_likelihoods)
        """
        responses = []

        with torch.no_grad():
            for _ in range(num_generations):
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )

                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs.sequences[0][input_length:]

                # Calculate token log likelihoods
                token_log_probs = []
                if outputs.scores:
                    for i, score in enumerate(outputs.scores):
                        log_probs = F.log_softmax(score[0], dim=-1)
                        token_id = generated_tokens[i].item()
                        token_log_probs.append(log_probs[token_id].item())

                response_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()

                responses.append((response_text, token_log_probs))

        return responses


class SePerEntailmentModel:
    """Entailment model wrapper for semantic equivalence checking."""

    def __init__(
        self, model_path: str = "microsoft/deberta-v2-xlarge-mnli", device: str = "cuda"
    ):
        self.model_path = model_path
        self.device = device

        logger.info(f"Loading entailment model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
            device
        )

        self.model.eval()

    def check_entailment(
        self,
        text1_batch: List[str],
        text2_batch: List[str],
        entailment_type: str = "loose",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Check entailment between two batches of texts.

        Params:
            text1_batch (List[str]): First batch of texts
            text2_batch (List[str]): Second batch of texts
            entailment_type (str): Type of entailment check ('bi', 'a_entails_b', 'b_entails_a', 'loose')

        Returns:
            Tuple of (semantic_equivalence_mask, entailment_logits_dict)
        """
        with torch.no_grad():
            # Check text1 -> text2 entailment
            inputs_1_to_2 = self.tokenizer(
                text1_batch,
                text2_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            outputs_1_to_2 = self.model(**inputs_1_to_2)
            logits_1_to_2 = outputs_1_to_2.logits
            implication_1_to_2 = torch.argmax(F.softmax(logits_1_to_2, dim=1), dim=1)

            # Check text2 -> text1 entailment
            inputs_2_to_1 = self.tokenizer(
                text2_batch,
                text1_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            outputs_2_to_1 = self.model(**inputs_2_to_1)
            logits_2_to_1 = outputs_2_to_1.logits
            implication_2_to_1 = torch.argmax(F.softmax(logits_2_to_1, dim=1), dim=1)

            # Determine semantic equivalence based on entailment type
            if entailment_type == "bi":
                # Both directions must be entailment (class 2)
                semantic_equivalent = (implication_1_to_2 == 2) & (
                    implication_2_to_1 == 2
                )
            elif entailment_type == "a_entails_b":
                # text1 entails text2
                semantic_equivalent = implication_1_to_2 == 2
            elif entailment_type == "b_entails_a":
                # text2 entails text1
                semantic_equivalent = implication_2_to_1 == 2
            elif entailment_type == "loose":
                # No contradiction and not both neutral
                no_contradiction = (implication_1_to_2 != 0) & (implication_2_to_1 != 0)
                not_both_neutral = ~(
                    (implication_1_to_2 == 1) & (implication_2_to_1 == 1)
                )
                semantic_equivalent = no_contradiction & not_both_neutral
            else:
                raise ValueError(f"Unknown entailment type: {entailment_type}")

            entailment_logits = {
                "a_entails_b": logits_1_to_2,
                "b_entails_a": logits_2_to_1,
            }

            return semantic_equivalent, entailment_logits


def create_prompt(
    question: str,
    context: Optional[str] = None,
    instruction: str = "Answer the following question as briefly as possible.\n",
    max_context_words: int = 512,
) -> str:
    """
    Create a prompt for question answering.

    Params:
        question (str): The question to answer
        context (Optional[str]): Optional context information
        instruction (str): Instruction text
        max_context_words (int): Maximum number of words in context

    Returns:
        Formatted prompt string
    """
    prompt = instruction

    if context:
        context_tokens = context.split()
        if len(context_tokens) > max_context_words:
            context = " ".join(context_tokens[:max_context_words])
        prompt += f"Context: {context}\n"

    prompt += f"Question: {question}\nAnswer:"
    return prompt
