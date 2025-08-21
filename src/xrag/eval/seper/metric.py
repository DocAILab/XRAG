"""Main SePer metric evaluator"""

import time
from dataclasses import dataclass
from ...utils import get_module_logger
from typing import List, Dict, Any, Optional
from .calculator import SePerCalculator, process_evaluation_data
from .models import SePerGenerationModel, SePerEntailmentModel, create_prompt

logger = get_module_logger(__name__)


@dataclass
class SePerResult:
    """Result of SePer evaluation."""

    question: str
    seper_with_context: float
    seper_without_context: float
    delta_seper: float
    computation_time: float
    context: Optional[str] = None
    num_responses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "question": self.question,
            "seper_with_context": self.seper_with_context,
            "seper_without_context": self.seper_without_context,
            "delta_seper": self.delta_seper,
            "computation_time": self.computation_time,
            "context": self.context,
            "num_responses": self.num_responses,
        }


class SePerEvaluator:
    """Main SePer evaluator for RAG systems."""

    def __init__(
        self,
        generation_model_path: str = "meta-llama/Llama-2-7b-chat-hf",
        entailment_model_path: str = "microsoft/deberta-v2-xlarge-mnli",
        device: str = "cuda",
        num_generations: int = 10,
        temperature: float = 1.0,
        max_new_tokens: int = 128,
        computation_chunk_size: int = 8,
        max_context_words: int = 512,
        prompt_type: str = "default",
        **kwargs,
    ):
        """
        Initialize SePer evaluator.

        Params:
            generation_model_path (str): Path to generation model
            entailment_model_path (str): Path to entailment model
            device (str): Device to run models on
            num_generations (int): Number of responses to generate
            temperature (float): Sampling temperature
            max_new_tokens (int): Maximum new tokens to generate
            computation_chunk_size (int): Batch size for entailment computation
            max_context_words (int): Maximum words in context
            prompt_type (str): Type of prompt to use
        """
        self.generation_model_path = generation_model_path
        self.entailment_model_path = entailment_model_path
        self.device = device
        self.num_generations = num_generations
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.computation_chunk_size = computation_chunk_size
        self.max_context_words = max_context_words
        self.prompt_type = prompt_type

        # Initialize models
        logger.info("Initializing SePer evaluator...")
        self._initialize_models(**kwargs)

        # Initialize calculator
        self.calculator = SePerCalculator(
            entailment_model=self.entailment_model,
            computation_chunk_size=computation_chunk_size,
        )

        logger.info("SePer evaluator initialized successfully")

    def _initialize_models(self, **kwargs):
        """Initialize generation and entailment models."""

        logger.info(f"Loading generation model: {self.generation_model_path}")
        self.generation_model = SePerGenerationModel(
            model_path=self.generation_model_path,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            **kwargs,
        )

        logger.info(f"Loading entailment model: {self.entailment_model_path}")
        self.entailment_model = SePerEntailmentModel(
            model_path=self.entailment_model_path, device=self.device
        )

    def evaluate_single(
        self,
        question: str,
        context: str,
        ground_truth_answers: List[str],
        use_soft_clustering: bool = True,
    ) -> SePerResult:
        """
        Evaluate a single question-context pair.

        Params:
            question (str): The question to evaluate
            context (str): Retrieved context
            ground_truth_answers (List[str]): List of ground truth answers
            use_soft_clustering (bool): Whether to use soft or hard clustering

        Returns:
            SePerResult object with evaluation results
        """
        start_time = time.time()

        logger.info(f"Evaluating question: {question}")

        # Generate responses with context
        prompt_with_context = create_prompt(
            question=question, context=context, max_context_words=self.max_context_words
        )
        responses_with_context = self.generation_model.generate_responses(
            prompt=prompt_with_context,
            num_generations=self.num_generations,
            temperature=self.temperature,
        )

        # Generate responses without context (baseline)
        prompt_without_context = create_prompt(
            question=question, context=None, max_context_words=self.max_context_words
        )
        responses_without_context = self.generation_model.generate_responses(
            prompt=prompt_without_context,
            num_generations=self.num_generations,
            temperature=self.temperature,
        )

        # Process data for SePer calculation
        data_with_context = process_evaluation_data(
            question=question,
            context=context,
            responses_with_likelihoods=responses_with_context,
            ground_truth_answers=ground_truth_answers,
        )
        data_without_context = process_evaluation_data(
            question=question,
            context="",
            responses_with_likelihoods=responses_without_context,
            ground_truth_answers=ground_truth_answers,
        )

        # Calculate SePer scores
        if use_soft_clustering:
            seper_with_context = self.calculator.calculate_seper_soft(
                questions=[data_with_context["question"]],
                response_texts=[data_with_context["response_texts"]],
                ground_truth_answers=[data_with_context["ground_truth_answers"]],
                response_likelihoods=[data_with_context["likelihoods"]],
            )[0]
            seper_without_context = self.calculator.calculate_seper_soft(
                questions=[data_without_context["question"]],
                response_texts=[data_without_context["response_texts"]],
                ground_truth_answers=[data_without_context["ground_truth_answers"]],
                response_likelihoods=[data_without_context["likelihoods"]],
            )[0]
        else:
            seper_with_context = self.calculator.calculate_seper_hard(
                question=data_with_context["question"],
                responses=data_with_context["response_texts"],
                ground_truth_answers=data_with_context["ground_truth_answers"],
                response_log_likelihoods=data_with_context["response_log_likelihoods"],
            )
            seper_without_context = self.calculator.calculate_seper_hard(
                question=data_without_context["question"],
                responses=data_without_context["response_texts"],
                ground_truth_answers=data_without_context["ground_truth_answers"],
                response_log_likelihoods=data_without_context[
                    "response_log_likelihoods"
                ],
            )

        # Calculate delta SePer
        delta_seper = seper_with_context - seper_without_context

        computation_time = time.time() - start_time
        logger.info(f"SePer evaluation completed in {computation_time:.2f}s")
        logger.info(
            f"Results - With context: {seper_with_context:.4f}, "
            f"Without context: {seper_without_context:.4f}, "
            f"ΔSePer: {delta_seper:.4f}"
        )

        return SePerResult(
            question=question,
            seper_with_context=seper_with_context,
            seper_without_context=seper_without_context,
            delta_seper=delta_seper,
            computation_time=computation_time,
            context=context,
            num_responses=self.num_generations,
        )

    def evaluate_batch(
        self,
        questions: List[str],
        contexts: List[str],
        ground_truth_answers_list: List[List[str]],
        use_soft_clustering: bool = True,
    ) -> List[SePerResult]:
        """
        Evaluate a batch of question-context pairs.

        Params:
            questions (List[str]): List of questions
            contexts (List[str]): List of contexts
            ground_truth_answers_list (List[List[str]]): List of ground truth answer lists
            use_soft_clustering (bool): Whether to use soft or hard clustering

        Returns:
            List of SePerResult objects
        """
        results = []

        for i, (question, context, gt_answers) in enumerate(
            zip(questions, contexts, ground_truth_answers_list)
        ):
            logger.info(f"Processing batch item {i+1}/{len(questions)}")

            try:
                result = self.evaluate_single(
                    question=question,
                    context=context,
                    ground_truth_answers=gt_answers,
                    use_soft_clustering=use_soft_clustering,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating item {i+1}: {str(e)}")
                # Create error result
                error_result = SePerResult(
                    question=question,
                    seper_with_context=0.0,
                    seper_without_context=0.0,
                    delta_seper=0.0,
                    computation_time=0.0,
                    context=context,
                    num_responses=0,
                )
                results.append(error_result)

        return results

    def cleanup(self):
        """Clean up model resources."""
        logger.info("Cleaning up SePer evaluator resources")
        # Clear CUDA cache if using GPU
        if self.device != "cpu":
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
