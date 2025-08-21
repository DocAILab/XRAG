"""SePer calculation utilities"""

import math
import random
import numpy as np
import torch.nn.functional as F
from ...utils import get_module_logger
from .models import SePerEntailmentModel
from typing import List, Dict, Any, Tuple

logger = get_module_logger(__name__)


def log_sum_exp_by_id(
    semantic_ids: List[int], log_likelihoods: List[float], agg: str = "sum_normalized"
) -> List[float]:
    """
    Sum probabilities with the same semantic id.
    Log-Sum-Exp because input and output probabilities in log space.

    Params:
        semantic_ids (List[int]): List of semantic cluster IDs for each response
        log_likelihoods (List[float]): List of log likelihoods for each response
        agg (str): Aggregation method ('sum_normalized')

    Returns:
        List of aggregated log likelihoods per cluster
    """
    log_likelihood_per_semantic_id = []

    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(
        range(len(unique_ids))
    ), "Semantic IDs must be consecutive integers starting from 0"

    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]

        if agg == "sum_normalized":
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise NotImplementedError(f"Aggregation method '{agg}' is not implemented.")

        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


class SePerCalculator:
    """Core SePer calculation engine."""

    def __init__(
        self, entailment_model: SePerEntailmentModel, computation_chunk_size: int = 128
    ):
        self.entailment_model = entailment_model
        self.computation_chunk_size = computation_chunk_size

    def calculate_seper_soft(
        self,
        questions: List[str],
        response_texts: List[List[str]],
        ground_truth_answers: List[List[str]],
        response_likelihoods: List[List[float]],
    ) -> List[float]:
        """
        Calculate SePer using soft clustering.

        Params:
            questions (List[str]): List of questions
            response_texts (List[List[str]]): List of response lists for each question
            ground_truth_answers (List[List[str]]): List of ground truth answer lists for each question
            response_likelihoods (List[List[float]]): List of likelihood lists for each question

        Returns:
            List of SePer scores for each question
        """
        batch_size = len(questions)
        seper_scores = []

        for batch_idx in range(batch_size):
            question = questions[batch_idx]
            responses = response_texts[batch_idx]
            gt_answers = ground_truth_answers[batch_idx]
            likelihoods = response_likelihoods[batch_idx]

            n_responses = len(responses)
            n_gt_answers = len(gt_answers)

            if n_gt_answers == 0:
                raise ValueError(
                    f"No ground truth answers provided for question: {question}"
                )

            # Prepare all response-answer pairs for entailment checking
            all_responses, all_answers_with_question = [], []
            for gt_answer in gt_answers:
                response_with_question = [f"{question} {resp}" for resp in responses]
                answer_with_question = f"{question} {gt_answer}"

                all_responses.extend(response_with_question)
                all_answers_with_question.extend([answer_with_question] * n_responses)

            # Batch entailment checking with chunking
            all_response_entails_answer, all_answer_entails_response = [], []
            for i in range(0, len(all_responses), self.computation_chunk_size):
                chunk_responses = all_responses[i : i + self.computation_chunk_size]
                chunk_answers = all_answers_with_question[
                    i : i + self.computation_chunk_size
                ]

                _, entailment_logits = self.entailment_model.check_entailment(
                    chunk_responses, chunk_answers, "loose"
                )

                # Extract entailment probabilities
                resp_entails_ans_probs = F.softmax(
                    entailment_logits["a_entails_b"], dim=-1
                )[:, 2]
                ans_entails_resp_probs = F.softmax(
                    entailment_logits["b_entails_a"], dim=-1
                )[:, 2]

                all_response_entails_answer.extend(
                    resp_entails_ans_probs.cpu().numpy().tolist()
                )
                all_answer_entails_response.extend(
                    ans_entails_resp_probs.cpu().numpy().tolist()
                )

            # Reshape entailment results
            response_entails_answer = np.array(all_response_entails_answer).reshape(
                n_responses, n_gt_answers
            )
            answer_entails_response = np.array(all_answer_entails_response).reshape(
                n_responses, n_gt_answers
            )

            likelihood_array = np.array(likelihoods)

            entailment_mass = np.stack(
                [
                    likelihood_array[:, np.newaxis] * response_entails_answer,
                    likelihood_array[:, np.newaxis] * answer_entails_response,
                ],
                axis=-1,
            )  # (n_responses, n_gt_answers, 2)

            max_entailment_per_pair = np.max(
                entailment_mass, axis=-1
            )  # (n_responses, n_gt_answers)
            total_entailment_per_answer = np.sum(
                max_entailment_per_pair, axis=0
            )  # (n_gt_answers,)
            seper_score = np.max(total_entailment_per_answer)
            seper_scores.append(float(seper_score))

        return seper_scores

    def calculate_seper_hard(
        self,
        question: str,
        responses: List[str],
        ground_truth_answers: List[str],
        response_log_likelihoods: List[List[float]],
        strict_entailment: bool = True,
    ) -> float:
        """
        Calculate SePer using hard clustering.

        Params:
            question (str): Single question (batch size must be 1!)
            responses (List[str]): List of response texts
            ground_truth_answers (List[str]): List of ground truth answers
            response_log_likelihoods (List[List[float]]): List of token log likelihood lists
            strict_entailment (bool): Whether to use strict ('bi') or loose entailment

        Returns:
            SePer score
        """
        n_gt_answers = len(ground_truth_answers)
        if n_gt_answers == 0:
            raise ValueError(
                f"No ground truth answers provided for question: {question}"
            )

        responses_with_question = [f"{question} {resp}" for resp in responses]

        log_likelihoods_agg = [
            np.mean(log_liks) for log_liks in response_log_likelihoods
        ]

        semantic_ids = self._cluster_responses_semantically(
            responses_with_question, strict_entailment
        )
        cluster_log_likelihoods = log_sum_exp_by_id(
            semantic_ids, log_likelihoods_agg, "sum_normalized"
        )

        n_clusters = len(set(semantic_ids))

        # Get representative text for each cluster
        cluster_representatives = []
        for cluster_id in range(n_clusters):
            for i, sid in enumerate(semantic_ids):
                if sid == cluster_id:
                    cluster_representatives.append(responses_with_question[i])
                    break

        cluster_entails_answers = self._check_cluster_answer_entailment(
            cluster_representatives, question, ground_truth_answers
        )

        seper_score = 0.0
        for cluster_id in range(n_clusters):
            if cluster_entails_answers[cluster_id]:
                seper_score += math.exp(cluster_log_likelihoods[cluster_id])

        return seper_score

    def _cluster_responses_semantically(
        self, responses: List[str], strict_entailment: bool = True
    ) -> List[int]:
        """
        Cluster responses by semantic equivalence.

        Params:
            responses (List[str]): List of response texts
            strict_entailment (bool): Whether to use strict or loose entailment

        Returns:
            List of cluster IDs for each response
        """
        n_responses = len(responses)
        semantic_ids = [-1] * n_responses
        next_id = 0

        entailment_type = "bi" if strict_entailment else "loose"

        for i, current_response in enumerate(responses):
            if semantic_ids[i] == -1:  # Unassigned
                semantic_ids[i] = next_id

                if i == n_responses - 1:
                    next_id += 1
                    break

                # Compare with remaining responses
                remaining_responses = responses[i + 1 :]
                current_responses = [current_response] * len(remaining_responses)

                equivalence_results, _ = self.entailment_model.check_entailment(
                    current_responses, remaining_responses, entailment_type
                )

                # Assign same cluster ID to equivalent responses
                for j, is_equivalent in enumerate(equivalence_results):
                    if is_equivalent:
                        semantic_ids[i + 1 + j] = next_id

                next_id += 1

        return semantic_ids

    def _check_cluster_answer_entailment(
        self,
        cluster_representatives: List[str],
        question: str,
        ground_truth_answers: List[str],
    ) -> List[bool]:
        """
        Check which clusters entail ground truth answers.

        Params:
            cluster_representatives (List[str]): Representative text for each cluster
            question (str): The original question
            ground_truth_answers (List[str]): List of ground truth answers

        Returns:
            List of boolean flags indicating which clusters entail answers
        """
        n_clusters = len(cluster_representatives)
        n_answers = len(ground_truth_answers)

        all_clusters = cluster_representatives * n_answers
        all_answers_with_question = []
        for answer in ground_truth_answers:
            all_answers_with_question.extend([f"{question} {answer}"] * n_clusters)

        entailment_results, _ = self.entailment_model.check_entailment(
            all_clusters, all_answers_with_question, "loose"
        )
        entailment_matrix = (
            entailment_results.reshape(n_clusters, n_answers).detach().cpu().numpy()
        )

        # A cluster entails answers if it entails at least one ground truth answer
        cluster_entails_answers = np.max(entailment_matrix, axis=1).astype(bool)

        return cluster_entails_answers.tolist()


def process_evaluation_data(
    question: str,
    context: str,
    responses_with_likelihoods: List[Tuple[str, List[float]]],
    ground_truth_answers: List[str],
    subsample: int = -1,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Process evaluation data into format expected by SePer calculator.

    Params:
        question (str): The question text
        context (str): Context information
        responses_with_likelihoods (List[Tuple[str, List[float]]]): List of (response_text, token_log_likelihoods) tuples
        ground_truth_answers (List[str]): List of ground truth answers
        subsample (int): Number of responses to subsample (-1 for all)
        random_seed (int): Random seed for subsampling

    Returns:
        Processed evaluation data dictionary
    """
    if subsample > 0 and len(responses_with_likelihoods) > subsample:
        random.seed(random_seed)
        responses_with_likelihoods = random.sample(
            responses_with_likelihoods, subsample
        )

    # Ensure question ends with '?'
    if not question.endswith("?"):
        question = question + "?"

    # Extract response texts and likelihoods
    response_texts = [resp[0] for resp in responses_with_likelihoods]
    response_log_likelihoods = [resp[1] for resp in responses_with_likelihoods]

    # Aggregate log likelihoods
    log_likelihoods_agg = [np.mean(log_liks) for log_liks in response_log_likelihoods]

    # Convert to normalized probabilities
    log_z = np.log(np.sum(np.exp(log_likelihoods_agg)))
    likelihoods = np.exp(np.array(log_likelihoods_agg) - log_z)

    return {
        "question": question,
        "context": context,
        "response_texts": response_texts,
        "response_log_likelihoods": response_log_likelihoods,
        "ground_truth_answers": ground_truth_answers,
        "likelihoods": likelihoods.tolist(),
        "log_likelihoods_agg": log_likelihoods_agg,
    }
