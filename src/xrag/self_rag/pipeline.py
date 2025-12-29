import numpy as np
from ..config import Config
from .model import SelfRAGModel
from ..utils import get_module_logger
from .retriever import SelfRAGRetriever
from typing import List, Dict, Any, Optional
from .utils import SelfRAGResponseParser, SelfRAGTokens

logger = get_module_logger(__name__)


class SelfRAGPipeline:
    """
    Complete Self-RAG pipeline.
    """

    def __init__(self, config: Optional[Config] = None, external_retriever: Optional[Any] = None):
        """
        Initialize the Self-RAG pipeline.

        Params:
            config (Config, optional): Configuration object. If None, uses global config.
        """
        self.config = config or Config()
        self.self_rag_config = self.config.config.get("self_rag", {})

        self.model = SelfRAGModel(self.config)
        self.retriever = SelfRAGRetriever(self.config)
        self.external_retriever = external_retriever
        self.parser = SelfRAGResponseParser()

        self.default_n_docs = 100
        self.retrieval_threshold = 0.5
        self.max_new_tokens = 15

        self.w_rel = 1.0  # relevance weight
        self.w_sup = 1.0  # support weight
        self.w_use = 0.5  # utility weight
        self.use_seqscore = False

        logger.info("Initialized SelfRAGPipeline (external retriever: %s)" % (self.external_retriever is not None))

    def setup_retriever(self, passages_path: str, embeddings_path: str = None):
        """
        Setup the retrieval component.

        Params:
            passages_path (str): Path to passages file
            embeddings_path (str, optional): Path to precomputed embeddings (optional)
        """
        logger.info("Setting up retriever...")
        try:
            self.retriever.setup_index(passages_path, embeddings_path)
            logger.info("Retriever setup completed successfully")
        except:
            logger.warning("Retriever setup incomplete. Some features may not work.")

    def query(
        self,
        query: str,
        n_docs: int = None,
        mode: str = "adaptive_retrieval",
        max_new_tokens: int = None,
        threshold: float = None,
    ) -> Dict[str, Any]:
        """
        Process a single query through the Self-RAG pipeline.

        Params:
            query (str): User query
            n_docs (int, optional): Number of documents to retrieve
            mode (str, optional): Retrieval mode ('adaptive_retrieval', 'always_retrieve', 'no_retrieval')
            max_new_tokens (int, optional): Maximum tokens to generate
            threshold (float, optional): Threshold for adaptive retrieval

        Returns:
            Dictionary containing the response and metadata
        """
        if n_docs is None:
            n_docs = self.default_n_docs
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if threshold is None:
            threshold = self.retrieval_threshold

        logger.info(f"Processing query: {query[:100]}...")

        # Get evidences/documents for retrieval
        retrieved_docs = []
        evidences = []

        if self.external_retriever is not None:
            try:
                retrieved_docs = self._external_search(query, n_docs)
            except Exception as e:
                logger.warning(f"External retriever failed ({e}); falling back to internal SelfRAG retriever")
                retrieved_docs = self.retriever.search_documents(query, n_docs)
        else:
            retrieved_docs = self.retriever.search_documents(query, n_docs)
        evidences = [
            {"title": doc.get("title", ""), "text": doc.get("text", "")}
            for doc in retrieved_docs
        ]
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        best_response, all_results, do_retrieve = (
            self._call_model_rerank_w_scores_batch(
                query, evidences, max_new_tokens, threshold, mode
            )
        )

        # Parse the best response for additional metadata
        parsed_best_response = self.parser.parse_response(best_response)

        return {
            "query": query,
            "response": best_response,
            "raw_response": best_response,
            "retrieved_documents": retrieved_docs,
            "retrieval_performed": do_retrieve,
            "mode": mode,
            "metadata": {
                "all_results": all_results,
                "retrieval_performed": do_retrieve,
                "parsed_response": {
                    "relevance": parsed_best_response["relevance"],
                    "support": parsed_best_response["support"],
                    "utility_score": parsed_best_response["utility_score"],
                    "special_tokens": parsed_best_response["special_tokens"],
                },
            },
            "all_responses": all_results,
        }

    def _external_search(self, query: str, n_docs: int) -> List[Dict[str, Any]]:
        """Run retrieval through provided external retriever (llama_index retriever)."""
        try:
            nodes = self.external_retriever.retrieve(query)  # llama_index retriever
        except AttributeError:
            base_ret = getattr(self.external_retriever, "_retriever", None)
            if base_ret is None:
                raise
            nodes = base_ret.retrieve(query)

        docs = []
        for rank, node_with_score in enumerate(nodes[:n_docs], start=1):
            node = getattr(node_with_score, "node", node_with_score)
            metadata = getattr(node, "metadata", {}) or {}
            doc_id = metadata.get("id", getattr(node, "node_id", str(rank)))
            title = metadata.get("title", f"doc_{doc_id}")
            text = getattr(node, "text", None) or getattr(node, "get_content", lambda: "")()
            docs.append({
                "id": str(doc_id),
                "title": title,
                "text": text,
                "score": float(getattr(node_with_score, "score", 1.0)),
                "rank": rank,
            })
        return docs

    def _call_model_rerank_w_scores_batch(
        self,
        prompt: str,
        evidences: List[Dict],
        max_new_tokens: int = 15,
        threshold: float = 0.5,
        mode: str = "adaptive_retrieval",
        closed: bool = False,
    ) -> tuple:
        """
        Implements adaptive retrieval, scoring, and response selection.

        Params:
            prompt (str): Input prompt/query
            evidences (List[Dict]): Retrieved documents with 'title' and 'text' keys
            max_new_tokens (int): Maximum tokens to generate
            threshold (float): Threshold for adaptive retrieval decision
            mode (str): Retrieval mode
            closed (bool): TODO

        Returns:
            Tuple of (best_response, all_results, do_retrieve)
        """
        results = {}

        ret_tokens, rel_tokens, grd_tokens, ut_tokens = self.model.get_special_tokens()

        if mode != "always_retrieve":
            no_retrieval_response = self.model.generate_with_logprobs(
                prompt, max_new_tokens=max_new_tokens
            )
            results["no_retrieval"] = no_retrieval_response["text"]

            pred_log_probs = no_retrieval_response.get("logprobs", [])
            if pred_log_probs:
                first_token_logprobs = pred_log_probs[0] if pred_log_probs else {}
                score_dict = {}
                for tok, token_id in ret_tokens.items():
                    if token_id not in first_token_logprobs:
                        score_dict[tok] = -100
                    else:
                        prob = first_token_logprobs[token_id]
                        score_dict[tok] = float(prob)

                if threshold is not None:
                    retrieval_score = score_dict.get(
                        SelfRAGTokens.RETRIEVAL.value, -100
                    )
                    no_retrieval_score = score_dict.get(
                        SelfRAGTokens.NO_RETRIEVAL.value, -100
                    )
                    do_retrieve = (
                        retrieval_score / (retrieval_score + no_retrieval_score)
                        > threshold
                    )
                else:
                    do_retrieve = (
                        SelfRAGTokens.RETRIEVAL.value in no_retrieval_response["text"]
                    )
            else:
                do_retrieve = True
        else:
            do_retrieve = mode == "always_retrieve"

        if do_retrieve and evidences:
            evidence_augmented_inputs = [
                prompt
                + f"[Retrieval]<paragraph>{para['title']}\n{para['text']}</paragraph>"
                for para in evidences
            ]

            overall_scores = {}
            for p_idx, augmented_prompt in enumerate(evidence_augmented_inputs):
                response = self.model.generate_with_logprobs(
                    augmented_prompt, max_new_tokens=max_new_tokens
                )

                pred_text = response["text"]
                pred_log_probs = response.get("logprobs", [])
                pred_token_ids = response.get("token_ids", [])
                cumulative_logprob = response.get("cumulative_logprob", 0.0)

                # Calculate sequence score
                seq_score = cumulative_logprob / max(len(pred_token_ids), 1)

                # Parse response
                parsed_response = self.parser.parse_response(pred_text)
                scoring_components = self.parser.get_scoring_components(pred_text)

                relevance_score_dict = {}
                grd_score_dict = {}
                ut_score_dict = {}

                # Primary scoring from logprobs
                if pred_log_probs and rel_tokens:
                    # Relevance tokens use first position (index 0)
                    first_token_logprobs = pred_log_probs[0] if pred_log_probs else {}
                    for tok, token_id in rel_tokens.items():
                        prob = first_token_logprobs.get(token_id, -100)
                        relevance_score_dict[tok] = np.exp(float(prob))

                if grd_tokens and pred_log_probs and pred_token_ids:
                    # Grounding tokens: search for where they appear in the sequence
                    groundness_token_appear_indices = []
                    for tok_idx, tok in enumerate(pred_token_ids):
                        if tok in list(grd_tokens.values()):
                            groundness_token_appear_indices.append(tok_idx)
                            break
                    if len(groundness_token_appear_indices) > 0:
                        idx = groundness_token_appear_indices[0]
                        if idx < len(pred_log_probs):
                            token_logprobs = (
                                pred_log_probs[idx] if pred_log_probs[idx] else {}
                            )
                            for token, token_id in grd_tokens.items():
                                prob = token_logprobs.get(token_id, -100)
                                grd_score_dict[token] = np.exp(float(prob))

                if ut_tokens and pred_log_probs and pred_token_ids:
                    # Utility tokens: search for where they appear in the sequence
                    utility_token_appear_indices = []
                    for tok_idx, tok in enumerate(pred_token_ids):
                        if tok in list(ut_tokens.values()):
                            utility_token_appear_indices.append(tok_idx)
                            break
                    if len(utility_token_appear_indices) > 0:
                        idx = utility_token_appear_indices[0]
                        if idx < len(pred_log_probs):
                            token_logprobs = (
                                pred_log_probs[idx] if pred_log_probs[idx] else {}
                            )
                            for token, token_id in ut_tokens.items():
                                prob = token_logprobs.get(token_id, -100)
                                ut_score_dict[token] = np.exp(float(prob))

                # Calculate scores from logprobs
                relevance_score = 0.0
                if relevance_score_dict:
                    total_rel = sum(relevance_score_dict.values())
                    if total_rel > 0:
                        relevance_score = (
                            relevance_score_dict.get(SelfRAGTokens.RELEVANT.value, 0)
                            / total_rel
                        )
                else:
                    relevance_score = scoring_components["relevance_score"]

                ground_score = 0.0
                if len(grd_score_dict) == 3:
                    gt_sum = sum(grd_score_dict.values())
                    if gt_sum > 0:
                        ground_score = (
                            grd_score_dict.get("[Fully supported]", 0) / gt_sum
                            + 0.5
                            * grd_score_dict.get("[Partially supported]", 0)
                            / gt_sum
                        )
                else:
                    ground_score = scoring_components["ground_score"]

                utility_score = 0.0
                if len(ut_score_dict) == 5:
                    ut_sum = sum(ut_score_dict.values())
                    if ut_sum > 0:
                        ut_scores = [-1, -0.5, 0, 0.5, 1]
                        utility_score = sum(
                            ut_scores[i]
                            * ut_score_dict.get(f"[Utility:{i+1}]", 0)
                            / ut_sum
                            for i in range(len(ut_scores))
                        )
                else:
                    utility_score = scoring_components["utility_score"]

                if self.use_seqscore:
                    final_score = (
                        np.exp(seq_score)
                        + self.w_rel * relevance_score
                        + self.w_sup * ground_score
                        + self.w_use * utility_score
                    )
                else:
                    final_score = (
                        self.w_rel * relevance_score
                        + self.w_sup * ground_score
                        + self.w_use * utility_score
                    )

                overall_scores[p_idx] = {
                    "final_score": final_score,
                    "relevance_score": relevance_score,
                    "ground_score": ground_score,
                    "utility_score": utility_score,
                }

                results[f"retrieval_{p_idx}"] = {
                    "pred": pred_text,
                    "score": final_score,
                    "ctx": evidences[p_idx],
                    "parsed_metadata": {
                        "relevance": parsed_response["relevance"],
                        "support": parsed_response["support"],
                        "utility_score": parsed_response["utility_score"],
                        "special_tokens": parsed_response["special_tokens"],
                        "content": parsed_response["content"],
                    },
                    "scores": {
                        "relevance_score": relevance_score,
                        "ground_score": ground_score,
                        "utility_score": utility_score,
                        "final_score": final_score,
                    },
                }

        elif not do_retrieve:
            prompt_no_retrieval = prompt + SelfRAGTokens.NO_RETRIEVAL.value
            response = self.model.generate_single(prompt_no_retrieval)
            # Parse the no-retrieval response
            parsed_response = self.parser.parse_response(response)
            results["no_retrieval"] = {
                "pred": response,
                "parsed_metadata": {
                    "relevance": parsed_response["relevance"],
                    "support": parsed_response["support"],
                    "utility_score": parsed_response["utility_score"],
                    "special_tokens": parsed_response["special_tokens"],
                    "content": parsed_response["content"],
                },
            }

        if len(results) == 1:
            key = list(results.keys())[0]
            if key == "no_retrieval":
                result_data = results[key]
                if isinstance(result_data, dict):
                    raw_response = result_data.get("pred", "")
                    parsed_metadata = result_data.get("parsed_metadata", {})
                    postprocessed_pred = parsed_metadata.get("content", raw_response)
                else:
                    # String format
                    raw_response = result_data
                    parsed_response = self.parser.parse_response(raw_response)
                    postprocessed_pred = parsed_response["content"]
                    # Update result to include metadata
                    results[key] = {
                        "pred": raw_response,
                        "parsed_metadata": {
                            "relevance": parsed_response["relevance"],
                            "support": parsed_response["support"],
                            "utility_score": parsed_response["utility_score"],
                            "special_tokens": parsed_response["special_tokens"],
                            "content": parsed_response["content"],
                        },
                    }
            else:
                # Retrieval result
                result_data = results[key]
                raw_response = result_data.get("pred", "")
                parsed_metadata = result_data.get("parsed_metadata", {})
                postprocessed_pred = parsed_metadata.get("content", raw_response)

            return postprocessed_pred, results, do_retrieve
        else:
            if closed:
                answer2score = {}
                for key, result in results.items():
                    if key == "no_retrieval":
                        continue
                    raw_pred = result["pred"]
                    parsed_metadata = result.get("parsed_metadata", {})
                    answer = parsed_metadata.get("content", raw_pred)
                    score = result["score"]

                    answer2score.setdefault(answer, 0)
                    answer2score[answer] += score

                if answer2score:
                    sorted_answers = sorted(
                        answer2score.items(), key=lambda x: x[1], reverse=True
                    )
                    best_option = sorted_answers[0][0]
                else:
                    best_option = ""
            else:
                path2score = {
                    key: item["score"]
                    for key, item in results.items()
                    if key != "no_retrieval"
                    and isinstance(item, dict)
                    and "score" in item
                }
                if path2score:
                    best_path = sorted(
                        path2score.items(), key=lambda x: x[1], reverse=True
                    )[0][0]
                    result_data = results[best_path]
                    parsed_metadata = result_data.get("parsed_metadata", {})
                    best_option = parsed_metadata.get(
                        "content", result_data.get("pred", "")
                    )
                else:
                    # Handle no_retrieval case
                    no_retrieval_result = results.get("no_retrieval", "")
                    if isinstance(no_retrieval_result, dict):
                        parsed_metadata = no_retrieval_result.get("parsed_metadata", {})
                        best_option = parsed_metadata.get(
                            "content", no_retrieval_result.get("pred", "")
                        )
                    else:
                        # Legacy string format
                        parsed_response = self.parser.parse_response(
                            no_retrieval_result
                        )
                        best_option = parsed_response["content"]

            return best_option, results, do_retrieve
