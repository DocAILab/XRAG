import numpy as np
import torch
from typing import Any, Dict, List, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from ..config import Config
from ..utils import get_module_logger
from .utils import (
    OpenRAGTokens,
    get_scoring_components,
    load_special_tokens,
    postprocess_answer_option_conditioned,
)

logger = get_module_logger(__name__)


class _StopOnTokenIds(StoppingCriteria):
    def __init__(self, stop_ids: List[int]):
        super().__init__()
        self.stop_ids = set(int(i) for i in stop_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_token_id = int(input_ids[0, -1].item())
        return last_token_id in self.stop_ids


class OpenRAGPipeline:
    def __init__(self, config: Optional[Config] = None, external_retriever: Optional[Any] = None):
        self.config = config or Config()
        self.open_rag_cfg = self.config.config.get("open_rag", {})
        self.external_retriever = external_retriever

        self.default_mode = self.open_rag_cfg.get("mode", "adaptive_retrieval")
        self.default_n_docs = int(self.open_rag_cfg.get("n_docs", 3))
        self.default_max_new_tokens = int(self.open_rag_cfg.get("max_new_tokens", 100))
        self.default_threshold = self.open_rag_cfg.get("threshold", 0.0)
        self.default_closed = bool(self.open_rag_cfg.get("closed", False))

        self.use_groundness = bool(self.open_rag_cfg.get("use_groundness", True))
        self.use_utility = bool(self.open_rag_cfg.get("use_utility", True))
        self.use_seqscore = bool(self.open_rag_cfg.get("use_seqscore", True))
        self.use_stopping_criteria = bool(self.open_rag_cfg.get("use_stopping_criteria", True))
        self.w_rel = float(self.open_rag_cfg.get("w_rel", 1.0))
        self.w_sup = float(self.open_rag_cfg.get("w_sup", 1.0))
        self.w_use = float(self.open_rag_cfg.get("w_use", 0.5))

        self._init_model()
        self.ret_tokens, self.rel_tokens, self.grd_tokens, self.ut_tokens = load_special_tokens(
            self.tokenizer,
            use_groundness=self.use_groundness,
            use_utility=self.use_utility,
        )
        self.stopping_criteria = self._build_stopping_criteria()

    def _build_stopping_criteria(self) -> Optional[StoppingCriteriaList]:
        if not self.use_stopping_criteria:
            return None
        try:
            stop_ids = self.tokenizer("</s>", add_special_tokens=False)["input_ids"]
            if stop_ids:
                # Keep behavior close to the reference implementation: stop on the last token id.
                return StoppingCriteriaList([_StopOnTokenIds(stop_ids=[stop_ids[-1]])])
        except Exception:
            pass
        return None

    def _init_model(self):
        model_name = self.open_rag_cfg.get("model_name", "shayekh/openrag_llama2_7b_8x135m")
        trust_remote_code = bool(self.open_rag_cfg.get("trust_remote_code", True))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(f"Loading OpenRAG model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.device = device

    def _generate_with_logprobs(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k_logprobs: Optional[int] = 5000,
        full_vocab_first_step_only: bool = False,
    ) -> Dict[str, Any]:
        enc = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **enc,
                do_sample=False,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self.stopping_criteria,
            )

        prompt_len = enc.input_ids.shape[1]
        generated_ids = outputs.sequences[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        logprobs_list: List[Dict[int, float]] = []
        cumulative_logprob = 0.0
        for step, (scores_step, token_id) in enumerate(zip(outputs.scores, generated_ids)):
            probs_log = torch.log_softmax(scores_step[0], dim=-1)
            token_id = int(token_id.item())
            cumulative_logprob += float(probs_log[token_id].item())

            if top_k_logprobs is None and not (full_vocab_first_step_only and step > 0):
                step_dict = {int(i): float(v) for i, v in enumerate(probs_log.tolist())}
            else:
                k = min(max(int(top_k_logprobs or 5000), 1), probs_log.size(0))
                top_vals, top_idx = torch.topk(probs_log, k)
                step_dict = {int(i): float(v) for i, v in zip(top_idx.tolist(), top_vals.tolist())}
            logprobs_list.append(step_dict)

        return {
            "text": text,
            "logprobs": logprobs_list,
            "token_ids": generated_ids.tolist(),
            "cumulative_logprob": cumulative_logprob,
        }

    def _external_search(self, query: str, n_docs: int) -> List[Dict[str, Any]]:
        if self.external_retriever is None:
            return []
        try:
            nodes = self.external_retriever.retrieve(query)
        except AttributeError:
            base_ret = getattr(self.external_retriever, "_retriever", None)
            if base_ret is None:
                raise
            nodes = base_ret.retrieve(query)

        docs = []
        for i, nws in enumerate(nodes[:n_docs], start=1):
            node = getattr(nws, "node", nws)
            md = getattr(node, "metadata", {}) or {}
            docs.append(
                {
                    "id": md.get("id", str(i)),
                    "title": md.get("title", f"doc_{i}"),
                    "text": getattr(node, "text", None) or getattr(node, "get_content", lambda: "")(),
                    "score": float(getattr(nws, "score", 0.0) or 0.0),
                }
            )
        return docs

    def query(
        self,
        query: str,
        n_docs: Optional[int] = None,
        mode: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        threshold: Optional[float] = None,
        closed: Optional[bool] = None,
    ) -> Dict[str, Any]:
        n_docs = self.default_n_docs if n_docs is None else n_docs
        mode = self.default_mode if mode is None else mode
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        threshold = self.default_threshold if threshold is None else threshold
        closed = self.default_closed if closed is None else bool(closed)

        retrieved_docs = self._external_search(query, n_docs)
        evidences = [{"title": d.get("title", ""), "text": d.get("text", "")} for d in retrieved_docs]

        prompt = f"### Instruction:\n{query}\n\n### Response:\n"
        best_response, all_results, do_retrieve, detailed_init_score, overall_scores = (
            self._call_model_rerank_w_scores_batch(
                prompt,
                evidences,
                max_new_tokens,
                threshold,
                mode,
                closed,
            )
        )
        cleaned = postprocess_answer_option_conditioned(best_response)

        return {
            "query": query,
            "response": cleaned,
            "raw_response": best_response,
            "retrieved_documents": retrieved_docs,
            "retrieval_performed": do_retrieve,
            "mode": mode,
            "metadata": {
                "all_results": all_results,
                "detailed_init_score": detailed_init_score,
                "overall_scores": overall_scores,
            },
        }

    def _call_model_rerank_w_scores_batch(
        self,
        prompt: str,
        evidences: List[Dict[str, Any]],
        max_new_tokens: int,
        threshold: Optional[float],
        mode: str,
        closed: bool = False,
    ):
        no_ret = None
        detailed_init_score = {
            "logproba_retrieval_thresh": 0.0,
            "proba_retrieval_thresh": 0.0,
            "pred_retrieval_decision": "",
            "do_retrieve": None,
            "proba_r": 0.0,
            "logr": 0.0,
            "lognr": 0.0,
            "proba_nr": 0.0,
            "seq_logprob": 0.0,
        }

        if mode != "always_retrieve":
            no_ret = self._generate_with_logprobs(
                prompt,
                max_new_tokens=max_new_tokens,
                top_k_logprobs=None,
                full_vocab_first_step_only=True,
            )

        if mode == "always_retrieve":
            do_retrieve = True
        elif mode == "no_retrieval":
            do_retrieve = False
        else:
            pred_log_probs = no_ret.get("logprobs", []) if no_ret else []
            if pred_log_probs:
                first = pred_log_probs[0]
                log_r = float(first.get(self.ret_tokens[OpenRAGTokens.RETRIEVAL], -100.0))
                log_nr = float(first.get(self.ret_tokens[OpenRAGTokens.NO_RETRIEVAL], -100.0))
                r = np.exp(log_r)
                nr = np.exp(log_nr)
                denom = max(r + nr, 1e-12)
                do_retrieve = (
                    (r / denom) > threshold
                    if threshold is not None
                    else (OpenRAGTokens.RETRIEVAL in no_ret["text"])
                )

                log_denom = log_r + log_nr
                log_ratio = float(log_r / log_denom) if abs(log_denom) > 1e-12 else 0.0
                detailed_init_score.update(
                    {
                        "logproba_retrieval_thresh": log_ratio,
                        "proba_retrieval_thresh": float(r / denom),
                        "pred_retrieval_decision": no_ret.get("text", ""),
                        "proba_r": float(r),
                        "logr": float(log_r),
                        "lognr": float(log_nr),
                        "proba_nr": float(nr),
                        "seq_logprob": float(
                            no_ret.get("cumulative_logprob", 0.0)
                            / max(len(no_ret.get("token_ids", [])), 1)
                        ),
                    }
                )
            else:
                do_retrieve = True

        detailed_init_score["do_retrieve"] = bool(do_retrieve)
        overall_scores: Dict[str, Dict[str, float]] = {}

        if not do_retrieve or not evidences:
            no_ret_with_tag = self._generate_with_logprobs(
                prompt + OpenRAGTokens.NO_RETRIEVAL,
                max_new_tokens=max_new_tokens,
                top_k_logprobs=5000,
            )
            pred_text = no_ret_with_tag.get("text", "")
            results = {
                "no_retrieval": pred_text
            }
            return pred_text, results, False, detailed_init_score, overall_scores

        results = {}
        path2score: Dict[str, float] = {}
        for cand_idx, ev in enumerate(evidences):
            aug_prompt = (
                prompt
                + f"{OpenRAGTokens.RETRIEVAL}{OpenRAGTokens.PARAGRAPH_START}{ev['title']}\n{ev['text']}{OpenRAGTokens.PARAGRAPH_END}"
            )
            out = self._generate_with_logprobs(aug_prompt, max_new_tokens=max_new_tokens, top_k_logprobs=5000)
            text = out["text"]
            pred_log_probs = out.get("logprobs", [])
            pred_token_ids = out.get("token_ids", [])
            seq_logprob = float(out.get("cumulative_logprob", 0.0) / max(len(pred_token_ids), 1))
            seq_score = float(np.exp(seq_logprob))

            parsed = get_scoring_components(text)
            relevance_score = parsed["relevance_score"]
            ground_score = parsed["ground_score"] if self.use_groundness else 0.0
            utility_score = parsed["utility_score"] if self.use_utility else 0.0

            if self.use_groundness and self.grd_tokens and pred_log_probs and pred_token_ids:
                grd_pos = next(
                    (i for i, t in enumerate(pred_token_ids) if t in self.grd_tokens.values()),
                    None,
                )
                if grd_pos is not None and grd_pos < len(pred_log_probs):
                    grd_prob = {
                        k: np.exp(pred_log_probs[grd_pos].get(v, -100.0))
                        for k, v in self.grd_tokens.items()
                    }
                    grd_sum = max(sum(grd_prob.values()), 1e-12)
                    ground_score = (
                        grd_prob.get(OpenRAGTokens.FULLY_SUPPORTED, 0.0) / grd_sum
                        + 0.5 * grd_prob.get(OpenRAGTokens.PARTIALLY_SUPPORTED, 0.0) / grd_sum
                    )

            if self.use_utility and self.ut_tokens and pred_log_probs and pred_token_ids:
                ut_pos = next(
                    (i for i, t in enumerate(pred_token_ids) if t in self.ut_tokens.values()),
                    None,
                )
                if ut_pos is not None and ut_pos < len(pred_log_probs):
                    ut_prob = {
                        k: np.exp(pred_log_probs[ut_pos].get(v, -100.0))
                        for k, v in self.ut_tokens.items()
                    }
                    ut_sum = max(sum(ut_prob.values()), 1e-12)
                    ut_scores = [-1, -0.5, 0, 0.5, 1]
                    ut_tokens = [
                        OpenRAGTokens.UTILITY_1,
                        OpenRAGTokens.UTILITY_2,
                        OpenRAGTokens.UTILITY_3,
                        OpenRAGTokens.UTILITY_4,
                        OpenRAGTokens.UTILITY_5,
                    ]
                    utility_score = sum(
                        ut_scores[i] * (ut_prob.get(ut_tokens[i], 0.0) / ut_sum)
                        for i in range(len(ut_scores))
                    )

            if pred_log_probs:
                first = pred_log_probs[0]
                rel_prob = {k: np.exp(first.get(v, -100.0)) for k, v in self.rel_tokens.items()}
                rel_sum = max(sum(rel_prob.values()), 1e-12)
                relevance_score = rel_prob.get(OpenRAGTokens.RELEVANT, 0.0) / rel_sum

            final = (
                self.w_rel * relevance_score
                + (self.w_sup * ground_score if self.use_groundness else 0.0)
                + (self.w_use * utility_score if self.use_utility else 0.0)
            )
            if self.use_seqscore:
                final += seq_score

            if text == OpenRAGTokens.END:
                final = -1.0

            path_key = f"retrieval_{cand_idx}"
            results[path_key] = {"pred": text, "score": float(final), "ctx": ev}
            path2score[path_key] = float(final)
            overall_scores[path_key] = {
                "final_score": float(final),
                "relevance_score": float(relevance_score),
                "ground_score": float(ground_score),
                "utility_score": float(utility_score),
                "log_seq_score_penalty": float(seq_logprob),
                "seq_score_penalty": float(seq_score),
                "seq_logprob": float(seq_logprob),
                "weighted_relevance_score": float(self.w_rel * relevance_score),
                "weighted_ground_score": float((self.w_sup * ground_score) if self.use_groundness else 0.0),
                "weighted_utility_score": float((self.w_use * utility_score) if self.use_utility else 0.0),
            }

        if len(results) == 1:
            only_key = next(iter(results))
            return results[only_key]["pred"], results, True, detailed_init_score, overall_scores

        if closed:
            answer2score: Dict[str, float] = {}
            for key, item in results.items():
                if key == "no_retrieval":
                    continue
                answer = postprocess_answer_option_conditioned(item.get("pred", ""))
                answer2score[answer] = answer2score.get(answer, 0.0) + float(item.get("score", 0.0))
            best_text = max(answer2score.items(), key=lambda x: x[1])[0] if answer2score else ""
        else:
            best_path = max(path2score.items(), key=lambda x: x[1])[0] if path2score else ""
            best_text = results.get(best_path, {}).get("pred", "")

        return best_text, results, True, detailed_init_score, overall_scores
