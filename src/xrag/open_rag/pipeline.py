import numpy as np
import torch
from typing import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import Config
from ..utils import get_module_logger
from .utils import OpenRAGTokens, load_special_tokens, postprocess_answer_option_conditioned, get_scoring_components

logger = get_module_logger(__name__)


class OpenRAGPipeline:
    def __init__(self, config: Optional[Config] = None, external_retriever: Optional[Any] = None):
        self.config = config or Config()
        self.open_rag_cfg = self.config.config.get("open_rag", {})
        self.external_retriever = external_retriever

        self.default_mode = self.open_rag_cfg.get("mode", "adaptive_retrieval")
        self.default_n_docs = int(self.open_rag_cfg.get("n_docs", 3))
        self.default_max_new_tokens = int(self.open_rag_cfg.get("max_new_tokens", 100))
        self.default_threshold = self.open_rag_cfg.get("threshold", 0.0)

        self.use_groundness = bool(self.open_rag_cfg.get("use_groundness", True))
        self.use_utility = bool(self.open_rag_cfg.get("use_utility", True))
        self.use_seqscore = bool(self.open_rag_cfg.get("use_seqscore", True))
        self.w_rel = float(self.open_rag_cfg.get("w_rel", 1.0))
        self.w_sup = float(self.open_rag_cfg.get("w_sup", 1.0))
        self.w_use = float(self.open_rag_cfg.get("w_use", 0.5))

        self._init_model()
        self.ret_tokens, self.rel_tokens, self.grd_tokens, self.ut_tokens = load_special_tokens(
            self.tokenizer, use_groundness=self.use_groundness, use_utility=self.use_utility
        )


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

    def _generate_with_logprobs(self, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
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
            )

        prompt_len = enc.input_ids.shape[1]
        generated_ids = outputs.sequences[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        logprobs_list = []
        cumulative_logprob = 0.0
        for step, (scores_step, token_id) in enumerate(zip(outputs.scores, generated_ids)):
            probs_log = torch.log_softmax(scores_step[0], dim=-1)
            token_id = int(token_id.item())
            cumulative_logprob += float(probs_log[token_id].item())

            k = min(5000, probs_log.size(0))
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
            docs.append({
                "id": md.get("id", str(i)),
                "title": md.get("title", f"doc_{i}"),
                "text": getattr(node, "text", None) or getattr(node, "get_content", lambda: "")(),
                "score": float(getattr(nws, "score", 0.0) or 0.0),
            })
        return docs

    def query(
        self,
        query: str,
        n_docs: Optional[int] = None,
        mode: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        n_docs = self.default_n_docs if n_docs is None else n_docs
        mode = self.default_mode if mode is None else mode
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        threshold = self.default_threshold if threshold is None else threshold

        retrieved_docs = self._external_search(query, n_docs)
        evidences = [{"title": d.get("title", ""), "text": d.get("text", "")} for d in retrieved_docs]

        prompt = f"### Instruction:\n{query}\n\n### Response:\n"
        best_response, all_results, do_retrieve = self._call_model_rerank_w_scores_batch(
            prompt, evidences, max_new_tokens, threshold, mode
        )
        cleaned = postprocess_answer_option_conditioned(best_response)

        return {
            "query": query,
            "response": cleaned,
            "raw_response": best_response,
            "retrieved_documents": retrieved_docs,
            "retrieval_performed": do_retrieve,
            "mode": mode,
            "metadata": {"all_results": all_results},
        }

    def _call_model_rerank_w_scores_batch(self, prompt, evidences, max_new_tokens, threshold, mode):
        no_ret = None
        if mode != "always_retrieve":
            no_ret = self._generate_with_logprobs(prompt, max_new_tokens=max_new_tokens)

        if mode == "always_retrieve":
            do_retrieve = True
        elif mode == "no_retrieval":
            do_retrieve = False
        else:
            pred_log_probs = no_ret.get("logprobs", []) if no_ret else []
            if pred_log_probs:
                first = pred_log_probs[0]
                r = np.exp(first.get(self.ret_tokens[OpenRAGTokens.RETRIEVAL], -100.0))
                nr = np.exp(first.get(self.ret_tokens[OpenRAGTokens.NO_RETRIEVAL], -100.0))
                denom = max(r + nr, 1e-12)
                do_retrieve = (r / denom) > threshold if threshold is not None else (OpenRAGTokens.RETRIEVAL in no_ret["text"])
            else:
                do_retrieve = True

        if not do_retrieve or not evidences:
            if no_ret is None:
                no_ret = self._generate_with_logprobs(prompt + OpenRAGTokens.NO_RETRIEVAL, max_new_tokens=max_new_tokens)
            return no_ret["text"], {"no_retrieval": no_ret["text"]}, False

        results = {}
        best_text = ""
        best_score = -1e9
        for cand_idx, ev in enumerate(evidences):
            aug_prompt = prompt + f"{OpenRAGTokens.RETRIEVAL}{OpenRAGTokens.PARAGRAPH_START}{ev['title']}\n{ev['text']}{OpenRAGTokens.PARAGRAPH_END}"
            out = self._generate_with_logprobs(aug_prompt, max_new_tokens=max_new_tokens)
            text = out["text"]
            pred_log_probs = out.get("logprobs", [])
            pred_token_ids = out.get("token_ids", [])
            seq_score = np.exp(out.get("cumulative_logprob", 0.0) / max(len(pred_token_ids), 1))

            parsed = get_scoring_components(text)
            relevance_score = parsed["relevance_score"]
            ground_score = parsed["ground_score"] if self.use_groundness else 0.0
            utility_score = parsed["utility_score"] if self.use_utility else 0.0

            if self.use_groundness and self.grd_tokens and pred_log_probs and pred_token_ids:
                grd_pos = next((i for i, t in enumerate(pred_token_ids) if t in self.grd_tokens.values()), None)
                if grd_pos is not None and grd_pos < len(pred_log_probs):
                    d = {k: np.exp(pred_log_probs[grd_pos].get(v,-100.0)) for k, v in self.grd_tokens.items()}
                    s = max(sum(d.values()),1e-12)
                    ground_score = d["[Fully supported]"]/s + 0.5*d["[Partially supported]"]/s

            if self.use_utility and self.ut_tokens and pred_log_probs and pred_token_ids:
                ut_pos = next((i for i,t in enumerate(pred_token_ids) if t in self.ut_tokens.values()), None)
                if ut_pos is not None and ut_pos < len(pred_log_probs):
                    d = {k: np.exp(pred_log_probs[ut_pos].get(v, -100.0)) for k, v in self.ut_tokens.items()}
                    s = max(sum(d.values()), 1e-12)
                    weights = [-1, -0.5, 0, 0.5, 1]
                    toks = ["[Utility:1]","[Utility:2]","[Utility:3]","[Utility:4]","[Utility:5]"]
                    utility_score = sum(weights[i] * (d[toks[i]] / s) for i in range(5))        

            if pred_log_probs:
                first = pred_log_probs[0]
                rel = {k: np.exp(first.get(v, -100.0)) for k, v in self.rel_tokens.items()}
                s = max(sum(rel.values()), 1e-12)
                relevance_score = rel.get(OpenRAGTokens.RELEVANT, 0.0) / s

            final = (
                self.w_rel * relevance_score
                + (self.w_sup * ground_score if self.use_groundness else 0.0)
                + (self.w_use * utility_score if self.use_utility else 0.0)
            )
            if self.use_seqscore:
                final += float(seq_score)

            results[f"retrieval_{cand_idx}"] = {"pred": text, "score": float(final), "ctx": ev}
            if final > best_score:
                best_score = final
                best_text = text

        return best_text, results, True
  


