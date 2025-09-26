import torch
from ..llms import get_llm
from threading import Lock
from ..config import Config
from ..utils import get_module_logger
from llama_index.core import Settings
from llama_index.core.schema import Document
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .text_utils import extract_final_answer_and_rationale, parse_query
from .prompts import (
    force_answer_prompt,
    search_query_prompt,
    abstain_force_answer_prompt,
)

logger = get_module_logger(__name__)


class SimRAGPipeline:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.sim_rag_config: Dict[str, Any] = self.cfg.config.get("sim_rag", {})
        self.max_turns: int = int(self.sim_rag_config.get("max_turns", 4))
        self.top_k: int = int(self.sim_rag_config.get("top_k", 2))
        self.search_query_prompt_str: str = search_query_prompt
        self.reason_prompt_str: str = force_answer_prompt
        self.question_type: str = "OEQ"
        self.use_abstain_first_turn: bool = bool(
            self.sim_rag_config.get("use_abstain_first_turn", False)
        )
        self.remove_repeat_docs: bool = bool(
            self.sim_rag_config.get("remove_repeat_docs", False)
        )

        self.gate_model_name: Optional[str] = self.sim_rag_config.get("gate_model")
        self.gate_device: str = self.sim_rag_config.get("device", "auto")
        self.gate_weighted: bool = bool(self.sim_rag_config.get("weighted", False))
        self.gate_max_new_tokens: int = int(
            self.sim_rag_config.get("max_new_tokens", 1)
        )

        # Setup XRAG LLM and embeddings
        llm = get_llm(self.cfg.llm)
        Settings.llm = llm

        logger.info("Initialized SimRAGPipeline")
        if not self.gate_model_name:
            raise ValueError("sim_rag.gate.model must be set to enable SIM-RAG gating.")
        tok_name = self.gate_model_name
        self._gate_tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if self.gate_device == "auto":
            self._gate_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.gate_model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        else:
            self._gate_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.gate_model_name
            )
            self._gate_model.to(self.gate_device)

    def _gate_decide(
        self, task_content: str, answer: str, rationale: str
    ) -> Dict[str, Any]:
        instruction = "Instruction: Predict if the following answer to the question and context should be accepted, 1, or rejected, 0, based on the rationale."
        text = (
            f"{instruction}\n{task_content} \nAnswer: {answer}\nRationale: {rationale}"
        )
        tok = self._gate_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        device = next(self._gate_model.parameters()).device
        tok = {k: v.to(device) for k, v in tok.items()}
        if self.gate_weighted:
            gen = self._gate_model.generate(
                input_ids=tok["input_ids"],
                attention_mask=tok.get("attention_mask"),
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=self.gate_max_new_tokens,
            )
            resp = self._gate_tokenizer.batch_decode(
                gen.sequences, skip_special_tokens=True
            )[0]
            last_logits = gen.scores[0]
            probs = torch.nn.functional.softmax(last_logits, dim=-1)
            t0 = self._gate_tokenizer.encode("0", add_special_tokens=False)[0]
            t1 = self._gate_tokenizer.encode("1", add_special_tokens=False)[0]
            p0 = probs[0, t0].item()
            p1 = probs[0, t1].item()
            return {
                "accepted": resp.strip().endswith("1"),
                "confidence_0": p0,
                "confidence_1": p1,
            }
        else:
            gen = self._gate_model.generate(
                input_ids=tok["input_ids"], attention_mask=tok.get("attention_mask")
            )
            resp = self._gate_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            return {"accepted": resp.strip().endswith("1")}

    def _llm_predict(self, system_prompt: str, content: str) -> str:
        prompt = f"{system_prompt}\n\n{content}"
        try:
            return Settings.llm.predict(prompt)
        except Exception as e:
            logger.exception("LLM prediction failed")
            raise e

    def _retrieve(
        self, retriever: Any, query: str, top_k: int
    ) -> List[Tuple[str, str, Any]]:
        nodes = retriever.retrieve(query)
        results: List[Tuple[str, str, Any]] = []
        for n in nodes[:top_k]:
            try:
                title = (
                    n.node.metadata.get("title")
                    or n.node.metadata.get("file_name")
                    or ""
                )
            except Exception:
                title = ""
            results.append((title, n.node.get_content(), n))
        return results

    def answer(self, question: str, retriever: Any) -> Dict[str, Any]:
        task_content = f"Question: {question}\nContext:\n"
        retrieved_nodes: List[Any] = []
        retrieved_ids: List[Any] = []
        all_turns: List[Dict[str, Any]] = []
        seen_titles: List[str] = []

        for turn in range(self.max_turns + 1):
            # Reason step
            reason_prompt = (
                abstain_force_answer_prompt
                if (turn == 0 and self.use_abstain_first_turn)
                else self.reason_prompt_str
            )
            response = self._llm_predict(reason_prompt, task_content)
            answer, rationale = extract_final_answer_and_rationale(
                response, self.question_type
            )

            # Decide if stop via gate or max turn
            gate_meta = self._gate_decide(task_content, answer, rationale)
            if turn == self.max_turns or gate_meta.get("accepted", False):
                all_turns.append(
                    {
                        "turn": turn,
                        "answer": answer,
                        "rationale": rationale,
                        "query": "",
                        "retrieved": [],
                        "gate": gate_meta,
                    }
                )
                break

            # Generate search query
            sq_prompt = self.search_query_prompt_str
            sq = self._llm_predict(sq_prompt, task_content)
            parsed_query = parse_query(sq)

            # Retrieve
            top_k = self.top_k
            results = self._retrieve(retriever, parsed_query, top_k)
            retrieved_titles: List[str] = []
            retrieved_texts: List[str] = []
            kept_nodes: List[Any] = []
            for t, txt, node in results:
                if self.remove_repeat_docs and t and t in seen_titles:
                    continue
                retrieved_titles.append(t or "")
                retrieved_texts.append(txt)
                kept_nodes.append(node)
                if t:
                    seen_titles.append(t)

            # Append kept nodes and ids
            retrieved_nodes.extend(kept_nodes)
            for n in kept_nodes:
                rid = None
                try:
                    rid = n.node.metadata.get("id")
                except Exception:
                    pass
                if rid is None:
                    rid = getattr(n.node, "node_id", None)
                retrieved_ids.append(rid)

            # Update task content
            if retrieved_titles:
                retrieved_docs_content = "\n".join(
                    [
                        f"Title: {title} Content: {text}" if title else text
                        for title, text in zip(retrieved_titles, retrieved_texts)
                    ]
                )
                task_content = (
                    task_content
                    + f"Query: {parsed_query}\nRetrieved Document: {retrieved_docs_content}\n"
                )

            all_turns.append(
                {
                    "turn": turn,
                    "answer": answer,
                    "rationale": rationale,
                    "query": parsed_query,
                    "retrieved": [
                        f"Title: {title} Content: {text}" if title else text
                        for title, text in zip(retrieved_titles, retrieved_texts)
                    ],
                    "gate": gate_meta,
                }
            )

        final_answer = all_turns[-1]["answer"] if all_turns else ""
        return {
            "response": final_answer,
            "all_turns": all_turns,
            "retrieved_nodes": retrieved_nodes,
            "retrieved_ids": [rid for rid in retrieved_ids if rid is not None],
            "retrieved_texts": [n.node.get_content() for n in retrieved_nodes],
        }


_SIMRAG_PIPELINE_SINGLETON: Optional[SimRAGPipeline] = None
_SIMRAG_PIPELINE_LOCK = Lock()


def get_simrag_pipeline(cfg: Optional[Config] = None) -> SimRAGPipeline:
    global _SIMRAG_PIPELINE_SINGLETON
    with _SIMRAG_PIPELINE_LOCK:
        if _SIMRAG_PIPELINE_SINGLETON is None:
            _SIMRAG_PIPELINE_SINGLETON = SimRAGPipeline(cfg)
        else:
            # If a different cfg object is explicitly passed in, refresh the pipeline
            if cfg is not None and _SIMRAG_PIPELINE_SINGLETON.cfg is not cfg:
                _SIMRAG_PIPELINE_SINGLETON = SimRAGPipeline(cfg)
        return _SIMRAG_PIPELINE_SINGLETON


def run_simrag(
    question: str,
    documents: Optional[List[Document]] = None,
    cfg: Optional[Config] = None,
    retriever: Any = None,
) -> Dict[str, Any]:
    pipeline = get_simrag_pipeline(cfg)
    return pipeline.answer(question, documents=documents, retriever=retriever)
