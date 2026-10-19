import argparse

from ..utils import get_module_logger

logger = get_module_logger(__name__)

load_tokenizer = []
llm_args = {"context_window": 4096, "max_new_tokens": 256,
                             "generate_kwargs": {"temperature": 0.7, "top_k": 50, "top_p": 0.95}}
def qwen_completion_to_prompt(completion):
    tokenizer = load_tokenizer[0]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": completion}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
class EvalModelAgent():
    def __init__(self, args, metrics=None):
        self.args = args
        selected_metrics = set(metrics if metrics is not None else getattr(args, "metrics", []))
        load_all = metrics is None and not selected_metrics
        needs_llama = load_all or any(metric.startswith("Llama_") for metric in selected_metrics)
        needs_deepeval = load_all or any(metric.startswith("DeepEval_") for metric in selected_metrics)
        needs_uptrain = load_all or any(metric.startswith("UpTrain_") for metric in selected_metrics)
        llamaIndex_LocalmodelName = self.args.llamaIndexEvaluateModel
        deepEval_LocalModelName = self.args.deepEvalEvaluateModel
        uptrain_LocalModelName = self.args.upTrainEvaluateModel
        api_name = self.args.api_name
        api_key = self.args.api_key
        api_base = self.args.api_base
        self.init_errors = {}
        logger.info("EvalModelName:")
        logger.info(api_name)
        logger.info("EvalModelAPI:")
        logger.info(api_key[0:4] + "***" + api_key[-4:])
        if needs_llama:
            try:
                self._init_llama(api_name, api_key, api_base, llamaIndex_LocalmodelName)
            except Exception as exc:
                logger.exception("LlamaIndex evaluator initialization failed")
                self.init_errors["Llama_"] = str(exc)
        if needs_deepeval:
            try:
                self._init_deepeval(api_name, api_key, api_base, deepEval_LocalModelName, llamaIndex_LocalmodelName)
            except Exception as exc:
                logger.exception("DeepEval evaluator initialization failed")
                self.init_errors["DeepEval_"] = str(exc)
        if needs_uptrain:
            try:
                self._init_uptrain(api_name, api_key, api_base, uptrain_LocalModelName)
            except Exception as exc:
                logger.exception("UpTrain evaluator initialization failed")
                self.init_errors["UpTrain_"] = str(exc)

    def _init_llama(self, api_name, api_key, api_base, model_name):
        if api_name == "":
            from ..llms.huggingface_model import get_huggingfacellm

            self.llamaModel = get_huggingfacellm(
                model_name,
                max_new_tokens=llm_args["max_new_tokens"],
                temperature=llm_args["generate_kwargs"]["temperature"],
            )
            self._llama_model = self.llamaModel.model
            self._llama_tokenizer = self.llamaModel.tokenizer
            load_tokenizer.append(self._llama_tokenizer)
        else:
            from llama_index.llms.openai import OpenAI

            self.llamaModel = OpenAI(api_key=api_key, api_base=api_base,
                      model=api_name)
    def _init_deepeval(self, api_name, api_key, api_base, model_name, llama_model_name):
        if api_name == "":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from .DeepEvalLocalModel import DeepEvalLocalModel

            if model_name == llama_model_name:
                if not (hasattr(self, "_llama_model") and hasattr(self, "_llama_tokenizer")):
                    self._llama_model = AutoModelForCausalLM.from_pretrained(
                        llama_model_name, torch_dtype=torch.float16, device_map="auto"
                    ).eval()
                    self._llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
                self._deepEval_model = self._llama_model
                self._deepEval_tokenizer = self._llama_tokenizer
            else:
                self._deepEval_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype=torch.float16,
                                                     device_map="auto").eval()
                self._deepEval_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.deepEvalModel = DeepEvalLocalModel(model=self._deepEval_model,
                                                    tokenizer=self._deepEval_tokenizer)
        else:
            from langchain_openai import ChatOpenAI
            from .DeepEvalLocalModel import DeepEvalLocalModel

            self._deepEval_model = ChatOpenAI(openai_api_key=api_key, openai_api_base=api_base,
                      model_name=api_name)
            self.deepEvalModel = DeepEvalLocalModel(model=self._deepEval_model,
                                                    tokenizer="")
    def _init_uptrain(self, api_name, api_key, api_base, model_name):
        if api_name == "":
            from uptrain import Settings

            self.uptrainSetting = Settings(model="ollama/"+model_name)
        else:
            from uptrain import Settings

            self.uptrainSetting = Settings(
                    model=api_name,
                    openai_api_key=api_key,
                    base_url=api_base,
                )
