import argparse

import deepeval.api
import torch
from deepeval.models import GPTModel
from langchain_openai import ChatOpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from uptrain import Settings
from .DeepEvalLocalModel import DeepEvalLocalModel

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
    def __init__(self, args):
        self.args = args
        llamaIndex_LocalmodelName = self.args.llamaIndexEvaluateModel
        deepEval_LocalModelName = self.args.deepEvalEvaluateModel
        uptrain_LocalModelName = self.args.upTrainEvaluateModel
        api_name = self.args.api_name
        api_key = self.args.api_key
        api_base = self.args.api_base
        print("EvalModelName:")
        print(api_name)
        print("EvalModelAPI:")
        print(api_key)
        if api_name == "":
            self._llama_model = AutoModelForCausalLM.from_pretrained(llamaIndex_LocalmodelName,
                                                     torch_dtype=torch.float16,
                                                     device_map="auto").eval()
            self._llama_tokenizer = AutoTokenizer.from_pretrained(llamaIndex_LocalmodelName)
            load_tokenizer.append(self._llama_tokenizer)
            self.llamaModel = HuggingFaceLLM(context_window=llm_args["context_window"],
                              max_new_tokens=llm_args["max_new_tokens"],
                              completion_to_prompt=qwen_completion_to_prompt,
                              generate_kwargs=llm_args["generate_kwargs"],
                              model=self._llama_model,
                              tokenizer=self._llama_tokenizer,
                              device_map="cuda:0",)
        else:
            self.llamaModel = OpenAI(api_key=api_key, api_base=api_base,
                      model=api_name)
        if api_name == "":
            if deepEval_LocalModelName == llamaIndex_LocalmodelName:
                self._deepEval_model = self._llama_model
                self._deepEval_tokenizer = self._llama_tokenizer
            else:
                self._deepEval_model = AutoModelForCausalLM.from_pretrained(deepEval_LocalModelName,
                                                     torch_dtype=torch.float16,
                                                     device_map="auto").eval()
                self._deepEval_tokenizer = AutoTokenizer.from_pretrained(deepEval_LocalModelName)
            self.deepEvalModel = DeepEvalLocalModel(model=self._deepEval_model,
                                                    tokenizer=self._deepEval_tokenizer)
        else:
            # 不再有效了
            # deepeval.api.API_BASE_URL = 'https://uiuiapi.com/v1'
            # self.deepEvalModel = api_name

            self._deepEval_model = ChatOpenAI(openai_api_key=api_key, openai_api_base=api_base,
                      model_name=api_name)
            self.deepEvalModel = DeepEvalLocalModel(model=self._deepEval_model,
                                                    tokenizer="")
        if api_name == "":
            self.uptrainSetting = Settings(model="ollama/"+uptrain_LocalModelName)
        else:
            self.uptrainSetting = Settings(
                    model=api_name,
                    openai_api_key=api_key,
                    base_url=api_base,
                )