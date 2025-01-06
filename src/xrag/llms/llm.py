from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from .huggingface_model import get_huggingfacellm
from ..config import Config


llm_dict = {
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "chatglm": "THUDM/chatglm3-6b",
    "qwen": "Qwen/Qwen1.5-7B-Chat",
    "qwen14_int8": "Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",
    "qwen7_int8": "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
    "qwen1.8": "Qwen/Qwen1.5-1.8B-Chat",
    "baichuan": "baichuan-inc/Baichuan2-7B-Chat",
    "falcon": "tiiuae/falcon-7b-instruct",
    "mpt": "mosaicml/mpt-7b-chat",
    "yi": "01-ai/Yi-6B-Chat",
}


def get_openai(api_base,api_key,api_name):
    return OpenAI(api_key=api_key,api_base=api_base, temperature=0,model=api_name)



def get_llm(name):
    if name == 'huggingface':
        return get_huggingfacellm(Config().huggingface_model)
    elif name == 'openai':
        return get_openai(Config().api_base,Config().api_key,Config().api_name)
    elif name == 'ollama':
        return Ollama(model=Config().ollama_model, request_timeout=60.0) 
    else:
        raise ValueError(f"no model name: {name}.")


