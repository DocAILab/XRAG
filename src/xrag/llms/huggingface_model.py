from functools import partial

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers.generation.utils import GenerationConfig
from llama_index.llms.huggingface import HuggingFaceLLM
# pip install llama-index-llms-huggingface
from ..config import Config
cfg = Config()

load_tokenizer = []


def llama_model_and_tokenizer(name, auth_token):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, token=auth_token, torch_dtype=torch.float16,
                                                 rope_scaling={"type": "dynamic", "factor": 2},
                                                 load_in_8bit=True, device_map="auto").eval()

    return tokenizer, model


def llama_completion_to_prompt(completion):
    return f"""<s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as 
        helpfully as possible, while being safe. Your answers should not include
        any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain 
        why instead of answering something not correct. If you don't know the answer 
        to a question, please don't share false information.

        Your goal is to provide answers relating to the financial performance of 
        the company.<</SYS>>
        {completion} [/INST]"""


def chatglm_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModel.from_pretrained(name, trust_remote_code=True).half().cuda().eval()

    return tokenizer, model


def chatglm_completion_to_prompt(completion):
    return "<|user|>\n " + completion + "<|assistant|>"


def qwen_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto").eval()

    return tokenizer, model


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


def baichuan_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, trust_remote_code=True)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto").eval()
    model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")

    return tokenizer, model


def baichuan_completion_to_prompt(completion):
    return "<reserved_106>" + completion + "<reserved_107>"  # "You are a helpful assistant.<reserved_106>" + completion + "<reserved_107>""


def falcon_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto").eval()

    return tokenizer, model


def falcon_completion_to_prompt(completion):
    return completion


def mpt_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto").eval()

    return tokenizer, model


def mpt_completion_to_prompt(completion):
    return completion


def yi_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    return tokenizer, model


def yi_completion_to_prompt(completion):
    return "<|im_start|> user\n" + completion + "<|im_end|> \n<|im_start|>assistant\n"


tokenizer_and_model_fn_dict = {
    "meta-llama/Llama-2-7b-chat-hf": partial(llama_model_and_tokenizer, auth_token=cfg.auth_token),
    "THUDM/chatglm3-6b": chatglm_model_and_tokenizer,
    "Qwen/Qwen1.5-7B-Chat": qwen_model_and_tokenizer,
    "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8": qwen_model_and_tokenizer,
    "baichuan-inc/Baichuan2-7B-Chat": baichuan_model_and_tokenizer,
    "tiiuae/falcon-7b-instruct": falcon_model_and_tokenizer,
    "mosaicml/mpt-7b-chat": mpt_model_and_tokenizer,
    "01-ai/Yi-6B-Chat": yi_model_and_tokenizer,
}

completion_to_prompt_dict = {
    "meta-llama/Llama-2-7b-chat-hf": llama_completion_to_prompt,
    "THUDM/chatglm3-6b": chatglm_completion_to_prompt,
    "Qwen/Qwen1.5-7B-Chat": qwen_completion_to_prompt,
    "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8": qwen_completion_to_prompt,
    "baichuan-inc/Baichuan2-7B-Chat": baichuan_completion_to_prompt,
    "tiiuae/falcon-7b-instruct": falcon_completion_to_prompt,
    "mosaicml/mpt-7b-chat": mpt_completion_to_prompt,
    "01-ai/Yi-6B-Chat": yi_completion_to_prompt,
}

# llm_argument_dict = {
#     "meta-llama/Llama-2-7b-chat-hf": {"context_window": 4096, "max_new_tokens": 256,
#                                       "generate_kwargs": {"temperature": 0.7, "top_k": 50, "top_p": 0.95}},
#     "THUDM/chatglm3-6b": {"context_window": 4096, "max_new_tokens": 256,
#                           "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95,
#                                               "eos_token_id": [2, 64795, 64797]}},
#     "Qwen/Qwen1.5-7B-Chat": {"context_window": 4096, "max_new_tokens": 256,
#                              "generate_kwargs": {"temperature": 0.7, "top_k": 50, "top_p": 0.95}},
#     "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8": {"context_window": 4096, "max_new_tokens": 256,
#                              "generate_kwargs": {"temperature": 0.7, "top_k": 50, "top_p": 0.95}},
#     "baichuan-inc/Baichuan2-7B-Chat": {"context_window": 4096, "max_new_tokens": 256, "generate_kwargs": None},
#     "tiiuae/falcon-7b-instruct": {"context_window": 4096, "max_new_tokens": 256,
#                                   "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50,
#                                                       "top_p": 0.95}},
#     "mosaicml/mpt-7b-chat": {"context_window": 4096, "max_new_tokens": 256,
#                              "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}},
#     "01-ai/Yi-6B-Chat": {"context_window": 4096, "max_new_tokens": 256,
#                          "generate_kwargs": {"temperature": 0.7, "top_k": 50, "top_p": 0.95}},
# }

llm_argument_dict = {
    "meta-llama/Llama-2-7b-chat-hf": {"context_window": 4096, "generate_kwargs": {"temperature": 0}},
    "THUDM/chatglm3-6b": {"context_window": 4096, "generate_kwargs": {"temperature": 0, "eos_token_id": [2, 64795, 64797]}},
    "Qwen/Qwen1.5-7B-Chat": {"context_window": 4096, "generate_kwargs": {"temperature": 0}},
    "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8": {"context_window": 4096, "generate_kwargs": {"temperature": 0}},
    "baichuan-inc/Baichuan2-7B-Chat": {"context_window": 4096, "generate_kwargs": {"temperature": 0}},
    "tiiuae/falcon-7b-instruct": {"context_window": 4096, "generate_kwargs": {"temperature": 0}},
    "mosaicml/mpt-7b-chat": {"context_window": 4096, "generate_kwargs": {"temperature": 0}},
    "01-ai/Yi-6B-Chat": {"context_window": 4096, "generate_kwargs": {"temperature": 0}},
}

def get_huggingfacellm(name):
    print("name is " + name)
    tokenizer, model = tokenizer_and_model_fn_dict[name](name)

    # Create a HF LLM using the llama index wrapper
    llm = HuggingFaceLLM(context_window=llm_argument_dict[name]["context_window"],
                         max_new_tokens=llm_argument_dict[name]["max_new_tokens"],
                         completion_to_prompt=completion_to_prompt_dict[name],
                         generate_kwargs=llm_argument_dict[name]["generate_kwargs"],
                         model=model,
                         tokenizer=tokenizer,
                         device_map="auto", )
    return llm
