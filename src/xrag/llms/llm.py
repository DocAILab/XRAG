from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.ollama import Ollama
from .huggingface_model import LEGACY_MODEL_ALIASES, get_huggingfacellm
from ..config import Config


# Kept for callers that imported the old alias mapping.
llm_dict = LEGACY_MODEL_ALIASES


def get_openai(api_base, api_key, api_name, temperature):
    """Create an OpenAI-compatible chat LLM without a model-name allowlist."""
    return OpenAILike(
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        model=api_name,
        context_window=128_000,
        is_chat_model=True,
    )


def get_ollama(model, url, port, request_timeout):
    """Create an Ollama LLM using the configured server address."""
    url = (url or "http://localhost").strip().rstrip("/")
    if "://" not in url:
        url = f"http://{url}"
    return Ollama(
        model=model,
        base_url=f"{url}:{int(port)}",
        request_timeout=request_timeout,
    )



def get_llm(name):
    if name == 'huggingface':
        cfg = Config()
        return get_huggingfacellm(
            cfg.huggingface_model,
            auth_token=getattr(cfg, "auth_token", None),
            template=getattr(cfg, "huggingface_template", "auto"),
            context_window=getattr(cfg, "huggingface_context_window", 0),
            max_new_tokens=getattr(cfg, "huggingface_max_new_tokens", 512),
            temperature=getattr(cfg, "temperature", 0.0),
            trust_remote_code=getattr(cfg, "huggingface_trust_remote_code", False),
        )
    elif name == 'openai':
        return get_openai(api_base=Config().api_base, api_key=Config().api_key, temperature=Config().temperature, api_name=Config().api_name)
    elif name == 'ollama':
        cfg = Config()
        return get_ollama(
            model=cfg.ollama_model,
            url=getattr(cfg, "ollama_url", "http://localhost"),
            port=getattr(cfg, "ollama_port", 11434),
            request_timeout=cfg.ollama_request_timeout,
        )
    else:
        raise ValueError(f"no model name: {name}.")


