from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.legacy.embeddings import HuggingFaceEmbedding


def _resolve_model_name(name):
    model_path = Path(name).expanduser()
    if model_path.exists():
        return str(model_path)

    try:
        cached_config = try_to_load_from_cache(name, "config.json")
    except (OSError, ValueError):
        cached_config = None

    if isinstance(cached_config, str):
        snapshot_path = Path(cached_config).parent
        if snapshot_path.is_dir():
            return str(snapshot_path)

    return name


def get_embedding(
    name,
    embed_batch_size=16,
    embedding_type="local",
    api_key=None,
    api_base=None,
    device=None,
):
    backend = (embedding_type or "local").strip().lower()
    if backend in {"local", "huggingface"}:
        kwargs = {"device": device} if device is not None else {}
        return HuggingFaceEmbedding(
            model_name=_resolve_model_name(name),
            embed_batch_size=embed_batch_size,
            **kwargs,
            # cache_folder="./embedding_model"
        )
    if backend == "openai":
        if not api_key:
            raise ValueError(
                "OpenAI-compatible embeddings require embedding_api_key in config.toml or the WebUI."
            )
        if not api_base:
            raise ValueError(
                "OpenAI-compatible embeddings require embedding_api_base in config.toml or the WebUI."
            )
        return OpenAIEmbedding(
            model_name=name,
            api_key=api_key,
            api_base=api_base.rstrip("/"),
            embed_batch_size=embed_batch_size,
        )
    raise ValueError(f"Unsupported embedding_type: {embedding_type}")
