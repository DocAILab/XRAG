"""Compatibility import for the former Zhipu-specific embedding backend."""

from llama_index.embeddings.openai import OpenAIEmbedding


class OpenAICompatibleEmbeddings(OpenAIEmbedding):
    """Embedding client for services implementing OpenAI's /embeddings API."""

    @classmethod
    def class_name(cls) -> str:
        return "OpenAICompatibleEmbedding"


# Keep old imports working while routing them through the generic backend.
ChatGLMEmbeddings = OpenAICompatibleEmbeddings
