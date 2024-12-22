from typing import Optional, List, Mapping, Any, Sequence, Dict
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from zhipuai import ZhipuAI

class ChatGLMEmbeddings(BaseEmbedding):
    model: str = Field(default='embedding-2', description="The ChatGlM model to use. embedding-2")
    api_key: str = Field(default=None, description="The ChatGLM API key.")
    reuse_client: bool = Field(default=True, description=(
            "Reuse the client between requests. When doing anything with large "
            "volumes of async API calls, setting this to false can improve stability."
        ),
    )

    _client: Optional[Any] = PrivateAttr()
    def __init__(
        self,
        model: str = 'embedding-2',
        reuse_client: bool = True,
        api_key: Optional[str] = None,
        **kwargs: Any,
    )-> None:
        super().__init__(
            model=model,
            api_key=api_key,
            reuse_client=reuse_client,
            **kwargs,
        )
        self._client = None

    def _get_client(self) -> ZhipuAI:
        if not self.reuse_client :
            return ZhipuAI(api_key=self.api_key)

        if self._client is None:
            self._client = ZhipuAI(api_key=self.api_key)
        return self._client

    @classmethod
    def class_name(cls) -> str:
        return "ChatGLMEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.get_general_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return self.get_general_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.get_general_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return self.get_general_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.get_general_text_embedding(text)
            embeddings_list.append(embeddings)

        return embeddings_list

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return self._get_text_embeddings(texts)

    def get_general_text_embedding(self, prompt: str) -> List[float]:
        response = self._get_client().embeddings.create(
            model=self.model, #填写需要调用的模型名称
            input=prompt,
        )
        return response.data[0].embedding
