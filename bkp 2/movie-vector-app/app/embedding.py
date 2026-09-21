from __future__ import annotations

from openai import AsyncOpenAI


class EmbeddingClient:
    def __init__(self, base_url: str, api_key: str, model: str, expected_dimension: int):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._expected_dimension = expected_dimension

    async def ping(self) -> bool:
        try:
            models = await self._client.models.list()
            return any(model.id == self._model for model in models.data)
        except Exception:
            return False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(model=self._model, input=texts)
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        for vector in vectors:
            if len(vector) != self._expected_dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {self._expected_dimension}, got {len(vector)}"
                )
        return vectors
