from __future__ import annotations

import asyncio

from .embedding import EmbeddingClient
from .repository import MovieRepository
from .vector_store import MovieVectorStore


class MovieIndexer:
    def __init__(
        self,
        repository: MovieRepository,
        embeddings: EmbeddingClient,
        store: MovieVectorStore,
        batch_size: int,
    ):
        self.repository = repository
        self.embeddings = embeddings
        self.store = store
        self.batch_size = batch_size
        self._lock = asyncio.Lock()

    async def reindex(self, *, recreate: bool = False) -> int:
        async with self._lock:
            if recreate:
                await asyncio.to_thread(self.store.reset_collection)

            records = await asyncio.to_thread(self.repository.fetch_all)
            total = 0

            for start in range(0, len(records), self.batch_size):
                batch = records[start : start + self.batch_size]

                # Each movie receives three independent dense representations.
                # The same query embedding is searched against all three fields.
                grouped_texts = [record.grouped_document() for record in batch]
                overview_texts = [record.overview_document() for record in batch]
                title_tagline_texts = [record.title_tagline_document() for record in batch]

                grouped_vectors, overview_vectors, title_tagline_vectors = await asyncio.gather(
                    self.embeddings.embed(grouped_texts),
                    self.embeddings.embed(overview_texts),
                    self.embeddings.embed(title_tagline_texts),
                )

                total += await asyncio.to_thread(
                    self.store.replace,
                    batch,
                    grouped_vectors,
                    overview_vectors,
                    title_tagline_vectors,
                )

            return total
