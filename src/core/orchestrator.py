import logging
from typing import Any, Dict, Optional
from src.services.ollama_service import OllamaService
from src.embeddings.embedder import Embedder
from src.caching.normalizer import Normalizer
from src.caching.persister import Persister
from src.caching.selector import Selector
from src.caching.evaluator import Evaluator
from src.caching.builder import Builder
from src.services.redis_service import RedisService

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Interface for coordinating the complete semantic caching workflow, including
    normalization, hashing, embedding generation, exact and semantic lookup,
    candidate evaluation, LLM fallback, and persistence of newly generated records.

    Attributes:
        normalizer (Normalizer): Component responsible for deterministic text normalization and hash computation.
        embedder (Embedder): Service used to generate embedding vectors for queries.
        persister (Persister): Handles retrieval, search, and storage of cache records.
        selector (Selector): Ranks and selects the best semantic candidate.
        evaluator (Evaluator): Validates cache candidates via TTL, confidence, and payload integrity checks.
        builder (Builder): Constructs new cache records for persistence.
        ollama (OllamaService): LLM backend used for fallback generation on cache MISS.

    Methods:
        query(text: str) -> str:
            Executes the full semantic cache pipeline for the given input text,
            returning either a cached answer or a newly generated LLM response.
    """

    def __init__(self) -> None:
        logger.info("Initializing semantic cache Orchestrator.")
        self.normalizer: Normalizer = Normalizer()
        self.embedder: Embedder = Embedder(OllamaService())
        self.persister: Persister = Persister(RedisService())
        self.selector: Selector = Selector()
        self.evaluator: Evaluator = Evaluator()
        self.builder: Builder = Builder()
        self.ollama: OllamaService = OllamaService()
        logger.info("Semantic cache Orchestrator initialized successfully.")

    async def query(self, text: str) -> str:
        logger.info("Starting semantic cache query workflow.")

        processed: Dict[str, str] = self.normalizer.process(text)
        normalized: str = processed["normalized"]
        query_hash: str = processed["query_hash"]

        logger.info("Checking exact match.")
        exact: Optional[Dict[str, Any]] = await self.persister.fetch_by_hash(query_hash)
        if exact and self.evaluator.is_valid(exact):
            logger.info("Exact match HIT.")
            return exact["payload"]["answer"]

        logger.info("Computing embedding for semantic search.")
        vector = await self.embedder.embed(normalized)

        logger.info("Searching semantic candidates.")
        semantic_candidates = await self.persister.search_by_vector(vector)
        best = self.selector.select(semantic_candidates)

        if best:
            logger.info(f"Semantic candidate selected: id={best.get('id')} score={best.get('score')}")
            clean_id: str = best["id"].replace("vec:", "")
            logger.info(f"Fetching full record for semantic candidate: {clean_id}")
            full: Optional[Dict[str, Any]] = await self.persister.fetch_by_hash(clean_id)

            if full:
                full["score"] = best["score"]
                if self.evaluator.is_valid(full):
                    logger.info("Semantic match HIT.")
                    return full["payload"]["answer"]

        logger.info("Cache MISS. Querying LLM.")
        llm_response: str = await self.ollama.chat(text)

        logger.info("Building new cache record.")
        record: Dict[str, Any] = self.builder.build(
            query_hash=query_hash,
            normalized=normalized,
            vector=vector,
            response=llm_response
        )

        logger.info("Saving new cache record.")
        await self.persister.save(record, ttl=record["payload"]["ttl"])

        logger.info("Semantic cache workflow completed.")
        return llm_response


_orchestrator_instance: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance