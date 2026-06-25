import logging
import time
from typing import List, Dict, Any
from src.services.ollama_service import OllamaService
from src.services.redis_service import RedisService
from src.warmup.loader import Loader
from src.embeddings.embedder import Embedder
from src.warmup.uploader import Uploader
from src.caching.normalizer import Normalizer

logger = logging.getLogger(__name__)


class Warmup:
    """
    Interface for orchestrating dataset loading, deterministic normalization, embedding
    generation, and batched insertion of vector‑payload records into Redis during the
    system initialization phase.

    Attributes:
        loader (Loader): Component responsible for retrieving structured dataset records.
        normalizer (Normalizer): Component performing deterministic text normalization and hashing.
        embedder (Embedder): Service used to compute embedding vectors for normalized questions.
        uploader (Uploader): Component responsible for uploading vector‑payload records to Redis.
    
    Methods:
        run() -> tuple[int, int]:
            Executes the warmup pipeline by loading dataset entries, normalizing questions,
            computing hashes, generating embeddings, constructing vector‑payload records,
            and uploading them to Redis in batch.
    """

    def __init__(self) -> None:
        self.loader: Loader = Loader()
        self.normalizer: Normalizer = Normalizer()
        self.embedder: Embedder = Embedder(OllamaService())
        self.uploader: Uploader = Uploader(RedisService())
        logger.info("Warmup component initialized.")

    async def run(self) -> tuple[int, int]:
        logger.info("Warmup process started.")

        records: List[Dict[str, Any]] = self.loader.load()
        logger.info(f"Retrieved {len(records)} dataset records.")

        enriched: List[Dict[str, Any]] = []

        for idx, entry in enumerate(records):
            question: str = entry["question"]
            answer: str = entry["answer"]

            logger.info(f"[{idx}] Normalizing question.")
            normalized: str = self.normalizer.normalize(question)

            logger.info(f"[{idx}] Computing hash.")
            query_hash: str = self.normalizer.compute_hash(normalized)

            logger.info(f"[{idx}] Generating embedding.")
            vector: List[float] = await self.embedder.embed(normalized)

            payload: Dict[str, Any] = {
                "question": normalized,
                "answer": answer,
                "ttl": 2592000,
                "timestamp": int(time.time())
            }

            record: Dict[str, Any] = {
                "id": query_hash,
                "vector": vector,
                "payload": payload,
            }

            logger.info(f"[{idx}] Record built: id={query_hash}")
            enriched.append(record)

        processed = len(records)
        logger.info(f"Computed embeddings for {processed} entries.")
        logger.info("Uploading records to Redis.")

        inserted = await self.uploader.upload(enriched)

        logger.info(
            f"Warmup process completed."
        )

        return processed, inserted
