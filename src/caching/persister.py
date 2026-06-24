import logging
from typing import Any, Dict, List, Optional
from src.services.redis_service import RedisService

logger = logging.getLogger(__name__)


class Persister:
    """
    Interface for managing persistence operations on cached vector records, providing
    structured access to exact-match retrieval, semantic search, record insertion, and
    deletion through an underlying RedisService instance.

    Attributes:
        redis (RedisService): Service responsible for executing Redis-based vector and
            payload operations.

    Methods:
        fetch_by_hash(query_hash: str) -> Optional[Dict[str, Any]]:
            Retrieves a cached record by its deterministic hash identifier.

        search_by_vector(vector: List[float], limit: int) -> List[Dict[str, Any]]:
            Executes a semantic similarity search using the provided embedding vector.

        save(record: Dict[str, Any], ttl: Optional[int]) -> None:
            Persists a structured vector record into Redis with optional TTL.

        delete(record_id: str) -> None:
            Removes a vector record from Redis by its identifier.
    """

    def __init__(self, redis_service: RedisService) -> None:
        self.redis: RedisService = redis_service
        logger.info("Persister initialized with RedisService instance.")

    async def fetch_by_hash(self, query_hash: str) -> Optional[Dict[str, Any]]:
        if not query_hash:
            raise ValueError("Query hash is empty or invalid.")
        logger.info("Fetching cache record by hash.")
        record: Optional[Dict[str, Any]] = await self.redis.get_vector(query_hash)
        logger.info("Fetch by hash completed.")
        return record

    async def search_by_vector(self, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        if not vector:
            raise ValueError("Vector for similarity search is empty or invalid.")
        logger.info("Executing vector similarity search.")
        results: List[Dict[str, Any]] = await self.redis.search_vector(vector, limit)
        logger.info("Vector similarity search completed.")
        return results

    async def save(self, record: Dict[str, Any], ttl: Optional[int] = None) -> None:
        if "id" not in record or "vector" not in record or "payload" not in record:
            raise ValueError("Record structure is invalid or incomplete.")
        logger.info("Saving cache record to Redis.")
        await self.redis.upsert_vector(
            record["id"],
            record["vector"],
            record["payload"],
            ttl=ttl
        )
        logger.info("Cache record saved successfully.")

    async def delete(self, record_id: str) -> None:
        if not record_id:
            raise ValueError("Record identifier is empty or invalid.")
        logger.info("Deleting cache record from Redis.")
        await self.redis.delete_vector(record_id)
        logger.info("Cache record deletion completed.")