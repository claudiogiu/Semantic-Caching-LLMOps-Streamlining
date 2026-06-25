import logging
from typing import List, Dict, Any
from src.services.redis_service import RedisService

logger = logging.getLogger(__name__)


class Uploader:
    """
    Interface for managing batched insertion of vector records into Redis,
    ensuring controlled upload throughput and reliable persistence of semantic cache entries.

    Attributes:
        redis (RedisService): Service instance responsible for executing write operations in Redis.
        batch_size (int): Fixed number of records processed per upload batch.
        ttl (int): Expiration time, in seconds, applied to all records inserted during warmup.

    Methods:
        upload(records: List[Dict[str, Any]]) -> int:
            Uploads vector records to Redis in fixed‑size batches, ensuring sequential
            persistence of all provided entries.
    """

    def __init__(self, redis_service: RedisService) -> None:
        self.redis: RedisService = redis_service
        self.batch_size: int = 1000
        self.ttl: int = 604800 
        logger.info("Uploader initialized with RedisService instance.")

    async def upload(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            raise ValueError("Record list is empty.")

        logger.info("Starting upload of semantic cache records to Redis.")

        await self.redis.create_vector_index_if_not_exists()

        total = len(records)
        logger.info(f"Uploading {total} records in batches of {self.batch_size}.")

        inserted = 0

        for start in range(0, total, self.batch_size):
            end = start + self.batch_size
            batch = records[start:end]

            logger.info(f"Uploading batch {start} to {end} ({len(batch)} records).")

            for entry in batch:
                await self.redis.upsert_vector(
                    entry["id"],
                    entry["vector"],
                    entry["payload"],
                    ttl=self.ttl
                )
                inserted += 1

        logger.info("Upload to Redis completed.")
        return inserted
