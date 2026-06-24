import logging
from typing import Any, Dict, List, Optional
import redis
import numpy as np
import json
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from src.config.constants import (
    REDIS_BASE_URL,
    REDIS_VECTOR_INDEX_NAME,
    REDIS_VECTOR_KEY_PREFIX,
    REDIS_VECTOR_DIM,
)

logger = logging.getLogger(__name__)


class RedisService:
    """
    Interface for managing Redis-based vector storage, including index creation,
    key–value operations, vector upsertion, similarity search, and structured
    retrieval of cached records.

    Attributes:
        redis_url (str): Connection URL for the Redis instance.
        index_name (str): Name of the RediSearch vector index.
        vector_key_prefix (str): Prefix applied to all vector record keys.
        vector_dim (int): Dimensionality of stored embedding vectors.
        _client (redis.Redis): Underlying Redis client instance.

    Methods:
        create_vector_index_if_not_exists() -> None:
            Ensures that the RediSearch vector index is created before use.

        set_key(key, value, ttl) -> None:
            Stores a simple key–value pair with optional TTL.

        get_key(key) -> Optional[str]:
            Retrieves a UTF‑8 decoded string value for a given key.

        upsert_vector(id, vector, payload, ttl) -> None:
            Stores or updates a vector record with metadata and optional TTL.

        get_vector(id) -> Optional[Dict[str, Any]]:
            Retrieves a full vector record including payload and decoded vector.

        search_vector(query_vector, limit) -> List[Dict[str, Any]]:
            Executes a KNN similarity search and returns ranked candidate records.

        delete_vector(id) -> None:
            Removes a vector record from Redis.

        ping() -> bool:
            Checks connectivity with the Redis server.

        close() -> None:
            Closes the underlying Redis client connection.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        index_name: Optional[str] = None,
        vector_key_prefix: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ) -> None:
        self.redis_url: str = redis_url or REDIS_BASE_URL
        self.index_name: str = index_name or REDIS_VECTOR_INDEX_NAME
        self.vector_key_prefix: str = vector_key_prefix or REDIS_VECTOR_KEY_PREFIX
        self.vector_dim: int = vector_dim or REDIS_VECTOR_DIM

        self._client: redis.Redis = redis.Redis.from_url(self.redis_url)

        logger.info(
            f"Redis service initialized. "
            f"URL={self.redis_url}, index={self.index_name}, dim={self.vector_dim}"
        )

    async def _index_exists(self) -> bool:
        try:
            self._client.ft(self.index_name).info()
            return True
        except Exception:
            return False

    async def create_vector_index_if_not_exists(self) -> None:
        if await self._index_exists():
            logger.info("Redis index already exists.")
            return

        logger.info("Creating Redis vector index.")

        schema = [
            TextField("id"),
            TextField("question"),
            TextField("answer"),
            VectorField(
                "vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dim,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        ]

        definition = IndexDefinition(
            prefix=[self.vector_key_prefix],
            index_type=IndexType.HASH,
        )

        self._client.ft(self.index_name).create_index(schema, definition=definition)
        logger.info("Redis vector index created successfully.")

    async def set_key(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        if ttl:
            self._client.set(key, value, ex=ttl)
        else:
            self._client.set(key, value)

    async def get_key(self, key: str) -> Optional[str]:
        value = self._client.get(key)
        return value.decode("utf-8") if value else None

    async def upsert_vector(
        self,
        id: str,
        vector: List[float],
        payload: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        key = f"{self.vector_key_prefix}{id}"

        vector_bytes: bytes = np.array(vector, dtype=np.float32).tobytes()

        data: Dict[str, Any] = {
            "id": id,
            "vector": vector_bytes,
            "payload": json.dumps(payload),
            "question": payload.get("question", ""),
            "answer": payload.get("answer", ""),
        }

        logger.info(f"Upserting vector record: {id}")
        self._client.hset(key, mapping=data)

        if ttl is not None:
            self._client.expire(key, ttl)
            logger.info(f"TTL applied to record {id}: {ttl}s")

    async def get_vector(self, id: str) -> Optional[Dict[str, Any]]:
        key = f"{self.vector_key_prefix}{id}"
        logger.info(f"Fetching vector record: {id}")

        data: Dict[bytes, bytes] = self._client.hgetall(key)
        if not data:
            logger.info(f"No record found for id={id}")
            return None

        vector: List[float] = np.frombuffer(data[b"vector"], dtype=np.float32).tolist()
        payload: Dict[str, Any] = json.loads(data[b"payload"].decode("utf-8"))

        logger.info(f"Record fetched successfully: {id}")
        return {
            "id": id,
            "vector": vector,
            "payload": payload,
            "score": 0.0,
        }

    async def search_vector(
        self,
        query_vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        logger.info("Executing vector similarity search.")

        query_bytes: bytes = np.array(query_vector, dtype=np.float32).tobytes()

        base_query: str = f"*=>[KNN {limit} @vector $vec AS score]"

        q: Query = (
            Query(base_query)
            .return_fields("id", "question", "answer", "score")
            .sort_by("score")
            .dialect(2)
        )

        params: Dict[str, Any] = {"vec": query_bytes}

        try:
            results = self._client.ft(self.index_name).search(q, query_params=params)
        except Exception as e:
            logger.error(f"Redis search error: {e}")
            return []

        output: List[Dict[str, Any]] = []

        for doc in results.docs:
            logger.info(f"Raw Redis doc: {doc.__dict__}")

            output.append(
                {
                    "id": doc.id,
                    "score": float(getattr(doc, "score", 1.0)),
                    "question": getattr(doc, "question", None),
                    "answer": getattr(doc, "answer", None),
                }
            )

        logger.info(f"Vector search completed. {len(output)} candidates found.")
        return output

    async def delete_vector(self, id: str) -> None:
        key = f"{self.vector_key_prefix}{id}"
        logger.info(f"Deleting vector record: {id}")
        self._client.delete(key)

    async def ping(self) -> bool:
        return bool(self._client.ping())

    async def close(self) -> None:
        logger.info("Closing Redis service connection.")
        try:
            self._client.close()
        except Exception:
            pass