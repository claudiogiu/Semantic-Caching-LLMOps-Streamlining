import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class Builder:
    """
    Interface for constructing fully‑formed semantic cache records by combining
    normalized queries, embedding vectors, and LLM responses into a structured
    payload suitable for persistence.

    Attributes:
        default_ttl (int): Expiration time, in seconds, applied to newly created
            cache records unless overridden.

    Methods:
        build(query_hash: str, normalized: str, vector: List[float], response: str) -> Dict[str, Any]:
            Assembles a complete cache record containing identifiers, embedding
            data, payload metadata, and temporal information.
    """

    def __init__(self, default_ttl: int = 604800) -> None:
        self.default_ttl: int = default_ttl
        logger.info("Builder initialized with default TTL configuration.")

    def build(
        self,
        query_hash: str,
        normalized: str,
        vector: List[float],
        response: str
    ) -> Dict[str, Any]:
        if not query_hash or not normalized or not vector or not response:
            raise ValueError("Record construction failed due to missing required fields.")

        logger.info("Starting construction of new cache record.")

        record: Dict[str, Any] = {
            "id": query_hash,
            "vector": vector,
            "payload": {
                "question": normalized,
                "answer": response,
                "timestamp": int(time.time()),
                "ttl": self.default_ttl,
            },
        }

        logger.info("Cache record construction completed.")
        return record