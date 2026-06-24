import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Interface for validating cached vector records through TTL verification,
    semantic‑confidence filtering, and payload‑integrity checks to ensure that
    only trustworthy and non‑expired entries are returned as cache hits.

    Attributes:
        ttl_tolerance (int): Maximum allowed age (in seconds) for a cached record
            before it is considered expired.
        min_confidence (float): Maximum acceptable cosine distance score for a
            semantic match to be considered valid.

    Methods:
        is_valid(record: Dict[str, Any]) -> bool:
            Executes TTL, confidence, and poisoning checks to determine whether
            a retrieved cache candidate is safe and suitable for reuse.
    """

    def __init__(self, ttl_tolerance: int = 604800, min_confidence: float = 0.20) -> None:
        self.ttl_tolerance: int = ttl_tolerance
        self.min_confidence: float = min_confidence
        logger.info("Evaluator initialized with TTL and confidence thresholds.")

    def _check_ttl(self, record: Dict[str, Any]) -> bool:
        timestamp: Optional[int] = record.get("payload", {}).get("timestamp")
        if timestamp is None:
            logger.info("Record rejected due to missing timestamp.")
            return False

        age: float = time.time() - timestamp
        if age > self.ttl_tolerance:
            logger.info("Record rejected due to TTL expiration.")
            return False

        return True

    def _check_confidence(self, record: Dict[str, Any]) -> bool:
        score: Optional[float] = record.get("score")
        if score is None:
            logger.info("Record rejected due to missing similarity score.")
            return False

        if score > self.min_confidence:
            logger.info("Record rejected due to insufficient semantic confidence.")
            return False

        return True

    def _check_poisoning(self, record: Dict[str, Any]) -> bool:
        payload: Dict[str, Any] = record.get("payload", {})
        answer: Optional[str] = payload.get("answer") or payload.get("response")

        if not isinstance(answer, str) or not answer.strip():
            logger.info("Record rejected due to empty or malformed payload content.")
            return False

        return True

    def is_valid(self, record: Dict[str, Any]) -> bool:
        logger.info("Starting evaluation of cache candidate.")

        if not self._check_ttl(record):
            return False

        if not self._check_confidence(record):
            return False

        if not self._check_poisoning(record):
            return False

        logger.info("Cache candidate validated successfully.")
        return True