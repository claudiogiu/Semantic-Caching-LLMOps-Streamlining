import logging
import hashlib
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Normalizer:
    """
    Interface for applying deterministic normalization and hash‑based identification
    of textual queries within semantic caching workflows.

    Methods:
        normalize(text: str) -> str:
            Applies canonical normalization to the input text.

        compute_hash(text: str) -> str:
            Computes a SHA‑256 hash for the provided normalized text.

        process(text: str) -> Dict[str, Any]:
            Executes the full normalization and hashing pipeline, returning
            both the normalized text and its associated hash.
    """

    def normalize(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text for normalization is empty or invalid.")

        logger.info("Starting normalization of input text.")
        normalized: str = " ".join(text.strip().lower().split())
        logger.info("Normalization completed.")
        return normalized

    def compute_hash(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text for hashing is empty or invalid.")

        logger.info("Computing SHA‑256 hash for normalized text.")
        digest: str = hashlib.sha256(text.encode("utf-8")).hexdigest()
        logger.info("Hash computation completed.")
        return digest

    def process(self, text: str) -> Dict[str, Any]:
        logger.info("Starting full normalization and hashing process.")
        normalized: str = self.normalize(text)
        query_hash: str = self.compute_hash(normalized)
        logger.info("Full normalization and hashing process completed.")
        return {
            "normalized": normalized,
            "query_hash": query_hash,
        }