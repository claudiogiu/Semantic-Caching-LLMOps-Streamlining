import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Selector:
    """
    Interface for ranking and selecting the most suitable cache candidate from a list
    of retrieved records based on deterministic similarity scoring.

    Methods:
        select(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            Validates the input list, filters structurally valid candidates, ranks them
            by ascending similarity score, and returns the highest‑ranked entry or None
            if no suitable candidate is available.
    """

    def select(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not candidates:
            logger.info("No candidates provided for selection.")
            return None

        logger.info("Starting selection among retrieved cache candidates.")

        valid: List[Dict[str, Any]] = [c for c in candidates if "score" in c]

        if not valid:
            logger.info("No valid candidates found.")
            return None

        logger.info("Ranking candidates by similarity score.")
        ranked: List[Dict[str, Any]] = sorted(valid, key=lambda x: x["score"])

        best: Dict[str, Any] = ranked[0]
        logger.info(f"Candidate selected: id={best.get('id')} score={best.get('score')}")
        return best