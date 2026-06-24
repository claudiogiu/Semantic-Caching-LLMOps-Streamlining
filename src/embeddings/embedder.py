import logging
from typing import List
from src.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)


class Embedder:
    """
    Interface for generating embedding vectors from textual inputs as part of
    the semantic processing workflow.

    Attributes:
        ollama (OllamaService): Service responsible for executing embedding
            requests for textual inputs.

    Methods:
        embed(text: str) -> List[float]:
            Generates an embedding vector for a single text input.

        embed_batch(texts: List[str]) -> List[List[float]]:
            Generates embedding vectors for a batch of text inputs, preserving
            input ordering and ensuring consistent transformation semantics.
    """

    def __init__(self, ollama_service: OllamaService) -> None:
        self.ollama = ollama_service
        logger.info("Embedder component initialized.")

    async def embed(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        logger.info("Processing embedding request for a single text input.")
        return await self.ollama.embed(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Input text list is empty.")

        logger.info(f"Processing embedding requests for {len(texts)} text inputs.")
        return [await self.ollama.embed(t) for t in texts]