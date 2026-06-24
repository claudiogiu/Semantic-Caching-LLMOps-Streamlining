import logging
from typing import List, Dict, Any, Optional
import httpx
from src.config.constants import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_CHAT_MODEL,
)

logger = logging.getLogger(__name__)


class OllamaService:
    """
    Interface for managing asynchronous communication with the Ollama inference server, 
    providing embedding generation and chat-based language model responses through HTTP endpoints.

    Attributes:
        base_url (str): Base URL of the Ollama server used for API requests.
        embed_model (str): Identifier of the embedding model used for vector generation.
        chat_model (str): Identifier of the chat model used for text generation.
        _client (httpx.AsyncClient): Asynchronous HTTP client configured for Ollama communication.

    Methods:
        _post(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            Sends a POST request to the specified Ollama endpoint and returns the parsed JSON response.

        embed(text: str) -> List[float]:
            Generates an embedding vector for the provided text using the configured embedding model.

        chat(prompt: str, temperature: float = 0.3, max_tokens: Optional[int] = None,
             system_prompt: Optional[str] = None) -> str:
            Produces a language model response using the configured chat model and optional system prompt.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        embed_model: Optional[str] = None,
        chat_model: Optional[str] = None,
        timeout: int = 60,
    ):
        self.base_url = base_url or OLLAMA_BASE_URL
        self.embed_model = embed_model or OLLAMA_EMBED_MODEL
        self.chat_model = chat_model or OLLAMA_CHAT_MODEL

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout
        )

        logger.info(
            f"Ollama service initialization completed. "
            f"Base endpoint: {self.base_url}. "
            f"Embedding model: {self.embed_model}. "
            f"Chat model: {self.chat_model}."
        )

    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Sending POST request to Ollama. Endpoint: {endpoint}.")
        response = await self._client.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        logger.info("Response received from Ollama service.")
        return data

    async def embed(self, text: str) -> List[float]:
        if not text:
            raise ValueError("The provided text for embedding generation is empty.")

        logger.info("Starting embedding generation through Ollama.")
        payload = {
            "model": self.embed_model,
            "input": text
        }

        data = await self._post("/api/embed", payload)
        embeddings = data.get("embeddings")

        if not embeddings or not isinstance(embeddings, list) or len(embeddings) == 0:
            raise RuntimeError("Ollama did not return a valid embedding.")

        embedding = embeddings[0]
        logger.info("Embedding successfully generated.")
        return embedding

    async def chat(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        logger.info("Starting LLM response generation through Ollama.")

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = await self._post("/api/chat", payload)

        if "message" in data:
            logger.info("LLM response successfully generated.")
            return data["message"]["content"]

        raise RuntimeError("The LLM response does not contain a valid message.")