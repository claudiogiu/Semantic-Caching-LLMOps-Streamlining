import os
from typing import Optional


OLLAMA_BASE_URL: Optional[str] = os.getenv("OLLAMA_BASE_URL")
OLLAMA_EMBED_MODEL: Optional[str] = os.getenv("OLLAMA_EMBED_MODEL")
OLLAMA_CHAT_MODEL: Optional[str] = os.getenv("OLLAMA_CHAT_MODEL")

REDIS_BASE_URL: Optional[str] = os.getenv("REDIS_BASE_URL")
REDIS_VECTOR_INDEX_NAME: Optional[str] = os.getenv("REDIS_VECTOR_INDEX_NAME")
REDIS_VECTOR_KEY_PREFIX: Optional[str] = os.getenv("REDIS_VECTOR_KEY_PREFIX")
REDIS_VECTOR_DIM: Optional[int] = (
    int(os.getenv("REDIS_VECTOR_DIM"))
    if os.getenv("REDIS_VECTOR_DIM") is not None
    else None
)

HF_DATASET_NAME: Optional[str] = os.getenv("HF_DATASET_NAME")
HF_DATASET_QUESTION_FIELD: Optional[str] = os.getenv("HF_DATASET_QUESTION_FIELD")
HF_DATASET_ANSWER_FIELD: Optional[str] = os.getenv("HF_DATASET_ANSWER_FIELD")