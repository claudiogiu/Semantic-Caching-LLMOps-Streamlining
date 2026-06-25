import logging
import time
from fastapi import APIRouter, HTTPException, Request
from src.api.schemas import (
    WarmupResponse,
    CacheQueryRequest,
    CacheQueryResponse,
    CacheMaintenanceResponse,
    DeleteRecordRequest,
)
from src.api.fields import (
    PipelineStatus,
    WarmupStatus,
    CacheOperation,
)
from src.core.orchestrator import get_orchestrator
from src.warmup.warmup import Warmup
from src.services.redis_service import RedisService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/cache/warmup", response_model=WarmupResponse, tags=["Cache"])
async def cache_warmup(request: Request):
    """
    Execute the warmup pipeline by generating embeddings and inserting them
    into the semantic cache.
    """

    start = time.time()

    try:
        warmup = Warmup()
        processed, inserted = await warmup.run()

        execution_time_ms = (time.time() - start) * 1000

        return WarmupResponse(
            status=WarmupStatus.SUCCESS,
            processed=processed,
            inserted=inserted,
            execution_time_ms=execution_time_ms,
        )

    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        raise HTTPException(status_code=500, detail="Warmup failed")


@router.post("/cache/query", response_model=CacheQueryResponse, tags=["Cache"])
async def cache_query(request: Request, payload: CacheQueryRequest):
    """
    Execute a semantic cache query. If a cache HIT occurs, return the cached answer.
    Otherwise, generate a new answer using the LLM and store it in the cache.
    """

    start = time.time()

    try:
        orchestrator = get_orchestrator()
        answer = await orchestrator.query(payload.query)

        execution_time_ms = (time.time() - start) * 1000

        return CacheQueryResponse(
            status=PipelineStatus.COMPLETED,
            answer=answer,
            execution_time_ms=execution_time_ms,
        )

    except Exception as e:
        logger.error(f"Cache query failed: {e}")
        raise HTTPException(status_code=500, detail="Cache query failed")


@router.delete("/cache", response_model=CacheMaintenanceResponse, tags=["Cache"])
async def cache_flush(request: Request):
    """
    Flush the entire semantic cache, removing all stored records.
    """

    start = time.time()

    try:
        redis = RedisService()
        redis._client.flushdb()

        execution_time_ms = (time.time() - start) * 1000

        return CacheMaintenanceResponse(
            status=PipelineStatus.COMPLETED,
            operation=CacheOperation.FLUSH_ALL,
            details="All cache records have been removed.",
            execution_time_ms=execution_time_ms,
        )

    except Exception as e:
        logger.error(f"Cache flush failed: {e}")
        raise HTTPException(status_code=500, detail="Cache flush failed")


@router.delete("/cache/record", response_model=CacheMaintenanceResponse, tags=["Cache"])
async def cache_delete_record(request: Request, payload: DeleteRecordRequest):
    """
    Delete a specific cache record identified by its hash‑based identifier.
    """
    
    start = time.time()

    try:
        redis = RedisService()
        redis._client.delete(f"vec:{payload.record_id}")

        execution_time_ms = (time.time() - start) * 1000

        return CacheMaintenanceResponse(
            status=PipelineStatus.COMPLETED,
            operation=CacheOperation.DELETE_RECORD,
            details=f"Record '{payload.record_id}' deleted successfully.",
            execution_time_ms=execution_time_ms,
        )

    except Exception as e:
        logger.error(f"Record deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Record deletion failed")
