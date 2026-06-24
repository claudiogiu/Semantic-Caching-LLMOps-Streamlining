from pydantic import BaseModel, Field
from src.api.fields import PipelineStatus, WarmupStatus, CacheOperation


class WarmupResponse(BaseModel):
    """Response model describing the outcome of the warmup procedure."""

    status: WarmupStatus = Field(..., description="Execution status of the warmup pipeline")
    processed: int = Field(..., description="Number of dataset entries processed during warmup")
    inserted: int = Field(..., description="Number of records successfully inserted into the cache")
    execution_time_ms: float = Field(..., description="Total execution time of the warmup procedure in milliseconds")


class CacheQueryRequest(BaseModel):
    """Request model for executing a semantic cache query."""

    query: str = Field(..., description="User query to be processed by the semantic cache pipeline")


class CacheQueryResponse(BaseModel):
    """Response model containing the generated answer and execution metadata."""

    status: PipelineStatus = Field(..., description="Execution status of the query pipeline")
    answer: str = Field(..., description="Final answer returned by the semantic cache or LLM backend")
    execution_time_ms: float = Field(..., description="Total execution time of the query pipeline in milliseconds")


class CacheMaintenanceResponse(BaseModel):
    """Response model describing the outcome of a cache maintenance operation."""

    status: PipelineStatus = Field(..., description="Execution status of the maintenance operation")
    operation: CacheOperation = Field(..., description="Type of cache maintenance operation performed")
    details: str = Field(..., description="Human‑readable description of the operation result")
    execution_time_ms: float = Field(..., description="Total execution time of the maintenance operation in milliseconds")


class DeleteRecordRequest(BaseModel):
    """Request model for deleting a specific cache record."""

    record_id: str = Field(..., description="Identifier of the cache record to be deleted")