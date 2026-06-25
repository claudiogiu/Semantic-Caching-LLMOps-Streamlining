import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api.routes import router
from src.core.orchestrator import get_orchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager responsible for initializing and
    shutting down the semantic cache orchestrator.
    """
    logger.info("Starting Semantic Cache API service")

    orchestrator = get_orchestrator()
    app.state.orchestrator = orchestrator

    logger.info("Orchestrator initialized successfully")

    yield

    try:
        await orchestrator.shutdown()
        logger.info("Semantic Cache services closed successfully")
    except Exception as e:
        logger.error(f"Shutdown failure: {e}", exc_info=True)


app = FastAPI(
    title="Semantic Cache API Service",
    description="API service providing warmup, semantic caching, LLM fallback, and cache maintenance operations",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "message": "Unexpected server error"}
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Semantic Cache API Service operational",
        "version": "1.0.0",
        "docs_url": "/docs",
        "api_version": "v1",
        "endpoints": {
            "warmup": "/api/v1/cache/warmup",
            "query": "/api/v1/cache/query",
            "flush": "/api/v1/cache",
            "delete_record": "/api/v1/cache/record",
        }
    }


app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Semantic Cache API server")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
