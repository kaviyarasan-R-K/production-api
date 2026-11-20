import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv

from app.config import get_settings
from app.models import (
    ChatRequest, ChatResponse, HealthResponse, MetricsResponse,
    ErrorResponse
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import JSONFormatter, MetricsCollector, RequestTimer, get_logger
from app.agent import ProductionAgent

load_dotenv()
logger = get_logger()

security: SecurityPipeline = None
cache: ResponseCache = None
metrics: MetricsCollector = None
agent: ProductionAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initiate all components on startup, clean up on shutdown
    This is new modern FastAPI pattern (replaces @app.on_event.)
    """
    global security, cache, metrics, agent

    settings = get_settings()
    logger.info("Starting Production API...", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "tracing": settings.langchain_tracing_V2
    }})

    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionAgent()

    logger.info("All components initialized, Ready to serve...")

    yield

    logger.info("Shutting down...", extra={"extra_data": metrics.get_summary()})


# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Production LangGraph API",
    description="A Production Ready chat API security, caching and observability",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
async def root():
    return {
        "service": "production-api",
        "status": "online",
        "docs": "/docs",
        "health": "/health"
    }

# Endpoints
@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.

    Flow:
    1. Security Check (injection + PII masking)
    2. Cache lookup
    3. LangGraph agent invoke (if cache misses)
    4. Output Validation
    5. Cache store
    6. Return Response
    """
    with RequestTimer() as timer:
        # 1. Security checkup
        security_notes = []

        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning("Request blocked by security", extra={"extra_data": {
                "reason": notes,
                "thread_id": body.thread_id,
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters"
            )

        # 2. Cache lookup
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info("Cache_hit", extra={"extra_data": {
                "thread_id": body.thread_id,
            }})
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
            )

        # 3. LangChain agent Invoke
        try:
            result = agent.invoke(cleaned_message)
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", extra={
                "extra_data": {
                    "thread_id": body.thread_id,
                    "error": str(e),
                }
            })
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request"
            )

        response_text = result["response"]
        model_used = result["model_used"]

        # Step 4: Output Validation
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # Step 5: Cache store (only cache real model responses, not error fallbacks)
        if model_used != "error_handler":
            cache.set(cleaned_message, validated_response)

        # Log and record metrics
        input_tokens = int(len(cleaned_message.split()) * 1.3)
        output_tokens = int(len(validated_response.split()) * 1.3)
        metrics.record_request(
            latency_ms=timer.elapsed_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=False,
        )

        if security_notes:
            logger.info("security notes", extra={
                "extra_data": {
                    "notes": security_notes,
                    "thread_id": body.thread_id
                }
            })
        logger.info("Request Completed", extra={
            "extra_data": {
                "thread_id": body.thread_id,
                "model_used": model_used,
                "latency_ms": round(timer.elapsed_ms, 2)
            }
        })

        return ChatResponse(
            response=validated_response,
            thread_id=body.thread_id,
            model_used=model_used,
            cached=False,
            processing_time_ms=round(timer.elapsed_ms, 2),
        )
@app.get("/health", response_model=HealthResponse)
async def health():
    
    components_ready = all([
        agent is not None,
        security is not None,
        cache is not None,
        metrics is not None,
    ])

    if not components_ready:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: core components not initialized"
        )

    return HealthResponse(
        status="healthy",
        environment=get_settings().app_env,
        model=get_settings().primary_model,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    summary = metrics.get_summary()
    return MetricsResponse(
        uptime_seconds=summary["uptime_seconds"],
        total_requests=summary["requests"]["total"],
        total_errors=summary["requests"]["errors"],
        error_rate=float(summary["requests"]["error_rate"].rstrip("%")),
        avg_latency_ms=summary["latency_ms"]["avg"],
        cache_hit_rate=float(summary["cache"]["hit_rate"].rstrip("%")),
        total_input_tokens=summary["tokens"]["input"],
        total_output_tokens=summary["tokens"]["output"],
    )


@app.get("/cache/stats")
async def get_cache_stats():

    return cache.stats


@app.delete("/cache")
async def clear_cache():
    
    cache.clear()
    logger.info("Cache cleared via API endpoint")
    return {"status": "cleared", "message": "Response cache has been flushed"}