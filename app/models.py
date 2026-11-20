"""
API Requests and Response models
Pydantic models input validationand response structure
"""

from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str = "default"

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        return v

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    model_used: str
    cached: bool=False
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class HealthResponse(BaseModel):
    status: str = "healthy"
    environment: str 
    version : str = "1.0.0"
    checks: dict = {}

class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    error_rate: float
    avg_latency_ms : float
    cache_hit_rate: float
    total_input_tokens: int
    total_output_tokens: int

class ErrorResponse(BaseModel):
    error: str
    detail: str | None
    request_id: str| None


