import logging
import json
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable
import threading

class JSONFormatter(logging.Formatter):

    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName
        }

        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)

def get_logger(name: str = "production-api") -> logging.Logger:

    logger= logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

class MetricsCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.time()

        self._requests_total: int = 0
        self._errors_total: int = 0
        self._latency_count: int = 0
        self._latency_sum: float = 0.0
        self._latency_max: float = 0.0
        self._tokens_input: int = 0
        self._tokens_output: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def record_request(
        self,
        latency_ms: float,
        success: bool = True,
        error: bool | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool | None = None,
    ) -> None:
        """Records an incoming API request, latency, errors, tokens, and cache status."""
        if error is not None:
            success = not error

        with self._lock:
            self._requests_total += 1
            if not success:
                self._errors_total += 1

            self._latency_count += 1
            self._latency_sum += latency_ms
            if latency_ms > self._latency_max:
                self._latency_max = latency_ms

            self._tokens_input += max(0, input_tokens)
            self._tokens_output += max(0, output_tokens)

            if cache_hit is True:
                self._cache_hits += 1
            elif cache_hit is False:
                self._cache_misses += 1

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Tracks prompt and completion token usage for LLM cost monitoring."""
        with self._lock:
            self._tokens_input += max(0, input_tokens)
            self._tokens_output += max(0, output_tokens)

    def record_cache(self, hit: bool) -> None:
        """Records whether a response was served from cache or missed."""
        with self._lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def get_summary(self) -> dict[str, Any]:
        """Returns a thread-safe snapshot of all operational metrics."""
        with self._lock:
            avg_latency = (
                self._latency_sum / self._latency_count
                if self._latency_count > 0
                else 0.0
            )
            error_rate = (
                self._errors_total / self._requests_total
                if self._requests_total > 0
                else 0.0
            )
            total_cache = self._cache_hits + self._cache_misses
            cache_hit_rate = (
                self._cache_hits / total_cache
                if total_cache > 0
                else 0.0
            )

            return {
                "uptime_seconds": round(time.time() - self._start_time, 2),
                "requests": {
                    "total": self._requests_total,
                    "errors": self._errors_total,
                    "error_rate": f"{error_rate:.1%}",
                },
                "latency_ms": {
                    "avg": round(avg_latency, 2),
                    "max": round(self._latency_max, 2),
                    "samples": self._latency_count,
                },
                "tokens": {
                    "input": self._tokens_input,
                    "output": self._tokens_output,
                    "total": self._tokens_input + self._tokens_output,
                },
                "cache": {
                    "hits": self._cache_hits,
                    "misses": self._cache_misses,
                    "hit_rate": f"{cache_hit_rate:.1%}",
                },
            }

    def reset(self) -> None:
        """Resets all counters safely (useful for testing or periodic flushes)."""
        with self._lock:
            self._start_time = time.time()
            self._requests_total = 0
            self._errors_total = 0
            self._latency_count = 0
            self._latency_sum = 0.0
            self._latency_max = 0.0
            self._tokens_input = 0
            self._tokens_output = 0
            self._cache_hits = 0
            self._cache_misses = 0

class RequestTimer:

    def __enter__(self):
        self.start = time.time()
        self._end = None
        return self

    def __exit__(self, *args):
        self._end = time.time()

    @property
    def elapsed_ms(self) -> float:
        end = self._end if self._end is not None else time.time()
        return (end - self.start) * 1000