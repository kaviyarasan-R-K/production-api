import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_api_key: str
    primary_model: str = "gemini-3.8-flash"
    fallback_model: str = "gemini-3.8-flash"

    langsmith_tracing: bool = True
    langchain_tracing_V2: bool = True
    langsmith_endpoint: str = "https://apac.api.smith.langchain.com"
    langsmith_api_key: str = ""
    langsmith_project: str = "production-api"
    langchain_api_key: str = ""
    langchain_project: str = "production-api"

    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    model_config = {"env_file": ".env", "extra": "ignore"}

@lru_cache
def get_settings() -> Settings:
    settings = Settings()

    api_key = settings.langsmith_api_key or settings.langchain_api_key
    project = settings.langsmith_project or settings.langchain_project

    if settings.langsmith_tracing and api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGCHAIN_PROJECT"] = project

    return settings
