"""Configuration for the Vital SDK using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class VitalConfig(BaseSettings):
    """SDK configuration loaded from environment variables.

    All fields prefixed with ``VITAL_`` in the environment.
    Example: ``VITAL_CORE_URL``, ``VITAL_API_KEY``, ``VITAL_SERVICE_NAME``.
    """

    model_config = SettingsConfigDict(env_prefix="VITAL_", env_file=".env")

    core_url: str = "http://localhost:8888"
    api_key: str = ""
    service_name: str = "codking"
    service_port: int = 8000
    redis_url: str = "redis://localhost:6379"
