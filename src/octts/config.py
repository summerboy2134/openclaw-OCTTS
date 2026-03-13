from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "OCTTS"
    app_env: str = "development"

    tushare_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("TINYSHARE_TOKEN", "TUSHARE_TOKEN"),
    )
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "DEEPSEEK_API_KEY"),
    )
    llm_model: str = Field(
        default="deepseek-ai/DeepSeek-V3",
        validation_alias=AliasChoices("LLM_MODEL", "DEEPSEEK_MODEL"),
    )
    llm_base_url: str = Field(
        default="https://api.modelverse.cn/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "DEEPSEEK_BASE_URL"),
    )
    wecom_webhook_url: str | None = Field(default=None, alias="WECOM_WEBHOOK_URL")
    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    openclaw_gateway_url: str | None = Field(default=None, alias="OPENCLAW_GATEWAY_URL")
    openclaw_agent_id: str = Field(default="octts", alias="OPENCLAW_AGENT_ID")
    openclaw_hooks_enabled: bool = Field(default=False, alias="OPENCLAW_HOOKS_ENABLED")

    memory_backend: Literal["redis", "file"] = Field(default="redis", alias="OCTTS_MEMORY_BACKEND")
    memory_file_path: str = Field(default="memory/latest_memory.json", alias="OCTTS_MEMORY_FILE_PATH")
    history_dir_path: str = Field(
        default="memory/history",
        validation_alias=AliasChoices("OCTTS_HISTORY_DIR_PATH", "OCTTS_HISTORY_FILE_PATH"),
    )
    history_limit_per_symbol: int = Field(default=30, alias="OCTTS_HISTORY_LIMIT_PER_SYMBOL")

    stock_pool_raw: str = Field(default="", alias="OCTTS_STOCK_POOL")
    default_lookback_days: int = Field(default=20, alias="OCTTS_DEFAULT_LOOKBACK_DAYS")
    minute_freq: str = Field(default="30MIN", alias="OCTTS_MINUTE_FREQ")

    request_timeout_seconds: int = Field(default=60, alias="OCTTS_REQUEST_TIMEOUT_SECONDS")
    llm_temperature: float = Field(default=0.2, alias="OCTTS_LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=3000, alias="OCTTS_LLM_MAX_TOKENS")
    llm_json_mode: bool = Field(default=True, alias="OCTTS_LLM_JSON_MODE")
    llm_retry_attempts: int = Field(default=3, alias="OCTTS_LLM_RETRY_ATTEMPTS")
    automation_enabled: bool = Field(default=False, alias="OCTTS_AUTOMATION_ENABLED")
    automation_timezone: str = Field(default="Asia/Shanghai", alias="OCTTS_AUTOMATION_TIMEZONE")
    automation_morning_time: str = Field(default="09:35", alias="OCTTS_AUTOMATION_MORNING_TIME")
    automation_afternoon_time: str = Field(default="14:35", alias="OCTTS_AUTOMATION_AFTERNOON_TIME")
    automation_review_time: str = Field(default="20:30", alias="OCTTS_AUTOMATION_REVIEW_TIME")
    automation_phases_raw: str = Field(default="review", alias="OCTTS_AUTOMATION_PHASES")
    automation_notify: bool = Field(default=True, alias="OCTTS_AUTOMATION_NOTIFY")

    @property
    def stock_pool(self) -> list[str]:
        return [item.strip() for item in self.stock_pool_raw.split(",") if item.strip()]

    @property
    def automation_phases(self) -> list[str]:
        allowed = {"morning", "afternoon", "review"}
        phases = [item.strip().lower() for item in self.automation_phases_raw.split(",") if item.strip()]
        valid_phases = [item for item in phases if item in allowed]
        return valid_phases or ["review"]

    @property
    def deepseek_api_key(self) -> str | None:
        return self.llm_api_key

    @property
    def deepseek_model(self) -> str:
        return self.llm_model

    @property
    def deepseek_base_url(self) -> str:
        return self.llm_base_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
