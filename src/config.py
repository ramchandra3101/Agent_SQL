"""Centralised settings for the SQL agent.

Reads from the local `.env` file (loaded once at import time) and exposes a
frozen `settings` singleton. Every other module imports from here — no module
is allowed to read environment variables directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level above this file's parent).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable: {key}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value or ""


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw not in (None, "") else default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


@dataclass(frozen=True)
class Settings:
    # ---- OpenAI ------------------------------------------------------------
    openai_api_key: str
    openai_model: str
    openai_embedding_model: str

    # ---- Database ----------------------------------------------------------
    database_url: str

    # ---- LangSmith ---------------------------------------------------------
    langsmith_tracing: bool
    langsmith_api_key: str
    langsmith_project: str

    # ---- Guardrail budgets -------------------------------------------------
    max_retries: int
    max_sql_generations: int
    row_scan_budget: int
    query_timeout_seconds: int

    # ---- Security ----------------------------------------------------------
    # Keywords rejected by the validator node before any SQL reaches Postgres.
    forbidden_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "DROP",
            "DELETE",
            "TRUNCATE",
            "ALTER",
            "INSERT",
            "UPDATE",
            "GRANT",
            "REVOKE",
            "CREATE",
            "REPLACE",
            "MERGE",
            "CALL",
            "EXEC",
            "EXECUTE",
            "COPY",
        )
    )

    # ---- Paths -------------------------------------------------------------
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)


def _load_settings() -> Settings:
    return Settings(
        openai_api_key=_env("OPENAI_API_KEY", required=True),
        openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embedding_model=_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        database_url=_env("DATABASE_URL", required=True),
        langsmith_tracing=_env_bool("LANGSMITH_TRACING", default=False),
        langsmith_api_key=_env("LANGSMITH_API_KEY", ""),
        langsmith_project=_env("LANGSMITH_PROJECT", "sql-agent-poc"),
        max_retries=_env_int("MAX_RETRIES", 2),
        max_sql_generations=_env_int("MAX_SQL_GENERATIONS", 3),
        row_scan_budget=_env_int("ROW_SCAN_BUDGET", 100_000),
        query_timeout_seconds=_env_int("QUERY_TIMEOUT_SECONDS", 15),
    )


settings: Settings = _load_settings()
