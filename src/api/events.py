"""SSE event payload schemas.

These Pydantic models exist **for OpenAPI documentation only** — they
describe the shape of each ``data:`` JSON payload the server emits on
its two SSE endpoints. The runtime path in ``server.py`` does not
instantiate them; it writes the JSON directly for speed.

Keeping the schemas here (rather than inline in ``server.py``) makes
the OpenAPI doc block compact and gives the frontend a single place to
generate TypeScript types from, via ``/openapi.json``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ThreadEvent(BaseModel):
    """``event: thread`` — first frame on /ask, carries the thread id."""
    thread_id: str


class RouteEvent(BaseModel):
    """``event: route`` — router decided sql-vs-rag."""
    route: Literal["sql", "rag"]
    reason: str = ""


class SchemaEvent(BaseModel):
    """``event: schema`` — tables the schema selector kept."""
    selected_tables: list[str]


class SqlEvent(BaseModel):
    """``event: sql`` — a SQL candidate from the generator."""
    sql: str
    generation_count: int


class GuardrailEvent(BaseModel):
    """``event: guardrail`` — validator + cost verdict."""
    validator_ok: bool | None = None
    cost_ok: bool | None = None
    risk_flags: list[str] = Field(default_factory=list)
    reason: str = ""


class RowsEvent(BaseModel):
    """``event: rows`` — executor returned a result set."""
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int


class ExecErrorEvent(BaseModel):
    """``event: exec_error`` — executor raised; may trigger a retry."""
    error: str


class RetryEvent(BaseModel):
    """``event: retry`` — self-correction emitted a hint."""
    hint: str
    retry_count: int


class RagEvent(BaseModel):
    """``event: rag`` — RAG fallback answer for off-schema questions."""
    answer: str


class AnswerDeltaEvent(BaseModel):
    """``event: answer_delta`` — a single token chunk of the NL summary."""
    text: str


class NeedsApprovalEvent(BaseModel):
    """``event: needs_approval`` — graph hit HITL; stream closes after this.

    The client should show the SQL + flags to the human, then POST to
    ``/resume`` with the chosen action to continue the same thread.
    """
    thread_id: str
    sql: str
    risk_flags: list[str]
    cost_reason: str = ""
    estimated_rows: int = 0
    execution_error: str | None = None


class DoneEvent(BaseModel):
    """``event: done`` — terminal frame on a successful run."""
    confidence: float
    visualization_code: str | None = None
    row_count: int = 0


class ErrorEvent(BaseModel):
    """``event: error`` — unexpected server-side failure."""
    message: str
