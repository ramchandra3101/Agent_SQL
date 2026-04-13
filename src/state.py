"""Graph state — the single source of truth passed between LangGraph nodes.

Every node receives the current `AgentState`, reads the fields it needs, and
returns a *partial* dict of the fields it updates. LangGraph merges the
partial into the canonical state between node runs.

Fields are grouped by lifecycle stage so it's obvious which node owns what.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# "sql"  → route through the SQL generation + execution pipeline
# "rag"  → off-schema question, route to the RAG fallback node
Route = Literal["sql", "rag"]

# What the human chose at a HITL breakpoint.
HumanAction = Literal["approve", "edit", "hint", "abort"]


class TableSchema(TypedDict):
    """A single table's contribution to the pruned schema prompt."""

    name: str
    columns: list[dict[str, str]]  # [{"name": "...", "type": "..."}]


class HumanFeedback(TypedDict, total=False):
    action: HumanAction
    edited_sql: str
    hint: str
    reason: str


class FinalResult(TypedDict, total=False):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    sql: str
    summary: str
    visualization_code: str
    confidence_score: float
    error: str


class AgentState(TypedDict, total=False):
    # ---- Input -------------------------------------------------------------
    user_query: str
    dialect: str  # "postgresql", "sqlite", ...
    thread_id: str  # LangGraph checkpoint thread — one per conversation

    # ---- Routing -----------------------------------------------------------
    route: Route
    route_reason: str

    # ---- Schema understanding ---------------------------------------------
    full_schema_summary: str
    selected_schema: list[TableSchema]

    # ---- Generation --------------------------------------------------------
    few_shot_examples: list[dict[str, str]]
    generated_sql: str
    correction_hint: str | None

    # ---- Guardrails --------------------------------------------------------
    validator_ok: bool
    validator_reason: str
    cost_ok: bool
    cost_reason: str
    risk_flags: list[str]            # e.g. ["cartesian_join", "no_where_clause"]
    estimated_rows: int              # planner estimate from EXPLAIN

    # ---- Human-in-the-loop -------------------------------------------------
    paused: bool
    human_feedback: HumanFeedback | None

    # ---- Execution ---------------------------------------------------------
    execution_error: str | None
    raw_rows: list[dict[str, Any]]
    columns: list[str]
    row_count: int

    # ---- Output ------------------------------------------------------------
    final_result: FinalResult
    natural_language_answer: str
    visualization_code: str | None
    confidence_score: float

    # ---- Loop accounting ---------------------------------------------------
    retry_count: int          # self-correction loops used
    generation_count: int     # total SQL generator invocations
    escalated: bool           # True when budgets are exhausted


def initial_state(user_query: str, thread_id: str) -> AgentState:
    """Build a zeroed-out state for a new user question."""
    return AgentState(
        user_query=user_query,
        thread_id=thread_id,
        dialect="postgresql",
        route="sql",
        route_reason="",
        full_schema_summary="",
        selected_schema=[],
        few_shot_examples=[],
        generated_sql="",
        correction_hint=None,
        validator_ok=False,
        validator_reason="",
        cost_ok=False,
        cost_reason="",
        risk_flags=[],
        estimated_rows=0,
        paused=False,
        human_feedback=None,
        execution_error=None,
        raw_rows=[],
        columns=[],
        row_count=0,
        final_result={},
        natural_language_answer="",
        visualization_code=None,
        confidence_score=0.0,
        retry_count=0,
        generation_count=0,
        escalated=False,
    )
