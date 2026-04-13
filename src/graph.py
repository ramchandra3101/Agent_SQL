"""LangGraph assembly — wires every node into the agent graph.

Topology (matches plan.md Step 20):

    START → router
    router         ──sql→ schema_selector
                   ──rag→ rag_fallback → END
    schema_selector → sql_generator → guardrail_node
    guardrail_node ──validator_fail→ formatter (error path)
                   ──cost_risk→      hitl → executor
                   ──ok→             executor
    executor       ──ok→              formatter → visualizer → END
                   ──err & budget→   self_correction → sql_generator
                   ──err & exhausted→ hitl → executor
    hitl           ──approve/edit→   executor
                   ──hint→           sql_generator
                   ──abort→          formatter

Persistence:

* ``PostgresSaver`` backed by the same ``SQL_POC`` database. A checkpoint
  row is written after every node, which is what lets HITL pause and later
  resume without losing state.
* ``interrupt_before=["hitl"]`` — the graph freezes *before* running the
  HITL node. The CLI reads the pending state, asks the human, writes
  ``human_feedback`` back, and resumes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg import Connection

from .config import settings
from .nodes.executor import execute_node
from .nodes.formatter import format_node
from .nodes.guardrail_node import guardrail_node
from .nodes.hitl import hitl_node
from .nodes.rag_fallback import rag_fallback_node
from .nodes.router import route_node
from .nodes.schema_selector import select_schema_node
from .nodes.self_correction import self_correction_node
from .nodes.sql_generator import generate_sql_node
from .nodes.visualizer import visualize_node
from .state import AgentState


# ---------------------------------------------------------------------------
#  Conditional edge functions — small, pure, easy to test.
# ---------------------------------------------------------------------------
def _after_router(state: AgentState) -> Literal["schema_selector", "rag_fallback"]:
    return "rag_fallback" if state.get("route") == "rag" else "schema_selector"


def _after_guardrail(state: AgentState) -> Literal["formatter", "hitl", "executor"]:
    # Destructive SQL from the model — surface as error rather than looping.
    if not state.get("validator_ok"):
        return "formatter"
    # Runnable but risky — human approves before executor sees it.
    if not state.get("cost_ok"):
        return "hitl"
    return "executor"


def _after_executor(
    state: AgentState,
) -> Literal["formatter", "self_correction", "hitl"]:
    if not state.get("execution_error"):
        return "formatter"
    # Retry budget left → coach the generator.
    if state.get("retry_count", 0) < settings.max_retries:
        return "self_correction"
    # Budget exhausted → ask the human.
    return "hitl"


def _after_hitl(state: AgentState) -> Literal["executor", "sql_generator", "formatter"]:
    # An abort sets escalated + final_result.error; go straight to formatter
    # which knows how to surface execution_error / aborted states.
    if state.get("escalated") and state.get("final_result", {}).get("error"):
        return "formatter"
    # The hitl node maps "hint" → correction_hint but does NOT set a new SQL.
    # Presence of correction_hint means we need the generator again.
    if state.get("correction_hint"):
        return "sql_generator"
    # approve/edit path — executor runs the (possibly edited) SQL.
    return "executor"


# ---------------------------------------------------------------------------
#  Checkpointer
# ---------------------------------------------------------------------------
def _checkpointer_conn_string() -> str:
    """Strip SQLAlchemy's ``+psycopg`` so langgraph-checkpoint-postgres is happy."""
    return settings.database_url.replace("postgresql+psycopg://", "postgresql://")


@lru_cache(maxsize=1)
def _get_checkpointer() -> PostgresSaver:
    """Build a PostgresSaver against ``SQL_POC`` and ensure its tables exist."""
    conn = Connection.connect(_checkpointer_conn_string(), autocommit=True)
    saver = PostgresSaver(conn)  # type: ignore[arg-type]
    saver.setup()  # idempotent: creates langgraph checkpoint tables on first run
    return saver


# ---------------------------------------------------------------------------
#  Graph builder
# ---------------------------------------------------------------------------
def build_graph() -> Any:
    """Compile and return the agent graph."""
    builder = StateGraph(AgentState)

    builder.add_node("router", route_node)
    builder.add_node("schema_selector", select_schema_node)
    builder.add_node("sql_generator", generate_sql_node)
    builder.add_node("guardrail_node", guardrail_node)
    builder.add_node("executor", execute_node)
    builder.add_node("self_correction", self_correction_node)
    builder.add_node("hitl", hitl_node)
    builder.add_node("formatter", format_node)
    builder.add_node("visualizer", visualize_node)
    builder.add_node("rag_fallback", rag_fallback_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _after_router,
        {"schema_selector": "schema_selector", "rag_fallback": "rag_fallback"},
    )
    builder.add_edge("schema_selector", "sql_generator")
    builder.add_edge("sql_generator", "guardrail_node")
    builder.add_conditional_edges(
        "guardrail_node",
        _after_guardrail,
        {"formatter": "formatter", "hitl": "hitl", "executor": "executor"},
    )
    builder.add_conditional_edges(
        "executor",
        _after_executor,
        {
            "formatter": "formatter",
            "self_correction": "self_correction",
            "hitl": "hitl",
        },
    )
    builder.add_edge("self_correction", "sql_generator")
    builder.add_conditional_edges(
        "hitl",
        _after_hitl,
        {
            "executor": "executor",
            "sql_generator": "sql_generator",
            "formatter": "formatter",
        },
    )
    builder.add_edge("formatter", "visualizer")
    builder.add_edge("visualizer", END)
    builder.add_edge("rag_fallback", END)

    return builder.compile(
        checkpointer=_get_checkpointer(),
        interrupt_before=["hitl"],
    )


# ---------------------------------------------------------------------------
#  Smoke test:  python -m src.graph
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .state import initial_state

    graph = build_graph()
    state = initial_state(
        "How many customers do we have in each country?",
        thread_id="smoke-1",
    )
    config = {"configurable": {"thread_id": state["thread_id"]}}
    final = graph.invoke(state, config=config)

    print("route:        ", final.get("route"))
    print("generated_sql:", final.get("generated_sql"))
    print("row_count:    ", final.get("row_count"))
    print("confidence:   ", final.get("confidence_score"))
    print()
    print("ANSWER:")
    print(final.get("natural_language_answer"))
