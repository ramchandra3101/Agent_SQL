"""End-to-end smoke test for the assembled graph.

Not a unit test — it hits the real ``SQL_POC`` database AND the OpenAI API.
Skipped automatically when either is unavailable so CI (or an offline run)
doesn't fail noisily. When green, it proves the whole pipeline
(router → schema_selector → sql_generator → guardrail → executor →
formatter → visualizer) holds together on plausible questions.

We assert only the shape and sanity of the result, not exact row counts —
the LLM can legitimately produce different-but-correct SQL across runs.
"""

from __future__ import annotations

import os
import uuid

import pytest

from src.config import settings
from src.db.connection import get_engine
from src.retrieval.golden_sql import count as golden_count
from src.retrieval.golden_sql import seed as golden_seed


# ---------------------------------------------------------------------------
#  Module-level skip gates — both DB and OpenAI must be reachable.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _require_db_and_openai() -> None:
    try:
        with get_engine().connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"SQL_POC database not available: {e}")

    if not settings.openai_api_key or not os.environ.get("OPENAI_API_KEY"):
        # settings reads from .env; also honor a missing env var.
        pytest.skip("OPENAI_API_KEY not set")

    # Ensure Golden SQL examples exist so few-shot retrieval is meaningful.
    if golden_count() == 0:
        golden_seed()


# ---------------------------------------------------------------------------
#  Lazy graph builder — reuse across tests in this module.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def graph():
    from src.graph import build_graph

    return build_graph()


def _run(graph, question: str) -> dict:
    from src.state import initial_state

    thread_id = f"smoke-{uuid.uuid4().hex[:8]}"
    state = initial_state(question, thread_id=thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(state, config=config)


# ---------------------------------------------------------------------------
#  Canned schema-grounded questions — each should return rows + a summary.
# ---------------------------------------------------------------------------
SQL_QUESTIONS = [
    "How many customers do we have in each country?",
    "What is the total revenue per country?",
    "List the top 5 products by revenue.",
]


@pytest.mark.parametrize("question", SQL_QUESTIONS)
def test_sql_questions_run_end_to_end(graph, question: str) -> None:
    final = _run(graph, question)

    # Should have taken the SQL route.
    assert final.get("route") == "sql", f"expected sql route, got {final.get('route')}"

    # No graph-level error and no HITL pause expected for these canned Qs.
    assert not final.get("escalated"), (
        f"run escalated: reason={final.get('final_result', {}).get('error')}"
    )

    # SQL was generated and rows came back.
    assert final.get("generated_sql"), "no SQL generated"
    assert final.get("row_count", 0) > 0, "executor returned zero rows"
    assert final.get("columns"), "no columns in result"

    # Formatter produced a grounded NL answer.
    answer = final.get("natural_language_answer", "")
    assert answer and len(answer) > 20, f"answer too short: {answer!r}"


# ---------------------------------------------------------------------------
#  Off-schema question — must take the RAG fallback path.
# ---------------------------------------------------------------------------
def test_off_schema_question_takes_rag_path(graph) -> None:
    final = _run(graph, "What is the capital of France?")

    assert final.get("route") == "rag", (
        f"expected rag route, got {final.get('route')}"
    )
    assert final.get("natural_language_answer"), "rag_fallback returned nothing"
    # Low confidence by design (ungrounded).
    assert final.get("confidence_score", 1.0) <= 0.5
