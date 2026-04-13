"""Self-correction node — turns a DB error into a corrective hint.

When the executor fails (bad column, type mismatch, missing table, etc.) we
*don't* ask the LLM to rewrite the SQL directly. Instead we ask it for a
short, instructive **hint** that the SQL generator consumes on its next
attempt. Separating "what went wrong" (this node) from "how to write the
query" (the generator) keeps each prompt focused and produces better fixes.

Flow:

1. Read the failed SQL, the Postgres error, and the selected schema.
2. LLM produces one paragraph of guidance (e.g. *"The column is
   ``line_total`` on ``order_items``, not ``orders``. Add a JOIN to
   ``order_items`` and aggregate that column instead."*).
3. Write the hint to ``correction_hint`` on state. The generator node
   (Step 12) reads and clears it on its next run.
4. Increment ``retry_count``. When ``retry_count > MAX_RETRIES`` we flip
   ``escalated=True`` so the graph routes to HITL instead of looping.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import settings
from ..db.introspect import table_ddl
from ..llm import get_llm
from ..state import AgentState


_SYSTEM_PROMPT = """You are the self-correction coach for a text-to-SQL agent.

A previous SQL query failed. You will be given:
  * the user's question
  * the schema that was used
  * the failed SQL
  * the database error message

Return ONE short paragraph (2-3 sentences) explaining how to fix the query.
Be specific: name the correct column, table, or join. Do NOT rewrite the
SQL — another node will do that. Do NOT use markdown or code fences.
"""


def self_correction_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: emit a corrective hint or escalate to HITL."""
    retry_count = state.get("retry_count", 0) + 1

    # Budget exhausted — escalate instead of looping forever.
    if retry_count > settings.max_retries:
        return {
            "retry_count": retry_count,
            "escalated": True,
            "paused": True,
            "correction_hint": None,
        }

    question = state.get("user_query", "")
    sql = state.get("generated_sql", "")
    error = state.get("execution_error") or "unknown error"
    table_names = [t["name"] for t in state.get("selected_schema") or []]
    ddl = table_ddl(table_names) if table_names else "(no schema selected)"

    llm = get_llm(temperature=0.0, max_tokens=250)
    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User question: {question}\n\n"
                    f"Schema:\n{ddl}\n\n"
                    f"Failed SQL:\n{sql}\n\n"
                    f"Database error:\n{error}\n\n"
                    f"Write the corrective hint now."
                )
            ),
        ]
    )
    hint = response.content.strip()

    return {
        "correction_hint": hint,
        "retry_count": retry_count,
        "escalated": False,
    }


# ---------------------------------------------------------------------------
#  Sanity check:  python -m src.nodes.self_correction
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .schema_selector import select_schema_node
    from ..state import initial_state

    state = initial_state("Revenue per country", thread_id="probe")
    state.update(select_schema_node(state))
    state["generated_sql"] = (
        "SELECT country, SUM(line_total) FROM customers GROUP BY country"
    )
    state["execution_error"] = (
        'column "line_total" does not exist'
    )

    out = self_correction_node(state)
    print(f"retry_count: {out['retry_count']}")
    print(f"escalated:   {out['escalated']}")
    print(f"hint:\n{out['correction_hint']}")
