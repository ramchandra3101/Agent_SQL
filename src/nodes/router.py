"""Router node — decides whether a question belongs on the SQL path or RAG path.

First node after the user enters the graph. Given the live schema summary
and the user's question, an LLM returns one of:

* ``"sql"`` — the schema can answer this. Continue to schema selection →
  generation → guardrails → execution.
* ``"rag"`` — the question is off-schema (policy, definitions, prose). Skip
  SQL entirely and go to the RAG fallback node.

The reason string is preserved on state so the formatter can tell the user
*why* we routed the way we did (transparency > black-box).

Design notes:

* The router sees the **schema summary only** (table names + columns), not
  row data. We don't want the LLM leaking sample values into its reasoning.
* Output is parsed as JSON to keep it deterministic and easy to validate.
  If the LLM fumbles the JSON, we fall back to ``"sql"`` — the downstream
  validator + cost guardrails will catch a bogus generation anyway, and a
  false-negative "rag" would silently drop legitimate questions.
* Temperature is 0. Routing should be reproducible for the same input.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..db.introspect import schema_summary
from ..llm import get_llm
from ..state import AgentState


_SYSTEM_PROMPT = """You are the router for a text-to-SQL agent.

Given the database schema summary and a user question, decide whether the
question can be answered by running a SQL query against that schema.

Return STRICT JSON with exactly two keys:
  - "route":  "sql"  if the schema contains the tables/columns needed
              "rag"  if the question is about policy, definitions, general
                     knowledge, or anything not present in the schema
  - "reason": one short sentence explaining your choice

Rules:
* If the question references a table or column that exists in the schema,
  choose "sql".
* If the question is a greeting, meta-question ("what can you do?"), or
  about concepts not in the schema, choose "rag".
* Do NOT include markdown, code fences, or any text outside the JSON object.
"""


def _parse_route(raw: str) -> tuple[str, str]:
    """Return (route, reason). Falls back to ('sql', ...) on parse failure."""
    try:
        data = json.loads(raw.strip())
        route = str(data.get("route", "")).lower().strip()
        reason = str(data.get("reason", "")).strip()
        if route in {"sql", "rag"}:
            return route, reason or f"router returned {route}"
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: assume SQL so the downstream guardrails get a chance to
    # reject a truly bad question, rather than silently dropping it.
    return "sql", f"router response unparseable; defaulted to sql. raw={raw[:120]!r}"


def route_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: classify the user question as 'sql' or 'rag'."""
    question = state["user_query"]
    summary = state.get("full_schema_summary") or schema_summary()

    llm = get_llm(temperature=0.0, max_tokens=200)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Schema summary:\n{summary}\n\n"
                f"User question: {question}\n\n"
                f"Respond with JSON only."
            )
        ),
    ]
    response = llm.invoke(messages)
    route, reason = _parse_route(response.content)

    return {
        "route": route,
        "route_reason": reason,
        # Cache the summary so downstream nodes don't re-introspect.
        "full_schema_summary": summary,
    }


# ---------------------------------------------------------------------------
#  Sanity check:  python -m src.nodes.router
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ..state import initial_state

    for q in [
        "How many customers do we have in each country?",
        "What is the capital of France?",
        "Show top 5 products by revenue.",
        "What can you do?",
    ]:
        state = initial_state(q, thread_id="probe")
        result = route_node(state)
        print(f"Q: {q}")
        print(f"  → route={result['route']}  reason={result['route_reason']}")
        print()
