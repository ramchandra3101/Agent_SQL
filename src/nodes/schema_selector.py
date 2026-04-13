"""Schema selector node — prunes the schema to just the relevant tables.

The SQL generator (Step 12) writes better SQL and burns fewer tokens when
its prompt contains only the tables the question actually needs. For a
question like *"top customers by spend"* we want
``[customers, orders, order_items]`` — not the full schema, and not
``[products]`` which isn't involved.

Design notes:

* The LLM sees the schema summary and returns a JSON list of table names.
* We **validate** the returned names against the real schema, dropping any
  hallucinated tables. If the LLM returns nothing valid, we fall back to
  the full table list — correctness over token savings.
* Output goes on state as ``selected_schema`` (list of ``TableSchema``
  entries matching the ``AgentState`` type) so downstream nodes can render
  prompts without re-introspecting.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..db.introspect import load_schema, schema_summary
from ..llm import get_llm
from ..state import AgentState, TableSchema


_SYSTEM_PROMPT = """You are the schema selector for a text-to-SQL agent.

Given the database schema and a user question, return the MINIMAL set of
tables required to answer the question. Include every table that must be
joined. Exclude tables that are not needed.

Return STRICT JSON with exactly one key:
  - "tables": an array of table names (strings)

Rules:
* Use only table names that appear in the provided schema.
* If joins are required, include every intermediate table.
* Do NOT include markdown, code fences, or any text outside the JSON object.
"""


def _parse_tables(raw: str, valid: set[str]) -> list[str]:
    """Parse the LLM response and keep only table names that really exist."""
    try:
        data = json.loads(raw.strip())
        names = data.get("tables", [])
        if isinstance(names, list):
            return [n for n in names if isinstance(n, str) and n in valid]
    except (json.JSONDecodeError, AttributeError):
        pass
    return []


def _tables_to_schema(names: list[str]) -> list[TableSchema]:
    """Project selected table names into the ``TableSchema`` shape on state."""
    full = load_schema()
    out: list[TableSchema] = []
    for n in names:
        table = full.get(n)
        if table is None:
            continue
        out.append(
            TableSchema(
                name=table["name"],
                columns=[
                    {"name": c["name"], "type": c["type"]}
                    for c in table["columns"]
                ],
            )
        )
    return out


def select_schema_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: pick the minimal table list for the question."""
    question = state["user_query"]
    summary = state.get("full_schema_summary") or schema_summary()
    all_tables = set(load_schema().keys())

    llm = get_llm(temperature=0.0, max_tokens=200)
    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Schema:\n{summary}\n\n"
                    f"User question: {question}\n\n"
                    f"Respond with JSON only."
                )
            ),
        ]
    )

    selected = _parse_tables(response.content, all_tables)

    # Fallback: if the LLM picked nothing valid, pass the full schema so
    # the generator at least has a chance. Correctness > token savings.
    if not selected:
        selected = sorted(all_tables)

    return {
        "selected_schema": _tables_to_schema(selected),
        "full_schema_summary": summary,
    }


# ---------------------------------------------------------------------------
#  Sanity check:  python -m src.nodes.schema_selector
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ..state import initial_state

    for q in [
        "Top customers by spend",
        "How many products are low on stock?",
        "Revenue per country",
        "List all orders for customer 1",
    ]:
        state = initial_state(q, thread_id="probe")
        result = select_schema_node(state)
        names = [t["name"] for t in result["selected_schema"]]
        print(f"Q: {q}\n  → {names}\n")
