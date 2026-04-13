"""SQL generator node — turns the question + pruned schema into a SELECT.

This is the core LLM call. Prompt inputs:

* **Dialect** — so the LLM picks the right date/string functions.
* **DDL for selected tables** — column types, primary keys, foreign keys.
  Everything the model needs to write correct JOINs without seeing every
  other table in the database.
* **Dynamic few-shot examples** — pulled from the Golden SQL pgvector store
  (Step 9). The retrieval is dialect- and schema-aware, so examples are
  always relevant.
* **Optional correction hint** — on a retry, the self-correction node puts
  the failed SQL and the DB error here so the model can fix it.

Token-accounting:

* Increments ``generation_count`` on state. The graph uses this to enforce
  the ``MAX_SQL_GENERATIONS`` budget from settings.
* Clears ``correction_hint`` after use so stale hints don't bleed into the
  next retry.

Output: the raw ``generated_sql`` string. The guardrail node (Step 13)
decides whether it's safe to run.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..db.introspect import table_ddl
from ..llm import get_llm
from ..retrieval.golden_sql import retrieve
from ..state import AgentState


_SYSTEM_PROMPT = """You are an expert SQL engineer.

Write a single SQL SELECT statement that answers the user's question using
ONLY the tables and columns in the provided schema. Follow these rules
strictly:

1. Output ONLY the SQL query. No markdown fences, no prose, no comments.
2. Produce exactly ONE statement. Do not chain multiple statements.
3. Read-only queries only. Never write INSERT / UPDATE / DELETE / DDL.
4. Use the specified SQL dialect's syntax.
5. When joining, use explicit JOIN ... ON ... syntax (never comma joins).
6. Prefer adding a LIMIT when the question implies a top-N answer.
7. If the question is ambiguous, make the most sensible interpretation.
"""


_SQL_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_sql_fences(text: str) -> str:
    """Remove markdown code fences the model may add despite instructions."""
    return _SQL_FENCE_RE.sub("", text).strip()


def _render_few_shot(examples: list[dict[str, Any]]) -> str:
    """Render retrieved Golden SQL examples as an in-prompt block."""
    if not examples:
        return ""
    lines = ["Here are similar questions we've answered before:"]
    for ex in examples:
        lines.append(f"\nQ: {ex['question']}\nSQL: {ex['sql']}")
    return "\n".join(lines)


def _render_correction_hint(hint: str | None) -> str:
    if not hint:
        return ""
    return (
        "\n\nPrevious attempt failed. Use this feedback to fix the query:\n"
        f"{hint}"
    )


def generate_sql_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: produce a single SELECT for ``state['user_query']``."""
    question = state["user_query"]
    dialect = state.get("dialect", "postgresql")
    selected = state.get("selected_schema") or []
    table_names = [t["name"] for t in selected]

    ddl = table_ddl(table_names) if table_names else ""

    # Dynamic few-shot from the Golden SQL store. Best-effort — if retrieval
    # errors out (DB down, store empty) we still run the generator without
    # examples rather than failing the whole graph.
    try:
        examples = retrieve(
            question, dialect=dialect, tables=table_names, k=3
        )
    except Exception:  # noqa: BLE001
        examples = []

    user_prompt = (
        f"Dialect: {dialect}\n\n"
        f"Schema:\n{ddl}\n\n"
        f"{_render_few_shot(examples)}\n\n"
        f"User question: {question}"
        f"{_render_correction_hint(state.get('correction_hint'))}\n\n"
        f"Write the SQL query now."
    )

    llm = get_llm(temperature=0.0, max_tokens=600)
    response = llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )
    sql = _strip_sql_fences(response.content).rstrip(";").strip()

    return {
        "generated_sql": sql,
        "few_shot_examples": [
            {"question": e["question"], "sql": e["sql"]} for e in examples
        ],
        "generation_count": state.get("generation_count", 0) + 1,
        # Consume the correction hint so it doesn't leak into a future retry.
        "correction_hint": None,
    }


# ---------------------------------------------------------------------------
#  Sanity check:  python -m src.nodes.sql_generator
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from .schema_selector import select_schema_node
    from ..state import initial_state

    for q in [
        "How many customers do we have in each country?",
        "Top 5 products by revenue",
        "Orders for customer 1",
    ]:
        state = initial_state(q, thread_id="probe")
        state.update(select_schema_node(state))
        out = generate_sql_node(state)
        print(f"Q: {q}")
        print(f"SQL:\n{out['generated_sql']}")
        print(f"(examples used: {len(out['few_shot_examples'])}, "
              f"gen_count: {out['generation_count']})")
        print()
