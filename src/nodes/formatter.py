"""Formatter node — turns raw rows into a human-friendly answer.

Final-answer node for the SQL path. Produces three artifacts on state:

* ``final_result`` — a dict with columns, rows, row_count, sql, summary,
  confidence_score. This is what the CLI / API returns to the caller.
* ``natural_language_answer`` — a short grounded summary of the rows for
  transparency. We pass the actual result rows (capped) into the prompt
  so the LLM can't confabulate numbers.
* ``confidence_score`` — a simple heuristic in [0, 1] based on whether the
  run went straight through (1.0) or needed retries / HITL edits (lower).
  Not a statistical guarantee — just a signal for the UI.

Design notes:

* We cap the rows sent to the LLM at 50. Past that, summaries don't get
  better and tokens balloon.
* We render a rich.Table to a string (rather than printing it directly) so
  the node stays pure — the CLI decides how to display.
"""

from __future__ import annotations

import io
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.table import Table

from ..llm import get_llm
from ..state import AgentState


_SYSTEM_PROMPT = """You are the answer writer for a text-to-SQL agent.

You will be given the user's question and the rows that the SQL query
returned. Write a short, factual answer (2-4 sentences) that references
the data. Rules:

* Only cite numbers that appear in the rows. Never invent values.
* If the result set is empty, say so plainly.
* No markdown, no code fences, no bullet lists — plain prose.
"""

_MAX_SUMMARY_ROWS = 50


def _render_table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Render rows as a Rich-formatted table string."""
    if not columns:
        return "(no columns)"
    table = Table(show_header=True, header_style="bold")
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*[str(row.get(c, "")) for c in columns])

    buf = io.StringIO()
    Console(file=buf, force_terminal=False, width=120).print(table)
    return buf.getvalue().rstrip()


def _confidence(state: AgentState) -> float:
    """Cheap heuristic: start at 1.0, penalize retries and HITL involvement."""
    score = 1.0
    score -= 0.15 * state.get("retry_count", 0)
    if state.get("human_feedback"):
        score -= 0.2
    if state.get("risk_flags"):
        score -= 0.05 * len(state["risk_flags"])
    return max(0.0, min(1.0, score))


def _summarize(question: str, columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Ask the LLM for a grounded one-paragraph summary of the rows."""
    if not rows:
        return "The query ran successfully but returned no rows."

    preview = rows[:_MAX_SUMMARY_ROWS]
    llm = get_llm(temperature=0.0, max_tokens=250)
    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User question: {question}\n\n"
                    f"Columns: {columns}\n"
                    f"Rows (up to {_MAX_SUMMARY_ROWS}):\n{preview}\n\n"
                    f"Write the answer now."
                )
            ),
        ]
    )
    return response.content.strip()


def format_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: build the final answer payload."""
    question = state.get("user_query", "")
    sql = state.get("generated_sql", "")
    columns = state.get("columns") or []
    rows = state.get("raw_rows") or []
    row_count = state.get("row_count", 0)

    # If the executor failed and we reached formatter anyway (escalation,
    # abort), surface the error clearly instead of inventing a summary.
    if state.get("execution_error"):
        err = state["execution_error"]
        return {
            "final_result": {
                "error": err,
                "sql": sql,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "summary": f"Query could not be executed: {err}",
                "confidence_score": 0.0,
            },
            "natural_language_answer": f"Query failed: {err}",
            "confidence_score": 0.0,
        }

    summary = _summarize(question, columns, rows)
    confidence = _confidence(state)
    table_str = _render_table(columns, rows[:_MAX_SUMMARY_ROWS])

    return {
        "final_result": {
            "columns": columns,
            "rows": rows,
            "row_count": row_count,
            "sql": sql,
            "summary": summary,
            "confidence_score": confidence,
            "table": table_str,
        },
        "natural_language_answer": summary,
        "confidence_score": confidence,
    }


# ---------------------------------------------------------------------------
#  Sanity check:  python -m src.nodes.formatter
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ..state import initial_state

    state = initial_state("How many customers per country?", thread_id="probe")
    state["generated_sql"] = "SELECT country, COUNT(*) AS n FROM customers GROUP BY country"
    state["columns"] = ["country", "n"]
    state["raw_rows"] = [
        {"country": "AU", "n": 30},
        {"country": "FR", "n": 25},
        {"country": "UK", "n": 25},
        {"country": "US", "n": 60},
        {"country": "DE", "n": 20},
    ]
    state["row_count"] = 5

    out = format_node(state)
    print(out["final_result"]["table"])
    print()
    print("Summary:", out["natural_language_answer"])
    print("Confidence:", out["confidence_score"])
