"""Executor node — actually runs the SQL against Postgres.

Everything up to this point has been analysis: routing, schema selection,
generation, guardrails. This node is the only place in the graph that
touches live data.

It delegates the heavy lifting to :func:`src.db.connection.run_query`,
which already:

* Asserts read-only + single statement (defence in depth on top of the
  validator).
* Wraps the execution in ``SET TRANSACTION READ ONLY`` with a statement
  timeout.
* Returns a structured result (never raises on query errors) so the
  self-correction node can read the error text and retry.

State updates:

* On success → ``raw_rows``, ``columns``, ``row_count``, and clears
  ``execution_error``.
* On failure → ``execution_error`` is set; row fields are cleared. The
  graph's retry edge will route to self-correction.
"""

from __future__ import annotations

from typing import Any

from ..config import settings
from ..db.connection import run_query
from ..state import AgentState


def execute_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: execute ``state['generated_sql']`` read-only."""
    sql = state.get("generated_sql", "")
    if not sql:
        return {
            "execution_error": "no SQL to execute",
            "raw_rows": [],
            "columns": [],
            "row_count": 0,
        }

    # Cap rows at the configured row-scan budget so an unexpectedly large
    # result set never blows up memory or the formatter's prompt.
    result = run_query(sql, row_limit=settings.row_scan_budget)

    if not result["ok"]:
        return {
            "execution_error": result["error"],
            "raw_rows": [],
            "columns": [],
            "row_count": 0,
        }

    return {
        "execution_error": None,
        "raw_rows": result["rows"],
        "columns": result["columns"],
        "row_count": result["row_count"],
    }


# ---------------------------------------------------------------------------
#  Sanity check:  python -m src.nodes.executor
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ..state import initial_state

    cases = [
        ("happy path",   "SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY n DESC LIMIT 5"),
        ("bad column",   "SELECT nonexistent FROM customers"),
        ("read-only guard", "DELETE FROM customers"),
    ]
    for label, sql in cases:
        state = initial_state("probe", thread_id="probe")
        state["generated_sql"] = sql
        out = execute_node(state)
        print(f"[{label}] sql={sql}")
        if out["execution_error"]:
            print(f"  error: {out['execution_error']}")
        else:
            print(f"  columns={out['columns']}  rows={out['row_count']}")
            for row in out["raw_rows"][:3]:
                print(f"    {row}")
        print()
