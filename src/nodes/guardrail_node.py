"""Guardrail node — runs the validator and cost checks as a single graph step.

Wraps the two guardrails built in Phase B:

* :func:`src.guardrails.validator.is_safe` — rejects destructive SQL.
* :func:`src.guardrails.cost.assess`       — estimates planner cost and flags
  cartesian joins, over-budget scans, unbounded queries.

The node writes three kinds of signal onto state:

* **Hard rejects** (``validator_ok=False``) — graph routes to self-correction
  or escalation. We never show the user an error; the agent retries.
* **Risky-but-runnable** (``cost_ok=False``) — graph pauses for HITL. The
  human sees the SQL, the flags, and the planner estimate, then approves or
  edits.
* **Informational flags** (``no_where_clause``, ``no_limit``) — travel with
  state so the formatter can surface them to the user for transparency,
  even when the query runs.

Skipping cost when validator fails is deliberate: EXPLAIN on a destructive
statement would execute it for some DDL variants, and we never want that.
"""

from __future__ import annotations

from typing import Any

from ..guardrails.cost import assess
from ..guardrails.validator import is_safe
from ..state import AgentState


def guardrail_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: run validator + cost and update state."""
    sql = state.get("generated_sql", "")

    # ---- 1. Destructive-keyword validator ---------------------------------
    v = is_safe(sql)
    update: dict[str, Any] = {
        "validator_ok": v.ok,
        "validator_reason": v.reason,
    }

    # If the validator rejected the SQL, skip cost analysis entirely. Running
    # EXPLAIN on destructive statements is risky for certain dialects/verbs
    # (some variants can still have side effects).
    if not v.ok:
        update.update(
            {
                "cost_ok": False,
                "cost_reason": "skipped: validator rejected SQL",
                "risk_flags": [],
                "estimated_rows": 0,
                "paused": False,  # hard reject — feeds self-correction, not HITL
            }
        )
        return update

    # ---- 2. Cost / risk guardrail -----------------------------------------
    c = assess(sql)
    update.update(
        {
            "cost_ok": c.ok,
            "cost_reason": c.reason,
            "risk_flags": list(c.flags),
            "estimated_rows": c.estimated_rows,
            # Runnable but risky → hand off to the human.
            "paused": not c.ok,
        }
    )
    return update


# ---------------------------------------------------------------------------
#  Sanity check:  python -m src.nodes.guardrail_node
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ..state import initial_state

    cases = [
        ("safe, indexed", "SELECT * FROM orders WHERE customer_id = 1"),
        ("destructive",   "DELETE FROM customers"),
        ("cartesian",     "SELECT * FROM customers, orders"),
        ("unbounded",     "SELECT * FROM customers"),
    ]
    for label, sql in cases:
        state = initial_state("probe", thread_id="probe")
        state["generated_sql"] = sql
        result = guardrail_node(state)
        print(f"[{label}] sql={sql}")
        print(f"  validator_ok={result['validator_ok']}  "
              f"cost_ok={result['cost_ok']}  "
              f"paused={result['paused']}  "
              f"flags={result['risk_flags']}")
        print(f"  reason: validator={result['validator_reason']!r} "
              f"cost={result['cost_reason']!r}")
        print()
