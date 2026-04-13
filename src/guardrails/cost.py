"""Cost / risk guardrail.

The validator (Step 7) decides *is this query allowed?*. This module decides
*is it safe to actually run?*. It uses Postgres' own planner output via
``EXPLAIN (FORMAT JSON)`` plus a small dose of SQL parsing to detect:

* **Over-budget scans** — the planner estimates more rows than
  ``ROW_SCAN_BUDGET`` (default 100,000).
* **Cartesian joins** — a ``Nested Loop`` with no join condition, or an
  explicit ``CROSS JOIN`` / comma-join in the SQL.
* **Unbounded queries** — no ``WHERE`` and no ``LIMIT``. Often harmless
  on small tables, but worth surfacing to the human for transparency.

Returns a :class:`CostAssessment`. The graph routes to HITL when ``ok``
is ``False``; informational flags (``no_where_clause``, ``no_limit``)
travel with the state for the formatter to display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sqlparse
from sqlparse.tokens import Keyword

from ..config import settings
from ..db.connection import explain_query_plan


@dataclass(frozen=True)
class CostAssessment:
    ok: bool
    reason: str
    flags: list[str] = field(default_factory=list)
    estimated_rows: int = 0


# ---------------------------------------------------------------------------
#  Plan-tree walker
# ---------------------------------------------------------------------------
def _walk(node: dict[str, Any], visit) -> None:
    """Depth-first walk over Postgres' EXPLAIN JSON plan nodes."""
    visit(node)
    for child in node.get("Plans", []) or []:
        _walk(child, visit)


def _join_condition_keys() -> set[str]:
    """Plan-node keys that indicate a join *has* a predicate."""
    return {
        "Join Filter",
        "Hash Cond",
        "Merge Cond",
        "Index Cond",
        "Recheck Cond",
    }


def _detect_plan_risks(plan_root: dict[str, Any]) -> tuple[int, list[str]]:
    """Return ``(top_level_estimated_rows, flags_from_plan)``."""
    flags: list[str] = []

    estimated_rows = int(plan_root.get("Plan Rows", 0))

    cond_keys = _join_condition_keys()

    def visit(node: dict[str, Any]) -> None:
        node_type = node.get("Node Type", "")
        # Cartesian: a Nested Loop with NO join condition. Postgres exposes
        # the absence by simply not setting any of the *Cond / Join Filter keys.
        if node_type == "Nested Loop" and not (cond_keys & node.keys()):
            flag = "cartesian_join"
            if flag not in flags:
                flags.append(flag)
        # Sequential scan on a sizeable table (>10k planner rows) — worth
        # surfacing even if total rows are under budget.
        if node_type == "Seq Scan":
            rel = node.get("Relation Name")
            rows = int(node.get("Plan Rows", 0))
            if rel and rows > 10_000:
                flag = f"full_table_scan:{rel}"
                if flag not in flags:
                    flags.append(flag)

    _walk(plan_root, visit)
    return estimated_rows, flags


# ---------------------------------------------------------------------------
#  SQL-text inspector (catches things the plan can hide)
# ---------------------------------------------------------------------------
def _detect_sql_risks(sql: str) -> list[str]:
    flags: list[str] = []

    parsed = sqlparse.parse(sql)[0]
    keywords = [
        token.normalized.upper()
        for token in parsed.flatten()
        if token.ttype in Keyword
    ]
    keyword_set = set(keywords)

    if "CROSS JOIN" in keyword_set and "cartesian_join" not in flags:
        flags.append("cartesian_join")

    if "WHERE" not in keyword_set:
        flags.append("no_where_clause")

    if "LIMIT" not in keyword_set:
        flags.append("no_limit")

    return flags


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------
def assess(sql: str) -> CostAssessment:
    """Estimate cost/risk for ``sql`` and return a structured assessment."""
    explain = explain_query_plan(sql)
    if not explain["ok"]:
        # Treat planner failures as cost failures so the executor never
        # runs a query the planner couldn't even parse.
        return CostAssessment(
            ok=False,
            reason=f"EXPLAIN failed: {explain['error']}",
            flags=["explain_failed"],
        )

    plan_payload = explain["plan"]
    # EXPLAIN (FORMAT JSON) returns a list with one dict carrying a "Plan" key.
    if not plan_payload or "Plan" not in plan_payload[0]:
        return CostAssessment(
            ok=False, reason="EXPLAIN returned an unexpected shape",
            flags=["explain_unparseable"],
        )

    estimated_rows, plan_flags = _detect_plan_risks(plan_payload[0]["Plan"])
    sql_flags = _detect_sql_risks(sql)

    flags: list[str] = []
    for f in plan_flags + sql_flags:
        if f not in flags:
            flags.append(f)

    # Decide blocking vs informational.
    blocking_reasons: list[str] = []
    if estimated_rows > settings.row_scan_budget:
        flag = "over_row_budget"
        if flag not in flags:
            flags.append(flag)
        blocking_reasons.append(
            f"planner estimates {estimated_rows:,} rows "
            f"(budget {settings.row_scan_budget:,})"
        )
    if "cartesian_join" in flags:
        blocking_reasons.append("query contains a cartesian join")

    if blocking_reasons:
        return CostAssessment(
            ok=False,
            reason="; ".join(blocking_reasons),
            flags=flags,
            estimated_rows=estimated_rows,
        )

    return CostAssessment(
        ok=True,
        reason=f"planner estimates {estimated_rows:,} rows",
        flags=flags,
        estimated_rows=estimated_rows,
    )
