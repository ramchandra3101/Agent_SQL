"""Tests for the cost / risk guardrail.

Requires a live, seeded ``SQL_POC`` Postgres database. If the DB isn't
reachable, every test in this file is skipped — they're not unit tests,
they're integration tests against the real planner.
"""

from __future__ import annotations

import pytest

from src.db.connection import get_engine
from src.guardrails.cost import assess


# ---------------------------------------------------------------------------
#  Skip the whole module if the DB isn't up.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _require_db() -> None:
    try:
        with get_engine().connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"SQL_POC database not available: {e}")


# ---------------------------------------------------------------------------
#  Cheap, well-bounded queries — must pass.
# ---------------------------------------------------------------------------
def test_indexed_lookup_passes() -> None:
    result = assess("SELECT * FROM orders WHERE customer_id = 1")
    assert result.ok, f"expected ok, got: {result.reason} (flags={result.flags})"
    assert result.estimated_rows < 1_000


def test_small_table_scan_passes() -> None:
    # 200-row table — well under the 100k budget.
    result = assess("SELECT * FROM customers")
    assert result.ok, result.reason
    # Informational flags are expected since there's no WHERE / LIMIT.
    assert "no_where_clause" in result.flags
    assert "no_limit" in result.flags


def test_join_with_predicate_passes() -> None:
    sql = """
        SELECT c.country, SUM(oi.line_total) AS revenue
        FROM customers c
        JOIN orders o      ON o.customer_id = c.customer_id
        JOIN order_items oi ON oi.order_id   = o.order_id
        GROUP BY c.country
    """
    result = assess(sql)
    assert result.ok, result.reason
    assert "cartesian_join" not in result.flags


# ---------------------------------------------------------------------------
#  Risky queries — must be blocked.
# ---------------------------------------------------------------------------
def test_comma_join_is_cartesian() -> None:
    # 200 customers × 1000 orders = 200,000 rows.
    result = assess("SELECT * FROM customers, orders")
    assert not result.ok
    assert "cartesian_join" in result.flags
    assert result.estimated_rows >= 100_000


def test_explicit_cross_join_is_cartesian() -> None:
    result = assess("SELECT * FROM customers CROSS JOIN orders")
    assert not result.ok
    assert "cartesian_join" in result.flags


def test_over_row_budget_is_blocked() -> None:
    # Same query as the comma-join test but verifies the budget flag separately.
    result = assess("SELECT * FROM customers, orders")
    assert "over_row_budget" in result.flags
    assert "rows" in result.reason.lower()


# ---------------------------------------------------------------------------
#  Planner failures — surface as cost failures (defence in depth).
# ---------------------------------------------------------------------------
def test_invalid_sql_is_blocked() -> None:
    # References a column that doesn't exist — planner fails.
    result = assess("SELECT nonexistent_column FROM customers")
    assert not result.ok
    assert "explain_failed" in result.flags
