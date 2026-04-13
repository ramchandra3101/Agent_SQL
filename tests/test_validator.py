"""Tests for the destructive-keyword validator.

The spec calls for a 100% rejection rate for destructive SQL keywords.
These tests lock that in across the variations attackers and LLMs both
produce: case mixing, comments, CTE wrappers, multi-statement payloads.

Equally important: these tests prove we **don't** false-positive on
legitimate SELECTs that happen to mention a forbidden word inside a
column name or string literal.
"""

from __future__ import annotations

import pytest

from src.guardrails.validator import is_safe


# ---------------------------------------------------------------------------
#  SAFE — these must pass
# ---------------------------------------------------------------------------
SAFE_QUERIES = [
    "SELECT 1",
    "SELECT * FROM customers",
    "SELECT * FROM customers WHERE country = 'US'",
    "SELECT name FROM customers ORDER BY name LIMIT 10",
    # Identifier that contains the substring 'UPDATE' — must NOT trip.
    "SELECT updated_at FROM events",
    # Identifier that contains 'DELETE' — must NOT trip.
    "SELECT deleted_flag FROM users",
    # Forbidden word inside a string literal — must NOT trip.
    "SELECT name FROM customers WHERE notes = 'do not delete'",
    # Forbidden word inside a SQL comment — must NOT trip.
    "SELECT 1 /* TODO: remember to DELETE old rows in cron */",
    "-- DELETE me later\nSELECT 1",
    # CTE that itself only reads.
    "WITH recent AS (SELECT * FROM orders WHERE order_date > '2024-01-01') SELECT COUNT(*) FROM recent",
    # Joins, aggregates, window functions — all just SELECT.
    """
    SELECT c.country, SUM(oi.line_total) AS revenue
    FROM customers c
    JOIN orders o      ON o.customer_id = c.customer_id
    JOIN order_items oi ON oi.order_id   = o.order_id
    GROUP BY c.country
    ORDER BY revenue DESC
    """,
]


@pytest.mark.parametrize("sql", SAFE_QUERIES)
def test_safe_queries_are_allowed(sql: str) -> None:
    result = is_safe(sql)
    assert result.ok, f"expected SAFE but got: {result.reason!r}\nSQL: {sql}"


# ---------------------------------------------------------------------------
#  DESTRUCTIVE — these must be rejected
# ---------------------------------------------------------------------------
DESTRUCTIVE_QUERIES = [
    # Plain DML / DDL.
    "DELETE FROM customers",
    "delete from customers",                 # lowercase
    "DeLeTe FROM customers",                 # mixed case
    "DROP TABLE customers",
    "TRUNCATE TABLE orders",
    "ALTER TABLE customers ADD COLUMN x INT",
    "INSERT INTO customers VALUES (1, 'x', 'x', 'x', NOW())",
    "UPDATE customers SET name = 'x' WHERE customer_id = 1",
    "GRANT SELECT ON customers TO public",
    "REVOKE ALL ON customers FROM public",
    "CREATE TABLE foo (id INT)",
    "MERGE INTO customers USING ...",
    "COPY customers TO '/tmp/leak.csv'",
    # Multi-statement injection — must reject because of count, not keyword.
    "SELECT 1; DELETE FROM customers",
    "SELECT 1; SELECT 2",
    # CTE with a write inside — Postgres allows this; we must not.
    "WITH gone AS (DELETE FROM customers RETURNING customer_id) SELECT * FROM gone",
    # Empty / whitespace.
    "",
    "   ",
    "\n\t",
]


@pytest.mark.parametrize("sql", DESTRUCTIVE_QUERIES)
def test_destructive_queries_are_rejected(sql: str) -> None:
    result = is_safe(sql)
    assert not result.ok, f"expected REJECT but was allowed.\nSQL: {sql}"
    assert result.reason, "rejection must include a non-empty reason"


# ---------------------------------------------------------------------------
#  100% rejection — explicit summary assertion the spec calls out
# ---------------------------------------------------------------------------
def test_100_percent_destructive_rejection() -> None:
    """Spec: '100% rejection rate for destructive SQL keywords.'"""
    rejections = sum(1 for q in DESTRUCTIVE_QUERIES if not is_safe(q).ok)
    assert rejections == len(DESTRUCTIVE_QUERIES), (
        f"only {rejections}/{len(DESTRUCTIVE_QUERIES)} destructive queries "
        f"were rejected — must be 100%"
    )
