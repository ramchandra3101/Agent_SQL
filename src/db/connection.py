"""Read-only SQL execution wrapper.

This is the **only** place in the app that talks to Postgres to run
user-generated SQL. Every query goes through :func:`run_query`, which:

1. Re-checks that the statement is a single read-only ``SELECT`` / ``WITH``
   (defence in depth on top of the validator node — this is the last line
   before Postgres sees the query).
2. Opens an explicit transaction with
   ``SET TRANSACTION READ ONLY`` + ``SET LOCAL statement_timeout``.
   Even though the POC connects as the ``postgres`` superuser, the
   transaction itself cannot write or run longer than the configured
   timeout.
3. Returns a **structured result dict** (never raises on query errors) so
   the self-correction node can inspect the failure and try again.

Two other helpers:

* :func:`get_engine`      — lazily builds a cached SQLAlchemy engine.
* :func:`explain_query_plan` — runs ``EXPLAIN (FORMAT JSON)`` for the cost
  guardrail (Step 8).
"""

from __future__ import annotations

import re, json
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import sqlparse
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from ..config import settings

# ----------------------------------------------------------------------------
#  Engine (singleton, lazily built)
# ----------------------------------------------------------------------------
_engine: Engine | None = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    """Return a process-wide cached SQLAlchemy engine for ``SQL_POC``."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = create_engine(
                settings.database_url,
                future=True,
                pool_pre_ping=True,
            )
    return _engine


def get_dialect() -> str:
    """Return the active SQL dialect name (e.g. 'postgresql')."""
    return get_engine().dialect.name


# ----------------------------------------------------------------------------
#  Read-only enforcement — the last line of defence before Postgres
# ----------------------------------------------------------------------------
_READ_ONLY_STATEMENT_TYPES = {"SELECT"}

# Very small blocklist used only as a belt-and-braces check in case the
# validator node (Step 7) is ever bypassed. The full, production-quality
# keyword list lives in settings.forbidden_keywords and is enforced by the
# validator itself.
_DESTRUCTIVE_KEYWORD_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|"
    r"REPLACE|MERGE|CALL|EXEC|EXECUTE|COPY)\b",
    re.IGNORECASE,
)


def assert_read_only(sql: str) -> None:
    """Raise ``ValueError`` unless ``sql`` is a single read-only statement.

    The validator node performs the primary check earlier in the graph;
    this function exists as a final, unconditional guard that runs regardless
    of which code path produced the SQL.
    """
    if not sql or not sql.strip():
        raise ValueError("Empty SQL statement.")

    # Single statement only.
    statements = [s for s in sqlparse.split(sql) if s.strip()]
    if len(statements) != 1:
        raise ValueError(
            f"Only a single SQL statement is allowed; got {len(statements)}."
        )

    stmt = statements[0]
    parsed = sqlparse.parse(stmt)[0]

    # Must be a SELECT (sqlparse reports CTEs as SELECT once resolved).
    stmt_type = parsed.get_type().upper()
    if stmt_type == "UNKNOWN":
        # Could be a WITH ... SELECT; walk tokens to find the leading DML/DDL.
        for token in parsed.flatten():
            if token.ttype is sqlparse.tokens.Keyword.CTE:
                stmt_type = "SELECT"
                break
            if token.ttype is sqlparse.tokens.Keyword.DML:
                stmt_type = token.normalized.upper()
                break

    if stmt_type not in _READ_ONLY_STATEMENT_TYPES:
        raise ValueError(
            f"Refusing to execute non-read-only statement type: {stmt_type}"
        )

    # Final keyword sweep (catches destructive statements hidden inside CTEs
    # like `WITH x AS (DELETE ... RETURNING ...) SELECT * FROM x`).
    # sqlparse strips string literals, so regex hits are real keywords.
    stripped = _strip_string_literals(stmt)
    match = _DESTRUCTIVE_KEYWORD_RE.search(stripped)
    if match:
        raise ValueError(
            f"Destructive keyword detected in SQL: {match.group(0).upper()}"
        )


def _strip_string_literals(sql: str) -> str:
    """Remove ``'...'`` and ``"..."`` literals so regex scans only hit code."""
    out: list[str] = []
    tokens = sqlparse.parse(sql)[0].flatten()
    for tok in tokens:
        if tok.ttype in (
            sqlparse.tokens.Literal.String.Single,
            sqlparse.tokens.Literal.String.Symbol,
        ):
            out.append(" ")
        else:
            out.append(str(tok))
    return "".join(out)


# ----------------------------------------------------------------------------
#  Transaction helper
# ----------------------------------------------------------------------------
@contextmanager
def _read_only_transaction() -> Iterator[Any]:
    """Yield a connection inside a read-only, time-bounded transaction."""
    eng = get_engine()
    with eng.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(
                text(
                    f"SET LOCAL statement_timeout = "
                    f"{settings.query_timeout_seconds * 1000}"
                )
            )
            yield conn
            trans.commit()
        except Exception:
            trans.rollback()
            raise


# ----------------------------------------------------------------------------
#  Public API — run a query, return a structured result
# ----------------------------------------------------------------------------
def run_query(sql: str, row_limit: int = 1000) -> dict[str, Any]:
    """Execute ``sql`` read-only and return a structured result dict.

    Returns one of::

        { "ok": True,
          "columns": ["col1", "col2", ...],
          "rows":    [{"col1": ..., "col2": ...}, ...],
          "row_count": int,
          "truncated": bool }

        { "ok": False, "error": "<message>" }

    Never raises on query errors — the self-correction node needs the error
    string. Only programmer errors (bad arguments) propagate.
    """
    try:
        assert_read_only(sql)
    except ValueError as e:
        return {"ok": False, "error": f"read-only guard: {e}"}

    try:
        with _read_only_transaction() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows: list[dict[str, Any]] = []
            truncated = False
            for i, row in enumerate(result):
                if i >= row_limit:
                    truncated = True
                    break
                rows.append(dict(zip(columns, row)))
            return {
                "ok": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": truncated,
            }
    except SQLAlchemyError as e:
        # Unwrap DB-API cause so the LLM gets the real Postgres message,
        # not SQLAlchemy's wrapper boilerplate.
        cause = e.__cause__ or e
        return {"ok": False, "error": str(cause).strip()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e).strip()}


# ----------------------------------------------------------------------------
#  EXPLAIN helper — used by the cost guardrail (Step 8)
# ----------------------------------------------------------------------------
def explain_query_plan(sql: str) -> dict[str, Any]:
    """Return the Postgres planner output for ``sql`` as a dict.

    Shape::

        { "ok": True,  "plan": <parsed JSON from EXPLAIN (FORMAT JSON)> }
        { "ok": False, "error": "<message>" }

    Uses ``EXPLAIN`` *without* ``ANALYZE`` so nothing is actually executed —
    we only want planner row estimates for the cost guardrail.
    """
    try:
        assert_read_only(sql)
    except ValueError as e:
        return {"ok": False, "error": f"read-only guard: {e}"}

    try:
        with _read_only_transaction() as conn:
            result = conn.execute(text(f"EXPLAIN (FORMAT JSON) {sql}"))
            row = result.fetchone()
            if not row:
                return {"ok": False, "error": "EXPLAIN returned no rows"}
            # psycopg3 parses JSON columns into Python automatically;
            # a `text()` query returns the JSON as a string for older versions,
            # so handle both.
            plan = row[0]
            if isinstance(plan, str):
                plan = json.loads(plan)
            return {"ok": True, "plan": plan}
    except SQLAlchemyError as e:
        cause = e.__cause__ or e
        return {"ok": False, "error": str(cause).strip()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e).strip()}
