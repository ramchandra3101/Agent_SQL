"""Golden SQL retrieval store (pgvector-backed).

The SQL generator (Step 13) does *dynamic* few-shot prompting: instead of a
hard-coded list of examples, it asks this module for the ``k`` examples most
semantically similar to the user's current question. This usually doubles
generation accuracy and lets the example set grow as new questions are
validated by humans (HITL "approve" → upsert here).

Layout choices for the POC:

* The store lives in the **same** Postgres database as the business tables
  (``SQL_POC``) — one connection, one set of credentials, one place to back
  up. The :func:`src.db.introspect.load_schema` function deliberately hides
  the ``golden_sql`` table from the agent so the LLM never sees it.
* Embeddings use ``text-embedding-3-small`` (1536 dims). The vector is sent
  to Postgres as the literal string ``'[0.12, -0.07, ...]'`` cast with
  ``::vector`` so we don't need to register the pgvector psycopg adapter.
* Filtering is dialect- *and* schema-aware: a Postgres question that touches
  ``customers``/``orders`` should not be primed with a SQLite example over
  ``invoices``. The SQL filter is ``dialect = :d`` plus an *optional*
  ``tables && :t`` overlap check (only applied if the caller passes tables).
* Distance metric is cosine (``<=>``) — matches how OpenAI embeddings are
  normalised and is what HNSW will use once we upgrade pgvector.

Public surface:

* :func:`upsert`  — embed and insert one example. Used by HITL approve.
* :func:`retrieve` — top-``k`` semantic neighbours for a question.
* :func:`seed`    — bulk-load ``data/golden_sql_seed.json`` (idempotent).
* :func:`count`   — row count, for diagnostics / tests.
* :func:`clear`   — wipe the table (tests only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import text

from ..db.connection import get_engine
from ..llm import get_embeddings


_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_sql_seed.json"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _vector_literal(vec: Sequence[float]) -> str:
    """Format a Python float list as a pgvector text literal: ``'[1,2,3]'``."""
    # repr() on floats keeps full precision; join with commas (no spaces, no
    # scientific-notation surprises that pgvector would reject).
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _embed(text_in: str) -> list[float]:
    """Return a 1536-dim embedding for ``text_in``."""
    return get_embeddings().embed_query(text_in)


# ---------------------------------------------------------------------------
#  Write path
# ---------------------------------------------------------------------------
def upsert(
    question: str,
    sql: str,
    *,
    dialect: str = "postgresql",
    tables: Sequence[str] = (),
) -> int:
    """Insert one (question, sql) example. Returns the new row id.

    "Upsert" is aspirational — the table has no natural unique key, so we
    insert. If you want true dedup later, add a UNIQUE on ``md5(question)``.
    """
    embedding = _embed(question)
    sql_insert = text(
        """
        INSERT INTO golden_sql (question, sql, dialect, tables, embedding)
        VALUES (:q, :s, :d, :t, CAST(:e AS vector))
        RETURNING id
        """
    )
    with get_engine().begin() as conn:
        row = conn.execute(
            sql_insert,
            {
                "q": question,
                "s": sql,
                "d": dialect,
                "t": list(tables),
                "e": _vector_literal(embedding),
            },
        ).fetchone()
    return int(row[0])


# ---------------------------------------------------------------------------
#  Read path
# ---------------------------------------------------------------------------
def retrieve(
    question: str,
    *,
    dialect: str = "postgresql",
    tables: Sequence[str] | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return top-``k`` semantically similar examples.

    Each row: ``{"id", "question", "sql", "tables", "distance"}`` —
    ``distance`` is cosine distance (0 = identical, 2 = opposite).
    """
    embedding = _embed(question)

    # Build the WHERE clause incrementally so the schema-overlap filter is
    # only applied when the caller passes tables.
    where = ["dialect = :d"]
    params: dict[str, Any] = {
        "d": dialect,
        "e": _vector_literal(embedding),
        "k": k,
    }
    if tables:
        where.append("tables && :t")
        params["t"] = list(tables)

    sql_select = text(
        f"""
        SELECT
            id,
            question,
            sql,
            tables,
            embedding <=> CAST(:e AS vector) AS distance
        FROM golden_sql
        WHERE {' AND '.join(where)}
        ORDER BY embedding <=> CAST(:e AS vector)
        LIMIT :k
        """
    )

    with get_engine().connect() as conn:
        result = conn.execute(sql_select, params)
        return [
            {
                "id": r.id,
                "question": r.question,
                "sql": r.sql,
                "tables": list(r.tables) if r.tables is not None else [],
                "distance": float(r.distance),
            }
            for r in result
        ]


# ---------------------------------------------------------------------------
#  Bulk seed
# ---------------------------------------------------------------------------
def seed(*, force: bool = False) -> int:
    """Load examples from ``data/golden_sql_seed.json``.

    Idempotent by default: if the table already has rows, returns 0 without
    re-embedding (each embed call is a paid OpenAI request). Pass
    ``force=True`` to wipe and re-seed.
    """
    if force:
        clear()
    elif count() > 0:
        return 0

    with _SEED_PATH.open() as f:
        examples = json.load(f)

    inserted = 0
    for ex in examples:
        upsert(
            ex["question"],
            ex["sql"],
            dialect=ex.get("dialect", "postgresql"),
            tables=ex.get("tables", []),
        )
        inserted += 1
    return inserted


# ---------------------------------------------------------------------------
#  Diagnostics
# ---------------------------------------------------------------------------
def count() -> int:
    with get_engine().connect() as conn:
        return int(conn.execute(text("SELECT COUNT(*) FROM golden_sql")).scalar() or 0)


def clear() -> None:
    with get_engine().begin() as conn:
        conn.execute(text("TRUNCATE TABLE golden_sql RESTART IDENTITY"))


# ---------------------------------------------------------------------------
#  CLI sanity check:  python -m src.retrieval.golden_sql
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"golden_sql rows before seed: {count()}")
    n = seed()
    print(f"seeded {n} examples (0 = already present)")
    print(f"golden_sql rows after seed:  {count()}")

    probe = "What's the revenue broken down by country?"
    print(f"\nProbe question: {probe!r}")
    for hit in retrieve(probe, k=3):
        print(f"  d={hit['distance']:.4f}  {hit['question']}")
