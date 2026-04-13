"""Schema introspection — live DB metadata for the router, schema selector,
and SQL generator nodes.

Three helpers:

* :func:`load_schema`   — full `{table: [{"name", "type"}, ...]}` map.
* :func:`schema_summary`— compact one-line-per-table text for prompts.
* :func:`table_ddl`     — richer DDL-ish rendering for a subset of tables,
  including foreign-key relationships so the SQL generator knows how to
  join them.

Metadata is cached for the lifetime of the process. Call :func:`reset_cache`
after a migration to force a re-read.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from sqlalchemy import inspect
from sqlalchemy.engine import Inspector

from .connection import get_engine

# Tables we never want to expose to the LLM, even if they exist in the DB.
# The Golden SQL store is agent-internal and should never be queryable via
# the natural-language interface.
_HIDDEN_TABLES = frozenset({"golden_sql"})


class Column(TypedDict):
    name: str
    type: str


class ForeignKey(TypedDict):
    columns: list[str]          # columns on the local table
    ref_table: str              # referenced table
    ref_columns: list[str]      # referenced columns


class Table(TypedDict):
    name: str
    columns: list[Column]
    primary_key: list[str]
    foreign_keys: list[ForeignKey]


def _inspector() -> Inspector:
    return inspect(get_engine())


# ----------------------------------------------------------------------------
#  Full schema load (cached)
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_schema() -> dict[str, Table]:
    """Return the visible schema as ``{table_name: Table}``."""
    insp = _inspector()
    schema: dict[str, Table] = {}

    for name in insp.get_table_names():
        if name in _HIDDEN_TABLES:
            continue

        columns: list[Column] = [
            {"name": c["name"], "type": str(c["type"])}
            for c in insp.get_columns(name)
        ]

        pk = insp.get_pk_constraint(name).get("constrained_columns", []) or []

        fks: list[ForeignKey] = []
        for fk in insp.get_foreign_keys(name):
            ref_table = fk.get("referred_table")
            if not ref_table or ref_table in _HIDDEN_TABLES:
                continue
            fks.append(
                {
                    "columns": list(fk.get("constrained_columns", [])),
                    "ref_table": ref_table,
                    "ref_columns": list(fk.get("referred_columns", [])),
                }
            )

        schema[name] = {
            "name": name,
            "columns": columns,
            "primary_key": list(pk),
            "foreign_keys": fks,
        }

    return schema


def reset_cache() -> None:
    """Clear the cached schema — call after DDL changes."""
    load_schema.cache_clear()


# ----------------------------------------------------------------------------
#  Prompt-facing renderers
# ----------------------------------------------------------------------------
def schema_summary() -> str:
    """Return a compact one-line-per-table summary for the router prompt.

    Format::

        - customers(customer_id INTEGER, name TEXT, email TEXT, ...)
        - orders(order_id INTEGER, customer_id INTEGER, order_date DATE, ...)
    """
    schema = load_schema()
    if not schema:
        return "(no tables found)"

    lines: list[str] = []
    for table in schema.values():
        cols = ", ".join(f"{c['name']} {c['type']}" for c in table["columns"])
        lines.append(f"- {table['name']}({cols})")
    return "\n".join(lines)


def table_ddl(tables: list[str]) -> str:
    """Render a DDL-ish description for the SQL generator prompt.

    Includes column types, primary keys, and foreign-key relationships so
    the LLM has everything it needs to write correct joins without the
    full schema.
    """
    schema = load_schema()
    out: list[str] = []

    for name in tables:
        table = schema.get(name)
        if table is None:
            continue

        out.append(f"TABLE {table['name']}")
        for col in table["columns"]:
            marker = "  PK" if col["name"] in table["primary_key"] else "    "
            out.append(f"  {marker} {col['name']} {col['type']}")

    # Foreign-key section — only edges where both ends are in `tables`.
    fk_lines: list[str] = []
    requested = set(tables)
    for name in tables:
        table = schema.get(name)
        if table is None:
            continue
        for fk in table["foreign_keys"]:
            if fk["ref_table"] not in requested:
                continue
            cols = ", ".join(fk["columns"])
            ref_cols = ", ".join(fk["ref_columns"])
            fk_lines.append(
                f"  {table['name']}({cols}) -> {fk['ref_table']}({ref_cols})"
            )

    if fk_lines:
        out.append("")
        out.append("FOREIGN KEYS")
        out.extend(fk_lines)

    return "\n".join(out)


# ----------------------------------------------------------------------------
#  Manual sanity check: `python -m src.db.introspect`
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SCHEMA SUMMARY")
    print("=" * 60)
    print(schema_summary())

    print()
    print("=" * 60)
    print("TABLE DDL — customers, orders, order_items")
    print("=" * 60)
    print(table_ddl(["customers", "orders", "order_items"]))
