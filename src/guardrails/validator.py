"""Destructive-keyword validator.

The first guardrail in the graph. Rejects any SQL that:

* Is empty / whitespace only.
* Contains more than one statement.
* Does not start with ``SELECT`` or ``WITH``.
* Mentions any keyword in :data:`config.settings.forbidden_keywords` as a
  real SQL keyword token (NOT as part of an identifier or string literal).

The check is done at the **token level** using ``sqlparse``, not regex.
Regex would false-positive on perfectly valid identifiers like
``updated_at`` (contains ``UPDATE``), ``deleted_flag`` (contains ``DELETE``),
or string literals like ``WHERE note = 'please do not delete'``.

Returns a :class:`ValidationResult` so downstream nodes can branch on
``ok`` and surface ``reason`` to the user.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlparse
from sqlparse.tokens import Comment, Keyword, Whitespace

from ..config import settings


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str


_ALLOWED_LEADING_TYPES = {"SELECT"}  # CTEs (`WITH ...`) resolve to SELECT


def _normalize_keyword(token: sqlparse.sql.Token) -> str | None:
    """Return the upper-cased keyword if ``token`` is a SQL keyword token.

    Returns ``None`` for identifiers, literals, comments, whitespace, etc.
    """
    if token.ttype is None:
        return None
    # Comments and whitespace are never destructive even if they contain the
    # word "DELETE".
    if token.ttype in Comment or token.ttype in Whitespace:
        return None
    # Keyword family covers Keyword, Keyword.DML, Keyword.DDL, Keyword.CTE,
    # Keyword.DCL — all of which are real SQL keywords as parsed by sqlparse.
    if token.ttype in Keyword:
        return token.normalized.upper()
    return None


def _leading_statement_type(parsed: sqlparse.sql.Statement) -> str:
    """Return the canonical leading statement type (e.g. ``SELECT``).

    ``sqlparse.Statement.get_type()`` reports ``UNKNOWN`` for CTEs because
    the leading keyword is ``WITH``; we walk the tokens to find the first
    real DML so ``WITH ... SELECT`` reports as ``SELECT`` and
    ``WITH ... DELETE`` reports as ``DELETE``.
    """
    base = parsed.get_type().upper()
    if base != "UNKNOWN":
        return base

    for token in parsed.flatten():
        kw = _normalize_keyword(token)
        if kw is None:
            continue
        if kw == "WITH":
            continue  # keep walking past the CTE marker
        if kw in {"SELECT", "INSERT", "UPDATE", "DELETE", "MERGE"}:
            return kw
    return "UNKNOWN"


def is_safe(sql: str) -> ValidationResult:
    """Return ``ValidationResult(ok=True/False, reason=...)`` for ``sql``."""
    if sql is None or not sql.strip():
        return ValidationResult(False, "empty SQL")

    statements = [s for s in sqlparse.split(sql) if s.strip()]
    if len(statements) != 1:
        return ValidationResult(
            False,
            f"only a single SQL statement is allowed; got {len(statements)}",
        )

    parsed = sqlparse.parse(statements[0])[0]

    # Token-level keyword sweep (catches `WITH x AS (DELETE ...) SELECT ...`).
    forbidden = set(settings.forbidden_keywords)
    for token in parsed.flatten():
        kw = _normalize_keyword(token)
        if kw and kw in forbidden:
            return ValidationResult(
                False, f"destructive keyword detected: {kw}"
            )

    # Leading statement must be a SELECT (or a CTE that resolves to SELECT).
    leading = _leading_statement_type(parsed)
    if leading not in _ALLOWED_LEADING_TYPES:
        return ValidationResult(
            False,
            f"only SELECT statements are allowed; got {leading}",
        )

    return ValidationResult(True, "ok")
