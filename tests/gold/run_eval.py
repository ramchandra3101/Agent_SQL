"""Gold-set accuracy harness.

For each question in ``questions.jsonl``:

1. Run the **expected** SQL directly to get the canonical result set.
2. Run the **agent** on the natural-language question and capture its
   result set.
3. Compare as **multisets of tuples** so row order and column order don't
   affect the match (SQL semantics don't guarantee either unless an
   ``ORDER BY`` is requested).

Prints a per-question pass/fail and an overall accuracy. Spec target: ≥80%.

Usage:

    python -m tests.gold.run_eval              # full set
    python -m tests.gold.run_eval --limit 3    # first 3 only
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from src.db.connection import run_query
from src.graph import build_graph
from src.state import initial_state


_QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")


def _load_questions() -> list[dict[str, Any]]:
    with _QUESTIONS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _row_multiset(rows: list[dict[str, Any]]) -> Counter:
    """Convert rows to a multiset of sorted-key tuples for order-free compare."""
    return Counter(
        tuple(sorted((k, _canonical(v)) for k, v in row.items())) for row in rows
    )


def _canonical(v: Any) -> Any:
    """Normalize numeric types so 30 == 30.0 == Decimal('30')."""
    from decimal import Decimal

    if isinstance(v, Decimal):
        # Drop trailing zeros then compare as float where possible.
        return float(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    return v


def _values_multiset(rows: list[dict[str, Any]]) -> Counter:
    """Fallback: compare by value bags when the agent picks different aliases."""
    return Counter(
        tuple(sorted(_canonical(v) for v in row.values() if v is not None))
        for row in rows
    )


def _matches(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
    if len(expected) != len(actual):
        return False
    if _row_multiset(expected) == _row_multiset(actual):
        return True
    # Column aliases often differ between expected and agent output
    # (``n`` vs ``customer_count``). Fall back to a value-bag match.
    return _values_multiset(expected) == _values_multiset(actual)


def _run_agent(graph: Any, question: str) -> list[dict[str, Any]]:
    thread_id = f"gold-{uuid.uuid4().hex[:8]}"
    state = initial_state(question, thread_id=thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    final = graph.invoke(state, config=config)
    return final.get("raw_rows") or []


def main(limit: int = typer.Option(0, help="Only run the first N questions.")) -> None:
    questions = _load_questions()
    if limit > 0:
        questions = questions[:limit]

    graph = build_graph()

    passed = 0
    for i, q in enumerate(questions, 1):
        question = q["question"]
        expected_sql = q["expected_sql"]

        expected_run = run_query(expected_sql, row_limit=10_000)
        if not expected_run["ok"]:
            print(f"[{i:>2}] SKIP (expected SQL failed): {question}")
            print(f"      error: {expected_run['error']}")
            continue
        expected_rows = expected_run["rows"]

        try:
            actual_rows = _run_agent(graph, question)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:>2}] FAIL (agent crashed): {question}")
            print(f"      error: {e}")
            continue

        ok = _matches(expected_rows, actual_rows)
        flag = "PASS" if ok else "FAIL"
        print(f"[{i:>2}] {flag}: {question}")
        if not ok:
            print(f"      expected ({len(expected_rows)}): {expected_rows[:3]}")
            print(f"      actual   ({len(actual_rows)}):   {actual_rows[:3]}")
        passed += int(ok)

    total = len(questions)
    accuracy = passed / total if total else 0.0
    print()
    print(f"Accuracy: {passed}/{total} = {accuracy:.0%}")
    print("Target: >= 80%" + ("  (MET)" if accuracy >= 0.8 else "  (NOT MET)"))


if __name__ == "__main__":
    typer.run(main)
