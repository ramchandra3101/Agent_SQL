"""CLI entrypoint — ``python -m src.main ask "..."`` and ``seed``.

Two commands:

* ``ask "<question>"`` — runs the graph. If the graph pauses for HITL, the
  CLI shows the SQL + risk flags, asks for an action (approve / edit /
  hint / abort), writes ``human_feedback`` onto state, and resumes.
* ``seed`` — loads the Golden SQL few-shot examples into pgvector
  (idempotent unless ``--force``).

Design:

* The ``PostgresSaver`` checkpointer persists every node, so a long HITL
  pause (user goes to lunch) doesn't lose state. The CLI uses a fresh
  ``thread_id`` per run (timestamp-based) by default; pass ``--thread``
  to resume an existing conversation.
* Rendering uses Rich for the table + a colored summary panel.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .graph import build_graph
from .retrieval.golden_sql import count as golden_count
from .retrieval.golden_sql import seed as golden_seed
from .state import initial_state


app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _render_table(columns: list[str], rows: list[dict[str, Any]]) -> Table:
    table = Table(show_header=True, header_style="bold cyan")
    for col in columns:
        table.add_column(col)
    for row in rows[:50]:
        table.add_row(*[str(row.get(c, "")) for c in columns])
    return table


def _show_hitl_prompt(state: dict[str, Any]) -> dict[str, Any]:
    """Show the pending SQL + risk flags and collect a human action."""
    console.rule("[bold yellow]Human approval required")
    console.print(f"[bold]SQL:[/bold]\n{state.get('generated_sql', '')}\n")
    if state.get("cost_reason"):
        console.print(f"[yellow]Reason:[/yellow] {state['cost_reason']}")
    if state.get("risk_flags"):
        console.print(f"[yellow]Flags:[/yellow]  {state['risk_flags']}")
    if state.get("estimated_rows"):
        console.print(f"[yellow]Est. rows:[/yellow] {state['estimated_rows']:,}")
    if state.get("execution_error"):
        console.print(f"[red]Last error:[/red] {state['execution_error']}")

    console.print()
    action = typer.prompt(
        "Action [approve/edit/hint/abort]", default="approve"
    ).strip().lower()

    if action == "edit":
        edited = typer.prompt("Edited SQL").strip()
        return {"action": "edit", "edited_sql": edited}
    if action == "hint":
        hint = typer.prompt("Hint for the generator").strip()
        return {"action": "hint", "hint": hint}
    if action == "abort":
        reason = typer.prompt("Reason", default="aborted by human").strip()
        return {"action": "abort", "reason": reason}
    return {"action": "approve"}


def _run_with_hitl(graph: Any, state: dict[str, Any], thread_id: str) -> dict[str, Any]:
    """Invoke the graph, resuming through any HITL interrupts."""
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(state, config=config)
    # After an interrupt_before=["hitl"], the graph returns WITHOUT running
    # the hitl node. graph.get_state(config).next tells us what would run.
    while True:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            return result
        if "hitl" not in snapshot.next:
            return result

        feedback = _show_hitl_prompt(snapshot.values)
        graph.update_state(config, {"human_feedback": feedback})
        result = graph.invoke(None, config=config)


# ---------------------------------------------------------------------------
#  Commands
# ---------------------------------------------------------------------------
@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question."),
    thread: str | None = typer.Option(None, help="Reuse a specific thread id."),
) -> None:
    """Ask the agent a question."""
    thread_id = thread or f"cli-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    console.print(f"[dim]thread: {thread_id}[/dim]")

    graph = build_graph()
    state = initial_state(question, thread_id=thread_id)
    final = _run_with_hitl(graph, state, thread_id)

    # ---- Render ---------------------------------------------------------
    columns = final.get("columns") or []
    rows = final.get("raw_rows") or []
    if columns and rows:
        console.print(_render_table(columns, rows))
    elif final.get("final_result", {}).get("error"):
        console.print(f"[red]Error:[/red] {final['final_result']['error']}")

    summary = final.get("natural_language_answer", "")
    if summary:
        console.print(Panel(summary, title="Answer", border_style="green"))

    sql = final.get("generated_sql")
    if sql:
        console.print(Panel(sql, title="SQL", border_style="dim"))

    if final.get("visualization_code"):
        console.print(
            Panel(
                final["visualization_code"],
                title="Suggested chart code",
                border_style="dim",
            )
        )

    conf = final.get("confidence_score", 0.0)
    console.print(f"[dim]confidence: {conf:.2f}   "
                  f"retries: {final.get('retry_count', 0)}   "
                  f"generations: {final.get('generation_count', 0)}[/dim]")


@app.command()
def seed(
    force: bool = typer.Option(False, "--force", help="Wipe and re-seed."),
) -> None:
    """Load Golden SQL few-shot examples into pgvector."""
    before = golden_count()
    n = golden_seed(force=force)
    after = golden_count()
    console.print(f"seeded {n} examples  (before={before}, after={after})")


if __name__ == "__main__":
    app()
