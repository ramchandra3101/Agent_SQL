"""Human-in-the-loop checkpoint node.

This node is deliberately simple — it's a **pause point**, not a decision
maker. The real work is done by LangGraph's ``interrupt_before=["hitl"]``:
when the graph arrives here it freezes its state via the checkpointer and
returns control to the caller (CLI, API, UI, etc.).

The caller shows the human:

  * the generated SQL
  * the guardrail flags (``cost_reason``, ``risk_flags``, ``estimated_rows``)
  * the validator/escalation reason if applicable

Then writes one of four actions to ``state["human_feedback"]`` before
resuming the graph:

  * ``approve`` — run the SQL as-is (cost_ok is flipped to True so the
    executor edge takes over).
  * ``edit``   — the human supplied an edited SQL; replace ``generated_sql``
    and clear the paused flag so the executor runs the human's version.
  * ``hint``   — the human couldn't fix it but left guidance; the
    self-correction path picks up ``correction_hint`` on the next loop.
  * ``abort``  — stop the run with a user-facing message.

This file implements only the **post-resume** half: reading the feedback
and projecting it back into state so the graph can route correctly. The
pause itself is configured in ``src/graph.py`` (Step 20).
"""

from __future__ import annotations

from typing import Any

from ..state import AgentState


def hitl_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: consume ``human_feedback`` and update state.

    If no feedback is present (first pass through, right before the
    interrupt), pass through unchanged so the checkpointer can pause.
    """
    feedback = state.get("human_feedback")
    if not feedback:
        # Pre-interrupt pass: nothing to do. The graph will pause here
        # thanks to interrupt_before=["hitl"].
        return {"paused": True}

    action = feedback.get("action")

    if action == "approve":
        # Human accepted the risk. Unblock the executor path.
        return {
            "paused": False,
            "cost_ok": True,
            "cost_reason": "approved by human",
            "escalated": False,
            "human_feedback": None,
        }

    if action == "edit":
        edited = feedback.get("edited_sql", "").strip()
        return {
            "paused": False,
            "cost_ok": True,
            "cost_reason": "edited by human",
            "generated_sql": edited,
            # Force re-validation + cost check on the edited SQL.
            "validator_ok": False,
            "escalated": False,
            "human_feedback": None,
        }

    if action == "hint":
        return {
            "paused": False,
            "correction_hint": feedback.get("hint", "").strip() or None,
            "escalated": False,
            "human_feedback": None,
        }

    if action == "abort":
        return {
            "paused": False,
            "escalated": True,
            "final_result": {
                "error": feedback.get("reason") or "aborted by human",
            },
            "human_feedback": None,
        }

    # Unknown action — stay paused rather than silently proceeding.
    return {"paused": True}


# ---------------------------------------------------------------------------
#  Sanity check:  python -m src.nodes.hitl
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ..state import initial_state

    base = initial_state("probe", thread_id="probe")
    base["generated_sql"] = "SELECT * FROM customers, orders"
    base["paused"] = True
    base["cost_ok"] = False
    base["cost_reason"] = "cartesian join"

    cases: list[dict[str, Any]] = [
        {"action": "approve"},
        {"action": "edit", "edited_sql": "SELECT * FROM customers LIMIT 10"},
        {"action": "hint", "hint": "add a join condition on customer_id"},
        {"action": "abort", "reason": "too expensive"},
        {},  # no feedback yet — should stay paused
    ]

    for fb in cases:
        state = dict(base)
        state["human_feedback"] = fb or None
        out = hitl_node(state)  # type: ignore[arg-type]
        print(f"feedback={fb}")
        print(f"  → {out}")
        print()
