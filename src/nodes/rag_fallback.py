"""RAG fallback node — handles questions the schema cannot answer.

When the router picks ``route="rag"`` (off-schema questions: policy, docs,
greetings, general knowledge), the graph lands here instead of the SQL
pipeline. For the POC this is a **stub**:

* It acknowledges the question.
* It explains the schema can't answer it.
* It emits a short LLM-written answer grounded only in the model's own
  knowledge (no document retrieval yet).

A production version would:

1. Load a document store (PDFs, wiki, policy docs) into pgvector.
2. Retrieve top-k passages for the question.
3. Stuff them into the prompt as context before the LLM answers.

Keeping it a stub unblocks end-to-end graph testing without adding a
whole document-ingestion pipeline to the POC.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from ..state import AgentState


_SYSTEM_PROMPT = """You are a helpful assistant for a data analytics tool.

The user asked a question that CANNOT be answered from the SQL database
this tool has access to. Do your best to give a short, helpful answer
from general knowledge. Keep it to 2-3 sentences. If you don't know,
say so plainly. No markdown, no code fences.
"""


def rag_fallback_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: answer off-schema questions with an LLM stub."""
    question = state.get("user_query", "")
    route_reason = state.get("route_reason", "")

    llm = get_llm(temperature=0.2, max_tokens=250)
    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
    )
    answer = response.content.strip()

    preamble = (
        "This question isn't answerable from the current database "
        f"({route_reason}). Answering from general knowledge instead:"
        if route_reason
        else "Answering from general knowledge:"
    )
    full_answer = f"{preamble}\n\n{answer}"

    return {
        "final_result": {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "sql": "",
            "summary": full_answer,
            "confidence_score": 0.3,  # low — no grounding in DB or docs
        },
        "natural_language_answer": full_answer,
        "confidence_score": 0.3,
    }


# ---------------------------------------------------------------------------
#  Sanity check:  python -m src.nodes.rag_fallback
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ..state import initial_state

    for q in [
        "What is the capital of France?",
        "What can this tool do?",
    ]:
        state = initial_state(q, thread_id="probe")
        state["route_reason"] = "question is not about the schema"
        out = rag_fallback_node(state)
        print(f"Q: {q}")
        print(out["natural_language_answer"])
        print()
