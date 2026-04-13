"""FastAPI server — SSE streaming around the LangGraph SQL agent.

OpenAPI docs live at ``/docs`` (Swagger UI), ``/redoc``, and the raw
schema at ``/openapi.json``. Because SSE doesn't map cleanly onto the
OpenAPI response model, the event payload schemas are documented
separately in ``src/api/events.py`` and referenced from the endpoint
docstrings below. The frontend can codegen TypeScript types from
``/openapi.json``.

Endpoints
---------
* ``POST /ask``    — start a new run; streams node-level events and a
  token-stream of the NL summary.
* ``POST /resume`` — feed ``human_feedback`` into an interrupted thread
  and keep streaming.
* ``GET  /health`` — cheap liveness probe.

Streaming implementation notes
------------------------------
* We pass ``stream_mode=["updates", "messages"]`` to ``graph.stream``.
  ``updates`` yields one dict per node with its state delta; ``messages``
  yields ``(chunk, metadata)`` pairs for every LLM token inside a node.
  We filter messages to the ``formatter`` node to drive ``answer_delta``.
* ``graph.stream`` is sync; we iterate it inside an async generator.
  That blocks the event loop per chunk — acceptable for a POC.
* CORS is wide open — tighten before anything other than local dev.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..graph import build_graph
from ..state import initial_state
from .events import (
    AnswerDeltaEvent,
    DoneEvent,
    ErrorEvent,
    ExecErrorEvent,
    GuardrailEvent,
    NeedsApprovalEvent,
    RagEvent,
    RetryEvent,
    RouteEvent,
    RowsEvent,
    SchemaEvent,
    SqlEvent,
    ThreadEvent,
)


# ---------------------------------------------------------------------------
#  App metadata  (drives /docs + /openapi.json)
# ---------------------------------------------------------------------------
API_DESCRIPTION = """
Self-correcting natural-language → SQL agent, exposed over HTTP with
Server-Sent Events for real-time progress.

### How a request flows

1. Client `POST /ask` with a question.
2. Server streams one SSE event per graph node (`route`, `schema`,
   `sql`, `guardrail`, `rows`, …) and token-streams the final NL
   summary as `answer_delta` events.
3. If the graph needs human approval, the stream closes after a
   `needs_approval` event. Client collects user input and calls
   `POST /resume` with the same `thread_id` to continue.
4. On success, the stream ends with a `done` event.

### Event schema

Every SSE frame is `event: <type>\\ndata: <json>`. Payload shapes live
in the `events` module and are surfaced in this spec under the
`SseEventPayloads` tag.

### Auth

None. Intended for local development against a local Postgres.
""".strip()


tags_metadata = [
    {"name": "agent", "description": "Run the SQL agent."},
    {"name": "health", "description": "Liveness probes."},
    {
        "name": "SseEventPayloads",
        "description": (
            "Payload schemas for each SSE `data:` frame. These endpoints "
            "never return JSON bodies directly; the models are published "
            "here so clients can codegen strongly-typed event handlers."
        ),
    },
]


app = FastAPI(
    title="SQL Agent API",
    version="0.1.0",
    description=API_DESCRIPTION,
    openapi_tags=tags_metadata,
    contact={"name": "SQL Agent POC"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
#  Request / response models
# ---------------------------------------------------------------------------
class AskBody(BaseModel):
    question: str = Field(
        ...,
        description="Natural-language question for the agent.",
        examples=["Top 5 customers by total spend"],
    )
    thread_id: str | None = Field(
        None,
        description=(
            "Optional conversation id. Omit to start a new thread; the "
            "server returns the generated id in the first `thread` SSE "
            "frame."
        ),
    )


class ResumeBody(BaseModel):
    thread_id: str = Field(
        ...,
        description="Thread id from the `needs_approval` event.",
    )
    action: str = Field(
        ...,
        description=(
            "Human decision: `approve` (run as-is), `edit` (run "
            "`edited_sql` instead), `hint` (regenerate with `hint`), or "
            "`abort` (stop with `reason`)."
        ),
        examples=["approve"],
    )
    edited_sql: str | None = Field(None, description="Required when action=`edit`.")
    hint: str | None = Field(None, description="Required when action=`hint`.")
    reason: str | None = Field(None, description="Optional abort reason.")


class HealthResponse(BaseModel):
    status: str = "ok"


# Schemas referenced purely so FastAPI registers them in components.
# They don't appear as responses on any real endpoint.
_SSE_EVENT_TYPES: list[type[BaseModel]] = [
    ThreadEvent,
    RouteEvent,
    SchemaEvent,
    SqlEvent,
    GuardrailEvent,
    RowsEvent,
    ExecErrorEvent,
    RetryEvent,
    RagEvent,
    AnswerDeltaEvent,
    NeedsApprovalEvent,
    DoneEvent,
    ErrorEvent,
]


# ---------------------------------------------------------------------------
#  SSE helpers
# ---------------------------------------------------------------------------
def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, default=str)}


def _node_event(node: str, update: dict[str, Any]) -> dict[str, str] | None:
    """Map a LangGraph node update to an SSE event (or None to skip)."""
    if node == "router":
        return _sse("route", {
            "route": update.get("route"),
            "reason": update.get("route_reason", ""),
        })
    if node == "schema_selector":
        tables = [t.get("name") for t in update.get("selected_schema", [])]
        return _sse("schema", {"selected_tables": tables})
    if node == "sql_generator":
        return _sse("sql", {
            "sql": update.get("generated_sql", ""),
            "generation_count": update.get("generation_count", 0),
        })
    if node == "guardrail_node":
        return _sse("guardrail", {
            "validator_ok": update.get("validator_ok"),
            "cost_ok": update.get("cost_ok"),
            "risk_flags": update.get("risk_flags", []),
            "reason": (
                update.get("cost_reason")
                or update.get("validator_reason")
                or ""
            ),
        })
    if node == "executor":
        if update.get("execution_error"):
            return _sse("exec_error", {"error": update["execution_error"]})
        return _sse("rows", {
            "columns": update.get("columns", []),
            "rows": update.get("raw_rows", []),
            "row_count": update.get("row_count", 0),
        })
    if node == "self_correction":
        return _sse("retry", {
            "hint": update.get("correction_hint", ""),
            "retry_count": update.get("retry_count", 0),
        })
    if node == "rag_fallback":
        return _sse("rag", {
            "answer": update.get("natural_language_answer", ""),
        })
    return None


def _stream_graph(graph: Any, input_state: Any, config: dict) -> Iterator[dict]:
    """Drive ``graph.stream`` and yield SSE event dicts."""
    try:
        for mode, chunk in graph.stream(
            input_state,
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "updates":
                for node, update in chunk.items():
                    evt = _node_event(node, update or {})
                    if evt:
                        yield evt
            elif mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") == "formatter":
                    text = getattr(msg, "content", "") or ""
                    if text:
                        yield _sse("answer_delta", {"text": text})

        snapshot = graph.get_state(config)
        if snapshot.next and "hitl" in snapshot.next:
            vals = snapshot.values
            yield _sse("needs_approval", {
                "thread_id": config["configurable"]["thread_id"],
                "sql": vals.get("generated_sql", ""),
                "risk_flags": vals.get("risk_flags", []),
                "cost_reason": vals.get("cost_reason", ""),
                "estimated_rows": vals.get("estimated_rows", 0),
                "execution_error": vals.get("execution_error"),
            })
            return

        final = snapshot.values
        yield _sse("done", {
            "confidence": final.get("confidence_score", 0.0),
            "visualization_code": final.get("visualization_code"),
            "row_count": final.get("row_count", 0),
        })
    except Exception as exc:
        yield _sse("error", {"message": str(exc)})


# ---------------------------------------------------------------------------
#  Shared OpenAPI response descriptor for the two SSE endpoints
# ---------------------------------------------------------------------------
_SSE_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": (
            "Server-Sent Events stream. Each frame is "
            "`event: <type>\\ndata: <json>`. See the `SseEventPayloads` "
            "schemas for per-type `data` shapes."
        ),
        "content": {
            "text/event-stream": {
                "schema": {
                    "type": "string",
                    "format": "event-stream",
                    "example": (
                        "event: thread\n"
                        'data: {"thread_id": "api-1712345678-abc123"}\n\n'
                        "event: route\n"
                        'data: {"route": "sql", "reason": "schema-grounded"}\n\n'
                        "event: sql\n"
                        'data: {"sql": "SELECT ...", "generation_count": 1}\n\n'
                        "event: answer_delta\n"
                        'data: {"text": "The top "}\n\n'
                        "event: done\n"
                        'data: {"confidence": 0.95, "row_count": 5}\n\n'
                    ),
                },
            }
        },
    }
}


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    tags=["health"],
    summary="Liveness probe",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/ask",
    tags=["agent"],
    summary="Ask a question (SSE stream)",
    description=(
        "Starts a new agent run and streams progress as Server-Sent "
        "Events. Event sequence on a clean run:\n\n"
        "`thread` → `route` → `schema` → `sql` → `guardrail` → "
        "`rows` → `answer_delta`* → `done`.\n\n"
        "On HITL the stream ends at `needs_approval` — resume via "
        "`POST /resume`."
    ),
    responses=_SSE_RESPONSES,
    response_class=EventSourceResponse,
)
async def ask(body: AskBody) -> EventSourceResponse:
    thread_id = body.thread_id or f"api-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    graph = build_graph()
    state = initial_state(body.question, thread_id=thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    async def gen():
        yield _sse("thread", {"thread_id": thread_id})
        for event in _stream_graph(graph, state, config):
            yield event

    return EventSourceResponse(gen())


@app.post(
    "/resume",
    tags=["agent"],
    summary="Resume a HITL-interrupted run (SSE stream)",
    description=(
        "Feeds `human_feedback` into an interrupted thread and "
        "continues streaming. Valid `action` values: `approve`, "
        "`edit` (needs `edited_sql`), `hint` (needs `hint`), `abort` "
        "(optional `reason`)."
    ),
    responses=_SSE_RESPONSES,
    response_class=EventSourceResponse,
)
async def resume(body: ResumeBody) -> EventSourceResponse:
    graph = build_graph()
    config = {"configurable": {"thread_id": body.thread_id}}

    feedback: dict[str, Any] = {"action": body.action}
    if body.edited_sql is not None:
        feedback["edited_sql"] = body.edited_sql
    if body.hint is not None:
        feedback["hint"] = body.hint
    if body.reason is not None:
        feedback["reason"] = body.reason

    async def gen():
        graph.update_state(config, {"human_feedback": feedback})
        for event in _stream_graph(graph, None, config):
            yield event

    return EventSourceResponse(gen())


# ---------------------------------------------------------------------------
#  Pin the SSE payload models into the OpenAPI components section.
#  Without a reference somewhere in the spec, FastAPI wouldn't emit them —
#  we override openapi() to splat them into `components.schemas`.
# ---------------------------------------------------------------------------
def _custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=tags_metadata,
    )
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    for model in _SSE_EVENT_TYPES:
        components[model.__name__] = model.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi  # type: ignore[assignment]


# ---------------------------------------------------------------------------
#  Local run:  python -m src.api.server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
