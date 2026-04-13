# SQL Agent

A self-correcting natural-language → SQL agent built on **LangGraph**, **LangChain**, and **PostgreSQL + pgvector**. Ask a question in English, get a SQL query, a result table, and a grounded natural-language answer — with guardrails, human-in-the-loop approval for risky queries, and automatic self-correction on execution errors.

Exposed as a Typer CLI and a FastAPI server with Server-Sent Events (ChatGPT-style token streaming).

---

## Features

- **Schema-aware generation** — LLM picks only the relevant tables before writing SQL.
- **Dialect-aware few-shot** — Golden SQL store in `pgvector`, filtered by dialect *and* table set.
- **Destructive-keyword rejection** — `sqlparse` token check blocks DML/DDL without false-positives on columns like `updated_at`.
- **Cost guardrail** — real `EXPLAIN (FORMAT JSON)` against Postgres; flags seq-scans on large tables, cartesian joins, missing `WHERE`/`LIMIT`, or planner row estimates above `ROW_SCAN_BUDGET`.
- **Read-only enforcement** — three layers: validator → `sqlparse` single-statement check → `SET TRANSACTION READ ONLY` + `statement_timeout` per connection.
- **Self-correction** — on execution failure the LLM emits a *hint* (not a rewritten query) and the generator retries, up to `MAX_RETRIES`.
- **Human-in-the-loop** — graph pauses before risky queries or budget exhaustion; human can approve, edit, hint, or abort.
- **Checkpointed** — `PostgresSaver` persists every node, so HITL pauses survive restarts.
- **Streaming API** — SSE endpoints emit per-node progress events plus token-level streaming of the final NL summary.

---

## Architecture

One `StateGraph` wires every node:

```
START → router ──rag→ rag_fallback → END
                 ──sql→ schema_selector → sql_generator → guardrail_node
guardrail_node ──validator_fail→ formatter        (destructive SQL: error, no retry)
               ──cost_risk→      hitl → executor  (risky: human approves)
               ──ok→             executor
executor ──ok→              formatter → visualizer → END
         ──err & retries_left→ self_correction → sql_generator
         ──err & exhausted→    hitl → executor
hitl ──approve|edit→ executor
     ──hint→         sql_generator
     ──abort→        formatter
```

Budgets are enforced by the graph's edge functions, not by nodes: `MAX_RETRIES=2` self-corrections, `MAX_SQL_GENERATIONS=3` total generations, 1 HITL escalation.

See [`plan.md`](plan.md) for the full design doc and build order, and [`CLAUDE.md`](CLAUDE.md) for a condensed architecture reference.

---

## Tech stack

| Layer            | Choice                                                  |
| ---------------- | ------------------------------------------------------- |
| Orchestration    | LangGraph                                               |
| LLM glue         | LangChain                                               |
| LLM / embeddings | OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)        |
| Database         | PostgreSQL 17 (`SQL_POC`), local install, no Docker     |
| Vector store     | `pgvector` — HNSW index on `vector(1536)`               |
| Persistence      | `PostgresSaver` (LangGraph checkpoints, same Postgres)  |
| API              | FastAPI + `sse-starlette`                               |
| CLI              | Typer + Rich                                            |
| Tests            | pytest                                                  |

---

## Prerequisites

1. **PostgreSQL 17** running locally.
2. Database `SQL_POC` created (case-sensitive — keep the underscore).
3. `pgvector` extension installed:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
4. **Python 3.11+**.
5. An **OpenAI API key**.

Verify Postgres + pgvector:

```bash
psql "postgresql://postgres@localhost:5432/SQL_POC" \
  -c "SELECT extname FROM pg_extension WHERE extname='vector';"
# → vector
```

---

## Setup

```bash
# 1. Clone and enter the project
cd sql-agent

# 2. Create & activate the virtualenv (project uses `pocenv/`, not `venv/`)
python -m venv pocenv
source pocenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# → edit .env and set OPENAI_API_KEY + DATABASE_URL password
```

### `.env` keys

| Variable                 | Purpose                                             |
| ------------------------ | --------------------------------------------------- |
| `OPENAI_API_KEY`         | Required.                                           |
| `OPENAI_MODEL`           | Default `gpt-4o-mini`.                              |
| `OPENAI_EMBEDDING_MODEL` | Default `text-embedding-3-small`.                   |
| `DATABASE_URL`           | `postgresql+psycopg://postgres:<pw>@localhost:5432/SQL_POC` |
| `LANGSMITH_TRACING`      | `true` / `false`.                                   |
| `LANGSMITH_API_KEY`      | Required when tracing is on.                        |
| `MAX_RETRIES`            | Self-correction loops after a failure. Default 2.   |
| `MAX_SQL_GENERATIONS`    | Total generator invocations per question. Default 3.|
| `ROW_SCAN_BUDGET`        | Planner-estimated rows above which HITL pauses. 100000. |
| `QUERY_TIMEOUT_SECONDS`  | `statement_timeout` per query. Default 15.          |

---

## Seed the database

Two seeding steps, both idempotent:

```bash
# 1. Create schema + insert the demo e-commerce data
python -m src.db.sample_data

# 2. Load the Golden SQL few-shot examples into pgvector
python -m src.main seed            # add --force to wipe and re-seed
```

After step 1, `\dt` in `psql SQL_POC` shows `customers`, `products`, `orders`, `order_items`, `golden_sql`.

---

## Usage

### CLI

```bash
python -m src.main ask "top 5 customers by total spend"
python -m src.main ask "how many orders per month last year?"

# Resume a specific thread (e.g. after a prior HITL session)
python -m src.main ask "..." --thread cli-1712345678-abc123
```

On risky queries the CLI prompts: `Action [approve/edit/hint/abort]`.

### HTTP API

```bash
python -m src.api.server            # http://localhost:8000
```

OpenAPI docs:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Raw schema: http://localhost:8000/openapi.json

Endpoints:

| Method | Path       | Purpose                                           |
| ------ | ---------- | ------------------------------------------------- |
| POST   | `/ask`     | Start a new run, stream SSE.                      |
| POST   | `/resume`  | Feed `human_feedback` into a HITL-interrupted thread, stream SSE. |
| GET    | `/health`  | Liveness probe.                                   |

Quick test with `curl`:

```bash
curl -N -H "Content-Type: application/json" \
  -d '{"question": "top 5 customers by spend"}' \
  http://localhost:8000/ask
```

#### SSE event types

Every frame is `event: <type>\ndata: <json>`. The payload schemas for every type are documented as Pydantic models in `src/api/events.py` and published in `/openapi.json`.

| Event             | Emitted when                                              |
| ----------------- | --------------------------------------------------------- |
| `thread`          | First frame on `/ask`; carries the thread id.             |
| `route`           | Router chose `sql` or `rag`.                              |
| `schema`          | Schema selector returned its table list.                  |
| `sql`             | Generator produced a SQL candidate.                       |
| `guardrail`       | Validator + cost verdict.                                 |
| `rows`            | Executor returned a result set.                           |
| `exec_error`      | Executor raised (may trigger a retry).                    |
| `retry`           | Self-correction emitted a hint.                           |
| `rag`             | RAG fallback answered an off-schema question.             |
| `answer_delta`    | Token chunk of the NL summary (streamed).                 |
| `needs_approval`  | Graph interrupted before `hitl`; stream closes next.      |
| `done`            | Terminal frame on a successful run.                       |
| `error`           | Unexpected server-side failure.                           |

#### HITL flow over HTTP

1. Client POSTs `/ask`.
2. If the query is risky, the stream closes after `needs_approval` with the pending SQL + flags.
3. Client collects the user's decision and POSTs `/resume`:
   ```json
   { "thread_id": "api-...", "action": "approve" }
   { "thread_id": "api-...", "action": "edit", "edited_sql": "SELECT ..." }
   { "thread_id": "api-...", "action": "hint", "hint": "join on customer_id" }
   { "thread_id": "api-...", "action": "abort", "reason": "too expensive" }
   ```
4. A fresh SSE stream continues the same thread.

---

## Testing

```bash
pytest                                  # full suite
pytest tests/test_validator.py          # destructive-keyword rejection
pytest tests/test_cost.py               # cost guardrail
pytest tests/test_graph_smoke.py        # end-to-end smoke
pytest tests/test_validator.py::test_rejects_update_keyword   # one test
```

---

## Project layout

```
sql-agent/
├── plan.md                  ← full design + 23-step build order
├── CLAUDE.md                ← condensed architecture reference
├── README.md                ← you are here
├── requirements.txt
├── .env.example
├── data/
│   └── golden_sql_seed.json ← few-shot examples
├── src/
│   ├── config.py            ← env-backed Settings
│   ├── state.py             ← AgentState TypedDict
│   ├── llm.py               ← OpenAI chat + embeddings factory
│   ├── graph.py             ← LangGraph assembly + PostgresSaver
│   ├── main.py              ← Typer CLI
│   ├── db/
│   │   ├── bootstrap.sql    ← CREATE EXTENSION + DDL
│   │   ├── sample_data.py   ← deterministic seeder
│   │   ├── connection.py    ← read-only engine + run_query
│   │   └── introspect.py    ← schema summary / table DDL
│   ├── guardrails/
│   │   ├── validator.py     ← destructive-keyword rejection
│   │   └── cost.py          ← EXPLAIN-based cost heuristics
│   ├── retrieval/
│   │   └── golden_sql.py    ← few-shot pgvector store
│   ├── nodes/
│   │   ├── router.py
│   │   ├── schema_selector.py
│   │   ├── sql_generator.py
│   │   ├── guardrail_node.py
│   │   ├── hitl.py
│   │   ├── executor.py
│   │   ├── self_correction.py
│   │   ├── formatter.py     ← NL summary (streamed)
│   │   ├── visualizer.py
│   │   └── rag_fallback.py
│   └── api/
│       ├── server.py        ← FastAPI + SSE
│       └── events.py        ← SSE payload schemas (OpenAPI)
└── tests/
```

---

## Security notes

The POC connects as the `postgres` superuser for simplicity. Read-only behavior is enforced in code via three layers (validator → `sqlparse` → `SET TRANSACTION READ ONLY`), but **before going to production**, create a least-privilege role (e.g. `sqlagent` with only `SELECT` on business tables) and switch `DATABASE_URL` to it. That adds defence-in-depth at the DB layer on top of the code-level checks.

CORS on the API is wide open (`allow_origins=["*"]`). Tighten it before exposing the server outside `localhost`.

`.env` is gitignored — never commit it.

---

## License

MIT. See [`LICENSE`](LICENSE).
