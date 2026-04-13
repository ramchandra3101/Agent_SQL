# Self-Correcting SQL Data Agent — Build Plan

A step-by-step build plan. Each step is small, runnable, and independently
verifiable. We don't move to the next step until the current one works.

---

## What we are building

A natural-language → SQL agent that:

1. Understands a user question.
2. Picks only the relevant tables from the schema.
3. Generates dialect-aware SQL (SQLite / PostgreSQL).
4. Checks the SQL for destructive keywords and high cost.
5. Executes through a read-only wrapper.
6. On failure, reflects on the error and rewrites (max 2 retries).
7. On risky queries, pauses for human approval.
8. Returns a table + a natural-language explanation (and a chart when useful).

Everything runs inside one LangGraph `StateGraph` with SQLite checkpointing.

---

## Tech stack (fixed)

| Layer | Choice |
|---|---|
| Orchestration | LangGraph |
| LLM glue | LangChain |
| Tracing | LangSmith |
| LLM | OpenAI (default, e.g. `gpt-4o-mini` / `gpt-4o`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| DB | **PostgreSQL 17** (database name: `SQL_POC`) — local install, no Docker |
| Persistence | `PostgresSaver` checkpoint store (same Postgres instance) |
| Vector store | **pgvector** — `golden_sql` table with `vector(1536)` + HNSW index (same Postgres instance) |
| CLI | Typer + Rich |
| Tests | pytest |

---


## File layout (target)

```
sql-agent/
├── plan.md                  ← this file
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py            ← env-backed Settings
│   ├── state.py             ← AgentState TypedDict
│   ├── llm.py               ← LLM factory (OpenAI chat + embeddings)
│   ├── db/
│   │   ├── bootstrap.sql    ← CREATE EXTENSION vector; schema DDL
│   │   ├── sample_data.py   ← seed SQL_POC with demo data
│   │   ├── connection.py    ← read-only engine + run_query
│   │   └── introspect.py    ← schema summary / table DDL
│   ├── guardrails/
│   │   ├── validator.py     ← destructive-keyword rejection
│   │   └── cost.py          ← cost/risk heuristics
│   ├── retrieval/
│   │   └── golden_sql.py    ← few-shot vector store
│   ├── nodes/
│   │   ├── router.py
│   │   ├── schema_selector.py
│   │   ├── sql_generator.py
│   │   ├── guardrail_node.py
│   │   ├── hitl.py
│   │   ├── executor.py
│   │   ├── self_correction.py
│   │   ├── formatter.py
│   │   ├── visualizer.py
│   │   └── rag_fallback.py
│   ├── graph.py             ← LangGraph assembly
│   └── main.py              ← CLI entrypoint
└── tests/
```

---

## Step-by-step build order

Each step ends with a concrete "done when…" checkpoint. We work one step at a
time and run it before moving on.

### Step 1 — Project scaffold + local Postgres

**Prerequisite — already done by the user:**

1. ✅ PostgreSQL 17 installed locally and running.
2. ✅ Database `SQL_POC` created.
3. ✅ `pgvector` extension installed (`CREATE EXTENSION vector` succeeded).
4. ✅ Connection role: **`postgres` superuser** (Option A — chosen for POC
   simplicity). Read-only enforcement is therefore handled entirely in
   Python code through three layers:
   1. Validator node rejects DML/DDL keywords.
   2. `sqlparse` single-statement check in the DB wrapper.
   3. Every connection issues `SET TRANSACTION READ ONLY` +
      `SET LOCAL statement_timeout` before executing user-generated SQL.

   > ⚠️ **Prod TODO:** before going to production, create a dedicated
   > least-privilege role (e.g. `sqlagent` with only `SELECT` on business
   > tables) and switch `DATABASE_URL` to it. This adds defence-in-depth at
   > the DB layer on top of the code-level checks.

**Repo-side work:**

- Create `requirements.txt`, `.env.example`, empty `src/`, `tests/`.
- `requirements.txt` uses `psycopg[binary]` + `pgvector` +
  `langgraph-checkpoint-postgres` (no SQLite/FAISS, no Docker).
- `.env.example` has
  `DATABASE_URL=postgresql+psycopg://postgres:<PASSWORD>@localhost:5432/SQL_POC`.
- **Done when:**
  1. `pip install -r requirements.txt` succeeds in a fresh venv.
  2. `psql "postgresql://postgres@localhost:5432/SQL_POC" -c "SELECT extname FROM pg_extension WHERE extname='vector';"`
     returns `vector`.

### Step 2 — Config + state

- `src/config.py`: frozen `Settings` dataclass. DB URL defaults to
  `postgresql+psycopg://postgres:<PASSWORD>@localhost:5432/SQL_POC`.
  Also: OpenAI model, `MAX_RETRIES=2`, `MAX_SQL_GENERATIONS=3`, row-scan
  budget, timeout, forbidden-keyword list.
- `src/state.py`: `AgentState` TypedDict with every field the spec calls for.
- **Done when:** `python -c "from src.config import settings; print(settings)"`
  prints the settings without errors.

### Step 3 — Sample database

- `src/db/bootstrap.sql`: `CREATE EXTENSION IF NOT EXISTS vector;` plus the
  e-commerce DDL and the `golden_sql` table (`id`, `question`, `sql`,
  `dialect`, `tables text[]`, `embedding vector(1536)`, HNSW index).
- `src/db/sample_data.py`: deterministic e-commerce seeder
  (`customers`, `products`, `orders`, `order_items`), fixed RNG seed. Runs
  `bootstrap.sql` first, then inserts data.
- **Done when:** `python -m src.db.sample_data` succeeds and
  `\dt` in `psql` shows all five tables inside `SQL_POC`.

### Step 4 — Read-only DB wrapper

- `src/db/connection.py`: cached SQLAlchemy engine pointing at
  `SQL_POC`. Each request opens a connection, issues
  `SET TRANSACTION READ ONLY` and
  `SET LOCAL statement_timeout = <QUERY_TIMEOUT_SECONDS * 1000>`, then runs
  the query. Single-statement enforcement via `sqlparse`. `run_query(sql)`
  returns `{ok, columns, rows, row_count}` or `{ok: False, error}`.
- **Done when:** a hand-written `SELECT` returns rows and a hand-written
  `INSERT` is rejected by the read-only transaction *and* by `sqlparse`.

### Step 5 — Schema introspection

- `src/db/introspect.py`: `load_schema()`, `schema_summary()`,
  `table_ddl(tables)` powered by SQLAlchemy's `inspect()`.
- **Done when:** `schema_summary()` prints a compact list of tables and columns.

### Step 6 — LLM factory

- `src/llm.py`: `get_llm()` returns `ChatOpenAI` (model from `OPENAI_MODEL`,
  default `gpt-4o-mini`). Reads `OPENAI_API_KEY` from `.env`.
- `get_embeddings()` returns `OpenAIEmbeddings` (`text-embedding-3-small`) for
  the Golden SQL FAISS store.
- **Done when:** a one-shot `get_llm().invoke("ping")` returns a response.

### Step 7 — Guardrail 1: destructive-keyword validator

- `src/guardrails/validator.py`: token-level check using `sqlparse`. Rejects
  DML/DDL keywords but must *not* false-positive on column names like
  `updated_at`.
- `tests/test_validator.py`: locks in 100% rejection across uppercase /
  lowercase / commented / CTE variants.
- **Done when:** `pytest tests/test_validator.py` is green.

### Step 8 — Guardrail 2: cost / risk heuristics

- `src/guardrails/cost.py`: runs `EXPLAIN (FORMAT JSON)` against Postgres,
  walks the plan tree to detect `Seq Scan` on large tables, `Nested Loop`
  without a join condition (cartesian), and absence of `WHERE` / `LIMIT`.
  Uses planner row estimates against `ROW_SCAN_BUDGET`. Returns
  `(ok, reason, flags)`.
- `tests/test_cost.py`: cartesian joins and missing-WHERE queries flagged;
  well-bounded queries pass.
- **Done when:** `pytest tests/test_cost.py` is green.

### Step 9 — Golden SQL few-shot store

- `src/retrieval/golden_sql.py`: reads/writes the `golden_sql` pgvector
  table. `upsert(question, sql, dialect, tables)` embeds the question with
  OpenAI and inserts a row. `retrieve(question, dialect, tables, k=5)` runs
  `ORDER BY embedding <=> :q_vec` with a `WHERE dialect = :d AND tables && :t`
  filter so few-shot examples are dialect- *and* schema-aware.
- Seed file `data/golden_sql_seed.json` loaded on first run.
- **Done when:** a known question retrieves a known-good SQL example from
  pgvector.

### Step 10 — Node: router (SQL vs RAG)

- `src/nodes/router.py`: LLM decides whether the schema can answer the
  question. Writes `{route, route_reason}`.
- **Done when:** a schema-grounded question returns `"sql"` and an
  off-schema question returns `"rag"`.

### Step 11 — Node: schema selector

- `src/nodes/schema_selector.py`: LLM picks the minimal table list given the
  schema summary and the question.
- **Done when:** "top customers by spend" returns `[customers, orders, order_items]`.

### Step 12 — Node: SQL generator

- `src/nodes/sql_generator.py`: prompt = system + dialect + selected schema +
  few-shot examples + optional correction hint + user question. Increments
  `generation_count`.
- **Done when:** given a sample question it returns a valid `SELECT`.

### Step 13 — Node: guardrail node (wraps Steps 7 & 8)

- `src/nodes/guardrail_node.py`: runs validator then cost; updates state with
  `validator_ok`, `cost_ok`, `risk_flags`, `paused`.
- **Done when:** destructive SQL sets `validator_ok=False`; cartesian joins set
  `paused=True`.

### Step 14 — Node: executor

- `src/nodes/executor.py`: calls `run_query`. Populates `raw_rows`,
  `columns`, `row_count` on success or `execution_error` on failure.
- **Done when:** both success and error paths set the expected fields.

### Step 15 — Node: self-correction

- `src/nodes/self_correction.py`: LLM emits a one-paragraph corrective *hint*
  (not a rewritten query). Increments `retry_count`. Escalates to HITL when
  `retry_count > MAX_RETRIES`.
- **Done when:** on a known-broken SQL it emits a sensible hint.

### Step 16 — Node: HITL checkpoint

- `src/nodes/hitl.py`: pure state pass-through. Graph uses
  `interrupt_before=["hitl"]`. On resume, reads
  `state["human_feedback"] = {action: approve|edit|hint, ...}`.
- **Done when:** graph pauses at `hitl` and resumes after feedback is provided.

### Step 17 — Node: formatter

- `src/nodes/formatter.py`: markdown table via Rich + LLM-generated NL summary
  grounded in the rows. Simple `confidence_score`.
- **Done when:** a completed run prints both a table and a summary.

### Step 18 — Node: visualizer

- `src/nodes/visualizer.py`: deterministic column-type inspection chooses a
  chart kind and emits templated Matplotlib/Plotly code (no free-form LLM
  codegen).
- **Done when:** a time-series result yields a line-chart snippet.

### Step 19 — Node: RAG fallback

- `src/nodes/rag_fallback.py`: stub responder used when the router picks
  `rag`. Just logs and returns an explanation for now.
- **Done when:** an off-schema question reaches this node without crashing.

### Step 20 — Graph assembly

- `src/graph.py`: wires every node with `PostgresSaver` (same
  `SQL_POC` database, under a `langgraph_checkpoints` schema) and
  `interrupt_before=["hitl"]`. Edges:

```
START → router
router       ──sql→ schema_selector
             ──rag→ rag_fallback → END
schema_selector → sql_generator → guardrail_node
guardrail_node ──ok→        executor
               ──cost_risk→ hitl → executor
               ──destructive→ formatter (error result)
executor       ──ok→  formatter → visualizer → END
               ──err & retries_left→ self_correction → sql_generator
               ──err & exhausted→    hitl → executor
```

- **Done when:** `graph.invoke({...})` returns a final state for a canned
  question.

### Step 21 — CLI entrypoint

- `src/main.py`: Typer CLI with `ask "question"` and `seed` commands.
  Handles interrupt → prompts user → resumes.
- **Done when:** `python -m src.main ask "top 5 customers by spend"` returns a
  table + summary end-to-end.

### Step 22 — End-to-end smoke test

- `tests/test_graph_smoke.py`: seeds DB, runs three canned questions, asserts
  non-empty rows and a non-empty NL answer.
- **Done when:** `pytest` is fully green.

### Step 23 — Gold-set accuracy harness

- `tests/gold/questions.jsonl` with `{question, expected_rows}`.
- A script that runs the graph on each and reports accuracy.
- **Done when:** ≥80% match on the gold set (spec target).

---

## Definition of done (from spec → file)

| Criterion | Enforced in |
|---|---|
| ≥80% accuracy on gold set | `tests/gold/` harness |
| Self-corrects within 2 retries | `graph.py` edges + `self_correction.py` |
| 100% destructive-keyword rejection | `guardrails/validator.py` + tests |
| Output = table + NL explanation | `nodes/formatter.py` |
| SQL vs RAG routing logged | `nodes/router.py` + LangSmith trace |
| Few-shot is dialect & schema aware | `retrieval/golden_sql.py` |
| Plottable results → viz code | `nodes/visualizer.py` |
| Per-query budget + override log | `guardrails/cost.py` + `nodes/hitl.py` |
| Max 3 generations / 2 retries / 1 escalation | `config.py` + `graph.py` |

---

## Working agreement

- One step at a time.
- I write the file, you review, we run it.
- If a step surfaces a question or change, we update this `plan.md` before
  continuing.
- Nothing moves to "done" until it runs cleanly.
