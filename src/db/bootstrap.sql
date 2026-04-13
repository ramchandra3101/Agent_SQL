-- ============================================================================
--  SQL Agent — database bootstrap
-- ----------------------------------------------------------------------------
--  Run by src/db/sample_data.py. Also safe to run manually with:
--      psql "postgresql://postgres:<PASSWORD>@localhost:5432/SQL_POC" -f bootstrap.sql
--
--  This file is idempotent — every DROP is guarded, every CREATE uses
--  IF NOT EXISTS where possible. Re-running it wipes and rebuilds the
--  business tables *and* the golden_sql store, so do not run it against
--  a database you care about.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ----------------------------------------------------------------------------
--  Business tables — the data the agent queries.
--  A tiny e-commerce schema: customers place orders, orders have items,
--  items reference products.
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders      CASCADE;
DROP TABLE IF EXISTS products    CASCADE;
DROP TABLE IF EXISTS customers   CASCADE;

CREATE TABLE customers (
    customer_id  SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    email        TEXT NOT NULL UNIQUE,
    country      TEXT NOT NULL,
    signup_date  DATE NOT NULL
);

CREATE TABLE products (
    product_id   SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,
    unit_price   NUMERIC(10, 2) NOT NULL
);

CREATE TABLE orders (
    order_id     SERIAL PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date   DATE    NOT NULL,
    status       TEXT    NOT NULL
);

CREATE TABLE order_items (
    item_id      SERIAL PRIMARY KEY,
    order_id     INTEGER NOT NULL REFERENCES orders(order_id),
    product_id   INTEGER NOT NULL REFERENCES products(product_id),
    quantity     INTEGER NOT NULL,
    line_total   NUMERIC(10, 2) NOT NULL
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date     ON orders(order_date);
CREATE INDEX idx_items_order     ON order_items(order_id);
CREATE INDEX idx_items_product   ON order_items(product_id);

-- ----------------------------------------------------------------------------
--  Golden SQL store — pgvector-backed few-shot examples.
--  Written by the agent whenever a user question is successfully answered.
--  Read by the SQL generator node to inject dialect- and schema-aware
--  few-shot examples into the prompt.
--
--  `tables` is a text[] so we can filter "show me examples that touched the
--   same tables as the current question" at retrieval time.
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS golden_sql CASCADE;

CREATE TABLE golden_sql (
    id          SERIAL PRIMARY KEY,
    question    TEXT         NOT NULL,
    sql         TEXT         NOT NULL,
    dialect     TEXT         NOT NULL DEFAULT 'postgresql',
    tables      TEXT[]       NOT NULL DEFAULT '{}',
    embedding   vector(1536) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- NOTE: no vector index on purpose.
--   pgvector 0.4.x (the version currently installed) does not support HNSW.
--   HNSW was added in pgvector 0.5.0. At POC scale the Golden SQL store
--   holds tens to low-hundreds of rows, so a sequential scan over the
--   `embedding` column is fast enough (microseconds).
--
-- TODO (prod): when pgvector is upgraded to >= 0.5.0, add:
--     CREATE INDEX golden_sql_embedding_hnsw
--         ON golden_sql
--         USING hnsw (embedding vector_cosine_ops);
--
-- B-tree on dialect so the retrieval "WHERE dialect = 'postgresql'" filter
-- stays cheap regardless of table size.
CREATE INDEX golden_sql_dialect_idx ON golden_sql(dialect);
