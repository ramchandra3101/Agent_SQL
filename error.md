# Known Issues — Vector Database (pgvector)

Errors we ran into while bootstrapping the Golden SQL pgvector store, the
current workaround, and the proper fix to apply before production.

---

## Issue #1 — HNSW index creation fails

### Error

```
sqlalchemy.exc.ProgrammingError: (psycopg.errors.UndefinedObject)
access method "hnsw" does not exist

[SQL: CREATE INDEX golden_sql_embedding_hnsw
    ON golden_sql
    USING hnsw (embedding vector_cosine_ops);]
```

### Where it happened

- File: `src/db/bootstrap.sql`
- Triggered by: `python -m src.db.sample_data` running the bootstrap SQL.

### Root cause

The local pgvector extension is **version 0.4.4**, installed by Homebrew.
HNSW (Hierarchical Navigable Small World) is an index access method that
was **added in pgvector 0.5.0** (released August 2023). Any version older
than 0.5.0 supports only `ivfflat` and sequential scan for vector search.

Confirmed with:

```bash
psql "postgresql://postgres:postgres@localhost:5432/SQL_POC" \
     -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
#  extversion
# ------------
#  0.4.4
```

And:

```bash
psql "postgresql://postgres:postgres@localhost:5432/SQL_POC" \
     -c "ALTER EXTENSION vector UPDATE;"
# NOTICE:  version "0.4.4" of extension "vector" is already installed
# ALTER EXTENSION
```

The `ALTER EXTENSION ... UPDATE` is a no-op because Homebrew's pgvector
formula did not have a newer version available at install time. The
extension binaries on disk are 0.4.4, so Postgres cannot offer any method
newer than what 0.4.4 ships.

### Current workaround (applied in the POC)

The HNSW index is commented out in `src/db/bootstrap.sql` and replaced with
a `TODO` for when pgvector is upgraded. The `embedding vector(1536)` column
is unchanged, and queries using the `<=>` cosine-distance operator still
work — they just seq-scan the table.

At POC scale (tens to low-hundreds of golden SQL rows) the seq-scan is
faster than any index lookup, so there is no functional or performance
impact today. This workaround is **not** acceptable for production.

Relevant snippet in `bootstrap.sql`:

```sql
-- NOTE: no vector index on purpose.
--   pgvector 0.4.x (the version currently installed) does not support HNSW.
--   HNSW was added in pgvector 0.5.0. ...
-- TODO (prod): when pgvector is upgraded to >= 0.5.0, add:
--     CREATE INDEX golden_sql_embedding_hnsw
--         ON golden_sql
--         USING hnsw (embedding vector_cosine_ops);
```

### Proper fix (to apply before production)

1. **Upgrade the pgvector binaries on the host**:

   ```bash
   brew update
   brew upgrade pgvector
   ```

   Verify the new version:

   ```bash
   brew info pgvector
   ```

   If Homebrew still ships a pre-0.5 formula, install from source:

   ```bash
   git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   make install        # may need sudo
   ```

   (Pin to the latest released tag — check
   <https://github.com/pgvector/pgvector/releases>.)

2. **Restart Postgres** so it picks up the new extension files:

   ```bash
   brew services restart postgresql@17
   ```

3. **Update the extension inside `SQL_POC`**:

   ```bash
   psql "postgresql://postgres:postgres@localhost:5432/SQL_POC" \
        -c "ALTER EXTENSION vector UPDATE;"
   psql "postgresql://postgres:postgres@localhost:5432/SQL_POC" \
        -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
   ```

   Expected version ≥ `0.5.0` (ideally the latest, currently `0.8.x`).

4. **Re-enable the HNSW index** in `src/db/bootstrap.sql`:

   ```sql
   CREATE INDEX golden_sql_embedding_hnsw
       ON golden_sql
       USING hnsw (embedding vector_cosine_ops);
   ```

5. **Re-seed** (optional — the seeder is idempotent):

   ```bash
   python -m src.db.sample_data
   ```

### Documentation to read later

- pgvector HNSW overview:
  <https://github.com/pgvector/pgvector#hnsw>
- HNSW tuning parameters (`m`, `ef_construction`, `ef_search`):
  <https://github.com/pgvector/pgvector#hnsw>
- pgvector distance operators (`<->`, `<=>`, `<#>`) and when to use each:
  <https://github.com/pgvector/pgvector#distances>
- pgvector extension upgrade notes:
  <https://github.com/pgvector/pgvector#updating>
- General Postgres extension upgrade workflow:
  <https://www.postgresql.org/docs/current/sql-alterextension.html>

---

## Issue #2 — `ALTER EXTENSION vector UPDATE` reports no-op

### Error

```
NOTICE:  version "0.4.4" of extension "vector" is already installed
ALTER EXTENSION
```

### Root cause

`ALTER EXTENSION ... UPDATE` only updates to whichever versions are
**available on disk**. If Homebrew has not upgraded the pgvector package,
there is no newer `vector--X.Y--X.Z.sql` upgrade script for Postgres to
apply, so the command is a silent no-op and prints the notice above.

### Fix

Same as Issue #1 — upgrade the pgvector binaries (via Homebrew or from
source), restart Postgres, then re-run `ALTER EXTENSION vector UPDATE`.

---

## Summary

| Status | Item |
|---|---|
| ✅ | `CREATE EXTENSION vector` succeeds |
| ✅ | `golden_sql` table created with `vector(1536)` column |
| ✅ | B-tree index on `golden_sql.dialect` |
| ❌ | HNSW index on `golden_sql.embedding` — **disabled** until pgvector ≥ 0.5.0 is installed |
| 📋 | Proper fix documented above — apply before production |
