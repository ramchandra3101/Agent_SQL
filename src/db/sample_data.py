"""Bootstrap and seed the Postgres database used by the POC.

Usage::

    python -m src.db.sample_data

This script is the one-stop way to get a clean demo database:

1. Runs ``bootstrap.sql`` (creates the pgvector extension, drops and recreates
   the business tables, and creates the ``golden_sql`` few-shot store).
2. Inserts deterministic sample data with a fixed RNG seed so every developer
   and every test run sees identical rows.

It is safe to re-run at any time — each run wipes and rebuilds everything.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ..config import settings

BOOTSTRAP_SQL_PATH = Path(__file__).with_name("bootstrap.sql")

COUNTRIES = ["US", "UK", "DE", "FR", "IN", "JP", "BR", "CA", "AU", "ES"]
CATEGORIES = ["Electronics", "Books", "Home", "Clothing", "Toys"]
ORDER_STATUSES = [
    "completed", "completed", "completed",  # weight toward completed
    "pending",
    "cancelled",
]

NUM_CUSTOMERS = 200
NUM_PRODUCTS = 50
NUM_ORDERS = 1_000
MAX_ITEMS_PER_ORDER = 4
MAX_QTY_PER_ITEM = 5


def _make_engine() -> Engine:
    """Build a SQLAlchemy engine that autocommits DDL + seed inserts."""
    return create_engine(settings.database_url, future=True)


def _run_bootstrap(engine: Engine) -> None:
    """Execute bootstrap.sql against the target database."""
    sql = BOOTSTRAP_SQL_PATH.read_text()
    # psycopg can run a multi-statement string when autocommit is on.
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        for raw_stmt in _split_sql(sql):
            conn.execute(text(raw_stmt))


def _split_sql(sql: str) -> list[str]:
    """Split a SQL script into individual statements.

    Postgres psycopg3 accepts multi-statement strings, but using explicit
    splits keeps error messages pointing at the right statement and avoids
    surprises with the ``vector`` extension DDL.
    """
    buf: list[str] = []
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            buf.append("\n".join(current))
            current = []
    if current:
        buf.append("\n".join(current))
    return buf


def _seed(engine: Engine) -> None:
    rng = random.Random(42)

    with engine.begin() as conn:
        # ---- customers -----------------------------------------------------
        conn.execute(
            text(
                """
                INSERT INTO customers (name, email, country, signup_date)
                VALUES (:name, :email, :country, :signup)
                """
            ),
            [
                {
                    "name": f"Customer {cid}",
                    "email": f"c{cid}@example.com",
                    "country": rng.choice(COUNTRIES),
                    "signup": date(2023, 1, 1) + timedelta(days=rng.randint(0, 700)),
                }
                for cid in range(1, NUM_CUSTOMERS + 1)
            ],
        )

        # ---- products ------------------------------------------------------
        conn.execute(
            text(
                """
                INSERT INTO products (name, category, unit_price)
                VALUES (:name, :category, :price)
                """
            ),
            [
                {
                    "name": f"Product {pid}",
                    "category": rng.choice(CATEGORIES),
                    "price": round(rng.uniform(5, 500), 2),
                }
                for pid in range(1, NUM_PRODUCTS + 1)
            ],
        )

        # pull product prices once so order-item totals are consistent
        price_rows = conn.execute(
            text("SELECT product_id, unit_price FROM products ORDER BY product_id")
        ).all()
        prices: dict[int, float] = {pid: float(price) for pid, price in price_rows}

        # ---- orders --------------------------------------------------------
        order_rows = [
            {
                "customer_id": rng.randint(1, NUM_CUSTOMERS),
                "order_date": date(2024, 1, 1) + timedelta(days=rng.randint(0, 830)),
                "status": rng.choice(ORDER_STATUSES),
            }
            for _ in range(NUM_ORDERS)
        ]
        conn.execute(
            text(
                """
                INSERT INTO orders (customer_id, order_date, status)
                VALUES (:customer_id, :order_date, :status)
                """
            ),
            order_rows,
        )
        # Fetch the generated order_ids back so the order_items insert can
        # reference them. `RETURNING` inside an executemany isn't reliably
        # returned to SQLAlchemy, so we query separately.
        order_ids = [
            row[0]
            for row in conn.execute(
                text("SELECT order_id FROM orders ORDER BY order_id")
            )
        ]

        # ---- order_items ---------------------------------------------------
        item_rows: list[dict[str, object]] = []
        for oid in order_ids:
            for _ in range(rng.randint(1, MAX_ITEMS_PER_ORDER)):
                pid = rng.randint(1, NUM_PRODUCTS)
                qty = rng.randint(1, MAX_QTY_PER_ITEM)
                item_rows.append(
                    {
                        "order_id": oid,
                        "product_id": pid,
                        "quantity": qty,
                        "line_total": round(prices[pid] * qty, 2),
                    }
                )
        conn.execute(
            text(
                """
                INSERT INTO order_items (order_id, product_id, quantity, line_total)
                VALUES (:order_id, :product_id, :quantity, :line_total)
                """
            ),
            item_rows,
        )


def _print_summary(engine: Engine) -> None:
    with engine.connect() as conn:
        tables = ["customers", "products", "orders", "order_items", "golden_sql"]
        print("Seeded SQL_POC:")
        for t in tables:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar_one()
            print(f"  {t:<12} {count:>8} rows")


def main() -> None:
    engine = _make_engine()
    _run_bootstrap(engine)
    _seed(engine)
    _print_summary(engine)


if __name__ == "__main__":
    main()
