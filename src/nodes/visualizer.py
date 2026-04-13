"""Visualizer node — emits chart code for the result set.

Intentionally **not** an LLM codegen node. Letting a model write matplotlib
from scratch produces fragile code, hallucinated columns, and a security
surface (arbitrary Python). Instead we use a tiny deterministic rule:

1. Inspect the first result row and classify each column as ``temporal``,
   ``numeric``, or ``categorical``.
2. Pick a chart kind:
     * temporal + numeric → ``line`` (time series)
     * categorical + numeric → ``bar``
     * two numerics → ``scatter``
     * single numeric → ``histogram``
     * otherwise → ``none``
3. Render a small, templated matplotlib snippet with the chosen columns
   hard-coded. The CLI can exec it, save it to a file, or show it.

The output is a **string of code**, not an image. That keeps the node pure
and lets the caller decide where to run it.
"""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from ..state import AgentState


_NUMERIC_TYPES = (int, float, Decimal)
_TEMPORAL_TYPES = (datetime, date, time)


def _classify(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):  # bool is subclass of int — check first
        return "categorical"
    if isinstance(value, _TEMPORAL_TYPES):
        return "temporal"
    if isinstance(value, _NUMERIC_TYPES):
        return "numeric"
    return "categorical"


def _column_kinds(columns: list[str], rows: list[dict[str, Any]]) -> dict[str, str]:
    """Infer a kind for each column from the first non-null value seen."""
    kinds: dict[str, str] = {c: "unknown" for c in columns}
    for col in columns:
        for row in rows:
            if row.get(col) is not None:
                kinds[col] = _classify(row[col])
                break
    return kinds


def _choose_chart(kinds: dict[str, str]) -> tuple[str, list[str]]:
    """Return ``(chart_kind, columns_used)``."""
    temporal = [c for c, k in kinds.items() if k == "temporal"]
    numeric = [c for c, k in kinds.items() if k == "numeric"]
    categorical = [c for c, k in kinds.items() if k == "categorical"]

    if temporal and numeric:
        return "line", [temporal[0], numeric[0]]
    if categorical and numeric:
        return "bar", [categorical[0], numeric[0]]
    if len(numeric) >= 2:
        return "scatter", [numeric[0], numeric[1]]
    if len(numeric) == 1:
        return "histogram", [numeric[0]]
    return "none", []


def _render_code(chart: str, cols: list[str]) -> str:
    """Return a templated matplotlib snippet for ``chart``."""
    if chart == "line":
        x, y = cols
        return (
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n\n"
            "df = pd.DataFrame(rows)\n"
            f"df = df.sort_values({x!r})\n"
            f"plt.plot(df[{x!r}], df[{y!r}], marker='o')\n"
            f"plt.xlabel({x!r})\n"
            f"plt.ylabel({y!r})\n"
            f"plt.title({y!r} + ' over ' + {x!r})\n"
            "plt.xticks(rotation=45)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    if chart == "bar":
        x, y = cols
        return (
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n\n"
            "df = pd.DataFrame(rows)\n"
            f"df = df.sort_values({y!r}, ascending=False)\n"
            f"plt.bar(df[{x!r}].astype(str), df[{y!r}])\n"
            f"plt.xlabel({x!r})\n"
            f"plt.ylabel({y!r})\n"
            f"plt.title({y!r} + ' by ' + {x!r})\n"
            "plt.xticks(rotation=45)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    if chart == "scatter":
        x, y = cols
        return (
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n\n"
            "df = pd.DataFrame(rows)\n"
            f"plt.scatter(df[{x!r}], df[{y!r}])\n"
            f"plt.xlabel({x!r})\n"
            f"plt.ylabel({y!r})\n"
            f"plt.title({y!r} + ' vs ' + {x!r})\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    if chart == "histogram":
        (x,) = cols
        return (
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n\n"
            "df = pd.DataFrame(rows)\n"
            f"plt.hist(df[{x!r}], bins=20)\n"
            f"plt.xlabel({x!r})\n"
            "plt.ylabel('count')\n"
            f"plt.title('Distribution of ' + {x!r})\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    return ""


def visualize_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: emit chart code for the current result set."""
    columns = state.get("columns") or []
    rows = state.get("raw_rows") or []

    if not columns or not rows:
        return {"visualization_code": None}

    kinds = _column_kinds(columns, rows)
    chart, cols = _choose_chart(kinds)
    if chart == "none":
        return {"visualization_code": None}

    code = _render_code(chart, cols)
    return {"visualization_code": code}


# ---------------------------------------------------------------------------
#  Sanity check:  python -m src.nodes.visualizer
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import date as _date

    from ..state import initial_state

    cases: list[tuple[str, list[str], list[dict[str, Any]]]] = [
        (
            "bar (categorical + numeric)",
            ["country", "revenue"],
            [
                {"country": "US", "revenue": 1200.0},
                {"country": "UK", "revenue": 800.0},
                {"country": "FR", "revenue": 500.0},
            ],
        ),
        (
            "line (temporal + numeric)",
            ["month", "orders"],
            [
                {"month": _date(2024, 1, 1), "orders": 90},
                {"month": _date(2024, 2, 1), "orders": 110},
                {"month": _date(2024, 3, 1), "orders": 130},
            ],
        ),
        (
            "histogram (single numeric)",
            ["age"],
            [{"age": 20}, {"age": 25}, {"age": 42}],
        ),
        (
            "none (all categorical)",
            ["country"],
            [{"country": "US"}, {"country": "UK"}],
        ),
    ]

    for label, cols, rows in cases:
        state = initial_state("probe", thread_id="probe")
        state["columns"] = cols
        state["raw_rows"] = rows
        out = visualize_node(state)
        print(f"[{label}]")
        print(out["visualization_code"] or "(no chart)")
        print()
