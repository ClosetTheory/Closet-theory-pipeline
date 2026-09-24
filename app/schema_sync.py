"""Additive schema sync: add columns the models declare but an existing table lacks.

Why this exists. The project has no migration tool; `Base.metadata.create_all()` runs at
startup and creates tables that do not exist yet. It never alters a table that does. So the
moment a deploy adds a column to a model whose table production already has, every query
touching that model fails with `UndefinedColumn` and the API returns 500s. This has now
happened twice:

- 2026-09-11: a column added to `users` broke production login.
- 2026-09-23: `outfit_votes` was created by the 11:19 UTC deploy with 11 columns; the 12:31 UTC
  deploy added seven more to the model (`comment`, `tags`, `feedback`, `garment_weights`,
  `pair_weights`, `counter_garment_ids`, `feedback_summary`). Every "Save score" on the review
  page then 500'd, because the vote ledger SELECT named columns the table did not have.

What this does. After `create_all`, compare every mapped table against the live database and
issue `ALTER TABLE ... ADD COLUMN` for each missing column. Strictly additive: it never drops,
renames, retypes or touches indexes and constraints. Those still need a deliberate migration.

NOT NULL columns need a value for the rows already in the table. The model's Python-side
default is used when it can be expressed as a literal: scalars directly, `dict`/`list`
defaults as `'{}'`/`'[]'` for JSON columns, datetime defaults as `CURRENT_TIMESTAMP`. A
NOT NULL column with no expressible default is added nullable instead, with a warning, so
startup never fails on it and the row-level fix can be made on purpose.

The DEFAULT stays on the column afterwards. It matches what the ORM would have written for a
new row anyway, and a server default that agrees with the model is harmless.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Iterable, List, Optional, Tuple

from sqlalchemy import Column, MetaData, Table, inspect, text
from sqlalchemy.engine import Connection, Dialect
from sqlalchemy.sql import sqltypes

logger = logging.getLogger(__name__)


def _quote(dialect: Dialect, name: str) -> str:
    return dialect.identifier_preparer.quote(name)


def _literal_for_default(column: Column, dialect: Dialect) -> Optional[str]:
    """SQL literal for the column's Python-side default, or None if it has none we can express."""
    default = column.default
    if default is None:
        return None
    if getattr(default, "is_sequence", False) or getattr(default, "is_clause_element", False):
        return None

    value = None
    if getattr(default, "is_scalar", False):
        value = default.arg
    elif getattr(default, "is_callable", False):
        # SQLAlchemy wraps zero-argument callables (dict, list, utc_now) so they accept the
        # execution context; passing None is how the ORM itself calls them for column defaults.
        try:
            value = default.arg(None)
        except Exception:  # noqa: BLE001 — a default we cannot evaluate is just "no default"
            return None
    else:
        return None

    if value is None:
        return None
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, _dt.datetime):
        # SQLite's ADD COLUMN refuses non-constant defaults ("Cannot add a column with
        # non-constant default"); there the column is added nullable instead. Postgres,
        # which is what production runs, back-fills with the current time.
        return None if dialect.name == "sqlite" else "CURRENT_TIMESTAMP"
    if isinstance(value, (dict, list)) and isinstance(column.type, sqltypes.JSON):
        return "'{}'" if isinstance(value, dict) else "'[]'"
    if isinstance(value, str):
        # Id-style generated defaults (uuid prefixes) are per-row and must not become a shared
        # server default; only allow strings for non-primary-key columns.
        if column.primary_key:
            return None
        return "'" + value.replace("'", "''") + "'"
    return None


def add_column_ddl(table: Table, column: Column, dialect: Dialect) -> Tuple[str, bool]:
    """Returns (DDL, added_nullable_instead_of_not_null)."""
    col_type = column.type.compile(dialect=dialect)
    parts = [
        f"ALTER TABLE {_quote(dialect, table.name)}",
        f"ADD COLUMN {_quote(dialect, column.name)} {col_type}",
    ]
    relaxed = False
    if not column.nullable and not column.primary_key:
        literal = _literal_for_default(column, dialect)
        if literal is None:
            relaxed = True  # cannot back-fill existing rows: add as nullable rather than fail startup
        else:
            parts.append(f"DEFAULT {literal} NOT NULL")
    elif column.default is not None and not column.primary_key:
        literal = _literal_for_default(column, dialect)
        if literal is not None:
            parts.append(f"DEFAULT {literal}")
    return " ".join(parts), relaxed


def sync_missing_columns(conn: Connection, metadata: MetaData, only_tables: Optional[Iterable[str]] = None) -> List[str]:
    """Adds every column in `metadata` that its (already existing) table lacks.

    Synchronous: call via `await async_conn.run_sync(sync_missing_columns, Base.metadata)`.
    Returns the list of "table.column" names it added, for logging and tests.
    """
    inspector = inspect(conn)
    existing_tables = set(inspector.get_table_names())
    wanted = set(only_tables) if only_tables is not None else None
    dialect = conn.dialect
    added: List[str] = []

    for table in metadata.sorted_tables:
        if table.name not in existing_tables:
            continue  # create_all owns brand-new tables
        if wanted is not None and table.name not in wanted:
            continue
        present = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            ddl, relaxed = add_column_ddl(table, column, dialect)
            conn.execute(text(ddl))
            added.append(f"{table.name}.{column.name}")
            if relaxed:
                logger.warning(
                    "Schema sync: added %s.%s as NULLABLE although the model says NOT NULL — it has no "
                    "default that can back-fill existing rows. Back-fill and tighten it deliberately.",
                    table.name, column.name,
                )
            else:
                logger.warning("Schema sync: added missing column %s.%s", table.name, column.name)
    return added
