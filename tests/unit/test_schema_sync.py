"""app/schema_sync.py — adding columns the model declares but the live table lacks.

Reproduces the production failure of 2026-09-23: `outfit_votes` existed with 11 columns, the
next deploy's model had 18, `create_all` did nothing, and every review save 500'd.
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, MetaData, String, Table, Text, create_engine, inspect, text

from app.schema_sync import add_column_ddl, sync_missing_columns


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _model_metadata() -> MetaData:
    """The table as the *current* model declares it (the wide version)."""
    md = MetaData()
    Table(
        "votes", md,
        Column("id", String(64), primary_key=True),
        Column("vote", String(8), nullable=False),
        # Everything below was added after the table first shipped.
        Column("comment", Text, nullable=True),
        Column("tags", JSON, default=list, nullable=False),
        Column("feedback", JSON, nullable=True),
        Column("garment_weights", JSON, default=dict, nullable=False),
        Column("weight", Float, default=1.0, nullable=False),
        Column("would_wear", Boolean, default=False, nullable=False),
        Column("source", String(32), default="review", nullable=False),
        Column("retries", Integer, default=0, nullable=False),
        Column("updated_at", DateTime(timezone=True), default=_utc_now, nullable=False),
        Column("no_default_required", String(16), nullable=False),  # NOT NULL, nothing to back-fill with
    )
    return md


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        # The table as production actually has it: the narrow, first-shipped shape, with a row in it.
        conn.execute(text("CREATE TABLE votes (id VARCHAR(64) PRIMARY KEY, vote VARCHAR(8) NOT NULL)"))
        conn.execute(text("INSERT INTO votes (id, vote) VALUES ('v1', 'up')"))
        # An unrelated table create_all would own; it must be left alone.
        conn.execute(text("CREATE TABLE other (id INTEGER PRIMARY KEY)"))
    return eng


def test_adds_every_missing_column_and_only_those(engine):
    md = _model_metadata()
    with engine.begin() as conn:
        added = sync_missing_columns(conn, md)

    assert set(added) == {
        "votes.comment", "votes.tags", "votes.feedback", "votes.garment_weights", "votes.weight",
        "votes.would_wear", "votes.source", "votes.retries", "votes.updated_at", "votes.no_default_required",
    }
    cols = {c["name"] for c in inspect(engine).get_columns("votes")}
    assert {"id", "vote"} | {a.split(".")[1] for a in added} == cols


def test_existing_rows_get_model_defaults_and_new_rows_insert_cleanly(engine):
    md = _model_metadata()
    with engine.begin() as conn:
        sync_missing_columns(conn, md)
        row = conn.execute(text(
            "SELECT tags, garment_weights, weight, would_wear, source, retries, updated_at, comment, no_default_required "
            "FROM votes WHERE id = 'v1'"
        )).one()
        assert row[0] == "[]"          # list default -> '[]'
        assert row[1] == "{}"          # dict default -> '{}'
        assert row[2] == 1.0
        assert row[3] in (0, False)    # bool default -> FALSE
        assert row[4] == "review"
        assert row[5] == 0
        assert row[6] is None          # SQLite cannot ADD COLUMN with CURRENT_TIMESTAMP: relaxed to nullable
        assert row[7] is None          # nullable, no default
        assert row[8] is None          # NOT NULL without a default was relaxed to nullable

        # The exact shape of the failing production query: an INSERT naming only the ORM's
        # columns, relying on defaults for the rest, must now succeed.
        conn.execute(text("INSERT INTO votes (id, vote, tags, garment_weights) VALUES ('v2', 'down', '[\"x\"]', '{}')"))
        assert conn.execute(text("SELECT count(*) FROM votes")).scalar() == 2


def test_second_run_is_a_no_op(engine):
    md = _model_metadata()
    with engine.begin() as conn:
        sync_missing_columns(conn, md)
    with engine.begin() as conn:
        assert sync_missing_columns(conn, md) == []


def test_missing_table_is_left_to_create_all(engine):
    md = MetaData()
    Table("brand_new", md, Column("id", Integer, primary_key=True), Column("x", Integer, nullable=False, default=0))
    with engine.begin() as conn:
        assert sync_missing_columns(conn, md) == []
    assert "brand_new" not in inspect(engine).get_table_names()


def test_not_null_without_default_is_relaxed_not_fatal(engine):
    md = MetaData()
    t = Table("votes", md, Column("id", String(64), primary_key=True), Column("strict", String(8), nullable=False))
    ddl, relaxed = add_column_ddl(t, t.c.strict, engine.dialect)
    assert relaxed is True
    assert "NOT NULL" not in ddl
    assert ddl.startswith('ALTER TABLE votes ADD COLUMN strict VARCHAR(8)') or ddl.startswith('ALTER TABLE "votes" ADD COLUMN "strict" VARCHAR(8)')


def test_generated_id_defaults_never_become_a_shared_server_default(engine):
    md = MetaData()
    t = Table("t", md, Column("id", String(64), primary_key=True, default=lambda: "fixed_id"))
    ddl, _ = add_column_ddl(t, t.c.id, engine.dialect)
    assert "DEFAULT" not in ddl


def test_real_models_are_in_sync_after_create_all():
    """Guard for the app's own metadata: create_all then sync must add nothing, otherwise a
    model has a column create_all cannot express and this module would fight it every boot."""
    from app.models.base import Base
    import app.models  # noqa: F401 — registers every table on Base.metadata

    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        Base.metadata.create_all(conn)
        assert sync_missing_columns(conn, Base.metadata) == []


def test_postgres_backfills_datetime_with_current_timestamp():
    from sqlalchemy.dialects import postgresql
    md = MetaData()
    t = Table("t", md, Column("id", String(64), primary_key=True),
              Column("updated_at", DateTime(timezone=True), default=_utc_now, nullable=False))
    ddl, relaxed = add_column_ddl(t, t.c.updated_at, postgresql.dialect())
    assert relaxed is False
    assert ddl == 'ALTER TABLE t ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL'


def test_postgres_ddl_for_the_actual_outfit_votes_columns():
    """The seven columns production was missing on 2026-09-23, compiled for Postgres."""
    from sqlalchemy.dialects import postgresql
    from app.models.outfit_vote import OutfitVote
    table = OutfitVote.__table__
    ddls = {name: add_column_ddl(table, table.c[name], postgresql.dialect())[0] for name in
            ("comment", "tags", "feedback", "garment_weights", "pair_weights", "counter_garment_ids", "feedback_summary")}
    assert ddls["tags"] == "ALTER TABLE outfit_votes ADD COLUMN tags JSON DEFAULT '[]' NOT NULL"
    assert ddls["garment_weights"] == "ALTER TABLE outfit_votes ADD COLUMN garment_weights JSON DEFAULT '{}' NOT NULL"
    assert ddls["comment"] == "ALTER TABLE outfit_votes ADD COLUMN comment TEXT"
    assert ddls["feedback"] == "ALTER TABLE outfit_votes ADD COLUMN feedback JSON"
    assert ddls["feedback_summary"] == "ALTER TABLE outfit_votes ADD COLUMN feedback_summary VARCHAR(300)"
