"""SQLite storage of analysis runs and per-file results."""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    joinedload,
    mapped_column,
    relationship,
    selectinload,
)

from ..types import FileResult, FrameScore, MediaKind, Segment, Verdict


class Base(DeclarativeBase):
    pass


# Bumped whenever we add columns or tables. ``_migrate_in_place`` records the
# current value in the ``schema_version`` table so older DB files can be
# detected and upgraded additively.
SCHEMA_VERSION = 3


class SchemaVersion(Base):
    __tablename__ = "schema_version"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    model_name: Mapped[str] = mapped_column(String(64))
    model_hash: Mapped[str] = mapped_column(String(64))
    config_json: Mapped[str] = mapped_column(Text)
    # `reverdict` creates derived runs that point back at the inference run
    # whose raw scores they reuse. NULL means this run produced its own scores.
    parent_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("runs.id"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(16), default="analyze")
    results: Mapped[list[Result]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    path: Mapped[str] = mapped_column(String(1024))
    kind: Mapped[str] = mapped_column(String(16))
    verdict: Mapped[str] = mapped_column(String(16))
    score: Mapped[float] = mapped_column(Float)
    faces_analyzed: Mapped[int] = mapped_column(Integer, default=0)
    processing_time_s: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    heatmap_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    run: Mapped[Run] = relationship(back_populates="results")
    frame_scores: Mapped[list[DBFrameScore]] = relationship(
        back_populates="result", cascade="all, delete-orphan", order_by="DBFrameScore.id"
    )
    segments: Mapped[list[DBSegment]] = relationship(
        back_populates="result", cascade="all, delete-orphan", order_by="DBSegment.id"
    )


class DBFrameScore(Base):
    __tablename__ = "frame_scores"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("results.id"), index=True)
    frame_index: Mapped[int] = mapped_column(Integer)
    timestamp_s: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    result: Mapped[Result] = relationship(back_populates="frame_scores")


class DBSegment(Base):
    __tablename__ = "segments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("results.id"), index=True)
    t_start: Mapped[float] = mapped_column(Float)
    t_end: Mapped[float] = mapped_column(Float)
    score_peak: Mapped[float] = mapped_column(Float)
    score_mean: Mapped[float] = mapped_column(Float)
    n_frames: Mapped[int] = mapped_column(Integer)
    result: Mapped[Result] = relationship(back_populates="segments")


def _columns(conn, table: str) -> set[str]:
    return {r[1] for r in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()}


def _read_version(conn) -> int:
    try:
        row = conn.execute(text("SELECT version FROM schema_version LIMIT 1")).fetchone()
    except OperationalError:  # table doesn't exist yet
        return 0
    return int(row[0]) if row else 0


def _json_list(raw) -> list[dict]:
    if not raw:
        return []
    try:
        val = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return []
    return val if isinstance(val, list) else []


def _results_query():
    """``SELECT results`` with children + parent run eagerly loaded (3 queries
    total instead of 1 + 3 per row)."""
    return select(Result).options(
        selectinload(Result.frame_scores),
        selectinload(Result.segments),
        joinedload(Result.run),
    )


class ResultsStore:
    """Thin façade so callers don't touch SQLAlchemy directly."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        self._migrate_in_place()

    def _migrate_in_place(self) -> None:
        """Bring older ``results.db`` files forward without Alembic.

        Every step is idempotent and checks the live schema rather than
        trusting the recorded version alone, so an interrupted migration
        simply resumes on the next open.

        * v1 → v2: ``runs.parent_run_id`` / ``runs.kind`` columns
          (``ALTER TABLE ADD COLUMN`` is rewrite-free in SQLite).
        * v2 → v3: per-frame scores and segments move out of JSON columns on
          ``results`` into the normalised ``frame_scores`` / ``segments``
          tables. SQLite can't drop columns cheaply, so ``results`` is rebuilt
          via rename → create → copy → drop.
        """
        with self.engine.begin() as conn:
            version = _read_version(conn)
            tables = set(inspect(conn).get_table_names())

            if "runs" in tables:
                existing = _columns(conn, "runs")
                for col, decl in (
                    ("parent_run_id", "INTEGER REFERENCES runs(id)"),
                    ("kind", "VARCHAR(16) DEFAULT 'analyze'"),
                ):
                    if col not in existing:
                        conn.execute(text(f"ALTER TABLE runs ADD COLUMN {col} {decl}"))

            if "results" in tables and "frame_scores" in _columns(conn, "results"):
                conn.execute(text("ALTER TABLE results RENAME TO results_old"))

        # Creates any missing tables (including the new ``results`` when we
        # just renamed the legacy one). Indexes are created with their tables.
        Base.metadata.create_all(self.engine)

        with self.engine.begin() as conn:
            if "results_old" in inspect(conn).get_table_names():
                self._copy_legacy_results(conn)
            # DBs created by the first (unindexed) v3 schema lack these; the
            # child-table lookups would otherwise be full scans.
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_frame_scores_result_id ON frame_scores (result_id)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_segments_result_id ON segments (result_id)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_results_run_id ON results (run_id)"
            ))

            current = conn.execute(text("SELECT version FROM schema_version LIMIT 1")).fetchone()
            if current is None:
                conn.execute(
                    text("INSERT INTO schema_version (version, updated_at) VALUES (:v, CURRENT_TIMESTAMP)"),
                    {"v": SCHEMA_VERSION},
                )
            elif current[0] != SCHEMA_VERSION or version != SCHEMA_VERSION:
                conn.execute(
                    text("UPDATE schema_version SET version = :v, updated_at = CURRENT_TIMESTAMP"),
                    {"v": SCHEMA_VERSION},
                )

    @staticmethod
    def _copy_legacy_results(conn) -> None:
        """Copy ``results_old`` (v≤2, JSON columns) into the v3 tables, then drop it.

        Runs in one transaction, so a crash leaves ``results_old`` intact and
        the copy is redone from scratch next time.
        """
        old_cols = _columns(conn, "results_old")
        new_cols = _columns(conn, "results")
        shared = ", ".join(sorted(old_cols & new_cols))
        conn.execute(text("DELETE FROM results"))  # in case a prior attempt half-ran
        conn.execute(text(f"INSERT INTO results ({shared}) SELECT {shared} FROM results_old"))

        seg_expr = "segments" if "segments" in old_cols else "NULL"
        rows = conn.execute(
            text(f"SELECT id, frame_scores, {seg_expr} FROM results_old")
        ).fetchall()
        frames: list[dict] = []
        segs: list[dict] = []
        for rid, fs_json, seg_json in rows:
            frames.extend(
                {"r": rid, "i": f.get("frame", 0), "t": f.get("t", 0.0), "s": f.get("score", 0.0)}
                for f in _json_list(fs_json)
            )
            segs.extend(
                {
                    "r": rid,
                    "ts": g.get("t_start", 0.0),
                    "te": g.get("t_end", 0.0),
                    "sp": g.get("score_peak", 0.0),
                    "sm": g.get("score_mean", 0.0),
                    "n": g.get("n_frames", 0),
                }
                for g in _json_list(seg_json)
            )
        # ``executemany`` — one round-trip per table instead of one per row.
        if frames:
            conn.execute(
                text("INSERT INTO frame_scores (result_id, frame_index, timestamp_s, score) "
                     "VALUES (:r, :i, :t, :s)"),
                frames,
            )
        if segs:
            conn.execute(
                text("INSERT INTO segments (result_id, t_start, t_end, score_peak, score_mean, n_frames) "
                     "VALUES (:r, :ts, :te, :sp, :sm, :n)"),
                segs,
            )
        conn.execute(text("DROP TABLE results_old"))

    def schema_version(self) -> int:
        """Return the schema version recorded in the DB (0 if absent)."""
        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT version FROM schema_version LIMIT 1")).fetchone()
            return int(row[0]) if row else 0

    def save_run(
        self,
        results: Iterable[FileResult],
        model_name: str,
        model_hash: str,
        config_dict: dict,
        parent_run_id: int | None = None,
        kind: str = "analyze",
    ) -> int:
        with Session(self.engine) as s:
            run = Run(
                model_name=model_name,
                model_hash=model_hash,
                config_json=json.dumps(config_dict, default=str),
                parent_run_id=parent_run_id,
                kind=kind,
            )
            for r in results:
                run.results.append(
                    Result(
                        path=str(r.path),
                        kind=r.kind.value if r.kind else "",
                        verdict=r.verdict.value,
                        score=r.score,
                        faces_analyzed=r.faces_analyzed,
                        processing_time_s=r.processing_time_s,
                        error=r.error,
                        heatmap_path=str(r.heatmap_path) if r.heatmap_path else None,
                        frame_scores=[
                            DBFrameScore(
                                frame_index=fs.frame_index, timestamp_s=fs.timestamp_s, score=fs.score
                            )
                            for fs in r.frame_scores
                        ],
                        segments=[
                            DBSegment(
                                t_start=seg.t_start,
                                t_end=seg.t_end,
                                score_peak=seg.score_peak,
                                score_mean=seg.score_mean,
                                n_frames=seg.n_frames,
                            )
                            for seg in (r.segments or [])
                        ],
                    )
                )
            s.add(run)
            s.commit()
            return run.id

    def save_reverdict_run(
        self,
        parent_run_id: int,
        results: Iterable[FileResult],
        threshold: float,
        uncertain_band: tuple[float, float],
    ) -> int:
        """Persist a derived run that re-thresholds an existing one.

        Reuses the parent's ``model_name``/``model_hash`` (the scores are
        unchanged) but stores the new threshold + band under ``config_json``
        so future ``runs`` / ``get_run`` calls show what verdict was applied.
        """
        with Session(self.engine) as s:
            parent = s.get(Run, parent_run_id)
            if parent is None:
                raise KeyError(f"No run with id={parent_run_id}")
            model_name, model_hash = parent.model_name, parent.model_hash
        cfg = {
            "threshold": threshold,
            "uncertain_band": list(uncertain_band),
            "derived_from": parent_run_id,
        }
        return self.save_run(
            results,
            model_name=model_name,
            model_hash=model_hash,
            config_dict=cfg,
            parent_run_id=parent_run_id,
            kind="reverdict",
        )

    def list_runs(self, limit: int = 50) -> list[dict]:
        # Count in SQL: ``len(run.results)`` would lazy-load every result row
        # (one query per run, materialising all of them) just to count.
        counts = (
            select(Result.run_id, func.count(Result.id).label("n"))
            .group_by(Result.run_id)
            .subquery()
        )
        stmt = (
            select(Run, func.coalesce(counts.c.n, 0))
            .outerjoin(counts, counts.c.run_id == Run.id)
            .order_by(Run.started_at.desc(), Run.id.desc())
            .limit(limit)
        )
        with Session(self.engine) as s:
            return [
                {
                    "id": r.id,
                    "started_at": r.started_at.isoformat(),
                    "model_name": r.model_name,
                    "model_hash": r.model_hash,
                    "n_results": n,
                    "kind": r.kind or "analyze",
                    "parent_run_id": r.parent_run_id,
                }
                for r, n in s.execute(stmt).all()
            ]

    def query(
        self,
        run_id: int | None = None,
        verdict: str | None = None,
        path_glob: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        limit: int = 200,
    ) -> list[FileResult]:
        """Filter stored results without re-running inference.

        Any combination of filters is allowed; ``None`` means "don't filter".
        ``path_glob`` is a SQL LIKE pattern (``%`` and ``_``) matched against
        the stored absolute path.
        """
        with Session(self.engine) as s:
            stmt = _results_query()
            if run_id is not None:
                stmt = stmt.where(Result.run_id == run_id)
            if verdict is not None:
                stmt = stmt.where(Result.verdict == verdict)
            if path_glob is not None:
                stmt = stmt.where(Result.path.like(path_glob))
            if min_score is not None:
                stmt = stmt.where(Result.score >= min_score)
            if max_score is not None:
                stmt = stmt.where(Result.score <= max_score)
            rows = s.execute(stmt.order_by(Result.score.desc()).limit(limit)).scalars().all()
            return [self._row_to_fileresult(r) for r in rows]

    def _row_to_fileresult(self, r: Result) -> FileResult:
        return FileResult(
            path=Path(r.path),
            kind=MediaKind(r.kind) if r.kind else None,
            verdict=Verdict(r.verdict),
            score=r.score,
            faces_analyzed=r.faces_analyzed,
            frame_scores=[
                FrameScore(
                    frame_index=fs.frame_index,
                    timestamp_s=fs.timestamp_s,
                    score=fs.score,
                )
                for fs in (r.frame_scores or [])
            ],
            segments=[
                Segment(
                    t_start=seg.t_start,
                    t_end=seg.t_end,
                    score_peak=seg.score_peak,
                    score_mean=seg.score_mean,
                    n_frames=seg.n_frames,
                )
                for seg in (r.segments or [])
            ],
            processing_time_s=r.processing_time_s,
            error=r.error,
            model_name=r.run.model_name,
            model_hash=r.run.model_hash,
            heatmap_path=Path(r.heatmap_path) if r.heatmap_path else None,
            created_at=r.created_at,
        )

    def get_run(self, run_id: int) -> list[FileResult]:
        with Session(self.engine) as s:
            rows = s.execute(
                _results_query().where(Result.run_id == run_id).order_by(Result.id)
            ).scalars().all()
            return [self._row_to_fileresult(r) for r in rows]
