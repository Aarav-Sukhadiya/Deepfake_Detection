"""SQLite storage of analysis runs and per-file results."""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

from ..types import FileResult, FrameScore, MediaKind, Segment, Verdict


class Base(DeclarativeBase):
    pass


# Bumped whenever we add columns or tables. ``_migrate_in_place`` records the
# current value in the ``schema_version`` table so older DB files can be
# detected and upgraded additively.
SCHEMA_VERSION = 2


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
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    path: Mapped[str] = mapped_column(String(1024))
    kind: Mapped[str] = mapped_column(String(16))
    verdict: Mapped[str] = mapped_column(String(16))
    score: Mapped[float] = mapped_column(Float)
    faces_analyzed: Mapped[int] = mapped_column(Integer, default=0)
    processing_time_s: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    heatmap_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    frame_scores: Mapped[dict] = mapped_column(JSON, default=list)
    segments: Mapped[dict] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    run: Mapped[Run] = relationship(back_populates="results")


class ResultsStore:
    """Thin façade so callers don't touch SQLAlchemy directly."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        Base.metadata.create_all(self.engine)
        self._migrate_in_place()

    def _migrate_in_place(self) -> None:
        """Best-effort additive migrations for pre-existing SQLite files.

        SQLite's ``ALTER TABLE ADD COLUMN`` is cheap and rewrite-free; we use it
        to bring older ``results.db`` schemas forward without Alembic. Each
        ``ADD COLUMN`` is idempotent: we list current columns first and only
        add the missing ones. After every successful migration we update the
        ``schema_version`` row so future readers can verify compatibility.
        """
        additions = {
            "runs": [
                ("parent_run_id", "INTEGER REFERENCES runs(id)"),
                ("kind", "VARCHAR(16) DEFAULT 'analyze'"),
            ],
            "results": [
                ("segments", "JSON DEFAULT '[]'"),
            ],
        }
        with self.engine.begin() as conn:
            for table, cols in additions.items():
                rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
                existing = {r[1] for r in rows}
                for col, decl in cols:
                    if col not in existing:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {decl}"))
            # Record current schema version (insert-or-update single row).
            current = conn.execute(text("SELECT version FROM schema_version LIMIT 1")).fetchone()
            if current is None:
                conn.execute(
                    text(
                        "INSERT INTO schema_version (version, updated_at) "
                        "VALUES (:v, CURRENT_TIMESTAMP)"
                    ),
                    {"v": SCHEMA_VERSION},
                )
            elif current[0] != SCHEMA_VERSION:
                conn.execute(
                    text("UPDATE schema_version SET version = :v, updated_at = CURRENT_TIMESTAMP"),
                    {"v": SCHEMA_VERSION},
                )

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
                        kind=r.kind.value,
                        verdict=r.verdict.value,
                        score=r.score,
                        faces_analyzed=r.faces_analyzed,
                        processing_time_s=r.processing_time_s,
                        error=r.error,
                        heatmap_path=str(r.heatmap_path) if r.heatmap_path else None,
                        frame_scores=[
                            {"frame": fs.frame_index, "t": fs.timestamp_s, "score": fs.score}
                            for fs in r.frame_scores
                        ],
                        segments=[
                            {
                                "t_start": seg.t_start,
                                "t_end": seg.t_end,
                                "score_peak": seg.score_peak,
                                "score_mean": seg.score_mean,
                                "n_frames": seg.n_frames,
                            }
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
            cfg = {
                "threshold": threshold,
                "uncertain_band": list(uncertain_band),
                "derived_from": parent_run_id,
            }
        return self.save_run(
            results,
            model_name=parent.model_name,
            model_hash=parent.model_hash,
            config_dict=cfg,
            parent_run_id=parent_run_id,
            kind="reverdict",
        )

    def list_runs(self, limit: int = 50) -> list[dict]:
        with Session(self.engine) as s:
            rows = s.execute(
                select(Run).order_by(Run.started_at.desc()).limit(limit)
            ).scalars().all()
            return [
                {
                    "id": r.id,
                    "started_at": r.started_at.isoformat(),
                    "model_name": r.model_name,
                    "model_hash": r.model_hash,
                    "n_results": len(r.results),
                    "kind": r.kind or "analyze",
                    "parent_run_id": r.parent_run_id,
                }
                for r in rows
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
            stmt = select(Result)
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
            kind=MediaKind(r.kind),
            verdict=Verdict(r.verdict),
            score=r.score,
            faces_analyzed=r.faces_analyzed,
            frame_scores=[
                FrameScore(
                    frame_index=fs["frame"],
                    timestamp_s=fs["t"],
                    score=fs["score"],
                )
                for fs in (r.frame_scores or [])
            ],
            segments=[
                Segment(
                    t_start=seg["t_start"],
                    t_end=seg["t_end"],
                    score_peak=seg["score_peak"],
                    score_mean=seg["score_mean"],
                    n_frames=seg["n_frames"],
                )
                for seg in (r.segments or [])
            ],
            processing_time_s=r.processing_time_s,
            error=r.error,
            heatmap_path=Path(r.heatmap_path) if r.heatmap_path else None,
            created_at=r.created_at,
        )

    def get_run(self, run_id: int) -> list[FileResult]:
        with Session(self.engine) as s:
            rows = s.execute(
                select(Result).where(Result.run_id == run_id)
            ).scalars().all()
            return [self._row_to_fileresult(r) for r in rows]
