"""dfdetect command-line interface (Typer + Rich)."""
from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .config import Config, load_config
from .types import FileResult, Verdict

app = typer.Typer(
    add_completion=False,
    help="Local deepfake detection for images and videos.",
    no_args_is_help=True,
)
console = Console()


_OUTPUT_FORMATS = ("json", "csv", "html")


def _setup_logging(level: str) -> None:
    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise typer.BadParameter(f"Unknown log level: {level}", param_hint="--log-level")
    logging.basicConfig(
        level=resolved,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, markup=True)],
    )


def _verdict_style(v: Verdict) -> str:
    return {
        Verdict.REAL: "green",
        Verdict.FAKE: "red bold",
        Verdict.UNCERTAIN: "yellow",
        Verdict.NO_FACE: "dim",
        Verdict.ERROR: "red italic",
    }[v]


def _summary_table(results: list[FileResult]) -> Table:
    t = Table(title="Results", show_lines=False)
    t.add_column("File", overflow="fold")
    t.add_column("Verdict")
    t.add_column("Score", justify="right")
    t.add_column("Faces", justify="right")
    t.add_column("Time", justify="right")
    for r in results:
        t.add_row(
            r.path.name,
            f"[{_verdict_style(r.verdict)}]{r.verdict.value}[/]",
            f"{r.score:.3f}",
            str(r.faces_analyzed),
            f"{r.processing_time_s:.2f}s",
        )
    return t


def _load(config: Path | None) -> Config:
    try:
        return load_config(config)
    except (FileNotFoundError, ValueError) as e:  # pydantic ValidationError is a ValueError
        raise typer.BadParameter(str(e), param_hint="--config") from e


def _apply_overrides(
    cfg: Config,
    threshold: float | None,
    gpu: bool | None,
    fps: float | None,
    batch_size: int | None,
    heatmaps: bool | None,
) -> Config:
    if threshold is not None:
        if not 0.0 <= threshold <= 1.0:
            raise typer.BadParameter("must be in [0, 1]", param_hint="--threshold")
        cfg.threshold = threshold
    if gpu is True:
        cfg.device = "cuda"
    if gpu is False:
        cfg.device = "cpu"
    if fps is not None:
        if fps <= 0:
            raise typer.BadParameter("must be > 0", param_hint="--frames-per-second")
        cfg.video.frames_per_second = fps
    if batch_size is not None:
        if batch_size < 1:
            raise typer.BadParameter("must be >= 1", param_hint="--batch-size")
        cfg.batch_size = batch_size
    if heatmaps is not None:
        cfg.report.heatmaps = heatmaps
    return cfg


@app.command()
def analyze(
    paths: list[Path] = typer.Argument(..., help="Files or directories to analyze."),
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config path."),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output format: json | csv | html (default: config report.default_format)."
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Where to write reports (default: config report.output_dir)."
    ),
    threshold: float | None = typer.Option(None, "--threshold", "-t"),
    recursive: bool | None = typer.Option(None, "--recursive/--no-recursive", "-r"),
    gpu: bool | None = typer.Option(None, "--gpu/--cpu", help="Force device."),
    fps: float | None = typer.Option(None, "--frames-per-second", "-f"),
    batch_size: int | None = typer.Option(None, "--batch-size", "-b"),
    heatmaps: bool | None = typer.Option(None, "--heatmaps/--no-heatmaps"),
    model_card: bool = typer.Option(
        False, "--model-card/--no-model-card",
        help="Render a model-card markdown alongside the report.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Analyze the given paths and write a report."""
    _setup_logging(log_level)
    cfg = _load(config)
    # CLI flags override the config only when actually passed.
    if recursive is not None:
        cfg.input.recursive = recursive
    if output_dir is not None:
        cfg.report.output_dir = output_dir
    cfg = _apply_overrides(cfg, threshold, gpu, fps, batch_size, heatmaps)
    output = (output or cfg.report.default_format).lower()
    # Validate before the (potentially hours-long) analysis, not after.
    if output not in _OUTPUT_FORMATS:
        raise typer.BadParameter(
            f"Unknown output format {output!r}; choose from {', '.join(_OUTPUT_FORMATS)}.",
            param_hint="--output",
        )

    # Defer heavy imports until after we've parsed CLI args.
    from .pipeline import Pipeline
    from .report.exporter import write_csv, write_html, write_json

    pipeline = Pipeline(cfg)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as bar:
        task_id = bar.add_task("Analyzing", total=None)

        def progress_cb(done: int, total: int, label: str):
            bar.update(task_id, total=total, completed=done, description=f"Analyzing {label}")

        results = pipeline.run(paths, progress=progress_cb)

    console.print(_summary_table(results))

    out_name = f"dfdetect-report.{output}"
    out_path = cfg.report.output_dir / out_name
    if output == "json":
        write_json(results, out_path)
    elif output == "csv":
        write_csv(results, out_path)
    else:
        write_html(
            results,
            out_path,
            threshold=cfg.threshold,
            uncertain_band=cfg.uncertain_band,
        )
    console.print(f"Wrote [cyan]{out_path}[/cyan]")

    if model_card:
        from .report.model_card import render_model_card
        card_path = cfg.report.output_dir / "model-card.md"
        render_model_card(cfg, results, card_path)
        console.print(f"Wrote [cyan]{card_path}[/cyan]")


@app.command()
def ui(
    config: Path | None = typer.Option(None, "--config", "-c"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    share: bool = typer.Option(False, "--share"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Launch the Gradio web UI."""
    _setup_logging(log_level)
    from .ui import launch
    launch(config_path=config, host=host, port=port, share=share)


@app.command()
def runs(
    config: Path | None = typer.Option(None, "--config", "-c"),
    limit: int = typer.Option(20, "--limit"),
):
    """List recent analysis runs from the local SQLite store."""
    cfg = _load(config)
    from .storage.db import ResultsStore
    store = ResultsStore(cfg.storage.db_path)
    rows = store.list_runs(limit=limit)
    t = Table(title=f"Recent runs ({len(rows)})")
    t.add_column("ID", justify="right")
    t.add_column("Started")
    t.add_column("Model")
    t.add_column("Hash")
    t.add_column("Files", justify="right")
    t.add_column("Kind")
    for r in rows:
        kind = r["kind"] if r["parent_run_id"] is None else f'{r["kind"]} (of {r["parent_run_id"]})'
        t.add_row(
            str(r["id"]), r["started_at"], r["model_name"], r["model_hash"], str(r["n_results"]), kind
        )
    console.print(t)


@app.command("reverdict")
def reverdict(
    run_id: int = typer.Argument(..., help="Run ID from `dfdetect runs`."),
    threshold: float | None = typer.Option(
        None, "--threshold", "-t", help="New threshold (default: config threshold)."
    ),
    uncertain_lo: float | None = typer.Option(
        None, "--uncertain-lo", help="Override uncertain-band lower bound."
    ),
    uncertain_hi: float | None = typer.Option(
        None, "--uncertain-hi", help="Override uncertain-band upper bound."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print updated verdicts without writing a derived run."
    ),
    config: Path | None = typer.Option(None, "--config", "-c"),
):
    """Re-apply a different threshold to an existing run without re-running inference.

    By default, a *derived* run is persisted referencing the original via
    ``parent_run_id``, so the new verdicts are queryable later. Use ``--dry-run``
    to preview without writing.
    """
    cfg = _load(config)
    from .inference.aggregator import reverdict as apply_verdict
    from .storage.db import ResultsStore

    if threshold is None:
        threshold = cfg.threshold
    if not 0.0 <= threshold <= 1.0:
        raise typer.BadParameter("must be in [0, 1]", param_hint="--threshold")
    lo = uncertain_lo if uncertain_lo is not None else cfg.uncertain_band[0]
    hi = uncertain_hi if uncertain_hi is not None else cfg.uncertain_band[1]
    if not (0.0 <= lo <= hi <= 1.0):
        raise typer.BadParameter(
            f"uncertain band must satisfy 0 <= lo <= hi <= 1 (got {lo}, {hi})"
        )

    store = ResultsStore(cfg.storage.db_path)
    results = store.get_run(run_id)
    if not results:
        console.print(f"[yellow]No results found for run {run_id}.[/]")
        return
    for r in results:
        r.verdict = apply_verdict(r, threshold, (lo, hi))
    console.print(_summary_table(results))

    if dry_run:
        console.print("[dim]--dry-run set; nothing written.[/]")
        return

    new_id = store.save_reverdict_run(
        parent_run_id=run_id,
        results=results,
        threshold=threshold,
        uncertain_band=(lo, hi),
    )
    console.print(
        f"Wrote derived run [cyan]{new_id}[/] (parent={run_id}, threshold={threshold})."
    )


@app.command("models")
def models_cmd():
    """List available registered models."""
    from .models.registry import list_models
    for m in list_models():
        console.print(f"- {m}")


@app.command("train")
def train_cmd(
    real_dir: Path = typer.Option(..., "--real", help="Folder of REAL labelled samples (walked recursively)."),
    fake_dir: Path = typer.Option(..., "--fake", help="Folder of FAKE labelled samples (walked recursively)."),
    output_dir: Path = typer.Option(Path("./checkpoints"), "--output-dir"),
    epochs: int = typer.Option(5, "--epochs"),
    learning_rate: float = typer.Option(1e-4, "--lr"),
    batch_size: int = typer.Option(16, "--batch-size"),
    val_split: float = typer.Option(0.1, "--val-split"),
    video_fps: float = typer.Option(1.0, "--video-fps", help="Frames per second to sample from video inputs."),
    max_frames_per_video: int = typer.Option(
        32, "--max-frames-per-video",
        help="Hard cap on frames sampled per video clip.",
    ),
    face_detection: bool = typer.Option(
        True, "--face-detect/--no-face-detect",
        help="Run MTCNN to crop the face from each input. Disable for already-face-cropped datasets (e.g. FFHQ/StyleGAN 256² faces).",
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume",
        help="Resume from output_dir/resume.pt if it exists and the config matches. Disable for a clean retrain.",
    ),
    device: str = typer.Option(
        "auto", "--device",
        help="Training device: 'auto' (CUDA if available, else CPU), 'cuda', 'cuda:0', or 'cpu'.",
    ),
    amp: bool = typer.Option(
        True, "--amp/--no-amp",
        help="Mixed-precision (float16) training on CUDA. ~2x faster on Volta+ GPUs; no-op on CPU.",
    ),
    check_only: bool = typer.Option(
        False, "--check-only",
        help="Only run the anti-bias constraint check; do not train.",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Fine-tune a backbone on labelled real/fake folders.

    Both folders are walked **recursively** and may contain images and videos
    at any depth. Enforces anti-bias constraints (min 50/class, fake ratio in
    [0.2, 0.8], >=5 distinct video sources for video-only classes) BEFORE any
    training. Use ``--check-only`` to validate a dataset without training.
    """
    _setup_logging(log_level)
    from .training import BiasConstraintError, enforce_constraints, summarise_dataset

    try:
        stats = enforce_constraints(summarise_dataset(real_dir, fake_dir))
    except BiasConstraintError as e:
        console.print(f"[red]Cannot train:[/] {e}")
        raise typer.Exit(code=1) from e
    console.print(
        f"[green]Dataset OK[/]: real={stats.real_count}, fake={stats.fake_count} "
        f"(ratio={stats.fake_ratio:.2f})"
    )
    for w in stats.warnings:
        console.print(f"[yellow]warn[/]: {w}")
    if check_only:
        return

    from .training.trainer import TrainConfig, train

    ckpt = train(
        TrainConfig(
            real_dir=real_dir,
            fake_dir=fake_dir,
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            val_split=val_split,
            video_fps=video_fps,
            max_frames_per_video=max_frames_per_video,
            face_detection=face_detection,
            resume=resume,
            device=device,
            amp=amp,
        ),
        stats=stats,  # already validated above; skip re-hashing every video
    )
    console.print(f"Wrote checkpoint [cyan]{ckpt}[/]")


@app.command("query")
def query_cmd(
    run_id: int | None = typer.Option(None, "--run", help="Filter by run id."),
    verdict: str | None = typer.Option(
        None, "--verdict", help="Filter by verdict (REAL/FAKE/UNCERTAIN/NO_FACE/ERROR)."
    ),
    path_glob: str | None = typer.Option(
        None, "--path", help="SQL LIKE pattern matched against the stored path (use %% as wildcard)."
    ),
    min_score: float | None = typer.Option(None, "--min-score"),
    max_score: float | None = typer.Option(None, "--max-score"),
    limit: int = typer.Option(200, "--limit"),
    config: Path | None = typer.Option(None, "--config", "-c"),
):
    """Filter persisted results without re-running inference."""
    cfg = _load(config)
    if verdict is not None:
        verdict = verdict.upper()
        if verdict not in Verdict.__members__:
            raise typer.BadParameter(
                f"choose from {', '.join(Verdict.__members__)}", param_hint="--verdict"
            )
    from .storage.db import ResultsStore
    store = ResultsStore(cfg.storage.db_path)
    results = store.query(
        run_id=run_id,
        verdict=verdict,
        path_glob=path_glob,
        min_score=min_score,
        max_score=max_score,
        limit=limit,
    )
    if not results:
        console.print("[yellow]No results match.[/]")
        return
    console.print(_summary_table(results))


if __name__ == "__main__":
    app()
