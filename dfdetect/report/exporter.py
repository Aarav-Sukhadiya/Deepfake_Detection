"""JSON / CSV / HTML exporters for FileResult lists."""
from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from functools import cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template

from ..types import FileResult

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


@cache
def _report_template() -> Template:
    # ``autoescape=True`` explicitly: ``select_autoescape(["html"])`` matches on
    # the file extension and so would NOT escape ``report.html.j2``, letting a
    # crafted filename inject markup/script into the report.
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=True)
    return env.get_template("report.html.j2")


def write_json(results: Iterable[FileResult], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    return out_path


def write_csv(results: Iterable[FileResult], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "kind",
        "verdict",
        "score",
        "faces_analyzed",
        "processing_time_s",
        "model_name",
        "model_hash",
        "error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            # Read fields off the dataclass directly — going through
            # ``to_dict`` would build frame_scores/segments lists per row,
            # which CSV throws away. Cheap to skip for long videos.
            w.writerow(
                {
                    "path": str(r.path),
                    "kind": r.kind.value if r.kind else "",
                    "verdict": r.verdict.value,
                    "score": round(r.score, 4),
                    "faces_analyzed": r.faces_analyzed,
                    "processing_time_s": round(r.processing_time_s, 3),
                    "model_name": r.model_name,
                    "model_hash": r.model_hash,
                    "error": r.error or "",
                }
            )
    return out_path


def write_html(
    results: Iterable[FileResult],
    out_path: Path,
    title: str = "dfdetect report",
    threshold: float = 0.5,
    uncertain_band: tuple[float, float] = (0.4, 0.6),
) -> Path:
    rows = [r.to_dict() for r in results]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _report_template().render(
            title=title,
            rows=rows,
            threshold=threshold,
            uncertain_lo=uncertain_band[0],
            uncertain_hi=uncertain_band[1],
        ),
        encoding="utf-8",
    )
    return out_path
