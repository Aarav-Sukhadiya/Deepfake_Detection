"""Render a model card markdown file summarising a run's model + settings.

A model card is a short doc that travels with a run: what model was used,
what data it was trained on (if known), known failure modes, calibration
status. Inspired by Mitchell et al.'s "Model Cards for Model Reporting"
(2019) and called out in the PRD §13.

This module renders the card from data the pipeline already records — it
does not invent claims about training data the user hasn't provided.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from ..config import Config
from ..types import FileResult, Verdict


def render_model_card(
    cfg: Config,
    results: Iterable[FileResult],
    out_path: Path,
    training_notes: str | None = None,
) -> Path:
    """Render a markdown model card alongside a run's report."""
    results = list(results)
    n = len(results)
    by_verdict = {v: 0 for v in Verdict}
    for r in results:
        by_verdict[r.verdict] = by_verdict.get(r.verdict, 0) + 1

    model_name = results[0].model_name if results else cfg.model.name
    model_hash = results[0].model_hash if results else "<unknown>"
    calib = cfg.model.calibration
    calibrated = calib.temperature != 1.0 or calib.platt_a is not None

    lines: list[str] = [
        f"# Model Card — `{model_name}`",
        "",
        f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}._",
        "",
        "## Model",
        "",
        f"- **Name:** `{model_name}`",
        f"- **Hash (truncated SHA-256 of weights):** `{model_hash}`",
        f"- **Ensemble:** {cfg.model.ensemble}",
    ]
    if cfg.model.ensemble and cfg.model.members:
        lines.append("- **Members:**")
        for m in cfg.model.members:
            src = m.weights or m.url or "<imagenet-fallback>"
            lines.append(f"  - `{m.name}` weight={m.weight} source=`{src}`")
    lines += [
        "",
        "## Calibration",
        "",
        f"- **Temperature:** {calib.temperature}",
        f"- **Platt a/b:** {calib.platt_a} / {calib.platt_b}",
        f"- **Active:** {'yes' if calibrated else 'no (raw softmax)'}",
    ]
    if not calibrated:
        lines.append(
            "- _Scores are raw post-softmax probabilities, not calibrated. "
            "Treat them as ranking signal, not probabilities._"
        )

    lines += [
        "",
        "## Decision policy",
        "",
        f"- **Threshold:** {cfg.threshold}",
        f"- **Uncertain band:** {list(cfg.uncertain_band)}",
        f"- **Per-frame aggregation:** {cfg.inference.aggregation}",
        f"- **Per-track aggregation:** {cfg.inference.track_aggregation}",
        "",
        "## Inputs in this run",
        "",
        f"- **Files analyzed:** {n}",
    ]
    for v in Verdict:
        if by_verdict.get(v, 0):
            lines.append(f"  - `{v.value}`: {by_verdict[v]}")

    lines += [
        "",
        "## Known limitations",
        "",
        "- The bundled EfficientNet-B4 falls back to ImageNet pretraining when "
        "no deepfake-trained weights are configured. Scores from that "
        "fallback are not meaningful for deepfake detection.",
        "- Audio deepfakes are not handled (IMPROVEMENTS.md #22).",
        "- Detection generalisation to manipulation families not present in "
        "the training data is bounded by the public checkpoint used. Always "
        "present scores alongside verdicts.",
    ]
    if training_notes:
        lines += ["", "## Training data / provenance", "", training_notes]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
