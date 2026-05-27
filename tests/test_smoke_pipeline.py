"""End-to-end pipeline smoke against the bundled synthetic samples.

Skipped when the heavy ML stack (torch / timm / facenet-pytorch) isn't
installed — the rest of the suite still passes in dep-light environments.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("facenet_pytorch")


def test_pipeline_smokes_against_samples(tmp_path: Path):
    from dfdetect.config import Config
    from dfdetect.pipeline import Pipeline
    from dfdetect.types import Verdict

    samples = Path(__file__).resolve().parent.parent / "samples"
    images = sorted(samples.glob("*.png"))
    if not images:
        pytest.skip("No sample images present.")

    cfg = Config()
    cfg.device = "cpu"
    cfg.batch_size = 2
    cfg.storage.db_path = tmp_path / "results.db"
    cfg.storage.models_dir = tmp_path / "models"
    cfg.report.output_dir = tmp_path / "reports"

    pipeline = Pipeline(cfg)
    results = pipeline.run(images)
    assert len(results) == len(images)
    for r in results:
        assert r.model_name == cfg.model.name
        assert r.verdict in {
            Verdict.REAL, Verdict.FAKE, Verdict.UNCERTAIN,
            Verdict.NO_FACE, Verdict.ERROR,
        }
