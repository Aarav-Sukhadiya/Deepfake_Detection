"""YAML-backed, pydantic-validated configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


class InputConfig(BaseModel):
    recursive: bool = False
    follow_symlinks: bool = False
    image_exts: list[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    video_exts: list[str] = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


class PreprocessConfig(BaseModel):
    face_detector: Literal["mtcnn", "retinaface"] = "mtcnn"
    face_margin: float = 0.30
    input_size: int = 224
    min_face_size: int = 40
    largest_face_only: bool = True


class VideoConfig(BaseModel):
    frames_per_second: float = 1.0
    max_frames: int = 64
    sampling_strategy: Literal["uniform", "scene", "hybrid"] = "uniform"
    # Lower = stricter (more boundaries flagged). HSV-histogram correlation
    # between consecutive thumbnails; values below this fire a boundary.
    scene_threshold: float = 0.6
    # Scene-detector thumbnail size and downsample stride. Detection runs on
    # every Nth decoded frame at this resolution to keep cost bounded even on
    # long videos. ``scene_stride: 0`` means "use the uniform step".
    scene_thumb_size: int = 64
    scene_stride: int = 0

    @field_validator("scene_threshold")
    @classmethod
    def _scene_threshold_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("scene_threshold must be in [0, 1]")
        return v


class WeightedConfig(BaseModel):
    mean: float = 0.6
    median: float = 0.2
    max: float = 0.2


class InferenceConfig(BaseModel):
    aggregation: Literal["mean", "median", "max", "weighted", "quality_weighted"] = "mean"
    weighted: WeightedConfig = Field(default_factory=WeightedConfig)
    # How to collapse per-track scores into a single per-file score when
    # multi-face videos produce more than one identity. ``max`` is the safe
    # default for deepfake detection — one fake speaker in a multi-person
    # interview should mark the whole clip as suspicious.
    track_aggregation: Literal["max", "mean", "off"] = "max"


class CalibrationConfig(BaseModel):
    """Probability calibration applied after raw softmax.

    ``temperature`` rescales logits before softmax (T>1 softens, T<1 sharpens).
    ``platt_a`` / ``platt_b`` apply a 1-D logistic on the post-softmax fake
    probability (``sigmoid(a * p + b)``) when both are set; combine with
    temperature for a two-stage fit.
    """

    temperature: float = 1.0
    platt_a: float | None = None
    platt_b: float | None = None

    @field_validator("temperature")
    @classmethod
    def _temp_positive(cls, v):
        if v <= 0:
            raise ValueError("temperature must be > 0")
        return v


class ModelMemberConfig(BaseModel):
    name: str
    weights: str | None = None
    url: str | None = None
    sha256: str | None = None
    num_classes: int = 2
    weight: float = 1.0          # ensemble averaging weight
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)


class ModelConfig(BaseModel):
    name: str = "efficientnet_b4"
    weights: str | None = None
    url: str | None = None       # remote source for the weights file
    sha256: str | None = None    # expected hex digest of the downloaded file
    num_classes: int = 2
    ensemble: bool = False
    # Ensemble member definitions used when ``ensemble=True``. Nested
    # ensembles are not supported.
    members: list[ModelMemberConfig] = Field(default_factory=list)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)


class StorageConfig(BaseModel):
    db_path: Path = Path("~/.cache/dfdetect/results.db").expanduser()
    models_dir: Path = Path("~/.cache/dfdetect/models").expanduser()

    @field_validator("db_path", "models_dir", mode="before")
    @classmethod
    def _expand(cls, v):
        return Path(str(v)).expanduser()


class ReportConfig(BaseModel):
    default_format: Literal["json", "csv", "html"] = "json"
    heatmaps: bool = False
    output_dir: Path = Path("./reports")


class LoggingConfig(BaseModel):
    level: str = "INFO"


class PipelineConfig(BaseModel):
    # Number of threads used to pre-extract face crops while the model is
    # running on the previous file. 0 = single-threaded (today's behaviour).
    workers: int = 0
    # Bound the in-flight queue so we don't fan out arbitrarily many files
    # when preprocessing is much faster than inference.
    prefetch: int = 2


class Config(BaseModel):
    device: Literal["auto", "cuda", "cpu"] = "auto"
    batch_size: int = 8
    threshold: float = 0.5
    uncertain_band: tuple[float, float] = (0.4, 0.6)

    input: InputConfig = Field(default_factory=InputConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @field_validator("threshold")
    @classmethod
    def _threshold_in_unit(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold must lie in [0, 1] (got {v})")
        return v

    @field_validator("uncertain_band", mode="before")
    @classmethod
    def _uncertain_band(cls, v):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            lo, hi = float(v[0]), float(v[1])
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(
                    f"uncertain_band must satisfy 0 <= lo <= hi <= 1 (got [{lo}, {hi}])"
                )
            return (lo, hi)
        raise ValueError("uncertain_band must be a 2-element list/tuple")

    @field_validator("batch_size")
    @classmethod
    def _batch_size_positive(cls, v):
        if v < 1:
            raise ValueError("batch_size must be >= 1")
        return v


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file, falling back to packaged defaults."""
    target = Path(path) if path else DEFAULT_CONFIG_PATH
    if not target.exists():
        return Config()
    with open(target) as f:
        data = yaml.safe_load(f) or {}
    return Config(**data)
