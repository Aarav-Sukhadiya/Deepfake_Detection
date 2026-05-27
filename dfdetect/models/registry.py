"""Model registry. Add a new model = `@register_model("name")` and it shows up everywhere."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..config import ModelConfig, ModelMemberConfig
from .base import DeepfakeModel

_REGISTRY: dict[str, Callable[..., DeepfakeModel]] = {}


def register_model(name: str) -> Callable[[Callable[..., DeepfakeModel]], Callable[..., DeepfakeModel]]:
    def deco(factory: Callable[..., DeepfakeModel]) -> Callable[..., DeepfakeModel]:
        _REGISTRY[name] = factory
        return factory

    return deco


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def _build_single(
    name: str,
    weights: str | None,
    url: str | None,
    sha256: str | None,
    device: str,
    weights_dir: Path | None,
) -> DeepfakeModel:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    weights_path = Path(weights) if weights else None
    return _REGISTRY[name](
        weights_path=weights_path,
        device=device,
        weights_dir=weights_dir,
        weights_url=url,
        weights_sha256=sha256,
    )


def build_model(cfg: ModelConfig, device: str = "cpu", weights_dir: Path | None = None) -> DeepfakeModel:
    # Importing the factories here forces decorator side-effects without circular imports.
    from . import efficientnet  # noqa: F401
    from .calibration import CalibratedModel
    from .ensemble import EnsembleModel

    def _wrap_calibration(model: DeepfakeModel, calib) -> DeepfakeModel:
        if calib.temperature == 1.0 and calib.platt_a is None and calib.platt_b is None:
            return model
        return CalibratedModel(
            model,
            temperature=calib.temperature,
            platt_a=calib.platt_a,
            platt_b=calib.platt_b,
        )

    if cfg.ensemble:
        if not cfg.members:
            raise ValueError(
                "model.ensemble=true but model.members is empty. "
                "List at least two member models under `model.members`."
            )
        members: list[tuple[DeepfakeModel, float]] = []
        for m in cfg.members:
            assert isinstance(m, ModelMemberConfig)
            inner = _build_single(m.name, m.weights, m.url, m.sha256, device, weights_dir)
            members.append((_wrap_calibration(inner, m.calibration), m.weight))
        ensemble = EnsembleModel(members)
        return _wrap_calibration(ensemble, cfg.calibration)

    base = _build_single(cfg.name, cfg.weights, cfg.url, cfg.sha256, device, weights_dir)
    return _wrap_calibration(base, cfg.calibration)
