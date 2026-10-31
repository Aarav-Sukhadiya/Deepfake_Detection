# Models

dfdetect's model layer is a small registry pattern. Adding a new backbone is one new file plus a config edit.

---

## Built-in models

| Name | Backbone | Input size | Notes |
|---|---|---|---|
| `efficientnet_b4` | `tf_efficientnet_b4` (timm) | 224 | Default. ImageNet-pretrained fallback when no checkpoint configured. |

Reserved for future versions per the PRD: `xception` (FaceForensics++), `mesonet` (lightweight fallback), and an ensemble averaging EfficientNet + Xception.

List what's currently registered with:

```bash
dfdetect models
```

---

## Wiring real deepfake weights

The bundled EfficientNet-B4 falls back to ImageNet weights so the pipeline runs end-to-end out of the box, but those scores are **not calibrated for deepfake detection**. There are two supported paths to a real checkpoint.

### Option A — local file

1. Obtain a DFDC- or FaceForensics++-trained `tf_efficientnet_b4` checkpoint (`.pt` or `.pth`), or train one with `dfdetect train` (see [USAGE.md](USAGE.md#27-train--fine-tune-on-labelled-folders)).
2. Place it somewhere stable, e.g. `~/.cache/dfdetect/models/efficientnet_b4_dfdc.pt`.
3. Point your config at it:

   ```yaml
   model:
     name: efficientnet_b4
     weights: ~/.cache/dfdetect/models/efficientnet_b4_dfdc.pt
   ```

### Option B — verified download

When you host (or trust) a URL for the checkpoint, set `url` and **strongly** prefer setting `sha256`:

```yaml
model:
  name: efficientnet_b4
  url: https://example.org/dfdc/efficientnet_b4_dfdc.pt
  sha256: 9f3a... (64 hex chars)
```

On the first run the file is streamed into `storage.models_dir/<model_name>.pt` with a `.part` sidecar; only after the SHA-256 matches is it atomically renamed into place. Subsequent runs reuse the cached file. A bad checksum deletes the partial download and raises `ChecksumError` rather than silently using corrupt weights. Set `sha256: null` to disable verification (not recommended).

You can also pre-register the source in code so users don't have to know the URL:

```python
from dfdetect.models.downloader import register_weights, WeightSource

register_weights(
    "efficientnet_b4",
    WeightSource(url="https://...", sha256="..."),
)
```

How checkpoints are loaded:

- `torch.load(..., weights_only=True)`: only tensors and plain containers are accepted, so a malicious `.pt` can't execute code on load. Checkpoints must therefore be **state dicts**, not pickled whole models.
- A `{"state_dict": ...}` wrapper is unwrapped, and a uniform `module.` (DataParallel/DDP) or `model.` prefix is stripped.
- **Missing parameters are an error.** A checkpoint from a different architecture raises `RuntimeError: … is not a tf_efficientnet_b4 checkpoint` instead of silently leaving the network randomly initialised. Extra, unexpected keys only log a warning.

After loading, the `model_hash` in your reports is the truncated SHA-256 of the actual file on disk, so you can match a stored run back to the exact weights that produced it.

---

## Adding a new model

Three steps. Total ~50 lines.

### 1. Create the wrapper

Drop a new file in `dfdetect/models/`:

```python
# dfdetect/models/xception.py
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

from .base import DeepfakeModel
from .registry import register_model


@register_model("xception")
class Xception(DeepfakeModel):
    name = "xception"
    input_size = 299

    # The registry passes weights_dir / weights_url / weights_sha256 too; accept
    # them (see efficientnet.py for how to use ensure_weights_file with them).
    def __init__(self, weights_path: Path | None = None, device: str = "cpu",
                 weights_dir: Path | None = None, weights_url: str | None = None,
                 weights_sha256: str | None = None):
        super().__init__(weights_path=weights_path, device=device)
        self._model = None

    def _build(self):
        import timm, torch
        m = timm.create_model("xception", pretrained=self.weights_path is None, num_classes=2)
        if self.weights_path and self.weights_path.exists():
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            m.load_state_dict(state)   # strict: fail loudly on a mismatched checkpoint
        m.eval().to(self.device)
        self._model = m

    def preprocess(self, faces: list[np.ndarray]) -> Any:
        # match the EfficientNet wrapper's preprocessing — see efficientnet.py
        ...

    def predict(self, batch: Any) -> list[float]:
        import torch
        if self._model is None: self._build()
        with torch.inference_mode():
            return torch.softmax(self._model(batch), dim=1)[:, 1].cpu().tolist()

    def grad_cam_target_layer(self) -> Any:
        if self._model is None: self._build()
        return self._model.conv4   # tweak to whatever layer makes sense
```

### 2. Register the import

`dfdetect/models/registry.py` imports the `efficientnet` module at `build_model` time to trigger its decorator. Add your module the same way:

```python
def build_model(cfg, device="cpu", weights_dir=None):
    from . import efficientnet  # noqa: F401
    from . import xception      # noqa: F401  ← add this
    ...
```

### 3. Update your config

```yaml
model:
  name: xception
  weights: ~/.cache/dfdetect/models/xception_ff.pt
```

`dfdetect models` should now list `xception` alongside `efficientnet_b4`.

---

## The `DeepfakeModel` contract

Subclasses must implement three abstract methods; the rest have working defaults. Returning the same shapes the rest of the pipeline expects is the only hard requirement.

| Method | Input | Output | Notes |
|---|---|---|---|
| `preprocess(faces)` | `list[np.ndarray]` of HxWx3 RGB uint8 face crops | model-ready batch tensor | Resize to `self.input_size`; normalize per the model's training scheme. |
| `predict(batch)` | tensor from `preprocess` | `list[float]` of FAKE probabilities, one per crop | Convention: class 1 = FAKE for binary. |
| `grad_cam_target_layer()` | — | a `torch.nn.Module` | The conv layer Grad-CAM should hook into. Usually the last conv block. |
| `grad_cam_model()` | — | `torch.nn.Module` or `None` | *Default provided:* returns `self._model`. Override in wrappers. |
| `grad_cam_preprocess(faces)` | same as `preprocess` | tensor | *Default provided:* calls `preprocess`. Override if your `preprocess` doesn't return a tensor (as `EnsembleModel` does). |
| `hash()` | — | `str` | *Default provided:* truncated SHA-256 of the weights file (memoised), or `"pretrained"` when none. |

Class index 1 must be FAKE: heatmaps explain logit 1, and `predict` must return its softmax probability.

Implementations should **lazy-import torch / timm** so the package stays usable without the full ML stack installed (this matters for the test suite).

---

## Calibration

`DeepfakeModel.predict()` returns post-softmax fake probabilities, but those probabilities are only well-calibrated on the data the model was trained against. `dfdetect.models.calibration.CalibratedModel` wraps any model and rescales its scores in two optional stages:

1. **Temperature scaling** on the underlying logit. Set `model.calibration.temperature: <T>`. `T > 1` softens an over-confident model, `T < 1` sharpens an under-confident one, `T = 1` is a no-op.
2. **Platt rescaling** — a 1-D logistic `sigmoid(a · p + b)` applied after temperature. Set both `platt_a` and `platt_b` to enable; leave either `null` to skip this stage.

Calibration applies to ensemble members individually *and* to the ensemble's averaged output, so you can compose a per-member fit with an outer correction:

```yaml
model:
  ensemble: true
  members:
    - { name: efficientnet_b4, weights: ./dfdc.pt, calibration: { temperature: 1.4 } }
    - { name: efficientnet_b4, weights: ./ff.pt,   calibration: { temperature: 1.2 } }
  calibration:
    platt_a: 1.1
    platt_b: -0.1
```

Fitting procedure (offline, one-time):

1. Pick a calibration set (a few hundred labeled examples not seen during training).
2. Sweep temperatures on a log grid and pick the one that minimises **NLL** or **ECE** on the held-out scores. `scipy.optimize.minimize_scalar` over `[0.1, 5.0]` is usually enough.
3. If residual miscalibration remains, fit Platt on the temperature-scaled scores: any sklearn `LogisticRegression(C=large)` fit on `p_T → label` gives you `(a, b)`.
4. Persist the fitted values alongside the weights (filename or config) so they travel together.

Sanity checks:

- The `uncertain_band` config (default `[0.4, 0.6]`) should narrow as your calibration improves — wider bands cover up an over-confident model.
- For aggregation, `weighted` with a heavier `max` term catches localized swaps; pure `mean` smooths over sparse forgeries.
- The `--heatmaps` flag is the fastest way to sanity-check that the model is attending to face regions and not, say, video compression artifacts in the background.

---

## Ensembles

Enable `model.ensemble: true` and list two or more `members`. Each member is built independently (with its own weights, URL, calibration), and `EnsembleModel` returns the per-face weighted average of their predictions. Member weights are normalised, so `weight: 1.0` everywhere = a simple mean.

```yaml
model:
  ensemble: true
  members:
    - name: efficientnet_b4
      weights: ~/.cache/dfdetect/models/dfdc_b4.pt
      weight: 2.0
    - name: efficientnet_b4
      url: https://example.org/ff_b4.pt
      sha256: 9f3a...
      weight: 1.0
```

Constraints:

- Nested ensembles are not supported (members can't themselves be ensembles).
- Grad-CAM uses the first member (both its network and its preprocessing), since CAM operates on a single conv graph.
- `model_hash` for an ensemble is `ens(<member1>|<member2>|...)`, so a stored run still uniquely identifies the bag of weights used.
