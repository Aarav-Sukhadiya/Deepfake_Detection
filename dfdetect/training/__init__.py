"""Fine-tuning entry points (IMPROVEMENTS.md #27).

The training module turns a pair of folders (``--real`` / ``--fake``) into a
new checkpoint that the inference path can load via ``model.weights``. Both
folders are walked **recursively** and may contain images and videos at any
depth. It's intentionally lightweight: one epoch loop, one optimizer,
balanced sampling, hard anti-bias constraints up front so users don't
accidentally train on a single-class or duplicate-only set.
"""
from .constraints import (  # noqa: F401
    BiasConstraintError,
    DatasetStats,
    enforce_constraints,
    summarise_dataset,
    walk_media,
)
