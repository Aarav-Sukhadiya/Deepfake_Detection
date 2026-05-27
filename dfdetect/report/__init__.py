"""Result exporters and Grad-CAM overlays."""
from .exporter import write_csv, write_html, write_json

__all__ = ["write_csv", "write_json", "write_html", "save_grad_cam"]


def save_grad_cam(*args, **kwargs):
    # Lazy import — pulls torch + grad-cam only when actually used.
    from .heatmap import save_grad_cam as _impl
    return _impl(*args, **kwargs)
