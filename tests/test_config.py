from pathlib import Path

from dfdetect.config import Config, load_config


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert 0.0 <= cfg.threshold <= 1.0
    assert cfg.video.max_frames > 0
    assert cfg.preprocess.input_size > 0


def test_uncertain_band_is_inside_threshold():
    cfg = load_config()
    lo, hi = cfg.uncertain_band
    assert 0.0 <= lo <= hi <= 1.0


def test_paths_expand(tmp_path: Path):
    cfg = Config()
    assert cfg.storage.db_path.is_absolute()
    assert cfg.storage.models_dir.is_absolute()
